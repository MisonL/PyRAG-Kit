import math
import time
from collections.abc import Mapping
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.__base__.model_provider import (
    RerankModel,
    is_retryable_error,
    normalize_client_timeout,
    validate_secret_free_options,
)
from src.utils.config import get_settings, is_double_convertible, safe_repr
from src.utils.log_manager import get_module_logger
from src.utils.security import safe_exception_text, validate_secret_free_payload

logger = get_module_logger(__name__)


# rerank 请求 ``top_n`` 的上界。上游 API 会自行截断，因此它不需要与文档数
# 挂钩；这个界的作用是挡住 TOML／调用方传进来的天文数字——它们会被静默接受，
# 直到 ``json.dumps`` 编码请求体时才抛与 rerank 无关的位数错误。
_MAX_RERANK_TOP_N = 10_000


class JinaProvider(RerankModel):
    """Jina Rerank Provider（HTTP JSON）。

    本类与 ``siliconflow_rerank.py`` 是近似副本（行级相似度约 0.78，7 个方法同名，
    ``_get_headers`` 逐字节相同）。两者**有意保持独立实现**，因为存在 4 处真实
    语义差异：``_OPTION_KEYS``（本类多 3 个 Jina 专有键）、``_base_url`` 来源
    （本类硬编码，SiliconFlow 走 settings）、构造期 base_url 校验、以及
    ``_parse_response`` 里 ``results`` 类型校验的时机。
    **修改任一侧的校验逻辑、payload 构造或响应解析时，必须同步另一侧。**

    已注入 CSE 性能传感器与 tenacity 重试机制。
    """

    _OPTION_KEYS = frozenset(
        {
            "return_documents",
            "max_chunks_per_doc",
            "late_chunking",
            "truncate",
            "embedding_type",
            "extra_body",
            "timeout",
        }
    )

    def __init__(self, model_name: str, options: dict[str, Any] | None = None):
        self._model_name = model_name
        self._options = validate_secret_free_options(options, "Jina Rerank")
        unknown = sorted(set(self._options).difference(self._OPTION_KEYS))
        if unknown:
            raise ValueError("Jina Rerank options 包含不支持的字段: " + ", ".join(unknown))
        settings = get_settings()
        self._api_key = settings.jina_api_key
        if not self._api_key:
            logger.error("Jina API Key 未设置。")
            raise ValueError("JINA_API_KEY is required for JinaProvider")
        self._base_url = "https://api.jina.ai/v1/rerank"
        logger.info(f"初始化 JinaProvider，模型: {model_name}")

    def _get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _prepare_payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        self._validate_inputs(query, documents, top_n)
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "model": self._model_name,
            "top_n": top_n,
            # 默认请求返回文档内容；与 SiliconFlow Rerank 保持一致的不可覆盖字段。
            "return_documents": True,
        }
        options = dict(self._options)
        options.pop("timeout", None)
        extra_body = validate_secret_free_payload(
            options.pop("extra_body", None),
            "Jina Rerank",
            "options.extra_body",
        )
        overlap = sorted(set(payload).intersection(options))
        if overlap:
            raise ValueError("Jina Rerank options 不允许覆盖请求字段: " + ", ".join(overlap))
        if extra_body:
            option_overlap = sorted(set(options).intersection(extra_body))
            if option_overlap:
                raise ValueError(
                    "Jina Rerank options 与 options.extra_body 重复: " + ", ".join(option_overlap)
                )
            body_overlap = sorted(set(payload).intersection(extra_body))
            if body_overlap:
                raise ValueError(
                    "Jina Rerank options.extra_body 不允许覆盖请求字段: " + ", ".join(body_overlap)
                )
            options.update(dict(extra_body))
        payload.update(options)
        return payload

    def _request_timeout(self) -> float:
        # 走共享 helper：它带**上界**。此前只有下界，实测 ``options.timeout``
        # 写成 ``10**400`` 时构造成功，到 httpx 比较超时才以形态各异的错误暴露。
        return normalize_client_timeout(
            self._options.get("timeout", 30.0), "Jina", "options.timeout"
        )

    @staticmethod
    def _validate_inputs(query: str, documents: list[str], top_n: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Jina Rerank query 必须是非空字符串。")
        if not isinstance(documents, list) or not documents:
            raise ValueError("Jina Rerank documents 必须是非空列表。")
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("Jina Rerank documents 必须全部是字符串。")
        # 上界此前缺失：``10**400`` 与 ``10**4301`` 都被静默接受，后者在
        # ``json.dumps`` 编码请求体时才抛「整数转字符串超限」——错误消息与
        # rerank 参数毫无关联，而同一份文件里 ``options.timeout`` 与响应
        # ``relevance_score`` 都已判过双精度范围，只有这里漏了。
        #
        # 上界取 ``_MAX_RERANK_TOP_N`` 而不是 ``len(documents)``：上游 API 自己
        # 会截断，``top_n`` 超过文档数是合法输入（由
        # ``test_rerank_accepts_top_n_larger_than_document_count`` 固定）。
        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or not 1 <= top_n <= _MAX_RERANK_TOP_N
        ):
            raise ValueError(
                f"Jina Rerank top_n 必须是 1 到 {_MAX_RERANK_TOP_N} 之间的整数，"
                f"但得到 {safe_repr(top_n)}。"
            )

    def _parse_response(
        self, results: list[dict], documents: list[str]
    ) -> tuple[list[int], list[float]]:
        if not isinstance(results, list):
            raise RuntimeError("Jina Rerank 响应缺少有效的 results 列表。")
        # 创建内容到索引的映射
        positions: dict[str, list[int]] = {}
        for index, content in enumerate(documents):
            positions.setdefault(content, []).append(index)
        indices = []
        scores = []
        for res in results:
            if not isinstance(res, Mapping):
                raise RuntimeError("Jina Rerank 响应包含无效结果项。")
            if "index" in res:
                direct_index = res["index"]
                if (
                    not isinstance(direct_index, int)
                    or isinstance(direct_index, bool)
                    or not 0 <= direct_index < len(documents)
                ):
                    raise RuntimeError("Jina Rerank 响应 index 无效。")
                if direct_index in indices:
                    raise RuntimeError("Jina Rerank 响应包含重复 index。")
                index = direct_index
                document = res.get("document")
                document_text = document.get("text") if isinstance(document, Mapping) else document
                if document_text is not None and document_text != documents[index]:
                    raise RuntimeError("Jina Rerank 响应 index 与 document 不匹配。")
                positions.get(documents[index], []).remove(index)
            else:
                document = res.get("document")
                doc_content = document.get("text") if isinstance(document, Mapping) else document
                if not isinstance(doc_content, str) or not positions.get(doc_content):
                    raise RuntimeError("Jina Rerank 响应无法映射到输入文档。")
                index = positions[doc_content].pop(0)
            # 取分数用 ``.get`` 而不是 ``[]``：``res["relevance_score"]`` 在字段
            # 缺失时抛 ``KeyError``，而它**不在**下面那个 ``try`` 的覆盖范围内
            # （那一段只包 ``float(...)``），于是本方法承诺的 ``RuntimeError``
            # 变成裸 ``KeyError: 'relevance_score'``。响应来自网络 JSON，一个
            # 不含该字段的结果项就能触发；调用方的错误分类与日志脱敏都按
            # ``RuntimeError`` 写，接不住 ``KeyError``。
            #
            # ``float(10**400)`` 抛 ``OverflowError``——它同样不是 ``ValueError``，
            # 因此下面这个 ``except`` 也接不住。响应来自网络 JSON，一个超大整数
            # 字面量就能触发。这里先判可转换性再交给 ``float``。
            raw_score = res.get("relevance_score") if isinstance(res, Mapping) else None
            if (
                isinstance(raw_score, int)
                and not isinstance(raw_score, bool)
                and not is_double_convertible(raw_score)
            ):
                raise RuntimeError("Jina Rerank 响应 relevance_score 超出双精度范围。")
            try:
                score = float(res["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Jina Rerank 响应 relevance_score 无效。") from exc
            if not math.isfinite(score):
                raise RuntimeError("Jina Rerank 响应 relevance_score 必须是有限数值。")
            indices.append(index)
            scores.append(score)
        return indices, scores

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    def rerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """同步 Rerank (CSE Sensor)。"""
        # ``len()`` 必须放在校验**之后**：``documents`` 是公开入口的参数，非序列
        # 取值在这里抛裸 ``TypeError: object of type 'int' has no len()``——而
        # 紧随其后的 ``_prepare_payload`` 对同一取值给的是带指引的
        # ``ValueError: documents 必须是非空列表``。同一判据的两半顺序反了，
        # 「参数写错」的收场就取决于日志级别是否够低（DEBUG 关闭时 f-string
        # 仍会求值，所以它与日志级别无关，只是位置错了）。
        self._validate_inputs(query, documents, top_n)
        logger.info(f"调用 Jina Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx

        start_time = time.perf_counter()

        try:
            with httpx.Client(timeout=self._request_timeout()) as client:
                response = client.post(
                    self._base_url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise RuntimeError("Jina Rerank 响应必须是 JSON 对象。")
                results = payload.get("results")
                if not isinstance(results, list):
                    raise RuntimeError("Jina Rerank 响应缺少有效的 results 列表。")

            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"Jina Rerank ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            error_text = safe_exception_text(e)
            logger.exception(
                "Jina Rerank (%s) 出错: %s",
                self._model_name,
                error_text,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> tuple[list[int], list[float]]:
        """异步 Rerank (CSE Sensor)。"""
        # 与同步入口同判据：``len()`` 必须在 ``_validate_inputs`` 之后。
        self._validate_inputs(query, documents, top_n)
        logger.info(f"异步调用 Jina Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout()) as aclient:
                response = await aclient.post(
                    self._base_url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise RuntimeError("Jina Rerank 响应必须是 JSON 对象。")
                results = payload.get("results")
                if not isinstance(results, list):
                    raise RuntimeError("Jina Rerank 响应缺少有效的 results 列表。")

            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"Jina Rerank ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            error_text = safe_exception_text(e)
            logger.exception(
                "Jina Rerank (%s) 异步出错: %s",
                self._model_name,
                error_text,
            )
            raise
