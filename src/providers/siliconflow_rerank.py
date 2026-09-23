import math
import time
from collections.abc import Mapping
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.__base__.model_provider import (
    RerankModel,
    is_retryable_error,
    validate_secret_free_options,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import redact_sensitive_text, validate_secret_free_payload

logger = get_module_logger(__name__)


class SiliconflowRerankProvider(RerankModel):
    """SiliconFlow Rerank Provider（HTTP JSON）。

    本类与 ``jina.py`` 是近似副本（行级相似度约 0.78，7 个方法同名，
    ``_get_headers`` 逐字节相同）。两者**有意保持独立实现**，因为存在 4 处真实
    语义差异：``_OPTION_KEYS``、``_base_url`` 来源（本类走 settings，Jina 硬编码）、
    构造期 base_url 校验（本类执行，Jina 不执行）、以及 ``_parse_response`` 里
    ``results`` 类型校验的时机（本类在 Mapping 循环之后）。
    **修改任一侧的校验逻辑、payload 构造或响应解析时，必须同步另一侧。**

    已注入 CSE 性能传感器与 tenacity 重试机制。
    """

    _OPTION_KEYS = frozenset({"return_documents", "truncate", "extra_body", "timeout"})

    def __init__(self, model_name: str, options: dict[str, Any] | None = None):
        self._model_name = model_name
        self._options = validate_secret_free_options(options, "SiliconFlow Rerank")
        unknown = sorted(set(self._options).difference(self._OPTION_KEYS))
        if unknown:
            raise ValueError("SiliconFlow Rerank options 包含不支持的字段: " + ", ".join(unknown))
        settings = get_settings()
        self._api_key = settings.siliconflow_api_key
        self._base_url = settings.siliconflow_base_url

        if not self._api_key:
            logger.error("SiliconFlow API Key 未设置。")
            raise ValueError("SILICONFLOW_API_KEY is required for SiliconflowRerankProvider")
        if not self._base_url:
            logger.error("SiliconFlow Base URL 未设置。")
            raise ValueError("SILICONFLOW_BASE_URL is required for SiliconflowRerankProvider")

        logger.info(f"初始化 SiliconflowRerankProvider，模型: {model_name}")

    def _get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _prepare_payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        self._validate_inputs(query, documents, top_n)
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "model": self._model_name,
            "top_n": top_n,
            "return_documents": True,
        }
        options = dict(self._options)
        options.pop("timeout", None)
        extra_body = validate_secret_free_payload(
            options.pop("extra_body", None),
            "SiliconFlow Rerank",
            "options.extra_body",
        )
        overlap = sorted(set(payload).intersection(options))
        if overlap:
            raise ValueError("SiliconFlow Rerank options 不允许覆盖请求字段: " + ", ".join(overlap))
        if extra_body:
            option_overlap = sorted(set(options).intersection(extra_body))
            if option_overlap:
                raise ValueError(
                    "SiliconFlow Rerank options 与 options.extra_body 重复: "
                    + ", ".join(option_overlap)
                )
            body_overlap = sorted(set(payload).intersection(extra_body))
            if body_overlap:
                raise ValueError(
                    "SiliconFlow Rerank options.extra_body 不允许覆盖请求字段: "
                    + ", ".join(body_overlap)
                )
            options.update(dict(extra_body))
        payload.update(options)
        return payload

    def _request_timeout(self) -> float:
        timeout = self._options.get("timeout", 30.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("SiliconFlow Rerank options.timeout 必须是正数。")
        return float(timeout)

    @staticmethod
    def _validate_inputs(query: str, documents: list[str], top_n: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("SiliconFlow Rerank query 必须是非空字符串。")
        if not isinstance(documents, list) or not documents:
            raise ValueError("SiliconFlow Rerank documents 必须是非空列表。")
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("SiliconFlow Rerank documents 必须全部是字符串。")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("SiliconFlow Rerank top_n 必须是大于等于 1 的整数。")

    def _parse_response(
        self, results: list[dict], documents: list[str]
    ) -> tuple[list[int], list[float]]:
        positions: dict[str, list[int]] = {}
        for index, content in enumerate(documents):
            positions.setdefault(content, []).append(index)
        indices = []
        scores = []
        if not isinstance(results, list):
            raise RuntimeError("SiliconFlow Rerank 响应缺少有效的 results 列表。")
        for res in results:
            if not isinstance(res, Mapping):
                raise RuntimeError("SiliconFlow Rerank 响应包含无效结果项。")
            if "index" in res:
                direct_index = res["index"]
                if (
                    not isinstance(direct_index, int)
                    or isinstance(direct_index, bool)
                    or not 0 <= direct_index < len(documents)
                ):
                    raise RuntimeError("SiliconFlow Rerank 响应 index 无效。")
                if direct_index in indices:
                    raise RuntimeError("SiliconFlow Rerank 响应包含重复 index。")
                index = direct_index
                document = res.get("document")
                document_text = document.get("text") if isinstance(document, Mapping) else document
                if document_text is not None and document_text != documents[index]:
                    raise RuntimeError("SiliconFlow Rerank 响应 index 与 document 不匹配。")
                positions.get(documents[index], []).remove(index)
            else:
                document = res.get("document")
                doc_content = document.get("text") if isinstance(document, Mapping) else document
                if not isinstance(doc_content, str) or not positions.get(doc_content):
                    raise RuntimeError("SiliconFlow Rerank 响应无法映射到输入文档。")
                index = positions[doc_content].pop(0)
            try:
                score = float(res["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("SiliconFlow Rerank 响应 relevance_score 无效。") from exc
            if not math.isfinite(score):
                raise RuntimeError("SiliconFlow Rerank 响应 relevance_score 必须是有限数值。")
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
        logger.info(f"调用 SiliconFlow Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx

        start_time = time.perf_counter()
        url = f"{self._base_url.rstrip('/')}/rerank"

        try:
            with httpx.Client(timeout=self._request_timeout()) as client:
                response = client.post(
                    url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise RuntimeError("SiliconFlow Rerank 响应必须是 JSON 对象。")
                results = payload.get("results")
                if not isinstance(results, list):
                    raise RuntimeError("SiliconFlow Rerank 响应缺少有效的 results 列表。")

            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"SiliconFlow Rerank ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "SiliconFlow Rerank (%s) 出错: %s",
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
        logger.info(f"异步调用 SiliconFlow Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx

        start_time = time.perf_counter()
        url = f"{self._base_url.rstrip('/')}/rerank"

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout()) as aclient:
                response = await aclient.post(
                    url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise RuntimeError("SiliconFlow Rerank 响应必须是 JSON 对象。")
                results = payload.get("results")
                if not isinstance(results, list):
                    raise RuntimeError("SiliconFlow Rerank 响应缺少有效的 results 列表。")

            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"SiliconFlow Rerank ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "SiliconFlow Rerank (%s) 异步出错: %s",
                self._model_name,
                error_text,
            )
            raise
