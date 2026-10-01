from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.providers.__base__.model_provider import close_resource_sync
from src.providers.factory import ModelProviderFactory
from src.runtime.contracts import RunConfig
from src.utils.config import (
    _MAX_EMBEDDING_DIMENSION,
    _MAX_KB_CHUNK_TOKENS,
    is_double_convertible,
    require_detail_attributes,
    safe_repr,
)


def _reject_unconvertible_numbers(values: Any, label: str) -> None:
    """拒绝会让 ``float()`` 抛 ``OverflowError`` 的整数。

    只递归检查 ``int``：``float`` 的 ``inf``／``nan`` 由调用方的
    ``np.isfinite`` 判据负责（消息更贴切），这里专管「形态是合法数字、值却
    超出 C double」这一类。``bool`` 是 ``int`` 子类，一并跳过——它是合法取值。
    """
    if isinstance(values, bool):
        return
    if isinstance(values, int):
        if not is_double_convertible(values):
            raise ValueError(
                f"{label}包含超出双精度范围的整数（{safe_repr(values)}）；"
                "请检查 embedding provider 的返回内容。"
            )
        return
    if isinstance(values, (list, tuple)):
        for item in values:
            _reject_unconvertible_numbers(item, label)
        return
    # ``ndarray`` 与「``ndarray`` 的 list」都是 ``_probe_dimension`` 显式支持的
    # 输入形态，因此它们会走到这里。不递归的话，``dtype=object`` 的数组带着
    # ``10**400`` 直接进 ``np.array(..., dtype=np.float32)``，由 numpy 抛裸
    # ``OverflowError``——同一批数据，``list`` 形态得到下面那条干净的
    # ``ValueError``，``ndarray`` 形态得到 ``ArithmeticError`` 子类，收场方式
    # 不一致，且穿透调用方的 ``except ValueError`` 分类与日志脱敏。
    # 只有 ``object`` dtype 能装下 Python 大整数；数值 dtype 的取值一定在
    # double 范围内，无需遍历（``tolist()`` 对高维数组代价不低）。
    if isinstance(values, np.ndarray) and values.dtype == object:
        for item in values.ravel().tolist():
            _reject_unconvertible_numbers(item, label)


class EmbeddingService:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self._embedding_model = None

    @property
    def embedding_provider_key(self) -> str:
        return self.run_config.default_embedding_provider

    @property
    def embedding_model_detail(self) -> Any:
        """按 key 取 embedding 模型详情，缺失时给出与工厂一致的 ``ValueError``。

        原实现是直接下标 ``configurations[key]``，而 ``default_embedding_provider``
        没有任何 validator（``Settings`` 里只是普通 ``str``）。打错一个 provider
        名时这里抛裸 ``KeyError``，而同一份配置走
        ``ModelProviderFactory.get_embedding_provider`` 得到的是带可选值列表的
        ``ValueError``——两条路径的报错契约本该一致，否则调用方按
        ``ValueError`` 写的错误分类与脱敏接不住 ``KeyError``。

        调用点包括 ``embed_query``（Google 渠道分支）与
        ``knowledge_build_service`` 的清单写入，都在检索/建库的主路径上。
        """
        configurations = self.run_config.embedding_configurations
        key = self.embedding_provider_key
        if not isinstance(configurations, Mapping) or key not in configurations:
            # 先渲染成字符串再排序，理由同 ``display_utils._require_model_detail``
            # 的同名位置：``sorted(configurations)`` 直接比较**原始键**，混合类型
            # 键抛裸 ``TypeError: '<' not supported between instances of 'str' and
            # 'int'``，而这条消息的用途正是「告诉用户可选值是什么」。
            # ``embedding_configurations`` 由 ``RunConfig`` 携带（frozen
            # dataclass、字段无校验器），嵌入方可自拼 ``RunConfig`` 送入任意键。
            available = (
                ", ".join(sorted(safe_repr(item) for item in configurations))
                if isinstance(configurations, Mapping)
                else "无"
            )
            raise ValueError(
                f"在Embedding配置中未找到key: {safe_repr(key)}。可选值: {available}。"
                " 请检查 config.toml 里的 default_embedding_provider 与"
                " embedding_configurations 是否对应。"
            )
        # 键在**不**代表值可用。``{"local-hash": None}`` 的键确实存在，于是不会
        # 走上面「未找到key」的分支，而是把 ``None`` 原样返回——两个调用点紧接着
        # 读 ``.provider``／``.model_name``，在那里抛裸 ``AttributeError``。
        # 与 ``ModelProviderFactory._get_model_detail`` 的同类情形对齐：同一个
        # ``{"k": None}`` 走工厂得到 ``ValueError``，走这里得到 ``AttributeError``，
        # 而调用方按 ``ValueError`` 写的错误分类与日志脱敏接不住后者。
        # 判据用属性存在性而非 ``isinstance(..., ModelDetail)``，理由见
        # ``require_detail_attributes`` 的 docstring（测试用 ``SimpleNamespace``
        # 构造等价配置）。
        detail = configurations[key]
        require_detail_attributes(
            detail,
            ("provider", "model_name"),
            field="embedding_configurations",
            key=key,
            role_label="Embedding",
        )
        return detail

    def _get_model(self):
        if self._embedding_model is None:
            self._embedding_model = ModelProviderFactory.get_embedding_provider(
                self.embedding_provider_key,
                self.run_config.embedding_configurations,
            )
        return self._embedding_model

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = await model.aembed_documents(texts)
        return self._as_float32_matrix(embeddings, expected_rows=len(texts))

    async def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        options: dict[str, Any] = {}
        if self.embedding_model_detail.provider == "google":
            options.setdefault("task_type", "RETRIEVAL_QUERY")
        vector = await model.aembed_query(query, **options)
        matrix = self._as_float32_matrix([vector], expected_rows=1)
        return matrix[0]

    async def embed_in_batches(self, texts: list[str]) -> np.ndarray:
        # 批次规模的可直写来源只有 ``RunConfig``——它是 ``frozen dataclass``，
        # 字段没有校验器，而生产路径的唯一构造点 ``build_run_config`` 直接照搬
        # ``Settings.kb_embedding_batch_size``（那里有 1.._MAX_KB_CHUNK_TOKENS 的
        # 双向界）。因此嵌入方若自己拼一个 ``RunConfig``，这里的 ``max(1, ...)``
        # 只挡得住下界：实测 ``"32"``／``None`` 抛裸
        # ``TypeError: '>' not supported between instances of ...``，位数超限的
        # 整数抛 ``ValueError: Exceeds the limit``——两条都与「批次配错了」无关，
        # 而本文件的其余判据都按 ``ValueError`` 收敛（见 ``_as_float32_matrix``）。
        #
        # 上界必须用**本字段自己的**语义上界 ``_MAX_KB_CHUNK_TOKENS``。
        # 不能复用 ``require_scale_field``：它的上界是
        # ``_MAX_RETRIEVAL_CANDIDATES``（2000，检索候选数的上界），而
        # ``Settings.validate_positive_sizes`` 对这个字段批的界是
        # 1.._MAX_KB_CHUNK_TOKENS（1_000_000）。实测 ``kb_embedding_batch_size
        # = 2001`` 被 ``Settings`` 正常接受，却会被那个 helper 判为「超出合理
        # 范围」——把一个合法配置在运行时拒掉，是比原缺陷更严重的过度收紧。
        batch_size_value = self.run_config.kb_embedding_batch_size
        if isinstance(batch_size_value, bool) or not isinstance(batch_size_value, int):
            raise ValueError(
                f"kb_embedding_batch_size 必须是整数，但得到 {safe_repr(batch_size_value)}。"
            )
        if not 1 <= batch_size_value <= _MAX_KB_CHUNK_TOKENS:
            raise ValueError(
                f"kb_embedding_batch_size 必须是 1 到 {_MAX_KB_CHUNK_TOKENS} 之间的整数，"
                f"但得到 {safe_repr(batch_size_value)}。"
            )
        batch_size = batch_size_value
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(await self.embed_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches)

    def close(self) -> None:
        """释放按需创建的 Embedding Provider。"""
        model = self._embedding_model
        self._embedding_model = None
        if model is None:
            return
        close_resource_sync(model, "Embedding Provider")

    async def aclose(self) -> None:
        """异步释放按需创建的 Embedding Provider。"""
        model = self._embedding_model
        self._embedding_model = None
        if model is None:
            return
        close = getattr(model, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return
        close = getattr(model, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _probe_dimension(embeddings: Any) -> int | None:
        """在不分配矩阵的前提下取出维度，取不到时返回 ``None``。

        上界校验必须在 ``np.array(...)`` **之前**：走到那一步时内存已经按该维度
        分配过了。对 ``list`` 形态（本项目所有 embedding provider 的返回）看首行
        长度即可，实测对 10⁷ 元素的列表耗时 9 微秒；``ndarray`` 直接读 shape；
        其余形态交给后面的正常路径处理。
        """
        if isinstance(embeddings, np.ndarray):
            return int(embeddings.shape[1]) if embeddings.ndim == 2 else None
        if isinstance(embeddings, (list, tuple)):
            if not embeddings:
                return None
            first = embeddings[0]
            if isinstance(first, np.ndarray):
                return int(first.shape[0]) if first.ndim == 1 else None
            if isinstance(first, (list, tuple)):
                return len(first)
        return None

    @staticmethod
    def _as_float32_matrix(embeddings: Any, expected_rows: int | None = None) -> np.ndarray:
        # 维度上界必须先于分配检查：``local-hash`` 的维度直接来自用户可写的
        # ``model_name`` 数字后缀，实测配成 ``local-hash-100000000`` 时一次
        # ``embed_texts(["测试"])`` 就分配 (1, 10⁸) 的 float32 矩阵——耗时 180 秒、
        # 峰值内存 3.9 GB；按真实知识库 115 个分块算，``np.vstack`` 需要约 4.3 GB，
        # 进程会被 OOM 杀掉而不是得到一条可读的错误。与已经加过界的
        # ``chat_top_k × retrieval_candidate_multiplier`` 同一类问题。
        probed = EmbeddingService._probe_dimension(embeddings)
        if probed is not None and probed > _MAX_EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding 维度超过上界："
                f"得到 {probed}，上界 {_MAX_EMBEDDING_DIMENSION}。"
                " 请检查 embedding_configurations 里的 model_name"
                "（local-hash 的维度取自它的数字后缀）或 provider 的返回内容。"
            )
        # 超大整数必须在 ``np.array`` **之前**拦下：numpy 走 ``float()`` 转换，
        # ``10**400`` 抛裸 ``OverflowError``（它不是 ``ValueError``，穿透调用方
        # 全部 ``except ValueError`` 分类与日志脱敏），而同一批输入里的 ``inf``
        # 却能得到下面那条干净的 ``ValueError``——收场方式不一致。
        # 与 ``RetrievalService._validate_rerank_output`` 的守卫对称：那里同样
        # 收敛 provider 返回的超大整数。
        _reject_unconvertible_numbers(embeddings, "EmbeddingService 收到的向量")
        matrix = np.array(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("EmbeddingService 期望二维向量矩阵。")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("EmbeddingService 收到空向量矩阵。")
        if expected_rows is not None and matrix.shape[0] != expected_rows:
            raise ValueError(
                "EmbeddingService 收到的向量数量与输入文本数量不一致。"
                # ``expected_rows`` 是调用方给的期望条数，形态不受控；``matrix``
                # 的 shape 来自刚建好的 ndarray，必是小的 Python int。前者要经
                # ``safe_repr``：裸插值在位数超限时抛
                # ``ValueError: Exceeds the limit``，把「条数不一致」这条诊断
                # 换成整数转换错误。实测 ``expected_rows=10**4301`` 时复现
                # （``10**4000`` 不会，位数在上限内）。
                f" 输入 {safe_repr(expected_rows)} 条，返回 {matrix.shape[0]} 条。"
            )
        # 有限性是硬要求，不是洁癖：FAISS 会把 NaN 向量照单收下（ntotal 正常增加），
        # 直到查询时才以 3.4e38 的哨兵距离暴露出来，而那时「哪个文档坏了」已经无法
        # 定位；inf 同理。另外 ``np.array(..., dtype=np.float32)`` 对超出 float32
        # 范围的值只发 RuntimeWarning 并静默溢出成 inf，所以在 cast 之后检查。
        if not np.isfinite(matrix).all():
            raise ValueError(
                "EmbeddingService 收到的向量包含非有限值（NaN 或无穷大）；"
                "请检查 embedding provider 的返回内容。"
            )
        return matrix
