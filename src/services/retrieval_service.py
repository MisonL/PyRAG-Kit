from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from rich.console import Console

from src.providers.__base__.model_provider import RerankModel, close_resource_sync
from src.providers.factory import ModelProviderFactory
from src.retrieval.vdb.base import VectorStoreBase
from src.runtime.contracts import SessionConfig
from src.services.embedding_service import EmbeddingService
from src.ui.display_utils import coerce_score, document_content
from src.utils.config import (
    _MAX_RETRIEVAL_CANDIDATES,
    RetrievalMethod,
    is_double_convertible,
    parse_decimal_int,
    require_detail_attributes,
    require_provider_name,
    require_rerank_enabled,
    require_retrieval_method,
    require_scale_field,
    safe_repr,
    to_finite_float,
)
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)
DEFAULT_RRF_K = 60.0

# RRF 分母里 rank 的上界。``rrf_k + rank`` 是实际参与浮点运算的派生值，两者各自
# 都能表示成 double 而和不能，因此 rank 需要一个能让和落在 double 范围内的界。
# 检索结果数远小于它（``chat_top_k`` 上界 2000，候选倍数上界 100），因此这个界
# 只可能被畸形数据触及，不会限制任何真实场景。
_MAX_RRF_RANK = 10**9


class HybridReranker:
    def __init__(
        self,
        vector_weight: float,
        keyword_weight: float,
        fusion_strategy: str = "rrf",
        rrf_k: float = DEFAULT_RRF_K,
    ):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        # 先判类型再调 ``.strip()``：非字符串取值抛 ``AttributeError``，而本类
        # 的契约是 ``ValueError``（``SessionConfig._validate_fusion_strategy``
        # 与 ``RecursiveTextSplitter.structure_mode`` 都已按同一判据修过）。
        # 本类是公开边界，``SessionConfig`` 之外还有库调用方。
        if not isinstance(fusion_strategy, str):
            raise ValueError(f"不支持的混合检索融合策略: {safe_repr(fusion_strategy)}")
        self.fusion_strategy = fusion_strategy.strip().lower()
        # ``rrf_k`` 必须有下界：它是 RRF 分母 ``rrf_k + rank`` 的一部分，
        # ``rrf_k = -rank`` 时抛 ``ZeroDivisionError``（实测 ``rrf_k=-60`` 配
        # ``semantic_rank=60``）。非有限值与超大整数同样要拦——``rrf_k=nan``
        # 会让每个得分都是 ``nan``，而 ``nan`` 与分数阈值的比较恒为 False，
        # 整批结果被静默丢弃；``rrf_k=10**400`` 抛裸 ``OverflowError``（它不是
        # ``ValueError``，穿透全部 ``except ValueError`` 分类）。
        #
        # 上界取 ``is_double_convertible``：``rrf_k`` 只作为分母出现，任何能
        # 表示成 C double 的正数都是安全的。
        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, (int, float))
            or not is_double_convertible(rrf_k)
        ):
            raise ValueError(f"rrf_k 必须是有限数值，但得到 {safe_repr(rrf_k)}。")
        if float(rrf_k) <= 0:
            raise ValueError(
                f"rrf_k 必须是正数，但得到 {safe_repr(rrf_k)}。"
                " 它作为 RRF 的分母 ``rrf_k + rank``，非正数会导致除零或负权重。"
            )
        self.rrf_k = float(rrf_k)
        if self.fusion_strategy not in {"rrf", "weighted"}:
            raise ValueError(f"不支持的混合检索融合策略: {safe_repr(fusion_strategy)}")

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]
        score_range = max_score - min_score
        return [max(0.0, min(1.0, (score - min_score) / score_range)) for score in scores]

    def rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 入参形态与构造器／``_weight_tuple`` 同判据：本类的 docstring 明确
        # 「库调用方可能绕过 ``Settings`` 与 ``SessionConfig`` 直接构造」，那两个
        # 入口都为此收敛了，唯独入参这一半漏了。两个内部分支都直接
        # ``for doc in documents`` 再对 ``doc.get(...)`` 取值，非列表取值的收场
        # 不在契约内：``5``／``None``／``object()`` 抛裸
        # ``TypeError: ... is not iterable``；``"abc"``／``{"a": 1}`` 抛
        # ``AttributeError: 'str' object has no attribute 'get'``（逐字符／逐键
        # 迭代）。两者都不是 ``ValueError``，穿透调用方按 ``ValueError`` 建立的
        # 分类。判据与生产路径共用 ``_require_document_list``。
        documents = _require_document_list(documents, "混合检索融合")
        if self.fusion_strategy == "weighted":
            return self._weighted_rerank(documents)
        return self._rrf_rerank(documents)

    def _weight_tuple(self) -> tuple[float, float]:
        """返回归一化后的 (向量权重, 关键词权重)。

        两个权重同时为零时显式报错而不是返回 ``(0.0, 0.0)``：后者会让每个
        文档得分都是 0.0，再被分数阈值全部过滤，表现为「检索无结果」而非
        配置错误。这是所有构造路径的最后一道兜底（``Settings`` 与
        ``SessionConfig`` 各自也有校验，但库调用方可能绕过两者直接构造）。
        """
        # 判据用 ``is_double_convertible`` 而不是 ``math.isfinite``：后者对超出
        # C double 的整数**自己就抛** ``OverflowError``（实测 ``math.isfinite(10**400)``
        # 抛错），拿它当守卫等于没有守卫。这是本方法作为「所有构造路径的最后一道
        # 兜底」必须自己守住的——库调用方可以绕过 ``Settings`` 与 ``SessionConfig``
        # 直接构造 ``HybridReranker``。
        if not is_double_convertible(self.vector_weight) or not is_double_convertible(
            self.keyword_weight
        ):
            raise ValueError(
                "混合检索的 vector_weight 与 keyword_weight 必须是和为正的有限数值，"
                f"但得到 vector_weight={safe_repr(self.vector_weight)}、"
                f"keyword_weight={safe_repr(self.keyword_weight)}。"
            )
        # ``total`` 是**派生值**，必须自己判一次可转换性：两个权重各自都在
        # double 范围内（上面那两处 ``is_double_convertible`` 放行），但它们的
        # **和**可以超出——实测 ``HybridReranker(10**308, 10**308)`` 构造成功，
        # 随后任意一次 ``rerank()`` 在 ``math.isfinite(total)`` 上抛
        # ``OverflowError``（``int`` 太大转不成 float）。该异常是
        # ``ArithmeticError`` 子类而**不是** ``ValueError``，穿透本类调用方的
        # 全部 ``except ValueError`` 分类，退化为裸 traceback。
        # 用 ``float()`` 相加而不是直接相加：两个 int 相加得到的是 int，其值
        # 可以远超 double 范围，下一行的 ``math.isfinite`` 随即抛
        # ``OverflowError``。先转 float 则溢出表现为 ``inf``，被 ``isfinite``
        # 正常判为非法，收场是带指引的 ``ValueError``。
        total = float(self.vector_weight) + float(self.keyword_weight)
        if not math.isfinite(total) or total <= 0:
            # ``nan`` 也走这一支：``nan <= 0`` 为 False，不显式判有限性就会
            # 漏过去，随后每个文档得分都是 ``nan``——排序结果不确定，且
            # ``nan`` 与分数阈值的比较恒为 False，整批结果被静默丢弃。
            raise ValueError(
                "混合检索的 vector_weight 与 keyword_weight 必须是和为正的有限数值，"
                f"但得到 vector_weight={safe_repr(self.vector_weight)}、"
                f"keyword_weight={safe_repr(self.keyword_weight)}。"
                " 两个权重同时为 0（或含非有限值）会让所有文档得分都是 0 或 nan，"
                "再被分数阈值过滤，检索会静默返回空结果。"
            )
        return self.vector_weight / total, self.keyword_weight / total

    def _weighted_rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        keyword_scores = [coerce_score(doc.get("keyword_score")) for doc in scored_documents]
        semantic_scores = [coerce_score(doc.get("semantic_score")) for doc in scored_documents]
        normalized_keyword = self._normalize_scores(keyword_scores)
        normalized_semantic = self._normalize_scores(semantic_scores)
        for index, doc in enumerate(scored_documents):
            doc["score"] = (
                vector_weight * normalized_semantic[index]
                + keyword_weight * normalized_keyword[index]
            )
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)

    def _rrf_rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        for doc in scored_documents:
            # rank 来自 store 返回的文档（``semantic_rank`` 由本文件写入，但
            # ``keyword_rank`` 可以是第三方 store 原样带进来的值），因此形态
            # 不受控。``int()`` 会抛裸 ``ValueError``（非十进制字符或位数超限），
            # 而 ``rrf_k + rank`` 是**派生值**：两个数各自都能表示成 double，
            # 其和不能——实测 ``semantic_rank = 10**400`` 让该分母抛
            # ``OverflowError``，穿透调用方的全部 ``except ValueError`` 分类。
            # 非正或超范围的 rank 视为「未参与该路检索」，与 0 同一处理。
            semantic_rank = _coerce_rank(doc.get("semantic_rank"))
            keyword_rank = _coerce_rank(doc.get("keyword_rank"))
            semantic_rrf = 1.0 / (self.rrf_k + semantic_rank) if semantic_rank > 0 else 0.0
            keyword_rrf = 1.0 / (self.rrf_k + keyword_rank) if keyword_rank > 0 else 0.0
            doc["score"] = vector_weight * semantic_rrf + keyword_weight * keyword_rrf
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)


def _is_hashable(value: Any) -> bool:
    """判断取值能否作为 ``cache_key`` 元组的一项。

    ``cache_key`` 会喂给 ``self._rerank_providers``（普通 ``dict``），元组里每
    一项都要可哈希，否则在 ``dict.get`` 处抛裸 ``TypeError: unhashable type:
    'list'``——它不是 ``ValueError``，穿透调用方的错误分类。

    判据用**可哈希**而不是 ``isinstance(..., str)``：这些字段只是用来区分缓存
    条目，非字符串但可哈希的取值（整数、元组）本来就能正常工作，按类型收窄会
    误伤它们；位数超限的整数是可哈希的，不必在此拦截。
    """
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _stable_fingerprint(value: Any) -> str:
    """把任意值渲染成**能区分不同取值**的稳定字符串，且绝不抛异常。

    ``repr()`` 对含超大整数的容器抛 ``ValueError``；``safe_repr`` 虽不抛，但对
    容器只退回 ``<dict>``／``<list>``，丢掉了区分度——用它做缓存键会让两份不同的
    ``options`` 撞到同一个 provider 实例。这里递归到叶子：整数走 ``safe_repr``
    的量级描述，其余保持 ``repr`` 的形态。

    dict／list／tuple／set／frozenset 都会**递归**（集合此前落到 ``repr()``
    兜底，含超大整数时整个塌成 ``<set>``），并沿当前路径记录容器 ``id`` 以截断
    自引用环——环上返回 ``<cycle>`` 而不是无限递归。

    非环的**深**嵌套仍受解释器递归深度限制（约 500 层），与修改前的行为一致；
    ``tomllib`` 在约 900 层解析期就先抛 ``RecursionError``，因此配置来源到不了
    那个深度。
    """
    return _fingerprint(value, set())


def _fingerprint(value: Any, seen: set[int]) -> str:
    """``_stable_fingerprint`` 的递归实现，``seen`` 记录当前路径上的容器身份。

    用「路径上的 id 集合」而不是全局去重：同一个对象在**兄弟**位置出现两次是
    正常取值（``{"a": shared, "b": shared}``），只有回到**祖先**才是环。
    """
    if isinstance(value, dict):
        if id(value) in seen:
            return "<cycle>"
        seen.add(id(value))
        try:
            return (
                "{"
                + ", ".join(
                    f"{_fingerprint(k, seen)}: {_fingerprint(v, seen)}" for k, v in value.items()
                )
                + "}"
            )
        finally:
            seen.discard(id(value))
    # 集合也要**递归**，不能落到 ``repr()`` 兜底：集合含位数超限的整数时
    # ``repr()`` 抛 ``ValueError``，整个集合会塌成 ``<set>``，区分度归零。
    # 实测 ``options={"x": {10**4301}}`` 与 ``{"x": {10**4302}}`` 都通过
    # ``validate_secret_free_options``，塌陷后两者得到同一个缓存键——两份不同的
    # rerank 配置静默复用同一个 provider，比崩溃更糟。
    #
    # 排序取稳定顺序：集合本身无序，``repr`` 的顺序还受哈希随机化影响，不排序
    # 会让**同一个**取值在两次调用间产生不同的指纹（缓存永远命中不了）。
    # 元素用各自渲染结果的 ``repr`` 排序，避免直接比较不可比较的元素类型。
    if isinstance(value, (set, frozenset)):
        if id(value) in seen:
            return "<cycle>"
        seen.add(id(value))
        try:
            items = sorted(_fingerprint(item, seen) for item in value)
            prefix = "set" if isinstance(value, set) else "frozenset"
            return f"{prefix}({{" + ", ".join(items) + "})"
        finally:
            seen.discard(id(value))
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return "<cycle>"
        seen.add(id(value))
        try:
            return "[" + ", ".join(_fingerprint(item, seen) for item in value) + "]"
        finally:
            seen.discard(id(value))
    try:
        return repr(value)
    except (ValueError, RuntimeError, RecursionError):
        # 只有位数超限的整数会走到这里。``safe_repr`` 的量级描述（「约 N 位
        # 十进制」）对**同位长**的不同整数是同一个字符串，做缓存键会碰撞；
        # 补上 ``bit_length`` 与 ``hash`` 使取值可区分。``hash`` 对 int 不经
        # 十进制转换，不会抛；缓存本身是进程内的，不要求跨进程稳定。
        if isinstance(value, int):
            return f"<int:{value.bit_length()}:{hash(value) & 0xFFFFFFFF:08x}>"
        return f"<{type(value).__name__}>"


def _coerce_threshold(value: Any) -> float:
    """把分数阈值收敛成有限浮点，非法形态给出可读 ``ValueError``。

    ``retrieve`` 直接算 ``coerce_score(...) >= session_config.score_threshold``，
    而该字段可被直写（见 ``require_scale_field``）。非数值形态抛裸
    ``TypeError: '>=' not supported between instances of 'float' and 'NoneType'``
    ——落在 ``retrieve`` 既有的错误分类之外。``bool`` 显式拒绝：``True`` 会静默
    变成 ``1.0``，看起来像一个真实阈值。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"score_threshold 必须是数值，但得到 {safe_repr(value)}。")
    number = to_finite_float(value)
    if number is None:
        raise ValueError(f"score_threshold 必须是有限数值，但得到 {safe_repr(value)}。")
    return number


def _coerce_rerank_enabled(value: Any) -> bool:
    """确认 ``rerank_enabled`` 是真正的布尔值。

    原先走 ``if not session_config.rerank_enabled``，真值语义会把 ``None``／``0``／
    ``[]`` 静默当成「禁用」、把 ``"no"`` 当成「启用」——用户看到的是 rerank 行为
    与配置不符，而不是一条配置错误。``SessionConfig`` 的校验器要求 ``bool``，
    这里对直写路径保持同一判据。

    判据与措辞都交给 ``require_rerank_enabled``：UI 的 5 个读取点此前用真值判断
    （直写 ``"false"`` 时配置表显示「启用」），现在与这里共用同一实现，读出来的
    收场与写进去的拒绝一致。
    """
    return require_rerank_enabled(value)


def _coerce_fusion_strategy(value: Any) -> str:
    """把融合策略收敛成规范形态，非法取值给出带字段名的 ``ValueError``。

    归一化（``.strip().lower()``）必须发生在这里，**且只发生在这里**：
    ``HybridReranker`` 的构造器对同一字段做 ``fusion_strategy.strip().lower()``，
    而 ``_should_apply_score_threshold`` 此前用**原始**取值与字面量 ``"rrf"``
    比较。于是 ``"RRF"``／``" rrf "`` 走的是 RRF 融合（分数是排名倒数之和），
    却**不**被识别为 RRF，阈值过滤照常套用——两种量纲混在一起，结果被静默
    清空。同一个字段在两处的判据必须一致。

    与 ``SessionConfig._validate_fusion_strategy`` 的关系：那里只接受已规范化
    的 ``"rrf"``／``"weighted"``，是配置边界的严格判据；这里是运行期把可直写
    的字段收敛到同一形态，宽容但不静默——无法识别成任一策略时仍抛错，由
    ``HybridReranker`` 给出「不支持的融合策略」。
    """
    if not isinstance(value, str):
        raise ValueError(f"hybrid_fusion_strategy 必须是字符串，但得到 {safe_repr(value)}。")
    return value.strip().lower()


def _coerce_rank(value: Any) -> int:
    """把文档里的 rank 收敛成非负整数，非法形态一律视为 0（未参与该路检索）。

    rank 来自 store 返回的文档，形态不受控。``int(value)`` 有两类失败：非十进制
    数字字符抛 ``ValueError``，位数超过 ``sys.get_int_max_str_digits()`` 同样抛
    ``ValueError``。而更大的问题在**下游的派生值**——``rrf_k + rank`` 对超过
    double 范围的 rank 抛 ``OverflowError``（``ArithmeticError`` 子类，不是
    ``ValueError``），穿透调用方的错误分类。这里把非法值收敛成 0：它的语义就是
    「该文档没有在这条检索路径上出现」，与 rank 缺失时一致，不会改变排序结果。
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return 0
    if isinstance(value, str):
        parsed = parse_decimal_int(value)
        if parsed is None:
            return 0
        value = parsed
    if not 0 <= value <= _MAX_RRF_RANK:
        return 0
    return value


def _document_metadata(document: Any) -> dict[str, Any]:
    """取出文档元数据，非映射时退回空 dict。

    ``document.get("metadata") or {}`` 只挡得住 ``None`` 与空值：**非空的非映射**
    会原样留下，随后 ``_deduplicate_parent_documents`` 里的 ``.get("parent_id")``
    抛 ``AttributeError: 'list' object has no attribute 'get'``——而那是检索管线
    内部，报错现场离「store 返回了什么」很远。

    这里退回空 dict 而不是报错：检索结果可以来自任何 ``VectorStoreBase`` 实现，
    元数据缺失/异常时仍能按内容检索出结果，比整批失败更有用；形态**类型**上的
    契约（必须是文档列表）由 ``_require_document_list`` 把关。
    """
    if not isinstance(document, dict):
        return {}
    metadata = document.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _require_document_list(results: Any, stage: str) -> list[dict[str, Any]]:
    """确认检索阶段返回的是「文档映射的列表」。

    ``RetrievalService`` 是三个消费者（聊天 prompt、召回测试表格、聊天详情
    表格）的共同上游，而检索结果可以来自任何 ``VectorStoreBase`` 实现——抽象
    基类正是为替换实现而存在的。实测返回 ``None`` 或 ``list[str]`` 的 store 会
    在这里抛 ``TypeError: 'NoneType' object is not iterable`` 或
    ``AttributeError: 'str' object has no attribute 'get'``，报错现场离
    「你的 store 返回了什么」很远；而在下游（``_promote_parent_context`` 等）
    才炸的话，位置更偏。

    这里统一转成可读的 ``TypeError``：这是**接口契约**问题，不是数据内容问题，
    因此不静默丢弃不合规的条目——丢弃会让「store 实现错了」表现为「检索结果
    变少了」。
    """
    if not isinstance(results, list):
        raise TypeError(
            f"{stage}返回的必须是文档列表，但得到 {type(results).__name__}。"
            " 请检查 VectorStoreBase 实现是否遵守 semantic_search/keyword_search 的契约。"
        )
    for position, document in enumerate(results):
        if not isinstance(document, dict):
            raise TypeError(
                f"{stage}返回的第 {position} 条不是文档映射"
                f"（{type(document).__name__}）。"
                " 请检查 VectorStoreBase 实现是否遵守契约。"
            )
    return results


def _unpack_rerank_result(result: Any) -> tuple[Any, Any]:
    """把 rerank provider 的返回值解包成 ``(索引, 分数)`` 二元组。

    不能直接写 ``a, b = await provider.arerank(...)``：``None``／单值抛裸
    ``TypeError``、三元组抛裸 ``ValueError``，而这两条都在
    ``_validate_rerank_output``（正是「校验 provider 返回值」的边界）**之前**，
    用户拿到的是「无法解包」而不是「provider 返回值不合法」——错误消息指向
    解包语法而非真正的契约违反。
    """
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            f"Rerank provider 必须返回 (索引列表, 分数列表) 二元组，但得到 {safe_repr(result)}。"
        )
    return result[0], result[1]


def _key_text(value: Any) -> str:
    """把元数据值收敛成去重键的一部分。

    不能直接写 ``str(value)``：元数据来自知识快照，而本项目加载 pickle 格式的
    快照——pickle 是二进制格式、不受 CPython 十进制位数限制，一个 4301 位的
    整数让 ``str()`` 抛 ``ValueError: Exceeds the limit``，于是「计算文档去重
    键」这一步在正要合并检索结果时崩掉。

    但 ``safe_repr`` 的降级描述**只保留量级**（``<整数，约 4302 位十进制>``），
    用它当键会让不同的超限整数塌成同一个字符串：实测 ``10**4301`` 与
    ``10**4301 + 1`` 得到同一个键，于是 ``_merge_hybrid_results`` 把两条不同的
    文档并成一条，其中一条静默消失。位数超限的整数**无法**转成十进制文本，
    但 ``int.to_bytes`` 不经过十进制转换，可以对它取摘要——键因此既稳定又唯一。

    非整数的失败形态（``repr`` 抛其它异常）退回 ``safe_repr``：那时没有更好的
    取值，而量级描述至少保持稳定。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        try:
            repr(value)
        except (ValueError, RuntimeError, RecursionError):
            # 位数超过 ``sys.get_int_max_str_digits()``。按二进制补码取摘要，
            # 符号一并编进去，避免 ``n`` 与 ``-n`` 碰撞。
            raw = value.to_bytes((value.bit_length() + 8) // 8, "big", signed=True)
            digest = hashlib.sha256(raw, usedforsecurity=False).hexdigest()
            return f"<整数，{digest}>"
    return safe_repr(value)


def _document_key(document: dict[str, Any]) -> str:
    """返回用于混合合并/去重的文档键。

    优先用 ``chunk_id``／``doc_id``，其次 ``source`` + ``chunk_index``。两者都
    缺失时**不能只截取内容前缀**：截断会让两份不同的长文档拿到同一个键——
    实测同来源、前 64 字符相同的两篇文档在 ``_merge_hybrid_results`` 里被并成
    一条，其中一篇的结果静默消失（``_deduplicate_parent_documents`` 同理，只是
    那里丢的是得分较低的那篇）。因此这里改用整段内容的摘要：既保持定长，又
    不因截断而碰撞。

    ``chunk_id`` 在真实快照里恒存在（分片阶段必然写入），所以这一支是兜底
    路径；但兜底路径同样不能产生错误结果。
    """
    metadata = _document_metadata(document)
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id:
        return _key_text(chunk_id)
    source = _key_text(metadata.get("source", ""))
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{_key_text(chunk_index)}"
    content = _key_text(document.get("page_content", ""))
    digest = hashlib.sha256(content.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{source}:{digest}"


def _merge_hybrid_results(
    semantic_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # 两个入参的形态判据与生产路径共用 ``_require_document_list``：本函数经
    # ``src/retrieval/retriever.py`` 的 ``__all__`` 对外导出，是公开边界。
    # ``retrieve`` 内部调用它之前已校验过两路结果，但库调用方直接调用这条路径
    # 没有——函数体第一件事就是 ``for rank, document in enumerate(...)`` 再对
    # ``document.get(...)`` 取值，非列表取值的收场不在契约内：``5``／``None``／
    # ``object()`` 抛裸 ``TypeError: ... is not iterable``；``"abc"``／
    # ``{"a": 1}`` 抛 ``AttributeError: 'str' object has no attribute 'get'``
    # （逐字符／逐键迭代）。两者都不是 ``ValueError``。
    semantic_results = _require_document_list(semantic_results, "语义检索")
    keyword_results = _require_document_list(keyword_results, "关键词检索")
    merged: dict[str, dict[str, Any]] = {}
    for rank, document in enumerate(semantic_results, start=1):
        merged_doc = copy.deepcopy(document)
        merged_doc["semantic_score"] = coerce_score(merged_doc.get("score"))
        merged_doc["semantic_rank"] = rank
        merged_doc["keyword_score"] = coerce_score(merged_doc.get("keyword_score"))
        # ``score`` 自身也要收敛：融合路径随后会由 ``HybridReranker`` 重算它，
        # 但把 ``nan``／``inf``／``dict`` 留在名为 ``score`` 的字段里，等于让
        # 一个中间产物带着不可用的值往下游走——任何读到它却没过 rerank 的调用方
        # （或将来调整了顺序的流水线）都会拿到坏分数。
        merged_doc["score"] = coerce_score(merged_doc.get("score"))
        merged[_document_key(merged_doc)] = merged_doc
    for rank, document in enumerate(keyword_results, start=1):
        key = _document_key(document)
        if key in merged:
            merged[key]["keyword_score"] = coerce_score(document.get("score"))
            merged[key]["keyword_rank"] = rank
        else:
            merged_doc = copy.deepcopy(document)
            merged_doc["semantic_score"] = coerce_score(merged_doc.get("semantic_score"))
            merged_doc["keyword_score"] = coerce_score(document.get("score"))
            merged_doc["score"] = coerce_score(merged_doc.get("score"))
            merged_doc["keyword_rank"] = rank
            merged[key] = merged_doc
    return list(merged.values())


def _promote_parent_context(
    documents: list[dict[str, Any]], resolver: Callable[[str | None], str | None] | None
) -> list[dict[str, Any]]:
    promoted_documents: list[dict[str, Any]] = []
    for document in documents:
        promoted_doc = copy.deepcopy(document)
        metadata = _document_metadata(promoted_doc)
        parent_content = resolver(metadata.get("parent_id")) if resolver else None
        if parent_content:
            # 收敛成字符串再写入：``page_content: str`` 是 ``Document`` 的契约
            # （``src/models/document.py``），``FaissStore._validate_document_shape``
            # 在快照载入期也按这条校验，而这里此前只判真值、不判类型。
            #
            # ``parent_resolver`` 取自 ``getattr(self.vector_store,
            # "resolve_parent_content", None)``，而 ``VectorStoreBase`` 正是为替换
            # 实现而存在的——签名声明 ``-> str | None``，第三方实现返回别的形态不会
            # 被任何地方拦下。实测返回列表让 ``page_content`` 变成 ``['a', 'b']``、
            # 返回整数变成 ``123``：展示层 ``document_content`` 把它收敛成 ``"N/A"``
            # （用户看到「无内容」而不是报错），而 prompt 构造对正文做 ``.lower()``
            # 时抛 ``AttributeError``，检索整体不可用。
            #
            # 用 ``_key_text`` 而不是裸 ``str()``：位数超限的整数（pickle 是二进制
            # 格式、不受十进制位数限制，畸形快照能带进来）会让 ``str()`` 抛
            # ``ValueError: Exceeds the limit``，而本函数的职责是「把父块内容放进
            # 正文」，不该在正要成功时崩。``_key_text`` 对这类值给量级描述。
            parent_text = _key_text(parent_content)
            if parent_text:
                metadata.setdefault("matched_chunk_content", promoted_doc.get("page_content", ""))
                promoted_doc["page_content"] = parent_text
        promoted_doc["metadata"] = metadata
        promoted_documents.append(promoted_doc)
    return promoted_documents


def _deduplicate_parent_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = _document_metadata(document)
        parent_key = metadata.get("parent_id")
        key = _key_text(parent_key) if parent_key else _document_key(document)
        existing = deduplicated.get(key)
        if existing is None or coerce_score(document.get("score")) > coerce_score(
            existing.get("score")
        ):
            deduplicated[key] = copy.deepcopy(document)
    return list(deduplicated.values())


class RetrievalService:
    def __init__(self, vector_store: VectorStoreBase, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._rerank_providers: dict[tuple[str, str, str, str], RerankModel] = {}

    async def retrieve(
        self, query: str, session_config: SessionConfig, console: Console | None = None
    ) -> list[dict[str, Any]]:
        # 检索模式同样要收敛：它是本函数里唯一没有对应助手的字段，而
        # ``==`` 判分支**不会**对未知取值报错——``"hybrid"``／``None``／``5``
        # 全部静默落进 else 分支，变成纯关键词检索（实测）。详见该助手的说明。
        retrieval_method = require_retrieval_method(session_config.retrieval_method)
        # 形态必须在这里收敛：``SessionConfig`` 的属性可被直写，而 ``max()`` 对
        # 非数值抛裸 ``TypeError``（详见 ``require_scale_field``）。
        top_k = require_scale_field(session_config.top_k, "top_k")
        multiplier = require_scale_field(
            session_config.retrieval_candidate_multiplier, "retrieval_candidate_multiplier"
        )
        # 乘积同样要判：两个字段各自合法、乘积越界是既有缺陷类（``Settings`` 与
        # ``__setitem__`` 都按组合判），这里兜住直写路径。
        if top_k * multiplier > _MAX_RETRIEVAL_CANDIDATES:
            raise ValueError(
                "top_k 与 retrieval_candidate_multiplier 的乘积超出合理范围："
                f" {top_k} × {multiplier} > {_MAX_RETRIEVAL_CANDIDATES}。"
                " 请检查这两项配置。"
            )
        effective_top_k = top_k * multiplier
        # 同一类前提的其余字段：都可被直写，读取时统一收敛（详见各 helper）。
        score_threshold = _coerce_threshold(session_config.score_threshold)
        rerank_enabled = _coerce_rerank_enabled(session_config.rerank_enabled)
        fusion_strategy = _coerce_fusion_strategy(session_config.hybrid_fusion_strategy)
        parent_resolver = getattr(self.vector_store, "resolve_parent_content", None)

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
            semantic_results, keyword_results = await self._gather_hybrid_results(
                query, effective_top_k
            )
            # 先校验两路结果的形态，再合并。``_require_document_list`` 原先只在
            # 合并与 rerank **之后**运行，而 ``_merge_hybrid_results`` 已经对每个
            # 条目调了 ``.get(...)``——store 返回 ``[None]`` 时抛的是
            # ``AttributeError: 'NoneType' object has no attribute 'get'``，
            # 而不是该校验承诺的可定位 ``TypeError``（附带是哪一路出的问题）。
            # 这也让「坏结果」的收场方式在三条检索路径之间保持一致。
            _require_document_list(semantic_results, "语义检索")
            _require_document_list(keyword_results, "关键词检索")
            reranker = HybridReranker(
                session_config.vector_weight,
                session_config.keyword_weight,
                fusion_strategy=fusion_strategy,
            )
            ranked_results = reranker.rerank(
                _merge_hybrid_results(semantic_results, keyword_results)
            )
        elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
            ranked_results = await self._semantic_retrieve(query, effective_top_k)
        else:
            ranked_results = await self._keyword_retrieve(query, effective_top_k)

        ranked_results = _require_document_list(ranked_results, "检索")

        if self._should_apply_score_threshold(session_config, fusion_strategy=fusion_strategy):
            ranked_results = [
                doc for doc in ranked_results if coerce_score(doc.get("score")) >= score_threshold
            ]

        ranked_results = _promote_parent_context(ranked_results, parent_resolver)
        ranked_results = _deduplicate_parent_documents(ranked_results)
        ranked_results = await self._rerank_if_needed(
            query, ranked_results, session_config, rerank_enabled=rerank_enabled, top_k=top_k
        )
        # 出口统一把 ``score`` 收敛成有限浮点。三条检索路径的来源不同：
        # ``FaissStore`` 自己写的是 float，第三方 store 可能是数字字符串或
        # 其它形态；融合路径会重算 score，而**语义／关键词单路不会**。不收敛的话
        # 同一份结果会出现「阈值按 0.8 通过、界面显示 0.0000」，而排序、去重与
        # 展示又各自按不同规则解释它。这里让出口的契约只有一种形态。
        for document in ranked_results:
            document["score"] = coerce_score(document.get("score"))
        # 用**已收敛**的 ``top_k`` 而不是重读原属性：``retrieve`` 开头就把
        # ``session_config.top_k`` 收敛进了局部变量，这里重读等于绕过那道校验。
        # 该字段可被直写，而 ``max()`` 对非数值抛裸 ``TypeError``——实测并发场景下
        # （``retrieve`` 是 async，中间有多处 ``await``）在检索飞行中把 ``top_k``
        # 直写成 ``"abc"``，本行抛
        # ``TypeError: '>' not supported between instances of 'str' and 'int'``，
        # 穿透调用方的 ``except ValueError`` 分类。
        return ranked_results[: max(1, top_k)]

    async def _gather_hybrid_results(
        self, query: str, top_k: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        semantic_results, keyword_results = await asyncio.gather(
            self._semantic_retrieve(query, top_k),
            self._keyword_retrieve(query, top_k),
        )
        return semantic_results, keyword_results

    async def _semantic_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        # Embedding failures must propagate. Only use the legacy path when the
        # vector store truly does not expose the newer semantic_search method;
        # an AttributeError/NotImplementedError raised inside a real provider is
        # a defect or an upstream failure, not a compatibility signal.
        if self._get_vector_store_method("semantic_search") is None:
            return await self._legacy_search(query, top_k, "semantic")
        query_embedding = await self.embedding_service.embed_query(query)
        return await self._call_vector_store("semantic_search", query_embedding, top_k)

    async def _keyword_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._get_vector_store_method("keyword_search") is None:
            return await self._legacy_search(query, top_k, "keyword")
        return await self._call_vector_store("keyword_search", query, top_k)

    def _get_vector_store_method(self, method_name: str) -> Callable[..., Any] | None:
        """Return an implemented vector-store method, excluding base stubs."""
        method = getattr(self.vector_store, method_name, None)
        if not callable(method):
            return None
        implementation = getattr(type(self.vector_store), method_name, None)
        base_implementation = getattr(VectorStoreBase, method_name, None)
        if implementation is base_implementation:
            return None
        return method

    async def _call_vector_store(self, method_name: str, *args) -> list[dict[str, Any]]:
        method = self._get_vector_store_method(method_name)
        if method is None:
            raise NotImplementedError(f"向量存储未实现 {method_name} 接口。")
        return await asyncio.to_thread(method, *args)

    async def _legacy_search(
        self, query: str, top_k: int, search_type: str
    ) -> list[dict[str, Any]]:
        async_search = getattr(self.vector_store, "asearch", None)
        if callable(async_search):
            return await async_search(query, top_k, search_type=search_type)
        sync_search = getattr(self.vector_store, "search", None)
        if not callable(sync_search):
            raise NotImplementedError("向量存储未实现兼容 search 接口。")
        return await asyncio.to_thread(sync_search, query, top_k, search_type)

    def _get_rerank_provider(
        self,
        provider_key: str,
        configurations: Mapping[str, Any] | None = None,
    ) -> RerankModel:
        # key 形态必须在 ``.get()`` **之前**收敛：``provider_key`` 来自可直写的
        # ``session_config.active_rerank_configuration``，不可哈希的取值在
        # ``dict.get`` 处就抛裸 ``TypeError: unhashable type: 'list'``——早于
        # ``ModelProviderFactory`` 的守卫，工厂那一侧的修复覆盖不到这条路径。
        # 位数超限的整数在下面构造 ``cache_key`` 的元组里同样要过哈希，但那种
        # 情况 ``hash`` 不抛，真正的失败在下游消息插值处，一并在这里拦掉。
        if not isinstance(provider_key, str):
            raise ValueError(
                "active_rerank_configuration 必须是字符串，"
                f"但得到 {safe_repr(provider_key)}。 请检查 rerank 配置项。"
            )
        # ``configurations`` 的形态同样要判：它来自可直写的
        # ``session_config.rerank_configurations``，非映射时 ``.get`` 抛裸
        # ``AttributeError: 'int' object has no attribute 'get'``。
        if configurations is not None and not isinstance(configurations, Mapping):
            raise ValueError(
                "rerank_configurations 必须是配置映射，"
                f"但得到 {type(configurations).__name__}。 请检查 rerank 配置项。"
            )
        # 配置**取值**的形态同样要判，三个失败面：
        #   * detail 不提供 ``provider``／``model_name``（``None``／dict／int，
        #     或属性被 ``del`` 掉）——``ModelProviderFactory.get_rerank_provider``
        #     里的 ``config.provider`` 抛裸 ``AttributeError``；
        #   * ``provider`` 非可哈希——它进 ``cache_key`` 元组，在
        #     ``self._rerank_providers.get(...)`` 处抛裸
        #     ``TypeError: unhashable type: 'list'``；
        #   * ``model_name`` 非可哈希——同上。
        # 来源是 ``session_config.rerank_configurations`` 的值，可直写。
        #
        # 判据用 ``provider_key in configurations`` 而**不是** ``detail is not None``：
        # 「key 不存在」与「key 存在但取值是 ``None``」是两件事。后者在工厂里走
        # ``configurations[key]`` 拿到 ``None``（键确实在），于是不报「未找到 key」，
        # 直接在 ``config.provider`` 上抛裸 ``AttributeError``。只有按成员判定才能
        # 把两者分开：前者保持原有「未找到 key」的 ``ValueError``，后者是形态错误。
        detail: Any = None
        provider_field = ""
        if configurations is not None and provider_key in configurations:
            detail = configurations[provider_key]
            # 属性**存在性**先判：``getattr`` 的默认值只挡缺失、不挡类型，
            # ``del detail.provider`` 兜出的空串会把「属性没了」伪装成
            # 「provider 名为空」，一路带到工厂再抛 ``AttributeError``。
            require_detail_attributes(
                detail,
                ("provider", "model_name"),
                field="rerank_configurations",
                key=provider_key,
                role_label="Rerank",
            )
            provider_field = require_provider_name(
                detail.provider, label=f"rerank 配置 {provider_key} 的 provider"
            )
            # ``model_name`` 同样进 ``cache_key`` 元组（``detail.model_name =
            # ["m"]`` 实测直接成功，会在这里抛裸 ``TypeError: unhashable type``）。
            model_name = detail.model_name
            if not _is_hashable(model_name):
                raise ValueError(
                    f"rerank 配置 {provider_key} 的 model_name 必须是可哈希的取值，"
                    f"但得到 {safe_repr(model_name)}。"
                )
        options = getattr(detail, "options", {}) if detail is not None else {}
        try:
            options_fingerprint = json.dumps(
                options or {}, sort_keys=True, default=str, separators=(",", ":")
            )
        except (TypeError, ValueError):
            # 兜底也必须用 ``safe_repr``：``repr(options)`` 抛的是**同一个**
            # ``ValueError``（位数超限），于是这里成了「正要兜底时再抛一次」。
            # 可达性已确认：``ModelDetail.options`` 是 ``dict[str, Any]``，
            # ``validate_secret_free_options`` 只查敏感键名、不做递归数值校验，
            # 实测 ``ModelDetail(..., options={"x": 10**4301})`` 被接受，随后
            # ``rerank_enabled=True`` 时每次检索都在构造缓存键处崩。
            #
            # ``json.dumps(default=str)`` 拦不住这类值：``default=`` 只对
            # **未知类型**生效，``int`` 是已知类型，直接走 ``int.__repr__``。
            #
            # 这里不能直接用 ``safe_repr(options)``：它对渲染失败的**容器**只
            # 退回 ``<dict>``，那会让所有含超大整数的 options 得到同一个缓存键，
            # 不同配置静默复用同一个 provider 实例——比崩溃更糟。逐元素渲染既
            # 保住区分度，又不抛。
            options_fingerprint = _stable_fingerprint(options)
        cache_key = (
            provider_key,
            provider_field,
            getattr(detail, "model_name", ""),
            options_fingerprint,
        )
        provider = self._rerank_providers.get(cache_key)
        if provider is None:
            # A configuration reload can keep the same key while changing the
            # endpoint, model or options. Drop the old instance before creating
            # its replacement so stale transports and credentials are not kept
            # alive indefinitely.
            stale_keys = [
                key for key in self._rerank_providers if key[0] == provider_key and key != cache_key
            ]
            for stale_key in stale_keys:
                stale_provider = self._rerank_providers.pop(stale_key)
                close_resource_sync(stale_provider, "Rerank Provider")
            if configurations is None:
                provider = ModelProviderFactory.get_rerank_provider(provider_key)
            else:
                provider = ModelProviderFactory.get_rerank_provider(provider_key, configurations)
            self._rerank_providers[cache_key] = provider
        return provider

    def clear_rerank_cache(self) -> None:
        """关闭并清空已缓存的 Rerank Provider。"""
        providers = list(self._rerank_providers.values())
        self._rerank_providers.clear()
        closed: set[int] = set()
        for provider in providers:
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            close_resource_sync(provider, "Rerank Provider")

    def close(self) -> None:
        """释放重排缓存及 EmbeddingService 持有的 Provider。"""
        self.clear_rerank_cache()
        close_embedding = getattr(self.embedding_service, "close", None)
        if callable(close_embedding):
            close_embedding()

    async def aclose(self) -> None:
        """异步释放重排缓存及 EmbeddingService 持有的 Provider。"""
        providers = list(self._rerank_providers.values())
        self._rerank_providers.clear()
        closed: set[int] = set()
        for provider in providers:
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            close = getattr(provider, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
                continue
            close = getattr(provider, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        close_embedding = getattr(self.embedding_service, "aclose", None)
        if not callable(close_embedding):
            close_embedding = getattr(self.embedding_service, "close", None)
        if callable(close_embedding):
            result = close_embedding()
            if inspect.isawaitable(result):
                await result

    async def _rerank_if_needed(
        self,
        query: str,
        ranked_results: list[dict[str, Any]],
        session_config: SessionConfig,
        *,
        rerank_enabled: bool | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """按需重排。

        ``rerank_enabled`` 由调用方传入**已收敛**的布尔：``retrieve`` 已经在
        检索开始前校验过（早于任何检索工作，失败更快），这里再读一次原属性会
        绕过那道校验——该字段可被直写。真值语义会把 ``None``／``0``／``[]``
        静默当成「禁用」、把 ``"no"`` 当成「启用」（见
        ``_coerce_rerank_enabled``）。默认 ``None`` 时自行收敛，供库调用方直接
        调本方法的场景使用。
        """
        enabled = (
            _coerce_rerank_enabled(session_config.rerank_enabled)
            if rerank_enabled is None
            else rerank_enabled
        )
        if not enabled or not ranked_results:
            return ranked_results

        # 与 ``top_k``／``rerank_enabled`` 同一处置：本方法的 docstring 明确
        # 「供库调用方直接调本方法」，而库调用方不经过 ``retrieve`` 里那道
        # ``_require_document_list``。入参形态不受控，实测非列表取值的收场都不在
        # 契约内：``"abc"`` 是合法可迭代对象，逐**字符**进入下面的
        # ``document["score"] = ...`` 后抛 ``TypeError: 'str' object does not
        # support item assignment``；``5``／``10**4301``／``object()`` 抛裸
        # ``TypeError: ... is not iterable``；``{"a": 1}`` 抛 ``KeyError: 0``
        # （字典迭代出的是键）。三种都不是 ``ValueError``，穿透调用方按
        # ``ValueError`` 建立的分类。判据与生产路径共用 ``_require_document_list``。
        ranked_results = _require_document_list(ranked_results, "重排")

        rerank_provider = self._get_rerank_provider(
            session_config.active_rerank_configuration,
            session_config.rerank_configurations,
        )
        # 内容形态由 ``document_content`` 收敛：直接取 ``page_content`` 会把
        # ``None``／``42`` 这类值交给 provider，由它的 ``_validate_inputs`` 抛
        # 「documents 必须全部是字符串」——错误信息指向 provider 的参数，而真正
        # 的原因是上游 store 返回了形态异常的文档。在这里收敛更靠近原因。
        docs_to_rerank = [document_content(doc, "") for doc in ranked_results]
        # Keep the requested output bounded by the available candidates after
        # thresholding and parent de-duplication.
        # 与 ``rerank_enabled`` 同一处置：优先用调用方传入的**已收敛**取值。
        # 本方法的 docstring 明确「供库调用方直接调本方法」，而库调用方不经过
        # ``retrieve`` 开头那道收敛——实测直写 ``top_k="abc"``／``None``／``[1]``
        # 时 ``min()`` 抛裸 ``TypeError: '<' not supported between instances of
        # 'int' and 'str'``，不是 ``ValueError``。
        effective_top_k = (
            require_scale_field(session_config.top_k, "top_k") if top_k is None else top_k
        )
        rerank_top_n = max(1, min(effective_top_k, len(docs_to_rerank)))
        rerank_result = await rerank_provider.arerank(query, docs_to_rerank, top_n=rerank_top_n)
        reranked_indices, reranked_scores = self._validate_rerank_output(
            *_unpack_rerank_result(rerank_result),
            document_count=len(ranked_results),
        )
        reranked_docs: list[dict[str, Any]] = []
        for index, score in zip(reranked_indices, reranked_scores):
            document = copy.deepcopy(ranked_results[index])
            document["score"] = score
            reranked_docs.append(document)
        return sorted(reranked_docs, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _validate_rerank_output(
        indices: Any,
        scores: Any,
        document_count: int,
    ) -> tuple[list[int], list[float]]:
        """Validate a provider result before it can alter retrieval ordering."""
        try:
            normalized_indices = list(indices)
            normalized_scores = list(scores)
        except TypeError as exc:
            raise RuntimeError("Rerank provider 返回值必须是可迭代的索引和分数。") from exc

        if len(normalized_indices) != len(normalized_scores):
            raise RuntimeError("Rerank provider 返回的索引和分数数量不一致。")
        if not normalized_indices:
            raise RuntimeError("Rerank provider 未返回可用结果。")

        seen_indices: set[int] = set()
        validated_scores: list[float] = []
        for position, (index, score) in enumerate(zip(normalized_indices, normalized_scores)):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < document_count
            ):
                raise RuntimeError(f"Rerank provider 返回的 index 无效: position={position}。")
            if index in seen_indices:
                raise RuntimeError(f"Rerank provider 返回重复 index: {index}。")
            seen_indices.add(index)
            if isinstance(score, bool):
                raise RuntimeError(f"Rerank provider 返回的 score 无效: index={index}。")
            # ``float(10**400)`` 抛 ``OverflowError``——它不是 ``ValueError``，
            # 因此下面这个 ``except`` 接不住。provider 返回值来自网络 JSON，
            # 里面出现一个超大整数字面量就足以走到这里。先判双精度范围，
            # 把它收敛成本方法承诺的 ``RuntimeError``。
            if isinstance(score, int) and not is_double_convertible(score):
                raise RuntimeError(f"Rerank provider 返回的 score 超出双精度范围: index={index}。")
            try:
                normalized_score = float(score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Rerank provider 返回的 score 无效: index={index}。") from exc
            if not math.isfinite(normalized_score):
                raise RuntimeError(f"Rerank provider 返回的 score 必须是有限数值: index={index}。")
            validated_scores.append(normalized_score)

        return normalized_indices, validated_scores

    @staticmethod
    def _should_apply_score_threshold(
        session_config: SessionConfig, *, fusion_strategy: str | None = None
    ) -> bool:
        """RRF 融合的分数是排名倒数之和，与阈值不同量纲，因此跳过阈值过滤。

        ``fusion_strategy`` 优先用调用方传入的**已收敛**取值：``retrieve`` 已经
        校验过。但兜底分支必须自己收敛，不能回落到原属性——本函数的判据是
        ``strategy == "rrf"``，而 ``"RRF"``／``" rrf "`` 走的是 RRF 融合却不等于
        这个字面量，阈值过滤会被错误地套用在排名倒数的量纲上（正是
        ``_coerce_fusion_strategy`` 的 docstring 记录的那个缺陷）。原先的兜底
        分支直接读原属性，等于把那条缺陷留在库调用方这条路径上（实测
        ``hybrid_fusion_strategy="RRF"`` 时返回 True）。

        ``retrieval_method`` 同理收敛：与字面量比较对未知取值**不会报错**，而是
        静默判定「不是混合检索」，于是阈值过滤照常套用。
        """
        strategy = (
            fusion_strategy
            if fusion_strategy is not None
            else _coerce_fusion_strategy(session_config.hybrid_fusion_strategy)
        )
        method = require_retrieval_method(session_config.retrieval_method)
        return not (method == RetrievalMethod.HYBRID_SEARCH and strategy == "rrf")
