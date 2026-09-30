from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from src.utils.config import (
    _MAX_RETRIEVAL_CANDIDATES,
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    ModelDetail,
    RetrievalMethod,
    Settings,
    is_double_convertible,
    require_chat_temperature,
    require_retrieval_method,
    safe_repr,
    to_finite_float,
)

SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class RunConfig:
    knowledge_base_path: Path
    legacy_pkl_path: Path
    snapshot_root: Path
    cache_path: Path
    log_path: Path
    default_vector_store: str
    default_llm_provider: str
    default_embedding_provider: str
    default_rerank_provider: str
    llm_configurations: dict[str, ModelDetail]
    embedding_configurations: dict[str, ModelDetail]
    rerank_configurations: dict[str, ModelDetail]
    chat_temperature: float
    kb_embedding_batch_size: int
    kb_chunk_size: int
    kb_chunk_overlap: int
    kb_child_chunk_size: int
    kb_child_chunk_overlap: int

    @property
    def active_snapshot_marker(self) -> Path:
        return self.snapshot_root / "ACTIVE_SNAPSHOT"


class SessionConfig(MutableMapping[str, Any]):
    def __init__(
        self,
        retrieval_method: RetrievalMethod,
        vector_weight: float,
        keyword_weight: float,
        hybrid_fusion_strategy: str,
        retrieval_candidate_multiplier: int,
        rerank_enabled: bool,
        top_k: int,
        score_threshold: float,
        active_llm_configuration: str,
        active_rerank_configuration: str,
        llm_configurations: dict[str, ModelDetail],
        rerank_configurations: dict[str, ModelDetail],
        chat_temperature: float,
    ):
        # 构造器走与 ``__setitem__`` **同一组**校验器。
        #
        # 类本身是公开的，库调用方可以直接 ``SessionConfig(...)`` 构造，只在
        # ``__setitem__`` 里守卫会留下一个完整的绕过口子——实测该构造器对
        # ``top_k=10**9``、``top_k=True``、``score_threshold=nan``、
        # ``chat_temperature=-1``、``vector_weight=nan`` 等**每一个**受校验
        # 字段都照单全收。其中 ``top_k`` 会一路走到 ``faiss_index.search()``，
        # FAISS 既不报错也不截断，按 10 亿条分配内存。
        #
        # 构造器自身是 ``build_session_config`` 的内部出口（值来自已校验的
        # ``Settings``），所以这条路径正常不会触发校验；这里校验的是「绕过
        # ``Settings`` 直接构造」的调用方。
        self.retrieval_method = self._validate_retrieval_method(retrieval_method)
        vector_weight = self._validate_vector_weight(vector_weight)
        keyword_weight = self._validate_keyword_weight(keyword_weight)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.hybrid_fusion_strategy = self._validate_fusion_strategy(hybrid_fusion_strategy)
        self.retrieval_candidate_multiplier = self._validate_candidate_multiplier(
            retrieval_candidate_multiplier
        )
        self.rerank_enabled = self._validate_rerank_enabled(rerank_enabled)
        self.top_k = self._validate_top_k(top_k)
        self.score_threshold = self._validate_score_threshold(score_threshold)
        self.active_llm_configuration = active_llm_configuration
        self.active_rerank_configuration = active_rerank_configuration
        self.llm_configurations = llm_configurations
        self.rerank_configurations = rerank_configurations
        self.chat_temperature = self._validate_chat_temperature(chat_temperature)

        # 组合校验放最后：单字段都合法之后才谈字段之间的约束。
        if vector_weight == 0.0 and keyword_weight == 0.0:
            raise ValueError(
                "vector_weight 与 keyword_weight 不能同时为 0，"
                "否则混合检索的所有得分都是 0 并被分数阈值全部过滤，静默返回空结果。"
            )
        self._reject_oversized_candidate_pool(self.top_k, self.retrieval_candidate_multiplier)

    # 可经 ``__setitem__`` 写入的字段及其校验器。``__init__`` 信任调用方
    # （``build_session_config`` 从已校验的 ``Settings`` 取值，合法），
    # 但 UI 与库调用方走 ``__setitem__``，必须逐字段校验。
    #
    # 这里不是重复 `Settings` 的工作：`Settings` 只在启动时校验一次，而
    # `SessionConfig` 是运行期可变的独立入口。修 `top_k` 上界之前实测
    # `chat_config["top_k"] = 10**9` 能一路走到 `faiss_index.search()`，
    # FAISS 既不报错也不截断，按 10 亿条分配。注册表在类体外填充（类体内直接
    # 引用 `cls._validate_x` 会在定义期拿不到绑定方法）。
    _VALIDATORS: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    # 已知字段名。``_require_known_field`` 用它做成员判断——不能用
    # ``key in self.to_dict()``，那会先把 key 拿去哈希、对不可哈希的取值抛
    # ``TypeError``。``to_dict`` 的键集就是字段集（下方 ``_VALIDATORS`` 的注释
    # 说明了二者的一致性），这里在类定义末尾由 ``_FIELD_NAMES`` 赋值填充。
    _FIELD_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._require_known_field(key)
        validator = self._VALIDATORS.get(key)
        validated = validator(value) if validator is not None else value
        if key in ("vector_weight", "keyword_weight"):
            # 兄弟字段可能绕过 validator 被属性直写（``SessionConfig`` 是公开类，
            # 普通属性没有 ``__setattr__`` 守卫）。``set_many`` 对同一情形有
            # ``_require_finite_number`` 收敛，这里必须用同一判据——否则
            # ``cfg.keyword_weight = nan`` 之后 ``cfg["vector_weight"] = 0.0``
            # 会静默放行（``nan == 0.0`` 恒为 False），两个权重同时失效，
            # 混合检索的所有得分都是 0 并被阈值全部过滤。
            other_key = "keyword_weight" if key == "vector_weight" else "vector_weight"
            self._require_finite_number(getattr(self, other_key, None), other_key)
            self._reject_both_weights_zero(key, validated)
        if key in ("top_k", "retrieval_candidate_multiplier"):
            # 改任一个都可能让乘积越界，因此按「改完之后」的组合判一次。
            # 兄弟字段同样可能未经 validator：``getattr`` 拿到 list/str 后
            # ``top_k * multiplier`` 与随后的 ``>`` 比较抛裸 ``TypeError``，
            # 穿透调用方按 ``ValueError`` 建立的分类。用与 ``set_many`` 同一套
            # ``_require_int_value`` 收敛。
            other_key = "retrieval_candidate_multiplier" if key == "top_k" else "top_k"
            self._reject_oversized_candidate_pool(
                self._require_int_value(
                    validated if key == "top_k" else getattr(self, "top_k", validated), "top_k"
                ),
                self._require_int_value(
                    validated
                    if key == "retrieval_candidate_multiplier"
                    else getattr(self, other_key, validated),
                    other_key,
                ),
            )
        setattr(self, key, validated)

    def set_many(self, values: Mapping[str, Any]) -> None:
        """批量写入，按**最终值**做组合校验，成功才整体生效。

        逐个调 ``__setitem__`` 会拿**副本当时的兄弟字段值**去校验——即尚未被
        这批更新覆盖的旧值。于是同一个合法终态能否被接受取决于 dict 的遍历
        顺序：初始 ``(top_k=5, mult=3)``、目标 ``(top_k=2000, mult=1)``（乘积
        2000 完全合法），先写 ``top_k`` 会拿旧的 ``mult=3`` 算出 6000 而拒绝，
        先写 ``mult`` 则通过。

        这不是理论问题：配置编辑器的动态上界会**放行**这两个字段（先设
        ``mult=1`` 时上界是 100，再设 ``top_k=2000`` 时上界正好 2000），
        而 ``to_dict()`` 的键顺序恰好是能通过的那个；但嵌入方传入自己的 dict、
        或将来调整字段顺序，就会变成「UI 放行、应用配置时报错并终止会话」。

        注意这与「两个权重同时为零」那条不同：那条是**等值**约束，最后一次写入
        必然能看到另一个字段的新值，因此逐个写也不会漏。乘积是**上界**约束，
        中间态可能瞬时越界而终态合法，所以必须按终态判。

        组合校验在单字段校验全部通过之后、任何字段落盘之前执行；因此被拒时
        不会留下改了一半的状态。
        """
        # 逐个过同一个 helper：``key not in self.to_dict()`` 对不可哈希的 key 抛
        # ``TypeError: unhashable type``，而消息里的 ``{key!r}`` 对位数超限的整数
        # 抛 ``ValueError: Exceeds the limit``。两条都与「字段名写错了」无关，
        # 而同文件的 ``_validate_fusion_strategy`` 早已认领同一类线索。
        # 容器形态先判：``for key in values`` 对不可迭代的取值抛裸
        # ``TypeError: 'int' object is not iterable``——而紧随其后的
        # ``_require_known_field`` 对**可迭代但字段名非法**的形态给的是带字段名
        # 的 ``ValueError``。同一判据的两半必须都在，否则「参数写错」的收场
        # 取决于它是标量还是序列。``str``／``bytes`` 要显式排除：它们是合法的
        # ``Sequence``，迭代得到的是字符（``"ab"`` 会被拆成 ``a``／``b`` 两个
        # 字段名，报出与调用方写法无关的 ``KeyError``）。
        # 取值来源是公开的 ``Chatbot.chat_config`` setter（标注
        # ``SessionConfig | dict``，但运行时不受约束）。
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Mapping):
            raise ValueError(f"批量写入必须是字段映射，但得到 {safe_repr(values)}。")
        for key in values:
            self._require_known_field(key)

        # 第一遍：逐字段校验，先在本地收敛成合法值（此时不改动 self）。
        validated: dict[str, Any] = {}
        for key, value in values.items():
            validator = self._VALIDATORS.get(key)
            validated[key] = validator(value) if validator is not None else value

        # 第二遍：组合校验按「这批更新全部生效之后」的取值判定。
        def final(field: str) -> Any:
            if field in validated:
                return validated[field]
            return getattr(self, field, validated.get(field))

        # 兄弟字段的取值可能**没经过 validator**：``SessionConfig`` 是公开类，
        # 普通属性没有 ``__setattr__`` 守卫，嵌入方（或携带超大整数的 pickle
        # 快照）可以直接 ``cfg.vector_weight = 10**400``。于是这里的
        # ``float(...)`` 抛 ``OverflowError``（``ArithmeticError`` 子类，**不是**
        # ``ValueError``）、``int(inf)`` 抛 ``OverflowError``、``int(nan)`` 抛
        # ``ValueError``——实测三条都复现，而调用方只按 ``ValueError`` 分类。
        # ``_require_finite_number`` 把取值收敛成有限浮点或给出可读的
        # ``ValueError``，校验与运算用的是同一个值。
        vector_weight = self._require_finite_number(final("vector_weight"), "vector_weight")
        keyword_weight = self._require_finite_number(final("keyword_weight"), "keyword_weight")
        if vector_weight == 0.0 and keyword_weight == 0.0:
            raise ValueError(
                "vector_weight 与 keyword_weight 不能同时为 0，"
                "否则混合检索的所有得分都是 0 并被分数阈值全部过滤，静默返回空结果。"
            )
        if "top_k" in validated or "retrieval_candidate_multiplier" in validated:
            self._reject_oversized_candidate_pool(
                self._require_int_value(final("top_k"), "top_k"),
                self._require_int_value(
                    final("retrieval_candidate_multiplier"), "retrieval_candidate_multiplier"
                ),
            )

        # 第三遍：全部通过，整体落盘。
        for key, value in validated.items():
            setattr(self, key, value)

    @staticmethod
    def _require_known_field(key: Any) -> None:
        """校验字段名已知；形态异常与未知字段都收敛成 ``KeyError``／``ValueError``。

        ``key not in self.to_dict()`` 对不可哈希的 key 抛 ``TypeError:
        unhashable type``，而 ``f"{key!r}"`` 对位数超限的整数抛 ``ValueError:
        Exceeds the limit``——两条都与「字段名写错了」无关。同文件的
        ``_validate_fusion_strategy`` 早已认领同一类线索。
        """
        if not isinstance(key, str):
            raise ValueError(f"SessionConfig 的字段名必须是字符串，但得到 {safe_repr(key)}。")
        if key not in SessionConfig._FIELD_NAMES:
            raise KeyError(f"SessionConfig 没有字段 {key!r}。")

    @staticmethod
    def _require_finite_number(value: Any, label: str) -> float:
        """把权重取值收敛成有限浮点，非法形态给出可读的 ``ValueError``。

        用于**兄弟字段**：它们可能绕过 validator 被属性直写。``float(10**400)``
        抛 ``OverflowError``——``ArithmeticError`` 子类，**不是** ``ValueError``。
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} 必须是数值，但得到 {safe_repr(value)}。")
        number = to_finite_float(value)
        if number is None:
            raise ValueError(f"{label} 必须是有限数值，但得到 {safe_repr(value)}。")
        return number

    @staticmethod
    def _require_int_value(value: Any, label: str) -> int:
        """把整数取值收敛成 ``int``，非法形态给出可读的 ``ValueError``。

        ``int(inf)`` 抛 ``OverflowError``、``int(nan)`` 抛 ``ValueError``——前者
        不是 ``ValueError``，会穿透调用方的错误分类。
        """
        if isinstance(value, bool) or not isinstance(value, int):
            number = to_finite_float(value)
            if number is None or not float(number).is_integer():
                raise ValueError(f"{label} 必须是整数，但得到 {safe_repr(value)}。")
            return int(number)
        return value

    @staticmethod
    def _require_bounded_int(value: Any, label: str, upper: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} 必须是整数，但得到 {safe_repr(value)}。")
        if not 1 <= value <= upper:
            # 消息里必须用 ``safe_repr``：``repr(10**4301)`` 超过
            # ``sys.get_int_max_str_digits()``（默认 4300），自己抛
            # ``ValueError: Exceeds the limit``——错误消息反过来把异常内容
            # 换成了与字段无关的「整数转字符串超限」。
            raise ValueError(f"{label} 必须是 1 到 {upper} 之间的整数，但得到 {safe_repr(value)}。")
        return value

    @classmethod
    def _validate_top_k(cls, value: Any) -> int:
        return cls._require_bounded_int(value, "top_k", _MAX_RETRIEVAL_TOP_K)

    @classmethod
    def _validate_candidate_multiplier(cls, value: Any) -> int:
        return cls._require_bounded_int(
            value, "retrieval_candidate_multiplier", _MAX_RETRIEVAL_MULTIPLIER
        )

    @staticmethod
    def _validate_unit_weight(value: Any, label: str) -> float:
        # ``float(10**400)`` 抛 ``OverflowError``——它是 ``ArithmeticError`` 的
        # 子类，**不是** ``ValueError``，会穿透调用方的 ``except ValueError``
        # 链（UI 与库调用方都只接 ``ValueError``）。判据必须用
        # ``is_double_convertible``：``math.isfinite(float(x))`` 在同一个输入上
        # 自己就抛 ``OverflowError``，拿它当守卫等于没有守卫。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} 必须是数值，但得到 {safe_repr(value)}。")
        if not is_double_convertible(value):
            raise ValueError(f"{label} 必须是有限数值，但得到 {safe_repr(value)}。")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} 必须是有限数值，但得到 {safe_repr(value)}。")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label} 必须在 0.0 到 1.0 之间，但得到 {safe_repr(value)}。")
        return number

    @classmethod
    def _validate_vector_weight(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "vector_weight")

    @classmethod
    def _validate_keyword_weight(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "keyword_weight")

    @staticmethod
    def _reject_oversized_candidate_pool(top_k: int, multiplier: int) -> None:
        """拒绝让候选规模超过上界的 ``top_k`` × ``retrieval_candidate_multiplier``。

        两个字段各自有上界，但**乘积**才是真正送进 ``faiss_index.search()`` 的
        k（见 ``RetrievalService.retrieve`` 的 ``effective_top_k``），而 FAISS
        对超过索引条数的 k 既不报错也不截断，按 k 全量分配。实测 dim=1536、
        k=10**6 时峰值 RSS 涨约 35 GB。``Settings`` 有同一条约束，这里是运行期
        入口的对应实现。
        """
        candidates = top_k * multiplier
        if candidates > _MAX_RETRIEVAL_CANDIDATES:
            # ``safe_repr`` 而不是裸 f-string：``top_k``／``multiplier`` 可能被
            # 属性直写成位数超限的整数（见 ``_require_int_value``），插值
            # ``{candidates}`` 会在**正要报「候选规模过大」**的那一步抛
            # ``ValueError: Exceeds the limit``——异常类型仍是 ``ValueError``，
            # 但内容与字段毫无关系。同文件 ``_require_bounded_int`` 早已用同一判据。
            raise ValueError(
                f"top_k × retrieval_candidate_multiplier 不能超过 "
                f"{_MAX_RETRIEVAL_CANDIDATES}，但得到 {safe_repr(candidates)}。"
                " 两者相乘才是实际送入向量检索的候选规模，过大会一次性申请巨量内存。"
            )

    def _reject_both_weights_zero(self, key: str, value: float) -> None:
        """拒绝把混合检索的两个权重同时设为零。

        ``Settings`` 在启动时校验过一次，但 ``SessionConfig`` 是运行期独立
        入口（UI 与库调用方走 ``__setitem__``），单字段界拦不住这种组合。
        两个权重同时为零时 ``weighted`` 融合的每个文档得分都是 0.0，随后被
        ``chat_score_threshold`` 全部过滤——检索静默返回空结果且无任何报错，
        正是「把配置错误伪装成检索质量差」的那类失败。
        """
        other = self.keyword_weight if key == "vector_weight" else self.vector_weight
        if value == 0.0 and other == 0.0:
            raise ValueError(
                "vector_weight 与 keyword_weight 不能同时为 0，"
                "否则混合检索的所有得分都是 0 并被分数阈值全部过滤，静默返回空结果。"
            )

    @classmethod
    def _validate_score_threshold(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "score_threshold")

    @classmethod
    def _validate_retrieval_method(cls, value: Any) -> RetrievalMethod:
        # 判据与措辞都交给 ``require_retrieval_method``：同一字段在配置边界
        # （这里）与运行期读取侧（``retrieval_service``／两个 UI 取用点）必须
        # 给出同一种收场，否则「写进去被拒」与「读出来报错」会分成两套消息。
        return require_retrieval_method(value)

    @staticmethod
    def _validate_fusion_strategy(value: Any) -> str:
        # 集合成员判断对不可哈希的值抛 ``TypeError: unhashable type``，而本类的
        # 契约是 ``ValueError``（``__setitem__`` 与构造器都据此分类）。输入可达：
        # 配置来自 TOML，``hybrid_fusion_strategy = ["rrf"]`` 是合法的 TOML 数组。
        # 实测修复前 ``cfg["hybrid_fusion_strategy"] = ["rrf"]`` 抛的是
        # ``TypeError``，绕过全部 ``except ValueError``。
        if not isinstance(value, str) or value not in {"rrf", "weighted"}:
            raise ValueError(
                f"hybrid_fusion_strategy 必须是 'rrf' 或 'weighted'，但得到 {safe_repr(value)}。"
            )
        return str(value)

    @staticmethod
    def _validate_rerank_enabled(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"rerank_enabled 必须是布尔值，但得到 {safe_repr(value)}。")
        return value

    @staticmethod
    def _validate_chat_temperature(value: Any) -> float:
        # 判据与措辞都交给 ``require_chat_temperature``：同一字段在配置边界（这里）
        # 与运行期读取侧（``ChatService.generate_reply``／``Chatbot.chat_async`` 的
        # 兜底分支）必须给出同一种收场，否则「写进去被拒」与「读出来报错」会分成
        # 两套消息。
        return require_chat_temperature(value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("SessionConfig 不支持删除字段。")

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_method": self.retrieval_method,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "hybrid_fusion_strategy": self.hybrid_fusion_strategy,
            "retrieval_candidate_multiplier": self.retrieval_candidate_multiplier,
            "rerank_enabled": self.rerank_enabled,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "active_llm_configuration": self.active_llm_configuration,
            "active_rerank_configuration": self.active_rerank_configuration,
            "llm_configurations": self.llm_configurations,
            "rerank_configurations": self.rerank_configurations,
            "chat_temperature": self.chat_temperature,
        }


# 每个可经 ``__setitem__`` 写入的字段都要在这里登记；未登记的字段（模型配置字典
# 等）按原样写入，由使用方负责其内部结构。
SessionConfig._VALIDATORS = {
    "retrieval_method": SessionConfig._validate_retrieval_method,
    "vector_weight": SessionConfig._validate_vector_weight,
    "keyword_weight": SessionConfig._validate_keyword_weight,
    "hybrid_fusion_strategy": SessionConfig._validate_fusion_strategy,
    "retrieval_candidate_multiplier": SessionConfig._validate_candidate_multiplier,
    "rerank_enabled": SessionConfig._validate_rerank_enabled,
    "top_k": SessionConfig._validate_top_k,
    "score_threshold": SessionConfig._validate_score_threshold,
    "chat_temperature": SessionConfig._validate_chat_temperature,
}

# ``to_dict()`` 的键集就是字段集——``__setitem__``／``set_many`` 的字段名校验
# 直接以它为准，避免两处各维护一份列表而分叉。
SessionConfig._FIELD_NAMES = frozenset(
    SessionConfig(
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        vector_weight=1.0,
        keyword_weight=0.0,
        hybrid_fusion_strategy="rrf",
        retrieval_candidate_multiplier=1,
        rerank_enabled=False,
        top_k=1,
        score_threshold=0.0,
        active_llm_configuration="",
        active_rerank_configuration="",
        llm_configurations={},
        rerank_configurations={},
        chat_temperature=0.0,
    ).to_dict()
)


def _manifest_count(value: Any, label: str) -> int:
    """把清单的计数字段收敛成非负整数，非法取值给出可读的 ``ValueError``。

    ``int(inf)`` 抛 ``OverflowError``——``ArithmeticError`` 子类而**不是**
    ``ValueError``，会穿透 ``write_manifest`` 调用方的错误分类；``int("abc")``
    抛 ``TypeError``。两者都收敛成带字段名的 ``ValueError``。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        number = to_finite_float(value)
        if number is None or not float(number).is_integer():
            raise ValueError(f"清单字段 {label} 必须是非负整数，但得到 {safe_repr(value)}。")
        value = int(number)
    if value < 0:
        raise ValueError(f"清单字段 {label} 必须是非负整数，但得到 {safe_repr(value)}。")
    # 位数上界：``f"{value}"`` 在 TOML 拼接处撞上 CPython 的整数位数上限
    # （默认 4300），于是「写清单」抛的是 ``Exceeds the limit`` 而字段名与原因
    # 都不在消息里。TOML 本身能表示任意精度的整数，但这个量级的计数没有任何
    # 现实意义——真实快照是几十到几万。
    if value.bit_length() > 64:
        raise ValueError(
            f"清单字段 {label} 超出合理范围（{safe_repr(value)}），快照计数不应超过 2**64。"
        )
    return value


def _manifest_text(value: Any) -> str:
    """把清单字段收敛成字符串，位数超限时降级为量级描述。

    与 ``from_mapping`` 的 docstring 同因：这些值来自磁盘上的清单文件，
    ``str()`` 对超过 ``sys.get_int_max_str_digits()`` 的整数自己会抛。
    """
    if isinstance(value, str):
        return value
    return safe_repr(value)


@dataclass(frozen=True)
class KnowledgeSnapshotManifest:
    schema_version: str
    snapshot_id: str
    created_at: str
    store_type: str
    embedding_provider: str
    embedding_model: str
    chunk_mode: str
    source_digest: str
    document_count: int
    chunk_count: int

    @classmethod
    def create(
        cls,
        snapshot_id: str,
        store_type: str,
        embedding_provider: str,
        embedding_model: str,
        chunk_mode: str,
        source_digest: str,
        document_count: int,
        chunk_count: int,
    ) -> KnowledgeSnapshotManifest:
        return cls(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            created_at=datetime.now(UTC).isoformat(),
            store_type=store_type,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_mode=chunk_mode,
            source_digest=source_digest,
            document_count=document_count,
            chunk_count=chunk_count,
        )

    @staticmethod
    def _toml_string(value: Any) -> str:
        r"""把一个值序列化成 TOML 基本字符串（含转义）。

        原实现用 ``f'key = "{value}"'`` 直接拼，没有做 TOML 字符串转义，而这些
        值都来自 ``config.toml``／``.env``：``embedding_model`` 可以是自建网关的
        模型路径（``models\qwen-7b``），``chunk_mode`` 与 ``snapshot_id`` 也由调用方
        决定。实测模型名含反斜杠时 ``write_manifest`` 正常落盘、``load_manifest``
        抛 ``TOMLDecodeError: Unescaped '\' in a string``——快照写得出、读不回，
        报错现场离真正的原因很远；含引号或换行的值则连落盘的 TOML 都不可解析。
        同一份 manifest 在内存里是好的，落盘一圈就坏了。

        这里按 TOML 1.0 的规定转义 ``"``、``\`` 与不可打印字符：可打印字符原样
        保留（含中文），其余走 ``\uXXXX``。``str`` 之外的值先转成字符串——写出来
        的是文本，读回时由 ``from_mapping`` 的 ``str(...)`` 负责，与既有行为一致。
        """
        # 非字符串走 ``_manifest_text`` 而不是 ``str``：``str(10**4301)`` 撞上
        # CPython 的整数位数上限，在**正要写清单**时抛 ``ValueError: Exceeds the
        # limit``——而读取侧的 ``from_mapping`` 早已按同一判据修过，只修一个方向
        # 会让「写得进、读不出」反过来变成「写得就崩」。写入侧的值同样来自
        # ``config.toml``／磁盘快照。
        text = value if isinstance(value, str) else _manifest_text(value)
        escaped = []
        for char in text:
            if char == "\\":
                escaped.append("\\\\")
            elif char == '"':
                escaped.append('\\"')
            elif char == "\b":
                escaped.append("\\b")
            elif char == "\t":
                escaped.append("\\t")
            elif char == "\n":
                escaped.append("\\n")
            elif char == "\f":
                escaped.append("\\f")
            elif char == "\r":
                escaped.append("\\r")
            elif char < " " or char == "\x7f":
                escaped.append(f"\\u{ord(char):04X}")
            else:
                escaped.append(char)
        return '"' + "".join(escaped) + '"'

    def to_toml(self) -> str:
        toml_string = self._toml_string
        return "\n".join(
            [
                f"schema_version = {toml_string(self.schema_version)}",
                f"snapshot_id = {toml_string(self.snapshot_id)}",
                f"created_at = {toml_string(self.created_at)}",
                f"store_type = {toml_string(self.store_type)}",
                f"embedding_provider = {toml_string(self.embedding_provider)}",
                f"embedding_model = {toml_string(self.embedding_model)}",
                f"chunk_mode = {toml_string(self.chunk_mode)}",
                f"source_digest = {toml_string(self.source_digest)}",
                # 计数同样可能来自磁盘（畸形快照）或调用方，``int()`` 对
                # ``inf`` 抛 ``OverflowError``（``ArithmeticError`` 子类，不是
                # ``ValueError``）、对位数超限的整数抛 ``ValueError``。这里经
                # ``_manifest_count`` 收敛，与 ``from_mapping`` 同源。
                f"document_count = {_manifest_count(self.document_count, 'document_count')!r}",
                f"chunk_count = {_manifest_count(self.chunk_count, 'chunk_count')!r}",
                "",
            ]
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> KnowledgeSnapshotManifest:
        """从清单映射构造，非字符串字段一律经 ``safe_repr`` 收敛。

        不能直接写 ``str(data[key])``：清单文件来自磁盘，而**本项目加载
        pickle 格式的快照**，pickle 是二进制格式、不受 CPython 的十进制位数
        限制（实测 ``pickle.loads(pickle.dumps(10**4301))`` 正常往返），因此
        一个畸形快照可以让 ``str()`` 在**正要报告清单内容**时抛
        ``ValueError: Exceeds the limit (4300 digits)``——异常类型与内容都与
        「清单字段不合法」无关，调用方的 ``except`` 分类随之失准。
        ``safe_repr`` 对位数超限的值给出量级描述，消息仍然可用。
        """
        return cls(
            schema_version=_manifest_text(data["schema_version"]),
            snapshot_id=_manifest_text(data["snapshot_id"]),
            created_at=_manifest_text(data["created_at"]),
            store_type=_manifest_text(data["store_type"]),
            embedding_provider=_manifest_text(data["embedding_provider"]),
            embedding_model=_manifest_text(data["embedding_model"]),
            chunk_mode=_manifest_text(data["chunk_mode"]),
            source_digest=_manifest_text(data["source_digest"]),
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
        )


def build_run_config(settings: Settings) -> RunConfig:
    return RunConfig(
        knowledge_base_path=Path(settings.knowledge_base_path),
        legacy_pkl_path=Path(settings.pkl_path),
        snapshot_root=Path(settings.snapshot_root),
        cache_path=Path(settings.cache_path),
        log_path=Path(settings.log_path),
        default_vector_store=settings.default_vector_store,
        default_llm_provider=settings.default_llm_provider,
        default_embedding_provider=settings.default_embedding_provider,
        default_rerank_provider=settings.default_rerank_provider,
        llm_configurations=deepcopy(settings.llm_configurations),
        embedding_configurations=deepcopy(settings.embedding_configurations),
        rerank_configurations=deepcopy(settings.rerank_configurations),
        chat_temperature=settings.chat_temperature,
        kb_embedding_batch_size=settings.kb_embedding_batch_size,
        kb_chunk_size=settings.kb_chunk_size,
        kb_chunk_overlap=settings.kb_chunk_overlap,
        kb_child_chunk_size=settings.kb_child_chunk_size,
        kb_child_chunk_overlap=settings.kb_child_chunk_overlap,
    )


def build_session_config(settings: Settings) -> SessionConfig:
    return SessionConfig(
        retrieval_method=settings.chat_retrieval_method,
        vector_weight=settings.chat_vector_weight,
        keyword_weight=settings.chat_keyword_weight,
        hybrid_fusion_strategy=settings.hybrid_fusion_strategy,
        retrieval_candidate_multiplier=settings.retrieval_candidate_multiplier,
        rerank_enabled=settings.chat_rerank_enabled,
        top_k=settings.chat_top_k,
        score_threshold=settings.chat_score_threshold,
        active_llm_configuration=settings.default_llm_provider,
        active_rerank_configuration=settings.default_rerank_provider,
        llm_configurations=deepcopy(settings.llm_configurations),
        rerank_configurations=deepcopy(settings.rerank_configurations),
        chat_temperature=settings.chat_temperature,
    )
