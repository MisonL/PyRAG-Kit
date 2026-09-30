import functools
import math
import re
import sys
import tomllib
import warnings
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.exceptions import SettingsError

from src.utils.security import redact_sensitive_text, validate_secret_free_options

# =================================================================
# 1. 基础定义 (DEFINITIONS)
# =================================================================


# 项目根目录
def resolve_app_root() -> Path:
    """
    返回应用根目录。

    - 源码运行时使用仓库根目录。
    - PyInstaller 冻结运行时使用可执行文件所在目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


ROOT_DIR = resolve_app_root()
# 配置文件路径。两者都以 ``ROOT_DIR`` 为基准，不用相对路径——相对路径由
# pydantic-settings 按**进程当前工作目录**解析，而冻结产物的可执行文件目录
# 与调用者 CWD 通常不同，会让用户放在安装目录旁的配置被静默忽略。
CONFIG_TOML_PATH = ROOT_DIR / "config.toml"
ENV_FILE_PATH = ROOT_DIR / ".env"


class RetrievalMethod(str, Enum):
    """定义知识库检索的策略枚举。"""

    SEMANTIC_SEARCH = "向量检索"
    FULL_TEXT_SEARCH = "全文检索"
    HYBRID_SEARCH = "混合检索"


class ModelProtocol(str, Enum):
    """模型渠道使用的线协议。"""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    GENERATE_CONTENT = "generate_content"
    ARK = "ark"


class ModelDetail(BaseModel):
    """定义单个模型配置的结构。"""

    provider: str
    model_name: str
    protocol: ModelProtocol | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        protected_namespaces=(),
        extra="forbid",
    )

    @field_validator("options", mode="before")
    @classmethod
    def validate_options(cls, value: Any, info: ValidationInfo) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("模型 options 必须是 TOML 表或字典。")
        provider = str(info.data.get("provider", "")).strip().lower()
        try:
            if provider == "google" and "http_options" in value:
                http_options = value["http_options"]
                remaining = {key: nested for key, nested in value.items() if key != "http_options"}
                validated = validate_secret_free_options(remaining, "模型")
                validated["http_options"] = validate_secret_free_options(
                    {"http_options": http_options},
                    "模型",
                    allowed_containers={"http_options", "headers"},
                )["http_options"]
                return validated
            return validate_secret_free_options(value, "模型")
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(
                    "模型 options 不允许包含凭证或请求头/query 配置",
                    "模型 options 不允许包含凭证或连接字段",
                )
            ) from exc

    @field_validator("protocol", mode="before")
    @classmethod
    def normalize_protocol(cls, value: Any) -> ModelProtocol | None:
        if value is None or value == "":
            return None
        if isinstance(value, ModelProtocol):
            return value

        # ``str(value)`` 自己会抛：``str(10**4301)`` 撞上 CPython 的整数位数上限
        # （``sys.get_int_max_str_digits()``，默认 4300），于是「协议不支持」的
        # 消息变成 ``Exceeds the limit (4300 digits)``。``protocol`` 是
        # ``config.toml`` 的字段，TOML 整数无上界，用户写错一个类型就会看到这句
        # 与协议毫无关系的话。先判类型，非字符串直接给带字段名的消息。
        # ``str(value)`` 自己会抛：``str(10**4301)`` 撞上 CPython 的整数位数上限
        # （``sys.get_int_max_str_digits()``，默认 4300），于是「协议不支持」的
        # 消息变成 ``Exceeds the limit (4300 digits)``。``protocol`` 是
        # ``config.toml`` 的字段，TOML 整数无上界，用户写错一个类型就会看到这句
        # 与协议毫无关系的话。先判类型，非字符串直接给带字段名的消息。
        # ``str(value)`` 对位数超限的整数自己会抛（``str(10**4301)`` 撞上
        # CPython 的 ``sys.get_int_max_str_digits()``，默认 4300），于是「协议
        # 不支持」的消息变成 ``Exceeds the limit (4300 digits)``。先判类型，
        # 非字符串直接给带字段名的消息。
        #
        # 这条守卫是**纵深防御**而非配置可达缺陷：``protocol`` 确实来自
        # ``config.toml``，但 ``tomllib`` 解析 4301 位的整数时自己就抛
        # ``ValueError``（``get_settings`` 已把它包成「配置解析失败」），位数在
        # 上限内的非字符串（``5``／``["responses"]``／``true``）原本就能得到正确
        # 消息。真正能触达的是 Python API 调用方
        # （``ModelDetail(protocol=10**4301)``），以及未来若换掉 TOML 解析器的
        # 情形。加它的理由是与同文件其它 validator 的判据保持一致——此前
        # ``log_level``／路径字段都已显式判类型，只有这里漏了。
        if not isinstance(value, str):
            supported = ", ".join(protocol.value for protocol in ModelProtocol)
            raise ValueError(f"不支持的模型协议: {safe_repr(value)}。可选值: {supported}")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "chat": ModelProtocol.CHAT_COMPLETIONS,
            "completion": ModelProtocol.CHAT_COMPLETIONS,
            "chat_completion": ModelProtocol.CHAT_COMPLETIONS,
            "chat_completions": ModelProtocol.CHAT_COMPLETIONS,
            "response": ModelProtocol.RESPONSES,
            "responses": ModelProtocol.RESPONSES,
            "anthropic": ModelProtocol.MESSAGES,
            "anthropic_messages": ModelProtocol.MESSAGES,
            "gemini": ModelProtocol.GENERATE_CONTENT,
            "generate": ModelProtocol.GENERATE_CONTENT,
            "generate_content": ModelProtocol.GENERATE_CONTENT,
        }
        try:
            alias = aliases.get(normalized)
            return alias if alias is not None else ModelProtocol(normalized)
        except ValueError as exc:
            supported = ", ".join(protocol.value for protocol in ModelProtocol)
            raise ValueError(f"不支持的模型协议: {safe_repr(value)}。可选值: {supported}") from exc


# =================================================================
# 2. 主配置模型 (MAIN SETTINGS MODEL)
# =================================================================


# 检索规模的上界。不是洁癖：``effective_top_k = chat_top_k *
# retrieval_candidate_multiplier`` 直通 ``faiss_index.search()``，FAISS 既不报错
# 也不截断。实测 ``chat_top_k=10**9`` 单次查询分配约 3.6 GB 数组、RSS 涨
# 10.3 GB、耗时 8.4 秒。
# 单字段上界与乘积上界必须一致：``top_k`` 单独就不该超过候选总规模，否则
# 「字段校验通过、组合校验拒绝」会让用户看到自相矛盾的提示。候选总规模取
# 2_000——实测 dim=1536、k=2000 时峰值 RSS 约 330 MB，而 k=100_000 涨到
# 1.6 GB。默认值是 5 × 3 = 15，留出约 130 倍余量。
# 向量维度的上界。``local-hash`` 的维度直接来自用户可写的 ``model_name``
# 数字后缀（``local-hash-256`` -> 256），而 provider 会**先按该维度分配出完整
# 向量**再返回：实测配成 ``local-hash-100000000`` 时单次 ``embed_texts(["测试"])``
# 耗时 136 秒、峰值内存 3.9 GB，按真实知识库 115 个分块算 ``np.vstack`` 需要约
# 4.3 GB，进程会被 OOM 杀掉而不是得到一条可读的错误。校验必须发生在分配**之前**，
# 因此 provider 侧也要用这个上界。
#
# 现有各家的最大维度是 3072（text-embedding-3-large），65536 留了 20 倍余量，
# 而 65536 维的 float32 单块只有 256 KB。
_MAX_EMBEDDING_DIMENSION = 65_536

_MAX_RETRIEVAL_TOP_K = 2_000
# 候选规模的上界。单看两个字段各自的上界是不够的——真正送进
# ``faiss_index.search()`` 的是**乘积** ``chat_top_k *
# retrieval_candidate_multiplier``（见 ``RetrievalService.retrieve``），而
# FAISS 对超过 ``ntotal`` 的 k 既不报错也不截断，按 k 全量分配。实测
# dim=1536、k=10**6 时峰值 RSS 涨约 35 GB。默认值是 5 × 3 = 15，这里留出
# 约 130 倍余量，覆盖「提高 top_k 并同时放大召回候选」的正常调优。
_MAX_RETRIEVAL_CANDIDATES = 2_000
# 日志保留天数与分片/批处理规模的上界。两者原来都无上界，实测
# ``log_retention_days=10**400``、``kb_chunk_size=10**400`` 均被接受，
# 直到真正参与运算（日期减法、内存申请）才炸。
_MAX_LOG_RETENTION_DAYS = 3650
_MAX_KB_CHUNK_TOKENS = 1_000_000
_MAX_RETRIEVAL_MULTIPLIER = 100


# 十进制数值字面量：可选负号、可选小数点、可选指数。不含下划线分组（``1_0``）、
# 不含正号（``+7``）、不含十六进制（``0x10``）。Python 的 ``int()``/``float()``
# 都会接受它们（``int("1_0") == 10``、``float("1_0") == 10.0``、
# ``float("+7") == 7.0``），静默读出一个「看起来成功」的错值，因此这里先做形状
# 校验再交给 ``float()`` 解析。
_DECIMAL_LITERAL_PATTERN = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_decimal_int(text: str) -> int | None:
    """解析十进制整数字面量；解析不了返回 ``None``，**绝不抛异常**。

    只接受十进制数字（可带前导负号），不做科学计数法/下划线/正号的宽松放行：
    ``int("1_0") == 10``、``int("+7") == 7`` 这类「安静的错值」正是本项目反复
    踩到的模式，宁愿在这里明确拒绝。

    ``int()`` 会抛 ``ValueError`` 的情形有**两**类，都必须收敛成 ``None``：

    1. 非十进制数字字符。判据用 ``str.isdecimal()`` 而不是 ``str.isdigit()``——
       后者对 128 个码位为真（``²`` ``①`` ``፩`` ``₅`` …）却让 ``int()`` 抛错。
    2. 位数超过 ``sys.get_int_max_str_digits()``（CPython 默认 4300 位）。
       ``"9" * 4301`` 是**合法**的十进制整数，``isdecimal()`` 也为真，但
       ``int()`` 照样抛 ``ValueError: Exceeds the limit (4300 digits)``。

    只堵住第 1 类是不够的：调用方拿到的仍然是一个可能冒到 prompt_toolkit
    事件循环或错误分类逻辑里的异常。这里用 ``try/except`` 一次性覆盖两类，
    不需要硬编码位数上界——``int()`` 抛 ``ValueError`` 本身就是判据。

    返回值是 ``int | None``，调用方据此分支，不必再包一层 ``try``。
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    digits = candidate.removeprefix("-")
    if not digits or not digits.isdecimal():
        return None
    try:
        return int(candidate)
    except ValueError:
        return None


# ``float()`` 与 ``math.isfinite()`` 对超出 C double 范围的整数都会先尝试转换，
# 失败时抛 ``OverflowError: int too large to convert to float``。
#
# ``OverflowError`` 继承 ``ArithmeticError``，**不是** ``ValueError``。本项目
# 的数值校验统一以 ``ValueError`` 作为契约（``Settings`` 的 validator、
# ``SessionConfig`` 的 ``__setitem__``、各 provider 的 options 校验、
# ``get_settings`` 的 ``except ValueError`` 兜底、questionary 的 validator），
# 因此 ``OverflowError`` 会穿透**全部**既有守卫，冒到解释器默认 handler。
#
# 判据不能写成 ``math.isfinite(x)``：它自己就会抛 ``OverflowError``（实测
# ``math.isfinite(10**400)`` 抛错），拿它当守卫等于没有守卫。这里先把
# 「是不是浮点」与「绝对值是否超出双精度」分开判，两者都过了才安全地调
# ``float()``。用 ``sys.float_info.max`` 做上界而不是硬编码字面量，避免在
# 极端平台上与 C double 的实际范围脱节。
_FLOAT_RANGE_BOUND = sys.float_info.max


def is_double_convertible(value: Any) -> bool:
    """``value`` 能否安全地转成 C double（即 ``float(value)`` 不会抛异常）。

    对 ``int`` 判绝对值上界，对 ``float`` 判有限性，其余类型一律返回 ``False``
    ——它们要么转换不了（``str``/``None``/容器），要么是 ``bool``（``bool`` 是
    ``int`` 的子类，``float(True) == 1.0`` 会静默变成 1.0，调用方必须显式拒绝）。
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return -_FLOAT_RANGE_BOUND <= value <= _FLOAT_RANGE_BOUND
    return False


def safe_repr(value: Any, *, limit: int = 80) -> str:
    """``repr(value)`` 的安全版本，供错误消息使用。

    ``repr`` 自己会抛异常：CPython 的 ``sys.get_int_max_str_digits()``（默认
    4300）让 ``repr(10**4301)`` 抛 ``ValueError: Exceeds the limit (4300
    digits)``。校验代码把它拼进消息时，用户看到的是「整数转字符串超限」，
    而真实原因是「数值非法」——错误消息反过来把异常类型也换掉了
    （``ValueError`` 变 ``ValueError``，但内容与字段毫无关系）。

    超长输出也一并截断：``repr(10**400)`` 是 401 个字符，一条校验消息不该
    刷屏。截断只影响显示，判据在调用方，因此不影响正确性。
    """
    try:
        text = repr(value)
    except (ValueError, RuntimeError, RecursionError):
        # 走到这里只有一种已知情形：整数的十进制位数超过
        # ``sys.get_int_max_str_digits()``。``bit_length()`` 不经过十进制
        # 转换，可以安全地给出量级。
        if isinstance(value, int):
            digits = value.bit_length() * 30103 // 100000 + 1  # log10(2) ≈ 0.30103
            return f"<整数，约 {digits} 位十进制>"
        return f"<{type(value).__name__}>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


_DETAIL_ATTRIBUTE_MISSING = object()


def require_detail_attributes(
    detail: Any,
    attributes: tuple[str, ...],
    *,
    field: str,
    key: Any,
    role_label: str,
) -> None:
    """校验模型配置的**取值**具备使用方要读的属性，否则抛可读的 ``ValueError``。

    ``llm_configurations``／``rerank_configurations`` 声明为
    ``dict[str, ModelDetail]``，但 ``SessionConfig`` 的 ``_VALIDATORS`` 明确不登记
    它们（注释写明「按原样写入，由使用方负责其内部结构」）。于是 ``key in
    mapping`` 只证明**键**在，不证明**值**可用：``{"k": None}`` 的键确实存在，
    取值被原样交给使用方，调用方的 ``config.provider`` 抛裸 ``AttributeError``
    ——它不是 ``ValueError``，会穿透调用方按 ``ValueError`` 做的错误分类。

    判据用**属性存在性**而不是 ``isinstance(detail, ModelDetail)``：使用方只要求
    取值提供这几个属性，任何具备它们的对象都能正常工作（仓库里的测试就用
    ``SimpleNamespace`` 构造等价配置）。按具体类型收窄会把可用的配置一并拒掉。

    属性存在但取值为 ``None`` 不在这里判——``None`` 是否能用作 provider 名、
    模型名由各使用方按自己的判据处理。
    """
    for attribute in attributes:
        if getattr(detail, attribute, _DETAIL_ATTRIBUTE_MISSING) is _DETAIL_ATTRIBUTE_MISSING:
            raise ValueError(
                f"{field} 中 {safe_repr(key)} 的取值必须是提供 {attribute} 属性的模型配置，"
                f"但得到 {type(detail).__name__}。 请检查 {role_label} 的当前配置项。"
            )


def require_retrieval_method(value: Any, *, label: str = "retrieval_method") -> RetrievalMethod:
    """把检索模式收敛成 ``RetrievalMethod`` 成员，非法取值给出带指引的 ``ValueError``。

    ``retrieval_method`` 声明为 ``RetrievalMethod``，但 ``SessionConfig`` 是公开类、
    普通属性没有 ``__setattr__`` 守卫，嵌入方（或携带异常取值的 pickle 快照）可以
    把它直写成任何形态。此前有三个取用点各自假设它是枚举成员：

    * ``retrieval_service.retrieve`` 用 ``==`` 判分支，且**没有任何形态收敛**——它是
      该函数里唯一没有 ``_coerce_*`` 助手的字段（``top_k``／候选倍率／
      ``score_threshold``／``rerank_enabled``／``hybrid_fusion_strategy`` 都有）。
      实测 ``cfg.retrieval_method = "hybrid"``／``None``／``5`` 时静默落进 else
      分支，变成**纯关键词检索**；只有恰好等于 ``"向量检索"`` 的取值才走语义检索。
      用户看到的是「混合检索配好了但只召回关键词命中」，而不是一条配置错误。
    * ``display_chat_config`` 读 ``.value``——它在 ``start_chat_session_async`` 里于
      任何提示符**之前**被调用，直写 ``5``／``None`` 时抛裸
      ``AttributeError: 'int' object has no attribute 'value'``，会话在启动横幅
      这一步就终止。
    * ``config_menu.edit_retrieval_params`` 读同一个 ``.value``，菜单渲染不出来。

    接受三类形态，判据是**属性存在性**而不是具体类型（与 ``require_detail_attributes``
    同一约定——使用方只要求取值提供 ``.value``，任何具备它的对象都能正常工作）：

    * ``RetrievalMethod`` 成员；
    * 它的**字符串取值**（``"混合检索"``）——本项目的既有契约，``config.toml`` 的
      ``chat_retrieval_method`` 与 ``Settings`` 走的就是这条路径，而
      ``RetrievalMethod`` 本身是 ``str`` 枚举，两种形态在 ``==`` 下等价；
    * 任何提供 ``.value`` 的等价替身。仓库里的测试用 ``SimpleNamespace(value="混合检索")``
      构造这种替身，按类型收窄会把它们一并拒掉。

    不做大小写／别名归一：``"hybrid"`` 不是合法取值，静默接受它等于把「配置写错了」
    变成「检索行为悄悄变了」。

    ``Enum`` 查找对不可哈希取值抛 ``ValueError``（``['a']``、``{'a': 1}`` 实测），
    但位数超限的整数抛的 ``ValueError`` 消息是 ``Exceeds the limit (4300 digits)``
    ——与「检索模式不合法」无关，因此先做一次形态收敛，把消息统一成带可选值的形态。
    """
    if isinstance(value, RetrievalMethod):
        return value
    candidate: Any = value
    if not isinstance(candidate, str):
        # ``getattr`` 而非 ``.value``：非字符串取值（``5``／``None``／列表）没有该
        # 属性，直接取会抛裸 ``AttributeError``。替身对象的 ``.value`` 也可能是
        # property（可以抛），因此整体包在 ``try`` 里。
        try:
            candidate = getattr(value, "value", _DETAIL_ATTRIBUTE_MISSING)
        except Exception:  # noqa: BLE001 - 取值不可用等同于「不是合法的检索模式」
            candidate = _DETAIL_ATTRIBUTE_MISSING
    if isinstance(candidate, str):
        try:
            return RetrievalMethod(candidate)
        except ValueError:
            pass
    allowed = ", ".join(member.value for member in RetrievalMethod)
    raise ValueError(f"{label} 必须是 {allowed} 之一，但得到 {safe_repr(value)}。")


def require_chat_temperature(value: Any) -> float:
    """把 ``chat_temperature`` 收敛成非负有限浮点，非法取值给出带字段名的 ``ValueError``。

    ``chat_temperature`` 有**完整**的校验器——``Settings.validate_chat_temperature``
    与 ``SessionConfig._validate_chat_temperature`` 各自一份，判据一致（拒 ``bool``、
    非数值、非有限、负数、超出双精度的整数）。但两个**读取点**都直读原属性，而
    ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫：

    * ``ChatService.generate_reply``：``session_config.chat_temperature`` 直接进
      ``CompletionRequest``，再原样成为 SDK 的 kwargs。实测 ``"abc"``／``None``／
      ``[0.5]``／``True``／``-1`` 全部**静默**通过——而它们正是
      ``SessionConfig["chat_temperature"] = ...`` 会明确拒绝的取值。同一个字段
      「写得进就合法、直写就非法」两种口径。非法值一路到服务端，用户看到的是
      接口返回的参数错误（或更糟：``-1`` 被某些网关当默认值接受），而不是一条
      本地配置错误。
    * ``Chatbot.chat_async`` 的兜底分支（``chat_service`` 缺失时）读
      ``self.session_config["chat_temperature"]``，同一个值。

    收敛语义与两处校验器一致（返回 ``float``，非负且有限）。与
    ``_validate_chat_temperature`` 的分工：那里是配置边界的**拒绝**，这里是运行期
    读取可直写字段时的**收敛**——两者共用同一判据，措辞也一致。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"chat_temperature 必须是数值，但得到 {safe_repr(value)}。")
    if not is_double_convertible(value):
        raise ValueError(f"chat_temperature 必须是非负有限数值，但得到 {safe_repr(value)}。")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"chat_temperature 必须是非负有限数值，但得到 {safe_repr(value)}。")
    return number


def to_finite_float(value: Any) -> float | None:
    """把数值收敛成有限 ``float``；不能安全转换时返回 ``None``，**绝不抛异常**。

    接受 ``float()`` 原本接受的三类输入：``int``（含超出双精度范围的，此时返回
    ``None`` 而不是抛 ``OverflowError``）、``float``（非有限返回 ``None``）、以及
    数字字符串。字符串必须继续接受——调用方此前走的就是 ``float(v)``，
    ``.env``/TOML 来源本来就以字符串形态到达。

    返回 ``None`` 的语义是「这个值不能当有限浮点用」，调用方据此抛自己的
    ``ValueError``（各处的字段名与措辞不同，消息由调用方决定）。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if not is_double_convertible(value):
        return None
    return float(value)


def _coerce_number(value: Any, field_label: str) -> int | float:
    """把 ``mode="before"`` 收到的原始输入转成数值。

    ``mode="before"`` 的 validator 拿到的是**未经 pydantic 强转**的原始值：
    ``.env``/TOML 来源是字符串（``"5"``），Python 调用方可能是 ``int``/``float``，
    而 ``bool`` 是 ``int`` 的子类（``int(True) == 1``）必须显式拒绝，否则
    ``Settings(chat_top_k=True)`` 会静默变成 ``1``。

    字符串只接受十进制数值字面量：``"1500"``、``"0.4"``、``"1e3"``。不接受
    ``"0x10"``（十六进制）与 ``"inf"``/``"nan"``（非有限值由各 validator 的
    ``math.isfinite`` 或上下界负责，但让 ``float()`` 接住它们会得到
    ``inf`` 这类「成功解析出的非法值」，不如直接在这里当解析失败处理）。
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_label} 不能是布尔值 {safe_repr(value)}。")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        parsed_int = parse_decimal_int(text)
        if parsed_int is not None:
            return parsed_int
        if not _DECIMAL_LITERAL_PATTERN.fullmatch(text):
            raise ValueError(f"无法把 {field_label} 的 {safe_repr(value)} 转换为数值。")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"无法把 {field_label} 的 {safe_repr(value)} 转换为数值。") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_label} 必须是有限数值，但得到 {safe_repr(value)}。")
        return number
    raise ValueError(f"{field_label} 必须是数值，但得到 {type(value).__name__}。")


class Settings(BaseSettings):
    """
    定义整个应用的配置，使用Pydantic进行类型校验和分层加载。
    加载顺序: 环境变量 > .env 文件 > config.toml 文件 > 模型中定义的默认值。
    """

    # --- [API_KEYS] ---
    anthropic_api_key: str | None = Field(default=None, repr=False)
    google_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)
    google_genai_use_vertexai: bool | None = Field(default=None)
    google_genai_use_enterprise: bool | None = Field(default=None)
    google_cloud_project: str | None = Field(default=None)
    google_cloud_location: str | None = Field(default=None)
    google_application_credentials: str | None = Field(default=None, repr=False)
    siliconflow_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    qwen_api_key: str | None = Field(default=None, repr=False)
    ark_api_key: str | None = Field(default=None, repr=False)
    volc_access_key: str | None = Field(default=None, repr=False)
    volc_secret_key: str | None = Field(default=None, repr=False)
    jina_api_key: str | None = Field(default=None, repr=False)
    deepseek_api_key: str | None = Field(default=None, repr=False)
    grok_api_key: str | None = Field(default=None, repr=False)
    lm_studio_api_key: str | None = Field(default=None, repr=False)

    _secret_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "anthropic_api_key",
            "google_api_key",
            "siliconflow_api_key",
            "gemini_api_key",
            "google_application_credentials",
            "openai_api_key",
            "qwen_api_key",
            "ark_api_key",
            "volc_access_key",
            "volc_secret_key",
            "jina_api_key",
            "deepseek_api_key",
            "grok_api_key",
            "lm_studio_api_key",
        }
    )

    def _safe_dump_exclude(self, exclude: Any, include_secrets: bool) -> Any:
        if include_secrets:
            return exclude
        secret_exclude = set(self._secret_fields)
        if exclude is None:
            return secret_exclude
        if isinstance(exclude, Mapping):
            return {**exclude, **{name: True for name in secret_exclude}}
        return set(exclude) | secret_exclude

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        include_secrets = bool(kwargs.pop("include_secrets", False))
        exclude = self._safe_dump_exclude(kwargs.pop("exclude", None), include_secrets)
        if exclude is not None:
            kwargs["exclude"] = exclude
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        include_secrets = bool(kwargs.pop("include_secrets", False))
        exclude = self._safe_dump_exclude(kwargs.pop("exclude", None), include_secrets)
        if exclude is not None:
            kwargs["exclude"] = exclude
        return super().model_dump_json(*args, **kwargs)

    # --- [BASE_URLS] ---
    openai_api_base: str = "https://api.openai.com/v1"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    deepseek_base_url: str = "https://api.deepseek.com"
    ollama_base_url: str = "http://localhost:11434/v1"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    volc_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    grok_base_url: str = "https://api.x.ai/v1"

    # --- [GENERAL] ---
    log_level: str = "WARNING"  # 新增 log_level 字段，默认级别调整为 WARNING
    cache_path: str = ".cache"  # 新增 cache_path 字段
    log_path: str = "data/logs"
    log_retention_days: int = 15

    # --- [PATHS] ---
    knowledge_base_path: str = "knowledge_base"
    pkl_path: str = "data/employee_kb.pkl"
    snapshot_root: str = "data/kb"

    # --- [KNOWLEDGE_BASE] ---
    kb_replace_whitespace: bool = False
    kb_remove_spaces: bool = False
    kb_remove_urls: bool = False
    kb_use_qa_segmentation: bool = False
    kb_splitter_separators: list[str] = Field(default_factory=lambda: ["###"])
    kb_chunk_size: int = 1500
    kb_chunk_overlap: int = 150
    kb_child_chunk_size: int = 300
    kb_child_chunk_overlap: int = 30
    kb_embedding_batch_size: int = 32

    # --- [BEHAVIOR] ---
    default_llm_provider: str = "google"
    default_embedding_provider: str = "local-hash"
    default_rerank_provider: str = "siliconflow"
    default_vector_store: str = "faiss"  # 新增向量存储默认提供商

    # --- [CHAT] ---
    chat_retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_SEARCH
    chat_vector_weight: float = 0.3
    chat_keyword_weight: float = 0.7
    hybrid_fusion_strategy: str = "rrf"
    retrieval_candidate_multiplier: int = 3
    chat_rerank_enabled: bool = False
    chat_top_k: int = 5
    chat_score_threshold: float = 0.4
    chat_temperature: float = 0.7  # 将 chat_temperature 移到这里

    # --- [MODEL_CONFIGURATIONS] ---
    embedding_configurations: dict[str, ModelDetail] = Field(
        default_factory=lambda: {
            # 与 llm_configurations 同理：兜底值必须写成当前有效的官方 ID。
            "local-hash": ModelDetail(provider="local-hash", model_name="local-hash-256"),
            "google": ModelDetail(provider="google", model_name="gemini-embedding-2"),
            "siliconflow": ModelDetail(provider="siliconflow", model_name="BAAI/bge-large-zh-v1.5"),
            "openai": ModelDetail(provider="openai", model_name="text-embedding-3-small"),
        }
    )
    rerank_configurations: dict[str, ModelDetail] = Field(
        default_factory=lambda: {
            "siliconflow": ModelDetail(
                provider="siliconflow", model_name="BAAI/bge-reranker-v2-m3"
            ),
        }
    )
    llm_configurations: dict[str, ModelDetail] = Field(
        default_factory=lambda: {
            # 默认条目只作为缺失配置时的兜底；模型名保持在写就时仍可用的现行 ID，
            # 避免新用户照抄到已退役模型。
            "google": ModelDetail(provider="google", model_name="gemini-2.5-flash"),
            "anthropic": ModelDetail(provider="anthropic", model_name="claude-sonnet-4-6"),
            "qwen": ModelDetail(provider="qwen", model_name="qwen3.8-max"),
            "deepseek": ModelDetail(provider="deepseek", model_name="deepseek-v4-pro"),
            "grok": ModelDetail(provider="grok", model_name="grok-4.6"),
            "volcengine": ModelDetail(
                provider="volcengine", model_name="doubao-seed-2-0-lite-260428"
            ),
            "siliconflow": ModelDetail(
                provider="siliconflow", model_name="deepseek-ai/DeepSeek-V3.2"
            ),
            "openai": ModelDetail(provider="openai", model_name="gpt-5.6-sol"),
            "ollama": ModelDetail(provider="ollama", model_name="llama3.1"),
            "lm-studio": ModelDetail(
                provider="lm-studio", model_name="LM-Studio-Community/Meta-Llama-3-8B-Instruct-GGUF"
            ),
        }
    )

    # --- [VALIDATORS] ---
    @field_validator("chat_top_k", mode="before")
    @classmethod
    def validate_chat_top_k(cls, value: Any) -> int:
        # 上界不是洁癖：effective_top_k = chat_top_k * retrieval_candidate_multiplier
        # 会直通 faiss_index.search()，FAISS 不报错也不截断。实测 chat_top_k=10**9
        # 时单次查询真实分配约 3.6 GB 数组、RSS 涨 10.3 GB 并挂起 8.4 秒。
        number = _coerce_number(value, "chat_top_k")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError(f"chat_top_k 必须是整数，但得到 {number}。")
        number = int(number)
        if not 1 <= number <= _MAX_RETRIEVAL_TOP_K:
            raise ValueError(f"chat_top_k 必须是 1 到 {_MAX_RETRIEVAL_TOP_K} 之间的整数。")
        return number

    @field_validator("chat_score_threshold", mode="before")
    @classmethod
    def validate_chat_score_threshold(cls, value: Any) -> float:
        number = _coerce_number(value, "chat_score_threshold")
        # 先做范围判断再转 float：``float(10**400)`` 抛的是裸 ``OverflowError``，
        # 它不是 ``ValidationError``，会穿透 ``get_settings`` 的 ``except`` 变成
        # 未脱敏的 traceback。范围判断对任意大的 int 都能给出正常结论。
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise ValueError(f"chat_score_threshold 必须是数值，但得到 {safe_repr(value)}。")
        if not 0 <= number <= 1:
            raise ValueError(
                f"chat_score_threshold 必须在 0 到 1 之间，但得到 {safe_repr(value)}。"
            )
        return float(number)

    @field_validator("log_retention_days", mode="before")
    @classmethod
    def validate_log_retention_days(cls, value: Any) -> int:
        number = _coerce_number(value, "log_retention_days")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError(f"log_retention_days 必须是整数，但得到 {number}。")
        number = int(number)
        # 上界不是洁癖：无上界时 10**400 会被接受（实测），而它接下来会被交给
        # 日期运算做 ``today - timedelta(days=N)``，直接 OverflowError。
        if not 1 <= number <= _MAX_LOG_RETENTION_DAYS:
            raise ValueError(
                f"log_retention_days 必须是 1 到 {_MAX_LOG_RETENTION_DAYS} 之间的整数。"
            )
        return number

    @field_validator(
        "kb_chunk_size",
        "kb_child_chunk_size",
        "kb_embedding_batch_size",
        mode="before",
    )
    @classmethod
    def validate_positive_sizes(cls, value: Any) -> int:
        number = _coerce_number(value, "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError(
                "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size 必须是整数。"
            )
        number = int(number)
        # 上界不是洁癖：无上界时 10**400 会被接受（实测），随后分片/批处理会把它
        # 当成真实规模去申请内存。tiktoken 编码前的分片上限取 100 万 token，比任何
        # 现实文档都宽，但仍挡住溢出量级。
        if not 1 <= number <= _MAX_KB_CHUNK_TOKENS:
            raise ValueError(
                "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size"
                f" 必须是 1 到 {_MAX_KB_CHUNK_TOKENS} 之间的整数。"
            )
        return number

    @field_validator("kb_chunk_overlap", "kb_child_chunk_overlap", mode="before")
    @classmethod
    def validate_non_negative_overlap(cls, value: Any) -> int:
        """overlap 必须落在 ``[0, _MAX_KB_CHUNK_TOKENS)``。

        上界与 ``kb_chunk_size`` 同源而不是各写一个数：overlap 的语义是「相邻
        分片重叠的 token 数」，它必须**小于** chunk_size，而 chunk_size 的上界
        就是 ``_MAX_KB_CHUNK_TOKENS``，因此这里取同一个常量作上界。

        缺上界是**配置可达**的：``config.toml`` 写
        ``kb_child_chunk_overlap = 10**400``（401 位，``tomllib`` 照常解析）会被
        ``Settings`` 静默接受（实测），配置加载成功，直到分片期才在
        ``RecursiveTextSplitter._validate_hierarchical_sizes`` 抛错——报错现场
        离写错的配置很远，而且消息里会打出整个 401 位数字。上界把这类错误拉回
        配置加载期，与同文件其它数值字段（``kb_chunk_size``／``chat_top_k``／
        ``retrieval_candidate_multiplier``）一致。
        """
        number = _coerce_number(value, "kb_chunk_overlap/kb_child_chunk_overlap")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError("kb_chunk_overlap/kb_child_chunk_overlap 必须是整数。")
        number = int(number)
        if number < 0:
            raise ValueError("kb_chunk_overlap/kb_child_chunk_overlap 必须是非负整数。")
        if number >= _MAX_KB_CHUNK_TOKENS:
            raise ValueError(
                "kb_chunk_overlap/kb_child_chunk_overlap 必须小于 "
                f"{_MAX_KB_CHUNK_TOKENS}（分片重叠必须小于分片上限）。"
            )
        return number

    @field_validator("chat_vector_weight", "chat_keyword_weight", mode="before")
    @classmethod
    def validate_hybrid_weights(cls, value: Any) -> float:
        # ``float(10**400)`` 抛 ``OverflowError``（**不是** ``ValueError``），
        # 会穿透 ``get_settings`` 的全部三个 handler。``to_finite_float`` 内部
        # 先判双精度范围，越界时返回 ``None``。
        number = to_finite_float(_coerce_number(value, "chat_vector_weight/chat_keyword_weight"))
        if number is None:
            raise ValueError(
                "chat_vector_weight/chat_keyword_weight 必须是有限数值，"
                f"但得到 {safe_repr(value)}。"
            )
        if not 0 <= number <= 1:
            raise ValueError("chat_vector_weight/chat_keyword_weight 必须在 0 到 1 之间。")
        return number

    @model_validator(mode="after")
    def validate_retrieval_size_bounds(self) -> "Settings":
        """检索规模必须有上界，否则单次查询就能耗尽内存。"""
        if self.retrieval_candidate_multiplier > _MAX_RETRIEVAL_MULTIPLIER:
            raise ValueError(
                f"retrieval_candidate_multiplier 必须不超过 {_MAX_RETRIEVAL_MULTIPLIER}。"
            )
        # 两个字段各自合法，乘积仍可能失控：10_000 × 100 = 10**6 会让 FAISS
        # 按 100 万条分配（实测 dim=1536 时峰值 RSS 约 35 GB）。约束必须落在
        # 乘积上——这才是真正送进 search() 的值。
        candidates = self.chat_top_k * self.retrieval_candidate_multiplier
        if candidates > _MAX_RETRIEVAL_CANDIDATES:
            raise ValueError(
                f"chat_top_k × retrieval_candidate_multiplier 不能超过 "
                f"{_MAX_RETRIEVAL_CANDIDATES}，但得到 {candidates}。"
                " 两者相乘才是实际送入向量检索的候选规模，过大会一次性申请巨量内存。"
            )
        return self

    @model_validator(mode="after")
    def validate_overlap_smaller_than_size(self) -> "Settings":
        """overlap 必须严格小于 size，且不得超过 size 的一半。

        ``chunk_size == chunk_overlap`` 会让切分器无法前进（得到空分片或
        死循环），且这一约束无法用单字段 validator 表达。

        比例上界（``overlap <= size / 2``）拦的是另一类失败：LangChain 的
        ``RecursiveCharacterTextSplitter`` 只在 ``overlap > size`` 时报错，
        **接近** size 时静默接受，而每个分片只前进 ``size - overlap`` 个字符，
        冗余随比例急剧放大。实测一篇 15000 字符的文档：

        ==========  ==========  ==========
        overlap     分片数      放大倍数
        ==========  ==========  ==========
        10%          14           1.1x
        50%          23           1.9x
        80%          56           4.7x
        99%          1651       137.6x
        99.9%        13751     1145.9x
        ==========  ==========  ==========

        这些分片全部进入 embedding 与 FAISS 写入，是一次「配置看着正常、代价
        上千倍」的静默放大。50% 是主流 RAG 实践的通行上界（它同时保证每个
        分片至少有一半是新内容），且实测在该点上放大仅 1.9x。
        """
        if self.kb_chunk_overlap >= self.kb_chunk_size:
            raise ValueError("kb_chunk_overlap 必须小于 kb_chunk_size。")
        if self.kb_chunk_overlap * 2 > self.kb_chunk_size:
            raise ValueError(
                "kb_chunk_overlap 不能超过 kb_chunk_size 的一半"
                f"（得到 overlap={self.kb_chunk_overlap}、size={self.kb_chunk_size}）。"
                " 重叠比例过高会让分片器几乎无法推进，产出成百上千倍的冗余分片。"
            )
        # 子分片的两条关系（``child_overlap < child_size``、
        # ``child_size < chunk_size``）**不在这里判**：它们只在层级模式下有
        # 意义，而模式是分片时的运行时参数（``build(splitter_structure_mode)``／
        # CLI 的 ``--mode``），``Settings`` 看不到。无条件拒绝会让一组
        # 「standard 下完全合法」的取值连配置都加载不了——用户只想用 standard，
        # 却必须先改掉用不到的字段（实测 ``kb_chunk_size=100,
        # kb_child_chunk_size=300`` 被拒，而 standard 模式根本不构造子分片器）。
        #
        # 判定移到 ``RecursiveTextSplitter._validate_hierarchical_sizes()``，
        # 那里模式已知、且与真正使用这些参数的位置一致。
        return self

    @model_validator(mode="after")
    def validate_hybrid_weights_not_both_zero(self) -> "Settings":
        """两个融合权重不得同时为 0，否则检索结果被静默全部丢弃。

        ``retrieval_service._weight_tuple()`` 在 ``total <= 0`` 时返回
        ``(0.0, 0.0)``，所有候选得分为 0；再叠加 ``weighted`` 策略会启用阈值
        过滤（默认 ``chat_score_threshold=0.4``），结果是空列表且无任何报错。
        单个为 0 是合法的（纯向量 / 纯关键词检索）。
        """
        if self.chat_vector_weight == 0 and self.chat_keyword_weight == 0:
            raise ValueError(
                "chat_vector_weight 与 chat_keyword_weight 不能同时为 0，否则检索结果会被全部丢弃。"
            )
        return self

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> str:
        """验证日志级别是否有效。

        ``config.toml`` 的 TOML 值可以是任意类型，因此必须先确认是字符串再调
        ``.upper()``：原实现直接 ``v.upper()``，``log_level = 123`` 抛的是裸
        ``AttributeError: 'int' object has no attribute 'upper'``——它不是
        ``ValueError`` 的子类，会穿透 ``get_settings`` 的全部三个 handler。这个
        字段尤其要紧：它在**模块导入期**就被 ``log_manager.get_module_logger``
        读取，早于任何入口的 ``try``，连脱敏边界都还没建立。
        """
        if not isinstance(v, str):
            raise ValueError(f"log_level 必须是字符串，但得到 {safe_repr(v)}。")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}. 必须是 {', '.join(valid_levels)} 中的一个。")
        return v.upper()

    @field_validator("embedding_configurations", "rerank_configurations")
    @classmethod
    def validate_non_llm_protocol(
        cls,
        value: dict[str, ModelDetail],
        info: ValidationInfo,
    ) -> dict[str, ModelDetail]:
        """Embedding/Rerank 配置不允许携带 LLM 线协议。"""
        invalid = sorted(key for key, detail in value.items() if detail.protocol is not None)
        if invalid:
            role = "Embedding" if info.field_name == "embedding_configurations" else "Rerank"
            raise ValueError(
                f"{role} 配置不支持 protocol；协议只能配置在 llm_configurations 中。"
                f" 无效条目: {', '.join(invalid)}"
            )
        return value

    @field_validator("chat_temperature", mode="before")
    @classmethod
    def validate_chat_temperature(cls, v: Any) -> float:
        """验证聊天温度在 0.0 到 1.0 之间。"""
        # ``bool`` 是 ``int`` 的子类，``float(True) == 1.0`` 会被静默接受。
        if isinstance(v, bool):
            raise ValueError(f"聊天温度必须是数字，不能是布尔值 {safe_repr(v)}。")
        # ``float(10**400)`` 抛的是 ``OverflowError``——它**不是** ``ValueError``，
        # 因此下面这个 ``except`` 接不住，裸异常会穿透 ``get_settings`` 的
        # 全部三个 handler。判据用 ``to_finite_float``（内部先判双精度范围）。
        value = to_finite_float(v)
        if value is None:
            raise ValueError(f"无法将聊天温度 {safe_repr(v)} 转换为数字。")

        if not (0.0 <= value <= 1.0):
            raise ValueError(f"聊天温度必须在 0.0 到 1.0 之间，但得到 {value}。")
        return value

    @field_validator("hybrid_fusion_strategy", mode="before")
    @classmethod
    def validate_hybrid_fusion_strategy(cls, v: Any) -> str:
        """验证混合检索融合策略。"""
        if not isinstance(v, str):
            raise ValueError("混合检索融合策略必须是字符串。")

        normalized = v.strip().lower()
        valid_strategies = {"rrf", "weighted"}
        if normalized not in valid_strategies:
            raise ValueError(
                f"无效的混合检索融合策略: {v}. 必须是 {', '.join(sorted(valid_strategies))}。"
            )
        return normalized

    @model_validator(mode="after")
    def warn_retired_base_urls(self) -> "Settings":
        """对已知失效的旧 base URL 发出显式升级警告。

        1.4.0 把 Qwen 默认端点从 ``dashscope.aliyuncs.com/api/v1`` 换成
        ``compatible-mode/v1``，并把火山默认域名从
        ``maas-api.ml-platform-cn-beijing.volces.com`` 换成
        ``ark.cn-beijing.volces.com/api/v3``。旧的 ``config.toml`` 会覆盖新默认值，
        使请求在运行期失败。这里在配置加载时就明确提示如何修正，而不是让用户从
        错误响应反推原因；显式配置的自建或代理端点不受影响。

        两处旧值的失效方式不同：DashScope 原生 ``/api/v1`` 本身仍在服务，但路径是
        ``/services/aigc/text-generation/generation``，而本项目按 OpenAI 兼容协议请求
        ``{base_url}/chat/completions``，因此该值在本项目内不可用；火山旧域名则确实
        已下线。

        只警告不阻断：旧值是可修复的配置问题而非安全边界，直接失败会让仍在使用
        旧配置的部署完全无法启动。
        """
        retired = {
            "qwen_base_url": (
                "https://dashscope.aliyuncs.com/api/v1",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "该端点不提供 {base_url}/chat/completions",
            ),
            "volc_base_url": (
                "https://maas-api.ml-platform-cn-beijing.volces.com",
                "https://ark.cn-beijing.volces.com/api/v3",
                "旧域名已停止服务",
            ),
        }
        for field_name, (old_url, new_url, reason) in retired.items():
            current = getattr(self, field_name, None)
            if isinstance(current, str) and current.rstrip("/") == old_url:
                warnings.warn(
                    f"{field_name} 指向旧端点 {old_url}（{reason}），请改为 {new_url}。",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    @field_validator("retrieval_candidate_multiplier", mode="before")
    @classmethod
    def validate_retrieval_candidate_multiplier(cls, v: Any) -> int:
        """验证检索候选过量招募倍率。"""
        # ``bool`` 是 ``int`` 的子类，``int(True) == 1`` 会被静默接受。
        if isinstance(v, bool):
            raise ValueError(f"检索候选倍率必须是整数，不能是布尔值 {safe_repr(v)}。")
        # ``inf`` 有**两条**独立入口，缺任何一条都会让程序以裸 traceback 退出：
        #
        # 1. ``config.toml`` 写 ``retrieval_candidate_multiplier = inf``——合法
        #    TOML，``tomllib`` 原样解析成 ``float('inf')``。下面这条 ``isfinite``
        #    守卫在第一行就把它挡住，给出「必须是整数，但得到 inf」这种能直接定位
        #    到字段的消息；同文件的 ``chat_top_k``／``log_retention_days``／
        #    ``kb_chunk_size`` 都有等价的前置守卫。
        # 2. 其它非有限数值类型（``Decimal('Infinity')`` 等，只能从 Python 调用方
        #    传入）。它们不是 ``float``，走不进上面那条守卫，由 ``int()`` 抛
        #    ``OverflowError``——它是 ``ArithmeticError`` 子类而**不是**
        #    ``ValueError``，必须在下面显式列出，否则穿透 ``get_settings`` 的全部
        #    三个 handler。
        #
        # 两条都必要：删掉守卫会让 ``inf`` 的报错变成泛化的转换失败；删掉
        # ``OverflowError`` 会让第二条路径裸崩。两条路径各有回归测试。
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"检索候选倍率必须是整数，但得到 {safe_repr(v)}。")
        # 与同文件的 ``chat_top_k``／``log_retention_days``／``kb_chunk_size``
        # 用同一套解析：走 ``_coerce_number``（内部用 ``parse_decimal_int``，
        # 只认纯十进制字面量）。此前这里直接 ``int(v)``，与其它整数字段**判据
        # 不一致**，两条后果都实测复现：
        #
        # * ``1.9`` 被静默截断成 ``1``（``chat_top_k=1.9`` 会被拒绝）；
        # * ``"1_0"`` 被 ``int()`` 接受为 10、``"+7"`` 被接受为 7——而
        #   ``parse_decimal_int`` 的 docstring 明确写着这两类「安静的错值」
        #   「正是本项目反复踩到的模式，宁愿在这里明确拒绝」。
        number = _coerce_number(v, "检索候选倍率")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError(f"检索候选倍率必须是整数，但得到 {safe_repr(v)}。")
        try:
            value = int(number)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"无法将检索候选倍率 '{safe_repr(v)}' 转换为整数。") from exc

        if value < 1:
            raise ValueError(f"检索候选倍率必须大于等于 1，但得到 {safe_repr(value)}。")
        return value

    @field_validator("kb_splitter_separators", mode="before")
    @classmethod
    def split_separators(cls, v: Any) -> list[str]:
        """
        如果分隔符是字符串，则按逗号分割成列表。
        如果输入值为空（None或空字符串），或者分割后为空列表，则使用字段的默认值。
        """
        field_info = cls.model_fields["kb_splitter_separators"]
        default_factory = field_info.default_factory
        if default_factory is not None:
            default_value = cast(Callable[[], Any], default_factory)()
        else:
            default_value = field_info.default
            if default_value is None:
                default_value = []

        # 如果输入为空（来自 .env 或环境变量的空字符串），则回退到默认值
        if v is None or v == "":
            return default_value

        if isinstance(v, str):
            # 按逗号分割，并过滤掉空的元素
            separators = [s.strip() for s in v.split(",") if s.strip()]
            # 如果分割后列表为空（例如，输入是" , "），也使用默认值
            return separators if separators else default_value

        # 如果输入已经是列表或其他类型，直接返回
        return v

    @field_validator("chat_retrieval_method", mode="before")
    @classmethod
    def validate_retrieval_method(cls, v: Any) -> Any:
        """允许使用枚举的键名（如HYBRID_SEARCH）或值（如'混合检索'）进行配置。"""
        if isinstance(v, str):
            # 尝试匹配枚举的键名 (e.g., "HYBRID_SEARCH")
            if v.upper() in RetrievalMethod.__members__:
                return RetrievalMethod[v.upper()]
            # 尝试匹配枚举的值 (e.g., "混合检索")
            for member in RetrievalMethod:
                if member.value == v:
                    return member
        # 如果已经是枚举成员或无法转换，则让默认验证器处理
        return v

    @field_validator(
        "knowledge_base_path", "pkl_path", "snapshot_root", "log_path", "cache_path", mode="before"
    )
    @classmethod
    def resolve_path(cls, v: Any, info: ValidationInfo) -> str:
        """将相对路径解析为绝对路径。

        非字符串必须显式拒绝：``Path(123)`` 抛的是裸
        ``TypeError: expected str, bytes or os.PathLike object, not int``，
        而 ``TypeError`` **不是** ``ValueError``，会穿透 ``get_settings`` 的全部
        三个 handler，用户看到的是解释器默认 handler 打出的多帧裸 traceback
        （实测 23 帧），其中没有任何字段名。
        """
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} 必须是字符串，但得到 {safe_repr(v)}。")
        # 空值必须显式拒绝，不能原样返回。此处原为 ``if not v: return v``，而
        # ``Path("")`` 在求值期等价于 ``Path(".")``——**运行时的当前工作目录**。
        # 两个后果：
        #
        # 1. 信任根扩到 CWD。``FaissStore.load_snapshot`` 拿 ``snapshot_root`` 当
        #    pickle 的来源信任根（``AGENTS.md`` 要求快照只从本项目生成的文件
        #    加载），实测 ``SNAPSHOT_ROOT=""`` 时信任根落在仓库根目录，在其中放
        #    一个带 ``ACTIVE_SNAPSHOT`` 标记的目录即可让加载被接受，边界等于
        #    不存在。
        # 2. 行为随 CWD 漂移。默认值（如 ``data/kb``）走下面的 ``ROOT_DIR`` 分支、
        #    不随 CWD 变，而空值原样返回后由 ``Path()`` 在求值期绑定到 CWD——同一
        #    份配置在仓库内与仓库外运行会指向不同目录。
        #
        # 纯空白一并拒绝：``Path("   ")`` 会解析成一个名为空格的**真实目录**，
        # 没有任何合法用途。路径字段的默认值都是非空相对路径，用户「不想设」时
        # 应删掉该键回落默认值，而不是写成空串。
        if not v.strip():
            raise ValueError(
                f"{info.field_name} 不能为空。"
                " 空值会解析成当前工作目录，使路径随运行位置漂移；"
                "如需使用默认值，请删除该配置项而不是置空。"
            )
        # NUL 字节也必须显式拒绝：``os.makedirs``／``open`` 对它抛
        # ``ValueError: embedded null byte``——**不是** ``OSError``，因此下游那些
        # 「目录不可写就降级」的 ``except OSError`` 全部接不住。实测
        # ``config.toml`` 写 ``log_path = "/tmp/a\u0000b"``（TOML 的 ``\u0000``
        # 转义）时，``cleanup.py`` 在**模块导入期**调用 ``get_module_logger``，
        # 程序连 ``run_cli`` 都到不了就崩。路径里出现 NUL 没有任何合法用途。
        if "\x00" in v:
            raise ValueError(f"{info.field_name} 不能包含 NUL 字节。")
        path = Path(v)
        if path.is_absolute():
            return str(path)
        return str((ROOT_DIR / v).resolve())

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """自定义配置加载源，保留环境变量、.env 和 config.toml 三层来源。"""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    model_config = SettingsConfigDict(
        # 见 ``ENV_FILE_PATH`` 的说明：与 ``CONFIG_TOML_PATH`` 同源。
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )


# ``options`` 的取值是原样交给 SDK／服务端的用户载荷（``response_format`` 的
# JSON Schema、``extra_body`` 的内容等），键名属于用户，不能按项目词汇小写化。
# 见 ``load_toml_config`` 里 ``normalize_mapping`` 的说明。
_OPAQUE_VALUE_CONTAINERS = frozenset({"options"})


def load_toml_config() -> dict[str, Any]:
    """
    从全局 CONFIG_TOML_PATH 路径加载 config.toml 文件配置。
    """
    if not CONFIG_TOML_PATH.exists():
        return {}

    with CONFIG_TOML_PATH.open("rb") as f:
        raw_config = tomllib.load(f)

    def normalize_mapping(
        value: Any, preserve_keys: set[str] | None = None, *, opaque: bool = False
    ) -> Any:
        """把 TOML 的键名归一成项目的小写词汇，但**不碰用户载荷**。

        归一化的理由是配置项名大小写不敏感（``CHAT_TOP_K``／``chat_top_k`` 等价），
        而这条理由**不适用于取值**：``options`` 里的 ``response_format``、
        ``extra_body`` 是原样交给 SDK／服务端的 JSON，键名属于用户。
        此前每一层嵌套字典的键都被小写化，实测
        ``required = ["OrderId"]`` 与 ``properties = { OrderId = ... }`` 被拆成
        ``["OrderId"]`` 与 ``["orderid"]``——schema 的 required 与 properties
        不再对应，结构化输出静默失效。

        * ``opaque=True``：整个子树原样保留键名（进入用户载荷之后）。
        * ``preserve_keys``：只保留**本层**键名（``llm_configurations`` 下的条目名
          由用户命名、别处按名查找），条目内部的字段名仍按项目词汇归一。
        * ``options``：本层键名是项目／SDK 的字段词汇（大小写不敏感，照旧归一，
          于是 ``MAX_TOKENS`` 仍然有效），其下的取值是用户载荷，原样保留。
        """
        if not isinstance(value, dict):
            return value

        preserved = preserve_keys or set()
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key) if opaque else str(raw_key).lower()
            if not isinstance(raw_value, dict):
                if isinstance(raw_value, list):
                    normalized[key] = [normalize_mapping(item, opaque=opaque) for item in raw_value]
                else:
                    normalized[key] = raw_value
                continue

            if opaque:
                normalized[key] = {
                    str(item_key): normalize_mapping(item_value, opaque=True)
                    for item_key, item_value in raw_value.items()
                }
            elif key in preserved:
                normalized[key] = {
                    str(item_key): normalize_mapping(item_value)
                    for item_key, item_value in raw_value.items()
                }
            elif key in _OPAQUE_VALUE_CONTAINERS:
                normalized[key] = {
                    str(item_key).lower(): normalize_mapping(item_value, opaque=True)
                    for item_key, item_value in raw_value.items()
                }
            else:
                normalized[key] = {
                    str(item_key).lower(): normalize_mapping(item_value)
                    for item_key, item_value in raw_value.items()
                }
        return normalized

    raw_config = normalize_mapping(
        raw_config,
        preserve_keys={"embedding_configurations", "rerank_configurations", "llm_configurations"},
    )

    flat_config: dict[str, Any] = {}
    scalar_keys = {
        "anthropic_api_key",
        "google_api_key",
        "gemini_api_key",
        # Vertex/Enterprise endpoint selection and project routing are not
        # credentials, so they may be supplied by config.toml as well as env.
        "google_genai_use_vertexai",
        "google_genai_use_enterprise",
        "google_cloud_project",
        "google_cloud_location",
        "siliconflow_api_key",
        "openai_api_key",
        "qwen_api_key",
        "ark_api_key",
        "volc_access_key",
        "volc_secret_key",
        "jina_api_key",
        "deepseek_api_key",
        "grok_api_key",
        "lm_studio_api_key",
        "openai_api_base",
        "siliconflow_base_url",
        "qwen_base_url",
        "deepseek_base_url",
        "ollama_base_url",
        "lm_studio_base_url",
        "volc_base_url",
        "grok_base_url",
        "log_level",
        "cache_path",
        "log_path",
        "log_retention_days",
        "knowledge_base_path",
        "pkl_path",
        "snapshot_root",
        "kb_replace_whitespace",
        "kb_remove_spaces",
        "kb_remove_urls",
        "kb_use_qa_segmentation",
        "kb_splitter_separators",
        "kb_chunk_size",
        "kb_chunk_overlap",
        "kb_child_chunk_size",
        "kb_child_chunk_overlap",
        "kb_embedding_batch_size",
        "default_llm_provider",
        "default_embedding_provider",
        "default_rerank_provider",
        "default_vector_store",
        "chat_retrieval_method",
        "chat_vector_weight",
        "chat_keyword_weight",
        "hybrid_fusion_strategy",
        "retrieval_candidate_multiplier",
        "chat_rerank_enabled",
        "chat_top_k",
        "chat_score_threshold",
        "chat_temperature",
    }

    for key in scalar_keys:
        if key in raw_config:
            flat_config[key] = raw_config[key]

    for config_key in ("embedding_configurations", "rerank_configurations", "llm_configurations"):
        if config_key in raw_config:
            flat_config[config_key] = raw_config[config_key]

    return flat_config


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    一个 pydantic-settings 的自定义源，用于从 config.toml 文件加载配置。
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # 在 __call__ 中处理所有逻辑，这里可以什么都不做
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """
        在被调用时加载并返回 TOML 配置。
        这确保了加载操作发生在 get_settings() 被调用时，
        此时 monkeypatch 已经生效。
        """
        return load_toml_config()


# =================================================================
# 3. 实例化并导出 (INSTANTIATE & EXPORT)
# =================================================================


@functools.lru_cache
def get_settings() -> Settings:
    """
    获取 Settings 实例的单例。
    加载顺序由 settings_customise_sources 定义。

    配置校验失败时重抛一个已脱敏的异常。``Settings()`` 在 import 期就会被调用
    （``log_manager.get_module_logger``），早于任何入口的 ``try``，pydantic 的
    默认渲染会把 ``input_value`` 明文交给解释器默认 handler——而 ``options`` 是
    文档指定的扩展入口，把 ``api_key`` 放进去恰好就是被校验拒绝的那类错误。
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ValueError(_redacted_settings_error(exc)) from None
    except SettingsError as exc:
        # pydantic-settings 在解析复杂字段（list/dict）时抛的是 SettingsError，
        # 它继承 ValueError 但**不是** ValidationError，因此上面那个分支接不住。
        # 实测 ``KB_SPLITTER_SEPARATORS='###'``（非 JSON 形态）走的正是这条路，
        # 用户看到的是解释器默认 handler 打出的多屏 traceback，而不是脱敏消息。
        raise ValueError(f"配置解析失败 - {redact_sensitive_text(str(exc))}") from None
    except tomllib.TOMLDecodeError as exc:
        # ``config.toml`` 语法错走的是 ``load_toml_config`` 里的 ``tomllib.load``，
        # pydantic-settings 把该异常原样放出来。``TOMLDecodeError`` 继承
        # ``ValueError``，但既不是 ``ValidationError`` 也不是 ``SettingsError``，
        # 上面两个分支都接不住——实测 import 期（``get_module_logger`` ->
        # ``get_settings``）直接抛裸 traceback，35 帧全打在终端上，早于任何入口
        # 的 ``try``。作为对照，同一份文件里字段**取值**非法时得到的是 11 帧的
        # 脱敏消息。同属「配置写错」，收场方式不该有两种。
        #
        # 消息里保留行号与列号：用户靠它定位是哪一行写坏了。tomllib 的消息只含
        # 位置与原因，不回显该行的内容，因此不涉及凭证回显；仍走一次脱敏以防
        # 上游改动，成本可忽略。
        raise ValueError(f"配置解析失败 - {redact_sensitive_text(str(exc))}") from None
    except RecursionError as exc:
        # ``tomllib`` 对**深层嵌套**的表/数组会递归解析，超过解释器递归上限时抛
        # ``RecursionError``。它继承 ``RuntimeError`` 而**不是** ``ValueError``，
        # 因此上面四个 handler 一个都接不住——实测 ``config.toml`` 写
        # ``a = [[[[...]]]]``（深度 500 即触发）后 ``import main`` 直接以
        # 2990 帧裸 traceback 终止，早于任何入口的 ``try``。
        #
        # 与下面 ``except ValueError`` 同属「配置写错」的收场，消息里带上原因：
        # 用户需要知道是嵌套层级而不是取值出的问题。
        raise ValueError(
            f"配置解析失败 - 嵌套层级过深（{redact_sensitive_text(str(exc))}）；"
            "请检查 config.toml 中是否存在过深的表或数组嵌套。"
        ) from None
    except ValueError as exc:
        # ``tomllib`` 解析 ``config.toml`` 时还有一类失败**不是**
        # ``TOMLDecodeError``：整数位数超过 ``sys.get_int_max_str_digits()``
        # （CPython 默认 4300 位）时，``_parser`` 内部直接 ``int()`` 并让
        # ``ValueError`` 穿出来。实测 ``chat_top_k = "9" * 4301`` 得到的是
        # 36 帧裸 traceback，消息只有 "Exceeds the limit (4300 digits)..."——
        # 既没有「配置解析失败」前缀，也没说是哪个文件写坏了，而同样属于
        # 「配置写错」的语法错是 5 帧带前缀的消息。这一支兜住它。
        #
        # 放在最后：``ValidationError``／``SettingsError``／``TOMLDecodeError``
        # 都继承 ``ValueError``，顺序颠倒会让前面的分支永远接不到。
        raise ValueError(f"配置解析失败 - {redact_sensitive_text(str(exc))}") from None


def _redacted_settings_error(exc: ValidationError) -> str:
    """把配置校验失败整理成一行脱敏消息。

    直接重抛 ``ValidationError`` 会让 pydantic 再包一层，原始报告的
    ``input_value`` 会以嵌套形式重复出现；这里逐条取 ``loc`` 与 ``msg``
    重建，只保留定位信息和校验原因。``msg`` 与 ``input`` **两者**都交给
    ``redact_sensitive_text``：validator 的消息里常常回显原始取值，
    只脱敏 ``input`` 会留下明文出口。
    """
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        # ``msg`` 也必须脱敏：validator 普遍把**原始取值**写进消息
        # （``f"无效的日志级别: {value}"``），因此只处理 ``input`` 会留下一条
        # 明文出口。实测 ``CHAT_TEMPERATURE=sk-live-...`` 时消息里原样带着该
        # 凭证，而这条消息经 ``get_settings`` 的 handler 抛出后进终端与日志
        # ——正是本函数 docstring 声称要挡住的那一类。
        message = redact_sensitive_text(str(error.get("msg", "校验失败")))
        # 用 ``safe_repr`` 而不是 ``str()``：``input`` 是 pydantic 记下的原始
        # 输入值，可以是任意大的整数。实测 ``Settings(chat_top_k=10**4301)``
        # （库调用方与测试都走这条构造路径）让**报告校验失败**这一步自己抛
        # ``ValueError: Exceeds the limit``——用户看到的是位数上限错误，而真正
        # 的问题是 ``chat_top_k`` 越界。
        value = redact_sensitive_text(safe_repr(error.get("input")))
        lines.append(f"{location}: {message} (input={value})" if location else f"{message}")
    return "配置校验失败 - " + "; ".join(lines) if lines else "配置校验失败"


# 导出 get_settings 函数，供其他模块在需要时调用
# 这样可以确保在测试中能够灵活地替换或模拟配置
# settings = get_settings() # 移除直接导出 settings 实例
