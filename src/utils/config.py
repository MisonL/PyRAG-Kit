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
# 配置文件路径
CONFIG_TOML_PATH = ROOT_DIR / "config.toml"


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

        normalized = str(value).strip().lower().replace("-", "_")
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
            raise ValueError(f"不支持的模型协议: {value}。可选值: {supported}") from exc


# =================================================================
# 2. 主配置模型 (MAIN SETTINGS MODEL)
# =================================================================


# 检索规模的上界。不是洁癖：``effective_top_k = chat_top_k *
# retrieval_candidate_multiplier`` 直通 ``faiss_index.search()``，FAISS 既不报错
# 也不截断。实测 ``chat_top_k=10**9`` 单次查询分配约 3.6 GB 数组、RSS 涨
# 10.3 GB、耗时 8.4 秒。10_000 远超任何真实场景（默认 top_k=5 × multiplier=3）。
_MAX_RETRIEVAL_TOP_K = 10_000
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


def _parse_int_literal(text: str) -> int | None:
    """严格解析十进制整数字面量，不做科学计数法/下划线/正号的宽松放行。

    只接受 ``[0-9]+``（可带前导负号）。``"1_0"`` 被静默读成 10 这类安静的
    成功正是本项目反复踩到的模式，所以宁愿在这里明确拒绝。
    """
    digits = text.removeprefix("-")
    if digits and all(char in "0123456789" for char in digits):
        return int(text)
    return None


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
        raise ValueError(f"{field_label} 不能是布尔值 {value!r}。")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        parsed_int = _parse_int_literal(text)
        if parsed_int is not None:
            return parsed_int
        if not _DECIMAL_LITERAL_PATTERN.fullmatch(text):
            raise ValueError(f"无法把 {field_label} 的 {value!r} 转换为数值。")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"无法把 {field_label} 的 {value!r} 转换为数值。") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_label} 必须是有限数值，但得到 {value!r}。")
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
            raise ValueError(f"chat_score_threshold 必须是数值，但得到 {value!r}。")
        if not 0 <= number <= 1:
            raise ValueError(f"chat_score_threshold 必须在 0 到 1 之间，但得到 {value!r}。")
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
        number = _coerce_number(value, "kb_chunk_overlap/kb_child_chunk_overlap")
        if isinstance(number, float) and not number.is_integer():
            raise ValueError("kb_chunk_overlap/kb_child_chunk_overlap 必须是整数。")
        number = int(number)
        if number < 0:
            raise ValueError("kb_chunk_overlap/kb_child_chunk_overlap 必须是非负整数。")
        return number

    @field_validator("chat_vector_weight", "chat_keyword_weight", mode="before")
    @classmethod
    def validate_hybrid_weights(cls, value: Any) -> float:
        number = float(_coerce_number(value, "chat_vector_weight/chat_keyword_weight"))
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
        return self

    @model_validator(mode="after")
    def validate_overlap_smaller_than_size(self) -> "Settings":
        """overlap 必须严格小于 size，否则分片无法推进。

        ``chunk_size == chunk_overlap`` 会让切分器无法前进（得到空分片或
        死循环），且这一约束无法用单字段 validator 表达。
        """
        if self.kb_chunk_overlap >= self.kb_chunk_size:
            raise ValueError("kb_chunk_overlap 必须小于 kb_chunk_size。")
        if self.kb_child_chunk_overlap >= self.kb_child_chunk_size:
            raise ValueError("kb_child_chunk_overlap 必须小于 kb_child_chunk_size。")
        return self

    @model_validator(mode="after")
    def validate_child_chunk_size_not_larger_than_parent(self) -> "Settings":
        """子分片不得大于父分片，否则层级结构静默退化为一层。

        实测 ``kb_chunk_size=300, kb_child_chunk_size=1500`` 被接受后，父块数
        从 2 涨到 10（每个父块只产出 1 个子块），「先粗后细」的父子检索意图
        失效，且没有任何报错。同属无法用单字段 validator 表达的约束。
        """
        # 相等同样要拒：实测 parent=child=300 时不同父块数 == 分块数（15/15），
        # 即每个父块只产出一个子块，层级与 child>parent 一样退化为一层。
        if self.kb_child_chunk_size >= self.kb_chunk_size:
            raise ValueError(
                "kb_child_chunk_size 必须小于 kb_chunk_size（子分片不得大于或等于父分片）。"
            )
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
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别是否有效。"""
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
            raise ValueError(f"聊天温度必须是数字，不能是布尔值 {v!r}。")
        try:
            value = float(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"无法将聊天温度 '{v}' 转换为数字。") from exc

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
            raise ValueError(f"检索候选倍率必须是整数，不能是布尔值 {v!r}。")
        try:
            value = int(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"无法将检索候选倍率 '{v}' 转换为整数。") from exc

        if value < 1:
            raise ValueError(f"检索候选倍率必须大于等于 1，但得到 {value}。")
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
    def resolve_path(cls, v: str) -> str:
        """将相对路径解析为绝对路径。"""
        if not v:
            return v
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
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )


def load_toml_config() -> dict[str, Any]:
    """
    从全局 CONFIG_TOML_PATH 路径加载 config.toml 文件配置。
    """
    if not CONFIG_TOML_PATH.exists():
        return {}

    with CONFIG_TOML_PATH.open("rb") as f:
        raw_config = tomllib.load(f)

    def normalize_mapping(value: Any, preserve_keys: set[str] | None = None) -> Any:
        if not isinstance(value, dict):
            return value

        preserved = preserve_keys or set()
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key in preserved and isinstance(raw_value, dict):
                normalized[key] = {
                    str(item_key): normalize_mapping(item_value)
                    for item_key, item_value in raw_value.items()
                }
                continue

            if isinstance(raw_value, dict):
                normalized[key] = {
                    str(item_key).lower(): normalize_mapping(item_value)
                    for item_key, item_value in raw_value.items()
                }
                continue

            if isinstance(raw_value, list):
                normalized[key] = [normalize_mapping(item) for item in raw_value]
                continue

            normalized[key] = raw_value
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


def _redacted_settings_error(exc: ValidationError) -> str:
    """把配置校验失败整理成一行脱敏消息。

    直接重抛 ``ValidationError`` 会让 pydantic 再包一层，原始报告的
    ``input_value`` 会以嵌套形式重复出现；这里逐条取 ``loc`` 与 ``msg``
    重建，只保留定位信息和校验原因，并把值整体交给 ``redact_sensitive_text``。
    """
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "校验失败"))
        value = redact_sensitive_text(str(error.get("input")))
        lines.append(f"{location}: {message} (input={value})" if location else f"{message}")
    return "配置校验失败 - " + "; ".join(lines) if lines else "配置校验失败"


# 导出 get_settings 函数，供其他模块在需要时调用
# 这样可以确保在测试中能够灵活地替换或模拟配置
# settings = get_settings() # 移除直接导出 settings 实例
