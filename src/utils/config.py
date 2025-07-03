# -*- coding: utf-8 -*-
import configparser
import json
import re
from enum import Enum
from pathlib import Path
import functools
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

# =================================================================
# 1. 基础定义 (DEFINITIONS)
# =================================================================

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
# 配置文件路径
CONFIG_PATH = ROOT_DIR / 'config.ini'

class RetrievalMethod(str, Enum):
    """定义知识库检索的策略枚举。"""
    SEMANTIC_SEARCH = "向量检索"
    FULL_TEXT_SEARCH = "全文检索"
    HYBRID_SEARCH = "混合检索"

class ModelDetail(BaseModel):
    """定义单个模型配置的结构。"""
    provider: str
    model_name: str

    model_config = SettingsConfigDict(protected_namespaces=())

# =================================================================
# 2. 主配置模型 (MAIN SETTINGS MODEL)
# =================================================================

class Settings(BaseSettings):
    """
    定义整个应用的配置，使用Pydantic进行类型校验和分层加载。
    加载顺序: 环境变量 > .env 文件 > config.ini 文件 > 模型中定义的默认值。
    """
    # --- [API_KEYS] ---
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    siliconflow_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    volc_access_key: Optional[str] = None
    volc_secret_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    lm_studio_api_key: Optional[str] = "lm-studio"

    # --- [BASE_URLS] ---
    openai_api_base: str = "https://api.openai.com/v1"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    deepseek_base_url: str = "https://api.deepseek.com"
    ollama_base_url: str = "http://localhost:11434/v1"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    volc_base_url: str = "https://maas-api.ml-platform-cn-beijing.volces.com"
    grok_base_url: str = "https://api.x.ai/v1"

    # --- [GENERAL] ---
    log_level: str = "INFO" # 新增 log_level 字段
    cache_path: str = ".cache" # 新增 cache_path 字段
    log_path: str = "data/logs"
    log_retention_days: int = 15

    # --- [PATHS] ---
    knowledge_base_path: str = "knowledge_base"
    pkl_path: str = "data/employee_kb.pkl"

    # --- [KNOWLEDGE_BASE] ---
    kb_replace_whitespace: bool = False
    kb_remove_spaces: bool = False
    kb_remove_urls: bool = False
    kb_use_qa_segmentation: bool = False
    kb_splitter_separators: List[str] = Field(default=["###"])
    kb_chunk_size: int = 1500
    kb_chunk_overlap: int = 150
    kb_embedding_batch_size: int = 32

    # --- [BEHAVIOR] ---
    default_llm_provider: str = "google"
    default_embedding_provider: str = "google"
    default_rerank_provider: str = "siliconflow"
    default_vector_store: str = "faiss" # 新增向量存储默认提供商

    # --- [CHAT] ---
    chat_retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_SEARCH
    chat_vector_weight: float = 0.3
    chat_keyword_weight: float = 0.7
    chat_rerank_enabled: bool = False
    chat_top_k: int = 5
    chat_score_threshold: float = 0.4
    chat_temperature: float = 0.7 # 将 chat_temperature 移到这里

    # --- [MODEL_CONFIGURATIONS] ---
    embedding_configurations: Dict[str, ModelDetail] = Field(default_factory=lambda: {
        "google": ModelDetail(provider="google", model_name="embedding-001"),
        "jina": ModelDetail(provider="jina", model_name="jina-embeddings-v2-base-zh"),
        "siliconflow": ModelDetail(provider="siliconflow", model_name="alibaba/bge-large-zh-v1.5"),
        "openai": ModelDetail(provider="openai", model_name="text-embedding-3-small"),
    })
    rerank_configurations: Dict[str, ModelDetail] = Field(default_factory=lambda: {
        "siliconflow": ModelDetail(provider="siliconflow", model_name="alibaba/bge-reranker-large"),
    })
    llm_configurations: Dict[str, ModelDetail] = Field(default_factory=lambda: {
        "google": ModelDetail(provider="google", model_name="gemini-1.5-pro-latest"),
        "anthropic": ModelDetail(provider="anthropic", model_name="claude-3-opus-20240229"),
        "qwen": ModelDetail(provider="qwen", model_name="qwen-turbo"),
        "deepseek": ModelDetail(provider="deepseek", model_name="deepseek-chat"),
        "grok": ModelDetail(provider="grok", model_name="grok-1"),
        "volcengine": ModelDetail(provider="volcengine", model_name="Doubao-pro-32k"),
        "siliconflow": ModelDetail(provider="siliconflow", model_name="deepseek-ai/DeepSeek-V2-Chat"),
        "openai": ModelDetail(provider="openai", model_name="gpt-4o"),
        "ollama": ModelDetail(provider="ollama", model_name="llama3"),
        "lm_studio": ModelDetail(provider="lm_studio", model_name="LM-Studio-Community/Meta-Llama-3-8B-Instruct-GGUF"),
    })

    # --- [VALIDATORS] ---
    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别是否有效。"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}. 必须是 {', '.join(valid_levels)} 中的一个。")
        return v.upper()

    @field_validator('chat_temperature', mode='before')
    @classmethod
    def validate_chat_temperature(cls, v: Any) -> float:
        """验证聊天温度在 0.0 到 1.0 之间。"""
        try:
            value = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"无法将聊天温度 '{v}' 转换为数字。")

        if not (0.0 <= value <= 1.0):
            raise ValueError(f"聊天温度必须在 0.0 到 1.0 之间，但得到 {value}。")
        return value

    @field_validator('kb_splitter_separators', mode='before')
    @classmethod
    def split_separators(cls, v: Any) -> List[str]:
        """
        如果分隔符是字符串，则按逗号分割成列表。
        如果输入值为空（None或空字符串），或者分割后为空列表，则使用字段的默认值。
        """
        # 通过访问类的模型字段来安全地获取默认值
        default_value = cls.model_fields['kb_splitter_separators'].default
        if default_value is None:
            default_value = [] # 以防万一没有设置默认值

        # 如果输入为空（来自 .env 或环境变量的空字符串），则回退到默认值
        if v is None or v == '':
            return default_value

        if isinstance(v, str):
            # 按逗号分割，并过滤掉空的元素
            separators = [s.strip() for s in v.split(',') if s.strip()]
            # 如果分割后列表为空（例如，输入是" , "），也使用默认值
            return separators if separators else default_value
        
        # 如果输入已经是列表或其他类型，直接返回
        return v

    @field_validator('chat_retrieval_method', mode='before')
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

    @field_validator('knowledge_base_path', 'pkl_path', 'log_path', 'cache_path', mode='before')
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
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        自定义配置加载源，加入对 config.ini 的支持。
        加载顺序:
        1. init_settings: 初始化时传入的参数
        2. env_settings: 环境变量
        3. dotenv_settings: .env 文件
        4. IniConfigSettingsSource: config.ini 文件
        5. file_secret_settings: Docker secrets
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            IniConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        protected_namespaces=(),
    )

def load_ini_config() -> Dict[str, Any]:
    """
    从全局 CONFIG_PATH 路径加载 .ini 文件配置，并将其扁平化为单个字典。
    """
    if not CONFIG_PATH.exists():
        return {}

    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH, encoding='utf-8')

    flat_config: Dict[str, Any] = {}
    json_keys = {'embedding_configurations', 'rerank_configurations', 'llm_configurations'}

    for section in parser.sections():
        for key, value in parser.items(section):
            # 移除可能存在的行内注释
            value = re.sub(r'\s*([#;]).*$', '', value).strip()
            # 移除值两端的单引号和双引号
            value = value.strip('\'"')

            if key in json_keys:
                if value:
                    try:
                        flat_config[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        flat_config[key] = {}
                else:
                    flat_config[key] = {}
            else:
                flat_config[key] = value
    return flat_config

class IniConfigSettingsSource(PydanticBaseSettingsSource):
    """
    一个 pydantic-settings 的自定义源，用于从 config.ini 文件加载配置。
    """
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        # 在 __call__ 中处理所有逻辑，这里可以什么都不做
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """
        在被调用时加载并返回 .ini 配置。
        这确保了加载操作发生在 get_settings() 被调用时，
        此时 monkeypatch 已经生效。
        """
        return load_ini_config()

# =================================================================
# 3. 实例化并导出 (INSTANTIATE & EXPORT)
# =================================================================

@functools.lru_cache()
def get_settings() -> Settings:
    """
    获取 Settings 实例的单例。
    加载顺序由 settings_customise_sources 定义。
    """
    return Settings()

# 导出 settings 实例，供其他模块直接导入
# 注意: 为了避免在导入时立即加载配置，通常更好的做法是
# 只导出 get_settings 函数，然后在需要的地方调用它。
settings = get_settings()

# =================================================================
# 4. 向后兼容层 (BACKWARD COMPATIBILITY LAYER)
# =================================================================
# 目标: 最小化对现有代码的侵入性。
# 策略: 保持旧的配置变量，但使其从新的settings实例派生。
# 后续重构中，应逐步淘汰这些变量，直接使用 `settings` 对象。

def get_backward_compatible_configs() -> Dict[str, Any]:
    """
    获取向后兼容的配置字典。
    """
    current_settings = get_settings()

    # --- 路径与环境配置 ---
    KB_PATH = Path(current_settings.knowledge_base_path)
    PKL_PATH = Path(current_settings.pkl_path)
    LOG_PATH = Path(current_settings.log_path)
    LOG_RETENTION_DAYS = current_settings.log_retention_days
    CACHE_PATH = Path(current_settings.cache_path)

    # --- 知识库构建配置 ---
    KB_CONFIG = {
        "replace_consecutive_whitespace": current_settings.kb_replace_whitespace,
        "remove_extra_spaces": current_settings.kb_remove_spaces,
        "remove_urls_and_emails": current_settings.kb_remove_urls,
        "text_splitter_separators": current_settings.kb_splitter_separators,
        "chunk_size": current_settings.kb_chunk_size,
        "chunk_overlap": current_settings.kb_chunk_overlap,
        "use_qa_segmentation": current_settings.kb_use_qa_segmentation,
        "embedding_configurations": current_settings.embedding_configurations,
        "active_embedding_configuration": current_settings.default_embedding_provider,
        "embedding_batch_size": current_settings.kb_embedding_batch_size,
        "kb_dir": str(KB_PATH),
        "output_file": str(PKL_PATH),
    }

    # --- 聊天机器人配置 (可动态修改) ---
    CHAT_CONFIG = {
        "active_llm_configuration": current_settings.default_llm_provider,
        "retrieval_method": current_settings.chat_retrieval_method,
        "vector_weight": current_settings.chat_vector_weight,
        "keyword_weight": current_settings.chat_keyword_weight,
        "rerank_enabled": current_settings.chat_rerank_enabled,
        "top_k": current_settings.chat_top_k,
        "score_threshold": current_settings.chat_score_threshold,
        "active_rerank_configuration": current_settings.default_rerank_provider,
        "rerank_configurations": current_settings.rerank_configurations,
        "llm_configurations": current_settings.llm_configurations,
    }

    # --- API密钥与URL配置 ---
    API_CONFIG = {
        "ANTHROPIC_API_KEY": current_settings.anthropic_api_key,
        "GOOGLE_API_KEY": current_settings.google_api_key,
        "SILICONFLOW_API_KEY": current_settings.siliconflow_api_key,
        "OPENAI_API_KEY": current_settings.openai_api_key,
        "QWEN_API_KEY": current_settings.qwen_api_key,
        "VOLC_ACCESS_KEY": current_settings.volc_access_key,
        "VOLC_SECRET_KEY": current_settings.volc_secret_key,
        "JINA_API_KEY": current_settings.jina_api_key,
        "DEEPSEEK_API_KEY": current_settings.deepseek_api_key,
        "GROK_API_KEY": current_settings.grok_api_key,
        "LM_STUDIO_API_KEY": current_settings.lm_studio_api_key,
        "OPENAI_API_BASE": str(current_settings.openai_api_base),
        "SILICONFLOW_BASE_URL": str(current_settings.siliconflow_base_url),
        "QWEN_BASE_URL": str(current_settings.qwen_base_url),
        "DEEPSEEK_BASE_URL": str(current_settings.deepseek_base_url),
        "OLLAMA_BASE_URL": str(current_settings.ollama_base_url),
        "LM_STUDIO_BASE_URL": str(current_settings.lm_studio_base_url),
        "VOLC_BASE_URL": str(current_settings.volc_base_url),
        "GROK_BASE_URL": str(current_settings.grok_base_url),
    }
    
    return {
        "KB_PATH": KB_PATH,
        "PKL_PATH": PKL_PATH,
        "LOG_PATH": LOG_PATH,
        "LOG_RETENTION_DAYS": LOG_RETENTION_DAYS,
        "CACHE_PATH": CACHE_PATH,
        "KB_CONFIG": KB_CONFIG,
        "CHAT_CONFIG": CHAT_CONFIG,
        "API_CONFIG": API_CONFIG,
        "EMBEDDING_CONFIGS": current_settings.embedding_configurations,
        "RERANK_CONFIGS": current_settings.rerank_configurations,
        "LLM_CONFIGS": current_settings.llm_configurations,
    }

# --- 为了解决循环导入问题，将模型配置的导出移到最后 ---
# 这些变量现在通过 get_backward_compatible_configs() 函数提供
# EMBEDDING_CONFIGS = settings.embedding_configurations
# RERANK_CONFIGS = settings.rerank_configurations
# LLM_CONFIGS = settings.llm_configurations