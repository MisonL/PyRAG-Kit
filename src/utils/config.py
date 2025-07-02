# -*- coding: utf-8 -*-
import configparser
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # --- [PATHS] ---
    knowledge_base_path: str = "knowledge_base"
    pkl_path: str = "data/employee_kb.pkl"
    log_path: str = "data/logs"
    log_retention_days: int = 15

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

    # --- [CHAT] ---
    chat_retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_SEARCH
    chat_vector_weight: float = 0.3
    chat_keyword_weight: float = 0.7
    chat_rerank_enabled: bool = False
    chat_top_k: int = 5
    chat_score_threshold: float = 0.4

    # --- [VALIDATORS] ---
    @field_validator('kb_splitter_separators', mode='before')
    @classmethod
    def split_separators(cls, v: Any) -> List[str]:
        """如果分隔符是字符串，则按逗号分割成列表。"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(',') if s.strip()]
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

    # --- [MODEL_CONFIGURATIONS] ---
    embedding_configurations: Dict[str, ModelDetail] = Field(default_factory=dict)
    rerank_configurations: Dict[str, ModelDetail] = Field(default_factory=dict)
    llm_configurations: Dict[str, ModelDetail] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore' # 忽略在模型中未定义的额外字段
    )

def load_ini_config() -> Dict[str, Any]:
    """
    从 config.ini 文件加载配置。
    通过禁用 configparser 的默认注释处理，并使用更精确的正则表达式，
    来确保像 '###' 这样的值不会被错误地当作注释移除。
    """
    if not CONFIG_PATH.exists():
        return {}

    config_dict: Dict[str, Any] = {}
    json_keys = {'embedding_configurations', 'rerank_configurations', 'llm_configurations'}
    
    # 初始化 ConfigParser，通过设置空的前缀来禁用默认的注释处理功能。
    # 这样，值中包含的 '#' 或 ';' 字符就不会被错误地处理。
    parser = configparser.ConfigParser(comment_prefixes=(), inline_comment_prefixes=())
    
    # 读取文件内容
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        # 忽略空行和以 '#' 或 ';' 开头的整行注释
        if not stripped_line or stripped_line.startswith('#') or stripped_line.startswith(';'):
            continue
        processed_lines.append(line)
    
    config_content = "".join(processed_lines)

    # 使用 read_string 来解析处理后的内容
    parser.read_string(config_content)

    config_dict: Dict[str, Any] = {}
    json_keys = {'embedding_configurations', 'rerank_configurations', 'llm_configurations'}

    # 遍历所有节
    for section in parser.sections():
        for key, value_with_comment in parser.items(section):
            # 这个正则表达式只会移除由空格隔开的'#'注释。
            # 例如 'value # comment' 会变成 'value'。
            # 而 '###' 或 'value#no-space' 不会受影响。
            value = re.sub(r'\s+#.*$', '', value_with_comment).strip()

            if key in json_keys:
                # 对于JSON字段，如果值为空，则初始化为空字典
                if not value:
                    config_dict[key] = {}
                    continue
                try:
                    config_dict[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # 如果JSON解析失败，也使用一个空字典作为回退
                    config_dict[key] = {}
            else:
                # 对于所有其他键，直接使用清理后的字符串值
                config_dict[key] = value
    return config_dict

# =================================================================
# 3. 实例化并导出 (INSTANTIATE & EXPORT)
# =================================================================

# 1. 从 INI 文件加载初始值
ini_values = load_ini_config()

# 2. 实例化 Settings，INI 值作为基础，但可被 .env 或环境变量覆盖
settings = Settings(**ini_values)


# =================================================================
# 4. 向后兼容层 (BACKWARD COMPATIBILITY LAYER)
# =================================================================
# 目标: 最小化对现有代码的侵入性。
# 策略: 保持旧的配置变量，但使其从新的settings实例派生。
# 后续重构中，应逐步淘汰这些变量，直接使用 `settings` 对象。

# --- 路径与环境配置 ---
KB_PATH = ROOT_DIR / settings.knowledge_base_path
PKL_PATH = ROOT_DIR / settings.pkl_path
LOG_PATH = ROOT_DIR / settings.log_path
LOG_RETENTION_DAYS = settings.log_retention_days
CACHE_PATH = ROOT_DIR / ".cache"

# --- 知识库构建配置 ---
KB_CONFIG = {
    "replace_consecutive_whitespace": settings.kb_replace_whitespace,
    "remove_extra_spaces": settings.kb_remove_spaces,
    "remove_urls_and_emails": settings.kb_remove_urls,
    "text_splitter_separators": settings.kb_splitter_separators,
    "chunk_size": settings.kb_chunk_size,
    "chunk_overlap": settings.kb_chunk_overlap,
    "use_qa_segmentation": settings.kb_use_qa_segmentation,
    "embedding_configurations": settings.embedding_configurations,
    "active_embedding_configuration": settings.default_embedding_provider,
    "embedding_batch_size": settings.kb_embedding_batch_size,
    "kb_dir": str(KB_PATH),
    "output_file": str(PKL_PATH),
}

# --- 聊天机器人配置 (可动态修改) ---
CHAT_CONFIG = {
    "active_llm_configuration": settings.default_llm_provider,
    "retrieval_method": settings.chat_retrieval_method,
    "vector_weight": settings.chat_vector_weight,
    "keyword_weight": settings.chat_keyword_weight,
    "rerank_enabled": settings.chat_rerank_enabled,
    "top_k": settings.chat_top_k,
    "score_threshold": settings.chat_score_threshold,
    "active_rerank_configuration": settings.default_rerank_provider,
    "rerank_configurations": settings.rerank_configurations,
    "llm_configurations": settings.llm_configurations,
}

# --- API密钥与URL配置 ---
API_CONFIG = {
    "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    "GOOGLE_API_KEY": settings.google_api_key,
    "SILICONFLOW_API_KEY": settings.siliconflow_api_key,
    "OPENAI_API_KEY": settings.openai_api_key,
    "QWEN_API_KEY": settings.qwen_api_key,
    "VOLC_ACCESS_KEY": settings.volc_access_key,
    "VOLC_SECRET_KEY": settings.volc_secret_key,
    "JINA_API_KEY": settings.jina_api_key,
    "DEEPSEEK_API_KEY": settings.deepseek_api_key,
    "GROK_API_KEY": settings.grok_api_key,
    "LM_STUDIO_API_KEY": settings.lm_studio_api_key,
    "OPENAI_API_BASE": str(settings.openai_api_base),
    "SILICONFLOW_BASE_URL": str(settings.siliconflow_base_url),
    "QWEN_BASE_URL": str(settings.qwen_base_url),
    "DEEPSEEK_BASE_URL": str(settings.deepseek_base_url),
    "OLLAMA_BASE_URL": str(settings.ollama_base_url),
    "LM_STUDIO_BASE_URL": str(settings.lm_studio_base_url),
    "VOLC_BASE_URL": str(settings.volc_base_url),
    "GROK_BASE_URL": str(settings.grok_base_url),
}

# --- 为了解决循环导入问题，将模型配置的导出移到最后 ---
EMBEDDING_CONFIGS = settings.embedding_configurations
RERANK_CONFIGS = settings.rerank_configurations
LLM_CONFIGS = settings.llm_configurations