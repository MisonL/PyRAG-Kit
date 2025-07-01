# -*- coding: utf-8 -*-
import os
import json
import configparser
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict

# =================================================================
# 1. 配置加载器 (CONFIGURATION LOADER)
# =================================================================
# 创建一个ConfigParser实例
config = configparser.ConfigParser()

# 定义config.ini文件的路径（位于项目根目录）
config_path = Path(__file__).parent.parent.parent / 'config.ini'

# 仅在文件存在时读取，以支持无文件、纯环境变量的部署方式
if config_path.exists():
    config.read(config_path, encoding='utf-8')


# =================================================================
# 2. 辅助函数与枚举 (HELPERS & ENUMS)
# =================================================================
def get_config(section: str, key: str, fallback: Any = None, type_converter: Callable = str) -> Any:
    """
    优先从环境变量获取配置，其次从config.ini文件获取。
    如果两处都未设置，则返回指定的fallback默认值。
    """
    # 1. 尝试从环境变量获取 (环境变量的键通常为大写)
    env_var = os.getenv(key.upper())
    if env_var is not None:
        value = env_var
    # 2. 如果环境变量不存在，则尝试从 .ini 文件获取
    elif config.has_option(section, key):
        value = config.get(section, key)
    # 3. 如果都不存在，则使用 fallback
    else:
        return fallback
    
    # 如果值是字符串，在转换类型前先清理行内注释
    if isinstance(value, str):
        value = value.split('#')[0].strip()

    # 对获取到的值进行类型转换
    try:
        return type_converter(value)
    except (ValueError, TypeError):
        return fallback

def str_to_bool(s: str) -> bool:
    """将字符串转换为布尔值。"""
    return str(s).lower() in ('true', '1', 't', 'y', 'yes')

def str_to_list(s: str, delimiter: str = ',') -> list:
    """将逗号分隔的字符串转换为列表。"""
    if isinstance(s, list):
        return s
    return [item.strip() for item in s.split(delimiter)]

class RetrievalMethod(Enum):
    """定义知识库检索的策略枚举。"""
    SEMANTIC_SEARCH = "向量检索"
    FULL_TEXT_SEARCH = "全文检索"
    HYBRID_SEARCH = "混合检索"

def str_to_retrieval_method(s: str) -> RetrievalMethod:
    """将字符串转换为RetrievalMethod枚举。"""
    s_upper = str(s).upper()
    try:
        return RetrievalMethod[s_upper]
    except KeyError:
        for member in RetrievalMethod:
            if member.value == s:
                return member
        return RetrievalMethod.HYBRID_SEARCH


# =================================================================
# 3. 路径与环境配置 (PATHS & ENVIRONMENT)
# =================================================================
ROOT_DIR = Path(__file__).parent.parent.parent
KB_PATH = ROOT_DIR / get_config("PATHS", "KNOWLEDGE_BASE_PATH", "knowledge_base")
PKL_PATH = ROOT_DIR / get_config("PATHS", "PKL_PATH", "data/employee_kb.pkl")
LOG_PATH = ROOT_DIR / get_config("PATHS", "LOG_PATH", "data/logs")
LOG_RETENTION_DAYS = get_config("PATHS", "LOG_RETENTION_DAYS", 15, int)
CACHE_PATH = ROOT_DIR / ".cache"


# =================================================================
# 4. 配置中心 (CONFIGURATION)
# =================================================================
def get_json_config(section: str, key: str) -> Dict:
    """优先从环境变量获取JSON配置，其次从config.ini文件获取。"""
    # 1. 尝试从环境变量获取
    json_str = os.getenv(key.upper())
    # 2. 如果环境变量不存在，则尝试从 .ini 文件获取
    if json_str is None and config.has_option(section, key):
        json_str = config.get(section, key)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {} # 解析失败则返回空字典
    return {} # 如果两处都没有，返回空字典

# --- 从配置源加载所有模型配置 ---
EMBEDDING_CONFIGS = get_json_config("MODEL_CONFIGURATIONS", "EMBEDDING_CONFIGURATIONS")
RERANK_CONFIGS = get_json_config("MODEL_CONFIGURATIONS", "RERANK_CONFIGURATIONS")
LLM_CONFIGS = get_json_config("MODEL_CONFIGURATIONS", "LLM_CONFIGURATIONS")


# --- 知识库构建配置 ---
KB_CONFIG = {
    "replace_consecutive_whitespace": get_config("KNOWLEDGE_BASE", "KB_REPLACE_WHITESPACE", False, str_to_bool),
    "remove_extra_spaces": get_config("KNOWLEDGE_BASE", "KB_REMOVE_SPACES", False, str_to_bool),
    "remove_urls_and_emails": get_config("KNOWLEDGE_BASE", "KB_REMOVE_URLS", False, str_to_bool),
    "text_splitter_separators": get_config("KNOWLEDGE_BASE", "KB_SPLITTER_SEPARATORS", ["###"], str_to_list),
    "chunk_size": get_config("KNOWLEDGE_BASE", "KB_CHUNK_SIZE", 1500, int),
    "chunk_overlap": get_config("KNOWLEDGE_BASE", "KB_CHUNK_OVERLAP", 150, int),
    "use_qa_segmentation": get_config("KNOWLEDGE_BASE", "KB_USE_QA_SEGMENTATION", False, str_to_bool),
    "embedding_configurations": EMBEDDING_CONFIGS,
    "active_embedding_configuration": get_config("BEHAVIOR", "DEFAULT_EMBEDDING_PROVIDER", "google", str),
    "embedding_batch_size": get_config("KNOWLEDGE_BASE", "KB_EMBEDDING_BATCH_SIZE", 32, int),
    "kb_dir": str(KB_PATH),
    "output_file": str(PKL_PATH),
}

# --- 聊天机器人配置 ---
# 这个字典在程序运行时会动态变化，以反映用户通过交互式菜单所做的临时配置更改。
CHAT_CONFIG = {
    # 程序启动时，使用BEHAVIOR中定义的默认LLM提供商来初始化
    "active_llm_configuration": get_config("BEHAVIOR", "DEFAULT_LLM_PROVIDER", "google", str),
    "retrieval_method": get_config("CHAT", "CHAT_RETRIEVAL_METHOD", RetrievalMethod.HYBRID_SEARCH, str_to_retrieval_method),
    "vector_weight": get_config("CHAT", "CHAT_VECTOR_WEIGHT", 0.3, float),
    "keyword_weight": get_config("CHAT", "CHAT_KEYWORD_WEIGHT", 0.7, float),
    "rerank_enabled": get_config("CHAT", "CHAT_RERANK_ENABLED", False, str_to_bool),
    "top_k": get_config("CHAT", "CHAT_TOP_K", 5, int),
    "score_threshold": get_config("CHAT", "CHAT_SCORE_THRESHOLD", 0.4, float),
    "active_rerank_configuration": get_config("BEHAVIOR", "DEFAULT_RERANK_PROVIDER", "siliconflow", str),
    "rerank_configurations": RERANK_CONFIGS,
    "llm_configurations": LLM_CONFIGS,
}

# --- API密钥与URL配置 ---
# 此处我们定义所有可能的API键和URL键，然后使用get_config来填充它们
# 这样可以确保环境变量优先的逻辑被正确应用
_api_keys = [
    'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'SILICONFLOW_API_KEY', 'OPENAI_API_KEY',
    'QWEN_API_KEY', 'VOLC_ACCESS_KEY', 'VOLC_SECRET_KEY', 'JINA_API_KEY',
    'DEEPSEEK_API_KEY', 'GROK_API_KEY', 'LM_STUDIO_API_KEY'
]
_base_urls = [
    'OPENAI_API_BASE', 'SILICONFLOW_BASE_URL', 'QWEN_BASE_URL', 'DEEPSEEK_BASE_URL',
    'OLLAMA_BASE_URL', 'LM_STUDIO_BASE_URL', 'VOLC_BASE_URL', 'GROK_BASE_URL'
]

API_CONFIG = {
    **{key: get_config('API_KEYS', key, fallback=None) for key in _api_keys},
    **{key: get_config('BASE_URLS', key, fallback=None) for key in _base_urls}
}