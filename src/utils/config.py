# -*- coding: utf-8 -*-
import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# --- 基础设置: 加载 .env 文件 ---
# .env 文件应该位于项目根目录
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# =================================================================
# 1. 辅助函数与枚举 (HELPERS & ENUMS)
# =================================================================
from typing import Any, Callable

def get_env(key: str, default: Any, type_converter: Callable) -> Any:
    """
    从环境变量获取值，如果未设置则返回默认值，并进行类型转换。
    支持布尔值、整数、浮点数和列表的转换。
    """
    value = os.getenv(key)
    if value is None:
        # 对于没有在.env中设置的变量，直接返回代码中定义的默认值
        return default
    
    # 对环境变量中的值进行类型转换
    try:
        return type_converter(value)
    except (ValueError, TypeError):
        # 如果转换失败，回退到代码中定义的默认值
        return default

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
    try:
        # 优先通过枚举的键（如 "HYBRID_SEARCH"）进行匹配
        return RetrievalMethod[s.upper()]
    except KeyError:
        # 如果失败，再尝试通过值（如 "混合检索"）进行匹配
        for member in RetrievalMethod:
            if member.value == s:
                return member
        # 如果都失败，返回默认值
        return RetrievalMethod.HYBRID_SEARCH

# =================================================================
# 2. 路径与环境配置 (PATHS & ENVIRONMENT)
# =================================================================
# 从环境变量获取路径，如果未设置则使用默认值
# 默认路径是相对于项目根目录的
ROOT_DIR = Path(__file__).parent.parent.parent
KB_PATH = ROOT_DIR / os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base")
PKL_PATH = ROOT_DIR / os.getenv("PKL_PATH", "data/employee_kb.pkl")
LOG_PATH = ROOT_DIR / os.getenv("LOG_PATH", "data/logs")
CACHE_PATH = ROOT_DIR / ".cache"


# =================================================================
# 3. 配置中心 (CONFIGURATION)
# =================================================================
import json

def get_json_env(key: str) -> dict:
    """从环境变量中读取JSON字符串并解析为字典。"""
    json_str = os.getenv(key)
    if not json_str:
        # 如果环境变量不存在或为空，则抛出错误，强制在.env中定义
        raise ValueError(f"错误：环境变量 '{key}' 未在 .env 文件中设置或为空。")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"错误：环境变量 '{key}' 中的JSON格式无效: {e}")

# --- 从环境变量加载所有模型配置 ---
EMBEDDING_CONFIGS = get_json_env("EMBEDDING_CONFIGURATIONS")
RERANK_CONFIGS = get_json_env("RERANK_CONFIGURATIONS")
LLM_CONFIGS = get_json_env("LLM_CONFIGURATIONS")


# --- 知识库构建配置 ---
# 所有配置均从环境变量驱动，get_env的第二个参数是当.env中未设置该值时的备用默认值
KB_CONFIG = {
    "replace_consecutive_whitespace": get_env("KB_REPLACE_WHITESPACE", False, str_to_bool),
    "remove_extra_spaces": get_env("KB_REMOVE_SPACES", False, str_to_bool),
    "remove_urls_and_emails": get_env("KB_REMOVE_URLS", False, str_to_bool),
    "text_splitter_separators": get_env("KB_SPLITTER_SEPARATORS", ["###"], str_to_list),
    "chunk_size": get_env("KB_CHUNK_SIZE", 1500, int),
    "chunk_overlap": get_env("KB_CHUNK_OVERLAP", 150, int),
    "use_qa_segmentation": get_env("KB_USE_QA_SEGMENTATION", False, str_to_bool),
    "embedding_configurations": EMBEDDING_CONFIGS,
    "active_embedding_configuration": get_env("KB_ACTIVE_EMBEDDING", "google", str),
    "embedding_batch_size": get_env("KB_EMBEDDING_BATCH_SIZE", 32, int),
    "kb_dir": str(KB_PATH),
    "output_file": str(PKL_PATH),
}

# --- 聊天机器人配置 ---
CHAT_CONFIG = {
    "retrieval_method": get_env("CHAT_RETRIEVAL_METHOD", RetrievalMethod.HYBRID_SEARCH, str_to_retrieval_method),
    "vector_weight": get_env("CHAT_VECTOR_WEIGHT", 0.3, float),
    "keyword_weight": get_env("CHAT_KEYWORD_WEIGHT", 0.7, float),
    "rerank_enabled": get_env("CHAT_RERANK_ENABLED", False, str_to_bool),
    "top_k": get_env("CHAT_TOP_K", 5, int),
    "score_threshold": get_env("CHAT_SCORE_THRESHOLD", 0.4, float),
    "active_rerank_configuration": get_env("CHAT_ACTIVE_RERANK", "siliconflow", str),
    "rerank_configurations": RERANK_CONFIGS,
    "active_llm_configuration": get_env("CHAT_ACTIVE_LLM", "google", str),
    "llm_configurations": LLM_CONFIGS,
}

# --- API密钥与URL配置 (完全由环境变量驱动) ---
API_CONFIG = {
    # API Keys
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "SILICONFLOW_API_KEY": os.getenv("SILICONFLOW_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "QWEN_API_KEY": os.getenv("QWEN_API_KEY"),
    "VOLC_ACCESS_KEY": os.getenv("VOLC_ACCESS_KEY"),
    "VOLC_SECRET_KEY": os.getenv("VOLC_SECRET_KEY"),
    "JINA_API_KEY": os.getenv("JINA_API_KEY"),
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    "GROK_API_KEY": os.getenv("GROK_API_KEY"),
    "LM_STUDIO_API_KEY": os.getenv("LM_STUDIO_API_KEY", "lm-studio"),

    # Base URLs
    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
    "SILICONFLOW_BASE_URL": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    "QWEN_BASE_URL": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
    "DEEPSEEK_BASE_URL": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "LM_STUDIO_BASE_URL": os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    "VOLC_BASE_URL": os.getenv("VOLC_BASE_URL", "https://maas-api.ml-platform-cn-beijing.volces.com"),
    "GROK_BASE_URL": os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
}