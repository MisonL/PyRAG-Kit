# -*- coding: utf-8 -*-
import pytest
import os
import json
from pathlib import Path
from pydantic import ValidationError
from src.utils.config import Settings, RetrievalMethod, get_settings, ROOT_DIR, CONFIG_PATH

# 模拟 config.ini 文件内容
MOCK_INI_CONTENT = """
[API_KEYS]
OPENAI_API_KEY = "ini_key"

[PATHS]
KNOWLEDGE_BASE_PATH = "ini_kb_path"

[CHAT]
CHAT_RETRIEVAL_METHOD = "全文检索"
CHAT_TOP_K = 10
HYBRID_FUSION_STRATEGY = "weighted"
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
KB_CHILD_CHUNK_SIZE = 180
KB_CHILD_CHUNK_OVERLAP = 18
"""

# 测试 Settings 模型的基本验证
def test_settings_model_validation():
    """测试 Settings 模型的基本数据验证逻辑。"""
    # 测试有效数据
    Settings(log_level="DEBUG", chat_temperature=0.5)
    
    # 测试无效日志级别
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID_LEVEL")
        
    # 测试无效温度
    with pytest.raises(ValidationError):
        Settings(chat_temperature=1.5)
        
    # 测试无效检索方法
    with pytest.raises(ValidationError):
        Settings(chat_retrieval_method="UNKNOWN_METHOD") # type: ignore

# 测试配置加载
@pytest.fixture(autouse=True)
def clear_cache_and_env(monkeypatch):
    """在每个测试运行前自动清理缓存和环境变量。"""
    get_settings.cache_clear()
    # 清理可能影响测试的环境变量
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KNOWLEDGE_BASE_PATH", raising=False)
    monkeypatch.delenv("CHAT_RETRIEVAL_METHOD", raising=False)
    monkeypatch.delenv("CHAT_TOP_K", raising=False)
    monkeypatch.delenv("HYBRID_FUSION_STRATEGY", raising=False)
    monkeypatch.delenv("RETRIEVAL_CANDIDATE_MULTIPLIER", raising=False)
    monkeypatch.delenv("KB_CHILD_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("KB_CHILD_CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    # 确保测试期间没有 .env 文件
    if (ROOT_DIR / ".env").exists():
        (ROOT_DIR / ".env").unlink()
    # 确保测试期间没有 config.ini 文件
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()

def test_settings_defaults():
    """测试在没有任何配置文件或环境变量时，加载的是默认值。"""
    settings = get_settings()
    assert settings.openai_api_key is None
    assert settings.chat_top_k == 5
    assert settings.chat_retrieval_method == RetrievalMethod.HYBRID_SEARCH
    assert settings.knowledge_base_path.endswith("knowledge_base")
    assert settings.hybrid_fusion_strategy == "rrf"
    assert settings.retrieval_candidate_multiplier == 3
    assert settings.kb_child_chunk_size == 300
    assert settings.kb_child_chunk_overlap == 30

def test_settings_from_ini(monkeypatch):
    """测试配置仅从 config.ini 文件加载。"""
    # 模拟 config.ini 文件
    monkeypatch.setattr("src.utils.config.CONFIG_PATH", ROOT_DIR / "test_config.ini")
    (ROOT_DIR / "test_config.ini").write_text(MOCK_INI_CONTENT)
    
    settings = get_settings()
    
    assert settings.openai_api_key == "ini_key"
    assert settings.chat_top_k == 10
    assert settings.chat_retrieval_method == RetrievalMethod.FULL_TEXT_SEARCH
    assert settings.knowledge_base_path.endswith("ini_kb_path")
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 4
    assert settings.kb_child_chunk_size == 180
    assert settings.kb_child_chunk_overlap == 18
    
    # 清理
    (ROOT_DIR / "test_config.ini").unlink()

def test_settings_from_dotenv(monkeypatch):
    """测试配置仅从 .env 文件加载。"""
    # 模拟 .env 文件
    (ROOT_DIR / ".env").write_text('OPENAI_API_KEY="dotenv_key"\nCHAT_TOP_K=15')
    
    settings = get_settings()
    
    assert settings.openai_api_key == "dotenv_key"
    assert settings.chat_top_k == 15

def test_settings_from_env_vars(monkeypatch):
    """测试配置仅从环境变量加载。"""
    monkeypatch.setenv("OPENAI_API_KEY", "env_var_key")
    monkeypatch.setenv("CHAT_TOP_K", "20")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "5")
    
    settings = get_settings()
    
    assert settings.openai_api_key == "env_var_key"
    assert settings.chat_top_k == 20
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 5

def test_settings_priority(monkeypatch):
    """测试加载优先级: 环境变量 > .env > config.ini > 默认值。"""
    # 1. 设置 config.ini (最低优先级)
    ini_content = """
[API_KEYS]
OPENAI_API_KEY = "ini_key"
[CHAT]
CHAT_TOP_K = 10
CHAT_TEMPERATURE = 0.7
[GENERAL]
LOG_LEVEL = "INFO"
"""
    monkeypatch.setattr("src.utils.config.CONFIG_PATH", ROOT_DIR / "test_config.ini")
    (ROOT_DIR / "test_config.ini").write_text(ini_content)

    # 2. 设置 .env (中等优先级)
    (ROOT_DIR / ".env").write_text('OPENAI_API_KEY="dotenv_key"\nCHAT_TOP_K=15\nCHAT_TEMPERATURE=0.8')
    
    # 3. 设置环境变量 (最高优先级)
    monkeypatch.setenv("OPENAI_API_KEY", "env_var_key")
    monkeypatch.setenv("CHAT_TEMPERATURE", "0.9")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "6")
    
    # --- 断言 ---
    settings = get_settings()
    
    # OPENAI_API_KEY: 被环境变量覆盖
    assert settings.openai_api_key == "env_var_key"
    # CHAT_TOP_K: 被 .env 文件覆盖
    assert settings.chat_top_k == 15
    # LOG_LEVEL: 来自 .ini 文件
    assert settings.log_level == "INFO"
    # CHAT_TEMPERATURE: 被环境变量覆盖，并正确转换为 float
    assert settings.chat_temperature == 0.9
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 6
    
    # 清理
    (ROOT_DIR / "test_config.ini").unlink()

def test_settings_singleton():
    """测试 get_settings() 返回的是单例。"""
    settings1 = get_settings()
    settings1.log_level = "CHANGED"
    
    settings2 = get_settings()
    
    assert settings1 is settings2
    assert settings2.log_level == "CHANGED"

def test_settings_path_resolution(monkeypatch):
    """测试路径属性的正确解析。"""
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "my_kb")
    
    settings = get_settings()
    
    assert os.path.isabs(settings.knowledge_base_path)
    assert settings.knowledge_base_path.endswith("my_kb")
