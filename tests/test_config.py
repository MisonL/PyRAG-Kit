# -*- coding: utf-8 -*-
import os

import pytest
from pydantic import ValidationError

from src.utils.config import RetrievalMethod, Settings, get_settings


MOCK_TOML_CONTENT = """
log_level = "INFO"
knowledge_base_path = "toml_kb_path"
chat_retrieval_method = "全文检索"
chat_top_k = 10
hybrid_fusion_strategy = "weighted"
retrieval_candidate_multiplier = 4
kb_child_chunk_size = 180
kb_child_chunk_overlap = 18

[embedding_configurations.google]
provider = "google"
model_name = "toml-embedding-model"

[llm_configurations.demo]
provider = "openai"
model_name = "demo-model"
"""


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    monkeypatch.setattr("src.utils.config.CONFIG_TOML_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KNOWLEDGE_BASE_PATH", raising=False)
    monkeypatch.delenv("CHAT_RETRIEVAL_METHOD", raising=False)
    monkeypatch.delenv("CHAT_TOP_K", raising=False)
    monkeypatch.delenv("HYBRID_FUSION_STRATEGY", raising=False)
    monkeypatch.delenv("RETRIEVAL_CANDIDATE_MULTIPLIER", raising=False)
    monkeypatch.delenv("KB_CHILD_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("KB_CHILD_CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("CHAT_TEMPERATURE", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    yield tmp_path
    get_settings.cache_clear()


def test_settings_model_validation():
    Settings(log_level="DEBUG", chat_temperature=0.5)

    with pytest.raises(ValidationError):
        Settings(log_level="INVALID_LEVEL")

    with pytest.raises(ValidationError):
        Settings(chat_temperature=1.5)

    with pytest.raises(ValidationError):
        Settings(chat_retrieval_method="UNKNOWN_METHOD")  # type: ignore[arg-type]


def test_settings_defaults():
    settings = get_settings()

    assert settings.openai_api_key is None
    assert settings.chat_top_k == 5
    assert settings.chat_retrieval_method == RetrievalMethod.HYBRID_SEARCH
    assert settings.knowledge_base_path.endswith("knowledge_base")
    assert settings.snapshot_root.endswith("data/kb")
    assert settings.hybrid_fusion_strategy == "rrf"
    assert settings.retrieval_candidate_multiplier == 3
    assert settings.kb_child_chunk_size == 300
    assert settings.kb_child_chunk_overlap == 30
    assert settings.default_embedding_provider == "local-hash"
    assert settings.embedding_configurations["local-hash"].model_name == "local-hash-256"
    assert settings.llm_configurations["iflow-qwen3-max"].model_name == "qwen3-max"


def test_settings_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(MOCK_TOML_CONTENT, encoding="utf-8")

    settings = get_settings()

    assert settings.log_level == "INFO"
    assert settings.chat_top_k == 10
    assert settings.chat_retrieval_method == RetrievalMethod.FULL_TEXT_SEARCH
    assert settings.knowledge_base_path.endswith("toml_kb_path")
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 4
    assert settings.kb_child_chunk_size == 180
    assert settings.kb_child_chunk_overlap == 18
    assert settings.embedding_configurations["google"].model_name == "toml-embedding-model"
    assert settings.llm_configurations["demo"].provider == "openai"
    assert settings.llm_configurations["demo"].model_name == "demo-model"


def test_settings_from_dotenv(tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text('OPENAI_API_KEY="dotenv_key"\nCHAT_TOP_K=15', encoding="utf-8")

    settings = get_settings()

    assert settings.openai_api_key == "dotenv_key"
    assert settings.chat_top_k == 15


def test_settings_from_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_var_key")
    monkeypatch.setenv("CHAT_TOP_K", "20")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "5")

    settings = get_settings()

    assert settings.openai_api_key == "env_var_key"
    assert settings.chat_top_k == 20
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 5


def test_settings_priority(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
log_level = "INFO"
chat_top_k = 10
chat_temperature = 0.7

[llm_configurations.demo]
provider = "openai"
model_name = "toml-model"
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        'OPENAI_API_KEY="dotenv_key"\nCHAT_TOP_K=15\nCHAT_TEMPERATURE=0.8',
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "env_var_key")
    monkeypatch.setenv("CHAT_TEMPERATURE", "0.9")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "6")

    settings = get_settings()

    assert settings.openai_api_key == "env_var_key"
    assert settings.chat_top_k == 15
    assert settings.log_level == "INFO"
    assert settings.chat_temperature == 0.9
    assert settings.hybrid_fusion_strategy == "weighted"
    assert settings.retrieval_candidate_multiplier == 6
    assert settings.llm_configurations["demo"].model_name == "toml-model"


def test_settings_singleton():
    settings1 = get_settings()
    settings1.log_level = "CHANGED"

    settings2 = get_settings()

    assert settings1 is settings2
    assert settings2.log_level == "CHANGED"


def test_settings_path_resolution(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", "my_kb")

    settings = get_settings()

    assert os.path.isabs(settings.knowledge_base_path)
    assert settings.knowledge_base_path.endswith("my_kb")
