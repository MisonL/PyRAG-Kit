import os

import pytest
from pydantic import ValidationError

from src.utils.config import (
    ModelDetail,
    ModelProtocol,
    RetrievalMethod,
    Settings,
    get_settings,
    resolve_app_root,
)

MOCK_TOML_CONTENT = """
log_level = "INFO"
knowledge_base_path = "toml_kb_path"
chat_retrieval_method = "全文检索"
chat_top_k = 10
hybrid_fusion_strategy = "weighted"
retrieval_candidate_multiplier = 4
kb_child_chunk_size = 180
kb_child_chunk_overlap = 18
google_genai_use_vertexai = true
gemini_api_key = "toml-gemini-key"
google_cloud_project = "toml-project"
google_cloud_location = "europe-west4"

[embedding_configurations.google]
provider = "google"
model_name = "toml-embedding-model"

[llm_configurations.demo]
provider = "openai"
model_name = "demo-model"
protocol = "responses"
options = { top_p = 0.8, response_format = { type = "json_object" } }
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

    with pytest.raises(ValidationError, match="chat_top_k"):
        Settings(chat_top_k=0)

    with pytest.raises(ValidationError, match="chat_score_threshold"):
        Settings(chat_score_threshold=1.1)


def test_settings_splitter_separators_empty_string_falls_back_to_default():
    settings = Settings(kb_splitter_separators="")
    assert settings.kb_splitter_separators == ["###"]


def test_settings_splitter_separators_blank_csv_falls_back_to_default():
    settings = Settings(kb_splitter_separators=" , ")
    assert settings.kb_splitter_separators == ["###"]


def test_settings_defaults(monkeypatch, tmp_path):
    # 直接构造，避免读取开发者本机的 config.toml：这些断言应验证模型内置默认值，
    # 而不是某个本地文件恰好写了什么。
    monkeypatch.setattr("src.utils.config.CONFIG_TOML_PATH", tmp_path / "absent.toml")
    settings = Settings(_env_file=None)

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
    assert settings.qwen_base_url.endswith("/compatible-mode/v1")
    assert settings.volc_base_url.endswith("/api/v3")
    assert "jina" not in settings.embedding_configurations


def test_settings_builtin_model_ids_are_current(monkeypatch, tmp_path):
    """兜底默认值必须是当前有效的官方 ID，避免新用户照抄到已失效模型。"""
    monkeypatch.setattr("src.utils.config.CONFIG_TOML_PATH", tmp_path / "absent.toml")
    settings = Settings(_env_file=None)

    # Google 的裸名 ``embedding-001`` 不是有效 ID；退役的 ``text-embedding-004``
    # 也不能作为兜底值。
    assert settings.embedding_configurations["google"].model_name == "gemini-embedding-2"
    # SiliconFlow 官方模型广场使用 ``BAAI/`` 命名空间，不存在 ``alibaba/`` 前缀；
    # 现行 rerank ID 是 ``bge-reranker-v2-m3``，没有 ``bge-reranker-large``。
    assert settings.embedding_configurations["siliconflow"].model_name == "BAAI/bge-large-zh-v1.5"
    assert settings.rerank_configurations["siliconflow"].model_name == "BAAI/bge-reranker-v2-m3"
    # Ollama 官方库的现行 tag 是带次版本号的 ``llama3.1``。
    assert settings.llm_configurations["ollama"].model_name == "llama3.1"


def test_settings_dump_hides_credentials_by_default():
    settings = Settings(openai_api_key="sk-test", lm_studio_api_key="local-secret")

    assert "openai_api_key" not in settings.model_dump()
    assert "lm_studio_api_key" not in settings.model_dump_json()
    assert settings.model_dump(include_secrets=True)["openai_api_key"] == "sk-test"


def test_settings_supports_gemini_api_key_alias_without_exposing_it_by_default():
    settings = Settings(gemini_api_key="gemini-test")

    assert settings.gemini_api_key == "gemini-test"
    assert "gemini_api_key" not in settings.model_dump()
    assert settings.model_dump(include_secrets=True)["gemini_api_key"] == "gemini-test"


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
    assert settings.google_genai_use_vertexai is True
    assert settings.gemini_api_key == "toml-gemini-key"
    assert settings.google_cloud_project == "toml-project"
    assert settings.google_cloud_location == "europe-west4"
    assert settings.embedding_configurations["google"].model_name == "toml-embedding-model"
    assert settings.llm_configurations["demo"].provider == "openai"
    assert settings.llm_configurations["demo"].model_name == "demo-model"
    assert settings.llm_configurations["demo"].protocol == ModelProtocol.RESPONSES
    assert settings.llm_configurations["demo"].options["top_p"] == 0.8


def test_model_protocol_aliases_and_invalid_values():
    assert (
        ModelDetail(provider="openai", model_name="gpt-test", protocol="response").protocol
        == ModelProtocol.RESPONSES
    )

    with pytest.raises(ValidationError, match="不支持的模型协议"):
        ModelDetail(provider="openai", model_name="gpt-test", protocol="unknown")


@pytest.mark.parametrize("field_name", ["embedding_configurations", "rerank_configurations"])
def test_settings_rejects_protocol_on_non_llm_configuration(field_name):
    with pytest.raises(ValidationError, match="配置不支持 protocol"):
        Settings(
            **{
                field_name: {
                    "demo": {
                        "provider": "openai",
                        "model_name": "demo-model",
                        "protocol": "responses",
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "X-API-Key",
        "X-Auth-Token",
        "Authorization",
        "access_token",
        "api_key",
        "Bearer",
    ],
)
def test_model_options_reject_credentials_nested_in_headers_or_query(sensitive_key):
    with pytest.raises(ValueError, match="凭证"):
        ModelDetail(
            provider="openai",
            model_name="gpt-test",
            options={"default_headers": {sensitive_key: "secret"}},
        )

    with pytest.raises(ValueError, match="凭证"):
        ModelDetail(
            provider="openai",
            model_name="gpt-test",
            options={"default_query": {sensitive_key: "secret"}},
        )


@pytest.mark.parametrize(
    "container",
    [
        "headers",
        "default_headers",
        "extra_headers",
        "query",
        "default_query",
        "extra_query",
        "http_options",
        "header",
        "params",
        "http_client",
        "httpx_client",
        "httpx_async_client",
        "aiohttp_client",
    ],
)
def test_model_options_reject_request_header_and_query_containers(container):
    with pytest.raises(ValueError, match="凭证或连接字段"):
        ModelDetail(
            provider="openai",
            model_name="gpt-test",
            options={container: {"X-Trace-ID": "trace"}},
        )


@pytest.mark.parametrize("connection_key", ["base_url", "client_args", "async_client_args"])
def test_model_options_reject_connection_overrides(connection_key):
    with pytest.raises(ValueError, match="凭证或连接字段"):
        ModelDetail(
            provider="openai",
            model_name="gpt-test",
            options={connection_key: "https://proxy.invalid"},
        )


def test_settings_from_dotenv(tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        'OPENAI_API_KEY="dotenv_key"\nCHAT_TOP_K=15\n'
        "GOOGLE_GENAI_USE_VERTEXAI=true\n"
        'GOOGLE_CLOUD_PROJECT="dotenv-project"\n'
        'GOOGLE_CLOUD_LOCATION="asia-east1"\n',
        encoding="utf-8",
    )

    settings = get_settings()

    assert settings.openai_api_key == "dotenv_key"
    assert settings.chat_top_k == 15
    assert settings.google_genai_use_vertexai is True
    assert settings.google_cloud_project == "dotenv-project"
    assert settings.google_cloud_location == "asia-east1"


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


def test_resolve_app_root_source_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.utils.config.sys", type("FakeSys", (), {"frozen": False, "executable": ""})()
    )
    monkeypatch.setattr("src.utils.config.__file__", str(tmp_path / "src" / "utils" / "config.py"))

    root = resolve_app_root()

    assert root == tmp_path


def test_resolve_app_root_frozen_mode(monkeypatch, tmp_path):
    executable = tmp_path / "PyRAG-Kit" / "PyRAG-Kit"
    monkeypatch.setattr(
        "src.utils.config.sys",
        type("FakeSys", (), {"frozen": True, "executable": str(executable)})(),
    )

    root = resolve_app_root()

    assert root == executable.parent


def test_settings_warns_on_retired_qwen_base_url():
    """旧 Qwen 端点会 404，加载配置时必须给出显式升级提示。"""
    with pytest.warns(UserWarning, match="qwen_base_url"):
        Settings.model_validate(
            {
                "qwen_base_url": "https://dashscope.aliyuncs.com/api/v1",
                "volc_base_url": "https://ark.cn-beijing.volces.com/api/v3",
            }
        )


def test_settings_warns_on_retired_volc_base_url():
    with pytest.warns(UserWarning, match="volc_base_url"):
        Settings.model_validate(
            {
                "qwen_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "volc_base_url": "https://maas-api.ml-platform-cn-beijing.volces.com",
            }
        )


def test_settings_accepts_current_and_custom_base_urls():
    """新默认值以及自建/代理端点都不应触发升级警告。"""
    import warnings as warnings_module

    for qwen, volc in [
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://ark.cn-beijing.volces.com/api/v3",
        ),
        ("https://my-proxy.internal/v1", "https://my-proxy.internal/v3"),
    ]:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            Settings.model_validate({"qwen_base_url": qwen, "volc_base_url": volc})
        assert not [w for w in caught if "base_url" in str(w.message)]
