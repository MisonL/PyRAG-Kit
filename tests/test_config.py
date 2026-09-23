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


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_message"),
    [
        ("log_retention_days", -5, "log_retention_days 必须是 1 到 3650 之间的整数"),
        ("log_retention_days", 0, "log_retention_days 必须是 1 到 3650 之间的整数"),
        ("kb_chunk_size", 0, "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size"),
        ("kb_chunk_size", -100, "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size"),
        ("kb_chunk_overlap", -1, "kb_chunk_overlap/kb_child_chunk_overlap 必须是非负整数"),
        ("kb_child_chunk_size", 0, "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size"),
        ("kb_child_chunk_overlap", -1, "kb_chunk_overlap/kb_child_chunk_overlap 必须是非负整数"),
        ("kb_embedding_batch_size", 0, "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size"),
        (
            "kb_embedding_batch_size",
            -1,
            "kb_chunk_size/kb_child_chunk_size/kb_embedding_batch_size",
        ),
        ("chat_vector_weight", -1.0, "chat_vector_weight/chat_keyword_weight 必须在 0 到 1 之间"),
        ("chat_vector_weight", 5.0, "chat_vector_weight/chat_keyword_weight 必须在 0 到 1 之间"),
        ("chat_keyword_weight", 2.0, "chat_vector_weight/chat_keyword_weight 必须在 0 到 1 之间"),
    ],
)
def test_numeric_settings_reject_illegal_values(field_name, bad_value, expected_message):
    """数值字段的非法值必须在加载期报错，而不是拖到分片/检索期。

    ``kb_chunk_size=0`` 此前会一路通过配置校验，直到 langchain 在分片阶段
    才抛 ``chunk_size must be > 0``；负权重会反向加成分数。UI 层
    （``src/ui/config_menu.py``）已有 0..1 校验，TOML/env 路径此前没有。

    断言用**该 validator 独有的消息片段**，而不是 ``match=field_name``：跨字段
    的 ``model_validator`` 报错文本里也会出现 ``kb_chunk_size`` 等字段名（例如
    「kb_chunk_overlap 必须小于 kb_chunk_size。」），用字段名匹配会让这些用例在
    被保护的 validator 被删除后依然变绿（假绿）。
    """
    with pytest.raises(ValidationError, match=expected_message):
        Settings(**{field_name: bad_value})


def test_chunk_overlap_must_be_strictly_smaller_than_chunk_size():
    """overlap 必须严格小于 size，否则分片无法推进（会得到空分片或死循环）。"""
    with pytest.raises(ValidationError, match="kb_chunk_overlap"):
        Settings(kb_chunk_size=100, kb_chunk_overlap=100)
    with pytest.raises(ValidationError, match="kb_child_chunk_overlap"):
        Settings(kb_child_chunk_size=50, kb_child_chunk_overlap=50)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("chat_top_k", 10**9),
        ("chat_top_k", 10**18),
        ("retrieval_candidate_multiplier", 10**9),
        ("retrieval_candidate_multiplier", 10**18),
    ],
)
def test_retrieval_sizes_reject_unbounded_values(field_name, bad_value):
    """检索规模必须有上界。

    ``effective_top_k = chat_top_k * retrieval_candidate_multiplier`` 直通
    ``faiss_index.search()``，FAISS 不报错也不截断：实测 ``chat_top_k=10**9``
    会被接受，单次查询真实分配约 3.6 GB 数组、RSS 涨 10.3 GB 并挂起 8.4 秒。
    """
    with pytest.raises(ValidationError, match=field_name):
        Settings(**{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name",
    [
        "log_retention_days",
        "kb_chunk_size",
        "kb_child_chunk_size",
        "kb_embedding_batch_size",
        "kb_chunk_overlap",
        "kb_child_chunk_overlap",
        "chat_top_k",
        "retrieval_candidate_multiplier",
        "chat_vector_weight",
        "chat_keyword_weight",
        "chat_score_threshold",
        "chat_temperature",
    ],
)
def test_int_and_float_settings_reject_bool(field_name):
    """``True``/``False`` 不得被当作 1/0 静默接受。

    Python 里 ``bool`` 是 ``int`` 的子类，不做 ``isinstance(value, bool)`` 短路
    的话 ``Settings(chat_top_k=True)`` 会静默变成 ``1``。这类配置错误应显式报错。
    """
    with pytest.raises(ValidationError, match=field_name):
        Settings(**{field_name: True})


def test_child_chunk_size_cannot_exceed_parent_chunk_size():
    """子分片不得大于父分片，否则层级结构静默退化为一层。

    实测 ``kb_chunk_size=300, kb_child_chunk_size=1500`` 被接受后，父块数从 2
    涨到 10（每个父块只产出 1 个子块），「先粗后细」的父子检索意图失效。
    """
    with pytest.raises(ValidationError, match="kb_child_chunk_size"):
        Settings(kb_chunk_size=300, kb_child_chunk_size=1500)
    with pytest.raises(ValidationError, match="kb_child_chunk_size"):
        Settings(kb_chunk_size=100, kb_child_chunk_size=100)
    # 相等也应拒绝：与 overlap<size 同理，层级需要严格的大小关系。
    with pytest.raises(ValidationError, match="kb_child_chunk_size"):
        Settings(kb_chunk_size=500, kb_child_chunk_size=500)


def test_hybrid_weights_cannot_both_be_zero():
    """两个融合权重不得同时为 0，否则检索结果被静默全部丢弃。

    ``retrieval_service._weight_tuple()`` 在 ``total <= 0`` 时返回 (0.0, 0.0)，
    所有候选得分为 0；再叠加 weighted 策略启用阈值过滤（默认 0.4），
    结果是空列表且无任何报错。
    """
    with pytest.raises(ValidationError, match="不能同时为 0"):
        Settings(chat_vector_weight=0.0, chat_keyword_weight=0.0)


def test_hybrid_weights_single_zero_is_still_allowed():
    """只有一个为 0 是合法配置（纯向量检索 / 纯关键词检索）。"""
    assert Settings(chat_vector_weight=0.0, chat_keyword_weight=1.0).chat_vector_weight == 0.0
    assert Settings(chat_vector_weight=1.0, chat_keyword_weight=0.0).chat_keyword_weight == 0.0


def test_numeric_settings_accept_legal_boundary_values():
    """合法边界值必须仍然可用，避免校验过紧。

    注意 ``kb_child_chunk_size`` 必须是**严格小于** ``kb_chunk_size``：层级分片
    需要这个严格关系，越小越退化。实测 ``parent=child=1`` 时不同父块数等于分块数
    （30/30），即每个父块只产出一个子块，与 ``child > parent`` 一样退化为一层。
    """
    settings = Settings(
        log_retention_days=1,
        kb_chunk_size=2,
        kb_chunk_overlap=0,
        kb_child_chunk_size=1,
        kb_child_chunk_overlap=0,
        kb_embedding_batch_size=1,
        chat_vector_weight=0.0,
        chat_keyword_weight=1.0,
    )
    assert settings.kb_chunk_overlap == 0
    assert settings.kb_child_chunk_size == 1
    assert settings.chat_vector_weight == 0.0
    assert settings.chat_keyword_weight == 1.0


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


# ── 回归：配置校验失败不能把凭证交给解释器默认 handler ──


def test_get_settings_redacts_credential_in_validation_error(isolated_config):
    """``Settings()`` 在 import 期就被调用（``log_manager.get_module_logger``），
    早于任何入口的 ``try``。pydantic 的默认渲染会把 ``input_value`` 明文交给
    解释器默认 handler，而 ``options`` 是文档指定的扩展入口——把 ``api_key``
    放进去恰好就是被校验拒绝的那类错误，即校验越严格越容易走到这条路径。
    """
    secret = "sk-proj-FAKE0000FAKE0000FAKE0000FAKE0000"
    (isolated_config / "config.toml").write_text(
        MOCK_TOML_CONTENT + f'\n[llm_configurations.leaky]\nprovider = "openai"\nmodel_name = "m"\n'
        f'options = {{ api_key = "{secret}" }}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    assert secret not in message
    assert "llm_configurations.leaky.options" in message
    assert "REDACTED" in message


def test_get_settings_reports_non_credential_validation_errors(isolated_config):
    """脱敏边界不能把普通配置错误也变成不可读的嵌套报告。"""
    (isolated_config / "config.toml").write_text(
        MOCK_TOML_CONTENT + "\nchat_top_k = 0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="chat_top_k"):
        get_settings()


# ── 回归：数值解析不得宽松放行「看起来成功」的形态 ──


@pytest.mark.parametrize(
    "raw",
    [
        "1_0",  # Python 下划线分组：int("1_0") == 10、float("1_0") == 10.0
        "+7",  # 正号：float("+7") == 7.0
        "0x10",  # 十六进制
        "inf",
        "-inf",
        "nan",
        "1 500",
        "1,500",
        "",
        "abc",
    ],
)
def test_numeric_settings_reject_non_decimal_string_forms(raw):
    """字符串数值只接受十进制字面量，不放行下划线分组/正号/十六进制/非有限值。

    ``int()`` 与 ``float()`` 都接受 ``"1_0"`` 这类 Python 字面量形态
    （``int("1_0") == 10``、``float("1_0") == 10.0``），配置里写错时会被静默
    读成另一个数——安静的成功正是本项目反复踩到的模式，因此在解析前先做形状校验。
    """
    with pytest.raises(ValidationError, match="无法把 kb_chunk_size"):
        Settings(kb_chunk_size=raw)


@pytest.mark.parametrize(
    "raw,expected",
    [("1500", 1500), (" 1500 ", 1500), ("2000", 2000), ("1500\n", 1500)],
)
def test_numeric_settings_accept_decimal_integer_strings(raw, expected):
    """收紧不能误伤：正常十进制字符串（含首尾空白）仍必须可用。

    取值都避开既有的交叉约束 ``kb_chunk_overlap < kb_chunk_size``：overlap 默认
    150，所以 ``kb_chunk_size="1"`` 会被它拒绝，与本次形状校验无关。
    """
    assert Settings(kb_chunk_size=raw).kb_chunk_size == expected


def test_numeric_settings_accept_scientific_notation_for_float_fields():
    """float 字段仍接受科学计数法——这是合法的十进制字面量。"""
    assert Settings(chat_score_threshold="1e-1").chat_score_threshold == pytest.approx(0.1)
    assert Settings(chat_temperature="1e0").chat_temperature == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("log_retention_days", 10**400),
        ("kb_chunk_size", 10**400),
        ("kb_child_chunk_size", 10**400),
        ("kb_embedding_batch_size", 10**400),
        ("log_retention_days", "1" + "0" * 400),
        ("kb_chunk_size", "1" + "0" * 400),
    ],
)
def test_integer_settings_reject_overflow_magnitudes(field_name, bad_value):
    """溢出量级的整数必须在加载期拒绝，而不是拖到运算期。

    实测修复前 ``log_retention_days=10**400`` 与 ``kb_chunk_size=10**400``
    都被接受（这两个字段当时只有下界），直到日期减法／内存申请才炸。
    """
    with pytest.raises(ValidationError, match=field_name):
        Settings(**{field_name: bad_value})


def test_score_threshold_overflow_is_a_validation_error_not_overflow_error():
    """``float(10**400)`` 抛的是裸 ``OverflowError``；它**不是** ``ValidationError``，
    会穿透 ``get_settings`` 的 ``except`` 变成未脱敏 traceback。因此范围判断必须在
    转 float 之前完成。

    这里用 ``Settings`` 直接断言异常类型：``get_settings`` 不接受 kwargs（它从
    TOML/env 取值），无法注入这种量级的值，而两条路径共用同一个 validator。
    """
    for bad in (10**400, "1" + "0" * 400):
        with pytest.raises(ValidationError, match="chat_score_threshold") as excinfo:
            Settings(chat_score_threshold=bad)
        assert not isinstance(excinfo.value.__cause__, OverflowError)


def test_get_settings_wraps_settings_error_from_unparseable_env_list(monkeypatch, isolated_config):
    """列表类 env 字段写错时 pydantic-settings 抛 ``SettingsError``，它继承
    ``ValueError`` 但**不是** ``ValidationError``，原先的 handler 接不住，用户看到
    的是解释器默认 handler 打出的多屏 traceback。现在必须变成一行脱敏消息。
    """
    monkeypatch.setenv("KB_SPLITTER_SEPARATORS", "###")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="配置解析失败") as excinfo:
        get_settings()

    get_settings.cache_clear()
    assert "kb_splitter_separators" in str(excinfo.value)


# ── 回归：仓库自身代码路径不得产生被吞掉的警告 ──


def test_repository_code_paths_emit_no_swallowed_warnings():
    """走一遍本仓库的配置加载路径，断言不产生 UserWarning/DeprecationWarning。

    ``pytest.ini`` 原先用全局 ``ignore::UserWarning`` / ``ignore::DeprecationWarning``
    把它们静音了：实测同一个探针用例在全局忽略下静默通过、在
    ``-W error::UserWarning`` 下失败，说明警告确实发生却被吞掉。现在过滤只按
    第三方 module 前缀限定，所以本仓库代码产生的警告必须在这里被显式盯住——
    否则「收窄过滤」本身也会在未来某次改动里悄悄失效。
    """
    import warnings

    get_settings.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            settings = get_settings()
            # 触发几条常见的配置读写路径
            settings.model_dump()
            build_runtime_paths = resolve_app_root()
            assert build_runtime_paths is not None
    finally:
        get_settings.cache_clear()

    offenders = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught
        if issubclass(warning.category, (UserWarning, DeprecationWarning))
    ]
    assert not offenders, f"本仓库代码产生了警告，请修复而不是加过滤: {offenders}"
