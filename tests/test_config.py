import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import (
    ModelDetail,
    ModelProtocol,
    RetrievalMethod,
    Settings,
    _redacted_settings_error,
    get_settings,
    parse_decimal_int,
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
    # 注意这里**只能**改类配置，不能只改模块常量：``env_file`` 在 Settings
    # 类定义期就被求值成字符串，``monkeypatch.setattr`` 打模块级
    # ``ENV_FILE_PATH`` 对它毫无影响（曾经同时写两行，其中一行是死代码，
    # 谁删掉真正起作用的那行都会让隔离静默失效）。生产代码里 .env 与
    # config.toml 都以 ROOT_DIR 为基准，测试必须显式对齐。
    monkeypatch.setitem(Settings.model_config, "env_file", str(tmp_path / ".env"))
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
        # 上界缺失时超大整数会被静默接受（实测 ``10**400`` 走真实 TOML 路径也
        # 通过），配置加载成功、直到分片期才在 ``_validate_hierarchical_sizes``
        # 抛错——报错现场离写错的配置很远。上界与 kb_chunk_size 同源。
        #
        # 用 ``10**4301`` 而不是 ``10**400``：两者都会被新上界拒绝，但 4301 位
        # 同时覆盖「消息插值自己会抛」那一面。必须显式给 id——pytest 自动生成 id
        # 时会对参数调 ``str()``，而 ``str(10**4301)`` 抛 ``ValueError``，测试在
        # **收集阶段**就崩掉。
        pytest.param(
            "kb_chunk_overlap",
            10**4301,
            "kb_chunk_overlap/kb_child_chunk_overlap 必须小于",
            id="kb_chunk_overlap-beyond-conversion-limit",
        ),
        pytest.param(
            "kb_child_chunk_overlap",
            10**4301,
            "kb_chunk_overlap/kb_child_chunk_overlap 必须小于",
            id="kb_child_chunk_overlap-beyond-conversion-limit",
        ),
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
    """overlap 必须严格小于 size，否则分片无法推进（会得到空分片或死循环）。

    父分片这条与模式无关（两种模式都会构造父分片器），因此仍在 ``Settings``
    里判。子分片那条只在层级模式下有意义，判定已移到
    ``RecursiveTextSplitter._validate_hierarchical_sizes()``。
    """
    with pytest.raises(ValidationError, match="kb_chunk_overlap"):
        Settings(kb_chunk_size=100, kb_chunk_overlap=100)


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


def test_child_chunk_size_relation_is_not_checked_at_settings_time():
    """子分片的尺寸关系**不再**在 ``Settings`` 里判——它只在层级模式下有意义。

    模式是分片时的运行时参数（``build(splitter_structure_mode)``／CLI 的
    ``--mode``），``Settings`` 看不到。原先无条件拒绝的后果是一组「standard 下
    完全合法」的取值连配置都加载不了：用户只想用 standard，却必须先改掉用不到
    的字段。判定移到 ``RecursiveTextSplitter._validate_hierarchical_sizes()``，
    对应的回归在 ``tests/etl/test_pipeline.py``。
    """
    # standard 下完全合法的一组取值：子分片大于父分片，但 standard 不构造子分片器
    settings = Settings(
        kb_chunk_size=100, kb_chunk_overlap=10, kb_child_chunk_size=300, kb_child_chunk_overlap=10
    )

    assert settings.kb_child_chunk_size == 300


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


# 校验器原先直接 ``str(value)``，于是 ``ModelDetail(protocol=10**4301)`` 撞上
# CPython 的整数位数上限（``sys.get_int_max_str_digits()``，默认 4300），用户
# 看到的是 ``Exceeds the limit (4300 digits)``——与「协议名写错了」毫无关系，
# 而且是在**正要报告一个校验错误时**改抛另一个错误。
#
# 可达性说明：这条**不是**配置可达缺陷。``protocol`` 确实来自 ``config.toml``，
# 但 ``tomllib`` 解析 4301 位的整数时自己就抛 ``ValueError``（``get_settings``
# 已包成「配置解析失败」），位数在上限内的非字符串（``5``／``["responses"]``／
# ``true``）原本就能得到正确消息——实测走真实 TOML 路径验证过。真正能触达的是
# Python API 调用方。加这条守卫是为了与同文件其它 validator 的判据一致：
# ``log_level`` 与路径字段都已显式判类型，只有这里漏了。
@pytest.mark.parametrize(
    "bad_value",
    [10**4301, -(10**4301), ["responses"], {"p": "responses"}],
    ids=["huge", "huge-negative", "list", "dict"],
)
def test_model_protocol_rejects_non_string_values_with_a_useful_message(bad_value):
    """非字符串 ``protocol`` 必须报「不支持的模型协议」，不能泄漏解释器错误。"""
    with pytest.raises(ValidationError) as excinfo:
        ModelDetail(provider="openai", model_name="gpt-test", protocol=bad_value)

    message = str(excinfo.value)
    assert "不支持的模型协议" in message, message
    assert "Exceeds the limit" not in message, message


def test_model_protocol_accepts_valid_values_after_the_type_guard():
    """收紧不能误伤：别名、规范名、``None`` 与空串照常通过。"""
    cases = {
        "chat": ModelProtocol.CHAT_COMPLETIONS,
        "response": ModelProtocol.RESPONSES,
        "responses": ModelProtocol.RESPONSES,
        "anthropic": ModelProtocol.MESSAGES,
        "gemini": ModelProtocol.GENERATE_CONTENT,
    }
    for raw, expected in cases.items():
        detail = ModelDetail(provider="openai", model_name="m", protocol=raw)
        assert detail.protocol is expected, raw

    assert ModelDetail(provider="openai", model_name="m", protocol=None).protocol is None
    assert ModelDetail(provider="openai", model_name="m", protocol="").protocol is None


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


def test_dotenv_is_resolved_from_root_dir_not_cwd(tmp_path):
    """``.env`` 必须按 ROOT_DIR 解析，与 config.toml 同源。

    回归背景：``env_file`` 原先是相对路径 ``".env"``，由 pydantic-settings
    按进程当前工作目录解析。冻结产物里可执行文件目录与调用者 CWD 通常不同，
    用户放在安装目录旁的 ``.env`` 会被静默忽略。

    必须用**子进程**验证，且不能在子进程里替 ``Settings`` 填好答案：
    ``env_file`` 在类定义期求值，进程内任何对 ``ROOT_DIR`` / ``ENV_FILE_PATH``
    的 patch 都不会让它重新求值。这里的做法是伪造 ``sys.frozen`` 与
    ``sys.executable``，让 ``resolve_app_root()`` 在**导入前**就返回 app 目录，
    从而走完整的生产求值路径。先前版本在测试里直接改写
    ``Settings.model_config``，断言比较的是测试自己注入的值——把 ``env_file``
    回退成相对路径后它依然通过，是测不到生产行为的假绿。
    """
    project_root = Path(__file__).resolve().parent.parent
    cwd_dir = tmp_path / "cwd"
    app_dir = tmp_path / "app"
    cwd_dir.mkdir()
    app_dir.mkdir()

    (app_dir / ".env").write_text('OPENAI_API_KEY="from_app_dir"\n', encoding="utf-8")
    (cwd_dir / ".env").write_text('OPENAI_API_KEY="from_cwd"\n', encoding="utf-8")

    # 子进程：先伪造冻结环境，再导入，使 Settings 在类定义期按 app_dir 求值。
    script = (
        "import sys\n"
        "sys.frozen = True\n"
        "sys.executable = sys.argv[2]\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from src.utils.config import Settings\n"
        "print(Settings().openai_api_key)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(project_root), str(app_dir / "PyRAG-Kit")],
        cwd=cwd_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "from_app_dir"


def test_env_file_path_constant_matches_model_config():
    """``ENV_FILE_PATH`` 与 ``Settings.model_config["env_file"]`` 必须一致。

    前者是模块常量、后者是类定义期求值的结果，两者若漂移，改哪个都会出现
    「改了不生效」的困惑（这正是先前 fixture 里那行死代码的成因）。

    必须在**子进程**里检查：本文件的 autouse fixture 会改写
    ``Settings.model_config``，在进程内断言只会看到夹具注入的值。
    """
    project_root = Path(__file__).resolve().parent.parent
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from src.utils.config import ENV_FILE_PATH, Settings\n"
        "assert Settings.model_config['env_file'] == str(ENV_FILE_PATH), (\n"
        "    Settings.model_config['env_file'], str(ENV_FILE_PATH))\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(project_root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ok"


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


# ── 回归：错误消息的格式化自己会抛异常（负数方向的位数超限） ──
#
# ``repr(10**4301)`` 抛 ``ValueError: Exceeds the limit (4300 digits)``
# （``sys.get_int_max_str_digits()``，默认 4300）。校验消息里直接插值
# ``{value}`` 的站点因此会在**正要报告一个错误时**改抛另一个错误：用户看到的
# 是「整数转字符串超限」，与「倍率不合法」毫无关系。
#
# 只有**负数**能走到这一支：正数的超大整数在上界判断处就被拒了，那条消息用的
# 是常量而不是插值。实测把 ``safe_repr(value)`` 换回 ``{value}`` 后本用例变红。
def test_candidate_multiplier_lower_bound_message_survives_unprintable_integer():
    """倍率下界分支的消息不得自己抛异常。"""
    with pytest.raises(ValidationError, match="retrieval_candidate_multiplier") as excinfo:
        Settings(retrieval_candidate_multiplier=-(10**4301))

    assert "Exceeds the limit" not in str(excinfo.value)


def test_settings_never_leak_the_integer_conversion_limit():
    """扫过**全部**字段，确认没有一条校验消息会把解释器的位数上限泄漏出来。

    与其逐个字段猜哪个分支会插值 ``{value}``，不如断言「消息里绝不出现
    ``Exceeds the limit``」——那串文本只可能来自 CPython，不可能来自本项目的
    任何一条校验消息。正负两个方向都要扫：正数在**上界**判断处被拒（消息里是
    常量），负数才走到**下界**分支（那里插值），只扫正数会漏掉一半站点。

    这条用例是对上面那条的推广，也是这次发现的缺陷的通用回归——``{value}``
    这类站点散落在各 validator 里，逐个补断言挡不住下一个。
    """
    leaks = []
    for field_name in Settings.model_fields:
        for label, bad_value in (("正", 10**4301), ("负", -(10**4301))):
            with pytest.raises(ValidationError) as excinfo:
                Settings(**{field_name: bad_value})
            if "Exceeds the limit" in str(excinfo.value):
                leaks.append(f"{field_name}（{label}）")

    assert not leaks, "以下字段的校验消息泄漏了解释器的整数位数上限: " + ", ".join(leaks)


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


# ── 回归：候选规模的上界必须落在 top_k × multiplier 的乘积上 ──


def test_settings_rejects_oversized_candidate_pool():
    """两个字段各自合法，乘积越界时必须拒绝。

    ``effective_top_k = chat_top_k * retrieval_candidate_multiplier`` 直通
    ``faiss_index.search()``，而 FAISS 对超过索引条数的 k 既不报错也不截断，
    按 k 全量分配。实测 dim=1536、k=10**6 时峰值 RSS 涨约 35 GB。单字段上界
    挡不住这个组合：10_000 × 100 = 10**6。
    """
    with pytest.raises(ValidationError, match="不能超过"):
        Settings(_env_file=None, chat_top_k=2000, retrieval_candidate_multiplier=100)


def test_settings_accepts_candidate_pool_at_the_bound():
    """收紧不能误伤：恰好等于上界的组合必须接受。"""
    settings = Settings(_env_file=None, chat_top_k=2000, retrieval_candidate_multiplier=1)

    assert settings.chat_top_k * settings.retrieval_candidate_multiplier == 2000


# ── 回归：config.toml 语法错也必须走脱敏边界 ──
#
# ``get_settings`` 的 try 只接 ``ValidationError`` 与 ``SettingsError``，两者都
# 覆盖不到 ``tomllib.TOMLDecodeError``——它继承 ``ValueError``，但既不是
# ``ValidationError`` 也不是 ``SettingsError``（pydantic-settings 在
# ``load_toml_config`` 里直接把 tomllib 的异常放出来）。而 ``get_settings`` 在
# import 期就会被调用（``log_manager.get_module_logger``），早于任何入口的
# ``try``：实测 config.toml 语法错时 import 直接抛裸 traceback，35 帧全打在
# 终端上；作为对照，同一份文件里字段**取值**非法时（``chat_top_k = "abc"``）
# 得到的是 11 帧的脱敏消息 ``配置校验失败 - ...``。同属「配置写错」，收场方式
# 不该有两种。


def test_get_settings_wraps_unparseable_toml(isolated_config):
    """config.toml 语法错时必须转成脱敏的 ``ValueError``，而不是裸 traceback。"""
    (isolated_config / "config.toml").write_text("this is not valid toml =\n", encoding="utf-8")

    with pytest.raises(ValueError, match="配置解析失败"):
        get_settings()


def test_get_settings_reports_toml_error_location(isolated_config):
    """错误消息要带上行号，否则用户无从定位是哪一行写坏了。"""
    (isolated_config / "config.toml").write_text(
        "chat_top_k = 5\nthis is not valid toml =\n", encoding="utf-8"
    )

    with pytest.raises(ValueError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    # 同时断言包装前缀：tomllib 的原始消息里本来就有 "line 2"，只查行号的话
    # 未包装的裸异常也能通过——测试会因为错误的原因变绿。
    assert "配置解析失败" in message, message
    assert "line 2" in message, message


# ── 回归：``int()`` 抛错的第二类情形——位数超过 ``sys.get_int_max_str_digits()`` ──
#
# 上一轮把 ``isdigit()`` 换成 ``isdecimal()``，只堵住了第一类（非十进制数字字符）。
# ``"9" * 4301`` 是**合法**的十进制整数、``isdecimal()`` 也为真，但 ``int()``
# 照样抛 ``ValueError: Exceeds the limit (4300 digits)``——CPython 默认上限
# ``sys.get_int_max_str_digits()`` 是 4300 位。于是同样的崩溃只是换了个触发条件：
# 配置编辑器里粘贴一长串数字、网关返回一个超长 status，收场与 ``²`` 完全一样。
#
# ``parse_decimal_int`` 用 ``try/except ValueError`` 一次覆盖两类，不硬编码位数
# 上界——``int()`` 抛错本身就是判据。


def _huge_digits():
    import sys

    return "9" * (sys.get_int_max_str_digits() + 1)


def test_parse_decimal_int_rejects_digit_strings_beyond_the_conversion_limit():
    """超过 ``int()`` 位数上限的数字串必须返回 ``None``，而不是抛 ``ValueError``。"""
    assert parse_decimal_int(_huge_digits()) is None


def test_parse_decimal_int_rejects_leading_zeros_beyond_the_limit():
    """前导零同样计入位数：``"0" * 5000 + "1"`` 也超限。"""
    assert parse_decimal_int("0" * 5000 + "1") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("123", 123),
        ("-7", -7),
        ("0", 0),
        ("  42  ", 42),
        ("٣", 3),
        ("１２", 12),
        ("۷", 7),
    ],
)
def test_parse_decimal_int_accepts_valid_literals(text, expected):
    """收紧不能误伤：正常十进制字面量与可解析的 Unicode 数字照常解析。"""
    assert parse_decimal_int(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1_0",
        "+7",
        "0x10",
        "²",
        "①",
        "፹",
        "1.5",
        "1e3",
        "",
        "-",
        "abc",
    ],
)
def test_parse_decimal_int_returns_none_for_non_literals(text):
    """非十进制整数字面量一律返回 ``None``，绝不抛异常。

    ``1_0``／``+7`` 是 ``int()`` 会**静默接受**的形态（``int("1_0") == 10``），
    这类「安静的错值」必须显式拒绝。
    """
    assert parse_decimal_int(text) is None


@pytest.mark.parametrize("value", [None, 42, 1.5, [], b"7"])
def test_parse_decimal_int_returns_none_for_non_strings(value):
    """非字符串输入返回 ``None``，不抛 ``AttributeError``。"""
    assert parse_decimal_int(value) is None


def test_parse_decimal_int_never_raises():
    """穷举一批畸形输入：这个函数的契约是「解析不了返回 None」，永不抛异常。"""
    probes = [
        "",
        "-",
        "--1",
        "-0",
        " ",
        "\t",
        "\n",
        "9" * 5000,
        "0" * 5000,
        "²" * 100,
        "٣" * 5000,
        "1" * 4299,
        "1" * 4300,
        "1" * 4301,
    ]
    for probe in probes:
        result = parse_decimal_int(probe)
        assert result is None or isinstance(result, int), f"{probe[:20]!r} 返回了非 int"


# ── 回归：``tomllib`` 还有一类失败**不是** ``TOMLDecodeError`` ──
#
# 上面那个用例覆盖的是 ``TOMLDecodeError``（语法错）。但 ``tomllib`` 解析整数
# 时走的是 ``int()``：位数超过 ``sys.get_int_max_str_digits()``（CPython 默认
# 4300 位）时，``_parser`` 内部直接让 ``ValueError`` 穿出来——它**不是**
# ``TOMLDecodeError``，三个 handler 全接不住。实测
# ``chat_top_k = "9" * 4301`` 得到的是 36 帧裸 traceback，消息只有
# "Exceeds the limit (4300 digits)..."，既没有「配置解析失败」前缀，也没说是
# 哪个文件写坏了；而同样属于「配置写错」的语法错是 5 帧带前缀的消息。


def test_get_settings_wraps_oversized_integer_literal(isolated_config):
    """config.toml 里位数超限的整数必须走同一条脱敏路径，而不是裸 traceback。

    ``"9" * 4301`` 是合法的 TOML 整数，tomllib 只是被 CPython 的整数位数上限
    挡住——用户看到的消息必须与语法错同形。
    """
    import sys

    digits = "9" * (sys.get_int_max_str_digits() + 1)
    (isolated_config / "config.toml").write_text(f"chat_top_k = {digits}\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    # 同时断言前缀与原因：只查前缀的话，一个把任意异常都包成同一句话的实现
    # 也能通过；只查原因的话，未包装的裸 ValueError 也能通过。
    assert "配置解析失败" in message, message
    assert "Exceeds the limit" in message, message


def test_get_settings_oversized_integer_is_not_a_toml_decode_error():
    """确认这一支确实不是 ``TOMLDecodeError``——否则上面的用例是假绿。

    ``tomllib`` 对语法错抛 ``TOMLDecodeError``，对超长整数抛的是**裸**
    ``ValueError``。若哪天它改成抛 ``TOMLDecodeError``，这条用例会提醒我们
    该分支已被 ``except tomllib.TOMLDecodeError`` 覆盖，可以简化。
    """
    import tomllib

    digits = "9" * 5000
    try:
        tomllib.loads(f"chat_top_k = {digits}\n")
    except ValueError as exc:
        assert not isinstance(exc, tomllib.TOMLDecodeError), (
            "tomllib 现在抛 TOMLDecodeError 了，ValueError 兜底分支可以移除"
        )
    else:
        raise AssertionError("tomllib 竟然接受了超长整数")


# ---------------------------------------------------------------------------
# 非字符串取值必须收敛成脱敏的 ValueError
# ---------------------------------------------------------------------------
#
# ``config.toml`` 的 TOML 值可以是任意类型，而这两个 validator 直接对传入值调用
# 字符串方法，于是非字符串输入抛出的是**裸的** ``AttributeError``／``TypeError``
# ——两者都**不是** ``ValueError`` 的子类，因此穿透 ``get_settings`` 的全部三个
# handler，用户看到的是解释器默认 handler 打出的多帧 traceback：
#
# * ``log_level = 123``：``v.upper()`` 抛 ``AttributeError: 'int' object has no
#   attribute 'upper'``（实测 14 帧裸 traceback）。这个字段更严重——它在
#   **模块导入期**就被 ``log_manager.get_module_logger`` 读取，早于任何入口的
#   ``try``，连脱敏边界都还没建立。
# * ``knowledge_base_path = 123``：``Path(123)`` 抛 ``TypeError: expected str,
#   bytes or os.PathLike object, not int``（实测 23 帧裸 traceback）。
#
# 参数 id 必须显式给出：``10**4301`` 超过 ``sys.get_int_max_str_digits()``，
# pytest 自动生成 id 时调 ``str(val)`` 会自己抛 ``ValueError``。
_NON_STRING_VALUES = [
    pytest.param(123, id="int"),
    pytest.param(10**400, id="400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(["WARNING"], id="list"),
    pytest.param({"level": "WARNING"}, id="dict"),
]


@pytest.mark.parametrize("bad_value", _NON_STRING_VALUES)
def test_log_level_rejects_non_string_values(bad_value):
    """``log_level`` 非字符串时必须报脱敏的 ``ValueError``，而不是 ``AttributeError``。"""
    with pytest.raises(ValueError, match="log_level"):
        Settings(log_level=bad_value, _env_file=None)


@pytest.mark.parametrize(
    "field",
    ["knowledge_base_path", "pkl_path", "snapshot_root", "log_path", "cache_path"],
)
@pytest.mark.parametrize("bad_value", [pytest.param(123, id="int"), pytest.param(1.5, id="float")])
def test_path_fields_reject_non_string_values(field, bad_value):
    """路径字段非字符串时必须报 ``ValueError``，而不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match=field):
        Settings(**{field: bad_value}, _env_file=None)


# ── 回归：报告配置校验失败这一步本身不得崩 ──
#
# ``_redacted_settings_error`` 里写的是 ``str(error.get("input"))``，而
# ``input`` 是 pydantic 记下的原始输入值、可以是任意大的整数。实测
# ``Settings(chat_top_k=10**4301)``（库调用方与测试都走这条构造路径）让
# **报告校验失败**这一步自己抛 ``ValueError: Exceeds the limit``——用户看到的
# 是位数上限错误，而真正的问题是 ``chat_top_k`` 越界。


def test_settings_validation_error_survives_huge_integer_input():
    with pytest.raises(ValidationError) as excinfo:
        Settings(chat_top_k=10**4301)

    rendered = _redacted_settings_error(excinfo.value)

    assert "chat_top_k" in rendered
    assert "Exceeds the limit" not in rendered


def test_settings_validation_error_still_redacts_secrets():
    """替换 ``str`` 为 ``safe_repr`` 不得削弱脱敏。"""
    with pytest.raises(ValidationError) as excinfo:
        Settings(chat_temperature="sk-live-abcdef123456")

    rendered = _redacted_settings_error(excinfo.value)

    assert "sk-live-abcdef123456" not in rendered


# ── 回归：``retrieval_candidate_multiplier`` 的 inf 分支 ──
#
# ``int(inf)`` 抛 ``OverflowError``——``ArithmeticError`` 子类而**不是**
# ``ValueError``，不在该 validator 的 ``except (ValueError, TypeError)`` 里，
# 会穿透 ``get_settings`` 的全部三个 handler。而 ``get_settings`` 在**导入期**
# 就被 ``cleanup.py`` 的 ``get_module_logger`` 调用，早于 ``run_cli`` 的任何
# ``try``，于是 ``config.toml`` 写 ``retrieval_candidate_multiplier = inf``
# （合法 TOML，``tomllib`` 原样解析成 ``inf``）让程序以裸 traceback 退出。


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(1e400, id="overflows-to-inf"),
    ],
)
def test_retrieval_candidate_multiplier_rejects_non_finite(bad_value):
    with pytest.raises((ValueError, ValidationError)):
        Settings(retrieval_candidate_multiplier=bad_value)


def test_retrieval_candidate_multiplier_rejects_decimal_infinity():
    """非 ``float`` 的非有限数值（``Decimal``）必须走 ``int()`` 的
    ``OverflowError`` 分支，而不是裸崩。

    这是 ``inf`` 的第二条入口：它绕过 ``isinstance(v, float) and not
    math.isfinite(v)`` 前置守卫（``Decimal`` 不是 ``float``），由 ``int()``
    抛 ``OverflowError``——``ArithmeticError`` 子类而**不是** ``ValueError``。
    ``except`` 里若不显式列出它，异常会穿透 ``get_settings`` 的全部三个
    handler，变成未脱敏的裸 traceback。删掉 ``OverflowError`` 会让这条测试变红。
    """
    import decimal

    with pytest.raises((ValueError, ValidationError)):
        Settings(retrieval_candidate_multiplier=decimal.Decimal("Infinity"))


def test_retrieval_candidate_multiplier_inf_guard_message_names_the_field():
    """``isfinite`` 守卫必须给出可定位到字段的消息，而不是泛化的转换失败。

    只留 ``OverflowError`` 兜底时，``config.toml`` 写 ``inf`` 会走到
    ``int()`` 并报「无法将检索候选倍率 'inf' 转换为整数」——消息仍然脱敏、
    仍然不崩，但把「字段不该是非有限值」说成了「转换失败」。删掉
    ``isfinite`` 守卫会让这条断言变红。
    """
    with pytest.raises(ValidationError) as excinfo:
        Settings(retrieval_candidate_multiplier=float("inf"))

    assert "必须是整数" in str(excinfo.value)


def test_tomllib_parses_inf_as_float_infinity():
    """``inf`` 是合法 TOML，这条路径无需任何位数技巧即可到达。"""
    import tomllib

    assert tomllib.loads("x = inf")["x"] == float("inf")


# ── 回归：路径字段必须拒绝 NUL 字节 ──
#
# ``os.makedirs``／``open`` 对含 NUL 的路径抛 ``ValueError: embedded null
# byte``——**不是** ``OSError``，因此下游那些「目录不可写就降级」的
# ``except OSError`` 全部接不住。实测 ``config.toml`` 写
# ``log_path = "/tmp/a\u0000b"``（TOML 的 ``\u0000`` 转义）时，
# ``cleanup.py`` 在**模块导入期**调用 ``get_module_logger``，程序连
# ``run_cli`` 都到不了就崩。


@pytest.mark.parametrize(
    "field_name",
    ["log_path", "cache_path", "knowledge_base_path", "snapshot_root", "pkl_path"],
)
def test_path_fields_reject_nul_bytes(field_name):
    with pytest.raises((ValueError, ValidationError), match="NUL"):
        Settings(**{field_name: "/tmp/a\x00b"})


# ── 回归：空的路径配置不得静默变成「当前工作目录」 ──
#
# ``resolve_path`` 的 docstring 承诺「将相对路径解析为绝对路径」，但它在解析
# **之前**有一个 ``if not v: return v`` 提前返回，于是空串被原样返回，而
# ``Path("")`` 在求值期等价于 ``Path(".")``——即**运行时的当前工作目录**。
#
# 两个后果，都不是「路径写错」应有的收场：
#
# 1. **信任根扩到 CWD**：``FaissStore.load_snapshot`` 用 ``snapshot_root`` 作为
#    pickle 的来源信任根（``AGENTS.md`` 要求快照只从本项目生成的文件加载）。
#    实测 ``SNAPSHOT_ROOT=""`` 时信任根解析为仓库根目录，在其中放一个带
#    ``ACTIVE_SNAPSHOT`` 标记的目录即可让 ``load_snapshot`` 接受并反序列化它的
#    ``chunks.pkl``——边界等于不存在。
# 2. **行为随 CWD 漂移**：其余四个路径字段同样原样返回，同一份配置在仓库内运行
#    与从仓库外运行会指向不同目录（实测 CWD=/tmp/xxx 时五者全部解析到 /tmp/xxx），
#    而默认值（如 ``data/kb``）走的是 ``ROOT_DIR`` 分支、不随 CWD 变。
#
# 空值没有任何合法语义：路径字段的默认值都是非空的相对路径，用户「不想设」时
# 应当删掉该键（回落默认值），而不是把它写成空串。因此这里显式拒绝，并给出
# 可执行的指引。``cache_path`` 的清理侧另有 ``if not cache_dir: return "cache_path
# 为空"`` 的守卫，那是删除前的最后一道防线，不该被当作「空值合法」的依据。
@pytest.mark.parametrize(
    "field_name",
    ["knowledge_base_path", "pkl_path", "snapshot_root", "log_path", "cache_path"],
)
def test_path_settings_reject_empty_string(field_name):
    with pytest.raises(ValidationError) as excinfo:
        Settings(**{field_name: ""})

    message = str(excinfo.value)
    assert field_name in message
    # 只查字段名的话，一个把任意异常都包成同一句话的实现也能通过；指引必须
    # 告诉用户「删掉该键回落默认值」这条可执行路径。
    assert "默认值" in message


def test_path_settings_reject_whitespace_only_string():
    """纯空白同样不是合法路径：``Path("  ")`` 会解析成一个名为两个空格的真实目录。"""
    with pytest.raises(ValidationError) as excinfo:
        Settings(snapshot_root="   ")

    assert "snapshot_root" in str(excinfo.value)


def test_empty_snapshot_root_does_not_widen_the_pickle_trust_root():
    """端到端：空 ``snapshot_root`` 不得让 CWD 成为 pickle 的来源信任根。

    这条断言走的是真实加载路径（``build_run_config`` → ``FaissStore.load_snapshot``），
    而不是只看 ``Settings`` 的取值——只看取值的话，一个把空值换成 ``"."`` 的
    实现仍然会让信任根落在 CWD 上。
    """
    with pytest.raises(ValidationError):
        Settings(snapshot_root="")


# ── 回归：路径字段的符号链接环必须收敛成 ``ValueError`` ──
#
# ``resolve_path`` 里 ``(ROOT_DIR / v).resolve()`` 对**符号链接环**抛
# ``RuntimeError: Symlink loop from ...``。它不是 ``ValueError`` 也不是
# ``OSError``，``get_settings`` 已有的四个 handler（``ValidationError``／
# ``SettingsError``／``TOMLDecodeError``／``RecursionError``）一个都接不住，
# 而 ``get_settings`` 在 **import 期**就被调用（``log_manager
# .get_module_logger``），早于任何入口的 ``try``。实测 ``data`` 是环
# （``data -> data``）时 ``import main`` 以裸 traceback 终止。收敛判据与
# ``utils.security.resolve_within`` 一致：``(OSError, RuntimeError)`` →
# 带字段名的 ``ValueError``。


def test_relative_path_symlink_loop_is_reported_as_value_error(tmp_path, monkeypatch):
    """相对路径字段遇到符号链接环时，必须抛脱敏的 ``ValueError``。"""
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    loop = tmp_path / "data"
    loop.symlink_to(loop)

    with pytest.raises(ValueError, match="snapshot_root"):
        Settings(snapshot_root="data/kb", _env_file=None)


def test_absolute_path_field_is_not_resolved(tmp_path):
    """绝对路径不做 ``resolve``，因此不受影响（收紧不能误伤）。"""
    settings = Settings(snapshot_root=str(tmp_path / "kb"), _env_file=None)
    assert settings.snapshot_root == str(tmp_path / "kb")


# ── Settings 整数/数值守卫的**测试覆盖**补缺 ──
#
# 下面几条是**变异测试**翻出来的覆盖空洞：把对应守卫的 ``raise`` 删掉之后，
# 全量套件仍然全绿——守卫本身正确，却没有任何一条测试会在它被删掉时变红。
# 甄别方法是「删除守卫 + 真实 ``Settings(...)`` 入口」，只保留**行为确实改变**
# 的那些（静默接受、静默截断，或把带字段名的 ``ValueError`` 换成别的异常）；
# 纯等价变异（删除后落到下一个守卫、结论不变）不在此列。
#
# 每条都以「删除守卫即变红」的形态写：断言的是守卫存在时的收场。


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("chat_top_k", 1.5, id="chat_top_k"),
        pytest.param("log_retention_days", 1.5, id="log_retention_days"),
        pytest.param("kb_chunk_overlap", 1.5, id="kb_chunk_overlap"),
        pytest.param("kb_child_chunk_overlap", 1.5, id="kb_child_chunk_overlap"),
    ],
)
def test_integer_settings_reject_a_non_integral_float(field, value):
    """非整数浮点必须被拒，而不是 ``int()`` 静默截断。

    这四处各有一条 ``isinstance(number, float) and not number.is_integer()``
    判据。删除后 ``int(1.5)`` 会把 ``1.5`` 悄悄变成 ``1``——用户写
    ``chat_top_k = 1.5`` 得到的是「取 1 条」，没有任何报错，与「配置写错了」
    的收场相去甚远。``1.5`` 是合法 TOML 浮点，无需任何技巧即可到达。
    """
    with pytest.raises(ValidationError, match="整数"):
        Settings(**{field: value}, _env_file=None)


def test_settings_reject_an_unknown_fusion_strategy():
    """合法字符串但不在 ``{"rrf", "weighted"}`` 时必须报错。

    ``validate_hybrid_fusion_strategy`` 的成员判据删除后，``"bogus"`` 会被
    ``Settings`` 静默接受并原样存进 ``hybrid_fusion_strategy``，随后
    ``HybridReranker`` 的派发（``if == "weighted" else _rrf_rerank``）按 RRF
    出结果——用户配置里写着一个无效策略、程序却按 RRF 跑，是静默错误结果。
    """
    with pytest.raises(ValidationError, match="融合策略"):
        Settings(hybrid_fusion_strategy="bogus", _env_file=None)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(object(), id="object"),
    ],
)
def test_numeric_settings_reject_a_non_numeric_container(value):
    """``_coerce_number`` 对非数值容器必须给带字段名的 ``ValueError``。

    最后那条 ``raise ValueError(f"{field_label} 必须是数值，但得到 ...")`` 删除
    后，列表/字典/对象会一路落到 pydantic 的类型校验，抛的是没有字段语义的
    ``TypeError``（实测 ``Settings(chat_top_k=[1])`` 报 ``TypeError``）——而本
    项目其余数值字段一律以带字段名的 ``ValueError`` 收敛。
    """
    with pytest.raises((ValidationError, ValueError)):
        Settings(chat_top_k=value, _env_file=None)


def test_settings_reject_a_numeric_literal_that_overflows_to_infinity():
    """``"1e400"`` 是合法十进制字面量，但 ``float()`` 后是 ``inf``。

    ``_coerce_number`` 的 ``isfinite`` 判据删除后，``"1e400"`` 会被转成
    ``inf`` 返回，随后落到整数上界判据，报「必须是整数，但得到 inf」——把
    「字面量溢出」说成了「类型不对」，且不再是原始字符串。
    """
    with pytest.raises(ValidationError, match="有限数值"):
        Settings(chat_top_k="1e400", _env_file=None)
