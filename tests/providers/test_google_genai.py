import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.providers.google as google_module
from src.providers.google import GoogleProvider


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """强制模拟 settings 并在环境中设置 key"""
    mock_settings_instance = MagicMock()
    mock_settings_instance.google_api_key = "fake_key"
    monkeypatch.setattr(google_module, "get_settings", lambda: mock_settings_instance)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    return mock_settings_instance


@pytest.fixture
def mock_genai_client():
    with patch.object(google_module.genai, "Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


def test_google_provider_init(mock_settings):
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    assert provider._model_name == "gemini-1.5-flash"
    assert provider._client is None


def test_google_provider_get_client(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    client = provider._get_client()
    assert client == mock_genai_client
    # Verify Client initialization
    from src.providers.google import genai

    genai.Client.assert_called_once_with(api_key="fake_key")


def test_google_provider_uses_credentials_without_api_key(monkeypatch):
    credentials = object()
    settings = MagicMock()
    settings.google_api_key = "should-not-be-forwarded"
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(
            model_name="gemini-1.5-flash",
            options={"credentials": credentials, "vertexai": True},
        )

        provider._get_client()

    client_class.assert_called_once_with(credentials=credentials, vertexai=True)


def test_google_provider_loads_credentials_file_and_forwards_it(monkeypatch):
    """设置里的凭证文件路径加载出的凭证必须传给 SDK（``"credentials" not in client_options``）。

    ``in`` 变异后仅在 ``client_options`` 已有 credentials 时才赋值——文件加载
    出的凭证被静默丢弃，SDK 落回 ADC/API key 路径，而配置看起来完全正常。
    """
    credentials = object()
    settings = MagicMock()
    settings.google_api_key = None
    settings.google_application_credentials = "/tmp/credentials.json"
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        GoogleProvider, "_load_credentials_file", staticmethod(lambda path: credentials)
    )
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(
            model_name="gemini-1.5-flash",
            options={"vertexai": True},
        )

        provider._get_client()

    client_class.assert_called_once_with(credentials=credentials, vertexai=True)


def test_google_provider_accepts_gemini_api_key_alias(monkeypatch):
    settings = SimpleNamespace(google_api_key=None, gemini_api_key="gemini-key")
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(model_name="gemini-1.5-flash")
        provider._get_client()

    client_class.assert_called_once_with(api_key="gemini-key")


def test_google_api_key_takes_precedence_over_gemini_alias(monkeypatch):
    settings = SimpleNamespace(google_api_key="google-key", gemini_api_key="gemini-key")
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(model_name="gemini-1.5-flash")
        provider._get_client()

    client_class.assert_called_once_with(api_key="google-key")


def test_google_provider_accepts_credentials_without_google_api_key(monkeypatch):
    credentials = object()
    settings = MagicMock()
    settings.google_api_key = None
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(
            model_name="gemini-1.5-flash", options={"credentials": credentials}
        )

        provider._get_client()

    client_class.assert_called_once_with(credentials=credentials, vertexai=True)


def test_google_credentials_reject_explicit_developer_api_mode(monkeypatch):
    credentials = object()
    settings = MagicMock()
    settings.google_api_key = None
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    provider = GoogleProvider(
        model_name="gemini-1.5-flash",
        options={"credentials": credentials, "vertexai": False},
    )
    with (
        patch.object(google_module.genai, "Client"),
        pytest.raises(ValueError, match="Vertex/Enterprise"),
    ):
        provider._get_client()


def test_google_vertex_uses_adc_without_api_key(monkeypatch):
    settings = MagicMock()
    settings.google_api_key = None
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(model_name="gemini-1.5-flash", options={"vertexai": True})

        provider._get_client()

    client_class.assert_called_once_with(vertexai=True)


def test_google_vertex_dotenv_settings_are_forwarded_to_sdk(monkeypatch):
    settings = SimpleNamespace(
        google_api_key=None,
        google_genai_use_vertexai=True,
        google_genai_use_enterprise=None,
        google_cloud_project="dotenv-project",
        google_cloud_location="asia-east1",
        google_application_credentials=None,
    )
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(model_name="gemini-1.5-flash")
        provider._get_client()

    client_class.assert_called_once_with(
        vertexai=True,
        project="dotenv-project",
        location="asia-east1",
    )


def test_google_get_client_requires_explicit_auth_without_key_or_vertex_mode(monkeypatch):
    settings = MagicMock()
    settings.google_api_key = None
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GoogleProvider(model_name="gemini-1.5-flash", options={})

    with (
        patch.object(google_module.genai, "Client") as client_class,
        pytest.raises(ValueError, match="GOOGLE_API_KEY|Vertex/Enterprise"),
    ):
        provider._get_client()

    client_class.assert_not_called()


@pytest.mark.parametrize(
    "http_options",
    [
        {"headers": {"Authorization": "Bearer secret"}},
        {"headers": {"X-API-Key": "secret"}},
        {"client_args": {"headers": {"X-Auth-Token": "secret"}}},
        {"client_args": {"params": {"access_token": "secret"}}},
        {"async_client_args": {"timeout": 1000}},
        {"async_client_args": {"params": {"trace_id": "trace"}}},
        {"base_url": "https://proxy.invalid/v1"},
        {"base_url": "https://proxy.invalid/v1?api_key=secret"},
        {"base_url": "https://user:password@proxy.invalid/v1"},
        {"httpx_client": object()},
    ],
)
def test_google_http_options_reject_credentials_or_custom_clients(http_options):
    with pytest.raises(ValueError, match="凭证|自定义客户端|连接覆盖"):
        GoogleProvider(
            model_name="gemini-1.5-flash",
            options={"http_options": http_options},
        )


def test_google_http_options_allow_safe_headers_and_sdk_object():
    provider = GoogleProvider(
        model_name="gemini-1.5-flash",
        options={"http_options": {"headers": {"X-Trace-ID": "trace"}}},
    )
    provider._validate_options(provider._options)
    sdk_options = google_module.types.HttpOptions(headers={"X-Trace-ID": "trace"})
    provider = GoogleProvider(
        model_name="gemini-1.5-flash",
        options={"http_options": sdk_options},
    )
    provider._validate_options(provider._options)


def test_google_http_options_scan_header_values_and_avoid_substring_false_positives():
    with pytest.raises(ValueError, match="凭证"):
        GoogleProvider(
            model_name="gemini-1.5-flash",
            options={"http_options": {"headers": {"X-Trace": "Bearer secret-token"}}},
        )

    provider = GoogleProvider(
        model_name="gemini-1.5-flash",
        options={
            "http_options": {
                "headers": {
                    "X-Authorization-Count": "1",
                    "X-Credential-Provider": "internal",
                }
            }
        },
    )
    provider._validate_options(provider._options)


def test_google_http_options_reject_connection_override_containers():
    for key in ("client_args", "async_client_args"):
        with pytest.raises(ValueError, match="连接覆盖"):
            GoogleProvider(
                model_name="gemini-1.5-flash",
                options={"http_options": {key: {"timeout": 1000}}},
            )


class _ModelDumpRaisesTypeError:
    """``model_dump(exclude_none=True)`` 抛 ``TypeError`` 的替身。

    真实 ``google-genai`` 的 HttpOptions 接受 ``exclude_none``；这里模拟的是
    SDK 签名变化或第三方对象只实现无参 ``model_dump`` 的情形。
    """

    def model_dump(self, *args, **kwargs):
        raise TypeError("model_dump() got an unexpected keyword argument 'exclude_none'")


class _ModelDumpRaisesOtherError:
    def model_dump(self, *args, **kwargs):
        raise KeyError("boom")


class _ModelDumpReturnsScalar:
    def model_dump(self, *args, **kwargs):
        return 5


class _ModelDumpTakesNoArguments:
    """无参 ``model_dump``：第一次带参调用抛 ``TypeError``，第二次成功。"""

    def model_dump(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("takes 1 positional argument but 2 were given")
        return {"headers": {"X-Trace": "ok"}}


# ``_validate_http_options`` 有两条输入路径：Mapping（直接 ``dict()``）与
# 对象（走 ``model_dump``）。两条路径上的形态判据在整套测试里从未执行——
# 既有用例只喂合法 Mapping 或真实 ``types.HttpOptions``。把它们的 ``raise``
# 换成 ``pass`` 后全量 3353 条仍全绿（只有 585 那条被
# ``test_google_http_options_reject_nested_non_string_key`` 钉住）。
@pytest.mark.parametrize(
    "value",
    [5, "abc", [1, 2], object()],
    ids=["int", "str", "list", "object"],
)
def test_google_http_options_reject_non_mapping_without_model_dump(value):
    """非 Mapping 且无 ``model_dump`` 必须给出可读消息。

    删除守卫后取值落到 ``model_dump`` 分支，得到「无法读取」——把一个**类型
    错误**说成**读取失败**，照提示排查的人会去查权限而不是配置形态。
    """
    with pytest.raises(ValueError, match="必须是对象表或 google-genai HttpOptions"):
        GoogleProvider._validate_http_options(value)


def test_google_http_options_reports_model_dump_failure_readably():
    """``model_dump`` 抛异常时必须收敛成 ``ValueError``，不能漏出 ``UnboundLocalError``。

    删除两条 ``except`` 里的守卫后，``payload`` 从未赋值，异常从
    「无法读取」退化成裸 ``UnboundLocalError: cannot access local variable
    'payload'``——调用方按 ``ValueError`` 分类，接不住它。
    """
    with pytest.raises(ValueError, match="无法读取"):
        GoogleProvider._validate_http_options(_ModelDumpRaisesOtherError())


def test_google_http_options_falls_back_to_no_argument_model_dump():
    """带参 ``model_dump`` 抛 ``TypeError`` 时回退到无参调用，不能吞掉合法对象。

    回退成功仍走完整校验，因此这里是**反向守卫**：合法对象照常通过。
    """
    GoogleProvider._validate_http_options(_ModelDumpTakesNoArguments())


def test_google_http_options_reports_dump_failure_after_fallback():
    """回退（无参）也失败时必须给「无法读取」，不能漏出 ``UnboundLocalError``。"""
    with pytest.raises(ValueError, match="无法读取"):
        GoogleProvider._validate_http_options(_ModelDumpRaisesTypeError())


def test_google_http_options_reject_model_dump_returning_non_mapping():
    """``model_dump`` 返回非 Mapping 必须显式报错，而不是裸 ``TypeError``。

    删除守卫后 ``"timeout" in payload`` 对整数抛裸
    ``TypeError: argument of type 'int' is not iterable``。
    """
    with pytest.raises(ValueError, match="http_options 必须是对象"):
        GoogleProvider._validate_http_options(_ModelDumpReturnsScalar())


def test_google_invoke_non_stream_refusal_is_explicit_error():
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=SimpleNamespace(parts=[]),
            )
        ]
    )
    client = MagicMock()
    client.models.generate_content.return_value = response

    with (
        patch.object(provider, "_get_client", return_value=client),
        pytest.raises(RuntimeError, match="安全策略|拒答"),
    ):
        list(provider.invoke("hello", stream=False))


def test_google_invoke_stream_refusal_is_explicit_error():
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=SimpleNamespace(parts=[]),
            )
        ]
    )
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter([chunk])

    with (
        patch.object(provider, "_get_client", return_value=client),
        pytest.raises(RuntimeError, match="安全策略|拒答"),
    ):
        list(provider.invoke("hello", stream=True))


def test_google_stream_events_emits_finish_for_top_level_finish_reason():
    events = GoogleProvider._stream_events_from_chunk({"finish_reason": "STOP"})

    finish_events = [event for event in events if event.type == "finish"]
    assert len(finish_events) == 1
    assert finish_events[0].finish_reason == "STOP"


def test_google_finish_reason_enum_is_normalized_to_wire_value():
    from google.genai import types

    candidate = SimpleNamespace(
        finish_reason=types.FinishReason.STOP,
        content=SimpleNamespace(parts=[]),
    )
    response = SimpleNamespace(candidates=[candidate])

    result = GoogleProvider._extract_result(response)
    events = GoogleProvider._stream_events_from_chunk(response)

    assert result.finish_reason == "STOP"
    assert [event.finish_reason for event in events if event.type == "finish"] == ["STOP"]


def test_google_ainvoke_non_stream_refusal_is_explicit_error():
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=SimpleNamespace(parts=[]),
            )
        ]
    )

    class Models:
        async def generate_content(self, **_kwargs):
            return response

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))

    async def consume():
        with patch.object(provider, "_get_client", return_value=client):
            return [chunk async for chunk in provider.ainvoke("hello", stream=False)]

    with pytest.raises(RuntimeError, match="安全策略|拒答"):
        asyncio.run(consume())


def test_google_ainvoke_stream_refusal_is_explicit_error():
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=SimpleNamespace(parts=[]),
            )
        ]
    )

    class AsyncChunks:
        def __aiter__(self):
            async def chunks():
                yield chunk

            return chunks()

    class Models:
        def generate_content_stream(self, **_kwargs):
            return AsyncChunks()

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))

    async def consume():
        with patch.object(provider, "_get_client", return_value=client):
            return [chunk async for chunk in provider.ainvoke("hello", stream=True)]

    with pytest.raises(RuntimeError, match="安全策略|拒答"):
        asyncio.run(consume())


def test_google_debug_config_mapping_is_converted_to_sdk_type(monkeypatch):
    settings = MagicMock()
    settings.google_api_key = "fake_key"
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(
            model_name="gemini-1.5-flash",
            options={"debug_config": {"client_mode": "replay", "replay_id": "r1"}},
        )
        provider._get_client()

    debug_config = client_class.call_args.kwargs["debug_config"]
    assert debug_config.client_mode == "replay"
    assert debug_config.replay_id == "r1"


def test_google_unknown_option_fails_at_provider_boundary(monkeypatch):
    settings = MagicMock()
    settings.google_api_key = "fake_key"
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="不支持的字段.*typo_option"):
        GoogleProvider(model_name="gemini-1.5-flash", options={"typo_option": True})


@pytest.mark.parametrize("value", ["false", "0", "off", "no", ""])
def test_google_vertex_mode_parses_false_string_values(value):
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._options = {"vertexai": value}

    assert provider._configured_vertex_mode() is False


def test_google_endpoint_modes_reject_conflicting_explicit_flags():
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._options = {"enterprise": True, "vertexai": False}

    with pytest.raises(ValueError, match="enterprise.*vertexai.*冲突"):
        provider._configured_vertex_mode()


def test_google_endpoint_modes_reject_conflicting_environment_flags(monkeypatch):
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._options = {}
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")

    with pytest.raises(ValueError, match="GOOGLE_GENAI_USE_ENTERPRISE.*冲突"):
        provider._configured_vertex_mode()


def test_google_get_client_reuses_falsey_client(monkeypatch):
    settings = MagicMock()
    settings.google_api_key = "fake_key"
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)

    class FalseyClient:
        def __bool__(self):
            return False

    client = FalseyClient()
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {}
    provider._client = client

    with patch.object(google_module.genai, "Client") as client_class:
        assert provider._get_client() is client
        client_class.assert_not_called()


def test_google_provider_invoke_non_stream(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="gemini-1.5-flash")

    # 直接模拟 _get_client 以规避 tenacity 装饰器可能带来的环境隔离问题
    with patch.object(GoogleProvider, "_get_client", return_value=mock_genai_client):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        # ``candidates`` 必须显式给成列表：真 SDK 的字段类型是
        # ``list[Candidate] | None``，而裸 ``MagicMock`` 的属性是一个可迭代的
        # ``MagicMock``（``iter()`` 返回空迭代器）——旧代码只做 ``or []`` 与迭代，
        # 这种不真实形态恰好能通过。响应容器的形态守卫（google.py 的
        # ``_require_candidates``）会把它拒掉，因此夹具要写成真实形状。
        mock_response.candidates = []
        mock_genai_client.models.generate_content.return_value = mock_response

        result = list(provider.invoke("test prompt", stream=False))

        assert result == ["Hello world"]
        mock_genai_client.models.generate_content.assert_called_once()


def test_google_provider_embed_documents(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="embedding-001")

    # 直接模拟 _get_client
    with patch.object(GoogleProvider, "_get_client", return_value=mock_genai_client):
        mock_emb_1 = MagicMock()
        mock_emb_1.values = [0.1, 0.2]
        mock_emb_2 = MagicMock()
        mock_emb_2.values = [0.3, 0.4]

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb_1, mock_emb_2]
        mock_genai_client.models.embed_content.return_value = mock_response

        texts = ["text1", "text2"]
        result = provider.embed_documents(texts)

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_genai_client.models.embed_content.assert_called_once()


def test_google_embedding_rejects_request_credentials_before_client_init(mock_settings):
    provider = GoogleProvider(model_name="embedding-001")

    with (
        patch.object(
            provider, "_get_client", side_effect=AssertionError("client must not initialize")
        ),
        pytest.raises(ValueError, match="凭证"),
    ):
        provider.embed_documents(
            ["text"],
            extra_headers={"Authorization": "Bearer secret-token"},
        )

    async def consume() -> None:
        with (
            patch.object(
                provider, "_get_client", side_effect=AssertionError("client must not initialize")
            ),
            pytest.raises(ValueError, match="凭证"),
        ):
            await provider.aembed_documents(
                ["text"],
                extra_headers={"Authorization": "Bearer secret-token"},
            )

    asyncio.run(consume())


# ── 回归：partial_args 的 json_path 索引位数超限时不得崩 ──
#
# ``_partial_arg_tokens`` 用 ``re.match(r"^\[(\d+)\]", rest)`` 取索引，再
# ``int(indexed.group(1))``。``\d+`` 对任意长度的数字串都匹配，而
# ``int("9" * 4301)`` 撞上 CPython 的 ``sys.get_int_max_str_digits()``
# （默认 4300）抛 ``ValueError: Exceeds the limit``——异常类型是本函数契约里的
# ``ValueError``，但**内容**是整数转换问题，与「json_path 不合法」无关，同函数
# 另外两条拒绝分支都带着原始 ``json_path`` 与指引。
#
# json_path 来自模型的流式 function call 分片（``partial_args``），形态不受控；
# 同一份响应里普通路径（``$.a``／``$[1]``）都能得到可读错误，只有这一支例外。


@pytest.mark.parametrize(
    "json_path",
    ["$[" + "9" * 4301 + "]", "$[" + "9" * 5000 + "]"],
    ids=["4301digits", "5000digits"],
)
def test_google_partial_args_rejects_oversized_index_readably(json_path):
    """超长索引必须给出带 json_path 的 ``ValueError``，不是整数转换超限。"""
    with pytest.raises(ValueError) as excinfo:
        GoogleProvider._partial_arg_tokens(json_path)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "json_path" in message


def test_google_partial_args_still_accepts_normal_paths():
    """收紧不能误伤：普通 JSONPath 子集照常解析。"""
    assert GoogleProvider._partial_arg_tokens("$.a[0].b") == ["a", 0, "b"]
    assert GoogleProvider._partial_arg_tokens("$['k'][2]") == ["k", 2]
    assert GoogleProvider._partial_arg_tokens("$") == []


# ── 回归：Google 覆写的 embed_query 必须与基类保持同一兜底 ──
#
# ``TextEmbeddingModel`` 的默认 ``embed_query`` 先判 ``if not vectors`` 再取
# ``[0]``，空结果得到的是带指引的 ``ValueError("Embedding provider 返回空结果。")``。
# ``GoogleProvider`` 为了设 ``task_type`` 覆写了这两个方法，却把那条兜底丢了，
# 直接下标取 ``[0]``——provider 返回空列表时抛裸
# ``IndexError: list index out of range``。
#
# 空返回不是假想输入：``values`` 数量校验是按 ``len(texts)`` 做的，传空 ``texts``
# 时 ``len(values) == len(texts) == 0`` 成立，方法正常返回 ``[]``。网关在截断
# 响应、或返回空 ``embeddings`` 字段时也会走到同一形态。
#
# ``IndexError`` 不是 ``ValueError``／``RuntimeError``，会穿透调用方按这两类做的
# 错误分类与日志脱敏，且消息与「provider 返回了什么」毫无关系。


class _EmptyEmbeddingGoogle(GoogleProvider):
    """绕过网络：让底层 ``embed_documents`` 返回空结果。"""

    def __init__(self):
        self._model_name = "m"

    def embed_documents(self, texts, **kwargs):
        self.seen_kwargs = kwargs
        return []

    async def aembed_documents(self, texts, **kwargs):
        self.seen_kwargs = kwargs
        return []


def test_google_embed_query_reports_empty_result_as_value_error():
    """空结果必须是 ``ValueError``，不是裸 ``IndexError``。"""
    provider = _EmptyEmbeddingGoogle()

    with pytest.raises(ValueError, match="空结果"):
        provider.embed_query("查询文本")


def test_google_aembed_query_reports_empty_result_as_value_error():
    """异步路径同判据。"""
    provider = _EmptyEmbeddingGoogle()

    with pytest.raises(ValueError, match="空结果"):
        asyncio.run(provider.aembed_query("查询文本"))


def test_google_embed_query_still_sets_retrieval_query_task_type():
    """收紧不能误伤：覆写存在的理由（``task_type``）必须保留。"""
    provider = _EmptyEmbeddingGoogle()

    with pytest.raises(ValueError):
        provider.embed_query("查询文本")

    assert provider.seen_kwargs["task_type"] == "RETRIEVAL_QUERY"


def test_google_aembed_query_still_sets_retrieval_query_task_type():
    provider = _EmptyEmbeddingGoogle()

    with pytest.raises(ValueError):
        asyncio.run(provider.aembed_query("查询文本"))

    assert provider.seen_kwargs["task_type"] == "RETRIEVAL_QUERY"


def test_google_embed_query_returns_the_first_vector():
    """正常路径不受影响。"""

    class _One(_EmptyEmbeddingGoogle):
        def embed_documents(self, texts, **kwargs):
            self.seen_kwargs = kwargs
            return [[0.5, 1.5]]

        async def aembed_documents(self, texts, **kwargs):
            self.seen_kwargs = kwargs
            return [[0.5, 1.5]]

    assert _One().embed_query("查询文本") == [0.5, 1.5]


# ── 回归：json_path 的拒绝消息自身用 ``!r`` 插值 ──
#
# ``_partial_arg_tokens`` 的两条拒绝分支写 ``f"...: {json_path!r}"``。判据是
# ``isinstance(json_path, str)``，因此**非字符串**取值会走到第一条分支——而
# ``repr(10**4301)`` 自己抛 ``ValueError: Exceeds the limit``：正要报出
# 「json_path 无效」的那一步变成整数转换错误。上一条回归只覆盖了「字符串形态、
# 索引位数超限」，这条覆盖「取值本身是位数超限的整数」。
#
# 可达路径：``_partial_args_to_mapping`` 读模型流式分片的 ``json_path`` 字段，
# 形态不受控。


@pytest.mark.parametrize(
    "json_path",
    [pytest.param(10**4301, id="huge-int"), pytest.param(["$"], id="list")],
)
def test_google_partial_args_rejects_malformed_json_path_readably(json_path):
    """非字符串 ``json_path`` 不得让「报告无效路径」自己抛裸异常。"""
    with pytest.raises(ValueError) as excinfo:
        GoogleProvider._partial_arg_tokens(json_path)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "json_path" in message


def test_google_partial_args_mapping_surfaces_malformed_json_path_readably():
    """从公开入口进入时同样给出可读消息。"""
    with pytest.raises(ValueError) as excinfo:
        GoogleProvider._partial_args_to_mapping([{"json_path": 10**4301, "number_value": 1}])

    assert "Exceeds the limit" not in str(excinfo.value)
