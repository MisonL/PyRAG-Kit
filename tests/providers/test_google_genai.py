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


def test_google_provider_accepts_gemini_api_key_alias(monkeypatch):
    settings = SimpleNamespace(google_api_key=None, gemini_api_key="gemini-key")
    monkeypatch.setattr(google_module, "get_settings", lambda: settings)
    with patch.object(google_module.genai, "Client") as client_class:
        client_class.return_value = MagicMock()
        provider = GoogleProvider(model_name="gemini-1.5-flash")
        provider._get_client()

    client_class.assert_called_once_with(api_key="gemini-key")


def test_google_api_key_takes_precedence_over_gemini_alias(monkeypatch):
    settings = SimpleNamespace(
        google_api_key="google-key", gemini_api_key="gemini-key"
    )
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
    with patch.object(google_module.genai, "Client"), pytest.raises(
        ValueError, match="Vertex/Enterprise"
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
        provider = GoogleProvider(
            model_name="gemini-1.5-flash", options={"vertexai": True}
        )

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

    with patch.object(google_module.genai, "Client") as client_class, pytest.raises(
        ValueError, match="GOOGLE_API_KEY|Vertex/Enterprise"
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

    with patch.object(provider, "_get_client", return_value=client), pytest.raises(
        RuntimeError, match="安全策略|拒答"
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

    with patch.object(provider, "_get_client", return_value=client), pytest.raises(
        RuntimeError, match="安全策略|拒答"
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
    with patch.object(GoogleProvider, '_get_client', return_value=mock_genai_client):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_genai_client.models.generate_content.return_value = mock_response
        
        result = list(provider.invoke("test prompt", stream=False))
        
        assert result == ["Hello world"]
        mock_genai_client.models.generate_content.assert_called_once()

def test_google_provider_embed_documents(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="embedding-001")
    
    # 直接模拟 _get_client
    with patch.object(GoogleProvider, '_get_client', return_value=mock_genai_client):
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
        patch.object(provider, "_get_client", side_effect=AssertionError("client must not initialize")),
        pytest.raises(ValueError, match="凭证"),
    ):
        provider.embed_documents(
            ["text"],
            extra_headers={"Authorization": "Bearer secret-token"},
        )

    async def consume() -> None:
        with (
            patch.object(provider, "_get_client", side_effect=AssertionError("client must not initialize")),
            pytest.raises(ValueError, match="凭证"),
        ):
            await provider.aembed_documents(
                ["text"],
                extra_headers={"Authorization": "Bearer secret-token"},
            )

    asyncio.run(consume())
