import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
from google.auth.credentials import AnonymousCredentials
from google.genai import types

from src.providers.__base__.model_provider import (
    CompletionRequest,
    LargeLanguageModel,
    TextEmbeddingModel,
    normalize_chat_messages,
    normalize_messages,
    normalize_responses_input,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.factory import ModelProviderFactory
from src.providers.google import GoogleProvider
from src.providers.openai import OpenAIProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.resources import AsyncGoogleResources, GoogleResources
from src.providers.volcengine import VolcengineProvider
from src.utils.config import ModelDetail
from src.utils.security import (
    find_sensitive_option_paths,
    is_sensitive_option_key,
    redact_sensitive_text,
    validate_secret_free_options,
    validate_secret_free_payload,
    validate_secret_free_resource_args,
    validate_secret_free_resource_kwargs,
)


def test_model_detail_options_are_explicit_and_secret_free():
    detail = ModelDetail(provider="openai", model_name="demo", options={"top_p": 0.8})
    assert detail.options == {"top_p": 0.8}
    with pytest.raises(ValueError, match="凭证"):
        ModelDetail(provider="openai", model_name="demo", options={"api_key": "secret"})
    with pytest.raises(ValueError, match="凭证"):
        ModelDetail(
            provider="openai",
            model_name="demo",
            options={"nested": {"headers": {"Authorization": "x"}}},
        )
    with pytest.raises(ValueError):
        ModelDetail(provider="openai", model_name="demo", unknown=True)


@pytest.mark.parametrize(
    "field_name",
    ["extra_headers", "extra_query"],
)
def test_completion_request_rejects_credentials_in_request_overrides(field_name):
    with pytest.raises(ValueError, match="请求头/query"):
        CompletionRequest(**{field_name: {"Authorization": "Bearer secret"}})


def test_completion_request_keeps_safe_request_overrides_isolated():
    headers = {"X-Trace-ID": "trace-1"}
    query = {"tenant": "demo"}
    request = CompletionRequest(extra_headers=headers, extra_query=query)

    assert request.extra_headers == headers
    assert request.extra_query == query
    assert request.extra_headers is not headers
    assert request.extra_query is not query


def test_completion_request_rejects_credentials_in_extra_body():
    with pytest.raises(ValueError, match="extra_body"):
        CompletionRequest(extra_body={"vendor": {"access_token": "secret"}})


def test_google_model_options_allow_safe_http_trace_headers():
    detail = ModelDetail(
        provider="google",
        model_name="gemini-test",
        options={"http_options": {"headers": {"X-Trace-ID": "trace"}}},
    )
    assert detail.options["http_options"]["headers"] == {"X-Trace-ID": "trace"}


def test_google_model_options_reject_sdk_http_credentials():
    http_options = types.HttpOptions(
        headers={"X-API-Key": "secret"},
    )
    with pytest.raises(ValueError, match="凭证"):
        ModelDetail(
            provider="google",
            model_name="gemini-test",
            options={"http_options": http_options},
        )


@pytest.mark.parametrize("timeout", ["1", 0, -1, float("nan"), float("inf")])
def test_google_request_timeout_must_be_finite_and_positive(timeout):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {}
    with pytest.raises(ValueError, match="timeout|正数"):
        provider._build_generation_config("system", None, 0.2, timeout=timeout)


def test_google_resource_config_rejects_nested_http_credentials():
    with pytest.raises(ValueError, match="凭证"):
        GoogleProvider._resource_config(
            lambda **values: values,
            {"http_options": {"headers": {"Authorization": "Bearer secret"}}},
            {},
        )


def test_google_options_extra_body_rejects_nested_credentials():
    with pytest.raises(ValueError, match="options.extra_body"):
        GoogleProvider(
            "gemini-test",
            options={"extra_body": {"routing": {"access_token": "secret"}}},
        )


@pytest.mark.parametrize("provider", ["openai", "qwen"])
def test_direct_openai_compatible_provider_rejects_request_header_and_query_options(provider):
    settings = SimpleNamespace(
        openai_api_key="key",
        qwen_api_key="key",
        qwen_base_url="https://qwen.invalid/v1",
        openai_api_base="https://api.openai.com/v1",
    )
    with pytest.raises(ValueError, match="请求头/query"):
        OpenAICompatibleProvider(
            "demo",
            provider,
            settings=settings,
            options={"default_headers": {"X-API-Key": "secret"}},
        )
    with pytest.raises(ValueError, match="请求头/query"):
        OpenAICompatibleProvider(
            "demo",
            provider,
            settings=settings,
            options={"default_query": {"access_token": "secret"}},
        )


def test_direct_openai_compatible_provider_rejects_base_url_and_http_client_options():
    settings = SimpleNamespace(
        qwen_api_key="key",
        qwen_base_url="https://qwen.invalid/v1",
    )
    with pytest.raises(ValueError, match="请求头/query"):
        OpenAICompatibleProvider(
            "demo",
            "qwen",
            settings=settings,
            options={"base_url": "https://proxy.invalid/v1"},
        )
    with pytest.raises(ValueError, match="请求头/query"):
        OpenAICompatibleProvider(
            "demo",
            "qwen",
            settings=settings,
            options={"http_client": object()},
        )


def test_direct_volcengine_provider_rejects_connection_override(monkeypatch):
    settings = SimpleNamespace(
        volc_access_key="ak",
        volc_secret_key="sk",
        ark_api_key=None,
        volc_base_url="https://ark.invalid/v3",
    )
    monkeypatch.setattr("src.providers.volcengine.get_settings", lambda: settings)
    with pytest.raises(ValueError, match="请求头/query"):
        VolcengineProvider(
            "doubao",
            options={"base_url": "https://proxy.invalid/v3"},
        )


def test_direct_anthropic_provider_rejects_request_header_and_query_options(monkeypatch):
    monkeypatch.setattr(
        "src.providers.anthropic.get_settings",
        lambda: SimpleNamespace(anthropic_api_key="key"),
    )
    with pytest.raises(ValueError, match="请求头/query"):
        AnthropicProvider(
            "claude",
            options={"default_headers": {"Authorization": "Bearer secret"}},
        )
    with pytest.raises(ValueError, match="请求头/query"):
        AnthropicProvider(
            "claude",
            options={"default_query": {"access_token": "secret"}},
        )


def test_direct_anthropic_provider_rejects_connection_override(monkeypatch):
    monkeypatch.setattr(
        "src.providers.anthropic.get_settings",
        lambda: SimpleNamespace(anthropic_api_key="key"),
    )
    with pytest.raises(ValueError, match="请求头/query"):
        AnthropicProvider("claude", options={"base_url": "https://proxy.invalid"})


def test_openai_chat_moderation_is_forwarded_only_for_official_provider():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    moderation = {"model": "omni-moderation-latest", "policy": {"input": {"mode": "block"}}}
    request = provider._build_chat_request(prompt="hi", moderation=moderation, stream=False)
    assert request["moderation"] == moderation

    provider._provider = "qwen"
    with pytest.raises(ValueError, match="moderation"):
        provider._build_chat_request(prompt="hi", moderation=moderation, stream=False)


def test_openai_compatibility_uses_endpoint_not_provider_name_for_official_fields():
    moderation = {"model": "omni-moderation-latest"}
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._options = {}
    provider._provider = "qwen"
    provider._base_url = "https://api.openai.com/v1"
    request = provider._build_chat_request(prompt="hi", moderation=moderation, stream=False)
    assert request["moderation"] == moderation

    provider._provider = "openai"
    provider._base_url = "https://gateway.example.com/v1"
    with pytest.raises(ValueError, match="moderation"):
        provider._build_chat_request(prompt="hi", moderation=moderation, stream=False)


def test_openai_chat_request_supports_messages_and_extended_parameters():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._provider = "openai"
    provider._options = {}
    request = provider._build_chat_request(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        temperature=0.2,
        max_tokens=128,
        top_p=0.9,
        seed=7,
        response_format={"type": "json_object"},
        stream=False,
    )
    assert request["messages"][0]["content"][0]["type"] == "text"
    assert request["max_completion_tokens"] == 128
    assert request["seed"] == 7
    assert request["response_format"]["type"] == "json_object"


def test_completion_request_omits_implicit_temperature_for_model_defaults():
    request = CompletionRequest(prompt="hi")
    assert request.temperature == 0.7
    assert "temperature" not in request.to_invoke_kwargs()

    explicit = CompletionRequest(prompt="hi", temperature=0.7)
    assert explicit.to_invoke_kwargs()["temperature"] == 0.7


def test_completion_request_copy_preserves_implicit_temperature_marker():
    request = CompletionRequest(prompt="hi")
    copied = request.copy_with(stream=False)

    assert copied.temperature == 0.7
    assert copied.temperature_is_explicit is False
    assert "temperature" not in copied.to_invoke_kwargs()


def test_openai_model_temperature_remains_the_default_when_request_omits_it():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {"temperature": 0.2}

    request = CompletionRequest(prompt="hi")
    params = provider._build_chat_request(**request.to_invoke_kwargs())

    assert params["temperature"] == 0.2


def test_google_model_temperature_remains_the_default_when_request_omits_it():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {"temperature": 0.2}

    request = CompletionRequest(prompt="hi")
    config = provider._build_generation_config(
        None,
        None,
        request.temperature if request.temperature_is_explicit else None,
    )

    assert config.temperature == 0.2


def test_legacy_openai_invoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {"temperature": 0.2}
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    assert list(provider.invoke("hello", stream=False)) == ["ok"]
    assert calls[0]["temperature"] == 0.2


def test_legacy_openai_ainvoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {"temperature": 0.2}
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [item async for item in provider.ainvoke("hello", stream=False)]

    assert asyncio.run(consume()) == ["ok"]
    assert calls[0]["temperature"] == 0.2


def test_legacy_google_invoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {"temperature": 0.2}
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="ok")

    client = SimpleNamespace(models=Models())
    provider._get_client = lambda: client
    provider._effective_vertex_mode = lambda _client: False

    assert list(provider.invoke("hello", stream=False)) == ["ok"]
    assert calls[0]["config"].temperature == 0.2


def test_legacy_google_ainvoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {"temperature": 0.2}
    calls = []

    class Models:
        async def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="ok")

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    provider._get_client = lambda: client
    provider._effective_vertex_mode = lambda _client: False

    async def consume():
        return [item async for item in provider.ainvoke("hello", stream=False)]

    assert asyncio.run(consume()) == ["ok"]
    assert calls[0]["config"].temperature == 0.2


def test_legacy_volcengine_invoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "chat_completions"
    provider._options = {"temperature": 0.2}
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    assert list(provider.invoke("hello", stream=False)) == ["ok"]
    assert calls[0]["temperature"] == 0.2


def test_legacy_volcengine_ainvoke_preserves_model_temperature_when_omitted():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "chat_completions"
    provider._options = {"temperature": 0.2}
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [item async for item in provider.ainvoke("hello", stream=False)]

    assert asyncio.run(consume()) == ["ok"]
    assert calls[0]["temperature"] == 0.2


def test_openai_compatible_rejects_conflicting_model_length_aliases():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {"max_tokens": 64, "max_completion_tokens": 128}

    with pytest.raises(ValueError, match="max_tokens.*max_completion_tokens"):
        provider._build_chat_request(prompt="hi", stream=False)


def test_openai_chat_extra_body_cannot_override_request_fields():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._options = {"extra_body": {"temperature": 0.1}}

    with pytest.raises(ValueError, match="extra_body.*temperature"):
        provider._build_chat_request(prompt="hi", temperature=0.7, stream=False)


def test_openai_request_builder_rejects_sensitive_http_overrides():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {}

    with pytest.raises(ValueError, match="请求头/query"):
        provider._build_chat_request(
            prompt="hi",
            extra_headers={"X-API-Key": "secret"},
            stream=False,
        )
    with pytest.raises(ValueError, match="请求头/query"):
        provider._build_responses_request(
            prompt="hi",
            extra_query={"access_token": "secret"},
            stream=False,
        )


def test_openai_resource_calls_fail_before_sdk_on_unknown_or_missing_arguments():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")

    with pytest.raises(ValueError, match="SDK.*unknown_option"):
        provider.create_upload(
            bytes=10,
            filename="demo.txt",
            mime_type="text/plain",
            purpose="assistants",
            unknown_option=True,
        )

    with pytest.raises(ValueError, match="SDK.*part_ids"):
        provider.complete_upload("upload-1")


def test_openai_direct_resource_calls_reject_sensitive_http_overrides():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")

    with pytest.raises(ValueError, match="请求头/query 不允许包含凭证"):
        provider.list_files(extra_headers={"X-API-Key": "secret"})
    with pytest.raises(ValueError, match="请求头/query 不允许包含凭证"):
        provider.list_files(extra_query={"access_token": "secret"})


def test_openai_vector_store_resource_calls_follow_installed_sdk_signature():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")

    with pytest.raises(ValueError, match="SDK.*unknown_option"):
        provider.create_vector_store(name="demo", unknown_option=True)


def test_openai_resource_creation_paths_reject_sensitive_overrides():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")
    cases = (
        (provider.create_vector_store, {"extra_headers": {"X-API-Key": "secret"}}),
        (provider.create_upload, {"extra_query": {"access_token": "secret"}}),
        (provider.complete_upload, {"upload_id": "upload-1", "extra_body": {"token": "secret"}}),
    )
    for operation, kwargs in cases:
        with pytest.raises(ValueError, match="凭证"):
            operation(**kwargs)


def test_openai_async_resource_creation_paths_reject_sensitive_overrides():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_aclient = lambda: openai.AsyncOpenAI(api_key="local-only")

    async def run() -> None:
        cases = (
            (provider.async_create_vector_store, {"extra_headers": {"X-API-Key": "secret"}}),
            (provider.async_create_upload, {"extra_query": {"access_token": "secret"}}),
            (
                provider.async_complete_upload,
                {"upload_id": "upload-1", "extra_body": {"token": "secret"}},
            ),
        )
        for operation, kwargs in cases:
            with pytest.raises(ValueError, match="凭证"):
                await operation(**kwargs)

    asyncio.run(run())


def test_openai_resource_groups_validate_unknown_arguments_before_dispatch():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")
    cases = (
        (provider.upload_file, (b"data",), {"purpose": "assistants", "unknown_option": True}),
        (provider.create_batch, ("file-1",), {"unknown_option": True}),
        (provider.create_conversation, (), {"unknown_option": True}),
        (provider.create_fine_tuning_job, (), {"unknown_option": True}),
        (provider.create_eval, (), {"unknown_option": True}),
    )
    for operation, args, kwargs in cases:
        with pytest.raises(ValueError, match="SDK.*unknown_option"):
            operation(*args, **kwargs)


def test_openai_resource_missing_required_arguments_are_project_errors():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")
    cases = (
        (provider.create_fine_tuning_job, (), {}, "model"),
        (provider.create_eval, (), {}, "data_source_config"),
        (provider.create_video, (), {}, "prompt"),
        (provider.update_conversation, ("conversation-1",), {}, "metadata"),
    )
    for operation, args, kwargs, missing in cases:
        with pytest.raises(ValueError, match=f"SDK.*{missing}"):
            operation(*args, **kwargs)


def test_openai_async_resource_groups_validate_unknown_arguments_before_dispatch():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_aclient = lambda: openai.AsyncOpenAI(api_key="local-only")

    async def run() -> None:
        cases = (
            (
                provider.async_upload_file,
                (b"data",),
                {"purpose": "assistants", "unknown_option": True},
            ),
            (provider.async_create_batch, ("file-1",), {"unknown_option": True}),
            (provider.async_create_fine_tuning_job, (), {"unknown_option": True}),
        )
        for operation, args, kwargs in cases:
            with pytest.raises(ValueError, match="SDK.*unknown_option"):
                await operation(*args, **kwargs)

    asyncio.run(run())


def test_openai_async_resource_missing_required_arguments_are_project_errors():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_aclient = lambda: openai.AsyncOpenAI(api_key="local-only")

    async def run() -> None:
        cases = (
            (provider.async_create_fine_tuning_job, (), {}, "model"),
            (provider.async_create_eval, (), {}, "data_source_config"),
            (provider.async_create_video, (), {}, "prompt"),
            (provider.async_update_conversation, ("conversation-1",), {}, "metadata"),
        )
        for operation, args, kwargs, missing in cases:
            with pytest.raises(ValueError, match=f"SDK.*{missing}"):
                await operation(*args, **kwargs)

    asyncio.run(run())


def test_openai_media_and_management_resources_validate_sdk_arguments():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    import openai

    provider._get_client = lambda: openai.OpenAI(api_key="local-only")
    cases = (
        (provider.create_container, (), {"name": "demo", "unknown_option": True}),
        (
            provider.create_fine_tuning_job,
            (),
            {"model": "demo", "training_file": "file-1", "unknown_option": True},
        ),
        (
            provider.create_eval,
            (),
            {
                "data_source_config": {},
                "testing_criteria": [],
                "unknown_option": True,
            },
        ),
        (provider.generate_image, ("hello",), {"unknown_option": True}),
        (
            provider.text_to_speech,
            ("hello", "tts-1", "alloy"),
            {"unknown_option": True},
        ),
        (provider.create_video, (), {"prompt": "hello", "unknown_option": True}),
        (provider.create_conversation, (), {"items": [], "unknown_option": True}),
    )
    for operation, args, kwargs in cases:
        with pytest.raises(ValueError, match="SDK.*unknown_option"):
            operation(*args, **kwargs)


def test_openai_responses_extra_body_cannot_override_request_fields():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._options = {
        "server_verified_protocols": ["responses"],
        "extra_body": {"input": "other"},
    }

    with pytest.raises(ValueError, match="extra_body.*input"):
        provider._build_responses_request(prompt="hi", stream=False)


@pytest.mark.parametrize("field", ["top_k", "repetition_penalty", "reasoning"])
def test_openai_chat_explicit_extra_body_extensions_cannot_be_silently_overridden(field):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {"extra_body": {field: {"effort": "low"} if field == "reasoning" else 1}}

    kwargs = {field: {"effort": "high"} if field == "reasoning" else 2}
    with pytest.raises(ValueError, match=field):
        provider._build_chat_request(prompt="hi", stream=False, **kwargs)


def test_ark_responses_rejects_instructions_with_enabled_caching():
    """官方规定 ``instructions`` 与 ``caching={"type": "enabled"}`` 互斥。

    配置 instructions 后本轮请求无法写入或使用缓存，caching 为 enabled 时
    服务端会直接报错；SDK 不做本地校验，因此这里要求在构造阶段显式失败。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    with pytest.raises(ValueError, match="instructions.*caching"):
        provider._build_responses_request(
            CompletionRequest(
                prompt="hi",
                system_prompt="你是助手",
                caching={"type": "enabled"},
                stream=False,
            )
        )


def test_ark_responses_accepts_system_prompt_without_enabled_caching():
    """未启用 caching 时 instructions 正常透传；caching 非 enabled 也不拦截。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    params = provider._build_responses_request(
        CompletionRequest(prompt="hi", system_prompt="你是助手", stream=False)
    )
    assert params["instructions"] == "你是助手"
    assert "caching" not in params

    params = provider._build_responses_request(
        CompletionRequest(
            prompt="hi", system_prompt="你是助手", caching={"type": "disabled"}, stream=False
        )
    )
    assert params["instructions"] == "你是助手"
    assert params["caching"] == {"type": "disabled"}


def test_ark_responses_enabled_caching_ok_without_instructions():
    """把系统提示放进 messages 并显式关闭 system_prompt 时，caching 可正常启用。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    params = provider._build_responses_request(
        CompletionRequest(
            messages=[
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "hi"},
            ],
            system_prompt=None,
            caching={"type": "enabled"},
            stream=False,
        )
    )
    assert "instructions" not in params
    assert params["caching"] == {"type": "enabled"}


@pytest.mark.parametrize("field", ["top_k", "seed", "stop", "reasoning"])
def test_openai_responses_explicit_extra_body_extensions_cannot_be_silently_overridden(field):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"], "extra_body": {field: 1}}

    kwargs = {field: {"effort": "high"} if field == "reasoning" else 2}
    with pytest.raises(ValueError, match=field):
        provider._build_responses_request(prompt="hi", stream=False, **kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_embedding_extra_body_conflicts_are_explicit(async_mode):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "embedding"
    provider._provider = "openai"
    provider._options = {"extra_body": {"tenant": "configured"}}
    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **_: None))
    provider._aclient = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **_: None))

    if async_mode:

        async def run():
            await provider.aembed_documents(["text"], extra_body={"tenant": "request"})

        with pytest.raises(ValueError, match="tenant"):
            asyncio.run(run())
    else:
        with pytest.raises(ValueError, match="tenant"):
            provider.embed_documents(["text"], extra_body={"tenant": "request"})


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "field_name, value",
    [
        ("extra_headers", {"Authorization": "Bearer secret"}),
        ("extra_headers", {"X-API-Key": "secret"}),
        ("extra_query", {"access_token": "secret"}),
        ("extra_query", {"api_key": "secret"}),
    ],
)
def test_openai_embedding_request_overrides_reject_credentials(async_mode, field_name, value):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "embedding"
    provider._provider = "openai"
    provider._options = {}
    provider._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: SimpleNamespace(data=[]))
    )

    kwargs = {field_name: value}
    if async_mode:

        async def run():
            await provider.aembed_documents(["text"], **kwargs)

        with pytest.raises(ValueError, match="请求头/query|凭证"):
            asyncio.run(run())
    else:
        with pytest.raises(ValueError, match="请求头/query|凭证"):
            provider.embed_documents(["text"], **kwargs)


def test_openai_embedding_rejects_unknown_model_options():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._options = {"dimensons": 512}

    with pytest.raises(ValueError, match="不支持请求参数: dimensons"):
        provider._embedding_options()


def test_openai_chat_rejects_unimplemented_request_fields():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._options = {}

    with pytest.raises(ValueError, match="不支持请求参数: background"):
        provider._build_chat_request(prompt="hi", background=True)


def test_non_openai_compatible_provider_requires_explicit_base_url(monkeypatch):
    settings = SimpleNamespace(qwen_api_key="token", qwen_base_url=None)
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: settings)

    with pytest.raises(ValueError, match="qwen_base_url"):
        OpenAICompatibleProvider("demo", "qwen")


def test_non_openai_chat_rejects_openai_only_request_fields():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "chat_completions"
    provider._options = {}

    with pytest.raises(ValueError, match="modalities"):
        provider._build_chat_request(prompt="hi", modalities=["audio"], stream=False)


def test_google_rejects_unimplemented_request_fields():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    with pytest.raises(ValueError, match="不支持请求参数: background"):
        provider.complete(CompletionRequest(prompt="hi", background=True))


def test_openai_complete_preserves_chat_text_tool_calls_and_usage():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="answer",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(name="lookup", arguments='{"id":1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )
    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    result = provider.complete(CompletionRequest(prompt="hi", stream=False))
    assert result.text == "answer"
    assert result.tool_calls[0]["name"] == "lookup"
    assert result.usage["total_tokens"] == 7


def test_openai_async_complete_preserves_responses_metadata():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._options = {}

    class Responses:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                id="resp-async",
                status="completed",
                output_text="异步答复",
                usage=SimpleNamespace(input_tokens=4, output_tokens=5, total_tokens=9),
            )

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    result = asyncio.run(provider.acomplete(CompletionRequest(prompt="hi")))

    assert result.text == "异步答复"
    assert result.finish_reason == "completed"
    assert result.usage["input_tokens"] == 4


def test_openai_provider_responses_dispatch_uses_inherited_responses_adapter():
    provider = object.__new__(OpenAIProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://api.openai.com/v1"
    provider._options = {}
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status="completed", output_text="答复")

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    result = provider.complete(CompletionRequest(prompt="hello"))

    assert result.text == "答复"
    assert calls == [
        {
            "model": "demo",
            "input": "hello",
            "stream": False,
            "instructions": "You are a helpful assistant.",
        }
    ]


def test_openai_responses_verbosity_uses_text_config():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._options = {}

    request = provider._build_responses_request(prompt="hi", stream=False, verbosity="low")

    assert request["text"] == {"verbosity": "low"}
    assert "verbosity" not in request


def test_openai_responses_parse_does_not_forward_internal_stream_flag():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    calls = []

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return "parsed"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    assert provider.parse(dict, CompletionRequest(prompt="hi")) == "parsed"
    assert calls[0]["text_format"] is dict
    assert "stream" not in calls[0]


def test_openai_async_responses_parse_does_not_forward_internal_stream_flag():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    calls = []

    class Responses:
        async def parse(self, **kwargs):
            calls.append(kwargs)
            return "parsed"

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())
    assert asyncio.run(provider.async_parse(dict, CompletionRequest(prompt="hi"))) == "parsed"
    assert calls[0]["text_format"] is dict
    assert "stream" not in calls[0]


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_responses_parse_rejects_background_mode(async_mode):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"background": True}
    calls = []

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return "parsed"

    class AsyncResponses:
        async def parse(self, **kwargs):
            calls.append(kwargs)
            return "parsed"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    provider._get_aclient = lambda: SimpleNamespace(responses=AsyncResponses())

    with pytest.raises(ValueError, match="background"):
        if async_mode:
            asyncio.run(provider.async_parse(dict, CompletionRequest(prompt="hi")))
        else:
            provider.parse(dict, CompletionRequest(prompt="hi"))

    assert calls == []


def test_openai_responses_normalizes_model_options_to_sdk_shape():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._options = {
        "max_tokens": 256,
        "verbosity": "low",
        "response_format": {"type": "json_object"},
    }

    request = provider._build_responses_request(prompt="hi", stream=False)

    assert request["max_output_tokens"] == 256
    assert request["text"] == {
        "format": {"type": "json_object"},
        "verbosity": "low",
    }
    assert "max_tokens" not in request
    assert "response_format" not in request
    assert "verbosity" not in request


def test_openai_responses_normalizes_completion_and_reasoning_options():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {
        "max_completion_tokens": 256,
        "reasoning_effort": "medium",
    }

    request = provider._build_responses_request(prompt="hi", stream=False)

    assert request["max_output_tokens"] == 256
    assert request["reasoning"] == {"effort": "medium"}
    assert "max_completion_tokens" not in request
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("field", ["stop", "seed", "top_k"])
def test_openai_responses_rejects_chat_only_sampling_fields(field):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}

    with pytest.raises(ValueError, match=field):
        provider._build_responses_request(prompt="hi", stream=False, **{field: 1})


def test_compatible_responses_keeps_vendor_sampling_extensions_in_extra_body():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._options = {}

    request = provider._build_responses_request(
        prompt="hi", stream=False, stop=["END"], seed=7, top_k=20
    )

    assert request["extra_body"] == {"stop": ["END"], "seed": 7, "top_k": 20}


def test_compatible_responses_allows_supported_openai_response_extensions():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._base_url = "https://gateway.invalid/v1"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._options = {}

    request = provider._build_responses_request(
        prompt="hi",
        stream=False,
        store=True,
        prompt_cache_key="cache-key",
        prompt_cache_options={"scope": "user"},
        prompt_cache_retention="24h",
        safety_identifier="user-1",
        moderation={"mode": "strict"},
        verbosity="low",
    )

    assert request["store"] is True
    assert request["prompt_cache_key"] == "cache-key"
    assert request["prompt_cache_options"] == {"scope": "user"}
    assert request["prompt_cache_retention"] == "24h"
    assert request["safety_identifier"] == "user-1"
    assert request["moderation"] == {"mode": "strict"}
    assert request["text"]["verbosity"] == "low"


def test_factory_protocol_status_uses_runtime_verified_options():
    assert ModelProviderFactory.protocol_status(
        "qwen",
        "responses",
        options={"server_verified_protocols": ["responses"]},
    ) == {
        "adapter_supported": True,
        "server_verified": True,
    }

    report = ModelProviderFactory.capability_report(
        "qwen",
        options={"server_verified_protocols": ["responses"]},
    )
    assert report["server_verified_protocols"] == ["responses"]
    assert report["protocol_status"]["responses"]["server_verified"] is True


@pytest.mark.parametrize(
    ("provider", "alias", "canonical"),
    [
        ("openai", "responses", "responses"),
        ("openai", "response", "responses"),
        ("anthropic", "anthropic", "messages"),
        ("anthropic", "anthropic_messages", "messages"),
        ("google", "gemini", "generate_content"),
        ("google", "generate", "generate_content"),
    ],
)
def test_factory_protocol_status_normalizes_protocol_aliases(provider, alias, canonical):
    assert ModelProviderFactory.protocol_status(
        provider, alias
    ) == ModelProviderFactory.protocol_status(provider, canonical)


def test_factory_google_protocol_status_accepts_safe_http_options():
    status = ModelProviderFactory.protocol_status(
        "google",
        "generate_content",
        options={"http_options": {"headers": {"X-Trace-ID": "trace"}}},
    )

    assert status == {"adapter_supported": True, "server_verified": False}
    report = ModelProviderFactory.capability_report(
        "google",
        options={"http_options": {"headers": {"X-Trace-ID": "trace"}}},
    )
    assert report["protocol_status"]["generate_content"]["server_verified"] is False


def test_factory_google_protocol_status_rejects_credentials_in_http_options():
    with pytest.raises(ValueError, match="http_options"):
        ModelProviderFactory.protocol_status(
            "google",
            "generate_content",
            options={"http_options": {"headers": {"Authorization": "Bearer secret"}}},
        )


def test_openai_responses_request_values_override_model_options():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._options = {
        "max_tokens": 256,
        "verbosity": "low",
        "response_format": {"type": "json_object"},
    }

    request = provider._build_responses_request(
        prompt="hi",
        stream=False,
        max_tokens=512,
        verbosity="high",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
            },
        },
    )

    assert request["max_output_tokens"] == 512
    assert request["text"]["verbosity"] == "high"
    assert request["text"]["format"]["name"] == "answer"


@pytest.mark.parametrize("field", ["thinking", "caching", "session", "expire_at"])
def test_openai_responses_rejects_unsupported_model_options(field):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._options = {field: {"enabled": True} if field != "expire_at" else 123}

    with pytest.raises(ValueError, match=field):
        provider._build_responses_request(prompt="hi", stream=False)


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_ark_rejects_unknown_model_options(protocol):
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = protocol
    provider._options = {"unknown_option": True}

    with pytest.raises(ValueError, match="unknown_option"):
        if protocol == "responses":
            provider._build_responses_request(CompletionRequest(prompt="hi", stream=False))
        else:
            provider._build_chat_request(prompt="hi", stream=False)


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_openai_compatible_rejects_unknown_model_options(protocol):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = protocol
    provider._server_verified_protocols = frozenset({"responses"})
    provider._options = {"unknown_option": True}

    with pytest.raises(ValueError, match="unknown_option"):
        if protocol == "responses":
            provider._build_responses_request(prompt="hi", stream=False)
        else:
            provider._build_chat_request(prompt="hi", stream=False)


def test_ark_responses_rejects_top_logprobs_at_request_and_model_boundaries():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"

    provider._options = {}
    with pytest.raises(ValueError, match="top_logprobs"):
        provider._build_responses_request(
            CompletionRequest(prompt="hi", stream=False, top_logprobs=2)
        )

    provider._options = {"top_logprobs": 2}
    with pytest.raises(ValueError, match="top_logprobs"):
        provider._build_responses_request(CompletionRequest(prompt="hi", stream=False))


def test_ark_responses_validates_session_mapping_and_conversation_conflicts():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {}

    with pytest.raises(ValueError, match="conversation.*对象"):
        provider._build_responses_request(
            CompletionRequest(prompt="hi", stream=False, conversation="conversation-id")
        )

    with pytest.raises(ValueError, match="conversation.*session"):
        provider._build_responses_request(
            CompletionRequest(
                prompt="hi",
                stream=False,
                conversation={"id": "conversation-id"},
                session={"id": "session-id"},
            )
        )

    request = provider._build_responses_request(
        CompletionRequest(prompt="hi", stream=False, conversation={"id": "conversation-id"})
    )
    assert request["session"] == {"id": "conversation-id"}


def test_ark_native_response_resource_requires_explicit_model_and_rejects_unknown_kwargs():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "configured-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return "response"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(ValueError, match="model"):
        provider.resources.create_response(input="raw")
    with pytest.raises(ValueError, match="unknown_option"):
        provider.resources.create_response(input="raw", model="caller-model", unknown_option=True)

    assert provider.resources.create_response(input="raw", model="caller-model") == "response"
    assert calls == [{"input": "raw", "model": "caller-model"}]


def test_secret_free_validation_isolates_nested_values():
    options = {"extra_body": {"routing": {"tenant": "demo"}}}
    validated_options = validate_secret_free_options(options, "Provider")
    validated_payload = validate_secret_free_payload(options, "Provider", "extra_body")

    assert validated_options is not options
    assert validated_options["extra_body"] is not options["extra_body"]
    assert validated_payload["extra_body"] is not options["extra_body"]

    options["extra_body"]["routing"]["tenant"] = "changed"
    assert validated_options["extra_body"]["routing"]["tenant"] == "demo"
    assert validated_payload["extra_body"]["routing"]["tenant"] == "demo"


def test_resource_security_validation_resolves_extension_key_variants():
    values = {
        "Extra-Headers": {"X-Trace": "trace"},
        "EXTRA_QUERY": {"tenant": "demo"},
        "extra-body": {"routing": {"tenant": "demo"}},
    }
    validated = validate_secret_free_resource_kwargs(values, "Provider")

    assert validated["Extra-Headers"] == {"X-Trace": "trace"}
    assert validated["EXTRA_QUERY"] == {"tenant": "demo"}
    assert validated["extra-body"] == {"routing": {"tenant": "demo"}}
    assert validated["Extra-Headers"] is not values["Extra-Headers"]
    assert validated["extra-body"] is not values["extra-body"]


@pytest.mark.parametrize(
    "key",
    [
        "openai_api_key",
        "api-token",
        "ApiToken",
        "token",
        "myClientSecret",
        "myApiKey",
        "myAPIKey",
        "private_key",
        "ssh_key",
        "ssh-private-key",
        "signing_key",
        "encryption_key",
    ],
)
def test_security_recognizes_separated_credential_key_variants(key):
    assert is_sensitive_option_key(key) is True


def test_security_allows_non_credential_header_descriptors():
    assert is_sensitive_option_key("X-Authorization-Count") is False
    assert is_sensitive_option_key("X-Credential-Provider") is False


def test_security_scans_string_values_for_embedded_credentials():
    found = find_sensitive_option_paths({"X-Trace": "Bearer secret-token", "tenant": "safe"})
    assert "X-Trace" in found


@pytest.mark.parametrize(
    "value",
    [
        '{"api_key": "secret-value"}',
        "{'token': 'secret-value'}",
    ],
)
def test_security_redacts_json_embedded_credentials(value):
    redacted = redact_sensitive_text(value)
    assert "secret-value" not in redacted
    assert "REDACTED" in redacted
    assert find_sensitive_option_paths({"message": value}) == ["message"]


@pytest.mark.parametrize(
    "value",
    [
        "token usage",
        "token verification failed",
        "what is the token budget",
        "token bucket algorithm",
        "renew authorization flow",
    ],
)
def test_security_does_not_treat_ordinary_text_as_credentials(value):
    assert redact_sensitive_text(value) == value
    assert find_sensitive_option_paths({"note": value}) == []
    assert validate_secret_free_options({"note": value}, "Provider")["note"] == value


def test_security_still_redacts_whitespace_separated_credential_like_values():
    value = "token 0123456789abcdef"
    assert "REDACTED" in redact_sensitive_text(value)
    assert find_sensitive_option_paths({"value": value}) == ["value"]


def test_security_redacts_credentials_in_non_string_exception_values():
    class CredentialError:
        def __str__(self):
            return "Authorization: Bearer secret-token-123456"

    assert "REDACTED" in redact_sensitive_text(CredentialError())


def test_secret_free_validation_deep_copies_sdk_models():
    http_options = types.HttpOptions(headers={"X-Trace-ID": "trace"})
    validated = validate_secret_free_options(
        {"http_options": http_options},
        "Google",
        allowed_containers={"http_options", "headers"},
    )

    http_options.headers["X-Trace-ID"] = "changed"
    assert validated["http_options"].headers["X-Trace-ID"] == "trace"
    assert validated["http_options"] is not http_options


def test_openai_responses_merges_native_text_options():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._options = {
        "text": {"verbosity": "low"},
        "response_format": {"type": "json_object"},
    }

    request = provider._build_responses_request(prompt="hi", stream=False)

    assert request["text"] == {
        "format": {"type": "json_object"},
        "verbosity": "low",
    }


def test_openai_chat_model_max_tokens_uses_current_sdk_field():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._provider = "openai"
    provider._options = {"max_tokens": 256}

    request = provider._build_chat_request(prompt="hi", stream=False, max_tokens=512)

    assert request["max_completion_tokens"] == 512
    assert "max_tokens" not in request


def test_openai_custom_gateway_keeps_legacy_chat_max_tokens_field():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._provider = "openai"
    provider._base_url = "https://gateway.example/v1"
    provider._options = {}

    request = provider._build_chat_request(prompt="hi", stream=False, max_tokens=512)

    assert request["max_tokens"] == 512
    assert "max_completion_tokens" not in request


@pytest.mark.parametrize("provider_name", ["qwen", "ollama", "lm-studio", "grok", "siliconflow"])
def test_compatible_chat_models_keep_max_tokens_field(provider_name):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = provider_name
    provider._protocol = "chat_completions"
    provider._options = {}

    request = provider._build_chat_request(prompt="hi", stream=False, max_tokens=256)

    assert request["max_tokens"] == 256
    assert "max_completion_tokens" not in request


def test_google_async_invoke_stream_and_non_stream_issue_one_request_each():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    calls = {"stream": 0, "complete": 0}

    class AsyncModels:
        def generate_content_stream(self, **_kwargs):
            calls["stream"] += 1
            return iter(
                [
                    SimpleNamespace(text="流"),
                    SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")]),
                ]
            )

        async def generate_content(self, **_kwargs):
            calls["complete"] += 1
            return SimpleNamespace(text="非流")

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=AsyncModels()))

    async def collect(stream):
        return [
            chunk async for chunk in provider.ainvoke(prompt="hi", stream=stream, temperature=None)
        ]

    assert asyncio.run(collect(True)) == ["流"]
    assert calls == {"stream": 1, "complete": 0}
    assert asyncio.run(collect(False)) == ["非流"]
    assert calls == {"stream": 1, "complete": 1}


def test_google_query_embedding_uses_query_task(monkeypatch):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "embedding"
    provider._options = {}
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0])])
    client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kwargs: response))
    provider._get_client = lambda: client
    assert provider.embed_documents(["question"], task_type="RETRIEVAL_QUERY") == [[1.0, 2.0]]


def test_google_http_extensions_map_to_sdk_http_options():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    config = provider._build_generation_config(
        "system",
        None,
        0.2,
        extra_headers={"X-Trace": "trace-1"},
        extra_body={"tenant": "demo"},
        timeout=1.25,
    )

    assert config.http_options.headers == {"X-Trace": "trace-1"}
    assert config.http_options.extra_body == {"tenant": "demo"}
    assert config.http_options.timeout == 1250
    with pytest.raises(ValueError, match="extra_query"):
        provider._build_generation_config("system", None, 0.2, extra_query={"tenant": "demo"})


@pytest.mark.parametrize("field", ["timeout", "extra_body"])
def test_google_http_options_conflict_with_model_level_http_fields_at_init(field):
    options = {"http_options": {"headers": {"X-Trace": "trace"}}}
    options[field] = 2 if field == "timeout" else {"tenant": "model"}

    with pytest.raises(ValueError, match="http_options.*模型级.*" + field):
        GoogleProvider("gemini", options=options)


def test_google_model_http_options_apply_to_embedding_and_token_count():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {"timeout": 2, "extra_body": {"tenant": "model"}}
    seen = {}

    class Models:
        def count_tokens(self, **kwargs):
            seen["count"] = kwargs
            return SimpleNamespace(total_tokens=3)

        def embed_content(self, **kwargs):
            seen["embed"] = kwargs
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0])])

    provider._get_client = lambda: SimpleNamespace(models=Models())

    assert provider.count_tokens(CompletionRequest(prompt="hi")) == 3
    assert provider.embed_documents(["doc"]) == [[1.0, 2.0]]
    for call in (seen["count"], seen["embed"]):
        assert call["config"].http_options.timeout == 2000
        assert call["config"].http_options.extra_body == {"tenant": "model"}


def test_google_request_headers_reject_sensitive_keys():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {}

    with pytest.raises(ValueError, match="请求头/query"):
        provider._build_generation_config(
            "system",
            None,
            0.2,
            extra_headers={"X-API-Key": "secret"},
        )


def test_google_async_token_count_and_embedding_forward_http_options():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    seen = {}

    class AsyncModels:
        async def count_tokens(self, **kwargs):
            seen["count"] = kwargs
            return SimpleNamespace(total_tokens=7)

        async def embed_content(self, **kwargs):
            seen["embed"] = kwargs
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=AsyncModels()))
    request = CompletionRequest(
        prompt="hi",
        extra_headers={"X-Trace": "trace-2"},
        extra_body={"tenant": "demo"},
        timeout=2,
    )

    assert asyncio.run(provider.async_count_tokens(request)) == 7
    count_http = seen["count"]["config"].http_options
    assert count_http.headers == {"X-Trace": "trace-2"}
    assert count_http.extra_body == {"tenant": "demo"}
    assert count_http.timeout == 2000

    assert asyncio.run(
        provider.aembed_documents(
            ["doc"],
            extra_headers={"X-Trace": "trace-3"},
            timeout=3,
        )
    ) == [[0.1, 0.2]]
    embed_config = seen["embed"]["config"]
    assert embed_config.http_options.headers == {"X-Trace": "trace-3"}
    assert embed_config.http_options.timeout == 3000


def test_google_compute_tokens_forwards_http_options_sync_and_async():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    seen = {}

    class Models:
        def compute_tokens(self, **kwargs):
            seen["sync"] = kwargs
            return "sync"

    class AsyncModels:
        async def compute_tokens(self, **kwargs):
            seen["async"] = kwargs
            return "async"

    provider._get_client = lambda: SimpleNamespace(
        models=Models(), aio=SimpleNamespace(models=AsyncModels())
    )
    request = CompletionRequest(
        prompt="hi",
        extra_headers={"X-Trace": "trace"},
        extra_body={"tenant": "demo"},
        timeout=2,
    )

    assert provider.compute_tokens(request) == "sync"
    assert asyncio.run(provider.async_compute_tokens(request)) == "async"
    for call in (seen["sync"], seen["async"]):
        assert call["model"] == "gemini"
        assert call["config"].http_options.headers == {"X-Trace": "trace"}
        assert call["config"].http_options.extra_body == {"tenant": "demo"}
        assert call["config"].http_options.timeout == 2000


def test_google_token_count_rejects_ignored_generation_fields():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    with pytest.raises(ValueError, match="max_tokens"):
        provider.count_tokens(CompletionRequest(prompt="hi", max_tokens=10))
    with pytest.raises(ValueError, match="tools"):
        provider.compute_tokens(CompletionRequest(prompt="hi", tools=[{"type": "function"}]))


def test_google_compute_tokens_rejects_gemini_developer_mode():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(
        _api_client=SimpleNamespace(vertexai=False),
        models=SimpleNamespace(compute_tokens=lambda **_: None),
    )
    with pytest.raises(ValueError, match="仅支持 Vertex"):
        provider.compute_tokens(CompletionRequest(prompt="hi"))


def test_openai_embedding_rejects_call_level_model_and_input_override():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "embedding"
    provider._provider = "openai"
    provider._options = {}
    provider._client = None

    with pytest.raises(ValueError, match="不允许覆盖请求字段"):
        provider.embed_documents(["text"], model="other")


def test_google_stream_events_cover_text_thought_tool_usage_and_finish():
    chunk = SimpleNamespace(
        response_id="gemini-1",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="思考", thought=True),
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="lookup", args={"id": 1})
                        ),
                        SimpleNamespace(text="答案", thought=False),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=2, candidates_token_count=3, total_token_count=5
        ),
    )

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [event.type for event in events] == [
        "reasoning_delta",
        "tool_call_delta",
        "text_delta",
        "finish",
        "usage",
    ]
    assert events[1].tool_call["arguments"] == {"id": 1}
    assert events[-1].usage["total_tokens"] == 5


def test_google_tool_call_merge_preserves_nested_partial_arguments():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "arguments": {"query": {"language": "zh"}}},
    )
    _, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "arguments": {"query": {"limit": 5}}},
    )

    assert merged["arguments"] == {
        "query": {"language": "zh", "limit": 5},
    }


def test_google_tool_call_merge_keeps_index_when_later_fragment_omits_id():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "index": 0, "name": "lookup", "arguments": {"first": 1}},
    )
    _, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"index": 0, "arguments": {"second": 2}},
    )

    assert len(accumulator) == 1
    assert merged["id"] == "call-1"
    assert merged["arguments"] == {"first": 1, "second": 2}


def test_google_tool_call_merge_keeps_id_when_later_fragment_has_no_identity_fields():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "name": "lookup", "arguments": {"first": 1}},
    )
    _, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"name": "lookup", "arguments": {"second": 2}},
    )

    assert len(accumulator) == 1
    assert merged["id"] == "call-1"
    assert merged["arguments"] == {"first": 1, "second": 2}


def test_google_tool_call_merge_migrates_index_to_late_id():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"index": 0, "name": "lookup", "arguments": {"first": 1}},
    )
    key, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "name": "lookup", "arguments": {"second": 2}},
    )

    assert key == ("index", 0)
    assert len(accumulator) == 1
    assert merged["id"] == "call-1"
    assert merged["index"] == 0
    assert merged["arguments"] == {"first": 1, "second": 2}


def test_google_tool_call_merge_keeps_index_when_late_id_also_has_index():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"index": 0, "name": "lookup", "arguments": {"first": 1}},
    )
    key, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "index": 0, "name": "lookup", "arguments": {"second": 2}},
    )

    assert key == ("index", 0)
    assert list(accumulator) == [("index", 0)]
    assert merged["id"] == "call-1"
    assert merged["arguments"] == {"first": 1, "second": 2}


def test_google_tool_call_merge_prefers_index_when_later_fragment_has_index():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"id": "call-1", "name": "lookup", "arguments": {"first": 1}},
    )
    key, merged = GoogleProvider._merge_tool_call(
        accumulator,
        {"index": 0, "name": "lookup", "arguments": {"second": 2}},
    )

    assert key == ("index", 0)
    assert len(accumulator) == 1
    assert merged["id"] == "call-1"
    assert merged["index"] == 0
    assert merged["arguments"] == {"first": 1, "second": 2}


def test_google_tool_call_merge_rejects_ambiguous_unidentified_parallel_calls():
    accumulator = {}
    GoogleProvider._merge_tool_call(
        accumulator,
        {"name": "lookup", "arguments": {"first": 1}},
    )

    with pytest.raises(RuntimeError, match="缺少稳定 id/index"):
        GoogleProvider._merge_tool_call(
            accumulator,
            {"name": "lookup", "arguments": {"first": 2}},
        )


def test_google_sync_stream_merges_tool_call_when_id_arrives_after_first_chunk():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    response = iter(
        [
            chunk(SimpleNamespace(name="lookup", args={"first": 1})),
            chunk(SimpleNamespace(id="call-1", args={"second": 2})),
            chunk(finish_reason="STOP"),
        ]
    )
    provider._get_client = lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_kwargs: response)
    )

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert len(completed) == 1
    assert [event.type for event in events][-2:] == ["tool_call_completed", "finish"]
    assert completed[0].response_id == "response-1"
    assert completed[0].raw.response_id == "response-1"
    assert completed[0].tool_call == {
        "type": "function",
        "arguments": {"first": 1, "second": 2},
        "name": "lookup",
        "id": "call-1",
    }


def test_google_sync_stream_defers_unidentified_tool_completion_until_finish():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    response = iter(
        [
            chunk(SimpleNamespace(name="lookup", args={"id": 1}, will_continue=False)),
            chunk(finish_reason="STOP"),
        ]
    )
    provider._get_client = lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_kwargs: response)
    )

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert len(completed) == 1
    assert [event.type for event in events][-2:] == ["tool_call_completed", "finish"]


def test_google_sync_stream_does_not_duplicate_string_tool_arguments_on_completion():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    function_call = SimpleNamespace(
        id="call-1",
        name="lookup",
        args='{"id":1}',
        will_continue=False,
    )
    chunk = SimpleNamespace(
        response_id="response-1",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)]),
                finish_reason="STOP",
            )
        ],
    )
    provider._get_client = lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_kwargs: iter([chunk]))
    )

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert len(completed) == 1
    assert completed[0].tool_call["arguments"] == '{"id":1}'


def test_google_async_stream_merges_tool_call_when_id_arrives_after_first_chunk():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    async def chunks():
        yield chunk(SimpleNamespace(name="lookup", args={"first": 1}))
        yield chunk(SimpleNamespace(id="call-1", args={"second": 2}))
        yield chunk(finish_reason="STOP")

    class AsyncModels:
        def generate_content_stream(self, **_kwargs):
            return chunks()

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=AsyncModels()))

    events = asyncio.run(_collect_google_async_events(provider))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert len(completed) == 1
    assert [event.type for event in events][-2:] == ["tool_call_completed", "finish"]
    assert completed[0].response_id == "response-1"
    assert completed[0].raw is not None
    assert completed[0].tool_call == {
        "type": "function",
        "arguments": {"first": 1, "second": 2},
        "name": "lookup",
        "id": "call-1",
    }


def test_google_async_stream_does_not_duplicate_string_tool_arguments_on_completion():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    function_call = SimpleNamespace(
        id="call-1",
        name="lookup",
        args='{"id":1}',
        will_continue=False,
    )
    chunk = SimpleNamespace(
        response_id="response-1",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)]),
                finish_reason="STOP",
            )
        ],
    )

    async def chunks():
        yield chunk

    class AsyncModels:
        def generate_content_stream(self, **_kwargs):
            return chunks()

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=AsyncModels()))

    events = asyncio.run(_collect_google_async_events(provider))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert len(completed) == 1
    assert completed[0].tool_call["arguments"] == '{"id":1}'


def test_google_sync_stream_keeps_parallel_tool_calls_separate_without_index():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    response = iter(
        [
            chunk(SimpleNamespace(id="call-1", name="lookup", args={"first": 1})),
            chunk(SimpleNamespace(id="call-2", name="search", args={"first": 2})),
            chunk(SimpleNamespace(id="call-1", args={"second": 1})),
            chunk(SimpleNamespace(id="call-2", args={"second": 2})),
            chunk(finish_reason="STOP"),
        ]
    )
    provider._get_client = lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_kwargs: response)
    )

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert [event.type for event in events][-3:] == [
        "tool_call_completed",
        "tool_call_completed",
        "finish",
    ]
    assert {event.tool_call["id"] for event in completed} == {"call-1", "call-2"}
    assert {event.tool_call["name"] for event in completed} == {"lookup", "search"}
    assert {event.tool_call["arguments"]["second"] for event in completed} == {1, 2}


def test_google_async_stream_keeps_parallel_tool_calls_separate_without_index():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    async def chunks():
        yield chunk(SimpleNamespace(id="call-1", name="lookup", args={"first": 1}))
        yield chunk(SimpleNamespace(id="call-2", name="search", args={"first": 2}))
        yield chunk(SimpleNamespace(id="call-1", args={"second": 1}))
        yield chunk(SimpleNamespace(id="call-2", args={"second": 2}))
        yield chunk(finish_reason="STOP")

    class AsyncModels:
        def generate_content_stream(self, **_kwargs):
            return chunks()

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=AsyncModels()))

    events = asyncio.run(_collect_google_async_events(provider))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert [event.type for event in events][-3:] == [
        "tool_call_completed",
        "tool_call_completed",
        "finish",
    ]
    assert {event.tool_call["id"] for event in completed} == {"call-1", "call-2"}
    assert {event.tool_call["name"] for event in completed} == {"lookup", "search"}
    assert {event.tool_call["arguments"]["second"] for event in completed} == {1, 2}


def test_google_sync_stream_migrates_parallel_unidentified_calls_by_name():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    def chunk(function_call=None, finish_reason=None):
        content = (
            SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            if function_call
            else None
        )
        return SimpleNamespace(
            response_id="response-1",
            candidates=[SimpleNamespace(content=content, finish_reason=finish_reason)],
        )

    response = iter(
        [
            chunk(SimpleNamespace(name="lookup", args={"first": 1})),
            chunk(SimpleNamespace(name="search", args={"first": 2})),
            chunk(SimpleNamespace(id="call-1", name="lookup", args={"second": 1})),
            chunk(SimpleNamespace(id="call-2", name="search", args={"second": 2})),
            chunk(finish_reason="STOP"),
        ]
    )
    provider._get_client = lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_kwargs: response)
    )

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]

    assert {event.tool_call["id"] for event in completed} == {"call-1", "call-2"}
    assert {event.tool_call["name"] for event in completed} == {"lookup", "search"}
    assert {event.tool_call["arguments"]["first"] for event in completed} == {1, 2}
    assert {event.tool_call["arguments"]["second"] for event in completed} == {1, 2}


def test_google_contents_links_function_call_content_to_followup_tool_result():
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"id": 1},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"value": 2}',
            },
        ],
        None,
        None,
    )

    function_call = contents[0].parts[0].function_call
    function_response = contents[1].parts[0].function_response
    assert function_call.id == "call-1"
    assert function_call.name == "lookup"
    assert function_response.id == "call-1"
    assert function_response.name == "lookup"
    assert function_response.response == {"value": 2}


@pytest.mark.parametrize("value", ["123abc", "-", "01", "-01"])
def test_google_tool_result_numeric_prefix_that_is_not_json_stays_text(value):
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "function_call", "id": "call-1", "name": "lookup", "arguments": {}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": value},
        ],
        None,
        None,
    )

    assert contents[1].parts[0].function_response.response == {"output": value}


@pytest.mark.parametrize("value", ["0", "-1", "1.25", "2e3"])
def test_google_tool_result_complete_json_number_is_decoded(value):
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "function_call", "id": "call-1", "name": "lookup", "arguments": {}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": value},
        ],
        None,
        None,
    )

    assert contents[1].parts[0].function_response.response == {"output": json.loads(value)}


def test_anthropic_stream_events_cover_text_thinking_tool_usage_and_finish():
    events = [
        AnthropicProvider._stream_event(
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="答")
            ),
            "msg-1",
        ),
        AnthropicProvider._stream_event(
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="想"),
            ),
            "msg-1",
        ),
        AnthropicProvider._stream_event(
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"id":1}'),
            ),
            "msg-1",
        ),
        AnthropicProvider._stream_event(
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(input_tokens=2, output_tokens=3),
            ),
            "msg-1",
        ),
    ]

    assert [event.type for event in events if event] == [
        "text_delta",
        "reasoning_delta",
        "tool_call_delta",
        "finish",
    ]
    assert events[3].finish_reason == "tool_use"
    assert events[3].usage["output_tokens"] == 3


def test_ark_stream_events_cover_chat_and_responses_terminal_metadata():
    chat = VolcengineProvider._chat_stream_event(
        SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="答", reasoning_content="想", tool_calls=None),
                    finish_reason=None,
                )
            ],
        )
    )
    provider = object.__new__(VolcengineProvider)
    response = provider._responses_stream_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="ark-1",
                status="completed",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        )
    )

    assert chat.type == "text_delta"
    assert chat.text == "答"
    assert response.type == "finish"
    assert response.finish_reason == "completed"
    assert (
        response.usage["total_tokens"]
        if "total_tokens" in response.usage
        else response.usage["input_tokens"]
    ) == 1


def test_ark_invoke_non_stream_rejects_failed_response_status():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_kwargs):
            return SimpleNamespace(status="failed", error=SimpleNamespace(message="ark failed"))

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match="ark failed"):
        list(provider.invoke("hi", stream=False))


def test_ark_ainvoke_non_stream_rejects_failed_response_status():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                status="failed", error=SimpleNamespace(message="ark async failed")
            )

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        return [item async for item in provider.ainvoke("hi", stream=False)]

    with pytest.raises(RuntimeError, match="ark async failed"):
        asyncio.run(collect())


def test_ark_non_stream_result_preserves_chat_reasoning_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="答复", reasoning_content="思考过程", tool_calls=[]
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    result = VolcengineProvider._extract_result(response)

    assert result.text == "答复"
    assert result.reasoning == "思考过程"


def test_ark_responses_result_does_not_replace_answer_with_reasoning_text():
    """正文与 reasoning 必须分开取。

    此前这里用 ``SimpleNamespace(output_text=...)`` 伪造响应，但 Ark 的
    ``Response`` 没有该字段——夹具替被测代码补上了它，掩盖了「非流式入口
    读不到正文」的缺陷。改用真实 SDK 模型构造。
    """
    response = _ark_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "最终答案", "annotations": []}],
            },
            {
                "type": "reasoning",
                "id": "r1",
                "summary": [{"type": "summary_text", "text": "思考过程"}],
            },
        ]
    )

    result = VolcengineProvider._extract_result(response)

    assert result.text == "最终答案"
    assert result.reasoning == "思考过程"


def test_ark_responses_error_and_cancelled_events_are_explicit():
    provider = object.__new__(VolcengineProvider)

    with pytest.raises(RuntimeError, match="rate limit"):
        provider._responses_stream_event(
            SimpleNamespace(type="response.error", error=SimpleNamespace(message="rate limit"))
        )

    with pytest.raises(RuntimeError, match="取消|cancelled"):
        provider._responses_stream_event(
            SimpleNamespace(type="response.cancelled", response=SimpleNamespace(status="cancelled"))
        )


def test_ark_responses_custom_tool_stream_uses_output_item_metadata():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="custom_tool_call",
                            id="item-1",
                            call_id="call-1",
                            name="run_command",
                            input="",
                        ),
                    ),
                    SimpleNamespace(
                        type="response.custom_tool_call_input.delta",
                        item_id="item-1",
                        output_index=0,
                        delta="echo hello",
                    ),
                    SimpleNamespace(
                        type="response.custom_tool_call_input.done",
                        item_id="item-1",
                        output_index=0,
                        input="echo hello",
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == [
        "tool_call_delta",
        "tool_call_completed",
        "finish",
    ]
    assert events[1].tool_call["type"] == "custom_tool_call"
    assert events[1].tool_call["name"] == "run_command"


def test_anthropic_rejects_seed_in_complete_request():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    with pytest.raises(ValueError, match="seed"):
        provider.complete(CompletionRequest(prompt="hi", seed=7))
    with pytest.raises(ValueError, match="seed"):
        provider.parse(dict, CompletionRequest(prompt="hi", seed=7))
    with pytest.raises(ValueError, match="seed"):
        asyncio.run(provider.async_parse(dict, CompletionRequest(prompt="hi", seed=7)))


def test_sdk_provider_close_and_aclose_release_clients():
    class SyncClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class AsyncClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    providers = [
        OpenAICompatibleProvider.__new__(OpenAICompatibleProvider),
        VolcengineProvider.__new__(VolcengineProvider),
        AnthropicProvider.__new__(AnthropicProvider),
    ]
    for provider in providers:
        sync = SyncClient()
        async_client = AsyncClient()
        provider._client = sync
        provider._aclient = async_client
        provider.close()
        assert sync.closed is True
        assert provider._client is None
        assert async_client.closed is True
        assert provider._aclient is None
        asyncio.run(provider.aclose())
        assert async_client.closed is True
        assert provider._aclient is None


def test_anthropic_sync_close_requires_aclose_inside_running_event_loop():
    class AsyncClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    provider = AnthropicProvider.__new__(AnthropicProvider)
    async_client = AsyncClient()
    provider._client = None
    provider._aclient = async_client

    async def run() -> None:
        with pytest.raises(RuntimeError, match="需要异步关闭.*aclose"):
            provider.close()
        assert provider._aclient is async_client
        await provider.aclose()

    asyncio.run(run())
    assert async_client.closed is True
    assert provider._aclient is None


def test_sdk_provider_aclose_closes_shared_client_once():
    class SharedClient:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    for provider_type in (
        OpenAICompatibleProvider,
        VolcengineProvider,
        AnthropicProvider,
    ):
        provider = provider_type.__new__(provider_type)
        shared = SharedClient()
        provider._client = shared
        provider._aclient = shared

        asyncio.run(provider.aclose())

        assert shared.close_calls == 1
        assert provider._client is None
        assert provider._aclient is None


def test_google_close_releases_sync_client_and_async_close_releases_aio_client():
    class SyncClient:
        def __init__(self):
            self.closed = False
            self.aio = self

        def close(self):
            self.closed = True

        async def aclose(self):
            self.closed = True

    provider = GoogleProvider.__new__(GoogleProvider)
    client = SyncClient()
    provider._client = client

    provider.close()
    assert client.closed is True
    assert provider._client is None

    provider._client = client
    asyncio.run(provider.aclose())
    assert client.closed is True
    assert provider._client is None


def test_google_vertex_gemini_embedding_uses_single_content_path():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-embedding-001"

    assert provider._vertex_embed_content_only() is True


def test_anthropic_complete_extracts_tool_blocks_and_usage():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="answer"),
            SimpleNamespace(type="tool_use", id="call-1", name="lookup", input={"id": 1}),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=5),
        stop_reason="tool_use",
    )
    provider._get_client = lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_: response)
    )
    result = provider.complete(CompletionRequest(prompt="hi", stream=False))
    assert result.text == "answer"
    assert result.tool_calls[0]["arguments"] == {"id": 1}
    assert result.usage == {"input_tokens": 4, "output_tokens": 5}


def test_anthropic_user_maps_to_sdk_user_profile_header():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    params = provider._build_message_params("hi", user="profile-1")
    assert params["user_profile_id"] == "profile-1"
    assert "user_id" not in params.get("metadata", {})


def test_openai_parse_and_compact_facades_forward_native_methods():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    calls = {}

    class Completions:
        def parse(self, **kwargs):
            calls["parse"] = kwargs
            return "parsed"

    class Responses:
        def compact(self, **kwargs):
            calls["compact"] = kwargs
            return "compacted"

    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()), responses=Responses()
    )
    assert provider.parse(dict, CompletionRequest(prompt="hi")) == "parsed"
    assert calls["parse"]["model"] == "demo"
    assert "stream" not in calls["parse"]
    provider._protocol = "responses"
    assert (
        provider.compact_responses(input="hello", previous_response_id="resp-1", timeout=2)
        == "compacted"
    )
    assert calls["compact"] == {
        "model": "demo",
        "input": "hello",
        "previous_response_id": "resp-1",
        "timeout": 2,
    }


def test_openai_compatible_responses_parse_and_compact_require_verified_server_capability():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    provider._options = {}

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.parse(dict, CompletionRequest(prompt="hi"))
    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.compact_responses(input="hello")
    with pytest.raises(ValueError, match="未验证 Responses"):
        asyncio.run(provider.async_parse(dict, CompletionRequest(prompt="hi")))
    with pytest.raises(ValueError, match="未验证 Responses"):
        asyncio.run(provider.async_compact_responses(input="hello"))


@pytest.mark.parametrize("interval", [0, -1, "0"])
def test_openai_responses_background_poll_interval_must_be_positive(interval):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._options = {"background_poll_interval": interval}
    with pytest.raises(ValueError, match="background_poll_interval.*正数"):
        provider._background_poll_config()


def test_openai_custom_base_url_requires_verified_responses_resource_capability():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://gateway.example/v1"
    provider._server_verified_protocols = frozenset()

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.retrieve_response("resp-1")


def test_openai_custom_base_url_blocks_regular_responses_requests():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://gateway.example/v1"
    provider._server_verified_protocols = frozenset()

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.complete(CompletionRequest(prompt="hi"))
    with pytest.raises(ValueError, match="未验证 Responses"):
        list(provider.invoke("hi"))
    with pytest.raises(ValueError, match="未验证 Responses"):
        list(provider.stream_events(CompletionRequest(prompt="hi")))


def test_openai_custom_base_url_blocks_responses_batch_requests():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://gateway.example/v1"
    provider._server_verified_protocols = frozenset()

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.create_batch("file-1")


def test_openai_official_base_url_allows_responses_resource_contract():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://api.openai.com/v1/"
    provider._server_verified_protocols = frozenset()
    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(retrieve=lambda response_id, **kwargs: (response_id, kwargs))
    )

    assert provider.retrieve_response("resp-1", include=["output"]) == (
        "resp-1",
        {"include": ["output"]},
    )


def test_openai_eval_output_item_retrieve_forwards_sync_and_async_resources():
    provider = object.__new__(OpenAIProvider)
    provider._model_name = "demo"
    calls = {}

    class OutputItems:
        def retrieve(self, output_item_id, **kwargs):
            calls["sync"] = (output_item_id, kwargs)
            return "retrieved"

    provider._get_client = lambda: SimpleNamespace(
        evals=SimpleNamespace(runs=SimpleNamespace(output_items=OutputItems()))
    )
    assert (
        provider.retrieve_eval_run_output_item("eval-1", "run-1", "item-1", limit=2) == "retrieved"
    )
    assert calls["sync"] == ("item-1", {"eval_id": "eval-1", "run_id": "run-1", "limit": 2})
    assert (
        provider.resources.retrieve_eval_run_output_item("eval-1", "run-1", "item-1") == "retrieved"
    )

    class AsyncOutputItems:
        async def retrieve(self, output_item_id, **kwargs):
            calls["async"] = (output_item_id, kwargs)
            return "async-retrieved"

    provider._get_aclient = lambda: SimpleNamespace(
        evals=SimpleNamespace(runs=SimpleNamespace(output_items=AsyncOutputItems()))
    )
    assert (
        asyncio.run(
            provider.async_retrieve_eval_run_output_item("eval-1", "run-1", "item-1", limit=3)
        )
        == "async-retrieved"
    )
    assert calls["async"] == ("item-1", {"eval_id": "eval-1", "run_id": "run-1", "limit": 3})
    assert (
        asyncio.run(
            provider.async_resources.retrieve_eval_run_output_item("eval-1", "run-1", "item-1")
        )
        == "async-retrieved"
    )


def test_resource_facade_exposes_native_sdk_clients():
    provider = object.__new__(OpenAICompatibleProvider)
    sync_client = object()
    async_client = object()
    provider._get_client = lambda: sync_client
    provider._get_aclient = lambda: async_client

    assert provider.resources.native is sync_client
    assert provider.async_resources.native is sync_client
    assert provider.async_resources.async_native is async_client


def test_official_resource_facade_keeps_missing_dynamic_attributes_normal():
    provider = object.__new__(OpenAIProvider)
    provider._get_client = lambda: SimpleNamespace(existing="value")
    provider._get_aclient = lambda: SimpleNamespace(existing="async-value")

    resources = provider.resources
    assert resources.existing == "value"
    assert not hasattr(resources, "missing_resource")
    assert not hasattr(resources, "__missing_resource__")


def test_openai_resources_expose_new_official_top_level_resources():
    provider = object.__new__(OpenAIProvider)
    sync_resources = SimpleNamespace(
        skills="skills",
        realtime="realtime",
        webhooks="webhooks",
        admin="admin",
        content_provenance_checks="provenance",
    )
    async_resources = SimpleNamespace(
        skills="async-skills",
        realtime="async-realtime",
        webhooks="async-webhooks",
        admin="async-admin",
        content_provenance_checks="async-provenance",
    )
    provider._get_client = lambda: sync_resources
    provider._get_aclient = lambda: async_resources

    assert provider.resources.skills == "skills"
    assert provider.resources.realtime == "realtime"
    assert provider.resources.webhooks == "webhooks"
    assert provider.resources.admin == "admin"
    assert provider.resources.content_provenance_checks == "provenance"
    assert provider.async_resources.skills == "async-skills"
    assert provider.async_resources.realtime == "async-realtime"
    assert provider.async_resources.webhooks == "async-webhooks"
    assert provider.async_resources.admin == "async-admin"
    assert provider.async_resources.content_provenance_checks == "async-provenance"


def test_openai_capabilities_include_current_top_level_resource_families():
    capabilities = ModelProviderFactory.capabilities("openai")
    assert {
        "skills",
        "realtime",
        "webhooks",
        "admin",
        "content_provenance_checks",
    }.issubset(capabilities)


def test_factory_capability_report_includes_sdk_and_protocol_evidence():
    report = ModelProviderFactory.capability_report("openai")

    assert report["provider"] == "openai"
    assert report["resolved_provider"] == "openai"
    assert report["role"] == "llm"
    assert report["integration"] == "native_sdk"
    assert report["sdk"] == {"package": "openai", "version": report["sdk"]["version"]}
    assert report["sdk"]["version"]
    assert report["protocol_status"]["responses"] == {
        "adapter_supported": True,
        "server_verified": False,
    }


def test_factory_capabilities_are_queryable_by_role_and_alias():
    embedding = ModelProviderFactory.capability_report("qwen", "embedding")
    assert embedding["role"] == "embedding"
    assert embedding["integration"] == "openai_compatible_sdk"
    assert embedding["sdk"]["package"] == "openai"
    assert "responses" not in embedding["protocol_status"]

    rerank = ModelProviderFactory.capability_report("siliconflow", "rerank")
    assert rerank["resolved_provider"] == "siliconflow_rerank"
    assert rerank["integration"] == "http_json"
    assert rerank["sdk"]["package"] == "httpx"
    assert rerank["protocols"] == []


def test_factory_capability_report_rejects_unknown_role():
    with pytest.raises(ValueError, match="不支持的模型角色"):
        ModelProviderFactory.capability_report("openai", "moderation")


def test_official_protocol_status_does_not_claim_unverified_custom_endpoints():
    assert ModelProviderFactory.protocol_status("openai", "chat_completions") == {
        "adapter_supported": True,
        "server_verified": False,
    }
    assert ModelProviderFactory.protocol_status("openai", "responses") == {
        "adapter_supported": True,
        "server_verified": False,
    }
    assert ModelProviderFactory.protocol_status("volcengine", "ark") == {
        "adapter_supported": True,
        "server_verified": False,
    }
    assert ModelProviderFactory.protocol_status("volcengine", "responses") == {
        "adapter_supported": True,
        "server_verified": False,
    }


def test_factory_rejects_server_verified_protocol_alias_not_supported_by_provider():
    with pytest.raises(ValueError, match="不适用于当前提供商"):
        ModelProviderFactory.capability_report(
            "qwen", options={"server_verified_protocols": ["gemini"]}
        )
    with pytest.raises(ValueError, match="不适用于当前提供商"):
        ModelProviderFactory.protocol_status(
            "qwen", "generate_content", options={"server_verified_protocols": ["gemini"]}
        )


def test_google_auth_tokens_facade_forwards_sync_and_async_clients():
    provider = object.__new__(GoogleProvider)
    calls = {}

    class Tokens:
        def create(self, **kwargs):
            calls["sync"] = kwargs
            return "token"

    class AsyncTokens:
        async def create(self, **kwargs):
            calls["async"] = kwargs
            return "async-token"

    sync_tokens = Tokens()
    async_tokens = AsyncTokens()
    provider._get_client = lambda: SimpleNamespace(
        auth_tokens=sync_tokens, aio=SimpleNamespace(auth_tokens=async_tokens)
    )

    # Facade 资源经过安全代理；显式 native 出口仍保留 SDK 原生身份。
    assert provider.resources.auth_tokens._value is sync_tokens
    assert provider.async_resources.auth_tokens._value is async_tokens
    assert provider.resources.native.auth_tokens is sync_tokens
    assert provider.async_resources.async_native.auth_tokens is async_tokens
    assert provider.resources.create_auth_token(config={}) == "token"
    assert asyncio.run(provider.async_resources.create_auth_token(config={})) == "async-token"
    assert calls == {"sync": {"config": {}}, "async": {"config": {}}}

    with pytest.raises(ValueError, match="凭证"):
        provider.resources.auth_tokens.create(extra_headers={"X-API-Key": "secret"})


def test_google_register_files_facade_allows_auth_business_parameter_sync_and_async():
    provider = object.__new__(GoogleProvider)
    auth = AnonymousCredentials()
    calls = {}

    class Files:
        def register_files(self, **kwargs):
            calls["sync"] = kwargs
            return "registered"

    class AsyncFiles:
        async def register_files(self, **kwargs):
            calls["async"] = kwargs
            return "async-registered"

    provider._get_client = lambda: SimpleNamespace(
        files=Files(), aio=SimpleNamespace(files=AsyncFiles())
    )

    assert provider.resources.register_files(auth=auth, uris=["gs://bucket/file"]) == "registered"
    assert (
        asyncio.run(provider.async_resources.register_files(auth=auth, uris=["gs://bucket/file"]))
        == "async-registered"
    )
    assert calls == {
        "sync": {"auth": auth, "uris": ["gs://bucket/file"]},
        "async": {"auth": auth, "uris": ["gs://bucket/file"]},
    }


def test_google_register_files_facade_rejects_non_credentials_auth_before_sdk_call():
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: pytest.fail("非法 auth 不应初始化 Google SDK 客户端")

    with pytest.raises(ValueError, match="Credentials"):
        provider.resources.register_files(
            auth={"api_key": "not-allowed"},
            uris=["gs://bucket/file"],
        )


def test_google_register_files_facade_still_rejects_credential_request_overrides():
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: pytest.fail("敏感请求覆盖不应初始化 Google SDK 客户端")

    with pytest.raises(ValueError, match="凭证"):
        provider.resources.register_files(
            auth=AnonymousCredentials(),
            uris=["gs://bucket/file"],
            extra_headers={"Authorization": "Bearer secret"},
        )


def test_anthropic_beta_resource_facade_forwards_current_sdk_tree():
    provider = object.__new__(AnthropicProvider)
    resources = {
        name: object()
        for name in (
            "agents",
            "deployments",
            "deployment_runs",
            "dreams",
            "environments",
            "files",
            "memory_stores",
            "models",
            "sessions",
            "skills",
            "tunnels",
            "user_profiles",
            "vaults",
            "webhooks",
        )
    }
    provider._get_client = lambda: SimpleNamespace(beta=SimpleNamespace(**resources))
    provider._get_aclient = lambda: SimpleNamespace(beta=SimpleNamespace(**resources))

    for name, value in resources.items():
        assert getattr(provider.resources, f"beta_{name}")._value is value
        assert getattr(provider.async_resources, f"beta_{name}")._value is value


def test_anthropic_direct_resource_calls_reject_sensitive_http_overrides():
    provider = object.__new__(AnthropicProvider)

    class Files:
        def list(self, **_kwargs):
            raise AssertionError("SDK should not be called")

    class Batches:
        def list(self, **_kwargs):
            raise AssertionError("SDK should not be called")

    provider._get_client = lambda: SimpleNamespace(
        beta=SimpleNamespace(files=Files()),
        messages=SimpleNamespace(batches=Batches()),
    )

    with pytest.raises(ValueError, match="请求头/query 不允许包含凭证"):
        provider.list_files(extra_headers={"X-API-Key": "secret"})
    with pytest.raises(ValueError, match="请求头/query 不允许包含凭证"):
        provider.list_batches(extra_query={"access_token": "secret"})


def test_google_resource_model_methods_forward_to_sdk():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    calls = {}

    class Models:
        def delete(self, **kwargs):
            calls["delete"] = kwargs
            return "deleted"

        def update(self, **kwargs):
            calls["update"] = kwargs
            return "updated"

    provider._get_client = lambda: SimpleNamespace(models=Models())
    assert provider.resources.delete_model("models/demo") == "deleted"
    assert (
        provider.resources.update_model("models/demo", config={"display_name": "demo"}) == "updated"
    )
    assert calls["delete"] == {"model": "models/demo"}
    assert calls["update"] == {"model": "models/demo", "config": {"display_name": "demo"}}


def test_google_operation_and_token_count_facades_match_current_sdk_signature():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    calls = {}

    class Operations:
        def get(self, operation, **kwargs):
            calls["operation"] = (operation, kwargs)
            return "operation"

    class Models:
        def count_tokens(self, **kwargs):
            calls["count"] = kwargs
            return SimpleNamespace(total_tokens=6)

    provider._get_client = lambda: SimpleNamespace(operations=Operations(), models=Models())
    assert provider.resources.get_operation("operations/1", config={}) == "operation"
    assert provider.resources.count_tokens(CompletionRequest(prompt="hi")) == 6
    assert calls["operation"] == ("operations/1", {"config": {}})
    assert calls["count"]["model"] == "gemini"


def test_google_async_operation_uses_sdk_config_keyword():
    provider = object.__new__(GoogleProvider)
    calls = {}

    class Operations:
        def get(self, operation, **kwargs):
            calls["operation"] = (operation, kwargs)
            return "operation"

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(operations=Operations()))

    assert asyncio.run(provider.async_get_operation("operations/2", timeout=2)) == "operation"
    operation, kwargs = calls["operation"]
    assert operation == "operations/2"
    assert kwargs["config"].http_options.timeout == 2000
    assert set(kwargs) == {"config"}


def test_google_chat_creation_wraps_generation_options_in_config():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = {}

    class Chats:
        def create(self, **kwargs):
            calls["sync"] = kwargs
            return "chat"

    provider._get_client = lambda: SimpleNamespace(chats=Chats())

    assert (
        provider.create_chat(
            history=[], temperature=0.2, max_output_tokens=64, extra_headers={"X-Trace": "1"}
        )
        == "chat"
    )
    assert calls["sync"]["model"] == "gemini-default"
    assert calls["sync"]["history"] == []
    assert calls["sync"]["config"].temperature == 0.2
    assert calls["sync"]["config"].max_output_tokens == 64
    assert calls["sync"]["config"].http_options.headers == {"X-Trace": "1"}
    assert set(calls["sync"]) == {"model", "history", "config"}


def test_google_async_chat_creation_wraps_generation_options_in_config():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = {}

    class Chats:
        def create(self, **kwargs):
            calls["async"] = kwargs
            return ("chat", kwargs)

    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace(chats=Chats()))

    result = asyncio.run(
        provider.async_create_chat(history=[], top_p=0.8, response_mime_type="application/json")
    )
    assert result[0] == "chat"
    assert calls["async"]["config"].top_p == 0.8
    assert calls["async"]["config"].response_mime_type == "application/json"
    assert set(calls["async"]) == {"model", "history", "config"}


def test_openai_compatible_capabilities_do_not_claim_official_resources():
    assert "files" not in OpenAICompatibleProvider.capabilities
    assert "batches" not in OpenAICompatibleProvider.capabilities
    assert "vector_stores" not in OpenAICompatibleProvider.capabilities
    assert "files" in ModelProviderFactory.capabilities("openai")
    assert "files" not in ModelProviderFactory.capabilities("qwen")


def test_openai_compatible_resources_only_expose_native_client():
    compatible = object.__new__(OpenAICompatibleProvider)
    compatible._get_client = lambda: object()
    compatible._get_aclient = lambda: object()
    assert hasattr(compatible.resources, "native")
    assert not hasattr(compatible.resources, "upload_file")

    official = object.__new__(OpenAIProvider)
    official._get_client = lambda: object()
    official._get_aclient = lambda: object()
    assert hasattr(official.resources, "upload_file")


def test_openai_compatible_direct_official_resources_require_declared_capability():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "qwen"
    provider._options = {}

    with pytest.raises(NotImplementedError, match="files"):
        provider.list_files()
    with pytest.raises(NotImplementedError, match="vector_stores"):
        provider.create_vector_store(name="demo")
    with pytest.raises(NotImplementedError, match="fine_tuning"):
        provider.list_fine_tuning_jobs()
    with pytest.raises(NotImplementedError, match="audio"):
        provider.text_to_speech("hello", "tts-1", "alloy")


def test_google_provider_missing_experimental_resources_is_explicit():
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="(?i)agents"):
        provider.create_agent(name="demo")
    with pytest.raises(NotImplementedError, match="(?i)auth token"):
        provider.create_auth_token()
    with pytest.raises(AttributeError, match="(?i)agents"):
        _missing_agents = provider.resources.agents


def test_factory_capabilities_follow_role_specific_provider_contracts():
    assert "embedding" not in ModelProviderFactory.capabilities("deepseek", "llm")
    assert ModelProviderFactory.capabilities("qwen", "embedding") == frozenset({"embedding"})
    assert ModelProviderFactory.capabilities("siliconflow", "rerank") == frozenset({"rerank"})


def test_factory_protocol_status_infers_non_llm_role_for_diagnostics():
    assert ModelProviderFactory.protocol_status("local-hash", "responses") == {
        "adapter_supported": False,
        "server_verified": False,
    }
    assert ModelProviderFactory.protocol_status("siliconflow", "responses", role="rerank") == {
        "adapter_supported": False,
        "server_verified": False,
    }


def test_protocol_status_separates_adapter_support_from_server_verification():
    assert ModelProviderFactory.protocol_status("qwen", "responses") == {
        "adapter_supported": True,
        "server_verified": False,
    }


@pytest.mark.parametrize("configured", [1, {}, ["responses", 1], ["responses", ""]])
def test_openai_verified_protocol_configuration_rejects_invalid_types(configured):
    provider = object.__new__(OpenAICompatibleProvider)
    with pytest.raises(ValueError, match="server_verified_protocols"):
        provider._normalize_verified_protocols(configured)


def test_openai_responses_token_count_forwards_full_request_extensions():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    calls = []

    class TokenCounter:
        def count(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(input_tokens=11)

    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(input_tokens=TokenCounter())
    )
    request = CompletionRequest(
        prompt="hi",
        response_format={"type": "json_object"},
        parallel_tool_calls=True,
        reasoning={"effort": "low"},
        truncation="auto",
        extra_body={"vendor_flag": True},
        extra_headers={"X-Trace": "trace-1"},
        extra_query={"tenant": "demo"},
        timeout=3,
    )

    assert provider.count_input_tokens(request) == 11
    assert calls == [
        {
            "model": "demo",
            "input": "hi",
            "instructions": "You are a helpful assistant.",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_object"}},
            "truncation": "auto",
            "extra_headers": {"X-Trace": "trace-1"},
            "extra_query": {"tenant": "demo"},
            "extra_body": {"vendor_flag": True},
            "timeout": 3,
        }
    ]


def test_openai_responses_token_count_converts_chat_tool_choice():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}

    calls = []

    class TokenCounter:
        def count(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(input_tokens=3)

    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(input_tokens=TokenCounter())
    )

    request = CompletionRequest(
        prompt="hi",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    assert provider.count_input_tokens(request) == 3
    assert calls[0]["tool_choice"] == {"type": "function", "name": "lookup"}


def test_openai_responses_token_count_rejects_generation_only_fields():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    with pytest.raises(ValueError, match="max_tokens"):
        provider.count_input_tokens(CompletionRequest(prompt="hi", max_tokens=16))


def test_openai_compatible_responses_resources_require_verified_server_capability():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.retrieve_response("resp-1")


def test_openai_compatible_verified_responses_resource_is_forwarded():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "qwen"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(retrieve=lambda response_id, **kwargs: (response_id, kwargs))
    )

    assert provider.retrieve_response("resp-1", include=["output"]) == (
        "resp-1",
        {"include": ["output"]},
    )


def test_anthropic_extra_body_and_request_extensions_keep_sdk_names():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    params = provider._build_message_params(
        prompt="hi",
        extra_body={"vendor_flag": True},
        extra_headers={"X-Trace": "trace-1"},
        extra_query={"tenant": "demo"},
    )

    assert params["extra_body"] == {"vendor_flag": True}
    assert params["extra_headers"] == {"X-Trace": "trace-1"}
    assert params["extra_query"] == {"tenant": "demo"}
    assert "vendor_flag" not in params


def test_anthropic_request_builder_rejects_sensitive_http_overrides():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="请求头/query"):
        provider._build_message_params(
            prompt="hi",
            extra_headers={"Authorization": "Bearer secret"},
        )
    with pytest.raises(ValueError, match="请求头/query"):
        provider._build_message_params(
            prompt="hi",
            extra_query={"access_token": "secret"},
        )


def test_openai_compatible_server_capability_option_is_not_sent_as_body():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._provider = "qwen"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._options = {"server_verified_protocols": ["responses"], "top_k": 20}

    request = provider._build_responses_request(prompt="hi", stream=False)

    assert request["extra_body"] == {"top_k": 20}
    assert "server_verified_protocols" not in request


def test_openai_compatible_responses_model_stop_and_seed_use_extra_body():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "responses"
    provider._provider = "qwen"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._options = {
        "server_verified_protocols": ["responses"],
        "stop": ["END"],
        "seed": 11,
    }

    request = provider._build_responses_request(prompt="hi", stream=False)

    assert request["extra_body"] == {"stop": ["END"], "seed": 11}
    assert "stop" not in request
    assert "seed" not in request


def test_anthropic_client_options_are_not_forwarded_to_message_request():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {
        "base_url": "https://proxy.invalid",
        "timeout": 3,
        "max_tokens": 8192,
        "top_p": 0.8,
    }
    params = provider._build_message_params(prompt="hi")
    assert params["max_tokens"] == 8192
    assert params["extra_body"]["top_p"] == 0.8
    assert "top_p" not in params
    assert "base_url" not in params
    assert "timeout" not in params
    assert provider._build_message_params(prompt="hi", timeout=4)["timeout"] == 4


def test_anthropic_sampling_fields_follow_installed_sdk_request_shape():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    params = provider._build_message_params(
        prompt="hi",
        temperature=0.2,
        top_p=0.8,
        top_k=32,
    )

    assert params["extra_body"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 32,
    }
    assert not {"temperature", "top_p", "top_k"}.intersection(params)


def test_anthropic_extra_body_sampling_conflicts_are_explicit():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="temperature.*extra_body"):
        provider._build_message_params(
            prompt="hi",
            temperature=0.2,
            extra_body={"temperature": 0.4},
        )


def test_anthropic_beta_tool_runner_rejects_unsupported_compaction_control():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="compaction_control"):
        provider.beta_tool_runner([], prompt="hi", compaction_control={"enabled": True})


def test_anthropic_output_format_follows_create_and_stream_sdk_shapes():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="create 不支持 output_format"):
        provider._build_message_params(prompt="hi", output_format=dict, stream=False)

    params = provider._build_message_params(prompt="hi", output_format=dict, stream=True)
    assert params["output_format"] is dict


def test_anthropic_beta_create_rejects_output_format():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="create 不支持 output_format"):
        provider.beta_create(CompletionRequest(prompt="hi", output_format={"type": "json_schema"}))


def test_anthropic_async_beta_create_rejects_output_format_before_sdk_call():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    calls = []

    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return "beta"

    provider._get_aclient = lambda: SimpleNamespace(beta=SimpleNamespace(messages=Messages()))

    with pytest.raises(ValueError, match="create 不支持 output_format"):
        asyncio.run(
            provider.async_beta_create(
                CompletionRequest(prompt="hi", output_format={"type": "json_schema"})
            )
        )
    assert calls == []


def test_anthropic_beta_token_count_rejects_output_format():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="token count 不支持 output_format"):
        provider.beta_count_tokens(
            CompletionRequest(prompt="hi", output_format={"type": "json_schema"})
        )


def test_anthropic_parse_ignores_configured_output_format():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {"output_format": {"type": "json_schema"}}
    calls = {}

    class Messages:
        def parse(self, **kwargs):
            calls.update(kwargs)
            return "parsed"

    provider._get_client = lambda: SimpleNamespace(messages=Messages())
    assert provider.parse(dict, CompletionRequest(prompt="hi")) == "parsed"
    assert calls["output_format"] is dict
    assert calls["output_format"] != provider._options["output_format"]


def test_anthropic_request_keys_match_installed_sdk_create_signatures():
    import anthropic

    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    params = provider._build_message_params(prompt="hi")

    create_keys = set(
        inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
    )
    assert set(params).issubset(create_keys)

    beta_params = provider._build_message_params(prompt="hi", beta=True)
    beta_keys = set(
        inspect.signature(anthropic.Anthropic(api_key="x").beta.messages.create).parameters
    )
    assert set(beta_params).issubset(beta_keys)


def test_anthropic_modern_models_omit_deprecated_sampling_defaults():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = {}

    params = provider._build_message_params(prompt="hi", temperature=0.7)

    assert "temperature" not in params.get("extra_body", {})
    with pytest.raises(ValueError, match="temperature"):
        provider._build_message_params(prompt="hi", temperature=0.2)
    with pytest.raises(ValueError, match="top_k"):
        provider._build_message_params(prompt="hi", top_k=5)


def test_anthropic_modern_models_allow_explicit_legacy_sampling_for_proxies():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = {"allow_deprecated_sampling": True}

    params = provider._build_message_params(
        prompt="hi",
        temperature=0.2,
        top_p=0.8,
        top_k=32,
    )

    assert params["extra_body"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 32,
    }
    assert provider.supports_sampling_option("temperature") is True
    assert provider.supports_sampling_option("top_p") is True
    assert provider.supports_sampling_option("top_k") is True


def test_anthropic_default_max_tokens_scales_with_model_family():
    provider = object.__new__(AnthropicProvider)
    provider._options = {}

    provider._model_name = "claude-3-opus-20240229"
    legacy = AnthropicProvider._build_message_params(provider, prompt="hi")
    provider._model_name = "claude-sonnet-4-6"
    modern = AnthropicProvider._build_message_params(provider, prompt="hi")

    assert legacy["max_tokens"] == 4096
    assert modern["max_tokens"] > legacy["max_tokens"]


@pytest.mark.parametrize(
    ("model_name", "deprecated", "max_tokens"),
    [
        ("claude-sonnet-4-20250514", False, 8192),
        ("claude-sonnet-4-5-20250929", True, 16384),
        ("claude-sonnet-5", True, 16384),
        ("claude-3-7-sonnet-20250219", False, 8192),
    ],
)
def test_anthropic_model_version_parser_ignores_dates_and_supports_major_only(
    model_name, deprecated, max_tokens
):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = model_name
    provider._options = {}

    assert provider._sampling_controls_deprecated(model_name) is deprecated
    assert provider._default_max_tokens() == max_tokens


def test_anthropic_file_resources_prefer_stable_sdk_surface():
    provider = object.__new__(AnthropicProvider)
    calls = []

    class Files:
        def upload(self, **kwargs):
            calls.append(("stable", kwargs))
            return "stable"

    class BetaFiles:
        def upload(self, **_kwargs):
            raise AssertionError("beta files should not be used for stable file operations")

    provider._get_client = lambda: SimpleNamespace(
        files=Files(), beta=SimpleNamespace(files=BetaFiles())
    )

    assert provider.upload_file(b"data") == "stable"
    assert calls == [("stable", {"file": b"data"})]


def test_google_cache_and_batch_resources_supply_default_model():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = []

    class Caches:
        def create(self, **kwargs):
            calls.append(("cache", kwargs))
            return "cache"

    class Batches:
        def create(self, **kwargs):
            calls.append(("batch", kwargs))
            return "batch"

    provider._get_client = lambda: SimpleNamespace(caches=Caches(), batches=Batches())
    assert provider.resources.create_cache(config={}) == "cache"
    assert provider.resources.create_batch([]) == "batch"
    assert calls == [
        ("cache", {"model": "gemini-default", "config": {}}),
        ("batch", {"model": "gemini-default", "src": []}),
    ]


def test_google_async_stream_and_chat_contracts_do_not_double_await():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}

    class AsyncModels:
        async def generate_content_stream(self, **_kwargs):
            async def chunks():
                yield SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])

            return chunks()

    class AsyncChats:
        def create(self, **kwargs):
            return ("chat", kwargs)

    client = SimpleNamespace(aio=SimpleNamespace(models=AsyncModels(), chats=AsyncChats()))
    provider._get_client = lambda: client
    events = asyncio.run(_collect_google_async_events(provider))
    assert [event.type for event in events] == ["finish"]
    assert asyncio.run(provider.async_create_chat(history=[])) == (
        "chat",
        {"model": "gemini", "history": []},
    )


def test_google_async_cache_and_batch_resources_supply_default_model():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = []

    class AsyncCaches:
        async def create(self, **kwargs):
            calls.append(("cache", kwargs))
            return "cache"

    class AsyncBatches:
        async def create(self, **kwargs):
            calls.append(("batch", kwargs))
            return "batch"

    provider._get_client = lambda: SimpleNamespace(
        aio=SimpleNamespace(caches=AsyncCaches(), batches=AsyncBatches())
    )
    assert asyncio.run(provider.async_resources.create_cache(config={})) == "cache"
    assert asyncio.run(provider.async_resources.create_batch([])) == "batch"
    assert calls == [
        ("cache", {"model": "gemini-default", "config": {}}),
        ("batch", {"model": "gemini-default", "src": []}),
    ]


def test_google_tuning_and_file_search_resources_forward_current_sdk_methods():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = []

    class Tunings:
        def tune(self, **kwargs):
            calls.append(("tune", kwargs))
            return "tuning"

        def validate_reward(self, **kwargs):
            calls.append(("reward", kwargs))
            return "reward"

    class FileSearchStores:
        def import_file(self, **kwargs):
            calls.append(("import", kwargs))
            return "imported"

    provider._get_client = lambda: SimpleNamespace(
        tunings=Tunings(), file_search_stores=FileSearchStores()
    )
    assert provider.resources.tune("gemini-base", {"examples": []}, config={}) == "tuning"
    assert (
        provider.resources.validate_tuning_reward("jobs/1", {"text": "ok"}, {"input": "x"})
        == "reward"
    )
    assert provider.resources.import_file_to_file_search_store("stores/1", "files/1") == "imported"
    assert calls == [
        (
            "tune",
            {
                "base_model": "gemini-base",
                "training_dataset": {"examples": []},
                "config": {},
            },
        ),
        (
            "reward",
            {"parent": "jobs/1", "sample_response": {"text": "ok"}, "example": {"input": "x"}},
        ),
        ("import", {"file_search_store_name": "stores/1", "file_name": "files/1"}),
    ]


def test_google_tuning_resource_missing_is_explicit():
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="调优资源"):
        provider.resources.list_tunings()


def test_resource_facade_rejects_sensitive_request_extensions_sync_and_async():
    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider.list_files = lambda **_: "should-not-run"
    provider.async_list_files = lambda **_: "should-not-run"

    with pytest.raises(ValueError, match="凭证"):
        provider.resources.list_files(extra_headers={"X-API-Key": "secret"})

    with pytest.raises(ValueError, match="凭证"):
        asyncio.run(provider.async_resources.list_files(extra_body={"token": "secret"}))


def test_direct_provider_resource_methods_reject_sensitive_extensions():
    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._get_client = lambda: SimpleNamespace(
        files=SimpleNamespace(list=lambda **kwargs: kwargs)
    )

    with pytest.raises(ValueError, match="凭证"):
        provider.list_files(extra_headers={"X-API-Key": "secret"})

    with pytest.raises(ValueError, match="凭证"):
        provider.list_files(extra_body={"nested": {"access_token": "secret"}})


def test_dynamic_native_resource_proxy_validates_nested_calls_and_keeps_results():
    calls = []

    class Files:
        def list(self, **kwargs):
            calls.append(kwargs)
            return "page"

    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    native = SimpleNamespace(experimental=SimpleNamespace(files=Files()))
    provider._get_client = lambda: native

    assert provider.resources.experimental.files.list(limit=2) == "page"
    assert calls == [{"limit": 2}]
    with pytest.raises(ValueError, match="凭证"):
        provider.resources.experimental.files.list(extra_query={"access_token": "secret"})


def test_dynamic_native_resource_proxy_rejects_sensitive_positional_arguments():
    calls = []

    class Responses:
        def connect(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "connection"

    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    assert (
        provider.resources.responses.connect({"tenant": "demo"}, {"X-Trace": "1"}) == "connection"
    )
    assert calls == [
        (({"tenant": "demo"}, {"X-Trace": "1"}), {}),
    ]
    for args in (
        ({"Authorization": "Bearer secret"},),
        ({"nested": {"access_token": "secret"}},),
        ("https://api.example.test/v1?api_key=secret",),
    ):
        with pytest.raises(ValueError, match="位置参数.*凭证"):
            provider.resources.responses.connect(*args)


def test_direct_openai_resource_call_rejects_sensitive_positional_arguments():
    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"

    def connect(*args, **kwargs):
        return args, kwargs

    with pytest.raises(ValueError, match="位置参数.*凭证"):
        provider._call_sdk_resource(
            connect,
            args=({"extra_headers": {"Authorization": "Bearer secret"}},),
            operation="OpenAI Responses 连接",
        )
    assert provider._call_sdk_resource(
        connect,
        args=("response-id",),
        operation="OpenAI Responses 连接",
    ) == (("response-id",), {})


def test_provider_resource_guard_rejects_sensitive_positional_arguments():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"

    class Models:
        def update(self, **_kwargs):
            raise AssertionError("SDK should not be called")

    provider._get_client = lambda: SimpleNamespace(models=Models())
    with pytest.raises(ValueError, match="位置参数.*凭证"):
        provider.update_model("models/demo", {"Authorization": "Bearer secret"})


def test_provider_resource_guard_normalizes_duplicate_sdk_arguments():
    provider = object.__new__(GoogleProvider)

    class Interactions:
        def get(self, *, id):
            return id

    provider._get_client = lambda: SimpleNamespace(interactions=Interactions())
    with pytest.raises(ValueError, match="资源 SDK 参数重复"):
        provider.get_interaction("interaction-1", id="interaction-2")


def test_resource_argument_validator_keeps_nested_business_values_isolated():
    original = {"query": {"text": "hello"}}
    validated = validate_secret_free_resource_args((original,), "Provider")
    assert validated == (original,)
    assert validated[0] is not original
    original["query"]["text"] = "changed"
    assert validated[0]["query"]["text"] == "hello"


def test_resource_kwargs_reject_duplicate_extension_spellings():
    with pytest.raises(ValueError, match="重复的扩展字段"):
        validate_secret_free_resource_kwargs(
            {"extra_headers": {"X-Trace": "1"}, "EXTRA_HEADERS": {"X-Trace": "2"}},
            "Provider",
        )


def test_dynamic_native_resource_proxy_preserves_native_client_identity():
    provider = object.__new__(OpenAIProvider)
    native = SimpleNamespace(experimental=SimpleNamespace(value="ok"))
    provider._get_client = lambda: native

    assert provider.resources.native is native
    assert provider.resources.experimental.value == "ok"


def test_dynamic_native_vector_store_search_allows_business_query_but_scans_nested_credentials():
    calls = []

    class VectorStores:
        def search(self, **kwargs):
            calls.append(kwargs)
            return "page"

    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._get_client = lambda: SimpleNamespace(vector_stores=VectorStores())

    assert provider.resources.vector_stores.search(vector_store_id="vs_1", query="refund") == "page"
    assert calls == [{"vector_store_id": "vs_1", "query": "refund"}]

    with pytest.raises(ValueError, match="凭证"):
        provider.resources.vector_stores.search(
            vector_store_id="vs_1", query={"Authorization": "Bearer secret"}
        )


def test_dynamic_native_resource_proxy_wraps_with_options_results_and_guards_credentials():
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return "response"

    class Client:
        def __init__(self):
            self.responses = Responses()

        def with_options(self, **kwargs):
            calls.append(("with_options", kwargs))
            return Client()

    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._get_client = Client

    configured = provider.resources.with_options(timeout=2)
    assert configured.responses.create(model="demo") == "response"
    assert calls == [
        ("with_options", {"timeout": 2}),
        ("create", {"model": "demo"}),
    ]

    for key in ("api_key", "admin_api_key", "webhook_secret", "default_headers", "base_url"):
        with pytest.raises(ValueError, match="凭证或请求头/query"):
            provider.resources.with_options(**{key: "secret"})

    with pytest.raises(ValueError, match="凭证"):
        configured.responses.create(extra_headers={"Authorization": "Bearer secret"})


def test_dynamic_async_native_resource_proxy_guards_nested_raw_and_streaming_resources():
    calls = []

    class RawResponses:
        async def create(self, **kwargs):
            calls.append(("raw", kwargs))
            return "raw-response"

    class StreamingResponses:
        async def create(self, **kwargs):
            calls.append(("streaming", kwargs))
            return "streaming-response"

    class Responses:
        def __init__(self):
            self._raw = RawResponses()
            self._streaming = StreamingResponses()

        @property
        def with_raw_response(self):
            return self._raw

        @property
        def with_streaming_response(self):
            return self._streaming

    class Client:
        def __init__(self):
            self.responses = Responses()

        def with_options(self, **kwargs):
            calls.append(("with_options", kwargs))
            return self

        def copy(self, **kwargs):
            calls.append(("copy", kwargs))
            return self

    client = Client()
    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._get_aclient = lambda: client

    configured = provider.async_resources.with_options(timeout=2)
    copied = configured.copy(max_retries=1)
    assert asyncio.run(copied.responses.with_raw_response.create(model="demo")) == "raw-response"
    assert (
        asyncio.run(copied.responses.with_streaming_response.create(model="demo"))
        == "streaming-response"
    )
    assert calls[:2] == [("with_options", {"timeout": 2}), ("copy", {"max_retries": 1})]

    with pytest.raises(ValueError, match="凭证"):
        copied.responses.with_raw_response.create(extra_headers={"Authorization": "Bearer secret"})
    with pytest.raises(ValueError, match="凭证"):
        copied.responses.with_streaming_response.create(
            extra_body={"nested": {"access_token": "secret"}}
        )

    for helper in ("with_options", "copy"):
        with pytest.raises(ValueError, match="凭证或请求头/query"):
            getattr(provider.async_resources, helper)(default_query={"token": "secret"})


def test_google_tuning_facades_forward_additional_sdk_kwargs():
    calls = []

    class Provider:
        def tune(self, base_model, training_dataset, config=None, **kwargs):
            calls.append(("sync", base_model, training_dataset, config, kwargs))
            return "tuning"

        async def async_tune(self, base_model, training_dataset, config=None, **kwargs):
            calls.append(("async", base_model, training_dataset, config, kwargs))
            return "async-tuning"

    provider = Provider()
    assert (
        GoogleResources(provider).tune("gemini-base", {"examples": []}, labels={"suite": "test"})
        == "tuning"
    )
    assert (
        asyncio.run(
            AsyncGoogleResources(provider).tune(
                "gemini-base", {"examples": []}, labels={"suite": "test"}
            )
        )
        == "async-tuning"
    )
    assert calls == [
        ("sync", "gemini-base", {"examples": []}, None, {"labels": {"suite": "test"}}),
        ("async", "gemini-base", {"examples": []}, None, {"labels": {"suite": "test"}}),
    ]


def test_google_next_generation_resources_forward_sync_and_async_calls():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = []

    class Resource:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                return f"{name}-result"

            return method

    provider._get_client = lambda: SimpleNamespace(
        interactions=Resource(),
        agents=Resource(),
        webhooks=Resource(),
        environments=Resource(),
        triggers=Resource(),
        aio=SimpleNamespace(
            interactions=Resource(),
            agents=Resource(),
            webhooks=Resource(),
            environments=Resource(),
            triggers=Resource(),
            live=SimpleNamespace(connect=lambda **kwargs: ("live", kwargs)),
        ),
    )

    assert provider.resources.create_interaction(input="hello") == "create-result"
    assert provider.resources.get_webhook("wh-1") == "get-result"
    assert provider.resources.run_trigger("tr-1", payload={"x": 1}) == "run-result"
    assert provider.resources.connect_live(config={}) == (
        "live",
        {"model": "gemini-default", "config": {}},
    )

    async def collect():
        return (
            await provider.async_resources.create_interaction(input="hello"),
            await provider.async_resources.get_agent("agent-1"),
            await provider.async_resources.delete_environment("env-1"),
            await provider.async_resources.list_trigger_executions("tr-1"),
        )

    assert asyncio.run(collect()) == (
        "create-result",
        "get-result",
        "delete_environment-result",
        "list_executions-result",
    )
    assert ("create", (), {"input": "hello"}) in calls
    assert ("get", ("wh-1",), {}) in calls


def test_provider_resource_facades_expose_beta_trees_without_copying_sdk_methods():
    anthropic_provider = object.__new__(AnthropicProvider)
    anthropic_client = SimpleNamespace(beta=SimpleNamespace(messages="beta-messages"))
    anthropic_provider._get_client = lambda: anthropic_client
    anthropic_provider._get_aclient = lambda: SimpleNamespace(beta="async-beta")
    assert anthropic_provider.resources.beta.messages == "beta-messages"
    assert anthropic_provider.async_resources.async_beta == "async-beta"

    compatible = object.__new__(OpenAICompatibleProvider)
    compatible._get_client = lambda: SimpleNamespace(beta="should-not-be-visible")
    compatible._get_aclient = lambda: SimpleNamespace(beta="should-not-be-visible")
    assert not hasattr(compatible.resources, "beta")
    assert not hasattr(compatible.resources, "async_beta")

    ark_provider = object.__new__(VolcengineProvider)
    ark_calls = []

    class Completions:
        def parse(self, **kwargs):
            ark_calls.append(("parse", kwargs))
            return "parsed"

        def stream(self, **kwargs):
            ark_calls.append(("stream", kwargs))
            return "stream"

    ark_provider._model_name = "ark-model"
    ark_provider._get_client = lambda: SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        bot_chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: ("bot", kwargs))
        ),
    )
    assert ark_provider.resources.beta_chat_parse(messages=[], response_format={}) == "parsed"
    assert ark_provider.resources.beta_chat_stream(messages=[]) == "stream"
    assert ark_provider.resources.bot_chat(messages=[]) == (
        "bot",
        {"messages": [], "model": "ark-model"},
    )
    assert ark_calls == [
        ("parse", {"messages": [], "response_format": {}, "model": "ark-model"}),
        ("stream", {"messages": [], "model": "ark-model"}),
    ]


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        ("upload_file", (b"data", "user_data"), {"extra_headers": {"X-API-Key": "secret"}}),
        ("list_files", (), {"extra_query": {"access_token": "secret"}}),
        ("retrieve_response", ("resp-1",), {"extra_body": {"token": "secret"}}),
        (
            "create_content_generation_task",
            (),
            {"content": [], "extra_body": {"password": "secret"}},
        ),
        ("beta_chat_parse", (), {"messages": [], "extra_query": {"api_key": "secret"}}),
        ("bot_chat", (), {"messages": [], "extra_headers": {"Authorization": "Bearer secret"}}),
        ("classify", ("query", ["label"]), {"extra_headers": {"X-Api-Key": "secret"}}),
    ],
)
def test_ark_resource_methods_reject_sensitive_extensions(method, args, kwargs):
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "responses"
    provider._model_name = "ark-model"

    class Resource:
        def __getattr__(self, _name):
            return lambda **_kwargs: "should-not-run"

    provider._get_client = lambda: SimpleNamespace(
        files=Resource(),
        responses=Resource(),
        content_generation=SimpleNamespace(tasks=Resource()),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=Resource())),
        bot_chat=SimpleNamespace(completions=Resource()),
        classification=Resource(),
    )

    with pytest.raises(ValueError, match="凭证"):
        getattr(provider, method)(*args, **kwargs)


def test_ark_async_beta_chat_stream_returns_async_context_manager_directly():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    calls = []

    class StreamManager:
        def __init__(self):
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return "stream"

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.exited = True

    stream_manager = StreamManager()

    class Completions:
        def stream(self, **kwargs):
            calls.append(kwargs)
            return stream_manager

    provider._get_aclient = lambda: SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )

    result = provider.async_resources.beta_chat_stream(messages=[])

    assert result is stream_manager
    assert not inspect.isawaitable(result)
    assert calls == [{"messages": [], "model": "ark-model"}]

    async def consume_stream():
        async with result as stream:
            return stream

    assert asyncio.run(consume_stream()) == "stream"
    assert stream_manager.entered is True
    assert stream_manager.exited is True


async def _collect_google_async_events(provider):
    return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]


def test_ark_async_input_items_use_top_level_resource():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = {}

    class InputItems:
        def list(self, response_id, **kwargs):
            calls["args"] = (response_id, kwargs)
            return "items"

    provider._get_aclient = lambda: SimpleNamespace(input_items=InputItems())
    assert asyncio.run(provider.async_list_response_input_items("resp-1", limit=2)) == "items"
    assert calls["args"] == ("resp-1", {"limit": 2})


@pytest.mark.parametrize(
    "method_name, args, kwargs",
    [
        ("retrieve_response", ("resp-1",), {"include": ["output"]}),
        ("delete_response", ("resp-1",), {"include": ["output"]}),
        ("list_response_input_items", ("resp-1",), {"bogus": True}),
        ("list_input_items", ("resp-1",), {"bogus": True}),
    ],
)
def test_ark_response_resources_reject_unknown_kwargs_before_sdk_call(method_name, args, kwargs):
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._get_client = lambda: pytest.fail("不支持的 Ark Responses 参数不应调用 SDK")

    with pytest.raises(ValueError, match="不支持请求参数"):
        getattr(provider, method_name)(*args, **kwargs)


@pytest.mark.parametrize(
    "method_name, args, kwargs",
    [
        ("async_retrieve_response", ("resp-1",), {"include": ["output"]}),
        ("async_delete_response", ("resp-1",), {"include": ["output"]}),
        ("async_list_response_input_items", ("resp-1",), {"bogus": True}),
        ("async_list_input_items", ("resp-1",), {"bogus": True}),
    ],
)
def test_ark_async_response_resources_reject_unknown_kwargs_before_sdk_call(
    method_name, args, kwargs
):
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._get_aclient = lambda: pytest.fail("不支持的 Ark Responses 参数不应调用 SDK")

    async def invoke():
        await getattr(provider, method_name)(*args, **kwargs)

    with pytest.raises(ValueError, match="不支持请求参数"):
        asyncio.run(invoke())


def test_ark_async_chat_stream_rejects_incomplete_tool_call():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        async def create(self, **_request):
            class Stream:
                def __aiter__(self):
                    self._events = iter(
                        [
                            SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        delta=SimpleNamespace(
                                            content=None,
                                            tool_calls=[
                                                SimpleNamespace(
                                                    id="call-1",
                                                    index=0,
                                                    type="function",
                                                    function=SimpleNamespace(
                                                        name="lookup", arguments=""
                                                    ),
                                                )
                                            ],
                                        ),
                                        finish_reason=None,
                                    )
                                ]
                            ),
                            SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        delta=SimpleNamespace(content=None, tool_calls=None),
                                        finish_reason="tool_calls",
                                    )
                                ]
                            ),
                        ]
                    )
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        raise StopAsyncIteration from exc

            return Stream()

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    with pytest.raises(RuntimeError, match="完整 arguments"):
        asyncio.run(collect())


def test_ark_responses_request_and_stream(monkeypatch):
    settings = SimpleNamespace(
        volc_access_key=None,
        volc_secret_key=None,
        ark_api_key="ark-key",
        volc_base_url="https://ark.example/v3",
    )
    monkeypatch.setitem(VolcengineProvider.__init__.__globals__, "get_settings", lambda: settings)
    provider = VolcengineProvider(
        "doubao",
        protocol="responses",
        options={"server_verified_protocols": ["responses"]},
    )
    assert (
        provider._build_responses_request(CompletionRequest(prompt="hi", stream=True))["input"]
        == "hi"
    )

    class Responses:
        def create(self, **_kwargs):
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="ok"),
                    SimpleNamespace(
                        type="response.completed", response=SimpleNamespace(status="completed")
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    assert list(provider.invoke("hi")) == ["ok"]


def test_ark_responses_direct_invocation_requires_server_verification():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {}
    provider._get_client = lambda: pytest.fail("未验证的 Ark Responses 不应初始化客户端")

    with pytest.raises(ValueError, match="server_verified_protocols"):
        list(provider.invoke("hello", stream=False))


def test_factory_ark_responses_requires_explicit_server_verification(monkeypatch):
    settings = SimpleNamespace(
        volc_access_key=None,
        volc_secret_key=None,
        ark_api_key="ark-key",
        volc_base_url="https://ark.example/v3",
    )
    monkeypatch.setitem(VolcengineProvider.__init__.__globals__, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ModelProviderFactory,
        "_get_provider_class",
        staticmethod(lambda _provider_name: VolcengineProvider),
    )

    with pytest.raises(ValueError, match="server_verified_protocols"):
        ModelProviderFactory._create_provider(
            "volcengine",
            "doubao-model",
            "llm",
            LargeLanguageModel,
            protocol="responses",
            options={},
        )

    provider = ModelProviderFactory._create_provider(
        "volcengine",
        "doubao-model",
        "llm",
        LargeLanguageModel,
        protocol="responses",
        options={"server_verified_protocols": ["responses"]},
    )
    assert isinstance(provider, VolcengineProvider)


def test_ark_ainvoke_preserves_explicit_none_system_prompt():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "chat_completions"
    provider._options = {}
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [item async for item in provider.ainvoke("hello", system_prompt=None, stream=False)]

    assert asyncio.run(consume()) == ["ok"]
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_ark_responses_normalizes_model_options_to_native_shape():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "high",
    }

    request = provider._build_responses_request(CompletionRequest(prompt="hi", stream=False))

    assert request["max_output_tokens"] == 1024
    assert request["text"] == {"format": {"type": "json_object"}}
    assert request["reasoning"] == {"effort": "high"}
    assert "max_tokens" not in request
    assert "response_format" not in request
    assert "reasoning_effort" not in request


def test_ark_responses_converts_configured_tool_choice_to_native_shape():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }

    request = provider._build_responses_request(CompletionRequest(prompt="hi", stream=False))

    assert request["tool_choice"] == {"type": "function", "name": "lookup"}


def test_ark_chat_model_options_move_sampling_extensions_into_extra_body():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"top_k": 20, "seed": 7}

    request = provider._build_chat_request(prompt="hi", stream=False)

    assert request["extra_body"] == {"top_k": 20, "seed": 7}
    assert "top_k" not in request
    assert "seed" not in request


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "function", "function": "lookup"},
        {"type": "function", "function": 1},
    ],
)
def test_ark_responses_tool_choice_rejects_non_object_function(tool_choice):
    with pytest.raises(ValueError, match="function tool_choice"):
        VolcengineProvider._convert_responses_tool_choice(tool_choice)


def test_ark_response_resource_create_uses_unified_request_and_native_defaults():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return "response"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    assert (
        provider.resources.create_response(CompletionRequest(prompt="hi", stream=False))
        == "response"
    )
    assert calls == [
        {
            "model": "ark-model",
            "input": "hi",
            "stream": False,
            "instructions": "You are a helpful assistant.",
        }
    ]
    with pytest.raises(ValueError, match="model"):
        provider.resources.create_response(input="raw")
    assert provider.resources.create_response(input="raw", model="caller-model") == "response"
    assert calls[-1] == {"input": "raw", "model": "caller-model"}


def test_ark_async_native_response_resource_requires_explicit_model_and_preserves_kwargs():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "configured-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = []

    class Responses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return "async-response"

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def invoke():
        with pytest.raises(ValueError, match="model"):
            await provider.async_resources.create_response(input="raw")
        result = await provider.async_resources.create_response(
            input="raw", model="caller-model", stream=False
        )
        return result

    assert asyncio.run(invoke()) == "async-response"
    assert calls == [{"input": "raw", "model": "caller-model", "stream": False}]


def test_google_environment_resource_names_and_async_live_contracts():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    calls = []

    class Files:
        def list(self, environment, path, **kwargs):
            calls.append(("files", (environment, path), kwargs))
            return "files"

    class AsyncFiles:
        async def list(self, environment, path, **kwargs):
            calls.append(("async-files", (environment, path), kwargs))
            return "async-files"

    class Environments:
        def create_environment(self, **kwargs):
            calls.append(("create", (), kwargs))
            return "created"

        def get_environment(self, **kwargs):
            calls.append(("get", (), kwargs))
            return "environment"

        def list_environments(self, **kwargs):
            calls.append(("list", (), kwargs))
            return "environments"

        def delete_environment(self, **kwargs):
            calls.append(("delete", (), kwargs))
            return "deleted"

        files = Files()

    class AsyncEnvironments(Environments):
        files = AsyncFiles()

    class Live:
        def connect(self, **kwargs):
            calls.append(("live", (), kwargs))
            return "live-manager"

    provider._get_client = lambda: SimpleNamespace(
        environments=Environments(),
        aio=SimpleNamespace(
            environments=AsyncEnvironments(),
            live=Live(),
        ),
    )
    assert provider.resources.create_environment(network={}) == "created"
    assert provider.resources.get_environment("env-1") == "environment"
    assert provider.resources.list_environments(page_size=10) == "environments"
    assert provider.resources.delete_environment("env-1") == "deleted"
    assert provider.resources.get_environment_files("env-1", "src", recursive=True) == "files"
    assert provider.async_resources.connect_live(config={}) == "live-manager"
    assert (
        asyncio.run(provider.async_resources.get_environment_files("env-1", "src", recursive=True))
        == "async-files"
    )
    assert calls == [
        ("create", (), {"network": {}}),
        ("get", (), {"id": "env-1"}),
        ("list", (), {"page_size": 10}),
        ("delete", (), {"id": "env-1"}),
        ("files", ("env-1", "src"), {"recursive": True}),
        ("live", (), {"model": "gemini-default", "config": {}}),
        ("async-files", ("env-1", "src"), {"recursive": True}),
    ]


@pytest.mark.parametrize(
    "method_name",
    [
        "create_auth_token",
        "create_interaction",
        "create_agent",
        "create_webhook",
        "create_environment",
        "create_trigger",
    ],
)
def test_google_experimental_resources_fail_with_capability_error_when_sdk_missing(method_name):
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="google-genai SDK"):
        getattr(provider, method_name)()


@pytest.mark.parametrize(
    "method_name",
    [
        "async_create_auth_token",
        "async_create_interaction",
        "async_create_agent",
        "async_create_webhook",
        "async_create_environment",
        "async_create_trigger",
    ],
)
def test_google_async_experimental_resources_fail_with_capability_error_when_sdk_missing(
    method_name,
):
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace())

    async def invoke():
        with pytest.raises(NotImplementedError, match="google-genai SDK"):
            await getattr(provider, method_name)()

    asyncio.run(invoke())


def test_google_live_connection_rejects_credentials_in_config():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    provider._get_client = lambda: SimpleNamespace(
        aio=SimpleNamespace(
            live=SimpleNamespace(connect=lambda **_: "should-not-run"),
        )
    )
    config = {"http_options": {"headers": {"Authorization": "Bearer secret"}}}
    with pytest.raises(ValueError, match="凭证"):
        provider.connect_live(config=config)
    with pytest.raises(ValueError, match="凭证"):
        provider.async_connect_live(config=config)


def test_google_live_connection_reports_missing_sdk_resource():
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-default"
    provider._get_client = lambda: SimpleNamespace(aio=SimpleNamespace())
    with pytest.raises(NotImplementedError, match="Live"):
        provider.connect_live()


def test_ark_content_generation_delete_uses_keyword_task_id_for_sync_and_async():
    provider = object.__new__(VolcengineProvider)
    calls = []

    class Tasks:
        def delete(self, *, task_id, **kwargs):
            calls.append(("sync", task_id, kwargs))
            return "deleted"

    class AsyncTasks:
        async def delete(self, *, task_id, **kwargs):
            calls.append(("async", task_id, kwargs))
            return "async-deleted"

    provider._get_client = lambda: SimpleNamespace(
        content_generation=SimpleNamespace(tasks=Tasks())
    )
    provider._get_aclient = lambda: SimpleNamespace(
        content_generation=SimpleNamespace(tasks=AsyncTasks())
    )

    assert provider.resources.delete_content_generation_task("task-1", timeout=4) == "deleted"
    assert (
        asyncio.run(provider.async_resources.delete_content_generation_task("task-2", timeout=5))
        == "async-deleted"
    )
    assert calls == [
        ("sync", "task-1", {"timeout": 4}),
        ("async", "task-2", {"timeout": 5}),
    ]


def test_ark_batch_chat_uses_current_nested_batch_resource():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return "batch"

    provider._get_client = lambda: SimpleNamespace(
        batch=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )
    assert provider.resources.create_batch_chat(messages=[]) == "batch"
    assert calls == {"messages": [], "model": "ark-model"}


def test_ark_async_batch_chat_forwards_sdk_supported_user_argument():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"

    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return "async-batch"

    provider._get_aclient = lambda: SimpleNamespace(
        batch=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )

    assert (
        asyncio.run(provider.async_create_batch_chat(messages=[], user="user-1")) == "async-batch"
    )
    assert calls == [{"messages": [], "model": "ark-model", "user": "user-1"}]


def test_ark_context_resources_validate_sdk_parameter_names_and_defaults():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    calls = []

    class Context:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return "context"

        class Completions:
            def create(self, **kwargs):
                calls.append(("complete", kwargs))
                return "completion"

        completions = Completions()

    provider._get_client = lambda: SimpleNamespace(context=Context())

    assert provider.create_context(messages=[], mode="session") == "context"
    assert provider.context_complete(context_id="ctx-1", messages=[], stream=False) == "completion"
    assert calls == [
        ("create", {"messages": [], "mode": "session", "model": "ark-model"}),
        (
            "complete",
            {"context_id": "ctx-1", "messages": [], "stream": False, "model": "ark-model"},
        ),
    ]

    with pytest.raises(ValueError, match="Context.*temperature"):
        provider.create_context(messages=[], temperature=0.2)
    with pytest.raises(ValueError, match="Context Completion.*unknown"):
        provider.context_complete(context_id="ctx-1", messages=[], unknown=None)


def test_ark_classification_validates_query_labels_model_and_kwargs():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._classification_resource = SimpleNamespace(create=lambda **kwargs: kwargs)
    provider._get_client = lambda: SimpleNamespace()

    result = provider.classify(" refund ", [" billing ", "support"])
    assert result == {
        "query": "refund",
        "model": "ark-model",
        "labels": ["billing", "support"],
    }
    with pytest.raises(ValueError, match="query"):
        provider.classify(" ", ["support"])
    with pytest.raises(ValueError, match="labels"):
        provider.classify("refund", [])
    with pytest.raises(ValueError, match="Classification.*unknown"):
        provider.classify("refund", ["support"], unknown=None)


def test_ark_batch_multimodal_embedding_rejects_sparse_embedding_before_sdk_call():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"

    class FailingResource:
        def create(self, **_kwargs):
            raise AssertionError("不应调用不支持 sparse_embedding 的 SDK 方法")

    provider._get_client = lambda: SimpleNamespace(
        batch=SimpleNamespace(multimodal_embeddings=FailingResource())
    )
    provider._get_aclient = lambda: SimpleNamespace(
        batch=SimpleNamespace(multimodal_embeddings=FailingResource())
    )

    with pytest.raises(
        ValueError, match="Ark Batch Multimodal Embedding 不支持请求参数: sparse_embedding"
    ):
        provider.create_batch_multimodal_embedding(input=[], sparse_embedding={"enabled": True})

    with pytest.raises(
        ValueError, match="Ark Batch Multimodal Embedding 不支持请求参数: sparse_embedding"
    ):
        asyncio.run(
            provider.async_create_batch_multimodal_embedding(
                input=[], sparse_embedding={"enabled": True}
            )
        )


def test_ark_async_generate_batch_chat_uses_matching_provider_method():
    provider = object.__new__(VolcengineProvider)
    calls = []

    async def create_batch_chat(**kwargs):
        calls.append(("create", kwargs))
        return "created"

    async def generate_batch_chat(**kwargs):
        calls.append(("generate", kwargs))
        return "generated"

    provider.async_create_batch_chat = create_batch_chat
    provider.async_generate_batch_chat = generate_batch_chat

    result = asyncio.run(provider.async_resources.generate_batch_chat(messages=[]))

    assert result == "generated"
    assert calls == [("generate", {"messages": []})]


def test_ark_tokenization_and_multimodal_embedding_reject_unknown_sdk_kwargs():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace()
    with pytest.raises(ValueError, match="Tokenization 不支持请求参数: temperature"):
        provider.count_tokens("hello", temperature=0.2)
    with pytest.raises(ValueError, match="Multimodal Embedding 不支持请求参数: temperature"):
        provider.multimodal_embed([], temperature=0.2)


def test_openai_responses_stream_helper_matches_installed_sdk_contract_sync_and_async():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    calls = []

    class Responses:
        def stream(self, **kwargs):
            calls.append(("sync", kwargs))
            return "sync-stream"

    class AsyncResponses:
        def stream(self, **kwargs):
            calls.append(("async", kwargs))
            return "async-stream"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    provider._get_aclient = lambda: SimpleNamespace(responses=AsyncResponses())

    assert provider.stream_responses(CompletionRequest(prompt="hi", stream=True)) == "sync-stream"
    assert (
        provider.async_stream_responses(CompletionRequest(prompt="hi", stream=True))
        == "async-stream"
    )
    assert calls == [
        (
            "sync",
            {
                "model": "demo",
                "input": "hi",
                "instructions": "You are a helpful assistant.",
            },
        ),
        (
            "async",
            {
                "model": "demo",
                "input": "hi",
                "instructions": "You are a helpful assistant.",
            },
        ),
    ]

    assert provider.stream_responses(input="hi", safety_identifier="user-1") == "sync-stream"
    assert calls[-1] == (
        "sync",
        {"input": "hi", "model": "demo", "safety_identifier": "user-1"},
    )
    assert provider.stream_responses(input="hi", verbosity="low") == "sync-stream"
    assert calls[-1] == (
        "sync",
        {"input": "hi", "model": "demo", "text": {"verbosity": "low"}},
    )


def test_openai_responses_stream_continuation_omits_model_sync_and_async():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    calls = []

    class Responses:
        def stream(self, **kwargs):
            if "response_id" in kwargs or "starting_after" in kwargs:
                assert "model" not in kwargs
            calls.append(("sync", kwargs))
            return "sync-stream"

    class AsyncResponses:
        def stream(self, **kwargs):
            if "response_id" in kwargs or "starting_after" in kwargs:
                assert "model" not in kwargs
            calls.append(("async", kwargs))
            return "async-stream"

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    provider._get_aclient = lambda: SimpleNamespace(responses=AsyncResponses())

    assert provider.stream_responses(response_id="resp-1") == "sync-stream"
    assert calls[-1] == ("sync", {"response_id": "resp-1"})
    assert provider.stream_responses(response_id="resp-1", starting_after=2) == "sync-stream"
    assert calls[-1] == (
        "sync",
        {"response_id": "resp-1", "starting_after": 2},
    )

    assert provider.async_stream_responses(response_id="resp-1", starting_after=2) == "async-stream"
    assert calls[-1] == (
        "async",
        {"response_id": "resp-1", "starting_after": 2},
    )


def test_openai_polling_helpers_match_installed_sdk_parameters_sync():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    calls = []

    class Files:
        def wait_for_processing(self, **kwargs):
            calls.append(("wait", kwargs))
            return "file"

    class VectorFiles:
        def poll(self, *args, **kwargs):
            calls.append(("vector_file", args, kwargs))
            return "vector-file"

    class VectorBatches:
        def poll(self, *args, **kwargs):
            calls.append(("vector_batch", args, kwargs))
            return "vector-batch"

        def upload_and_poll(self, **kwargs):
            calls.append(("upload_batch", kwargs))
            return "uploaded-batch"

    class Videos:
        def poll(self, *args, **kwargs):
            calls.append(("video", args, kwargs))
            return "video"

    provider._get_client = lambda: SimpleNamespace(
        files=Files(),
        vector_stores=SimpleNamespace(files=VectorFiles(), file_batches=VectorBatches()),
        videos=Videos(),
    )

    assert provider.wait_for_file("file-1", poll_interval=1.5, max_wait_seconds=20) == "file"
    assert (
        provider.poll_vector_store_file("store-1", "file-1", poll_interval_ms=250) == "vector-file"
    )
    assert (
        provider.poll_vector_store_file_batch("store-1", "batch-1", poll_interval_ms=300)
        == "vector-batch"
    )
    assert (
        provider.upload_vector_store_file_batch_and_poll(
            "store-1",
            ["a"],
            max_concurrency=2,
            file_ids=["f-1"],
            poll_interval_ms=400,
            chunking_strategy={"type": "auto"},
        )
        == "uploaded-batch"
    )
    assert provider.poll_video("video-1", poll_interval_ms=500) == "video"

    assert calls == [
        ("wait", {"id": "file-1", "poll_interval": 1.5, "max_wait_seconds": 20}),
        ("vector_file", ("file-1",), {"vector_store_id": "store-1", "poll_interval_ms": 250}),
        ("vector_batch", ("batch-1",), {"vector_store_id": "store-1", "poll_interval_ms": 300}),
        (
            "upload_batch",
            {
                "vector_store_id": "store-1",
                "files": ["a"],
                "max_concurrency": 2,
                "file_ids": ["f-1"],
                "poll_interval_ms": 400,
                "chunking_strategy": {"type": "auto"},
            },
        ),
        ("video", ("video-1",), {"poll_interval_ms": 500}),
    ]

    with pytest.raises(ValueError, match="文件等待.*timeout"):
        provider.wait_for_file("file-1", timeout=1)
    with pytest.raises(ValueError, match="向量库文件轮询.*extra_headers"):
        provider.poll_vector_store_file("store-1", "file-1", extra_headers={"X-Test": "1"})
    with pytest.raises(ValueError, match="向量库文件批次上传轮询.*timeout"):
        provider.upload_vector_store_file_batch_and_poll("store-1", [], timeout=1)
    with pytest.raises(ValueError, match="视频轮询.*extra_query"):
        provider.poll_video("video-1", extra_query={"trace": "1"})


def test_openai_vector_store_upload_and_container_file_match_sdk_parameters():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class VectorFiles:
        def upload(self, **kwargs):
            calls.append(("upload", kwargs))
            return "uploaded"

    class ContainerFiles:
        def create(self, *args, **kwargs):
            calls.append(("container", args, kwargs))
            return "container-file"

    provider._get_client = lambda: SimpleNamespace(
        vector_stores=SimpleNamespace(files=VectorFiles()),
        containers=SimpleNamespace(files=ContainerFiles()),
    )
    assert (
        provider.upload_vector_store_file("store-1", "file", chunking_strategy={"type": "auto"})
        == "uploaded"
    )
    assert provider.create_container_file("container-1", file_id="file-1") == "container-file"
    assert calls == [
        (
            "upload",
            {
                "vector_store_id": "store-1",
                "file": "file",
                "chunking_strategy": {"type": "auto"},
            },
        ),
        ("container", ("container-1",), {"file_id": "file-1"}),
    ]
    with pytest.raises(ValueError, match="向量库文件上传.*timeout"):
        provider.upload_vector_store_file("store-1", "file", timeout=1)


def test_openai_vector_store_upload_and_poll_matches_installed_sdk_parameters_sync():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class VectorFiles:
        def upload_and_poll(self, **kwargs):
            calls.append(kwargs)
            return "uploaded-and-polled"

    provider._get_client = lambda: SimpleNamespace(
        vector_stores=SimpleNamespace(files=VectorFiles())
    )

    assert (
        provider.upload_vector_store_file_and_poll(
            "store-1",
            "file",
            attributes={"source": "docs", "rank": 1},
            poll_interval_ms=250,
            chunking_strategy={"type": "auto"},
        )
        == "uploaded-and-polled"
    )
    assert calls == [
        {
            "vector_store_id": "store-1",
            "file": "file",
            "attributes": {"source": "docs", "rank": 1},
            "poll_interval_ms": 250,
            "chunking_strategy": {"type": "auto"},
        }
    ]
    with pytest.raises(ValueError, match="向量库文件上传轮询.*timeout"):
        provider.upload_vector_store_file_and_poll("store-1", "file", timeout=1)


def test_openai_vector_store_file_update_requires_sdk_attributes_parameter():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class VectorFiles:
        def update(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "updated"

    provider._get_client = lambda: SimpleNamespace(
        vector_stores=SimpleNamespace(files=VectorFiles())
    )

    assert (
        provider.update_vector_store_file(
            "store-1", "file-1", attributes={"source": "docs", "rank": 1}
        )
        == "updated"
    )
    assert calls == [
        (
            ("file-1",),
            {
                "vector_store_id": "store-1",
                "attributes": {"source": "docs", "rank": 1},
            },
        )
    ]
    with pytest.raises(TypeError):
        provider.update_vector_store_file("store-1", "file-1")


def test_openai_upload_file_chunked_matches_installed_sdk_contract_sync_and_async():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class Uploads:
        def upload_file_chunked(self, **kwargs):
            calls.append(("sync", kwargs))
            return "uploaded"

    class AsyncUploads:
        async def upload_file_chunked(self, **kwargs):
            calls.append(("async", kwargs))
            return "async-uploaded"

    provider._get_client = lambda: SimpleNamespace(uploads=Uploads())
    provider._get_aclient = lambda: SimpleNamespace(uploads=AsyncUploads())

    assert (
        provider.upload_file_chunked(
            file=b"payload",
            mime_type="text/plain",
            purpose="assistants",
            filename="notes.txt",
            bytes=7,
            part_size=4,
            md5="md5-value",
        )
        == "uploaded"
    )

    async def run():
        return await provider.async_upload_file_chunked(
            file=b"payload",
            mime_type="text/plain",
            purpose="assistants",
            filename="notes.txt",
            bytes=7,
            part_size=4,
            md5="md5-value",
        )

    assert asyncio.run(run()) == "async-uploaded"
    assert calls == [
        (
            "sync",
            {
                "file": b"payload",
                "mime_type": "text/plain",
                "purpose": "assistants",
                "filename": "notes.txt",
                "bytes": 7,
                "part_size": 4,
                "md5": "md5-value",
            },
        ),
        (
            "async",
            {
                "file": b"payload",
                "mime_type": "text/plain",
                "purpose": "assistants",
                "filename": "notes.txt",
                "bytes": 7,
                "part_size": 4,
                "md5": "md5-value",
            },
        ),
    ]

    with pytest.raises(ValueError, match="分片文件上传.*timeout"):
        provider.upload_file_chunked(
            file=b"payload", mime_type="text/plain", purpose="assistants", timeout=1
        )

    async def run_invalid():
        with pytest.raises(ValueError, match="分片文件上传.*timeout"):
            await provider.async_upload_file_chunked(
                file=b"payload", mime_type="text/plain", purpose="assistants", timeout=1
            )

    asyncio.run(run_invalid())


def test_openai_container_file_content_reports_missing_sdk_resource():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._get_client = lambda: SimpleNamespace(
        containers=SimpleNamespace(files=SimpleNamespace())
    )

    with pytest.raises(NotImplementedError, match="容器文件内容读取资源"):
        provider.container_file_content("container-1", "file-1")


def test_openai_vector_store_upload_and_container_file_match_sdk_parameters_async():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class VectorFiles:
        async def upload(self, **kwargs):
            calls.append(("upload", kwargs))
            return "uploaded"

    class ContainerFiles:
        async def create(self, *args, **kwargs):
            calls.append(("container", args, kwargs))
            return "container-file"

    provider._get_aclient = lambda: SimpleNamespace(
        vector_stores=SimpleNamespace(files=VectorFiles()),
        containers=SimpleNamespace(files=ContainerFiles()),
    )

    async def run():
        assert (
            await provider.async_upload_vector_store_file(
                "store-1", "file", chunking_strategy={"type": "auto"}
            )
            == "uploaded"
        )
        assert (
            await provider.async_create_container_file("container-1", file_id="file-1")
            == "container-file"
        )
        with pytest.raises(ValueError, match="向量库文件上传.*timeout"):
            await provider.async_upload_vector_store_file("store-1", "file", timeout=1)

    asyncio.run(run())
    assert calls == [
        (
            "upload",
            {
                "vector_store_id": "store-1",
                "file": "file",
                "chunking_strategy": {"type": "auto"},
            },
        ),
        ("container", ("container-1",), {"file_id": "file-1"}),
    ]


def test_openai_vector_store_upload_and_poll_matches_installed_sdk_parameters_async():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class VectorFiles:
        async def upload_and_poll(self, **kwargs):
            calls.append(kwargs)
            return "uploaded-and-polled"

    provider._get_aclient = lambda: SimpleNamespace(
        vector_stores=SimpleNamespace(files=VectorFiles())
    )

    async def run():
        assert (
            await provider.async_upload_vector_store_file_and_poll(
                "store-1",
                "file",
                attributes={"source": "docs", "rank": 1},
                poll_interval_ms=250,
                chunking_strategy={"type": "auto"},
            )
            == "uploaded-and-polled"
        )
        with pytest.raises(ValueError, match="向量库文件上传轮询.*timeout"):
            await provider.async_upload_vector_store_file_and_poll("store-1", "file", timeout=1)

    asyncio.run(run())
    assert calls == [
        {
            "vector_store_id": "store-1",
            "file": "file",
            "attributes": {"source": "docs", "rank": 1},
            "poll_interval_ms": 250,
            "chunking_strategy": {"type": "auto"},
        }
    ]


def test_openai_async_container_file_content_reports_missing_sdk_resource():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._get_aclient = lambda: SimpleNamespace(
        containers=SimpleNamespace(files=SimpleNamespace())
    )

    async def run():
        with pytest.raises(NotImplementedError, match="容器文件内容读取资源"):
            await provider.async_container_file_content("container-1", "file-1")

    asyncio.run(run())


def test_openai_polling_helpers_match_installed_sdk_parameters_async():
    provider = object.__new__(OpenAICompatibleProvider)
    calls = []

    class Files:
        async def wait_for_processing(self, **kwargs):
            calls.append(("wait", kwargs))
            return "file"

    class VectorFiles:
        async def poll(self, *args, **kwargs):
            calls.append(("vector_file", args, kwargs))
            return "vector-file"

    class VectorBatches:
        async def poll(self, *args, **kwargs):
            calls.append(("vector_batch", args, kwargs))
            return "vector-batch"

        async def upload_and_poll(self, **kwargs):
            calls.append(("upload_batch", kwargs))
            return "uploaded-batch"

    class Videos:
        async def poll(self, *args, **kwargs):
            calls.append(("video", args, kwargs))
            return "video"

    provider._get_aclient = lambda: SimpleNamespace(
        files=Files(),
        vector_stores=SimpleNamespace(files=VectorFiles(), file_batches=VectorBatches()),
        videos=Videos(),
    )

    async def run():
        assert (
            await provider.async_wait_for_file("file-1", poll_interval=1.5, max_wait_seconds=20)
            == "file"
        )
        assert (
            await provider.async_poll_vector_store_file("store-1", "file-1", poll_interval_ms=250)
            == "vector-file"
        )
        assert (
            await provider.async_poll_vector_store_file_batch(
                "store-1", "batch-1", poll_interval_ms=300
            )
            == "vector-batch"
        )
        assert (
            await provider.async_upload_vector_store_file_batch_and_poll(
                "store-1",
                ["a"],
                max_concurrency=2,
                file_ids=["f-1"],
                poll_interval_ms=400,
                chunking_strategy={"type": "auto"},
            )
            == "uploaded-batch"
        )
        assert await provider.async_poll_video("video-1", poll_interval_ms=500) == "video"
        with pytest.raises(ValueError, match="文件等待.*timeout"):
            await provider.async_wait_for_file("file-1", timeout=1)
        with pytest.raises(ValueError, match="向量库文件轮询.*extra_headers"):
            await provider.async_poll_vector_store_file(
                "store-1", "file-1", extra_headers={"X-Test": "1"}
            )
        with pytest.raises(ValueError, match="向量库文件批次上传轮询.*timeout"):
            await provider.async_upload_vector_store_file_batch_and_poll("store-1", [], timeout=1)
        with pytest.raises(ValueError, match="视频轮询.*extra_query"):
            await provider.async_poll_video("video-1", extra_query={"trace": "1"})

    asyncio.run(run())
    assert calls == [
        ("wait", {"id": "file-1", "poll_interval": 1.5, "max_wait_seconds": 20}),
        ("vector_file", ("file-1",), {"vector_store_id": "store-1", "poll_interval_ms": 250}),
        ("vector_batch", ("batch-1",), {"vector_store_id": "store-1", "poll_interval_ms": 300}),
        (
            "upload_batch",
            {
                "vector_store_id": "store-1",
                "files": ["a"],
                "max_concurrency": 2,
                "file_ids": ["f-1"],
                "poll_interval_ms": 400,
                "chunking_strategy": {"type": "auto"},
            },
        ),
        ("video", ("video-1",), {"poll_interval_ms": 500}),
    ]


def test_ark_file_wait_helpers_match_installed_sdk_parameters_sync_and_async():
    provider = object.__new__(VolcengineProvider)
    calls = []

    class Files:
        def wait_for_processing(self, **kwargs):
            calls.append(("sync", kwargs))
            return "file"

    class AsyncFiles:
        async def wait_for_processing(self, **kwargs):
            calls.append(("async", kwargs))
            return "async-file"

    provider._get_client = lambda: SimpleNamespace(files=Files())
    provider._get_aclient = lambda: SimpleNamespace(files=AsyncFiles())

    assert provider.wait_for_file("file-1", poll_interval=2, max_wait_seconds=30) == "file"
    assert (
        asyncio.run(provider.async_wait_for_file("file-2", poll_interval=1, max_wait_seconds=40))
        == "async-file"
    )
    assert calls == [
        ("sync", {"id": "file-1", "poll_interval": 2, "max_wait_seconds": 30}),
        ("async", {"id": "file-2", "poll_interval": 1, "max_wait_seconds": 40}),
    ]

    with pytest.raises(ValueError, match="文件等待.*timeout"):
        provider.wait_for_file("file-1", timeout=1)
    with pytest.raises(ValueError, match="文件等待.*extra_headers"):
        asyncio.run(provider.async_wait_for_file("file-2", extra_headers={"X-Test": "1"}))


@pytest.mark.parametrize(
    ("file_id", "poll_interval", "max_wait_seconds", "match"),
    [
        ("", 1, 10, "file_id"),
        ("file-1", 0, 10, "poll_interval"),
        ("file-1", float("inf"), 10, "poll_interval"),
        ("file-1", 1, 0, "max_wait_seconds"),
        ("file-1", 1, float("nan"), "max_wait_seconds"),
    ],
)
def test_ark_file_wait_helpers_validate_identifiers_and_timeouts(
    file_id, poll_interval, max_wait_seconds, match
):
    provider = object.__new__(VolcengineProvider)
    provider._get_client = lambda: pytest.fail("无效轮询参数不应调用 SDK")
    provider._get_aclient = lambda: pytest.fail("无效轮询参数不应调用异步 SDK")

    with pytest.raises(ValueError, match=match):
        provider.wait_for_file(
            file_id, poll_interval=poll_interval, max_wait_seconds=max_wait_seconds
        )

    with pytest.raises(ValueError, match=match):
        asyncio.run(
            provider.async_wait_for_file(
                file_id,
                poll_interval=poll_interval,
                max_wait_seconds=max_wait_seconds,
            )
        )


def test_ark_file_wait_helpers_reject_unknown_none_arguments():
    provider = object.__new__(VolcengineProvider)
    provider._get_client = lambda: pytest.fail("未知参数不应调用 SDK")

    with pytest.raises(ValueError, match="文件等待.*timeout"):
        provider.wait_for_file("file-1", timeout=None)


def test_ark_classification_resource_forwards_sync_and_async_calls():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    calls = []

    class Classification:
        def create(self, **kwargs):
            calls.append(("sync", kwargs))
            return "classified"

    class AsyncClassification:
        async def create(self, **kwargs):
            calls.append(("async", kwargs))
            return "async-classified"

    provider._get_client = lambda: SimpleNamespace()
    provider._get_aclient = lambda: SimpleNamespace()
    provider._classification_resource = Classification()
    provider._async_classification_resource = AsyncClassification()

    assert provider.resources.classify("refund", ["billing", "support"]) == "classified"
    assert (
        asyncio.run(provider.async_resources.classify("refund", ["billing", "support"]))
        == "async-classified"
    )
    assert calls == [
        ("sync", {"query": "refund", "model": "ark-model", "labels": ["billing", "support"]}),
        ("async", {"query": "refund", "model": "ark-model", "labels": ["billing", "support"]}),
    ]


def test_ark_close_invalidates_cached_classification_resources():
    provider = object.__new__(VolcengineProvider)
    provider._client = None
    provider._aclient = None
    provider._classification_resource = object()
    provider._async_classification_resource = object()

    provider.close()
    assert provider._classification_resource is None
    assert provider._async_classification_resource is None

    provider._classification_resource = object()
    provider._async_classification_resource = object()
    asyncio.run(provider.aclose())
    assert provider._classification_resource is None
    assert provider._async_classification_resource is None


def test_ark_classification_is_declared_as_provider_capability():
    assert "classification" in VolcengineProvider.capabilities


def test_ark_native_resource_wrappers_reject_credentials_before_sdk_dispatch():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"

    def fail_if_called():
        raise AssertionError("SDK client must not be constructed for rejected credentials")

    provider._get_client = fail_if_called
    provider._get_aclient = fail_if_called

    with pytest.raises(ValueError, match="凭证"):
        provider.list_files(extra_headers={"X-API-Key": "secret"})
    with pytest.raises(ValueError, match="凭证"):
        provider.create_context(messages=[], extra_body={"token": "secret"})

    async def run() -> None:
        with pytest.raises(ValueError, match="凭证"):
            await provider.async_list_files(extra_query={"access_token": "secret"})
        with pytest.raises(ValueError, match="凭证"):
            await provider.async_create_context(
                messages=[], extra_headers={"Authorization": "Bearer secret"}
            )

    asyncio.run(run())


def test_anthropic_token_count_rejects_non_counting_fields():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    with pytest.raises(ValueError, match="不支持请求参数: background"):
        provider.count_tokens(CompletionRequest(prompt="hi", background=True))


def test_anthropic_token_count_rejects_output_format_before_sdk_call():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    calls = {}

    class Messages:
        def count_tokens(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(input_tokens=5)

    provider._get_client = lambda: SimpleNamespace(messages=Messages())
    with pytest.raises(ValueError, match="token count 不支持 output_format"):
        provider.count_tokens(CompletionRequest(prompt="hi", output_format=dict))
    assert calls == {}

    async def run() -> None:
        with pytest.raises(ValueError, match="token count 不支持 output_format"):
            await provider.async_count_tokens(CompletionRequest(prompt="hi", output_format=dict))

    asyncio.run(run())


def test_anthropic_token_count_forwards_supported_extra_body_sync_and_async():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    calls = {}

    class SyncMessages:
        def count_tokens(self, **kwargs):
            calls["sync"] = kwargs
            return SimpleNamespace(input_tokens=5)

    class AsyncMessages:
        async def count_tokens(self, **kwargs):
            calls["async"] = kwargs
            return SimpleNamespace(input_tokens=6)

    provider._get_client = lambda: SimpleNamespace(
        messages=SyncMessages(),
        beta=SimpleNamespace(messages=SyncMessages()),
    )
    provider._get_aclient = lambda: SimpleNamespace(
        messages=AsyncMessages(),
        beta=SimpleNamespace(messages=AsyncMessages()),
    )

    request = CompletionRequest(prompt="hi", extra_body={"tenant": "demo"})
    assert provider.count_tokens(request) == 5
    assert provider.beta_count_tokens(request) == 5
    assert asyncio.run(provider.async_count_tokens(request)) == 6
    assert asyncio.run(provider.async_beta_count_tokens(request)) == 6
    assert calls["sync"]["extra_body"] == {"tenant": "demo"}
    assert calls["async"]["extra_body"] == {"tenant": "demo"}


def test_anthropic_beta_facade_does_not_leak_stream_flag():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}
    calls = {}

    class Messages:
        def create(self, **kwargs):
            calls.update(kwargs)
            return "beta"

        def count_tokens(self, **kwargs):
            calls["count"] = kwargs
            return SimpleNamespace(input_tokens=8)

    provider._get_client = lambda: SimpleNamespace(beta=SimpleNamespace(messages=Messages()))
    assert provider.resources.beta_create(CompletionRequest(prompt="hi")) == "beta"
    assert "stream" not in calls
    assert provider.resources.beta_count_tokens(CompletionRequest(prompt="hi")) == 8


def test_async_embedding_query_default_works_for_legacy_provider():
    class Embedding(TextEmbeddingModel):
        def embed_documents(self, texts, **_kwargs):
            return [[float(len(texts[0]))]]

        async def aembed_documents(self, texts, **_kwargs):
            return [[float(len(texts[0]))]]

    assert asyncio.run(Embedding().aembed_query("abc")) == [3.0]


def test_chat_completions_normalizes_flat_tool_calls_to_openai_shape():
    """我们产出的扁平 tool_calls 必须能在回填时转换成 OpenAI 规范结构。

    ``CompletionResult.tool_calls`` 使用扁平的 ``{"id","type","name",
    "arguments"}`` 形状；OpenAI Chat Completions 要求 assistant 历史的
    ``tool_calls`` 使用嵌套的 ``{"id","type","function":{"name",
    "arguments"}}``。适配器必须在请求边界完成转换，否则多轮工具调用会
    被服务端以 400 拒绝。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    flat = {"id": "call-1", "type": "function", "name": "lookup", "arguments": '{"id":1}'}
    request = provider._build_chat_request(
        messages=[
            {"role": "user", "content": "查询"},
            {"role": "assistant", "content": "", "tool_calls": [flat]},
            {"role": "tool", "tool_call_id": "call-1", "content": "结果"},
        ],
        stream=False,
    )

    tool_calls = request["messages"][1]["tool_calls"]
    assert tool_calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":1}'},
        }
    ], "扁平 tool_calls 必须转换为 OpenAI 嵌套结构"


def test_chat_completions_preserves_native_tool_calls():
    """已经是 OpenAI 原生嵌套结构的历史必须原样保留。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    native = {
        "id": "call-2",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"id":2}'},
    }
    request = provider._build_chat_request(
        messages=[
            {"role": "user", "content": "查询"},
            {"role": "assistant", "content": "", "tool_calls": [native]},
            {"role": "tool", "tool_call_id": "call-2", "content": "结果"},
        ],
        stream=False,
    )

    assert request["messages"][1]["tool_calls"] == [native]


def test_chat_completions_rejects_tool_call_without_function_name():
    """缺少函数名的工具调用必须显式报错，不能静默发出无效请求。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    with pytest.raises(ValueError, match="函数名"):
        provider._build_chat_request(
            messages=[
                {"role": "user", "content": "查询"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-3", "type": "function", "arguments": "{}"}],
                },
            ],
            stream=False,
        )


def test_normalize_messages_accepts_id_less_tool_calls_for_non_chat_protocols():
    """Google 这类协议必须接受无 id 的历史工具调用。

    Gemini 的 ``FunctionCall.id`` 是可选字段且 SDK 默认 ``None``，其
    ``_contents`` 明确允许无 id 的调用。公共的 ``normalize_messages`` 因此不能
    强制要求 id，否则多轮工具调用会在到达 SDK 之前就被本地拒绝。
    """
    normalized = normalize_messages(
        None,
        None,
        [
            {"role": "user", "content": "北京天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": None,
                        "type": "function",
                        "name": "get_weather",
                        "arguments": {"city": "BJ"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "get_weather",
                "name": "get_weather",
                "content": '{"t":20}',
            },
        ],
    )

    tool_call = normalized[1]["tool_calls"][0]
    assert "id" not in tool_call
    assert tool_call["type"] == "function"
    assert tool_call["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"BJ"}',
    }


def test_google_multi_turn_replay_accepts_id_less_tool_calls():
    """把上一轮无 id 的工具调用原样回填给 Google 必须成功。"""
    contents = GoogleProvider._contents(
        normalize_messages(
            None,
            None,
            [
                {"role": "user", "content": "北京天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": None,
                            "type": "function",
                            "name": "get_weather",
                            "arguments": {"city": "BJ"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "get_weather",
                    "name": "get_weather",
                    "content": '{"t":20}',
                },
            ],
        ),
        None,
        None,
    )

    function_call = next(
        part.function_call for content in contents for part in content.parts if part.function_call
    )
    assert function_call.name == "get_weather"


def test_chat_tool_call_normalization_reads_input_alias():
    """Responses/Anthropic 用 ``input`` 承载参数，回填时不能静默变空。"""
    normalized = normalize_chat_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "custom_tool_call", "name": "n", "input": '{"a":1}'}
                ],
            }
        ]
    )

    assert normalized[0]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "n", "arguments": '{"a":1}'}}
    ]


@pytest.mark.parametrize(
    "tool_type",
    ["function", "function_call", "custom_tool_call", "tool_call"],
)
def test_chat_tool_call_normalization_maps_type_to_function(tool_type):
    """Chat Completions 只接受 ``type="function"``，其它来源取值都要映射。"""
    normalized = normalize_chat_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "x", "type": tool_type, "name": "n", "arguments": "{}"}],
            }
        ]
    )

    assert normalized[0]["tool_calls"][0]["type"] == "function"


def test_chat_completions_still_requires_tool_call_id():
    """Chat Completions 的严格校验不能被放宽，缺 id 必须显式报错。"""
    with pytest.raises(ValueError, match="缺少 id/call_id"):
        normalize_messages(
            None,
            None,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": None, "type": "function", "name": "n", "arguments": "{}"}
                    ],
                }
            ],
            require_tool_call_ids=True,
        )


def test_responses_input_still_requires_tool_call_id():
    """Responses 输入项需要 call_id，缺 id 仍要在请求前显式报错。"""
    with pytest.raises(ValueError, match="缺少 id/call_id"):
        normalize_responses_input(
            None,
            None,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": None, "type": "function", "name": "n", "arguments": "{}"}
                    ],
                }
            ],
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
        "claude-mythos-5-1",
        "claude-mythos-preview",
    ],
)
def test_anthropic_sampling_deprecation_covers_new_family_names(model_name):
    """5 代新增家族（fable/mythos）同样拒绝 legacy 采样控制字段。

    Python SDK v1.0+ 已从请求签名移除 temperature/top_p/top_k，识别失败会让
    请求直接抛 TypeError。家族名不能只匹配 opus/sonnet/haiku。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = model_name
    provider._options = {}

    assert provider._sampling_controls_deprecated(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-sonnet-4",
        "claude-opus-4",
    ],
)
def test_anthropic_sampling_deprecation_keeps_legacy_models_permissive(model_name):
    """4.5 之前的模型仍接受采样控制字段，不能被新家族规则误伤。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = model_name
    provider._options = {}

    assert provider._sampling_controls_deprecated(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    ["notclaude-sonnet-4-6", "xclaude-opus-5", "myclaude-fable-5"],
)
def test_anthropic_version_parser_requires_a_word_boundary(model_name):
    """``claude`` 必须是独立的名称段，不能被别的字符串前缀带出误判。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = model_name
    provider._options = {}

    assert provider._sampling_controls_deprecated(model_name) is False


# ── 回归：Ark Responses 非流式正文提取 ──


def _ark_response(output, *, status="completed"):
    """用真实 Ark SDK 模型构造响应，避免伪造线上不存在的字段。

    此前测试普遍使用 ``SimpleNamespace(output_text=...)``，但 Ark 的
    ``Response`` 没有 ``output_text``（OpenAI SDK 才把它实现为聚合 property），
    正文位于 ``output[].content[].text``。伪造该字段会让测试永远通过，掩盖
    真实环境下的空正文缺陷。
    """
    from volcenginesdkarkruntime.types.responses.response import Response

    return Response.model_validate(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "model": "ark-model",
            "status": status,
            "tools": [],
            "output": output,
        }
    )


def test_ark_responses_non_stream_extracts_text_from_output_content():
    """Ark 的 Response 没有 ``output_text`` 属性，正文必须从 output 内容块取。

    只读 ``output_text`` 会让所有非流式入口返回空文本且不报错，上层
    ``identify_intent`` 随后抛出「意图识别返回空结果」。
    """
    response = _ark_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "这是真实回答。", "annotations": []}],
            }
        ]
    )

    assert VolcengineProvider._extract_result(response).text == "这是真实回答。"


def test_ark_response_model_has_no_output_text_attribute():
    """固定 SDK 事实：Ark 的 Response 不提供 ``output_text``。

    该断言失败说明 SDK 改变了响应形状，届时可以重新评估提取逻辑。
    """
    from volcenginesdkarkruntime.types.responses.response import Response

    assert "output_text" not in Response.model_fields
    assert not hasattr(Response, "output_text")


def test_ark_responses_completed_response_without_any_content_fails_loudly():
    """标记 completed 却无正文、工具调用、拒答与推理时，必须显式失败。

    项目规则禁止用占位结果掩盖失败：静默返回空文本会让调用方拿到空串后
    继续执行，最终表现为难以定位的「空结果」错误。
    """
    with pytest.raises(RuntimeError, match="未包含任何正文"):
        VolcengineProvider._extract_result(_ark_response([]))


def test_ark_responses_reasoning_only_response_is_still_valid():
    """只有推理内容、没有正文的响应是合法的，不能被 fail-closed 误伤。"""
    response = _ark_response(
        [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Ark 思考"}],
            }
        ]
    )

    result = VolcengineProvider._extract_result(response)

    assert result.text == ""
    assert result.reasoning == "Ark 思考"


# ── 回归：动态资源树必须与显式 Facade 共用能力门禁 ──


def _ark_resources_without_verified_protocols(calls):
    """构造未登记 server_verified_protocols 的 Ark provider 与资源 Facade。"""
    from src.providers.volcengine import ArkResources

    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    provider._options = {}
    object.__setattr__(
        provider,
        "_client",
        SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: calls.append(kwargs) or "remote",
                retrieve=lambda **kwargs: calls.append(kwargs) or "remote",
            ),
            files=SimpleNamespace(create=lambda **kwargs: "local-file"),
        ),
    )
    return provider, ArkResources(provider)


def test_ark_dynamic_responses_resource_enforces_server_verification():
    """``resources.responses.create(...)`` 必须与 ``create_response()`` 同样受检。

    动态资源树此前只做凭证校验，绕过了 server_verified_protocols 门禁，会把
    本地 SDK 资源直接发往未验证的服务端。
    """
    calls: list[dict[str, object]] = []
    _, resources = _ark_resources_without_verified_protocols(calls)

    with pytest.raises(ValueError, match="未验证 Responses"):
        resources.responses.create(model="ark-model", input="hi")
    with pytest.raises(ValueError, match="未验证 Responses"):
        resources.responses.retrieve(response_id="resp-1")

    assert calls == []


def test_ark_dynamic_non_responses_resources_are_not_gated():
    """能力门禁只针对 responses 资源，files 等同名资源不能被误拦。"""
    calls: list[dict[str, object]] = []
    _, resources = _ark_resources_without_verified_protocols(calls)

    assert resources.files.create(file="f") == "local-file"


def test_ark_dynamic_responses_resource_passes_once_protocol_is_verified():
    """登记 server_verified_protocols 后动态路径应正常放行。"""
    calls: list[dict[str, object]] = []
    provider, resources = _ark_resources_without_verified_protocols(calls)
    provider._server_verified_protocols = frozenset({"responses"})

    assert resources.responses.create(model="ark-model", input="hi") == "remote"
    assert len(calls) == 1


def test_ark_resource_guard_keeps_credential_check_before_capability_check():
    """凭证问题必须优先报出，不能被能力错误掩盖成配置问题。"""
    calls: list[dict[str, object]] = []
    _, resources = _ark_resources_without_verified_protocols(calls)

    with pytest.raises(ValueError, match="凭证"):
        resources.responses.create(model="ark-model", input="hi", api_key="sk-secret")


# ── 回归：Ark instructions × caching 互斥必须覆盖所有构造路径 ──


def _ark_provider_for_request_build(options=None):
    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = options or {}
    return provider


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"prompt": "hi", "caching": {"type": "enabled"}}, id="top-level-caching"),
        pytest.param(
            {"prompt": "hi", "extra_body": {"caching": {"type": "enabled"}}},
            id="request-extra-body",
        ),
    ],
)
def test_ark_responses_rejects_instructions_with_enabled_caching_on_every_path(request_kwargs):
    """``caching`` 也可从 ``extra_body`` 进来，Ark SDK 会把它合并进请求体，
    服务端看到的仍是 ``caching=enabled``。只查顶层参数会漏掉这条路径。"""
    provider = _ark_provider_for_request_build()

    with pytest.raises(ValueError, match="互斥"):
        provider._build_responses_request(CompletionRequest(**request_kwargs))


def test_ark_responses_rejects_caching_enabled_via_model_options_extra_body():
    """模型级 options.extra_body 同样会被并入请求体。"""
    provider = _ark_provider_for_request_build({"extra_body": {"caching": {"type": "enabled"}}})

    with pytest.raises(ValueError, match="互斥"):
        provider._build_responses_request(CompletionRequest(prompt="hi"))


def test_ark_native_response_kwargs_reject_instructions_with_enabled_caching():
    """原生 create_response/async_create_response 走 kwargs 校验，也必须受检。"""
    with pytest.raises(ValueError, match="互斥"):
        VolcengineProvider._validate_native_response_kwargs(
            {
                "model": "ark-model",
                "input": "hi",
                "instructions": "system",
                "caching": {"type": "enabled"},
            }
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"model": "m", "input": "hi", "caching": {"type": "enabled"}}, id="no-instructions"
        ),
        pytest.param(
            {"model": "m", "input": "hi", "instructions": "s", "caching": {"type": "disabled"}},
            id="caching-disabled",
        ),
        pytest.param({"model": "m", "input": "hi", "instructions": "s"}, id="no-caching"),
    ],
)
def test_ark_native_response_kwargs_allow_non_conflicting_combinations(kwargs):
    """合法组合不能被互斥检查误伤。"""
    assert VolcengineProvider._validate_native_response_kwargs(kwargs)["model"] == "m"


# ── 回归：Ark 专属内容块不能经顶层 item 形态绕过 variant 守卫 ──


@pytest.mark.parametrize(
    "item",
    [
        {"type": "input_audio", "audio": "AAAA"},
        {"type": "input_video", "video": "BBBB"},
        {"type": "input_image", "image_url": "u", "image_pixel_limit": {"max_pixels": 100}},
    ],
)
def test_openai_responses_rejects_ark_only_blocks_as_top_level_items(item):
    """同一个 Ark 专属块写成 content 会被拒、写成顶层 item 却直通请求体，
    等于绕过了 variant 守卫。两种形态必须一致拒绝。"""
    with pytest.raises(ValueError):
        normalize_responses_input(None, None, [dict(item)], provider="openai")


def test_ark_provider_allows_ark_only_blocks_as_top_level_items():
    """Ark 渠道本身要能发这些块，守卫不能把合法用法一并拦下。"""
    converted = normalize_responses_input(
        None,
        None,
        [{"type": "input_audio", "audio_url": "https://example.invalid/a.mp3"}],
        provider="ark",
    )

    assert converted[0]["type"] == "input_audio"


# ── 回归：custom_tool_call 回填不能被强制 JSON 解析 ──


def test_responses_custom_tool_call_round_trip_preserves_free_form_input():
    """Responses custom tool 的 ``input`` 是任意文本（SDK 契约无 JSON 约束），
    强制 json.loads 会让多轮 custom tool 在请求边界直接失败。"""
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "custom_tool_call",
                        "id": "ctc_1",
                        "call_id": "call_abc",
                        "name": "my_custom_tool",
                        "arguments": "any free-form text, not JSON",
                    }
                ],
            }
        ],
        provider="openai",
    )

    assert converted[0]["type"] == "custom_tool_call"
    assert converted[0]["input"] == "any free-form text, not JSON"
    assert converted[0]["call_id"] == "call_abc"


def test_responses_custom_tool_call_accepts_empty_input():
    """空 input 是合法值（custom tool 无参数），不能被当成无效 JSON 拒绝。"""
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "custom_tool_call", "call_id": "c", "name": "f", "arguments": ""}
                ],
            }
        ],
        provider="openai",
    )

    assert converted[0]["input"] == ""


def test_responses_function_call_still_normalizes_json_arguments():
    """function_call 必须继续做 JSON 规范化，不能被 custom 分支影响。"""
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a":1}'}
                ],
            }
        ],
        provider="openai",
    )

    assert converted[0]["type"] == "function_call"
    assert converted[0]["arguments"] == '{"a":1}'


def test_chat_messages_do_not_leak_responses_only_type_marker():
    """Chat Completions 路径不能带上 Responses 专属的内部字段，否则它会随
    请求体发给 API。"""
    normalized = normalize_chat_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "custom_tool_call", "name": "n", "input": '{"a":1}'}
                ],
            }
        ]
    )

    assert normalized[0]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "n", "arguments": '{"a":1}'}}
    ]


def test_responses_round_trip_keeps_call_id_for_streamed_tool_calls():
    """流式事件同时给出 ``id``（输出项 ID）与 ``call_id``（调用 ID），两者不同。

    Responses 用 ``call_id`` 关联 ``function_call`` 与 ``function_call_output``，
    若 Chat 归一化只保留 ``id``，回填时输出项就配不上调用。
    """
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "fc_0123",
                        "call_id": "call_abc",
                        "type": "function_call",
                        "name": "f",
                        "arguments": "{}",
                    }
                ],
            }
        ],
        provider="openai",
    )

    assert converted[0]["call_id"] == "call_abc"
    assert converted[0]["id"] == "fc_0123"


def test_responses_function_call_output_pairs_with_preserved_call_id():
    """回填的 function_call 与 function_call_output 必须共享同一 call_id。"""
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "fc_0123",
                        "call_id": "call_abc",
                        "type": "function_call",
                        "name": "f",
                        "arguments": "{}",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "result"},
        ],
        provider="openai",
    )

    assert converted[0]["call_id"] == converted[1]["call_id"] == "call_abc"


# ── 回归：动态资源路径的门禁与参数校验 ──


def _ark_provider_with_recording_client(protocol="chat_completions", verified=()):
    """构造带记录型客户端的 Ark 资源 Facade，返回 (resources, 捕获列表)。

    用 ``object.__new__`` 绕过 ``__init__``：本组测试只关心门禁与参数校验的
    路径判断，不需要真实凭证，也不该依赖环境变量。
    """
    from src.providers.volcengine import ArkResources

    captured: list[tuple[str, dict]] = []
    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._model_name = "ark-model"
    provider._protocol = protocol
    provider._server_verified_protocols = frozenset(verified)
    provider._options = {}
    object.__setattr__(
        provider,
        "_client",
        SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: captured.append(("responses.create", kwargs)) or "remote",
            ),
            input_items=SimpleNamespace(
                list=lambda *args, **kwargs: captured.append(("input_items.list", kwargs)) or [],
            ),
            files=SimpleNamespace(
                create=lambda **kwargs: captured.append(("files.create", kwargs)) or "local"
            ),
        ),
    )
    return ArkResources(provider), captured


def test_ark_dynamic_input_items_resource_enforces_responses_gate():
    """``input_items`` 是 Responses 的子资源，路径分段不含 ``responses``
    （``volcengine.input_items.list`` 打的是 ``/responses/{id}/input_items``）。
    只看分段会让它在未验证 responses 的渠道上把请求发出去。
    """
    resources, captured = _ark_provider_with_recording_client(protocol="chat_completions")

    with pytest.raises(ValueError, match="responses 协议"):
        resources.input_items.list("resp_1")

    assert captured == []


def test_ark_dynamic_files_resource_is_not_gated_as_responses():
    """门禁不能误伤同名无关的资源：``files.create`` 与 Responses 无关。"""
    resources, captured = _ark_provider_with_recording_client(protocol="chat_completions")

    resources.files.create(file=("a.txt", b"x"), purpose="assistants")

    assert [name for name, _ in captured] == ["files.create"]


def test_ark_dynamic_responses_create_enforces_parameter_validation():
    """动态路径不经过 Facade，Facade 上的参数校验必须同等执行，否则
    ``resources.responses.create(...)`` 会带着互斥参数直接发出。"""
    resources, captured = _ark_provider_with_recording_client(
        protocol="responses", verified=["responses"]
    )

    with pytest.raises(ValueError, match="互斥"):
        resources.responses.create(
            model="ark-model", input="x", instructions="sys", caching={"type": "enabled"}
        )

    assert captured == []


def test_ark_dynamic_responses_create_requires_model():
    """原生调用必须显式提供 model，动态路径同样要拦。"""
    resources, captured = _ark_provider_with_recording_client(
        protocol="responses", verified=["responses"]
    )

    with pytest.raises(ValueError, match="model"):
        resources.responses.create(input="x")

    assert captured == []


def test_ark_dynamic_responses_create_allows_valid_request():
    """补了参数校验后，合法调用不能被误拦。"""
    resources, captured = _ark_provider_with_recording_client(
        protocol="responses", verified=["responses"]
    )

    resources.responses.create(model="ark-model", input="x")

    assert [name for name, _ in captured] == ["responses.create"]


# ── 回归：内置工具调用项是合法中间态，不能被 fail-closed 判为结构不符 ──


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "web_search_call",
            "id": "w1",
            "action": {"type": "search", "query": "q"},
            "status": "completed",
        },
        {"type": "mcp_call", "id": "c1", "name": "n", "arguments": "{}", "server_label": "s"},
        {"type": "mcp_list_tools", "id": "m1", "server_label": "s", "tools": []},
        {"type": "reasoning", "id": "r1", "summary": []},
    ],
)
def test_ark_builtin_tool_items_are_not_treated_as_malformed(item):
    """内置工具（web_search_call、mcp_call 等）的调用项既无正文也不是
    function_call，但都是 SDK 输出项联合的正式成员，属正常中间态——没有工具
    调用就不可能有后续轮次。判为「结构不符」会让这类响应无法处理。
    """
    response = _ark_response([item])

    result = VolcengineProvider._extract_result(response)

    assert result.text == ""


def test_ark_empty_completed_response_still_fails_loudly():
    """放宽内置工具项后，真正空白的 completed 响应仍必须显式失败。"""
    response = _ark_response([])

    with pytest.raises(RuntimeError, match="未包含任何正文"):
        VolcengineProvider._extract_result(response)


# ── 回归：非字符串 tool type 不能把「类型不合法」变成崩溃 ──


@pytest.mark.parametrize("bad_type", [["function"], {"a": 1}, ["custom_tool_call"]])
def test_chat_tool_call_item_accepts_non_string_type_without_crashing(bad_type):
    """``source_type in _CHAT_TOOL_CALL_TYPES`` 对 unhashable 值抛
    ``TypeError: unhashable type``，把可诊断的类型错误变成崩溃。"""
    from src.providers.__base__.model_provider import _chat_tool_call_item

    item = _chat_tool_call_item(
        {"id": "c1", "type": bad_type, "name": "n", "arguments": "{}"}, "loc"
    )

    assert item["type"] == "function"


# ── 回归：内部标记不能随请求体发出 ──


@pytest.mark.parametrize("role", [None, "user", "system", "developer"])
def test_responses_input_strips_internal_markers_for_non_assistant_roles(role):
    """``_responses_tool_call_type`` 是内部判定用的标记。assistant.tool_calls
    分支会消费它，但兜底分支原样复制消息，标记会随请求体发给服务端。
    """
    message = {
        "content": "go",
        "tool_calls": [{"id": "c1", "type": "custom_tool_call", "name": "n", "input": "raw"}],
    }
    if role is not None:
        message["role"] = role

    converted = normalize_responses_input(None, None, [message], provider="ark")

    assert "_responses_tool_call_type" not in json.dumps(converted)


def test_responses_input_keeps_custom_tool_semantics_for_assistant():
    """剥离内部标记不能影响 assistant 路径的 custom_tool_call 语义。"""
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "custom_tool_call", "name": "n", "input": "raw text"}
                ],
            }
        ],
        provider="ark",
    )

    assert converted[0]["type"] == "custom_tool_call"
    assert converted[0]["input"] == "raw text"
    assert "_responses_tool_call_type" not in json.dumps(converted)


# ── 回归：流式工具调用的 id 语义必须与非流式路径一致 ──


def test_responses_stream_tool_call_id_is_stable_across_delta_and_done():
    """``id`` 必须是调用标识符，与非流式路径（``call_id or id``）一致。

    取 ``item_id``（输出项 ID）会让同一轮工具调用在 delta 与 done 事件里得到
    不同的 id，调用方按 ``tool_call["id"]`` 回填 ``role=tool`` 时就会配不上。
    """
    metadata: dict = {}
    OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(
                type="custom_tool_call", id="item-1", call_id="call-1", name="run", input=""
            ),
        ),
        metadata,
    )

    delta = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.custom_tool_call_input.delta",
            item_id="item-1",
            output_index=1,
            delta="echo ",
        ),
        metadata,
    )
    done = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.custom_tool_call_input.done",
            item_id="item-1",
            output_index=1,
            input="hello world",
        ),
        metadata,
    )

    assert delta is not None and done is not None
    assert delta.tool_call["id"] == done.tool_call["id"] == "call-1"


def test_responses_function_call_item_does_not_duplicate_call_id_as_output_id():
    """归一化后的流式形状里 ``id == call_id``。把它当输出项 ID 发出，等于把
    调用标识符填进了 SDK 期望输出项标识符的位置。"""
    from src.providers.__base__.model_provider import _responses_function_call_item

    item = _responses_function_call_item(
        {
            "id": "call-1",
            "call_id": "call-1",
            "type": "function_call",
            "name": "n",
            "arguments": "{}",
        },
        "loc",
    )

    assert item["call_id"] == "call-1"
    assert "id" not in item


def test_ark_responses_public_entrypoints_return_text_from_output_content():
    """驱动公开入口，而不只是私有 ``_extract_result``。

    缺陷原文是「complete / acomplete / invoke(stream=False) /
    ainvoke(stream=False) 全部返回空文本」。只钉住私有助手的话，将来某个入口
    绕开 ``_extract_result`` 或丢弃其返回值，测试不会发现。
    """
    import asyncio

    response = _ark_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "这是真实回答。", "annotations": []}],
            }
        ]
    )

    def make_provider():
        provider = object.__new__(VolcengineProvider)
        provider._provider = "volcengine"
        provider._model_name = "ark-model"
        provider._protocol = "responses"
        provider._server_verified_protocols = frozenset({"responses"})
        provider._options = {}

        async def _acreate(**_kwargs):
            return response

        object.__setattr__(
            provider,
            "_client",
            SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: response)),
        )
        object.__setattr__(
            provider,
            "_aclient",
            SimpleNamespace(responses=SimpleNamespace(create=_acreate)),
        )
        return provider

    # 四个非流式公开入口。``invoke``/``ainvoke`` 的 ``stream=False`` 走
    # ``_extract_result`` 分支，正是此前返回空文本的那条路径。
    assert make_provider().complete(CompletionRequest(prompt="hi")).text == "这是真实回答。"
    assert list(make_provider().invoke(prompt="hi", stream=False)) == ["这是真实回答。"]

    async def _collect_ainvoke():
        return [chunk async for chunk in make_provider().ainvoke(prompt="hi", stream=False)]

    assert (
        asyncio.run(make_provider().acomplete(CompletionRequest(prompt="hi"))).text
        == "这是真实回答。"
    )
    assert asyncio.run(_collect_ainvoke()) == ["这是真实回答。"]
