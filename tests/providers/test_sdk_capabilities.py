import asyncio
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
from google.auth.credentials import AnonymousCredentials
from google.genai import types

from src.providers.__base__.model_provider import (
    CompletionRequest,
    LargeLanguageModel,
    StreamEvent,
    TextEmbeddingModel,
    _normalize_responses_native_item,
    _responses_content_part,
    normalize_chat_messages,
    normalize_messages,
    normalize_responses_input,
    raise_for_stream_error_event,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.factory import ModelProviderFactory
from src.providers.google import GoogleProvider
from src.providers.openai import OpenAIProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.resources import AsyncGoogleResources, GoogleResources
from src.providers.volcengine import (
    _ARK_BUILTIN_TOOL_ITEM_TYPES,
    VolcengineProvider,
    _require_ark_content,
    _require_ark_list,
)
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


# `_build_chat_request` 用三条「与模型 options.extra_body 重复」判据挡住同一字段
# 被两处配置：显式参数（`reasoning` 的 effort 分支与无 effort 分支各一条）与
# 请求级 `extra_body`。三条在整套测试里从未执行（`sys.settrace` 行覆盖证实；
# 把它们的 `raise` 换成 `pass` 后全量 3212 条仍然全绿）。删除后行为确实改变
# （实测）：
#   * 1167（`reasoning` 带 effort）与 1171（`reasoning` 无 effort）→ **静默接受**，
#     显式 `reasoning` 覆盖模型 options 里的 `reasoning`，两处配置谁生效取决于
#     调用顺序，没有任何提示；
#   * 1185（请求级 `extra_body` 与模型 options 的键重复）→ **静默覆盖**，
#     实测 `{"foo": 1}` 被 `{"foo": 2}` 顶掉。
# 同批的 1145（`seed` 重复）经甄别是**等价变异**：删除后仍抛 `ValueError`，
# 只是消息换成 `_validated_extra_body` 那条「不允许覆盖请求字段: seed」，结论不变，
# 故不补测试。
def _chat_request_provider(**options):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._protocol = "chat_completions"
    provider._provider = "openai"
    provider._options = options
    return provider


@pytest.mark.parametrize(
    "reasoning",
    [{"effort": "high"}, {"summary": "auto"}],
    ids=["with-effort", "without-effort"],
)
def test_openai_chat_request_rejects_reasoning_duplicating_model_options(reasoning):
    """显式 ``reasoning`` 与模型 options 里的 ``reasoning`` 重复时必须报错，不能静默覆盖。"""
    provider = _chat_request_provider(extra_body={"reasoning": {"configured": True}})

    with pytest.raises(ValueError, match="reasoning 与模型 options.extra_body 重复"):
        provider._build_chat_request(prompt="hi", stream=False, reasoning=reasoning)


def test_openai_chat_request_rejects_request_extra_body_overriding_model_options():
    """请求级 ``extra_body`` 与模型 options 的同名键重复时必须报错，不能静默覆盖。"""
    provider = _chat_request_provider(extra_body={"foo": 1})

    with pytest.raises(ValueError, match="extra_body 与模型 options 重复: foo"):
        provider._build_chat_request(prompt="hi", stream=False, extra_body={"foo": 2})


def test_openai_chat_request_keeps_distinct_reasoning_and_extra_body():
    """收紧不能误伤：不重复的 ``reasoning`` 与 ``extra_body`` 照常进请求。"""
    provider = _chat_request_provider(extra_body={"foo": 1})

    request = provider._build_chat_request(
        prompt="hi", stream=False, reasoning={"effort": "high"}, extra_body={"bar": 2}
    )

    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {"foo": 1, "bar": 2}


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


# `_build_responses_request` 与 `_build_chat_request` 同属一组「同一字段两处配置」
# 判据，下面五条在整套测试里从未执行（`sys.settrace` 行覆盖证实；把它们的 `raise`
# 换成 `pass` 后全量 3216 条仍然全绿）。删除后行为确实改变（实测）：
#   * 1407 `options.text` 非对象 → 带字段名的 `ValueError` 变成裸
#     `TypeError: 'int' object is not iterable`（`dict(configured_text)`）；
#   * 1456 显式 `frequency_penalty`/`presence_penalty` 与模型 options 的
#     `extra_body` 重复 → **静默接受**，两处配置谁生效取决于调用顺序；
#   * 1596 `background` 非布尔 → **静默接受**，`"yes"` 原样进请求体；
#   * 1598 `background=true` 配流式 → **静默接受**，把官方不支持的组合发出去；
#   * 1610 请求级 `extra_body` 与模型 options 的同名键重复 → **静默覆盖**。
# 同函数的 1415（`options.stop`/`seed` 与 extra_body 重复）与 1553
# （`reasoning_effort` 与 extra_body 重复）经甄别是**等价变异**，不补测试：
# 删除后仍抛 `ValueError`，只是消息换成 `_merge_extra_body`／`_validated_extra_body`
# 那条，结论不变。
def _responses_request_provider(**options):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = "https://gateway.example.com/v1"
    provider._options = options
    return provider


def test_openai_responses_rejects_non_object_options_text():
    """`options.text` 非对象时必须给带字段名的 ``ValueError``，不是裸 ``TypeError``。"""
    provider = _responses_request_provider(text=5)

    with pytest.raises(ValueError, match="options.text 必须是对象"):
        provider._build_responses_request(prompt="hi", stream=False)


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
def test_openai_responses_rejects_sampling_field_duplicating_model_extra_body(field):
    """显式采样字段与模型 options 的 ``extra_body`` 重复时必须报错，不能静默接受。"""
    provider = _responses_request_provider(extra_body={field: 0.5})

    with pytest.raises(ValueError, match=rf"{field} 与模型 options.extra_body 重复"):
        provider._build_responses_request(prompt="hi", stream=False, **{field: 0.5})


@pytest.mark.parametrize("background", ["yes", 1, [], {"a": 1}])
def test_openai_responses_rejects_non_boolean_background(background):
    """``background`` 必须是布尔值，不能把任意取值原样发给服务端。"""
    provider = _responses_request_provider()

    with pytest.raises(ValueError, match="background 必须是布尔值"):
        provider._build_responses_request(prompt="hi", stream=False, background=background)


def test_openai_responses_rejects_background_with_streaming():
    """``background=true`` 与流式互斥，必须在构造阶段拦住。"""
    provider = _responses_request_provider()

    with pytest.raises(ValueError, match="background 请求不支持流式模式"):
        provider._build_responses_request(prompt="hi", stream=True, background=True)


def test_openai_responses_request_extra_body_cannot_override_model_options():
    """请求级 ``extra_body`` 与模型 options 的同名键重复时必须报错，不能静默覆盖。"""
    provider = _responses_request_provider(extra_body={"foo": 1})

    with pytest.raises(ValueError, match="extra_body 与模型 options 重复: foo"):
        provider._build_responses_request(prompt="hi", stream=False, extra_body={"foo": 2})


def test_openai_responses_accepts_distinct_extra_body_and_background():
    """收紧不能误伤：不重复的 ``extra_body`` 与非流式 ``background`` 照常进请求。"""
    provider = _responses_request_provider(extra_body={"foo": 1})

    request = provider._build_responses_request(
        prompt="hi", stream=False, extra_body={"bar": 2}, background=True
    )

    assert request["extra_body"] == {"foo": 1, "bar": 2}
    assert request["background"] is True


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


# `_merge_tool_call` 有五条「分片指向多个调用／缺少稳定标识」的守卫在整套测试里
# 从未执行（`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3283 条
# 仍然全绿）。既有用例 `test_google_tool_call_merge_rejects_ambiguous_unidentified_
# parallel_calls` 命中的只是「先建一条、再补一条」这条路径，下面三条全绕开了。
# 删除后行为确实改变（实测）：
#   * 2303 同 index 已有两条 name 不同的记录 → **静默接受**，两个并行调用的参数
#     被并进同一条记录；
#   * 2341 同 id 已挂在两条不同 key 的记录上 → **静默接受**，同样合并；
#   * 2326 index 分片匹配到多条「无 index 且同名」的记录 → **静默接受**。
# 这三处的后果与源码注释写明的意图正相反：注释明确「绝不能从分片位置合成
# index——该位置每个 chunk 都会重置，会把并行的独立调用合并在一起」，而合并后
# 两次工具调用的参数会互相污染。
# 同批的 2316 与 2376 经甄别是**等价变异**，不补测试：删除后落到相邻守卫，仍抛
# `RuntimeError` 且消息仍指向同一类问题（实测分别是「index 指向多个调用」与
# 「无法确认是否为同一调用」），结论不变。
@pytest.mark.parametrize(
    ("existing", "fragment"),
    [
        pytest.param(
            [
                (("index", 0), {"type": "function", "index": 0, "name": "lookup"}),
                (("index", 0, "b"), {"type": "function", "index": 0, "name": "search"}),
            ],
            {"index": 0, "arguments": {"third": 3}},
            id="same-index-different-names",
        ),
        pytest.param(
            [
                (("id", "call-1"), {"type": "function", "id": "call-1", "name": "lookup"}),
                (("index", 7), {"type": "function", "index": 7, "id": "call-1", "name": "search"}),
            ],
            {"id": "call-1", "arguments": {"third": 3}},
            id="same-id-two-keys",
        ),
    ],
)
def test_google_tool_call_merge_rejects_fragments_pointing_at_multiple_calls(existing, fragment):
    """分片同时匹配多条已累积记录时必须 ``RuntimeError``，不能把两次调用合并成一条。"""
    # 手工构造累积状态：走 ``_merge_tool_call`` 建这两条时它们会被同名规则合并，
    # 构造不出「两条独立记录」这个前提。
    accumulator = {key: {**value, "arguments": {}} for key, value in existing}

    with pytest.raises(RuntimeError, match="指向多个调用"):
        GoogleProvider._merge_tool_call(accumulator, fragment)

    assert len(accumulator) == len(existing)


def test_google_tool_call_merge_rejects_index_fragment_matching_multiple_unidentified_records():
    """index 分片匹配到多条「无 index 且同名」的记录时必须报错，不能静默合并。"""
    # 同上，手工构造两条同名且都无 index 的记录，模拟上游把它们当作两次独立调用
    accumulator = {
        ("position", 0): {"type": "function", "index": None, "name": "lookup", "arguments": {}},
        ("position", 1): {"type": "function", "index": None, "name": "lookup", "arguments": {}},
    }

    with pytest.raises(RuntimeError, match="缺少稳定 id/index"):
        GoogleProvider._merge_tool_call(accumulator, {"index": 3, "arguments": {"third": 3}})

    assert len(accumulator) == 2


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


def test_ark_chat_completions_tool_calls_carry_type_key():
    """Ark 的 Chat Completions 分支必须产出契约形状的 tool_calls。

    ``CompletionResult.tool_calls`` 的契约（``model_provider.py:343`` 与
    ``docs/user_guide/llm-providers.md:108``）是扁平的
    ``{"id", "type", "name", "arguments"}``。此前 Ark 的 CC 分支漏掉
    ``type``，而 Responses 分支（同一函数内）却有——两条分支形状不一致。
    本测试直接钉住该键，并断言它与 OpenAI 兼容侧产出完全一致。
    """
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            # 故意用非 "function" 的 type：夹具若写 "function"，
                            # 硬编码 "function" 的变异体与透传 call.type 输出相同，
                            # 测试对「值是否真来自 SDK」零区分度。
                            type="custom_tool_call",
                            function=SimpleNamespace(name="lookup", arguments='{"id": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
        output=None,
        output_text=None,
    )

    result = VolcengineProvider._extract_result(response)

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    # 必须是 SDK 回传的原值，而不是被硬编码成 "function"
    assert call["type"] == "custom_tool_call"
    assert set(call) == {"id", "type", "name", "arguments"}

    # 与 OpenAI 兼容侧逐键对齐，防止两条路径再次分叉。
    oai_calls = OpenAICompatibleProvider._extract_tool_calls(response)
    assert call == oai_calls[0]

    # 交叉断言必须也覆盖「缺 type 时两侧各自的默认值」：夹具若显式给了 type，
    # 上面的对齐只验证透传路径，两条路径的 default 一旦分叉（例如一侧被改成
    # "function_MUTATED"）测试仍然全绿。
    missing_type = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="lookup", arguments='{"id": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
        output=None,
        output_text=None,
    )
    assert (
        VolcengineProvider._extract_result(missing_type).tool_calls[0]
        == OpenAICompatibleProvider._extract_tool_calls(missing_type)[0]
    )


def test_ark_chat_completions_tool_calls_default_type_to_function():
    """Ark 未回传 ``type`` 时也要补 ``function``，与 openai_compatible 一致。"""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-2",
                            function=SimpleNamespace(name="lookup", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
        output=None,
        output_text=None,
    )

    result = VolcengineProvider._extract_result(response)

    assert result.tool_calls[0]["type"] == "function"


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
    """不支持的 ``compaction_control`` 必须报「SDK 不支持」，不能误报成参数混用。

    删除守卫后 ``compaction_control`` 被透传给 ``coerce_completion_request``，
    消息变成「不能同时传 request 和关键字参数」（或「SDK 参数不匹配」）——把
    「Beta tool_runner 不支持这个能力」说成「你调用的方式不对」，用户改调用形态
    也解决不了。旧断言只匹配 ``compaction_control`` 子串，两条消息都含它，变异
    后仍然变绿（变异扫描证实），必须钉住「当前 SDK 不支持」前缀。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match="当前 SDK 不支持 compaction_control"):
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
    """Beta 入口必须报 Beta 的消息，不能退化成下游普通 ``create`` 的指引。

    删除 ``beta_create`` 的前置守卫后落到 ``_build_message_params`` 里的判据，
    消息变成「Anthropic Messages.create 不支持 output_format；请使用 parse 或
    stream。」——指引指向的 ``beta_parse``/``beta_stream`` 被换成普通入口，用户
    按提示改会再次撞墙。变异扫描证实宽松的 ``match="create 不支持"`` 在变异后
    仍然变绿（两条消息都含该子串）。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude"
    provider._options = {}

    with pytest.raises(ValueError, match=r"Beta Messages\.create 不支持 output_format"):
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

    # 异步侧是**另一处**守卫（async_beta_create 各自判一次），必须单独钉住 Beta
    # 消息：同步用例打不到这里，删除后同样退化成下游普通 ``create`` 的指引。
    with pytest.raises(ValueError, match=r"Beta Messages\.create 不支持 output_format"):
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


# `_normalize_sampling_value` 的三条「必须是数字」判据在整套测试里从未执行
# （`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3251 条仍全绿）。
# 既有用例 `test_anthropic_modern_models_omit_deprecated_sampling_defaults` 只喂
# **合法数字**（`temperature=0.2`／`top_k=5`），所以三条都绕开了。删除后行为确实
# 改变（实测）：
#   * 299 `temperature="abc"` → 消息从「Anthropic temperature 必须是数字。」换成
#     「当前 Claude 模型不支持该 temperature；请省略该字段或使用 1.0。」——把
#     **类型错误**说成**值域错误**，照着提示改的人会把 `abc` 换成 `1.0`，而那与
#     真正的问题无关；
#   * 309 `top_p="abc"` → `to_finite_float` 返回 `None`，`None >= 0.99` 抛裸
#     `TypeError: '>=' not supported between instances of 'NoneType' and 'float'`；
#   * 312 `top_p=0.5`（合法数字但低于阈值）→ 落到函数末尾的 top_k 分支，消息变成
#     「当前 Claude 模型不支持 top_k；请移除该字段。」——用户传的是 `top_p`，
#     报错却说 `top_k`，改错字段。
@pytest.mark.parametrize("value", ["abc", [1], {"a": 1}, object()])
def test_anthropic_modern_models_reject_non_numeric_temperature(value):
    """`temperature` 非数字时消息要直指「必须是数字」，不能说成「值不受支持」。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = {}

    with pytest.raises(ValueError, match="temperature 必须是数字"):
        provider._normalize_sampling_value("temperature", value)


@pytest.mark.parametrize("value", ["abc", [1], {"a": 1}, object()])
def test_anthropic_modern_models_reject_non_numeric_top_p(value):
    """`top_p` 非数字时给带字段名的 ``ValueError``，不是裸 ``TypeError``。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = {}

    with pytest.raises(ValueError, match="top_p 必须是数字"):
        provider._normalize_sampling_value("top_p", value)


def test_anthropic_modern_models_reject_low_top_p_with_the_right_field_name():
    """`top_p` 低于阈值时消息要指向 `top_p`，不能误报成 `top_k`。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = {}

    with pytest.raises(ValueError, match="该 top_p"):
        provider._normalize_sampling_value("top_p", 0.5)


# `_build_message_params` 的六条「配置与请求参数不冲突」判据在整套测试里从未执行
# （`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3260 条仍全绿）。
# 删除后行为确实改变（实测，全在 `src/providers/anthropic.py`）：
#   * 916 `allow_deprecated_sampling="yes"` → **静默接受**，非空字符串为真，等同于
#     显式打开「转发已废弃采样字段」，与调用方写 `False` 的意图正好相反；
#   * 920 `options.response_format` 非对象 → 带字段名的 `ValueError` 变成裸
#     `TypeError: 'int' object is not iterable`；
#   * 922／930 `response_format`／`output_format` 同时出现在 options 与请求参数 →
#     **静默接受**，请求侧取值被丢弃，两处配置冲突无从察觉；
#   * 962 请求级 `extra_body` 与模型 options 的 `extra_body` 有同名键 → **静默覆盖**；
#   * 995 `extra_body` 的键与已建好的请求字段（如 `model`）重名 → **静默接受**，
#     该字段会被 `extra_body` 覆盖掉再发给服务端。
def _anthropic_options_provider(**options):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    provider._options = options
    return provider


@pytest.mark.parametrize("value", ["yes", 1, 0, [], {}])
def test_anthropic_rejects_non_boolean_allow_deprecated_sampling(value):
    """`allow_deprecated_sampling` 必须是布尔值，不能靠真值语义静默生效。"""
    provider = _anthropic_options_provider(allow_deprecated_sampling=value)

    with pytest.raises(ValueError, match="allow_deprecated_sampling 必须是布尔值"):
        provider._build_message_params(prompt="hi")


@pytest.mark.parametrize("value", [5, "json_object", ["a"]])
def test_anthropic_rejects_non_object_options_response_format(value):
    """`options.response_format` 非对象时给带字段名的 ``ValueError``，不是裸 ``TypeError``。"""
    provider = _anthropic_options_provider(response_format=value)

    with pytest.raises(ValueError, match="response_format 必须是对象"):
        provider._build_message_params(prompt="hi")


@pytest.mark.parametrize("field", ["response_format", "output_format"])
def test_anthropic_rejects_field_configured_in_both_options_and_request(field):
    """同一字段同时配在 options 与请求参数时必须报错，不能静默丢掉一侧。"""
    request_value = {"type": "json_object"} if field == "response_format" else str
    provider = _anthropic_options_provider(**{field: request_value})

    with pytest.raises(ValueError, match=rf"{field} 与配置项重复"):
        provider._build_message_params(prompt="hi", stream=True, **{field: request_value})


def test_anthropic_rejects_request_extra_body_overriding_configured_extra_body():
    """请求级 `extra_body` 与模型 options 的 `extra_body` 同名键必须报错，不能静默覆盖。"""
    provider = _anthropic_options_provider(extra_body={"temperature": 0.5})

    with pytest.raises(ValueError, match="extra_body 不允许覆盖配置字段: temperature"):
        provider._build_message_params(prompt="hi", extra_body={"temperature": 0.9})


def test_anthropic_rejects_extra_body_overriding_a_request_field():
    """`extra_body` 的键与已建好的请求字段重名时必须报错，不能静默顶掉该字段。"""
    provider = _anthropic_options_provider(extra_body={"model": "other-model"})

    with pytest.raises(ValueError, match="extra_body 不允许覆盖请求字段: model"):
        provider._build_message_params(prompt="hi")


def test_anthropic_accepts_distinct_configured_and_request_values():
    """收紧不能误伤：不冲突的 options／请求参数组合照常构造请求。"""
    provider = _anthropic_options_provider(extra_body={"metadata": {"team": "search"}})

    params = provider._build_message_params(
        prompt="hi", response_format={"type": "json_object"}, extra_body={"seed": 7}
    )

    assert params["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }
    assert params["extra_body"] == {"metadata": {"team": "search"}, "seed": 7}


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


def test_call_sdk_resource_contracts_signature_mismatch_to_valueerror():
    """签名 ``bind`` 失败必须收敛成带渠道名的 ``ValueError``，不能透出裸 ``TypeError``。

    删除守卫后未知关键字参数直接打到 ``method(...)``，抛的是裸
    ``TypeError: ... unexpected keyword argument``——与本模块承诺的
    「SDK 参数不匹配」错误分类不一致。变异扫描证实该守卫经
    ``_call_sdk_resource`` 真实转发路径没有反向覆盖。
    """

    def method(a, b):
        return a, b

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"

    with pytest.raises(ValueError, match=r"chat\.create SDK 参数不匹配: "):
        provider._call_sdk_resource(method, (), {"zzz": 1}, "chat.create")

    # 收紧不能误伤：签名匹配的调用照常转发
    assert provider._call_sdk_resource(method, (), {"a": 1, "b": 2}, "chat.create") == (1, 2)


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


# ``_guard_provider_resource_method`` 把 SDK 抛出的 ``TypeError`` 收敛成带渠道名
# 与原始消息的 ``ValueError``，分三个子类：unexpected keyword / multiple values /
# missing required；同步与异步各有一份（2016/2020/2024 与 2046/2048/2050）。
# 既有用例只钉住**同步**的「multiple values」那一条，其余五条在整套测试里从未
# 执行（行覆盖证实；把它们换成 ``pass`` 后全量 3397 条仍全绿）。
#
# 删除后收场从 ``ValueError`` 退化成裸 ``TypeError``：调用方的错误分类按
# ``ValueError`` 写，而 SDK 的 ``TypeError`` 文本只说「哪个参数不对」，没有
# 渠道名——排查时无法定位到是哪个 provider 的资源方法。


def _resource_guard_sync_provider(sdk_call):
    class Interactions:
        def get(self, *, id, **kwargs):
            return sdk_call(id=id, **kwargs)

    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._provider = "test"
    provider._get_client = lambda: SimpleNamespace(interactions=Interactions())
    return provider


def _resource_guard_async_provider(sdk_call):
    class Caches:
        async def get(self, *, name, **kwargs):
            return sdk_call(name=name, **kwargs)

    class Aio:
        caches = Caches()

    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._provider = "test"
    provider._get_client = lambda: SimpleNamespace(aio=Aio())
    return provider


def _unexpected_keyword(**_kwargs):
    raise TypeError("get() got an unexpected keyword argument 'tenant'")


def _duplicate_argument(id=None, **_kwargs):
    raise TypeError("get() got multiple values for argument 'id'")


def _missing_required_argument(**_kwargs):
    raise TypeError("get() missing 1 required positional argument: 'name'")


_RESOURCE_GUARD_CASES = [
    pytest.param(_unexpected_keyword, "不匹配", id="unexpected-keyword"),
    pytest.param(_duplicate_argument, "重复", id="duplicate-argument"),
    pytest.param(_missing_required_argument, "不匹配", id="missing-required"),
]


@pytest.mark.parametrize(("sdk_call", "expected"), _RESOURCE_GUARD_CASES)
def test_resource_guard_reports_sync_type_errors_readably(sdk_call, expected):
    """同步资源入口的 ``TypeError`` 必须收敛成带渠道名的 ``ValueError``。"""
    provider = _resource_guard_sync_provider(sdk_call)

    with pytest.raises(ValueError, match=f"test 资源 SDK 参数{expected}"):
        provider.get_interaction("interaction-1", extra_body={"tenant": "t"})


@pytest.mark.parametrize(("sdk_call", "expected"), _RESOURCE_GUARD_CASES)
def test_resource_guard_reports_async_type_errors_readably(sdk_call, expected):
    """异步资源入口是**另一份包装器**，三条判据同样必须生效。

    同步侧的用例打不到这里——``async_wrapper`` 与 ``wrapper`` 是两个独立的闭包。
    """
    provider = _resource_guard_async_provider(sdk_call)

    async def run():
        return await provider.async_get_cache("caches/1", extra_body={"tenant": "t"})

    with pytest.raises(ValueError, match=f"test 资源 SDK 参数{expected}"):
        asyncio.run(run())


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_resource_guard_keeps_unrelated_type_errors_intact(async_mode):
    """反向守卫：与参数无关的 ``TypeError`` 必须原样透出，不能被误分类。

    守卫按消息内容分类；一条与 SDK 参数无关的 ``TypeError`` 若被收敛成
    「资源 SDK 参数不匹配」，排查方向会被带偏。
    """

    def unrelated(**_kwargs):
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

    if async_mode:
        provider = _resource_guard_async_provider(unrelated)
        coroutine = provider.async_get_cache("caches/1", extra_body={"tenant": "t"})
        with pytest.raises(TypeError, match="unsupported operand"):
            asyncio.run(coroutine)
    else:
        provider = _resource_guard_sync_provider(unrelated)
        with pytest.raises(TypeError, match="unsupported operand"):
            provider.get_interaction("interaction-1", extra_body={"tenant": "t"})


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


@pytest.mark.parametrize("bad_tool_call", [5, "oops", ["x"]])
def test_chat_tool_call_normalization_rejects_non_mapping_entries(bad_tool_call):
    """``tool_calls`` 的元素必须是对象，非 Mapping 不能走到 ``dict()`` 或迭代。

    删除该守卫后，整数元素抛裸 ``TypeError: 'int' object is not iterable``，
    字符串元素抛 ``dictionary update sequence element #0 has length 1``——
    两者都与「工具调用写错了」无关，且 ``TypeError`` 会穿透只捕获 ``ValueError``
    的调用方。
    """
    with pytest.raises(ValueError, match=r"必须是对象"):
        normalize_chat_messages(
            [{"role": "assistant", "content": "", "tool_calls": [bad_tool_call]}]
        )


def test_chat_tool_call_normalization_rejects_non_mapping_function_field():
    """嵌套 ``function`` 非对象时必须报错，不能静默改用顶层同名字段。

    输入里顶层 ``name``/``arguments`` 都合法，只有 ``function`` 是字符串。
    删除守卫后本函数会退回 ``normalized`` 取值并正常返回，把一个明显畸形的
    工具调用当作有效请求发出。
    """
    with pytest.raises(ValueError, match=r"\.function 必须是对象"):
        normalize_chat_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "name": "n",
                            "arguments": "{}",
                            "function": "bad",
                        }
                    ],
                }
            ]
        )


def test_responses_function_call_item_rejects_non_mapping_function_field():
    """Responses 原生 ``function_call`` 的嵌套 ``function`` 同样要校验。

    这条守卫与 ``_chat_tool_call_item`` 里的是**两份独立代码**（969 与 384），
    Chat 路径的测试杀不掉它。删除后原生 item 分支静默接受，把字符串 ``function``
    丢弃、改用顶层字段拼出请求。
    """
    with pytest.raises(ValueError, match=r"\.function 必须是对象"):
        normalize_responses_input(
            None,
            None,
            [
                {
                    "role": "assistant",
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "n",
                    "arguments": "{}",
                    "function": "bad",
                }
            ],
        )


def test_responses_function_call_item_requires_function_name():
    """原生 ``function_call`` 缺函数名必须在请求前报错。

    删除守卫后返回 ``name=None`` 的 item，把无名调用发给 Responses。
    """
    with pytest.raises(ValueError, match="缺少函数名"):
        normalize_responses_input(
            None,
            None,
            [
                {
                    "role": "assistant",
                    "type": "function_call",
                    "call_id": "c1",
                    "arguments": "{}",
                }
            ],
        )


@pytest.mark.parametrize(
    "message",
    [
        {
            "role": "assistant",
            "type": "function_call",
            "call_id": "c1",
            "name": "n",
            "arguments": {"k": object()},
        },
        {"type": "function_call_output", "call_id": "c1", "output": {"k": object()}},
    ],
)
def test_responses_json_text_rejects_unserializable_payload(message):
    """工具参数与输出必须可 JSON 序列化，否则 ``json.dumps`` 会抛裸 ``TypeError``。

    删除守卫后异常不再收敛为带 ``location`` 的 ``ValueError``；实测在
    ``_responses_json_text`` 内部先撞 ``TypeError``，调用方只捕获 ``ValueError``
    时无法给出「哪个字段写错了」的消息。
    """
    with pytest.raises(ValueError, match="必须可序列化为 JSON"):
        normalize_responses_input(None, None, [message])


def test_validate_complete_tool_call_requires_name_in_stream_events():
    """流式工具调用在发出 completed 事件前必须带函数名。

    ``validate_complete_tool_call`` 的 name 判据是流式两条路径（Chat Completions
    与 Responses）共用的第一道闸。删除后无名调用被当作完整事件交付。
    """
    finish = StreamEvent(type="finish", finish_reason="tool_calls", usage={})
    with pytest.raises(RuntimeError, match="缺少 name"):
        OpenAICompatibleProvider._chat_completion_events(
            finish,
            {0: {"id": "t1", "type": "function", "arguments": "{}"}},
            set(),
        )
    with pytest.raises(RuntimeError, match="缺少 name"):
        OpenAICompatibleProvider._responses_completion_event(
            StreamEvent(
                type="tool_call_completed",
                tool_call={"id": "t1", "type": "function", "arguments": "{}"},
            ),
            {},
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
    等于绕过了 variant 守卫。两种形态必须一致拒绝。

    断言带上 ``match``：只查异常类型的话，被测代码因别的原因抛 ``ValueError``
    也会让测试通过。
    """
    with pytest.raises(ValueError, match="不受|缺少|必须"):
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
        {"type": "agent_tool_call", "id": "a1", "name": "agent", "status": "completed"},
    ],
)
def test_ark_builtin_tool_items_are_not_treated_as_malformed(item):
    """内置工具（web_search_call、mcp_call 等）的调用项既无正文也不是
    function_call，但都是 SDK 输出项联合的正式成员，属正常中间态——没有工具
    调用就不可能有后续轮次。判为「结构不符」会让这类响应无法处理。

    ``reasoning`` 不在豁免之列：它不是工具调用，本轮有摘要时走 ``reasoning``
    字段放行；空摘要且无正文、无工具调用时确实什么都没有，报错才对。
    """
    response = _ark_response([item])

    result = VolcengineProvider._extract_result(response)

    assert result.text == ""


def test_ark_empty_reasoning_item_is_not_a_silent_success():
    """空摘要的 reasoning 项不构成有效响应。

    把它列入豁免会让「completed 但什么都没有」静默返回空成功，掩盖失败
    （项目规则禁止）。
    """
    response = _ark_response([{"type": "reasoning", "id": "r1", "summary": []}])

    with pytest.raises(RuntimeError, match="未包含任何正文"):
        VolcengineProvider._extract_result(response)


def test_ark_builtin_item_whitelist_covers_sdk_union():
    """白名单必须覆盖 SDK 输出项联合的全部成员（除已单独处理的）。

    手工维护的清单会随 SDK 演进漏掉新成员，那类响应会突然无法处理——这正是
    ``image_process``、``agent_tool_call`` 曾经的状态。这里对差集断言。
    """
    import typing

    from volcenginesdkarkruntime.types.responses.response import ResponseOutputItem

    def flatten(union, seen=None):
        seen = seen or set()
        found = []
        for member in typing.get_args(union):
            if member in seen:
                continue
            seen.add(member)
            if typing.get_origin(member) is typing.Union:
                found.extend(flatten(member, seen))
            else:
                found.append(member)
        return found

    union_types = set()
    for member in flatten(ResponseOutputItem):
        type_field = getattr(member, "model_fields", {}).get("type")
        if type_field is None:
            continue
        literals = typing.get_args(type_field.annotation)
        if literals:
            union_types.add(literals[0])

    # ``message`` 由正文提取处理，``function_call`` 归入 tool_calls，
    # ``reasoning`` 有摘要时经 reasoning 字段放行。
    handled = {"message", "function_call", "reasoning"}

    assert union_types - handled - _ARK_BUILTIN_TOOL_ITEM_TYPES == set()


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


# ── 回归：顶层 item 形态不能绕过内容块的变体校验 ──


@pytest.mark.parametrize(
    "message",
    [
        {"type": "input_audio", "audio_url": "u"},
        # 加一个无关的 ``content`` 键就能改走消息分支，从而绕过顶层校验。
        {"type": "input_audio", "audio_url": "u", "content": "x"},
        {"role": "user", "content": "x", "type": "input_audio", "audio_url": "u"},
        {"type": "input_image", "image_url": {"url": "u", "image_pixel_limit": {"m": 1}}},
    ],
)
def test_openai_responses_rejects_ark_only_blocks_with_or_without_content_key(message):
    """Ark 专属块写成顶层 item 时同样要拒。

    分流若以「有没有 ``content`` 键」为依据，攻击者加一个无关的 ``content``
    键即可改走另一条路径，绕过全部变体校验。
    """
    with pytest.raises(ValueError, match="不受|缺少"):
        normalize_responses_input(None, None, [message], provider="openai")


@pytest.mark.parametrize(
    "message",
    [
        {"type": "input_audio", "content": "x"},
        {"type": "input_file", "filename": "a.txt"},
        {"type": "input_video", "video_url": "u", "fps": "fast"},
    ],
)
def test_ark_responses_validates_native_item_required_fields(message):
    """顶层 item 形态也要做必填字段与取值范围校验，不能只查类型。

    断言带上 ``match``，把「必填字段/取值校验报的错」与「其它 ValueError」
    区分开。
    """
    with pytest.raises(ValueError, match="缺少|必须|不受"):
        normalize_responses_input(None, None, [message], provider="ark")


@pytest.mark.parametrize(
    "message",
    [
        {"type": "input_audio", "audio_url": "https://x/a.mp3"},
        {"type": "input_video", "video_url": "https://x/v.mp4", "fps": 2},
        {
            "type": "input_image",
            "image_url": "https://x/i.png",
            "image_pixel_limit": {"max_pixels": 100},
        },
    ],
)
def test_ark_responses_still_accepts_valid_native_items(message):
    """补全校验后，Ark 合法块不能被误拦。"""
    converted = normalize_responses_input(None, None, [message], provider="ark")

    assert converted[0]["type"] == message["type"]


# ── 回归：instructions 与 caching 的互斥检查要覆盖 extra_body ──


@pytest.mark.parametrize(
    "request_kwargs",
    [
        # instructions 在 extra_body、caching 在顶层
        {
            "prompt": "hi",
            "system_prompt": None,
            "extra_body": {"instructions": "sys"},
            "caching": {"type": "enabled"},
        },
        # 顶层 instructions、caching 在 extra_body
        {
            "prompt": "hi",
            "system_prompt": "sys",
            "extra_body": {"caching": {"type": "enabled"}},
        },
        # 两者都在 extra_body
        {
            "prompt": "hi",
            "system_prompt": None,
            "extra_body": {"instructions": "sys", "caching": {"type": "enabled"}},
        },
    ],
)
def test_ark_responses_mutual_exclusion_covers_extra_body(request_kwargs):
    """``instructions`` 与 ``caching`` 各有顶层和 ``extra_body`` 两条来源。

    SDK 会把 ``extra_body`` 合并进请求体，服务端看到的参数与顶层写法相同，
    因此只查顶层会漏掉这些组合。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    with pytest.raises(ValueError, match="互斥"):
        provider._build_responses_request(CompletionRequest(stream=False, **request_kwargs))


def test_ark_responses_mutual_exclusion_allows_either_alone():
    """互斥检查不能误伤只出现一个的合法请求。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    only_instructions = provider._build_responses_request(
        CompletionRequest(prompt="hi", system_prompt="sys", stream=False)
    )
    only_caching = provider._build_responses_request(
        CompletionRequest(
            prompt="hi", system_prompt=None, caching={"type": "enabled"}, stream=False
        )
    )

    assert only_instructions["instructions"] == "sys"
    assert only_caching["caching"] == {"type": "enabled"}


# ── 回归：custom tool 的结构化 input 编码要与 assistant 路径一致 ──


@pytest.mark.parametrize(
    "value,expected",
    [({"a": 1}, '{"a":1}'), ([1, 2], "[1,2]"), ("plain text", "plain text")],
)
def test_responses_custom_tool_input_uses_json_encoding(value, expected):
    """``str()`` 对 dict/list 产出 Python repr（单引号），而同一份数据经
    ``assistant.tool_calls`` 路径会被 JSON 编码，两条路径给出不同文本。"""
    from src.providers.__base__.model_provider import _responses_function_call_item

    item = _responses_function_call_item(
        {"id": "c1", "type": "custom_tool_call", "name": "n", "input": value}, "loc"
    )

    assert item["input"] == expected


def test_responses_custom_tool_input_matches_assistant_path_encoding():
    """两条路径对同一输入必须给出同一文本。"""
    from src.providers.__base__.model_provider import _responses_function_call_item

    payload = {"nested": {"a": 1}, "list": [1, 2]}

    via_item = _responses_function_call_item(
        {"id": "c1", "type": "custom_tool_call", "name": "n", "input": payload}, "loc"
    )
    via_assistant = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "custom_tool_call", "name": "n", "input": payload}
                ],
            }
        ],
        provider="ark",
    )

    assert via_item["input"] == via_assistant[0]["input"]


def test_ark_responses_output_text_supports_mapping_responses():
    """``field()`` 同时支持属性访问与 Mapping。dict 形态响应（非标准网关或
    中间层透传的 JSON）也要走同一条提取路径与类型过滤。
    """
    from src.providers.volcengine import _ark_responses_output_text

    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "正文"},
                    {"type": "refusal", "refusal": "拒绝内容"},
                ],
            },
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "思考"}]},
        ]
    }

    # 只取 output_text 块：refusal 与 reasoning 不能混进正文。
    assert _ark_responses_output_text(response) == "正文"


def test_ark_responses_native_item_proxy_hides_implementation_slots():
    """``__slots__`` 里的名字默认出现在 ``dir()`` 中，IDE 会把 ``_value``
    提示成公共 API；而真正的出口是文档化的 ``.native``。
    """
    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    provider._options = {}
    object.__setattr__(
        provider,
        "_client",
        SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: "remote")),
    )

    proxy = provider.resources.responses

    for internal in ("_value", "_path", "_provider", "_children"):
        assert internal not in dir(proxy), internal
    # 隐藏的是自动补全，不是访问能力。
    assert proxy._value is provider._client.responses


# ── 回归：资源代理不能经私有名转发到底层 SDK ──


def test_native_resource_proxy_blocks_private_attribute_forwarding():
    """``__slots__`` 封闭了自有 ``__dict__``，但 ``__getattr__`` 会把私有名
    转发给底层 SDK 节点——``proxy.__dict__["_client"]`` 能拿到未包装的原始
    客户端，绕开全部凭证扫描与能力门禁。出口应是文档化的 ``.native``。
    """
    import httpx
    from volcenginesdkarkruntime import Ark

    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._model_name = "ark-model"
    provider._protocol = "chat_completions"
    provider._server_verified_protocols = frozenset()
    provider._options = {}
    object.__setattr__(
        provider,
        "_client",
        Ark(
            api_key="test-key",
            base_url="https://ark.example.invalid/api/v3",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200, json={"id": "r", "status": "completed", "output": []}
                    )
                )
            ),
        ),
    )

    proxy = provider.resources.responses

    for private in ("__dict__", "_client", "_post", "_get", "_delete"):
        with pytest.raises(AttributeError):
            getattr(proxy, private)
    # 文档化的出口仍可用。
    assert provider.resources.native is provider._client


def test_native_resource_proxy_dir_lists_only_public_resource_names():
    """``dir()`` 不能提示可直达原始 SDK 的私有通路。"""
    provider = object.__new__(VolcengineProvider)
    provider._provider = "volcengine"
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset()
    provider._options = {}
    object.__setattr__(
        provider,
        "_client",
        SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: "remote", retrieve=lambda **_k: "r")
        ),
    )

    listed = dir(provider.resources.responses)

    assert not [name for name in listed if name.startswith("_")], listed
    assert "create" in listed


# ── 回归：caching 取值的大小写与空白变体 ──


@pytest.mark.parametrize(
    "caching",
    [
        {"type": "enabled"},
        {"type": "ENABLED"},
        {"type": "Enabled"},
        {"type": " enabled"},
        {"type": "  ENABLED  "},
    ],
)
def test_ark_responses_mutual_exclusion_normalizes_caching_type(caching):
    """SDK 的 ``ResponseCaching.type`` 是类型注解、不做运行时校验，
    ``"ENABLED"`` 会原样发到服务端。精确比较会让大小写变体绕过互斥检查。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    with pytest.raises(ValueError, match="互斥"):
        provider._build_responses_request(
            CompletionRequest(prompt="hi", system_prompt="sys", caching=caching, stream=False)
        )


@pytest.mark.parametrize("caching", [{"type": "disabled"}, {"type": "DISABLED"}, {}])
def test_ark_responses_mutual_exclusion_allows_non_enabled_caching(caching):
    """未启用 caching 时 instructions 正常透传。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    params = provider._build_responses_request(
        CompletionRequest(prompt="hi", system_prompt="sys", caching=caching, stream=False)
    )

    assert params["instructions"] == "sys"


# ── 回归：动态路径的参数校验不按动词白名单 ──


def test_ark_dynamic_responses_validation_is_not_verb_based():
    """按 ``{"create", "generate"}`` 判断会漏掉 ``async_create`` 这类 SDK
    演进后新增的写法；应按参数里是否出现受校验字段判断。"""
    resources, captured = _ark_provider_with_recording_client(
        protocol="responses", verified=["responses"]
    )
    # 模拟 SDK 演进后新增的动词：按动词白名单判断会漏掉它。
    resources.provider._client.responses.async_create = (  # type: ignore[attr-defined]
        lambda **kwargs: captured.append(("responses.async_create", kwargs)) or "remote"
    )

    with pytest.raises(ValueError, match="互斥"):
        resources.responses.async_create(
            model="ark-model", input="x", instructions="sys", caching={"type": "enabled"}
        )

    assert captured == []


# ── 回归：动态路径的能力映射要覆盖全部能力 ──


@pytest.mark.parametrize(
    ("path", "capability"),
    [
        ("deepseek.batches.create", "batches"),
        ("deepseek.vector_stores.create", "vector_stores"),
        ("deepseek.images.generate", "images"),
        ("deepseek.containers.create", "containers"),
        ("deepseek.audio.speech.create", "audio"),
        ("deepseek.files.create", "files"),
        # uploads 与 files 是两项独立能力，不能坍缩。
        ("deepseek.uploads.create", "uploads"),
    ],
)
def test_openai_compatible_dynamic_path_resolves_capability(path, capability):
    """只覆盖 responses/files 会让其余动态路径拿到 ``None`` 直接放行，而显式
    Facade 名会被拦——同一能力因书写形式不同而区别对待。"""
    assert OpenAICompatibleProvider._resource_capability_for_path(path) == capability


def test_openai_compatible_dynamic_path_moderations_matches_explicit_name():
    """SDK 客户端的属性是复数 ``moderations``，而显式 Facade 名
    ``create_moderation`` 解析出的能力名是单数。段映射只写单数时
    ``resources.moderations.create`` 拿到 ``None`` 直接放行，同一能力因
    书写形式不同而区别对待。
    """
    assert OpenAICompatibleProvider._resource_capability_for_path(
        "deepseek.moderations.create"
    ) == ("moderation")
    assert OpenAICompatibleProvider._resource_capability_for_method("create_moderation") == (
        "moderation"
    )


def test_openai_compatible_segment_map_stays_within_declared_capabilities():
    """段映射给出声明集之外的能力名，会把官方端点从「放行」变成
    ``NotImplementedError``——而 ``OpenAIProvider.capabilities`` 里那些资源
    （skills/realtime/webhooks/admin/content_provenance_checks）声明为可用，
    用户看到可用、调用却被拦。
    """
    declared = OpenAICompatibleProvider._OFFICIAL_OPENAI_RESOURCE_CAPABILITIES
    # ``responses`` 走 ``_require_responses_resource`` 专用分支，不受声明集约束。
    extra = set(OpenAICompatibleProvider._RESOURCE_SEGMENTS_TO_CAPABILITY.values()) - set(declared)
    assert extra == {"responses"}, extra


@pytest.mark.parametrize(
    "path",
    [
        "openai.skills.list",
        "openai.realtime.client_secrets.create",
        "openai.webhooks.list",
        "openai.admin.api_keys.list",
        "openai.content_provenance_checks.create",
    ],
)
def test_official_openai_declared_resources_are_not_blocked_by_dynamic_path(path):
    """``capabilities`` 声明这些资源可用，动态路径就不能拦下它们。"""
    provider = object.__new__(OpenAIProvider)
    provider._provider = "openai"
    provider._require_provider_resource(path)


def test_ark_dynamic_retrieve_with_extra_body_is_not_treated_as_create():
    """``extra_body`` 是 retrieve/delete/list 的合法参数，不该触发创建校验。

    把它放进触发集会让 ``resources.responses.retrieve("r1", extra_body={...})``
    被要求提供 ``model`` 与 ``input``，而同样的调用走 Facade 是通过的——
    动态路径与 Facade 的新不对称。
    """
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "responses"
    provider._server_verified_protocols = ("responses",)

    provider._require_provider_resource("volcengine.responses.retrieve", {"extra_body": {"a": 1}})
    provider._require_provider_resource("volcengine.responses.delete", {"extra_body": {"a": 1}})


def test_ark_dynamic_create_still_rejects_conflicting_instructions_and_caching():
    """去掉 ``extra_body`` 触发器不能削弱创建路径的互斥校验。"""
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "responses"
    provider._server_verified_protocols = ("responses",)

    with pytest.raises(ValueError, match="instructions"):
        provider._require_provider_resource(
            "volcengine.responses.create",
            {"model": "m", "input": "x", "instructions": "sys", "caching": {"type": "ENABLED"}},
        )


@pytest.mark.parametrize(
    "path",
    [
        "google.interactions.create",
        "google.agents.list",
        "google.webhooks.create",
        "google.environments.get",
        "google.triggers.list",
    ],
)
def test_google_dynamic_experimental_paths_are_gated(path):
    """动态路径是点分形式，资源名是独立分段；只按 ``startswith``/``_singular``
    匹配会让所有动态路径落到 ``None`` 直接放行——用户拿一个不含该资源的客户端
    走 ``resources.interactions`` 就能绕开这道门禁。
    """
    provider = object.__new__(GoogleProvider)
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="(?i)资源"):
        provider._require_provider_resource(path)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "hi", "input_audio": {"data": "x"}},
        {"role": "user", "content": "hi", "input_video": {"data": "x"}},
        {"role": "user", "content": "hi", "audio_url": "u"},
        {"role": "user", "content": "hi", "video_url": "u"},
    ],
)
def test_openai_responses_rejects_ark_only_top_level_fields(message):
    """带一个无关的 ``content`` 键就绕开了类型分流，Ark 专属块原样发给
    不支持它的端点。"""
    with pytest.raises(ValueError, match="不受 OpenAI Responses SDK 支持"):
        normalize_responses_input(None, None, [message], provider="openai")


@pytest.mark.parametrize("field", ["input_audio", "input_video", "audio_url", "video_url"])
def test_normalize_native_item_rejects_ark_only_fields_on_its_own(field):
    """助手函数要能独立拦住 Ark 专属顶层字段。

    ``normalize_responses_input`` 在进入原生 item 分支前已查过一遍，因此这层
    守卫对当前唯一调用点是冗余的——但没有测试能区分它的存在，等于死代码。
    这里直接调用助手，把它锚定住：若有人删掉这层自校验，助手被单独复用时
    就会静默放行 Ark 专属字段。
    """
    with pytest.raises(ValueError, match="不受 OpenAI Responses SDK 支持"):
        _normalize_responses_native_item(
            {"role": "user", "content": "hi", field: "x"}, "loc", "openai"
        )


# ── 回归：顶层 Ark 专属字段与带 role 的 Ark 专属块 ──


@pytest.mark.parametrize(
    "message",
    [
        {"image_pixel_limit": {"max_pixels": 1}},
        {"type": "message", "role": "user", "content": "hi", "image_pixel_limit": {"m": 1}},
        {"role": "user", "content": "hi", "image_pixel_limit": {"m": 1}},
        {"role": "user", "content": "x", "type": "input_audio", "audio_url": "u"},
        {"role": "system", "content": "x", "type": "input_video", "video_url": "u"},
    ],
)
def test_openai_responses_rejects_ark_only_fields_in_every_position(message):
    """Ark 专属字段与 ``type``/``content``/``role`` 都无关：写成内容块、原生
    item 还是普通消息顶层，OpenAI 兼容渠道都不能发出。只查一条路径会留下旁路。
    """
    with pytest.raises(ValueError, match="不受|缺少"):
        normalize_responses_input(None, None, [message], provider="openai")


@pytest.mark.parametrize("bad_type", [["input_text"], {"a": 1}, 123])
def test_responses_content_type_must_be_hashable_string(bad_type):
    """``type`` 是任意 JSON 值，直接做集合查找会抛
    ``TypeError: unhashable type``，把可诊断的类型错误变成崩溃。"""
    with pytest.raises(ValueError, match="type 必须是非空字符串"):
        normalize_responses_input(None, None, [{"type": bad_type, "text": "x"}], provider="openai")


def test_ark_responses_accepts_ark_only_fields():
    """补全校验后 Ark 渠道自身仍要放行。"""
    converted = normalize_responses_input(
        None,
        None,
        [{"type": "input_image", "image_url": "u", "image_pixel_limit": {"m": 1}}],
        provider="ark",
    )

    assert converted[0]["image_pixel_limit"] == {"m": 1}


# ---------------------------------------------------------------------------
# 客户端级 max_retries
# ---------------------------------------------------------------------------
#
# ``timeout`` 与 ``max_retries`` 是 openai SDK 的**客户端构造参数**，本项目把它们
# 从 ``options`` 直接透传给 ``openai.OpenAI(...)``（见 ``_get_client``）。SDK 自己
# 不做任何校验，于是 ``options`` 里的取值可以造成两类失败：
#
# 1. ``max_retries`` 为**负数**：SDK 的重试循环是 ``range(max_retries + 1)``，
#    负数让循环体一次都不执行，请求根本没发出去，最后抛的是 SDK 内部的
#    ``AssertionError: could not resolve response (should never happen)``——
#    实测 ``max_retries=-1`` 时请求数为 **0**。
# 2. ``max_retries`` 为**超大值**（``10**400`` 这类，``config.toml`` 的 TOML
#    整数无上界）：等价于无上限重试，网络故障下会一直重试下去。
#
# 非整数（``"abc"``／``1.5``／``True``）则是第三种：SDK 把它们存进
# ``self.max_retries``，直到真正重试时才在 ``max_retries - retries_taken`` 上
# 抛 ``TypeError``，报错点离配置处很远。
#
# Ark 渠道的 ``_normalize_client_options`` 早就校验了 ``max_retries``（要求
# ``>= 0`` 的整数），OpenAI 兼容渠道漏了同一道。这里补齐，并给上界——重试次数
# 是配置项，不是可以取到天文数字的量。
_CLIENT_OPTION_PROVIDER = "qwen"
_CLIENT_OPTION_SETTINGS = SimpleNamespace(
    qwen_api_key="key",
    qwen_base_url="https://qwen.invalid/v1",
)


def _build_client_option_provider(**options):
    return OpenAICompatibleProvider(
        "demo", _CLIENT_OPTION_PROVIDER, settings=_CLIENT_OPTION_SETTINGS, options=options
    )


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(-(10**400), id="negative-beyond-double-range"),
        pytest.param(10**400, id="400digits-over-upper-bound"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param("abc", id="non-numeric-string"),
        pytest.param("", id="empty-string"),
    ],
)
def test_openai_compatible_client_max_retries_must_be_bounded_non_negative_int(bad_value):
    """``options.max_retries`` 必须在构造期收敛成有上界的非负整数。"""
    with pytest.raises(ValueError, match="max_retries"):
        _build_client_option_provider(max_retries=bad_value)


@pytest.mark.parametrize(
    ("good_value", "expected"),
    [
        (0, 0),
        (1, 1),
        (10, 10),
        (100, 100),
        # 数字字符串必须继续接受：``.env`` 与 TOML 来源本来就以字符串形态到达，
        # 且 Ark 渠道的同一套判据也接受 ``"5"``。
        ("5", 5),
        (5.0, 5),
    ],
)
def test_openai_compatible_client_max_retries_accepts_valid_values(good_value, expected):
    provider = _build_client_option_provider(max_retries=good_value)
    assert provider._client_options()["max_retries"] == expected


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(10**400, id="400digits-beyond-double-range"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(True, id="bool"),
        pytest.param("abc", id="non-numeric-string"),
    ],
)
def test_openai_compatible_client_timeout_must_be_finite_positive(bad_value):
    """``options.timeout`` 是客户端构造参数，同样要在构造期拒绝非法取值。

    原实现把 ``timeout`` 原样透传给 ``openai.OpenAI``，而 SDK 不校验：实测
    ``timeout="abc"``、``timeout=-1``、``timeout=0`` 全部被接受并存进
    ``client.timeout``，直到 httpx 层才以形态各异的错误暴露。
    """
    with pytest.raises(ValueError, match="timeout"):
        _build_client_option_provider(timeout=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(101, id="just-over-upper-bound"),
        pytest.param(10**400, id="400digits-over-upper-bound"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param("abc", id="non-numeric-string"),
    ],
)
def test_ark_client_max_retries_must_be_bounded_non_negative_int(bad_value):
    """Ark 的 ``max_retries`` 与 OpenAI 兼容渠道走同一套判据。

    此前的内联实现只要求「``>= 0`` 的整数」，**没有上界**：实测
    ``max_retries = 10**400`` 被原样接受并传给 Ark SDK，而 SDK 的重试循环按该值
    决定次数，等价于无限重试。
    """
    with pytest.raises(ValueError, match="max_retries"):
        VolcengineProvider._normalize_client_options({"max_retries": bad_value})


@pytest.mark.parametrize(
    ("good_value", "expected"),
    [(0, 0), (1, 1), (100, 100), ("5", 5), (5.0, 5)],
)
def test_ark_client_max_retries_accepts_valid_values(good_value, expected):
    normalized = VolcengineProvider._normalize_client_options({"max_retries": good_value})
    assert normalized["max_retries"] == expected


# ---------------------------------------------------------------------------
# Anthropic 的客户端级 timeout / max_retries
# ---------------------------------------------------------------------------
#
# 与 OpenAI 兼容渠道同一类问题：``timeout``／``max_retries`` 被原样透传给
# ``anthropic.Anthropic(...)``，SDK 不做任何校验。Anthropic SDK 的重试循环同样是
# ``for retries_taken in range(max_retries + 1)``，因此 ``max_retries=-1`` 时循环体
# 一次都不执行——实测 HTTP 请求数为 **0**，最终抛
# ``RuntimeError: could not resolve response (should never happen)``。
_ANTHROPIC_CLIENT_OPTION_SETTINGS = SimpleNamespace(anthropic_api_key="sk-ant-test")


def _patch_anthropic_settings(monkeypatch) -> None:
    """把 ``get_settings`` 打到 ``AnthropicProvider`` 真正读取的那个模块字典上。

    不能用字符串形式 ``monkeypatch.setattr("src.providers.anthropic.get_settings", ...)``：
    ``tests/providers/test_factory.py`` 会从 ``sys.modules`` 里删除
    ``src.providers.anthropic`` 再重新导入，于是字符串形式解析到的是**新**模块对象，
    而 ``AnthropicProvider.__init__`` 仍从**旧**模块字典取 ``get_settings``——patch
    落空，构造器读到真实配置（无 key）后抛 ``ANTHROPIC_API_KEY is required``。
    单独跑本文件时不会触发，只有全量运行才暴露。
    """
    module_globals = AnthropicProvider.__init__.__globals__
    monkeypatch.setitem(module_globals, "get_settings", lambda: _ANTHROPIC_CLIENT_OPTION_SETTINGS)


def _build_anthropic_provider(**options):
    return AnthropicProvider("claude-sonnet-4-6", options=options or None)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(101, id="just-over-upper-bound"),
        pytest.param(10**400, id="400digits-over-upper-bound"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param("abc", id="non-numeric-string"),
    ],
)
def test_anthropic_client_max_retries_must_be_bounded_non_negative_int(monkeypatch, bad_value):
    _patch_anthropic_settings(monkeypatch)
    with pytest.raises(ValueError, match="max_retries"):
        _build_anthropic_provider(max_retries=bad_value)


@pytest.mark.parametrize(
    ("good_value", "expected"),
    [(0, 0), (1, 1), (100, 100), ("5", 5)],
)
def test_anthropic_client_max_retries_accepts_valid_values(monkeypatch, good_value, expected):
    _patch_anthropic_settings(monkeypatch)
    provider = _build_anthropic_provider(max_retries=good_value)
    assert provider._normalized_client_options["max_retries"] == expected


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(10**400, id="400digits-beyond-double-range"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(True, id="bool"),
        pytest.param("abc", id="non-numeric-string"),
    ],
)
def test_anthropic_client_timeout_must_be_finite_positive(monkeypatch, bad_value):
    _patch_anthropic_settings(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        _build_anthropic_provider(timeout=bad_value)


def test_anthropic_client_accepts_httpx_timeout_object(monkeypatch):
    """``httpx.Timeout`` 形态必须保留——SDK 接受它，且无法用 TOML 表达。"""
    import httpx2

    _patch_anthropic_settings(monkeypatch)
    provider = _build_anthropic_provider(timeout=httpx2.Timeout(5.0))
    assert hasattr(provider._normalized_client_options["timeout"], "connect")


# ── 回归：客户端 timeout 的**上界**与派生值 ──
#
# 四个渠道原本只判「有限正数」，没有上界。失效点不在超时值本身，而在派生值：
# Google 是唯一把秒换算成毫秒再交给 SDK 的渠道（``_build_http_options`` 的
# ``int(timeout * 1000)``），超过 ``DBL_MAX/1000`` 的秒数让该派生抛
# ``OverflowError``——它是 ``ArithmeticError`` 的子类，**不是** ``ValueError``，
# 穿透项目全部以 ``ValueError`` 为契约的 handler，以裸 traceback 终止会话。
# 实测 ``config.toml`` 写 ``timeout = 1e306`` 即可复现：配置加载成功、provider
# 构造成功，首次提问才炸。

_BEYOND_MILLISECOND_RANGE = [
    pytest.param(1e306, id="306exponent"),
    pytest.param(1e308, id="308exponent"),
    pytest.param(86_401.0, id="just-over-one-day"),
]


@pytest.mark.parametrize("bad_value", _BEYOND_MILLISECOND_RANGE)
def test_google_request_timeout_has_upper_bound(bad_value):
    """``options.timeout`` 必须在上界内，否则毫秒派生值会抛 OverflowError。"""
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-test"
    provider._options = {}
    with pytest.raises(ValueError, match="不能超过"):
        provider._build_generation_config("system", None, 0.2, timeout=bad_value)


@pytest.mark.parametrize("bad_value", _BEYOND_MILLISECOND_RANGE)
def test_openai_compatible_client_timeout_has_upper_bound(bad_value):
    with pytest.raises(ValueError, match="不能超过"):
        _build_client_option_provider(timeout=bad_value)


@pytest.mark.parametrize("bad_value", _BEYOND_MILLISECOND_RANGE)
def test_anthropic_client_timeout_has_upper_bound(monkeypatch, bad_value):
    _patch_anthropic_settings(monkeypatch)
    with pytest.raises(ValueError, match="不能超过"):
        _build_anthropic_provider(timeout=bad_value)


def test_ark_client_timeout_has_upper_bound():
    with pytest.raises(ValueError, match="不能超过"):
        VolcengineProvider._normalize_client_options({"timeout": 1e306})


def test_google_http_options_timeout_rejects_beyond_millisecond_range():
    """``options.http_options.timeout`` 是毫秒，同样需要上界。

    此前判据只要求「正整数毫秒」，实测 ``1e308`` 能通过，到真实
    ``genai.Client(...)`` 构造时才抛 pydantic 的
    ``Unable to parse input string as an integer``——错误消息与 Google 配置
    毫无关联。
    """
    with pytest.raises(ValueError, match="正整数毫秒"):
        GoogleProvider._validate_http_options({"timeout": 1e308})


@pytest.mark.parametrize("good_value", [1, 30.0, "60"])
def test_client_timeout_accepts_values_within_bound(good_value):
    """上界之内的正常取值必须继续接受（Ark 渠道的字符串形态也在其中）。"""
    assert VolcengineProvider._normalize_client_options({"timeout": good_value})["timeout"] > 0


# ── 回归：Ark ``count_tokens`` 与其余七个 token 计数入口同契约 ──
#
# 八个 provider 的 token 计数入口都已改用 ``require_token_count``，只有 Ark 的
# 同步/异步 ``count_tokens`` 漏了，写的是 ``int(field(item, "total_tokens", 0))``：
# ``"abc"`` 抛裸 ``ValueError``、``None`` 抛裸 ``TypeError``、位数超限的数字串抛
# 与 token 计数无关的位数错误，且消息里连字段名都没有；``0`` 默认值还把
# 「响应不完整」静默算成「0 个 token」。


def _ark_provider_with_tokenization_response(items):
    """构造只带假 tokenization 资源的 Ark provider。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(
        tokenization=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(data=items),
        )
    )
    return provider


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("abc", id="non-numeric-string"),
        pytest.param(None, id="missing-field"),
        pytest.param(1.5, id="fractional"),
        pytest.param(-5, id="negative"),
        pytest.param("9" * 4301, id="4301digits-over-int-max-str-digits"),
    ],
)
def test_ark_count_tokens_reports_field_name_for_bad_values(bad_value):
    provider = _ark_provider_with_tokenization_response([SimpleNamespace(total_tokens=bad_value)])
    with pytest.raises(RuntimeError, match="total_tokens"):
        provider.count_tokens("hello")


def test_ark_count_tokens_sums_valid_values():
    provider = _ark_provider_with_tokenization_response(
        [SimpleNamespace(total_tokens=3), SimpleNamespace(total_tokens=4)]
    )
    assert provider.count_tokens("hello") == 7


# ── 回归：不可哈希取值不得让集合成员守卫抛 TypeError ──
#
# ``field(x, "type") in {"a", "b"}`` 对 ``list``/``dict`` 抛
# ``TypeError: unhashable type``，而调用方按 ``ValueError``/``RuntimeError``
# 分类错误。这些值来自网关 JSON：openai SDK 对 SSE chunk 的未知顶层字段用
# ``extra="allow"`` 原样保留，实测一个 ``{"type": ["error"]}`` 就能构造出来，
# 于是「网关返回了形态异常的事件」变成裸 ``TypeError``。


@pytest.mark.parametrize(
    "bad_value",
    [pytest.param(["error"], id="list"), pytest.param({"a": 1}, id="dict")],
)
def test_responses_stream_event_type_must_not_raise_type_error(bad_value):
    """非字符串 ``type`` 按「未知事件」处理，不得抛 TypeError。"""
    assert raise_for_stream_error_event({"type": bad_value}, "openai") is None


@pytest.mark.parametrize(
    "bad_value",
    [pytest.param(["auto"], id="list"), pytest.param({"a": 1}, id="dict")],
)
@pytest.mark.parametrize("part_type", ["input_image", "input_file"])
def test_responses_content_part_detail_must_not_raise_type_error(part_type, bad_value):
    """``detail`` 非字符串时必须报 ``ValueError``，而不是 ``TypeError``。"""
    part = {"type": part_type, "detail": bad_value}
    if part_type == "input_image":
        part["image_url"] = "https://x/y.png"
    else:
        part["file_data"] = "data:x"
    with pytest.raises(ValueError, match="detail"):
        _responses_content_part(part, "input[0]", provider="openai")


# `_responses_content_part` 的二十六条形态判据在整套测试里从未执行（`sys.settrace`
# 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3286 条仍然全绿）。既有用例
# `test_responses_content_part_detail_must_not_raise_type_error` 只钉住两个分支里
# 的 `detail` 成员判据，其余形态判据全绕开了。删除后行为确实改变（实测，全在
# `src/providers/__base__/model_provider.py`）：**二十五条静默接受**——畸形取值
# 原样进 `converted` 字典再发给上游（如 `text: 5`、`file_id: 5`、`detail: "ultra"`、
# 整块缺 URI 却产出 `{"type": "input_video"}`）；**两条变成裸 `TypeError`**——
# `dict(...)` 对非映射取值抛 `TypeError: 'int' object is not iterable`。
# 最值得记的是四条 `prompt_cache_breakpoint` 守卫（700／810／858／901）：它们
# 拦住的是「把 OpenAI 专属字段发给 Ark SDK」——删掉后 Ark 侧静默丢弃该字段，
# 而调用方以为缓存已生效。
_DATA_PNG = "data:image/png;base64,AAA"


@pytest.mark.parametrize(
    ("part", "provider", "expected"),
    [
        pytest.param(5, "openai", r"必须是字符串或对象", id="part-not-str-or-object"),
        pytest.param(
            {"type": "text", "text": 5}, "openai", r"\.text 必须是字符串", id="text-not-string"
        ),
        pytest.param(
            {"type": "text", "text": "a", "prompt_cache_breakpoint": 1},
            "ark",
            "prompt_cache_breakpoint 不受 Ark",
            id="text-pcb-on-ark",
        ),
        pytest.param(
            {"type": "input_audio", "audio_url": "https://x/a.wav", "file_id": 5},
            "ark",
            r"\.file_id 必须是字符串",
            id="audio-file-id-not-string",
        ),
        pytest.param(
            {"type": "input_audio", "audio_url": "https://x/a.wav", "chunking_strategy": 5},
            "ark",
            r"chunking_strategy 必须是对象",
            id="audio-chunking-not-mapping",
        ),
        pytest.param(
            {"type": "input_video", "video_url": {"url": 5}},
            "ark",
            r"video_url\.url 必须是字符串",
            id="video-url-not-string",
        ),
        pytest.param(
            {"type": "input_video", "video_url": {"url": "https://x/v.mp4"}, "file_id": 5},
            "ark",
            r"\.file_id 必须是字符串",
            id="video-file-id-not-string",
        ),
        pytest.param(
            {"type": "input_video", "video_url": {}},
            "ark",
            "缺少 video_url 或 file_id",
            id="video-missing-url-and-file-id",
        ),
        pytest.param(
            {"type": "image_url", "image_url": 5},
            "openai",
            r"\.image_url 必须是字符串或对象",
            id="image-url-not-str-or-object",
        ),
        pytest.param(
            {"type": "image_url", "image_url": {"url": 5}},
            "openai",
            r"image_url\.url 必须是字符串",
            id="image-url-value-not-string",
        ),
        pytest.param(
            {"type": "image_url", "file_id": 5},
            "openai",
            r"\.file_id 必须是字符串",
            id="image-file-id-not-string",
        ),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "detail": "ultra"},
            "openai",
            r"\.detail 必须是",
            id="image-detail-not-allowed",
        ),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "image_pixel_limit": {"min_pixels": 1}},
            "openai",
            "image_pixel_limit 不受 OpenAI",
            id="image-pixel-limit-on-openai",
        ),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "image_pixel_limit": 5},
            "ark",
            r"image_pixel_limit 必须是对象",
            id="image-pixel-limit-not-mapping",
        ),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "prompt_cache_breakpoint": 1},
            "ark",
            "prompt_cache_breakpoint 不受 Ark",
            id="image-pcb-on-ark",
        ),
        pytest.param(
            {"type": "input_image", "image_url": 5},
            "openai",
            r"\.image_url 必须是字符串",
            id="input-image-url-not-string",
        ),
        pytest.param(
            {"type": "input_image", "image_url": _DATA_PNG, "file_id": 5},
            "openai",
            r"\.file_id 必须是字符串",
            id="input-image-file-id-not-string",
        ),
        pytest.param(
            {"type": "input_image"},
            "openai",
            "缺少 image_url 或 file_id",
            id="input-image-missing-url-and-file-id",
        ),
        pytest.param(
            {"type": "input_image", "image_url": _DATA_PNG, "image_pixel_limit": 5},
            "ark",
            r"image_pixel_limit 必须是对象",
            id="input-image-pixel-limit-not-mapping",
        ),
        pytest.param(
            {"type": "input_image", "image_url": _DATA_PNG, "prompt_cache_breakpoint": 1},
            "ark",
            "prompt_cache_breakpoint 不受 Ark",
            id="input-image-pcb-on-ark",
        ),
        pytest.param(
            {"type": "file", "file": 5},
            "openai",
            r"\.file 必须是字符串或对象",
            id="file-not-str-or-object",
        ),
        pytest.param(
            {"type": "file"},
            "openai",
            "缺少 file_id、file_data 或 file_url",
            id="file-missing-all-three",
        ),
        pytest.param(
            {"type": "file", "file_id": 5},
            "openai",
            r"\.file_id 必须是字符串",
            id="file-id-not-string",
        ),
        pytest.param(
            {"type": "file", "file_id": "f", "detail": "auto"},
            "ark",
            r"\.detail 不受 Ark",
            id="file-detail-on-ark",
        ),
        pytest.param(
            {"type": "file", "file_id": "f", "prompt_cache_breakpoint": 1},
            "ark",
            "prompt_cache_breakpoint 不受 Ark",
            id="file-pcb-on-ark",
        ),
        pytest.param(
            {"type": "input_file", "file_id": 5},
            "openai",
            r"\.file_id 必须是字符串",
            id="input-file-id-not-string",
        ),
        pytest.param(
            {"type": "input_file", "file_id": "f", "prompt_cache_breakpoint": 1},
            "ark",
            "prompt_cache_breakpoint 不受 Ark",
            id="input-file-pcb-on-ark",
        ),
    ],
)
def test_responses_content_part_rejects_malformed_blocks(part, provider, expected):
    """内容块形态非法时必须给带字段名的 ``ValueError``，不能静默接受或抛裸 ``TypeError``。"""
    with pytest.raises(ValueError, match=expected):
        _responses_content_part(part, "input[0]", provider=provider)


@pytest.mark.parametrize(
    ("part", "provider"),
    [
        pytest.param({"type": "text", "text": "hi"}, "openai", id="text"),
        pytest.param(
            {"type": "text", "text": "hi", "prompt_cache_breakpoint": True},
            "openai",
            id="text-pcb-openai",
        ),
        pytest.param(
            {
                "type": "input_audio",
                "audio_url": "https://x/a.wav",
                "chunking_strategy": {"size": 1},
            },
            "ark",
            id="audio-chunking",
        ),
        pytest.param({"type": "input_video", "video_url": "https://x/v.mp4"}, "ark", id="video"),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "detail": "original"},
            "openai",
            id="image",
        ),
        pytest.param(
            {"type": "image_url", "image_url": _DATA_PNG, "image_pixel_limit": {"min_pixels": 1}},
            "ark",
            id="image-pixel-limit-ark",
        ),
        pytest.param({"type": "input_image", "image_url": _DATA_PNG}, "openai", id="input-image"),
        pytest.param({"type": "file", "file_id": "file-1"}, "openai", id="file"),
        pytest.param(
            {"type": "file", "file_url": "https://x/f.pdf", "detail": "high"},
            "openai",
            id="file-detail",
        ),
        pytest.param({"type": "input_file", "file_id": "file-1"}, "openai", id="input-file"),
    ],
)
def test_responses_content_part_accepts_well_formed_blocks(part, provider):
    """收紧不能误伤：各分支的合法内容块照常转换。"""
    converted = _responses_content_part(dict(part), "input[0]", provider=provider)

    assert converted["type"].startswith("input_")


def test_responses_stream_delta_survives_unhashable_event_type():
    """网关 SSE chunk 的 ``type`` 是 list 时不得抛 TypeError。

    openai SDK 对未知顶层字段用 ``extra="allow"`` 原样保留，因此
    ``{"type": ["error"]}`` 会以 list 形态到达 ``_stream_delta``。
    """
    chunk = SimpleNamespace(type=["error"], delta="")
    assert OpenAICompatibleProvider._stream_delta(chunk) == ""


# ── 回归：错误消息插值不得因超大整数改抛位数错误 ──
#
# ``f"{10**4300}"`` 抛 ``ValueError: Exceeds the limit (4300 digits)``。
# 这些字段（``status``／``code``／``message``）来自网关 JSON，而 openai SDK 能
# 解析出 4300 位的整数——于是「网关返回了业务错误」变成「整数转字符串超限」。


@pytest.mark.parametrize(
    "field_name",
    [
        pytest.param("status", id="status"),
        pytest.param("code", id="code"),
        pytest.param("error_code", id="error_code"),
        pytest.param("message", id="message"),
    ],
)
def test_chat_response_error_message_survives_huge_integer_field(field_name):
    response = SimpleNamespace(choices=None, **{field_name: 10**4300})
    with pytest.raises(RuntimeError, match="Chat Completions"):
        OpenAICompatibleProvider._raise_for_chat_response_error(response)


def test_chat_response_error_status_still_compares_numerically():
    """``status`` 仍须参与 ``>= 400`` 的数值比较，不能被收敛成空串。"""
    response = SimpleNamespace(choices=None, status=503)
    with pytest.raises(RuntimeError, match="status=503"):
        OpenAICompatibleProvider._raise_for_chat_response_error(response)


# ── 回归：Embedding ``options.dimensions`` 必须校验 ──
#
# 这个值从 ``config.toml`` 直达请求体，此前**完全没有校验**：``-1``／``0``／
# ``"abc"``／``1.5``／``True`` 全部原样发给远端；``10**4301`` 在 ``json.dumps``
# 编码请求体时才抛位数错误。项目里 ``_MAX_EMBEDDING_DIMENSION`` 早就是这个量
# 的上界（``local_hash`` 与 ``embedding_service`` 都在用）。


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(10**400, id="400digits-over-upper-bound"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param("abc", id="non-numeric-string"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_embedding_dimensions_option_is_validated(bad_value):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._options = {"dimensions": bad_value}
    provider._provider = "openai"
    with pytest.raises(ValueError, match="dimensions"):
        provider._embedding_options()


def test_embedding_dimensions_option_accepts_valid_value():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._options = {"dimensions": 1536}
    provider._provider = "openai"
    assert provider._embedding_options()["dimensions"] == 1536


# ── 回归：Google 整数型生成参数必须校验 ──
#
# 此前只做键名白名单，取值原样透传给 ``types.GenerateContentConfig``：实测
# ``options.max_output_tokens = 10**400`` 时构造与调用都成功，请求体里就带着
# 那个 401 位整数发给服务端（MockTransport 抓包确认）；``candidate_count``
# 的 ``-1``／``0`` 同样静默通过。


@pytest.mark.parametrize("field_name", ["max_output_tokens", "candidate_count"])
@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(10**400, id="400digits-over-upper-bound"),
        pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_google_generation_options_are_validated(field_name, bad_value):
    provider = object.__new__(GoogleProvider)
    provider._options = {field_name: bad_value}
    with pytest.raises(ValueError, match=field_name):
        provider._generation_options()


def test_google_generation_options_accept_valid_values():
    provider = object.__new__(GoogleProvider)
    provider._options = {"max_output_tokens": 4096, "candidate_count": 1}
    values = provider._generation_options()
    assert values == {"max_output_tokens": 4096, "candidate_count": 1}


# ── 回归：Ark Classification 的动态资源路径必须与 Facade 同约束 ──
#
# AGENTS.md 要求 Ark Classification 资源「需通过显式 Facade/入口调用，不能把
# 缺少资源挂载静默当成成功」。Facade 侧做对了（``_get_classification_resource``
# 在 SDK 未挂载时抛带指引的 ``NotImplementedError``），但动态路径
# ``resources.classification.create(...)`` 既不过 Facade、也不做参数白名单，
# 实测 ``恶意字段=1`` 一路放行，最终拿到 SDK 内部的
# ``AttributeError: 'Ark' object has no attribute 'classification'``。


def _ark_provider_with_classification_resource():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {}
    provider._classification_resource = SimpleNamespace(create=lambda **kwargs: kwargs)
    provider._get_client = lambda: SimpleNamespace()
    return provider


def test_ark_classification_dynamic_path_shares_facade_kwargs_whitelist():
    provider = _ark_provider_with_classification_resource()
    with pytest.raises(ValueError, match="Classification"):
        provider._require_provider_resource(
            "volcengine.classification.create",
            {"unknown": 1},
        )


def test_ark_classification_dynamic_path_accepts_facade_supported_kwargs():
    provider = _ark_provider_with_classification_resource()
    # 不抛错即通过；具体资源可用性由 SDK 决定，本测试只固定「不受门禁误伤」。
    provider._require_provider_resource(
        "volcengine.classification.create",
        {"user": "u1"},
    )


# ── 回归：诊断入口对形态异常的参数必须给可读消息，而不是裸的整数转换错误 ──
#
# ``protocol_status`` 与 ``capability_report`` 是公开的诊断 API（配置界面与
# 排障工具会调），参数来自调用方而非 ``Settings``——``ModelDetail.protocol``
# 有 validator，但这两个入口收的是**裸字符串**，没有任何校验。
#
# 两处都把取值直接喂给 ``str(...).strip().lower()`` 做协议名归一化：
# ``10**4301`` 撞上 CPython 的整数位数上限，抛的是
# ``ValueError: Exceeds the limit (4300 digits) for integer string conversion``
# ——既没有「哪个参数错了」，也没有「可选值是什么」，用户拿到的是与协议诊断
# 毫无关系的整数转换错误。同一文件的其它入口（``_normalize_verified_protocols``
# 的非字符串分支）早已给出带指引的消息，这里补齐一致性。
#
# ``server_verified_protocols`` 里的取值同理：它是 ``options`` 里的任意值，
# 而 ``validate_secret_free_options`` 只查敏感键名、不做类型校验。
# 用显式 id：pytest 默认把参数值 ``str()`` 成用例 id，而 ``10**4301`` 在
# 收集期就会抛 ``ValueError``，整个文件无法导入。
@pytest.mark.parametrize(
    "hostile_protocol",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
        pytest.param(None, id="none"),
        pytest.param(["responses"], id="list"),
    ],
)
def test_protocol_status_rejects_malformed_protocol_with_a_readable_message(hostile_protocol):
    with pytest.raises(ValueError) as excinfo:
        ModelProviderFactory.protocol_status("openai", hostile_protocol)

    message = str(excinfo.value)
    # 只断言「抛 ValueError」不够：未修复的实现抛的也是 ValueError（整数转换
    # 错误），必须断言消息指向**参数名**。用参数名而不是「协议」二字，是因为
    # 诊断入口的调用方要靠它定位自己传错了哪个实参。
    assert "protocol" in message
    assert "必须是字符串" in message


@pytest.mark.parametrize(
    "hostile_value",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_protocol_status_rejects_malformed_verified_entry_with_a_readable_message(hostile_value):
    with pytest.raises(ValueError) as excinfo:
        ModelProviderFactory.protocol_status(
            "openai",
            "responses",
            options={"server_verified_protocols": [hostile_value]},
        )

    message = str(excinfo.value)
    assert "server_verified_protocols" in message
    assert "必须是字符串" in message
    # 关键区分：未修复的实现抛的是 CPython 的整数转换错误，用户拿不到任何
    # 「哪个参数错了」的线索。断言消息里**不出现**那段原始文本。
    assert "Exceeds the limit" not in message


def test_protocol_status_still_accepts_valid_protocols_after_the_guard():
    """反向守卫：收紧不得误伤正常取值与别名。"""
    assert ModelProviderFactory.protocol_status("openai", "responses") == {
        "adapter_supported": True,
        "server_verified": False,
    }
    assert ModelProviderFactory.protocol_status("openai", "  RESPONSES  ") == (
        ModelProviderFactory.protocol_status("openai", "responses")
    )


# ── 回归：协议名归一化的另外三处同样不能抛裸的整数转换错误 ──
#
# 与 ``factory`` 那两处同一根因：把取值直接喂给 ``str(...).strip().lower()``，
# ``10**4301`` 撞上 CPython 的整数位数上限，抛的是与「协议名不支持」毫无关系的
# ``ValueError: Exceeds the limit``。三处都可达（前两处经公开构造器，第三处经
# ``_normalize_responses_native_item`` 的入口分流）。
@pytest.mark.parametrize(
    "hostile_protocol",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(None, id="none"),
        pytest.param(["responses"], id="list"),
    ],
)
def test_openai_compatible_normalize_protocol_rejects_non_string(hostile_protocol):
    from src.providers.openai_compatible import OpenAICompatibleProvider

    with pytest.raises(ValueError) as excinfo:
        OpenAICompatibleProvider._normalize_protocol(hostile_protocol)

    message = str(excinfo.value)
    assert "必须是字符串" in message
    assert "Exceeds the limit" not in message
    # 诊断入口要给出可选值，否则用户只知道「传错了」。
    assert "chat_completions" in message


@pytest.mark.parametrize(
    "hostile_protocol",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(None, id="none"),
        pytest.param(["responses"], id="list"),
    ],
)
def test_volcengine_rejects_non_string_protocol(hostile_protocol):
    from src.providers.volcengine import VolcengineProvider

    with pytest.raises(ValueError) as excinfo:
        VolcengineProvider(model_name="demo", protocol=hostile_protocol)

    message = str(excinfo.value)
    assert "必须是字符串" in message
    assert "Exceeds the limit" not in message


def test_responses_provider_variant_falls_back_instead_of_raising():
    """纯分流判断不该抛错：非字符串一律走 openai 变体。

    它抛在 ``_normalize_responses_native_item`` 的入口，会把本该给出的
    「字段类型不对」消息换成一条与协议无关的整数转换错误。
    """
    from src.providers.__base__.model_provider import _responses_provider_variant

    for hostile in (10**4301, None, ["ark"], 42):
        assert _responses_provider_variant(hostile) == "openai"
    # 反向守卫：正常取值仍要正确分流。
    assert _responses_provider_variant("ark") == "ark"
    assert _responses_provider_variant("volcengine") == "ark"
    assert _responses_provider_variant("  ARK  ") == "ark"
    assert _responses_provider_variant("openai") == "openai"


# ── 回归：``normalize_messages`` 的 prompt / system_prompt 形态 ──
#
# 本函数是四个 provider 公共 ``invoke`` 的**统一请求边界**（``google.py:1497``、
# ``anthropic.py:834/1746/…``、``openai_compatible.py:1024``、``volcengine.py:1001``
# 都调它）。``messages`` 那条分支有完整判据（非序列、非 Mapping 条目都给
# ``ValueError``），而 ``prompt``／``system_prompt`` 这一半**完全没有**：
#
#   * ``prompt=5``／``["a"]`` 静默生成 ``{"role": "user", "content": 5}``，非字符串
#     字面量被当成真实用户输入发给模型（与 ``ChatService._build_prompt`` 钉住的
#     ``用户问题: None`` 是同一形态）；
#   * ``prompt=10**4301`` 在 ``str(prompt).strip()`` 处抛 ``ValueError: Exceeds the
#     limit``——消息与「请求内容写错了」无关，而 provider 的错误分类按
#     ``ValueError``／``RuntimeError`` 写；
#   * ``system_prompt`` 同理：非字符串直接进 ``content``，位数超限的整数在
#     ``normalize_responses_input`` 的日志插值处炸。
#
# 可达路径：``invoke``／``ainvoke`` 是公开方法，参数一路透传到 ``normalize_messages``；
# 本仓测试就有 ``ainvoke("hello", system_prompt=None, stream=False)`` 这类直调。


@pytest.mark.parametrize(
    "hostile_prompt",
    [
        pytest.param(10**4301, id="huge-int"),
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
    ],
)
def test_normalize_messages_rejects_non_string_prompt_readably(hostile_prompt):
    """非字符串 ``prompt`` 必须给带指引的 ``ValueError``，不是静默进 content 或裸整数转换错。"""
    with pytest.raises(ValueError) as excinfo:
        normalize_messages(hostile_prompt, None, None)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "prompt" in message


@pytest.mark.parametrize(
    "hostile_system",
    [
        pytest.param(10**4301, id="huge-int"),
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
    ],
)
def test_normalize_messages_rejects_non_string_system_prompt_readably(hostile_system):
    """非字符串 ``system_prompt`` 同样必须被拒绝，而不是进 ``content``。"""
    with pytest.raises(ValueError) as excinfo:
        normalize_messages("hi", hostile_system, None)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "system_prompt" in message


def test_normalize_messages_still_accepts_string_prompt_and_system_prompt():
    """收紧不能误伤：字符串 prompt／system_prompt 照常构造消息。"""
    normalized = normalize_messages("北京天气", "你是助手", None)

    assert normalized == [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "北京天气"},
    ]


# ── 回归：Responses background 轮询超时消息的 id 形态 ──
#
# ``_poll_background_response``／``_poll_background_response_async`` 在超时分支
# 里用裸 f-string 插值 ``response_id``（``f"...响应 {response_id} 在 {timeout:g}s
# 内未完成。"``）。``response_id`` 来自网关 JSON 的 ``id`` 字段，形态不受控：
# ``10**4301`` 让 f-string 抛 ``ValueError: Exceeds the limit``——**正要报告超时**
# 的那一步变成整数转换错误，而调用方按 ``TimeoutError``／``RuntimeError`` 分类。
# 判据与同文件 ``_error_message``／``_extract_refusal`` 一致（字符串原样，其余走
# ``safe_repr``）。


def _background_poll_provider():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "demo"
    provider._options = {}
    provider._background_poll_config = lambda: (0.0, 0.0)
    provider._response_retrieve_kwargs = lambda params: {}
    return provider


@pytest.mark.parametrize(
    "hostile_id",
    [pytest.param(10**4301, id="huge-int"), pytest.param(["a"], id="list")],
)
def test_responses_background_timeout_reports_malformed_id_readably(hostile_id):
    """超时消息不得因非字符串 ``id`` 抛裸异常。"""
    provider = _background_poll_provider()
    response = {"status": "queued", "id": hostile_id}

    with pytest.raises(TimeoutError) as excinfo:
        provider._poll_background_response(response, {})

    assert "Exceeds the limit" not in str(excinfo.value)


@pytest.mark.parametrize(
    "hostile_id",
    [pytest.param(10**4301, id="huge-int"), pytest.param(["a"], id="list")],
)
def test_responses_background_timeout_reports_malformed_id_readably_async(hostile_id):
    """异步分支同判据。"""
    provider = _background_poll_provider()
    response = {"status": "queued", "id": hostile_id}

    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(provider._poll_background_response_async(response, {}))

    assert "Exceeds the limit" not in str(excinfo.value)


def test_responses_background_timeout_still_reports_string_id():
    """收紧不能误伤：字符串 id 仍原样出现在消息里。"""
    provider = _background_poll_provider()
    response = {"status": "queued", "id": "resp_abc"}

    with pytest.raises(TimeoutError) as excinfo:
        provider._poll_background_response(response, {})

    assert "resp_abc" in str(excinfo.value)


# ── 流终止与拒答守卫 ──
#
# ``OpenAICompatibleProvider`` 与 ``VolcengineProvider`` 各自有四条「流在终结事件
# 之前结束」与四条「返回拒答」守卫，分布在 ``invoke``／``ainvoke``（纯文本生成）
# 与 ``stream_events``／``astream_events``（结构化事件）两套入口上，合计 30 条。
# 这些判据在整套测试里从未执行（`sys.settrace` 行覆盖证实）；把它们的 ``raise``
# 换成 ``pass`` 后全量 3361 条仍然全绿——既有用例只钉住 Responses 异步流的一条
# 拒 delta 收敛（``test_openai_responses_async_streaming_refusal_delta_is_coerced``）。
#
# 删除后的后果分两类（实测，均经公开入口）：
# * **流提前结束**：静默正常返回 ``["hi"]``（invoke 侧）或一串 ``StreamEvent``
#   （stream_events 侧）。调用方无法区分「模型答完了」与「连接断了」，半截答案
#   当成完整答案进入对话历史；
# * **拒答**：非流式静默返回空列表，流式退化成相邻的「流提前结束」消息——把
#   「模型明确拒绝回答」说成「流断了」，两者需要完全不同的处置。
PROVIDER_STREAM_MATRIX = [
    (OpenAICompatibleProvider, "responses", "responses"),
    (OpenAICompatibleProvider, "chat_completions", "chat"),
    (VolcengineProvider, "responses", "responses"),
    (VolcengineProvider, "chat_completions", "chat"),
]

_RESPONSES_REFUSAL_BODY = {
    "id": "r1",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", "refusal": "no", "content": []}],
}
# Ark 的 ``_extract_result`` 只认 ``output[].type == "refusal"``，与 OpenAI 侧的
# ``output[].refusal`` 字段形态不同（见两处 ``_extract_result`` 的对比）。
_ARK_REFUSAL_BODY = {
    "id": "r1",
    "status": "completed",
    "output": [{"type": "refusal", "refusal": "no"}],
}
_RESPONSES_TEXT_BODY = {
    "id": "r1",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    ],
}
_RESPONSES_REFUSAL_EVENT = {
    "type": "response.refusal.delta",
    "item_id": "i1",
    "output_index": 0,
    "delta": "no",
}
_RESPONSES_TEXT_EVENT = {
    "type": "response.output_text.delta",
    "item_id": "i1",
    "output_index": 0,
    "delta": "hi",
}
_RESPONSES_DONE_EVENT = {"type": "response.completed", "response": {"id": "r1"}}
_CHAT_REFUSAL_BODY = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "finish_reason": "content_filter",
            "message": {"role": "assistant", "refusal": "no"},
        }
    ],
}
_CHAT_REFUSAL_CHUNK = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "choices": [{"index": 0, "delta": {"refusal": "no"}}],
}
_CHAT_TEXT_CHUNK = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "choices": [{"index": 0, "delta": {"content": "hi"}}],
}
_CHAT_FINISH_CHUNK = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
}


def _sync_iterator(items):
    return iter(items)


def _async_iterator(items):
    async def gen():
        for item in items:
            yield item

    return gen()


def _stream_client(payload, kind, *, is_async):
    """构造只区分 ``stream=`` 的假 SDK 客户端。

    非流式调用返回响应对象本身、流式调用返回迭代器——两者不能混，否则
    ``_raise_for_*`` 会拿到迭代器并把「响应缺字段」误报成非法响应。
    """

    def create(**kwargs):
        if kwargs.get("stream"):
            return _async_iterator(payload) if is_async else _sync_iterator(payload)
        return payload[0]

    if kind == "responses":
        return SimpleNamespace(responses=SimpleNamespace(create=create))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _stream_provider(provider_type, protocol, kind, payload, *, is_async):
    provider = object.__new__(provider_type)
    provider._model_name = "demo"
    provider._provider = "demo"
    provider._protocol = protocol
    provider._options = {"server_verified_protocols": ["responses", "chat_completions"]}
    provider._server_verified_protocols = frozenset({"responses", "chat_completions"})
    client = _stream_client(payload, kind, is_async=is_async)
    if is_async:
        provider._get_aclient = lambda: client
    else:
        provider._get_client = lambda: client
    return provider


def _refusal_body(provider_type, kind):
    if kind != "responses":
        return _CHAT_REFUSAL_BODY
    if provider_type is VolcengineProvider:
        return _ARK_REFUSAL_BODY
    return _RESPONSES_REFUSAL_BODY


def _unterminated_payload(kind):
    if kind == "responses":
        return [_RESPONSES_TEXT_EVENT]
    return [_CHAT_TEXT_CHUNK]


def _terminated_payload(kind):
    if kind == "responses":
        return [_RESPONSES_TEXT_EVENT, _RESPONSES_DONE_EVENT]
    return [_CHAT_TEXT_CHUNK, _CHAT_FINISH_CHUNK]


@pytest.mark.parametrize(
    ("provider_type", "protocol", "kind"),
    PROVIDER_STREAM_MATRIX,
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_refusal_is_explicit(provider_type, protocol, kind, async_mode):
    """拒答必须报成「返回拒答」，不能退化成「流提前结束」或静默返回空。

    删除流式拒答守卫后，收场从「返回拒答: no」变成「流在 … 之前结束」——把
    模型明确拒绝回答说成连接中断，两者需要完全不同的处置；删除非流式守卫后
    则**静默返回空列表**，调用方把「拒答」当成「模型没话说」。
    """
    if kind == "responses":
        payload = [_RESPONSES_REFUSAL_EVENT]
    else:
        payload = [_CHAT_REFUSAL_CHUNK]
    provider = _stream_provider(provider_type, protocol, kind, payload, is_async=async_mode)

    if async_mode:

        async def run():
            return [x async for x in provider.ainvoke(prompt="hi", stream=True)]

        with pytest.raises(RuntimeError, match="返回拒答"):
            asyncio.run(run())
    else:
        with pytest.raises(RuntimeError, match="返回拒答"):
            list(provider.invoke(prompt="hi", stream=True))


@pytest.mark.parametrize(
    ("provider_type", "protocol", "kind"),
    PROVIDER_STREAM_MATRIX,
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_non_streaming_refusal_is_explicit(provider_type, protocol, kind, async_mode):
    """非流式拒答同样必须显式报错。

    删除后实测静默返回 ``[]``——一次明确的拒答变成「模型没有输出」。
    """
    payload = [_refusal_body(provider_type, kind)]
    provider = _stream_provider(provider_type, protocol, kind, payload, is_async=async_mode)

    if async_mode:

        async def run():
            return [x async for x in provider.ainvoke(prompt="hi", stream=False)]

        with pytest.raises(RuntimeError, match="返回拒答"):
            asyncio.run(run())
    else:
        with pytest.raises(RuntimeError, match="返回拒答"):
            list(provider.invoke(prompt="hi", stream=False))


@pytest.mark.parametrize(
    ("provider_type", "protocol", "kind"),
    PROVIDER_STREAM_MATRIX,
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_without_terminal_event_is_explicit(provider_type, protocol, kind, async_mode):
    """流在终结事件之前结束必须报错，不能静默交付半截答案。

    删除后实测静默返回 ``["hi"]``：调用方无法区分「模型答完了」与「连接断了」。
    """
    provider = _stream_provider(
        provider_type, protocol, kind, _unterminated_payload(kind), is_async=async_mode
    )

    if async_mode:

        async def run():
            return [x async for x in provider.ainvoke(prompt="hi", stream=True)]

        with pytest.raises(RuntimeError, match="之前结束"):
            asyncio.run(run())
    else:
        with pytest.raises(RuntimeError, match="之前结束"):
            list(provider.invoke(prompt="hi", stream=True))


@pytest.mark.parametrize(
    ("provider_type", "protocol", "kind"),
    PROVIDER_STREAM_MATRIX,
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_stream_events_requires_terminal_event(provider_type, protocol, kind, async_mode):
    """``stream_events``／``astream_events`` 是另一套实现，终止判据同样要生效。

    删除后实测静默产出一串 ``StreamEvent``——结构化事件流的调用方连「流是否
    完整」都无从判断，而它们正是靠这条区分「完整事件」与「半截事件」。
    """
    provider = _stream_provider(
        provider_type, protocol, kind, _unterminated_payload(kind), is_async=async_mode
    )

    if async_mode:

        async def run():
            return [x async for x in provider.astream_events(prompt="hi")]

        with pytest.raises(RuntimeError, match="之前结束"):
            asyncio.run(run())
    else:
        with pytest.raises(RuntimeError, match="之前结束"):
            list(provider.stream_events(prompt="hi"))


@pytest.mark.parametrize(
    ("provider_type", "protocol", "kind"),
    PROVIDER_STREAM_MATRIX,
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_terminated_stream_still_yields_content(provider_type, protocol, kind):
    """反向守卫：带终结事件的正常流照常产出内容与 finish 事件。"""
    provider = _stream_provider(
        provider_type, protocol, kind, _terminated_payload(kind), is_async=False
    )

    events = list(provider.stream_events(prompt="hi"))

    assert any(event.type == "finish" for event in events)


# ── 消息归一化的形态判据 ──
#
# ``normalize_chat_messages``／``normalize_messages``／``normalize_responses_input``
# 是四个 provider 公共的**统一请求边界**，上面六条形态判据在整套测试里从未执行
# （行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后全量 3405 条仍全绿）。既有用例
# 覆盖的是元素**内部**字段（缺 id、缺 name、非法 arguments），这几条**形态**判据
# 全绕开了。删除后行为确实改变（实测）：
# * 消息元素非对象 → 整数抛裸 ``TypeError: 'int' object is not iterable``、字符串抛
#   ``dictionary update sequence element #0 has length 1``；
# * ``tool_calls`` 非序列 → 裸 ``TypeError``，或对字符串退化成元素级错误消息；
# * ``messages`` 非序列 → 裸 ``TypeError``（字符串则退化成元素级消息）；
# * ``prompt`` 为空／全空白 → **静默接受**，产出 ``{"role": "user", "content": None}``
#   或全空白内容发给模型；
# * Responses ``content`` 非序列 → 裸 ``TypeError``；
# * ``role="function"`` → **静默接受**，产出一个没有 ``tool_call_id`` 的 function
#   消息，Responses 无法把它关联回任何调用。


@pytest.mark.parametrize("bad_message", [5, "oops", ["x"]], ids=["int", "str", "list"])
def test_chat_messages_reject_non_mapping_entries(bad_message):
    """``messages`` 的元素必须是对象。"""
    with pytest.raises(ValueError, match=r"messages\[0\] 必须是对象"):
        normalize_chat_messages([bad_message])


@pytest.mark.parametrize("bad_tool_calls", [5, "abc", {"a": 1}], ids=["int", "str", "dict"])
def test_chat_messages_reject_non_sequence_tool_calls(bad_tool_calls):
    """``tool_calls`` 必须是序列；字符串虽可迭代但语义不同，须显式排除。"""
    with pytest.raises(ValueError, match=r"tool_calls 必须是序列"):
        normalize_chat_messages(
            [{"role": "assistant", "content": "", "tool_calls": bad_tool_calls}]
        )


@pytest.mark.parametrize("bad_messages", [5, "abc", {"a": 1}], ids=["int", "str", "dict"])
def test_normalize_messages_rejects_non_sequence_container(bad_messages):
    """``messages`` 容器本身必须是序列，不能是字符串或映射。"""
    with pytest.raises(ValueError, match="messages 必须是消息对象序列"):
        normalize_messages(None, None, bad_messages)


@pytest.mark.parametrize("prompt", [None, "", "   "], ids=["none", "empty", "blank"])
def test_normalize_messages_requires_prompt_or_messages(prompt):
    """既没有 ``messages`` 也没有非空 ``prompt`` 时必须报错。

    删除守卫后实测产出 ``{"role": "user", "content": None}``（或全空白内容）并
    发给模型——一次「请求内容为空」的配置错误变成真实提问。
    """
    with pytest.raises(ValueError, match="必须提供 prompt 或 messages"):
        normalize_messages(prompt, None, None)


@pytest.mark.parametrize("bad_content", [5, {"a": 1}, object()], ids=["int", "dict", "object"])
def test_responses_input_rejects_non_sequence_content(bad_content):
    """Responses 消息的 ``content`` 必须是字符串或内容块序列。"""
    with pytest.raises(ValueError, match="content 必须是字符串或内容块序列"):
        normalize_responses_input(None, None, [{"role": "user", "content": bad_content}])


@pytest.mark.parametrize("bad_content", [5, {"a": 1}], ids=["int", "dict"])
def test_responses_input_rejects_invalid_assistant_content_alongside_tool_calls(bad_content):
    """带 ``tool_calls`` 的 assistant 消息，其「有值但非法」的 ``content`` 必须显式报错。

    ``_responses_has_content`` 末行 ``return True`` 反向成 ``False`` 后实测：
    int/dict 这类非 ``None``、非字符串、非序列的 content 被判为「无内容」而整段
    丢弃，只剩 ``function_call`` 发给服务端——助手文本无声消失。既有
    ``test_responses_input_rejects_non_sequence_content`` 只走 ``role="user"``
    路径（另一处守卫），从未触发这里。
    """
    with pytest.raises(ValueError, match="content 必须是字符串或内容块序列"):
        normalize_responses_input(
            None,
            None,
            [
                {
                    "role": "assistant",
                    "content": bad_content,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ],
        )


def test_responses_input_rejects_function_role_without_tool_call_id():
    """``role="function"`` 的消息必须带 ``tool_call_id``。

    删除守卫后实测静默接受，产出没有 ``tool_call_id`` 的 function 消息——
    Responses 侧无法把它关联回任何一次函数调用。
    """
    with pytest.raises(ValueError, match="function 消息缺少 tool_call_id"):
        normalize_responses_input(None, None, [{"role": "function", "name": "f"}])


def test_normalization_still_accepts_well_formed_messages():
    """反向守卫：合法消息形态照常归一，不受上述收紧影响。"""
    chat = normalize_chat_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    )
    assert [message["role"] for message in chat] == ["user", "assistant"]

    responses = normalize_responses_input(
        None, None, [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    )
    assert responses[0]["role"] == "user"

    assert normalize_messages("hi", None, None) == [{"role": "user", "content": "hi"}]


# ── Ark Chat／Embedding／Classification 的形态判据 ──
#
# 六条判据在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后
# 全量 3434 条仍全绿）。删除后行为确实改变（实测，均经公开入口）：
# * 1027 ``max_tokens`` 与 ``max_completion_tokens`` 同时出现在模型 options →
#   **静默接受**，只有 ``max_tokens`` 进请求，另一处配置被无声丢弃；
# * 1404 原生 Responses 调用缺 ``input``／``input=None`` → **静默接受**，
#   请求不带 ``input`` 直接发给服务端；
# * 1420 ``session`` 非对象 → 带字段名的 ``ValueError`` 变成裸 ``TypeError:
#   'int' object is not iterable``（``dict(session)``）；
# * 2249／2296 Embedding 的 ``embed_documents``／``aembed_documents`` 请求参数
#   覆盖 ``input``／``model`` → 消息从「不允许覆盖请求字段」退化成
#   ``require_text_list`` 抛的「不支持请求参数」，把「参数冲突」说成「不支持的
#   参数」；
# * 2978 ``async_classify`` 的 ``labels`` 形态判据 → 非列表抛裸 ``TypeError``、
#   元素非字符串抛裸 ``AttributeError``。


def test_ark_chat_rejects_conflicting_max_tokens_options():
    """两处输出长度配置同时出现必须报错，不能静默只取其一。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "chat_completions"
    provider._options = {"max_tokens": 10, "max_completion_tokens": 20}

    with pytest.raises(ValueError, match="不能同时设置 max_tokens 和 max_completion_tokens"):
        provider._build_chat_request(prompt="hi", stream=False)


@pytest.mark.parametrize(
    "params",
    [{"model": "m"}, {"model": "m", "input": None}],
    ids=["missing-input", "none-input"],
)
def test_ark_native_responses_requires_input(params):
    """原生 Responses 调用缺 ``input`` 必须在请求前报错。"""
    with pytest.raises(ValueError, match="创建需要 input"):
        VolcengineProvider._validate_native_response_kwargs(params)


def test_ark_native_responses_rejects_non_object_session():
    """``session`` 非对象必须给出可读消息，不能漏出裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="session 必须是对象"):
        VolcengineProvider._validate_native_response_kwargs(
            {"model": "m", "input": "hi", "session": 5}
        )


def test_ark_native_responses_still_accepts_valid_session():
    """反向守卫：合法 ``session`` 照常归一为 dict。"""
    params = VolcengineProvider._validate_native_response_kwargs(
        {"model": "m", "input": "hi", "session": {"a": 1}}
    )

    assert params["session"] == {"a": 1}


def _ark_embedding_provider():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-embedding"
    provider._protocol = "chat_completions"
    provider._options = {}
    return provider


@pytest.mark.parametrize("field", ["input", "model"])
def test_ark_embedding_rejects_request_field_overrides(field):
    """请求参数覆盖 ``input``／``model`` 必须报「不允许覆盖请求字段」。

    删除守卫后消息退化成 ``require_text_list`` 的「不支持请求参数」——把参数
    冲突说成参数不受支持，排查方向被带偏。
    """
    provider = _ark_embedding_provider()

    with pytest.raises(ValueError, match="不允许覆盖请求字段"):
        provider.embed_documents(["a"], **{field: ["b"] if field == "input" else "other"})


@pytest.mark.parametrize("field", ["input", "model"])
def test_ark_async_embedding_rejects_request_field_overrides(field):
    """异步入口同判据（另一份实现）。"""
    provider = _ark_embedding_provider()

    async def run():
        return await provider.aembed_documents(
            ["a"], **{field: ["b"] if field == "input" else "other"}
        )

    with pytest.raises(ValueError, match="不允许覆盖请求字段"):
        asyncio.run(run())


@pytest.mark.parametrize(
    "labels",
    [5, [], [5], ["  "], ["ok", None]],
    ids=["int", "empty", "non-string-entry", "blank-entry", "none-entry"],
)
def test_ark_async_classify_rejects_malformed_labels(labels):
    """异步 Classification 的 ``labels`` 必须是非空字符串列表。

    删除守卫后实测非列表抛裸 ``TypeError: 'int' object is not iterable``、元素
    非字符串抛裸 ``AttributeError: 'int' object has no attribute 'strip'``。
    """
    provider = _ark_embedding_provider()

    async def run():
        return await provider.async_classify(query="q", labels=labels)

    with pytest.raises(ValueError, match="labels 必须是非空字符串列表"):
        asyncio.run(run())


@pytest.mark.parametrize(
    "labels",
    [5, [], [5], ["  "]],
    ids=["int", "empty", "non-string-entry", "blank-entry"],
)
def test_ark_sync_classify_rejects_malformed_labels(labels):
    """同步 Classification 是另一份实现，同样要有判据（既有用例已覆盖，这里作对照）。"""
    provider = _ark_embedding_provider()

    with pytest.raises(ValueError, match="labels 必须是非空字符串列表"):
        provider.classify(query="q", labels=labels)


# ── Anthropic 流终止与 parse 必填参数的八条守卫 ──
#
# 这八条在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后
# 全量 3479 条仍全绿）。既有用例只钉住 ``beta_tool_runner`` **同步**侧的
# ``compaction_control`` 与 ``output_format`` 在 create/stream 两条路径上的判据。
# 删除后行为确实改变（实测）：
# * 1175／1210 ``invoke``／``ainvoke`` 的流在 ``message_stop`` 之前结束 → **静默
#   返回已收到的片段**（实测 ``["hi"]``），调用方无法区分「答完了」与「断了」；
# * 1541／1634 ``stream_events``／``astream_events`` 同判据 → 静默产出一串
#   ``StreamEvent``，结构化事件流的调用方连「流是否完整」都无从判断；
# * 1663／2146 ``parse``／``async_parse`` 与 1815／2032 ``beta_parse``／
#   ``async_beta_parse`` 缺 ``output_format`` → 消息从「必须提供 output_format」
#   退化成 ``normalize_messages`` 的「聊天请求必须提供 prompt 或 messages」，
#   把「缺少结构化输出」说成「缺少提问」，排查方向被带偏；
# * 944 ``output_format`` 传字典 schema（流式路径）→ **静默接受**，把整个字典
#   当作 Python 类型原样放进请求的 ``output_format``；
# * 2124 ``async_beta_tool_runner`` 的 ``compaction_control``（异步侧是**另一处**）
#   → 消息退化成 ``CompletionRequest.__init__() got an unexpected keyword``，
#   既不提 Anthropic 也不提 compaction_control。

_ANTHROPIC_TEXT_EVENT = {
    "type": "content_block_delta",
    "index": 0,
    "delta": {"type": "text_delta", "text": "hi"},
}
_ANTHROPIC_STOP_EVENT = {"type": "message_stop"}


class _AnthropicSyncStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)


class _AnthropicAsyncStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _anthropic_stream_provider(events, *, is_async):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-fable-5"
    provider._options = {}
    provider._provider = "anthropic"

    class SyncContext:
        def __enter__(self):
            return _AnthropicSyncStream(events)

        def __exit__(self, *exc):
            return False

    class AsyncContext:
        async def __aenter__(self):
            return _AnthropicAsyncStream(events)

        async def __aexit__(self, *exc):
            return False

    if is_async:

        class Messages:
            @staticmethod
            def stream(**_kwargs):
                return AsyncContext()

        client = SimpleNamespace(aio=SimpleNamespace(messages=Messages()), messages=Messages())
    else:

        class Messages:
            @staticmethod
            def stream(**_kwargs):
                return SyncContext()

        client = SimpleNamespace(messages=Messages())

    provider._get_client = lambda: client
    provider._get_aclient = lambda: client
    return provider


def _anthropic_invoke_text(events, *, is_async):
    # ``_iter_invoke_stream_text``／``_aiter_invoke_stream_text`` 是 classmethod，
    # 直接消费响应流，不经过 provider 实例。
    response = _AnthropicAsyncStream(events) if is_async else _AnthropicSyncStream(events)
    if is_async:

        async def run():
            return [item async for item in AnthropicProvider._aiter_invoke_stream_text(response)]

        return asyncio.run(run())
    return list(AnthropicProvider._iter_invoke_stream_text(response))


def _anthropic_stream_events(events, *, is_async):
    provider = _anthropic_stream_provider(events, is_async=is_async)
    if is_async:

        async def run():
            return [item async for item in provider.astream_events(prompt="hi")]

        return asyncio.run(run())
    return list(provider.stream_events(prompt="hi"))


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_anthropic_invoke_stream_requires_message_stop(is_async):
    """``invoke`` 的文本流必须在 ``message_stop`` 之后才算完整。

    删除守卫后实测静默返回已收到的片段——半截答案当成完整答案交给调用方。
    """
    with pytest.raises(RuntimeError, match="message_stop 之前结束"):
        _anthropic_invoke_text([_ANTHROPIC_TEXT_EVENT], is_async=is_async)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_anthropic_stream_events_requires_message_stop(is_async):
    """``stream_events`` 是另一套实现，终止判据同样要生效。"""
    with pytest.raises(RuntimeError, match="message_stop 之前结束"):
        _anthropic_stream_events([_ANTHROPIC_TEXT_EVENT], is_async=is_async)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_anthropic_invoke_stream_still_yields_text_when_stopped(is_async):
    """反向守卫：带 ``message_stop`` 的正常流照常产出全部文本。"""
    assert _anthropic_invoke_text(
        [_ANTHROPIC_TEXT_EVENT, _ANTHROPIC_STOP_EVENT], is_async=is_async
    ) == ["hi"]


def _anthropic_parse_provider():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-fable-5"
    provider._options = {}
    provider._provider = "anthropic"
    return provider


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("parse", "Anthropic parse 必须提供 output_format"),
        ("beta_parse", "Anthropic Beta parse 必须提供 output_format"),
    ],
)
def test_anthropic_parse_requires_output_format(method, message):
    """同步两个 parse 入口缺 ``output_format`` 必须给专��消息。"""
    provider = _anthropic_parse_provider()

    with pytest.raises(ValueError, match="必须提供 output_format"):
        getattr(provider, method)(None)


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("async_parse", "Anthropic parse 必须提供 output_format"),
        ("async_beta_parse", "Anthropic Beta parse 必须提供 output_format"),
    ],
)
def test_anthropic_async_parse_requires_output_format(method, message):
    """异步两个 parse 入口是**另一份**实现，同样要有专��消息。"""
    provider = _anthropic_parse_provider()

    with pytest.raises(ValueError, match="必须提供 output_format"):
        asyncio.run(getattr(provider, method)(None))


def test_anthropic_stream_rejects_mapping_output_format():
    """流式路径的 ``output_format`` 必须是 Python 类型，不能是字典 schema。

    删除守卫后实测字典被原样放进请求的 ``output_format``——SDK 只接受类型对象，
    这个请求到服务端才失败。
    """
    provider = _anthropic_parse_provider()

    with pytest.raises(ValueError, match="必须是 Python 类型"):
        provider._build_message_params(prompt="hi", output_format={"type": "object"}, stream=True)


def test_anthropic_async_beta_tool_runner_rejects_compaction_control():
    """异步 tool runner 的 ``compaction_control`` 判据是**另一处**守卫。

    同步侧已有 ``test_anthropic_beta_tool_runner_rejects_unsupported_compaction_control``，
    异步侧此前从未执行。
    """
    provider = _anthropic_parse_provider()

    with pytest.raises(ValueError, match="当前 SDK 不支持 compaction_control"):
        asyncio.run(
            provider.async_beta_tool_runner([], prompt="hi", compaction_control={"enabled": True})
        )


# ── SDK 客户端属性的三条守卫 ──
#
# ``sdk_client``／``async_sdk_client`` 是 provider 基类上「供原生资源调用」的出口，
# 三条判据在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass``
# 后全量 3540 条仍全绿）。删除后行为确实改变（实测）：
# * 2140 同步侧没有 ``_get_client`` → 带 provider 类名的 ``AttributeError``
#   变成裸 ``TypeError: 'NoneType' object is not callable``；
# * 2150 ``_get_aclient()`` 返回 ``None`` → **静默返回 None**，调用方拿到
#   ``None`` 当客户端用，在更远处才崩；
# * 2158 客户端没有 ``aio`` 节点 → **静默返回 None**，同样把「没有异步客户端」
#   说成「有，但它是 None」。
#
# 同批的 2155（``client_getter()`` 返回 ``None``）经甄别是**等价变异**，不补
# 测试：删掉后 ``getattr(None, "aio", None)`` 得到 ``None``，随即被 2158 以
# **同一条消息**拦住，结论不变。


class _ClientlessProvider(LargeLanguageModel):
    """没有 ``_get_client``／``_get_aclient`` 的 provider——两者都是纯基类实现。"""

    def invoke(self, *_args, **_kwargs):
        return iter(())

    async def ainvoke(self, *_args, **_kwargs):
        return
        yield  # pragma: no cover - 让函数成为异步生成器

    def stream_events(self, *_args, **_kwargs):
        return iter(())

    async def astream_events(self, *_args, **_kwargs):
        return
        yield  # pragma: no cover - 让函数成为异步生成器


def test_sdk_client_reports_missing_getter_readably():
    """没有 ``_get_client`` 时必须报出 provider 类名，而不是裸 ``TypeError``。"""
    with pytest.raises(AttributeError, match="_ClientlessProvider 没有同步 SDK 客户端"):
        _ = _ClientlessProvider().sdk_client


def test_async_sdk_client_reports_none_client_readably():
    """``_get_aclient()`` 返回 ``None`` 时必须报错，不能静默交出 ``None``。"""
    provider = _ClientlessProvider()
    provider._get_aclient = lambda: None

    with pytest.raises(AttributeError, match="_ClientlessProvider 没有异步 SDK 客户端"):
        _ = provider.async_sdk_client


def test_async_sdk_client_reports_missing_aio_node_readably():
    """客户端没有 ``aio`` 节点时必须报错，不能静默交出 ``None``。

    这是「没有异步客户端」与「有但它是 None」的分界——两条守卫消息相同但位置
    不同，少一条就会把一种说成另一种。
    """
    provider = _ClientlessProvider()
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(AttributeError, match="_ClientlessProvider 没有异步 SDK 客户端"):
        _ = provider.async_sdk_client


def test_sdk_client_still_resolves_through_aio_node():
    """反向守卫：同步客户端可用时，异步出口从 ``aio`` 节点解析。"""
    provider = _ClientlessProvider()
    aio = object()
    provider._get_client = lambda: SimpleNamespace(aio=aio)

    assert provider.async_sdk_client is aio


# ── OpenAI 兼容渠道 openai_compatible.py 覆盖缺口守卫 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证：把它们的 ``raise`` 换成
# ``pass`` 后全量 3552 条仍全绿，唯一被抓住的 1208／1706 是覆盖器的假阳性、
# 既有用例已覆盖）。删除后行为确实改变（实测，均经真实入口）：
# * 289 构造期缺 ``api_key`` → **静默接受**，无凭证的 provider 被构造出来；
# * 343 ``_normalize_protocol`` 未知协议 → **静默返回 None**，``_protocol`` 变
#   ``None``，后续 ``== "responses"`` 判断全部落空；
# * 478 ``_validate_sdk_resource_call`` 的 ``Signature.bind`` 抛 ``ValueError``
#   （位置专用参数）→ **静默跳过**参数校验，未知参数一路发到 SDK；
# * 568 ``_background_poll_config`` 的 ``timeout`` 非正／非数值 → 0 与负值被
#   **静默接受**，``None`` 抛裸 ``TypeError``，超大整数抛裸 ``OverflowError``；
# * 585／617 后台响应缺 ``id`` → 裸 ``AttributeError``（``_client``／``_aclient``）；
# * 920 ``_require_responses_resource`` 非 responses 协议 → 不再拦，把本地 SDK
#   资源当远端 Responses 能力；
# * 936 ``_merge_options`` options 覆盖请求字段 → **静默覆盖** ``model``／``input``；
# * 987 ``_merge_resource_kwargs`` 调用方覆盖 Facade 已绑定字段 → **静默覆盖**；
# * 1145 Chat ``seed`` 与 options.extra_body 重复 → 消息退化成「extra_body 不允许
#   覆盖请求字段」；
# * 1264 ``_convert_responses_tool_choice`` 既非字符串也非 Mapping → 裸 ``TypeError``；
# * 1273／1279 function tool_choice 缺 name → 1273 删掉后**静默产出**
#   ``{"type": "function", "name": None}``，1279 删掉后**静默返回 None**；
# * 1627 ``_convert_responses_format`` 的 json_schema 非对象 → 裸 ``AttributeError``；
# * 1649 response_format 的 type 不支持且无 format → **静默返回 None**；
# * 1885 Chat 错误体含 ``error`` 且状态码正常 → 消息退化成「响应缺少有效 choices」；
# * 1906 ``allow_empty_choices`` 下 ``choices=None`` 且有 detail → **静默接受**；
# * 2290 Responses 终止事件非 ``response.completed`` → **静默产出一个 finish 事件**；
# * 2580／2617 ``parse``／``async_parse`` 缺 text_format → 消息退化成
#   ``normalize_messages`` 的「聊天请求必须提供 prompt 或 messages」；
# * 2598／2632 ``parse`` 同时用 request.response_format 与 text_format →
#   **静默接受**，两者之一被无声丢弃；
# * 3078／3137／3117 Embedding 请求参数与模型 options 重复／覆盖 ``model`` →
#   **静默接受**或消息退化；
# * 3195／4465 ``file_content`` 缺 ``files.content`` 资源 → 裸
#   ``TypeError: 'NoneType' object is not callable``，同构的
#   ``container_file_content`` 却有专属 ``NotImplementedError``（sibling 缺口）；
# * 3355／4612 ``stream_responses``／``async_stream_responses`` 同时传 request 与
#   原生关键字参数 → 裸 ``AttributeError``。
#
# 同批的 951（``_validated_extra_body`` 非对象）经甄别是**等价变异**，不补测试：
# 删掉后落到下游 ``validate_secret_free_payload``，抛同一条消息的同一个
# ``ValueError``，结论不变。


def _openai_compatible_provider(**options):
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._base_url = None
    provider._options = options
    return provider


def test_openai_compatible_constructor_requires_api_key():
    """构造期缺 ``api_key`` 必须报错，不能静默构造出无凭证的 provider。"""
    settings = SimpleNamespace(demo_api_key=None, demo_base_url="https://demo.invalid/v1")

    with pytest.raises(ValueError, match="demo_api_key is required for demo"):
        OpenAICompatibleProvider("m", "demo", settings=settings, options={})


@pytest.mark.parametrize(
    "protocol",
    [
        pytest.param("grpc", id="unknown"),
        pytest.param("", id="empty"),
        pytest.param("chat-completions-x", id="near-miss"),
    ],
)
def test_openai_compatible_protocol_rejects_unsupported_names(protocol):
    """未知协议名必须报错，不能静默返回 ``None`` 让后续分支全部落空。"""
    with pytest.raises(ValueError, match="不支持 .* 协议"):
        OpenAICompatibleProvider._normalize_protocol(protocol)


def test_openai_compatible_sdk_parameter_validation_failure_is_not_skipped():
    """``Signature.bind`` 抛 ``ValueError`` 时不得静默跳过校验。"""

    class _ValueErrorSignature(inspect.Signature):
        def bind(self, *args, **kwargs):
            raise ValueError("positional-only argument passed as keyword")

        def bind_partial(self, *args, **kwargs):
            raise ValueError("positional-only argument passed as keyword")

    def method(x):
        return x

    method.__signature__ = _ValueErrorSignature()

    with pytest.raises(ValueError, match="SDK 参数无法验证"):
        OpenAICompatibleProvider._validate_sdk_resource_call(method, (), {}, "OpenAI 资源")


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param("0", id="string"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
        pytest.param(10**4301, id="huge-int"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_openai_compatible_background_timeout_must_be_positive_number(timeout):
    """``background_timeout`` 非正／非数值必须报错，不是静默接受或裸异常。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._options = {"background_timeout": timeout}

    with pytest.raises(ValueError, match="background_timeout 必须是正数"):
        provider._background_poll_config()


@pytest.mark.parametrize(
    "status",
    [pytest.param("queued", id="queued"), pytest.param("in_progress", id="in-progress")],
)
def test_openai_compatible_background_poll_requires_response_id(status):
    """后台响应缺 ``id`` 必须报 ``RuntimeError``，不是裸 ``AttributeError``。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "demo"

    with pytest.raises(RuntimeError, match="background 响应缺少 id"):
        provider._poll_background_response({"status": status, "id": ""}, {})


def test_openai_compatible_background_poll_requires_response_id_async():
    """异步分支同判据。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "demo"

    with pytest.raises(RuntimeError, match="background 响应缺少 id"):
        asyncio.run(provider._poll_background_response_async({"status": "queued", "id": ""}, {}))


def test_openai_compatible_responses_resource_rejects_chat_protocol():
    """非 responses 协议下不得把本地 SDK 资源当成远端 Responses 能力。"""
    provider = _openai_compatible_provider()
    provider._protocol = "chat_completions"
    provider._get_client = lambda: pytest.fail("非 responses 协议不应初始化客户端")

    with pytest.raises(ValueError, match="仅适用于 responses 协议"):
        provider.retrieve_response("resp-1")


@pytest.mark.parametrize("field", ["model", "input", "messages"])
def test_openai_compatible_merge_options_rejects_reserved_field_override(field):
    """``options`` 覆盖请求保留字段必须报错，不能静默覆盖。"""
    with pytest.raises(ValueError, match="Provider options 不允许覆盖请求字段"):
        OpenAICompatibleProvider._merge_options({"model": "a", "input": "b"}, {field: "x"})


def test_openai_compatible_merge_resource_kwargs_rejects_caller_override():
    """调用方覆盖 Facade 已绑定的资源字段必须报错，不能静默覆盖。"""
    with pytest.raises(ValueError, match="资源 SDK 参数重复"):
        OpenAICompatibleProvider._merge_resource_kwargs(
            {"response_id": "r1"}, {"response_id": "r2"}, "拉取响应"
        )


def test_openai_compatible_chat_seed_duplicating_extra_body_reports_seed():
    """Chat ``seed`` 与 options.extra_body 重复时消息必须点名 ``seed``。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {"extra_body": {"seed": 1}}
    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace()))
    )

    with pytest.raises(ValueError, match="Chat Completions seed 与模型 options.extra_body 重复"):
        provider._build_chat_request(prompt="hi", seed=2)


@pytest.mark.parametrize(
    "tool_choice",
    [pytest.param(5, id="int"), pytest.param(1.5, id="float"), pytest.param([1], id="list")],
)
def test_openai_compatible_responses_tool_choice_rejects_non_string_non_mapping(tool_choice):
    """``tool_choice`` 既非字符串也非 Mapping 必须给带字段名的 ``ValueError``。"""
    with pytest.raises(ValueError, match="必须是字符串或对象"):
        OpenAICompatibleProvider._convert_responses_tool_choice(tool_choice)


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param({"type": "function", "function": {}}, id="empty-function"),
        pytest.param({"type": "function", "function": {"name": ""}}, id="blank-name"),
    ],
)
def test_openai_compatible_responses_tool_choice_requires_function_name(tool_choice):
    """``function`` 形态的 tool_choice 缺 name 必须报错，不能静默产出无名对象。"""
    with pytest.raises(ValueError, match="function tool_choice 缺少 name"):
        OpenAICompatibleProvider._convert_responses_tool_choice(tool_choice)


def test_openai_compatible_responses_tool_choice_function_not_mapping_without_name():
    """``function`` 非 Mapping 且顶层无 name 时，兜底判据必须拦住，不能静默返回 None。"""
    with pytest.raises(ValueError, match="function tool_choice 缺少 name"):
        OpenAICompatibleProvider._convert_responses_tool_choice({"type": "function", "function": 5})


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param({"type": "json_schema", "json_schema": 5}, id="int"),
        pytest.param({"type": "json_schema", "json_schema": "s"}, id="str"),
    ],
)
def test_openai_compatible_responses_format_rejects_non_object_json_schema(response_format):
    """``json_schema`` 非对象必须报错，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="json_schema 必须是对象"):
        OpenAICompatibleProvider._convert_responses_format(response_format)


@pytest.mark.parametrize(
    "response_format",
    [pytest.param({"type": "xml"}, id="xml"), pytest.param({}, id="empty")],
)
def test_openai_compatible_responses_format_rejects_unsupported_type(response_format):
    """不支持的 ``type`` 必须报错，不能静默返回 ``None`` 丢掉结构化输出约束。"""
    with pytest.raises(ValueError, match="仅支持 json_schema 或 json_object"):
        OpenAICompatibleProvider._convert_responses_format(response_format)


def test_openai_compatible_chat_error_body_reports_detail():
    """Chat 错误体含 ``error`` 且状态码正常时必须报 detail，不是「缺少 choices」。"""
    with pytest.raises(RuntimeError, match="返回错误: boom"):
        OpenAICompatibleProvider._raise_for_chat_response_error(
            {"status": 200, "error": {"message": "boom"}}, provider="openai"
        )


def test_openai_compatible_chat_error_with_empty_choices_and_detail():
    """``allow_empty_choices`` 下 ``choices=None`` 且有 detail 时必须报错。"""
    with pytest.raises(RuntimeError, match="返回错误"):
        OpenAICompatibleProvider._raise_for_chat_response_error(
            {"choices": None, "msg": "boom"}, allow_empty_choices=True, provider="openai"
        )


def test_openai_compatible_terminal_event_without_error_status_is_rejected():
    """终止事件非 ``response.completed`` 必须报错，不能静默产出 finish 事件。"""
    with pytest.raises(RuntimeError, match="收到终止事件: response.incomplete"):
        OpenAICompatibleProvider._responses_stream_event(
            {"type": "response.incomplete", "response": {}}, error_provider="openai"
        )


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_compatible_parse_requires_text_format(async_mode):
    """``parse``／``async_parse`` 缺 ``text_format`` 必须报专属消息。"""
    provider = _openai_compatible_provider()

    with pytest.raises(ValueError, match="必须提供 text_format"):
        if async_mode:
            asyncio.run(provider.async_parse(None))
        else:
            provider.parse(None)


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_compatible_parse_rejects_dual_response_format(async_mode):
    """``request.response_format`` 与 ``text_format`` 同时使用必须报错，不能静默接受。"""
    provider = _openai_compatible_provider()
    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kw: "ok")
    )

    async def async_parse():
        return await provider.async_parse(
            dict, CompletionRequest(prompt="hi", response_format={"type": "json_object"})
        )

    with pytest.raises(ValueError, match="不能同时使用 request.response_format 和 text_format"):
        if async_mode:
            asyncio.run(async_parse())
        else:
            provider.parse(
                dict, CompletionRequest(prompt="hi", response_format={"type": "json_object"})
            )


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_compatible_embedding_rejects_request_option_overlap(async_mode):
    """Embedding 请求参数与模型 options 重复时必须报错，不能静默接受。"""
    provider = _openai_compatible_provider(dimensions=128)
    provider._get_client = lambda: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[]))
    )
    provider._get_aclient = lambda: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[]))
    )

    async def run():
        return await provider.aembed_documents(["x"], dimensions=64)

    with pytest.raises(ValueError, match="Embedding 请求参数与模型 options 重复"):
        if async_mode:
            asyncio.run(run())
        else:
            provider.embed_documents(["x"], dimensions=64)


def test_openai_compatible_embedding_rejects_reserved_field_override():
    """Embedding 请求参数覆盖 ``model`` 必须报「不允许覆盖请求字段」。"""
    provider = _openai_compatible_provider()
    provider._get_aclient = lambda: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[]))
    )

    async def run():
        return await provider.aembed_documents(["x"], model="other")

    with pytest.raises(ValueError, match="Embedding 不允许覆盖请求字段"):
        asyncio.run(run())


def test_openai_compatible_file_content_reports_missing_sdk_resource():
    """``file_content`` 缺 ``files.content`` 必须报 ``NotImplementedError``。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._get_client = lambda: SimpleNamespace(files=SimpleNamespace())

    with pytest.raises(NotImplementedError, match=r"files\.content"):
        provider.file_content("file-1")


def test_openai_compatible_async_file_content_reports_missing_sdk_resource():
    """异步分支同判据。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._get_aclient = lambda: SimpleNamespace(files=SimpleNamespace())

    async def run():
        return await provider.async_file_content("file-1")

    with pytest.raises(NotImplementedError, match=r"files\.content"):
        asyncio.run(run())


@pytest.mark.parametrize("async_mode", [False, True])
def test_openai_compatible_stream_responses_rejects_request_and_kwargs(async_mode):
    """``stream_responses`` 同时传 request 与原生关键字参数必须报错。"""
    provider = _openai_compatible_provider()
    provider._get_client = lambda: pytest.fail("双来源必须在构造请求前拦住")

    with pytest.raises(ValueError, match="不能同时传 request 和原生关键字参数"):
        if async_mode:
            provider.async_stream_responses(CompletionRequest(prompt="hi"), model="x")
        else:
            provider.stream_responses(CompletionRequest(prompt="hi"), model="x")


# ── Ark／Volcengine 渠道 volcengine.py 覆盖缺口守卫 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3598 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 106／125 ``_require_ark_list``／``_require_ark_content`` 非列表 → 裸
#   ``TypeError: 'int' object is not iterable``（str 还会被**逐字符**当成元素）；
# * 558 ``__init__`` 未知协议 → 不再拦，``_protocol`` 落空；
# * 582 ``__init__`` 缺全部凭证 → **静默构造**出无凭证的 provider；
# * 605 ``_client_options`` 认证不完整 → **静默返回空字典**，SDK 拿到无认证客户端；
# * 648 ``_normalize_client_options`` region 非字符串 → 裸 ``AttributeError``，
#   空白 → **静默接受**、region 变成空串；
# * 660／671 ``_normalize_verified_protocols`` 非序列 → 裸 ``TypeError``，元素
#   非法 → 裸 ``AttributeError``；
# * 675 登记未知协议 → **静默接受**，``frozenset({'grpc'})`` 通过；
# * 801 ``_require_classification_resource`` 资源未挂载 → **静默返回 None**；
# * 857／889 ``_convert_responses_format``／``_convert_responses_tools`` 形态 →
#   裸 ``AttributeError``／``TypeError``；
# * 1035／1040／1084／1088 Ark Chat 的 top_k／seed 在模型 options 与 extra_body
#   两处配置 → **静默接受**，其中一个被无声丢弃；
# * 1108 extra_body 与模型 options 重复 → **静默覆盖**；
# * 1507 ``_prepare_batch_kwargs`` 缺必填参数 → 消息退化成「不支持请求参数」，
#   把「缺字段」说成「字段不受支持」；
# * 1510 资源 model 非字符串／空白 → **静默接受**，非法 model 一路发给 SDK；
# * 1663 Responses 状态 ``incomplete`` → **静默返回 None**，不完整响应当成成功；
# * 2396／2718 ``create_response``／``async_create_response`` 同时传 request 与
#   原生关键字参数 → 裸 ``AttributeError``；
# * 2525／2881 ``count_tokens``／``async_count_tokens`` 响应缺 ``data`` →
#   **静默返回 0**，把「响应不完整」报成「没有 token」；
# * 2608／2957 Classification 资源导入失败 → 裸 ``UnboundLocalError``；
# * 3028／3051 ``close``／``aclose`` 首个关闭异常 → **静默吞掉**，不向上报告。
#
# 同批的 578（部分 AK/SK）与 631（timeout 非数值）经甄别是**等价变异**，不补测试：
# 578 删掉后落到 582 的同一条消息，631 删掉后落到下游 ``normalize_client_timeout``
# 的同一条消息，结论都不变。


def _volcengine_provider(**options):
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "demo"
    provider._provider = "volcengine"
    provider._protocol = "chat_completions"
    provider._options = options
    provider._base_url = None
    provider._api_key = "ak"
    provider._secret_key = "sk"
    provider._ark_api_key = "ark-key"
    return provider


def _volcengine_init(settings):
    """用给定 settings 跑一遍真实构造路径（构造期凭证判据在 ``get_settings`` 之后）。

    补丁打在 ``VolcengineProvider`` 所属类的 ``__globals__`` 上，而**不是**
    重新 ``import src.providers.volcengine``：``tests/providers/test_factory.py``
    的 autouse 夹具会把该模块从 ``sys.modules`` 删除并重新导入，此时
    ``import ... as m`` 拿到的是**新模块对象**，而类仍从**旧模块字典**取全局名
    ——补丁会静默落空，走真实 ``get_settings``（全量套件下才复现）。
    """
    module_globals = VolcengineProvider.__init__.__globals__

    original = module_globals["get_settings"]
    module_globals["get_settings"] = lambda: settings
    try:
        return VolcengineProvider("m", protocol="ark", options={})
    finally:
        module_globals["get_settings"] = original


async def _async_value(value):
    return value


class _ClassificationImportBlocker:
    """拦截 ``volcenginesdkarkruntime.resources.classification`` 的导入。"""

    _TARGET = "volcenginesdkarkruntime.resources.classification"

    def find_spec(self, name, path=None, target=None):
        if name == self._TARGET:
            raise ImportError(f"No module named {name!r}")


def _with_blocked_classification_import(call):
    # 必须先逐出 ``sys.modules`` 里的缓存：全套运行时更早的用例可能已经导入过该
    # 模块，此后 ``from ... import`` 直接命中缓存、根本不会走 ``find_spec``，拦截器
    # 形同虚设（单跑时缓存为空才通过）。逐出后再装拦截器，才能稳定复现「SDK 未
    # 提供该资源」这条分支。
    target = _ClassificationImportBlocker._TARGET
    cached = sys.modules.pop(target, None)
    blocker = _ClassificationImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        return call()
    finally:
        sys.meta_path.remove(blocker)
        if cached is not None:
            sys.modules[target] = cached


@pytest.mark.parametrize(
    "value",
    [pytest.param(5, id="int"), pytest.param(1.5, id="float"), pytest.param({"a": 1}, id="dict")],
)
def test_ark_require_list_rejects_non_sequence(value):
    """Ark 容器字段非列表必须报 ``RuntimeError``，不是裸 ``TypeError``。"""
    with pytest.raises(RuntimeError, match="必须是列表"):
        _require_ark_list(value, label="output")


def test_ark_require_list_treats_string_as_sequence_of_characters():
    """字符串会被**逐字符**当成元素——必须显式拒绝，不能静默拆成字符列表。"""
    with pytest.raises(RuntimeError, match="必须是列表"):
        _require_ark_list("ab", label="output")


def test_ark_require_content_rejects_non_sequence():
    """嵌套 ``content`` 非列表必须报 ``RuntimeError``。"""
    with pytest.raises(RuntimeError, match="必须是列表"):
        _require_ark_content({"content": 5}, label="content")


@pytest.mark.parametrize(
    "protocol",
    [pytest.param("grpc", id="unknown"), pytest.param("", id="empty")],
)
def test_volcengine_init_rejects_unsupported_protocol(protocol):
    """未知协议必须报错，不能静默留下落空的 ``_protocol``。"""
    with pytest.raises(ValueError, match="仅支持 ark/chat_completions 或 responses"):
        VolcengineProvider("m", protocol=protocol)


def test_volcengine_init_requires_credentials():
    """缺全部凭证必须报错，不能静默构造出无凭证的 provider。"""
    settings = SimpleNamespace(
        volc_access_key=None, volc_secret_key=None, ark_api_key=None, volc_base_url=None
    )

    with pytest.raises(ValueError, match="需要配置 ARK_API_KEY"):
        _volcengine_init(settings)


def test_volcengine_client_options_require_complete_credentials():
    """``_client_options`` 认证不完整必须报错，不能静默返回空字典。"""
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._base_url = None
    provider._api_key = None
    provider._secret_key = None
    provider._ark_api_key = None

    with pytest.raises(ValueError, match="认证配置不完整"):
        provider._client_options()


def test_volcengine_client_options_reject_non_string_region():
    """region 非字符串必须报错，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="region 必须是非空字符串"):
        VolcengineProvider._normalize_client_options({"region": 5})


def test_volcengine_client_options_reject_blank_region():
    """region 纯空白必须报错，不能静默收敛成空串。"""
    with pytest.raises(ValueError, match="region 必须是非空字符串"):
        VolcengineProvider._normalize_client_options({"region": "   "})


@pytest.mark.parametrize(
    "configured",
    [pytest.param(5, id="int"), pytest.param(1.5, id="float")],
)
def test_volcengine_verified_protocols_reject_non_sequence(configured):
    """``server_verified_protocols`` 非序列必须报错，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="必须是字符串或协议序列"):
        VolcengineProvider._normalize_verified_protocols(configured)


def test_volcengine_verified_protocols_reject_non_string_element():
    """登记项非字符串必须报错，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="中的协议必须是非空字符串"):
        VolcengineProvider._normalize_verified_protocols([5])


def test_volcengine_verified_protocols_reject_unknown_name():
    """登记未知协议必须报错，不能静默接受 ``frozenset({'grpc'})``。"""
    with pytest.raises(ValueError, match="不支持登记服务端协议: grpc"):
        VolcengineProvider._normalize_verified_protocols(["grpc"])


def test_ark_classification_guard_reports_missing_resource():
    """分类资源未挂载时必须报错，不能静默返回 ``None``。"""
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._client = object()
    provider._get_classification_resource = lambda: (_ for _ in ()).throw(
        NotImplementedError("no classification")
    )

    with pytest.raises(NotImplementedError, match="Ark classify"):
        provider._require_classification_resource("classify")


@pytest.mark.parametrize(
    "response_format",
    [pytest.param(5, id="int"), pytest.param("s", id="str")],
)
def test_ark_responses_format_rejects_non_object(response_format):
    """``response_format`` 非对象必须报错，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="response_format 必须是对象"):
        VolcengineProvider._convert_responses_format(response_format)


@pytest.mark.parametrize("tools", [pytest.param(5, id="int"), pytest.param(1.5, id="float")])
def test_ark_responses_tools_reject_non_sequence(tools):
    """``tools`` 非列表必须报错，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="工具定义必须是列表"):
        VolcengineProvider._convert_responses_tools(tools)


@pytest.mark.parametrize("field", ["top_k", "seed"])
def test_ark_chat_model_options_reject_duplicate_with_extra_body(field):
    """模型 options 的 top_k／seed 与 extra_body 重复时必须报错，不能静默丢弃。"""
    provider = _volcengine_provider(extra_body={field: 1}, **{field: 2})

    with pytest.raises(ValueError, match=rf"模型 options 的 {field} 与 extra_body 重复"):
        provider._build_chat_request("hi")


@pytest.mark.parametrize("field", ["top_k", "seed"])
def test_ark_chat_request_rejects_duplicate_with_model_extra_body(field):
    """请求级 top_k／seed 与模型 options.extra_body 重复时必须报错。"""
    provider = _volcengine_provider(extra_body={field: 1})

    with pytest.raises(ValueError, match=rf"{field} 与模型 options.extra_body 重复"):
        provider._build_chat_request("hi", **{field: 2})


def test_ark_chat_extra_body_duplicate_with_model_options():
    """请求级 extra_body 与模型 options 重复时必须报错，不能静默覆盖。"""
    provider = _volcengine_provider(extra_body={"a": 1})

    with pytest.raises(ValueError, match="extra_body 与模型 options 重复"):
        provider._build_chat_request("hi", extra_body={"a": 2})


def test_ark_batch_kwargs_reports_missing_required_parameter():
    """缺必填参数必须点名，不能退化成「不支持请求参数」。"""
    with pytest.raises(ValueError, match="缺少必填参数: name"):
        VolcengineProvider._prepare_batch_kwargs(
            {}, "demo", "知识库", frozenset({"name"}), ("name",)
        )


@pytest.mark.parametrize("model", [pytest.param(5, id="int"), pytest.param("   ", id="blank")])
def test_ark_batch_kwargs_rejects_invalid_model(model):
    """资源 model 非字符串／空白必须报错，不能静默发给 SDK。"""
    with pytest.raises(ValueError, match="model 必须是非空字符串"):
        VolcengineProvider._prepare_batch_kwargs(
            {"model": model, "name": "k"},
            "demo",
            "知识库",
            frozenset({"model", "name"}),
            ("name",),
        )


def test_ark_responses_incomplete_status_is_rejected():
    """状态 ``incomplete`` 必须报错，不能静默当成成功。"""
    with pytest.raises(RuntimeError, match="响应不完整"):
        VolcengineProvider._raise_for_response_error(
            {"status": "incomplete", "incomplete_details": {"reason": "max_tokens"}}
        )


def test_ark_responses_cancelled_status_is_rejected():
    """状态 ``cancelled`` 必须报错，不能静默返回 None。

    删除守卫后 ``status == "cancelled"`` 的响应不再抛错，调用方把「请求已取消」
    当成成功响应继续用（``_raise_for_response_error`` 静默返回 ``None``）。
    变异扫描证实该守卫在整套测试里没有反向覆盖（既有 ``cancelled`` 用例打的是
    流式事件路径 ``_responses_stream_event``，不是本函数）。
    """
    with pytest.raises(RuntimeError, match="已取消"):
        VolcengineProvider._raise_for_response_error({"status": "cancelled"})


def test_ark_responses_incomplete_reason_with_huge_int_reports_readably():
    """``incomplete`` 原因是大整数时消息不得抛位数超限的 ``ValueError``。"""
    with pytest.raises(RuntimeError) as excinfo:
        VolcengineProvider._raise_for_response_error(
            {"status": "incomplete", "incomplete_details": {"reason": 10**4300}}
        )

    assert "Exceeds the limit" not in str(excinfo.value)


@pytest.mark.parametrize("async_mode", [False, True])
def test_ark_create_response_rejects_request_and_kwargs(async_mode):
    """``create_response`` 同时传 request 与原生关键字参数必须报错。"""
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._server_verified_protocols = frozenset({"responses"})
    provider._get_client = lambda: pytest.fail("双来源必须在构造请求前拦住")
    provider._get_aclient = lambda: pytest.fail("双来源必须在构造请求前拦住")

    with pytest.raises(ValueError, match="不能同时传 request 和原生关键字参数"):
        if async_mode:
            asyncio.run(provider.async_create_response(CompletionRequest(prompt="hi"), model="x"))
        else:
            provider.create_response(CompletionRequest(prompt="hi"), model="x")


@pytest.mark.parametrize(
    "items",
    [pytest.param([], id="empty"), pytest.param(None, id="missing-field")],
)
def test_ark_count_tokens_requires_data(items):
    """tokenization 响应缺 ``data`` 必须报错，不能静默返回 0。"""
    provider = _ark_provider_with_tokenization_response(items)

    with pytest.raises(RuntimeError, match="tokenization 响应缺少 data"):
        provider.count_tokens("hi")


@pytest.mark.parametrize(
    "items",
    [pytest.param([], id="empty"), pytest.param(None, id="missing-field")],
)
def test_ark_async_count_tokens_requires_data(items):
    """异步分支同判据。"""
    provider = _ark_provider_with_tokenization_response(items)
    provider._get_aclient = lambda: SimpleNamespace(
        tokenization=SimpleNamespace(
            create=lambda **kwargs: _async_value(SimpleNamespace(data=items))
        )
    )

    async def run():
        return await provider.async_count_tokens("hi")

    with pytest.raises(RuntimeError, match="tokenization 响应缺少 data"):
        asyncio.run(run())


def test_ark_classification_import_failure_reports_readably():
    """Classification 模块导入失败时必须报 ``NotImplementedError``，不是裸 ``UnboundLocalError``。"""
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._classification_resource = None
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="Ark Classification 资源"):
        _with_blocked_classification_import(provider._get_classification_resource)


def test_ark_async_classification_import_failure_reports_readably():
    """异步分支同判据。"""
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._async_classification_resource = None
    provider._get_aclient = lambda: SimpleNamespace()

    async def run():
        return await provider._get_async_classification_resource()

    with pytest.raises(NotImplementedError, match="Ark Async Classification 资源"):
        _with_blocked_classification_import(lambda: asyncio.run(run()))


@pytest.mark.parametrize("async_mode", [False, True])
def test_volcengine_close_reports_first_error(async_mode):
    """``close``／``aclose`` 首个关闭异常必须向上报告，不能静默吞掉。"""

    class _BadClose:
        def close(self):
            raise RuntimeError("close 失败")

    provider = object.__new__(VolcengineProvider)
    provider._client = _BadClose()
    provider._aclient = None
    provider._classification_resource = None
    provider._async_classification_resource = None

    with pytest.raises(RuntimeError, match="close 失败"):
        if async_mode:
            asyncio.run(provider.aclose())
        else:
            provider.close()


# ── OpenAICompatibleProvider Responses 扩展字段重复 / close 的覆盖缺口 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3698 条仍全绿）。删除后行为确实改变（实测，经
# ``_build_responses_request`` 真实入口）：
# * 1415 ``options.stop``／``options.seed`` 与 ``extra_body`` 重复 → 消息从
#   「Responses options.X 与 extra_body 重复」退化成下游
#   ``_merge_extra_body`` 的「模型 options 的扩展字段重复」：同一个重复被报两次，
#   但**点名的是合并器而不是字段来源**，调用方按消息定位要改的地方会改错；
# * 1553 ``options.reasoning_effort`` 与 ``extra_body.reasoning`` 重复 → 消息
#   退化成「extra_body 不允许覆盖请求字段: reasoning」，把「两个来源配了同一个
#   意图」说成「extra_body 越权覆盖请求字段」；
# * 5678／5699 ``close``／``aclose`` 首个关闭异常 → **静默吞掉**（Google 与
#   Volcengine 的两个入口都有覆盖，OpenAI 兼容渠道没有——sibling 缺口）。
def _responses_provider(**options):
    provider = _openai_compatible_provider(**options)
    provider._server_verified_protocols = frozenset({"responses"})
    return provider


@pytest.mark.parametrize("field,value", [("stop", ["x"]), ("seed", 7)])
def test_openai_compatible_responses_rejects_option_duplicated_in_extra_body(field, value):
    """``options`` 的 stop／seed 与 ``extra_body`` 重复时必须点名是哪个字段重复。

    只覆盖 ``configured_options`` 那一半：``_compat_extra_options()`` 里的
    ``top_k``／``repetition_penalty`` 走的是下游 ``_merge_extra_body``，报的是
    「模型 options 的扩展字段重复」，与本条不是同一判据。
    """
    provider = _responses_provider(**{field: value, "extra_body": {field: "other"}})

    with pytest.raises(ValueError, match=rf"Responses options\.{field} 与 extra_body 重复"):
        provider._build_responses_request(prompt="hi")


def test_openai_compatible_responses_rejects_reasoning_effort_duplicated_in_extra_body():
    """``reasoning_effort`` 与 ``extra_body.reasoning`` 重复时必须点名来源。"""
    provider = _responses_provider(
        reasoning_effort="high", extra_body={"reasoning": {"effort": "low"}}
    )

    with pytest.raises(
        ValueError, match="Responses reasoning_effort 与模型 options.extra_body 重复"
    ):
        provider._build_responses_request(prompt="hi")


def test_openai_responses_rejects_reasoning_duplicating_model_extra_body():
    """显式 ``reasoning`` 与 ``extra_body.reasoning`` 重复时必须点名来源。

    ``_build_chat_request`` 的同名判据早有覆盖，但 Responses 路径（1549）是
    **另一处**独立判据，变异扫描证实它在整套测试里存活：删除后消息退化成
    ``_validated_extra_body`` 的「不允许覆盖请求字段: reasoning」——只说
    「冲突」，不再指出冲突的是「模型 options 里的 extra_body」，用户不知道该
    去改配置文件还是改调用参数。同函数 1553 的 ``reasoning_effort`` 分支已有
    用例钉住（变异后变红），本条补对称的 ``reasoning`` 分支。
    """
    provider = _responses_provider(extra_body={"reasoning": {"effort": "low"}})

    with pytest.raises(ValueError, match="Responses reasoning 与模型 options.extra_body 重复"):
        provider._build_responses_request(prompt="hi", reasoning={"effort": "high"})


class _BadOpenAIClose:
    def close(self):
        raise RuntimeError("close 失败")


@pytest.mark.parametrize("method_name", ["close", "aclose"])
def test_openai_compatible_close_reports_first_error(method_name):
    """``close``／``aclose`` 首个关闭异常必须向上报告，不能静默吞掉。"""
    provider = _openai_compatible_provider()
    provider._client = _BadOpenAIClose() if method_name == "close" else None
    provider._aclient = _BadOpenAIClose() if method_name == "aclose" else None

    with pytest.raises(RuntimeError, match="close 失败"):
        result = getattr(provider, method_name)()
        if method_name == "aclose":
            asyncio.run(result)


# ── VolcengineProvider Ark SDK 缺失时的报错形态 ──
#
# ``_load_ark_clients`` 在导入失败时抛带指引的 ``RuntimeError``。这条判据在整套
# 测试里从未执行（行覆盖 + 变异扫描双证；把 ``raise`` 换成 ``pass`` 后全量 3698
# 条仍全绿）——SDK 在开发机与 CI 上都已安装，这条分支只在缺依赖的部署里走到。
# 删除后行为确实改变（实测）：``Ark`` 未绑定，收场从
# ``RuntimeError: 未安装 volcengine-python-sdk[ark]...`` 变成裸
# ``UnboundLocalError: cannot access local variable 'Ark'``——调用方按
# ``RuntimeError`` 分类，接不住。
class _ArkSdkImportBlocker:
    """拦截 ``volcenginesdkarkruntime`` 顶层包的导入。"""

    _TARGET = "volcenginesdkarkruntime"

    def find_spec(self, name, path=None, target=None):
        if name == self._TARGET:
            raise ImportError(f"No module named {name!r}")


def _with_blocked_ark_sdk_import(call):
    # 同 ``_with_blocked_classification_import``：先逐出缓存再装拦截器，否则
    # ``from volcenginesdkarkruntime import ...`` 直接命中 ``sys.modules``、
    # 拦截器形同虚设。顶层包与全部子模块都要逐出——``from X import Y`` 命中
    # 缓存时走的是 ``X`` 本身。
    target = _ArkSdkImportBlocker._TARGET
    cached = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == target or name.startswith(target + ".")
    }
    blocker = _ArkSdkImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        return call()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(cached)


def test_volcengine_ark_sdk_missing_reports_readably():
    """缺 Ark SDK 时必须报带安装指引的 ``RuntimeError``，不是裸 ``UnboundLocalError``。"""
    import src.providers.volcengine as volcengine_module

    with pytest.raises(RuntimeError, match=r"未安装 volcengine-python-sdk\[ark\]"):
        _with_blocked_ark_sdk_import(volcengine_module._load_ark_clients)


def _ark_provider_for_logging_tests():
    """构造一个最小可用的 Ark provider（不发真实请求）。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao"
    provider._options = {}
    provider._protocol = "chat_completions"
    return provider


@pytest.mark.parametrize(
    "method_name, is_async",
    [
        pytest.param("invoke", False, id="invoke"),
        pytest.param("ainvoke", True, id="ainvoke"),
        pytest.param("embed_documents", False, id="embed_documents"),
        pytest.param("aembed_documents", True, id="aembed_documents"),
    ],
)
def test_ark_call_paths_log_failures_before_reraising(caplog, method_name, is_async):
    """Ark 的四条调用路径失败时必须留下日志。

    另外三家（anthropic/google/openai_compatible）的 ``invoke``/``ainvoke``／
    ``embed_documents``/``aembed_documents`` 都有同一层包装：
    ``except Exception: logger.exception(...); raise``——记日志后**重新抛出**，
    不吞掉异常。Ark 是同一批适配器里的第四个渠道，四处全部缺失：失败时零日志，
    用户只看到异常，排查无从下手（``safe_exception_text`` 的脱敏与截断也一并失效）。
    """
    import logging

    provider = _ark_provider_for_logging_tests()

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("Ark 调用失败")

    provider._get_client = lambda: _Boom()
    provider._get_aclient = lambda: _Boom()

    with (
        caplog.at_level(logging.ERROR, logger="src.providers.volcengine"),
        pytest.raises(RuntimeError, match="Ark 调用失败"),
    ):
        if is_async:
            if method_name == "ainvoke":
                asyncio.run(_consume_async(provider.ainvoke(prompt="hi")))
            else:
                asyncio.run(provider.aembed_documents(["文本"]))
        else:
            if method_name == "invoke":
                list(provider.invoke(prompt="hi"))
            else:
                provider.embed_documents(["文本"])
    assert caplog.records, f"{method_name} 失败时没有留下任何日志"


async def _consume_async(agen):
    async for _ in agen:
        pass


def test_volcengine_init_partial_credentials_names_both_keys():
    """只配一半 VOLC 凭证时必须点名缺的另一半，而不是泛化消息。

    ``except`` 前的分支区分「完全没有凭证」与「只配了一半」：前者提示
    ``ARK_API_KEY``，后者提示「需要同时配置 VOLC_ACCESS_KEY 和
    VOLC_SECRET_KEY」。删除部分凭证分支后两者共用泛化消息，用户按提示
    再配一个 ARK key 仍然报同样的错。
    """
    settings = SimpleNamespace(
        volc_access_key="ak", volc_secret_key=None, ark_api_key=None, volc_base_url=None
    )

    with pytest.raises(ValueError, match="需要同时配置 VOLC_ACCESS_KEY 和 VOLC_SECRET_KEY"):
        _volcengine_init(settings)


class _ClassificationAttributeErrorBlocker:
    """让 ``classification`` 模块**导入成功但模块体抛 AttributeError**。

    与 ``_ClassificationImportBlocker`` 的区别：那个在 ``find_spec`` 抛
    ``ImportError``，覆盖的是「模块不存在」；这个在 ``exec_module`` 抛
    ``AttributeError``，覆盖的是「模块存在但初始化引用了缺失属性」——
    真实的 SDK 版本错配常是后者。
    """

    _TARGET = "volcenginesdkarkruntime.resources.classification"

    class _Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise AttributeError("模块体引用了缺失属性")

    def find_spec(self, name, path=None, target=None):
        if name == self._TARGET:
            import importlib.util

            return importlib.util.spec_from_loader(name, self._Loader())
        return None


def _with_classification_attribute_error(call):
    import sys

    target = _ClassificationAttributeErrorBlocker._TARGET
    cached = sys.modules.pop(target, None)
    blocker = _ClassificationAttributeErrorBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        return call()
    finally:
        sys.meta_path.remove(blocker)
        if cached is not None:
            sys.modules[target] = cached


def test_ark_classification_attribute_error_reports_readably():
    """模块体抛 ``AttributeError`` 时同样收敛成 ``NotImplementedError``。

    ``except (ImportError, AttributeError)`` 的 ``AttributeError`` 此前无测试
    触发——收窄成 ``except ImportError`` 后，模块体 AttributeError 冒泡，
    用户拿到的是 SDK 内部错误而不是能力缺失提示。
    """
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._classification_resource = None
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="Ark Classification 资源"):
        _with_classification_attribute_error(provider._get_classification_resource)


def test_ark_async_classification_attribute_error_reports_readably():
    """异步分支同判据。"""
    provider = _volcengine_provider()
    provider._protocol = "responses"
    provider._async_classification_resource = None
    provider._get_aclient = lambda: SimpleNamespace()

    async def run():
        return await provider._get_async_classification_resource()

    with pytest.raises(NotImplementedError, match="Ark Async Classification 资源"):
        _with_classification_attribute_error(lambda: asyncio.run(run()))


def test_google_http_options_timeout_rejects_zero_and_accepts_positive():
    """``timeout <= 0`` 拒绝：0 与负数都必须拦下（``<`` 不是 ``<=``）。"""
    from src.providers.google import GoogleProvider

    for bad in (0, -1, 0.0):
        with pytest.raises(ValueError, match="正整数毫秒"):
            GoogleProvider._validate_http_options({"timeout": bad})
    GoogleProvider._validate_http_options({"timeout": 1})


# ── 回归：chat 请求构造必须把 ``require_tool_call_ids=True`` 传给规范化层 ──
#
# openai 1048 与 volcengine 1020 的布尔字面量互换后全量测试仍全绿：两者调用
# ``normalize_messages(..., require_tool_call_ids=True)`` 的**调用点标志**无人
# 看守——既有的 ``test_chat_completions_still_requires_tool_call_id`` 直接调
# ``normalize_messages`` 本身，不经过构造器。反向成 ``False`` 后缺 id 的
# assistant 历史被静默放进请求，网关按自己那一套处理（或直接 400），而 Chat
# Completions 协议要求 assistant 历史的每个 ``tool_call`` 都带 id（构造器上方
# 的注释写明了这一点）。


@pytest.mark.parametrize(
    "provider_cls,protocol",
    [
        pytest.param(OpenAICompatibleProvider, "chat_completions", id="openai"),
        pytest.param(VolcengineProvider, "chat_completions", id="volcengine"),
    ],
)
def test_build_chat_request_rejects_tool_calls_without_id(provider_cls, protocol):
    provider = object.__new__(provider_cls)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = protocol
    provider._options = {}
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
    ]

    with pytest.raises(ValueError, match="缺少 id/call_id"):
        provider._build_chat_request(prompt="hi", messages=messages)


# ── 回归：动态资源 Facade 的 ``_delegate_native`` 必须逐类声明 ──
#
# 四个类的 ``_delegate_native = True`` 互换后全量测试仍全绿（boolc 扫描）：
# 既有用例只通过 ``dir(resources_module)`` 取**第一个** OpenAI 命名的 Facade
# 来验证动态资源代理，其余类从不实例化。反向成 ``False`` 后 ``__getattr__``
# 与 ``beta``／``async_beta`` 直接抛 ``AttributeError``——官方 SDK 在模块发布
# 之后新增的资源（Realtime／Webhooks／Interactions 等）对这几个渠道整棵树消失，
# 而这是 Facade 存在的唯一理由。


@pytest.mark.parametrize(
    "facade_name,client_attr",
    [
        pytest.param("GoogleResources", "sdk_client", id="google"),
        pytest.param("AsyncGoogleResources", "async_sdk_client", id="async-google"),
        pytest.param("AsyncArkResources", "async_sdk_client", id="async-ark"),
    ],
)
def test_facade_delegates_native_resources_for_every_declaring_class(facade_name, client_attr):
    """每个声明 ``_delegate_native = True`` 的 Facade 都要能转发动态资源。"""
    import src.providers.resources as resources_module

    facade_cls = getattr(resources_module, facade_name)
    assert facade_cls._delegate_native is True

    provider = SimpleNamespace(
        _provider="demo",
        sdk_client=SimpleNamespace(brand_new="SYNC", beta="SYNC-BETA"),
        async_sdk_client=SimpleNamespace(brand_new="ASYNC", beta="ASYNC-BETA"),
    )
    facade = facade_cls(provider)

    # ``__getattr__`` 走 ``_delegate_client_name``，而 ``beta``／``async_beta``
    # 两个属性方法固定读同步／异步客户端——三者的客户端来源不同，分开断言。
    expected = "ASYNC" if client_attr == "async_sdk_client" else "SYNC"
    assert facade.brand_new == expected
    assert facade.beta == "SYNC-BETA"
    assert facade.async_beta == "ASYNC-BETA"
