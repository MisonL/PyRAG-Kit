import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.providers.factory as factory_module
from src.providers.__base__.model_provider import (
    CompletionRequest,
    LargeLanguageModel,
    iterate_async,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.factory import ModelProviderFactory
from src.providers.google import GoogleProvider
from src.providers.grok import GrokProvider
from src.providers.openai import OpenAIProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.volcengine import VolcengineProvider
from src.utils.config import ModelDetail


class FailingAsyncClient:
    class chat:
        class completions:
            @staticmethod
            async def create(*_args, **_kwargs):
                raise RuntimeError("boom-chat")

    class embeddings:
        @staticmethod
        async def create(*_args, **_kwargs):
            raise RuntimeError("boom-embed")


class RequestCaptureModel(LargeLanguageModel):
    def invoke(self, **_kwargs):
        yield "ok"

    async def ainvoke(self, **_kwargs):
        yield "ok"


def test_iterate_async_awaits_async_close_on_sync_iterable():
    class AsyncClosableIterable:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter(["ok"])

        async def close(self):
            self.closed = True

    value = AsyncClosableIterable()

    async def consume() -> list[str]:
        return [item async for item in iterate_async(value)]

    assert asyncio.run(consume()) == ["ok"]
    assert value.closed is True


def test_base_request_and_kwargs_conflict_is_explicit():
    provider = RequestCaptureModel()
    request = CompletionRequest(prompt="from-request")

    with pytest.raises(ValueError, match="不能同时传 request"):
        provider.complete(request, prompt="from-kwargs")
    with pytest.raises(ValueError, match="不能同时传 request"):
        asyncio.run(provider.acomplete(request, prompt="from-kwargs"))
    with pytest.raises(ValueError, match="不能同时传 request"):
        list(provider.stream_events(request, prompt="from-kwargs"))

    async def consume() -> None:
        async for _event in provider.astream_events(request, prompt="from-kwargs"):
            pass

    with pytest.raises(ValueError, match="不能同时传 request"):
        asyncio.run(consume())


@pytest.mark.parametrize(
    ("provider_class", "method_name"),
    [
        (OpenAICompatibleProvider, "complete"),
        (OpenAICompatibleProvider, "stream_events"),
        (AnthropicProvider, "complete"),
        (AnthropicProvider, "stream_events"),
        (GoogleProvider, "complete"),
        (GoogleProvider, "count_tokens"),
        (VolcengineProvider, "complete"),
        (VolcengineProvider, "stream_events"),
    ],
)
def test_provider_request_and_kwargs_conflict_is_explicit(provider_class, method_name):
    provider = object.__new__(provider_class)
    request = CompletionRequest(prompt="from-request")
    method = getattr(provider, method_name)

    with pytest.raises(ValueError, match="不能同时传 request"):
        result = method(request, prompt="from-kwargs")
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
            list(result)


def test_openai_provider_async_failure_raises(monkeypatch):
    fake_settings = SimpleNamespace(openai_api_key="token", openai_api_base="http://example.com")
    monkeypatch.setattr("src.providers.openai.get_settings", lambda: fake_settings)

    provider = OpenAIProvider("demo")
    provider._aclient = FailingAsyncClient()

    async def consume() -> None:
        async for _chunk in provider.ainvoke("hello", stream=False):
            pass

    with pytest.raises(RuntimeError, match="boom-chat"):
        asyncio.run(consume())

    with pytest.raises(RuntimeError, match="boom-embed"):
        asyncio.run(provider.aembed_documents(["doc"]))


def test_openai_compatible_provider_async_failure_raises(monkeypatch):
    fake_settings = SimpleNamespace(
        deepseek_api_key="token", deepseek_base_url="http://example.com"
    )
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)

    provider = OpenAICompatibleProvider("demo", "deepseek")
    provider._aclient = FailingAsyncClient()

    async def consume() -> None:
        async for _chunk in provider.ainvoke("hello", stream=False):
            pass

    with pytest.raises(RuntimeError, match="boom-chat"):
        asyncio.run(consume())

    with pytest.raises(RuntimeError, match="boom-embed"):
        asyncio.run(provider.aembed_documents(["doc"]))


def test_openai_provider_reuses_compatible_adapter(monkeypatch):
    fake_settings = SimpleNamespace(openai_api_key="token", openai_api_base="http://example.com")
    monkeypatch.setattr("src.providers.openai.get_settings", lambda: fake_settings)

    provider = OpenAIProvider("demo")

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._base_url == "http://example.com"


def test_openai_provider_preserves_missing_key_error(monkeypatch):
    monkeypatch.setattr(
        "src.providers.openai.get_settings",
        lambda: SimpleNamespace(openai_api_key=None, openai_api_base="http://example.com"),
    )

    with pytest.raises(ValueError, match="OpenAI配置不完整：缺少 OPENAI_API_KEY"):
        OpenAIProvider("demo")


def test_openai_compatible_provider_normalizes_hyphenated_settings(monkeypatch):
    fake_settings = SimpleNamespace(
        lm_studio_api_key=None,
        lm_studio_base_url="http://localhost:1234/v1",
    )
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)

    provider = OpenAICompatibleProvider("demo", "lm-studio")

    assert provider._api_key == "no-key-required"
    assert provider._base_url == "http://localhost:1234/v1"


def test_openai_compatible_provider_forwards_tools(monkeypatch):
    fake_settings = SimpleNamespace(
        deepseek_api_key="token", deepseek_base_url="http://example.com"
    )
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)
    provider = OpenAICompatibleProvider("demo", "deepseek")

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    provider._client = client
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    assert list(provider.invoke("hello", tools=tools, stream=False)) == ["ok"]

    request = client.chat.completions.create.call_args.kwargs
    assert request["tools"] == tools
    assert request["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]


def test_grok_provider_can_be_loaded_by_factory(monkeypatch):
    fake_settings = SimpleNamespace(
        llm_configurations={"grok-test": ModelDetail(provider="grok", model_name="grok-1")},
        grok_api_key="token",
        grok_base_url="https://api.x.ai/v1",
    )
    monkeypatch.setattr(factory_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)

    provider = factory_module.ModelProviderFactory.get_llm_provider("grok-test")

    assert provider.__class__.__name__ == GrokProvider.__name__
    assert provider.__class__.__module__ == "src.providers.grok"


def test_factory_caches_provider_class(monkeypatch):
    factory = factory_module.ModelProviderFactory
    factory._get_provider_class.cache_clear()
    monkeypatch.setattr(
        factory,
        "_provider_map",
        {"cached": {"module": "fake.provider", "class": "CachedProvider"}},
    )
    module = SimpleNamespace(CachedProvider=object)
    imported_modules = []

    def import_module(module_name):
        imported_modules.append(module_name)
        return module

    monkeypatch.setattr(factory_module.importlib, "import_module", import_module)

    first = factory._get_provider_class("cached")
    second = factory._get_provider_class("cached")

    assert first is second
    assert imported_modules == ["fake.provider"]


@pytest.mark.parametrize(
    ("provider_name", "module_name", "class_name"),
    [
        (provider_name, provider_info["module"], provider_info["class"])
        for provider_name, provider_info in ModelProviderFactory._provider_map.items()
    ],
)
def test_provider_map_modules_are_importable(provider_name, module_name, class_name):
    module = importlib.import_module(module_name)
    provider_class = getattr(module, class_name)

    assert provider_class is not None, provider_name


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ({"error": {"message": "bad key sk-FAKE0000badkey1234567890"}}, "Responses API 返回错误"),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "bad key sk-FAKE0000badkey1234567890"},
            },
            "响应状态为 incomplete",
        ),
    ],
)
def test_responses_error_path_redacts_server_message(response, match):
    """Responses 错误出口与 Chat Completions 出口必须同样脱敏。

    服务端错误消息常回显请求头或 URL；异常文本会进日志与终端，是本项目
    凭证最容易泄漏的出口。此前 Responses 分支直接拼接原始消息。
    """
    with pytest.raises(RuntimeError, match=match) as excinfo:
        OpenAICompatibleProvider._raise_for_response_error(response)

    assert "sk-FAKE0000badkey1234567890" not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_responses_error_path_reports_status_without_message(status):
    """失败/取消状态不含服务端消息，仍须显式报错而非静默返回。"""
    with pytest.raises(RuntimeError, match=f"响应状态为 {status}"):
        OpenAICompatibleProvider._raise_for_response_error({"status": status})


@pytest.mark.parametrize("status", ["completed", "in_progress", None])
def test_responses_error_path_passes_through_when_not_terminal_failure(status):
    """非失败状态不是错误，不得抛异常。"""
    assert OpenAICompatibleProvider._raise_for_response_error({"status": status}) is None
