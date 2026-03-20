# -*- coding: utf-8 -*-
import asyncio
import importlib
from types import SimpleNamespace

import pytest

import src.providers.factory as factory_module
from src.providers.factory import ModelProviderFactory
from src.providers.grok import GrokProvider
from src.providers.openai import OpenAIProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
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
    fake_settings = SimpleNamespace(deepseek_api_key="token", deepseek_base_url="http://example.com")
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
