import asyncio

import pytest

from src.services.chat_service import ChatService


class _FailingModel:
    async def ainvoke(self, **_kwargs):
        raise AttributeError("provider response bug")
        yield ""


class _EmptyModel:
    async def ainvoke(self, **_kwargs):
        if False:
            yield ""


class _ModernAnthropicModel:
    def __init__(self):
        self.calls = []

    def supports_sampling_option(self, name):
        assert name == "temperature"
        return False

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        yield "检索"


class _ResponsesReasoningModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        yield "检索"


class _SamplingModel(_ResponsesReasoningModel):
    def supports_sampling_option(self, name):
        assert name == "temperature"
        return True


def test_identify_intent_propagates_provider_failure():
    service = ChatService(_FailingModel(), retrieval_service=object())

    with pytest.raises(RuntimeError, match="意图识别失败"):
        asyncio.run(service.identify_intent("问题"))


def test_identify_intent_rejects_empty_provider_response():
    service = ChatService(_EmptyModel(), retrieval_service=object())

    with pytest.raises(RuntimeError, match="返回空结果"):
        asyncio.run(service.identify_intent("问题"))


def test_identify_intent_omits_sampling_for_models_that_reject_temperature():
    model = _ModernAnthropicModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert model.calls == [{"prompt": model.calls[0]["prompt"], "stream": False}]


def test_identify_intent_omits_sampling_without_capability_declaration():
    model = _ResponsesReasoningModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert "temperature" not in model.calls[0]


def test_generate_reply_omits_sampling_without_capability_declaration():
    model = _ResponsesReasoningModel()
    service = ChatService(model, retrieval_service=object())

    async def consume():
        return [
            chunk
            async for chunk in service.generate_reply(
                "问题",
                "意图",
                [],
                type("SessionConfig", (), {"chat_temperature": 0.7})(),
            )
        ]

    assert asyncio.run(consume()) == ["检索"]
    assert "temperature" not in model.calls[0]


def test_chat_service_passes_sampling_when_capability_is_declared():
    model = _SamplingModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert model.calls[0]["temperature"] == 0.1

    async def consume():
        return [
            chunk
            async for chunk in service.generate_reply(
                "问题",
                "意图",
                [],
                type("SessionConfig", (), {"chat_temperature": 0.4})(),
            )
        ]

    assert asyncio.run(consume()) == ["检索"]
    assert model.calls[1]["temperature"] == 0.4


def test_build_prompt_handles_documents_without_source_metadata():
    prompt = ChatService._build_prompt(
        "问题",
        "意图",
        [{"page_content": "知识内容"}],
    )

    assert "来源: 未知来源" in prompt
    assert "内容: 知识内容" in prompt
