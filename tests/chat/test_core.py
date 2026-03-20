import asyncio
from types import SimpleNamespace

from rich.console import Console

from src.chat.core import Chatbot, start_chat_session_async


class FakePromptSession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def prompt_async(self, *_args, **_kwargs):
        return next(self._responses)


class FakeChatbot:
    def __init__(self, _console: Console):
        self.llm_model = object()
        self.chat_config = {
            "active_llm_configuration": "iflow-qwen3-max",
            "active_rerank_configuration": "siliconflow",
            "retrieval_method": SimpleNamespace(value="混合检索"),
            "vector_weight": 0.3,
            "keyword_weight": 0.7,
            "hybrid_fusion_strategy": "rrf",
            "retrieval_candidate_multiplier": 3,
            "rerank_enabled": False,
            "top_k": 5,
            "score_threshold": 0.4,
            "llm_configurations": {
                "iflow-qwen3-max": SimpleNamespace(provider="openai", model_name="qwen3-max"),
            },
            "rerank_configurations": {
                "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank"),
            },
            "chat_temperature": 0.7,
        }

    def reload_llm(self):
        return True

    async def chat_async(self, _user_input):
        if False:
            yield ""


def test_start_chat_session_async_can_exit(monkeypatch):
    monkeypatch.setattr("src.chat.core.Chatbot", FakeChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)

    asyncio.run(start_chat_session_async())


def test_start_chat_session_async_can_open_config(monkeypatch):
    config_calls = []

    monkeypatch.setattr("src.chat.core.Chatbot", FakeChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.chat.core.launch_config_editor",
        lambda chat_config: (False, chat_config),
    )

    async def fake_to_thread(func, *args, **kwargs):
        config_calls.append(func(*args, **kwargs))
        return config_calls[-1]

    monkeypatch.setattr("src.chat.core.asyncio.to_thread", fake_to_thread)

    asyncio.run(start_chat_session_async())

    assert len(config_calls) == 1


def test_chat_async_retrieves_with_intent():
    retrieval_queries = []

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

        def error(self, *_args, **_kwargs):
            return None

    class DummyLLM:
        async def ainvoke(self, *_args, **_kwargs):
            yield "回答"

    async def fake_identify(self, _user_query):
        return "清除浏览器缓存的操作步骤"

    async def fake_retrieve(self, retrieval_query):
        retrieval_queries.append(retrieval_query)
        return []

    bot = object.__new__(Chatbot)
    bot.console = Console()
    bot.vector_store = object()
    bot.llm_model = DummyLLM()
    bot.logger = DummyLogger()
    bot.chat_config = {"chat_temperature": 0.7}
    bot._identify_intent_async = fake_identify.__get__(bot, Chatbot)
    bot._retrieve_knowledge_async = fake_retrieve.__get__(bot, Chatbot)
    bot._display_retrieved_docs = lambda _docs: None

    async def consume():
        chunks = []
        async for chunk in bot.chat_async("如何清除浏览器缓存？只回答关键步骤。"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(consume())

    assert retrieval_queries == ["清除浏览器缓存的操作步骤"]
    assert chunks == ["回答"]
