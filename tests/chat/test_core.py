import asyncio
from types import SimpleNamespace

from rich.console import Console

from src.chat.core import Chatbot, start_chat_session_async
from src.runtime.contracts import SessionConfig
from src.utils.config import ModelDetail, RetrievalMethod
from src.utils.log_manager import RedactingFormatter
from src.utils.security import safe_exception_text


def test_safe_exception_text_redacts_credentials():
    text = safe_exception_text(RuntimeError("Authorization: Bearer sk-secret-token"))
    assert "sk-secret-token" not in text
    assert "REDACTED" in text


def test_safe_exception_text_redacts_json_credentials():
    text = safe_exception_text(RuntimeError('{"api_key": "secret-value"}'))
    assert "secret-value" not in text
    assert "REDACTED" in text


def test_log_formatter_redacts_credentials_from_traceback():
    import logging
    import sys

    try:
        raise RuntimeError("Authorization: Bearer sk-secret-token")
    except RuntimeError:
        record = logging.LogRecord(
            "chat_logger",
            logging.ERROR,
            __file__,
            1,
            "request failed",
            (),
            sys.exc_info(),
        )

    rendered = RedactingFormatter("%(message)s").format(record)
    assert "sk-secret-token" not in rendered
    assert "[REDACTED]" in rendered


def test_chatbot_close_disposes_llm_and_retrieval_service():
    llm = SimpleNamespace(close=lambda: setattr(llm, "closed", True), closed=False)
    retrieval = SimpleNamespace(close=lambda: setattr(retrieval, "closed", True), closed=False)
    bot = object.__new__(Chatbot)
    bot.llm_model = llm
    bot.chat_service = object()
    bot.retrieval_service = retrieval

    bot.close()

    assert llm.closed is True
    assert retrieval.closed is True
    assert bot.llm_model is None
    assert bot.retrieval_service is None


class FakePromptSession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def prompt_async(self, *_args, **_kwargs):
        return next(self._responses)


class FakeChatbot:
    def __init__(self, _console: Console):
        self.llm_model = object()
        self.chat_config = {
            "active_llm_configuration": "openai",
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
                "openai": SimpleNamespace(provider="openai", model_name="gpt-4o"),
            },
            "rerank_configurations": {
                "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank"),
            },
            "chat_temperature": 0.7,
        }

    def reload_llm(self):
        return True

    def apply_config_update(self, updated_config, _llm_needs_reload):
        self.chat_config = updated_config

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


def test_start_chat_session_async_skips_empty_input(monkeypatch):
    chat_calls = []

    class EmptySafeChatbot(FakeChatbot):
        async def chat_async(self, user_input):
            chat_calls.append(user_input)
            if False:
                yield ""

    monkeypatch.setattr("src.chat.core.Chatbot", EmptySafeChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["", "   ", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)

    asyncio.run(start_chat_session_async())

    assert chat_calls == []


def test_chat_async_retrieves_with_intent():
    retrieval_queries = []
    model_calls = []

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

        def error(self, *_args, **_kwargs):
            return None

    class DummyLLM:
        async def ainvoke(self, *_args, **kwargs):
            model_calls.append(kwargs)
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
    assert "temperature" not in model_calls[0]


def test_reload_llm_keeps_existing_model_on_failure(monkeypatch):
    class OldModel:
        async def ainvoke(self, *_args, **_kwargs):
            if False:
                yield ""

    bot = object.__new__(Chatbot)
    bot.console = Console()
    bot.session_config = SimpleNamespace(active_llm_configuration="new-model")
    bot.retrieval_service = object()
    bot.llm_model = old_model = OldModel()
    bot.chat_service = old_chat_service = object()

    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider",
        lambda _llm_key, _configurations=None: (_ for _ in ()).throw(RuntimeError("reload failed")),
    )

    assert bot.reload_llm() is False
    assert bot.llm_model is old_model
    assert bot.chat_service is old_chat_service


def test_apply_config_update_async_keeps_runtime_on_chat_service_failure(monkeypatch):
    class ClosableModel:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    old_model = ClosableModel()
    new_model = ClosableModel()
    old_chat_service = object()
    messages = []

    class DummySessionConfig:
        active_llm_configuration = "new-model"

        def __setitem__(self, key, value):
            setattr(self, key, value)

    bot = object.__new__(Chatbot)
    bot.console = SimpleNamespace(print=lambda message: messages.append(str(message)))
    bot.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    bot.session_config = DummySessionConfig()
    bot.retrieval_service = object()
    bot.llm_model = old_model
    bot.chat_service = old_chat_service

    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider",
        lambda *_args, **_kwargs: new_model,
    )
    monkeypatch.setattr(
        "src.chat.core.ChatService",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("service init failed")),
    )

    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))

    assert bot.llm_model is old_model
    assert bot.chat_service is old_chat_service
    assert new_model.closed is True
    assert any("LLM 切换失败" in message for message in messages)


def test_apply_config_update_reverts_llm_key_when_reload_fails(monkeypatch):
    class DummySessionConfig:
        def __init__(self, active_llm_configuration: str):
            self.active_llm_configuration = active_llm_configuration

        def __setitem__(self, key, value):
            setattr(self, key, value)

    messages = []
    bot = object.__new__(Chatbot)
    bot.console = SimpleNamespace(print=lambda message: messages.append(str(message)))
    bot.session_config = DummySessionConfig(active_llm_configuration="old-model")
    bot.llm_model = object()
    bot.chat_service = object()
    monkeypatch.setattr(bot, "reload_llm", lambda: False)

    bot.apply_config_update({"active_llm_configuration": "new-model"}, llm_needs_reload=True)

    assert bot.session_config.active_llm_configuration == "old-model"
    assert any("LLM 切换失败" in message for message in messages)


def test_apply_config_update_restores_all_llm_configurations_on_reload_failure(monkeypatch):
    old_detail = ModelDetail(provider="openai", model_name="old-model")
    new_detail = ModelDetail(provider="qwen", model_name="new-model")
    bot = object.__new__(Chatbot)
    bot.console = SimpleNamespace(print=lambda _message: None)
    bot.session_config = SessionConfig(
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        vector_weight=0.3,
        keyword_weight=0.7,
        hybrid_fusion_strategy="rrf",
        retrieval_candidate_multiplier=3,
        rerank_enabled=False,
        top_k=5,
        score_threshold=0.4,
        active_llm_configuration="old-model",
        active_rerank_configuration="siliconflow",
        llm_configurations={"old-model": old_detail},
        rerank_configurations={},
        chat_temperature=0.7,
    )
    bot.llm_model = object()
    bot.chat_service = object()
    monkeypatch.setattr(bot, "reload_llm", lambda: False)

    bot.apply_config_update(
        {
            "active_llm_configuration": "new-model",
            "llm_configurations": {"new-model": new_detail},
        },
        llm_needs_reload=True,
    )

    assert bot.session_config.active_llm_configuration == "old-model"
    assert bot.session_config.llm_configurations == {"old-model": old_detail}


def test_start_chat_session_async_reverts_llm_after_editor_mutation(monkeypatch):
    class RealishChatbot(FakeChatbot):
        last_instance = None

        def __init__(self, _console: Console):
            RealishChatbot.last_instance = self
            self.console = _console
            self.llm_model = object()
            self.session_config = SessionConfig(
                retrieval_method=RetrievalMethod.HYBRID_SEARCH,
                vector_weight=0.3,
                keyword_weight=0.7,
                hybrid_fusion_strategy="rrf",
                retrieval_candidate_multiplier=3,
                rerank_enabled=False,
                top_k=5,
                score_threshold=0.4,
                active_llm_configuration="old-model",
                active_rerank_configuration="siliconflow",
                llm_configurations={"old-model": ModelDetail(provider="openai", model_name="gpt-4o")},
                rerank_configurations={"siliconflow": ModelDetail(provider="siliconflow", model_name="rerank")},
                chat_temperature=0.7,
            )
            self.chat_service = object()

        @property
        def chat_config(self):
            return self.session_config

        @chat_config.setter
        def chat_config(self, value):
            if isinstance(value, SessionConfig):
                self.session_config = value
                return
            for key, item in value.items():
                self.session_config[key] = item

        def reload_llm(self):
            return False

        def apply_config_update(self, updated_config, llm_needs_reload):
            return Chatbot.apply_config_update(self, updated_config, llm_needs_reload)

    monkeypatch.setattr("src.chat.core.Chatbot", RealishChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)

    edited_configs = []

    def fake_launch(chat_config):
        edited_configs.append(chat_config)
        chat_config["active_llm_configuration"] = "new-model"
        return True, chat_config

    monkeypatch.setattr("src.chat.core.launch_config_editor", fake_launch)

    asyncio.run(start_chat_session_async())

    assert edited_configs[0]["active_llm_configuration"] == "new-model"
    assert RealishChatbot.last_instance.session_config.active_llm_configuration == "old-model"


def test_redacting_formatter_scrubs_cached_traceback_for_later_handlers():
    """脱敏 handler 之后的普通 handler 不能拿到原始 traceback。

    ``logging.Formatter`` 把格式化后的 traceback 缓存在 ``record.exc_text``，
    后续 handler 会复用该缓存。脱敏 formatter 必须在写回缓存前完成脱敏，否则
    宿主挂在同一 logger 上的普通 handler 仍会输出原始凭证。
    """
    import io
    import logging
    import sys

    try:
        raise RuntimeError("Authorization: Bearer sk-FAKE0000TRACEBACK0000FAKE0000")
    except RuntimeError:
        exc_info = sys.exc_info()

    redacting = logging.StreamHandler(io.StringIO())
    redacting.setFormatter(RedactingFormatter("%(message)s"))
    plain = logging.StreamHandler(io.StringIO())
    plain.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("redaction-cache-test")
    logger.propagate = False
    logger.handlers = [redacting, plain]
    logger.setLevel(logging.ERROR)
    logger.error("request failed", exc_info=exc_info)

    assert "sk-FAKE0000TRACEBACK0000FAKE0000" not in redacting.stream.getvalue()
    assert "sk-FAKE0000TRACEBACK0000FAKE0000" not in plain.stream.getvalue()
