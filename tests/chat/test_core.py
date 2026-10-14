import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from src.chat.core import Chatbot, start_chat_session_async
from src.runtime.contracts import SessionConfig
from src.utils.config import ModelDetail, RetrievalMethod
from src.utils.log_manager import RedactingFormatter
from src.utils.security import safe_exception_text

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_start_chat_session_async_survives_a_rejected_config_update(monkeypatch):
    """``/config`` 应用失败时必须留在会话里，而不是终止整个聊天。

    内层 ``try`` 只接 ``(KeyboardInterrupt, EOFError)``，而 ``SessionConfig`` 的
    校验器会拒绝非法取值（``ValueError``）。配置编辑器的动态上界与
    ``SessionConfig`` 的守卫是两套独立实现，任一侧演化就会命中这条路径——实测
    编辑器返回「非法字段 + 非法值」时异常穿透到 ``finally``，用户被丢回主菜单
    并丢失会话。同文件的 ``chat_async`` 对异常是恢复的（打印后继续下一轮），
    这里与之对齐。
    """
    prompts = []

    class RejectingChatbot(FakeChatbot):
        def apply_config_update(self, updated_config, llm_needs_reload):
            raise ValueError("retrieval_method 必须是 向量检索, 全文检索, 混合检索 之一。")

    class RecordingPromptSession(FakePromptSession):
        async def prompt_async(self, *args, **kwargs):
            value = await super().prompt_async(*args, **kwargs)
            prompts.append(value)
            return value

    monkeypatch.setattr("src.chat.core.Chatbot", RejectingChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: RecordingPromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.chat.core.launch_config_editor",
        lambda chat_config: (False, {**chat_config, "retrieval_method": "坏模式"}),
    )

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("src.chat.core.asyncio.to_thread", fake_to_thread)

    # 会话必须走完（回到提示符并正常退出），而不是抛异常终止。
    asyncio.run(start_chat_session_async())

    # 两个输入都被读到，说明被拒之后确实回到了提示符。
    assert prompts == ["/config", "/quit"]


def test_start_chat_session_async_reports_rejected_config_without_credentials(monkeypatch):
    """被拒的配置错误也要走脱敏出口。"""

    class RejectingChatbot(FakeChatbot):
        def apply_config_update(self, updated_config, llm_needs_reload):
            raise ValueError("非法取值: sk-live-abcdef123456")

    monkeypatch.setattr("src.chat.core.Chatbot", RejectingChatbot)
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
        return func(*args, **kwargs)

    monkeypatch.setattr("src.chat.core.asyncio.to_thread", fake_to_thread)

    console = Console(record=True, width=120, force_terminal=False)
    monkeypatch.setattr("src.chat.core.Console", lambda: console)

    asyncio.run(start_chat_session_async())

    rendered = console.export_text()
    assert "sk-live-abcdef123456" not in rendered


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
    # 走 LLM 异步生成分支时必须**流式**请求：stream 标志是这一分支的契约
    # （与 ChatService.generate_reply 同判据），反向成 False 后整段回复在
    # provider 内部一次性缓冲再返回，逐块 yield 名存实亡。
    assert model_calls[0]["stream"] is True
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
                llm_configurations={
                    "old-model": ModelDetail(provider="openai", model_name="gpt-4o")
                },
                rerank_configurations={
                    "siliconflow": ModelDetail(provider="siliconflow", model_name="rerank")
                },
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


# ── 回归：chat_config setter 收 dict 时必须是原子的 ──


def test_chat_config_setter_is_atomic_on_invalid_field():
    """批量更新里任一字段非法时，整批都不应生效。

    直接循环写 ``self.session_config`` 不是原子的：``{"top_k": 42,
    "retrieval_method": "不存在"}`` 会留下「top_k 已改成 42、检索模式还是
    旧的」这种改了一半的状态，而调用方只看到一个异常，以为整批都没生效。
    """
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    bot = object.__new__(Chatbot)
    bot.session_config = build_session_config(Settings(_env_file=None))
    before = bot.session_config.to_dict()

    with pytest.raises(ValueError, match="retrieval_method"):
        bot.chat_config = {"top_k": 42, "retrieval_method": "不存在的模式"}

    assert bot.session_config.to_dict() == before, "配置处于改了一半的状态"


def test_chat_config_setter_applies_a_valid_batch():
    """收紧不能误伤：合法批量更新必须整体生效。"""
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    bot = object.__new__(Chatbot)
    bot.session_config = build_session_config(Settings(_env_file=None))

    bot.chat_config = {"top_k": 42, "rerank_enabled": True}

    assert bot.session_config["top_k"] == 42
    assert bot.session_config["rerank_enabled"] is True


# ── 回归：聊天路径的终端输出不得被文档内容里的标记形态打断 ──
#
# ``console.print(str)``／``Table.add_row(str)`` 的字符串会被 rich 按标记解析。
# 检索到的文档正文、来源路径、意图文本都可能含字面量 ``[/]``（知识库文档里
# 实测存在「匹配斜杠用 [/] 表示」这类写法），修复前会让聊天会话以
# ``MarkupError`` 崩溃——而该异常在 ``chat_async`` 的 except 里再次走同一个
# 渲染通道，于是「报告错误」本身也抛，最终以裸 traceback 终止会话。


class _SilentLogger:
    """吞掉所有日志调用，只让测试关心终端输出。"""

    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def exception(self, *_args, **_kwargs):
        return None


def _silent_bot(buffer):
    bot = object.__new__(Chatbot)
    bot.console = Console(file=buffer, width=200, force_terminal=False, no_color=True)
    bot.logger = _SilentLogger()
    bot.session_config = {"chat_temperature": 0.7}
    return bot


def test_display_retrieved_docs_survives_markup_like_content(monkeypatch):
    """预览与来源含标记形态时必须照常渲染，且内容原样显示。"""
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    bot._display_retrieved_docs(
        [
            {
                "page_content": "配置说明：匹配斜杠用 [/] 表示。",
                "metadata": {"source": "/kb/[/]说明.md"},
                "score": 0.9,
            }
        ]
    )

    assert "[/]" in buffer.getvalue()


def test_chat_async_survives_markup_like_intent(monkeypatch):
    """意图文本含标记形态时，输出与错误处理都不得抛错。"""
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    async def fake_identify(_query):
        return "斜杠 [/] 的用法"

    async def fake_retrieve(_query):
        return []

    class Model:
        async def ainvoke(self, *_args, **_kwargs):
            yield "回答"

    bot.llm_model = Model()
    bot.chat_service = None
    # 直接赋普通函数：实例属性不做绑定，调用时只收到查询本身。
    bot._identify_intent_async = fake_identify
    bot._retrieve_knowledge_async = fake_retrieve

    async def consume():
        return [chunk async for chunk in bot.chat_async("斜杠怎么用")]

    chunks = asyncio.run(consume())

    assert chunks == ["回答"]
    assert "斜杠 [/] 的用法" in buffer.getvalue()


def _inline_branch_bot(documents, captured):
    """构造走 ``chat_async`` 内联分支（``chat_service=None``）的 bot。

    该分支只在 ``chat_service`` 缺失而 ``llm_model`` 仍在时进入——例如
    ``reload_llm`` 之外的库调用方手工装配，或依赖被替换后的降级路径。它不是
    死代码：``test_chat_async_survives_markup_like_intent`` 同样走这一支。
    """
    from io import StringIO

    bot = _silent_bot(StringIO())

    class Model:
        async def ainvoke(self, **kwargs):
            captured["prompt"] = kwargs.get("prompt")
            yield "回答"

    async def fake_identify(query):
        return query

    async def fake_retrieve(_query):
        return documents

    bot.llm_model = Model()
    bot.chat_service = None
    bot._identify_intent_async = fake_identify
    bot._retrieve_knowledge_async = fake_retrieve
    return bot


def test_chat_async_inline_branch_does_not_feed_malformed_content_to_the_model():
    """内联分支必须与 ``ChatService._build_prompt`` 收敛同一条判据。

    ``_build_prompt`` 对检索结果里的文档字段已经收敛（``page_content=None``
    时整条跳过，绝不把字面量 ``内容: None`` 喂给模型），而 ``chat_async`` 的
    内联分支把整份 ``retrieved_docs`` 直接 f-string 进 prompt——实测
    ``page_content=None`` 生成
    ``知识: [{'page_content': None, ...}]``，模型会把字面量 ``None`` 当成
    一条真实知识。检索结果可以来自任何 ``VectorStoreBase`` 实现，第三方 store
    不会做载入期归一化。
    """
    captured = {}
    bot = _inline_branch_bot([{"page_content": None, "metadata": {"source": "a.md"}}], captured)

    async def consume():
        return [chunk async for chunk in bot.chat_async("问题")]

    chunks = asyncio.run(consume())

    assert chunks == ["回答"]
    assert "None" not in captured["prompt"]


def test_chat_async_inline_branch_survives_oversized_score():
    """位数超限的分数字段不得让整次问答失败。

    内联分支 f-string 整个 ``retrieved_docs``；文档里的 ``score=10**4301``
    会抛裸 ``ValueError: Exceeds the limit``，``chat_async`` 捕获后只回一句
    「抱歉，处理请求时遇到错误」——用户看到的是莫名其妙的失败，而不是那条
    知识被正常跳过。
    """
    captured = {}
    bot = _inline_branch_bot(
        [{"page_content": "正文", "metadata": {"source": "a.md"}, "score": 10**4301}],
        captured,
    )

    async def consume():
        return [chunk async for chunk in bot.chat_async("问题")]

    chunks = asyncio.run(consume())

    assert chunks == ["回答"]
    assert "正文" in captured["prompt"]


def test_chat_async_error_path_survives_markup_like_exception(monkeypatch):
    """异常消息含标记形态时，错误提示必须能打出来（级联失效的回归）。"""
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    async def failing_identify(_query):
        raise RuntimeError("bad tag [/] in message")

    bot._identify_intent_async = failing_identify

    async def consume():
        return [chunk async for chunk in bot.chat_async("问题")]

    chunks = asyncio.run(consume())

    assert chunks == ["抱歉，处理请求时遇到错误。"]
    rendered = buffer.getvalue()
    # 断言的是**原异常**而不是任意异常：签名写错会抛 TypeError，那条路径
    # 同样会打印「LLM 异步生成出错」，测试会因为错误的原因变绿。
    assert "bad tag [/] in message" in rendered


@pytest.mark.parametrize(
    "document",
    [
        {"page_content": None, "metadata": {"source": "a.md"}, "score": 0.5},
        {"page_content": ["x"], "metadata": {"source": "a.md"}, "score": 0.5},
        {"page_content": "x", "metadata": None, "score": 0.5},
        {"page_content": "x", "metadata": ["a"], "score": 0.5},
        {"page_content": "x", "metadata": {"source": ["a"]}, "score": 0.5},
        {"page_content": "x", "metadata": {"source": "a.md"}, "score": None},
        {"page_content": "x", "metadata": {"source": "a.md"}, "score": "0.5"},
        "不是映射",
        None,
    ],
)
def test_display_retrieved_docs_survives_malformed_documents(document):
    """字段形态异常时聊天侧的检索详情也必须照常渲染。

    检索结果可以来自任何 ``VectorStoreBase`` 实现——``FaissStore`` 会在载入期
    校验并归一化，第三方 store 不会，而抽象基类正是为替换实现而存在的。
    """
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    bot._display_retrieved_docs([document])

    assert buffer.getvalue()


# ── 回归：``_display_retrieved_docs`` 的顶层容器形态 ──
#
# 与召回测试侧的 ``display_results`` 同判据——两者消费的是检索管线输出的同一种
# 值，而那一侧已有这条守卫，这一侧此前漏了。实测非列表取值两种收场都不在契约内：
#   * ``"abc"`` 是合法可迭代对象，逐**字符**迭代后每个字符经 ``document_*`` 助手
#     收敛成占位符，渲染出一张 3 行来源与预览都是空串的伪结果表格（用户看到
#     「检索详情」里有 3 条命中，而实际返回的是字符串）；
#   * ``5``／``True`` 抛裸 ``TypeError: 'int' object is not iterable``，而本方法在
#     ``chat_async`` 的 ``try`` 内，异常会被兜成「LLM 异步生成出错」——一次成功
#     检索的展示失败被显示成模型故障。


@pytest.mark.parametrize(
    "docs",
    [
        pytest.param("abc", id="str"),
        pytest.param(5, id="int"),
        pytest.param(True, id="bool"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(None, id="none"),
    ],
)
def test_display_retrieved_docs_treats_non_list_as_empty(docs):
    """非列表的 ``docs`` 一律按「无相关文档」收敛，不得渲染伪结果或裸抛。"""
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    bot._display_retrieved_docs(docs)

    output = buffer.getvalue()
    assert "无相关文档" in output
    # ``"abc"`` 修复前会渲染出一张 3 行的伪表格；这里断言表格标题不出现。
    assert "检索详情" not in output


def test_display_retrieved_docs_still_renders_a_normal_list():
    """收紧不能误伤：正常文档列表照常逐条渲染。"""
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    bot._display_retrieved_docs(
        [{"page_content": "正文", "metadata": {"source": "a.md"}, "score": 0.5}]
    )

    output = buffer.getvalue()
    assert "检索详情" in output
    assert "a.md" in output


# ── 回归：批量更新要按**最终值**校验，而不是逐个对旧值校验 ──
#
# ``chat_config`` 的 setter 在副本上逐个写字段，但每个字段的校验读的是**副本当时
# 的兄弟字段值**——即尚未被这批更新覆盖的旧值。于是同一个合法终态能否被接受，
# 取决于 dict 的遍历顺序：
#
#   初始 (top_k=5, mult=3)，目标 (top_k=2000, mult=1)，乘积 2000 完全合法。
#   先写 top_k -> 用旧的 mult=3 校验，乘积 6000 -> 拒绝。
#   先写 mult  -> 用旧的 top_k=5 校验，随后 top_k 用新 mult=1 校验 -> 接受。
#
# 这不是理论问题：配置编辑器的动态上界会**放行**这两个字段（先设 mult=1 时上界
# 是 100，再设 top_k=2000 时上界正好 2000），而 ``SessionConfig.to_dict()`` 的键
# 顺序把 ``retrieval_candidate_multiplier`` 排在 ``top_k`` **之前**——正好是能
# 通过的那个顺序；但嵌入方传入自己的 dict、或将来调整字段顺序，就会变成
# 「UI 放行、应用时报错并终止会话」。


def test_chat_config_setter_validates_against_the_final_state():
    """合法终态必须被接受，且与 dict 的键顺序无关。"""
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    for label, payload in [
        ("top_k 在前", {"top_k": 2000, "retrieval_candidate_multiplier": 1}),
        ("mult 在前", {"retrieval_candidate_multiplier": 1, "top_k": 2000}),
    ]:
        bot = object.__new__(Chatbot)
        bot.session_config = build_session_config(Settings(_env_file=None))
        bot.session_config["top_k"] = 5
        bot.session_config["retrieval_candidate_multiplier"] = 3

        bot.chat_config = payload

        assert bot.session_config["top_k"] == 2000, label
        assert bot.session_config["retrieval_candidate_multiplier"] == 1, label


def test_chat_config_setter_still_rejects_an_illegal_final_state():
    """收紧不能误伤：终态本身越界时，任何键顺序都要拒绝，且不留痕。"""
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    for label, payload in [
        ("top_k 在前", {"top_k": 2000, "retrieval_candidate_multiplier": 100}),
        ("mult 在前", {"retrieval_candidate_multiplier": 100, "top_k": 2000}),
    ]:
        bot = object.__new__(Chatbot)
        bot.session_config = build_session_config(Settings(_env_file=None))
        bot.session_config["top_k"] = 5
        bot.session_config["retrieval_candidate_multiplier"] = 3
        before = bot.session_config.to_dict()

        with pytest.raises(ValueError, match="不能超过"):
            bot.chat_config = payload

        assert bot.session_config.to_dict() == before, f"{label}: 配置处于改了一半的状态"


# ── 「服务尚未初始化」三条判据的覆盖缺口 ──
#
# 三条判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3725 条仍全绿）。删除后行为确实改变（实测，经真实入口）：
# ``reload_llm``／``apply_config_update_async``（260）在 ``retrieval_service is
# None`` 时不再报错，而是拿 ``None`` 去构造 ``ChatService``——崩在构造器内部，
# 消息与真实原因（服务没装配好）无关；``chat_async``（332）在 ``chat_service``
# 为 ``None`` 时静默走完整个生成器，用户拿到空回答而不是「聊天服务尚未初始化」。
def test_reload_llm_reports_missing_retrieval_service(monkeypatch):
    """``retrieval_service`` 未初始化时必须报错，不能拿 None 去构造 ChatService。"""
    bot = object.__new__(Chatbot)
    bot.console = Console()
    bot.session_config = SimpleNamespace(active_llm_configuration="new-model")
    bot.retrieval_service = None
    bot.llm_model = None
    bot.chat_service = None

    class _Model:
        pass

    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider",
        lambda _llm_key, _configurations=None: _Model(),
    )

    assert bot.reload_llm() is False


def test_chat_async_reports_missing_chat_service(monkeypatch):
    """``chat_service`` 与 ``llm_model`` 都未初始化时用户必须收到显式错误。

    删除守卫后生成器**什么都不产出**就结束——用户拿到空回答，与「模型真的
    回答为空」无法区分；守卫在场时异常被外层 ``except`` 收敛成致歉文本。
    """
    bot = object.__new__(Chatbot)
    bot.console = Console()
    bot.session_config = {
        "chat_temperature": 0.7,
        "active_llm_configuration": "openai",
        "llm_configurations": {},
    }
    bot.chat_service = None
    bot.llm_model = None
    bot.retrieval_service = None
    bot.logger = MagicMock()

    async def _drain():
        return [chunk async for chunk in bot.chat_async("你好")]

    chunks = asyncio.run(_drain())

    assert chunks, "两个服务都缺失时必须产出显式错误文本，不能是空回答"


def test_apply_config_update_async_reports_missing_retrieval_service(monkeypatch):
    """异步配置更新路径上 ``retrieval_service`` 未初始化也必须报错。

    与 ``test_reload_llm_reports_missing_retrieval_service`` 是**两条独立路径**
    （同步 ``reload_llm`` 与 ``apply_config_update_async`` 各有自己的判据），
    既有用例只覆盖了后者且 ``retrieval_service`` 非 ``None``。
    """

    class _Model:
        pass

    class DummySessionConfig:
        active_llm_configuration = "new-model"

        def __setitem__(self, key, value):
            setattr(self, key, value)

    messages = []
    bot = object.__new__(Chatbot)
    bot.console = SimpleNamespace(print=lambda message: messages.append(str(message)))
    bot.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    bot.session_config = DummySessionConfig()
    bot.retrieval_service = None
    bot.llm_model = object()
    bot.chat_service = object()

    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider",
        lambda *_args, **_kwargs: _Model(),
    )

    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))

    assert any("LLM 切换失败" in message for message in messages)


# --- 否定变异（not / not in / is not）扫描的覆盖缺口 -------------------------
#
# 把守卫里的 ``is not`` 换成 ``is``、``not`` 去掉后全量仍绿的存活点。判据是
# 「只换掉其中一个 ``is not``」——同步分支大多有测试，异步兄弟方法没有。


class _ClosableModel:
    def __init__(self, name="m"):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def _disposal_bot(model):
    """构造一个只够跑 ``reload_llm``/``apply_config_update_async`` 的 Chatbot。"""
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    bot = object.__new__(Chatbot)
    bot.console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
    bot.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    bot.session_config = build_session_config(Settings(_env_file=None))
    bot.session_config.active_llm_configuration = "new-model"
    bot.retrieval_service = object()
    bot.llm_model = model
    bot.chat_service = object()
    return bot


def test_chatbot_aclose_disposes_llm_model():
    """``aclose()`` 释放 LLM 模型（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 ``_adispose_model`` 永不执行，异步会话退出时模型
    不被关闭——连接池泄漏。同步 ``close()`` 有测试，异步兄弟方法此前没有。
    """
    llm = _ClosableModel()
    retrieval = SimpleNamespace(aclose=lambda: None)
    bot = object.__new__(Chatbot)
    bot.llm_model = llm
    bot.chat_service = object()
    bot.retrieval_service = retrieval
    bot.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)

    asyncio.run(bot.aclose())

    assert llm.closed is True
    assert bot.llm_model is None


def test_reload_llm_disposes_previous_model_only_when_replaced(monkeypatch):
    """``reload_llm`` 只在**换成了不同对象**时释放旧模型。

    137 行有两个 ``is not``：换第一个（``is None``）后不同对象的旧模型不再
    被释放；换第二个（``is``）后工厂返回同一对象时会把刚装上的模型关掉。
    """
    # (a) 工厂返回**同一对象**：不得释放
    same = _ClosableModel()
    bot = _disposal_bot(same)
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider", lambda *_a, **_k: same
    )
    assert bot.reload_llm() is True
    assert same.closed is False

    # (b) 换成**不同对象**：必须释放旧模型
    old, new = _ClosableModel("old"), _ClosableModel("new")
    bot = _disposal_bot(old)
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider", lambda *_a, **_k: new
    )
    assert bot.reload_llm() is True
    assert old.closed is True
    assert new.closed is False
    assert bot.llm_model is new


def test_reload_llm_failure_disposes_only_uninstalled_model(monkeypatch):
    """``reload_llm`` 失败时释放**未装上**的新模型，保留在用的旧模型。

    ``new_model is not self.llm_model`` 换成 ``is`` 后新模型不再被释放——
    初始化失败留下的 Provider 泄漏连接池。
    """
    old, new = _ClosableModel("old"), _ClosableModel("new")
    bot = _disposal_bot(old)
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider", lambda *_a, **_k: new
    )
    monkeypatch.setattr(
        "src.chat.core.ChatService",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("service init failed")),
    )

    assert bot.reload_llm() is False
    assert new.closed is True
    assert old.closed is False
    assert bot.llm_model is old


def test_apply_config_update_async_disposes_previous_model_only_when_replaced(monkeypatch):
    """异步配置更新与同步 ``reload_llm`` 同判据（两个 ``is not`` 都要判）。

    264 行的两个 ``is not`` 此前都无测试：换成 ``is None`` 后旧模型不释放；
    换成 ``is`` 后工厂返回同一对象时把在用的模型关掉。
    """
    # (a) 工厂返回**同一对象**：不得释放
    same = _ClosableModel()
    bot = _disposal_bot(same)
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider", lambda *_a, **_k: same
    )
    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))
    assert same.closed is False

    # (b) 换成**不同对象**：必须释放旧模型
    old, new = _ClosableModel("old"), _ClosableModel("new")
    bot = _disposal_bot(old)
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider", lambda *_a, **_k: new
    )
    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))
    assert old.closed is True
    assert new.closed is False
    assert bot.llm_model is new


def test_apply_config_update_async_rolls_back_session_config_on_failure(monkeypatch):
    """异步重载失败时回滚 ``session_config``（``is not None`` 不是 ``is None``）。

    ``llm_needs_reload`` 为真时 ``previous_config`` 必非 ``None``（假值已在
    前面提前返回），因此 ``is None`` 变异后回滚永不执行——用户看到「已保留
    当前模型配置」，实际留下的是失败的那份配置。
    """
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    bot = _disposal_bot(_ClosableModel("old"))
    bot.session_config = build_session_config(Settings(_env_file=None))
    bot.session_config.active_llm_configuration = "old-model"
    monkeypatch.setattr(
        "src.chat.core.ModelProviderFactory.get_llm_provider",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("reload failed")),
    )

    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))

    assert bot.session_config.active_llm_configuration == "old-model"


def test_apply_config_update_async_passes_llm_configurations_when_present(monkeypatch):
    """``llm_configurations`` 存在时传给工厂（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后走单参分支，用户的模型定义（provider/model_name）被
    整个丢掉。
    """
    from src.runtime.contracts import build_session_config
    from src.utils.config import Settings

    bot = _disposal_bot(_ClosableModel("old"))
    bot.session_config = build_session_config(Settings(_env_file=None))
    bot.session_config.active_llm_configuration = "old-model"
    bot.session_config.llm_configurations = {
        "new-model": ModelDetail(provider="openai", model_name="gpt-4o")
    }
    received = []

    def _factory(*args, **_kwargs):
        received.append(args)
        return _ClosableModel("new")

    monkeypatch.setattr("src.chat.core.ModelProviderFactory.get_llm_provider", _factory)

    asyncio.run(bot.apply_config_update_async({"active_llm_configuration": "new-model"}, True))

    assert received == [
        ("new-model", {"new-model": ModelDetail(provider="openai", model_name="gpt-4o")})
    ]


def test_importing_chat_core_does_not_start_interactive_session():
    """import ``src.chat.core`` 不得触发 ``__main__`` 分支（``==`` 不是 ``!=``）。

    ``if __name__ == "__main__"`` 的 ``==``→``!=`` 变异后守卫常真：任何 import
    都会把模块当作主脚本执行（实测子进程直接进入会话，打出「客服已就绪」与
    「感谢使用，再见！」）——所有 import 该模块的调用方（含本仓库其它测试）
    都会被拖进交互式会话。
    """
    result = subprocess.run(
        [sys.executable, "-c", "import src.chat.core"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "客服已就绪" not in output
    assert "感谢使用，再见！" not in output


def test_display_retrieved_docs_renders_the_column_header_row():
    """检索详情表必须渲染**列头**（``show_header=True``）。

    ``show_header`` 反向成 ``False`` 后 rich 不再输出列头行，表格只剩数据行：
    第一列的数字与第三列的小数失去列名（``来源``／``预览``／``得分``），
    用户看到三列中间夹着省略号却无从判断哪列是分数。
    """
    from io import StringIO

    buffer = StringIO()
    bot = _silent_bot(buffer)

    bot._display_retrieved_docs(
        [{"page_content": "正文", "metadata": {"source": "a.md"}, "score": 0.5}]
    )

    output = buffer.getvalue()
    for column in ("来源", "预览", "得分"):
        assert column in output, f"列头 {column!r} 未渲染：{output!r}"


class _SurvivalBot(FakeChatbot):
    """带真实 ``SessionConfig`` 的替身，rerank 段为空表。"""

    def __init__(self, _console: Console):
        self.llm_model = object()
        self.chat_config = SessionConfig(
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
            rerank_configurations={},
            chat_temperature=0.7,
        )

    def apply_config_update(self, updated_config, _llm_needs_reload):
        for key, item in updated_config.items():
            self.chat_config[key] = item

    def reload_llm(self):
        return False

    async def chat_async(self, _user_input):
        if False:
            yield ""


def test_start_chat_session_async_survives_config_display_failure(monkeypatch):
    """配置已生效后，回显抛 ``ValueError`` 也必须让会话存活。

    实测路径（真实 ``SessionConfig`` + 真实 ``display_chat_config``）：rerank 段
    为空表时，编辑器保存 ``rerank_enabled=True`` 通过 ``set_many`` 校验、apply
    「成功」，随后 ``display_chat_config`` 在 ``chat/core.py`` 的回显调用点抛
    ``ValueError: 在Rerank配置中未找到key: ...``。修复前该调用点在恢复 ``try``
    之外，异常穿透内层只接 ``(KeyboardInterrupt, EOFError)`` 的 ``try``——
    配置已生效、会话却被终止，用户丢失全部对话，直接违反同块开头声明的
    「配置应用失败不能终止整个会话」契约。
    """
    from src.ui.display_utils import display_chat_config as real_display

    monkeypatch.setattr("src.chat.core.Chatbot", _SurvivalBot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    # 启动横幅里的回显不炸（rerank_enabled=False 时不读 rerank 段），
    # 只有 /config 保存后的回显要炸——用计数器区分两次调用。
    display_calls = []

    def exploding_display(_console, chat_config):
        display_calls.append(1)
        if len(display_calls) > 1:
            raise ValueError("在Rerank配置中未找到key: 'siliconflow'")
        real_display(_console, chat_config)

    monkeypatch.setattr("src.chat.core.display_chat_config", exploding_display)
    monkeypatch.setattr(
        "src.chat.core.launch_config_editor",
        lambda chat_config: (False, {"rerank_enabled": True}),
    )

    asyncio.run(start_chat_session_async())

    # 启动回显 + /config 后的回显都被调用过——后者抛了异常而会话仍走到 /quit。
    assert len(display_calls) == 2


def test_start_chat_session_async_survives_config_editor_failure(monkeypatch):
    """编辑器自身抛 ``ValueError``（空候选/配置段非映射）也必须让会话存活。"""
    monkeypatch.setattr("src.chat.core.Chatbot", _SurvivalBot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)

    def exploding_launch(_chat_config):
        raise ValueError("A list of choices needs to be provided.")

    monkeypatch.setattr("src.chat.core.launch_config_editor", exploding_launch)

    asyncio.run(start_chat_session_async())


def test_start_chat_session_async_survives_failing_teardown(monkeypatch):
    """``finally`` 里的 ``aclose`` 抛异常时不得顶替正常收尾。

    替身 ``aclose`` 是受支持的契约（同函数用 ``getattr`` 探测它）；裸 ``await``
    会让替身的错误顶替「再见」输出并向上穿透到菜单兜底（实测复现）。
    释放失败只记录，会话仍正常收尾。
    """

    class ExplodingTeardownBot(FakeChatbot):
        last_instance = None

        def __init__(self, console):
            super().__init__(console)
            ExplodingTeardownBot.last_instance = self
            self.closed = False

        async def aclose(self):
            self.closed = True
            raise RuntimeError("替身的 aclose 失败")

    monkeypatch.setattr("src.chat.core.Chatbot", ExplodingTeardownBot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_args, **_kwargs: None)

    asyncio.run(start_chat_session_async())

    assert ExplodingTeardownBot.last_instance.closed is True


def test_start_chat_session_survives_banner_display_failure(monkeypatch):
    """启动横幅回显失败（rerank 启用而段为空）只报黄字，会话照常启动。

    修复前 ``display_chat_config`` 在任何提示符之前裸抛 ``ValueError``，
    以「程序运行期间发生错误」打回主菜单——可修复的配置问题被顶成程序级
    错误，与 ``/config`` 路径对同一类错误的恢复处理不对称。
    """
    from src.ui.display_utils import display_chat_config as real_display

    def exploding_display(_console, chat_config):
        real_display(_console, chat_config)
        raise ValueError("在Rerank配置中未找到key: 'siliconflow'")

    monkeypatch.setattr("src.chat.core.Chatbot", _SurvivalBot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", exploding_display)

    asyncio.run(start_chat_session_async())


def test_ready_line_survives_none_chat_config(monkeypatch):
    """替身 bot 的 ``chat_config=None`` 在就绪行抛 ``TypeError``——必须在捕获面内。

    实测 ``bot.chat_config["active_llm_configuration"]`` 对 ``None`` 抛
    ``TypeError: 'NoneType' object is not subscriptable``，不在
    ``(ValueError, KeyError)`` 捕获面——注释声称要堵的形态恰好漏接。
    """

    class NoneConfigBot(FakeChatbot):
        def __init__(self, console):
            self.chat_config = None
            self.llm_model = object()

        async def chat_async(self, _user_input):
            if False:
                yield ""

    monkeypatch.setattr("src.chat.core.Chatbot", NoneConfigBot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr(
        "src.chat.core.display_chat_config",
        lambda *_args, **_kwargs: None,
    )

    asyncio.run(start_chat_session_async())
