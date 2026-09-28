import asyncio
from types import SimpleNamespace

import pytest
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
