"""「吞掉型 except」变异扫描的对话循环覆盖缺口。"""

import asyncio

from src.chat.core import start_chat_session_async
from tests.chat.test_core import FakeChatbot


class _PromptSession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def prompt_async(self, *_args, **_kwargs):
        item = next(self._responses)
        if isinstance(item, BaseException):
            raise item
        return item


def test_chat_session_breaks_cleanly_on_interrupt(monkeypatch):
    """流式渲染期间 Ctrl-C 干净退出，不作为异常抛出。

    删除吞掉后 ``KeyboardInterrupt`` 冒泡——退出对话变成一次崩溃
    （``start_chat_session_async`` 的 ``finally`` 仍会关闭，但调用方收到
    KeyboardInterrupt）。
    """
    from src.chat import core as chat_core

    monkeypatch.setattr(chat_core, "Chatbot", FakeChatbot)
    monkeypatch.setattr(
        chat_core,
        "PromptSession",
        lambda: _PromptSession([KeyboardInterrupt(), "/quit"]),
    )
    monkeypatch.setattr(chat_core, "display_chat_config", lambda *_a, **_k: None)
    asyncio.run(start_chat_session_async())


# --- except 元组收窄：第二类型（KeyError / EOFError）必须被触发 -------------


def test_start_chat_session_survives_keyerror_from_config_update(monkeypatch):
    """``apply_config_update`` 抛 ``KeyError`` 时留在会话里，而不是终止。

    内层 ``except (ValueError, KeyError)`` 的 ``KeyError`` 此前无测试触发——
    收窄成 ``except ValueError`` 后，配置更新里的缺键异常穿透到 ``finally``，
    用户被丢出会话。
    """
    import asyncio

    from tests.chat.test_core import FakeChatbot, FakePromptSession

    class KeyErrorChatbot(FakeChatbot):
        def apply_config_update(self, updated_config, llm_needs_reload):
            raise KeyError("active_llm_configuration")

    monkeypatch.setattr("src.chat.core.Chatbot", KeyErrorChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.chat.core.launch_config_editor",
        lambda chat_config: (False, dict(chat_config)),
    )
    # 不抛异常即表示会话存活到 /quit。
    asyncio.run(__import__("src.chat.core", fromlist=["x"]).start_chat_session_async())


def test_start_chat_session_exits_cleanly_on_eof(monkeypatch):
    """``prompt_async`` 抛 ``EOFError`` 时干净退出，而不是冒泡。

    收窄成 ``except KeyboardInterrupt`` 后 ``EOFError`` 穿透——非交互输入
    （管道、CI）下会话以未捕获异常终止。
    """
    import asyncio

    from tests.chat.test_core import FakeChatbot, FakePromptSession

    class EofSession(FakePromptSession):
        async def prompt_async(self, *args, **kwargs):
            raise EOFError()

    monkeypatch.setattr("src.chat.core.Chatbot", FakeChatbot)
    monkeypatch.setattr("src.chat.core.PromptSession", lambda: EofSession([]))
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_a, **_k: None)
    asyncio.run(__import__("src.chat.core", fromlist=["x"]).start_chat_session_async())


# --- 否定变异：``if not callable(apply_sync)`` 去掉 not ----------------------


def test_start_chat_session_survives_bot_without_config_update_methods(monkeypatch):
    """bot 两个配置更新方法都没有时报错并留在会话里，而不是终止。

    ``if not callable(apply_sync)`` 去掉 ``not`` 后，缺方法的替身走
    ``apply_sync(...)`` → 裸 ``TypeError``；它不落在 ``except (ValueError,
    KeyError)`` 里，会一路穿透到 ``finally``，用户被丢出会话。
    """
    from tests.chat.test_core import FakeChatbot, FakePromptSession

    class NoUpdateChatbot(FakeChatbot):
        apply_config_update = None
        apply_config_update_async = None

        def __getattribute__(self, name):
            # 两个方法都不存在：走 ``getattr(bot, ..., None)`` 的 None 分支。
            if name in {"apply_config_update", "apply_config_update_async"}:
                return None
            return super().__getattribute__(name)

    monkeypatch.setattr("src.chat.core.Chatbot", NoUpdateChatbot)
    monkeypatch.setattr(
        "src.chat.core.PromptSession",
        lambda: FakePromptSession(["/config", "/quit"]),
    )
    monkeypatch.setattr("src.chat.core.display_chat_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.chat.core.launch_config_editor",
        lambda chat_config: (False, dict(chat_config)),
    )
    # 不抛异常即表示会话存活到 /quit。
    asyncio.run(__import__("src.chat.core", fromlist=["x"]).start_chat_session_async())
