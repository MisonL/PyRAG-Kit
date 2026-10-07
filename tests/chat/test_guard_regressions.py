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
