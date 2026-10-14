"""补齐「吞掉异常」处理器（fail-open / 静默降级）的测试覆盖缺口（Provider 批）。

来源：对 src/ 下 87 条「吞掉型 except」（body 无 raise）做变异扫描——把
handler body 替换成 ``raise``（取消吞掉、让异常冒泡）后全量测试仍绿的存活项。
逐条用真实入口确认「取消吞掉后行为确实改变」，再按守卫存在时的行为写断言。
"""

import asyncio
import types

from src.providers.__base__.model_provider import (
    _is_provider_resource_method,
    _provider_call_accepts_arguments,
    retry_async_stream,
    retry_async_stream_context,
    retry_sync_stream_context,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.volcengine import VolcengineProvider

# --- close()/aclose()：一个资源失败不得阻止其余资源被释放 -----------------


class _SyncFailing:
    def close(self):
        raise RuntimeError("同步关闭失败")


class _AsyncFailing:
    def close(self):
        raise RuntimeError("异步关闭失败")


class _Recording:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class _AsyncRecording:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def _check_sync_close(provider):
    """close()：同步客户端失败后异步客户端仍要关闭，且报告第一个错误。"""
    provider._client = _SyncFailing()
    recorder = _Recording()
    provider._aclient = recorder
    try:
        provider.close()
    except RuntimeError as exc:
        assert "同步关闭失败" in str(exc)
    else:  # pragma: no cover - 守卫被删时才会走到
        raise AssertionError("close() 必须报告第一个错误")
    assert recorder.closed == 1, "同步客户端失败后异步客户端必须仍被关闭"


def _check_async_close(provider):
    """aclose()：第一个客户端失败后第二个仍要关闭，且报告第一个错误。"""
    provider._aclient = _AsyncFailing()
    recorder = _Recording()
    provider._client = recorder
    try:
        asyncio.run(provider.aclose())
    except RuntimeError as exc:
        assert "异步关闭失败" in str(exc)
    else:  # pragma: no cover - 守卫被删时才会走到
        raise AssertionError("aclose() 必须报告第一个错误")
    assert recorder.closed == 1, "第一个客户端失败后第二个必须仍被关闭"


def test_anthropic_close_and_aclose_release_all_clients():
    provider = object.__new__(AnthropicProvider)
    _check_sync_close(provider)
    provider = object.__new__(AnthropicProvider)
    _check_async_close(provider)


def test_google_close_and_aclose_release_all_clients():
    class _AioFailing:
        async def aclose(self):
            raise RuntimeError("异步关闭失败")

    class _ClientWithFailingClose:
        def __init__(self):
            self.aio = _AsyncRecording()

        def close(self):
            raise RuntimeError("同步关闭失败")

    provider = object.__new__(GoogleProvider)
    client = _ClientWithFailingClose()
    provider._client = client
    try:
        provider.close()
    except RuntimeError as exc:
        assert "同步关闭失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("close() 必须报告第一个错误")
    assert client.aio.closed == 1, "同步客户端失败后 aio 必须仍被关闭"

    class _ClientWithFailingAio:
        def __init__(self):
            self.aio = _AioFailing()
            self.closed = 0

        def close(self):
            self.closed += 1

    provider = object.__new__(GoogleProvider)
    client = _ClientWithFailingAio()
    provider._client = client
    try:
        asyncio.run(provider.aclose())
    except RuntimeError as exc:
        assert "异步关闭失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("aclose() 必须报告第一个错误")
    assert client.closed == 1, "aio 失败后同步客户端必须仍被关闭"


def test_openai_compatible_close_and_aclose_release_all_clients():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    _check_sync_close(provider)
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    _check_async_close(provider)


def test_volcengine_close_and_aclose_release_all_clients():
    provider = object.__new__(VolcengineProvider)
    provider._provider = None
    _check_sync_close(provider)
    provider = object.__new__(VolcengineProvider)
    provider._provider = None
    _check_async_close(provider)


# --- 基础层：空流、签名探测 ------------------------------------------------


def test_retry_async_stream_empty_stream_ends_cleanly():
    """空异步流以 ``StopAsyncIteration`` 正常收尾。

    取消吞掉 ``StopAsyncIteration`` 后，空流被当成「重试未返回结果」抛
    ``RuntimeError``——一个合法的空流变成失败。
    """

    class _Empty:
        def __aiter__(self):
            async def gen():
                if False:
                    yield 1

            return gen()

    async def consume():
        stream = retry_async_stream(lambda: _Empty())
        with pytest_raises_stop():
            await stream.__anext__()

    asyncio.run(consume())


def pytest_raises_stop():
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        try:
            yield
        except StopAsyncIteration:
            return
        raise AssertionError("空流必须以 StopAsyncIteration 收尾")

    return _ctx()


def test_provider_call_accepts_arguments_returns_false_for_unsignaturable():
    """签名不可读的可调用对象返回 False（不抛）。

    取消吞掉后 ``inspect.signature`` 的 ``ValueError`` 直接冒泡——本函数
    的契约是「探测」，形态奇怪的目标应当返回 False。
    """

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

        def __call__(self):
            pass

    assert _provider_call_accepts_arguments(_NoSignature(), object(), (), {}) is False


def test_is_provider_resource_method_returns_false_for_unsignaturable():
    """签名不可读的可调用对象不算资源方法。"""

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

        def __call__(self):
            pass

    assert _is_provider_resource_method("op", _NoSignature()) is False


# --- Anthropic：类型注解解析、快照拒答探测 ---------------------------------


def test_anthropic_safe_type_hint_returns_none_for_unresolvable_annotation():
    """无法解析的注解返回 None，不影响其余成员。

    取消吞掉后 ``get_type_hints`` 的异常直接冒泡——一个成员的模型结构
    变化会让整张表构造失败。
    """

    from src.providers.anthropic import _safe_type_hint

    class _Bad:
        __annotations__ = {"x": "NotDefined.Anywhere"}

    assert _safe_type_hint(_Bad, "x") is None


def test_anthropic_snapshot_refusal_probe_tolerates_missing_attribute():
    """响应对象没有 ``current_message_snapshot`` 时静默返回。

    取消吞掉后 ``AttributeError`` 冒泡——测试替身与旧版 SDK 的响应对象
    都会让流消费在第一次探测时崩溃。
    """

    class _NoSnapshot:
        pass

    AnthropicProvider._raise_for_stream_snapshot_refusal(_NoSnapshot())


# --- Google：HTTP 配置 URL 解析、响应文本回退 ------------------------------


def test_google_validate_http_options_skips_unparsable_urls():
    """无法解析的 URL 跳过检查而不是抛错。

    取消吞掉后 ``urlsplit`` 的 ``ValueError: Invalid IPv6 URL`` 冒泡——
    校验函数本身成了崩溃点。
    """
    GoogleProvider._validate_http_options({"headers": {"X": "http://[bad"}})


def test_google_response_text_falls_back_when_accessor_raises():
    """``response.text`` 取值抛错时回退到候选部件提取。

    取消吞掉后异常冒泡——响应解析在用户已拿到部分输出后崩溃。
    """

    class _Boom:
        def __init__(self):
            self.candidates: list = []

        @property
        def text(self):
            raise RuntimeError("SDK 取值失败")

    assert GoogleProvider._response_text(_Boom()) == ""


# --- OpenAI 兼容：SDK 校验回退、端口解析、工具键跳过 -----------------------


def test_openai_validate_sdk_resource_call_skips_unsignaturable_method():
    """方法签名不可读时跳过校验而不是抛错。"""

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

    provider._validate_sdk_resource_call(_NoSignature(), (), {}, "op")


def test_openai_official_endpoint_detection_handles_bad_port():
    """URL 端口越界时返回 False 而不是抛 ``ValueError``。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    provider._base_url = "https://api.openai.com:99999/v1"
    assert provider._uses_official_openai_endpoint() is False


def test_openai_response_tool_keys_skips_unhashable_values():
    """不可哈希的分量被跳过，其余分量仍进键元组。

    取消吞掉后 ``hash()`` 的 ``TypeError: unhashable type`` 冒泡。
    """
    keys = OpenAICompatibleProvider._response_tool_keys(
        {"item_id": [1, 2], "output_index": 0, "call_id": "c1"}
    )
    assert ("call_id", "c1") in keys
    assert all(not isinstance(value, list) for _, value in keys)


def test_close_aggregates_errors_reporting_the_first_failure():
    """两个客户端都关闭失败时报告**第一个**错误，而不是被第二个顶替。

    删除 ``first_error = first_error or exc`` 后，后发生的错误覆盖先发生的
    ——调用方看到的失败原因与实际顺序不符。
    """

    class _Failing:
        def __init__(self, message):
            self._message = message

        def close(self):
            raise RuntimeError(self._message)

    cases = [
        (AnthropicProvider, "Anthropic"),
        (OpenAICompatibleProvider, "OpenAI"),
        (VolcengineProvider, "Ark"),
    ]
    for provider_class, label in cases:
        provider = object.__new__(provider_class)
        if provider_class is not AnthropicProvider:
            provider._provider = None
        provider._client = _Failing("第一个失败")
        provider._aclient = _Failing("第二个失败")
        try:
            provider.close()
        except RuntimeError as exc:
            assert "第一个失败" in str(exc), f"{label} 报告了被顶替的错误"
        else:  # pragma: no cover
            raise AssertionError(f"{label} close() 必须报告错误")


def test_google_close_aggregates_errors_reporting_the_first_failure():
    """Google 的 close()/aclose() 同样报告第一个错误。"""

    class _FailingSync:
        def close(self):
            raise RuntimeError("同步失败")

    class _FailingAsync:
        async def aclose(self):
            raise RuntimeError("aio 失败")

        def close(self):
            raise RuntimeError("同步失败")

    class _ClientBothFailing:
        def __init__(self):
            # aio 的失败消息必须与同步的区分：两处都叫「同步失败」时，
            # 「保留第一个错误」与「被第二个顶替」不可观测（等价变异）。
            self.aio = types.SimpleNamespace(
                close=lambda: (_ for _ in ()).throw(RuntimeError("aio 失败"))
            )

        def close(self):
            raise RuntimeError("同步失败")

    provider = object.__new__(GoogleProvider)
    provider._client = _ClientBothFailing()
    try:
        provider.close()
    except RuntimeError as exc:
        assert "同步失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("close() 必须报告错误")

    # aio 是唯一失败源时，错误必须被报告出来（而不是被静默吞掉）。
    # 注意 close_resource_sync 优先调 close()，所以这个 aio 只有 aclose。
    class _OnlyAcloseFails:
        async def aclose(self):
            raise RuntimeError("aio 失败")

    class _ClientOnlyAioFails:
        def __init__(self):
            self.aio = _OnlyAcloseFails()
            self.closed = 0

        def close(self):
            self.closed += 1

    provider = object.__new__(GoogleProvider)
    only_aio = _ClientOnlyAioFails()
    provider._client = only_aio
    try:
        provider.close()
    except RuntimeError as exc:
        assert "aio 失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("aio 失败必须被报告")
    assert only_aio.closed == 1

    class _ClientBothFailingAio:
        def __init__(self):
            self.aio = types.SimpleNamespace(aclose=_failing_async_close)

        def close(self):
            raise RuntimeError("同步失败")

    async def _failing_async_close():
        raise RuntimeError("aio 失败")

    provider = object.__new__(GoogleProvider)
    provider._client = _ClientBothFailingAio()
    try:
        asyncio.run(provider.aclose())
    except RuntimeError as exc:
        assert "aio 失败" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("aclose() 必须报告错误")


def test_stream_context_exit_suppresses_body_error_and_still_closes():
    """``__exit__`` 返回 True 表示抑制：body 异常不上抛，且上下文必须退出。

    删除吞掉后 body 异常直接上抛、``__exit__`` 不被调用——资源清理被跳过，
    调用方也无法声明「我处理了这个错误」。
    """

    class _Manager:
        def __init__(self):
            self.exited = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.exited += 1
            return True

    manager = _Manager()
    with retry_sync_stream_context(lambda: manager):
        raise RuntimeError("body failed")
    assert manager.exited == 1

    class _AsyncManager:
        def __init__(self):
            self.exited = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.exited += 1
            return True

    async_manager = _AsyncManager()

    async def consume():
        async with retry_async_stream_context(lambda: async_manager):
            raise RuntimeError("body failed")

    asyncio.run(consume())
    assert async_manager.exited == 1


def test_sanitize_resource_call_keeps_call_shape_for_uninspectable_method():
    """签名不可读的动态 SDK 包装仍走 keyword/positional 校验路径。

    取消吞掉后 ``inspect.signature`` 的 ``ValueError`` 冒泡——返回时参数
    仍会被清洗，但整个调用被诊断异常打断。
    """

    from src.providers.__base__.model_provider import _sanitize_resource_call

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

    args, kwargs = _sanitize_resource_call(_NoSignature(), object(), (1,), {"model": "m"})
    assert args == (1,)
    assert kwargs == {"model": "m"}
