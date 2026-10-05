import asyncio
import base64
import importlib
import math
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

import openai
import pytest

import src.providers.factory as factory_module
from src.providers.__base__.model_provider import (
    CompletionRequest,
    LargeLanguageModel,
    StreamEventError,
    _responses_content_part,
    is_retryable_error,
    iterate_async,
    normalize_embedding_vector,
    raise_for_stream_error_event,
    require_token_count,
    retry_async_stream,
    retry_sync_stream,
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


# ── 回归：空的 base_url 必须与「未配置」同义，而不是把请求打到相对路径 ──
#
# ``openai_compatible.__init__`` 用 ``getattr(settings, "<provider>_base_url")``
# 取值，随后只判 ``if provider != "openai" and not self._base_url``——**openai
# 被显式排除在外**（它允许回落到 SDK 自带的默认地址）。而字段的默认值非空、
# ``.env`` 里又有 ``OPENAI_API_BASE``，于是「显式置空」这条路径没人管：
# 实测 ``OPENAI_API_BASE=""`` 时 ``self._base_url`` 是空串（不是 ``None``），
# 判断 ``is None`` 的回落分支不触发，``openai.OpenAI(base_url="")`` 得到
# ``URL('')``，请求 URL 变成相对路径 ``chat/completions``——没有主机名，
# 失败信息与「地址配错」毫无关系。
#
# 判据要覆盖三个取值：``""``（显式置空）、``"   "``（空白，SDK 会把它编码成
# ``%20%20%20/``）、以及 ``None``（真正未配置，应当回落 SDK 默认）。只判
# ``is None`` 会漏掉前两者，只判 falsy 又会把 ``None`` 一并拒掉。
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_openai_compatible_provider_rejects_blank_base_url(monkeypatch, blank_value):
    fake_settings = SimpleNamespace(
        openai_api_key="token",
        openai_api_base=blank_value,
    )
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)

    with pytest.raises(ValueError) as excinfo:
        OpenAICompatibleProvider("demo", "openai")

    message = str(excinfo.value)
    assert "openai_api_base" in message
    # 只查键名的话，一个把任意异常都包成同一句话的实现也能通过；消息必须
    # 指出「空值」这一真实原因。
    assert "空" in message


def test_openai_compatible_provider_still_defaults_when_base_url_is_unset(monkeypatch):
    """反向守卫：``None`` 是「未配置」，必须继续回落 SDK 默认地址。

    把判据写成「一律拒绝空值」会连这条一起拒掉，而 ``openai`` 渠道依赖它。
    """
    fake_settings = SimpleNamespace(openai_api_key="token", openai_api_base=None)
    monkeypatch.setattr("src.providers.openai_compatible.get_settings", lambda: fake_settings)

    provider = OpenAICompatibleProvider("demo", "openai")

    # 与「不传 base_url」的 SDK 客户端逐字节一致：``openai.base_url`` 这个模块级
    # 常量是 ``None``（不是默认地址），拿它作基准的话未修复的实现也能通过。
    assert provider._get_client().base_url == openai.OpenAI(api_key="token").base_url


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


# ── 流式重试助手：只重试「首事件之前」的建立阶段 ──
# 这段逻辑是生产流式路径的重试入口，此前完全没有覆盖。


def test_retry_sync_stream_retries_establishment_failure():
    """建立阶段抛错应重试，且只重试一次即可成功。"""
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("建立失败")
        return iter(["a", "b"])

    assert list(retry_sync_stream(factory)) == ["a", "b"]
    assert len(calls) == 2


def test_retry_sync_stream_does_not_retry_after_first_event():
    """首事件已产出后不得重试，否则会重复已输出的内容。"""
    calls = []

    def factory():
        calls.append(1)

        def gen():
            yield "first"
            raise ConnectionError("消费阶段失败")

        return gen()

    with pytest.raises(ConnectionError):
        for _ in retry_sync_stream(factory):
            pass

    assert len(calls) == 1


def test_retry_sync_stream_first_event_validator_sees_first_event():
    """首事件必须先过校验器，且校验通过后原样产出。"""
    seen = []

    def validate(event):
        seen.append(event)

    assert list(retry_sync_stream(lambda: iter([{"ok": 1}, {"ok": 2}]), validate)) == [
        {"ok": 1},
        {"ok": 2},
    ]
    assert seen == [{"ok": 1}]


def test_retry_sync_stream_retries_when_first_event_validator_raises_retryable():
    """首事件是 5xx/429 错误事件时，重试发生在消费之前，不会重复输出内容。"""
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            return iter([{"type": "error", "error": {"message": "overloaded", "status_code": 503}}])
        return iter([{"ok": 1}])

    def validate(event):
        raise_for_stream_error_event(event, "Test")

    assert list(retry_sync_stream(factory, validate)) == [{"ok": 1}]
    assert len(calls) == 2


def test_retry_sync_stream_does_not_retry_non_retryable_validator_error():
    """校验器抛不可重试错误（如参数错误）时不得重试。"""
    calls = []

    def factory():
        calls.append(1)
        return iter([{"bad": True}])

    def validate(event):
        raise ValueError("不可重试")

    with pytest.raises(ValueError, match="不可重试"):
        list(retry_sync_stream(factory, validate))

    assert len(calls) == 1


def test_retry_sync_stream_empty_stream_is_not_an_error():
    """空流不是失败，不得重试也不得抛错。"""
    calls = []

    def factory():
        calls.append(1)
        return iter([])

    assert list(retry_sync_stream(factory)) == []
    assert len(calls) == 1


def test_retry_sync_stream_closes_iterator_on_consumer_abort():
    """消费方提前退出时须关闭底层迭代器，避免连接泄漏。"""
    closed = []

    class ClosingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return "item"

        def close(self):
            closed.append(True)

    assert next(iter(retry_sync_stream(lambda: ClosingIterator()))) == "item"

    assert closed == [True]


def test_retry_async_stream_retries_establishment_failure():
    calls = []

    async def factory():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("建立失败")

        async def gen():
            yield "a"
            yield "b"

        return gen()

    async def collect():
        return [item async for item in retry_async_stream(factory)]

    assert asyncio.run(collect()) == ["a", "b"]
    assert len(calls) == 2


def test_retry_async_stream_does_not_retry_after_first_event():
    calls = []

    async def factory():
        calls.append(1)

        async def gen():
            yield "first"
            raise ConnectionError("消费阶段失败")

        return gen()

    async def consume():
        async for _ in retry_async_stream(factory):
            pass

    with pytest.raises(ConnectionError):
        asyncio.run(consume())

    assert len(calls) == 1


def test_retry_async_stream_closes_iterator_on_consumer_abort():
    closed = []

    class ClosingAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return "item"

        async def aclose(self):
            closed.append(True)

    async def consume_one():
        async for _ in retry_async_stream(lambda: ClosingAsyncIterator()):
            return

    asyncio.run(consume_one())

    assert closed == [True]


# ── 回归：错误分类不得被「看起来像数字」的 Unicode 码位打断 ──
#
# ``is_retryable_error`` 与 ``raise_for_stream_error_event`` 都用 ``str.isdigit()``
# 判断 SDK/网关给的 status 是不是数字串。``isdigit()`` 对 128 个 Unicode 码位
# 为真（``²`` ``①`` ``፩`` ``₅`` …）而 ``int()`` 对它们一律抛 ``ValueError``——
# 于是一个形态奇怪的 status 值会让**错误分类本身**崩掉：本该重试的 5xx 变成
# 一条解析异常，本该报出的流错误也变成一条解析异常。
#
# 全码位穷举：``isdigit()`` 为真的 788 个码位里 128 个 ``int()`` 抛错；
# ``isdecimal()`` 为真的 660 个码位 ``int()`` 全部可解析。


@pytest.mark.parametrize("status", ["²", "①", "፹", "₅", "4²0"])
def test_is_retryable_error_survives_unparseable_digit_status(status):
    """status 是「看起来像数字但 ``int()`` 解析不了」时必须给出布尔判断。"""
    error = SimpleNamespace(status_code=status)

    assert is_retryable_error(error) is False


@pytest.mark.parametrize(
    "status,expected",
    [("503", True), ("429", True), ("404", False), ("200", False)],
)
def test_is_retryable_error_still_classifies_numeric_status(status, expected):
    """收紧不能误伤：正常数字串的分类结果不得改变。"""
    error = SimpleNamespace(status_code=status)

    assert is_retryable_error(error) is expected


@pytest.mark.parametrize("status", ["²", "①", "₅"])
def test_stream_error_event_survives_unparseable_digit_status(status):
    """流错误事件的 status 解析不了时必须照常报出流错误，而不是解析异常。"""
    event = {"type": "error", "error": {"message": "overloaded", "status_code": status}}

    with pytest.raises(StreamEventError, match="overloaded") as excinfo:
        raise_for_stream_error_event(event, "Test")

    assert excinfo.value.status_code is None


def test_stream_error_event_still_extracts_numeric_status():
    """收紧不能误伤：正常数字串仍要提取成 status_code。"""
    event = {"type": "error", "error": {"message": "overloaded", "status_code": "503"}}

    with pytest.raises(StreamEventError) as excinfo:
        raise_for_stream_error_event(event, "Test")

    assert excinfo.value.status_code == 503


@pytest.mark.parametrize("status", ["²", "①", "4²0"])
def test_chat_status_is_error_survives_unparseable_digit_status(status):
    """网关响应体里的 status 解析不了时，错误分类必须照常给出布尔判断。

    ``_chat_status_is_error`` 是 ``_raise_for_chat_response_error`` 的一环，
    用于识别「HTTP 200 但响应体里写着失败」的兼容网关。它崩掉意味着这类网关
    的业务错误被误报成空文本。
    """
    assert OpenAICompatibleProvider._chat_status_is_error(status) is False


def test_chat_status_is_error_still_classifies_numeric_status():
    """收紧不能误伤：正常数字串的分类结果不得改变。"""
    assert OpenAICompatibleProvider._chat_status_is_error("500") is True
    assert OpenAICompatibleProvider._chat_status_is_error("200") is False
    assert OpenAICompatibleProvider._chat_status_is_error("error") is True


def _huge_digits():
    import sys

    return "9" * (sys.get_int_max_str_digits() + 1)


def test_is_retryable_error_survives_digit_strings_beyond_the_conversion_limit():
    """超长数字 status 必须给出布尔判断，而不是让重试判定抛 ``ValueError``。"""
    error = SimpleNamespace(status_code=_huge_digits())

    assert is_retryable_error(error) is False


def test_stream_error_event_survives_digit_strings_beyond_the_conversion_limit():
    """超长数字 status 必须照常报出流错误，而不是解析异常。"""
    event = {"type": "error", "error": {"message": "overloaded", "status_code": _huge_digits()}}

    with pytest.raises(StreamEventError, match="overloaded") as excinfo:
        raise_for_stream_error_event(event, "Test")

    assert excinfo.value.status_code is None


def test_chat_status_is_error_survives_digit_strings_beyond_the_conversion_limit():
    """超长数字 status 必须给出布尔判断，而不是让错误分类抛 ``ValueError``。"""
    assert OpenAICompatibleProvider._chat_status_is_error(_huge_digits()) is False


# ── 回归：token 计数响应只判了 None，形态异常会抛与「token 计数」无关的异常 ──
#
# 八个方法（Anthropic 同步/异步 × 普通/Beta、Google 同步/异步、OpenAI Responses
# 同步/异步）此前都是「判 None 后直接 ``int(value)``」。那只挡住了缺字段：
# SDK 或兼容网关返回 ``"abc"``／``1.5``／``[1]``／位数超限的数字串时，``int()``
# 抛裸 ``ValueError``／``TypeError``／位数超限错误，调用方拿不到是哪个字段出的
# 问题。非负整数才是这份数据的真实契约——token 计数不可能为负，也不可能带小数。


@pytest.mark.parametrize(
    "value,label",
    [
        (None, "缺字段"),
        ("abc", "非数值字符串"),
        ("1.5", "浮点字符串"),
        ("9" * 5000, "位数超限的数字串"),
        (1.5, "浮点数"),
        (float("inf"), "无穷大"),
        (float("nan"), "NaN"),
        ([1], "列表"),
        ({"a": 1}, "映射"),
        (True, "布尔值"),
        (-5, "负数"),
    ],
)
def test_require_token_count_reports_malformed_values(value, label):
    """形态异常必须报成带 provider 与字段名的 ``RuntimeError``。"""
    with pytest.raises(RuntimeError) as excinfo:
        require_token_count(value, "Test", "total_tokens")

    message = str(excinfo.value)
    assert "Test" in message, label
    assert "total_tokens" in message, label


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (12, 12), (12.0, 12), ("12", 12), ("0", 0)],
)
def test_require_token_count_accepts_valid_counts(value, expected):
    """收紧不能误伤：正常计数照常通过（含整数值的浮点与数字字符串）。"""
    assert require_token_count(value, "Test", "total_tokens") == expected


def test_require_token_count_never_raises_a_non_runtime_error():
    """契约是「要么返回 int，要么抛 RuntimeError」，不能漏出 ``OverflowError``。

    ``int(float("inf"))`` 抛的是 ``OverflowError``——它既不是 ``ValueError`` 也
    不是 ``TypeError``，任何按后者写的处理都接不住。

    ``10**400`` 与 ``"9" * 5000`` 走的是**正数**方向，负数方向另有
    ``test_require_token_count_negative_branch_survives_unprintable_integer``
    覆盖：那条路径上的消息格式化自己也会抛，是本文件里唯一一处「抛异常时再抛
    异常」的站点。
    """
    for value in [float("inf"), float("-inf"), float("nan"), 10**400, "9" * 5000]:
        try:
            require_token_count(value, "Test", "total_tokens")
        except RuntimeError:
            pass
        except BaseException as exc:
            raise AssertionError(f"{value!r} 抛出了 {type(exc).__name__}: {exc}") from exc


# ``repr(10**4301)`` 自己抛 ``ValueError: Exceeds the limit (4300 digits)``
# （CPython 的 ``sys.get_int_max_str_digits()``，默认 4300）。错误消息里凡是
# 直接插值 ``{numeric}`` 的地方，都会在**正要报告一个错误时**改抛另一个错误，
# 把契约里承诺的 ``RuntimeError`` 换成与字段无关的 ``ValueError``。
#
# 这条路径只有**负数**能走到：正数在 ``numeric < 0`` 之前就返回了，因此
# 上面那条「永不抛非 RuntimeError」的用例（探针全是正数）碰不到它。实测把
# ``safe_repr(numeric)`` 换回 ``{numeric}`` 后本用例变红。
def test_require_token_count_negative_branch_survives_unprintable_integer():
    """负数分支的消息格式化不得自己抛异常。"""
    huge_negative = -(10**4301)

    with pytest.raises(RuntimeError) as excinfo:
        require_token_count(huge_negative, "Test", "total_tokens")

    message = str(excinfo.value)
    assert "Test" in message
    assert "total_tokens" in message
    assert "Exceeds the limit" not in message, message


# ---------------------------------------------------------------------------
# 超出双精度范围的整数
# ---------------------------------------------------------------------------
#
# ``math.isfinite(x)`` 与 ``float(x)`` 对任意大的 Python int 都会先尝试转成
# C double，超出 ``DBL_MAX``（约 1.8e308）时抛 ``OverflowError: int too large
# to convert to float``。``OverflowError`` 是 ``ArithmeticError`` 的子类，
# **不是** ``ValueError``，会穿透这些方法承诺的异常契约与调用方的
# ``except ValueError``。
#
# 输入可达：``config.toml`` 的整数是无上界的 TOML 整数，``tomllib`` 原样解析
# 成 Python int；provider 的 ``options`` 又是文档指定的扩展入口。provider
# 返回值一侧同理——网络 JSON 里出现一个超大整数字面量就能走到。
#
# 参数 id 必须显式给出：``10**4301`` 超过 ``sys.get_int_max_str_digits()``，
# pytest 自动生成 id 时调 ``str(val)`` 会自己抛 ``ValueError``，测试在收集
# 阶段就崩掉。
_BEYOND_DOUBLE_INTS = [
    pytest.param(10**400, id="400digits"),
    pytest.param(-(10**400), id="negative-400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
]


@pytest.mark.parametrize("value", _BEYOND_DOUBLE_INTS)
def test_normalize_embedding_vector_rejects_ints_beyond_double_range(value):
    """SDK 返回的向量里出现超大整数时，必须报成 ``ValueError`` 而不是 ``OverflowError``。"""
    with pytest.raises(ValueError, match="超出双精度范围"):
        normalize_embedding_vector([1.0, value])


def test_normalize_embedding_vector_still_passes_through_non_finite_floats():
    """只收敛 ``OverflowError``：``inf``/``nan`` 的既有契约不变。

    base64 分支（``struct.unpack``）本来就可能产出 ``inf``/``nan``，有限性由下游
    ``EmbeddingService`` 的 ``np.isfinite`` 检查统一负责。在这里另加一道拒绝会让
    两个分支的契约不一致。
    """
    assert normalize_embedding_vector([float("inf"), 1.0]) == [float("inf"), 1.0]
    assert math.isnan(normalize_embedding_vector([float("nan")])[0])


# `normalize_embedding_vector` 的 base64／类型分支有三条判据在整套测试里从未执行
# （`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3323 条仍然全绿）。
# 既有用例只喂 list 形态（走 `isinstance(value, (list, tuple))` 分支），三条都绕开了。
# 删除后行为确实改变（实测，全在 `src/providers/__base__/model_provider.py`）：
#   * 1397 非法 base64 → `encoded` 未赋值，抛 `UnboundLocalError`（不是
#     `ValueError`，穿透 `EmbeddingService` 的 `except ValueError`）；
#   * 1399 不支持的类型 → 同样抛 `UnboundLocalError`，把「SDK 返回了奇怪类型」
#     说成一个与类型无关的名字错误；
#   * 1402 空字节串 → **静默返回空列表**（一个零维 embedding 被当成合法结果）；
#     长度非 4 的倍数 → 裸 `struct.error: unpack requires a buffer of 4 bytes`。
def test_normalize_embedding_vector_rejects_invalid_base64():
    """非法 base64 必须报 ``ValueError``，不是 ``UnboundLocalError``。"""
    with pytest.raises(ValueError, match="base64 内容无效"):
        normalize_embedding_vector("!!!not-base64!!!")


@pytest.mark.parametrize("value", [5, None, {"a": 1}, object()])
def test_normalize_embedding_vector_rejects_unsupported_types(value):
    """SDK 返回不支持的类型时必须给带类型名的 ``TypeError``，不是 ``UnboundLocalError``。"""
    with pytest.raises(TypeError, match="不支持的类型"):
        normalize_embedding_vector(value)


@pytest.mark.parametrize(
    "value", [b"", b"12345", b"\x00\x00\x00"], ids=["empty", "not-multiple-of-4", "three-bytes"]
)
def test_normalize_embedding_vector_rejects_non_float32_byte_lengths(value):
    """字节长度不是 4 的倍数时必须 ``ValueError``，不能静默返回空列表或抛 ``struct.error``。"""
    with pytest.raises(ValueError, match="不是有效的 float32 字节序列"):
        normalize_embedding_vector(value)


def test_normalize_embedding_vector_accepts_valid_base64_and_bytes():
    """收紧不能误伤：合法的 base64 与字节串照常解出浮点列表。"""
    encoded = base64.b64encode(struct.pack("<2f", 1.5, -2.5)).decode()

    assert normalize_embedding_vector(encoded) == [1.5, -2.5]
    assert normalize_embedding_vector(struct.pack("<2f", 1.5, -2.5)) == [1.5, -2.5]


@pytest.mark.parametrize("value", _BEYOND_DOUBLE_INTS)
def test_responses_video_fps_beyond_double_range_is_explicit(value):
    """Ark Responses 的 ``fps`` 超出双精度范围时必须是 ``ValueError``。"""
    with pytest.raises(ValueError, match=r"fps 必须是有限数字"):
        _responses_content_part(
            {
                "type": "input_video",
                "video_url": {"url": "https://example.invalid/v.mp4"},
                "fps": value,
            },
            "params",
            provider="ark",
        )


# ── 回归：Ark Responses 失败出口的 message 形态 ──
#
# ``VolcengineProvider._raise_for_response_error`` 的 ``status == "failed"``
# 分支写 ``raise RuntimeError(str(message))``，``message`` 是网关 JSON 的
# ``error``（或它的 ``message`` 字段）。位数超限的整数让裸 ``str()`` 抛
# ``ValueError: Exceeds the limit``——**正要报告网关错误**的那一步变成整数转换
# 错误，而调用方按 ``RuntimeError`` 分类。同函数的 ``incomplete`` 分支与
# ``openai_compatible._error_message`` 都已按 ``safe_repr`` 收敛，这一支漏了。


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({"status": "failed", "error": 10**4301}, id="huge-int"),
        pytest.param({"status": "failed", "error": {"message": 10**4301}}, id="huge-int-message"),
    ],
)
def test_ark_responses_failed_reports_malformed_error_readably(response):
    """非字符串 ``error`` 不得让「报告网关错误」自己抛裸异常。"""
    with pytest.raises(RuntimeError) as excinfo:
        VolcengineProvider._raise_for_response_error(response)

    assert "Exceeds the limit" not in str(excinfo.value)


def test_ark_responses_failed_still_reports_string_error():
    """收紧不能误伤：字符串 ``error`` 仍原样出现在消息里。"""
    with pytest.raises(RuntimeError, match="boom"):
        VolcengineProvider._raise_for_response_error({"status": "failed", "error": "boom"})


# ── 回归：流错误事件兜底用裸 ``code`` 插值 ──
#
# ``raise_for_stream_error_event`` 在 message 提取失败后回落到
# ``field(error, "code") or f"{provider} 流返回错误"``。``code`` 来自网关 JSON，
# 形态不受控：位数超限的整数让 f-string 抛 ``ValueError: Exceeds the limit``——
# 正要报出流错误的那一步变成整数转换错误，而调用方按 ``StreamEventError`` 分类。
# 同函数提取 ``message`` 的那条路径只收字符串，这一支漏了。


@pytest.mark.parametrize(
    "code",
    [pytest.param(10**4301, id="huge-int"), pytest.param(["a"], id="list")],
)
def test_stream_error_event_reports_malformed_code_readably(code):
    """非字符串 ``code`` 不得让「报告流错误」自己抛裸异常。"""
    event = {"type": "error", "error": {"code": code}}

    with pytest.raises(StreamEventError) as excinfo:
        raise_for_stream_error_event(event, "Test")

    assert "Exceeds the limit" not in str(excinfo.value)


def test_stream_error_event_still_reports_string_code():
    """收紧不能误伤：字符串 ``code`` 仍原样出现在消息里。"""
    event = {"type": "error", "error": {"code": "invalid_api_key"}}

    with pytest.raises(StreamEventError, match="invalid_api_key"):
        raise_for_stream_error_event(event, "Test")


# ── 回归：错误分类不得被「形态奇怪的 status」打断（容器持大整数）──
#
# ``_chat_status_is_error`` 的 docstring 明确承诺「一个形态奇怪的 status 值不能
# 让错误分类本身崩掉」，它也确实挡住了「字符串形态但位数超限」的取值。但收敛
# 只做了一半：非 ``int`` 的非字符串走 ``str(status).strip().lower()``，而
# ``str([10**4301])`` 自己抛 ``ValueError: Exceeds the limit``——于是
# 「网关在响应体里返回了业务错误」变成一条整数转换错误，调用方的错误分类与日志
# 脱敏都接不住。``status`` 来自网关 JSON，形态不受控。


@pytest.mark.parametrize(
    "status",
    [
        pytest.param([10**4301], id="list-with-huge-int"),
        pytest.param({"code": 10**4301}, id="dict-with-huge-int"),
    ],
)
def test_chat_status_is_error_survives_container_with_huge_int(status):
    """容器里含位数超限整数时，错误分类必须照常给出布尔判断。"""
    assert OpenAICompatibleProvider._chat_status_is_error(status) is False


def test_chat_response_error_reports_container_status_readably():
    """从 ``_raise_for_chat_response_error`` 进入时同样不抛整数转换错误。"""
    response = {"status": [10**4301], "choices": None}

    with pytest.raises(RuntimeError) as excinfo:
        OpenAICompatibleProvider._raise_for_chat_response_error(response)

    assert "Exceeds the limit" not in str(excinfo.value)


# ── 回归：provider 构造器对非字符串 model_name 无形态守卫 ──
#
# 六个 provider 的 ``__init__`` 都把 ``model_name`` 原样存下、不做形态判据：
#   * google / anthropic / jina / siliconflow_rerank / openai_compatible 在
#     ``logger.info(f"...模型: {model_name}")`` 处抛裸
#     ``ValueError: Exceeds the limit``（位数超限的整数），而该消息与「模型名
#     写错了」无关；列表/``None`` 则静默通过，原样发给 SDK；
#   * volcengine 连日志都没有，非字符串全程静默接受。
#
# 这些类都是**公开类**，嵌入方可以绕过工厂直接构造（本仓测试大量这么用），而
# 工厂路径已用 ``require_model_name`` 收敛（``cc10585``）。同一个 ``model_name``
# 走工厂得到带指引的 ``ValueError``，直接构造却得到整数转换错误或静默接受——
# 判据只做了一半。


def _model_name_settings():
    return SimpleNamespace(
        openai_api_key="k",
        openai_api_base="https://api.openai.com/v1",
        anthropic_api_key="k",
        google_api_key="k",
        jina_api_key="k",
        siliconflow_api_key="k",
        siliconflow_base_url="https://api.siliconflow.cn/v1",
        ark_api_key="k",
        volc_access_key="ak",
        volc_secret_key="sk",
        volc_base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


def _constructible_providers(monkeypatch):
    """构造六个 provider 的工厂，各自用最小的合法凭证设置。

    provider **类**必须从本次新导入的模块取（而不是文件顶部的导入）：
    ``tests/providers/test_factory.py`` 会把 ``src.providers.*`` 从 ``sys.modules``
    里删掉再重新导入，于是文件顶部那个类属于**旧**模块对象，而
    ``__init__`` 仍从旧模块字典取 ``get_settings``——patch 落在新模块上，类却读
    旧模块，全量运行时读到真实配置（无 key）后抛 ``API_KEY is required``。单独跑
    本文件不触发，只有全量运行才暴露。
    """
    settings = _model_name_settings()
    cases = [
        ("google", "GoogleProvider", lambda cls, v: cls(v)),
        ("anthropic", "AnthropicProvider", lambda cls, v: cls(v)),
        ("jina", "JinaProvider", lambda cls, v: cls(v)),
        ("siliconflow_rerank", "SiliconflowRerankProvider", lambda cls, v: cls(v)),
        ("volcengine", "VolcengineProvider", lambda cls, v: cls(v)),
        (
            "openai_compatible",
            "OpenAICompatibleProvider",
            lambda cls, v: cls(v, "openai", settings=settings),
        ),
    ]
    factories = []
    for module_name, class_name, build in cases:
        module = importlib.import_module(f"src.providers.{module_name}")
        monkeypatch.setitem(module.__dict__, "get_settings", lambda: settings)
        provider_class = getattr(module, class_name)
        factories.append((class_name, lambda v, build=build, cls=provider_class: build(cls, v)))
    return factories


@pytest.mark.parametrize(
    "bad_name",
    [pytest.param(10**4301, id="huge-int"), pytest.param(["m"], id="list")],
)
def test_provider_constructors_reject_non_string_model_name_readably(monkeypatch, bad_name):
    """非字符串 model_name 必须给出带字段名的 ``ValueError``，不得静默接受。"""
    for label, build in _constructible_providers(monkeypatch):
        with pytest.raises(ValueError) as excinfo:
            build(bad_name)
        message = str(excinfo.value)
        assert "Exceeds the limit" not in message, f"{label}: {message}"
        assert "model_name" in message, f"{label}: {message}"


def test_provider_constructors_still_accept_string_model_names(monkeypatch):
    """收紧不能误伤：字符串 model_name 照常构造。"""
    for label, build in _constructible_providers(monkeypatch):
        build("demo-model")


# ── 回归：OpenAI 兼容构造器的 ``provider`` 参数无形态守卫 ──
#
# 与 ``model_name`` 同一处、同一类：``__init__`` 里 ``provider.lower()`` 直接调用，
# 非字符串抛裸 ``AttributeError: 'list' object has no attribute 'lower'``。而
# ``provider`` 随后要拼出 ``{provider}_api_key`` 字段名——本类的其余拒绝路径
# （缺 key／缺 base_url／协议非法）都给带指引的 ``ValueError``。工厂路径已用
# ``require_provider_name`` 收敛，直接构造这条路径漏了。


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param(10**4301, id="huge-int"),
        pytest.param(["openai"], id="list"),
        pytest.param(None, id="none"),
    ],
)
def test_openai_compatible_rejects_non_string_provider_readably(provider):
    """非字符串 provider 必须给出带字段名的 ``ValueError``。"""
    settings = _model_name_settings()

    with pytest.raises(ValueError) as excinfo:
        OpenAICompatibleProvider("demo", provider, settings=settings)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "provider" in message, message


def test_openai_compatible_still_accepts_string_provider():
    """收紧不能误伤：字符串 provider 照常构造。"""
    settings = _model_name_settings()

    provider = OpenAICompatibleProvider("demo", "openai", settings=settings)

    assert provider._provider == "openai"
