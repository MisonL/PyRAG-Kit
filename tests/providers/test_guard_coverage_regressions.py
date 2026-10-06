"""补齐「删守卫后全量测试仍绿」的非 raise 守卫覆盖缺口（Provider 批）。

每一条用例对应一条经变异扫描存活的守卫：把它的 ``if`` 整体替换成 ``pass``
后，3804 条测试全部通过——即整套测试没有任何断言能察觉该守卫消失。每条都
经真实入口探测确认删除后行为确实改变（崩溃、静默取值或消息内容），再按
**守卫存在时的行为**写成断言。详见 CHANGELOG 对应条目。
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.providers.__base__.model_provider import (
    CompletionRequest,
    StreamEvent,
    _guard_provider_resource_method,
    _responses_content_part,
    _responses_has_content,
    _responses_json_text,
    content_to_text,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.volcengine import VolcengineProvider

# --- OpenAI 兼容渠道 -------------------------------------------------------


def test_chat_stream_events_returns_usage_only_event_for_empty_choices():
    """``choices=[]`` 的收尾块（常见于 usage-only 块）不能索引 ``choices[0]``。

    删除守卫后 ``choices[0]`` 抛裸 ``IndexError: list index out of range``；
    真实网关在流的最后一块用空 ``choices`` 携带 usage，这是常规形态。
    """
    events = OpenAICompatibleProvider._chat_stream_events(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
    )
    assert [event.type for event in events] == ["usage"]
    assert OpenAICompatibleProvider._chat_stream_events({"choices": []}) == []


def test_chat_completion_events_does_not_flush_accumulator_on_non_finish():
    """非 finish 事件原样返回，不得提前冲刷累积中的工具调用。

    删除守卫后 text_delta 也会走进冲刷分支：未完成的工具调用被提前发出
    （或以 ``RuntimeError`` 中断流），且 ``completed`` 被污染使 finish 时
    无法再补齐。
    """
    accumulator = {"call_1": {"name": "f", "arguments": "{}", "id": "call_1"}}
    completed: set = set()
    event = StreamEvent(type="text_delta", text="hi")
    assert OpenAICompatibleProvider._chat_completion_events(event, accumulator, completed) == [
        event
    ]
    assert completed == set()
    assert list(accumulator) == ["call_1"]


def test_responses_completion_event_ignores_non_tool_events():
    """只有 ``tool_call_completed`` 且带载荷的事件才参与合并。

    删除守卫后文本增量事件也会进入合并分支，在 ``None`` 上取 ``get`` 抛裸
    ``AttributeError: 'NoneType' object has no attribute 'get'``。
    """
    assert (
        OpenAICompatibleProvider._responses_completion_event(
            StreamEvent(type="text_delta", text="hi"), {}
        )
        is None
    )
    assert (
        OpenAICompatibleProvider._responses_completion_event(
            StreamEvent(type="tool_call_completed", tool_call=None), {}
        )
        is None
    )


def test_extract_refusal_does_not_misreport_normal_text_as_refusal():
    """普通 ``output_text`` 内容不得被当成拒答文本返回。

    删除守卫后所有内容块都按「refusal 或 text」取值，正常回答会被当作拒答
    上抛给调用方。
    """
    normal = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "正常回答"}]}]
    }
    assert OpenAICompatibleProvider._extract_refusal(normal) is None
    refusal = {
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "我不行"}]}]
    }
    assert OpenAICompatibleProvider._extract_refusal(refusal) == "我不行"


def test_convert_responses_tool_choice_accepts_top_level_function_name():
    """Responses 的 function tool_choice 允许顶层 ``name``（无嵌套 ``function``）。

    删除守卫后这种形态落到「缺少 name」的 ``ValueError``，合法请求被拒绝。
    """
    choice = {"type": "function", "name": "get_weather"}
    assert OpenAICompatibleProvider._convert_responses_tool_choice(choice) == choice


def test_convert_responses_format_passes_through_already_normalized_format():
    """已是 ``{"format": ...}`` 形态的 response_format 原样返回。

    删除守卫后抛「仅支持 json_schema 或 json_object」的 ``ValueError``。
    """
    fmt = {"format": {"type": "json_object"}}
    assert OpenAICompatibleProvider._convert_responses_format(fmt) == fmt


def test_error_message_preserves_string_and_code_details():
    """错误消息提取：字符串错误与 ``code`` 字段都不能丢。

    删除 2649 的守卫后字符串错误退化成默认消息；删除 2658 的守卫后
    ``{"code": ...}`` 形态退化成默认消息——调用方看不到真实原因。
    """
    assert OpenAICompatibleProvider._error_message("rate limited", "默认消息") == "rate limited"
    assert OpenAICompatibleProvider._error_message({"code": "E42"}, "默认消息") == "E42"
    assert OpenAICompatibleProvider._error_message({}, "默认消息") == "默认消息"


def test_normalize_verified_protocols_none_returns_empty_set():
    """``server_verified_protocols`` 未配置（None）时返回空集。

    删除守卫后 None 落到「必须是字符串或协议序列」的 ``ValueError``，
    默认配置无法通过。
    """
    assert OpenAICompatibleProvider._normalize_verified_protocols(None) == frozenset()


def test_resource_capability_table_maps_responses_methods():
    """``retrieve_response`` 等 Responses 方法必须映射到 ``responses`` 能力。

    删除守卫后这些方法的能力映射变成 ``None``（未知方法），资源校验会在
    调用期把它们当成未挂载资源拒绝。
    """
    assert (
        OpenAICompatibleProvider._resource_capability_for_method("retrieve_response") == "responses"
    )
    assert (
        OpenAICompatibleProvider._resource_capability_for_method("cancel_response") == "responses"
    )


def test_response_tool_keys_skips_none_components():
    """键元组只包含实际存在的分量；全缺时返回空列表。

    删除守卫后返回 ``[("item_id", None), ("index", None), ("call_id", None)]``，
    合并逻辑会把这些「全 None」键当成真实键参与匹配。
    """
    assert OpenAICompatibleProvider._response_tool_keys(None) == []


# --- Anthropic -------------------------------------------------------------


def test_anthropic_convert_tool_choice_accepts_dict_auto_any_none():
    """``{"type": "auto"|"any"|"none"}`` 字典形态原样返回。

    删除守卫后这三种字典形态落到「必须是 auto、any、none、tool 或 function
    对象」的 ``ValueError``，只有字符串形态能通过。
    """
    for value in ("auto", "any", "none"):
        assert AnthropicProvider._convert_tool_choice({"type": value}) == {"type": value}


def test_anthropic_top_p_at_compat_threshold_is_omitted():
    """不高于 0.99 的兼容 top_p 取值返回 None（省略字段）而不是抛错。

    删除守卫后 0.99/1.0 也抛「不支持该 top_p」，模型声明支持兼容值却无法
    通过校验。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-6"
    assert provider._normalize_sampling_value("top_p", 0.99) is None
    assert provider._normalize_sampling_value("top_p", 1.0) is None
    with pytest.raises(ValueError, match="不支持该 top_p"):
        provider._normalize_sampling_value("top_p", 0.5)


def test_anthropic_user_profile_fields_without_user_returns_inputs_unchanged():
    """``user=None`` 时不追加 user-profiles beta 头。

    删除守卫后即使没有 user 也会把 ``user-profiles-2026-08-18`` 塞进
    betas/headers，向服务端声明了一个没有主体的 beta 资源。
    """
    assert AnthropicProvider._user_profile_fields(None, None, None, beta=True) == (None, None)
    headers, betas = AnthropicProvider._user_profile_fields("user_123", None, None, beta=True)
    assert headers is None
    assert betas == ["user-profiles-2026-08-18"]


def test_anthropic_stream_event_emits_usage_and_finish_events():
    """``message_delta`` 的 usage 与 ``message_stop`` 的 finish 都必须发出。

    删除 1411 的守卫后 usage 增量返回 None（usage 丢失）；删除 1414 的守卫
    后 message_stop 返回 None（流消费者永远等不到 finish）。
    """
    usage_event = AnthropicProvider._stream_event(
        {"type": "message_delta", "usage": {"output_tokens": 7}}, "resp_1"
    )
    assert usage_event is not None and usage_event.type == "usage"
    assert usage_event.usage == {"output_tokens": 7}
    finish_event = AnthropicProvider._stream_event({"type": "message_stop"}, "resp_1")
    assert finish_event is not None and finish_event.type == "finish"
    assert AnthropicProvider._stream_event({"type": "ping"}, "resp_1") is None


def test_anthropic_iter_invoke_stream_skips_non_text_events():
    """``ping``/``content_block_start`` 等非文本事件被跳过而不是解引用 None。

    删除守卫后 ``converted.type`` 在 ``None`` 上抛裸 ``AttributeError``——
    真实 Anthropic 流的每个响应都含 ping 与 content_block_* 事件。
    """

    class FakeStream:
        def __iter__(self):
            return iter(
                [
                    {"type": "ping"},
                    {"type": "content_block_start", "index": 0},
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "你好"},
                    },
                    {"type": "message_stop"},
                ]
            )

    assert list(AnthropicProvider._iter_invoke_stream_text(FakeStream())) == ["你好"]


# --- Google ----------------------------------------------------------------


def test_google_client_vertex_mode_reads_test_double_attribute():
    """客户端替身上直接暴露的 ``vertexai`` 属性要被读到。

    删除守卫后返回 None（回退到配置推断），而配置推断与替身实际行为可能
    不一致——本函数的用途正是优先读真实客户端。
    """

    class FakeClient:
        def __init__(self):
            self.vertexai = True

    class PlainClient:
        pass

    assert GoogleProvider._client_vertex_mode(FakeClient()) is True
    assert GoogleProvider._client_vertex_mode(PlainClient()) is None


def test_google_token_count_system_prompt_prefers_message_system():
    """token 统计的 system prompt 优先取消息里的 system 内容。

    删除守卫后回退到 ``request.system_prompt``，两者同时存在时取错来源，
    统计的 token 数与实际发送的请求不一致。
    """
    from src.providers.__base__.model_provider import CompletionRequest

    both = CompletionRequest(
        messages=[
            {"role": "system", "content": "来自消息"},
            {"role": "user", "content": "hi"},
        ],
        system_prompt="来自请求",
    )
    assert GoogleProvider._token_count_system_prompt(both) == "来自消息"
    only_request = CompletionRequest(
        messages=[{"role": "user", "content": "hi"}], system_prompt="只有请求"
    )
    assert GoogleProvider._token_count_system_prompt(only_request) == "只有请求"


def test_google_http_options_for_request_returns_configured_options():
    """模型级 ``options.http_options`` 配置优先于重建的请求级配置。

    删除守卫后配置被静默丢弃，``timeout``/``extra_body`` 等模型级 HTTP
    配置不生效。
    """
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-2.0-flash"
    provider._options = {"http_options": {"timeout": 12345}}
    assert provider._http_options_for_request() == {"timeout": 12345}


def test_google_contents_returns_prompt_directly_for_prompt_only_requests():
    """只给 prompt 的请求原样返回字符串，不经过消息归一化。

    删除守卫后返回值变成 ``Content`` 列表（类型变化），依赖字符串形态的
    调用方行为改变。
    """
    assert GoogleProvider._contents(None, "你好", None) == "你好"


def test_google_contents_skips_system_messages():
    """``role=system`` 的消息不进入 contents（system 走独立通道）。

    删除守卫后 system 内容被并入 user 部件，模型收到重复的系统提示。
    """
    contents = GoogleProvider._contents(
        [{"role": "system", "content": "系统提示"}, {"role": "user", "content": "hi"}],
        None,
        None,
    )
    assert len(contents) == 1
    assert [part.text for part in contents[0].parts] == ["hi"]


def test_google_candidate_refusal_reads_block_reason_message():
    """``block_reason_message`` 是拒答文本的来源之一。

    删除守卫后该字段被忽略，只带 block_reason_message 的候选返回 None。
    """

    class FakeCandidate:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    candidate = FakeCandidate(block_reason_message="内容被屏蔽")
    assert GoogleProvider._candidate_refusal(candidate) == "内容被屏蔽"


def test_google_stream_chunk_terminal_detection_covers_both_shapes():
    """顶层 ``finish_reason`` 与 ``prompt_feedback.block_reason`` 都算终止。

    删除 2670 后带顶层 finish_reason 的块不再终止；删除 2673 后
    prompt_feedback 拦截的块不再终止——流消费方会继续等待永远不会来的事件。
    """

    class FakeChunk:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    assert GoogleProvider._stream_chunk_has_terminal_event(FakeChunk(finish_reason="STOP")) is True
    assert (
        GoogleProvider._stream_chunk_has_terminal_event(
            FakeChunk(prompt_feedback=FakeChunk(block_reason="SAFETY"))
        )
        is True
    )
    assert GoogleProvider._stream_chunk_has_terminal_event(FakeChunk(text="hi")) is False


def test_google_close_without_client_is_noop():
    """未构造过客户端时 ``close``/``aclose`` 直接返回。

    删除守卫后 ``None.close()`` / ``None.aio`` 抛裸 ``AttributeError``；
    清理路径上的崩溃会打断退出流程。
    """
    provider = object.__new__(GoogleProvider)
    provider._client = None
    provider.close()
    asyncio.run(provider.aclose())


def test_google_partial_args_to_mapping_accepts_mapping():
    """``partial_args`` 已是 Mapping 时直接转 dict。

    删除守卫后 Mapping 落到「必须是列表」的 ``ValueError``，合法输入被拒。
    """
    assert GoogleProvider._partial_args_to_mapping({"a": 1}) == {"a": 1}


def test_google_coerce_debug_config_passes_through_none_and_non_mapping():
    """``debug_config`` 为 None 或非 Mapping 时原样返回。

    删除守卫后 ``for key in None`` / ``for key in 5`` 抛裸 ``TypeError``，
    未配置 debug 的默认路径直接崩溃。
    """
    assert GoogleProvider._coerce_debug_config(None) is None
    assert GoogleProvider._coerce_debug_config(5) == 5


# --- Volcengine（Ark）------------------------------------------------------


def test_ark_responses_format_and_tool_choice_default_none():
    """``response_format=None`` 返回 ``{}``、``tool_choice=None`` 返回 None。

    删除守卫后 None 分别落到「必须是对象」与「必须是 auto、none、required
    或工具对象」的 ``ValueError``，未配置这两个字段的请求全部被拒。
    """
    assert VolcengineProvider._convert_responses_format(None) == {}
    assert VolcengineProvider._convert_responses_tool_choice(None) is None


def test_ark_chat_stream_events_returns_usage_only_event_for_empty_choices():
    """Ark 流块的 ``choices=[]`` 形态同样不能索引 ``choices[0]``。

    删除守卫后抛裸 ``IndexError``。
    """
    events = VolcengineProvider._chat_stream_events(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
    )
    assert [event.type for event in events] == ["usage"]
    assert VolcengineProvider._chat_stream_events({"choices": []}) == []


# --- 基础层（__base__）-----------------------------------------------------


def test_responses_has_content_rejects_empty_sequence():
    """空序列不算「有内容」。

    删除守卫后空列表落到 ``return True``，空内容被当成有内容，调用方不再
    走回退取值路径。
    """
    assert _responses_has_content("") is False
    assert _responses_has_content([]) is False
    assert _responses_has_content("x") is True
    assert _responses_has_content([1]) is True


def test_responses_content_part_accepts_plain_string():
    """字符串部件转换为 ``input_text`` 块。

    删除守卫后字符串落到「必须是字符串或对象」的 ``ValueError``，最常见的
    部件形态被拒绝。
    """
    assert _responses_content_part("hi", "loc") == {"type": "input_text", "text": "hi"}
    with pytest.raises(ValueError, match="必须是字符串或对象"):
        _responses_content_part(5, "loc")


def test_content_to_text_joins_sequence_parts():
    """序列内容拼接为文本而不是丢弃。

    删除守卫后列表内容返回空串——多部件消息的文本静默消失。
    """
    assert content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert content_to_text(["a", "b"]) == "ab"
    assert content_to_text("hi") == "hi"


def test_responses_json_text_empty_handling():
    """``allow_empty=True`` 时空值/空串返回空串；缺省时 None 序列化为 ``{}``。

    删除 536 后 None+allow 变成 ``'{}'``；删除 540 后空串+allow 抛
    「必须是有效 JSON」——两种形态的语义都被改变。
    """
    assert _responses_json_text(None, "loc", allow_empty=True) == ""
    assert _responses_json_text("", "loc", allow_empty=True) == ""
    assert _responses_json_text(None, "loc") == "{}"
    with pytest.raises(ValueError, match="必须是有效 JSON"):
        _responses_json_text("", "loc")


def test_guard_provider_resource_method_is_idempotent():
    """已包装过的方法再次包装时原样返回。

    删除守卫后每次调用都套一层新包装：参数校验被执行多次，且
    ``_resource_guarded`` 的同一性判断失效。
    """

    def plain(a, b):
        return (a, b)

    guarded = _guard_provider_resource_method(plain)
    assert _guard_provider_resource_method(guarded) is guarded


# --- 收口批：轮询、文件内容、消息转换、异步流与流终止去重 -------------------


def test_openai_background_poll_returns_terminal_response_without_id():
    """终态响应没有 ``id`` 时原样返回（不进入轮询、也不需要 id）。

    删除守卫后终态响应落到「缺少 id，无法轮询」的 ``RuntimeError``——网关
    返回 ``status=completed`` 但省略 id 时，本来已完成的响应被当成错误。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    terminal = {"status": "completed"}
    assert provider._poll_background_response(terminal, {}) is terminal
    assert asyncio.run(provider._poll_background_response_async(terminal, {})) is terminal


def test_openai_retrieve_file_content_uses_sdk_method_when_available():
    """SDK 提供 ``files.retrieve_content`` 时走 SDK 资源通道。

    删除守卫后直接落到 ``file_content``（SDK 无该资源时抛
    ``NotImplementedError``），已挂载的 SDK 方法被绕过。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    calls = []

    def fake_call(method, args=(), kwargs=None, operation=""):
        calls.append(operation)
        return "sdk-result"

    async def fake_async_call(method, args=(), kwargs=None, operation=""):
        calls.append(operation)
        return "sdk-result"

    provider._call_sdk_resource = fake_call
    provider._call_async_sdk_resource = fake_async_call
    provider._get_client = lambda: _files_client()
    provider._get_aclient = _files_client
    assert provider.retrieve_file_content("file_1") == "sdk-result"
    assert asyncio.run(provider.async_retrieve_file_content("file_1")) == "sdk-result"
    assert calls == ["OpenAI 文件文本读取", "OpenAI 文件文本读取"]


def _files_client():
    import types

    return types.SimpleNamespace(
        files=types.SimpleNamespace(retrieve_content=lambda *args, **kwargs: "sdk")
    )


def test_anthropic_convert_messages_skips_system_messages():
    """``role=system`` 的消息不进入 messages（system 走独立参数）。

    删除守卫后系统提示被转换成 user 消息，与 ``system`` 参数里的提示重复，
    模型收到两份系统指令。
    """
    converted = AnthropicProvider._convert_messages(
        [{"role": "system", "content": "系统提示"}, {"role": "user", "content": "hi"}]
    )
    assert converted == [{"role": "user", "content": "hi"}]


def test_anthropic_async_iter_invoke_stream_skips_non_text_events():
    """异步流同样跳过 ``ping`` 等非文本事件。

    删除守卫后 ``converted.type`` 在 ``None`` 上抛裸 ``AttributeError``。
    """

    class AsyncStream:
        def __aiter__(self):
            async def events():
                yield {"type": "ping"}
                yield {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "你好"},
                }
                yield {"type": "message_stop"}

            return events()

    async def consume():
        return [text async for text in AnthropicProvider._aiter_invoke_stream_text(AsyncStream())]

    assert asyncio.run(consume()) == ["你好"]


def test_google_validate_http_options_allows_none_and_none_valued_sensitive_keys():
    """``http_options=None`` 通过校验；值为 None 的敏感键不算凭证。

    删除 502 的守卫后 None 落到「必须是对象表或 HttpOptions」的
    ``ValueError``；删除 590 的守卫后 ``{"api_key": None}`` 被判成携带凭证
    ——None 值没有可泄露的内容，占位形态应当放行。
    """
    GoogleProvider._validate_http_options(None)
    GoogleProvider._validate_http_options({"api_key": None})
    with pytest.raises(ValueError, match="不允许包含凭证"):
        GoogleProvider._validate_http_options({"api_key": "sk-1"})


def test_ark_normalize_verified_protocols_none_returns_empty_set():
    """Ark 的 ``server_verified_protocols`` 未配置时返回空集。

    删除守卫后 None 落到「必须是字符串或协议序列」的 ``ValueError``。
    """
    assert VolcengineProvider._normalize_verified_protocols(None) == frozenset()


def _responses_stream_client(events):
    import types

    class Responses:
        def create(self, **request):
            return iter(events)

    return types.SimpleNamespace(responses=Responses())


def test_openai_responses_stream_ignores_events_after_terminal():
    """``response.completed`` 之后的事件被跳过（流排空阶段不重复发 finish）。

    删除守卫后重复的终止事件会再次转换成 finish 事件——SDK 排空阶段重复
    送达的终止块让调用方收到两个 finish。
    """
    import types

    terminal = types.SimpleNamespace(
        type="response.completed", response=types.SimpleNamespace(status="completed")
    )
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_client = lambda: _responses_stream_client([terminal, terminal])
    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    assert [event.type for event in events].count("finish") == 1


def test_ark_responses_stream_ignores_events_after_terminal():
    """Ark 的 Responses 流同样跳过终止后的事件。"""
    import types

    terminal = types.SimpleNamespace(
        type="response.completed", response=types.SimpleNamespace(status="completed")
    )
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "m"
    provider._provider = "volcengine"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_client = lambda: _responses_stream_client([terminal, terminal])
    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    assert [event.type for event in events].count("finish") == 1


def test_anthropic_async_stream_deduplicates_finish_events():
    """异步流的 message_delta(finish) 与 message_stop 只产生一个 finish。

    删除守卫后两个事件各转换一次 finish，调用方收到重复的终止事件。
    """

    class AsyncStreamManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def __aiter__(self):
            async def events():
                yield SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason="end_turn"),
                    usage=SimpleNamespace(output_tokens=1),
                )
                yield SimpleNamespace(type="message_stop")

            return events()

    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    provider._get_aclient = lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: AsyncStreamManager())
    )

    async def consume():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    events = asyncio.run(consume())
    assert [event.type for event in events].count("finish") == 1


def test_google_stream_deduplicates_completed_tool_calls():
    """同一工具调用的完成事件只交付一次（重复块被跳过）。

    删除守卫后重复的 function_call 块再次交付 ``tool_call_completed``——
    同一轮工具调用被执行两次。
    """
    import types

    function_call = types.SimpleNamespace(
        id="call-1", name="lookup", args='{"id":1}', will_continue=False
    )
    chunk = types.SimpleNamespace(
        response_id="response-1",
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(function_call=function_call)]
                ),
                finish_reason=None,
            )
        ],
    )
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._options = {}
    terminal = types.SimpleNamespace(
        response_id="response-1",
        candidates=[
            types.SimpleNamespace(content=types.SimpleNamespace(parts=[]), finish_reason="STOP")
        ],
    )
    provider._get_client = lambda: types.SimpleNamespace(
        models=types.SimpleNamespace(
            generate_content_stream=lambda **_kwargs: iter([chunk, chunk, terminal])
        )
    )
    events = list(provider.stream_events(CompletionRequest(prompt="hi")))
    completed = [event for event in events if event.type == "tool_call_completed"]
    assert len(completed) == 1


def _async_responses_client(events):
    import types

    class Stream:
        def __init__(self):
            self._events = iter(events)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def close(self):
            return None

    class Responses:
        async def create(self, **_request):
            return Stream()

    return types.SimpleNamespace(responses=Responses())


def test_openai_async_responses_stream_ignores_events_after_terminal():
    """异步流的终止后事件同样被跳过。"""
    import types

    terminal = types.SimpleNamespace(
        type="response.completed", response=types.SimpleNamespace(status="completed")
    )
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_aclient = lambda: _async_responses_client([terminal, terminal])

    async def consume():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    events = asyncio.run(consume())
    assert [event.type for event in events].count("finish") == 1


def test_ark_async_responses_stream_ignores_events_after_terminal():
    """Ark 异步流的终止后事件同样被跳过。"""
    import types

    terminal = types.SimpleNamespace(
        type="response.completed", response=types.SimpleNamespace(status="completed")
    )
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "m"
    provider._provider = "volcengine"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_aclient = lambda: _async_responses_client([terminal, terminal])

    async def consume():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    events = asyncio.run(consume())
    assert [event.type for event in events].count("finish") == 1


def test_google_async_stream_deduplicates_completed_tool_calls():
    """Google 异步流同样只交付一次工具调用完成事件。"""
    import types

    function_call = types.SimpleNamespace(
        id="call-1", name="lookup", args='{"id":1}', will_continue=False
    )
    chunk = types.SimpleNamespace(
        response_id="response-1",
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(function_call=function_call)]
                ),
                finish_reason=None,
            )
        ],
    )
    terminal = types.SimpleNamespace(
        response_id="response-1",
        candidates=[
            types.SimpleNamespace(content=types.SimpleNamespace(parts=[]), finish_reason="STOP")
        ],
    )

    class AsyncStream:
        def __init__(self):
            self._events = iter([chunk, chunk, terminal])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._provider = "google"
    provider._options = {}
    aio = types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content_stream=lambda **_kwargs: AsyncStream())
    )
    provider._client = types.SimpleNamespace(aio=aio)
    provider._get_aclient = lambda: provider._client

    async def consume():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    events = asyncio.run(consume())
    completed = [event for event in events if event.type == "tool_call_completed"]
    assert len(completed) == 1
