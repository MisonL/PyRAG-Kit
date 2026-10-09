"""补齐「删守卫后全量测试仍绿」的非 raise 守卫覆盖缺口（Provider 批）。

每一条用例对应一条经变异扫描存活的守卫：把它的 ``if`` 整体替换成 ``pass``
后，3804 条测试全部通过——即整套测试没有任何断言能察觉该守卫消失。每条都
经真实入口探测确认删除后行为确实改变（崩溃、静默取值或消息内容），再按
**守卫存在时的行为**写成断言。详见 CHANGELOG 对应条目。
"""

import asyncio
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

from src.providers.__base__ import model_provider as model_provider_module
from src.providers.__base__.model_provider import (
    CompletionRequest,
    LargeLanguageModel,
    StreamEvent,
    _guard_provider_resource_method,
    _responses_content_part,
    _responses_has_content,
    _responses_json_text,
    content_to_text,
    normalize_chat_messages,
    normalize_embedding_vector,
    normalize_messages,
    normalize_responses_input,
    normalize_tool_arguments,
    retry_async_call,
    retry_sync_call,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.local_hash import LocalHashEmbeddingProvider
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
    # ``getattr(method, "_resource_guarded", False)`` 的默认值若反向成 ``True``，
    # 未包装的方法也会被判为「已安装」——守卫被静默跳过，凭证与能力校验全部失效。
    assert getattr(guarded, "_resource_guarded", None) is True


def test_guard_provider_resource_method_is_idempotent_for_coroutines():
    """**协程**方法同样必须打上 ``_resource_guarded`` 标记并保持幂等。

    ``_guard_provider_resource_method`` 对 ``inspect.iscoroutinefunction`` 走
    ``async_wrapper`` 分支，其 ``setattr(async_wrapper, "_resource_guarded",
    True)`` 与同步分支的赋值是**两处独立**的字面量。上面那条用例用的是同步
    函数，只覆盖同步分支；实测把异步分支的标记反向成 ``False`` 后全量套件仍
    绿——异步资源入口每次被重复包装都会多套一层，``_resource_guarded`` 的
    幂等判断对它们失效。
    """

    async def plain(a, b):
        return (a, b)

    guarded = _guard_provider_resource_method(plain)
    assert getattr(guarded, "_resource_guarded", None) is True
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


# --- except 元组收窄：声称捕获的第二类型必须有测试触发 ----------------------
#
# 来源：把 ``except (A, B, ...)`` 收窄成 ``except A``（只留第一个类型）后
# 全量测试仍绿的存活点。逐条用探针确认「命中被丢弃类型时行为确实改变」，
# 再按守卫存在时的行为写成断言。等价变异（被丢弃类型与第一个类型收场逐字
# 相同）与不可达类型不在此列。


def test_sanitize_resource_call_skips_valueerror_from_signature_property():
    """``__signature__`` 抛 ``ValueError`` 时同样跳过校验，而不是让异常冒泡。

    测试替身此前没有 ``__call__``，``inspect.signature`` 对它抛的是
    ``TypeError``——docstring 声称覆盖的 ``ValueError`` 分支从未被走到。
    收窄成 ``except TypeError`` 后本用例变红。
    """

    from src.providers.__base__.model_provider import _sanitize_resource_call

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

        def __call__(self):  # 有 __call__ 才会走到 ValueError 分支
            pass

    args, kwargs = _sanitize_resource_call(_NoSignature(), object(), (1,), {"model": "m"})
    assert args == (1,)
    assert kwargs == {"model": "m"}


def test_openai_validate_sdk_resource_call_skips_valueerror_from_signature_property():
    """``__signature__`` 抛 ``ValueError`` 时跳过校验而不是抛错。"""

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None

    class _NoSignature:
        @property
        def __signature__(self):
            raise ValueError("无签名")

        def __call__(self):
            pass

    provider._validate_sdk_resource_call(_NoSignature(), (), {}, "op")


def test_responses_json_text_reports_circular_reference_as_json_error():
    """循环引用的取值收敛成「必须可序列化为 JSON」，而不是裸 ``ValueError``。

    删除 ``except`` 里的 ``ValueError``（``json.dumps`` 对环抛
    ``ValueError: Circular reference detected``）后，消息变成 SDK 内部文本。
    """
    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="必须可序列化为 JSON"):
        _responses_json_text(cyclic, "参数")


@pytest.mark.parametrize("payload", ["中文", "é"])
def test_normalize_embedding_vector_reports_non_ascii_base64_readably(payload):
    """非 ASCII 字符串让 ``b64decode`` 抛 ``ValueError``，收敛成本模块的消息。

    收窄成 ``except binascii.Error`` 后，消息变成
    ``string argument should contain only ASCII characters``。
    """
    from src.providers.__base__.model_provider import normalize_embedding_vector

    with pytest.raises(ValueError, match="Embedding base64 内容无效"):
        normalize_embedding_vector(payload)


def test_anthropic_convert_content_reports_non_ascii_base64_readably():
    """Anthropic 图片 data URI 的非 ASCII base64 收敛成本模块的消息。"""
    with pytest.raises(ValueError, match="Anthropic 图片 data URI 不是有效 Base64"):
        AnthropicProvider._convert_content(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,中文"}}]
        )


def test_google_decode_data_uri_reports_non_ascii_base64_readably():
    """Google data URI 的非 ASCII base64 收敛成本模块的消息。"""
    with pytest.raises(ValueError, match="data URI 不是有效 Base64"):
        GoogleProvider._decode_data_uri("data:image/png;base64,中文", "图片")


def test_anthropic_snapshot_refusal_probe_tolerates_notimplementederror():
    """``current_message_snapshot`` 抛 ``NotImplementedError`` 时静默返回。

    收窄成 ``except AttributeError`` 后，未实现该属性的 SDK 响应对象会让流
    消费在第一次探测时崩溃。
    """

    class _Resp:
        @property
        def current_message_snapshot(self):
            raise NotImplementedError("SDK 未实现")

    AnthropicProvider._raise_for_stream_snapshot_refusal(_Resp())


def test_google_require_resource_converts_notimplementederror_property():
    """资源属性抛 ``NotImplementedError`` 时转成统一的能力错误。"""

    class _Client:
        @property
        def models(self):
            raise NotImplementedError("SDK 未实现")

    with pytest.raises(NotImplementedError, match="当前 google-genai SDK 未提供"):
        GoogleProvider._require_resource(_Client(), "models", "模型")


def test_delegated_resource_converts_notimplementederror_to_capability_error():
    """动态资源属性抛 ``NotImplementedError`` 时转成带 provider 名的能力错误。

    收窄成 ``except AttributeError`` 后，SDK 版本未实现该资源时
    ``NotImplementedError`` 冒泡，消息从「openai SDK 未提供 beta 资源」
    退化成 SDK 内部文本。
    """
    import src.providers.resources as resources_module

    facade_cls = next(
        cls
        for name in dir(resources_module)
        if isinstance(cls := getattr(resources_module, name), type)
        and getattr(cls, "_delegate_native", False)
        and "OpenAI" in name
    )

    class _BadClient:
        @property
        def beta(self):
            raise NotImplementedError("SDK 未实现")

    class _Provider:
        sdk_client = _BadClient()
        async_sdk_client = _BadClient()

        def __init__(self):
            self._provider = "openai"

    facade = object.__new__(facade_cls)
    object.__setattr__(facade, "provider", _Provider())
    with pytest.raises(NotImplementedError, match="SDK 未提供 beta 资源"):
        facade.beta  # noqa: B018 - 触发属性访问即目的


def test_facade_getattr_converts_notimplementederror_to_attributeerror():
    """动态节点属性抛 ``NotImplementedError`` 时收敛成 ``AttributeError``。

    收窄成 ``except AttributeError`` 后，``NotImplementedError`` 冒泡——
    动态资源探测（``getattr`` 探测新资源）把它当成「资源存在但失败」。
    """
    import src.providers.resources as resources_module

    facade_cls = next(
        cls
        for name in dir(resources_module)
        if isinstance(cls := getattr(resources_module, name), type)
        and getattr(cls, "_delegate_native", False)
        and "OpenAI" in name
    )

    class _BadClient:
        @property
        def brand_new_resource(self):
            raise NotImplementedError("SDK 未实现")

    class _Provider:
        sdk_client = _BadClient()
        async_sdk_client = _BadClient()

        def __init__(self):
            self._provider = "openai"

    facade = object.__new__(facade_cls)
    object.__setattr__(facade, "provider", _Provider())
    with pytest.raises(AttributeError, match="brand_new_resource"):
        facade.brand_new_resource  # noqa: B018 - 触发属性访问即目的


def test_google_require_provider_resource_converts_async_notimplementederror():
    """``client.aio`` 抛 ``NotImplementedError`` 时转成带资源名的能力错误。

    收窄成 ``except AttributeError`` 后，SDK 未提供异步通道时
    ``NotImplementedError`` 冒泡，消息退化成 SDK 内部文本。
    """
    provider = object.__new__(GoogleProvider)

    class _Client:
        @property
        def aio(self):
            raise NotImplementedError("SDK 未实现")

    provider._client = _Client()
    with pytest.raises(NotImplementedError, match="当前 google-genai SDK 未提供异步 Webhooks"):
        provider._require_provider_resource("async_webhooks")


def test_google_connect_live_converts_async_notimplementederror():
    """``connect_live`` 对 ``client.aio`` 抛 ``NotImplementedError`` 时转成能力错误。"""
    provider = object.__new__(GoogleProvider)

    class _Client:
        @property
        def aio(self):
            raise NotImplementedError("SDK 未实现")

    provider._client = _Client()
    with pytest.raises(NotImplementedError, match="当前 google-genai SDK 未提供异步 Live"):
        provider.connect_live(model="m")


def test_background_poll_times_out_when_remaining_is_exactly_zero(monkeypatch):
    """剩余时间恰好为 0 即超时（``<= 0`` 不是 ``< 0``），不得再发起 retrieve。

    时钟注入只替换**本模块**引用的 ``time``：``asyncio.run`` 自身也要读真实
    时钟，直接打补丁到全局 ``time.monotonic`` 会让它的调度序列错位。
    变异成 ``< 0`` 后 ``remaining == 0`` 这一轮不超时，会多调用一次
    ``retrieve``（断言 ``calls == []`` 即红）。
    """
    import time as _time
    from types import SimpleNamespace

    import src.providers.openai_compatible as oc

    sync_calls: list[str] = []
    async_calls: list[str] = []

    class _SyncResponses:
        def retrieve(self, response_id, **kwargs):
            sync_calls.append(response_id)
            return SimpleNamespace(status="queued", id=response_id)

    class _AsyncResponses:
        async def retrieve(self, response_id, **kwargs):
            async_calls.append(response_id)
            return SimpleNamespace(status="queued", id=response_id)

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = None
    provider._options = {"background_poll_interval": 0.5, "background_timeout": 10.0}
    provider._get_client = lambda: SimpleNamespace(responses=_SyncResponses())
    provider._get_aclient = lambda: SimpleNamespace(responses=_AsyncResponses())
    pending = {"status": "in_progress", "id": "resp-1"}

    # deadline 读表 100.0（deadline=110）；remaining 读表恰好 110.0 → == 0。
    clock = iter([100.0, 110.0, 121.0, 121.0])
    monkeypatch.setattr(
        oc, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=_time.sleep)
    )
    with pytest.raises(TimeoutError):
        provider._poll_background_response(pending, {})
    assert sync_calls == [], "remaining == 0 即超时，不得再发起 retrieve"

    clock = iter([100.0, 110.0, 121.0, 121.0])
    monkeypatch.setattr(
        oc, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=_time.sleep)
    )

    async def consume():
        await provider._poll_background_response_async(pending, {})

    with pytest.raises(TimeoutError):
        asyncio.run(consume())
    assert async_calls == [], "remaining == 0 即超时，不得再发起 retrieve"


# --- 布尔运算符互换：and/or 互换后仍绿的存活点 -----------------------------
#
# 来源：把守卫里的 and↔or 互换后全量仍绿的存活点。存活即说明「只满足第一个
# 条件」或「只满足第二个条件」的组合从未被测试覆盖。逐条用探针确认行为改变
# 后按守卫存在时的行为写断言。


def test_anthropic_output_format_requires_beta_for_tool_runner_only():
    """只开 ``for_tool_runner``（beta=False）时 output_format 仍被拒绝。

    ``beta and for_tool_runner`` 改 ``or`` 后，只开 tool_runner 的组合会被
    误放行，output_format 静默进请求。
    """
    provider = object.__new__(AnthropicProvider)
    provider._options = {}
    provider._model_name = "claude-sonnet-4-5"
    with pytest.raises(ValueError, match="请使用 parse 或 stream"):
        provider._build_message_params(
            prompt="hi", output_format=str, stream=False, beta=False, for_tool_runner=True
        )
    # 两个都开时照常放行。
    params = provider._build_message_params(
        prompt="hi", output_format=str, stream=False, beta=True, for_tool_runner=True
    )
    assert "output_format" in params


def test_anthropic_stream_ignores_empty_refusal_delta():
    """空 refusal 的 ``refusal_delta`` 事件不得报成拒答（``and`` 不是 ``or``）。

    ``content_block_start``（type=refusal、无文本）转换出 ``refusal=''`` 的
    事件；收窄成 ``or`` 后把它误报为拒答，用户看到空拒答消息。
    """

    def gen():
        yield {"type": "content_block_start", "index": 0, "content_block": {"type": "refusal"}}
        yield {"type": "message_stop"}

    class _Resp:
        def __iter__(self):
            return gen()

    assert list(AnthropicProvider._iter_invoke_stream_text(_Resp())) == []


def test_anthropic_async_stream_ignores_empty_refusal_delta():
    """异步分支同判据。"""

    async def agen():
        yield {"type": "content_block_start", "index": 0, "content_block": {"type": "refusal"}}
        yield {"type": "message_stop"}

    class _Resp:
        def __aiter__(self):
            return agen()

    async def collect():
        return [t async for t in AnthropicProvider._aiter_invoke_stream_text(_Resp())]

    assert asyncio.run(collect()) == []


def test_google_generation_config_accepts_only_max_output_tokens():
    """只设 ``max_output_tokens``（不设 ``max_tokens``）时不得报重复。

    ``and`` 改 ``or`` 后只设一个别名也会触发「重复」误报。
    """
    provider = object.__new__(GoogleProvider)
    provider._options = {"max_output_tokens": 100}
    provider._model_name = "gemini-2.0-flash"
    config = provider._build_generation_config(None, None, None)
    assert getattr(config, "max_output_tokens", None) == 100


def test_google_count_tokens_devapi_without_system_prompt_proceeds():
    """Developer API 且**未**提供 system prompt 时不得报「不支持 system_prompt」。

    ``vertexai is False and effective_system`` 改 ``or`` 后，只要走 Developer
    API 就误报——普通 token 计数直接失败。
    """
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import CompletionRequest

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._model_name = "gemini-2.0-flash"
    provider._effective_vertex_mode = lambda _client: False
    provider._token_count_system_prompt = lambda _request: None
    provider._validate_token_count_fields = lambda _request: None
    models = SimpleNamespace(count_tokens=lambda **_kw: SimpleNamespace(total_tokens=5))
    provider._get_client = lambda: SimpleNamespace(models=models)
    assert provider.count_tokens(CompletionRequest(prompt="hi")) == 5


def test_google_count_tokens_devapi_without_tools_proceeds():
    """Developer API 且未提供 tools 时不得报「不支持 tools」。"""
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import CompletionRequest

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._model_name = "gemini-2.0-flash"
    provider._effective_vertex_mode = lambda _client: False
    provider._token_count_system_prompt = lambda _request: None
    provider._validate_token_count_fields = lambda _request: None
    models = SimpleNamespace(count_tokens=lambda **_kw: SimpleNamespace(total_tokens=7))
    provider._get_client = lambda: SimpleNamespace(models=models)
    assert provider.count_tokens(CompletionRequest(prompt="hi")) == 7


def test_google_compute_tokens_vertex_without_system_prompt_proceeds():
    """Vertex 模式且未提供 system prompt 时不得报「不支持 system_prompt」。"""
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import CompletionRequest

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._model_name = "gemini-2.0-flash"
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: None
    provider._validate_compute_token_fields = lambda _request: None
    sentinel = SimpleNamespace(total_tokens=9)
    models = SimpleNamespace(compute_tokens=lambda **_kw: sentinel)
    provider._get_client = lambda: SimpleNamespace(models=models)
    assert provider.compute_tokens(CompletionRequest(prompt="hi")) is sentinel


def test_google_async_count_tokens_devapi_without_system_prompt_proceeds():
    """异步 count_tokens 同判据。"""
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import CompletionRequest

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._model_name = "gemini-2.0-flash"
    provider._effective_vertex_mode = lambda _client: False
    provider._token_count_system_prompt = lambda _request: None
    provider._validate_token_count_fields = lambda _request: None

    async def count_tokens(**_kw):
        return SimpleNamespace(total_tokens=11)

    models = SimpleNamespace(count_tokens=count_tokens)
    provider._get_client = lambda: SimpleNamespace(
        aio=SimpleNamespace(models=models), models=models
    )

    async def run():
        return await provider.async_count_tokens(CompletionRequest(prompt="hi"))

    assert asyncio.run(run()) == 11


def test_google_async_compute_tokens_vertex_without_system_prompt_proceeds():
    """异步 compute_tokens 同判据。"""
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import CompletionRequest

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._model_name = "gemini-2.0-flash"
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: None
    provider._validate_compute_token_fields = lambda _request: None

    sentinel = SimpleNamespace(total_tokens=13)

    async def compute_tokens(**_kw):
        return sentinel

    models = SimpleNamespace(compute_tokens=compute_tokens)
    provider._get_client = lambda: SimpleNamespace(
        aio=SimpleNamespace(models=models), models=models
    )

    async def run():
        return await provider.async_compute_tokens(CompletionRequest(prompt="hi"))

    assert asyncio.run(run()) is sentinel


def test_google_validate_tuning_reward_accepts_only_composite_config():
    """只传 ``composite_reward_config`` 时不得报「不能同时传入」。"""
    from types import SimpleNamespace

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._get_client = lambda: object()
    tunings = SimpleNamespace(validate_reward=lambda **_kw: "ok")
    provider._require_resource = lambda *_a, **_kw: tunings
    provider._resource_config_kwargs = lambda _cls, _config, _kwargs: {}
    assert (
        provider.validate_tuning_reward(
            "t1", sample_response={}, example={}, composite_reward_config={"b": 2}
        )
        == "ok"
    )


def test_google_async_validate_tuning_reward_accepts_only_composite_config():
    """异步分支同判据。"""
    from types import SimpleNamespace

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(aio=object())

    async def validate_reward(**_kw):
        return "ok-async"

    tunings = SimpleNamespace(validate_reward=validate_reward)
    provider._require_resource = lambda *_a, **_kw: tunings
    provider._resource_config_kwargs = lambda _cls, _config, _kwargs: {}

    async def run():
        return await provider.async_validate_tuning_reward(
            "t1", sample_response={}, example={}, composite_reward_config={"b": 2}
        )

    assert asyncio.run(run()) == "ok-async"


def test_openai_responses_format_rejects_empty_schema():
    """``json_schema.schema`` 为空对象时必须拒绝（``not json_schema`` 不可省）。

    ``or`` 改 ``and`` 后空 schema 被静默接受——客户端以为拿到了结构化输出
    约束，实际发的是空约束（与 Ark 副本修复前同一缺陷形态）。
    """
    with pytest.raises(ValueError, match="缺少 schema 字段"):
        OpenAICompatibleProvider._convert_responses_format(
            {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        )


def test_facade_getattr_allows_dunder_suffix_without_prefix():
    """只以 ``__`` 结尾（不以 ``__`` 开头）的属性名照常转发。

    ``startswith('__') and endswith('__')`` 改 ``or`` 后，``beta__`` 这类
    正常资源名被误判成 dunder 探测而拒绝。
    """
    import src.providers.resources as resources_module

    facade_cls = next(
        cls
        for name in dir(resources_module)
        if isinstance(cls := getattr(resources_module, name), type)
        and getattr(cls, "_delegate_native", False)
        and "OpenAI" in name
    )

    class _Client:
        beta__ = "ok"

    class _Provider:
        sdk_client = _Client()
        async_sdk_client = _Client()

        def __init__(self):
            self._provider = "openai"

    facade = object.__new__(facade_cls)
    object.__setattr__(facade, "provider", _Provider())
    assert facade.beta__ == "ok"


def test_volcengine_chat_request_accepts_only_max_completion_tokens():
    """只设 ``max_completion_tokens`` 时不得报「不能同时设置」。"""
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._model_name = "doubao"
    provider._protocol = "chat_completions"
    provider._request_options = lambda: {"max_completion_tokens": 20}
    request = provider._build_chat_request(prompt="hi", stream=False)
    assert request["max_completion_tokens"] == 20


def test_volcengine_chat_request_top_k_param_without_extra_body():
    """请求级 ``top_k``（模型 options 未设）时不得报重复。"""
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._model_name = "doubao"
    provider._protocol = "chat_completions"
    provider._request_options = dict
    request = provider._build_chat_request(prompt="hi", stream=False, top_k=5)
    assert request["extra_body"]["top_k"] == 5


def test_volcengine_chat_request_seed_param_without_extra_body():
    """请求级 ``seed``（模型 options 未设）时不得报重复。"""
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._model_name = "doubao"
    provider._protocol = "chat_completions"
    provider._request_options = dict
    request = provider._build_chat_request(prompt="hi", stream=False, seed=7)
    assert request["extra_body"]["seed"] == 7


def test_volcengine_responses_request_seed_param_without_extra_body():
    """Responses 请求级 ``seed`` 同判据。"""
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._model_name = "doubao"
    provider._protocol = "responses"
    provider._request_options = dict
    from src.providers.__base__.model_provider import CompletionRequest

    request = provider._build_responses_request(CompletionRequest(prompt="hi", seed=7))
    assert request["extra_body"]["seed"] == 7


def test_native_item_keeps_id_and_status_metadata():
    """原生内容块 item 的 ``id``/``status`` 元数据必须保留（``not in`` 不是 ``in``）。

    内容块转换只产出 SDK 输入联合里的字段，item 级元数据不在其中；
    ``key not in converted`` 换成 ``key in converted`` 后两个字段都被丢掉。
    """
    from src.providers.__base__.model_provider import _normalize_responses_native_item

    result = _normalize_responses_native_item(
        {"type": "input_text", "text": "hi", "id": "i1", "status": "s1"}, "loc", "openai"
    )

    assert result["id"] == "i1"
    assert result["status"] == "s1"
    assert result["text"] == "hi"


def test_responses_content_part_extracts_nested_image_file_id():
    """嵌套 ``image_url`` 对象里的 ``file_id`` 要提取出来（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 file_id 被丢掉：同时带 url 时静默少发一个字段，
    只带 file_id 时报「缺少 image_url 或 file_id」——用户按文档写嵌套形态
    却被拒。
    """
    from src.providers.__base__.model_provider import _responses_content_part

    both = _responses_content_part(
        {"type": "input_image", "image_url": {"url": "https://x/y.png", "file_id": "f1"}}, "loc"
    )
    assert both["image_url"] == "https://x/y.png"
    assert both["file_id"] == "f1"

    only_id = _responses_content_part(
        {"type": "input_image", "image_url": {"file_id": "f2"}}, "loc"
    )
    assert only_id["file_id"] == "f2"


# --- 否定变异（not / not in / is not）扫描的覆盖缺口 -------------------------
#
# 把 ``is not`` 换成 ``is``、``not in`` 换成 ``in``、去掉 ``not`` 后全量仍绿的
# 存活点。这批集中在 Responses 输入转换与 Anthropic 参数装配——两处都是
# 「可选字段有就给、没有就不给」的形态，判据反向时字段被静默丢掉。


def test_function_call_output_keeps_status_for_list_output():
    """列表 ``output`` 的 ``function_call_output`` 保留 ``status``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 ``status`` 被丢掉——工具结果的状态元数据静默消失。
    """
    from src.providers.__base__.model_provider import _responses_function_output_item

    item = _responses_function_output_item(
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "hi"}],
            "status": "completed",
        },
        "loc",
    )

    assert item["status"] == "completed"


def test_function_call_output_keeps_status_for_text_output():
    """纯文本 ``output`` 的分支同样保留 ``status``（两处判据逐字相同）。

    1062 行与 1051 行是两份独立拷贝，只测其中一份时另一份反向仍绿。
    """
    from src.providers.__base__.model_provider import _responses_function_output_item

    item = _responses_function_output_item(
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "hi",
            "status": "completed",
        },
        "loc",
    )

    assert item["status"] == "completed"


def test_responses_content_part_extracts_nested_image_url():
    """嵌套 ``image_url`` 对象里的 ``url`` 要提取出来（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后走 ``source.pop("image_url")`` 分支，url 被丢掉。
    """
    from src.providers.__base__.model_provider import _responses_content_part

    converted = _responses_content_part(
        {"type": "input_image", "image_url": {"url": "https://x/y.png"}}, "loc"
    )

    assert converted["image_url"] == "https://x/y.png"


def test_responses_content_part_extracts_nested_file_url():
    """``{"type": "file", "file": {"url": ...}}`` 的 ``url`` 映射成 ``file_url``（``not in`` 不是 ``in``）。

    ``not in`` 换成 ``in`` 后嵌套 url 不再提升到顶层，整条输入被
    「缺少 file_id、file_data 或 file_url」拒绝——用户按文档写嵌套形态却被拒。
    """
    from src.providers.__base__.model_provider import _responses_content_part

    converted = _responses_content_part({"type": "file", "file": {"url": "https://x/y.pdf"}}, "loc")

    assert converted == {"type": "input_file", "file_url": "https://x/y.pdf"}


def test_input_file_detail_allowlist_accepts_only_known_values():
    """``input_file`` 的 ``detail`` 白名单只放行 auto/low/high（``not in`` 不是 ``in``）。

    ``not in`` 换成 ``in`` 后判据反转：合法的 ``high`` 被拒、任意的 ``weird``
    被放行发给 SDK。
    """
    from src.providers.__base__.model_provider import _responses_content_part

    accepted = _responses_content_part(
        {"type": "input_file", "file_id": "f1", "detail": "high"}, "loc", provider="openai"
    )
    assert accepted["detail"] == "high"

    with pytest.raises(ValueError, match="detail 不受 Openai Responses SDK 支持"):
        _responses_content_part(
            {"type": "input_file", "file_id": "f1", "detail": "weird"}, "loc", provider="openai"
        )


def test_retry_async_stream_closes_iterator_on_early_exit():
    """提前退出时关闭被包装的异步迭代器（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 ``aclose()`` 永不调用——消费方 ``break`` 之后底层
    连接不会被释放。
    """
    from src.providers.__base__.model_provider import retry_async_stream

    closed: list[bool] = []

    def _make():
        async def _gen():
            try:
                yield 1
                yield 2
                yield 3
            finally:
                closed.append(True)

        return _gen()

    async def _consume_one():
        stream = retry_async_stream(_make)
        first = await stream.__anext__()
        await stream.aclose()
        # 快照必须在协程**内**取：``asyncio.run`` 的收尾会关闭仍在引用的
        # 异步生成器，把「未显式关闭」的差异掩盖成相同结果。
        return first, list(closed)

    first, closed_after_close = asyncio.run(_consume_one())
    assert first == 1
    assert closed_after_close == [True]


def test_merge_tool_call_fragment_keeps_existing_text_arguments():
    """已有文本 ``arguments`` 时不被子片段的非字符串取值覆盖（去掉 ``not`` 后反转）。

    ``not merged.get("arguments")`` 的 ``not`` 去掉后，非字符串分片会覆盖
    已累积的文本参数——流式拼接出的参数从中间被替换成对象。
    """
    from src.providers.__base__.model_provider import merge_tool_call_fragment

    accumulator: dict = {}
    merge_tool_call_fragment(accumulator, {"index": 0, "arguments": "abc"})
    _, merged = merge_tool_call_fragment(accumulator, {"index": 0, "arguments": {"obj": 1}})

    assert merged["arguments"] == "abc"


# --- Anthropic 可选参数：``is not None`` 判据 ---------------------------------


def _anthropic_provider():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._default_max_tokens = lambda: 1024
    return provider


def test_anthropic_message_params_keep_stop_sequences():
    """``stop`` 给出时装配 ``stop_sequences``（``is not None`` 不是 ``is None``）。"""
    params = _anthropic_provider()._build_message_params(prompt="hi", stop="END")

    assert params["stop_sequences"] == ["END"]


def test_anthropic_message_params_keep_tool_choice():
    """``tool_choice`` 给出时装配（``is not None`` 不是 ``is None``）。"""
    params = _anthropic_provider()._build_message_params(prompt="hi", tool_choice={"type": "auto"})

    assert params["tool_choice"] == {"type": "auto"}


def test_anthropic_message_params_keep_container_and_inference_geo():
    """``container``／``inference_geo`` 给出时装配（两个 ``is not None`` 各自要判）。"""
    params = _anthropic_provider()._build_message_params(
        prompt="hi", container="c1", inference_geo="us"
    )

    assert params["container"] == "c1"
    assert params["inference_geo"] == "us"


def test_anthropic_count_tokens_forwards_extra_query():
    """``count_tokens`` 转发 ``extra_query``（``is not None`` 不是 ``is None``）。"""
    captured: dict = {}

    def _client():
        def count_tokens(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(input_tokens=3)

        return SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens))

    provider = _anthropic_provider()
    provider._get_client = _client

    provider.count_tokens(CompletionRequest(prompt="hi", extra_query={"k": "v"}))

    assert captured["extra_query"] == {"k": "v"}


# --- Anthropic 四个 count_tokens 兄弟方法的可选字段透传 -----------------------
#
# 四个方法（``count_tokens``／``beta_count_tokens``／``async_count_tokens``／
# ``async_beta_count_tokens``）逐字重复同一组 ``if request.X is not None:
# params[...] = X``。否定变异扫描里这 40 余处**全部**存活：没有任何测试断言
# 这些可选字段被转发到 SDK 调用。表驱动覆盖，避免逐条重复。

_ANTHROPIC_TOKEN_COUNT_COMMON_CASES = [
    ({"tool_choice": {"type": "auto"}}, "tool_choice", {"type": "auto"}),
    ({"cache_control": {"type": "ephemeral"}}, "cache_control", {"type": "ephemeral"}),
    ({"user": "u1"}, "user_profile_id", "u1"),
    (
        {"output_config": {"format": {"type": "json_schema", "name": "x", "schema": {}}}},
        "output_config",
        {"format": {"type": "json_schema", "name": "x", "schema": {}}},
    ),
    (
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        "thinking",
        {"type": "enabled", "budget_tokens": 1024},
    ),
    ({"extra_body": {"custom": "v"}}, "extra_body", {"custom": "v"}),
    ({"extra_query": {"k": "v"}}, "extra_query", {"k": "v"}),
    ({"timeout": 5.0}, "timeout", 5.0),
    ({"extra_headers": {"X-Trace": "1"}}, "extra_headers", {"X-Trace": "1"}),
]

_ANTHROPIC_TOKEN_COUNT_BETA_CASES = [
    ({"context_management": {"edits": []}}, "context_management", {"edits": []}),
    ({"mcp_servers": [{"url": "https://x"}]}, "mcp_servers", [{"url": "https://x"}]),
    ({"speed": "fast"}, "speed", "fast"),
]


def _capturing_anthropic_provider(captured: dict, *, beta: bool):
    """构造一个把 ``count_tokens`` 参数记进 ``captured`` 的 Anthropic Provider。"""
    provider = _anthropic_provider()

    def _count_tokens(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(input_tokens=3)

    messages = SimpleNamespace(count_tokens=_count_tokens)
    client = (
        SimpleNamespace(beta=SimpleNamespace(messages=messages))
        if beta
        else SimpleNamespace(messages=messages)
    )
    provider._get_client = lambda: client
    provider._get_aclient = lambda: client
    return provider


@pytest.mark.parametrize(
    "method_name,beta",
    [
        ("count_tokens", False),
        ("beta_count_tokens", True),
        ("async_count_tokens", False),
        ("async_beta_count_tokens", True),
    ],
)
@pytest.mark.parametrize("request_kwargs,param_key,expected", _ANTHROPIC_TOKEN_COUNT_COMMON_CASES)
def test_anthropic_token_count_forwards_common_optional_field(
    method_name, beta, request_kwargs, param_key, expected
):
    """四个 count_tokens 方法都转发通用可选字段（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后字段被静默丢掉——调用方设置了 ``timeout``／``user``
    却不会生效。
    """
    captured: dict = {}
    provider = _capturing_anthropic_provider(captured, beta=beta)

    result = getattr(provider, method_name)(CompletionRequest(prompt="hi", **request_kwargs))
    if method_name.startswith("async_"):
        asyncio.run(result)

    assert captured[param_key] == expected


@pytest.mark.parametrize("method_name", ["beta_count_tokens", "async_beta_count_tokens"])
@pytest.mark.parametrize("request_kwargs,param_key,expected", _ANTHROPIC_TOKEN_COUNT_BETA_CASES)
def test_anthropic_beta_token_count_forwards_beta_only_field(
    method_name, request_kwargs, param_key, expected
):
    """Beta 专属字段只在 beta 方法上转发（``is not None`` 不是 ``is None``）。"""
    captured: dict = {}
    provider = _capturing_anthropic_provider(captured, beta=True)

    result = getattr(provider, method_name)(CompletionRequest(prompt="hi", **request_kwargs))
    if method_name.startswith("async_"):
        asyncio.run(result)

    assert captured[param_key] == expected


@pytest.mark.parametrize("method_name", ["beta_count_tokens", "async_beta_count_tokens"])
def test_anthropic_beta_token_count_merges_profile_beta(method_name):
    """``betas`` 与 user profile 头合并后转发（``is not None`` 不是 ``is None``）。"""
    captured: dict = {}
    provider = _capturing_anthropic_provider(captured, beta=True)

    result = getattr(provider, method_name)(
        CompletionRequest(prompt="hi", user="u1", betas=["custom-beta"])
    )
    if method_name.startswith("async_"):
        asyncio.run(result)

    assert captured["betas"] == ["custom-beta", "user-profiles-2026-08-18"]


# --- Google ``_build_generation_config`` 的可选字段透传 ----------------------
#
# 与 Anthropic 的 count_tokens 同一形态：一串 ``if X is not None: values[...] = X``
# 在否定变异扫描里全部存活——没有任何测试断言这些生成参数真的进了
# ``GenerateContentConfig``。

_GOOGLE_GENERATION_CONFIG_CASES = [
    ({"top_p": 0.9}, "top_p", 0.9),
    ({"top_k": 7}, "top_k", 7),
    ({"frequency_penalty": 0.5}, "frequency_penalty", 0.5),
    ({"presence_penalty": 0.4}, "presence_penalty", 0.4),
    ({"seed": 11}, "seed", 11),
    ({"modalities": ["TEXT"]}, "response_modalities", ["TEXT"]),
    ({"stop": "END"}, "stop_sequences", ["END"]),
]


def _google_provider(options=None):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-2.0-flash"
    provider._options = options or {}
    return provider


@pytest.mark.parametrize("kwargs,attribute,expected", _GOOGLE_GENERATION_CONFIG_CASES)
def test_google_generation_config_forwards_optional_field(kwargs, attribute, expected):
    """生成参数逐个进 ``GenerateContentConfig``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后参数被静默丢掉——调用方设置的 ``seed``／``stop``
    不会到达服务端。
    """
    config = _google_provider()._build_generation_config(None, None, None, **kwargs)

    assert getattr(config, attribute) == expected


def test_google_generation_config_forwards_tool_choice():
    """``tool_choice`` 转成 ``tool_config``（``is not None`` 不是 ``is None``）。"""
    config = _google_provider()._build_generation_config(
        None, None, None, tool_choice={"type": "auto"}
    )

    assert config.tool_config is not None
    assert config.tool_config.function_calling_config.mode.name == "AUTO"


def test_google_generation_config_forwards_options_thinking():
    """模型 options 里的 ``thinking`` 转成 ``thinking_config``（``is not None`` 不是 ``is None``）。"""
    provider = _google_provider({"thinking": {"thinking_budget": 1024}})

    config = provider._build_generation_config(None, None, None)

    assert config.thinking_config is not None


# --- Google / OpenAI 兼容：去掉 ``not`` 或反转 ``is not`` 的存活点 ------------


def test_google_tool_choice_accepts_non_empty_function_name():
    """``function.name`` 合法时照常转换（第二个 ``not`` 去掉后判据反转）。

    ``not name.strip()`` 的 ``not`` 去掉后，非空名字反而被拒——合法的
    function tool_choice 一律抛「缺少 function.name」。
    """
    from src.providers.google import GoogleProvider

    config = GoogleProvider._convert_tool_choice(
        {"type": "function", "function": {"name": "get_weather"}}
    )

    assert config.function_calling_config.mode.name == "ANY"
    assert config.function_calling_config.allowed_function_names == ["get_weather"]


def test_google_tool_result_resolves_name_from_call_id():
    """工具结果缺 ``name`` 时按 ``tool_call_id`` 反查（``not part_name`` 的 ``not`` 去掉后不再反查）。

    反查被跳过时报「缺少可关联的 tool_call_id/name」——而 id 恰好是调用方
    唯一给出的关联信息。
    """
    from src.providers.google import GoogleProvider

    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        None,
        None,
    )

    assert "lookup" in str(contents)


def test_google_candidate_refusal_falls_back_to_part_text():
    """``type=refusal`` 且 ``refusal`` 为空时退回 ``text``（``not value`` 的 ``not`` 去掉后不再退回）。

    退回被跳过时拒答被读成 ``None``，安全拦截表现为一次正常回复。
    """
    from src.providers.google import GoogleProvider

    candidate = {
        "content": {"parts": [{"type": "refusal", "text": "无法回答该问题"}]},
    }

    assert GoogleProvider._candidate_refusal(candidate) == "无法回答该问题"


def test_google_merge_tool_arguments_merges_into_existing_element():
    """列表参数在**已有**元素上继续递归合并（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后已有元素被新值整体替换——分片追加变成覆盖，前一
    分片里的字段丢失。
    """
    from src.providers.google import GoogleProvider

    merged = GoogleProvider._merge_tool_arguments([{"a": 1}], [{"b": 2}])

    assert merged == [{"a": 1, "b": 2}]


def test_chat_request_forwards_stop_parameter():
    """Chat Completions 转发 ``stop``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后停止序列被静默丢掉——调用方设置的停止词不生效。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "gpt-4o"
    provider._provider = "openai"
    provider._options = {}

    request = provider._build_chat_request(prompt="hi", stop=["END"])

    assert request["stop"] == ["END"]


def test_anthropic_beta_create_rejects_output_format():
    """``beta_create`` 拒绝 ``output_format``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后守卫不触发，落到 ``_build_message_params`` 的同名拒绝——
    两者消息前缀相同、**指引不同**（这里指向 ``beta_parse``，那里指向 ``parse``），
    断言必须带上指引部分。
    """
    provider = _anthropic_provider()

    with pytest.raises(ValueError, match="请使用 beta_parse 或 beta_stream"):
        provider.beta_create(CompletionRequest(prompt="hi", output_format={"type": "json_schema"}))


def test_anthropic_async_beta_create_rejects_output_format():
    """异步 ``async_beta_create`` 与同步版同判据（``is not None`` 不是 ``is None``）。"""
    provider = _anthropic_provider()

    async def _call():
        await provider.async_beta_create(
            CompletionRequest(prompt="hi", output_format={"type": "json_schema"})
        )

    with pytest.raises(ValueError, match="请使用 beta_parse 或 beta_stream"):
        asyncio.run(_call())


def test_anthropic_async_count_tokens_prefers_explicit_output_config():
    """显式 ``output_config`` 不被 ``response_format`` 覆盖（``not in`` 不是 ``in``）。

    ``not in`` 换成 ``in`` 后 fallback 无条件覆盖，用户显式给的 output_config
    被 ``response_format`` 派生的值替换。
    """
    captured: dict = {}
    provider = _capturing_anthropic_provider(captured, beta=False)

    asyncio.run(
        provider.async_count_tokens(
            CompletionRequest(
                prompt="hi",
                output_config={"explicit": True},
                response_format={"type": "json_object"},
            )
        )
    )

    assert captured["output_config"] == {"explicit": True}


def test_chat_request_forwards_remaining_optional_fields():
    """Chat Completions 转发 ``top_p``／``user``／``extra_query``（三个 ``is not None`` 各自要判）。

    与 ``stop`` 同形态：``is None`` 变异后字段被静默丢掉。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "gpt-4o"
    provider._provider = "openai"
    provider._options = {}

    request = provider._build_chat_request(
        prompt="hi", top_p=0.9, user="u1", extra_query={"k": "v"}
    )

    assert request["top_p"] == 0.9
    assert request["user"] == "u1"
    assert request["extra_query"] == {"k": "v"}


def test_jina_rerank_accepts_valid_results_list(monkeypatch):
    """合法响应（``results`` 是列表）照常解析（去掉 ``not`` 后反转成「是列表就报错」）。

    判据反转后每一次**成功**的 rerank 都被当成响应非法，检索直接失败。
    """
    import httpx

    from src.providers.jina import JinaProvider

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    provider = object.__new__(JinaProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "https://api.jina.ai/v1/rerank"
    provider._get_headers = dict

    indices, scores = provider.rerank("q", ["a"], 1)

    assert indices == [0]
    assert scores == [0.9]


def test_jina_arerank_accepts_valid_results_list(monkeypatch):
    """异步 ``arerank`` 与同步版同判据（去掉 ``not`` 后同样反转）。"""
    import httpx

    from src.providers.jina import JinaProvider

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
    provider = object.__new__(JinaProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "https://api.jina.ai/v1/rerank"
    provider._get_headers = dict

    indices, scores = asyncio.run(provider.arerank("q", ["a"], 1))

    assert indices == [0]
    assert scores == [0.9]


def test_google_safety_block_chunk_emits_finish_event():
    """安全拦截块要产出 ``finish`` 事件（``not any`` 的 ``not`` 去掉后不再产出）。

    去掉 ``not`` 后判据反转成「已有 finish 才追加」，而此处恰恰是**没有**
    finish 的路径——被拦截的流丢失终止事件，消费方看不到结束标记。
    """
    from src.providers.google import GoogleProvider

    chunk = SimpleNamespace(
        candidates=[],
        prompt_feedback=SimpleNamespace(block_reason="SAFETY", block_reason_message="m"),
    )

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [event.type for event in events] == ["refusal_delta", "finish"]
    assert events[-1].finish_reason == "SAFETY"


def test_google_vertex_count_tokens_keeps_system_instruction():
    """Vertex 模式下 ``count_tokens`` 带上 ``system_instruction``（``is not False`` 不是 ``is False``）。

    ``is False`` 变异后该分支只在 Developer API 下成立——而那条路径已在上面
    被拒，于是 system_instruction 永远进不了 config。
    """
    captured: dict = {}

    def _client():
        def count_tokens(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_tokens=3)

        return SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens))

    provider = _google_provider()
    provider._get_client = _client
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: "sys"

    provider.count_tokens(CompletionRequest(prompt="hi"))

    assert captured["config"].system_instruction == "sys"


def test_google_async_vertex_count_tokens_keeps_system_instruction():
    """异步 ``async_count_tokens`` 与同步版同判据（``is not False`` 不是 ``is False``）。"""
    captured: dict = {}

    def _client():
        def count_tokens(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_tokens=3)

        aio = SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens))
        return SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens), aio=aio)

    provider = _google_provider()
    provider._get_client = _client
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: "sys"

    asyncio.run(provider.async_count_tokens(CompletionRequest(prompt="hi")))

    assert captured["config"].system_instruction == "sys"


def test_chat_request_forwards_response_format_tool_choice_and_stream_options():
    """Chat Completions 转发 ``response_format``／``tool_choice``／``stream_options``。

    三者与 ``stop`` 同形态，``is None`` 变异后字段被静默丢掉。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "gpt-4o"
    provider._provider = "openai"
    provider._options = {}

    request = provider._build_chat_request(
        prompt="hi",
        response_format={"type": "json_object"},
        tool_choice="auto",
        stream_options={"include_usage": True},
    )

    assert request["response_format"] == {"type": "json_object"}
    assert request["tool_choice"] == "auto"
    assert request["stream_options"] == {"include_usage": True}


def test_chat_request_forwards_extra_headers():
    """Chat Completions 转发 ``extra_headers``（``is not None`` 不是 ``is None``）。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "gpt-4o"
    provider._provider = "openai"
    provider._options = {}

    request = provider._build_chat_request(prompt="hi", extra_headers={"X-Trace": "1"})

    assert request["extra_headers"] == {"X-Trace": "1"}


def test_chat_request_uses_configured_max_completion_tokens_when_request_omits_it():
    """请求未给长度时回落到配置的 ``max_completion_tokens``。

    ``is not None`` 变异后回落不生效，配置的输出长度被忽略。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "gpt-4o"
    provider._provider = "openai"
    provider._options = {"max_completion_tokens": 256}

    request = provider._build_chat_request(prompt="hi")

    assert request.get("max_completion_tokens") == 256


def test_verified_protocols_reject_blank_string_entry():
    """``server_verified_protocols`` 拒绝空白字符串条目（``not value.strip()`` 的 ``not`` 去掉后反转）。

    反转后合法协议反而被拒，而空白串被放行并归一化成无效协议。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"

    with pytest.raises(ValueError, match="必须是非空字符串"):
        provider._normalize_verified_protocols(["chat_completions", "  "])

    assert provider._normalize_verified_protocols(["chat_completions"]) == {"chat_completions"}


def test_response_tool_keys_defaults_to_event_when_item_missing():
    """``_response_tool_keys`` 未给 ``item`` 时退回 ``event``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后一律用 ``event``，显式传入的 ``item`` 被忽略——关联键
    取自错误的对象。
    """
    event = {"output_index": None}
    item = {"id": "item-1"}

    keys = OpenAICompatibleProvider._response_tool_keys(event, item)

    assert ("item_id", "item-1") in keys


def test_remember_response_tool_metadata_prefers_arguments_field():
    """``_remember_response_tool_metadata`` 从 ``arguments`` 取参数（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 ``arguments`` 与 ``input`` 都不再看，工具调用元数据
    丢失参数。
    """
    metadata: dict = {}
    event = {
        "item_id": "k1",
        "item": {"id": "k1", "type": "function_call", "name": "lookup", "arguments": '{"a": 1}'},
    }

    OpenAICompatibleProvider._remember_response_tool_metadata(event, metadata)

    assert metadata[("item_id", "k1")]["arguments"] == '{"a": 1}'


def test_vector_store_file_update_keeps_none_attributes():
    """``attributes=None`` 按 SDK 契约原样传 ``None``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 ``attributes`` 变成 ``None`` 而不是传入的对象，
    SDK 侧「清空属性」与「不传」的语义被混淆。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._aclient = SimpleNamespace(
        vector_stores=SimpleNamespace(files=SimpleNamespace(update=object()))
    )
    captured: dict = {}

    async def _call(_method, *, kwargs=None, **_rest):
        captured.update(kwargs or {})
        return "ok"

    provider._call_async_sdk_resource = _call

    asyncio.run(provider.async_update_vector_store_file("vs1", "f1", attributes={"k": "v"}))

    assert captured["attributes"] == {"k": "v"}


def test_google_generation_config_forwards_thinking_and_service_tier():
    """``thinking``／``service_tier`` 参数进 ``GenerateContentConfig``。

    与其它生成参数同形态，``is None`` 变异后字段被静默丢掉。
    """
    config = _google_provider()._build_generation_config(
        None, None, None, thinking={"thinking_budget": 512}, service_tier="flex"
    )

    assert config.thinking_config is not None
    assert config.service_tier == "flex"


def test_google_vertex_count_tokens_keeps_tools():
    """Vertex 模式下 ``count_tokens`` 带上 ``tools``（``is not False`` 不是 ``is False``）。"""
    captured: dict = {}

    def _client():
        def count_tokens(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_tokens=3)

        return SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens))

    provider = _google_provider()
    provider._get_client = _client
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: None

    provider.count_tokens(
        CompletionRequest(prompt="hi", tools=[{"type": "function", "function": {"name": "f"}}])
    )

    assert captured["config"].tools


def test_google_async_vertex_count_tokens_keeps_tools():
    """异步 ``async_count_tokens`` 与同步版同判据（``is not False`` 不是 ``is False``）。"""
    captured: dict = {}

    def _client():
        def count_tokens(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(total_tokens=3)

        aio = SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens))
        return SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens), aio=aio)

    provider = _google_provider()
    provider._get_client = _client
    provider._effective_vertex_mode = lambda _client: True
    provider._token_count_system_prompt = lambda _request: None

    asyncio.run(
        provider.async_count_tokens(
            CompletionRequest(prompt="hi", tools=[{"type": "function", "function": {"name": "f"}}])
        )
    )

    assert captured["config"].tools


def test_google_options_generation_count_rejects_negative_at_construction():
    """构造期校验 ``options`` 里的 ``max_output_tokens``／``candidate_count``。

    ``options[key] is not None`` 换成 ``is None`` 后构造期校验被跳过，
    非法取值要等首次提问才暴露。
    """
    from src.providers.google import GoogleProvider

    with pytest.raises(ValueError, match="max_output_tokens"):
        GoogleProvider("gemini-2.0-flash", {"max_output_tokens": -1})

    with pytest.raises(ValueError, match="candidate_count"):
        GoogleProvider("gemini-2.0-flash", {"candidate_count": 0})


def test_google_extract_refusal_prefers_block_reason_message():
    """``_extract_refusal`` 优先用 ``block_reason_message``（``not message.strip()`` 的 ``not`` 去掉后反转）。

    反转后合法消息被跳过，退回 ``message`` 字段——取到的是另一条文案。
    """
    from src.providers.google import GoogleProvider

    feedback = SimpleNamespace(
        block_reason="SAFETY", block_reason_message="被安全策略拦截", message="其它消息"
    )

    assert (
        GoogleProvider._extract_refusal(SimpleNamespace(candidates=[], prompt_feedback=feedback))
        == "被安全策略拦截"
    )


def test_google_merge_tool_call_fragment_keeps_non_string_arguments():
    """非字符串参数直接写入（``arguments is not None`` 不是 ``is None``）。

    ``is None`` 变异后非字符串参数（对象形态的 JSON 参数）被丢弃。
    """
    from src.providers.google import GoogleProvider

    accumulator: dict = {}
    _, merged = GoogleProvider._merge_tool_call(accumulator, {"index": 0, "arguments": {"a": 1}})

    assert merged["arguments"] == {"a": 1}


def test_google_stream_chunk_keeps_refusal_from_part_text():
    """流块里 ``type=refusal`` 且无 ``refusal`` 字段时退回 ``text``（去掉 ``not`` 后不再退回）。"""
    from src.providers.google import GoogleProvider

    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(type="refusal", text="拒答内容")])
            )
        ]
    )

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert any(event.type == "refusal_delta" for event in events)


def test_google_stream_chunk_emits_candidate_refusal_once():
    """candidate 拒答只发一次（``not refusal_emitted`` 的 ``not`` 去掉后判据反转）。

    去掉 ``not`` 后 ``refusal_emitted`` 为假时才发——首个候选之后不再发出，
    多候选响应丢掉后续拒答。
    """
    from src.providers.google import GoogleProvider

    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(refusal="第一条")])),
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(refusal="第二条")])),
        ]
    )

    events = GoogleProvider._stream_events_from_chunk(chunk)
    refusals = [event.refusal for event in events if event.type == "refusal_delta"]

    assert refusals[0] == "第一条"


def test_siliconflow_rerank_accepts_valid_results_list(monkeypatch):
    """SiliconFlow 合法响应照常解析（去掉 ``not`` 后反转成「是列表就报错」）。

    与 Jina 同一份近似副本、同一处判据，两处都要各自覆盖。
    """
    import httpx

    from src.providers.siliconflow_rerank import SiliconflowRerankProvider

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    provider = object.__new__(SiliconflowRerankProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "https://api.siliconflow.cn/v1/rerank"
    provider._get_headers = dict

    indices, scores = provider.rerank("q", ["a"], 1)

    assert indices == [0]
    assert scores == [0.9]


def test_siliconflow_arerank_accepts_valid_results_list(monkeypatch):
    """异步 ``arerank`` 与同步版同判据。"""
    import httpx

    from src.providers.siliconflow_rerank import SiliconflowRerankProvider

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _AsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
    provider = object.__new__(SiliconflowRerankProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "https://api.siliconflow.cn/v1/rerank"
    provider._get_headers = dict

    indices, scores = asyncio.run(provider.arerank("q", ["a"], 1))

    assert indices == [0]
    assert scores == [0.9]


def test_retry_async_stream_invokes_first_event_validator():
    """首个事件交给 ``first_event_validator``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后校验器永不调用——流建立阶段的错误事件不再被拦。
    """
    from src.providers.__base__.model_provider import retry_async_stream

    seen: list = []

    def _make():
        async def _gen():
            yield {"type": "error", "message": "boom"}

        return _gen()

    async def _consume():
        stream = retry_async_stream(_make, first_event_validator=seen.append)
        return await stream.__anext__()

    asyncio.run(_consume())

    assert seen == [{"type": "error", "message": "boom"}]


def test_retry_async_stream_closes_iterator_when_first_event_raises():
    """首事件校验抛错时关闭迭代器（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后建立阶段的失败不释放底层流。
    """
    from src.providers.__base__.model_provider import retry_async_stream

    closed: list[bool] = []

    def _make():
        async def _gen():
            try:
                yield {"type": "error"}
            finally:
                closed.append(True)

        return _gen()

    def _boom(_event):
        raise ValueError("bad first event")

    async def _consume():
        stream = retry_async_stream(_make, first_event_validator=_boom)
        try:
            await stream.__anext__()
        except ValueError:
            pass
        return list(closed)

    assert asyncio.run(_consume()) == [True]


def test_merge_tool_call_fragment_keeps_non_string_arguments_without_existing():
    """已有参数为空时非字符串分片直接写入（``not merged.get("arguments")`` 的 ``not`` 去掉后反转）。"""
    from src.providers.__base__.model_provider import merge_tool_call_fragment

    accumulator: dict = {}
    _, merged = merge_tool_call_fragment(accumulator, {"index": 0, "arguments": {"a": 1}})

    assert merged["arguments"] == {"a": 1}


# --- 其余否定变异存活点 -------------------------------------------------------


def test_resource_proxy_dir_lists_cached_children():
    """``_NativeResourceProxy.__dir__`` 列出已缓存的子节点（``not name.startswith("_")`` 的 ``not`` 去掉后反转）。

    反转后只列出下划线开头的名字——资源名一个都不出现，``dir(proxy)``
    与 ``__getattr__`` 的可达名字不一致。
    """
    from src.providers.resources import _NativeResourceProxy

    proxy = _NativeResourceProxy(SimpleNamespace(), None, "root")
    proxy._children["messages"] = object()

    names = dir(proxy)

    assert "messages" in names


def test_anthropic_message_params_keep_cache_control_thinking_and_service_tier():
    """``cache_control``／``thinking``／``service_tier`` 给出时装配（三个 ``is not None`` 各自要判）。

    与 ``stop``／``tool_choice`` 同形态：``is None`` 变异后字段被静默丢掉。
    """
    params = _anthropic_provider()._build_message_params(
        prompt="hi",
        cache_control={"type": "ephemeral"},
        thinking={"type": "enabled", "budget_tokens": 1024},
        service_tier="standard_only",
    )

    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert params["service_tier"] == "standard_only"


def _ark_responses_provider():
    provider = object.__new__(VolcengineProvider)
    provider._options = {}
    provider._model_name = "doubao"
    provider._protocol = "responses"
    provider._request_options = dict
    return provider


def test_ark_responses_rejects_unsupported_field():
    """Ark Responses 拒绝不支持的请求字段（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后字典推导只收集**值为 None** 的字段，非空的不支持字段
    不再进入拒绝列表——用户设置了 ``container`` 却被静默忽略。
    """
    provider = _ark_responses_provider()

    with pytest.raises(ValueError, match="container"):
        provider._build_responses_request(CompletionRequest(prompt="hi", container="c1"))


def test_ark_responses_forwards_explicit_temperature():
    """显式 ``temperature`` 进请求（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后显式温度被丢掉，模型回落到服务端默认值。
    """
    provider = _ark_responses_provider()

    request = provider._build_responses_request(CompletionRequest(prompt="hi", temperature=0.25))

    assert request["temperature"] == 0.25


def test_ark_responses_rejects_credentials_in_extra_body():
    """Ark Responses 校验 ``extra_body`` 不含凭证（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后校验被跳过，凭证原样进请求体。
    """
    provider = _ark_responses_provider()

    with pytest.raises(ValueError, match="extra_body"):
        provider._build_responses_request(
            CompletionRequest(prompt="hi", extra_body={"api_key": "sk-live-SECRET"})
        )


def test_anthropic_options_reject_unknown_key():
    """模型 ``options`` 里的未知键被拒（``key not in allowed_options`` 不是 ``key in``）。

    ``not in`` 换成 ``in`` 后判据反转：**未知键被放行**、已知键反被拒——
    用户写错一个字段名会被静默忽略。
    """
    provider = _anthropic_provider()
    provider._options = {"unknown_option": 1}

    with pytest.raises(ValueError, match="unknown_option"):
        provider._build_message_params(prompt="hi")


def test_google_async_validate_tuning_reward_forwards_single_config():
    """只传 ``single_reward_config`` 时转发它（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 kwargs 里什么都不放，SDK 收到空配置。
    """
    from types import SimpleNamespace

    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(aio=object())
    captured: dict = {}

    async def validate_reward(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider._require_resource = lambda *_a, **_kw: SimpleNamespace(validate_reward=validate_reward)
    provider._resource_config_kwargs = lambda _cls, _config, _kwargs: {}

    asyncio.run(
        provider.async_validate_tuning_reward(
            "t1", sample_response={}, example={}, single_reward_config={"a": 1}
        )
    )

    assert captured["single_reward_config"] == {"a": 1}


def test_anthropic_stream_error_message_reads_nested_non_string_values():
    """嵌套错误对象里**非字符串**取值也要继续下钻（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后非字符串取值被丢弃、不再入队，嵌套消息读不出来，
    用户只看到兜底文案「Anthropic Messages 流返回错误」。
    """
    provider = AnthropicProvider

    nested = SimpleNamespace(message=SimpleNamespace(message="深层消息"))

    assert provider._stream_error_message(nested) == "深层消息"


def test_google_merge_tool_call_keeps_non_string_arguments():
    """``_merge_tool_call`` 保留非字符串参数（``arguments is not None`` 不是 ``is None``）。

    ``is None`` 变异后对象／列表形态的参数被丢弃，``arguments`` 变成空字典。
    """
    from src.providers.google import GoogleProvider

    _, merged = GoogleProvider._merge_tool_call({}, {"index": 0, "arguments": [1, 2]})
    assert merged["arguments"] == [1, 2]

    _, merged_int = GoogleProvider._merge_tool_call({}, {"index": 0, "arguments": 42})
    assert merged_int["arguments"] == 42


# ── 否定变异扫描（face 11）补的覆盖缺口 ──
#
# 把守卫里的 ``not``／``not in``／``is not`` 变异后全量仍绿的 39 个存活点，
# 逐条用真实入口探针确认行为改变后补测试；27 处确认为真缺口，其余 12 处为
# 等价变异（如 ``reject_unsupported_kwargs`` 自身已按 ``is not None`` 过滤，
# 上游再过滤一层不改变收场）。


def test_anthropic_duplicate_sampling_field_with_extra_body():
    """``top_p`` 同时在 options 与 extra_body 时必须报错（两个 ``is not None`` 都要判）。

    ``sampling_options[key] is not None`` 变异成 ``is None`` 后，只有 options
    一侧非空时不再报重复，请求字段被 ``extra_body`` 静默顶掉。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-3-5-sonnet"
    provider._options = {"top_p": 0.9}
    provider._default_max_tokens = lambda: 1024

    with pytest.raises(ValueError, match="top_p 与 extra_body 重复"):
        provider._build_message_params(prompt="hi", extra_body={"top_p": 0.5})


def test_anthropic_beta_only_fields_rejected_without_beta():
    """非 beta 调用传 beta 专属字段必须拒绝（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后字典推导只收集**值为 None** 的 beta 字段，``mcp_servers``
    这类非空字段不再进入 ``ignored``——参数被静默丢弃。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-3-5-sonnet"
    provider._options = {}
    provider._default_max_tokens = lambda: 1024

    with pytest.raises(ValueError, match="mcp_servers"):
        provider._build_message_params(prompt="hi", mcp_servers=[{"url": "https://x"}])


def test_anthropic_tool_content_list_is_preserved():
    """``role=tool`` 的列表内容原样透传（``not isinstance`` 不是 ``isinstance``）。

    ``not`` 删除后列表被 ``content_to_text`` 压平成字符串，多块内容（文本＋
    图片）退化成一段拼接文本，结构化信息丢失。
    """
    converted = AnthropicProvider._convert_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {"type": "text", "text": "part-a"},
                    {"type": "text", "text": "part-b"},
                ],
            },
        ]
    )

    assert converted[1]["content"][0]["content"] == [
        {"type": "text", "text": "part-a"},
        {"type": "text", "text": "part-b"},
    ]


def test_anthropic_supports_sampling_option_tracks_model_generation():
    """旧模型支持采样、新模型不支持（``not deprecated or allow`` 的 ``not`` 不可删）。

    ``not`` 删除后判据反转：旧模型被判成不支持采样（调用方丢掉 temperature），
    新模型被判成支持（把已弃用字段发给服务端换回 400）。
    """
    legacy = object.__new__(AnthropicProvider)
    legacy._model_name = "claude-3-5-sonnet"
    legacy._options = {}
    modern = object.__new__(AnthropicProvider)
    modern._model_name = "claude-sonnet-4-6"
    modern._options = {}

    assert legacy.supports_sampling_option("temperature") is True
    assert modern.supports_sampling_option("temperature") is False


def test_anthropic_async_stream_emits_completed_tool_call():
    """``astream_events`` 在 finish 前交付 ``tool_call_completed``。

    ``key not in completed_tool_calls`` 变异成 ``in`` 后判据反转，工具调用
    永远不满足条件——调用方只看到 delta，拿不到完整调用。
    """
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-3-5-sonnet"
    provider._options = {}
    provider._api_key = "k"
    provider._normalized_client_options = {}
    provider._default_max_tokens = lambda: 1024

    class _Stream:
        def __init__(self, agen):
            self._agen = agen.__aiter__()

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self._agen.__anext__()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Messages:
        def stream(self, **_kwargs):
            async def gen():
                yield {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup"},
                }
                yield {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                }
                yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
                yield {"type": "message_stop"}

            return _Stream(gen())

    provider._aclient = type("A", (), {"messages": _Messages()})()

    async def run():
        return [
            event.type
            async for event in provider.astream_events(CompletionRequest(prompt="hi", stream=True))
        ]

    assert asyncio.run(run()) == [
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_completed",
        "finish",
    ]


def test_google_tool_result_part_resolves_name_from_call_id():
    """``tool_result`` 只有 ``tool_call_id`` 时从已登记的调用名解析（``not part_name`` 不可删）。

    ``not`` 删除后回退分支不再执行，``part_name`` 保持 ``None``，随后被判成
    「缺少可关联的 tool_call_id/name」——标准形态的并行工具结果全部报错。
    """
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "lookup_order", "arguments": "{}"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_call_id": "call-1", "response": {"ok": True}}
                ],
            },
        ],
        None,
        None,
    )

    response = contents[1].parts[0].function_response
    assert response.name == "lookup_order"
    assert response.id == "call-1"


def test_google_tool_result_accepts_valid_name():
    """显式给出合法 ``name`` 的 ``tool_result`` 必须通过（``not part_name.strip()`` 的 ``not`` 不可删）。

    ``not`` 删除后判据反转：非空名字反而触发「缺少可关联的 tool_call_id/name」，
    只有空名字才被接受——恰好与契约相反。
    """
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "name": "lookup",
                        "response": {"ok": True},
                    }
                ],
            },
        ],
        None,
        None,
    )

    assert contents[1].parts[0].function_response.name == "lookup"


def test_google_stream_refusal_part_uses_text_fallback():
    """``type=refusal`` 的块缺 ``refusal`` 字段时回退到 ``text``（``not part_refusal`` 不可删）。

    ``not`` 删除后回退分支不再执行，``part_refusal`` 保持 ``None``；同一块的
    ``text`` 又被当成普通文本交付——调用方同时收到 ``text_delta`` 与
    ``refusal_delta``，把拒答内容当正文显示。
    """
    part = SimpleNamespace(type="refusal", text="I cannot help with that.", refusal=None)
    chunk = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [(event.type, event.refusal) for event in events] == [
        ("refusal_delta", "I cannot help with that.")
    ]


def test_google_stream_candidate_refusal_emitted_once():
    """candidate 级拒答原因在未发过 ``refusal_delta`` 时交付（``not refusal_emitted`` 不可删）。

    ``not`` 删除后判据反转：只有**已经发过**拒答才补发，candidate 级的
    ``finish_message`` 永远不会被交付——拒答原因静默丢失。
    """
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(type="text", text="hi", refusal=None)]),
        finish_message="blocked by safety",
    )
    chunk = SimpleNamespace(candidates=[candidate])

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [event.type for event in events] == ["text_delta", "refusal_delta"]
    assert events[1].refusal == "blocked by safety"


def test_google_async_validate_tuning_reward_forwards_composite_config():
    """只传 ``composite_reward_config`` 时转发它（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 kwargs 里什么都不放，SDK 收到空配置。
    """
    provider = object.__new__(GoogleProvider)
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(aio=object())
    captured: dict = {}

    async def validate_reward(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider._require_resource = lambda *_a, **_kw: SimpleNamespace(validate_reward=validate_reward)
    provider._resource_config_kwargs = lambda _cls, _config, _kwargs: {}

    asyncio.run(
        provider.async_validate_tuning_reward(
            "t1", sample_response={}, example={}, composite_reward_config={"b": 2}
        )
    )

    assert captured["composite_reward_config"] == {"b": 2}


def test_openai_chat_request_forwards_metadata():
    """``metadata`` 给出时进请求（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后 metadata 被静默丢弃。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    request = provider._build_chat_request(prompt="hi", metadata={"trace": "t1"})

    assert request["metadata"] == {"trace": "t1"}


def test_openai_responses_configured_top_logprobs_used_as_fallback():
    """请求未给 ``top_logprobs`` 时回退到 options（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后回退取值恒为 ``None``，配置里的 top_logprobs 永远不生效。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"top_logprobs": 3}

    request = provider._build_responses_request(prompt="hi")

    assert request["top_logprobs"] == 3


def test_openai_responses_official_endpoint_rejects_configured_extra_body_stop():
    """官方端点下 options.extra_body 的 ``stop`` 必须拒绝（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后字典推导只收集**值为 None** 的项，非空的 ``stop``
    不再进入拒绝列表——官方端点会把它当未知字段或静默忽略。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"extra_body": {"stop": "END"}}

    with pytest.raises(ValueError, match="stop"):
        provider._build_responses_request(prompt="hi")


def test_openai_response_retrieve_kwargs_drops_none_values():
    """``_response_retrieve_kwargs`` 过滤 ``None`` 值（``params[key] is not None`` 不是 ``is None``）。

    ``is None`` 变异后只保留**值为 None** 的项，真正的取值被丢掉。
    """
    out = OpenAICompatibleProvider._response_retrieve_kwargs({"timeout": None, "include": ["a"]})

    assert out == {"include": ["a"]}


def test_openai_responses_sync_emits_completed_before_later_events():
    """``output_item.done`` 的完整调用**立即**交付，不推迟到 finish。

    ``key not in completed_tool_calls`` 变异成 ``in`` 后判据反转，完成事件被
    推迟到 ``finish`` 才补发——调用方在中间收到文本增量，顺序错乱。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )
            return iter(
                [
                    SimpleNamespace(type="response.output_item.done", item=item, output_index=0),
                    SimpleNamespace(type="response.output_text.delta", delta="后续文本"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))

    assert [event.type for event in events] == ["tool_call_completed", "text_delta", "finish"]


def test_openai_responses_async_emits_completed_before_later_events():
    """异步分支同判据。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )

            async def gen():
                yield SimpleNamespace(type="response.output_item.done", item=item, output_index=0)
                yield SimpleNamespace(type="response.output_text.delta", delta="后续文本")
                yield SimpleNamespace(
                    type="response.completed", response=SimpleNamespace(status="completed")
                )

            return gen()

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def run():
        return [
            event.type async for event in provider.astream_events(CompletionRequest(prompt="hi"))
        ]

    assert asyncio.run(run()) == ["tool_call_completed", "text_delta", "finish"]


def test_openai_async_compact_responses_forwards_only_given_fields():
    """``async_compact_responses`` 只转发显式给出的字段（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后判据反转，只转发**值为 None** 的字段——SDK 收到一堆
    空参数，真正的 ``input`` 与 ``timeout`` 全被丢掉。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    captured: dict = {}

    class Responses:
        async def compact(self, **kwargs):
            captured.update(kwargs)
            return "ok"

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())
    provider._call_async_sdk_resource = lambda fn, kwargs=None, operation="": fn(**(kwargs or {}))

    asyncio.run(provider.async_compact_responses(input="hi", timeout=9))

    assert set(captured) == {"model", "input", "timeout"}


def test_openai_async_update_vector_store_file_drops_none_extras():
    """``async_update_vector_store_file`` 过滤未给出的扩展参数（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后只转发**值为 None** 的扩展，SDK 收到一堆 ``extra_*``
    空参数，而显式给出的 ``timeout`` 被丢掉。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    captured: dict = {}

    class _Files:
        async def update(self, *args, **kwargs):
            captured.update(kwargs)
            return "ok"

    provider._get_aclient = lambda: SimpleNamespace(vector_stores=SimpleNamespace(files=_Files()))
    provider._call_async_sdk_resource = lambda fn, args=(), kwargs=None, operation="": fn(
        *args, **(kwargs or {})
    )

    asyncio.run(
        provider.async_update_vector_store_file(
            "vs1", "f1", attributes={"a": 1}, extra_headers=None, timeout=9
        )
    )

    assert set(captured) == {"vector_store_id", "attributes", "timeout"}


def test_ark_responses_rejects_unsupported_request_field():
    """拒绝不支持的 Ark Responses 请求字段（``getattr(...) is not None`` 不是 ``is None``）。

    ``is None`` 变异后字典推导只收集**取值恰为 None** 的字段（即未设置的），
    非空的 ``fallbacks`` 不再进入拒绝列表——用户设置了却被静默忽略。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "m"
    provider._options = {}

    with pytest.raises(ValueError, match="fallbacks"):
        provider._build_responses_request(CompletionRequest(prompt="hi", fallbacks=[{"a": 1}]))


def test_ark_responses_native_kwargs_validate_extra_body_secrets():
    """原生入口的 ``extra_body`` 必须过凭证校验（``params["extra_body"] is not None`` 不是 ``is None``）。

    ``is None`` 变异后校验分支不再执行，携带 ``api_key`` 的 extra_body
    被原样放行——凭证会随请求发往远端。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "m"
    provider._options = {}

    with pytest.raises(ValueError, match="api_key"):
        provider._validate_native_response_kwargs(
            {"model": "m", "input": "hi", "extra_body": {"api_key": "sk-abcdef1234567890"}}
        )


def test_ark_responses_sync_emits_completed_before_later_events():
    """Ark 同步流同判据：``output_item.done`` 立即交付。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )
            return iter(
                [
                    SimpleNamespace(type="response.output_item.done", item=item, output_index=0),
                    SimpleNamespace(type="response.output_text.delta", delta="后续文本"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))

    assert [event.type for event in events] == ["tool_call_completed", "text_delta", "finish"]


def test_ark_responses_async_emits_completed_before_later_events():
    """Ark 异步流同判据。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )

            async def gen():
                yield SimpleNamespace(type="response.output_item.done", item=item, output_index=0)
                yield SimpleNamespace(type="response.output_text.delta", delta="后续文本")
                yield SimpleNamespace(
                    type="response.completed", response=SimpleNamespace(status="completed")
                )

            return gen()

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def run():
        return [
            event.type async for event in provider.astream_events(CompletionRequest(prompt="hi"))
        ]

    assert asyncio.run(run()) == ["tool_call_completed", "text_delta", "finish"]


def test_ark_responses_async_deduplicates_tool_call_delivered_by_finish():
    """finish 分支补发 pending 时不得重复已交付的调用（``key not in completed`` 不是 ``in``）。

    ``in`` 变异后 finish 分支对**已交付**的调用再发一次，调用方收到两个
    相同 id 的 ``tool_call_completed``。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )

            async def gen():
                yield SimpleNamespace(type="response.output_item.done", item=item, output_index=0)
                yield SimpleNamespace(
                    type="response.function_call_arguments.done", item=item, output_index=0
                )
                yield SimpleNamespace(
                    type="response.completed", response=SimpleNamespace(status="completed")
                )

            return gen()

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def run():
        return [
            event.type async for event in provider.astream_events(CompletionRequest(prompt="hi"))
        ]

    assert asyncio.run(run()).count("tool_call_completed") == 1


def test_ark_responses_sync_deduplicates_tool_call_delivered_by_finish():
    """Ark 同步流同判据：finish 分支不得重复交付已发过的调用。"""
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call", call_id="call-1", name="lookup", arguments='{"id":1}'
            )
            return iter(
                [
                    SimpleNamespace(type="response.output_item.done", item=item, output_index=0),
                    SimpleNamespace(
                        type="response.function_call_arguments.done", item=item, output_index=0
                    ),
                    SimpleNamespace(
                        type="response.completed", response=SimpleNamespace(status="completed")
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))

    assert [event.type for event in events].count("tool_call_completed") == 1


def test_openai_compatible_merge_resource_kwargs_keeps_none_by_default():
    """默认（``omit_none=False``）必须原样保留 ``None`` 取值。

    ``not omit_none`` 的 ``not`` 被删后默认路径也进入过滤：调用方显式给出的
    ``None`` 字段被静默丢弃，SDK 收到的是「未提供」而不是「显式置空」。
    """
    out = OpenAICompatibleProvider._merge_resource_kwargs({"a": 1, "b": None}, None, "T")

    assert out == {"a": 1, "b": None}


def test_openai_compatible_merge_resource_kwargs_omits_none_when_requested():
    """``omit_none=True`` 时过滤 ``None``（``value is not None`` 不是 ``is None``）。

    ``is None`` 变异后判据反转：只保留值为 ``None`` 的字段，真正的取值全部
    被丢掉。
    """
    out = OpenAICompatibleProvider._merge_resource_kwargs(
        {"a": 1, "b": None}, None, "T", omit_none=True
    )

    assert out == {"a": 1}


def test_openai_responses_none_max_output_tokens_not_forwarded():
    """值为 ``None`` 的 ``max_output_tokens`` 不得进入请求体。

    原先只在取值非 ``None`` 时才把它移出 ``configured_options``；显式配置成
    ``None`` 时键留在 options 里、经 ``_merge_options`` 原样并入请求，SDK 收到
    ``max_output_tokens: null``。改为无条件移出：真实取值经 ``configured_limits``
    重新写回，``None`` 不落进请求。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"max_output_tokens": None}

    request = provider._build_responses_request(prompt="hi")

    assert "max_output_tokens" not in request


# --- face 12（== ↔ != 互换扫描）存活位点 -------------------------------------
#
# 358 个 eqneq 候选全量扫描后 27 个存活；下列每条经真实入口探针确认 base 与
# 变异体收场逐字不同，按守卫存在时的行为写断言。


def test_normalize_messages_inserts_system_only_when_history_lacks_it():
    """消息没有 system 角色时补插 system_prompt（``not any(...)`` 不是 ``any(...)``）。

    ``==``→``!=`` 变异后判据反转：有 system 时重复插入、没有时反而丢弃
    system_prompt——实测变异体输出只剩 ``[{'role': 'user', ...}]``。
    """
    from src.providers.__base__.model_provider import normalize_messages

    out = normalize_messages(None, "你在做客服。", [{"role": "user", "content": "hi"}])

    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "你在做客服。"


def test_responses_content_part_original_detail_only_for_openai_variant():
    """``detail="original`` 只对 openai 变体合法（``variant == "openai"``）。

    ``==``→``!=`` 变异后判据反转：openai 被拒、ark 反而接受 ``original``——
    两侧的可选值集合互换，与 Responses 契约（原图只在官方端点存在）相反。
    """
    from src.providers.__base__.model_provider import _responses_content_part

    part = {"type": "input_image", "image_url": "https://x/y.png", "detail": "original"}

    openai_out = _responses_content_part(dict(part), "L", provider="openai")
    assert openai_out.get("detail") == "original"
    with pytest.raises(ValueError, match="L.detail 必须"):
        _responses_content_part(dict(part), "L", provider="ark")


def test_capability_report_verified_protocols_cleared_for_non_llm_role():
    """非 llm 角色不得继承注册表里的 ``server_verified_protocols``（``== "llm"``）。

    ``==``→``!=`` 变异后判据反转：注册的验收证据从 llm 报告里消失、却出现在
    embedding 报告里——与 `protocol_status` 对齐的注释约定（协议只对 llm 适用）
    相反。注册表项默认是空集，测试里临时写入非空值再还原。
    """
    from src.providers.factory import ModelProviderFactory

    info = ModelProviderFactory._provider_map["openai"]
    original = info.get("server_verified_protocols")
    info["server_verified_protocols"] = {"responses"}
    try:
        llm = ModelProviderFactory.capability_report("openai", role="llm")
        embedding = ModelProviderFactory.capability_report("openai", role="embedding")
    finally:
        info["server_verified_protocols"] = original

    assert llm["server_verified_protocols"] == ["responses"]
    assert llm["protocol_status"]["responses"]["server_verified"] is True
    assert embedding["server_verified_protocols"] == []
    assert embedding["protocol_status"] == {}


def test_google_as_bool_coerces_numeric_zero_and_nonzero():
    """数值强制转换 ``value != 0``：0→False，非 0→True。

    ``!=``→``==`` 变异后判据反转：``_as_bool(0)`` 为 True、``_as_bool(1)`` 为
    False——配置里的数值布尔被静默取反。
    """
    from src.providers.google import GoogleProvider

    assert GoogleProvider._as_bool(0) is False
    assert GoogleProvider._as_bool(1) is True
    assert GoogleProvider._as_bool(1.5) is True
    assert GoogleProvider._as_bool(-2.0) is True
    assert GoogleProvider._as_bool(True) is True
    assert GoogleProvider._as_bool(False) is False


def test_google_stream_prompt_feedback_emits_refusal_alongside_content():
    """同一 chunk 既有内容又带安全拦截时，refusal_delta 仍要发出一次。

    ``==``→``!=`` 变异后去重判据反转：已有 refusal 才追加——常见形态（内容与
    prompt_feedback 同 chunk）下安全拦截被静默丢弃，上层看不到拒答原因。
    """
    part = SimpleNamespace(type="text", text="部分内容", refusal=None)
    cand = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=None)
    feedback = SimpleNamespace(block_reason="SAFETY", block_reason_message=None, message=None)
    chunk = SimpleNamespace(candidates=[cand], prompt_feedback=feedback, usage_metadata=None)

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [(e.type, e.refusal) for e in events] == [
        ("text_delta", None),
        ("refusal_delta", "Google 提示被安全策略拦截: SAFETY"),
        ("finish", None),
    ]


def test_google_stream_top_level_finish_kept_when_chunk_has_content():
    """内容与顶层 finish_reason 同 chunk 时 finish 仍要发出（``==`` 去重判据）。

    ``==``→``!=`` 变异后反转：已有 finish 才追加——实测变异体只剩
    ``[('text_delta', None)]``，finish_reason 丢失，上层永远等不到终止事件。
    """
    part = SimpleNamespace(type="text", text="回答", refusal=None)
    cand = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=None)
    chunk = SimpleNamespace(
        candidates=[cand], prompt_feedback=None, usage_metadata=None, finish_reason="STOP"
    )

    events = GoogleProvider._stream_events_from_chunk(chunk)

    assert [(e.type, e.finish_reason) for e in events] == [("text_delta", None), ("finish", "STOP")]


def test_local_hash_embedding_matches_reference_sign_pattern():
    """哈希符号位 ``digest[0] % 2 == 0`` / ``digest[15] % 2 == 0`` 决定落点符号。

    任一 ``==``→``!=`` 变异都会改变既有文本的嵌入值——快照指纹随之漂移，
    与「离线可复现」的契约矛盾。期望值由 base 源码实算得出。
    """
    provider = LocalHashEmbeddingProvider("local-hash-8")

    vec = provider._embed_text("hello world")

    assert [round(v, 4) for v in vec] == [0.0, 0.0, 0.7071, -0.7071, 0.0, 0.0, 0.0, 0.0]


def test_openai_chat_invoke_stream_yields_text_delta():
    """同步 invoke 流式路径按 ``event.type == "text_delta"`` 交付文本。

    ``==``→``!=`` 变异后文本全部被丢弃——实测变异体 ``list(...)`` 为 ``[]``，
    聊天界面收到空回复；finish 守卫（独立判据）仍绿，所以只有显式断言文本
    才能抓住它。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="答", tool_calls=None),
                                finish_reason=None,
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None, tool_calls=None),
                                finish_reason="stop",
                            )
                        ]
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    assert list(provider.invoke("hi", stream=True)) == ["答"]


def test_openai_chat_ainvoke_stream_yields_text_delta():
    """异步 invoke 流式路径同判据（``==``→``!=`` 变异后返回空列表）。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        async def create(self, **_request):
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="答", tool_calls=None),
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                ),
            ]

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hi", stream=True)]

    assert asyncio.run(consume()) == ["答"]


def test_openai_responses_batch_guard_covers_explicit_endpoint():
    """显式 ``endpoint="/v1/responses"`` 同样要过未验证服务端的门禁。

    判据是 ``endpoint.rstrip("/") == "/v1/responses"``；``==``→``!=`` 变异后
    网关（未验证 Responses）带着显式端点直接调到 SDK ``batches.create``——
    实测变异体返回 ``batch``，远端 404/405 的保护被绕开。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    provider._base_url = "https://gateway.example/v1"
    provider._server_verified_protocols = frozenset()

    with pytest.raises(ValueError, match="未验证 Responses"):
        provider.create_batch("file-1", endpoint="/v1/responses")


def test_openai_responses_batch_default_endpoint_is_responses():
    """responses 协议下缺省 endpoint 是 ``/v1/responses``，不是 chat 端点。

    ``==``→``!=`` 变异后判据反转：默认请求打到 ``/v1/chat/completions``——
    batch 与聊天端点混用，服务端按错误路由处理。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    provider._base_url = "https://api.openai.com/v1/"
    provider._server_verified_protocols = frozenset()
    captured = {}

    class Batches:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "batch"

    provider._get_client = lambda: SimpleNamespace(batches=Batches())

    assert provider.create_batch("file-1") == "batch"
    assert captured["endpoint"] == "/v1/responses"


def test_openai_async_responses_batch_guard_covers_explicit_endpoint():
    """异步 create_batch 同判据（``==``→``!=`` 变异后门禁被绕开）。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    provider._base_url = "https://gateway.example/v1"
    provider._server_verified_protocols = frozenset()

    class Batches:
        async def create(self, **_kwargs):
            return "batch"

    provider._get_aclient = lambda: SimpleNamespace(batches=Batches())

    with pytest.raises(ValueError, match="未验证 Responses"):
        asyncio.run(provider.async_create_batch("file-1", endpoint="/v1/responses"))


def test_openai_async_responses_batch_default_endpoint_is_responses():
    """异步缺省 endpoint 同判据（``==``→``!=`` 变异后打到 chat 端点）。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {}
    provider._base_url = "https://api.openai.com/v1/"
    provider._server_verified_protocols = frozenset()
    captured = {}

    class Batches:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return "batch"

    provider._get_aclient = lambda: SimpleNamespace(batches=Batches())

    assert asyncio.run(provider.async_create_batch("file-1")) == "batch"
    assert captured["endpoint"] == "/v1/responses"


def _ark_chat_chunks():
    return [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "t1",
                                "function": {"name": "f", "arguments": '{"a":1}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]


def _ark_chat_provider():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._provider = "volcengine"
    provider._protocol = "chat_completions"
    provider._options = {}
    return provider


def test_ark_chat_stream_events_flush_completed_after_delta():
    """finish 前先冲刷：``tool_call_delta`` 先于 ``tool_call_completed`` 发出。

    冲刷判据 ``converted.type == "finish"`` 若变异成 ``!=``，冲刷提前到首个
    delta 上——实测事件序列变成 ``[tool_call_completed, tool_call_delta,
    finish]``，消费方在看到片段之前先收到完成事件。
    """
    provider = _ark_chat_provider()

    class Completions:
        def create(self, **_request):
            return iter(_ark_chat_chunks())

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    events = list(provider.stream_events(CompletionRequest(prompt="hi")))

    assert [e.type for e in events] == ["tool_call_delta", "tool_call_completed", "finish"]


def test_ark_async_chat_stream_events_flush_completed_after_delta():
    """异步 stream_events 同判据（``==``→``!=`` 变异后完成事件抢在片段前）。"""
    provider = _ark_chat_provider()

    class Completions:
        async def create(self, **_request):
            return _ark_chat_chunks()

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def collect():
        return [e.type async for e in provider.astream_events(CompletionRequest(prompt="hi"))]

    assert asyncio.run(collect()) == ["tool_call_delta", "tool_call_completed", "finish"]


def test_ark_invoke_stream_yields_text_delta():
    """Ark 同步 invoke 按 ``event.type == "text_delta"`` 交付文本。

    ``==``→``!=`` 变异后文本全部丢弃（实测 ``[]``），聊天界面收到空回复。
    """
    provider = _ark_chat_provider()

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    {"choices": [{"delta": {"content": "答"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    assert list(provider.invoke("hi", stream=True, temperature=0.5)) == ["答"]


def test_ark_ainvoke_stream_yields_text_delta():
    """Ark 异步 invoke 同判据（``==``→``!=`` 变异后返回空列表）。"""
    provider = _ark_chat_provider()

    class Completions:
        async def create(self, **_request):
            return [
                {"choices": [{"delta": {"content": "答"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hi", stream=True, temperature=0.5)]

    assert asyncio.run(consume()) == ["答"]


# --- face 13（True ↔ False 互换扫描）存活位点 ---------------------------------


class _RecordingLLM(LargeLanguageModel):
    """记录 ``invoke``／``ainvoke`` 收到的 ``stream`` 实参。"""

    def __init__(self):
        self.seen: list[tuple[str, bool]] = []

    def invoke(self, prompt=None, system_prompt=None, tools=None, stream=True, temperature=0.7):
        self.seen.append(("invoke", stream))
        yield "x"

    async def ainvoke(
        self, prompt=None, system_prompt=None, tools=None, stream=True, temperature=0.7
    ):
        self.seen.append(("ainvoke", stream))
        yield "x"


def test_chat_tool_call_item_defaults_to_no_call_id_requirement():
    """``require_call_id`` 默认 ``False``：无 id 的历史工具调用必须被原样接受。

    缺 id 的 Gemini 历史经 ``_chat_tool_call_item`` 默认路径必须通过；
    ``require_call_id=False`` 反向成 ``True`` 后该路径开始拒绝本可发送的请求。
    """
    item = model_provider_module._chat_tool_call_item(
        {"type": "function", "function": {"name": "f", "arguments": "{}"}}, "loc"
    )

    assert item == {"type": "function", "function": {"name": "f", "arguments": "{}"}}


def test_chat_tool_call_item_does_not_leak_responses_marker_by_default():
    """``keep_responses_type`` 默认 ``False``：内部标记不得进入 Chat 请求体。

    该默认值反向成 ``True`` 后，``_chat_tool_call_item`` 会给
    ``type != function`` 的工具调用挂上 ``_responses_tool_call_type``——一个
    只在 Responses 内部判定用的字段随之进入 Chat Completions 请求体。
    """
    item = model_provider_module._chat_tool_call_item(
        {"type": "custom_tool_call", "name": "f", "input": "raw"}, "loc"
    )

    assert "_responses_tool_call_type" not in item
    assert item == {"type": "function", "function": {"name": "f", "arguments": "raw"}}


def test_responses_input_rejects_tool_call_id_without_call_id_for_output():
    """只有 Chat 的 ``tool_call_id`` 不能满足 Responses 的 ``call_id`` 要求。

    缺 ``call_id`` 的 ``function_call_output`` 必须被拒；
    ``require_call_id=True`` 反向成 ``False`` 后，``tool_call_id`` 被当成
    ``call_id`` 的别名回填，而 Responses 的关联键是 ``call_id``，回填值与
    调用方的 ``function_call`` 对不上，工具输出静默失联。
    """
    messages = [{"type": "function_call_output", "tool_call_id": "tc1", "output": "{}"}]

    with pytest.raises(ValueError, match="tool_call_id/call_id"):
        normalize_responses_input(None, None, messages)


@pytest.mark.parametrize("method", ["complete", "acomplete", "stream_events", "astream_events"])
def test_compat_layer_forces_stream_flag_for_its_call_shape(method):
    """兼容层必须按自己的调用形态固定 ``stream``，不能沿用请求里的取值。

    ``complete``／``acomplete`` 要**非流式**（一次性聚合），``stream_events``／
    ``astream_events`` 要**流式**（逐块产出）。``CompletionRequest.stream`` 的
    默认值 ``True`` 被反向成 ``False`` 后，``complete``／``acomplete`` 的
    ``copy_with(stream=False)`` 仍显式固定，但 ``stream_events``／
    ``astream_events`` 的 ``copy_with(stream=True)`` 变成 ``False``——两条事件
    流路径静默退化成非流式，逐块产出变成一次性返回。既有用例只断言了
    ``complete`` 的文本结果，事件流两条路径的 ``stream`` 实参从未被观察。
    """
    provider = _RecordingLLM()

    if method == "complete":
        provider.complete(prompt="hi")
    elif method == "acomplete":
        asyncio.run(provider.acomplete(prompt="hi"))
    elif method == "stream_events":
        list(provider.stream_events(prompt="hi"))
    else:

        async def _drain():
            return [event async for event in provider.astream_events(prompt="hi")]

        asyncio.run(_drain())

    expected_method = "ainvoke" if method in ("acomplete", "astream_events") else "invoke"
    expected_stream = method in ("stream_events", "astream_events")
    assert provider.seen == [(expected_method, expected_stream)]


def test_completion_request_stream_defaults_to_true():
    """``CompletionRequest.stream`` 的默认值是 ``True``（``bool`` 不是 ``int``）。

    默认值反向成 ``False`` 后，未显式指定 ``stream`` 的请求会静默走非流式；
    ``to_invoke_kwargs`` 原样转发该字段，调用方拿到的 ``stream`` 与契约相反。
    """
    assert CompletionRequest(prompt="hi").stream is True
    assert CompletionRequest(prompt="hi").to_invoke_kwargs()["stream"] is True


def test_completion_request_temperature_flag_defaults_off_and_stays_private():
    """``_temperature_explicit`` 的 dataclass 默认值是 ``False``，且字段保持私有。

    默认值反向成 ``True`` 后：``repr`` 里出现该字段（``repr=False`` 失效）、
    两个 ``CompletionRequest`` 不再相等（``compare=False`` 失效）、而
    ``__post_init__`` 仍按真实显式性赋值——``temperature_is_explicit`` 与
    ``repr``／``==`` 的语义互相矛盾。``init=False`` 使该字段不能从构造器传入，
    因此 ``repr``／``==`` 是观察默认值的唯一入口。
    """
    request = CompletionRequest(prompt="hi")
    assert "_temperature_explicit" not in repr(request)
    assert request == CompletionRequest(prompt="hi", temperature=0.7)


def test_completion_request_temperature_field_default_is_false_in_the_dataclass():
    """``_temperature_explicit`` 的**字段声明默认值**必须是 ``False``。

    上面那条用例断言的是 ``repr``／``==`` 的行为，而这两个面由
    ``repr=False``／``compare=False`` 决定——把 ``default=False`` 反向成
    ``True`` 后它们逐字不变，实测全量套件仍绿。声明默认值本身是
    ``dataclasses.fields()`` 可直接读到的契约，也是「字段默认关」这条
    docstring 承诺的落点，必须单独钉住。
    """
    field_info = {item.name: item for item in dataclasses.fields(CompletionRequest)}[
        "_temperature_explicit"
    ]
    assert field_info.default is False
    assert field_info.init is False


def test_completion_request_temperature_field_cannot_be_passed_to_the_constructor():
    """``init=False`` 必须让私有字段无法从构造器注入。

    ``init=False`` 反向成 ``True`` 后，``CompletionRequest(prompt="hi",
    _temperature_explicit=True)`` 被接受——调用方可以伪造「温度是显式传入的」
    这一标记，而 ``__post_init__`` 随后会按真实取值覆盖它，标记与温度自相矛盾。
    """
    with pytest.raises(TypeError):
        CompletionRequest(prompt="hi", _temperature_explicit=True)


def test_normalize_chat_messages_requires_tool_call_ids_only_when_asked():
    """缺 ``id`` 的历史工具调用：默认放行，``require_tool_call_ids=True`` 才拒绝。

    ``require_tool_call_ids`` 默认 ``False`` 被反向成 ``True`` 后，**默认路径**
    开始拒绝无 id 的 assistant 历史——Gemini 这类不带 id 的历史在 Chat 路径上
    被静默挡下；而显式要求 id 的路径（Responses）行为不变。既有用例只覆盖了
    ``require_tool_call_ids=True`` 与带 id 的输入，默认路径这一半从未被触达。
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
    ]

    normalized = normalize_chat_messages(messages)
    assert normalized[0]["tool_calls"][0]["type"] == "function"

    with pytest.raises(ValueError, match="缺少 id/call_id"):
        normalize_chat_messages(messages, require_tool_call_ids=True)


def test_normalize_messages_keeps_custom_tool_call_type_only_when_asked():
    """``custom_tool_call`` 的原始类型只在 ``keep_responses_type=True`` 时保留。

    默认 ``False`` 被反向成 ``True`` 后，Chat 路径会把 Responses 专属的
    ``custom_tool_call`` 原样带进请求体——Chat Completions 的
    ``tool_calls[].type`` 只接受 ``function``。既有用例只覆盖了
    ``keep_responses_type=True`` 与 ``function`` 类型，默认路径这一半从未被触达。
    """
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "custom_tool_call", "name": "f", "input": "raw text"}
            ],
        }
    ]

    normalized = normalize_messages(None, None, messages)
    assert normalized[0]["tool_calls"][0]["type"] == "function"
    assert "_responses_tool_call_type" not in normalized[0]["tool_calls"][0]

    kept = normalize_messages(None, None, messages, keep_responses_type=True)
    assert kept[0]["tool_calls"][0]["type"] == "function"
    assert kept[0]["tool_calls"][0]["_responses_tool_call_type"] == "custom_tool_call"


def test_responses_input_requires_call_id_for_function_call_output():
    """Responses 的 ``function_call_output`` 必须带 ``call_id``（``require_call_id=True``）。

    该默认值反向成 ``False`` 后，缺 ``call_id`` 的 ``function_call_output`` 不再
    被拒——下游按 ``call_id`` 关联工具调用时拿到 ``None``，关联静默失效。
    """
    messages = [{"type": "function_call_output", "output": "{}"}]

    with pytest.raises(ValueError, match="call_id"):
        normalize_responses_input(None, None, messages)


def test_responses_json_text_keeps_non_ascii_verbatim():
    """``json.dumps`` 的 ``ensure_ascii=False`` 必须保留非 ASCII 字符。

    反向成 ``True`` 后中文被转义成 ``\\uXXXX`` 形态——字节数翻倍、下游按原文匹配
    的调用方拿不到原值。既有用例只断言了「是 JSON 字符串」，没有断言转义形态。
    """
    assert model_provider_module._responses_json_text({"a": "中文"}, "loc") == '{"a":"中文"}'


def test_normalize_tool_arguments_keeps_non_ascii_verbatim():
    """工具参数序列化同样必须 ``ensure_ascii=False``。"""
    assert normalize_tool_arguments({"a": "中文"}) == '{"a":"中文"}'


def test_normalize_embedding_vector_rejects_invalid_base64_with_value_error():
    """非法 base64 必须报 ``ValueError``，不能漏出 ``binascii.Error``。

    ``validate=True`` 反向成 ``False`` 后，``base64.b64decode`` 忽略非法字符，
    ``"YWJj!!!"`` 被解成 ``b"abc"`` 并继续走 float32 解析——错误从「内容无效」
    变成「不是有效的 float32 字节序列」，调用方按前者写的错误分类接不住。
    """
    with pytest.raises(ValueError, match="base64 内容无效"):
        normalize_embedding_vector("YWJj!!!")


@pytest.mark.parametrize("retry", [retry_sync_call, retry_async_call])
def test_retry_helpers_reraise_original_error(retry):
    """重试耗尽后必须抛出**原始异常**，不是 ``RetryError``。

    ``reraise=True`` 反向成 ``False`` 后，tenacity 把最后一次异常包成
    ``RetryError`` 抛出——它既不是 ``ValueError`` 也不是 ``RuntimeError``，
    调用方按后者写的错误分类全部接不住。既有用例把 ``retry_*`` 整体 monkeypatch
    掉了，真实重试路径的收场从未被观察。
    """

    class _TimeoutLike(Exception):
        pass

    def _boom():
        raise _TimeoutLike("boom")

    async def _aboom():
        raise _TimeoutLike("aboom")

    target = _boom if retry is retry_sync_call else _aboom
    with pytest.raises(_TimeoutLike):
        if retry is retry_sync_call:
            retry(target)
        else:
            asyncio.run(retry(target))


def test_responses_has_content_treats_empty_containers_as_absent():
    """空串／空列表／``0`` 都算「没有内容」（``return True`` 兜底）。

    兜底反向成 ``False`` 后，``0`` 与空容器被判为无内容——``_responses_content_part``
    随后走空内容分支，``0`` 这样的合法取值被静默丢弃。既有用例只覆盖了
    ``None` 与空串，``0` 与空容器这一半从未被触达。
    """
    assert _responses_has_content(None) is False
    assert _responses_has_content("") is False
    assert _responses_has_content("x") is True
    assert _responses_has_content([]) is False
    assert _responses_has_content([1]) is True
    assert _responses_has_content(0) is True
    assert _responses_has_content(1) is True
    assert _responses_has_content({"a": 1}) is True


# --- face 13（True ↔ False 互换扫描）：Anthropic 入口与数据形态 ---------------


class _AnyAttr:
    """任意属性访问都返回可调用对象，调用结果仍是 ``_AnyAttr``。"""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: _AnyAttr()

    def __iter__(self):
        return iter(())

    def __aiter__(self):
        async def _gen():
            for _item in ():
                yield _item

        return _gen()


def _anthropic_flag_provider(recorder: dict):
    """记录各入口传给 ``_build_message_params``／``_parse_params`` 的开关。"""
    provider = _anthropic_provider()

    def _build(**kwargs):
        recorder.setdefault("build", []).append(
            {key: kwargs.get(key) for key in ("stream", "beta", "for_parse", "for_tool_runner")}
        )
        return {}

    def _parse(params, *, beta):
        recorder.setdefault("parse", []).append(beta)
        return {}

    provider._build_message_params = _build
    provider._parse_params = _parse
    provider._validate_token_count_request = lambda request, *, beta: recorder.setdefault(
        "token_count", []
    ).append(beta)
    provider._get_client = lambda: _AnyAttr()
    provider._get_aclient = lambda: _AnyAttr()
    return provider


def _consume(call):
    """调用入口并消费可能的生成器；返回是否抛异常。"""
    try:
        result = call()
        if inspect.isawaitable(result):
            asyncio.run(result)
        elif inspect.isgenerator(result):
            list(result)
        elif inspect.isasyncgen(result):

            async def _drain():
                return [item async for item in result]

            asyncio.run(_drain())
    except Exception:  # noqa: BLE001 - 入口内部实现与本用例无关
        return True
    return False


@pytest.mark.parametrize(
    "method_name,expected",
    [
        ("complete", {"stream": False, "beta": None, "for_parse": None, "for_tool_runner": None}),
        ("acomplete", {"stream": False, "beta": None, "for_parse": None, "for_tool_runner": None}),
        ("parse", {"stream": False, "beta": None, "for_parse": True, "for_tool_runner": None}),
        (
            "async_parse",
            {"stream": False, "beta": None, "for_parse": True, "for_tool_runner": None},
        ),
        (
            "beta_create",
            {"stream": False, "beta": True, "for_parse": None, "for_tool_runner": None},
        ),
        (
            "async_beta_create",
            {"stream": False, "beta": True, "for_parse": None, "for_tool_runner": None},
        ),
        ("beta_parse", {"stream": False, "beta": True, "for_parse": True, "for_tool_runner": None}),
        (
            "async_beta_parse",
            {"stream": False, "beta": True, "for_parse": True, "for_tool_runner": None},
        ),
        ("beta_stream", {"stream": True, "beta": True, "for_parse": None, "for_tool_runner": None}),
        (
            "async_beta_stream",
            {"stream": True, "beta": True, "for_parse": None, "for_tool_runner": None},
        ),
        (
            "beta_tool_runner",
            {"stream": True, "beta": True, "for_parse": None, "for_tool_runner": True},
        ),
        (
            "async_beta_tool_runner",
            {"stream": True, "beta": True, "for_parse": None, "for_tool_runner": True},
        ),
    ],
)
def test_anthropic_entries_pass_expected_stream_and_beta_flags(method_name, expected):
    """每个公共入口传给 ``_build_message_params`` 的开关必须与它的形态一致。

    这些开关都是布尔字面量，反向成另一个取值后：``complete``／``acomplete``／
    ``parse`` 三条**非流式**入口变成流式装配、``beta_*`` 不再启用 Beta 资源、
    ``parse`` 不再走 ``for_parse`` 分支（``output_format`` 校验与配置覆盖的
    行为随之改变）。既有用例只断言了各入口的最终结果或错误消息，装配开关
    本身从未被观察。
    """
    recorder: dict = {}
    provider = _anthropic_flag_provider(recorder)
    request = CompletionRequest(prompt="hi")
    calls = {
        "complete": lambda: provider.complete(request),
        "acomplete": lambda: provider.acomplete(request),
        "parse": lambda: provider.parse({}, request),
        "async_parse": lambda: provider.async_parse({}, request),
        "beta_create": lambda: provider.beta_create(request),
        "async_beta_create": lambda: provider.async_beta_create(request),
        "beta_parse": lambda: provider.beta_parse({}, request),
        "async_beta_parse": lambda: provider.async_beta_parse({}, request),
        "beta_stream": lambda: provider.beta_stream(request),
        "async_beta_stream": lambda: provider.async_beta_stream(request),
        "beta_tool_runner": lambda: provider.beta_tool_runner([], request),
        "async_beta_tool_runner": lambda: provider.async_beta_tool_runner([], request),
    }

    _consume(calls[method_name])

    assert recorder["build"] == [expected]


@pytest.mark.parametrize(
    "method_name,expected_beta",
    [
        ("count_tokens", False),
        ("beta_count_tokens", True),
        ("async_count_tokens", False),
        ("async_beta_count_tokens", True),
    ],
)
def test_anthropic_token_count_entries_declare_beta_consistently(method_name, expected_beta):
    """``count_tokens`` 家族必须按入口声明是否走 Beta 资源（``beta=False`` 不是 ``True``）。

    反向成另一个取值后，普通入口被当成 Beta 调用（``beta=False`` 的 token
    计数会去申请 Beta 专属资源），或反过来让 Beta 入口走非 Beta 校验——
    两边的参数约束与资源挂载不一致。既有用例只断言了返回的 token 数。
    """
    recorder: dict = {}
    provider = _anthropic_flag_provider(recorder)
    request = CompletionRequest(prompt="hi")

    _consume(lambda: getattr(provider, method_name)(request))

    assert recorder["token_count"] == [expected_beta]


def test_anthropic_convert_content_rejects_illegal_base64_characters():
    """data URI 的 base64 必须 ``validate=True``，非法字符不能被静默忽略。

    ``validate=False`` 时 ``base64.b64decode`` 丢弃非法字符，``"YWJj!!!"`` 被解成
    ``b"abc"`` 并作为图片内容发给上游——损坏的图片被当成合法数据发送，而
    ``"中文"`` 这类非 ASCII 输入则抛裸 ``ValueError: string argument should
    contain only ASCII characters``（不是本模块的指引消息）。
    """
    with pytest.raises(ValueError, match="不是有效 Base64"):
        AnthropicProvider._convert_content(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj!!!"}}]
        )


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_anthropic_invoke_stream_returns_quietly_for_empty_stream(is_async):
    """空事件流必须静默返回，不能报「message_stop 之前结束」。

    ``saw_event`` 初值反向成 ``True`` 后，空响应被当成「收到过事件」并要求
    ``message_stop``——实测抛 RuntimeError，而「响应对象没有可迭代事件」是
    被支持的既有形态（下面的 ``current_message_snapshot``／``text_stream``
    两条兜底路径正是为它准备的）。
    """

    class _EmptyStream:
        def __iter__(self):
            return iter(())

        def __aiter__(self):
            async def _gen():
                for _item in ():
                    yield _item

            return _gen()

        @property
        def text_stream(self):
            return ()

    response = _EmptyStream()
    if is_async:

        async def run():
            return [item async for item in AnthropicProvider._aiter_invoke_stream_text(response)]

        assert asyncio.run(run()) == []
    else:
        assert list(AnthropicProvider._iter_invoke_stream_text(response)) == []


def test_anthropic_stream_event_keeps_non_ascii_tool_arguments_verbatim():
    """``tool_use`` 的初始参数序列化必须 ``ensure_ascii=False``。

    反向成 ``True`` 后中文参数被转义成 ``\\uXXXX`` 形态——回填给模型的参数
    与调用方给出的原值不同，按原文匹配的下游拿不到原值。
    """
    event = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": "中文"}},
    }

    converted = AnthropicProvider._stream_event(event)

    assert converted is not None
    assert converted.tool_call["arguments"] == '{"a":"中文"}'


# ── face 13（True ↔ False 互换）扫描在 Google 入口的存活位点 ──
# 实测：574／1483／1900 反向后全量 4269 条仍全绿，用能触达守卫的判别输入
# 确认行为确实改变（EQUIV 桶逐条反证，见 1922/1287 等处的甄别记录）。


def test_google_validate_http_options_keeps_blank_query_values():
    """``keep_blank_values=True``：query 里值为空的敏感参数同样要被报出来。

    反向成 ``False`` 后 ``parse_qsl`` 丢弃 ``token=``（值为空）这一项，整个 query
    归一成空对——「配置里放了一个空 token」被静默接受，与 ``redact_sensitive_text``
    对非空值的拦截（同一行的前半判据）只判了一半。实测 base 报
    ``headers.X?token``，变异体接受。
    """
    with pytest.raises(ValueError, match=r"headers\.X\?token\b"):
        GoogleProvider._validate_http_options({"headers": {"X": "http://h/p?token="}})


def test_google_decode_data_uri_rejects_illegal_base64_characters():
    """Google data URI 的 base64 必须 ``validate=True``，与 Anthropic 侧同判据。

    反向成 ``False`` 后 ``base64.b64decode`` 丢弃非法字符，``"YWJj!!!"`` 被解成
    ``b"abc"`` 照常返回——损坏的图片被当成合法数据继续走。既有用例断言的是
    非 ASCII 输入（走 ``ValueError``），ASCII 非法字符这一半此前无人看守。
    """
    with pytest.raises(ValueError, match="data URI 不是有效 Base64"):
        GoogleProvider._decode_data_uri("data:image/png;base64,YWJj!!!", "图片")


def test_google_response_text_reaches_the_accessor_fallback_when_parts_are_empty():
    """``saw_parts`` 初值必须是 ``False``：parts 为空时才够得着 ``response.text`` 兜底。

    反向成 ``True`` 后空 parts 也被当成「看过部件」，直接返回空串，随后
    ``response.text`` 里的正文被跳过——SDK 构造的**无 parts** 响应
    （只有 ``text`` 属性）此前拿得到正文，变异后拿到空字符串。
    """
    from types import SimpleNamespace as NS

    response = NS(candidates=[NS(content=NS(parts=[]))], text="兜底正文")

    assert GoogleProvider._response_text(response) == "兜底正文"


class _AnthropicRecorder:
    """路径感知的假 Anthropic SDK 客户端：属性链成路径，末端调用被记录。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._path = ""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = _AnthropicRecorder()
        child.calls = self.calls
        child._path = f"{self._path}.{name}"
        return child

    def __call__(self, *args, **kwargs):
        self.calls.append((self._path.lstrip("."), kwargs))
        return SimpleNamespace(
            input_tokens=3,
            total_tokens=3,
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )


@pytest.mark.parametrize(
    "method_name,is_async",
    [("parse", False), ("async_parse", True)],
)
def test_anthropic_parse_rejects_beta_only_params(method_name, is_async):
    """非 Beta 的 ``parse`` 入口必须拒绝 ``cache_control`` 这类 Beta 专属参数。

    ``_parse_params(params, beta=False)`` 的 ``beta`` 反向成 ``True`` 后，允许
    集合被扩成 Beta 全集，``cache_control``／``speed`` 等字段被原样交给
    非 Beta 的 ``messages.parse``——SDK 侧要么报未知参数、要么把 Beta 专属语义
    发给普通端点。实测 base 抛「不支持请求参数: cache_control」，变异体照常调用
    ``messages.parse``。
    """
    recorder = _AnthropicRecorder()
    provider = _anthropic_provider()
    provider._get_client = lambda: recorder
    provider._get_aclient = lambda: recorder
    request = CompletionRequest(prompt="hi", cache_control={"type": "ephemeral"})

    with pytest.raises(ValueError, match=r"Anthropic Messages parse 不支持请求参数: cache_control"):
        result = getattr(provider, method_name)({}, request)
        if is_async:
            asyncio.run(result)


@pytest.mark.parametrize(
    "method_name,is_async",
    [("beta_parse", False), ("async_beta_parse", True)],
)
def test_anthropic_beta_parse_forwards_beta_only_params(method_name, is_async):
    """Beta 的 ``beta_parse`` 入口必须放行 ``speed`` 这类 Beta 专属参数。

    ``_parse_params(params, beta=True)`` 的 ``beta`` 反向成 ``False`` 后，允许
    集合退化成普通集合，``speed``（经 ``beta_values`` 进入 params）被判为
    「不支持请求参数」——合法的 Beta 调用直接失败。实测 base 调用
    ``beta.messages.parse``，变异体抛 ``ValueError``。
    """
    recorder = _AnthropicRecorder()
    provider = _anthropic_provider()
    provider._get_client = lambda: recorder
    provider._get_aclient = lambda: recorder
    request = CompletionRequest(prompt="hi", speed="fast")

    result = getattr(provider, method_name)({}, request)
    if is_async:
        asyncio.run(result)

    assert [name for name, _ in recorder.calls] == ["beta.messages.parse"]
    assert recorder.calls[0][1]["speed"] == "fast"


def test_anthropic_async_count_tokens_declares_user_profile_beta_header():
    """``async_count_tokens`` 的 ``_user_profile_fields(beta=False)`` 必须走头合并分支。

    该调用的 ``beta=False`` 反向成 ``True`` 后，``_user_profile_fields`` 改走
    Beta 资源分支：``user-profiles-2026-08-18`` 被塞进**第二个**返回值
    （``betas``），而本处只取第一个返回值（``profile_headers``）——Beta 声明被
    丢弃，``extra_headers`` 里不再有 ``anthropic-beta``。实测 base 的
    ``messages.count_tokens`` 收到该头，变异体收到 ``None``。同步入口
    （``count_tokens``）已被既有用例覆盖，异步这一支此前无人看守。
    """
    recorder = _AnthropicRecorder()
    provider = _anthropic_provider()
    provider._get_aclient = lambda: recorder

    asyncio.run(provider.async_count_tokens(CompletionRequest(prompt="hi", user="u1")))

    assert recorder.calls == [
        (
            "messages.count_tokens",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are a helpful assistant.",
                "user_profile_id": "u1",
                "extra_headers": {"anthropic-beta": "user-profiles-2026-08-18"},
            },
        )
    ]


def test_anthropic_build_message_params_rejects_output_format_without_stream():
    """非流式装配拒绝 ``output_format``，除非调用方显式声明是 tool runner。

    ``for_tool_runner`` 的默认值 ``False`` 反向成 ``True`` 后，任何**没有**声明
    tool runner 的调用都满足 ``not (beta and for_tool_runner)``，于是
    ``beta=True`` + ``stream=False`` + ``output_format`` 的组合不再报
    「Messages.create 不支持 output_format」，``output_format`` 被原样塞进
    ``messages.create`` 的参数。实测 base 抛 ``ValueError``，变异体返回含
    ``output_format`` 的 params。
    """
    provider = _anthropic_provider()

    with pytest.raises(ValueError, match="不支持 output_format"):
        provider._build_message_params(prompt="hi", beta=True, stream=False, output_format=int)


@pytest.mark.parametrize(
    "provider_cls",
    [AnthropicProvider, GoogleProvider, OpenAICompatibleProvider, VolcengineProvider],
)
def test_provider_invoke_and_ainvoke_default_to_streaming(provider_cls):
    """所有 Provider 的 ``invoke``／``ainvoke`` 都必须默认 ``stream=True``。

    ``stream`` 的默认值反向成 ``False`` 后，不传该参数的调用静默退化成非流式
    ——``chat/core.py`` 与 ``ChatService`` 都靠显式传参自保，但公共签名承诺的
    默认形态被改变。抽象基类 ``LargeLanguageModel`` 上的两处默认值是各 Provider
    共同契约的落点，同样必须钉住（实测全量套件对这两处无感）。
    """
    for method_name in ("invoke", "ainvoke"):
        parameters = inspect.signature(getattr(provider_cls, method_name)).parameters
        assert parameters["stream"].default is True, f"{provider_cls.__name__}.{method_name}"


def test_abstract_base_invoke_signatures_default_to_streaming():
    """抽象基类的 ``invoke``／``ainvoke`` 签名默认值同样是 ``True``。

    这两处（``model_provider.py`` 2167／2178 行）是各 Provider 覆盖时的契约来源，
    反向成 ``False`` 后仅靠 ``inspect.signature`` 可观察；没有这条断言时全量
    套件对它无感。
    """
    for method_name in ("invoke", "ainvoke"):
        parameters = inspect.signature(getattr(LargeLanguageModel, method_name)).parameters
        assert parameters["stream"].default is True, f"LargeLanguageModel.{method_name}"


# =========================================================================
# face 13：boolc（True↔False 字面量互换）扫描确认的真缺口
# -------------------------------------------------------------------------
# 以下用例对应 ``face13_auth68.log`` 中 68 个存活位点里经真实入口探针
# 判定为**真缺口**的部分（另 18 处等价变异见 CHANGELOG 条目）。
# =========================================================================


# --- Google：原生工具 ``is True`` 与分片 ``will_continue`` -----------------


def test_google_native_tool_accepts_boolean_true_marker():
    """``{"type": "google_search", "google_search": True}`` 必须转成合法 ``Tool``。

    ``native_value is True or native_value is None`` 反向成 ``is False`` 后，
    布尔 ``True`` 既不是 ``False`` 也不是 ``None``，被原样交给
    ``types.Tool(google_search=True)``——pydantic 校验失败，一条合法的
    「打开内置搜索」声明直接抛 ``ValidationError``。
    """
    from google.genai import types

    tools = GoogleProvider._convert_tools([{"type": "google_search", "google_search": True}])

    assert len(tools) == 1
    assert isinstance(tools[0], types.Tool)
    assert tools[0].google_search is not None


def test_google_native_tool_rejects_boolean_false_marker():
    """``google_search=False`` 必须被拒（``is True or is None`` 不是「任意布尔」）。

    ``is True`` 反向成 ``is False`` 后，``False`` 落进「当作空配置」分支，
    本应报错的显式关闭标记被静默接受成 ``Tool(google_search=GoogleSearch())``。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GoogleProvider._convert_tools([{"type": "google_search", "google_search": False}])


def test_google_tool_call_fragment_uses_will_continue_as_identity():
    """无 id/index 的分片靠 ``will_continue is True`` 关联到已有候选。

    ``tool_call.get("will_continue") is True`` 反向成 ``is False`` 后，一个
    显式声明 ``will_continue=True`` 的分片不再被认作可关联——本可合并的流式
    分片抛 ``RuntimeError: ... 无法确认是否为同一调用``。
    """
    accumulator = {("position", 0): {"type": "function", "name": "f"}}

    key, merged = GoogleProvider._merge_tool_call(
        accumulator, {"name": "f", "will_continue": True, "arguments": "{}"}
    )

    assert key == ("position", 0)
    assert merged["name"] == "f"


def test_google_tool_call_fragment_rejects_explicit_false_continuation():
    """显式 ``will_continue=False`` 的分片不得被当成可关联。

    ``is True`` 反向成 ``is False`` 后，``False is False`` 成立——一个显式声明
    「本次调用到此为止」的分片被误认作可关联，两个同名并行调用被静默合并。
    """
    accumulator = {("position", 0): {"type": "function", "name": "f"}}

    with pytest.raises(RuntimeError, match="无法确认是否为同一调用"):
        GoogleProvider._merge_tool_call(
            accumulator, {"name": "f", "will_continue": False, "arguments": "{}"}
        )


# --- OpenAI 兼容：``strict`` 默认值、空状态判定、流终止、background --------


def test_responses_tools_default_strict_to_false():
    """原生 function 工具未写 ``strict`` 时补 ``False``（严格模式默认关）。

    ``setdefault("strict", False)`` 反向成 ``True`` 后，未显式要求严格模式的
    工具被强制开启 ``strict``——服务端按严格 schema 校验，宽松的工具定义
    开始报错。
    """
    converted = OpenAICompatibleProvider._convert_responses_tools(
        [{"type": "function", "name": "f", "parameters": {"type": "object"}}]
    )

    assert converted[0]["strict"] is False


def test_responses_tools_preserve_explicit_strict_true():
    """显式 ``strict=True`` 必须保留（默认只补缺省，不覆盖调用方）。"""
    converted = OpenAICompatibleProvider._convert_responses_tools(
        [{"type": "function", "name": "f", "parameters": {"type": "object"}, "strict": True}]
    )

    assert converted[0]["strict"] is True


@pytest.mark.parametrize("status", ["", "   ", " \t\n"])
def test_chat_status_is_error_treats_blank_status_as_not_error(status):
    """空白字符串状态不算错误（``if not normalized: return False``）。

    ``return False`` 反向成 ``return True`` 后，网关在 HTTP 200 里放一个
    空白 ``status`` 就会被判成业务失败——一次正常响应被报成 ``RuntimeError``。
    """
    assert OpenAICompatibleProvider._chat_status_is_error(status) is False


def test_openai_async_chat_stream_requires_terminal_finish():
    """异步 Chat 流缺少 finish 块时必须报错（``terminal_seen = True`` 初值）。

    初值反向成 ``True`` 后，被截断的流不再报「finish 之前结束」——调用方
    拿到一段不完整文本却以为流正常收尾。
    """

    def _rows(with_finish):
        out = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None), finish_reason=None
                    )
                ],
                usage=None,
                id="r",
            )
        ]
        if with_finish:
            out.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                    id="r",
                )
            )
        return out

    def _provider(with_finish):
        provider = object.__new__(OpenAICompatibleProvider)
        provider._model_name = "m"
        provider._provider = "openai"
        provider._protocol = "chat_completions"
        provider._options = {}

        class Completions:
            async def create(self, **_kw):
                return _rows(with_finish)

        provider._get_aclient = lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        return provider

    async def _collect(provider):
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    complete = asyncio.run(_collect(_provider(True)))
    assert [event.type for event in complete] == ["text_delta", "finish"]

    with pytest.raises(RuntimeError, match="finish 事件之前结束"):
        asyncio.run(_collect(_provider(False)))


def test_openai_responses_stream_records_terminal_completed():
    """收到 ``response.completed`` 的完整流必须正常收尾（不误报截断）。

    ``terminal_seen = True`` 反向成 ``False`` 后，**已经**收到终止事件的完整流
    仍被判为「response.completed 之前结束」——一次正常响应被报成 ``RuntimeError``。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="hi"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed", error=None),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    assert list(provider._invoke_responses({"stream": True})) == ["hi"]


@pytest.mark.parametrize(
    "method_name,is_async",
    [("parse", False), ("async_parse", True)],
)
def test_openai_parse_rejects_background_requests(method_name, is_async):
    """``parse`` 必须显式拒绝 ``background=True``（先于流式开关判定）。

    ``request.copy_with(stream=False)`` 反向成 ``stream=True`` 后，
    ``_build_responses_request`` 因 ``background and stream`` 抛出**另一条**
    「background 请求不支持流式模式」——指引与 ``parse`` 语义不符，调用方按
    「parse 不支持 background」写的分类接不住。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class _Never:
        @staticmethod
        def parse(**_kw):
            return "unused"

    provider._get_client = lambda: SimpleNamespace(
        responses=_Never(), chat=SimpleNamespace(completions=_Never())
    )
    provider._get_aclient = lambda: SimpleNamespace(
        responses=_Never(), chat=SimpleNamespace(completions=_Never())
    )
    request = CompletionRequest(prompt="hi", background=True)

    with pytest.raises(ValueError, match="不支持 background=true"):
        result = getattr(provider, method_name)(text_format=int, request=request)
        if is_async:
            asyncio.run(result)


@pytest.mark.parametrize(
    "method_name,is_async",
    [("stream_responses", False), ("async_stream_responses", True)],
)
def test_openai_stream_responses_rejects_background_requests(method_name, is_async):
    """``stream_responses`` 必须拒绝 ``background=True``。

    ``request.copy_with(stream=True)`` 反向成 ``stream=False`` 后，
    ``background`` 与 ``stream`` 不再冲突，后台请求被放行交给
    ``responses.stream``——一个本应「非流式轮询」的请求被当成流式发出。
    """
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class _Never:
        @staticmethod
        async def stream(**_kw):
            return "unused"

    provider._get_client = lambda: SimpleNamespace(responses=_Never())
    provider._get_aclient = lambda: SimpleNamespace(responses=_Never())
    request = CompletionRequest(prompt="hi", background=True)

    with pytest.raises(ValueError, match="background 请求不支持流式模式"):
        result = getattr(provider, method_name)(request=request)
        if is_async:
            asyncio.run(result)


# --- Ark（Volcengine）：异步 Responses 流终止 -------------------------------


def test_ark_async_responses_stream_records_terminal_completed():
    """Ark 异步 Responses 收到 ``response.completed`` 的完整流必须正常收尾。

    ``terminal_seen = True`` 反向成 ``False`` 后，**已经**收到终止事件的完整流
    仍被判为「response.completed 之前结束」——一次正常响应被报成 ``RuntimeError``。
    """
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "m"
    provider._provider = "volcengine"
    provider._protocol = "responses"
    provider._options = {}
    provider._server_verified_protocols = frozenset({"responses"})

    class Responses:
        @staticmethod
        def create(**_kw):
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="hi"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed", error=None),
                    ),
                ]
            )

    provider._aclient = SimpleNamespace(responses=Responses())

    async def _collect():
        return [chunk async for chunk in provider.ainvoke("hi")]

    assert asyncio.run(_collect()) == ["hi"]


# --- 私有请求装配器的 stream 默认值（与 ``_chat_tool_call_item`` 同类） ----


def test_openai_chat_builder_defaults_to_streaming():
    """``_build_chat_request`` 的 ``stream`` 默认值必须是 ``True``。"""
    parameters = inspect.signature(OpenAICompatibleProvider._build_chat_request).parameters
    assert parameters["stream"].default is True


def test_openai_responses_builder_defaults_to_streaming():
    """``_build_responses_request`` 的 ``stream`` 默认值必须是 ``True``。"""
    parameters = inspect.signature(OpenAICompatibleProvider._build_responses_request).parameters
    assert parameters["stream"].default is True


def test_ark_chat_builder_defaults_to_streaming():
    """Ark ``_build_chat_request`` 的 ``stream`` 默认值必须是 ``True``。"""
    parameters = inspect.signature(VolcengineProvider._build_chat_request).parameters
    assert parameters["stream"].default is True


# --- reraise=True：重试耗尽后抛原始异常，不是 tenacity 的 RetryError --------
#
# 同 ``test_retry_helpers_reraise_original_error``：``reraise=True`` 反向成
# ``False`` 后，最后一次异常被包成 ``RetryError``——调用方按
# ``TimeoutError``／``ConnectionError`` 写的错误分类接不住。这里覆盖各
# Provider 的 ``embed_documents``／``aembed_documents``／``complete`` 装饰器
# （每一处都是**独立**的 ``@retry(...)``，必须逐个看守）。


def _patch_retry_sleep(monkeypatch, decorated):
    """掏掉退避等待（tenacity 的 ``sleep`` 是 retry 对象的实例属性）。"""
    monkeypatch.setattr(decorated.retry, "sleep", lambda _seconds: None)


@pytest.mark.parametrize(
    "decorated",
    [
        OpenAICompatibleProvider.embed_documents,
        OpenAICompatibleProvider.aembed_documents,
        GoogleProvider.embed_documents,
        GoogleProvider.aembed_documents,
        VolcengineProvider.complete,
        VolcengineProvider.embed_documents,
        VolcengineProvider.aembed_documents,
    ],
    ids=["oc.embed", "oc.aembed", "g.embed", "g.aembed", "v.complete", "v.embed", "v.aembed"],
)
def test_provider_retry_decorators_reraise_original_error(decorated):
    """每一处 ``@retry(reraise=True)`` 耗尽后都必须抛出原始异常。"""
    assert decorated.retry.reraise is True


def test_openai_embed_reraise_surfaces_timeout_error(monkeypatch):
    """行为面确认：``embed_documents`` 耗尽后抛 ``TimeoutError`` 而非 ``RetryError``。"""
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "m"
    provider._provider = "openai"
    provider._options = {}
    _patch_retry_sleep(monkeypatch, OpenAICompatibleProvider.embed_documents)

    class Embeddings:
        @staticmethod
        def create(**_kw):
            raise TimeoutError("boom")

    provider._get_client = lambda: SimpleNamespace(embeddings=Embeddings())

    with pytest.raises(TimeoutError):
        provider.embed_documents(["a"])


def test_google_embed_reraise_surfaces_timeout_error(monkeypatch):
    """行为面确认：Google ``embed_documents`` 耗尽后抛 ``TimeoutError``。"""
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-2.0-flash"
    provider._options = {}
    _patch_retry_sleep(monkeypatch, GoogleProvider.embed_documents)

    class Models:
        @staticmethod
        def embed_content(**_kw):
            raise TimeoutError("boom")

    class Client:
        vertexai = False
        models = Models()

    provider._get_client = lambda: Client()

    with pytest.raises(TimeoutError):
        provider.embed_documents(["a"])
