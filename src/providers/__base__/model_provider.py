import asyncio
import base64
import binascii
import inspect
import json
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.security import (
    validate_secret_free_options as _validate_secret_free_options,
)
from src.utils.security import (
    validate_secret_free_payload as _validate_secret_free_payload,
)
from src.utils.security import (
    validate_secret_free_request_overrides as _validate_secret_free_request_overrides,
)
from src.utils.security import (
    validate_secret_free_resource_args as _validate_secret_free_resource_args,
)
from src.utils.security import (
    validate_secret_free_resource_kwargs as _validate_secret_free_resource_kwargs,
)

_UNSET_TEMPERATURE = object()


@dataclass
class CompletionRequest:
    """统一聊天请求。

    `prompt` 和 `system_prompt` 保留旧版便捷调用；需要多轮或多模态输入时
    使用 `messages`。供应商专属参数放在 `extra_body` 或 Provider 配置的
    `options` 中，适配器会在请求边界进行校验和转换。未显式传入
    `system_prompt` 时沿用兼容默认值 `You are a helpful assistant.`；
    如需完全不带系统提示，请显式传入 `None`。
    """

    prompt: str | None = None
    system_prompt: str | None = "You are a helpful assistant."
    messages: Sequence[Mapping[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    stream: bool = True
    # Keep the historical public default while distinguishing an omitted
    # value from an explicit ``temperature=0.7`` for model-level defaults.
    temperature: float | None = _UNSET_TEMPERATURE  # type: ignore[assignment]
    _temperature_explicit: bool = dataclass_field(
        init=False, default=False, repr=False, compare=False
    )
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    n: int | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    modalities: list[str] | None = None
    audio: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    web_search_options: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | Sequence[str] | None = None
    response_format: dict[str, Any] | None = None
    tool_choice: Any | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    timeout: float | None = None
    # OpenAI Responses/Ark Responses 的会话与推理参数。其它 Provider 会在
    # 请求边界忽略不适用字段，但不会把它们误传给各自的 SDK。
    previous_response_id: str | None = None
    # OpenAI Responses input token count 的模型风格预设。当前 SDK 仅在
    # `responses.input_tokens.count` 支持该字段，普通 Responses 创建会显式拒绝。
    personality: str | None = None
    include: Sequence[str] | None = None
    reasoning: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    store: bool | None = None
    background: bool | None = None
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None = None
    conversation: str | dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    caching: dict[str, Any] | None = None
    expire_at: int | None = None
    prompt_cache_options: dict[str, Any] | None = None
    safety_identifier: str | None = None
    moderation: dict[str, Any] | None = None
    verbosity: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    service_tier: str | None = None
    truncation: str | None = None
    stream_options: dict[str, Any] | None = None
    # Anthropic Messages/Beta Messages 的资源级请求字段。它们保留在统一
    # 请求对象中，只有对应 Provider 会转发到 SDK。
    cache_control: dict[str, Any] | None = None
    container: str | dict[str, Any] | None = None
    inference_geo: str | None = None
    mcp_servers: Sequence[Mapping[str, Any]] | None = None
    output_config: dict[str, Any] | None = None
    output_format: dict[str, Any] | None = None
    speed: str | None = None
    betas: Sequence[str] | None = None
    diagnostics: dict[str, Any] | None = None
    fallback_credit_token: str | None = None
    fallbacks: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """在统一请求边界校验并复制 HTTP 扩展，避免凭证绕过 Provider。"""

        self._temperature_explicit = self.temperature is not _UNSET_TEMPERATURE
        if not self._temperature_explicit:
            self.temperature = 0.7

        headers, query = validate_secret_free_request_overrides(
            self.extra_headers,
            self.extra_query,
            "CompletionRequest",
        )
        self.extra_headers = headers  # type: ignore[assignment]
        self.extra_query = query
        if self.extra_body is not None:
            self.extra_body = _validate_secret_free_payload(
                self.extra_body,
                "CompletionRequest",
                "extra_body",
            )

    @property
    def temperature_is_explicit(self) -> bool:
        """返回调用方是否显式提供了温度参数。"""
        return self._temperature_explicit

    def copy_with(self, **changes: Any) -> "CompletionRequest":
        """复制请求并保留隐式温度的显式性标记。

        ``dataclasses.replace`` 会把已经规范化的 ``0.7`` 再当成显式值，
        从而覆盖模型级 temperature。Provider 在切换 stream 或调整其它
        请求字段时应使用此方法，避免复制过程改变调用方语义。
        """
        if not self._temperature_explicit and "temperature" not in changes:
            changes["temperature"] = _UNSET_TEMPERATURE
        return replace(self, **changes)

    def to_invoke_kwargs(self) -> dict[str, Any]:
        """转换为 Provider `invoke` 的关键字参数。"""
        values = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "tools": self.tools,
            "stream": self.stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "n": self.n,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "modalities": self.modalities,
            "audio": self.audio,
            "prediction": self.prediction,
            "web_search_options": self.web_search_options,
            "seed": self.seed,
            "stop": self.stop,
            "response_format": self.response_format,
            "tool_choice": self.tool_choice,
            "extra_body": self.extra_body,
            "extra_headers": self.extra_headers,
            "extra_query": self.extra_query,
            "metadata": self.metadata,
            "user": self.user,
            "timeout": self.timeout,
            "previous_response_id": self.previous_response_id,
            "personality": self.personality,
            "include": self.include,
            "reasoning": self.reasoning,
            "thinking": self.thinking,
            "store": self.store,
            "background": self.background,
            "parallel_tool_calls": self.parallel_tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "conversation": self.conversation,
            "session": self.session,
            "context_management": self.context_management,
            "caching": self.caching,
            "expire_at": self.expire_at,
            "prompt_cache_options": self.prompt_cache_options,
            "safety_identifier": self.safety_identifier,
            "moderation": self.moderation,
            "verbosity": self.verbosity,
            "prompt_cache_key": self.prompt_cache_key,
            "prompt_cache_retention": self.prompt_cache_retention,
            "service_tier": self.service_tier,
            "truncation": self.truncation,
            "stream_options": self.stream_options,
            "cache_control": self.cache_control,
            "container": self.container,
            "inference_geo": self.inference_geo,
            "mcp_servers": self.mcp_servers,
            "output_config": self.output_config,
            "output_format": self.output_format,
            "speed": self.speed,
            "betas": self.betas,
            "diagnostics": self.diagnostics,
            "fallback_credit_token": self.fallback_credit_token,
            "fallbacks": self.fallbacks,
        }
        if not self._temperature_explicit:
            values.pop("temperature", None)
        return {key: value for key, value in values.items() if value is not None}


@dataclass
class CompletionResult:
    """非流式或聚合后的统一聊天结果。"""

    text: str = ""
    tool_calls: list[dict[str, Any]] = dataclass_field(default_factory=list)
    usage: dict[str, Any] = dataclass_field(default_factory=dict)
    finish_reason: str | None = None
    response_id: str | None = None
    refusal: str | None = None
    reasoning: str = ""
    raw: Any = None


@dataclass
class StreamEvent:
    """统一流式事件。

    `invoke`/`ainvoke` 仍只返回文本，以兼容现有聊天界面；需要工具调用、
    usage、reasoning 或完成原因时使用 `stream_events`/`astream_events`。
    """

    type: str
    text: str = ""
    reasoning: str = ""
    tool_call: dict[str, Any] | None = None
    usage: dict[str, Any] = dataclass_field(default_factory=dict)
    finish_reason: str | None = None
    response_id: str | None = None
    refusal: str | None = None
    raw: Any = None


def merge_tool_call_fragment(
    accumulator: dict[Any, dict[str, Any]],
    tool_call: Mapping[str, Any],
    fallback_index: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """累积 OpenAI/Anthropic 风格的流式工具调用片段。

    供应商通常只在第一个片段发送 id/name，后续片段只发送 index 和
    partial arguments。统一在适配层保留这些字段，避免调用方自行拼接。
    """
    index = tool_call.get("index", fallback_index)
    call_id = tool_call.get("id") or tool_call.get("call_id")
    if index is not None:
        key: Any = ("index", index)
    elif call_id:
        key = ("id", call_id)
    else:
        key = ("position", len(accumulator))

    merged = accumulator.setdefault(
        key,
        {
            "index": index,
            "type": tool_call.get("type", "function") or "function",
            "arguments": "",
        },
    )
    if index is not None:
        merged["index"] = index
    for name in ("id", "call_id", "type", "name"):
        value = tool_call.get(name)
        if value is not None and value != "":
            merged[name] = value

    fragment = tool_call.get("arguments")
    if isinstance(fragment, str):
        if fragment:
            current = merged.get("arguments", "")
            merged["arguments"] = (current if isinstance(current, str) else "") + fragment
    elif fragment is not None and not merged.get("arguments"):
        merged["arguments"] = fragment
    return key, merged


_CHAT_TOOL_CALL_TYPES = frozenset({"function"})


def _chat_tool_call_item(
    tool_call: Mapping[str, Any],
    location: str,
    *,
    require_call_id: bool = False,
    keep_responses_type: bool = False,
) -> dict[str, Any]:
    """将工具调用归一化为 OpenAI Chat Completions 的嵌套结构。

    ``CompletionResult.tool_calls`` 使用扁平的 ``{"id", "type", "name",
    "arguments"}`` 形状，而 Chat Completions 要求 assistant 历史的
    ``tool_calls`` 使用 ``{"id", "type", "function": {"name", "arguments"}}``。
    两种输入都在这里统一，调用方可以把上一轮的 ``tool_calls`` 原样回填。

    历史工具调用并不总是带 ``id``：Gemini 的 ``FunctionCall.id`` 是可选字段，
    SDK 默认 ``None``，其 ``_contents`` 也允许无 id 的调用。因此默认只在调用方
    确实提供 id 时保留该字段，不用本地错误打断本可发送的请求；只有确实要求
    ``id`` 的协议（Chat Completions）才传入 ``require_call_id=True`` 显式校验。
    """
    if not isinstance(tool_call, Mapping):
        raise ValueError(f"{location} 必须是对象。")
    normalized = dict(tool_call)

    function = normalized.get("function")
    if function is not None and not isinstance(function, Mapping):
        raise ValueError(f"{location}.function 必须是对象。")
    function_values = function if isinstance(function, Mapping) else normalized

    name = function_values.get("name") or normalized.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{location} 缺少函数名。")
    arguments = function_values.get("arguments")
    if arguments is None:
        arguments = normalized.get("arguments")
    if arguments is None:
        # ``input`` 是 Responses ``custom_tool_call`` 与 Anthropic ``tool_use``
        # 的参数别名，回退顺序与 Responses 适配保持一致。
        arguments = normalized.get("input")
    # 复用统一助手：非字符串参数按 JSON 文本发送，None 保持空串，
    # 与 Responses 适配和流式工具调用使用同一套规则。
    arguments = normalize_tool_arguments(arguments)

    call_id = normalized.get("id") or normalized.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        if require_call_id:
            raise ValueError(f"{location} 缺少 id/call_id。")
        call_id = None

    # Chat Completions 的 ``tool_calls[].type`` 只接受 ``function``；Responses 的
    # ``function_call``/``custom_tool_call`` 与流式片段都要映射回该取值。
    source_type = normalized.get("type")
    tool_type = source_type if source_type in _CHAT_TOOL_CALL_TYPES else "function"

    item: dict[str, Any] = {
        "type": tool_type,
        "function": {"name": name, "arguments": arguments},
    }
    if call_id is not None:
        item["id"] = call_id
    # Responses 需要区分 ``custom_tool_call``（其 ``input`` 是任意文本，不做
    # JSON 规范化），但 Chat Completions 只有 ``function`` 一种取值。调用方若
    # 来自 Responses 路径，就在类型被改写时记录原值，避免多轮 custom tool
    # 回填被误当 function_call。默认不记录，防止该内部字段进入 Chat 请求体。
    if keep_responses_type and source_type != tool_type:
        item["_responses_tool_call_type"] = source_type
    # Responses 用 ``call_id`` 关联 ``function_call`` 与 ``function_call_output``，
    # 而流式事件会同时给出 ``id``（输出项 ID）与 ``call_id``（调用 ID）两个不同
    # 的值。Chat 归一化只保留 ``id``，会把 ``call_id`` 丢掉，导致回填 Responses
    # 时输出项配不上调用。这里在 Responses 路径上一并保留。
    if keep_responses_type:
        source_call_id = normalized.get("call_id")
        if isinstance(source_call_id, str) and source_call_id.strip():
            item["call_id"] = source_call_id
    return item


def normalize_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    location: str = "messages",
    *,
    require_tool_call_ids: bool = False,
    keep_responses_type: bool = False,
) -> list[dict[str, Any]]:
    """归一化 Chat Completions 消息，转换工具调用并保留其它字段。

    ``keep_responses_type`` 由 Responses 路径开启，用于保留
    ``custom_tool_call`` 这类 Chat Completions 不支持的原始调用类型；
    Chat 路径保持默认关闭，避免内部字段进入请求体。
    """
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{location}[{index}] 必须是对象。")
        item = dict(message)
        tool_calls = item.get("tool_calls")
        if tool_calls is not None:
            if isinstance(tool_calls, (str, bytes, bytearray)) or not isinstance(
                tool_calls, Sequence
            ):
                raise ValueError(f"{location}[{index}].tool_calls 必须是序列。")
            item["tool_calls"] = [
                _chat_tool_call_item(
                    tool_call,
                    f"{location}[{index}].tool_calls[{call_index}]",
                    require_call_id=require_tool_call_ids,
                    keep_responses_type=keep_responses_type,
                )
                for call_index, tool_call in enumerate(tool_calls)
            ]
        normalized.append(item)
    return normalized


def normalize_messages(
    prompt: str | None,
    system_prompt: str | None,
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    require_tool_call_ids: bool = False,
    keep_responses_type: bool = False,
) -> list[dict[str, Any]]:
    """生成不修改调用方对象的标准消息列表。

    ``require_tool_call_ids`` 只应由确实要求 assistant 历史工具调用带 ``id``
    的协议（Chat Completions）开启；其它协议继续接受 Gemini 这类无 id 的历史。
    """
    if messages is not None:
        if isinstance(messages, (str, bytes, bytearray)) or not isinstance(messages, Sequence):
            raise ValueError("messages 必须是消息对象序列。")
        normalized = normalize_chat_messages(
            messages,
            require_tool_call_ids=require_tool_call_ids,
            keep_responses_type=keep_responses_type,
        )
        if system_prompt and not any(message.get("role") == "system" for message in normalized):
            normalized.insert(0, {"role": "system", "content": system_prompt})
        return normalized

    if prompt is None or not str(prompt).strip():
        raise ValueError("聊天请求必须提供 prompt 或 messages。")
    result: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    if system_prompt:
        result.insert(0, {"role": "system", "content": system_prompt})
    return result


def _responses_json_text(value: Any, location: str, *, allow_empty: bool = False) -> str:
    """将工具参数或输出规范为 Responses 接受的 JSON 文本。"""
    if value is None:
        if allow_empty:
            return ""
        return "{}"
    if isinstance(value, str):
        if not value and allow_empty:
            return ""
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location} 必须是有效 JSON。") from exc
    else:
        parsed = value
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} 必须可序列化为 JSON。") from exc


def _responses_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return bool(value)
    return True


def _responses_provider_variant(provider: str) -> str:
    """规范化 Responses 目标 SDK 变体。"""
    normalized = str(provider).strip().lower().replace("-", "_")
    return "ark" if normalized in {"ark", "volcengine", "volc_engine"} else "openai"


# Ark 专属的 Responses 内容块类型；OpenAI 兼容端点不接受这些块。
_ARK_ONLY_RESPONSES_ITEM_TYPES = frozenset(
    {"input_audio", "audio_url", "input_video", "video_url"}
)


def _reject_unsupported_responses_item(
    item: Mapping[str, Any],
    location: str,
    provider: str,
) -> None:
    """拒绝经顶层 item 形态绕过 variant 守卫的供应商专属内容块。

    ``normalize_responses_input`` 对带 ``content`` 的消息走
    ``_responses_message_content``，那里会按 provider 变体拒绝 Ark 专属块。
    但原生 Responses item（``{"type": "input_audio", ...}``）没有 ``content``，
    会直通到请求体，因此需要在这里做等价检查，避免同一种块因书写位置不同
    而一个被拒、一个静默发给不支持它的端点。
    """
    if _responses_provider_variant(provider) == "ark":
        return
    item_type = item.get("type")
    if item_type in _ARK_ONLY_RESPONSES_ITEM_TYPES:
        raise ValueError(
            f"{location} 的 {item_type} 内容块当前不受 Responses SDK 支持。"
        )
    if "image_pixel_limit" in item:
        raise ValueError(
            f"{location}.image_pixel_limit 不受 OpenAI Responses SDK 支持。"
        )


def _responses_content_part(
    part: Any,
    location: str,
    *,
    provider: str = "openai",
) -> dict[str, Any]:
    """将 Chat 风格多模态内容块转换为 Responses 内容块。

    OpenAI/Ark Responses 不接受 Chat Completions 的 ``text``、
    ``image_url`` 和 ``file`` 类型；其输入内容分别使用
    ``input_text``、``input_image`` 和 ``input_file``。Ark 还支持
    ``input_audio``、``input_video`` 与 ``image_pixel_limit``，由
    ``provider`` 参数选择对应变体。不在目标 SDK 输入联合类型中的内容块
    统一显式失败，不能把无效请求静默发送到服务端。
    """
    if isinstance(part, str):
        return {"type": "input_text", "text": part}
    if not isinstance(part, Mapping):
        raise ValueError(f"{location} 必须是字符串或对象。")

    source = dict(part)
    part_type = source.get("type")
    variant = _responses_provider_variant(provider)
    if part_type in {"text", "input_text", "output_text"}:
        text = source.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{location}.text 必须是字符串。")
        converted = {"type": "input_text", "text": text}
        if "prompt_cache_breakpoint" in source and variant == "openai":
            converted["prompt_cache_breakpoint"] = source["prompt_cache_breakpoint"]
        elif "prompt_cache_breakpoint" in source:
            raise ValueError(f"{location}.prompt_cache_breakpoint 不受 Ark Responses SDK 支持。")
        return converted

    if part_type in {"audio_url", "input_audio"}:
        if variant != "ark":
            raise ValueError(
                f"{location} 的 {part_type} 内容块当前不受 Responses SDK 支持。"
            )
        audio_value = source.get("audio_url")
        if isinstance(audio_value, Mapping):
            audio_url = audio_value.get("url") or audio_value.get("audio_url")
            audio_file_id = source.get("file_id") or audio_value.get("file_id")
            chunking_strategy = source.get(
                "chunking_strategy", audio_value.get("chunking_strategy")
            )
        else:
            audio_url = audio_value
            audio_file_id = source.get("file_id")
            chunking_strategy = source.get("chunking_strategy")
        if not isinstance(audio_url, str) or not audio_url.strip():
            raise ValueError(f"{location} 缺少有效 audio_url。")
        if audio_file_id is not None and not isinstance(audio_file_id, str):
            raise ValueError(f"{location}.file_id 必须是字符串。")
        if chunking_strategy is not None and not isinstance(chunking_strategy, Mapping):
            raise ValueError(f"{location}.chunking_strategy 必须是对象。")
        converted_audio: dict[str, Any] = {"type": "input_audio", "audio_url": audio_url}
        if audio_file_id:
            converted_audio["file_id"] = audio_file_id
        if chunking_strategy is not None:
            converted_audio["chunking_strategy"] = dict(chunking_strategy)
        return converted_audio

    if part_type in {"video_url", "input_video"}:
        if variant != "ark":
            raise ValueError(
                f"{location} 的 {part_type} 内容块当前不受 Responses SDK 支持。"
            )
        video_value = source.get("video_url")
        if isinstance(video_value, Mapping):
            video_url = video_value.get("url") or video_value.get("video_url")
            video_file_id = source.get("file_id") or video_value.get("file_id")
            fps = source.get("fps", video_value.get("fps"))
        else:
            video_url = video_value
            video_file_id = source.get("file_id")
            fps = source.get("fps")
        if video_url is not None and not isinstance(video_url, str):
            raise ValueError(f"{location}.video_url.url 必须是字符串。")
        if video_file_id is not None and not isinstance(video_file_id, str):
            raise ValueError(f"{location}.file_id 必须是字符串。")
        if not video_url and not video_file_id:
            raise ValueError(f"{location} 缺少 video_url 或 file_id。")
        if fps is not None and (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
        ):
            raise ValueError(f"{location}.fps 必须是有限数字。")
        converted_video: dict[str, Any] = {"type": "input_video"}
        if video_url:
            converted_video["video_url"] = video_url
        if video_file_id:
            converted_video["file_id"] = video_file_id
        if fps is not None:
            converted_video["fps"] = fps
        return converted_video

    if part_type == "image_url":
        image_value = source.pop("image_url", None)
        image_url: Any = None
        image_file_id: Any = source.get("file_id")
        detail: Any = source.get("detail")
        image_pixel_limit: Any = source.get("image_pixel_limit")
        if isinstance(image_value, Mapping):
            image_url = image_value.get("url") or image_value.get("image_url")
            image_file_id = image_file_id or image_value.get("file_id")
            detail = detail or image_value.get("detail")
            image_pixel_limit = image_pixel_limit or image_value.get("image_pixel_limit")
        elif isinstance(image_value, str):
            image_url = image_value
        elif image_value is not None:
            raise ValueError(f"{location}.image_url 必须是字符串或对象。")
        fallback_url = source.pop("url", None)
        image_url = image_url or fallback_url
        if image_url is not None and not isinstance(image_url, str):
            raise ValueError(f"{location}.image_url.url 必须是字符串。")
        if image_file_id is not None and not isinstance(image_file_id, str):
            raise ValueError(f"{location}.file_id 必须是字符串。")
        if not image_url and not image_file_id:
            raise ValueError(f"{location} 缺少 image_url 或 file_id。")
        if detail is None:
            detail = "auto"
        allowed_details = {"auto", "low", "high"}
        if variant == "openai":
            allowed_details.add("original")
        if not isinstance(detail, str) or detail not in allowed_details:
            allowed = "、".join(sorted(allowed_details))
            raise ValueError(f"{location}.detail 必须是 {allowed}。")
        converted_image: dict[str, Any] = {"type": "input_image", "detail": detail}
        if image_url:
            converted_image["image_url"] = image_url
        if image_file_id:
            converted_image["file_id"] = image_file_id
        if image_pixel_limit is not None:
            if variant != "ark":
                raise ValueError(f"{location}.image_pixel_limit 不受 OpenAI Responses SDK 支持。")
            if not isinstance(image_pixel_limit, Mapping):
                raise ValueError(f"{location}.image_pixel_limit 必须是对象。")
            converted_image["image_pixel_limit"] = dict(image_pixel_limit)
        if "prompt_cache_breakpoint" in source and variant == "openai":
            converted_image["prompt_cache_breakpoint"] = source["prompt_cache_breakpoint"]
        elif "prompt_cache_breakpoint" in source:
            raise ValueError(f"{location}.prompt_cache_breakpoint 不受 Ark Responses SDK 支持。")
        return converted_image

    if part_type == "input_image":
        image_value = source.get("image_url")
        if isinstance(image_value, Mapping):
            image_url = image_value.get("url") or image_value.get("image_url")
            file_id = source.get("file_id") or image_value.get("file_id")
            if source.get("detail") is None and image_value.get("detail") is not None:
                source["detail"] = image_value["detail"]
            if (
                source.get("image_pixel_limit") is None
                and image_value.get("image_pixel_limit") is not None
            ):
                source["image_pixel_limit"] = image_value["image_pixel_limit"]
            if image_url is not None:
                source["image_url"] = image_url
            else:
                source.pop("image_url", None)
            if file_id is not None:
                source["file_id"] = file_id
        if source.get("image_url") is not None and not isinstance(source["image_url"], str):
            raise ValueError(f"{location}.image_url 必须是字符串。")
        if source.get("file_id") is not None and not isinstance(source["file_id"], str):
            raise ValueError(f"{location}.file_id 必须是字符串。")
        if not source.get("image_url") and not source.get("file_id"):
            raise ValueError(f"{location} 缺少 image_url 或 file_id。")
        if source.get("detail") is None:
            source["detail"] = "auto"
        allowed_details = {"auto", "low", "high"}
        if variant == "openai":
            allowed_details.add("original")
        if source["detail"] not in allowed_details:
            allowed = "、".join(sorted(allowed_details))
            raise ValueError(f"{location}.detail 必须是 {allowed}。")
        converted_input_image: dict[str, Any] = {"type": "input_image", "detail": source["detail"]}
        if source.get("image_pixel_limit") is not None:
            if variant != "ark":
                raise ValueError(f"{location}.image_pixel_limit 不受 OpenAI Responses SDK 支持。")
            if not isinstance(source["image_pixel_limit"], Mapping):
                raise ValueError(f"{location}.image_pixel_limit 必须是对象。")
            converted_input_image["image_pixel_limit"] = dict(source["image_pixel_limit"])
        for key in ("image_url", "file_id"):
            if source.get(key) is not None:
                converted_input_image[key] = source[key]
        if "prompt_cache_breakpoint" in source and variant == "openai":
            converted_input_image["prompt_cache_breakpoint"] = source["prompt_cache_breakpoint"]
        elif "prompt_cache_breakpoint" in source:
            raise ValueError(f"{location}.prompt_cache_breakpoint 不受 Ark Responses SDK 支持。")
        return converted_input_image

    if part_type == "file":
        nested_file = source.pop("file", None)
        nested: dict[str, Any] = {}
        if isinstance(nested_file, Mapping):
            nested.update(nested_file)
        elif isinstance(nested_file, str):
            if nested_file.startswith(("http://", "https://")):
                nested["file_url"] = nested_file
            else:
                nested["file_id"] = nested_file
        elif nested_file is not None:
            raise ValueError(f"{location}.file 必须是字符串或对象。")
        for key in ("file_id", "file_data", "file_url", "filename", "detail"):
            if key not in source and key in nested:
                source[key] = nested[key]
        if "file_url" not in source and "url" in nested:
            source["file_url"] = nested["url"]
        if "file_url" not in source and isinstance(source.get("url"), str):
            source["file_url"] = source["url"]
        if not any(source.get(key) for key in ("file_id", "file_data", "file_url")):
            raise ValueError(
                f"{location} 缺少 file_id、file_data 或 file_url。"
            )
        for key in ("file_id", "file_data", "file_url", "filename"):
            if source.get(key) is not None and not isinstance(source[key], str):
                raise ValueError(f"{location}.{key} 必须是字符串。")
        converted = {"type": "input_file"}
        if source.get("detail") is not None:
            if variant != "openai" or source["detail"] not in {"auto", "low", "high"}:
                raise ValueError(f"{location}.detail 不受 {variant.title()} Responses SDK 支持。")
            converted["detail"] = source["detail"]
        for key in ("file_id", "file_data", "file_url", "filename"):
            if source.get(key) is not None:
                converted[key] = source[key]
        if "prompt_cache_breakpoint" in source and variant == "openai":
            converted["prompt_cache_breakpoint"] = source["prompt_cache_breakpoint"]
        elif "prompt_cache_breakpoint" in source:
            raise ValueError(f"{location}.prompt_cache_breakpoint 不受 Ark Responses SDK 支持。")
        return converted

    if part_type == "input_file":
        for key in ("file_id", "file_data", "file_url", "filename"):
            if source.get(key) is not None and not isinstance(source[key], str):
                raise ValueError(f"{location}.{key} 必须是字符串。")
        if not any(source.get(key) for key in ("file_id", "file_data", "file_url")):
            raise ValueError(
                f"{location} 缺少 file_id、file_data 或 file_url。"
            )
        converted = {"type": "input_file"}
        if source.get("detail") is not None:
            if variant != "openai" or source["detail"] not in {"auto", "low", "high"}:
                raise ValueError(f"{location}.detail 不受 {variant.title()} Responses SDK 支持。")
            converted["detail"] = source["detail"]
        for key in ("file_id", "file_data", "file_url", "filename"):
            if source.get(key) is not None:
                converted[key] = source[key]
        if "prompt_cache_breakpoint" in source and variant == "openai":
            converted["prompt_cache_breakpoint"] = source["prompt_cache_breakpoint"]
        elif "prompt_cache_breakpoint" in source:
            raise ValueError(f"{location}.prompt_cache_breakpoint 不受 Ark Responses SDK 支持。")
        return converted

    if not isinstance(part_type, str) or not part_type.strip():
        raise ValueError(f"{location}.type 必须是非空字符串。")
    raise ValueError(
        f"{location} 的内容类型 {part_type} 不受 Responses SDK 支持。"
    )


def _responses_message_content(
    value: Any,
    location: str,
    *,
    provider: str = "openai",
) -> Any:
    """复制并归一化 Responses 消息的 content 字段。"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{location}.content 必须是字符串或内容块序列。")
    return [
        _responses_content_part(
            part,
            f"{location}.content[{index}]",
            provider=provider,
        )
        for index, part in enumerate(value)
    ]


def _responses_function_call_item(
    tool_call: Mapping[str, Any],
    location: str,
    *,
    require_call_id: bool = False,
) -> dict[str, Any]:
    """将 Chat Completions 工具调用转换为 Responses function_call item。"""
    explicit_call_id = tool_call.get("call_id")
    call_id = explicit_call_id
    if call_id is None and not require_call_id:
        call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError(f"{location} 缺少 call_id/id。")

    function = tool_call.get("function")
    if function is not None and not isinstance(function, Mapping):
        raise ValueError(f"{location}.function 必须是对象。")
    function_values = function if isinstance(function, Mapping) else tool_call
    name = function_values.get("name") or tool_call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{location} 缺少函数名。")

    raw_arguments = function_values.get(
        "arguments",
        tool_call.get("arguments", tool_call.get("input")),
    )
    # Responses 的 custom tool 按 SDK 契约接受任意文本（``input: str``，无 JSON
    # 约束），Provider 自己产出的 ``custom_tool_call`` 也带任意文本。若把它当
    # function_call 强制 json.loads，多轮 custom tool 回填会在请求边界直接失败。
    # 因此按类型分流：custom 保留原始文本，function 才做 JSON 规范化。
    #
    # 优先取 _chat_tool_call_item 保留的原始类型：``type`` 已被归一化为 Chat
    # Completions 唯一允许的 ``function``，直接读它会丢掉 custom 语义。
    source_type = tool_call.get("_responses_tool_call_type") or tool_call.get("type")
    if source_type == "custom_tool_call":
        custom_input = "" if raw_arguments is None else str(raw_arguments)
        item: dict[str, Any] = {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name,
            "input": custom_input,
        }
    else:
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": _responses_json_text(raw_arguments, f"{location}.arguments"),
        }
    # A Chat Completions ``id`` is the call identifier, not the Responses
    # output-item identifier. Do not duplicate it as ``id`` unless the caller
    # supplied a distinct Responses-style ``call_id`` explicitly.
    if explicit_call_id is not None and tool_call.get("id") is not None:
        item["id"] = tool_call["id"]
    if tool_call.get("status") is not None:
        item["status"] = tool_call["status"]
    return item


def _responses_function_output_item(
    message: Mapping[str, Any],
    location: str,
    *,
    require_call_id: bool = False,
    provider: str = "openai",
) -> dict[str, Any]:
    call_id = message.get("call_id")
    if call_id is None and not require_call_id:
        call_id = message.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError(f"{location} 缺少 tool_call_id/call_id。")
    output = message.get("output", message.get("content"))
    if output is None:
        output_text = ""
    elif isinstance(output, str):
        output_text = output
    elif isinstance(output, Sequence) and not isinstance(output, (bytes, bytearray)):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [
                _responses_content_part(
                    part,
                    f"{location}.output[{index}]",
                    provider=provider,
                )
                for index, part in enumerate(output)
            ],
            **(
                {"status": message["status"]}
                if "status" in message and message["status"] is not None
                else {}
            ),
        }
    else:
        output_text = _responses_json_text(output, f"{location}.output")
    item: dict[str, Any] = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output_text,
    }
    if "status" in message and message["status"] is not None:
        item["status"] = message["status"]
    return item


def normalize_responses_input(
    prompt: str | None,
    system_prompt: str | None,
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    provider: str = "openai",
) -> str | list[dict[str, Any]]:
    """将统一消息转换为 OpenAI/Ark Responses 输入项。

    Chat Completions 的 assistant.tool_calls 与 role=tool 不是 Responses
    输入消息类型，必须分别变成 function_call 和 function_call_output。
    原生 Responses item 会保留，但其关键字段仍在边界处校验。``provider``
    为 ``ark``/``volcengine`` 时启用 Ark 独有的音视频内容块。

    Responses 输入项要求每个 ``function_call`` 带 ``call_id``，因此这里强制
    assistant 历史的工具调用提供 id；缺少关联 ID 会在请求前显式报错。
    """
    normalized = normalize_messages(
        prompt,
        system_prompt,
        messages,
        require_tool_call_ids=True,
        # Responses 需要区分 custom_tool_call 与 function_call：前者的 input
        # 是任意文本，不做 JSON 规范化。Chat 路径不开启该开关。
        keep_responses_type=True,
    )
    if messages is None and prompt is not None:
        return prompt

    converted: list[dict[str, Any]] = []
    for index, message in enumerate(normalized):
        location = f"Responses messages[{index}]"
        message_type = message.get("type")
        if message_type == "function_call":
            converted.append(
                _responses_function_call_item(
                    message,
                    location,
                    require_call_id=True,
                )
            )
            continue
        if message_type == "function_call_output":
            converted.append(
                _responses_function_output_item(
                    message,
                    location,
                    require_call_id=True,
                    provider=provider,
                )
            )
            continue

        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls is not None:
            if (
                isinstance(tool_calls, (str, bytes, bytearray))
                or not isinstance(tool_calls, Sequence)
            ):
                raise ValueError(f"{location}.tool_calls 必须是工具调用序列。")
            content = message.get("content")
            if _responses_has_content(content):
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": _responses_message_content(
                        content,
                        location,
                        provider=provider,
                    ),
                }
                for key in ("type", "status", "phase"):
                    if key in message:
                        assistant_message[key] = message[key]
                converted.append(assistant_message)
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    raise ValueError(
                        f"{location}.tool_calls[{call_index}] 必须是对象。"
                    )
                converted.append(
                    _responses_function_call_item(
                        tool_call,
                        f"{location}.tool_calls[{call_index}]",
                    )
                )
            continue
        if role == "tool":
            converted.append(
                _responses_function_output_item(
                    message,
                    location,
                    provider=provider,
                )
            )
            continue
        if role == "function":
            raise ValueError(f"{location} 的 function 消息缺少 tool_call_id。")

        normalized_message = dict(message)
        if "content" in normalized_message:
            normalized_message["content"] = _responses_message_content(
                normalized_message["content"],
                location,
                provider=provider,
            )
        else:
            # 原生 Responses item 形态（例如 ``{"type": "input_audio", ...}``）
            # 不带 role/content，因此不会经过 _responses_message_content 的
            # 内容块转换。同一个 Ark 专属块写成 content 会被拒、写成顶层 item
            # 却静默发出，等于绕过了 variant 守卫，因此这里补一次等价检查。
            _reject_unsupported_responses_item(normalized_message, location, provider)
        converted.append(normalized_message)
    return converted



def normalize_usage(usage: Any) -> dict[str, Any]:
    """将常见供应商 usage 结构规范为统一 token 字段。"""
    if usage is None:
        return {}
    def usage_key(prefix: str) -> str:
        return f"{prefix}_tokens"

    aliases = {
        "prompt_token_count": usage_key("input"),
        "candidates_token_count": usage_key("output"),
        "total_token_count": usage_key("total"),
        "input_token_count": usage_key("input"),
        "output_token_count": usage_key("output"),
        "cached_content_token_count": usage_key("cached"),
        "thoughts_token_count": usage_key("reasoning"),
        "tool_use_prompt_token_count": usage_key("tool_use_prompt"),
    }
    result: dict[str, Any] = {}
    for source, target in aliases.items():
        value = field(usage, source)
        if value is not None:
            result[target] = value
    for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
        "output_tokens", "cached_tokens", "reasoning_tokens",
        "tool_use_prompt_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens",
    ):
        value = field(usage, key)
        if value is not None:
            result[key] = value
    input_details = field(usage, "input_tokens_details")
    output_details = field(usage, "output_tokens_details")
    if (value := field(input_details, "cached_tokens")) is not None:
        result.setdefault("cached_tokens", value)
    if (value := field(output_details, "reasoning_tokens")) is not None:
        result.setdefault("reasoning_tokens", value)
    return result


def extract_system_prompt(messages: Sequence[Mapping[str, Any]] | None) -> str | None:
    """提取消息列表中的 system 内容，供无 system role 的 SDK 使用。"""
    if not messages:
        return None
    values = [content_to_text(message.get("content", "")) for message in messages if message.get("role") == "system"]
    value = "\n\n".join(item for item in values if item)
    return value or None


def content_to_text(content: Any) -> str:
    """从常见 SDK 内容块中提取文本，保留未知块的显式空结果。"""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        return "".join(content_to_text(item) for item in content)
    return ""


def normalize_tool_arguments(value: Any) -> str:
    """将 Responses 工具参数统一为 JSON 字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("工具调用 arguments 必须可序列化为 JSON。") from exc


def validate_complete_tool_call(tool_call: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """校验流式工具调用在发送 completed 事件前已经完整。"""
    name = tool_call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"{provider} 工具调用缺少 name，流提前结束。")
    tool_type = tool_call.get("type", "function_call")
    arguments = tool_call.get("arguments")
    # Responses custom tools intentionally accept arbitrary text, including an
    # empty input. Function calls still require a complete JSON arguments value.
    if tool_type != "custom_tool_call" and (arguments is None or arguments == ""):
        raise RuntimeError(f"{provider} 工具调用 {name} 缺少完整 arguments，流提前结束。")
    if tool_type == "function_call" and isinstance(arguments, str):
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{provider} 工具调用 {name} 的 arguments 不是完整 JSON。"
            ) from exc
    return dict(tool_call)


def normalize_embedding_vector(value: Any) -> list[float]:
    """将 SDK 返回的 list 或 OpenAI base64 embedding 统一为浮点列表。"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    elif isinstance(value, str):
        try:
            encoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Embedding base64 内容无效。") from exc
    else:
        raise TypeError(f"Embedding 返回了不支持的类型: {type(value).__name__}")

    if not encoded or len(encoded) % 4:
        raise ValueError("Embedding base64 内容不是有效的 float32 字节序列。")
    count = len(encoded) // 4
    return list(struct.unpack(f"<{count}f", encoded))


def field(value: Any, name: str, default: Any = None) -> Any:
    """兼容 Pydantic SDK 对象和字典响应。"""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def is_retryable_error(error: BaseException) -> bool:
    """只对连接、限流和服务端错误重试，避免参数/认证错误重复发送请求。"""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status_values = (
        field(error, "status_code"),
        field(error, "code"),
        field(field(error, "response"), "status_code"),
    )
    for status in status_values:
        if isinstance(status, bool):
            continue
        if isinstance(status, int):
            return status == 429 or status >= 500
        if isinstance(status, str) and status.isdigit():
            numeric_status = int(status)
            return numeric_status == 429 or numeric_status >= 500
    name = error.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "ratelimit", "internalserver", "serviceunavailable"))


def reject_unsupported_kwargs(provider: str, values: Mapping[str, Any]) -> None:
    """拒绝适配器未实现的非空请求字段，避免参数被静默丢弃。"""
    unsupported = sorted(key for key, value in values.items() if value is not None)
    if unsupported:
        raise ValueError(f"{provider} 不支持请求参数: {', '.join(unsupported)}")


def validate_secret_free_options(
    options: Mapping[str, Any] | None,
    provider: str,
) -> dict[str, Any]:
    """校验直接构造 Provider 时的模型级选项不会携带凭证。"""
    return _validate_secret_free_options(options, provider)


def validate_secret_free_request_overrides(
    extra_headers: Mapping[str, Any] | None = None,
    extra_query: Mapping[str, Any] | None = None,
    provider: str = "Provider",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """校验并复制请求级 HTTP 扩展。"""

    return _validate_secret_free_request_overrides(
        extra_headers,
        extra_query,
        provider,
    )


def coerce_completion_request(
    request: CompletionRequest | None,
    kwargs: Mapping[str, Any],
    operation: str = "请求",
) -> CompletionRequest:
    """规范化统一请求，并拒绝混用对象参数和关键字参数。

    两种调用形式分别用于不同的调用场景。混用时如果静默丢弃关键字参数，
    调用方看到的请求和实际发送的请求会不一致，因此在进入 SDK 前显式失败。
    """
    if request is not None and kwargs:
        raise ValueError(f"{operation} 不能同时传 request 和关键字参数。")
    if request is not None:
        return request
    return CompletionRequest(**dict(kwargs))


async def iterate_async(value: Any) -> AsyncGenerator[Any, None]:
    """统一消费 SDK 的异步流和同步迭代器返回值。

    官方异步客户端通常返回 ``AsyncIterable``，但某些 SDK 版本、兼容网关
    或测试替身会直接返回普通迭代器。两者都应保持同一异步 Provider 契约。
    """
    try:
        if isinstance(value, AsyncIterable) or hasattr(value, "__aiter__"):
            async for item in value:
                yield item
            return
        if isinstance(value, Iterable):
            for item in value:
                yield item
            return
        raise TypeError("SDK 流式响应既不是异步迭代器也不是同步迭代器。")
    finally:
        close = getattr(value, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        else:
            close = getattr(value, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result


T = TypeVar("T")
_STREAM_END = object()


class StreamEventError(RuntimeError):
    """SSE 错误事件，保留 HTTP 状态供首帧安全重试判断。"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def raise_for_stream_error_event(event: Any, provider: str) -> None:
    """将协议层 error 事件转换为带状态码的异常。"""
    event_type = field(event, "type", "")
    embedded_error = field(event, "error")
    if event_type not in {"error", "response.error"} and embedded_error is None:
        return
    error = embedded_error or event
    pending: list[Any] = [error]
    visited: set[int] = set()
    message: str | None = None
    while pending and not message:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, str) and current.strip():
            message = current.strip()
            break
        if isinstance(current, Mapping):
            values = [current.get(key) for key in ("message", "detail", "reason", "error", "body")]
        else:
            values = [field(current, key) for key in ("message", "detail", "reason", "error", "body")]
        for value in values:
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
        if not message:
            pending.extend(value for value in values if value is not None)
    if not message:
        message = field(error, "code") or f"{provider} 流返回错误"
    status: int | None = None
    for value in (
        field(error, "status_code"),
        field(error, "status"),
        field(event, "status_code"),
        field(event, "status"),
    ):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            status = value
            break
        if isinstance(value, str) and value.isdigit():
            status = int(value)
            break
    raise StreamEventError(f"{provider} 流返回错误: {message}", status)


def retry_sync_call(call: Callable[[], T]) -> T:
    """在生成器消费阶段重试同步 SDK 请求建立。"""
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    ):
        with attempt:
            return call()
    raise RuntimeError("同步 SDK 请求重试未返回结果。")


async def retry_async_call(call: Callable[[], Any]) -> Any:
    """在生成器消费阶段重试异步 SDK 请求建立。"""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    ):
        with attempt:
            result = call()
            return await result if inspect.isawaitable(result) else result
    raise RuntimeError("异步 SDK 请求重试未返回结果。")


def retry_sync_stream(
    factory: Callable[[], Any],
    first_event_validator: Callable[[Any], None] | None = None,
) -> Generator[Any, None, None]:
    """只重试同步流在首个事件前的建立阶段，避免重复已输出内容。"""
    def open_and_peek() -> tuple[Any, Any]:
        iterator = iter(factory())
        try:
            first = next(iterator)
            if first is not _STREAM_END and first_event_validator is not None:
                first_event_validator(first)
            return iterator, first
        except StopIteration:
            return iterator, _STREAM_END
        except BaseException:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            raise

    iterator, first = retry_sync_call(open_and_peek)
    try:
        if first is not _STREAM_END:
            yield first
        yield from iterator
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


async def retry_async_stream(
    factory: Callable[[], Any],
    first_event_validator: Callable[[Any], None] | None = None,
) -> AsyncGenerator[Any, None]:
    """只重试异步流在首个事件前的建立阶段，避免重复已输出内容。"""
    async def open_and_peek() -> tuple[Any, Any]:
        response = factory()
        response = await response if inspect.isawaitable(response) else response
        iterator = iterate_async(response).__aiter__()
        try:
            first = await iterator.__anext__()
            if first is not _STREAM_END and first_event_validator is not None:
                first_event_validator(first)
            return iterator, first
        except StopAsyncIteration:
            return iterator, _STREAM_END
        except BaseException:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
            raise

    iterator, first = await retry_async_call(open_and_peek)
    try:
        if first is not _STREAM_END:
            yield first
        async for item in iterator:
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


@contextmanager
def retry_sync_stream_context(factory: Callable[[], Any]) -> Any:
    """重试同步流上下文的进入阶段，不重试已经开始的事件消费。"""
    def enter() -> tuple[Any, Any]:
        manager = factory()
        try:
            return manager, manager.__enter__()
        except BaseException:
            close = getattr(manager, "close", None)
            if close is not None:
                close()
            raise

    manager, response = retry_sync_call(enter)
    try:
        yield response
    except BaseException as error:
        if not manager.__exit__(type(error), error, error.__traceback__):
            raise
    else:
        manager.__exit__(None, None, None)


@asynccontextmanager
async def retry_async_stream_context(factory: Callable[[], Any]) -> Any:
    """重试异步流上下文的进入阶段，不重试已经开始的事件消费。"""
    async def enter() -> tuple[Any, Any]:
        manager = factory()
        try:
            return manager, await manager.__aenter__()
        except BaseException:
            close = getattr(manager, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
            raise

    manager, response = await retry_async_call(enter)
    try:
        yield response
    except BaseException as error:
        if not await manager.__aexit__(type(error), error, error.__traceback__):
            raise
    else:
        await manager.__aexit__(None, None, None)


_RESOURCE_GUARD_EXCLUDED = frozenset(
    {
        # These entry points already coerce and validate a CompletionRequest;
        # wrapping them would validate the same values twice and would make
        # their streaming generators harder to introspect.
        "complete",
        "acomplete",
        "invoke",
        "ainvoke",
        "stream_events",
        "astream_events",
        "embed_documents",
        "aembed_documents",
        "embed_query",
        "aembed_query",
    }
)


def _provider_label(provider: Any) -> str:
    return str(getattr(provider, "_provider", None) or type(provider).__name__)


def _provider_call_accepts_arguments(
    method: Callable[..., Any],
    provider: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    """判断 Provider 自身的调用签名是否完整。"""

    try:
        inspect.signature(method).bind(provider, *args, **dict(kwargs))
    except (TypeError, ValueError):
        return False
    return True


def _is_missing_required_argument_error(error: TypeError) -> bool:
    message = str(error).lower()
    return "missing" in message and "required" in message and "argument" in message


def _is_duplicate_argument_error(error: TypeError) -> bool:
    """识别 SDK 调用中由显式参数与 ``**kwargs`` 重复造成的 TypeError。"""

    message = str(error).lower()
    return "multiple values for" in message and (
        "argument" in message or "keyword" in message
    )


def _sanitize_resource_call(
    method: Callable[..., Any],
    provider: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """校验资源方法的显式扩展参数和 ``**kwargs``。"""

    label = f"{_provider_label(provider)} 资源"
    copied_args = _validate_secret_free_resource_args(args, label)
    copied_kwargs = dict(kwargs)
    try:
        signature = inspect.signature(method)
        bound = signature.bind_partial(provider, *copied_args, **copied_kwargs)
    except (TypeError, ValueError):
        # Dynamic SDK wrappers can expose no inspectable signature.  The
        # keyword and positional paths are still safe and preserve the
        # original call shape.
        return copied_args, _validate_secret_free_resource_kwargs(copied_kwargs, label)

    extension_names = {"extra_headers", "extra_query", "extra_body"}
    explicit = {
        name: bound.arguments[name]
        for name in extension_names
        if name in bound.arguments
    }
    if explicit:
        bound.arguments.update(
            _validate_secret_free_resource_kwargs(explicit, label)
        )
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and name in bound.arguments:
            bound.arguments[name] = _validate_secret_free_resource_kwargs(
                bound.arguments[name], label
            )
    # ``bound.args`` includes the instance we supplied for binding; the
    # wrapper passes that instance separately when invoking the original
    # function.
    return bound.args[1:], dict(bound.kwargs)


def _guard_provider_resource_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """给 Provider 资源入口安装统一的请求扩展校验。"""

    if getattr(method, "_resource_guarded", False):
        return method
    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def async_wrapper(provider: Any, *args: Any, **kwargs: Any) -> Any:
            # 凭证校验先行：它约束参数本身，与渠道是否具备该能力无关。
            # 顺序反了会让携带凭证的调用报出能力错误，把安全问题掩盖成配置问题。
            safe_args, safe_kwargs = _sanitize_resource_call(
                method, provider, args, kwargs
            )
            resource_guard = getattr(provider, "_require_provider_resource", None)
            if callable(resource_guard):
                resource_guard(method.__name__)
            provider_call_is_valid = _provider_call_accepts_arguments(
                method, provider, safe_args, safe_kwargs
            )
            try:
                return await method(provider, *safe_args, **safe_kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc):
                    raise ValueError(
                        f"{_provider_label(provider)} 资源 SDK 参数不匹配: {exc}"
                    ) from exc
                if _is_duplicate_argument_error(exc):
                    raise ValueError(
                        f"{_provider_label(provider)} 资源 SDK 参数重复: {exc}"
                    ) from exc
                if provider_call_is_valid and _is_missing_required_argument_error(exc):
                    raise ValueError(
                        f"{_provider_label(provider)} 资源 SDK 参数不匹配: {exc}"
                    ) from exc
                raise

        setattr(async_wrapper, "_resource_guarded", True)  # noqa: B010
        return async_wrapper

    @wraps(method)
    def wrapper(provider: Any, *args: Any, **kwargs: Any) -> Any:
        # 与 async_wrapper 保持同一顺序：先凭证，后能力。
        safe_args, safe_kwargs = _sanitize_resource_call(
            method, provider, args, kwargs
        )
        resource_guard = getattr(provider, "_require_provider_resource", None)
        if callable(resource_guard):
            resource_guard(method.__name__)
        provider_call_is_valid = _provider_call_accepts_arguments(
            method, provider, safe_args, safe_kwargs
        )
        try:
            return method(provider, *safe_args, **safe_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc):
                raise ValueError(
                    f"{_provider_label(provider)} 资源 SDK 参数不匹配: {exc}"
                ) from exc
            if _is_duplicate_argument_error(exc):
                raise ValueError(
                    f"{_provider_label(provider)} 资源 SDK 参数重复: {exc}"
                ) from exc
            if provider_call_is_valid and _is_missing_required_argument_error(exc):
                raise ValueError(
                    f"{_provider_label(provider)} 资源 SDK 参数不匹配: {exc}"
                ) from exc
            raise

    setattr(wrapper, "_resource_guarded", True)  # noqa: B010
    return wrapper


def _is_provider_resource_method(
    name: str, method: Any
) -> bool:
    if name.startswith("_") or name in _RESOURCE_GUARD_EXCLUDED:
        return False
    if not callable(method):
        return False
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name in {"extra_headers", "extra_query", "extra_body"}
        for parameter in parameters
    )


def close_resource_sync(resource: Any, label: str) -> None:
    """在同步生命周期中可靠释放 Provider/服务资源。"""
    close = getattr(resource, "close", None)
    if not callable(close):
        close = getattr(resource, "aclose", None)
    if not callable(close):
        return
    result = close()
    if not inspect.isawaitable(result):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_resource_close(result))
        return
    # A synchronous close invoked from an active event loop cannot block on
    # the async client without risking a nested-loop error. Close a coroutine
    # object where possible to avoid a RuntimeWarning, then fail explicitly.
    close_coroutine = getattr(result, "close", None)
    if callable(close_coroutine):
        close_coroutine()
    raise RuntimeError(f"{label} 需要异步关闭；请调用 aclose()。")


async def _await_resource_close(result: Awaitable[Any]) -> None:
    """将任意可等待关闭结果包装为 asyncio.run 可接受的协程。"""
    await result

class LargeLanguageModel(ABC):
    """语言模型抽象基类"""

    capabilities = frozenset({"chat", "stream"})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """为 Provider 的原生资源入口统一安装安全边界。"""

        super().__init_subclass__(**kwargs)
        for name, method in list(cls.__dict__.items()):
            if _is_provider_resource_method(name, method):
                setattr(cls, name, _guard_provider_resource_method(method))

    @classmethod
    def supports(cls, capability: str) -> bool:
        return capability in cls.capabilities

    def _require_provider_resource(self, method_name: str) -> None:
        """为 Provider 原生资源入口提供可选的能力门禁。"""
        return

    @property
    def sdk_client(self) -> Any:
        """返回底层同步 SDK 客户端，供原生资源调用。"""
        getter = getattr(self, "_get_client", None)
        if getter is None:
            raise AttributeError(f"{type(self).__name__} 没有同步 SDK 客户端。")
        return getter()

    @property
    def async_sdk_client(self) -> Any:
        """返回底层异步 SDK 客户端，供原生资源调用。"""
        getter = getattr(self, "_get_aclient", None)
        if getter is not None:
            client = getter()
            if client is None:
                raise AttributeError(f"{type(self).__name__} 没有异步 SDK 客户端。")
            return client
        client_getter = getattr(self, "_get_client", None)
        client = client_getter() if client_getter is not None else None
        if client is None:
            raise AttributeError(f"{type(self).__name__} 没有异步 SDK 客户端。")
        aio = getattr(client, "aio", None)
        if aio is None:
            raise AttributeError(f"{type(self).__name__} 没有异步 SDK 客户端。")
        return aio

    @abstractmethod
    def invoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """同步调用语言模型。"""

    @abstractmethod
    def ainvoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """异步调用语言模型。"""

    def complete(self, request: CompletionRequest | None = None, **kwargs: Any) -> CompletionResult:
        """兼容层：聚合文本流；具体 Provider 可覆盖以返回 tool calls/usage。"""
        request = coerce_completion_request(request, kwargs, "complete")
        # 不修改调用方传入的 dataclass；Provider 的 complete 实现通常需要
        # 将流式请求改为非流式，但请求对象可能会被复用。
        request = request.copy_with(stream=False)
        invoke_kwargs = request.to_invoke_kwargs()
        accepted = inspect.signature(self.invoke).parameters
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()):
            invoke_kwargs = {key: value for key, value in invoke_kwargs.items() if key in accepted}
        text = "".join(self.invoke(**invoke_kwargs))
        return CompletionResult(text=text)

    async def acomplete(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> CompletionResult:
        """异步兼容层；具体 Provider 可覆盖以返回完整元数据。"""
        request = coerce_completion_request(request, kwargs, "acomplete")
        request = request.copy_with(stream=False)
        chunks: list[str] = []
        invoke_kwargs = request.to_invoke_kwargs()
        accepted = inspect.signature(self.ainvoke).parameters
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()):
            invoke_kwargs = {key: value for key, value in invoke_kwargs.items() if key in accepted}
        async for chunk in self.ainvoke(**invoke_kwargs):
            chunks.append(chunk)
        return CompletionResult(text="".join(chunks))

    def stream_events(self, request: CompletionRequest | None = None, **kwargs: Any) -> Generator[StreamEvent, None, None]:
        """以统一事件形式读取流；旧 Provider 默认包装文本流。"""
        request = coerce_completion_request(request, kwargs, "stream_events")
        request = request.copy_with(stream=True)
        invoke_kwargs = request.to_invoke_kwargs()
        accepted = inspect.signature(self.invoke).parameters
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()):
            invoke_kwargs = {key: value for key, value in invoke_kwargs.items() if key in accepted}
        for chunk in self.invoke(**invoke_kwargs):
            yield StreamEvent(type="text_delta", text=chunk)

    async def astream_events(self, request: CompletionRequest | None = None, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        """异步统一事件流；旧 Provider 默认包装文本流。"""
        request = coerce_completion_request(request, kwargs, "astream_events")
        request = request.copy_with(stream=True)
        invoke_kwargs = request.to_invoke_kwargs()
        accepted = inspect.signature(self.ainvoke).parameters
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()):
            invoke_kwargs = {key: value for key, value in invoke_kwargs.items() if key in accepted}
        async for chunk in self.ainvoke(**invoke_kwargs):
            yield StreamEvent(type="text_delta", text=chunk)

    def close(self) -> None:
        """释放 Provider 持有的同步 SDK 客户端。"""
        return

    async def aclose(self) -> None:
        """释放 Provider 持有的异步 SDK 客户端。"""
        return

class TextEmbeddingModel(ABC):
    """文本向量化模型抽象基类"""

    @abstractmethod
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """同步将文档列表向量化。"""

    @abstractmethod
    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """异步将文档列表向量化。"""

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """默认的单文本查询向量化实现。"""
        vectors = self.embed_documents([text], **kwargs)
        if not vectors:
            raise ValueError("Embedding provider 返回空结果。")
        return vectors[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        vectors = await self.aembed_documents([text], **kwargs)
        if not vectors:
            raise ValueError("Embedding provider 返回空结果。")
        return vectors[0]

    def close(self) -> None:
        """释放向量模型持有的同步资源；默认实现为空操作。"""
        return

    async def aclose(self) -> None:
        """释放向量模型持有的异步资源；默认实现为空操作。"""
        return

class RerankModel(ABC):
    """Rerank模型抽象基类"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """同步对文档列表进行重排序。"""

    @abstractmethod
    async def arerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """异步对文档列表进行重排序。"""

    def close(self) -> None:
        """释放重排模型持有的同步资源；默认实现为空操作。"""
        return

    async def aclose(self) -> None:
        """释放重排模型持有的异步资源；默认实现为空操作。"""
        return
