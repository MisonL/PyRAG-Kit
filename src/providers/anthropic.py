import base64
import binascii
import inspect
import json
import re
import time
from collections.abc import AsyncGenerator, Generator, Mapping
from typing import Any

import anthropic

from src.providers.__base__.model_provider import (
    CompletionRequest,
    CompletionResult,
    LargeLanguageModel,
    StreamEvent,
    close_resource_sync,
    coerce_completion_request,
    content_to_text,
    extract_system_prompt,
    field,
    iterate_async,
    merge_tool_call_fragment,
    normalize_messages,
    normalize_usage,
    raise_for_stream_error_event,
    reject_unsupported_kwargs,
    retry_async_call,
    retry_async_stream_context,
    retry_sync_call,
    retry_sync_stream_context,
    validate_secret_free_options,
    validate_secret_free_request_overrides,
)
from src.providers.resources import AnthropicResources, AsyncAnthropicResources
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import (
    redact_sensitive_text,
    validate_secret_free_payload,
    validate_secret_free_resource_kwargs,
)

logger = get_module_logger(__name__)


class AnthropicProvider(LargeLanguageModel):
    """
    Anthropic模型提供商，处理Claude系列模型。
    已增加 tenacity 重试机制与性能传感器。
    """

    capabilities = frozenset(
        {
            "chat",
            "stream",
            "messages",
            "multimodal",
            "tools",
            "structured_output",
            "usage",
            "token_count",
            "batches",
            "files",
            "models",
            "parse",
        }
    )

    # Anthropic 同时使用两种模型命名：``claude-<family>-<major>``（如
    # ``claude-sonnet-4-6``）与 ``claude-<major>[-<minor>]-<family>``（如
    # ``claude-3-5-sonnet-20240620``）。家族名不限于 opus/sonnet/haiku——
    # 5 代引入了 ``claude-fable-5``、``claude-mythos-5`` 等新家族，因此家族段
    # 按任意字母串匹配，避免新家族漏判采样控制字段的弃用契约。
    _MODEL_VERSION_RE = re.compile(
        r"(?<![a-z0-9])claude[-_]"
        r"(?:"
        r"(?P<family_name>[a-z]+)[-_](?P<family_major>\d+)"
        r"(?:[-_.](?P<family_minor>\d{1,2}))?"
        r"(?:[-_]\d{8})?"
        r"|"
        r"(?P<version_major>\d+)"
        r"(?:[-_.](?P<version_minor>\d{1,2}))?"
        r"[-_](?:[a-z]+)"
        r"(?:[-_]\d{8})?"
        r")"
        r"(?=$|[-_.])",
        re.IGNORECASE,
    )

    # 无版本号的现役家族名（如 ``claude-mythos-preview``）。这些是 5 代命名，
    # 官方端点同样拒绝 legacy 采样控制字段。
    _UNVERSIONED_MODERN_MODEL_RE = re.compile(
        r"(?<![a-z0-9])claude[-_](?:fable|mythos)(?:[-_][a-z0-9]+)*$",
        re.IGNORECASE,
    )

    def __init__(self, model_name: str, options: dict[str, Any] | None = None):
        self._model_name = model_name
        self._options = validate_secret_free_options(options, "Anthropic")
        settings = get_settings()
        self._api_key = settings.anthropic_api_key

        if not self._api_key:
            logger.error("Anthropic API Key 未设置。")
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")

        self._client: anthropic.Anthropic | None = None
        self._aclient: anthropic.AsyncAnthropic | None = None
        logger.info(f"初始化 AnthropicProvider，模型: {model_name}")

    @property
    def resources(self) -> AnthropicResources:
        return AnthropicResources(self)

    @property
    def async_resources(self) -> AsyncAnthropicResources:
        return AsyncAnthropicResources(self)

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            validate_secret_free_options(self._options, "Anthropic")
            client_options = {
                key: value
                for key, value in self._options.items()
                if key in {"timeout", "max_retries"}
            }
            self._client = anthropic.Anthropic(api_key=self._api_key, **client_options)
        return self._client

    def _get_aclient(self) -> anthropic.AsyncAnthropic:
        if self._aclient is None:
            validate_secret_free_options(self._options, "Anthropic")
            client_options = {
                key: value
                for key, value in self._options.items()
                if key in {"timeout", "max_retries"}
            }
            self._aclient = anthropic.AsyncAnthropic(api_key=self._api_key, **client_options)
        return self._aclient

    @staticmethod
    async def _resolve_async_result(result: Any) -> Any:
        """兼容异步客户端返回的协程和异步分页器。"""
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _safe_resource_kwargs(kwargs: Mapping[str, Any], operation: str) -> dict[str, Any]:
        """校验直连资源方法的请求扩展，避免绕过 Resource Facade。"""
        return validate_secret_free_resource_kwargs(kwargs, f"Anthropic {operation}")

    @classmethod
    def _sampling_controls_deprecated(cls, model_name: str) -> bool:
        """识别 Claude 4.5+ 模型对采样控制字段的弃用契约。

        4.7 起官方端点对非默认值直接返回 400，且 Python SDK v1.0+ 已从请求
        签名移除 ``temperature``/``top_p``/``top_k``，透传会抛 ``TypeError``。
        因此识别失败会让请求直接失败，必须覆盖全部命名形式。
        """
        name = str(model_name)
        match = cls._MODEL_VERSION_RE.search(name)
        if match is None:
            # 无版本号的家族名（如 ``claude-mythos-preview``）出现在 5 代命名
            # 中，按当前主力家族处理，避免把弃用字段透传给新模型。
            return cls._UNVERSIONED_MODERN_MODEL_RE.search(name) is not None
        major = int(match.group("family_major") or match.group("version_major"))
        minor_value = match.group("family_minor") or match.group("version_minor")
        minor = int(minor_value) if minor_value is not None else 0
        return major > 4 or (major == 4 and minor >= 5)

    def supports_sampling_option(self, name: str) -> bool:
        """供上层判断当前模型是否仍接受 temperature/top_p/top_k。"""
        if name not in {"temperature", "top_p", "top_k"}:
            raise ValueError(f"Anthropic 不支持的采样字段: {name}")
        return (
            not self._sampling_controls_deprecated(self._model_name)
            or getattr(self, "_options", {}).get("allow_deprecated_sampling") is True
        )

    def _normalize_sampling_value(self, name: str, value: Any) -> Any:
        if value is None or not self._sampling_controls_deprecated(self._model_name):
            return value
        # Official Claude 4.5+ endpoints reject legacy sampling controls, but
        # deployments behind an Anthropic-compatible gateway may intentionally
        # retain them. Keep the strict default and require an explicit model
        # option before forwarding those values.
        if getattr(self, "_options", {}).get("allow_deprecated_sampling") is True:
            return value
        if name == "temperature":
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Anthropic temperature 必须是数字。") from exc
            if numeric in {0.7, 1.0}:
                # 0.7 is the application's historical default; 1.0 is the
                # backwards-compatible server value. Omitting either avoids a
                # request field that modern Claude models no longer use.
                return None
            raise ValueError("当前 Claude 模型不支持该 temperature；请省略该字段或使用 1.0。")
        if name == "top_p":
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Anthropic top_p 必须是数字。") from exc
            if numeric >= 0.99:
                return None
            raise ValueError(
                "当前 Claude 模型不支持该 top_p；仅允许省略或使用不低于 0.99 的兼容值。"
            )
        raise ValueError("当前 Claude 模型不支持 top_k；请移除该字段。")

    def _default_max_tokens(self) -> int:
        configured = getattr(self, "_options", {}).get("max_tokens")
        if configured is not None:
            if isinstance(configured, bool) or not isinstance(configured, (int, float)):
                raise ValueError("Anthropic options.max_tokens 必须是正整数。")
            configured_int = int(configured)
            if configured_int <= 0 or configured_int != configured:
                raise ValueError("Anthropic options.max_tokens 必须是正整数。")
            return configured_int
        model = str(getattr(self, "_model_name", "")).lower()
        if "claude-3-opus" in model or "claude-3-haiku" in model:
            return 4096
        if "claude-3-5" in model or "claude-3.5" in model:
            return 8192
        if self._sampling_controls_deprecated(model):
            return 16384
        return 8192

    @staticmethod
    def _user_profile_fields(
        user: str | None,
        extra_headers: Mapping[str, str] | None = None,
        betas: Any = None,
        *,
        beta: bool,
    ) -> tuple[dict[str, str] | None, Any]:
        """补齐 user profile 所需的 SDK beta 资源声明。"""
        if user is None:
            return dict(extra_headers) if extra_headers is not None else None, betas
        header = "user-profiles-2026-08-18"
        if beta:
            if isinstance(betas, str):
                values = [betas]
            else:
                values = list(betas or [])
            if header not in values:
                values.append(header)
            return dict(extra_headers) if extra_headers is not None else None, values
        headers = dict(extra_headers or {})
        existing = headers.get("anthropic-beta", "")
        values = [item.strip() for item in existing.split(",") if item.strip()]
        if header not in values:
            values.append(header)
        headers["anthropic-beta"] = ",".join(values)
        return headers, betas

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        converted: list[dict[str, Any]] = []
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                raise ValueError(f"Anthropic 工具定义[{index}] 必须是对象。")
            tool_type = tool.get("type")
            # Anthropic 的 server_* 工具不是 function schema，原样传递其
            # 专属字段，避免统一 OpenAI schema 破坏服务端工具配置。
            if isinstance(tool_type, str) and (
                tool_type.startswith("server_")
                or tool_type
                in {
                    "computer_20250124",
                    "bash_20250124",
                    "text_editor_20250124",
                }
            ):
                converted.append(dict(tool))
                continue
            function = tool.get("function", tool)
            if not isinstance(function, Mapping):
                raise ValueError(f"Anthropic 工具定义[{index}].function 必须是对象。")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Anthropic 工具定义[{index}] 缺少 function.name。")
            description = function.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise ValueError(f"Anthropic 工具定义[{index}].description 必须是字符串。")
            input_schema = function.get(
                "parameters",
                function.get("input_schema", {"type": "object", "properties": {}}),
            )
            if not isinstance(input_schema, Mapping):
                raise ValueError(
                    f"Anthropic 工具定义[{index}] 的 parameters/input_schema 必须是对象。"
                )
            converted.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": dict(input_schema),
                }
            )
        return converted

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized in {"auto", "any", "none"}:
                return {"type": normalized}
            if normalized in {"required", "force"}:
                return {"type": "any"}
            raise ValueError(f"Anthropic 不支持 tool_choice: {tool_choice}")
        if not isinstance(tool_choice, Mapping):
            raise ValueError("Anthropic tool_choice 必须是字符串或对象。")
        choice = dict(tool_choice)
        choice_type = choice.get("type")
        if choice_type in {"auto", "any", "none"}:
            return choice
        if choice_type == "tool":
            name = choice.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Anthropic tool_choice 的 tool 缺少 name。")
            return choice
        if choice_type == "function":
            function = choice.get("function", choice)
            if not isinstance(function, Mapping):
                raise ValueError("Anthropic tool_choice.function 必须是对象。")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Anthropic tool_choice 的 function 缺少 name。")
            return {"type": "tool", "name": name}
        if choice_type is None and "name" in choice:
            name = choice.get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "tool", "name": name}
        raise ValueError("Anthropic tool_choice 必须是 auto、any、none、tool 或 function 对象。")

    @staticmethod
    def _convert_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        converted = []
        for part in content:
            if not isinstance(part, dict):
                converted.append(part)
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                converted.append({"type": "text", "text": part.get("text", "")})
            elif part_type in {"image_url", "input_image"}:
                image_url = part.get("image_url", part.get("url"))
                if isinstance(image_url, Mapping):
                    image_url = image_url.get("url", image_url.get("image_url"))
                if not isinstance(image_url, str) or not image_url.strip():
                    raise ValueError("Anthropic 图片内容缺少有效 image_url。")
                image_url = image_url.strip()
                if image_url.startswith("data:"):
                    if "," not in image_url:
                        raise ValueError("Anthropic 图片 data URI 缺少逗号分隔符。")
                    header, data = image_url.split(",", 1)
                    header_parts = header.split(";", 1)
                    media_type = header_parts[0].removeprefix("data:")
                    if (
                        not media_type
                        or len(header_parts) == 1
                        or header_parts[1].lower() != "base64"
                    ):
                        raise ValueError(
                            "Anthropic 图片 data URI 必须包含 MIME 类型和 base64 标记。"
                        )
                    try:
                        base64.b64decode(data, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError("Anthropic 图片 data URI 不是有效 Base64。") from exc
                    converted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                else:
                    converted.append({"type": "image", "source": {"type": "url", "url": image_url}})
            elif part_type == "tool_result":
                converted.append(part)
            else:
                converted.append(part)
        return converted

    @classmethod
    def _convert_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                converted.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                continue
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                    raise ValueError(
                        "Anthropic 工具结果缺少 tool_call_id，无法关联对应的 tool_use。"
                    )
                tool_content = content
                if not isinstance(tool_content, (str, list)):
                    tool_content = content_to_text(tool_content)
                tool_result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
                }
                if message.get("is_error") is not None:
                    tool_result["is_error"] = bool(message["is_error"])
                pending_tool_results.append(tool_result)
                continue

            flush_tool_results()
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[Any] = []
                if content:
                    blocks.extend(
                        cls._convert_content(content)
                        if isinstance(content, list)
                        else [{"type": "text", "text": content}]
                    )
                for index, call in enumerate(message.get("tool_calls", [])):
                    if not isinstance(call, Mapping):
                        raise ValueError(f"Anthropic assistant tool_calls[{index}] 必须是对象。")
                    function = call.get("function", call)
                    if not isinstance(function, Mapping):
                        raise ValueError(
                            f"Anthropic assistant tool_calls[{index}].function 必须是对象。"
                        )
                    call_id = call.get("id")
                    name = function.get("name")
                    if not isinstance(call_id, str) or not call_id.strip():
                        raise ValueError(f"Anthropic assistant tool_calls[{index}] 缺少 id。")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError(
                            f"Anthropic assistant tool_calls[{index}] 缺少 function.name。"
                        )
                    arguments = function.get("arguments", function.get("input", {}))
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Anthropic assistant tool_calls[{index}] 的 arguments 不是有效 JSON。"
                            ) from exc
                    if not isinstance(arguments, Mapping):
                        raise ValueError(
                            f"Anthropic assistant tool_calls[{index}] 的 arguments 必须是 JSON 对象。"
                        )
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": dict(arguments),
                        }
                    )
                content = blocks
            converted.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": cls._convert_content(content),
                }
            )
        flush_tool_results()
        return converted

    @staticmethod
    def _response_format_config(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
        if not response_format:
            return None
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", response_format)
            return {"type": "json_schema", "schema": schema.get("schema", schema)}
        if response_format.get("type") == "json_object":
            return {"type": "json_schema", "schema": {"type": "object"}}
        return response_format

    def _build_message_params(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        *,
        stream: bool | None = None,
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        user: str | None = None,
        timeout: float | None = None,
        cache_control: dict[str, Any] | None = None,
        container: str | dict[str, Any] | None = None,
        inference_geo: str | None = None,
        output_config: dict[str, Any] | None = None,
        output_format: dict[str, Any] | None = None,
        beta: bool = False,
        for_parse: bool = False,
        for_tool_runner: bool = False,
        context_management: dict[str, Any] | None = None,
        mcp_servers: Any = None,
        speed: str | None = None,
        betas: Any = None,
        diagnostics: dict[str, Any] | None = None,
        fallback_credit_token: str | None = None,
        fallbacks: Any = None,
        **ignored: Any,
    ) -> dict[str, Any]:
        extra_headers, extra_query = validate_secret_free_request_overrides(
            extra_headers,
            extra_query,
            "Anthropic Messages",
        )
        beta_values = {
            "context_management": context_management,
            "mcp_servers": mcp_servers,
            "speed": speed,
            "betas": betas,
            "diagnostics": diagnostics,
            "fallback_credit_token": fallback_credit_token,
            "fallbacks": fallbacks,
        }
        if seed is not None:
            raise ValueError("Anthropic Messages 不支持请求参数: seed")
        if not beta:
            ignored = {
                **ignored,
                **{key: value for key, value in beta_values.items() if value is not None},
            }
        reject_unsupported_kwargs("Anthropic Messages", ignored)
        normalized = normalize_messages(prompt, system_prompt, messages)
        params: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": max_tokens if max_tokens is not None else self._default_max_tokens(),
            "messages": self._convert_messages(normalized),
        }
        effective_system = extract_system_prompt(messages) or system_prompt
        if effective_system:
            params["system"] = effective_system
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
        if stop is not None:
            params["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if tool_choice is not None:
            params["tool_choice"] = self._convert_tool_choice(tool_choice)
        if metadata is not None:
            params["metadata"] = metadata
        if cache_control is not None:
            params["cache_control"] = cache_control
        if container is not None:
            params["container"] = container
        if inference_geo is not None:
            params["inference_geo"] = inference_geo
        if thinking is not None:
            params["thinking"] = thinking
        if service_tier is not None:
            params["service_tier"] = service_tier
        if user is not None:
            if for_parse and not beta:
                raise ValueError("Anthropic Messages parse 不支持 user；请改用 beta_parse。")
            # The SDK translates this argument into the required header. Add
            # the beta opt-in explicitly because ordinary Messages calls do
            # not otherwise advertise user-profile support.
            params["user_profile_id"] = user
            extra_headers, configured_betas = self._user_profile_fields(
                user, extra_headers, beta_values.get("betas"), beta=beta
            )
            if beta:
                beta_values["betas"] = configured_betas
        if timeout is not None:
            params["timeout"] = timeout
        if extra_headers is not None:
            params["extra_headers"] = extra_headers
        if extra_query is not None:
            params["extra_query"] = extra_query
        # Client construction options must never leak into messages.create.
        # Anthropic's SDK validates request kwargs locally and would otherwise
        # report a misleading unknown parameter error.
        # Keep this defensive filter for callers that construct a Provider
        # through ``object.__new__`` or mutate an existing options mapping
        # after initialization. Normal construction rejects these fields at
        # the configuration boundary, but they must never reach messages.create.
        client_only = {
            "base_url",
            "timeout",
            "max_retries",
            "default_headers",
            "default_query",
            "http_client",
        }
        options = {
            key: value
            for key, value in getattr(self, "_options", {}).items()
            if key not in client_only
        }
        allow_deprecated_sampling = options.pop("allow_deprecated_sampling", False)
        if not isinstance(allow_deprecated_sampling, bool):
            raise ValueError("Anthropic options.allow_deprecated_sampling 必须是布尔值。")
        configured_response_format = options.get("response_format")
        if configured_response_format is not None:
            if not isinstance(configured_response_format, Mapping):
                raise ValueError("Anthropic options.response_format 必须是对象。")
            if response_format is not None:
                raise ValueError("Anthropic response_format 与配置项重复。")
            response_format = dict(configured_response_format)
        options.pop("response_format", None)
        configured_output_format = options.pop("output_format", None)
        if for_parse:
            configured_output_format = None
        if configured_output_format is not None:
            if output_format is not None:
                raise ValueError("Anthropic output_format 与配置项重复。")
            output_format = configured_output_format
        format_config = self._response_format_config(response_format)
        if format_config:
            params["output_config"] = {"format": format_config}
        if output_config is not None:
            params["output_config"] = output_config
        if not for_parse:
            if output_format is not None and stream is not True and not (beta and for_tool_runner):
                raise ValueError(
                    "Anthropic Messages.create 不支持 output_format；请使用 parse 或 stream。"
                )
            if output_format is not None and (stream is True or (beta and for_tool_runner)):
                if isinstance(output_format, Mapping):
                    raise ValueError(
                        "Anthropic output_format 必须是 Python 类型；字典 schema 请放入 output_config。"
                    )
                params["output_format"] = output_format
        if beta:
            params.update({key: value for key, value in beta_values.items() if value is not None})
        configured_body = validate_secret_free_payload(
            options.pop("extra_body", None),
            "Anthropic Messages",
            "options.extra_body",
        )
        request_body = validate_secret_free_payload(
            extra_body,
            "Anthropic Messages",
            "extra_body",
        )
        overlap = sorted(set(configured_body).intersection(request_body))
        if overlap:
            raise ValueError(f"Anthropic extra_body 不允许覆盖配置字段: {', '.join(overlap)}")

        sampling_options = {
            key: options.pop(key, None) for key in ("temperature", "top_p", "top_k")
        }
        sampling_values = {
            "temperature": (
                temperature
                if temperature is not None
                else sampling_options["temperature"]
                if sampling_options["temperature"] is not None
                else None
            ),
            "top_p": top_p,
            "top_k": top_k,
        }
        sampling_keys = set(sampling_values)
        for key in sampling_keys:
            if key in configured_body or key in request_body:
                if sampling_options[key] is not None or sampling_values[key] is not None:
                    raise ValueError(f"Anthropic 请求字段 {key} 与 extra_body 重复。")
                continue
            value = sampling_values[key]
            if value is None:
                value = sampling_options[key]
            value = self._normalize_sampling_value(key, value)
            if value is not None:
                request_body[key] = value

        body = {**configured_body, **request_body}
        if body:
            reserved = set(params).intersection(body)
            if reserved:
                raise ValueError(
                    f"Anthropic extra_body 不允许覆盖请求字段: {', '.join(sorted(reserved))}"
                )
            params["extra_body"] = body
        for key in (
            "model",
            "messages",
            "system",
            "max_tokens",
            "tool_choice",
            "tools",
        ):
            options.pop(key, None)
        options.pop("max_tokens", None)
        option_aliases = {
            "stop": "stop_sequences",
            "user": "user_profile_id",
        }
        allowed_options = {
            "metadata",
            "cache_control",
            "container",
            "inference_geo",
            "thinking",
            "service_tier",
            "extra_headers",
            "extra_query",
            "output_config",
            "stop_sequences",
            "user_profile_id",
        }
        if beta:
            allowed_options.update(
                {
                    "context_management",
                    "mcp_servers",
                    "speed",
                    "betas",
                    "diagnostics",
                    "fallback_credit_token",
                    "fallbacks",
                }
            )
        allowed_options.update(option_aliases)
        reject_unsupported_kwargs(
            "Anthropic Messages options",
            {key: value for key, value in options.items() if key not in allowed_options},
        )
        for key, value in options.items():
            params.setdefault(option_aliases.get(key, key), value)
        return params

    @classmethod
    def _extract_result(cls, response: Any) -> CompletionResult:
        blocks = field(response, "content", []) or []
        text = "".join(
            field(block, "text", "") for block in blocks if field(block, "type") == "text"
        )
        reasoning = "".join(
            field(block, "thinking", "")
            for block in blocks
            if field(block, "type") in {"thinking", "redacted_thinking"}
        )
        tool_calls = [
            {
                "id": field(block, "id"),
                "type": "function",
                "name": field(block, "name"),
                "arguments": field(block, "input", {}),
            }
            for block in blocks
            if field(block, "type") == "tool_use"
        ]
        usage_dict = normalize_usage(field(response, "usage"))
        refusal = next(
            (
                field(block, "reason", "") or field(block, "text", "")
                for block in blocks
                if field(block, "type") == "refusal"
            ),
            None,
        )
        if not refusal:
            stop_reason = field(response, "stop_reason")
            stop_details = field(response, "stop_details") or field(
                response, "refusal_stop_details"
            )
            details_type = field(stop_details, "type")
            if stop_reason == "refusal" or details_type == "refusal":
                explanation = field(stop_details, "explanation")
                category = field(stop_details, "category")
                refusal = str(explanation or category or "Anthropic 请求被安全策略拒绝")
        return CompletionResult(
            text=text,
            tool_calls=tool_calls,
            usage=usage_dict,
            finish_reason=field(response, "stop_reason"),
            response_id=field(response, "id"),
            reasoning=reasoning,
            refusal=refusal,
            raw=response,
        )

    @classmethod
    def _raise_for_refusal(cls, response: Any) -> None:
        """将 Anthropic 的拒答/安全拦截转换为显式异常。"""
        result = cls._extract_result(response)
        if result.refusal:
            raise RuntimeError(f"Anthropic 请求被拒绝: {result.refusal}")

    @classmethod
    def _raise_for_stream_snapshot_refusal(cls, response: Any) -> None:
        """在文本流输出前检查 SDK 当前快照中的拒答。"""
        try:
            snapshot = response.current_message_snapshot
        except (AttributeError, NotImplementedError):
            return
        cls._raise_for_refusal(snapshot)

    @classmethod
    def _iter_invoke_stream_text(cls, response: Any) -> Generator[str, None, None]:
        """消费消息流文本，并保留拒答事件的可观测性。

        官方 MessageStream 暴露 ``current_message_snapshot``，文本迭代完成后
        从最终快照读取 stop_details；简单测试替身通常只提供 text_stream 或
        原始事件迭代，因此分别兼容这两种形态。
        """
        stream_iter = getattr(response, "__iter__", None)
        if callable(stream_iter):
            saw_event = False
            message_stop_seen = False
            for event in response:
                saw_event = True
                message_stop_seen = message_stop_seen or field(event, "type") == "message_stop"
                converted = cls._stream_event(event)
                if converted is None:
                    continue
                if converted.type == "refusal_delta" and converted.refusal:
                    raise RuntimeError(f"Anthropic 请求被拒绝: {converted.refusal}")
                if converted.type == "text_delta" and converted.text:
                    yield converted.text
            if saw_event:
                if not message_stop_seen:
                    raise RuntimeError("Anthropic Messages 流在 message_stop 之前结束。")
                return

        if hasattr(type(response), "current_message_snapshot"):
            cls._raise_for_stream_snapshot_refusal(response)
            for text in response.text_stream:
                cls._raise_for_stream_snapshot_refusal(response)
                if text:
                    yield text
            cls._raise_for_stream_snapshot_refusal(response)
            return

        for text in response.text_stream:
            if text:
                yield text

    @classmethod
    async def _aiter_invoke_stream_text(cls, response: Any) -> AsyncGenerator[str, None]:
        """异步版本的消息流文本消费与拒答检查。"""
        stream_iter = getattr(response, "__aiter__", None)
        if callable(stream_iter):
            saw_event = False
            message_stop_seen = False
            async for event in iterate_async(response):
                saw_event = True
                message_stop_seen = message_stop_seen or field(event, "type") == "message_stop"
                converted = cls._stream_event(event)
                if converted is None:
                    continue
                if converted.type == "refusal_delta" and converted.refusal:
                    raise RuntimeError(f"Anthropic 请求被拒绝: {converted.refusal}")
                if converted.type == "text_delta" and converted.text:
                    yield converted.text
            if saw_event:
                if not message_stop_seen:
                    raise RuntimeError("Anthropic Messages 流在 message_stop 之前结束。")
                return

        if hasattr(type(response), "current_message_snapshot"):
            cls._raise_for_stream_snapshot_refusal(response)
            async for text in iterate_async(response.text_stream):
                cls._raise_for_stream_snapshot_refusal(response)
                if text:
                    yield text
            cls._raise_for_stream_snapshot_refusal(response)
            return

        async for text in iterate_async(response.text_stream):
            if text:
                yield text

    @classmethod
    def _stream_event(cls, event: Any, response_id: str | None = None) -> StreamEvent | None:
        event_type = field(event, "type", "")
        if event_type == "message_start":
            message = field(event, "message")
            message_id = field(message, "id") or response_id
            usage = normalize_usage(field(message, "usage"))
            return StreamEvent(type="start", usage=usage, response_id=message_id, raw=event)
        if event_type == "content_block_start":
            block = field(event, "content_block")
            block_type = field(block, "type")
            if block_type == "refusal" or field(block, "refusal"):
                refusal = field(block, "refusal") or field(block, "text") or field(block, "reason")
                return StreamEvent(
                    type="refusal_delta",
                    refusal=refusal if isinstance(refusal, str) else str(refusal or ""),
                    response_id=response_id,
                    raw=event,
                )
            if block_type == "tool_use":
                initial_input = field(block, "input", {})
                if isinstance(initial_input, Mapping):
                    initial_arguments = (
                        json.dumps(initial_input, ensure_ascii=False, separators=(",", ":"))
                        if initial_input
                        else ""
                    )
                else:
                    initial_arguments = str(initial_input) if initial_input else ""
                return StreamEvent(
                    type="tool_call_delta",
                    tool_call={
                        "id": field(block, "id"),
                        "index": field(event, "index"),
                        "name": field(block, "name"),
                        "arguments": initial_arguments,
                    },
                    response_id=response_id,
                    raw=event,
                )
            if block_type in {"thinking", "redacted_thinking"}:
                thinking = field(block, "thinking", "")
                return StreamEvent(
                    type="reasoning_delta",
                    reasoning=thinking if isinstance(thinking, str) else "",
                    response_id=response_id,
                    raw=event,
                )
            return None
        if event_type == "content_block_delta":
            delta = field(event, "delta")
            delta_type = field(delta, "type")
            if delta_type == "text_delta":
                return StreamEvent(
                    type="text_delta",
                    text=field(delta, "text", ""),
                    response_id=response_id,
                    raw=event,
                )
            if delta_type == "thinking_delta":
                return StreamEvent(
                    type="reasoning_delta",
                    reasoning=field(delta, "thinking", ""),
                    response_id=response_id,
                    raw=event,
                )
            if delta_type in {"refusal_delta", "refusal"}:
                refusal = field(delta, "refusal") or field(delta, "text") or field(delta, "reason")
                return StreamEvent(
                    type="refusal_delta",
                    refusal=refusal if isinstance(refusal, str) else str(refusal or ""),
                    response_id=response_id,
                    raw=event,
                )
            if delta_type == "input_json_delta":
                return StreamEvent(
                    type="tool_call_delta",
                    tool_call={
                        "index": field(event, "index"),
                        "arguments": field(delta, "partial_json", ""),
                    },
                    response_id=response_id,
                    raw=event,
                )
            return None
        if event_type == "message_delta":
            usage = normalize_usage(field(event, "usage"))
            delta = field(event, "delta")
            reason = field(delta, "stop_reason")
            stop_details = field(delta, "stop_details") or field(delta, "refusal_stop_details")
            details_type = field(stop_details, "type")
            if reason == "refusal" or details_type == "refusal":
                explanation = field(stop_details, "explanation")
                category = field(stop_details, "category")
                refusal = str(explanation or category or "Anthropic 请求被安全策略拒绝")
                return StreamEvent(
                    type="refusal_delta",
                    refusal=refusal,
                    response_id=response_id,
                    raw=event,
                )
            if reason:
                return StreamEvent(
                    type="finish",
                    usage=usage,
                    finish_reason=str(reason),
                    response_id=response_id,
                    raw=event,
                )
            if usage:
                return StreamEvent(type="usage", usage=usage, response_id=response_id, raw=event)
            return None
        if event_type == "message_stop":
            return StreamEvent(type="finish", response_id=response_id, raw=event)
        if event_type == "error" or event_type.endswith(".error"):
            raise_for_stream_error_event(event, "Anthropic Messages")
            message = cls._stream_error_message(event)
            raise RuntimeError(redact_sensitive_text(message))
        return None

    @staticmethod
    def _stream_error_message(event: Any) -> str:
        """从 SDK 错误事件的嵌套对象中保留最具体的错误信息。"""
        pending: list[Any] = [event]
        visited: set[int] = set()
        preferred_keys = ("message", "detail", "reason", "error", "body")
        while pending:
            current = pending.pop(0)
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            if isinstance(current, str) and current.strip():
                return current.strip()
            if isinstance(current, Mapping):
                for key in preferred_keys:
                    value = current.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                pending.extend(current.get(key) for key in preferred_keys if key in current)
                continue
            for key in preferred_keys:
                value = field(current, key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if value is not None:
                    pending.append(value)
        return "Anthropic Messages 流返回错误"

    def stream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Generator[StreamEvent, None, None]:
        request = coerce_completion_request(request, kwargs, "Anthropic stream_events")
        request = request.copy_with(stream=True)
        params = self._build_message_params(**request.to_invoke_kwargs())
        if not request.stream:
            result = self.complete(request)
            if result.text:
                yield StreamEvent(
                    type="text_delta",
                    text=result.text,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.reasoning:
                yield StreamEvent(
                    type="reasoning_delta",
                    reasoning=result.reasoning,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.refusal:
                yield StreamEvent(
                    type="refusal_delta",
                    refusal=result.refusal,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            for call in result.tool_calls:
                yield StreamEvent(
                    type="tool_call_delta",
                    tool_call=call,
                    response_id=result.response_id,
                    raw=result.raw,
                )
                yield StreamEvent(
                    type="tool_call_completed",
                    tool_call=call,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.usage:
                yield StreamEvent(
                    type="usage", usage=result.usage, response_id=result.response_id, raw=result.raw
                )
            yield StreamEvent(
                type="finish",
                finish_reason=result.finish_reason,
                response_id=result.response_id,
                raw=result.raw,
            )
            return
        response_id: str | None = None
        finish_emitted = False
        message_stop_seen = False
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_tool_calls: set[Any] = set()
        with retry_sync_stream_context(
            lambda: self._get_client().messages.stream(**params)
        ) as stream:
            for event in stream:
                if field(event, "type") == "message_stop":
                    message_stop_seen = True
                converted = self._stream_event(event, response_id)
                if converted:
                    if converted.type == "tool_call_delta" and converted.tool_call:
                        merge_tool_call_fragment(tool_calls, converted.tool_call)
                    elif converted.type == "finish":
                        for key, tool_call in tool_calls.items():
                            if key not in completed_tool_calls:
                                completed_tool_calls.add(key)
                                yield StreamEvent(
                                    type="tool_call_completed",
                                    tool_call=dict(tool_call),
                                    response_id=response_id,
                                    raw=converted.raw,
                                )
                    if converted.type == "finish":
                        if finish_emitted:
                            continue
                        finish_emitted = True
                    response_id = converted.response_id or response_id
                    yield converted
            if not message_stop_seen:
                raise RuntimeError("Anthropic Messages 流在 message_stop 之前结束。")

    async def astream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        request = coerce_completion_request(request, kwargs, "Anthropic astream_events")
        request = request.copy_with(stream=True)
        params = self._build_message_params(**request.to_invoke_kwargs())
        if not request.stream:
            result = await self.acomplete(request)
            if result.text:
                yield StreamEvent(
                    type="text_delta",
                    text=result.text,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.reasoning:
                yield StreamEvent(
                    type="reasoning_delta",
                    reasoning=result.reasoning,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.refusal:
                yield StreamEvent(
                    type="refusal_delta",
                    refusal=result.refusal,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            for call in result.tool_calls:
                yield StreamEvent(
                    type="tool_call_delta",
                    tool_call=call,
                    response_id=result.response_id,
                    raw=result.raw,
                )
                yield StreamEvent(
                    type="tool_call_completed",
                    tool_call=call,
                    response_id=result.response_id,
                    raw=result.raw,
                )
            if result.usage:
                yield StreamEvent(
                    type="usage", usage=result.usage, response_id=result.response_id, raw=result.raw
                )
            yield StreamEvent(
                type="finish",
                finish_reason=result.finish_reason,
                response_id=result.response_id,
                raw=result.raw,
            )
            return
        response_id: str | None = None
        finish_emitted = False
        message_stop_seen = False
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_tool_calls: set[Any] = set()
        async with retry_async_stream_context(
            lambda: self._get_aclient().messages.stream(**params)
        ) as stream:
            async for event in iterate_async(stream):
                if field(event, "type") == "message_stop":
                    message_stop_seen = True
                converted = self._stream_event(event, response_id)
                if converted:
                    if converted.type == "tool_call_delta" and converted.tool_call:
                        merge_tool_call_fragment(tool_calls, converted.tool_call)
                    elif converted.type == "finish":
                        for key, tool_call in tool_calls.items():
                            if key not in completed_tool_calls:
                                completed_tool_calls.add(key)
                                yield StreamEvent(
                                    type="tool_call_completed",
                                    tool_call=dict(tool_call),
                                    response_id=response_id,
                                    raw=converted.raw,
                                )
                    if converted.type == "finish":
                        if finish_emitted:
                            continue
                        finish_emitted = True
                    response_id = converted.response_id or response_id
                    yield converted
            if not message_stop_seen:
                raise RuntimeError("Anthropic Messages 异步流在 message_stop 之前结束。")

    def complete(self, request: CompletionRequest | None = None, **kwargs: Any) -> CompletionResult:
        request = coerce_completion_request(request, kwargs, "Anthropic complete")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, stream=False)
        response = retry_sync_call(lambda: self._get_client().messages.create(**params))
        return self._extract_result(response)

    async def acomplete(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> CompletionResult:
        """使用 AsyncAnthropic 聚合完整结果，保留工具、thinking 和 usage。"""
        request = coerce_completion_request(request, kwargs, "Anthropic acomplete")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, stream=False)
        response = await retry_async_call(lambda: self._get_aclient().messages.create(**params))
        return self._extract_result(response)

    def parse(
        self,
        output_format: Any,
        request: CompletionRequest | None = None,
        **kwargs: Any,
    ) -> Any:
        """调用 Anthropic Messages 的原生结构化输出解析入口。"""
        if output_format is None:
            raise ValueError("Anthropic parse 必须提供 output_format。")
        request = coerce_completion_request(request, kwargs, "Anthropic parse")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        values.pop("output_format", None)
        params = self._build_message_params(**values, stream=False, for_parse=True)
        if request.output_config is None:
            params.pop("output_config", None)
        params = self._parse_params(params, beta=False)
        return self._get_client().messages.parse(output_format=output_format, **params)

    @staticmethod
    def _parse_params(params: dict[str, Any], *, beta: bool) -> dict[str, Any]:
        """只保留当前 SDK 的 Messages.parse 参数。"""
        allowed = {
            "model",
            "max_tokens",
            "messages",
            "metadata",
            "output_config",
            "output_format",
            "service_tier",
            "stop_sequences",
            "system",
            "thinking",
            "tool_choice",
            "tools",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
        if beta:
            allowed.update(
                {
                    "cache_control",
                    "container",
                    "context_management",
                    "diagnostics",
                    "fallback_credit_token",
                    "fallbacks",
                    "inference_geo",
                    "mcp_servers",
                    "speed",
                    "betas",
                    "user_profile_id",
                }
            )
        reject_unsupported_kwargs(
            "Anthropic Beta Messages parse" if beta else "Anthropic Messages parse",
            {key: value for key, value in params.items() if key not in allowed},
        )
        return {key: value for key, value in params.items() if key in allowed}

    @staticmethod
    def _validate_token_count_request(request: CompletionRequest, *, beta: bool) -> None:
        """拒绝不会被当前 Messages count_tokens SDK 使用的请求字段。"""
        allowed = {
            "prompt",
            "system_prompt",
            "messages",
            "tools",
            "tool_choice",
            "cache_control",
            "response_format",
            "output_config",
            # CompletionRequest defaults temperature for generation. It has no
            # effect on token counting, so accept the shared default without
            # forwarding it to the count_tokens endpoint.
            "thinking",
            "user",
            "extra_body",
            "extra_headers",
            "extra_query",
            "timeout",
            "stream",
            "temperature",
        }
        if request.output_format is not None:
            mode = "Beta " if beta else ""
            raise ValueError(
                f"Anthropic {mode}Messages token count 不支持 output_format；请使用 output_config。"
            )
        if beta:
            allowed.update({"context_management", "mcp_servers", "speed", "betas"})
        reject_unsupported_kwargs(
            "Anthropic Beta Messages token count" if beta else "Anthropic Messages token count",
            {key: value for key, value in request.to_invoke_kwargs().items() if key not in allowed},
        )

    def count_tokens(self, request: CompletionRequest | None = None, **kwargs: Any) -> int:
        request = coerce_completion_request(request, kwargs, "Anthropic count_tokens")
        self._validate_token_count_request(request, beta=False)
        normalized = normalize_messages(request.prompt, request.system_prompt, request.messages)
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._convert_messages(normalized),
        }
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        if effective_system:
            params["system"] = effective_system
        tools = self._convert_tools(request.tools)
        if tools:
            params["tools"] = tools
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        if request.cache_control is not None:
            params["cache_control"] = request.cache_control
        if request.user is not None:
            params["user_profile_id"] = request.user
        if request.output_config is not None:
            params["output_config"] = request.output_config
        else:
            format_config = self._response_format_config(request.response_format)
            if format_config:
                params["output_config"] = {"format": format_config}
        if request.thinking is not None:
            params["thinking"] = request.thinking
        if request.extra_body is not None:
            params["extra_body"] = validate_secret_free_payload(
                request.extra_body,
                "Anthropic Messages token count",
                "extra_body",
            )
        profile_headers, _ = self._user_profile_fields(
            request.user, request.extra_headers, beta=False
        )
        if profile_headers is not None:
            params["extra_headers"] = profile_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        if request.timeout is not None:
            params["timeout"] = request.timeout
        response = self._get_client().messages.count_tokens(**params)
        value = field(response, "input_tokens")
        if value is None:
            raise RuntimeError("Anthropic token count 响应缺少 input_tokens。")
        return int(value)

    def beta_create(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        """调用 Anthropic Beta Messages，保留 Beta 专属参数转发能力。"""
        request = coerce_completion_request(request, kwargs, "Anthropic beta_create")
        if request.output_format is not None:
            raise ValueError(
                "Anthropic Beta Messages.create 不支持 output_format；请使用 beta_parse 或 beta_stream。"
            )
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, beta=True, stream=False)
        return self._get_client().beta.messages.create(**params)

    def beta_parse(
        self, output_format: Any, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        if output_format is None:
            raise ValueError("Anthropic Beta parse 必须提供 output_format。")
        request = coerce_completion_request(request, kwargs, "Anthropic beta_parse")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        values.pop("output_format", None)
        params = self._build_message_params(**values, beta=True, stream=False, for_parse=True)
        if request.output_config is None:
            params.pop("output_config", None)
        params.pop("output_format", None)
        params = self._parse_params(params, beta=True)
        return self._get_client().beta.messages.parse(output_format=output_format, **params)

    def beta_count_tokens(self, request: CompletionRequest | None = None, **kwargs: Any) -> int:
        request = coerce_completion_request(request, kwargs, "Anthropic beta_count_tokens")
        self._validate_token_count_request(request, beta=True)
        normalized = normalize_messages(request.prompt, request.system_prompt, request.messages)
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._convert_messages(normalized),
        }
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        if effective_system:
            params["system"] = effective_system
        if request.tools:
            params["tools"] = self._convert_tools(request.tools)
        if request.cache_control is not None:
            params["cache_control"] = request.cache_control
        if request.context_management is not None:
            params["context_management"] = request.context_management
        if request.mcp_servers is not None:
            params["mcp_servers"] = request.mcp_servers
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        if request.thinking is not None:
            params["thinking"] = request.thinking
        if request.output_config is not None:
            params["output_config"] = request.output_config
        else:
            format_config = self._response_format_config(request.response_format)
            if format_config:
                params["output_config"] = {"format": format_config}
        if request.speed is not None:
            params["speed"] = request.speed
        if request.user is not None:
            params["user_profile_id"] = request.user
        _, profile_betas = self._user_profile_fields(
            request.user, request.extra_headers, request.betas, beta=True
        )
        if profile_betas is not None:
            params["betas"] = profile_betas
        if request.extra_body is not None:
            params["extra_body"] = validate_secret_free_payload(
                request.extra_body,
                "Anthropic Beta Messages token count",
                "extra_body",
            )
        if request.extra_headers is not None:
            params["extra_headers"] = request.extra_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        if request.timeout is not None:
            params["timeout"] = request.timeout
        response = self._get_client().beta.messages.count_tokens(**params)
        value = field(response, "input_tokens")
        if value is None:
            raise RuntimeError("Anthropic Beta token count 响应缺少 input_tokens。")
        return int(value)

    def beta_stream(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        """返回 Anthropic Beta Messages 流式上下文管理器。"""
        request = coerce_completion_request(request, kwargs, "Anthropic beta_stream")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, beta=True, stream=True)
        return retry_sync_stream_context(lambda: self._get_client().beta.messages.stream(**params))

    def beta_tool_runner(
        self, tools: Any, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        """创建 Anthropic Beta 工具运行器，保留 SDK 原生 runner 对象。"""
        if "compaction_control" in kwargs:
            raise ValueError("Anthropic Beta tool_runner 当前 SDK 不支持 compaction_control。")
        runner_options = {key: kwargs.pop(key) for key in ("max_iterations",) if key in kwargs}
        request = coerce_completion_request(request, kwargs, "Anthropic beta_tool_runner")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        values.pop("tools", None)
        params = self._build_message_params(
            **values, beta=True, stream=request.stream, for_tool_runner=True
        )
        params["tools"] = tools
        params["stream"] = request.stream
        params.update(runner_options)
        return self._get_client().beta.messages.tool_runner(**params)

    def create_batch(self, requests: Any, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.create(requests=requests, **kwargs)

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.retrieve(batch_id, **kwargs)

    def batch_results(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.results(batch_id, **kwargs)

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.cancel(batch_id, **kwargs)

    def list_batches(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.list(**kwargs)

    def delete_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return self._get_client().messages.batches.delete(batch_id, **kwargs)

    def upload_file(self, file: Any, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return self._get_client().files.upload(file=file, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return self._get_client().files.list(**kwargs)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return self._get_client().files.retrieve_metadata(file_id, **kwargs)

    def download_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return self._get_client().files.download(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return self._get_client().files.delete(file_id, **kwargs)

    def list_models(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "模型")
        return self._get_client().models.list(**kwargs)

    def retrieve_model(self, model_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "模型")
        return self._get_client().models.retrieve(model_id, **kwargs)

    async def async_count_tokens(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> int:
        request = coerce_completion_request(request, kwargs, "Anthropic async_count_tokens")
        self._validate_token_count_request(request, beta=False)
        normalized = normalize_messages(request.prompt, request.system_prompt, request.messages)
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._convert_messages(normalized),
        }
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        if effective_system:
            params["system"] = effective_system
        tools = self._convert_tools(request.tools)
        if tools:
            params["tools"] = tools
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        if request.cache_control is not None:
            params["cache_control"] = request.cache_control
        if request.user is not None:
            params["user_profile_id"] = request.user
        if request.output_config is not None:
            params["output_config"] = request.output_config
        format_config = self._response_format_config(request.response_format)
        if format_config and "output_config" not in params:
            params["output_config"] = {"format": format_config}
        if request.thinking is not None:
            params["thinking"] = request.thinking
        if request.extra_body is not None:
            params["extra_body"] = validate_secret_free_payload(
                request.extra_body,
                "Anthropic Messages token count",
                "extra_body",
            )
        profile_headers, _ = self._user_profile_fields(
            request.user, request.extra_headers, beta=False
        )
        if profile_headers is not None:
            params["extra_headers"] = profile_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        if request.timeout is not None:
            params["timeout"] = request.timeout
        response = await self._resolve_async_result(
            self._get_aclient().messages.count_tokens(**params)
        )
        value = field(response, "input_tokens")
        if value is None:
            raise RuntimeError("Anthropic token count 响应缺少 input_tokens。")
        return int(value)

    async def async_beta_create(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        request = coerce_completion_request(request, kwargs, "Anthropic async_beta_create")
        if request.output_format is not None:
            raise ValueError(
                "Anthropic Beta Messages.create 不支持 output_format；请使用 beta_parse 或 beta_stream。"
            )
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, beta=True, stream=False)
        return await self._resolve_async_result(self._get_aclient().beta.messages.create(**params))

    async def async_beta_parse(
        self, output_format: Any, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        if output_format is None:
            raise ValueError("Anthropic Beta parse 必须提供 output_format。")
        request = coerce_completion_request(request, kwargs, "Anthropic async_beta_parse")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, beta=True, stream=False, for_parse=True)
        if request.output_config is None:
            params.pop("output_config", None)
        params.pop("output_format", None)
        params = self._parse_params(params, beta=True)
        return await self._resolve_async_result(
            self._get_aclient().beta.messages.parse(output_format=output_format, **params)
        )

    async def async_beta_count_tokens(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> int:
        request = coerce_completion_request(request, kwargs, "Anthropic async_beta_count_tokens")
        self._validate_token_count_request(request, beta=True)
        normalized = normalize_messages(request.prompt, request.system_prompt, request.messages)
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._convert_messages(normalized),
        }
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        if effective_system:
            params["system"] = effective_system
        if request.tools:
            params["tools"] = self._convert_tools(request.tools)
        if request.cache_control is not None:
            params["cache_control"] = request.cache_control
        if request.context_management is not None:
            params["context_management"] = request.context_management
        if request.mcp_servers is not None:
            params["mcp_servers"] = request.mcp_servers
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        if request.thinking is not None:
            params["thinking"] = request.thinking
        if request.output_config is not None:
            params["output_config"] = request.output_config
        else:
            format_config = self._response_format_config(request.response_format)
            if format_config:
                params["output_config"] = {"format": format_config}
        if request.speed is not None:
            params["speed"] = request.speed
        if request.user is not None:
            params["user_profile_id"] = request.user
        _, profile_betas = self._user_profile_fields(
            request.user, request.extra_headers, request.betas, beta=True
        )
        if profile_betas is not None:
            params["betas"] = profile_betas
        if request.extra_body is not None:
            params["extra_body"] = validate_secret_free_payload(
                request.extra_body,
                "Anthropic Beta Messages token count",
                "extra_body",
            )
        if request.extra_headers is not None:
            params["extra_headers"] = request.extra_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        if request.timeout is not None:
            params["timeout"] = request.timeout
        response = await self._resolve_async_result(
            self._get_aclient().beta.messages.count_tokens(**params)
        )
        value = field(response, "input_tokens")
        if value is None:
            raise RuntimeError("Anthropic Beta token count 响应缺少 input_tokens。")
        return int(value)

    def async_beta_stream(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        """返回 AsyncAnthropic Beta 流式上下文管理器。"""
        request = coerce_completion_request(request, kwargs, "Anthropic async_beta_stream")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        params = self._build_message_params(**values, beta=True, stream=True)
        return retry_async_stream_context(
            lambda: self._get_aclient().beta.messages.stream(**params)
        )

    def async_beta_tool_runner(
        self, tools: Any, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        """创建 AsyncAnthropic Beta 工具运行器。"""
        if "compaction_control" in kwargs:
            raise ValueError("Anthropic Beta tool_runner 当前 SDK 不支持 compaction_control。")
        runner_options = {key: kwargs.pop(key) for key in ("max_iterations",) if key in kwargs}
        request = coerce_completion_request(request, kwargs, "Anthropic async_beta_tool_runner")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        values.pop("tools", None)
        params = self._build_message_params(
            **values, beta=True, stream=request.stream, for_tool_runner=True
        )
        params["tools"] = tools
        params["stream"] = request.stream
        params.update(runner_options)
        return self._get_aclient().beta.messages.tool_runner(**params)

    async def async_parse(
        self,
        output_format: Any,
        request: CompletionRequest | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步调用 Anthropic Messages 的原生结构化输出解析入口。"""
        if output_format is None:
            raise ValueError("Anthropic parse 必须提供 output_format。")
        request = coerce_completion_request(request, kwargs, "Anthropic async_parse")
        values = request.to_invoke_kwargs()
        values.pop("stream", None)
        values.pop("output_format", None)
        params = self._build_message_params(**values, stream=False, for_parse=True)
        if request.output_config is None:
            params.pop("output_config", None)
        params = self._parse_params(params, beta=False)
        return await self._resolve_async_result(
            self._get_aclient().messages.parse(output_format=output_format, **params)
        )

    def close(self) -> None:
        """释放同步和异步 Anthropic SDK 客户端。

        异步客户端在没有运行中事件循环时通过同步生命周期桥接器关闭；
        若当前已处于事件循环，则保留引用并显式要求调用 ``aclose``。
        """
        client = self._client
        async_client = self._aclient
        self._client = None
        first_error: Exception | None = None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                first_error = exc
        if async_client is not None:
            try:
                close_resource_sync(async_client, "Anthropic 异步客户端")
            except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                first_error = first_error or exc
            else:
                self._aclient = None
        else:
            self._aclient = None
        if first_error is not None:
            raise first_error

    async def aclose(self) -> None:
        async_client = self._aclient
        sync_client = self._client
        self._aclient = None
        self._client = None
        first_error: Exception | None = None
        closed_clients: set[int] = set()
        for client in (async_client, sync_client):
            if client is None or id(client) in closed_clients:
                continue
            closed_clients.add(id(client))
            try:
                result = client.close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    async def async_create_batch(self, requests: Any, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(
            self._get_aclient().messages.batches.create(requests=requests, **kwargs)
        )

    async def async_retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(
            self._get_aclient().messages.batches.retrieve(batch_id, **kwargs)
        )

    async def async_batch_results(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(
            self._get_aclient().messages.batches.results(batch_id, **kwargs)
        )

    async def async_cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(
            self._get_aclient().messages.batches.cancel(batch_id, **kwargs)
        )

    async def async_list_batches(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(self._get_aclient().messages.batches.list(**kwargs))

    async def async_delete_batch(self, batch_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "批处理")
        return await self._resolve_async_result(
            self._get_aclient().messages.batches.delete(batch_id, **kwargs)
        )

    async def async_upload_file(self, file: Any, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return await self._resolve_async_result(
            self._get_aclient().files.upload(file=file, **kwargs)
        )

    async def async_list_files(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return await self._resolve_async_result(self._get_aclient().files.list(**kwargs))

    async def async_retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return await self._resolve_async_result(
            self._get_aclient().files.retrieve_metadata(file_id, **kwargs)
        )

    async def async_download_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return await self._resolve_async_result(
            self._get_aclient().files.download(file_id, **kwargs)
        )

    async def async_delete_file(self, file_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "文件")
        return await self._resolve_async_result(self._get_aclient().files.delete(file_id, **kwargs))

    async def async_list_models(self, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "模型")
        return await self._resolve_async_result(self._get_aclient().models.list(**kwargs))

    async def async_retrieve_model(self, model_id: str, **kwargs: Any) -> Any:
        kwargs = self._safe_resource_kwargs(kwargs, "模型")
        return await self._resolve_async_result(
            self._get_aclient().models.retrieve(model_id, **kwargs)
        )

    def invoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = None,
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        user: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """同步调用 Anthropic Claude LLM (CSE Sensor)。"""
        logger.info(f"调用 Anthropic LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()

        try:
            params = self._build_message_params(
                prompt,
                system_prompt,
                tools,
                temperature,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                response_format=response_format,
                tool_choice=tool_choice,
                extra_body=extra_body,
                metadata=metadata,
                thinking=thinking,
                service_tier=service_tier,
                user=user,
                timeout=timeout,
                stream=stream,
                **kwargs,
            )

            if stream:
                with retry_sync_stream_context(
                    lambda: client.messages.stream(**params)
                ) as response:
                    yield from self._iter_invoke_stream_text(response)
            else:
                response = retry_sync_call(lambda: client.messages.create(**params))
                result = self._extract_result(response)
                if result.refusal:
                    raise RuntimeError(f"Anthropic 请求被拒绝: {result.refusal}")
                if result.text:
                    yield result.text

            duration = time.perf_counter() - start_time
            logger.info(f"Anthropic LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Anthropic LLM (%s) 出错: %s",
                self._model_name,
                error_text,
            )
            raise

    async def ainvoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = None,
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        user: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """异步调用 Anthropic Claude LLM (CSE Sensor)。"""
        logger.info(f"异步调用 Anthropic LLM ({self._model_name})，流式: {stream}")
        aclient = self._get_aclient()
        start_time = time.perf_counter()

        try:
            params = self._build_message_params(
                prompt,
                system_prompt,
                tools,
                temperature,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                response_format=response_format,
                tool_choice=tool_choice,
                extra_body=extra_body,
                metadata=metadata,
                thinking=thinking,
                service_tier=service_tier,
                user=user,
                timeout=timeout,
                stream=stream,
                **kwargs,
            )

            if stream:
                async with retry_async_stream_context(
                    lambda: aclient.messages.stream(**params)
                ) as response:
                    async for text in self._aiter_invoke_stream_text(response):
                        yield text
            else:
                response = await retry_async_call(lambda: aclient.messages.create(**params))
                result = self._extract_result(response)
                if result.refusal:
                    raise RuntimeError(f"Anthropic 请求被拒绝: {result.refusal}")
                if result.text:
                    yield result.text

            duration = time.perf_counter() - start_time
            logger.info(f"Anthropic LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Anthropic LLM (%s) 异步出错: %s",
                self._model_name,
                error_text,
            )
            raise
