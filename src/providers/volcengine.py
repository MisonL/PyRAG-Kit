import inspect
import math
from collections.abc import AsyncGenerator, Collection, Generator, Mapping
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.providers.__base__.model_provider import (
    _UNSET_TEMPERATURE,
    CompletionRequest,
    CompletionResult,
    LargeLanguageModel,
    StreamEvent,
    TextEmbeddingModel,
    close_resource_sync,
    coerce_completion_request,
    content_to_text,
    field,
    is_retryable_error,
    merge_tool_call_fragment,
    normalize_embedding_vector,
    normalize_messages,
    normalize_responses_input,
    normalize_tool_arguments,
    normalize_usage,
    raise_for_stream_error_event,
    reject_unsupported_kwargs,
    retry_async_call,
    retry_async_stream,
    retry_sync_call,
    retry_sync_stream,
    validate_complete_tool_call,
    validate_secret_free_options,
    validate_secret_free_request_overrides,
)
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.resources import ArkResources, AsyncArkResources
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import validate_secret_free_payload

logger = get_module_logger(__name__)


def _load_ark_clients() -> tuple[type[Any], type[Any]]:
    """按需加载 Ark SDK，避免导入 provider 时扫描整个生成式类型树。"""
    try:
        from volcenginesdkarkruntime import (  # type: ignore[import-untyped]
            Ark,
            AsyncArk,
        )
    except ImportError as exc:
        raise RuntimeError(
            "未安装 volcengine-python-sdk[ark]，无法使用 VolcengineProvider。"
        ) from exc
    return Ark, AsyncArk


# Responses 的 output 里，除 message 与 function_call 外还有内置工具的调用项。
# 这些项既无正文也不是 function_call，但都是 SDK ``ResponseOutputItem`` 联合的
# 正式成员，且都是「需要后续轮次」的中间态——调用方要读 output 里的工具调用
# 才能继续，因此返回空正文是正确的。
#
# 不含 ``reasoning``：它不是工具调用，本轮有摘要时 ``reasoning`` 字段非空，
# 前面 ``and not reasoning`` 已经放行；只有空摘要且无正文、无工具调用时才走到
# 失败分支，那确实是什么都没有，报错才对。把 reasoning 列入豁免会让这种响应
# 静默返回空成功，掩盖失败（项目规则禁止）。
#
# 白名单需与 SDK 联合成员保持同步，`test_ark_builtin_item_whitelist_covers_sdk_union`
# 会对差集断言。
_ARK_BUILTIN_TOOL_ITEM_TYPES = frozenset(
    {
        "web_search_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "knowledge_search_call",
        "doubao_app_call",
        "image_process",
        "agent_tool_call",
    }
)


def _ark_responses_has_builtin_tool_items(response: Any) -> bool:
    """判断响应里是否含 Ark 内置工具的调用项。"""
    for item in field(response, "output", []) or []:
        if field(item, "type") in _ARK_BUILTIN_TOOL_ITEM_TYPES:
            return True
    return False


def _ark_responses_output_text(response: Any) -> str:
    """从 Responses 的 ``output[].content[]`` 提取助手正文。

    Ark 的 ``Response`` 模型没有 ``output_text`` 字段（OpenAI SDK 把它实现为
    聚合 property），正文只在 ``output`` 项的 ``content`` 里。只取
    ``output_text``/``text`` 类型的内容块，避免把 reasoning 与 refusal 文本
    混进正文。
    """
    chunks: list[str] = []
    for item in field(response, "output", []) or []:
        for content in field(item, "content", []) or []:
            if field(content, "type") not in {"output_text", "text"}:
                continue
            piece = field(content, "text")
            if isinstance(piece, str):
                chunks.append(piece)
    return "".join(chunks)


class VolcengineProvider(LargeLanguageModel, TextEmbeddingModel):
    """火山引擎 Ark Provider，支持 Chat Completions、Responses 和 Embeddings。"""

    _ARK_CHAT_MODEL_OPTION_KEYS = frozenset(
        {
            "frequency_penalty",
            "function_call",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "reasoning_effort",
            "repetition_penalty",
            "response_format",
            "service_tier",
            "stop",
            "stream_options",
            "temperature",
            "thinking",
            "tool_choice",
            "top_logprobs",
            "top_p",
            "user",
            "extra_body",
            "top_k",
            "seed",
        }
    )
    _ARK_RESPONSES_MODEL_OPTION_KEYS = frozenset(
        {
            "caching",
            "context_management",
            "conversation",
            "expire_at",
            "extra_body",
            "frequency_penalty",
            "max_completion_tokens",
            "max_output_tokens",
            "max_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "previous_response_id",
            "presence_penalty",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "service_tier",
            "session",
            "store",
            "temperature",
            "text",
            "thinking",
            "tool_choice",
            "top_k",
            "top_p",
            "seed",
        }
    )
    _ARK_RESPONSES_CREATE_KEYS = frozenset(
        {
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "parallel_tool_calls",
            "previous_response_id",
            "thinking",
            "store",
            "caching",
            "stream",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "max_tool_calls",
            "context_management",
            "expire_at",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
            "reasoning",
            "session",
            "service_tier",
        }
    )
    _ARK_RESPONSES_RETRIEVE_KEYS = frozenset(
        {"extra_headers", "extra_query", "extra_body", "timeout"}
    )
    _ARK_RESPONSES_INPUT_ITEMS_KEYS = frozenset(
        {
            "after",
            "before",
            "include",
            "limit",
            "order",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_BATCH_MULTIMODAL_EMBEDDING_KEYS = frozenset(
        {
            "input",
            "model",
            "encoding_format",
            "dimensions",
            "instructions",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_BATCH_CHAT_KEYS = frozenset(
        {
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "presence_penalty",
            "stop",
            "temperature",
            "tools",
            "top_logprobs",
            "top_p",
            "repetition_penalty",
            "n",
            "parallel_tool_calls",
            "service_tier",
            "tool_choice",
            "response_format",
            "thinking",
            "max_completion_tokens",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_BATCH_EMBEDDING_KEYS = frozenset(
        {
            "input",
            "model",
            "encoding_format",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_BATCH_CHAT_ASYNC_KEYS = _ARK_BATCH_CHAT_KEYS
    _ARK_CONTEXT_CREATE_KEYS = frozenset(
        {
            "model",
            "messages",
            "ttl",
            "mode",
            "truncation_strategy",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_CONTEXT_COMPLETION_KEYS = frozenset(
        {
            "context_id",
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "presence_penalty",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tools",
            "top_logprobs",
            "top_p",
            "repetition_penalty",
            "n",
            "tool_choice",
            "response_format",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_CLASSIFICATION_KEYS = frozenset(
        {
            "query",
            "model",
            "labels",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_CONTENT_GENERATION_CREATE_KEYS = frozenset(
        {
            "model",
            "content",
            "safety_identifier",
            "callback_url",
            "return_last_frame",
            "service_tier",
            "execution_expires_after",
            "priority",
            "generate_audio",
            "draft",
            "camera_fixed",
            "watermark",
            "seed",
            "resolution",
            "ratio",
            "duration",
            "frames",
            "tools",
            "output_format",
            "omni_reference_task_type",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_CONTENT_GENERATION_LIST_KEYS = frozenset(
        {
            "page_num",
            "page_size",
            "status",
            "task_ids",
            "model",
            "service_tier",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_FILE_CREATE_KEYS = frozenset(
        {
            "expire_at",
            "preprocess_configs",
            "url",
            "tos",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_FILE_LIST_KEYS = frozenset(
        {
            "after",
            "limit",
            "order",
            "purpose",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_IMAGE_GENERATE_KEYS = frozenset(
        {
            "model",
            "prompt",
            "image",
            "response_format",
            "size",
            "seed",
            "guidance_scale",
            "watermark",
            "optimize_prompt",
            "optimize_prompt_options",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
            "sequential_image_generation",
            "sequential_image_generation_options",
            "tools",
            "output_format",
            "layer_decomposition",
            "stream",
        }
    )
    _ARK_BETA_CHAT_KEYS = frozenset(
        {
            "messages",
            "model",
            "response_format",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "service_tier",
            "stop",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "reasoning_effort",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )
    _ARK_BOT_CHAT_KEYS = frozenset(
        {
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "presence_penalty",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tools",
            "top_logprobs",
            "top_p",
            "repetition_penalty",
            "n",
            "parallel_tool_calls",
            "service_tier",
            "tool_choice",
            "response_format",
            "user",
            "metadata",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
    )

    capabilities = frozenset(
        {
            "chat",
            "stream",
            "messages",
            "multimodal",
            "tools",
            "structured_output",
            "usage",
            "embedding",
            "responses",
            "files",
            "batches",
            "token_count",
            "images",
            "multimodal_embedding",
            "context",
            "content_generation",
            "beta_chat",
            "bot_chat",
            "classification",
        }
    )

    def __init__(
        self, model_name: str, protocol: str = "ark", options: dict[str, Any] | None = None
    ):
        normalized = str(protocol).strip().lower().replace("-", "_")
        if normalized in {"ark", "chat", "chat_completion", "chat_completions"}:
            self._protocol = "chat_completions"
        elif normalized in {"response", "responses"}:
            self._protocol = "responses"
        else:
            raise ValueError("Volcengine 仅支持 ark/chat_completions 或 responses 协议。")
        self._model_name = model_name
        self._options = validate_secret_free_options(options, "Volcengine")
        self._options = self._normalize_client_options(self._options)
        self._server_verified_protocols = self._normalize_verified_protocols(
            self._options.get("server_verified_protocols", ())
        )
        settings = get_settings()
        self._api_key = settings.volc_access_key
        self._secret_key = getattr(settings, "volc_secret_key", None)
        self._ark_api_key = getattr(settings, "ark_api_key", None)
        self._base_url = getattr(settings, "volc_base_url", None)
        has_ak_sk = bool(self._api_key and self._secret_key)
        has_partial_ak_sk = bool(self._api_key or self._secret_key)
        if not has_ak_sk and not self._ark_api_key:
            if has_partial_ak_sk:
                raise ValueError(
                    "VolcengineProvider 需要同时配置 VOLC_ACCESS_KEY 和 "
                    "VOLC_SECRET_KEY，或改用 ARK_API_KEY。"
                )
            raise ValueError(
                "VolcengineProvider 需要配置 ARK_API_KEY，或同时配置 "
                "VOLC_ACCESS_KEY 和 VOLC_SECRET_KEY。"
            )
        self._client: Any | None = None
        self._aclient: Any | None = None

    @property
    def resources(self) -> ArkResources:
        return ArkResources(self)

    @property
    def async_resources(self) -> AsyncArkResources:
        return AsyncArkResources(self)

    def _client_options(self) -> dict[str, Any]:
        options = self._normalize_client_options(getattr(self, "_options", {}) or {})
        result: dict[str, Any] = {"base_url": self._base_url} if self._base_url else {}
        if self._api_key and self._secret_key:
            result.update({"ak": self._api_key, "sk": self._secret_key})
        elif getattr(self, "_ark_api_key", None):
            result["api_key"] = self._ark_api_key
        else:
            raise ValueError(
                "VolcengineProvider 认证配置不完整：需要 ARK_API_KEY，或同时提供 "
                "VOLC_ACCESS_KEY 和 VOLC_SECRET_KEY。"
            )
        for key in ("timeout", "max_retries", "region"):
            if key in options:
                result[key] = options[key]
        return result

    @staticmethod
    def _normalize_client_options(options: Mapping[str, Any]) -> dict[str, Any]:
        """规范化 Ark 客户端级 timeout、重试次数和 region 参数。"""
        normalized = dict(options)
        if "timeout" in normalized:
            value = normalized["timeout"]
            if isinstance(value, bool):
                raise ValueError("Volcengine options.timeout 必须是正数。")
            if isinstance(value, str):
                try:
                    value = float(value.strip())
                except ValueError as exc:
                    raise ValueError("Volcengine options.timeout 必须是正数。") from exc
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                # httpx.Timeout is accepted by the Ark SDK but cannot be
                # represented in TOML. Preserve such explicit SDK objects.
                if not hasattr(value, "connect"):
                    raise ValueError("Volcengine options.timeout 必须是正数或 httpx.Timeout。")
            elif float(value) <= 0:
                raise ValueError("Volcengine options.timeout 必须是正数。")
            else:
                normalized["timeout"] = float(value)

        if "max_retries" in normalized:
            value = normalized["max_retries"]
            if isinstance(value, bool):
                raise ValueError("Volcengine options.max_retries 必须是大于等于 0 的整数。")
            if isinstance(value, str):
                text = value.strip()
                try:
                    parsed = int(text)
                except ValueError as exc:
                    raise ValueError(
                        "Volcengine options.max_retries 必须是大于等于 0 的整数。"
                    ) from exc
                if text not in {str(parsed), f"+{parsed}"}:
                    raise ValueError("Volcengine options.max_retries 必须是大于等于 0 的整数。")
                value = parsed
            elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or value < 0:
                raise ValueError("Volcengine options.max_retries 必须是大于等于 0 的整数。")
            normalized["max_retries"] = value

        if "region" in normalized:
            value = normalized["region"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Volcengine options.region 必须是非空字符串。")
            normalized["region"] = value.strip()
        return normalized

    @classmethod
    def _normalize_verified_protocols(cls, configured: Any) -> frozenset[str]:
        """规范化 Ark 渠道的服务端协议登记。"""
        if configured is None:
            return frozenset()
        if isinstance(configured, str):
            configured = (configured,)
        elif not isinstance(configured, (list, tuple, set, frozenset)):
            raise ValueError("server_verified_protocols 必须是字符串或协议序列。")
        aliases = {
            "chat": "chat_completions",
            "completion": "chat_completions",
            "chat_completion": "chat_completions",
            "response": "responses",
        }
        supported = {"ark", "chat_completions", "responses"}
        normalized: set[str] = set()
        for value in configured:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("server_verified_protocols 中的协议必须是非空字符串。")
            protocol = value.strip().lower().replace("-", "_")
            protocol = aliases.get(protocol, protocol)
            if protocol not in supported:
                raise ValueError(
                    f"Volcengine 不支持登记服务端协议: {value}。"
                    "可选值: ark, chat_completions, responses"
                )
            normalized.add(protocol)
        return frozenset(normalized)

    # Responses 资源树下的子资源。它们的路径不含 ``responses`` 分段
    # （``volcengine.input_items.list`` 打的是 ``/responses/{id}/input_items``），
    # 只看分段会把它们漏掉；但 ``files.create`` 这类同名无关调用必须放行，
    # 因此用精确集合而不是子串匹配。
    # 这些字段出现即说明调用意图是「创建/发起 Responses 请求」，无论方法名
    # 是什么。用字段而非动词判断，避免 SDK 新增入口时漏检。
    _ARK_RESPONSES_VALIDATED_KWARGS = frozenset(
        {"input", "instructions", "caching", "model", "tools", "extra_body"}
    )

    _ARK_RESPONSES_SUBRESOURCES = frozenset(
        {
            "input_items",
            "input_tokens",
            "output_items",
        }
    )

    def _require_provider_resource(
        self, method_name: str, kwargs: Mapping[str, Any] | None = None
    ) -> None:
        """在原生资源方法触达 SDK 前校验 Ark 渠道能力与参数约束。

        ``method_name`` 可能是显式方法名（``create_response``），也可能是动态
        资源树路径（``volcengine.responses.create``）。按路径分段判断，避免把
        ``files.create`` 这类同名但无关的调用一并拦下。

        动态路径不经过 Facade，因此 Facade 上的参数校验必须在这里对同一组
        ``kwargs`` 再执行一次，否则 ``resources.responses.create(...)`` 可以
        带着 ``instructions`` 与 ``caching={"type": "enabled"}`` 直接发出。
        """
        segments = method_name.split(".")
        if (
            "responses" in segments
            or method_name.endswith("_response")
            or "response" in segments
            or any(segment in self._ARK_RESPONSES_SUBRESOURCES for segment in segments)
        ):
            self._require_responses_resource(method_name)
            # 不按动词白名单判断（``{"create", "generate"}`` 会漏掉
            # ``async_create`` 这类 SDK 演进后新增的写法）。凡是参数里出现
            # 受校验字段就执行同一套校验——这正是「动态路径与 Facade 受同一
            # 约束」的判据。
            if kwargs and self._ARK_RESPONSES_VALIDATED_KWARGS.intersection(kwargs):
                self._validate_native_response_kwargs(kwargs)

    def _require_responses_resource(self, operation: str) -> None:
        """阻止未验证的 Ark Responses 请求到达远端。"""
        if self._protocol != "responses":
            raise ValueError(f"Ark {operation} 仅适用于 responses 协议。")
        verified = getattr(self, "_server_verified_protocols", None)
        if verified is None:
            options = getattr(self, "_options", {}) or {}
            verified = self._normalize_verified_protocols(
                options.get("server_verified_protocols", ())
            )
        if "responses" not in verified:
            raise ValueError(
                f"Ark 未验证 Responses {operation} 的服务端能力；"
                "请在 provider options 中显式配置 "
                "server_verified_protocols=['responses']。"
            )

    def _get_client(self) -> Any:
        if self._client is None:
            ark_class, _ = _load_ark_clients()
            self._client = ark_class(**self._client_options())
        return self._client

    def _get_aclient(self) -> Any:
        if self._aclient is None:
            _, async_ark_class = _load_ark_clients()
            self._aclient = async_ark_class(**self._client_options())
        return self._aclient

    @staticmethod
    async def _resolve_async_result(result: Any) -> Any:
        """兼容 Ark 异步客户端的协程和同步分页器返回值。"""
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _merge_options(request: dict[str, Any], options: dict[str, Any]) -> None:
        reserved = {"model", "messages", "input", "stream", "instructions", "tools"}
        overlap = sorted(reserved.intersection(options))
        if overlap:
            raise ValueError(f"Provider options 不允许覆盖请求字段: {', '.join(overlap)}")
        for key, value in options.items():
            request.setdefault(key, value)

    @staticmethod
    def _validate_model_options(
        options: Mapping[str, Any],
        allowed: frozenset[str],
        endpoint: str,
    ) -> None:
        unknown = {key: value for key, value in options.items() if key not in allowed}
        reject_unsupported_kwargs(endpoint, unknown)

    def _request_options(self) -> dict[str, Any]:
        """返回不会误传给 Ark 请求的模型级选项。"""
        validate_secret_free_options(getattr(self, "_options", {}) or {}, "Volcengine")
        return {
            key: value
            for key, value in self._options.items()
            if key
            not in {
                "timeout",
                "max_retries",
                "region",
                "server_verified_protocols",
            }
        }

    @staticmethod
    def _validated_extra_body(
        value: Any,
        endpoint: str,
        reserved: set[str],
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{endpoint} extra_body 必须是对象。")
        value = validate_secret_free_payload(value, endpoint, "extra_body")
        overlap = sorted(set(value).intersection(reserved))
        if overlap:
            raise ValueError(f"{endpoint} extra_body 不允许覆盖请求字段: {', '.join(overlap)}")
        return dict(value)

    @staticmethod
    def _merge_extra_body(
        configured: Mapping[str, Any] | None,
        extensions: Mapping[str, Any] | None,
        endpoint: str,
    ) -> dict[str, Any]:
        """合并模型级扩展，拒绝同一字段的静默覆盖。"""
        configured_values = dict(configured or {})
        extension_values = dict(extensions or {})
        overlap = sorted(set(configured_values).intersection(extension_values))
        if overlap:
            raise ValueError(f"{endpoint} 模型 options 的扩展字段重复: {', '.join(overlap)}")
        return {**configured_values, **extension_values}

    @staticmethod
    def _convert_responses_format(response_format: dict[str, Any]) -> dict[str, Any]:
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", response_format)
            if not isinstance(schema, dict):
                raise ValueError("Ark Responses 的 json_schema 必须是对象。")
            return {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("name", "response"),
                        "schema": schema.get("schema", response_format.get("schema", {})),
                        "strict": schema.get("strict", True),
                    },
                }
            }
        if response_format.get("type") == "json_object":
            return {"format": {"type": "json_object"}}
        if "format" in response_format:
            return response_format
        raise ValueError("Ark Responses 的 response_format 仅支持 json_schema 或 json_object。")

    @staticmethod
    def _convert_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """将统一的 Chat 工具 schema 转换为 Ark Responses 工具 schema。"""
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                raise ValueError(f"Ark Responses 工具定义[{index}] 必须是对象。")
            if "function" not in tool:
                native_tool = dict(tool)
                tool_type = native_tool.get("type")
                if not isinstance(tool_type, str) or not tool_type.strip():
                    raise ValueError(f"Ark Responses 工具定义[{index}] 缺少有效 type。")
                if tool_type == "function":
                    name = native_tool.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError(f"Ark Responses 工具定义[{index}] 缺少 name。")
                    parameters = native_tool.get("parameters", {"type": "object", "properties": {}})
                    if not isinstance(parameters, Mapping):
                        raise ValueError(f"Ark Responses 工具定义[{index}].parameters 必须是对象。")
                    native_tool["parameters"] = dict(parameters)
                converted.append(native_tool)
                continue
            function = tool["function"]
            if not isinstance(function, Mapping):
                raise ValueError(f"Ark Responses 工具定义[{index}].function 必须是对象。")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ark Responses 工具定义缺少 function.name。")
            parameters = function.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, Mapping):
                raise ValueError(
                    f"Ark Responses 工具定义[{index}].function.parameters 必须是对象。"
                )
            item: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": dict(parameters),
            }
            if function.get("description") is not None:
                item["description"] = function["description"]
            if "strict" in function or "strict" in tool:
                item["strict"] = function.get("strict", tool.get("strict"))
            converted.append(item)
        return converted

    @staticmethod
    def _convert_responses_tool_choice(tool_choice: Any) -> Any:
        """将 OpenAI 风格的工具选择转换为 Ark Responses schema。"""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            aliases = {"required": "required", "auto": "auto", "none": "none"}
            normalized = tool_choice.lower()
            if normalized in aliases:
                return aliases[normalized]
            raise ValueError(f"Ark Responses 不支持 tool_choice: {tool_choice}")
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function = tool_choice.get("function", tool_choice)
                if not isinstance(function, Mapping):
                    raise ValueError("Ark Responses 的 function tool_choice 必须是对象。")
                name = function.get("name")
                if not name:
                    raise ValueError("Ark Responses 的 function tool_choice 缺少 name。")
                return {"type": "function", "name": name}
            if tool_choice.get("type") in {"mcp", "web_search", "knowledge_search"}:
                return dict(tool_choice)
        raise ValueError("Ark Responses 的 tool_choice 必须是 auto、none、required 或工具对象。")

    def _build_chat_request(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        stream: bool = True,
        *,
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        user: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        service_tier: str | None = None,
        thinking: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        timeout: float | None = None,
        **ignored: Any,
    ) -> dict[str, Any]:
        reject_unsupported_kwargs("Ark Chat Completions", ignored)
        extra_headers, extra_query = validate_secret_free_request_overrides(
            extra_headers,
            extra_query,
            "Ark Chat Completions",
        )
        request: dict[str, Any] = {
            "model": self._model_name,
            # Ark Chat Completions 同样要求 assistant 历史的 tool_call 带 id。
            "messages": normalize_messages(
                prompt,
                system_prompt,
                messages,
                require_tool_call_ids=True,
            ),
            "stream": stream,
        }
        configured_options = self._request_options()
        self._validate_model_options(
            configured_options,
            self._ARK_CHAT_MODEL_OPTION_KEYS,
            "Ark Chat Completions",
        )
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Ark Chat Completions",
            {"model", "messages", "input", "stream"},
        )
        configured_max_tokens = configured_options.pop("max_tokens", None)
        configured_max_completion_tokens = configured_options.pop("max_completion_tokens", None)
        if configured_max_tokens is not None and configured_max_completion_tokens is not None:
            raise ValueError(
                "Ark Chat Completions options 不能同时设置 max_tokens 和 max_completion_tokens。"
            )
        configured_top_k = configured_options.pop("top_k", None)
        configured_seed = configured_options.pop("seed", None)
        configured_scalar_extra_keys: set[str] = set()
        if configured_top_k is not None:
            if "top_k" in configured_extra_body:
                raise ValueError("Ark Chat Completions 模型 options 的 top_k 与 extra_body 重复。")
            configured_extra_body["top_k"] = configured_top_k
            configured_scalar_extra_keys.add("top_k")
        if configured_seed is not None:
            if "seed" in configured_extra_body:
                raise ValueError("Ark Chat Completions 模型 options 的 seed 与 extra_body 重复。")
            configured_extra_body["seed"] = configured_seed
            configured_scalar_extra_keys.add("seed")
        self._merge_options(request, configured_options)
        if temperature is None and "temperature" not in request:
            request["temperature"] = 0.7
        if max_tokens is None:
            if configured_max_tokens is not None:
                request["max_tokens"] = configured_max_tokens
            elif configured_max_completion_tokens is not None:
                request["max_completion_tokens"] = configured_max_completion_tokens
        if configured_extra_body:
            request["extra_body"] = dict(configured_extra_body)
        if temperature is not None:
            request["temperature"] = temperature
        if tools:
            request["tools"] = tools
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if top_p is not None:
            request["top_p"] = top_p
        for key, value in {
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "n": n,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "parallel_tool_calls": parallel_tool_calls,
            "service_tier": service_tier,
            "thinking": thinking,
            "reasoning_effort": (reasoning or {}).get("effort")
            if isinstance(reasoning, dict)
            else None,
            "stream_options": stream_options,
            "timeout": timeout,
            "extra_headers": extra_headers,
            "extra_query": extra_query,
        }.items():
            if value is not None:
                request[key] = value
        if top_k is not None:
            if "top_k" in configured_extra_body and "top_k" not in configured_scalar_extra_keys:
                raise ValueError("Ark Chat Completions top_k 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["top_k"] = top_k
        if seed is not None:
            if "seed" in configured_extra_body and "seed" not in configured_scalar_extra_keys:
                raise ValueError("Ark Chat Completions seed 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["seed"] = seed
        if stop is not None:
            request["stop"] = stop
        if response_format is not None:
            request["response_format"] = response_format
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if user is not None:
            request["user"] = user
        request_extra_body = self._validated_extra_body(
            extra_body,
            "Ark Chat Completions",
            set(request).difference({"extra_body"}),
        )
        if request_extra_body:
            configured_overlap = sorted(
                set(request.get("extra_body", {})).intersection(request_extra_body)
            )
            if configured_overlap:
                raise ValueError(
                    "Ark Chat Completions extra_body 与模型 options 重复: "
                    + ", ".join(configured_overlap)
                )
            request.setdefault("extra_body", {}).update(request_extra_body)
        if request.get("extra_body"):
            request["extra_body"] = self._validated_extra_body(
                request["extra_body"],
                "Ark Chat Completions",
                set(request).difference({"extra_body"}),
            )
        return request

    def _build_responses_request(self, request: CompletionRequest) -> dict[str, Any]:
        validate_secret_free_request_overrides(
            getattr(request, "extra_headers", None),
            getattr(request, "extra_query", None),
            "Ark Responses",
        )
        unsupported_fields = (
            "repetition_penalty",
            "n",
            "logit_bias",
            "logprobs",
            "modalities",
            "audio",
            "prediction",
            "web_search_options",
            "stop",
            "include",
            "background",
            "metadata",
            "user",
            "moderation",
            "prompt_cache_options",
            "top_logprobs",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "truncation",
            "stream_options",
            "cache_control",
            "container",
            "inference_geo",
            "mcp_servers",
            "output_config",
            "output_format",
            "speed",
            "betas",
            "diagnostics",
            "fallback_credit_token",
            "fallbacks",
        )
        reject_unsupported_kwargs(
            "Ark Responses",
            {
                key: getattr(request, key)
                for key in unsupported_fields
                if getattr(request, key, None) is not None
            },
        )
        normalized_input = normalize_responses_input(
            request.prompt,
            request.system_prompt,
            request.messages,
            provider="ark",
        )
        params: dict[str, Any] = {
            "model": self._model_name,
            "input": normalized_input,
            "stream": request.stream,
        }
        configured_options = self._request_options()
        self._validate_model_options(
            configured_options,
            self._ARK_RESPONSES_MODEL_OPTION_KEYS,
            "Ark Responses",
        )
        configured_max_tokens = configured_options.pop("max_tokens", None)
        configured_max_completion_tokens = configured_options.pop("max_completion_tokens", None)
        configured_max_output_tokens = configured_options.pop("max_output_tokens", None)
        configured_limits = {
            key: value
            for key, value in {
                "max_output_tokens": configured_max_output_tokens,
                "max_tokens": configured_max_tokens,
                "max_completion_tokens": configured_max_completion_tokens,
            }.items()
            if value is not None
        }
        if len(configured_limits) > 1:
            raise ValueError(
                "Ark Responses options 不能同时设置多个输出长度字段: "
                + ", ".join(configured_limits)
            )
        configured_response_format = configured_options.pop("response_format", None)
        configured_reasoning_effort = configured_options.pop("reasoning_effort", None)
        configured_frequency_penalty = configured_options.pop("frequency_penalty", None)
        configured_presence_penalty = configured_options.pop("presence_penalty", None)
        configured_tool_choice = configured_options.pop("tool_choice", None)
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Ark Responses",
            {"model", "messages", "input", "stream", "instructions", "tools"},
        )
        configured_top_k = configured_options.pop("top_k", None)
        configured_seed = configured_options.pop("seed", None)
        configured_conversation = configured_options.pop("conversation", None)
        configured_session = configured_options.pop("session", None)
        if configured_conversation is not None and configured_session is not None:
            raise ValueError("Ark Responses 模型 options 不能同时设置 conversation 和 session。")
        configured_scalar_extra_keys: set[str] = set()
        if configured_top_k is not None:
            if "top_k" in configured_extra_body:
                raise ValueError("Ark Responses 模型 options 的 top_k 与 extra_body 重复。")
            configured_extra_body["top_k"] = configured_top_k
            configured_scalar_extra_keys.add("top_k")
        if configured_seed is not None:
            if "seed" in configured_extra_body:
                raise ValueError("Ark Responses 模型 options 的 seed 与 extra_body 重复。")
            configured_extra_body["seed"] = configured_seed
            configured_scalar_extra_keys.add("seed")
        if request.system_prompt and request.messages is None:
            params["instructions"] = request.system_prompt
        if getattr(request, "_temperature_explicit", True) and request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_output_tokens"] = request.max_tokens
        elif configured_limits:
            params["max_output_tokens"] = next(iter(configured_limits.values()))
        if request.top_p is not None:
            params["top_p"] = request.top_p
        sampling_extensions = {
            "frequency_penalty": (
                request.frequency_penalty
                if request.frequency_penalty is not None
                else configured_frequency_penalty
            ),
            "presence_penalty": (
                request.presence_penalty
                if request.presence_penalty is not None
                else configured_presence_penalty
            ),
        }
        for key, value in sampling_extensions.items():
            if value is None:
                continue
            if key in configured_extra_body:
                raise ValueError(f"Ark Responses {key} 与模型 options.extra_body 重复。")
            configured_extra_body[key] = value
        reject_unsupported_kwargs("Ark Responses", {"top_logprobs": request.top_logprobs})
        if request.top_k is not None:
            if "top_k" in configured_extra_body and "top_k" not in configured_scalar_extra_keys:
                raise ValueError("Ark Responses top_k 与模型 options.extra_body 重复。")
            configured_extra_body["top_k"] = request.top_k
        if request.seed is not None:
            if "seed" in configured_extra_body and "seed" not in configured_scalar_extra_keys:
                raise ValueError("Ark Responses seed 与模型 options.extra_body 重复。")
            configured_extra_body["seed"] = request.seed
        converted_tools = self._convert_responses_tools(request.tools)
        if converted_tools:
            params["tools"] = converted_tools
        effective_response_format = request.response_format or configured_response_format
        if effective_response_format:
            params["text"] = self._convert_responses_format(effective_response_format)
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_responses_tool_choice(request.tool_choice)
        elif configured_tool_choice is not None:
            params["tool_choice"] = self._convert_responses_tool_choice(configured_tool_choice)
        for key in (
            "previous_response_id",
            "reasoning",
            "thinking",
            "store",
            "caching",
            "parallel_tool_calls",
            "max_tool_calls",
            "context_management",
            "expire_at",
            "service_tier",
        ):
            value = getattr(request, key, None)
            if value is not None:
                params[key] = value
        if request.reasoning is None and configured_reasoning_effort is not None:
            params["reasoning"] = {"effort": configured_reasoning_effort}
        if request.conversation is not None and request.session is not None:
            raise ValueError("Ark Responses 不能同时传 conversation 和 session。")
        # Ark calls the response conversation state a session. Keep
        # `conversation` as the portable request field for OpenAI-compatible
        # channels while mapping it to the SDK's actual parameter here.
        request_session = request.session
        request_conversation = request.conversation
        if request_session is not None:
            if not isinstance(request_session, Mapping):
                raise ValueError("Ark Responses session 必须是对象。")
            params["session"] = dict(request_session)
        elif request_conversation is not None:
            if not isinstance(request_conversation, Mapping):
                raise ValueError("Ark Responses conversation 必须是对象。")
            params["session"] = dict(request_conversation)
        elif configured_session is not None:
            if not isinstance(configured_session, Mapping):
                raise ValueError("Ark Responses 模型 options.session 必须是对象。")
            params["session"] = dict(configured_session)
        elif configured_conversation is not None:
            if not isinstance(configured_conversation, Mapping):
                raise ValueError("Ark Responses 模型 options.conversation 必须是对象。")
            params["session"] = dict(configured_conversation)
        if request.timeout is not None:
            params["timeout"] = request.timeout
        if request.extra_headers is not None:
            params["extra_headers"] = request.extra_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        self._merge_options(params, configured_options)
        if configured_extra_body:
            params["extra_body"] = dict(configured_extra_body)
        request_extra_body = self._validated_extra_body(
            request.extra_body,
            "Ark Responses",
            set(params).difference({"extra_body"}),
        )
        if request_extra_body:
            configured_overlap = sorted(
                set(params.get("extra_body", {})).intersection(request_extra_body)
            )
            if configured_overlap:
                raise ValueError(
                    "Ark Responses extra_body 与模型 options 重复: " + ", ".join(configured_overlap)
                )
            params.setdefault("extra_body", {}).update(request_extra_body)
        if params.get("extra_body"):
            params["extra_body"] = self._validated_extra_body(
                params["extra_body"],
                "Ark Responses",
                set(params).difference({"extra_body"}),
            )
        self._reject_instructions_with_enabled_caching(params)
        return params

    @staticmethod
    def _reject_instructions_with_enabled_caching(params: Mapping[str, Any]) -> None:
        """拒绝 ``instructions`` 与 ``caching={"type": "enabled"}`` 同时出现。

        官方文档明确二者互斥：配置 ``instructions`` 后本轮请求无法写入或使用
        缓存，``caching`` 为 ``enabled`` 时服务端直接报错。SDK 不做本地校验，
        会原样发到服务端，因此在构造阶段显式拒绝，避免用户从远端 400 反推。

        ``instructions`` 与 ``caching`` 各有两条来源：顶层参数，以及
        ``extra_body``（Ark SDK 在 ``_base_client`` 里把 ``extra_body`` 合并
        进请求体，服务端看到的仍是同名参数）。只查顶层会漏掉
        ``extra_body={"instructions": ...}`` 配顶层 ``caching`` 这种组合。
        """
        extra_body = params.get("extra_body")
        instructions = params.get("instructions")
        if instructions is None and isinstance(extra_body, Mapping):
            instructions = extra_body.get("instructions")
        if instructions is None:
            return
        candidates = [params.get("caching")]
        if isinstance(extra_body, Mapping):
            candidates.append(extra_body.get("caching"))
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            # SDK 的 ``ResponseCaching.type`` 是 ``Literal["disabled","enabled"]``，
            # 但那是类型注解、不做运行时校验：``"ENABLED"`` / ``" enabled"`` 会
            # 原样发到服务端。归一化后再比较，否则大小写与空白变体可绕过互斥检查。
            candidate_type = candidate.get("type")
            if isinstance(candidate_type, str) and candidate_type.strip().lower() == "enabled":
                raise ValueError(
                    'Ark Responses 的 instructions 与 caching={"type": "enabled"} 互斥：'
                    "官方规定配置 instructions 后本轮请求无法写入或使用缓存，caching 为 "
                    "enabled 时请求会直接报错。instructions 来自 system_prompt，"
                    "注意 CompletionRequest.system_prompt 有兼容默认值"
                    '（未显式传入时为 "You are a helpful assistant."）；'
                    "请显式传入 system_prompt=None 并改用 messages 携带系统提示，"
                    "或移除 caching。"
                )

    @classmethod
    def _validate_native_response_kwargs(cls, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """校验 Ark SDK 原生 Responses 参数，不改变其必填字段语义。"""
        params = dict(kwargs)
        unsupported = {
            key: value for key, value in params.items() if key not in cls._ARK_RESPONSES_CREATE_KEYS
        }
        reject_unsupported_kwargs("Ark Responses", unsupported)
        if (
            "model" not in params
            or not isinstance(params.get("model"), str)
            or not params["model"].strip()
        ):
            raise ValueError("Ark Responses 原生调用必须显式提供 model。")
        if "input" not in params or params.get("input") is None:
            raise ValueError("Ark Responses 创建需要 input。")
        headers, query = validate_secret_free_request_overrides(
            params.get("extra_headers"),
            params.get("extra_query"),
            "Ark Responses",
        )
        if "extra_headers" in params:
            params["extra_headers"] = headers
        if "extra_query" in params:
            params["extra_query"] = query
        if "extra_body" in params and params["extra_body"] is not None:
            params["extra_body"] = validate_secret_free_payload(
                params["extra_body"], "Ark Responses", "extra_body"
            )
        if "session" in params and params["session"] is not None:
            if not isinstance(params["session"], Mapping):
                raise ValueError("Ark Responses session 必须是对象。")
            params["session"] = dict(params["session"])
        # 原生入口同样要执行 instructions 与 caching 的互斥校验，否则用户
        # 绕过 Facade 直接调用 create_response/async_create_response 时会从
        # 远端 400 才得知参数冲突。
        cls._reject_instructions_with_enabled_caching(params)
        return params

    def _embedding_options(self) -> dict[str, Any]:
        """返回 Ark Embeddings 接受的模型级参数，排除聊天选项。"""
        supported = {"dimensions", "encoding_format", "user", "extra_body"}
        return {key: value for key, value in self._options.items() if key in supported}

    @staticmethod
    def _validate_resource_kwargs(
        resource: str,
        kwargs: dict[str, Any],
        supported: Collection[str],
    ) -> None:
        validate_secret_free_request_overrides(
            kwargs.get("extra_headers"),
            kwargs.get("extra_query"),
            f"Ark {resource}",
        )
        if "extra_body" in kwargs:
            validate_secret_free_payload(
                kwargs.get("extra_body"),
                f"Ark {resource}",
                "extra_body",
            )
        unsupported = sorted(key for key in kwargs if key not in supported)
        if unsupported:
            raise ValueError(f"Ark {resource} 不支持请求参数: {', '.join(unsupported)}")

    @staticmethod
    def _validate_file_wait_args(
        file_id: str,
        poll_interval: float | None,
        max_wait_seconds: float | None,
    ) -> tuple[str, float | None, float | None]:
        """校验 Ark 文件轮询参数，避免无界等待或底层类型错误。"""
        file_id = VolcengineProvider._require_non_empty_string(
            file_id,
            "Ark 文件等待的 file_id",
        )

        def positive_finite(value: float | None, name: str) -> float | None:
            if value is None:
                return None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"Ark 文件等待的 {name} 必须是有限正数。")
            return float(value)

        return (
            file_id,
            positive_finite(poll_interval, "poll_interval"),
            positive_finite(max_wait_seconds, "max_wait_seconds"),
        )

    @staticmethod
    def _require_non_empty_string(value: Any, label: str) -> str:
        """校验 SDK 资源 ID、模型名等字符串参数。"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} 必须是非空字符串。")
        return value.strip()

    @classmethod
    def _prepare_batch_kwargs(
        cls,
        kwargs: dict[str, Any],
        model_name: str,
        resource: str,
        supported: frozenset[str],
        required: tuple[str, ...],
    ) -> dict[str, Any]:
        params = dict(kwargs)
        if params.get("model") is None:
            params["model"] = model_name
        missing = [name for name in required if name not in params or params[name] is None]
        if missing:
            raise ValueError(f"Ark {resource} 缺少必填参数: {', '.join(missing)}")
        model = params.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Ark {resource} 的 model 必须是非空字符串。")
        cls._validate_resource_kwargs(resource, params, supported)
        return params

    @staticmethod
    def _extract_result(response: Any) -> CompletionResult:
        choices = field(response, "choices", []) or []
        text = ""
        calls: list[dict[str, Any]] = []
        if choices:
            message = field(choices[0], "message")
            text = content_to_text(field(message, "content", ""))
            for call in field(message, "tool_calls", []) or []:
                fn = field(call, "function")
                calls.append(
                    {
                        "id": field(call, "id"),
                        "name": field(fn, "name"),
                        "arguments": normalize_tool_arguments(field(fn, "arguments", "")),
                    }
                )
        output_text = field(response, "output_text")
        if isinstance(output_text, str):
            text = output_text
        if not text:
            # Ark 的 Response 没有 OpenAI 那样的 ``output_text`` 聚合属性，
            # 正文位于 ``output[].content[].text``。缺了这一步，所有非流式
            # 入口都会返回空文本而不报错。仅在没有 choices 文本时启用，避免
            # 覆盖 Chat Completions 分支已经取到的内容。
            text = _ark_responses_output_text(response)
        for item in field(response, "output", []) or []:
            item_type = field(item, "type")
            if item_type in {"function_call", "custom_tool_call"}:
                calls.append(
                    {
                        "id": field(item, "call_id") or field(item, "id"),
                        "type": item_type,
                        "name": field(item, "name"),
                        "arguments": normalize_tool_arguments(
                            field(item, "arguments", field(item, "input", ""))
                        ),
                    }
                )
        usage_dict = normalize_usage(field(response, "usage"))
        reasoning = ""
        refusal: str | None = None
        if choices:
            message = field(choices[0], "message")
            for name in ("reasoning_content", "reasoning"):
                value = field(message, name)
                if isinstance(value, str):
                    reasoning += value
            candidate_refusal = field(message, "refusal")
            if isinstance(candidate_refusal, str) and candidate_refusal:
                refusal = candidate_refusal
        for item in field(response, "output", []) or []:
            if field(item, "type") in {"reasoning", "reasoning_item"}:
                summary = field(item, "summary")
                if isinstance(summary, str):
                    reasoning += summary
                elif isinstance(summary, (list, tuple)):
                    for entry in summary:
                        reasoning_text = field(entry, "text")
                        if isinstance(reasoning_text, str):
                            reasoning += reasoning_text
                for content in field(item, "content", []) or []:
                    reasoning_text = field(content, "text")
                    if isinstance(reasoning_text, str):
                        reasoning += reasoning_text
                fallback = field(item, "text")
                if isinstance(fallback, str):
                    reasoning += fallback
            if field(item, "type") == "refusal":
                candidate_refusal = field(item, "refusal") or field(item, "text")
                if isinstance(candidate_refusal, str) and candidate_refusal:
                    refusal = candidate_refusal
            for content in field(item, "content", []) or []:
                if field(content, "type") != "refusal":
                    continue
                candidate_refusal = field(content, "refusal") or field(content, "text")
                if isinstance(candidate_refusal, str) and candidate_refusal:
                    refusal = candidate_refusal
        if (
            not choices
            and not text
            and not calls
            and not refusal
            and not reasoning
            and field(response, "status") == "completed"
            and not _ark_responses_has_builtin_tool_items(response)
        ):
            # 项目规则禁止用占位结果掩盖失败。Responses 响应标记为 completed
            # 却既无正文、无工具调用、也无拒答与推理，说明响应结构与预期不符
            # （例如 SDK 改了字段形状）。此时静默返回空文本会让上层拿到空串后
            # 继续，最终表现为难以定位的「空结果」错误，因此在边界显式失败。
            #
            # 但内置工具（web_search_call、mcp_call、mcp_list_tools 等）的
            # 调用项既不是 function_call 也没有正文，而它们是 Ark 内置工具
            # 的正常中间态——没有工具调用就不可能有后续轮次，调用方需要读
            # output 才能继续。把它们判为「结构不符」会让这类响应无法处理。
            raise RuntimeError(
                "Ark Responses 响应已完成但未包含任何正文、工具调用或拒答内容；"
                "请检查响应结构与 SDK 版本是否匹配。"
            )
        return CompletionResult(
            text=text,
            tool_calls=calls,
            usage=usage_dict,
            finish_reason=field(choices[0], "finish_reason")
            if choices
            else field(response, "status"),
            response_id=field(response, "id"),
            refusal=refusal,
            reasoning=reasoning,
            raw=response,
        )

    @staticmethod
    def _raise_for_response_error(response: Any) -> None:
        status = field(response, "status")
        if status == "failed":
            error = field(response, "error")
            message = (
                error
                if isinstance(error, str)
                else field(error, "message", "Ark Responses 请求失败")
            )
            raise RuntimeError(str(message))
        if status == "incomplete":
            details = field(response, "incomplete_details")
            reason = field(details, "reason", "unknown")
            raise RuntimeError(f"Ark Responses 响应不完整: {reason}")
        if status == "cancelled":
            raise RuntimeError("Ark Responses 请求已取消")

    @staticmethod
    def _chat_stream_events(chunk: Any) -> list[StreamEvent]:
        raise_for_stream_error_event(chunk, "Ark Chat Completions")
        OpenAICompatibleProvider._raise_for_chat_response_error(
            chunk,
            allow_empty_choices=True,
            provider="Ark Chat Completions",
        )
        choices = field(chunk, "choices", []) or []
        usage = normalize_usage(field(chunk, "usage"))
        if not choices:
            return [StreamEvent(type="usage", usage=usage, raw=chunk)] if usage else []
        choice = choices[0]
        delta = field(choice, "delta")
        text = content_to_text(field(delta, "content", ""))
        reasoning = field(delta, "reasoning_content") or field(delta, "reasoning") or ""
        refusal = field(delta, "refusal")
        calls = field(delta, "tool_calls", []) or []
        finish_reason = field(choice, "finish_reason")
        events: list[StreamEvent] = []
        if isinstance(refusal, str) and refusal:
            events.append(StreamEvent(type="refusal_delta", refusal=refusal, raw=chunk))
        response_id = field(chunk, "id")
        if text:
            events.append(
                StreamEvent(type="text_delta", text=text, response_id=response_id, raw=chunk)
            )
        if isinstance(reasoning, str) and reasoning:
            events.append(
                StreamEvent(
                    type="reasoning_delta", reasoning=reasoning, response_id=response_id, raw=chunk
                )
            )
        if calls:
            for position, call in enumerate(calls):
                function = field(call, "function")
                events.append(
                    StreamEvent(
                        type="tool_call_delta",
                        tool_call={
                            "id": field(call, "id"),
                            "index": field(call, "index", position),
                            "type": field(call, "type", "function"),
                            "name": field(function, "name"),
                            "arguments": field(function, "arguments", ""),
                        },
                        response_id=response_id,
                        raw=chunk,
                    )
                )
        if finish_reason:
            events.append(
                StreamEvent(
                    type="finish",
                    usage=usage,
                    finish_reason=finish_reason,
                    response_id=field(chunk, "id"),
                    raw=chunk,
                )
            )
        elif usage:
            if events:
                events[-1].usage = usage
            else:
                events.append(StreamEvent(type="usage", usage=usage, raw=chunk))
        return events

    @staticmethod
    def _chat_stream_event(chunk: Any) -> StreamEvent | None:
        events = VolcengineProvider._chat_stream_events(chunk)
        return events[0] if events else None

    @staticmethod
    def _merge_response_tool_call(
        accumulator: dict[Any, dict[str, Any]],
        tool_call: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        index = tool_call.get("index")
        call_id = tool_call.get("call_id") or tool_call.get("id")
        key: Any = ("index", index) if index is not None else ("id", call_id)
        if key == ("id", None):
            key = ("position", len(accumulator))
        merged = accumulator.setdefault(key, {})
        for name in ("id", "call_id", "index", "type", "name"):
            value = tool_call.get(name)
            if value is not None and value != "":
                merged[name] = value
        arguments = tool_call.get("arguments")
        if arguments is not None and arguments != "":
            merged["arguments"] = normalize_tool_arguments(arguments)
        merged.setdefault("type", "function_call")
        merged.setdefault("arguments", "")
        return key, merged

    @classmethod
    def _responses_completion_event(
        cls,
        converted: StreamEvent,
        accumulator: dict[Any, dict[str, Any]],
    ) -> tuple[Any, StreamEvent] | None:
        if converted.type != "tool_call_completed" or not converted.tool_call:
            return None
        key, merged = cls._merge_response_tool_call(accumulator, converted.tool_call)
        validate_complete_tool_call(merged, "Ark Responses")
        return key, StreamEvent(
            type="tool_call_completed",
            tool_call=dict(merged),
            response_id=converted.response_id,
            raw=converted.raw,
        )

    def _responses_stream_event(
        self,
        event: Any,
        tool_metadata: dict[Any, dict[str, Any]] | None = None,
    ) -> StreamEvent | None:
        """复用 Responses 事件转换，确保 Ark 与 OpenAI 的工具关联一致。"""
        return OpenAICompatibleProvider._responses_stream_event(
            event,
            tool_metadata,
            error_provider="Ark Responses",
            response_error_handler=self._raise_for_response_error,
        )

    def stream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Generator[StreamEvent, None, None]:
        request = coerce_completion_request(request, kwargs, "Ark stream_events").copy_with(
            stream=True,
        )
        if self._protocol == "responses":
            self._require_responses_resource("stream_events")
            response_tool_metadata: dict[Any, dict[str, Any]] = {}
            response_tool_calls: dict[Any, dict[str, Any]] = {}
            pending_completions: dict[Any, StreamEvent] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            for event in retry_sync_stream(
                lambda: self._get_client().responses.create(
                    **self._build_responses_request(request)
                ),
                lambda event: raise_for_stream_error_event(event, "Ark Responses"),
            ):
                if terminal_seen:
                    continue
                event_type = field(event, "type", "")
                converted = self._responses_stream_event(event, response_tool_metadata)
                if converted:
                    if converted.type == "finish":
                        terminal_seen = True
                        for key, pending in pending_completions.items():
                            if key not in completed_tool_calls:
                                completed_tool_calls.add(key)
                                yield pending
                        for key, tool_call in response_tool_calls.items():
                            if key not in completed_tool_calls:
                                validate_complete_tool_call(tool_call, "Ark Responses")
                                completed_tool_calls.add(key)
                                yield StreamEvent(
                                    type="tool_call_completed",
                                    tool_call=dict(tool_call),
                                    response_id=converted.response_id,
                                    raw=converted.raw,
                                )
                        yield converted
                        # Drain the SDK stream to EOF before closing it.  Some
                        # async transports keep a response-body generator open
                        # after the terminal event; breaking here can surface
                        # noisy generator-close errors and leak a connection.
                        continue
                    if converted.type == "tool_call_delta" and converted.tool_call:
                        merge_tool_call_fragment(response_tool_calls, converted.tool_call)
                        yield converted
                        continue
                    if converted.type == "tool_call_completed":
                        completed = self._responses_completion_event(converted, response_tool_calls)
                        if completed is None:
                            continue
                        key, completed_event = completed
                        if event_type in {
                            "response.function_call_arguments.done",
                            "response.custom_tool_call_input.done",
                        }:
                            pending_completions[key] = completed_event
                            continue
                        pending_completions.pop(key, None)
                        if key not in completed_tool_calls:
                            completed_tool_calls.add(key)
                            yield completed_event
                        continue
                    yield converted
            if not terminal_seen:
                raise RuntimeError("Ark Responses 流在 response.completed 之前结束。")
            return
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_chat_tool_calls: set[Any] = set()
        terminal_seen = False
        for chunk in retry_sync_stream(
            lambda: self._get_client().chat.completions.create(
                **self._build_chat_request(**request.to_invoke_kwargs())
            ),
            lambda chunk: raise_for_stream_error_event(chunk, "Ark Chat Completions"),
        ):
            raise_for_stream_error_event(chunk, "Ark Chat Completions")
            for converted in self._chat_stream_events(chunk):
                if converted.type == "finish":
                    terminal_seen = True
                if converted.type == "tool_call_delta" and converted.tool_call:
                    merge_tool_call_fragment(tool_calls, converted.tool_call)
                if converted.type == "finish":
                    for key, tool_call in tool_calls.items():
                        if key not in completed_chat_tool_calls:
                            validate_complete_tool_call(tool_call, "Ark Chat Completions")
                            completed_chat_tool_calls.add(key)
                            yield StreamEvent(
                                type="tool_call_completed",
                                tool_call=dict(tool_call),
                                response_id=converted.response_id,
                                raw=converted.raw,
                            )
                yield converted
        if not terminal_seen:
            raise RuntimeError("Ark Chat Completions 流在 finish 事件之前结束。")

    async def astream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        request = coerce_completion_request(request, kwargs, "Ark astream_events").copy_with(
            stream=True,
        )
        if self._protocol == "responses":
            self._require_responses_resource("astream_events")
            response_tool_metadata: dict[Any, dict[str, Any]] = {}
            response_tool_calls: dict[Any, dict[str, Any]] = {}
            pending_completions: dict[Any, StreamEvent] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            async for event in retry_async_stream(
                lambda: self._get_aclient().responses.create(
                    **self._build_responses_request(request)
                ),
                lambda event: raise_for_stream_error_event(event, "Ark Responses"),
            ):
                if terminal_seen:
                    continue
                event_type = field(event, "type", "")
                converted = self._responses_stream_event(event, response_tool_metadata)
                if converted:
                    if converted.type == "finish":
                        terminal_seen = True
                        for key, pending in pending_completions.items():
                            if key not in completed_tool_calls:
                                completed_tool_calls.add(key)
                                yield pending
                        for key, tool_call in response_tool_calls.items():
                            if key not in completed_tool_calls:
                                validate_complete_tool_call(tool_call, "Ark Responses")
                                completed_tool_calls.add(key)
                                yield StreamEvent(
                                    type="tool_call_completed",
                                    tool_call=dict(tool_call),
                                    response_id=converted.response_id,
                                    raw=converted.raw,
                                )
                        yield converted
                        # Drain the SDK stream to EOF before closing it. Some
                        # async transports keep the response body generator
                        # open after the terminal event.
                        continue
                    if converted.type == "tool_call_delta" and converted.tool_call:
                        merge_tool_call_fragment(response_tool_calls, converted.tool_call)
                        yield converted
                        continue
                    if converted.type == "tool_call_completed":
                        completed = self._responses_completion_event(converted, response_tool_calls)
                        if completed is None:
                            continue
                        key, completed_event = completed
                        if event_type in {
                            "response.function_call_arguments.done",
                            "response.custom_tool_call_input.done",
                        }:
                            pending_completions[key] = completed_event
                            continue
                        pending_completions.pop(key, None)
                        if key not in completed_tool_calls:
                            completed_tool_calls.add(key)
                            yield completed_event
                        continue
                    yield converted
            if not terminal_seen:
                raise RuntimeError("Ark Responses 异步流在 response.completed 之前结束。")
            return
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_chat_tool_calls: set[Any] = set()
        terminal_seen = False
        async for chunk in retry_async_stream(
            lambda: self._get_aclient().chat.completions.create(
                **self._build_chat_request(**request.to_invoke_kwargs())
            ),
            lambda chunk: raise_for_stream_error_event(chunk, "Ark Chat Completions"),
        ):
            raise_for_stream_error_event(chunk, "Ark Chat Completions")
            for converted in self._chat_stream_events(chunk):
                if converted.type == "finish":
                    terminal_seen = True
                if converted.type == "tool_call_delta" and converted.tool_call:
                    merge_tool_call_fragment(tool_calls, converted.tool_call)
                if converted.type == "finish":
                    for key, tool_call in tool_calls.items():
                        if key not in completed_chat_tool_calls:
                            validate_complete_tool_call(tool_call, "Ark Chat Completions")
                            completed_chat_tool_calls.add(key)
                            yield StreamEvent(
                                type="tool_call_completed",
                                tool_call=dict(tool_call),
                                response_id=converted.response_id,
                                raw=converted.raw,
                            )
                yield converted
        if not terminal_seen:
            raise RuntimeError("Ark Chat Completions 异步流在 finish 事件之前结束。")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    def complete(self, request: CompletionRequest | None = None, **kwargs: Any) -> CompletionResult:
        request = coerce_completion_request(request, kwargs, "Ark complete")
        request = request.copy_with(stream=False)
        if self._protocol == "responses":
            self._require_responses_resource("complete")
            response = self._get_client().responses.create(**self._build_responses_request(request))
            self._raise_for_response_error(response)
        else:
            response = self._get_client().chat.completions.create(
                **self._build_chat_request(**request.to_invoke_kwargs())
            )
            OpenAICompatibleProvider._raise_for_chat_response_error(
                response,
                provider="Ark Chat Completions",
            )
        return self._extract_result(response)

    async def acomplete(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> CompletionResult:
        """使用 AsyncArk 聚合完整结果，保留 Responses/Chat 元数据。"""
        request = coerce_completion_request(request, kwargs, "Ark acomplete")
        request = request.copy_with(stream=False)
        if self._protocol == "responses":
            self._require_responses_resource("acomplete")
            response = await retry_async_call(
                lambda: self._get_aclient().responses.create(
                    **self._build_responses_request(request)
                )
            )
            self._raise_for_response_error(response)
        else:
            response = await retry_async_call(
                lambda: self._get_aclient().chat.completions.create(
                    **self._build_chat_request(**request.to_invoke_kwargs())
                )
            )
            OpenAICompatibleProvider._raise_for_chat_response_error(
                response,
                provider="Ark Chat Completions",
            )
        return self._extract_result(response)

    def invoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = _UNSET_TEMPERATURE,  # type: ignore[assignment]
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        request = CompletionRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **kwargs,
        )
        if self._protocol == "responses":
            self._require_responses_resource("invoke")
            if tools:
                raise ValueError(
                    "Ark Responses invoke 仅返回文本，不支持 tools；"
                    "请使用 complete 或 stream_events 获取工具调用。"
                )
            if stream:
                terminal_seen = False
                for event in retry_sync_stream(
                    lambda: self._get_client().responses.create(
                        **self._build_responses_request(request)
                    ),
                    lambda event: raise_for_stream_error_event(event, "Ark Responses"),
                ):
                    raise_for_stream_error_event(event, "Ark Responses")
                    converted = self._responses_stream_event(event)
                    if field(event, "type") == "response.completed":
                        terminal_seen = True
                    if converted and converted.type == "refusal_delta" and converted.refusal:
                        raise RuntimeError(f"Ark Responses 返回拒答: {converted.refusal}")
                    if converted and converted.text:
                        yield converted.text
                if not terminal_seen:
                    raise RuntimeError("Ark Responses 流在 response.completed 之前结束。")
            else:
                response = retry_sync_call(
                    lambda: self._get_client().responses.create(
                        **self._build_responses_request(request)
                    )
                )
                self._raise_for_response_error(response)
                result = self._extract_result(response)
                if result.refusal:
                    raise RuntimeError(f"Ark Responses 返回拒答: {result.refusal}")
                if result.text:
                    yield result.text
            return
        if stream:
            terminal_seen = False
            for chunk in retry_sync_stream(
                lambda: self._get_client().chat.completions.create(
                    **self._build_chat_request(**request.to_invoke_kwargs())
                ),
                lambda chunk: raise_for_stream_error_event(chunk, "Ark Chat Completions"),
            ):
                raise_for_stream_error_event(chunk, "Ark Chat Completions")
                for event in self._chat_stream_events(chunk):
                    if event.type == "finish":
                        terminal_seen = True
                    if event.type == "refusal_delta" and event.refusal:
                        raise RuntimeError(f"Ark Chat Completions 返回拒答: {event.refusal}")
                    if event.type == "text_delta" and event.text:
                        yield event.text
            if not terminal_seen:
                raise RuntimeError("Ark Chat Completions 流在 finish 事件之前结束。")
        else:
            response = retry_sync_call(
                lambda: self._get_client().chat.completions.create(
                    **self._build_chat_request(**request.to_invoke_kwargs())
                )
            )
            OpenAICompatibleProvider._raise_for_chat_response_error(
                response,
                provider="Ark Chat Completions",
            )
            result = self._extract_result(response)
            if result.refusal:
                raise RuntimeError(f"Ark Chat Completions 返回拒答: {result.refusal}")
            if result.text:
                yield result.text

    async def ainvoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = _UNSET_TEMPERATURE,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        if prompt is not None:
            kwargs["prompt"] = prompt
        kwargs["system_prompt"] = system_prompt
        if tools is not None:
            kwargs["tools"] = tools
        kwargs["stream"] = stream
        kwargs["temperature"] = temperature
        request = CompletionRequest(**kwargs)
        if self._protocol == "responses":
            self._require_responses_resource("ainvoke")
            if request.tools:
                raise ValueError(
                    "Ark Responses ainvoke 仅返回文本，不支持 tools；"
                    "请使用 acomplete 或 astream_events 获取工具调用。"
                )
            if request.stream:
                terminal_seen = False
                async for event in retry_async_stream(
                    lambda: self._get_aclient().responses.create(
                        **self._build_responses_request(request)
                    ),
                    lambda event: raise_for_stream_error_event(event, "Ark Responses"),
                ):
                    raise_for_stream_error_event(event, "Ark Responses")
                    converted = self._responses_stream_event(event)
                    if field(event, "type") == "response.completed":
                        terminal_seen = True
                    if converted and converted.type == "refusal_delta" and converted.refusal:
                        raise RuntimeError(f"Ark Responses 返回拒答: {converted.refusal}")
                    if converted and converted.text:
                        yield converted.text
                if not terminal_seen:
                    raise RuntimeError("Ark Responses 异步流在 response.completed 之前结束。")
            else:
                response = await retry_async_call(
                    lambda: self._get_aclient().responses.create(
                        **self._build_responses_request(request)
                    )
                )
                self._raise_for_response_error(response)
                result = self._extract_result(response)
                if result.refusal:
                    raise RuntimeError(f"Ark Responses 返回拒答: {result.refusal}")
                if result.text:
                    yield result.text
            return
        if request.stream:
            terminal_seen = False
            async for chunk in retry_async_stream(
                lambda: self._get_aclient().chat.completions.create(
                    **self._build_chat_request(**request.to_invoke_kwargs())
                ),
                lambda chunk: raise_for_stream_error_event(chunk, "Ark Chat Completions"),
            ):
                raise_for_stream_error_event(chunk, "Ark Chat Completions")
                for event in self._chat_stream_events(chunk):
                    if event.type == "finish":
                        terminal_seen = True
                    if event.type == "refusal_delta" and event.refusal:
                        raise RuntimeError(f"Ark Chat Completions 返回拒答: {event.refusal}")
                    if event.type == "text_delta" and event.text:
                        yield event.text
            if not terminal_seen:
                raise RuntimeError("Ark Chat Completions 异步流在 finish 事件之前结束。")
        else:
            response = await retry_async_call(
                lambda: self._get_aclient().chat.completions.create(
                    **self._build_chat_request(**request.to_invoke_kwargs())
                )
            )
            OpenAICompatibleProvider._raise_for_chat_response_error(
                response,
                provider="Ark Chat Completions",
            )
            result = self._extract_result(response)
            if result.refusal:
                raise RuntimeError(f"Ark Chat Completions 返回拒答: {result.refusal}")
            if result.text:
                yield result.text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        overlap = sorted({"input", "model"}.intersection(kwargs))
        if overlap:
            raise ValueError(f"Ark Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
        request: dict[str, Any] = {"input": texts, "model": self._model_name}
        configured_options = self._embedding_options()
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Ark Embedding",
            set(request).union(configured_options),
        )
        request_options = dict(kwargs)
        self._validate_resource_kwargs(
            "Embedding",
            request_options,
            {
                "encoding_format",
                "dimensions",
                "user",
                "extra_body",
                "extra_headers",
                "extra_query",
                "timeout",
            },
        )
        request_extra_body = self._validated_extra_body(
            request_options.pop("extra_body", None),
            "Ark Embedding",
            set(request).union(configured_options).union(request_options),
        )
        request.update(configured_options)
        request.update(request_options)
        merged_extra_body = self._merge_extra_body(
            configured_extra_body, request_extra_body, "Ark Embedding"
        )
        if merged_extra_body:
            request["extra_body"] = merged_extra_body
        response = self._get_client().embeddings.create(**request)
        return [normalize_embedding_vector(item.embedding) for item in response.data]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        overlap = sorted({"input", "model"}.intersection(kwargs))
        if overlap:
            raise ValueError(f"Ark Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
        request: dict[str, Any] = {"input": texts, "model": self._model_name}
        configured_options = self._embedding_options()
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Ark Embedding",
            set(request).union(configured_options),
        )
        request_options = dict(kwargs)
        self._validate_resource_kwargs(
            "Embedding",
            request_options,
            {
                "encoding_format",
                "dimensions",
                "user",
                "extra_body",
                "extra_headers",
                "extra_query",
                "timeout",
            },
        )
        request_extra_body = self._validated_extra_body(
            request_options.pop("extra_body", None),
            "Ark Embedding",
            set(request).union(configured_options).union(request_options),
        )
        request.update(configured_options)
        request.update(request_options)
        merged_extra_body = self._merge_extra_body(
            configured_extra_body, request_extra_body, "Ark Embedding"
        )
        if merged_extra_body:
            request["extra_body"] = merged_extra_body
        response = await self._resolve_async_result(
            self._get_aclient().embeddings.create(**request)
        )
        return [normalize_embedding_vector(item.embedding) for item in response.data]

    def upload_file(self, file: Any, purpose: str, **kwargs: Any) -> Any:
        purpose = self._require_non_empty_string(purpose, "Ark 文件 purpose")
        self._validate_resource_kwargs("文件创建", kwargs, self._ARK_FILE_CREATE_KEYS)
        return self._get_client().files.create(file=file, purpose=purpose, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("文件列表", kwargs, self._ARK_FILE_LIST_KEYS)
        return self._get_client().files.list(**kwargs)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        file_id = self._require_non_empty_string(file_id, "Ark 文件 file_id")
        self._validate_resource_kwargs(
            "文件获取", kwargs, {"extra_headers", "extra_query", "extra_body", "timeout"}
        )
        return self._get_client().files.retrieve(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        file_id = self._require_non_empty_string(file_id, "Ark 文件 file_id")
        self._validate_resource_kwargs(
            "文件删除", kwargs, {"extra_headers", "extra_query", "extra_body", "timeout"}
        )
        return self._get_client().files.delete(file_id, **kwargs)

    def wait_for_file(
        self,
        file_id: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """等待 Ark 文件处理完成，仅转发 SDK 的轮询参数。"""
        file_id, poll_interval, max_wait_seconds = self._validate_file_wait_args(
            file_id,
            poll_interval,
            max_wait_seconds,
        )
        self._validate_resource_kwargs("文件等待", kwargs, set())
        poll_kwargs = {
            key: value
            for key, value in {
                "poll_interval": poll_interval,
                "max_wait_seconds": max_wait_seconds,
            }.items()
            if value is not None
        }
        return self._get_client().files.wait_for_processing(id=file_id, **poll_kwargs)

    def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("retrieve_response")
        self._validate_resource_kwargs(
            "Responses retrieve_response", kwargs, self._ARK_RESPONSES_RETRIEVE_KEYS
        )
        return self._get_client().responses.retrieve(response_id, **kwargs)

    def create_response(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        """创建 Ark Responses；可传统一请求或原生 SDK 参数。"""
        self._require_responses_resource("create_response")
        if request is not None:
            if kwargs:
                raise ValueError("Ark Responses 不能同时传 request 和原生关键字参数。")
            params = self._build_responses_request(request)
        else:
            params = self._validate_native_response_kwargs(kwargs)
        return self._get_client().responses.create(**params)

    def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("delete_response")
        self._validate_resource_kwargs(
            "Responses delete_response", kwargs, self._ARK_RESPONSES_RETRIEVE_KEYS
        )
        return self._get_client().responses.delete(response_id, **kwargs)

    def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("list_response_input_items")
        self._validate_resource_kwargs(
            "Responses list_response_input_items",
            kwargs,
            self._ARK_RESPONSES_INPUT_ITEMS_KEYS,
        )
        return self._get_client().input_items.list(response_id, **kwargs)

    def create_batch_chat(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Chat",
            self._ARK_BATCH_CHAT_KEYS,
            ("messages",),
        )
        return self._get_client().batch.chat.completions.create(**params)

    def create_batch_embedding(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Embedding",
            self._ARK_BATCH_EMBEDDING_KEYS,
            ("input",),
        )
        return self._get_client().batch.embeddings.create(**params)

    def create_batch_multimodal_embedding(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Multimodal Embedding",
            self._ARK_BATCH_MULTIMODAL_EMBEDDING_KEYS,
            ("input",),
        )
        return self._get_client().batch.multimodal_embeddings.create(**params)

    def create_context(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Context",
            self._ARK_CONTEXT_CREATE_KEYS,
            ("messages",),
        )
        return self._get_client().context.create(**params)

    def context_complete(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Context Completion",
            self._ARK_CONTEXT_COMPLETION_KEYS,
            ("context_id", "messages"),
        )
        params["context_id"] = self._require_non_empty_string(
            params["context_id"],
            "Ark Context Completion 的 context_id",
        )
        return self._get_client().context.completions.create(**params)

    def create_content_generation_task(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Content Generation",
            self._ARK_CONTENT_GENERATION_CREATE_KEYS,
            ("content",),
        )
        return self._get_client().content_generation.tasks.create(**params)

    def get_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        task_id = self._require_non_empty_string(task_id, "Ark Content Generation 的 task_id")
        self._validate_resource_kwargs(
            "Content Generation 获取",
            kwargs,
            {"extra_headers", "extra_query", "extra_body", "timeout"},
        )
        return self._get_client().content_generation.tasks.get(task_id=task_id, **kwargs)

    def list_content_generation_tasks(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs(
            "Content Generation 列表",
            kwargs,
            self._ARK_CONTENT_GENERATION_LIST_KEYS,
        )
        return self._get_client().content_generation.tasks.list(**kwargs)

    def delete_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        task_id = self._require_non_empty_string(task_id, "Ark Content Generation 的 task_id")
        self._validate_resource_kwargs(
            "Content Generation 删除",
            kwargs,
            {"extra_headers", "extra_query", "extra_body", "timeout"},
        )
        return self._get_client().content_generation.tasks.delete(task_id=task_id, **kwargs)

    def count_tokens(self, text: str | list[str], **kwargs: Any) -> int:
        self._validate_resource_kwargs(
            "Tokenization",
            kwargs,
            {"user", "extra_headers", "extra_query", "extra_body", "timeout"},
        )
        response = self._get_client().tokenization.create(
            text=text, model=self._model_name, **kwargs
        )
        values = field(response, "data", []) or []
        if not values:
            raise RuntimeError("Ark tokenization 响应缺少 data。")
        return sum(int(field(item, "total_tokens", 0)) for item in values)

    def multimodal_embed(self, inputs: Any, **kwargs: Any) -> Any:
        self._validate_resource_kwargs(
            "Multimodal Embedding",
            kwargs,
            {
                "encoding_format",
                "dimensions",
                "instructions",
                "sparse_embedding",
                "extra_headers",
                "extra_query",
                "extra_body",
                "timeout",
            },
        )
        return self._get_client().multimodal_embeddings.create(
            input=inputs, model=self._model_name, **kwargs
        )

    def list_input_items(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("list_input_items")
        self._validate_resource_kwargs(
            "Responses list_input_items", kwargs, self._ARK_RESPONSES_INPUT_ITEMS_KEYS
        )
        return self._get_client().input_items.list(response_id, **kwargs)

    def generate_batch_chat(self, **kwargs: Any) -> Any:
        """兼容旧名称，转发到唯一的 Batch Chat 实现。"""
        return self.create_batch_chat(**kwargs)

    def generate_image(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("Images", kwargs, self._ARK_IMAGE_GENERATE_KEYS)
        kwargs.setdefault("model", self._model_name)
        return self._get_client().images.generate(**kwargs)

    def beta_chat_parse(self, **kwargs: Any) -> Any:
        """调用 Ark beta.chat 的结构化解析接口。"""
        self._validate_resource_kwargs("Beta Chat Parse", kwargs, self._ARK_BETA_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return self._get_client().beta.chat.completions.parse(**kwargs)

    def beta_chat_stream(self, **kwargs: Any) -> Any:
        """调用 Ark beta.chat 的原生流式上下文管理器。"""
        self._validate_resource_kwargs("Beta Chat Stream", kwargs, self._ARK_BETA_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return self._get_client().beta.chat.completions.stream(**kwargs)

    def bot_chat(self, **kwargs: Any) -> Any:
        """调用 Ark Bot Chat 原生接口。"""
        self._validate_resource_kwargs("Bot Chat", kwargs, self._ARK_BOT_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return self._get_client().bot_chat.completions.create(**kwargs)

    def _get_classification_resource(self) -> Any:
        """返回 Ark Classification 资源。

        Ark SDK 当前版本包含 classification 模块，但客户端构造函数没有将其
        挂载到顶层。优先使用未来版本可能提供的顶层资源，否则显式实例化
        SDK 官方资源类；导入失败时保留可定位的 NotImplementedError。
        """
        client = self._get_client()
        resource = getattr(client, "classification", None)
        if resource is not None:
            return resource
        resource = getattr(self, "_classification_resource", None)
        if resource is not None:
            return resource
        try:
            from volcenginesdkarkruntime.resources.classification import (  # type: ignore[import-untyped]
                Classification,
            )
        except (ImportError, AttributeError) as exc:
            raise NotImplementedError(
                "当前 volcengine-python-sdk 未提供 Ark Classification 资源。"
            ) from exc
        resource = Classification(client)
        self._classification_resource = resource
        return resource

    def classify(
        self,
        query: str,
        labels: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """调用 Ark Classification 接口。"""
        query = self._require_non_empty_string(query, "Ark Classification 的 query")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError("Ark Classification 的 labels 必须是非空字符串列表。")
        labels = [label.strip() for label in labels]
        effective_model = (
            self._model_name
            if model is None
            else self._require_non_empty_string(model, "Ark Classification 的 model")
        )
        self._validate_resource_kwargs(
            "Classification",
            kwargs,
            {"user", "extra_headers", "extra_query", "extra_body", "timeout"},
        )
        return self._get_classification_resource().create(
            query=query,
            model=effective_model,
            labels=labels,
            **kwargs,
        )

    async def async_upload_file(self, file: Any, purpose: str, **kwargs: Any) -> Any:
        purpose = self._require_non_empty_string(purpose, "Ark 文件 purpose")
        self._validate_resource_kwargs("文件创建", kwargs, self._ARK_FILE_CREATE_KEYS)
        return await self._resolve_async_result(
            self._get_aclient().files.create(file=file, purpose=purpose, **kwargs)
        )

    async def async_list_files(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("文件列表", kwargs, self._ARK_FILE_LIST_KEYS)
        return await self._resolve_async_result(self._get_aclient().files.list(**kwargs))

    async def async_retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        file_id = self._require_non_empty_string(file_id, "Ark 文件 file_id")
        self._validate_resource_kwargs(
            "文件获取", kwargs, {"extra_headers", "extra_query", "extra_body", "timeout"}
        )
        return await self._resolve_async_result(
            self._get_aclient().files.retrieve(file_id, **kwargs)
        )

    async def async_delete_file(self, file_id: str, **kwargs: Any) -> Any:
        file_id = self._require_non_empty_string(file_id, "Ark 文件 file_id")
        self._validate_resource_kwargs(
            "文件删除", kwargs, {"extra_headers", "extra_query", "extra_body", "timeout"}
        )
        return await self._resolve_async_result(self._get_aclient().files.delete(file_id, **kwargs))

    async def async_wait_for_file(
        self,
        file_id: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步等待 Ark 文件处理完成，仅转发 SDK 的轮询参数。"""
        file_id, poll_interval, max_wait_seconds = self._validate_file_wait_args(
            file_id,
            poll_interval,
            max_wait_seconds,
        )
        self._validate_resource_kwargs("文件等待", kwargs, set())
        poll_kwargs = {
            key: value
            for key, value in {
                "poll_interval": poll_interval,
                "max_wait_seconds": max_wait_seconds,
            }.items()
            if value is not None
        }
        return await self._resolve_async_result(
            self._get_aclient().files.wait_for_processing(id=file_id, **poll_kwargs)
        )

    async def async_retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("retrieve_response")
        self._validate_resource_kwargs(
            "Responses retrieve_response", kwargs, self._ARK_RESPONSES_RETRIEVE_KEYS
        )
        return await self._resolve_async_result(
            self._get_aclient().responses.retrieve(response_id, **kwargs)
        )

    async def async_create_response(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        """异步创建 Ark Responses；可传统一请求或原生 SDK 参数。"""
        self._require_responses_resource("create_response")
        if request is not None:
            if kwargs:
                raise ValueError("Ark Responses 不能同时传 request 和原生关键字参数。")
            params = self._build_responses_request(request)
        else:
            params = self._validate_native_response_kwargs(kwargs)
        return await self._resolve_async_result(self._get_aclient().responses.create(**params))

    async def async_delete_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("delete_response")
        self._validate_resource_kwargs(
            "Responses delete_response", kwargs, self._ARK_RESPONSES_RETRIEVE_KEYS
        )
        return await self._resolve_async_result(
            self._get_aclient().responses.delete(response_id, **kwargs)
        )

    async def async_list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("list_response_input_items")
        self._validate_resource_kwargs(
            "Responses list_response_input_items",
            kwargs,
            self._ARK_RESPONSES_INPUT_ITEMS_KEYS,
        )
        return await self._resolve_async_result(
            self._get_aclient().input_items.list(response_id, **kwargs)
        )

    async def async_list_input_items(self, response_id: str, **kwargs: Any) -> Any:
        """Ark 顶层 input_items 资源的异步直观别名。"""
        self._require_responses_resource("list_input_items")
        self._validate_resource_kwargs(
            "Responses list_input_items", kwargs, self._ARK_RESPONSES_INPUT_ITEMS_KEYS
        )
        return await self._resolve_async_result(
            self._get_aclient().input_items.list(response_id, **kwargs)
        )

    async def async_create_batch_chat(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Chat",
            self._ARK_BATCH_CHAT_ASYNC_KEYS,
            ("messages",),
        )
        return await self._resolve_async_result(
            self._get_aclient().batch.chat.completions.create(**params)
        )

    async def async_generate_batch_chat(self, **kwargs: Any) -> Any:
        """兼容旧名称，转发到当前 Ark Batch Chat 异步资源。"""
        return await self.async_create_batch_chat(**kwargs)

    async def async_create_batch_embedding(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Embedding",
            self._ARK_BATCH_EMBEDDING_KEYS,
            ("input",),
        )
        return await self._resolve_async_result(
            self._get_aclient().batch.embeddings.create(**params)
        )

    async def async_create_batch_multimodal_embedding(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Batch Multimodal Embedding",
            self._ARK_BATCH_MULTIMODAL_EMBEDDING_KEYS,
            ("input",),
        )
        return await self._resolve_async_result(
            self._get_aclient().batch.multimodal_embeddings.create(**params)
        )

    async def async_create_context(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Context",
            self._ARK_CONTEXT_CREATE_KEYS,
            ("messages",),
        )
        return await self._resolve_async_result(self._get_aclient().context.create(**params))

    async def async_context_complete(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Context Completion",
            self._ARK_CONTEXT_COMPLETION_KEYS,
            ("context_id", "messages"),
        )
        params["context_id"] = self._require_non_empty_string(
            params["context_id"],
            "Ark Context Completion 的 context_id",
        )
        return await self._resolve_async_result(
            self._get_aclient().context.completions.create(**params)
        )

    async def async_create_content_generation_task(self, **kwargs: Any) -> Any:
        params = self._prepare_batch_kwargs(
            kwargs,
            self._model_name,
            "Content Generation",
            self._ARK_CONTENT_GENERATION_CREATE_KEYS,
            ("content",),
        )
        return await self._resolve_async_result(
            self._get_aclient().content_generation.tasks.create(**params)
        )

    async def async_get_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        task_id = self._require_non_empty_string(task_id, "Ark Content Generation 的 task_id")
        self._validate_resource_kwargs(
            "Content Generation 获取",
            kwargs,
            {"extra_headers", "extra_query", "extra_body", "timeout"},
        )
        return await self._resolve_async_result(
            self._get_aclient().content_generation.tasks.get(task_id=task_id, **kwargs)
        )

    async def async_list_content_generation_tasks(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs(
            "Content Generation 列表",
            kwargs,
            self._ARK_CONTENT_GENERATION_LIST_KEYS,
        )
        return await self._resolve_async_result(
            self._get_aclient().content_generation.tasks.list(**kwargs)
        )

    async def async_delete_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        task_id = self._require_non_empty_string(task_id, "Ark Content Generation 的 task_id")
        self._validate_resource_kwargs(
            "Content Generation 删除",
            kwargs,
            {"extra_headers", "extra_query", "extra_body", "timeout"},
        )
        return await self._resolve_async_result(
            self._get_aclient().content_generation.tasks.delete(task_id=task_id, **kwargs)
        )

    async def async_count_tokens(self, text: str | list[str], **kwargs: Any) -> int:
        self._validate_resource_kwargs(
            "Tokenization",
            kwargs,
            {"user", "extra_headers", "extra_query", "extra_body", "timeout"},
        )
        response = await self._resolve_async_result(
            self._get_aclient().tokenization.create(text=text, model=self._model_name, **kwargs)
        )
        values = field(response, "data", []) or []
        if not values:
            raise RuntimeError("Ark tokenization 响应缺少 data。")
        return sum(int(field(item, "total_tokens", 0)) for item in values)

    async def async_multimodal_embed(self, inputs: Any, **kwargs: Any) -> Any:
        self._validate_resource_kwargs(
            "Multimodal Embedding",
            kwargs,
            {
                "encoding_format",
                "dimensions",
                "instructions",
                "sparse_embedding",
                "extra_headers",
                "extra_query",
                "extra_body",
                "timeout",
            },
        )
        return await self._resolve_async_result(
            self._get_aclient().multimodal_embeddings.create(
                input=inputs, model=self._model_name, **kwargs
            )
        )

    async def async_generate_image(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("Images", kwargs, self._ARK_IMAGE_GENERATE_KEYS)
        kwargs.setdefault("model", self._model_name)
        return await self._resolve_async_result(self._get_aclient().images.generate(**kwargs))

    async def async_beta_chat_parse(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("Beta Chat Parse", kwargs, self._ARK_BETA_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return await self._resolve_async_result(
            self._get_aclient().beta.chat.completions.parse(**kwargs)
        )

    def async_beta_chat_stream(self, **kwargs: Any) -> Any:
        """返回 Ark 异步 Beta Chat 的原生流式上下文管理器。

        Ark SDK 的 ``AsyncCompletions.stream`` 是同步方法，但返回的对象实现
        ``__aenter__``/``__aexit__``。这里不能再包一层协程，否则调用方必须
        先无意义地 await，无法直接按 SDK 约定使用 ``async with``。
        """
        self._validate_resource_kwargs("Beta Chat Stream", kwargs, self._ARK_BETA_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return self._get_aclient().beta.chat.completions.stream(**kwargs)

    async def async_bot_chat(self, **kwargs: Any) -> Any:
        self._validate_resource_kwargs("Bot Chat", kwargs, self._ARK_BOT_CHAT_KEYS)
        kwargs.setdefault("model", self._model_name)
        return await self._resolve_async_result(
            self._get_aclient().bot_chat.completions.create(**kwargs)
        )

    async def _get_async_classification_resource(self) -> Any:
        """返回 Ark Async Classification 资源，保持与同步入口相同的显式边界。"""
        client = self._get_aclient()
        resource = getattr(client, "classification", None)
        if resource is not None:
            return resource
        resource = getattr(self, "_async_classification_resource", None)
        if resource is not None:
            return resource
        try:
            from volcenginesdkarkruntime.resources.classification import (  # type: ignore[import-untyped]
                AsyncClassification,
            )
        except (ImportError, AttributeError) as exc:
            raise NotImplementedError(
                "当前 volcengine-python-sdk 未提供 Ark Async Classification 资源。"
            ) from exc
        resource = AsyncClassification(client)
        self._async_classification_resource = resource
        return resource

    async def async_classify(
        self,
        query: str,
        labels: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步调用 Ark Classification 接口。"""
        query = self._require_non_empty_string(query, "Ark Classification 的 query")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError("Ark Classification 的 labels 必须是非空字符串列表。")
        labels = [label.strip() for label in labels]
        effective_model = (
            self._model_name
            if model is None
            else self._require_non_empty_string(model, "Ark Classification 的 model")
        )
        self._validate_resource_kwargs(
            "Classification",
            kwargs,
            {"user", "extra_headers", "extra_query", "extra_body", "timeout"},
        )
        resource = await self._get_async_classification_resource()
        return await self._resolve_async_result(
            resource.create(
                query=query,
                model=effective_model,
                labels=labels,
                **kwargs,
            )
        )

    def close(self) -> None:
        """释放同步和异步 Ark SDK 客户端及分类资源。"""
        client = self._client
        async_client = self._aclient
        self._client = None
        self._classification_resource = None
        self._async_classification_resource = None
        first_error: Exception | None = None
        closed_ids: set[int] = set()
        if client is not None:
            try:
                client.close()
                closed_ids.add(id(client))
            except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                first_error = exc
        if async_client is not None:
            if id(async_client) in closed_ids:
                self._aclient = None
            else:
                try:
                    close_resource_sync(async_client, "Volcengine 异步客户端")
                except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                    first_error = first_error or exc
                else:
                    self._aclient = None
        else:
            self._aclient = None
        if first_error is not None:
            raise first_error

    async def aclose(self) -> None:
        """释放同步和异步 Ark SDK 客户端。"""
        async_client = self._aclient
        sync_client = self._client
        self._aclient = None
        self._client = None
        self._classification_resource = None
        self._async_classification_resource = None
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
