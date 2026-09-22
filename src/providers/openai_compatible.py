import asyncio
import inspect
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator, Mapping
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

import openai
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
from src.providers.resources import (
    AsyncOpenAICompatibleResources,
    OpenAICompatibleResources,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import (
    redact_sensitive_text,
    validate_secret_free_payload,
    validate_secret_free_resource_args,
    validate_secret_free_resource_kwargs,
)

logger = get_module_logger(__name__)


class OpenAICompatibleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    处理所有与OpenAI API格式兼容的提供商的通用逻辑。
    已注入 CSE 性能传感器与 tenacity 重试机制。
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
            "embedding",
            "responses",
        }
    )
    # OpenAI's resource methods live on this shared adapter for code reuse, but
    # only the official provider (or an explicitly verified Responses endpoint)
    # may expose them.  Keeping the declaration separate from the general
    # capability set prevents compatible channels from accidentally advertising
    # or dispatching official-only resources.
    _resource_capabilities: frozenset[str] = frozenset()
    # Tests and lightweight integrations sometimes instantiate this shared
    # adapter directly with ``provider="openai"`` instead of using the
    # OpenAIProvider subclass.  Keep that explicit provider identity
    # equivalent to the official adapter for resource gating, while leaving
    # all other compatible providers fail-closed.
    _OFFICIAL_OPENAI_RESOURCE_CAPABILITIES = frozenset(
        {
            "files",
            "batches",
            "vector_stores",
            "models",
            "moderation",
            "images",
            "audio",
            "videos",
            "uploads",
            "conversations",
            "containers",
            "fine_tuning",
            "evals",
        }
    )
    # These fields are accepted by OpenAI Chat Completions but are not part of
    # the Responses request shape. Keep this guard protocol-specific: the
    # current OpenAI SDK also exposes the fields below on ``responses.create``
    # (with ``verbosity`` normalized into ``text`` by this adapter).
    _OPENAI_ONLY_CHAT_FIELDS = frozenset(
        {
            "modalities",
            "audio",
            "prediction",
            "web_search_options",
        }
    )
    _OPENAI_ONLY_RESPONSES_FIELDS: frozenset[str] = frozenset()
    _OPENAI_ONLY_CHAT_COMPAT_FIELDS = frozenset(
        {
            "store",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "safety_identifier",
            "moderation",
            "verbosity",
        }
    )
    _BACKGROUND_OPTION_KEYS = frozenset(
        {
            "background",
            "background_poll_interval",
            "background_timeout",
            "responses_poll_interval",
            "responses_poll_timeout",
        }
    )
    # Model-level options are defaults for the corresponding SDK request. Keep
    # the allow-list separate from request-level ``extra_body`` so a typo cannot
    # reach the OpenAI SDK as an unknown keyword argument.
    _CHAT_MODEL_OPTION_KEYS = frozenset(
        {
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "modalities",
            "audio",
            "prediction",
            "web_search_options",
            "seed",
            "stop",
            "response_format",
            "tool_choice",
            "metadata",
            "user",
            "reasoning_effort",
            "stream_options",
            "parallel_tool_calls",
            "store",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "safety_identifier",
            "moderation",
            "verbosity",
            "extra_body",
        }
    )
    _RESPONSES_MODEL_OPTION_KEYS = frozenset(
        {
            "background",
            "context_management",
            "conversation",
            "include",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "moderation",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "reasoning",
            "reasoning_effort",
            "safety_identifier",
            "service_tier",
            "store",
            "stream_options",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
            # Portable/common names normalized by this adapter.
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stop",
            "seed",
            "verbosity",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "extra_body",
        }
    )
    _EMBEDDING_MODEL_OPTION_KEYS = frozenset(
        {"dimensions", "encoding_format", "user", "extra_body"}
    )
    _EMBEDDING_CLIENT_OPTION_KEYS = frozenset(
        {"timeout", "max_retries", "server_verified_protocols"}
    )

    def __init__(
        self,
        model_name: str,
        provider: str,
        settings: Any | None = None,
        protocol: str = "chat_completions",
        options: dict[str, Any] | None = None,
    ):
        self._model_name = model_name
        self._provider = provider
        self._protocol = self._normalize_protocol(protocol)
        self._options = validate_secret_free_options(options, provider)
        self._server_verified_protocols = self._normalize_verified_protocols(
            self._options.get("server_verified_protocols", ())
        )
        settings = settings if settings is not None else get_settings()

        settings_prefix = provider.lower().replace("-", "_")
        api_key_name = f"{settings_prefix}_api_key"
        base_url_name = f"{settings_prefix}_base_url"

        self._api_key = getattr(settings, api_key_name, None)
        self._base_url = getattr(settings, base_url_name, None)
        if self._base_url is None:
            # OpenAI 旧配置字段使用 `openai_api_base`，继续兼容现有配置文件。
            self._base_url = getattr(settings, f"{settings_prefix}_api_base", None)

        if provider in ["ollama", "lm-studio"] and not self._api_key:
            self._api_key = "no-key-required"

        if not self._api_key:
            logger.error(f"{provider} API Key 未设置。")
            raise ValueError(f"{api_key_name} is required for {provider}")
        if provider != "openai" and not self._base_url:
            logger.error(f"{provider} Base URL 未设置。")
            raise ValueError(f"{base_url_name} is required for {provider}")

        self._client: openai.OpenAI | None = None
        self._aclient: openai.AsyncOpenAI | None = None
        logger.info(f"初始化 OpenAICompatibleProvider ({provider})，模型: {model_name}")

    @property
    def resources(self) -> OpenAICompatibleResources:
        return OpenAICompatibleResources(self)

    @property
    def async_resources(self) -> AsyncOpenAICompatibleResources:
        return AsyncOpenAICompatibleResources(self)

    @staticmethod
    def _normalize_protocol(protocol: str) -> str:
        normalized = str(protocol).strip().lower().replace("-", "_")
        if normalized in {"chat", "completion", "chat_completion", "chat_completions"}:
            return "chat_completions"
        if normalized in {"response", "responses"}:
            return "responses"
        raise ValueError(
            f"OpenAI 兼容渠道不支持 {protocol} 协议。可选值: chat_completions, responses"
        )

    @classmethod
    def _normalize_verified_protocols(cls, configured: Any) -> frozenset[str]:
        """规范化服务端协议登记，并在配置边界给出明确错误。"""
        if configured is None:
            return frozenset()
        if isinstance(configured, str):
            configured = (configured,)
        elif not isinstance(configured, (list, tuple, set, frozenset)):
            raise ValueError("server_verified_protocols 必须是字符串或协议序列。")
        normalized: set[str] = set()
        for value in configured:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("server_verified_protocols 中的协议必须是非空字符串。")
            normalized.add(cls._normalize_protocol(value))
        return frozenset(normalized)

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            client_options = {
                key: value
                for key, value in self._options.items()
                if key in {"timeout", "max_retries"}
            }
            self._client = openai.OpenAI(
                api_key=self._api_key, base_url=self._base_url, **client_options
            )
        return self._client

    def _get_aclient(self) -> openai.AsyncOpenAI:
        if self._aclient is None:
            client_options = {
                key: value
                for key, value in self._options.items()
                if key in {"timeout", "max_retries"}
            }
            self._aclient = openai.AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url, **client_options
            )
        return self._aclient

    @staticmethod
    async def _resolve_async_result(result: Any) -> Any:
        """兼容 SDK 的协程方法和同步分页器返回值。"""
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _validate_sdk_resource_call(
        method: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        operation: str,
    ) -> None:
        """在调用资源 SDK 前校验参数名和必填字段。

        官方 SDK 方法通常没有 ``**kwargs``，因此 ``Signature.bind`` 可以在
        请求发出前同时拦截未知参数和缺失必填参数。测试替身或未来 SDK 若
        暴露动态签名，则保留原有转发能力。
        """
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return
        try:
            # Bind partially first so an unknown keyword is reported even when
            # the same call also omits another required argument. This keeps
            # the actionable caller error stable instead of depending on the
            # order in which ``inspect.Signature`` enumerates parameters.
            signature.bind_partial(*args, **dict(kwargs))
            signature.bind(*args, **dict(kwargs))
        except TypeError as exc:
            raise ValueError(f"{operation} SDK 参数不匹配: {exc}") from exc
        except ValueError as exc:
            # A few extension/builtin signatures raise ValueError while
            # binding positional-only arguments. Do not silently skip this
            # validation boundary; surface the SDK contract failure instead.
            raise ValueError(f"{operation} SDK 参数无法验证: {exc}") from exc

    def _call_sdk_resource(
        self,
        method: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        operation: str = "OpenAI 资源",
    ) -> Any:
        """校验并调用一个 OpenAI SDK 资源方法。

        Provider 资源入口保留 ``**kwargs`` 是为了兼容 SDK 的扩展字段；
        这里把实际 SDK 方法签名作为最后一道边界，确保未知参数和缺失的
        必填参数在请求发出前转换成项目统一的 ``ValueError``。对无法
        introspect 的第三方替身仍保持原有转发能力。
        """
        call_args = validate_secret_free_resource_args(
            args,
            f"{getattr(self, '_provider', 'OpenAI')} {operation}",
        )
        call_kwargs = validate_secret_free_resource_kwargs(
            kwargs or {},
            f"{getattr(self, '_provider', 'OpenAI')} {operation}",
        )
        self._validate_sdk_resource_call(method, call_args, call_kwargs, operation)
        return method(*call_args, **call_kwargs)

    async def _call_async_sdk_resource(
        self,
        method: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        operation: str = "OpenAI 资源",
    ) -> Any:
        """异步版本的 SDK 资源签名校验与调用。"""
        return await self._resolve_async_result(
            self._call_sdk_resource(method, args, kwargs, operation)
        )

    def _request_options(self) -> dict[str, Any]:
        """返回调用方配置的受控扩展参数；凭证不会进入 ModelDetail.options。"""
        validate_secret_free_options(
            getattr(self, "_options", {}) or {},
            getattr(self, "_provider", "openai"),
        )
        client_only = {"timeout", "max_retries", "server_verified_protocols"}
        non_openai_top_level = {"top_k", "repetition_penalty"}
        return {
            key: value
            for key, value in (getattr(self, "_options", {}) or {}).items()
            if key not in client_only
            and key not in non_openai_top_level
            and key not in self._BACKGROUND_OPTION_KEYS
        }

    @staticmethod
    def _validate_model_options(
        options: Mapping[str, Any],
        allowed: frozenset[str],
        endpoint: str,
    ) -> None:
        unknown = {key: value for key, value in options.items() if key not in allowed}
        reject_unsupported_kwargs(endpoint, unknown)

    def _background_poll_config(self) -> tuple[float, float]:
        options = getattr(self, "_options", {}) or {}
        interval = options.get(
            "background_poll_interval",
            options.get("responses_poll_interval", 1.0),
        )
        timeout = options.get(
            "background_timeout",
            options.get("responses_poll_timeout", 300.0),
        )
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(float(interval))
            or interval <= 0
        ):
            raise ValueError("Responses background_poll_interval 必须是正数。")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("Responses background_timeout 必须是正数。")
        return float(interval), float(timeout)

    @classmethod
    def _response_retrieve_kwargs(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: params[key]
            for key in ("include", "extra_headers", "extra_query", "extra_body", "timeout")
            if key in params and params[key] is not None
        }

    def _poll_background_response(self, response: Any, params: Mapping[str, Any]) -> Any:
        status = self._field(response, "status")
        if status not in {"queued", "in_progress", "pending"}:
            return response
        response_id = self._field(response, "id")
        if not response_id:
            raise RuntimeError("Responses background 响应缺少 id，无法轮询。")
        interval, timeout = self._background_poll_config()
        deadline = time.monotonic() + timeout
        retrieve_kwargs = self._response_retrieve_kwargs(params)
        while status in {"queued", "in_progress", "pending"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Responses background 响应 {response_id} 在 {timeout:g}s 内未完成。"
                )
            if interval:
                time.sleep(min(interval, remaining))
            response = retry_sync_call(
                lambda: self._get_client().responses.retrieve(response_id, **retrieve_kwargs)
            )
            status = self._field(response, "status")
        return response

    async def _poll_background_response_async(
        self, response: Any, params: Mapping[str, Any]
    ) -> Any:
        status = self._field(response, "status")
        if status not in {"queued", "in_progress", "pending"}:
            return response
        response_id = self._field(response, "id")
        if not response_id:
            raise RuntimeError("Responses background 响应缺少 id，无法轮询。")
        interval, timeout = self._background_poll_config()
        deadline = time.monotonic() + timeout
        retrieve_kwargs = self._response_retrieve_kwargs(params)
        while status in {"queued", "in_progress", "pending"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Responses background 响应 {response_id} 在 {timeout:g}s 内未完成。"
                )
            if interval:
                await asyncio.sleep(min(interval, remaining))
            response = await retry_async_call(
                lambda: self._get_aclient().responses.retrieve(response_id, **retrieve_kwargs)
            )
            status = self._field(response, "status")
        return response

    def _compat_extra_options(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (getattr(self, "_options", {}) or {}).items()
            if key in {"top_k", "repetition_penalty"}
        }

    def _validate_openai_field_compatibility(
        self, values: Mapping[str, Any], endpoint: str, protocol: str | None = None
    ) -> None:
        """拒绝当前协议不支持或兼容端点未声明的 OpenAI 字段。"""
        if self._uses_official_openai_endpoint():
            return
        effective_protocol = protocol or getattr(self, "_protocol", "chat_completions")
        if effective_protocol == "responses":
            unsupported_fields = self._OPENAI_ONLY_RESPONSES_FIELDS
        else:
            unsupported_fields = (
                self._OPENAI_ONLY_CHAT_FIELDS | self._OPENAI_ONLY_CHAT_COMPAT_FIELDS
            )
        unsupported = {
            key: value
            for key, value in values.items()
            if key in unsupported_fields and value is not None
        }
        reject_unsupported_kwargs(
            f"{getattr(self, '_provider', 'compatible')} {endpoint}",
            unsupported,
        )

    def _validate_chat_field_compatibility(self, values: Mapping[str, Any]) -> None:
        self._validate_openai_field_compatibility(values, "Chat Completions")

    @staticmethod
    def _resource_capability_for_method(method_name: str) -> str | None:
        """将 OpenAI 资源入口映射到统一的能力名称。"""
        name = method_name.removeprefix("async_")
        if name in {
            "retrieve_response",
            "cancel_response",
            "delete_response",
            "list_response_input_items",
            "connect_responses",
            "stream_responses",
            "count_input_tokens",
            "compact_responses",
        }:
            return "responses"
        if name in {
            "upload_file",
            "list_files",
            "retrieve_file",
            "file_content",
            "retrieve_file_content",
            "delete_file",
            "wait_for_file",
        }:
            return "files"
        if name in {
            "create_upload",
            "complete_upload",
            "cancel_upload",
            "create_upload_part",
            "upload_file_chunked",
        }:
            return "uploads"
        if name == "text_to_speech":
            return "audio"
        prefixes = (
            ("vector_store", "vector_stores"),
            ("fine_tuning", "fine_tuning"),
            ("eval", "evals"),
            ("conversation", "conversations"),
            ("container", "containers"),
            ("upload", "uploads"),
            ("video", "videos"),
            ("image", "images"),
            ("audio", "audio"),
            ("moderation", "moderation"),
            ("model", "models"),
            ("batch", "batches"),
        )
        for prefix, capability in prefixes:
            if name.startswith(prefix) or f"_{prefix}" in name:
                return capability
        return None

    # 动态资源树路径（``OpenAIProvider.responses.create``）的末段是方法名，
    # 与显式 Facade 名（``create_response``）不同形。只看显式名会让动态路径
    # 拿到 capability=None 直接放行，正是 server_verified_protocols 门禁要堵的
    # 旁路。这里按路径分段识别资源树。
    # 资源树分段 → 能力名。必须与 ``_resource_capability_for_method`` 的显式
    # 映射覆盖同一批能力：只覆盖 responses/files 会让
    # ``resources.batches.create`` 这类写法拿到 capability=None 直接放行，
    # 而显式名 ``create_batch`` 会被拦——同一能力因书写形式不同而区别对待，
    # 正是本门禁要消灭的旁路。
    #
    # ``uploads`` 与 ``files`` 是两项独立能力（SDK 里也分属不同资源），
    # 不能坍缩成同一个名字。
    #
    # 这里的键必须落在官方声明集 ``_OFFICIAL_OPENAI_RESOURCE_CAPABILITIES``
    # 内：段映射一旦给出声明集之外的能力名，官方端点会从「放行」变成
    # ``NotImplementedError``——而 ``OpenAIProvider.capabilities`` 里那些
    # 资源（skills/realtime/webhooks/admin/content_provenance_checks）是
    # 声明为可用的，用户看到可用、调用却被拦。
    _RESOURCE_SEGMENTS_TO_CAPABILITY: ClassVar[Mapping[str, str]] = {
        "responses": "responses",
        "input_items": "responses",
        "input_tokens": "responses",
        "files": "files",
        "uploads": "uploads",
        "batches": "batches",
        "vector_stores": "vector_stores",
        "models": "models",
        # SDK 客户端的属性是复数 ``moderations``（``client.moderations``），
        # 而显式 Facade 名 ``create_moderation`` 解析出的能力名是单数。两个
        # 拼写都要映射到同一能力，否则 ``resources.moderations.create`` 拿到
        # capability=None 直接放行，而 ``create_moderation`` 会被拦。
        "moderations": "moderation",
        "images": "images",
        "audio": "audio",
        "videos": "videos",
        "conversations": "conversations",
        "containers": "containers",
        "fine_tuning": "fine_tuning",
        "evals": "evals",
    }

    @classmethod
    def _resource_capability_for_path(cls, method_name: str) -> str | None:
        """从动态资源树路径推断能力名称。"""
        for segment in method_name.split("."):
            capability = cls._RESOURCE_SEGMENTS_TO_CAPABILITY.get(segment)
            if capability is not None:
                return capability
        return None

    def _require_provider_resource(
        self, method_name: str, kwargs: Mapping[str, Any] | None = None
    ) -> None:
        """在原生资源方法真正触达 SDK 前校验渠道能力。"""
        capability = self._resource_capability_for_method(method_name)
        if capability is None:
            # 显式 Facade 名匹配不上时，再按动态资源树路径判断。
            capability = self._resource_capability_for_path(method_name)
        if capability is None:
            return
        if capability == "responses":
            self._require_responses_resource(method_name)
            return
        # An object created with ``object.__new__`` has no instance metadata;
        # in that case there is no provider identity to gate and the SDK
        # signature/resource checks below should remain testable.  A real
        # OpenAI provider is identified explicitly by ``_provider`` and gets
        # the same resource set as the dedicated subclass.
        provider_name = getattr(self, "_provider", None)
        if provider_name is None:
            return
        declared: frozenset[str] = getattr(self, "_resource_capabilities", frozenset())
        if provider_name == "openai" and not declared:
            declared = self._OFFICIAL_OPENAI_RESOURCE_CAPABILITIES
        if capability not in declared:
            raise NotImplementedError(f"{provider_name} 未声明 {capability} 资源能力。")

    def _chat_max_tokens_key(self) -> str:
        """返回当前 Chat Completions 端点的输出长度字段名。"""
        # OpenAI 官方 SDK 3.x 已将该字段更名为 max_completion_tokens。
        # 绝大多数兼容网关（Qwen、DeepSeek、Ollama、LM Studio、Grok、
        # SiliconFlow）仍按 Chat Completions 契约使用 max_tokens。
        provider = getattr(self, "_provider", "openai")
        return (
            "max_completion_tokens"
            if provider == "openai" and self._uses_official_openai_endpoint()
            else "max_tokens"
        )

    def _embedding_options(self) -> dict[str, Any]:
        """校验并返回 Embeddings 端点接受的模型级参数。

        聊天参数不能复用到 Embeddings，否则 OpenAI SDK 会在本地校验阶段
        抛出 TypeError，兼容渠道也可能收到无效字段。客户端级配置仍可
        放在同一份 provider options 中，但不会被转发为 Embeddings 请求字段。
        """
        options = getattr(self, "_options", {}) or {}
        supported = self._EMBEDDING_MODEL_OPTION_KEYS
        allowed = supported | self._EMBEDDING_CLIENT_OPTION_KEYS
        reject_unsupported_kwargs(
            f"{getattr(self, '_provider', 'OpenAI-compatible')} Embedding 模型 options",
            {key: value for key, value in options.items() if key not in allowed},
        )
        return {key: value for key, value in options.items() if key in supported}

    @staticmethod
    def _validate_embedding_kwargs(kwargs: Mapping[str, Any]) -> None:
        allowed = {
            "dimensions",
            "encoding_format",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
        client_only = {
            "base_url",
            "default_headers",
            "default_query",
            "http_client",
            "max_retries",
            "server_verified_protocols",
        }
        unsupported = {
            key: value for key, value in kwargs.items() if key not in allowed or key in client_only
        }
        reject_unsupported_kwargs("OpenAI Embedding", unsupported)
        validate_secret_free_request_overrides(
            kwargs.get("extra_headers"),
            kwargs.get("extra_query"),
            "OpenAI Embedding",
        )

    def _uses_official_openai_endpoint(self) -> bool:
        """只把 OpenAI 官方 Responses 端点视为默认已知资源契约。

        ``provider='openai'`` 也可用于代理或自建兼容网关，因此不能仅凭
        provider 名称放行 SDK 的专属资源。缺少 ``base_url`` 时沿用 SDK
        默认端点语义；显式 URL 则必须精确指向 ``https://api.openai.com/v1``。
        """
        base_url = getattr(self, "_base_url", None)
        if not base_url:
            # 只有 OpenAI Provider 省略 Base URL 时才会落到 SDK 官方默认端点。
            return getattr(self, "_provider", "openai") == "openai"
        try:
            parsed = urlsplit(str(base_url).strip())
            port = parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme.lower() == "https"
            and parsed.hostname is not None
            and parsed.hostname.lower() == "api.openai.com"
            and (port in (None, 443))
            and parsed.path.rstrip("/") == "/v1"
            and not parsed.query
            and not parsed.fragment
        )

    def _require_responses_resource(self, operation: str) -> None:
        """阻止兼容渠道把本地 SDK 资源误当成远端 Responses 能力。"""
        # ``object.__new__`` 构造的实例没有 ``_protocol``；此时没有协议可校验，
        # 交给下面的服务端验证门禁判断（与 ``_provider`` 缺失时的处理一致）。
        protocol = getattr(self, "_protocol", None)
        if protocol is not None and protocol != "responses":
            raise ValueError(f"{operation} 仅适用于 responses 协议。")
        # 官方端点由其 SDK 契约覆盖；代理、自建网关和其它兼容渠道必须
        # 显式声明已验证的服务端协议，避免本地 SDK 资源导致远端 404/405。
        provider = getattr(self, "_provider", "openai")
        verified = "responses" in getattr(self, "_server_verified_protocols", ())
        if not verified and not self._uses_official_openai_endpoint():
            raise ValueError(
                f"{provider} 未验证 Responses {operation} 的服务端能力；"
                "请在 provider options 中显式配置 server_verified_protocols=['responses']。"
            )

    @staticmethod
    def _merge_options(request: dict[str, Any], options: dict[str, Any]) -> None:
        reserved = {"model", "messages", "input", "stream", "instructions", "tools"}
        overlap = sorted(reserved.intersection(options))
        if overlap:
            raise ValueError(f"Provider options 不允许覆盖请求字段: {', '.join(overlap)}")
        # options 是模型级默认值；调用级字段在后续构造阶段覆盖它们。
        for key, value in options.items():
            request.setdefault(key, value)

    @staticmethod
    def _validated_extra_body(
        value: Any,
        endpoint: str,
        reserved: set[str],
    ) -> dict[str, Any]:
        """校验 SDK 的扩展请求体，避免字符串或保留字段穿透到请求边界。"""
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
    def _merge_resource_kwargs(
        fixed: Mapping[str, Any],
        overrides: Mapping[str, Any] | None,
        operation: str,
        *,
        omit_none: bool = False,
    ) -> dict[str, Any]:
        """合并资源入口参数，拒绝调用方覆盖 Facade 已绑定的字段。"""
        fixed_values = {
            key: value for key, value in fixed.items() if not omit_none or value is not None
        }
        override_values = dict(overrides or {})
        overlap = sorted(set(fixed_values).intersection(override_values))
        if overlap:
            raise ValueError(f"{operation} 资源 SDK 参数重复: {', '.join(overlap)}")
        return {**fixed_values, **override_values}

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
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
        prediction: dict[str, Any] | None = None,
        web_search_options: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        reasoning: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        store: bool | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        prompt_cache_retention: str | None = None,
        safety_identifier: str | None = None,
        moderation: dict[str, Any] | None = None,
        service_tier: str | None = None,
        verbosity: str | None = None,
        **ignored: Any,
    ) -> dict[str, Any]:
        reject_unsupported_kwargs("OpenAI Chat Completions", ignored)
        extra_headers, extra_query = validate_secret_free_request_overrides(
            extra_headers,
            extra_query,
            f"{getattr(self, '_provider', 'openai')} Chat Completions",
        )
        # Chat Completions 要求 assistant 历史的每个 tool_call 都带 id；
        # 这里显式校验，其它协议继续接受 Gemini 这类无 id 的历史。
        normalized_messages = normalize_messages(
            prompt,
            system_prompt,
            messages,
            require_tool_call_ids=True,
        )

        request: dict[str, Any] = {
            "model": self._model_name,
            "messages": normalized_messages,
            "stream": stream,
        }
        configured_options = self._request_options()
        self._validate_model_options(
            configured_options,
            self._CHAT_MODEL_OPTION_KEYS,
            f"{getattr(self, '_provider', 'compatible')} Chat Completions",
        )
        configured_max_tokens = configured_options.pop("max_tokens", None)
        configured_max_completion_tokens = configured_options.pop("max_completion_tokens", None)
        if configured_max_tokens is not None and configured_max_completion_tokens is not None:
            raise ValueError(
                "Chat Completions options 不能同时设置 max_tokens 和 max_completion_tokens。"
            )
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Chat Completions",
            {"model", "messages", "input", "stream", "instructions", "tools"},
        )
        configured_explicit_extra_keys = set(configured_extra_body)
        self._validate_chat_field_compatibility(configured_options)
        self._merge_options(request, configured_options)
        configured_extra = self._compat_extra_options()
        merged_configured_extra = self._merge_extra_body(
            configured_extra_body, configured_extra, "Chat Completions"
        )
        if merged_configured_extra:
            request["extra_body"] = merged_configured_extra
        if temperature is not None:
            request["temperature"] = temperature
        elif "temperature" not in request:
            request["temperature"] = 0.7
        if tools:
            request["tools"] = tools
        max_tokens_key = self._chat_max_tokens_key()
        if max_tokens is not None:
            request[max_tokens_key] = max_tokens
            request.pop(
                "max_tokens" if max_tokens_key != "max_tokens" else "max_completion_tokens", None
            )
        elif configured_max_tokens is not None and max_tokens_key not in request:
            request[max_tokens_key] = configured_max_tokens
            request.pop(
                "max_tokens" if max_tokens_key != "max_tokens" else "max_completion_tokens", None
            )
        elif configured_max_completion_tokens is not None and max_tokens_key not in request:
            request[max_tokens_key] = configured_max_completion_tokens
            request.pop(
                "max_tokens" if max_tokens_key != "max_tokens" else "max_completion_tokens", None
            )
        if top_p is not None:
            request["top_p"] = top_p
        request_extensions = {
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "modalities": modalities,
            "audio": audio,
            "prediction": prediction,
            "web_search_options": web_search_options,
            "parallel_tool_calls": parallel_tool_calls,
            "store": store,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_options": prompt_cache_options,
            "prompt_cache_retention": prompt_cache_retention,
            "safety_identifier": safety_identifier,
            "moderation": moderation,
            "service_tier": service_tier,
            "verbosity": verbosity,
        }
        self._validate_chat_field_compatibility(request_extensions)
        for key, value in request_extensions.items():
            if value is not None:
                request[key] = value
        if repetition_penalty is not None:
            # repetition_penalty 是部分 OpenAI 兼容服务的扩展参数，
            # OpenAI 官方 Chat Completions 不接受它作为顶层字段。
            if "repetition_penalty" in configured_explicit_extra_keys:
                raise ValueError(
                    "Chat Completions repetition_penalty 与模型 options.extra_body 重复。"
                )
            request.setdefault("extra_body", {})["repetition_penalty"] = repetition_penalty
        if top_k is not None and self._protocol == "chat_completions":
            if "top_k" in configured_explicit_extra_keys:
                raise ValueError("Chat Completions top_k 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["top_k"] = top_k
        if seed is not None:
            if "seed" in configured_explicit_extra_keys:
                raise ValueError("Chat Completions seed 与模型 options.extra_body 重复。")
            request["seed"] = seed
        if stop is not None:
            request["stop"] = stop
        if response_format is not None:
            request["response_format"] = response_format
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if metadata is not None:
            request["metadata"] = metadata
        if user is not None:
            request["user"] = user
        if timeout is not None:
            request["timeout"] = timeout
        if extra_headers is not None:
            request["extra_headers"] = extra_headers
        if extra_query is not None:
            request["extra_query"] = extra_query
        if reasoning is not None:
            effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
            if effort is not None:
                if "reasoning" in configured_explicit_extra_keys:
                    raise ValueError("Chat Completions reasoning 与模型 options.extra_body 重复。")
                request["reasoning_effort"] = effort
            else:
                if "reasoning" in configured_explicit_extra_keys:
                    raise ValueError("Chat Completions reasoning 与模型 options.extra_body 重复。")
                request.setdefault("extra_body", {})["reasoning"] = reasoning
        if stream_options is not None:
            request["stream_options"] = stream_options
        request_extra_body = self._validated_extra_body(
            extra_body,
            "Chat Completions",
            set(request).difference({"extra_body"}),
        )
        if request_extra_body:
            configured_overlap = sorted(
                set(request.get("extra_body", {})).intersection(request_extra_body)
            )
            if configured_overlap:
                raise ValueError(
                    "Chat Completions extra_body 与模型 options 重复: "
                    + ", ".join(configured_overlap)
                )
            request.setdefault("extra_body", {}).update(request_extra_body)
        if request.get("extra_body"):
            request["extra_body"] = self._validated_extra_body(
                request["extra_body"],
                "Chat Completions",
                set(request).difference({"extra_body"}),
            )
        return request

    @staticmethod
    def _convert_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        converted: list[dict[str, Any]] = []
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                raise ValueError(f"Responses 工具定义[{index}] 必须是对象。")
            if "function" not in tool:
                native_tool = dict(tool)
                tool_type = native_tool.get("type")
                if not isinstance(tool_type, str) or not tool_type.strip():
                    raise ValueError(f"Responses 工具定义[{index}] 缺少有效 type。")
                if tool_type == "function":
                    name = native_tool.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError(f"Responses 工具定义[{index}] 缺少 name。")
                    parameters = native_tool.get("parameters", {"type": "object", "properties": {}})
                    if not isinstance(parameters, Mapping):
                        raise ValueError(f"Responses 工具定义[{index}].parameters 必须是对象。")
                    native_tool["parameters"] = dict(parameters)
                    native_tool.setdefault("strict", False)
                converted.append(native_tool)
                continue

            function = tool["function"]
            if not isinstance(function, Mapping):
                raise ValueError(f"Responses 工具定义[{index}].function 必须是对象。")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Responses 工具定义[{index}] 缺少 function.name。")
            parameters = function.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, Mapping):
                raise ValueError(f"Responses 工具定义[{index}].function.parameters 必须是对象。")
            response_tool: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": dict(parameters),
                "strict": function.get("strict", tool.get("strict", False)),
            }
            if "description" in function:
                response_tool["description"] = function["description"]
            for key in ("allowed_callers", "defer_loading", "output_schema"):
                if key in function:
                    response_tool[key] = function[key]
            converted.append(response_tool)
        return converted

    @staticmethod
    def _convert_responses_tool_choice(tool_choice: Any) -> Any:
        """将 Chat Completions 风格的工具选择转换为 Responses 形状。

        Responses 的指定函数形式是扁平的 ``{type, name}``，而统一请求
        也接受 ``{type: function, function: {name}}``。原生 Responses
        工具选择以及字符串模式保持不变，便于调用方直接使用 SDK 形状。
        """
        if tool_choice is None or isinstance(tool_choice, str):
            return tool_choice
        if not isinstance(tool_choice, Mapping):
            raise ValueError("Responses tool_choice 必须是字符串或对象。")

        choice = dict(tool_choice)
        if choice.get("type") != "function":
            return choice
        function = choice.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if not name:
                raise ValueError("Responses 的 function tool_choice 缺少 name。")
            converted = {key: value for key, value in choice.items() if key != "function"}
            converted["name"] = name
            return converted
        if isinstance(choice.get("name"), str) and choice["name"]:
            return choice
        raise ValueError("Responses 的 function tool_choice 缺少 name。")

    def _build_responses_request(
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
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        previous_response_id: str | None = None,
        include: list[str] | None = None,
        reasoning: dict[str, Any] | None = None,
        store: bool | None = None,
        background: bool | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        conversation: str | dict[str, Any] | None = None,
        session: dict[str, Any] | None = None,
        context_management: dict[str, Any] | None = None,
        caching: dict[str, Any] | None = None,
        expire_at: int | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        safety_identifier: str | None = None,
        moderation: dict[str, Any] | None = None,
        top_logprobs: int | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        service_tier: str | None = None,
        truncation: str | None = None,
        stream_options: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        **ignored: Any,
    ) -> dict[str, Any]:
        extra_headers, extra_query = validate_secret_free_request_overrides(
            extra_headers,
            extra_query,
            f"{getattr(self, '_provider', 'openai')} Responses",
        )
        reject_unsupported_kwargs(
            "OpenAI Responses",
            {
                "thinking": thinking,
                "caching": caching,
                "session": session,
                "expire_at": expire_at,
            },
        )
        reject_unsupported_kwargs("OpenAI Responses", ignored)
        input_value = normalize_responses_input(
            prompt,
            system_prompt,
            messages,
            provider=getattr(self, "_provider", "openai"),
        )
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": input_value,
            "stream": stream,
        }
        configured_options = self._request_options()
        self._validate_model_options(
            configured_options,
            self._RESPONSES_MODEL_OPTION_KEYS,
            f"{getattr(self, '_provider', 'compatible')} Responses",
        )
        # Keep the common model configuration ergonomic while emitting the
        # current Responses SDK shape. These names are valid for Chat
        # Completions, but Responses expects them under a different field.
        configured_response_format = configured_options.pop("response_format", None)
        configured_verbosity = configured_options.pop("verbosity", None)
        configured_max_tokens = configured_options.pop("max_tokens", None)
        configured_max_completion_tokens = configured_options.pop("max_completion_tokens", None)
        configured_max_output_tokens = configured_options.get("max_output_tokens")
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
                "Responses options 不能同时设置多个输出长度字段: " + ", ".join(configured_limits)
            )
        if configured_max_output_tokens is not None:
            configured_options.pop("max_output_tokens", None)
        configured_frequency_penalty = configured_options.pop("frequency_penalty", None)
        configured_presence_penalty = configured_options.pop("presence_penalty", None)
        configured_top_logprobs = configured_options.pop("top_logprobs", None)
        configured_unsupported = {
            key: configured_options.pop(key, None)
            for key in ("thinking", "caching", "session", "expire_at")
        }
        reject_unsupported_kwargs("OpenAI Responses", configured_unsupported)
        configured_background = (getattr(self, "_options", {}) or {}).get("background")
        configured_stop = configured_options.pop("stop", None)
        configured_seed = configured_options.pop("seed", None)
        configured_reasoning_effort = configured_options.pop("reasoning_effort", None)
        configured_text = configured_options.pop("text", None)
        configured_extra_body = self._validated_extra_body(
            configured_options.pop("extra_body", None),
            "Responses",
            {"model", "messages", "input", "stream", "instructions", "tools"},
        )
        configured_explicit_extra_keys = set(configured_extra_body)
        self._validate_openai_field_compatibility(configured_options, "Responses")
        if configured_text is not None:
            if not isinstance(configured_text, Mapping):
                raise ValueError("Responses options.text 必须是对象。")
            request["text"] = dict(configured_text)
        self._merge_options(request, configured_options)
        configured_extra = self._compat_extra_options()
        for key, value in (("stop", configured_stop), ("seed", configured_seed)):
            if value is None:
                continue
            if key in configured_extra_body or key in configured_extra:
                raise ValueError(f"Responses options.{key} 与 extra_body 重复。")
            configured_extra[key] = value
        merged_configured_extra = self._merge_extra_body(
            configured_extra_body, configured_extra, "Responses"
        )
        if merged_configured_extra:
            request["extra_body"] = merged_configured_extra
        if temperature is not None:
            request["temperature"] = temperature
        if system_prompt and messages is None:
            request["instructions"] = system_prompt
        converted_tools = self._convert_responses_tools(tools)
        if converted_tools:
            request["tools"] = converted_tools
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens
        elif configured_limits:
            request["max_output_tokens"] = next(iter(configured_limits.values()))
        if top_p is not None:
            request["top_p"] = top_p
        effective_frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else configured_frequency_penalty
        )
        effective_presence_penalty = (
            presence_penalty if presence_penalty is not None else configured_presence_penalty
        )
        unsupported_sampling = {
            key: value
            for key, value in {
                "frequency_penalty": effective_frequency_penalty,
                "presence_penalty": effective_presence_penalty,
            }.items()
            if value is not None
        }
        # OpenAI Responses 3.3 has no top-level stop/seed/frequency/presence
        # fields. Keep these useful extensions for compatible gateways, but fail
        # early for the official endpoint instead of sending a misleading request.
        if self._uses_official_openai_endpoint():
            reject_unsupported_kwargs("OpenAI Responses", unsupported_sampling)
        for key, value in unsupported_sampling.items():
            if key in configured_explicit_extra_keys:
                raise ValueError(f"Responses {key} 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})[key] = value
        # OpenAI Responses 3.3 has no top-level stop/seed/top_k fields. Keep
        # these useful extensions for compatible gateways, but fail early for
        # the official endpoint instead of sending a misleading request.
        unsupported_official = {
            key: value
            for key, value in {
                "stop": stop,
                "seed": seed,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }.items()
            if value is not None
        }
        configured_unsupported_official: dict[str, Any] = {
            key: value
            for key, value in {
                **{
                    key: self._options.get(key)
                    for key in ("stop", "seed", "top_k", "frequency_penalty", "presence_penalty")
                    if key in getattr(self, "_options", {})
                },
                **{
                    key: merged_configured_extra.get(key)
                    for key in (
                        "stop",
                        "seed",
                        "top_k",
                        "frequency_penalty",
                        "presence_penalty",
                    )
                    if key in merged_configured_extra
                },
            }.items()
            if value is not None
        }
        if configured_unsupported_official and self._uses_official_openai_endpoint():
            reject_unsupported_kwargs("OpenAI Responses", configured_unsupported_official)
        if unsupported_official and self._uses_official_openai_endpoint():
            reject_unsupported_kwargs("OpenAI Responses", unsupported_official)
        if top_k is not None:
            if "top_k" in configured_explicit_extra_keys:
                raise ValueError("Responses top_k 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["top_k"] = top_k
        if seed is not None:
            if "seed" in configured_explicit_extra_keys:
                raise ValueError("Responses seed 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["seed"] = seed
        if stop is not None:
            if "stop" in configured_explicit_extra_keys:
                raise ValueError("Responses stop 与模型 options.extra_body 重复。")
            request.setdefault("extra_body", {})["stop"] = stop
        effective_response_format = (
            response_format if response_format is not None else configured_response_format
        )
        if effective_response_format is not None:
            request.setdefault("text", {}).update(
                self._convert_responses_format(effective_response_format)
            )
        response_extensions = {
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_options": prompt_cache_options,
            "prompt_cache_retention": prompt_cache_retention,
            "safety_identifier": safety_identifier,
            "moderation": moderation,
            "verbosity": verbosity,
            "store": store,
        }
        self._validate_openai_field_compatibility(
            response_extensions, "Responses", protocol="responses"
        )
        if tool_choice is not None:
            request["tool_choice"] = self._convert_responses_tool_choice(tool_choice)
        elif "tool_choice" in request:
            request["tool_choice"] = self._convert_responses_tool_choice(request["tool_choice"])
        if metadata is not None:
            request["metadata"] = metadata
        if user is not None:
            request["user"] = user
        if timeout is not None:
            request["timeout"] = timeout
        if extra_headers is not None:
            request["extra_headers"] = extra_headers
        if extra_query is not None:
            request["extra_query"] = extra_query
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        if include is not None:
            request["include"] = include
        if reasoning is not None:
            if "reasoning" in configured_explicit_extra_keys:
                raise ValueError("Responses reasoning 与模型 options.extra_body 重复。")
            request["reasoning"] = reasoning
        elif configured_reasoning_effort is not None:
            if "reasoning" in configured_explicit_extra_keys:
                raise ValueError("Responses reasoning_effort 与模型 options.extra_body 重复。")
            request["reasoning"] = {"effort": configured_reasoning_effort}
        if store is not None:
            request["store"] = store
        if background is not None:
            request["background"] = background
        if parallel_tool_calls is not None:
            request["parallel_tool_calls"] = parallel_tool_calls
        if max_tool_calls is not None:
            request["max_tool_calls"] = max_tool_calls
        if conversation is not None:
            request["conversation"] = conversation
        if context_management is not None:
            request["context_management"] = context_management
        if prompt_cache_options is not None:
            request["prompt_cache_options"] = prompt_cache_options
        if safety_identifier is not None:
            request["safety_identifier"] = safety_identifier
        if moderation is not None:
            request["moderation"] = moderation
        effective_top_logprobs = (
            top_logprobs if top_logprobs is not None else configured_top_logprobs
        )
        if effective_top_logprobs is not None:
            request["top_logprobs"] = effective_top_logprobs
        effective_verbosity = verbosity if verbosity is not None else configured_verbosity
        if effective_verbosity is not None:
            # openai>=3 accepts Responses verbosity inside `text`; it is not
            # a valid top-level argument for `responses.create()`.
            request.setdefault("text", {})["verbosity"] = effective_verbosity
        if prompt_cache_key is not None:
            request["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            request["prompt_cache_retention"] = prompt_cache_retention
        if service_tier is not None:
            request["service_tier"] = service_tier
        if truncation is not None:
            request["truncation"] = truncation
        if stream_options is not None:
            request["stream_options"] = stream_options
        effective_background = background if background is not None else configured_background
        if effective_background is not None:
            if not isinstance(effective_background, bool):
                raise ValueError("Responses background 必须是布尔值。")
            if effective_background and stream:
                raise ValueError("Responses background 请求不支持流式模式。")
            request["background"] = effective_background
        request_extra_body = self._validated_extra_body(
            extra_body,
            "Responses",
            set(request).difference({"extra_body"}),
        )
        if request_extra_body:
            configured_overlap = sorted(
                set(request.get("extra_body", {})).intersection(request_extra_body)
            )
            if configured_overlap:
                raise ValueError(
                    "Responses extra_body 与模型 options 重复: " + ", ".join(configured_overlap)
                )
            request.setdefault("extra_body", {}).update(request_extra_body)
        if request.get("extra_body"):
            request["extra_body"] = self._validated_extra_body(
                request["extra_body"],
                "Responses",
                set(request).difference({"extra_body"}),
            )
        return request

    @staticmethod
    def _convert_responses_format(response_format: dict[str, Any]) -> dict[str, Any]:
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", response_format)
            json_schema = schema.get("schema", {}) if isinstance(schema, Mapping) else {}
            return {
                "format": {
                    "type": "json_schema",
                    "name": schema.get("name", "response")
                    if isinstance(schema, Mapping)
                    else "response",
                    "schema": json_schema or response_format.get("schema", schema),
                    "strict": schema.get("strict", True) if isinstance(schema, Mapping) else True,
                }
            }
        if response_format.get("type") == "json_object":
            return {"format": {"type": "json_object"}}
        if "format" in response_format:
            return response_format
        raise ValueError("Responses response_format 仅支持 json_schema 或 json_object。")

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _extract_response_text(cls, response: Any) -> str:
        output_text = cls._field(response, "output_text")
        if isinstance(output_text, str):
            return output_text

        chunks: list[str] = []
        choices = cls._field(response, "choices", []) or []
        for choice in choices:
            message = cls._field(choice, "message") or cls._field(choice, "delta")
            text = cls._field(message, "content")
            if text:
                chunks.append(content_to_text(text))
        if chunks:
            return "".join(chunks)
        for item in cls._field(response, "output", []) or []:
            for content in cls._field(item, "content", []) or []:
                text = cls._field(content, "text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    @classmethod
    def _extract_tool_calls(cls, response: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        choices = cls._field(response, "choices", []) or []
        for choice in choices:
            message = cls._field(choice, "message") or cls._field(choice, "delta")
            for call in cls._field(message, "tool_calls", []) or []:
                function = cls._field(call, "function")
                calls.append(
                    {
                        "id": cls._field(call, "id"),
                        "type": cls._field(call, "type", "function"),
                        "name": cls._field(function, "name"),
                        "arguments": normalize_tool_arguments(
                            cls._field(function, "arguments", "")
                        ),
                    }
                )
        for item in cls._field(response, "output", []) or []:
            if cls._field(item, "type") in {"function_call", "custom_tool_call"}:
                calls.append(
                    {
                        "id": cls._field(item, "call_id") or cls._field(item, "id"),
                        "type": cls._field(item, "type"),
                        "name": cls._field(item, "name"),
                        "arguments": normalize_tool_arguments(
                            cls._field(item, "arguments", cls._field(item, "input", ""))
                        ),
                    }
                )
        return calls

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, Any]:
        return normalize_usage(cls._field(response, "usage"))

    @classmethod
    def _extract_reasoning(cls, response: Any) -> str:
        values: list[str] = []
        choices = cls._field(response, "choices", []) or []
        for choice in choices:
            message = cls._field(choice, "message") or cls._field(choice, "delta")
            for name in ("reasoning_content", "reasoning"):
                value = cls._field(message, name)
                if isinstance(value, str):
                    values.append(value)
        for item in cls._field(response, "output", []) or []:
            if cls._field(item, "type") in {"reasoning", "reasoning_item"}:
                summary = cls._field(item, "summary")
                if isinstance(summary, str):
                    values.append(summary)
                elif isinstance(summary, (list, tuple)):
                    for entry in summary:
                        text = cls._field(entry, "text")
                        if isinstance(text, str):
                            values.append(text)
                for content in cls._field(item, "content", []) or []:
                    text = cls._field(content, "text")
                    if isinstance(text, str):
                        values.append(text)
                fallback = cls._field(item, "text")
                if isinstance(fallback, str):
                    values.append(fallback)
        return "".join(values)

    @classmethod
    def _extract_refusal(cls, response: Any) -> str | None:
        """提取 Chat Completions 与 Responses 的统一拒答文本。"""
        choices = cls._field(response, "choices", []) or []
        for choice in choices:
            message = cls._field(choice, "message") or cls._field(choice, "delta")
            refusal = cls._field(message, "refusal")
            if isinstance(refusal, str) and refusal:
                return refusal
        for item in cls._field(response, "output", []) or []:
            refusal = cls._field(item, "refusal")
            if isinstance(refusal, str) and refusal:
                return refusal
            for content in cls._field(item, "content", []) or []:
                if cls._field(content, "type") != "refusal":
                    continue
                refusal = cls._field(content, "refusal") or cls._field(content, "text")
                if isinstance(refusal, str) and refusal:
                    return refusal
        return None

    @classmethod
    def _chat_status_is_error(cls, status: Any) -> bool:
        """识别兼容网关放在 HTTP 200 响应体中的业务失败状态。"""
        if status is None or isinstance(status, bool):
            return False
        if isinstance(status, int):
            return status >= 400
        normalized = str(status).strip().lower()
        if not normalized:
            return False
        if normalized.isdigit():
            return int(normalized) >= 400
        return normalized in {
            "error",
            "failed",
            "failure",
            "cancelled",
            "canceled",
            "incomplete",
            "rejected",
            "denied",
        } or normalized.startswith(("error_", "failed_", "failure_"))

    @classmethod
    def _chat_response_error_detail(cls, response: Any) -> str | None:
        """提取兼容网关常见的顶层业务错误字段。"""
        for name in ("error", "msg", "message", "detail", "reason", "code", "error_code"):
            value = cls._field(response, name)
            if value is None or value == "":
                continue
            if name == "error":
                return redact_sensitive_text(cls._error_message(value, "未知错误。"))
            return redact_sensitive_text(str(value))
        return None

    @classmethod
    def _raise_for_chat_response_error(
        cls,
        response: Any,
        *,
        allow_empty_choices: bool = False,
        provider: str = "Chat Completions",
    ) -> None:
        """把 Chat Completions 的协议错误和网关业务错误转换为异常。

        部分兼容网关会以 HTTP 200 返回 status/msg，并将 choices 设为
        None。如果只按 SDK 的异常或 choices 提取，这类失败会被误报为空文本。
        流式响应允许没有 choices 的 usage chunk，但只要出现错误状态或错误
        字段仍必须显式失败。
        """
        detail = cls._chat_response_error_detail(response)
        status = cls._field(response, "status")
        if detail is not None and cls._field(response, "error") is not None:
            raise RuntimeError(f"{provider} 返回错误: {detail}")
        if cls._chat_status_is_error(status):
            suffix = f": {detail}" if detail else "。"
            raise RuntimeError(f"{provider} 返回业务错误 (status={status}){suffix}")

        choices = cls._field(response, "choices")
        if allow_empty_choices:
            if choices is None and detail is not None:
                raise RuntimeError(f"{provider} 返回错误: {detail}")
            return
        if choices is None or (isinstance(choices, (list, tuple)) and not choices):
            suffix = f": {detail}" if detail else "。"
            raise RuntimeError(f"{provider} 响应缺少有效 choices{suffix}")

    def complete(self, request: CompletionRequest | None = None, **kwargs: Any) -> CompletionResult:
        request = coerce_completion_request(request, kwargs, "OpenAI-compatible complete")
        request = request.copy_with(stream=False)
        if self._protocol == "responses":
            self._require_responses_resource("complete")
            params = self._build_responses_request(**request.to_invoke_kwargs())
            response = retry_sync_call(lambda: self._get_client().responses.create(**params))
            if params.get("background"):
                response = self._poll_background_response(response, params)
            self._raise_for_response_error(response)
        else:
            params = self._build_chat_request(**request.to_invoke_kwargs())
            response = retry_sync_call(lambda: self._get_client().chat.completions.create(**params))
            self._raise_for_chat_response_error(
                response,
                provider=f"{getattr(self, '_provider', 'OpenAI-compatible')} Chat Completions",
            )
        choices = self._field(response, "choices", []) or []
        finish_reason = (
            self._field(response, "status")
            if self._protocol == "responses"
            else self._field(choices[0], "finish_reason")
            if choices
            else None
        )
        return CompletionResult(
            text=self._extract_response_text(response),
            tool_calls=self._extract_tool_calls(response),
            usage=self._extract_usage(response),
            finish_reason=finish_reason,
            response_id=self._field(response, "id"),
            refusal=self._extract_refusal(response),
            reasoning=self._extract_reasoning(response),
            raw=response,
        )

    async def acomplete(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> CompletionResult:
        """使用 AsyncOpenAI 聚合完整结果，保留 SDK 元数据。"""
        request = coerce_completion_request(request, kwargs, "OpenAI-compatible acomplete")
        request = request.copy_with(stream=False)
        if self._protocol == "responses":
            self._require_responses_resource("acomplete")
            params = self._build_responses_request(**request.to_invoke_kwargs())
            response = await retry_async_call(
                lambda: self._get_aclient().responses.create(**params)
            )
            if params.get("background"):
                response = await self._poll_background_response_async(response, params)
            self._raise_for_response_error(response)
        else:
            params = self._build_chat_request(**request.to_invoke_kwargs())
            response = await retry_async_call(
                lambda: self._get_aclient().chat.completions.create(**params)
            )
            self._raise_for_chat_response_error(
                response,
                provider=f"{getattr(self, '_provider', 'OpenAI-compatible')} Chat Completions",
            )
        choices = self._field(response, "choices", []) or []
        finish_reason = (
            self._field(response, "status")
            if self._protocol == "responses"
            else self._field(choices[0], "finish_reason")
            if choices
            else None
        )
        return CompletionResult(
            text=self._extract_response_text(response),
            tool_calls=self._extract_tool_calls(response),
            usage=self._extract_usage(response),
            finish_reason=finish_reason,
            response_id=self._field(response, "id"),
            refusal=self._extract_refusal(response),
            reasoning=self._extract_reasoning(response),
            raw=response,
        )

    @classmethod
    def _chat_stream_events(cls, chunk: Any) -> list[StreamEvent]:
        raise_for_stream_error_event(chunk, "Chat Completions")
        cls._raise_for_chat_response_error(chunk, allow_empty_choices=True)
        choices = cls._field(chunk, "choices", []) or []
        usage = cls._extract_usage(chunk)
        if not choices:
            return [StreamEvent(type="usage", usage=usage, raw=chunk)] if usage else []
        choice = choices[0]
        delta = cls._field(choice, "delta")
        text = content_to_text(cls._field(delta, "content", ""))
        reasoning = cls._field(delta, "reasoning_content") or cls._field(delta, "reasoning") or ""
        refusal = cls._field(delta, "refusal")
        tool_calls = cls._field(delta, "tool_calls", []) or []
        finish_reason = cls._field(choice, "finish_reason")
        events: list[StreamEvent] = []

        if isinstance(refusal, str) and refusal:
            events.append(StreamEvent(type="refusal_delta", refusal=refusal, raw=chunk))

        response_id = cls._field(chunk, "id")
        if text:
            events.append(
                StreamEvent(
                    type="text_delta",
                    text=text,
                    response_id=response_id,
                    raw=chunk,
                )
            )
        if isinstance(reasoning, str) and reasoning:
            events.append(
                StreamEvent(
                    type="reasoning_delta",
                    reasoning=reasoning,
                    response_id=response_id,
                    raw=chunk,
                )
            )
        if tool_calls:
            for position, call in enumerate(tool_calls):
                function = cls._field(call, "function")
                events.append(
                    StreamEvent(
                        type="tool_call_delta",
                        tool_call={
                            "id": cls._field(call, "id"),
                            "index": cls._field(call, "index", position),
                            "type": cls._field(call, "type", "function"),
                            "name": cls._field(function, "name"),
                            "arguments": cls._field(function, "arguments", ""),
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
                    response_id=cls._field(chunk, "id"),
                    raw=chunk,
                )
            )
        elif usage:
            if events:
                events[-1].usage = usage
            else:
                events.append(StreamEvent(type="usage", usage=usage, raw=chunk))
        return events

    @classmethod
    def _chat_stream_event(cls, chunk: Any) -> StreamEvent | None:
        """兼容旧调用方，返回当前 chunk 的第一个统一事件。"""
        events = cls._chat_stream_events(chunk)
        return events[0] if events else None

    @classmethod
    def _response_tool_keys(cls, event: Any, item: Any = None) -> list[Any]:
        """返回 Responses 工具事件可用的稳定关联键。"""
        item = item if item is not None else event
        item_id = cls._field(event, "item_id")
        if item_id is None:
            item_id = cls._field(item, "id")
        output_index = cls._field(event, "output_index")
        if output_index is None:
            output_index = cls._field(item, "output_index")
        call_id = cls._field(event, "call_id")
        if call_id is None:
            call_id = cls._field(item, "call_id")
        keys: list[Any] = []
        for prefix, value in (
            ("item_id", item_id),
            ("index", output_index),
            ("call_id", call_id),
        ):
            if value is None:
                continue
            try:
                hash(value)
            except TypeError:
                continue
            keys.append((prefix, value))
        return keys

    @classmethod
    def _remember_response_tool_metadata(
        cls,
        event: Any,
        metadata: dict[Any, dict[str, Any]],
    ) -> None:
        """记录 output_item.added 中后续 delta/done 事件需要的工具元数据。"""
        item = cls._field(event, "item") or event
        item_type = cls._field(item, "type")
        if item_type not in {"function_call", "custom_tool_call"}:
            return
        item_id = cls._field(item, "id") or cls._field(event, "item_id")
        call_id = cls._field(item, "call_id") or cls._field(event, "call_id")
        output_index = cls._field(event, "output_index")
        if output_index is None:
            output_index = cls._field(item, "output_index")
        values: dict[str, Any] = {
            "id": call_id or item_id,
            "call_id": call_id,
            "index": output_index,
            "type": item_type,
            "name": cls._field(item, "name"),
        }
        for field_name in ("arguments", "input"):
            value = cls._field(item, field_name)
            if value is not None:
                values["arguments"] = normalize_tool_arguments(value)
                break
        for key in cls._response_tool_keys(event, item):
            stored = metadata.setdefault(key, {})
            for name, value in values.items():
                if value is not None and value != "":
                    stored[name] = value

    @classmethod
    def _response_tool_metadata(
        cls,
        event: Any,
        metadata: Mapping[Any, Mapping[str, Any]] | None,
        item: Any = None,
    ) -> dict[str, Any]:
        """按 item_id/output_index/call_id 合并已记录的工具元数据。"""
        if not metadata:
            return {}
        merged: dict[str, Any] = {}
        for key in cls._response_tool_keys(event, item):
            values = metadata.get(key)
            if not values:
                continue
            for name, value in values.items():
                merged.setdefault(name, value)
        return merged

    @classmethod
    def _responses_stream_event(
        cls,
        event: Any,
        tool_metadata: dict[Any, dict[str, Any]] | None = None,
        *,
        error_provider: str = "Responses API",
        response_error_handler: Any | None = None,
    ) -> StreamEvent | None:
        event_type = cls._field(event, "type", "")
        if event_type == "response.output_item.added":
            if tool_metadata is not None:
                cls._remember_response_tool_metadata(event, tool_metadata)
            return None
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = cls._field(event, "delta")
            is_refusal = event_type.endswith("refusal.delta")
            return StreamEvent(
                type="refusal_delta" if is_refusal else "text_delta",
                text="" if is_refusal else (delta if isinstance(delta, str) else ""),
                refusal=delta if is_refusal and isinstance(delta, str) else None,
                raw=event,
            )
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            delta = cls._field(event, "delta")
            return StreamEvent(
                type="reasoning_delta", reasoning=delta if isinstance(delta, str) else "", raw=event
            )
        if event_type.endswith(".delta") and (
            "function_call_arguments" in event_type or "custom_tool_call_input" in event_type
        ):
            metadata = cls._response_tool_metadata(event, tool_metadata)
            item_type = metadata.get(
                "type",
                "custom_tool_call" if "custom_tool_call_input" in event_type else "function_call",
            )
            item_id = cls._field(event, "item_id")
            call_id = cls._field(event, "call_id") or metadata.get("call_id")
            output_index = cls._field(event, "output_index")
            if output_index is None:
                output_index = metadata.get("index")
            return StreamEvent(
                type="tool_call_delta",
                tool_call={
                    # ``id`` 在流式路径上必须与 ``output_item.done`` 及非流式
                    # 路径取同一语义（call_id）。取 ``item_id`` 会让同一轮工具
                    # 调用在 delta 与 done 两个事件里得到不同的 id，且是否一致
                    # 取决于事件到达顺序；调用方按 ``tool_call["id"]`` 回填
                    # ``role=tool`` 时就会配不上。
                    "id": call_id or item_id or metadata.get("id"),
                    "call_id": call_id,
                    "index": output_index
                    if output_index is not None
                    else cls._field(event, "index"),
                    "type": item_type,
                    "name": cls._field(event, "name") or metadata.get("name"),
                    "arguments": normalize_tool_arguments(cls._field(event, "delta", "")),
                },
                raw=event,
            )
        if event_type in {
            "response.output_item.done",
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.done",
        }:
            item = cls._field(event, "item")
            metadata = cls._response_tool_metadata(event, tool_metadata, item)
            item_type = cls._field(item, "type") if item is not None else None
            item_type = item_type or metadata.get("type")
            if (
                item_type in {"function_call", "custom_tool_call"}
                or "function_call_arguments" in event_type
                or "custom_tool_call_input" in event_type
            ):
                call_id = (
                    cls._field(item, "call_id")
                    or cls._field(event, "call_id")
                    or metadata.get("call_id")
                )
                output_index = cls._field(event, "output_index")
                if output_index is None:
                    output_index = metadata.get("index")
                argument_value: Any = None
                for source in (item, event):
                    for field_name in ("arguments", "input"):
                        value = cls._field(source, field_name)
                        if value is not None:
                            argument_value = value
                            break
                    if argument_value is not None:
                        break
                if argument_value is None:
                    argument_value = metadata.get("arguments", "")
                tool_call: dict[str, Any] = {
                    # 与非流式路径（``call_id or id``）和 delta 事件保持同一
                    # 语义：``id`` 是调用标识符，不是 Responses 输出项标识符。
                    # 取 ``item.id`` 或 ``item_id`` 会让同一轮工具调用在 delta
                    # 与 done 事件里得到不同的 id，调用方按 ``tool_call["id"]``
                    # 回填 ``role=tool`` 时就会配不上。上面解析出的 ``call_id``
                    # 才是权威来源。
                    "id": call_id
                    or cls._field(item, "id")
                    or cls._field(event, "item_id")
                    or metadata.get("id"),
                    "index": output_index
                    if output_index is not None
                    else cls._field(event, "index"),
                    "type": item_type
                    if item_type in {"function_call", "custom_tool_call"}
                    else "function_call",
                    "name": cls._field(item, "name")
                    or cls._field(event, "name")
                    or metadata.get("name"),
                    "arguments": normalize_tool_arguments(argument_value),
                }
                if call_id is not None:
                    tool_call["call_id"] = call_id
                return StreamEvent(
                    type="tool_call_completed",
                    tool_call=tool_call,
                    raw=event,
                )
        if event_type in {"error", "response.error"}:
            raise_for_stream_error_event(event, error_provider)
        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }:
            response = cls._field(event, "response") or event
            usage = cls._extract_usage(response)
            status = cls._field(response, "status")
            if status in {"failed", "incomplete", "cancelled"}:
                handler = response_error_handler or cls._raise_for_response_error
                handler(response)
            if event_type != "response.completed":
                raise RuntimeError(f"{error_provider} 收到终止事件: {event_type}。")
            return StreamEvent(
                type="finish",
                usage=usage,
                finish_reason=status,
                response_id=cls._field(response, "id"),
                raw=event,
            )
        return None

    @classmethod
    def _merge_response_tool_call(
        cls,
        accumulator: dict[Any, dict[str, Any]],
        tool_call: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """合并 Responses 的完整工具调用元数据。

        ``function_call_arguments.done`` 使用 item_id，而
        ``output_item.done`` 通常带 call_id；两者必须按 output_index 归并。
        """
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
        # ``id`` 必须与调用标识符一致（非流式路径用 ``call_id or id``，delta
        # 事件也用 call_id）。``output_item.done`` 带的 ``id`` 是输出项 ID，
        # 按上面顺序会覆盖 delta 事件写入的 call_id，使同一轮工具调用在
        # delta 与 completed 两个事件里得到不同的 id。这里统一回 call_id。
        if merged.get("call_id"):
            merged["id"] = merged["call_id"]
        arguments = tool_call.get("arguments")
        if arguments is not None and arguments != "":
            merged["arguments"] = normalize_tool_arguments(arguments)
        merged.setdefault("type", "function_call")
        merged.setdefault("arguments", "")
        return key, merged

    @classmethod
    def _chat_completion_events(
        cls,
        converted: StreamEvent,
        accumulator: dict[Any, dict[str, Any]],
        completed: set[Any],
    ) -> list[StreamEvent]:
        """在 Chat Completions finish 前发出一次完整工具调用事件。"""
        if converted.type == "tool_call_delta" and converted.tool_call:
            merge_tool_call_fragment(accumulator, converted.tool_call)
            return [converted]
        if converted.type != "finish":
            return [converted]

        events: list[StreamEvent] = []
        for key, tool_call in accumulator.items():
            if key in completed:
                continue
            validate_complete_tool_call(tool_call, "OpenAI Chat Completions")
            completed.add(key)
            events.append(
                StreamEvent(
                    type="tool_call_completed",
                    tool_call=dict(tool_call),
                    usage={},
                    finish_reason=converted.finish_reason,
                    response_id=converted.response_id,
                    raw=converted.raw,
                )
            )
        events.append(converted)
        return events

    @classmethod
    def _responses_completion_event(
        cls,
        converted: StreamEvent,
        accumulator: dict[Any, dict[str, Any]],
    ) -> tuple[Any, StreamEvent] | None:
        if converted.type != "tool_call_completed" or not converted.tool_call:
            return None
        key, merged = cls._merge_response_tool_call(accumulator, converted.tool_call)
        validate_complete_tool_call(merged, "OpenAI Responses")
        return key, StreamEvent(
            type="tool_call_completed",
            tool_call=dict(merged),
            response_id=converted.response_id,
            raw=converted.raw,
        )

    def stream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Generator[StreamEvent, None, None]:
        request = coerce_completion_request(
            request, kwargs, "OpenAI-compatible stream_events"
        ).copy_with(
            stream=True,
        )
        if self._protocol == "responses":
            self._require_responses_resource("stream_events")
            params = self._build_responses_request(**request.to_invoke_kwargs())
            response_tool_metadata: dict[Any, dict[str, Any]] = {}
            response_tool_calls: dict[Any, dict[str, Any]] = {}
            pending_completions: dict[Any, StreamEvent] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            for event in retry_sync_stream(
                lambda: self._get_client().responses.create(**params),
                lambda event: raise_for_stream_error_event(event, self._provider),
            ):
                if terminal_seen:
                    continue
                event_type = self._field(event, "type", "")
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
                                validate_complete_tool_call(tool_call, "OpenAI Responses")
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
                raise RuntimeError("Responses API 流在 response.completed 之前结束。")
            return
        params = self._build_chat_request(**request.to_invoke_kwargs())
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_chat_tool_calls: set[Any] = set()
        terminal_seen = False
        for chunk in retry_sync_stream(
            lambda: self._get_client().chat.completions.create(**params),
            lambda chunk: raise_for_stream_error_event(
                chunk, getattr(self, "_provider", "OpenAI-compatible")
            ),
        ):
            raise_for_stream_error_event(chunk, getattr(self, "_provider", "OpenAI-compatible"))
            for converted in self._chat_stream_events(chunk):
                if converted.type == "finish":
                    terminal_seen = True
                yield from self._chat_completion_events(
                    converted, tool_calls, completed_chat_tool_calls
                )
        if not terminal_seen:
            raise RuntimeError("Chat Completions 流在 finish 事件之前结束。")

    async def astream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        request = coerce_completion_request(
            request, kwargs, "OpenAI-compatible astream_events"
        ).copy_with(
            stream=True,
        )
        if self._protocol == "responses":
            self._require_responses_resource("astream_events")
            params = self._build_responses_request(**request.to_invoke_kwargs())
            response_tool_metadata: dict[Any, dict[str, Any]] = {}
            response_tool_calls: dict[Any, dict[str, Any]] = {}
            pending_completions: dict[Any, StreamEvent] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            async for event in retry_async_stream(
                lambda: self._get_aclient().responses.create(**params),
                lambda event: raise_for_stream_error_event(event, self._provider),
            ):
                if terminal_seen:
                    continue
                event_type = self._field(event, "type", "")
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
                                validate_complete_tool_call(tool_call, "OpenAI Responses")
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
                raise RuntimeError("Responses API 异步流在 response.completed 之前结束。")
            return
        params = self._build_chat_request(**request.to_invoke_kwargs())
        tool_calls: dict[Any, dict[str, Any]] = {}
        completed_chat_tool_calls: set[Any] = set()
        terminal_seen = False
        async for chunk in retry_async_stream(
            lambda: self._get_aclient().chat.completions.create(**params),
            lambda chunk: raise_for_stream_error_event(
                chunk, getattr(self, "_provider", "OpenAI-compatible")
            ),
        ):
            raise_for_stream_error_event(chunk, getattr(self, "_provider", "OpenAI-compatible"))
            for converted in self._chat_stream_events(chunk):
                if converted.type == "finish":
                    terminal_seen = True
                for event in self._chat_completion_events(
                    converted, tool_calls, completed_chat_tool_calls
                ):
                    yield event
        if not terminal_seen:
            raise RuntimeError("Chat Completions 异步流在 finish 事件之前结束。")

    def parse(
        self,
        text_format: Any,
        request: CompletionRequest | None = None,
        **kwargs: Any,
    ) -> Any:
        """调用 OpenAI Chat/Responses 的原生结构化解析入口。"""
        if text_format is None:
            raise ValueError("OpenAI parse 必须提供 text_format。")
        request = coerce_completion_request(request, kwargs, "OpenAI-compatible parse")
        request = request.copy_with(stream=False)
        values = request.to_invoke_kwargs()
        if self._protocol == "responses":
            self._require_responses_resource("parse")
            params = self._build_responses_request(**values)
            if params.get("background"):
                raise ValueError(
                    "Responses parse 不支持 background=true；"
                    "请使用 complete/acomplete 等待后台任务完成后再处理结果。"
                )
            # ``stream`` is an internal unified-request control flag. The
            # Responses parse helper always returns one parsed result and its
            # SDK signature does not accept this flag.
            params.pop("stream", None)
            text_config = params.pop("text", None)
            if isinstance(text_config, Mapping) and "format" in text_config:
                raise ValueError(
                    "Responses parse 不能同时使用 request.response_format 和 text_format。"
                )
            if text_config:
                params["text"] = text_config
            return self._get_client().responses.parse(text_format=text_format, **params)
        params = self._build_chat_request(**values)
        params.pop("response_format", None)
        params.pop("stream", None)
        return self._get_client().chat.completions.parse(response_format=text_format, **params)

    async def async_parse(
        self,
        text_format: Any,
        request: CompletionRequest | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步调用 OpenAI Chat/Responses 的原生结构化解析入口。"""
        if text_format is None:
            raise ValueError("OpenAI parse 必须提供 text_format。")
        request = coerce_completion_request(request, kwargs, "OpenAI-compatible async_parse")
        request = request.copy_with(stream=False)
        values = request.to_invoke_kwargs()
        if self._protocol == "responses":
            self._require_responses_resource("parse")
            params = self._build_responses_request(**values)
            if params.get("background"):
                raise ValueError(
                    "Responses async_parse 不支持 background=true；"
                    "请使用 acomplete 等待后台任务完成后再处理结果。"
                )
            params.pop("stream", None)
            text_config = params.pop("text", None)
            if isinstance(text_config, Mapping) and "format" in text_config:
                raise ValueError(
                    "Responses parse 不能同时使用 request.response_format 和 text_format。"
                )
            if text_config:
                params["text"] = text_config
            return await self._resolve_async_result(
                self._get_aclient().responses.parse(text_format=text_format, **params)
            )
        params = self._build_chat_request(**values)
        params.pop("response_format", None)
        params.pop("stream", None)
        return await self._resolve_async_result(
            self._get_aclient().chat.completions.parse(response_format=text_format, **params)
        )

    @classmethod
    def _error_message(cls, error: Any, default: str) -> str:
        if isinstance(error, str) and error:
            return error
        message = cls._field(error, "message")
        if message:
            return str(message)
        code = cls._field(error, "code")
        if code:
            return f"{code}"
        return default

    @classmethod
    def _raise_for_response_error(cls, response: Any) -> None:
        error = cls._field(response, "error")
        if error:
            raise RuntimeError(f"Responses API 返回错误: {cls._error_message(error, '未知错误。')}")

        status = cls._field(response, "status")
        if status not in {"failed", "incomplete", "cancelled"}:
            return

        if status == "incomplete":
            details = cls._field(response, "incomplete_details")
            reason = cls._field(details, "reason")
            detail = f"原因: {reason}" if reason else "未提供原因。"
        else:
            detail = "未提供原因。"
        raise RuntimeError(f"Responses API 响应状态为 {status}: {detail}")

    @classmethod
    def _stream_delta(cls, event: Any) -> str:
        event_type = cls._field(event, "type", "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = cls._field(event, "delta")
            return delta if isinstance(delta, str) else ""
        if event_type in {"error", "response.error"}:
            raise_for_stream_error_event(event, "Responses API")
        if event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.cancelled",
        }:
            response = cls._field(event, "response") or event
            cls._raise_for_response_error(response)
            if event_type != "response.completed":
                raise RuntimeError(f"Responses API 收到终止事件: {event_type}。")
        return ""

    def _invoke_responses(self, request: dict[str, Any]) -> Iterator[str]:
        if request["stream"]:
            terminal_seen = False
            for event in retry_sync_stream(
                lambda: self._get_client().responses.create(**request),
                lambda event: raise_for_stream_error_event(event, self._provider),
            ):
                event_type = self._field(event, "type", "")
                if event_type == "response.refusal.delta":
                    refusal = self._field(event, "delta")
                    if refusal:
                        raise RuntimeError(f"{self._provider} Responses 返回拒答: {refusal}")
                delta = self._stream_delta(event)
                if delta:
                    yield delta
                if event_type == "response.completed":
                    terminal_seen = True
            if not terminal_seen:
                raise RuntimeError("Responses API 流在 response.completed 之前结束。")
            return
        response = retry_sync_call(lambda: self._get_client().responses.create(**request))
        if request.get("background"):
            response = self._poll_background_response(response, request)
        self._raise_for_response_error(response)
        refusal = self._extract_refusal(response)
        if refusal:
            raise RuntimeError(f"{self._provider} Responses 返回拒答: {refusal}")
        text = self._extract_response_text(response)
        if text:
            yield text

    async def _ainvoke_responses(self, request: dict[str, Any]) -> AsyncIterator[str]:
        if request["stream"]:
            terminal_seen = False
            async for event in retry_async_stream(
                lambda: self._get_aclient().responses.create(**request),
                lambda event: raise_for_stream_error_event(event, self._provider),
            ):
                event_type = self._field(event, "type", "")
                if event_type == "response.refusal.delta":
                    refusal = self._field(event, "delta")
                    if refusal:
                        raise RuntimeError(f"{self._provider} Responses 返回拒答: {refusal}")
                delta = self._stream_delta(event)
                if delta:
                    yield delta
                if event_type == "response.completed":
                    terminal_seen = True
            if not terminal_seen:
                raise RuntimeError("Responses API 异步流在 response.completed 之前结束。")
            return
        response = await retry_async_call(lambda: self._get_aclient().responses.create(**request))
        if request.get("background"):
            response = await self._poll_background_response_async(response, request)
        self._raise_for_response_error(response)
        refusal = self._extract_refusal(response)
        if refusal:
            raise RuntimeError(f"{self._provider} Responses 返回拒答: {refusal}")
        text = self._extract_response_text(response)
        if text:
            yield text

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
        top_k: int | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """同步调用 OpenAI 兼容 LLM (CSE Sensor)。"""
        temperature = None if temperature is _UNSET_TEMPERATURE else temperature
        logger.info(f"调用 {self._provider} LLM ({self._model_name})，流式: {stream}")
        if self._protocol == "responses":
            self._require_responses_resource("invoke")
            if tools:
                raise ValueError(
                    f"{self._provider} Responses invoke 仅返回文本，不支持 tools；"
                    "请使用 complete 或 stream_events 获取工具调用。"
                )
        client = self._get_client()
        start_time = time.perf_counter()

        try:
            if self._protocol == "responses":
                yield from self._invoke_responses(
                    self._build_responses_request(
                        prompt,
                        system_prompt,
                        tools,
                        temperature,
                        stream,
                        messages=messages,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        stop=stop,
                        response_format=response_format,
                        tool_choice=tool_choice,
                        extra_body=extra_body,
                        metadata=metadata,
                        user=user,
                        timeout=timeout,
                        **kwargs,
                    )
                )
                return

            request = self._build_chat_request(
                prompt,
                system_prompt,
                tools,
                temperature,
                stream,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                top_k=top_k,
                seed=seed,
                response_format=response_format,
                tool_choice=tool_choice,
                extra_body=extra_body,
                metadata=metadata,
                user=user,
                timeout=timeout,
                **kwargs,
            )
            if stream:
                terminal_seen = False
                for chunk in retry_sync_stream(
                    lambda: client.chat.completions.create(**request),
                    lambda chunk: raise_for_stream_error_event(chunk, self._provider),
                ):
                    raise_for_stream_error_event(chunk, self._provider)
                    for event in self._chat_stream_events(chunk):
                        if event.type == "finish":
                            terminal_seen = True
                        if event.type == "refusal_delta" and event.refusal:
                            raise RuntimeError(
                                f"{self._provider} Chat Completions 返回拒答: {event.refusal}"
                            )
                        if event.type == "text_delta" and event.text:
                            yield event.text
                if not terminal_seen:
                    raise RuntimeError("Chat Completions 流在 finish 事件之前结束。")
            else:
                response = retry_sync_call(lambda: client.chat.completions.create(**request))
                self._raise_for_chat_response_error(
                    response,
                    provider=f"{getattr(self, '_provider', 'OpenAI-compatible')} Chat Completions",
                )
                refusal = self._extract_refusal(response)
                if refusal:
                    raise RuntimeError(f"{self._provider} Chat Completions 返回拒答: {refusal}")
                choices = self._field(response, "choices", []) or []
                if choices:
                    message = self._field(choices[0], "message")
                    content = self._field(message, "content")
                    if content:
                        yield content_to_text(content)

            duration = time.perf_counter() - start_time
            logger.info(
                f"{self._provider} LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s"
            )
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "%s LLM (%s) 出错: %s",
                self._provider,
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
        temperature: float | None = _UNSET_TEMPERATURE,  # type: ignore[assignment]
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """异步调用 OpenAI 兼容 LLM (CSE Sensor)。"""
        temperature = None if temperature is _UNSET_TEMPERATURE else temperature
        logger.info(f"异步调用 {self._provider} LLM ({self._model_name})，流式: {stream}")
        if self._protocol == "responses":
            self._require_responses_resource("ainvoke")
            if tools:
                raise ValueError(
                    f"{self._provider} Responses ainvoke 仅返回文本，不支持 tools；"
                    "请使用 acomplete 或 astream_events 获取工具调用。"
                )
        aclient = self._get_aclient()
        start_time = time.perf_counter()

        try:
            if self._protocol == "responses":
                async for chunk in self._ainvoke_responses(
                    self._build_responses_request(
                        prompt,
                        system_prompt,
                        tools,
                        temperature,
                        stream,
                        messages=messages,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        stop=stop,
                        response_format=response_format,
                        tool_choice=tool_choice,
                        extra_body=extra_body,
                        metadata=metadata,
                        user=user,
                        timeout=timeout,
                        **kwargs,
                    )
                ):
                    yield chunk
                return

            request = self._build_chat_request(
                prompt,
                system_prompt,
                tools,
                temperature,
                stream,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                top_k=top_k,
                seed=seed,
                response_format=response_format,
                tool_choice=tool_choice,
                extra_body=extra_body,
                metadata=metadata,
                user=user,
                timeout=timeout,
                **kwargs,
            )
            if stream:
                terminal_seen = False
                async for chunk in retry_async_stream(
                    lambda: aclient.chat.completions.create(**request),
                    lambda chunk: raise_for_stream_error_event(chunk, self._provider),
                ):
                    raise_for_stream_error_event(chunk, self._provider)
                    for event in self._chat_stream_events(chunk):
                        if event.type == "finish":
                            terminal_seen = True
                        if event.type == "refusal_delta" and event.refusal:
                            raise RuntimeError(
                                f"{self._provider} Chat Completions 返回拒答: {event.refusal}"
                            )
                        if event.type == "text_delta" and event.text:
                            yield event.text
                if not terminal_seen:
                    raise RuntimeError("Chat Completions 异步流在 finish 事件之前结束。")
            else:
                response = await retry_async_call(
                    lambda: aclient.chat.completions.create(**request)
                )
                self._raise_for_chat_response_error(
                    response,
                    provider=f"{getattr(self, '_provider', 'OpenAI-compatible')} Chat Completions",
                )
                refusal = self._extract_refusal(response)
                if refusal:
                    raise RuntimeError(f"{self._provider} Chat Completions 返回拒答: {refusal}")
                choices = self._field(response, "choices", []) or []
                if choices:
                    message = self._field(choices[0], "message")
                    content = self._field(message, "content")
                    if content:
                        yield content_to_text(content)

            duration = time.perf_counter() - start_time
            logger.info(
                f"{self._provider} LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s"
            )
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "%s LLM (%s) 异步出错: %s",
                self._provider,
                self._model_name,
                error_text,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """同步向量化文档。"""
        logger.info(f"调用 {self._provider} 嵌入 ({self._model_name})，数量: {len(texts)}")
        start_time = time.perf_counter()
        overlap = sorted({"input", "model"}.intersection(kwargs))
        if overlap:
            raise ValueError(f"{self._provider} Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
        self._validate_embedding_kwargs(kwargs)
        client = self._get_client()

        try:
            request: dict[str, Any] = {"input": texts, "model": self._model_name}
            configured_options = self._embedding_options()
            configured_extra_body = self._validated_extra_body(
                configured_options.pop("extra_body", None),
                f"{self._provider} Embedding",
                set(request).union(configured_options),
            )
            request_options = dict(kwargs)
            request_extra_body = self._validated_extra_body(
                request_options.pop("extra_body", None),
                f"{self._provider} Embedding",
                set(request).union(configured_options).union(request_options),
            )
            option_overlap = sorted(set(configured_options).intersection(request_options))
            if option_overlap:
                raise ValueError(
                    f"{self._provider} Embedding 请求参数与模型 options 重复: "
                    + ", ".join(option_overlap)
                )
            request.update(configured_options)
            request.update(request_options)
            merged_extra_body = self._merge_extra_body(
                configured_extra_body, request_extra_body, f"{self._provider} Embedding"
            )
            if merged_extra_body:
                request["extra_body"] = merged_extra_body
            response = client.embeddings.create(**request)
            embeddings = [normalize_embedding_vector(item.embedding) for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(f"{self._provider} 嵌入 ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "%s 嵌入 (%s) 出错: %s",
                self._provider,
                self._model_name,
                error_text,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """异步向量化文档。"""
        logger.info(f"异步调用 {self._provider} 嵌入 ({self._model_name})，数量: {len(texts)}")
        start_time = time.perf_counter()
        overlap = sorted({"input", "model"}.intersection(kwargs))
        if overlap:
            raise ValueError(f"{self._provider} Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
        self._validate_embedding_kwargs(kwargs)
        aclient = self._get_aclient()

        try:
            request: dict[str, Any] = {"input": texts, "model": self._model_name}
            configured_options = self._embedding_options()
            configured_extra_body = self._validated_extra_body(
                configured_options.pop("extra_body", None),
                f"{self._provider} Embedding",
                set(request).union(configured_options),
            )
            request_options = dict(kwargs)
            request_extra_body = self._validated_extra_body(
                request_options.pop("extra_body", None),
                f"{self._provider} Embedding",
                set(request).union(configured_options).union(request_options),
            )
            option_overlap = sorted(set(configured_options).intersection(request_options))
            if option_overlap:
                raise ValueError(
                    f"{self._provider} Embedding 请求参数与模型 options 重复: "
                    + ", ".join(option_overlap)
                )
            request.update(configured_options)
            request.update(request_options)
            merged_extra_body = self._merge_extra_body(
                configured_extra_body, request_extra_body, f"{self._provider} Embedding"
            )
            if merged_extra_body:
                request["extra_body"] = merged_extra_body
            response = await self._resolve_async_result(aclient.embeddings.create(**request))
            embeddings = [normalize_embedding_vector(item.embedding) for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(
                f"{self._provider} 嵌入 ({self._model_name}) 异步完成，耗时: {duration:.2f}s"
            )
            return embeddings
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "%s 嵌入 (%s) 异步出错: %s",
                self._provider,
                self._model_name,
                error_text,
            )
            raise

    def upload_file(self, file: Any, purpose: str = "assistants", **kwargs: Any) -> Any:
        method = self._get_client().files.create
        return self._call_sdk_resource(
            method,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "purpose": cast(Any, purpose)},
                kwargs,
                "OpenAI 文件上传",
            ),
            operation="OpenAI 文件上传",
        )

    def list_files(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().files.list, kwargs=kwargs, operation="OpenAI 文件列表"
        )

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().files.retrieve,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件读取",
        )

    def file_content(self, file_id: str, **kwargs: Any) -> Any:
        # openai>=3 exposes binary/text file content as `files.content`.
        files = getattr(self._get_client(), "files", None)
        method = getattr(files, "content", None)
        if not callable(method):
            raise NotImplementedError("当前 OpenAI SDK 不提供文件内容读取资源（files.content）。")
        return self._call_sdk_resource(
            method,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件内容读取",
        )

    def retrieve_file_content(self, file_id: str, **kwargs: Any) -> Any:
        """读取文件文本内容；兼容 SDK 的 retrieve_content 命名。"""
        method = getattr(self._get_client().files, "retrieve_content", None)
        if method is not None:
            return self._call_sdk_resource(
                method,
                args=(file_id,),
                kwargs=kwargs,
                operation="OpenAI 文件文本读取",
            )
        return self.file_content(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().files.delete,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件删除",
        )

    def wait_for_file(
        self,
        file_id: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """等待文件处理完成，仅转发 OpenAI SDK 的轮询参数。"""
        reject_unsupported_kwargs("OpenAI 文件等待", kwargs)
        poll_kwargs = {
            key: value
            for key, value in {
                "poll_interval": poll_interval,
                "max_wait_seconds": max_wait_seconds,
            }.items()
            if value is not None
        }
        return self._call_sdk_resource(
            self._get_client().files.wait_for_processing,
            kwargs=self._merge_resource_kwargs({"id": file_id}, poll_kwargs, "OpenAI 文件等待"),
            operation="OpenAI 文件等待",
        )

    def create_batch(self, input_file_id: str, endpoint: str | None = None, **kwargs: Any) -> Any:
        if self._protocol == "responses" and (
            endpoint is None or endpoint.rstrip("/") == "/v1/responses"
        ):
            self._require_responses_resource("create_batch")
        if endpoint is None:
            endpoint = "/v1/responses" if self._protocol == "responses" else "/v1/chat/completions"
        kwargs.setdefault("completion_window", "24h")
        return self._call_sdk_resource(
            self._get_client().batches.create,
            kwargs=self._merge_resource_kwargs(
                {
                    "input_file_id": input_file_id,
                    "endpoint": cast(Any, endpoint),
                },
                kwargs,
                "OpenAI Batch 创建",
            ),
            operation="OpenAI Batch 创建",
        )

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().batches.retrieve,
            args=(batch_id,),
            kwargs=kwargs,
            operation="OpenAI Batch 读取",
        )

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().batches.cancel,
            args=(batch_id,),
            kwargs=kwargs,
            operation="OpenAI Batch 取消",
        )

    def list_batches(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().batches.list, kwargs=kwargs, operation="OpenAI Batch 列表"
        )

    def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("retrieve_response")
        return self._call_sdk_resource(
            self._get_client().responses.retrieve,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 读取",
        )

    def cancel_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("cancel_response")
        return self._call_sdk_resource(
            self._get_client().responses.cancel,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 取消",
        )

    def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("delete_response")
        return self._call_sdk_resource(
            self._get_client().responses.delete,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 删除",
        )

    def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("list_response_input_items")
        return self._call_sdk_resource(
            self._get_client().responses.input_items.list,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 输入项列表",
        )

    def connect_responses(self, **kwargs: Any) -> Any:
        """打开官方 Responses WebSocket 连接管理器。"""
        self._require_responses_resource("connect_responses")
        return self._call_sdk_resource(
            self._get_client().responses.connect,
            kwargs=kwargs,
            operation="OpenAI Responses 连接",
        )

    def _prepare_responses_stream_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """将统一参数收敛到 ``responses.stream`` 的 SDK 契约。"""
        # The SDK uses response_id/starting_after to continue an existing
        # response and rejects model in that mode. A native caller may pass
        # either value explicitly, so remove the provider default as well as
        # an explicitly supplied model before dispatch.
        if "response_id" in params or "starting_after" in params:
            params.pop("model", None)
        else:
            params.setdefault("model", self._model_name)
        params.pop("stream", None)
        if "verbosity" in params:
            verbosity = params.pop("verbosity")
            if verbosity is not None:
                params.setdefault("text", {}).setdefault("verbosity", verbosity)
        return params

    def stream_responses(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        """返回官方 Responses 流上下文管理器，支持统一请求或原生参数。"""
        self._require_responses_resource("stream_responses")
        if request is not None and kwargs:
            raise ValueError("stream_responses 不能同时传 request 和原生关键字参数。")
        if request is not None:
            params = self._build_responses_request(
                **request.copy_with(stream=True).to_invoke_kwargs()
            )
            params.pop("stream", None)
        else:
            params = dict(kwargs)
        # `responses.stream` is a helper with a narrower contract than
        # `responses.create`: it does not accept the create-only `stream`
        # switch, and it uses `starting_after` for continuation. Keep the
        # provider's unified request shape while validating the helper boundary.
        params = self._prepare_responses_stream_params(params)
        return self._call_sdk_resource(
            self._get_client().responses.stream,
            kwargs=params,
            operation="OpenAI Responses 流",
        )

    def _build_token_count_request(self, request: CompletionRequest) -> dict[str, Any]:
        validate_secret_free_request_overrides(
            request.extra_headers,
            request.extra_query,
            f"{getattr(self, '_provider', 'openai')} Responses input token count",
        )
        unsupported = {
            key: value
            for key, value in request.to_invoke_kwargs().items()
            if key
            not in {
                "prompt",
                "system_prompt",
                "messages",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "reasoning",
                "previous_response_id",
                "personality",
                "response_format",
                "truncation",
                "conversation",
                "extra_headers",
                "extra_query",
                "extra_body",
                "timeout",
                # Shared generation defaults are not part of the token-count
                # request, but accepting and dropping them is necessary for
                # CompletionRequest compatibility. They are never forwarded.
                "stream",
                "temperature",
            }
        }
        reject_unsupported_kwargs("OpenAI Responses input token count", unsupported)
        params: dict[str, Any] = {
            "model": self._model_name,
            "input": normalize_responses_input(
                request.prompt,
                request.system_prompt,
                request.messages,
                provider=getattr(self, "_provider", "openai"),
            ),
        }
        if request.system_prompt and request.messages is None:
            params["instructions"] = request.system_prompt
        if request.conversation is not None:
            params["conversation"] = request.conversation
        if request.tools:
            params["tools"] = self._convert_responses_tools(request.tools)
        if request.tool_choice is not None:
            params["tool_choice"] = self._convert_responses_tool_choice(request.tool_choice)
        if request.parallel_tool_calls is not None:
            params["parallel_tool_calls"] = request.parallel_tool_calls
        if request.reasoning is not None:
            params["reasoning"] = request.reasoning
        if request.previous_response_id is not None:
            params["previous_response_id"] = request.previous_response_id
        if request.personality is not None:
            params["personality"] = request.personality
        if request.response_format is not None:
            params["text"] = self._convert_responses_format(request.response_format)
        if request.truncation is not None:
            params["truncation"] = request.truncation
        if request.extra_headers is not None:
            params["extra_headers"] = request.extra_headers
        if request.extra_query is not None:
            params["extra_query"] = request.extra_query
        if request.extra_body is not None:
            params["extra_body"] = request.extra_body
        if request.timeout is not None:
            params["timeout"] = request.timeout
        return params

    def count_input_tokens(self, request: CompletionRequest | None = None, **kwargs: Any) -> int:
        request = coerce_completion_request(request, kwargs, "OpenAI-compatible count_input_tokens")
        self._require_responses_resource("count_input_tokens")
        params = self._build_token_count_request(request)
        response = self._call_sdk_resource(
            self._get_client().responses.input_tokens.count,
            kwargs=params,
            operation="OpenAI Responses 输入 token 统计",
        )
        value = self._field(response, "input_tokens")
        if value is None:
            raise RuntimeError("OpenAI token count 响应缺少 input_tokens。")
        return int(value)

    def create_vector_store(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.create,
            kwargs=kwargs,
            operation="OpenAI Vector Store 创建",
        )

    def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.retrieve,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 读取",
        )

    def list_vector_stores(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.list,
            kwargs=kwargs,
            operation="OpenAI Vector Store 列表",
        )

    def search_vector_store(
        self, vector_store_id: str, query: str | list[str], **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.search,
            args=(vector_store_id,),
            kwargs=self._merge_resource_kwargs(
                {"query": query}, kwargs, "OpenAI Vector Store 搜索"
            ),
            operation="OpenAI Vector Store 搜索",
        )

    def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.delete,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 删除",
        )

    def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.update,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 更新",
        )

    def create_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.create,
            args=(vector_store_id,),
            kwargs=self._merge_resource_kwargs(
                {"file_id": file_id}, kwargs, "OpenAI Vector Store 文件创建"
            ),
            operation="OpenAI Vector Store 文件创建",
        )

    def create_vector_store_file_and_poll(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.create_and_poll,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件创建轮询",
            ),
            operation="OpenAI Vector Store 文件创建轮询",
        )

    def upload_vector_store_file(
        self,
        vector_store_id: str,
        file: Any,
        *,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """上传并挂载向量库文件，仅转发 SDK 支持的分块策略。"""
        reject_unsupported_kwargs("OpenAI 向量库文件上传", kwargs)
        upload_kwargs = (
            {"chunking_strategy": chunking_strategy} if chunking_strategy is not None else {}
        )
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.upload,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "file": file},
                upload_kwargs,
                "OpenAI Vector Store 文件上传",
            ),
            operation="OpenAI Vector Store 文件上传",
        )

    def upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        file: Any,
        *,
        attributes: Mapping[str, str | float | bool] | None = None,
        poll_interval_ms: int | None = None,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """上传单个文件到向量库并等待处理完成。"""
        reject_unsupported_kwargs("OpenAI 向量库文件上传轮询", kwargs)
        upload_kwargs = {
            key: value
            for key, value in {
                "attributes": dict(attributes) if attributes is not None else None,
                "poll_interval_ms": poll_interval_ms,
                "chunking_strategy": chunking_strategy,
            }.items()
            if value is not None
        }
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.upload_and_poll,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "file": file},
                upload_kwargs,
                "OpenAI Vector Store 文件上传轮询",
            ),
            operation="OpenAI Vector Store 文件上传轮询",
        )

    def retrieve_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.retrieve,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件读取",
            ),
            operation="OpenAI Vector Store 文件读取",
        )

    def list_vector_store_files(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.list,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件列表",
        )

    def update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, str | float | bool] | None,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        """更新向量库文件属性，严格遵循 SDK 的必填 ``attributes`` 契约。"""
        reject_unsupported_kwargs("OpenAI 向量库文件更新", kwargs)
        request = {
            "vector_store_id": vector_store_id,
            "attributes": dict(attributes) if attributes is not None else None,
        }
        request.update(
            {
                key: value
                for key, value in {
                    "extra_headers": extra_headers,
                    "extra_query": extra_query,
                    "extra_body": extra_body,
                    "timeout": timeout,
                }.items()
                if value is not None
            }
        )
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.update,
            args=(file_id,),
            kwargs=request,
            operation="OpenAI Vector Store 文件更新",
        )

    def delete_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.delete,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件删除",
            ),
            operation="OpenAI Vector Store 文件删除",
        )

    def vector_store_file_content(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.content,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件内容读取",
            ),
            operation="OpenAI Vector Store 文件内容读取",
        )

    def poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """轮询向量库文件，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 向量库文件轮询", kwargs)
        return self._call_sdk_resource(
            self._get_client().vector_stores.files.poll,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                {"poll_interval_ms": poll_interval_ms} if poll_interval_ms is not None else {},
                "OpenAI Vector Store 文件轮询",
            ),
            operation="OpenAI Vector Store 文件轮询",
        )

    def create_vector_store_file_batch(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.create,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件批次创建",
        )

    def create_vector_store_file_batch_and_poll(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.create_and_poll,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件批次创建轮询",
        )

    def retrieve_vector_store_file_batch(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.retrieve,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次读取",
            ),
            operation="OpenAI Vector Store 文件批次读取",
        )

    def cancel_vector_store_file_batch(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.cancel,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次取消",
            ),
            operation="OpenAI Vector Store 文件批次取消",
        )

    def poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """轮询向量库文件批次，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 向量库文件批次轮询", kwargs)
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.poll,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                {"poll_interval_ms": poll_interval_ms} if poll_interval_ms is not None else {},
                "OpenAI Vector Store 文件批次轮询",
            ),
            operation="OpenAI Vector Store 文件批次轮询",
        )

    def list_vector_store_file_batch_files(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().vector_stores.file_batches.list_files,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次文件列表",
            ),
            operation="OpenAI Vector Store 文件批次文件列表",
        )

    def upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        files: Any,
        *,
        max_concurrency: int | None = None,
        file_ids: Any | None = None,
        poll_interval_ms: int | None = None,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """上传并轮询向量库文件批次，严格匹配 SDK 辅助方法参数。"""
        reject_unsupported_kwargs("OpenAI 向量库文件批次上传轮询", kwargs)
        poll_kwargs = {
            key: value
            for key, value in {
                "max_concurrency": max_concurrency,
                "file_ids": file_ids,
                "poll_interval_ms": poll_interval_ms,
                "chunking_strategy": chunking_strategy,
            }.items()
            if value is not None
        }
        upload_and_poll = cast(Any, self._get_client().vector_stores.file_batches.upload_and_poll)
        return self._call_sdk_resource(
            upload_and_poll,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "files": files},
                poll_kwargs,
                "OpenAI Vector Store 文件批次上传轮询",
            ),
            operation="OpenAI Vector Store 文件批次上传轮询",
        )

    def list_models(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().models.list, kwargs=kwargs, operation="OpenAI 模型列表"
        )

    def retrieve_model(self, model: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().models.retrieve,
            args=(model,),
            kwargs=kwargs,
            operation="OpenAI 模型读取",
        )

    def delete_model(self, model: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().models.delete,
            args=(model,),
            kwargs=kwargs,
            operation="OpenAI 模型删除",
        )

    def create_moderation(self, input: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().moderations.create,
            kwargs=self._merge_resource_kwargs({"input": input}, kwargs, "OpenAI Moderation 创建"),
            operation="OpenAI Moderation 创建",
        )

    def generate_image(self, prompt: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().images.generate,
            kwargs=self._merge_resource_kwargs({"prompt": prompt}, kwargs, "OpenAI 图片生成"),
            operation="OpenAI 图片生成",
        )

    def edit_image(self, image: Any, prompt: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().images.edit,
            kwargs=self._merge_resource_kwargs(
                {"image": image, "prompt": prompt}, kwargs, "OpenAI 图片编辑"
            ),
            operation="OpenAI 图片编辑",
        )

    def create_image_variation(self, image: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().images.create_variation,
            kwargs=self._merge_resource_kwargs({"image": image}, kwargs, "OpenAI 图片变体"),
            operation="OpenAI 图片变体",
        )

    def text_to_speech(self, text: str, model: str, voice: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().audio.speech.create,
            kwargs=self._merge_resource_kwargs(
                {"input": text, "model": model, "voice": voice},
                kwargs,
                "OpenAI 语音合成",
            ),
            operation="OpenAI 语音合成",
        )

    def transcribe_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().audio.transcriptions.create,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "model": model}, kwargs, "OpenAI 音频转写"
            ),
            operation="OpenAI 音频转写",
        )

    def translate_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().audio.translations.create,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "model": model}, kwargs, "OpenAI 音频翻译"
            ),
            operation="OpenAI 音频翻译",
        )

    def create_video(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.create, kwargs=kwargs, operation="OpenAI 视频创建"
        )

    def create_video_and_poll(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.create_and_poll,
            kwargs=kwargs,
            operation="OpenAI 视频创建轮询",
        )

    def retrieve_video(self, video_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.retrieve,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频读取",
        )

    def list_videos(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.list, kwargs=kwargs, operation="OpenAI 视频列表"
        )

    def delete_video(self, video_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.delete,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频删除",
        )

    def download_video(self, video_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.download_content,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频下载",
        )

    def create_video_character(self, name: str, video: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.create_character,
            kwargs=self._merge_resource_kwargs(
                {"name": name, "video": video}, kwargs, "OpenAI 视频角色创建"
            ),
            operation="OpenAI 视频角色创建",
        )

    def retrieve_video_character(self, character_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.get_character,
            args=(character_id,),
            kwargs=kwargs,
            operation="OpenAI 视频角色读取",
        )

    def edit_video(self, prompt: str, video: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.edit,
            kwargs=self._merge_resource_kwargs(
                {"prompt": prompt, "video": video}, kwargs, "OpenAI 视频编辑"
            ),
            operation="OpenAI 视频编辑",
        )

    def extend_video(self, prompt: str, seconds: Any, video: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.extend,
            kwargs=self._merge_resource_kwargs(
                {"prompt": prompt, "seconds": seconds, "video": video},
                kwargs,
                "OpenAI 视频续写",
            ),
            operation="OpenAI 视频续写",
        )

    def remix_video(self, video_id: str, prompt: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().videos.remix,
            args=(video_id,),
            kwargs=self._merge_resource_kwargs({"prompt": prompt}, kwargs, "OpenAI 视频混剪"),
            operation="OpenAI 视频混剪",
        )

    def poll_video(
        self,
        video_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """轮询视频处理状态，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 视频轮询", kwargs)
        return self._call_sdk_resource(
            self._get_client().videos.poll,
            args=(video_id,),
            kwargs={
                "poll_interval_ms": poll_interval_ms,
            }
            if poll_interval_ms is not None
            else {},
            operation="OpenAI 视频轮询",
        )

    def create_upload(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().uploads.create,
            kwargs=kwargs,
            operation="OpenAI Upload 创建",
        )

    def complete_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().uploads.complete,
            args=(upload_id,),
            kwargs=kwargs,
            operation="OpenAI Upload 完成",
        )

    def cancel_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().uploads.cancel,
            args=(upload_id,),
            kwargs=kwargs,
            operation="OpenAI Upload 取消",
        )

    def create_upload_part(self, upload_id: str, data: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().uploads.parts.create,
            args=(upload_id,),
            kwargs=self._merge_resource_kwargs({"data": data}, kwargs, "OpenAI Upload 分片创建"),
            operation="OpenAI Upload 分片创建",
        )

    def upload_file_chunked(
        self,
        *,
        file: Any,
        mime_type: str,
        purpose: str,
        filename: str | None = None,
        bytes: int | None = None,
        part_size: int | None = None,
        md5: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """按 SDK 的窄参数契约执行分片上传。"""
        reject_unsupported_kwargs("OpenAI 分片文件上传", kwargs)
        request = {
            key: value
            for key, value in {
                "file": file,
                "mime_type": mime_type,
                "purpose": purpose,
                "filename": filename,
                "bytes": bytes,
                "part_size": part_size,
                "md5": md5,
            }.items()
            if value is not None
        }
        return self._call_sdk_resource(
            self._get_client().uploads.upload_file_chunked,
            kwargs=request,
            operation="OpenAI 分片文件上传",
        )

    def compact_responses(
        self,
        *,
        model: str | None = None,
        input: Any = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: Any = None,
        prompt_cache_retention: str | None = None,
        service_tier: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        """压缩 Responses 会话上下文，严格匹配 OpenAI SDK 参数。"""
        self._require_responses_resource("compact_responses")
        reject_unsupported_kwargs("OpenAI Responses compact", kwargs)
        compact_kwargs = {
            key: value
            for key, value in {
                "model": model or self._model_name,
                "input": input,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_options": prompt_cache_options,
                "prompt_cache_retention": prompt_cache_retention,
                "service_tier": service_tier,
                "extra_headers": extra_headers,
                "extra_query": extra_query,
                "extra_body": extra_body,
                "timeout": timeout,
            }.items()
            if value is not None
        }
        return self._call_sdk_resource(
            self._get_client().responses.compact,
            kwargs=compact_kwargs,
            operation="OpenAI Responses compact",
        )

    def create_conversation(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.create,
            kwargs=kwargs,
            operation="OpenAI Conversation 创建",
        )

    def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.retrieve,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 读取",
        )

    def update_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.update,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 更新",
        )

    def delete_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.delete,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 删除",
        )

    def list_conversation_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.items.list,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 项目列表",
        )

    def create_conversation_items(self, conversation_id: str, items: Any, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.items.create,
            args=(conversation_id,),
            kwargs=self._merge_resource_kwargs(
                {"items": items}, kwargs, "OpenAI Conversation 项目创建"
            ),
            operation="OpenAI Conversation 项目创建",
        )

    def retrieve_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.items.retrieve,
            args=(item_id,),
            kwargs=self._merge_resource_kwargs(
                {"conversation_id": conversation_id},
                kwargs,
                "OpenAI Conversation 项目读取",
            ),
            operation="OpenAI Conversation 项目读取",
        )

    def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().conversations.items.delete,
            args=(item_id,),
            kwargs=self._merge_resource_kwargs(
                {"conversation_id": conversation_id},
                kwargs,
                "OpenAI Conversation 项目删除",
            ),
            operation="OpenAI Conversation 项目删除",
        )

    def create_container(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.create,
            kwargs=kwargs,
            operation="OpenAI Container 创建",
        )

    def retrieve_container(self, container_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.retrieve,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 读取",
        )

    def list_containers(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.list,
            kwargs=kwargs,
            operation="OpenAI Container 列表",
        )

    def delete_container(self, container_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.delete,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 删除",
        )

    def create_container_file(
        self,
        container_id: str,
        file: Any = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        file_kwargs = {
            key: value
            for key, value in {"file": file, "file_id": file_id}.items()
            if value is not None
        }
        return self._call_sdk_resource(
            self._get_client().containers.files.create,
            args=(container_id,),
            kwargs=self._merge_resource_kwargs(
                file_kwargs,
                kwargs,
                "OpenAI Container 文件创建",
            ),
            operation="OpenAI Container 文件创建",
        )

    def list_container_files(self, container_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.files.list,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 文件列表",
        )

    def retrieve_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.files.retrieve,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件读取",
            ),
            operation="OpenAI Container 文件读取",
        )

    def delete_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().containers.files.delete,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件删除",
            ),
            operation="OpenAI Container 文件删除",
        )

    def container_file_content(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        files = getattr(getattr(self._get_client(), "containers", None), "files", None)
        content = getattr(files, "content", None)
        method = getattr(content, "retrieve", None)
        if not callable(method):
            raise NotImplementedError(
                "当前 OpenAI SDK 不提供容器文件内容读取资源 （containers.files.content.retrieve）。"
            )
        return self._call_sdk_resource(
            method,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件内容读取",
            ),
            operation="OpenAI Container 文件内容读取",
        )

    def create_fine_tuning_job(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.create,
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 创建",
        )

    def list_fine_tuning_jobs(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.list,
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 列表",
        )

    def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.retrieve,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 读取",
        )

    def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.cancel,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 取消",
        )

    def pause_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.pause,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 暂停",
        )

    def resume_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.resume,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 恢复",
        )

    def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().fine_tuning.jobs.list_events,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 事件列表",
        )

    def create_eval(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.create,
            kwargs=kwargs,
            operation="OpenAI Eval 创建",
        )

    def list_evals(self, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.list,
            kwargs=kwargs,
            operation="OpenAI Eval 列表",
        )

    def retrieve_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.retrieve,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 读取",
        )

    def update_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.update,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 更新",
        )

    def delete_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.delete,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 删除",
        )

    def create_eval_run(self, eval_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.create,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval Run 创建",
        )

    def list_eval_runs(self, eval_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.list,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval Run 列表",
        )

    def retrieve_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.retrieve,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 读取"
            ),
            operation="OpenAI Eval Run 读取",
        )

    def cancel_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.cancel,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 取消"
            ),
            operation="OpenAI Eval Run 取消",
        )

    def delete_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.delete,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 删除"
            ),
            operation="OpenAI Eval Run 删除",
        )

    def list_eval_run_output_items(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.output_items.list,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id},
                kwargs,
                "OpenAI Eval Run 输出项列表",
            ),
            operation="OpenAI Eval Run 输出项列表",
        )

    def retrieve_eval_run_output_item(
        self, eval_id: str, run_id: str, output_item_id: str, **kwargs: Any
    ) -> Any:
        return self._call_sdk_resource(
            self._get_client().evals.runs.output_items.retrieve,
            args=(output_item_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id, "run_id": run_id},
                kwargs,
                "OpenAI Eval Run 输出项读取",
            ),
            operation="OpenAI Eval Run 输出项读取",
        )

    async def async_upload_file(self, file: Any, purpose: str = "assistants", **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().files.create,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "purpose": cast(Any, purpose)},
                kwargs,
                "OpenAI 文件上传",
            ),
            operation="OpenAI 文件上传",
        )

    async def async_list_files(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().files.list, kwargs=kwargs, operation="OpenAI 文件列表"
        )

    async def async_retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().files.retrieve,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件读取",
        )

    async def async_file_content(self, file_id: str, **kwargs: Any) -> Any:
        files = getattr(self._get_aclient(), "files", None)
        method = getattr(files, "content", None)
        if not callable(method):
            raise NotImplementedError("当前 OpenAI SDK 不提供文件内容读取资源（files.content）。")
        return await self._call_async_sdk_resource(
            method,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件内容读取",
        )

    async def async_retrieve_file_content(self, file_id: str, **kwargs: Any) -> Any:
        files = self._get_aclient().files
        method = getattr(files, "retrieve_content", None)
        if method is not None:
            return await self._call_async_sdk_resource(
                method,
                args=(file_id,),
                kwargs=kwargs,
                operation="OpenAI 文件文本读取",
            )
        return await self.async_file_content(file_id, **kwargs)

    async def async_delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().files.delete,
            args=(file_id,),
            kwargs=kwargs,
            operation="OpenAI 文件删除",
        )

    async def async_wait_for_file(
        self,
        file_id: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步等待文件处理完成，仅转发 AsyncOpenAI SDK 的轮询参数。"""
        reject_unsupported_kwargs("OpenAI 文件等待", kwargs)
        poll_kwargs = {
            key: value
            for key, value in {
                "poll_interval": poll_interval,
                "max_wait_seconds": max_wait_seconds,
            }.items()
            if value is not None
        }
        return await self._call_async_sdk_resource(
            self._get_aclient().files.wait_for_processing,
            kwargs=self._merge_resource_kwargs({"id": file_id}, poll_kwargs, "OpenAI 文件等待"),
            operation="OpenAI 文件等待",
        )

    async def async_create_batch(
        self, input_file_id: str, endpoint: str | None = None, **kwargs: Any
    ) -> Any:
        if self._protocol == "responses" and (
            endpoint is None or endpoint.rstrip("/") == "/v1/responses"
        ):
            self._require_responses_resource("create_batch")
        if endpoint is None:
            endpoint = "/v1/responses" if self._protocol == "responses" else "/v1/chat/completions"
        kwargs.setdefault("completion_window", "24h")
        return await self._call_async_sdk_resource(
            self._get_aclient().batches.create,
            kwargs=self._merge_resource_kwargs(
                {
                    "input_file_id": input_file_id,
                    "endpoint": cast(Any, endpoint),
                },
                kwargs,
                "OpenAI Batch 创建",
            ),
            operation="OpenAI Batch 创建",
        )

    async def async_retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().batches.retrieve,
            args=(batch_id,),
            kwargs=kwargs,
            operation="OpenAI Batch 读取",
        )

    async def async_cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().batches.cancel,
            args=(batch_id,),
            kwargs=kwargs,
            operation="OpenAI Batch 取消",
        )

    async def async_list_batches(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().batches.list, kwargs=kwargs, operation="OpenAI Batch 列表"
        )

    async def async_retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("retrieve_response")
        return await self._call_async_sdk_resource(
            self._get_aclient().responses.retrieve,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 读取",
        )

    async def async_cancel_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("cancel_response")
        return await self._call_async_sdk_resource(
            self._get_aclient().responses.cancel,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 取消",
        )

    async def async_delete_response(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("delete_response")
        return await self._call_async_sdk_resource(
            self._get_aclient().responses.delete,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 删除",
        )

    async def async_list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        self._require_responses_resource("list_response_input_items")
        return await self._call_async_sdk_resource(
            self._get_aclient().responses.input_items.list,
            args=(response_id,),
            kwargs=kwargs,
            operation="OpenAI Responses 输入项列表",
        )

    def async_connect_responses(self, **kwargs: Any) -> Any:
        """返回 AsyncOpenAI Responses WebSocket 连接管理器。"""
        self._require_responses_resource("connect_responses")
        return self._call_sdk_resource(
            self._get_aclient().responses.connect,
            kwargs=kwargs,
            operation="OpenAI Responses 连接",
        )

    def async_stream_responses(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        """返回 AsyncOpenAI Responses 流上下文管理器。"""
        self._require_responses_resource("stream_responses")
        if request is not None and kwargs:
            raise ValueError("async_stream_responses 不能同时传 request 和原生关键字参数。")
        if request is not None:
            params = self._build_responses_request(
                **request.copy_with(stream=True).to_invoke_kwargs()
            )
            params.pop("stream", None)
        else:
            params = dict(kwargs)
        params = self._prepare_responses_stream_params(params)
        return self._call_sdk_resource(
            self._get_aclient().responses.stream,
            kwargs=params,
            operation="OpenAI Responses 流",
        )

    async def async_count_input_tokens(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> int:
        request = coerce_completion_request(
            request, kwargs, "OpenAI-compatible async_count_input_tokens"
        )
        self._require_responses_resource("count_input_tokens")
        params = self._build_token_count_request(request)
        response = await self._call_async_sdk_resource(
            self._get_aclient().responses.input_tokens.count,
            kwargs=params,
            operation="OpenAI Responses 输入 token 统计",
        )
        value = self._field(response, "input_tokens")
        if value is None:
            raise RuntimeError("OpenAI token count 响应缺少 input_tokens。")
        return int(value)

    async def async_create_vector_store(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.create,
            kwargs=kwargs,
            operation="OpenAI Vector Store 创建",
        )

    async def async_retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.retrieve,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 读取",
        )

    async def async_list_vector_stores(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.list,
            kwargs=kwargs,
            operation="OpenAI Vector Store 列表",
        )

    async def async_search_vector_store(
        self, vector_store_id: str, query: Any, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.search,
            args=(vector_store_id,),
            kwargs=self._merge_resource_kwargs(
                {"query": query}, kwargs, "OpenAI Vector Store 搜索"
            ),
            operation="OpenAI Vector Store 搜索",
        )

    async def async_delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.delete,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 删除",
        )

    async def async_update_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.update,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 更新",
        )

    async def async_create_vector_store_file(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.create,
            args=(vector_store_id,),
            kwargs=self._merge_resource_kwargs(
                {"file_id": file_id}, kwargs, "OpenAI Vector Store 文件创建"
            ),
            operation="OpenAI Vector Store 文件创建",
        )

    async def async_create_vector_store_file_and_poll(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.create_and_poll,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件创建轮询",
            ),
            operation="OpenAI Vector Store 文件创建轮询",
        )

    async def async_upload_vector_store_file(
        self,
        vector_store_id: str,
        file: Any,
        *,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步上传并挂载向量库文件，仅转发 SDK 支持的分块策略。"""
        reject_unsupported_kwargs("OpenAI 向量库文件上传", kwargs)
        upload_kwargs = (
            {"chunking_strategy": chunking_strategy} if chunking_strategy is not None else {}
        )
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.upload,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "file": file},
                upload_kwargs,
                "OpenAI Vector Store 文件上传",
            ),
            operation="OpenAI Vector Store 文件上传",
        )

    async def async_upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        file: Any,
        *,
        attributes: Mapping[str, str | float | bool] | None = None,
        poll_interval_ms: int | None = None,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步上传单个文件到向量库并等待处理完成。"""
        reject_unsupported_kwargs("OpenAI 向量库文件上传轮询", kwargs)
        upload_kwargs = {
            key: value
            for key, value in {
                "attributes": dict(attributes) if attributes is not None else None,
                "poll_interval_ms": poll_interval_ms,
                "chunking_strategy": chunking_strategy,
            }.items()
            if value is not None
        }
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.upload_and_poll,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "file": file},
                upload_kwargs,
                "OpenAI Vector Store 文件上传轮询",
            ),
            operation="OpenAI Vector Store 文件上传轮询",
        )

    async def async_retrieve_vector_store_file(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.retrieve,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件读取",
            ),
            operation="OpenAI Vector Store 文件读取",
        )

    async def async_list_vector_store_files(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.list,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件列表",
        )

    async def async_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Mapping[str, str | float | bool] | None,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        """异步更新向量库文件属性，严格遵循 SDK 的必填 ``attributes`` 契约。"""
        reject_unsupported_kwargs("OpenAI 向量库文件更新", kwargs)
        request = {
            "vector_store_id": vector_store_id,
            "attributes": dict(attributes) if attributes is not None else None,
        }
        request.update(
            {
                key: value
                for key, value in {
                    "extra_headers": extra_headers,
                    "extra_query": extra_query,
                    "extra_body": extra_body,
                    "timeout": timeout,
                }.items()
                if value is not None
            }
        )
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.update,
            args=(file_id,),
            kwargs=request,
            operation="OpenAI Vector Store 文件更新",
        )

    async def async_delete_vector_store_file(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.delete,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件删除",
            ),
            operation="OpenAI Vector Store 文件删除",
        )

    async def async_vector_store_file_content(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.content,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件内容读取",
            ),
            operation="OpenAI Vector Store 文件内容读取",
        )

    async def async_poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步轮询向量库文件，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 向量库文件轮询", kwargs)
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.files.poll,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                {"poll_interval_ms": poll_interval_ms} if poll_interval_ms is not None else {},
                "OpenAI Vector Store 文件轮询",
            ),
            operation="OpenAI Vector Store 文件轮询",
        )

    async def async_create_vector_store_file_batch(
        self, vector_store_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.create,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件批次创建",
        )

    async def async_create_vector_store_file_batch_and_poll(
        self, vector_store_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.create_and_poll,
            args=(vector_store_id,),
            kwargs=kwargs,
            operation="OpenAI Vector Store 文件批次创建轮询",
        )

    async def async_retrieve_vector_store_file_batch(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.retrieve,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次读取",
            ),
            operation="OpenAI Vector Store 文件批次读取",
        )

    async def async_cancel_vector_store_file_batch(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.cancel,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次取消",
            ),
            operation="OpenAI Vector Store 文件批次取消",
        )

    async def async_poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步轮询向量库文件批次，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 向量库文件批次轮询", kwargs)
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.poll,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                {"poll_interval_ms": poll_interval_ms} if poll_interval_ms is not None else {},
                "OpenAI Vector Store 文件批次轮询",
            ),
            operation="OpenAI Vector Store 文件批次轮询",
        )

    async def async_list_vector_store_file_batch_files(
        self, vector_store_id: str, batch_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().vector_stores.file_batches.list_files,
            args=(batch_id,),
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id},
                kwargs,
                "OpenAI Vector Store 文件批次文件列表",
            ),
            operation="OpenAI Vector Store 文件批次文件列表",
        )

    async def async_upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        files: Any,
        *,
        max_concurrency: int | None = None,
        file_ids: Any | None = None,
        poll_interval_ms: int | None = None,
        chunking_strategy: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步上传并轮询向量库文件批次，严格匹配 SDK 参数。"""
        reject_unsupported_kwargs("OpenAI 向量库文件批次上传轮询", kwargs)
        poll_kwargs = {
            key: value
            for key, value in {
                "max_concurrency": max_concurrency,
                "file_ids": file_ids,
                "poll_interval_ms": poll_interval_ms,
                "chunking_strategy": chunking_strategy,
            }.items()
            if value is not None
        }
        upload_and_poll = cast(Any, self._get_aclient().vector_stores.file_batches.upload_and_poll)
        return await self._call_async_sdk_resource(
            upload_and_poll,
            kwargs=self._merge_resource_kwargs(
                {"vector_store_id": vector_store_id, "files": files},
                poll_kwargs,
                "OpenAI Vector Store 文件批次上传轮询",
            ),
            operation="OpenAI Vector Store 文件批次上传轮询",
        )

    async def async_list_models(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().models.list, kwargs=kwargs, operation="OpenAI 模型列表"
        )

    async def async_retrieve_model(self, model: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().models.retrieve,
            args=(model,),
            kwargs=kwargs,
            operation="OpenAI 模型读取",
        )

    async def async_delete_model(self, model: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().models.delete,
            args=(model,),
            kwargs=kwargs,
            operation="OpenAI 模型删除",
        )

    async def async_create_moderation(self, input: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().moderations.create,
            kwargs=self._merge_resource_kwargs({"input": input}, kwargs, "OpenAI Moderation 创建"),
            operation="OpenAI Moderation 创建",
        )

    async def async_generate_image(self, prompt: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().images.generate,
            kwargs=self._merge_resource_kwargs({"prompt": prompt}, kwargs, "OpenAI 图片生成"),
            operation="OpenAI 图片生成",
        )

    async def async_edit_image(self, image: Any, prompt: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().images.edit,
            kwargs=self._merge_resource_kwargs(
                {"image": image, "prompt": prompt}, kwargs, "OpenAI 图片编辑"
            ),
            operation="OpenAI 图片编辑",
        )

    async def async_create_image_variation(self, image: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().images.create_variation,
            kwargs=self._merge_resource_kwargs({"image": image}, kwargs, "OpenAI 图片变体"),
            operation="OpenAI 图片变体",
        )

    async def async_text_to_speech(self, text: str, model: str, voice: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().audio.speech.create,
            kwargs=self._merge_resource_kwargs(
                {"input": text, "model": model, "voice": voice},
                kwargs,
                "OpenAI 语音合成",
            ),
            operation="OpenAI 语音合成",
        )

    async def async_transcribe_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().audio.transcriptions.create,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "model": model}, kwargs, "OpenAI 音频转写"
            ),
            operation="OpenAI 音频转写",
        )

    async def async_translate_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().audio.translations.create,
            kwargs=self._merge_resource_kwargs(
                {"file": file, "model": model}, kwargs, "OpenAI 音频翻译"
            ),
            operation="OpenAI 音频翻译",
        )

    async def async_create_video(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.create,
            kwargs=kwargs,
            operation="OpenAI 视频创建",
        )

    async def async_create_video_and_poll(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.create_and_poll,
            kwargs=kwargs,
            operation="OpenAI 视频创建轮询",
        )

    async def async_retrieve_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.retrieve,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频读取",
        )

    async def async_list_videos(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.list, kwargs=kwargs, operation="OpenAI 视频列表"
        )

    async def async_delete_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.delete,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频删除",
        )

    async def async_download_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.download_content,
            args=(video_id,),
            kwargs=kwargs,
            operation="OpenAI 视频下载",
        )

    async def async_create_video_character(self, name: str, video: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.create_character,
            kwargs=self._merge_resource_kwargs(
                {"name": name, "video": video}, kwargs, "OpenAI 视频角色创建"
            ),
            operation="OpenAI 视频角色创建",
        )

    async def async_retrieve_video_character(self, character_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.get_character,
            args=(character_id,),
            kwargs=kwargs,
            operation="OpenAI 视频角色读取",
        )

    async def async_edit_video(self, prompt: str, video: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.edit,
            kwargs=self._merge_resource_kwargs(
                {"prompt": prompt, "video": video}, kwargs, "OpenAI 视频编辑"
            ),
            operation="OpenAI 视频编辑",
        )

    async def async_extend_video(self, prompt: str, seconds: Any, video: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.extend,
            kwargs=self._merge_resource_kwargs(
                {"prompt": prompt, "seconds": seconds, "video": video},
                kwargs,
                "OpenAI 视频续写",
            ),
            operation="OpenAI 视频续写",
        )

    async def async_remix_video(self, video_id: str, prompt: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.remix,
            args=(video_id,),
            kwargs=self._merge_resource_kwargs({"prompt": prompt}, kwargs, "OpenAI 视频混剪"),
            operation="OpenAI 视频混剪",
        )

    async def async_poll_video(
        self,
        video_id: str,
        *,
        poll_interval_ms: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步轮询视频处理状态，仅支持 SDK 的 ``poll_interval_ms``。"""
        reject_unsupported_kwargs("OpenAI 视频轮询", kwargs)
        return await self._call_async_sdk_resource(
            self._get_aclient().videos.poll,
            args=(video_id,),
            kwargs={
                "poll_interval_ms": poll_interval_ms,
            }
            if poll_interval_ms is not None
            else {},
            operation="OpenAI 视频轮询",
        )

    async def async_create_upload(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().uploads.create,
            kwargs=kwargs,
            operation="OpenAI Upload 创建",
        )

    async def async_complete_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().uploads.complete,
            args=(upload_id,),
            kwargs=kwargs,
            operation="OpenAI Upload 完成",
        )

    async def async_cancel_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().uploads.cancel,
            args=(upload_id,),
            kwargs=kwargs,
            operation="OpenAI Upload 取消",
        )

    async def async_create_upload_part(self, upload_id: str, data: Any, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().uploads.parts.create,
            args=(upload_id,),
            kwargs=self._merge_resource_kwargs({"data": data}, kwargs, "OpenAI Upload 分片创建"),
            operation="OpenAI Upload 分片创建",
        )

    async def async_upload_file_chunked(
        self,
        *,
        file: Any,
        mime_type: str,
        purpose: str,
        filename: str | None = None,
        bytes: int | None = None,
        part_size: int | None = None,
        md5: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步分片上传，严格匹配 SDK 的窄参数契约。"""
        reject_unsupported_kwargs("OpenAI 分片文件上传", kwargs)
        request = {
            key: value
            for key, value in {
                "file": file,
                "mime_type": mime_type,
                "purpose": purpose,
                "filename": filename,
                "bytes": bytes,
                "part_size": part_size,
                "md5": md5,
            }.items()
            if value is not None
        }
        return await self._call_async_sdk_resource(
            self._get_aclient().uploads.upload_file_chunked,
            kwargs=request,
            operation="OpenAI 分片文件上传",
        )

    async def async_compact_responses(
        self,
        *,
        model: str | None = None,
        input: Any = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: Any = None,
        prompt_cache_retention: str | None = None,
        service_tier: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        """异步压缩 Responses 会话，严格匹配 AsyncOpenAI SDK 参数。"""
        self._require_responses_resource("compact_responses")
        reject_unsupported_kwargs("OpenAI Responses compact", kwargs)
        compact_kwargs = {
            key: value
            for key, value in {
                "model": model or self._model_name,
                "input": input,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_options": prompt_cache_options,
                "prompt_cache_retention": prompt_cache_retention,
                "service_tier": service_tier,
                "extra_headers": extra_headers,
                "extra_query": extra_query,
                "extra_body": extra_body,
                "timeout": timeout,
            }.items()
            if value is not None
        }
        return await self._call_async_sdk_resource(
            self._get_aclient().responses.compact,
            kwargs=compact_kwargs,
            operation="OpenAI Responses compact",
        )

    async def async_create_conversation(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.create,
            kwargs=kwargs,
            operation="OpenAI Conversation 创建",
        )

    async def async_retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.retrieve,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 读取",
        )

    async def async_update_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.update,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 更新",
        )

    async def async_delete_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.delete,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 删除",
        )

    async def async_list_conversation_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.items.list,
            args=(conversation_id,),
            kwargs=kwargs,
            operation="OpenAI Conversation 项目列表",
        )

    async def async_create_conversation_items(
        self, conversation_id: str, items: Any, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.items.create,
            args=(conversation_id,),
            kwargs=self._merge_resource_kwargs(
                {"items": items}, kwargs, "OpenAI Conversation 项目创建"
            ),
            operation="OpenAI Conversation 项目创建",
        )

    async def async_retrieve_conversation_item(
        self, conversation_id: str, item_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.items.retrieve,
            args=(item_id,),
            kwargs=self._merge_resource_kwargs(
                {"conversation_id": conversation_id},
                kwargs,
                "OpenAI Conversation 项目读取",
            ),
            operation="OpenAI Conversation 项目读取",
        )

    async def async_delete_conversation_item(
        self, conversation_id: str, item_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().conversations.items.delete,
            args=(item_id,),
            kwargs=self._merge_resource_kwargs(
                {"conversation_id": conversation_id},
                kwargs,
                "OpenAI Conversation 项目删除",
            ),
            operation="OpenAI Conversation 项目删除",
        )

    async def async_create_container(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.create,
            kwargs=kwargs,
            operation="OpenAI Container 创建",
        )

    async def async_retrieve_container(self, container_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.retrieve,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 读取",
        )

    async def async_list_containers(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.list,
            kwargs=kwargs,
            operation="OpenAI Container 列表",
        )

    async def async_delete_container(self, container_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.delete,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 删除",
        )

    async def async_create_container_file(
        self,
        container_id: str,
        file: Any = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        file_kwargs = {
            key: value
            for key, value in {"file": file, "file_id": file_id}.items()
            if value is not None
        }
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.files.create,
            args=(container_id,),
            kwargs=self._merge_resource_kwargs(
                file_kwargs,
                kwargs,
                "OpenAI Container 文件创建",
            ),
            operation="OpenAI Container 文件创建",
        )

    async def async_list_container_files(self, container_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.files.list,
            args=(container_id,),
            kwargs=kwargs,
            operation="OpenAI Container 文件列表",
        )

    async def async_retrieve_container_file(
        self, container_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.files.retrieve,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件读取",
            ),
            operation="OpenAI Container 文件读取",
        )

    async def async_delete_container_file(
        self, container_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().containers.files.delete,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件删除",
            ),
            operation="OpenAI Container 文件删除",
        )

    async def async_container_file_content(
        self, container_id: str, file_id: str, **kwargs: Any
    ) -> Any:
        files = getattr(getattr(self._get_aclient(), "containers", None), "files", None)
        content = getattr(files, "content", None)
        method = getattr(content, "retrieve", None)
        if not callable(method):
            raise NotImplementedError(
                "当前 OpenAI SDK 不提供容器文件内容读取资源 （containers.files.content.retrieve）。"
            )
        return await self._call_async_sdk_resource(
            method,
            args=(file_id,),
            kwargs=self._merge_resource_kwargs(
                {"container_id": container_id},
                kwargs,
                "OpenAI Container 文件内容读取",
            ),
            operation="OpenAI Container 文件内容读取",
        )

    async def async_create_fine_tuning_job(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.create,
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 创建",
        )

    async def async_list_fine_tuning_jobs(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.list,
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 列表",
        )

    async def async_retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.retrieve,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 读取",
        )

    async def async_cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.cancel,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 取消",
        )

    async def async_pause_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.pause,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 暂停",
        )

    async def async_resume_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.resume,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 恢复",
        )

    async def async_list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().fine_tuning.jobs.list_events,
            args=(job_id,),
            kwargs=kwargs,
            operation="OpenAI Fine-tuning 事件列表",
        )

    async def async_create_eval(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.create,
            kwargs=kwargs,
            operation="OpenAI Eval 创建",
        )

    async def async_list_evals(self, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.list,
            kwargs=kwargs,
            operation="OpenAI Eval 列表",
        )

    async def async_retrieve_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.retrieve,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 读取",
        )

    async def async_update_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.update,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 更新",
        )

    async def async_delete_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.delete,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval 删除",
        )

    async def async_create_eval_run(self, eval_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.create,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval Run 创建",
        )

    async def async_list_eval_runs(self, eval_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.list,
            args=(eval_id,),
            kwargs=kwargs,
            operation="OpenAI Eval Run 列表",
        )

    async def async_retrieve_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.retrieve,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 读取"
            ),
            operation="OpenAI Eval Run 读取",
        )

    async def async_cancel_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.cancel,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 取消"
            ),
            operation="OpenAI Eval Run 取消",
        )

    async def async_delete_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.delete,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id}, kwargs, "OpenAI Eval Run 删除"
            ),
            operation="OpenAI Eval Run 删除",
        )

    async def async_list_eval_run_output_items(
        self, eval_id: str, run_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.output_items.list,
            args=(run_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id},
                kwargs,
                "OpenAI Eval Run 输出项列表",
            ),
            operation="OpenAI Eval Run 输出项列表",
        )

    async def async_retrieve_eval_run_output_item(
        self, eval_id: str, run_id: str, output_item_id: str, **kwargs: Any
    ) -> Any:
        return await self._call_async_sdk_resource(
            self._get_aclient().evals.runs.output_items.retrieve,
            args=(output_item_id,),
            kwargs=self._merge_resource_kwargs(
                {"eval_id": eval_id, "run_id": run_id},
                kwargs,
                "OpenAI Eval Run 输出项读取",
            ),
            operation="OpenAI Eval Run 输出项读取",
        )

    def close(self) -> None:
        """释放同步和异步 OpenAI SDK 客户端。

        同步上下文中会桥接等待异步客户端的关闭；活动事件循环中则显式
        要求调用 ``aclose``，并保留未关闭的异步客户端供后续清理。
        """
        client = self._client
        async_client = self._aclient
        self._client = None
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
                    close_resource_sync(async_client, "OpenAI 异步客户端")
                except Exception as exc:  # noqa: BLE001 - close all resources before reporting
                    first_error = first_error or exc
                else:
                    self._aclient = None
        else:
            self._aclient = None
        if first_error is not None:
            raise first_error

    async def aclose(self) -> None:
        """释放同步和异步 OpenAI SDK 客户端，即使其中一个关闭失败。"""
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
