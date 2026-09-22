import base64
import binascii
import inspect
import json
import math
import os
import re
import time
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar
from urllib.parse import parse_qsl, quote, urlsplit

from google import genai
from google.auth.credentials import Credentials
from google.genai import types
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
    extract_system_prompt,
    field,
    is_retryable_error,
    iterate_async,
    normalize_embedding_vector,
    normalize_messages,
    normalize_usage,
    raise_for_stream_error_event,
    reject_unsupported_kwargs,
    retry_async_call,
    retry_async_stream,
    retry_sync_call,
    retry_sync_stream,
    validate_complete_tool_call,
)
from src.providers.resources import AsyncGoogleResources, GoogleResources
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import (
    _model_dump_mapping,
    redact_sensitive_text,
    validate_secret_free_payload,
    validate_secret_free_request_overrides,
)

logger = get_module_logger(__name__)


class GoogleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    Google模型提供商，统一处理Gemini LLM和Embedding。
    已升级为使用最新的 google-genai SDK (v1.0+)。
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
            "embedding",
            "files",
            "batches",
            "cache",
            "models",
            "tuning",
            "images",
            "videos",
            "file_search",
            "interactions",
            "agents",
            "webhooks",
            "environments",
            "triggers",
            "live",
            "auth_tokens",
            "operations",
            "chats",
        }
    )

    _EXPERIMENTAL_RESOURCE_LABELS: ClassVar[dict[str, str]] = {
        "auth_tokens": "Auth Token",
        "interactions": "Interactions",
        "agents": "Agents",
        "webhooks": "Webhooks",
        "environments": "Environments",
        "triggers": "Triggers",
    }

    _CLIENT_OPTION_KEYS = frozenset(
        {
            "enterprise",
            "vertexai",
            "project",
            "location",
            "credentials",
            "http_options",
            "debug_config",
        }
    )
    _OPTION_KEYS = _CLIENT_OPTION_KEYS | frozenset(
        {
            "extra_body",
            "response_format",
            "timeout",
            "max_tokens",
            "task_type",
            "output_dimensionality",
            "title",
            "mime_type",
            "auto_truncate",
            "document_ocr",
            "audio_track_extraction",
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "max_output_tokens",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_mime_type",
            "response_schema",
            "response_json_schema",
            "safety_settings",
            "cached_content",
            "response_modalities",
            "thinking",
            "thinking_config",
            "automatic_function_calling",
            "speech_config",
            "image_config",
            "labels",
            "routing_config",
            "model_selection_config",
            "media_resolution",
            "enable_enhanced_civic_answers",
            "service_tier",
            "logprobs",
            "response_logprobs",
            "audio_timestamp",
            "audio_transcription_config",
            "model_armor_config",
            "should_return_http_response",
        }
    )

    _HTTP_OPTIONS_SENSITIVE_EXACT = frozenset(
        {
            "apikey",
            "xapikey",
            "accesskey",
            "secretkey",
            "token",
            "password",
            "passwd",
            "authorization",
            "auth",
            "bearer",
            "accesstoken",
            "xaccesstoken",
            "authtoken",
            "xauthtoken",
            "authkey",
            "credential",
            "credentials",
            "cookie",
            "setcookie",
            "proxyauthorization",
            "secret",
            "privatekey",
            "httpclient",
            "httpxclient",
            "httpxasyncclient",
            "aiohttpclient",
            "clientargs",
            "asyncclientargs",
            "baseurl",
            "baseurlresourcescope",
            "proxy",
            "proxies",
            "transport",
            "verify",
            "cert",
            "mounts",
        }
    )

    def __init__(self, model_name: str, options: dict[str, Any] | None = None):
        self._model_name = model_name
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("Google options 必须是对象。")
        self._options = dict(options or {})
        self._validate_options(self._options)
        logger.info(f"初始化 GoogleProvider (google-genai)，模型: {model_name}")
        self._client: Any | None = None

    @property
    def resources(self) -> GoogleResources:
        return GoogleResources(self)

    @property
    def async_resources(self) -> AsyncGoogleResources:
        return AsyncGoogleResources(self)

    def _get_client(self) -> Any:
        """延迟初始化并返回 genai.Client 实例。"""
        # Some test doubles and SDK client wrappers define false-y equality or
        # length semantics.  Only ``None`` means that initialization has not
        # happened yet; a valid but false-y client must be reused.
        if self._client is None:
            self._validate_options(self._options)
            settings = get_settings()
            credentials = self._options.get("credentials")
            vertexai = self._configured_vertex_mode()
            if credentials is None and os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None:
                credential_path = self._configured_setting(
                    settings,
                    "google_application_credentials",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                )
                if credential_path:
                    credentials = self._load_credentials_file(credential_path)
            api_key = self._configured_api_key(settings)
            if credentials is not None and vertexai is False:
                raise ValueError(
                    "Google credentials 需要 Vertex/Enterprise 模式；"
                    "请将 vertexai 或 enterprise 设为 true。"
                )
            if credentials is not None and vertexai is None:
                vertexai = True
            if credentials is None and not api_key and vertexai is not True:
                logger.error("Google API Key 未设置。")
                raise ValueError(
                    "GOOGLE_API_KEY is required for GoogleProvider unless options.credentials is provided "
                    "or Vertex/Enterprise mode enables Application Default Credentials"
                )

            logger.debug("正在初始化 google-genai Client")
            client_options = {
                key: value
                for key, value in self._options.items()
                if key in self._CLIENT_OPTION_KEYS
            }
            for key in ("enterprise", "vertexai"):
                if key in client_options:
                    client_options[key] = self._as_bool(client_options[key])
            if "enterprise" not in client_options and "vertexai" not in client_options:
                configured_flags = self._configured_environment_vertex_flags(settings)
                if "GOOGLE_GENAI_USE_ENTERPRISE" in configured_flags:
                    client_options["enterprise"] = configured_flags["GOOGLE_GENAI_USE_ENTERPRISE"]
                elif "GOOGLE_GENAI_USE_VERTEXAI" in configured_flags:
                    client_options["vertexai"] = configured_flags["GOOGLE_GENAI_USE_VERTEXAI"]
            if "debug_config" in client_options:
                client_options["debug_config"] = self._coerce_debug_config(
                    client_options["debug_config"]
                )
            if credentials is not None:
                if "credentials" not in client_options:
                    client_options["credentials"] = credentials
                if (
                    vertexai is True
                    and "enterprise" not in client_options
                    and "vertexai" not in client_options
                ):
                    # google-genai treats explicit credentials as Vertex ADC
                    # credentials only when the Vertex transport is selected.
                    # Make that implication explicit instead of letting the SDK
                    # fall through to Developer API key validation.
                    client_options["vertexai"] = True
            if vertexai is True:
                for option_name, setting_name, env_name in (
                    ("project", "google_cloud_project", "GOOGLE_CLOUD_PROJECT"),
                    ("location", "google_cloud_location", "GOOGLE_CLOUD_LOCATION"),
                ):
                    if option_name not in client_options:
                        configured_value = self._configured_setting(
                            settings, setting_name, env_name
                        )
                        if configured_value:
                            client_options[option_name] = configured_value
            # ``google-genai`` supports both ADC and Vertex express-mode API
            # keys. Only explicit credentials and an API key are mutually
            # exclusive; an API key remains valid when the Vertex endpoint is
            # selected and must not be dropped when it came from config.toml.
            if credentials is None and api_key:
                self._client = genai.Client(api_key=api_key, **client_options)
            else:
                # google-genai can resolve Vertex ADC when neither api_key nor
                # explicit credentials is supplied; credentials and api_key
                # must never be passed together.
                self._client = genai.Client(**client_options)
            logger.info("google-genai Client 初始化成功。")
        return self._client

    @staticmethod
    def _configured_api_key(settings: Any) -> str | None:
        """读取 Google SDK 支持的两个环境键，优先 GOOGLE_API_KEY。"""
        for name in ("google_api_key", "gemini_api_key"):
            value = getattr(settings, name, None)
            # MagicMock-style test doubles expose arbitrary attributes; only
            # actual strings are valid SDK API keys here.
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _configured_setting(settings: Any, field_name: str, env_name: str) -> str | None:
        """读取环境变量或 Settings 中的非空字符串配置。"""
        value = os.getenv(env_name)
        if value is None:
            value = getattr(settings, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _load_credentials_file(path: str) -> Any:
        """从 Settings 中配置的服务账号路径加载 ADC 凭证。"""
        try:
            from google.auth import load_credentials_from_file
        except ImportError as exc:  # pragma: no cover - google-genai supplies google-auth
            raise RuntimeError("当前环境缺少 google-auth，无法加载 Google ADC 凭证。") from exc
        try:
            credentials, _ = load_credentials_from_file(path)
        except Exception as exc:
            raise ValueError("Google GOOGLE_APPLICATION_CREDENTIALS 文件无法加载。") from exc
        return credentials

    @staticmethod
    def _as_bool(value: Any) -> bool:
        """将配置文件中的布尔字符串规范为 SDK 需要的 bool。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        if not isinstance(options, Mapping):
            raise ValueError("Google options 必须是对象。")
        unknown = sorted(set(options).difference(cls._OPTION_KEYS))
        if unknown:
            raise ValueError("Google options 包含不支持的字段: " + ", ".join(unknown))
        if "extra_body" in options:
            validate_secret_free_payload(
                options["extra_body"],
                "Google",
                "options.extra_body",
            )
        if "http_options" in options:
            cls._validate_http_options(options["http_options"])
        if "timeout" in options:
            cls._validate_timeout(options["timeout"], "Google options.timeout")
        configured_body = options.get("extra_body")
        model_conflicts: list[str] = []
        if options.get("http_options") is not None:
            if options.get("timeout") is not None:
                model_conflicts.append("timeout")
            if configured_body:
                model_conflicts.append("extra_body")
        if model_conflicts:
            raise ValueError(
                "Google options.http_options 不能与模型级 options 同时配置: "
                + ", ".join(model_conflicts)
            )

    @staticmethod
    def _require_resource(client: Any, name: str, label: str) -> Any:
        """读取 SDK 资源并把版本差异转换成明确的能力错误。"""

        try:
            resource = getattr(client, name)
        except (AttributeError, NotImplementedError) as exc:
            raise NotImplementedError(
                f"当前 google-genai SDK 未提供 {label}资源；请升级 SDK 或使用官方支持的接口。"
            ) from exc
        if resource is None:
            raise NotImplementedError(
                f"当前 google-genai SDK 未提供 {label}资源；请升级 SDK 或使用官方支持的接口。"
            )
        return resource

    def _require_provider_resource(
        self, method_name: str, kwargs: Mapping[str, Any] | None = None
    ) -> None:
        """把实验性资源的版本差异转换成明确的能力错误。

        ``kwargs`` 未使用：本 Provider 的原生资源方法不做参数校验，动态路径
        没有可绕过的参数约束。凭证边界由 ``_NativeResourceProxy.__call__``
        在调用本钩子前统一执行。
        """
        name = method_name.removeprefix("async_")
        resource_name: str | None = None
        # 显式 Facade 名（``create_interaction``）与动态资源树路径
        # （``google.interactions.create``）形状不同：前者把资源名嵌在方法名里，
        # 后者是点分路径，资源名是一个独立分段。只按 ``startswith``/``_singular``
        # 匹配会让所有动态路径都落到 ``None`` 直接放行——用户拿一个不含该资源的
        # 客户端走 ``resources.interactions`` 就能绕开这道门禁。
        segments = name.split(".")
        for candidate in self._EXPERIMENTAL_RESOURCE_LABELS:
            singular = candidate.removesuffix("s")
            if (
                name.startswith(singular)
                or f"_{singular}" in name
                or candidate in segments
                or singular in segments
            ):
                resource_name = candidate
                break
        if resource_name is None:
            if name == "create_auth_token":
                resource_name = "auth_tokens"
            else:
                return
        client = self._get_client()
        if method_name.startswith("async_"):
            try:
                client = client.aio
            except (AttributeError, NotImplementedError) as exc:
                raise NotImplementedError(
                    f"当前 google-genai SDK 未提供异步 {self._EXPERIMENTAL_RESOURCE_LABELS[resource_name]}资源。"
                ) from exc
        self._require_resource(
            client,
            resource_name,
            self._EXPERIMENTAL_RESOURCE_LABELS[resource_name],
        )

    @classmethod
    def _validate_http_options(cls, value: Any) -> None:
        """校验 Google SDK HTTP 扩展，允许普通追踪头但拒绝凭证。"""
        if value is None:
            return
        if isinstance(value, Mapping):
            payload: Any = dict(value)
        else:
            model_dump = getattr(value, "model_dump", None)
            if not callable(model_dump):
                raise ValueError(
                    "Google options.http_options 必须是对象表或 google-genai HttpOptions。"
                )
            try:
                payload = model_dump(exclude_none=True)
            except TypeError:
                try:
                    payload = model_dump()
                except Exception as exc:
                    raise ValueError("Google options.http_options 无法读取。") from exc
            except Exception as exc:
                raise ValueError("Google options.http_options 无法读取。") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Google options.http_options 必须是对象。")

        if "timeout" in payload and payload["timeout"] is not None:
            timeout = payload["timeout"]
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0
                or int(float(timeout)) != float(timeout)
            ):
                raise ValueError("Google options.http_options.timeout 必须是正整数毫秒。")

        found: list[str] = []

        def normalized_key(key: Any) -> str:
            return "".join(char for char in str(key).lower() if char.isalnum())

        def is_sensitive_key(key: Any) -> bool:
            normalized = normalized_key(key)
            if normalized in cls._HTTP_OPTIONS_SENSITIVE_EXACT:
                return True
            return any(
                normalized.endswith(marker)
                for marker in (
                    "apikey",
                    "accesskey",
                    "accesstoken",
                    "authtoken",
                    "secret",
                    "secretkey",
                    "clientsecret",
                    "password",
                    "credential",
                    "cookie",
                )
            )

        def check_url_query(value: Any, path: str) -> None:
            if not isinstance(value, str):
                return
            try:
                query = parse_qsl(urlsplit(value).query, keep_blank_values=True)
            except ValueError:
                return
            for key, _ in query:
                if is_sensitive_key(key):
                    found.append(f"{path}?{key}")

        def visit(item: Any, path: str = "") -> None:
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    current = f"{path}.{key}" if path else str(key)
                    if nested is None:
                        continue
                    if is_sensitive_key(key):
                        found.append(current)
                        continue
                    visit(nested, current)
            elif isinstance(item, (list, tuple, set, frozenset)):
                for index, nested in enumerate(item):
                    visit(nested, f"{path}[{index}]")
            elif isinstance(item, str):
                if redact_sensitive_text(item) != item:
                    found.append(path or "value")
                check_url_query(item, path)
                try:
                    parsed = urlsplit(item)
                except ValueError:
                    parsed = None
                if parsed is not None and (
                    parsed.username is not None or parsed.password is not None
                ):
                    found.append(f"{path} userinfo")

        visit(payload)
        if found:
            raise ValueError(
                "Google options.http_options 不允许包含凭证、自定义客户端或连接覆盖: "
                + ", ".join(sorted(found))
            )

    @staticmethod
    def _validate_timeout(value: Any, field_name: str) -> float:
        """将请求 timeout 规范为有限正数，避免底层比较抛出 TypeError。"""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{field_name} 必须是正数。")
        return float(value)

    @classmethod
    def _validate_resource_config(cls, config: Any) -> None:
        """检查资源 config 中的 HTTP 配置，但不误判业务字段。"""
        if config is None:
            return
        if isinstance(config, Mapping):
            payload: Any = dict(config)
        else:
            payload = _model_dump_mapping(config)
        if not isinstance(payload, Mapping):
            return

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    normalized = "".join(char for char in str(key).lower() if char.isalnum())
                    if normalized == "httpoptions":
                        cls._validate_http_options(nested)
                        continue
                    visit(nested)
                return
            dumped = _model_dump_mapping(value)
            if dumped is not None:
                visit(dumped)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for nested in value:
                    visit(nested)

        visit(payload)

    @staticmethod
    def _coerce_debug_config(value: Any) -> Any:
        """将配置文件中的 DebugConfig 表转换为 SDK 对象。"""
        if value is None or not isinstance(value, Mapping):
            return value
        allowed = {"client_mode", "replays_directory", "replay_id"}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError("Google debug_config 包含不支持的字段: " + ", ".join(unknown))
        try:
            from google.genai.client import DebugConfig
        except ImportError as exc:
            raise RuntimeError("当前 google-genai SDK 不支持 debug_config。") from exc
        try:
            return DebugConfig(**dict(value))
        except Exception as exc:
            raise ValueError("Google debug_config 配置无效。") from exc

    def _configured_vertex_mode(self) -> bool | None:
        """读取显式或 SDK 支持的环境级 Vertex/Enterprise 开关。"""
        options = getattr(self, "_options", {}) or {}
        explicit_values = {
            key: self._as_bool(options[key])
            for key in ("enterprise", "vertexai")
            if key in options and options[key] is not None
        }
        if explicit_values:
            if len(set(explicit_values.values())) > 1:
                raise ValueError(
                    "Google options.enterprise 与 options.vertexai 的值冲突；"
                    "请只设置一个，或将两者设为相同值。"
                )
            return next(iter(explicit_values.values()))

        settings = get_settings()
        environment_values = self._configured_environment_vertex_flags(settings)
        if len(set(environment_values.values())) > 1:
            raise ValueError(
                "GOOGLE_GENAI_USE_ENTERPRISE 与 GOOGLE_GENAI_USE_VERTEXAI 的值冲突；"
                "请只保留一个，或将两者设为相同值。"
            )
        # 未配置开关时保持 unknown，而不是武断地当作 Developer API。
        # 这样测试替身、旧 SDK 或由 SDK 自己决定 endpoint 的客户端仍能
        # 继续执行能力探测；明确设置 false 时才拒绝 Vertex-only 能力。
        return next(iter(environment_values.values())) if environment_values else None

    @classmethod
    def _configured_environment_vertex_flags(cls, settings: Any) -> dict[str, bool]:
        """读取进程环境或 ``.env`` 载入的 Vertex/Enterprise 开关。"""

        values: dict[str, bool] = {}
        for env_name, field_name in (
            ("GOOGLE_GENAI_USE_ENTERPRISE", "google_genai_use_enterprise"),
            ("GOOGLE_GENAI_USE_VERTEXAI", "google_genai_use_vertexai"),
        ):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                raw_value = getattr(settings, field_name, None)
                # MagicMock-style test doubles expose arbitrary attributes;
                # those are not valid Settings values and must be ignored.
                if not isinstance(raw_value, (bool, int, float, str)):
                    raw_value = None
            if raw_value is not None:
                values[env_name] = cls._as_bool(raw_value)
        return values

    @staticmethod
    def _client_vertex_mode(client: Any) -> bool | None:
        """返回已构造客户端的真实模式；测试替身没有该属性时返回 None。"""
        client_values = vars(client) if hasattr(client, "__dict__") else {}
        api_client = client_values.get("_api_client")
        api_values = vars(api_client) if hasattr(api_client, "__dict__") else {}
        if api_client is not None and "vertexai" in api_values:
            return bool(api_client.vertexai)
        if "vertexai" in client_values:
            return bool(client_values["vertexai"])
        return None

    def _effective_vertex_mode(self, client: Any) -> bool | None:
        """优先读取 SDK 实际模式，测试替身或旧 SDK 缺失时回退到配置。"""
        actual = self._client_vertex_mode(client)
        return self._configured_vertex_mode() if actual is None else actual

    def _vertex_embed_content_only(self) -> bool:
        """判断当前模型是否只能通过 Vertex 的单内容 embedContent 调用。"""
        model = self._model_name.lower()
        return ("gemini" in model) or "maas" in model

    @staticmethod
    def _validate_token_count_fields(request: CompletionRequest) -> None:
        """拒绝 token 统计不会接受或不会生效的生成参数。"""
        unsupported = {
            key: value
            for key, value in request.to_invoke_kwargs().items()
            if key
            in {
                "max_tokens",
                "top_p",
                "top_k",
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
                "reasoning",
                "thinking",
                "store",
                "background",
                "parallel_tool_calls",
                "max_tool_calls",
                "conversation",
                "session",
                "context_management",
                "caching",
                "expire_at",
                "prompt_cache_options",
                "safety_identifier",
                "moderation",
                "verbosity",
                "prompt_cache_key",
                "prompt_cache_retention",
                "service_tier",
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
                "user",
                "personality",
            }
            and value is not None
        }
        reject_unsupported_kwargs("Google token count", unsupported)

    @staticmethod
    def _validate_compute_token_fields(request: CompletionRequest) -> None:
        """拒绝 computeTokens 不接受或不会生效的请求字段。"""
        unsupported = {
            key: value
            for key, value in request.to_invoke_kwargs().items()
            if key
            in {
                "max_tokens",
                "top_p",
                "top_k",
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
                "reasoning",
                "thinking",
                "store",
                "background",
                "parallel_tool_calls",
                "max_tool_calls",
                "conversation",
                "session",
                "context_management",
                "caching",
                "expire_at",
                "prompt_cache_options",
                "safety_identifier",
                "moderation",
                "verbosity",
                "prompt_cache_key",
                "prompt_cache_retention",
                "service_tier",
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
                "user",
                "personality",
                "tools",
            }
            and value is not None
        }
        reject_unsupported_kwargs("Google compute tokens", unsupported)

    @staticmethod
    def _token_count_system_prompt(request: CompletionRequest) -> str | None:
        """返回 token 统计请求中真正显式提供的 system prompt。"""
        message_system = extract_system_prompt(request.messages)
        if message_system:
            return message_system
        default_system_prompt = CompletionRequest.__dataclass_fields__["system_prompt"].default
        return None if request.system_prompt == default_system_prompt else request.system_prompt

    def _embedding_options(self) -> dict[str, Any]:
        supported = {
            "task_type",
            "output_dimensionality",
            "title",
            "mime_type",
            "auto_truncate",
            "document_ocr",
            "audio_track_extraction",
        }
        return {key: value for key, value in self._options.items() if key in supported}

    @staticmethod
    def _resource_config(config_type: Any, config: Any, values: dict[str, Any]) -> Any:
        """将 Facade 的关键字配置转换为 google-genai 的 ``config=`` 参数。"""
        GoogleProvider._validate_resource_config(config)
        values = dict(values)
        request_options = {
            key: values.pop(key, None)
            for key in ("extra_headers", "extra_body", "extra_query", "timeout")
        }
        http_options = GoogleProvider._build_http_options(**request_options)
        if http_options is not None:
            if "http_options" in values:
                raise ValueError("Google SDK 的 http_options 不能与请求级 HTTP 配置同时传入。")
            values["http_options"] = http_options
        if config is not None:
            if values:
                raise ValueError("Google SDK 的 config 不能与同级配置关键字同时传入。")
            return config
        if not values:
            return None
        return config_type(**values)

    @classmethod
    def _resource_config_kwargs(
        cls, config_type: Any, config: Any, values: dict[str, Any]
    ) -> dict[str, Any]:
        built = cls._resource_config(config_type, config, values)
        return {"config": built} if built is not None else {}

    @staticmethod
    def _validate_embedding_kwargs(kwargs: dict[str, Any]) -> None:
        allowed = {
            "task_type",
            "output_dimensionality",
            "title",
            "mime_type",
            "auto_truncate",
            "document_ocr",
            "audio_track_extraction",
        }
        unsupported = sorted(
            key for key, value in kwargs.items() if value is not None and key not in allowed
        )
        if unsupported:
            raise ValueError(f"Google Embedding 不支持请求参数: {', '.join(unsupported)}")

    @staticmethod
    def _validate_request(request: CompletionRequest) -> None:
        supported = {
            "prompt",
            "system_prompt",
            "messages",
            "tools",
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "seed",
            "modalities",
            "stop",
            "response_format",
            "tool_choice",
            "extra_body",
            "extra_headers",
            "extra_query",
            "thinking",
            "service_tier",
            "timeout",
        }
        reject_unsupported_kwargs(
            "Google SDK",
            {
                key: value
                for key, value in request.to_invoke_kwargs().items()
                if key not in supported
            },
        )

    @staticmethod
    def _build_http_options(
        *,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_query: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any | None:
        """将统一请求的 HTTP 扩展转换为 google-genai 的 HttpOptions。"""
        extra_headers, extra_query = validate_secret_free_request_overrides(
            extra_headers,
            extra_query,
            "Google",
        )
        if extra_query is not None:
            raise ValueError(
                "Google SDK 不支持请求级 extra_query；请在 options.http_options 中配置。"
            )
        extra_body = (
            validate_secret_free_payload(
                extra_body,
                "Google",
                "extra_body",
            )
            if extra_body is not None
            else None
        )
        values: dict[str, Any] = {}
        if extra_headers is not None:
            values["headers"] = dict(extra_headers)
        if extra_body is not None:
            values["extra_body"] = dict(extra_body)
        if timeout is not None:
            timeout = GoogleProvider._validate_timeout(timeout, "Google 请求 timeout")
            values["timeout"] = max(1, int(timeout * 1000))
        return types.HttpOptions(**values) if values else None

    @staticmethod
    async def _resolve_async_result(result: Any) -> Any:
        return await result if inspect.isawaitable(result) else result

    def _generation_options(self) -> dict[str, Any]:
        supported = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "max_output_tokens",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "response_mime_type",
            "response_schema",
            "response_json_schema",
            "safety_settings",
            "cached_content",
            "response_modalities",
            "thinking_config",
            "automatic_function_calling",
            "speech_config",
            "image_config",
            "labels",
            "routing_config",
            "model_selection_config",
            "media_resolution",
            "enable_enhanced_civic_answers",
            "service_tier",
            "logprobs",
            "response_logprobs",
            "audio_timestamp",
            "audio_transcription_config",
            "model_armor_config",
            "should_return_http_response",
        }
        return {key: value for key, value in self._options.items() if key in supported}

    @staticmethod
    def _response_format_values(response_format: Any) -> dict[str, Any]:
        """将统一 response_format 转换为 Gemini GenerateContentConfig 字段。"""
        if not isinstance(response_format, Mapping):
            raise ValueError("Google response_format 必须是对象。")
        format_type = response_format.get("type")
        if format_type == "json_object":
            return {"response_mime_type": "application/json"}
        if format_type == "json_schema":
            schema = response_format.get("json_schema", response_format)
            if not isinstance(schema, Mapping):
                raise ValueError("Google response_format.json_schema 必须是对象。")
            payload = schema.get("schema", response_format.get("schema"))
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Google response_format.json_schema.schema 必须是 JSON Schema 对象。"
                )
            return {
                "response_mime_type": "application/json",
                "response_json_schema": dict(payload),
            }
        raise ValueError("Google response_format 仅支持 json_object 或 json_schema。")

    def _generation_http_options(
        self,
        *,
        extra_body: dict[str, Any] | None,
        extra_headers: dict[str, str] | None,
        extra_query: dict[str, Any] | None,
        timeout: float | None,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, str] | None,
        dict[str, Any] | None,
        float | None,
        Any | None,
    ]:
        """合并模型级与调用级 HTTP 扩展，拒绝同名字段静默覆盖。"""
        options = getattr(self, "_options", {}) or {}
        configured_body = options.get("extra_body")
        if configured_body is not None and not isinstance(configured_body, Mapping):
            raise ValueError("Google options.extra_body 必须是对象。")
        configured_body = validate_secret_free_payload(
            configured_body,
            "Google",
            "options.extra_body",
        )
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise ValueError("Google extra_body 必须是对象。")
        configured_values = dict(configured_body or {})
        request_values = validate_secret_free_payload(
            extra_body,
            "Google",
            "extra_body",
        )
        overlap = sorted(set(configured_values).intersection(request_values))
        if overlap:
            raise ValueError("Google extra_body 与模型 options 重复: " + ", ".join(overlap))
        merged_body = {**configured_values, **request_values} or None
        effective_timeout = timeout
        if effective_timeout is None:
            effective_timeout = options.get("timeout")
        configured_http_options = options.get("http_options")
        request_body_configured = request_values if request_values else None
        if configured_http_options is not None and any(
            value is not None
            for value in (extra_headers, request_body_configured, extra_query, timeout)
        ):
            raise ValueError(
                "Google options.http_options 不能与请求级 extra_headers、extra_body、"
                "extra_query 或 timeout 同时传入。"
            )
        return (
            merged_body,
            dict(extra_headers) if extra_headers is not None else None,
            dict(extra_query) if extra_query is not None else None,
            effective_timeout,
            configured_http_options,
        )

    def _http_options_for_request(
        self,
        *,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any | None:
        """构造模型级与请求级 HTTP 配置共用的 ``HttpOptions``。

        Embedding、token-count 和 compute-token 接口不经过生成配置构造器，
        因此必须显式复用同一套模型级 ``extra_body``、``timeout`` 和
        ``http_options`` 语义，避免这些入口悄悄丢弃配置。
        """
        merged_body, request_headers, request_query, effective_timeout, configured = (
            self._generation_http_options(
                extra_body=extra_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
            )
        )
        if configured is not None:
            return configured
        return self._build_http_options(
            extra_headers=request_headers,
            extra_body=merged_body,
            extra_query=request_query,
            timeout=effective_timeout,
        )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
        if not tools:
            return None

        converted_tools: list[Any] = []
        declarations = []
        for tool in tools:
            if not isinstance(tool, dict):
                converted_tools.append(tool)
                continue
            native_keys = {
                "retrieval",
                "computer_use",
                "file_search",
                "google_search",
                "google_maps",
                "code_execution",
                "enterprise_web_search",
                "google_search_retrieval",
                "parallel_ai_search",
                "url_context",
                "mcp_servers",
                "exa_ai_search",
            }
            type_name = tool.get("type")
            type_aliases = {
                "google_search": "google_search",
                "code_execution": "code_execution",
                "url_context": "url_context",
                "file_search": "file_search",
            }
            if type_name in type_aliases:
                native_value = tool.get(type_name, {})
                if native_value is True or native_value is None:
                    native_value = {}
                converted_tools.append(types.Tool(**{type_aliases[type_name]: native_value}))
                continue
            if native_keys.intersection(tool):
                converted_tools.append(types.Tool(**tool))
                continue
            function = tool.get("function", tool)
            name = function.get("name")
            if not name:
                raise ValueError("Google 工具定义缺少 function.name。")
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=function.get("description"),
                    parameters_json_schema=function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                )
            )
        if declarations:
            converted_tools.insert(0, types.Tool(function_declarations=declarations))
        return converted_tools

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict) and "function_calling_config" in tool_choice:
            return types.ToolConfig(**tool_choice)
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function = tool_choice.get("function", {})
                if not isinstance(function, Mapping):
                    raise ValueError("Google function tool_choice 的 function 必须是对象。")
                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Google function tool_choice 缺少 function.name。")
                return types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=[name],
                    )
                )
            tool_choice = tool_choice.get("type", tool_choice.get("mode", "AUTO"))
        if isinstance(tool_choice, str):
            mode = {"auto": "AUTO", "required": "ANY", "any": "ANY", "none": "NONE"}.get(
                tool_choice.lower(), tool_choice.upper()
            )
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode(mode)
                )
            )
        return tool_choice

    def _build_generation_config(
        self,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        candidate_count: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        modalities: list[str] | None = None,
        stop: Sequence[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        values: dict[str, Any] = self._generation_options()
        configured_max_tokens = self._options.get("max_tokens")
        configured_max_output_tokens = values.get("max_output_tokens")
        if configured_max_tokens is not None and configured_max_output_tokens is not None:
            raise ValueError("Google options.max_tokens 与 options.max_output_tokens 重复。")
        if configured_max_tokens is not None:
            values["max_output_tokens"] = configured_max_tokens
        configured_thinking = self._options.get("thinking")
        configured_thinking_config = values.get("thinking_config")
        if configured_thinking is not None and configured_thinking_config is not None:
            raise ValueError("Google options.thinking 与 options.thinking_config 重复。")
        if configured_thinking is not None:
            values["thinking_config"] = configured_thinking
        configured_response_format = self._options.get("response_format")
        if configured_response_format is not None:
            if response_format is not None:
                raise ValueError("Google response_format 与模型 options 重复。")
            response_format = configured_response_format
        values.update(
            {
                "system_instruction": system_prompt,
                "tools": self._convert_tools(tools),
            }
        )
        if temperature is not None or "temperature" not in values:
            values["temperature"] = 0.7 if temperature is None else temperature
        if max_tokens is not None:
            values["max_output_tokens"] = max_tokens
        if top_p is not None:
            values["top_p"] = top_p
        if top_k is not None:
            values["top_k"] = top_k
        if candidate_count is not None:
            values["candidate_count"] = candidate_count
        if frequency_penalty is not None:
            values["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            values["presence_penalty"] = presence_penalty
        if seed is not None:
            values["seed"] = seed
        if modalities is not None:
            values["response_modalities"] = modalities
        if stop is not None:
            values["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if response_format is not None:
            values.update(self._response_format_values(response_format))
        if tool_choice is not None:
            values["tool_config"] = self._convert_tool_choice(tool_choice)
        if thinking is not None:
            values["thinking_config"] = thinking
        if service_tier is not None:
            values["service_tier"] = service_tier
        merged_body, request_headers, request_query, effective_timeout, configured_http_options = (
            self._generation_http_options(
                extra_body=extra_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
            )
        )
        if configured_http_options is not None:
            values["http_options"] = configured_http_options
        else:
            http_options = self._build_http_options(
                extra_headers=request_headers,
                extra_body=merged_body,
                extra_query=request_query,
                timeout=effective_timeout,
            )
            if http_options is not None:
                values["http_options"] = http_options
        return types.GenerateContentConfig(
            **{key: value for key, value in values.items() if value is not None}
        )

    @staticmethod
    def _decode_data_uri(value: str, context: str) -> tuple[str, bytes]:
        """严格解码 data URI，避免 ``b64decode`` 静默丢弃非法字符。"""
        if "," not in value:
            raise ValueError(f"Google {context} 的 data URI 缺少逗号分隔符。")
        header, encoded = value.split(",", 1)
        mime_type = header.split(";", 1)[0].removeprefix("data:")
        if not mime_type:
            raise ValueError(f"Google {context} 的 data URI 缺少 MIME 类型。")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Google {context} 的 data URI 不是有效 Base64。") from exc
        return mime_type, decoded

    @staticmethod
    def _contents(
        messages: Any,
        prompt: str | None,
        system_prompt: str | None,
        *,
        vertex_mode: bool | None = None,
    ) -> Any:
        """将统一消息转换为 Gemini 内容，并按客户端模式校验媒体 URI。

        Developer API 支持 Gemini Files API 的 ``file_id`` 映射；Vertex/Enterprise
        只接受 Cloud Storage URI 或可公开读取的 HTTPS URI。模式未知时保持
        Developer API 的历史兼容行为，实际 provider 调用会传入 SDK 检测结果。
        """
        normalized = normalize_messages(prompt, system_prompt, messages)
        if messages is None and prompt is not None:
            return prompt
        contents: list[Any] = []
        tool_call_names: dict[str, str] = {}
        pending_tool_parts: list[Any] = []

        def append_content(role: str, parts: list[Any]) -> None:
            """合并连续同角色消息，满足 Gemini user/model 交替约束。"""
            if contents and field(contents[-1], "role") == role:
                contents[-1].parts.extend(parts)
                return
            contents.append(types.Content(role=role, parts=parts))

        def file_uri(value: Any, context: str) -> str:
            """规范化媒体 URI，并拒绝当前端点不支持的凭证引用。"""
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Google {context} 缺少有效 URI。")
            normalized = value.strip()
            if normalized.startswith("gs://"):
                return normalized
            google_files_prefix = "https://generativelanguage.googleapis.com/v1beta/files/"
            file_name = normalized.removeprefix("files/")
            if normalized.startswith("files/") or "://" not in normalized:
                if (
                    not file_name
                    or "/" in file_name
                    or not re.fullmatch(r"[A-Za-z0-9._~-]+", file_name)
                ):
                    raise ValueError(f"Google {context} 的 file_id 无效。")
                if vertex_mode is True:
                    raise ValueError(
                        f"Google {context} 在 Vertex/Enterprise 模式不支持 Gemini Files file_id；"
                        "请使用 gs:// URI 或可公开访问的 HTTPS URI。"
                    )
                return google_files_prefix + quote(file_name, safe="-._~")
            try:
                parsed = urlsplit(normalized)
            except ValueError as exc:
                raise ValueError(f"Google {context} URI 无效。") from exc
            is_google_files_uri = (
                parsed.scheme.lower() == "https"
                and parsed.hostname
                and parsed.hostname.lower() == "generativelanguage.googleapis.com"
                and re.search(r"/files/[A-Za-z0-9._~-]+$", parsed.path)
                and not parsed.query
                and not parsed.fragment
            )
            if is_google_files_uri:
                if vertex_mode is True:
                    raise ValueError(
                        f"Google {context} 在 Vertex/Enterprise 模式不支持 Gemini Files URI；"
                        "请使用 gs:// URI 或可公开访问的 HTTPS URI。"
                    )
                return normalized
            if (
                vertex_mode is True
                and parsed.scheme.lower() == "https"
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not parsed.fragment
            ):
                # Vertex accepts externally hosted media only when the service
                # can fetch it. We cannot prove public reachability locally,
                # but reject malformed/userinfo URLs before handing them to
                # the SDK; the service remains the authority on accessibility.
                return normalized
            if vertex_mode is True:
                raise ValueError(
                    f"Google {context} 在 Vertex/Enterprise 模式仅支持 gs:// URI 或可公开访问的 HTTPS URI。"
                )
            raise ValueError(
                f"Google {context} 仅支持 data: URI、gs:// URI 或 Google Files URI，也可传 file_id。"
            )

        def media_mime_type(*sources: Any) -> str | None:
            """读取媒体块显式提供的 MIME 类型，保持 SDK 的 MIME 透传。"""
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                value = source.get(
                    "mime_type",
                    source.get("mimeType", source.get("media_type")),
                )
                if value is None:
                    continue
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Google 媒体内容的 mime_type 必须是非空字符串。")
                return value.strip()
            return None

        def file_data(value: Any, context: str, mime_type: str | None = None) -> Any:
            values: dict[str, Any] = {"file_uri": file_uri(value, context)}
            if mime_type is not None:
                values["mime_type"] = mime_type
            return types.FileData(**values)

        def function_response_value(value: Any) -> Any:
            if isinstance(value, str):
                stripped = value.strip()
                # Tool results are often plain text. Attempt JSON decoding for
                # JSON-shaped values so malformed structured results still fail
                # loudly, while preserving ordinary text as a valid output.
                numeric_json = bool(
                    stripped
                    and stripped[0] in "-0123456789"
                    and re.fullmatch(
                        r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?",
                        stripped,
                    )
                )
                looks_like_json = (
                    stripped.startswith(("{", "[", '"'))
                    or stripped in {"true", "false", "null"}
                    or numeric_json
                )
                if not looks_like_json:
                    return {"output": value}
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError("Google 工具结果的 content 不是有效 JSON。") from exc
                return dict(parsed) if isinstance(parsed, Mapping) else {"output": parsed}
            if isinstance(value, Mapping):
                return dict(value)
            return {"output": value}

        for message in normalized:
            if message.get("role") == "system":
                continue
            message_role = message.get("role")
            if message_role == "tool":
                tool_call_id = message.get("tool_call_id") or message.get("id")
                explicit_name = message.get("name") or message.get("tool_name")
                name = explicit_name
                if not name and isinstance(tool_call_id, str):
                    name = tool_call_names.get(tool_call_id)
                if not name:
                    raise ValueError("Google 工具结果缺少可关联的 tool_call_id/name。")
                raw_content = message.get("content", message.get("response", {}))
                response_parts = raw_content if isinstance(raw_content, list) else [raw_content]
                parts = []
                for response_part in response_parts:
                    if isinstance(response_part, dict) and response_part.get("type") in {
                        "tool_result",
                        "function_result",
                        "function_response",
                    }:
                        response = response_part.get("response", response_part.get("content", {}))
                        part_name = response_part.get("name") or name
                        part_id = response_part.get("tool_call_id") or tool_call_id
                    else:
                        response = response_part
                        part_name = name
                        part_id = tool_call_id
                    response_value = function_response_value(response)
                    if part_id is not None:
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=part_id,
                                    name=part_name,
                                    response=response_value,
                                )
                            )
                        )
                    else:
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=part_name,
                                    response=response_value,
                                )
                            )
                        )
                # Gemini represents parallel function results as one user turn.
                # Keep consecutive standard ``role=tool`` messages together so
                # the model receives a single FunctionResponse content block.
                pending_tool_parts.extend(parts)
                continue

            if pending_tool_parts:
                append_content("user", pending_tool_parts)
                pending_tool_parts = []

            role = "model" if message_role == "assistant" else "user"
            raw_content = message.get("content", "")
            if raw_content is None:
                parts = []
            elif isinstance(raw_content, str):
                parts = [types.Part(text=raw_content)]
            elif isinstance(raw_content, list):
                parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        parts.append(types.Part(text=part))
                        continue
                    if not isinstance(part, dict):
                        parts.append(part)
                        continue
                    part_type = part.get("type")
                    if part_type in {"text", "input_text"}:
                        parts.append(types.Part(text=part.get("text", "")))
                    elif part_type in {"image_url", "input_image"}:
                        image_value = part.get("image_url")
                        image_file_id = part.get("file_id")
                        if isinstance(image_value, Mapping):
                            image_url = image_value.get(
                                "url",
                                image_value.get("image_url", image_value.get("file_uri")),
                            )
                            image_file_id = image_file_id or image_value.get("file_id")
                        else:
                            image_url = image_value or part.get("url")
                        if image_url is None and image_file_id is not None:
                            image_url = image_file_id
                        if not isinstance(image_url, str) or not image_url.strip():
                            raise ValueError("Google 图片内容缺少有效 URI。")
                        image_url = image_url.strip()
                        if isinstance(image_url, str) and image_url.startswith("data:"):
                            mime_type, data = GoogleProvider._decode_data_uri(image_url, "图片")
                            parts.append(
                                types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))
                            )
                        else:
                            parts.append(
                                types.Part(
                                    file_data=file_data(
                                        image_url,
                                        "图片",
                                        media_mime_type(part, image_value),
                                    )
                                )
                            )
                    elif part_type in {
                        "audio_url",
                        "input_audio",
                        "video_url",
                        "input_video",
                        "file",
                    }:
                        media_value = part.get(
                            "audio_url",
                            part.get("video_url", part.get("url", part.get("file_uri"))),
                        )
                        media_file_id = part.get("file_id")
                        if part_type == "file" and media_value is None:
                            media_value = part.get("file")
                        if isinstance(media_value, Mapping):
                            media = media_value.get(
                                "url",
                                media_value.get(
                                    "file_uri",
                                    media_value.get("audio_url", media_value.get("video_url")),
                                ),
                            )
                            media_file_id = media_file_id or media_value.get("file_id")
                        else:
                            media = media_value
                        if media is None and media_file_id is not None:
                            media = media_file_id
                        if not isinstance(media, str) or not media.strip():
                            raise ValueError("Google 媒体内容缺少有效 URI。")
                        media = media.strip()
                        if isinstance(media, str) and media.startswith("data:"):
                            mime_type, data = GoogleProvider._decode_data_uri(media, "媒体")
                            parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type=mime_type,
                                        data=data,
                                    )
                                )
                            )
                        else:
                            parts.append(
                                types.Part(
                                    file_data=file_data(
                                        media,
                                        "媒体",
                                        media_mime_type(part, media_value),
                                    )
                                )
                            )
                    elif part_type in {"tool_result", "function_result"}:
                        part_id = part.get("tool_call_id") or part.get("id")
                        part_name = part.get("name") or part.get("tool_name")
                        if not part_name and isinstance(part_id, str):
                            part_name = tool_call_names.get(part_id)
                        if not isinstance(part_name, str) or not part_name.strip():
                            raise ValueError("Google 工具结果缺少可关联的 tool_call_id/name。")
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=part_id,
                                    name=part_name,
                                    response=function_response_value(
                                        part.get("response", part.get("content", {}))
                                    ),
                                )
                            )
                        )
                    elif part_type == "function_call":
                        call_id = part.get("id") or part.get("call_id")
                        function_name = part.get("name")
                        if not isinstance(function_name, str) or not function_name.strip():
                            raise ValueError("Google function_call 缺少 name。")
                        if call_id and function_name:
                            tool_call_names[call_id] = function_name
                        arguments = part.get("arguments", part.get("args", {}))
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError as exc:
                                raise ValueError(
                                    "Google function_call 的 arguments 不是有效 JSON。"
                                ) from exc
                        if not isinstance(arguments, Mapping):
                            raise ValueError("Google function_call 的 arguments 必须是 JSON 对象。")
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=call_id, name=function_name, args=dict(arguments)
                                )
                            )
                        )
                    elif part_type == "function_response":
                        part_id = part.get("tool_call_id") or part.get("id")
                        part_name = part.get("name") or part.get("tool_name")
                        if not part_name and isinstance(part_id, str):
                            part_name = tool_call_names.get(part_id)
                        if not isinstance(part_name, str) or not part_name.strip():
                            raise ValueError(
                                "Google function_response 缺少可关联的 tool_call_id/name。"
                            )
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=part_id,
                                    name=part_name,
                                    response=function_response_value(part.get("response", {})),
                                )
                            )
                        )
                    else:
                        raise ValueError(f"Google 不支持的消息内容块类型: {part_type!r}。")
            else:
                parts = [types.Part(text=str(raw_content))]
            for index, call in enumerate(message.get("tool_calls", []) or []):
                if not isinstance(call, Mapping):
                    raise ValueError(f"Google assistant tool_calls[{index}] 必须是对象。")
                function = call.get("function", call)
                if not isinstance(function, Mapping):
                    raise ValueError(f"Google assistant tool_calls[{index}].function 必须是对象。")
                call_id = call.get("id") or call.get("call_id")
                function_name = function.get("name", "")
                if not isinstance(function_name, str) or not function_name.strip():
                    raise ValueError(f"Google assistant tool_calls[{index}] 缺少 function.name。")
                if call_id and function_name:
                    tool_call_names[call_id] = function_name
                arguments = function.get("arguments", function.get("args", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Google assistant tool_calls[{index}] 的 arguments 不是有效 JSON。"
                        ) from exc
                if not isinstance(arguments, Mapping):
                    raise ValueError("Google assistant tool call 的 arguments 必须是 JSON 对象。")
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id, name=function.get("name", ""), args=dict(arguments)
                        )
                    )
                )
            append_content(role, parts)
        if pending_tool_parts:
            append_content("user", pending_tool_parts)
        return contents

    @staticmethod
    def _response_text(response: Any) -> str:
        """读取文本而不触发 Google SDK 纯工具响应的 ValueError。"""
        candidates = field(response, "candidates", []) or []
        if candidates:
            chunks: list[str] = []
            saw_parts = False
            for candidate in candidates:
                content = field(candidate, "content")
                for part in field(content, "parts", []) or []:
                    saw_parts = True
                    if field(part, "thought", False):
                        continue
                    text = field(part, "text", "")
                    if isinstance(text, str):
                        chunks.append(text)
            if saw_parts:
                return "".join(chunks)
        try:
            value = field(response, "text", "")
        except Exception:  # noqa: BLE001 - SDK response accessors are dynamic
            value = ""
        if isinstance(value, str):
            return value
        chunks = []
        for candidate in candidates:
            content = field(candidate, "content")
            for part in field(content, "parts", []) or []:
                if field(part, "thought", False):
                    continue
                text = field(part, "text", "")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    _REFUSAL_FINISH_REASONS = frozenset(
        {
            "SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "RECITATION",
            "IMAGE_SAFETY",
            "MODEL_ARMOR",
        }
    )

    @classmethod
    def _candidate_refusal(cls, candidate: Any) -> str | None:
        """读取 Gemini 内容块或安全拦截元数据中的拒答原因。"""
        content = field(candidate, "content")
        for part in field(content, "parts", []) or []:
            value = field(part, "refusal")
            if not value and field(part, "type") == "refusal":
                value = field(part, "text") or field(part, "reason")
            if isinstance(value, str) and value:
                return value

        for key in ("finish_message", "block_reason_message"):
            value = field(candidate, key)
            if isinstance(value, str) and value:
                return value

        finish_reason = field(candidate, "finish_reason")
        normalized_reason = cls._normalize_finish_reason(finish_reason) or ""
        if normalized_reason in cls._REFUSAL_FINISH_REASONS:
            return f"Google 内容被安全策略拦截: {normalized_reason}"
        return None

    @staticmethod
    def _normalize_finish_reason(value: Any) -> str | None:
        """将 google-genai 枚举或原始值统一为协议中的字符串。"""
        if value is None:
            return None
        name = getattr(value, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip().upper()
        raw = getattr(value, "value", value)
        text = str(raw).strip()
        if not text:
            return None
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.upper()

    @classmethod
    def _extract_refusal(cls, response: Any) -> str | None:
        """提取 Gemini 响应中的显式拒答或安全拦截信息。"""
        for candidate in field(response, "candidates", []) or []:
            refusal = cls._candidate_refusal(candidate)
            if refusal:
                return refusal
        feedback = field(response, "prompt_feedback")
        block_reason = field(feedback, "block_reason")
        if isinstance(block_reason, Enum):
            block_reason_text = block_reason.name or str(block_reason)
        elif isinstance(block_reason, str):
            block_reason_text = block_reason.strip()
        else:
            block_reason_text = ""
        if block_reason_text:
            message = field(feedback, "block_reason_message")
            if not isinstance(message, str) or not message.strip():
                message = field(feedback, "message")
            return str(message or f"Google 提示被安全策略拦截: {block_reason_text}")
        return None

    @staticmethod
    def _partial_arg_value(partial: Any) -> Any:
        """读取 Vertex ``PartialArg`` 的标量值。"""
        for name in ("bool_value", "number_value", "string_value"):
            value = field(partial, name)
            if value is not None:
                return value
        if field(partial, "null_value") is not None:
            return None
        raise ValueError("Google FunctionCall.partial_args 缺少有效值。")

    @staticmethod
    def _partial_arg_tokens(json_path: str) -> list[str | int]:
        """解析 Vertex PartialArg 使用的常见 JSONPath 子集。"""
        if not isinstance(json_path, str) or not json_path.startswith("$"):
            raise ValueError(f"Google FunctionCall.partial_args 的 json_path 无效: {json_path!r}")
        if json_path in {"", "$"}:
            return []
        rest = json_path[1:]
        tokens: list[str | int] = []
        while rest:
            dotted = re.match(r"^\.([A-Za-z_][A-Za-z0-9_]*)", rest)
            if dotted:
                tokens.append(dotted.group(1))
                rest = rest[dotted.end() :]
                continue
            indexed = re.match(r"^\[(\d+)\]", rest)
            if indexed:
                tokens.append(int(indexed.group(1)))
                rest = rest[indexed.end() :]
                continue
            quoted = re.match(r"^\[['\"]([^'\"]+)['\"]\]", rest)
            if quoted:
                tokens.append(quoted.group(1))
                rest = rest[quoted.end() :]
                continue
            raise ValueError(f"Google FunctionCall.partial_args 不支持的 json_path: {json_path!r}")
        return tokens

    @classmethod
    def _partial_args_to_mapping(cls, partial_args: Any) -> dict[str, Any]:
        """将 Vertex 的 PartialArg 列表还原为普通 JSON 对象。"""
        if isinstance(partial_args, Mapping):
            return dict(partial_args)
        if not isinstance(partial_args, (list, tuple)):
            raise ValueError("Google FunctionCall.partial_args 必须是列表。")
        result: dict[str, Any] = {}
        for partial in partial_args:
            json_path = field(partial, "json_path")
            tokens = cls._partial_arg_tokens(json_path or "$")
            value = cls._partial_arg_value(partial)
            if not tokens:
                if not isinstance(value, Mapping):
                    raise ValueError("Google FunctionCall.partial_args 的根值必须是 JSON 对象。")
                result.update(dict(value))
                continue
            current: Any = result
            for position, token in enumerate(tokens):
                last = position == len(tokens) - 1
                if isinstance(token, int):
                    if not isinstance(current, list) or token < 0:
                        raise ValueError(
                            f"Google FunctionCall.partial_args 的 json_path 无法写入: {json_path!r}"
                        )
                    while len(current) <= token:
                        current.append(None)
                    if last:
                        current[token] = value
                    elif current[token] is None:
                        current[token] = [] if isinstance(tokens[position + 1], int) else {}
                    current = current[token]
                else:
                    if not isinstance(current, dict):
                        raise ValueError(
                            f"Google FunctionCall.partial_args 的 json_path 无法写入: {json_path!r}"
                        )
                    if last:
                        current[token] = value
                    else:
                        current.setdefault(
                            token,
                            [] if isinstance(tokens[position + 1], int) else {},
                        )
                        current = current[token]
        return result

    @classmethod
    def _function_call_arguments(cls, function_call: Any) -> Any:
        """统一读取完整 args 与 Vertex 流式 partial_args。"""
        arguments = field(function_call, "args")
        if arguments is not None:
            return arguments
        partial_args = field(function_call, "partial_args")
        if partial_args is not None:
            return cls._partial_args_to_mapping(partial_args)
        return {}

    @staticmethod
    def _function_call_index(function_call: Any) -> Any:
        """读取仅由兼容字典载荷提供的工具调用 index。

        ``google-genai`` 的 ``types.FunctionCall`` 没有 index 字段；并行
        调用的 index 不能用 Candidate.index 代替，因为后者标识候选答案，
        不是候选内部的工具调用。保留字典读取仅用于中转网关的扩展载荷。
        """
        return function_call.get("index") if isinstance(function_call, Mapping) else None

    @classmethod
    def _merge_tool_arguments(cls, existing: Any, incoming: Any) -> Any:
        """递归合并工具参数分片，保留不同分片中的嵌套字段。"""
        if isinstance(existing, Mapping) and isinstance(incoming, Mapping):
            merged_mapping: dict[Any, Any] = dict(existing)
            for key, value in incoming.items():
                merged_mapping[key] = (
                    cls._merge_tool_arguments(merged_mapping[key], value)
                    if key in merged_mapping
                    else value
                )
            return merged_mapping
        if isinstance(existing, list) and isinstance(incoming, list):
            merged_list: list[Any] = list(existing)
            for index, value in enumerate(incoming):
                if index >= len(merged_list):
                    merged_list.extend([None] * (index + 1 - len(merged_list)))
                if value is not None:
                    merged_list[index] = (
                        cls._merge_tool_arguments(merged_list[index], value)
                        if merged_list[index] is not None
                        else value
                    )
            return merged_list
        return incoming

    @classmethod
    def _merge_tool_call(
        cls,
        accumulator: dict[Any, dict[str, Any]],
        tool_call: Mapping[str, Any],
        fallback_index: Any = None,
        *,
        merge_arguments: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """累积 Gemini FunctionCall 分片，兼容字典参数与字符串参数。

        ``tool_call_completed`` 有时会紧跟同一个 chunk 的 delta，并重复携带
        完整 arguments。完成事件只需要复用已累积的元数据；调用方可将
        ``merge_arguments`` 设为 ``False``，避免字符串参数被追加两次。
        """
        call_id = tool_call.get("id") or tool_call.get("call_id")
        index = tool_call.get("index", fallback_index)
        name = tool_call.get("name")

        def has_identity(candidate: Mapping[str, Any]) -> bool:
            return bool(candidate.get("id") or candidate.get("call_id"))

        def candidate_matches_id(candidate: Mapping[str, Any]) -> bool:
            candidate_id = candidate.get("id") or candidate.get("call_id")
            return bool(call_id and candidate_id == call_id)

        def name_matches(candidate: Mapping[str, Any]) -> bool:
            candidate_name = candidate.get("name")
            return name is None or candidate_name in {None, name}

        def move_candidate(candidate_key: Any, target_key: Any) -> Any:
            if candidate_key == target_key:
                return target_key
            if target_key in accumulator:
                raise RuntimeError("Google Gemini 工具调用分片的 id/index 指向多个调用。")
            accumulator[target_key] = accumulator.pop(candidate_key)
            return target_key

        # google-genai's FunctionCall does not expose an index in the current
        # SDK.  Never synthesize one from a part position: that position resets
        # in every response chunk and would merge independent parallel calls.
        # An explicit index is the most stable stream identity.  If a later
        # fragment supplies an id, attach it to the existing indexed record
        # instead of moving the accumulated arguments to a new id key.
        if index is not None:
            candidates = [
                candidate_key
                for candidate_key, candidate in accumulator.items()
                if candidate.get("index") == index and name_matches(candidate)
            ]
            if len(candidates) > 1:
                raise RuntimeError("Google Gemini 工具调用分片的 index 指向多个调用。")
            if candidates:
                key = candidates[0]
            else:
                # A provider may emit the id first and the index only on a
                # later fragment. Match that id across the whole accumulator
                # before falling back to a name-only, index-less record.
                id_candidates = [
                    candidate_key
                    for candidate_key, candidate in accumulator.items()
                    if candidate_matches_id(candidate)
                ]
                if len(id_candidates) > 1:
                    raise RuntimeError("Google Gemini 工具调用分片的 id 指向多个调用。")
                if id_candidates:
                    key = move_candidate(id_candidates[0], ("index", index))
                else:
                    candidates = [
                        candidate_key
                        for candidate_key, candidate in accumulator.items()
                        if candidate.get("index") is None and name_matches(candidate)
                    ]
                    if len(candidates) > 1:
                        raise RuntimeError(
                            "Google Gemini 工具调用分片缺少稳定 id/index，无法关联并行调用。"
                        )
                    if len(candidates) == 1:
                        key = move_candidate(candidates[0], ("index", index))
                    else:
                        key = ("index", index)
        elif call_id:
            exact_key = ("id", call_id)
            id_candidates = [
                candidate_key
                for candidate_key, candidate in accumulator.items()
                if candidate_matches_id(candidate)
            ]
            if len(id_candidates) > 1:
                raise RuntimeError("Google Gemini 工具调用分片的 id 指向多个调用。")
            if id_candidates:
                # Keep an existing explicit index as the accumulator key. This
                # handles streams where the first fragment has both id/index
                # and later fragments repeat only the id.
                key = id_candidates[0]
            else:
                candidates = [
                    candidate_key
                    for candidate_key, candidate in accumulator.items()
                    if not has_identity(candidate) and name_matches(candidate)
                ]
                if len(candidates) > 1:
                    raise RuntimeError(
                        "Google Gemini 工具调用分片缺少稳定 id/index，无法关联并行调用。"
                    )
                if len(candidates) == 1:
                    # Preserve an existing explicit index.  A position-only
                    # record has no stronger identity, so promote it to id.
                    candidate_key = candidates[0]
                    candidate = accumulator[candidate_key]
                    key = (
                        candidate_key
                        if candidate.get("index") is not None
                        else move_candidate(candidate_key, exact_key)
                    )
                else:
                    key = exact_key
        else:
            candidates = [
                candidate_key
                for candidate_key, candidate in accumulator.items()
                if name_matches(candidate)
            ]
            if len(candidates) > 1:
                raise RuntimeError(
                    "Google Gemini 工具调用分片缺少稳定 id/index，无法关联并行调用。"
                )
            if candidates:
                candidate = accumulator[candidates[0]]
                if (
                    has_identity(candidate)
                    or tool_call.get("will_continue") is True
                    or candidate.get("will_continue") is True
                ):
                    key = candidates[0]
                else:
                    raise RuntimeError(
                        "Google Gemini 工具调用分片缺少稳定 id/index，无法确认是否为同一调用。"
                    )
            else:
                position = len(accumulator)
                key = ("position", position)
                while key in accumulator:
                    position += 1
                    key = ("position", position)
        merged = accumulator.setdefault(
            key,
            {"type": "function", "arguments": {}},
        )
        existing_id = merged.get("id") or merged.get("call_id")
        if call_id and existing_id and existing_id != call_id:
            raise RuntimeError("Google Gemini 工具调用分片的 id 指向多个调用。")
        existing_index = merged.get("index")
        if index is not None and existing_index is not None and existing_index != index:
            raise RuntimeError("Google Gemini 工具调用分片的 index 指向多个调用。")
        existing_name = merged.get("name")
        if name and existing_name and existing_name != name:
            raise RuntimeError("Google Gemini 工具调用分片的 name 指向多个调用。")
        for name in ("id", "call_id", "type", "name", "index", "will_continue"):
            value = tool_call.get(name)
            if value is not None and value != "":
                merged[name] = value
        if merge_arguments:
            arguments = tool_call.get("arguments")
            if isinstance(arguments, Mapping):
                existing = merged.get("arguments")
                if not isinstance(existing, Mapping):
                    existing = {}
                merged["arguments"] = cls._merge_tool_arguments(existing, arguments)
            elif isinstance(arguments, str) and arguments:
                existing = merged.get("arguments")
                merged["arguments"] = (existing if isinstance(existing, str) else "") + arguments
            elif arguments is not None:
                merged["arguments"] = arguments
        return key, merged

    @staticmethod
    def _complete_stream_tool_calls(
        accumulator: dict[Any, dict[str, Any]],
        completed: set[Any],
        response_id: str | None,
        raw: Any,
    ) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for key, tool_call in accumulator.items():
            if key in completed:
                continue
            validate_complete_tool_call(tool_call, "Google Gemini")
            completed.add(key)
            events.append(
                StreamEvent(
                    type="tool_call_completed",
                    tool_call=dict(tool_call),
                    response_id=response_id,
                    raw=raw,
                )
            )
        return events

    @staticmethod
    def _extract_result(response: Any) -> CompletionResult:
        usage = field(response, "usage_metadata") or field(response, "usage")
        usage_dict = normalize_usage(usage)
        calls = []
        reasoning: list[str] = []
        for candidate in field(response, "candidates", []) or []:
            content = field(candidate, "content")
            for part in field(content, "parts", []) or []:
                function_call = field(part, "function_call")
                if function_call:
                    calls.append(
                        {
                            "id": field(function_call, "id"),
                            "type": "function",
                            "name": field(function_call, "name"),
                            "arguments": GoogleProvider._function_call_arguments(function_call),
                        }
                    )
                if field(part, "thought") and isinstance(field(part, "text"), str):
                    reasoning.append(field(part, "text"))
        return CompletionResult(
            text=GoogleProvider._response_text(response),
            tool_calls=calls,
            usage=usage_dict,
            finish_reason=GoogleProvider._normalize_finish_reason(
                field((field(response, "candidates", []) or [None])[0], "finish_reason")
            ),
            response_id=field(response, "response_id"),
            refusal=GoogleProvider._extract_refusal(response),
            reasoning="".join(reasoning),
            raw=response,
        )

    @classmethod
    def _stream_events_from_chunk(cls, chunk: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        response_id = field(chunk, "response_id")
        for candidate in field(chunk, "candidates", []) or []:
            content = field(candidate, "content")
            parts = field(content, "parts", []) or []
            refusal_emitted = False
            for part in parts:
                function_call = field(part, "function_call")
                if function_call:
                    tool_call = {
                        "id": field(function_call, "id"),
                        "index": cls._function_call_index(function_call),
                        "type": "function",
                        "name": field(function_call, "name"),
                        "arguments": cls._function_call_arguments(function_call),
                    }
                    events.append(
                        StreamEvent(
                            type="tool_call_delta",
                            tool_call=tool_call,
                            response_id=response_id,
                            raw=chunk,
                        )
                    )
                    # Without id/index there is no safe way to distinguish an
                    # immediate completion event from a duplicate delta.  The
                    # stream accumulator emits the completed call at finish,
                    # where the same ambiguity is checked explicitly.
                    if field(function_call, "will_continue") is False and (
                        field(function_call, "id")
                        or field(function_call, "call_id")
                        or cls._function_call_index(function_call) is not None
                    ):
                        events.append(
                            StreamEvent(
                                type="tool_call_completed",
                                tool_call=tool_call,
                                response_id=response_id,
                                raw=chunk,
                            )
                        )

                part_type = field(part, "type")
                part_refusal = field(part, "refusal")
                if not part_refusal and part_type == "refusal":
                    part_refusal = field(part, "text") or field(part, "reason")
                if isinstance(part_refusal, str) and part_refusal:
                    events.append(
                        StreamEvent(
                            type="refusal_delta",
                            refusal=part_refusal,
                            response_id=response_id,
                            raw=chunk,
                        )
                    )
                    refusal_emitted = True
                    continue

                text = field(part, "text")
                if isinstance(text, str) and text:
                    if field(part, "thought"):
                        events.append(
                            StreamEvent(
                                type="reasoning_delta",
                                reasoning=text,
                                response_id=response_id,
                                raw=chunk,
                            )
                        )
                    else:
                        events.append(
                            StreamEvent(
                                type="text_delta",
                                text=text,
                                response_id=response_id,
                                raw=chunk,
                            )
                        )

            candidate_refusal = cls._candidate_refusal(candidate)
            if candidate_refusal and not refusal_emitted:
                events.append(
                    StreamEvent(
                        type="refusal_delta",
                        refusal=candidate_refusal,
                        response_id=response_id,
                        raw=chunk,
                    )
                )
            finish_reason = field(candidate, "finish_reason")
            if finish_reason:
                events.append(
                    StreamEvent(
                        type="finish",
                        finish_reason=cls._normalize_finish_reason(finish_reason),
                        response_id=response_id,
                        raw=chunk,
                    )
                )

        # Safety blocks can arrive as ``prompt_feedback`` with no candidates.
        # Treat that metadata as an explicit refusal and terminal event instead
        # of allowing an empty stream to look like a successful response.
        feedback = field(chunk, "prompt_feedback")
        block_reason = field(feedback, "block_reason")
        if block_reason:
            message = field(feedback, "block_reason_message") or field(feedback, "message")
            refusal = str(message or f"Google 提示被安全策略拦截: {block_reason}")
            if not any(event.type == "refusal_delta" for event in events):
                events.append(
                    StreamEvent(
                        type="refusal_delta",
                        refusal=refusal,
                        response_id=response_id,
                        raw=chunk,
                    )
                )
            if not any(event.type == "finish" for event in events):
                events.append(
                    StreamEvent(
                        type="finish",
                        finish_reason=cls._normalize_finish_reason(block_reason),
                        response_id=response_id,
                        raw=chunk,
                    )
                )
        top_level_finish_reason = field(chunk, "finish_reason")
        if top_level_finish_reason is not None and not any(
            event.type == "finish" for event in events
        ):
            events.append(
                StreamEvent(
                    type="finish",
                    finish_reason=cls._normalize_finish_reason(top_level_finish_reason),
                    response_id=response_id,
                    raw=chunk,
                )
            )
        usage = normalize_usage(field(chunk, "usage_metadata") or field(chunk, "usage"))
        if usage:
            events.append(
                StreamEvent(type="usage", usage=usage, response_id=response_id, raw=chunk)
            )
        return events

    @staticmethod
    def _stream_chunk_has_terminal_event(chunk: Any) -> bool:
        """判断 Gemini 流块是否明确表示本次生成已经终止。"""
        if field(chunk, "finish_reason", None) is not None:
            return True
        feedback = field(chunk, "prompt_feedback")
        if field(feedback, "block_reason", None) is not None:
            return True
        return any(
            field(candidate, "finish_reason", None) is not None
            for candidate in field(chunk, "candidates", []) or []
        )

    def complete(self, request: CompletionRequest | None = None, **kwargs: Any) -> CompletionResult:
        request = coerce_completion_request(request, kwargs, "Google complete")
        self._validate_request(request)
        system_prompt = extract_system_prompt(request.messages) or request.system_prompt
        config = self._build_generation_config(
            system_prompt,
            request.tools,
            request.temperature if request.temperature_is_explicit else None,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            candidate_count=request.n,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            modalities=request.modalities,
            stop=request.stop,
            response_format=request.response_format,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
            extra_headers=request.extra_headers,
            extra_query=request.extra_query,
            thinking=request.thinking,
            service_tier=request.service_tier,
            timeout=request.timeout,
        )
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        response = retry_sync_call(
            lambda: client.models.generate_content(
                model=self._model_name,
                contents=self._contents(
                    request.messages,
                    request.prompt,
                    request.system_prompt,
                    vertex_mode=vertexai,
                ),
                config=config,
            )
        )
        return self._extract_result(response)

    async def acomplete(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> CompletionResult:
        """使用 google-genai 异步模型接口聚合完整结果。"""
        request = coerce_completion_request(request, kwargs, "Google acomplete")
        self._validate_request(request)
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        config = self._build_generation_config(
            effective_system,
            request.tools,
            request.temperature if request.temperature_is_explicit else None,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            candidate_count=request.n,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            modalities=request.modalities,
            stop=request.stop,
            response_format=request.response_format,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
            extra_headers=request.extra_headers,
            extra_query=request.extra_query,
            thinking=request.thinking,
            service_tier=request.service_tier,
            timeout=request.timeout,
        )
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        response = await retry_async_call(
            lambda: client.aio.models.generate_content(
                model=self._model_name,
                contents=self._contents(
                    request.messages,
                    request.prompt,
                    request.system_prompt,
                    vertex_mode=vertexai,
                ),
                config=config,
            )
        )
        return self._extract_result(response)

    def count_tokens(self, request: CompletionRequest | None = None, **kwargs: Any) -> int:
        request = coerce_completion_request(request, kwargs, "Google count_tokens")
        self._validate_token_count_fields(request)
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        config_values: dict[str, Any] = {}
        # ``CompletionRequest`` keeps the legacy chat default system prompt for
        # normal generation. It must not silently become an explicit
        # ``system_instruction`` for a plain Developer API token-count call,
        # where that field is unsupported. A caller-provided message-level
        # system prompt remains explicit and is still validated below.
        configured_system = self._token_count_system_prompt(request)
        effective_system = configured_system
        if vertexai is False and effective_system:
            raise ValueError(
                "Gemini Developer API 的 count_tokens 不支持 system_prompt；"
                "请设置 system_prompt=None，或改用 Vertex/Enterprise 模式。"
            )
        if vertexai is False and request.tools:
            raise ValueError(
                "Gemini Developer API 的 count_tokens 不支持 tools；请改用 Vertex/Enterprise 模式。"
            )
        if vertexai is not False and effective_system:
            config_values["system_instruction"] = effective_system
        if vertexai is not False and request.tools:
            config_values["tools"] = self._convert_tools(request.tools)
        http_options = self._http_options_for_request(
            extra_headers=request.extra_headers,
            extra_body=request.extra_body,
            extra_query=request.extra_query,
            timeout=request.timeout,
        )
        if http_options is not None:
            config_values["http_options"] = http_options
        contents = self._contents(
            request.messages,
            request.prompt,
            None if vertexai is False else configured_system,
            vertex_mode=vertexai,
        )
        response = retry_sync_call(
            lambda: client.models.count_tokens(
                model=self._model_name,
                contents=contents,
                config=types.CountTokensConfig(**config_values) if config_values else None,
            )
        )
        value = field(response, "total_tokens")
        if value is None:
            raise RuntimeError("Google token count 响应缺少 total_tokens。")
        return int(value)

    def compute_tokens(self, request: CompletionRequest | None = None, **kwargs: Any) -> Any:
        request = coerce_completion_request(request, kwargs, "Google compute_tokens")
        self._validate_compute_token_fields(request)
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        if vertexai is False:
            raise ValueError(
                "Google compute_tokens 仅支持 Vertex/Enterprise 模式；"
                "Gemini Developer API 不支持该能力。"
            )
        effective_system = self._token_count_system_prompt(request)
        if vertexai is True and effective_system:
            raise ValueError(
                "Google compute_tokens 不支持 system_prompt；请在 contents 中显式传入要统计的文本。"
            )
        if vertexai is True and request.tools:
            raise ValueError(
                "Google compute_tokens 不支持 tools；请在 contents 中显式传入要统计的内容。"
            )
        http_options = self._http_options_for_request(
            extra_headers=request.extra_headers,
            extra_body=request.extra_body,
            extra_query=request.extra_query,
            timeout=request.timeout,
        )
        config = (
            types.ComputeTokensConfig(http_options=http_options)
            if http_options is not None
            else None
        )
        contents = self._contents(
            request.messages,
            request.prompt,
            None,
            vertex_mode=vertexai,
        )
        return retry_sync_call(
            lambda: client.models.compute_tokens(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        )

    def upload_file(self, file: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().files.upload(
            file=file,
            **self._resource_config_kwargs(types.UploadFileConfig, config, kwargs),
        )

    @staticmethod
    def _validate_register_files_auth(auth: Any) -> Credentials:
        """限制文件注册凭证为 google-auth 的官方 Credentials 对象。"""
        if not isinstance(auth, Credentials):
            raise ValueError(
                "Google register_files auth 必须是 google.auth.credentials.Credentials 对象。"
            )
        return auth

    def register_files(self, auth: Any, uris: list[str], config: Any = None, **kwargs: Any) -> Any:
        auth = self._validate_register_files_auth(auth)
        return self._get_client().files.register_files(
            auth=auth,
            uris=uris,
            **self._resource_config_kwargs(types.RegisterFilesConfig, config, kwargs),
        )

    def get_file(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().files.get(
            name=name,
            **self._resource_config_kwargs(types.GetFileConfig, config, kwargs),
        )

    def list_files(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().files.list(
            **self._resource_config_kwargs(types.ListFilesConfig, config, kwargs),
        )

    def download_file(self, file: Any, config: Any = None, **kwargs: Any) -> bytes:
        return self._get_client().files.download(
            file=file,
            **self._resource_config_kwargs(types.DownloadFileConfig, config, kwargs),
        )

    def delete_file(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().files.delete(
            name=name,
            **self._resource_config_kwargs(types.DeleteFileConfig, config, kwargs),
        )

    def list_file_search_documents(self, parent: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.documents.list(
            parent=parent,
            **self._resource_config_kwargs(types.ListDocumentsConfig, config, kwargs),
        )

    def get_file_search_document(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.documents.get(
            name=name,
            **self._resource_config_kwargs(types.GetDocumentConfig, config, kwargs),
        )

    def delete_file_search_document(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.documents.delete(
            name=name,
            **self._resource_config_kwargs(types.DeleteDocumentConfig, config, kwargs),
        )

    async def async_upload_file(self, file: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.files.upload(
                file=file,
                **self._resource_config_kwargs(types.UploadFileConfig, config, kwargs),
            )
        )

    async def async_register_files(
        self, auth: Any, uris: list[str], config: Any = None, **kwargs: Any
    ) -> Any:
        auth = self._validate_register_files_auth(auth)
        return await self._resolve_async_result(
            self._get_client().aio.files.register_files(
                auth=auth,
                uris=uris,
                **self._resource_config_kwargs(types.RegisterFilesConfig, config, kwargs),
            )
        )

    async def async_get_file(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.files.get(
                name=name,
                **self._resource_config_kwargs(types.GetFileConfig, config, kwargs),
            )
        )

    async def async_list_files(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.files.list(
                **self._resource_config_kwargs(types.ListFilesConfig, config, kwargs),
            )
        )

    async def async_download_file(self, file: Any, config: Any = None, **kwargs: Any) -> bytes:
        return await self._resolve_async_result(
            self._get_client().aio.files.download(
                file=file,
                **self._resource_config_kwargs(types.DownloadFileConfig, config, kwargs),
            )
        )

    async def async_delete_file(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.files.delete(
                name=name,
                **self._resource_config_kwargs(types.DeleteFileConfig, config, kwargs),
            )
        )

    async def async_list_file_search_documents(
        self, parent: str, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.documents.list(
                parent=parent,
                **self._resource_config_kwargs(types.ListDocumentsConfig, config, kwargs),
            )
        )

    async def async_get_file_search_document(
        self, name: str, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.documents.get(
                name=name,
                **self._resource_config_kwargs(types.GetDocumentConfig, config, kwargs),
            )
        )

    async def async_delete_file_search_document(
        self, name: str, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.documents.delete(
                name=name,
                **self._resource_config_kwargs(types.DeleteDocumentConfig, config, kwargs),
            )
        )

    async def async_count_tokens(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> int:
        request = coerce_completion_request(request, kwargs, "Google async_count_tokens")
        self._validate_token_count_fields(request)
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        config_values: dict[str, Any] = {}
        configured_system = self._token_count_system_prompt(request)
        effective_system = configured_system
        if vertexai is False and effective_system:
            raise ValueError(
                "Gemini Developer API 的 count_tokens 不支持 system_prompt；"
                "请设置 system_prompt=None，或改用 Vertex/Enterprise 模式。"
            )
        if vertexai is False and request.tools:
            raise ValueError(
                "Gemini Developer API 的 count_tokens 不支持 tools；请改用 Vertex/Enterprise 模式。"
            )
        if vertexai is not False and effective_system:
            config_values["system_instruction"] = effective_system
        if vertexai is not False and request.tools:
            config_values["tools"] = self._convert_tools(request.tools)
        http_options = self._http_options_for_request(
            extra_headers=request.extra_headers,
            extra_body=request.extra_body,
            extra_query=request.extra_query,
            timeout=request.timeout,
        )
        if http_options is not None:
            config_values["http_options"] = http_options
        contents = self._contents(
            request.messages,
            request.prompt,
            None if vertexai is False else configured_system,
            vertex_mode=vertexai,
        )
        response = await retry_async_call(
            lambda: client.aio.models.count_tokens(
                model=self._model_name,
                contents=contents,
                config=types.CountTokensConfig(**config_values) if config_values else None,
            )
        )
        value = field(response, "total_tokens")
        if value is None:
            raise RuntimeError("Google token count 响应缺少 total_tokens。")
        return int(value)

    async def async_compute_tokens(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Any:
        request = coerce_completion_request(request, kwargs, "Google async_compute_tokens")
        self._validate_compute_token_fields(request)
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        if vertexai is False:
            raise ValueError(
                "Google compute_tokens 仅支持 Vertex/Enterprise 模式；"
                "Gemini Developer API 不支持该能力。"
            )
        effective_system = self._token_count_system_prompt(request)
        if vertexai is True and effective_system:
            raise ValueError(
                "Google compute_tokens 不支持 system_prompt；请在 contents 中显式传入要统计的文本。"
            )
        if vertexai is True and request.tools:
            raise ValueError(
                "Google compute_tokens 不支持 tools；请在 contents 中显式传入要统计的内容。"
            )
        http_options = self._http_options_for_request(
            extra_headers=request.extra_headers,
            extra_body=request.extra_body,
            extra_query=request.extra_query,
            timeout=request.timeout,
        )
        config = (
            types.ComputeTokensConfig(http_options=http_options)
            if http_options is not None
            else None
        )
        contents = self._contents(
            request.messages,
            request.prompt,
            None,
            vertex_mode=vertexai,
        )
        return await retry_async_call(
            lambda: client.aio.models.compute_tokens(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        )

    def stream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> Generator[StreamEvent, None, None]:
        request = coerce_completion_request(request, kwargs, "Google stream_events")
        request = request.copy_with(stream=True)
        self._validate_request(request)
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        config = self._build_generation_config(
            effective_system,
            request.tools,
            request.temperature if request.temperature_is_explicit else None,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            candidate_count=request.n,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            modalities=request.modalities,
            stop=request.stop,
            response_format=request.response_format,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
            extra_headers=request.extra_headers,
            extra_query=request.extra_query,
            thinking=request.thinking,
            service_tier=request.service_tier,
            timeout=request.timeout,
        )
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        contents = self._contents(
            request.messages,
            request.prompt,
            request.system_prompt,
            vertex_mode=vertexai,
        )
        if request.stream:
            response = retry_sync_stream(
                lambda: client.models.generate_content_stream(
                    model=self._model_name, contents=contents, config=config
                ),
                first_event_validator=lambda event: raise_for_stream_error_event(
                    event, "Google Gemini"
                ),
            )
            tool_calls: dict[Any, dict[str, Any]] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            for chunk in response:
                for event in self._stream_events_from_chunk(chunk):
                    if event.type == "finish":
                        terminal_seen = True
                        yield from self._complete_stream_tool_calls(
                            tool_calls,
                            completed_tool_calls,
                            event.response_id,
                            event.raw,
                        )
                        yield event
                        continue
                    if event.type == "tool_call_delta" and event.tool_call:
                        key, merged = self._merge_tool_call(tool_calls, event.tool_call)
                        event.tool_call = dict(merged)
                        yield event
                        continue
                    if event.type == "tool_call_completed" and event.tool_call:
                        previous_entries = tuple(tool_calls.values())
                        key, merged = self._merge_tool_call(
                            tool_calls,
                            event.tool_call,
                            merge_arguments=False,
                        )
                        if not any(merged is entry for entry in previous_entries):
                            # No matching delta was accumulated.  Keep the
                            # normal merge path so a completion-only event
                            # still carries its arguments exactly once.
                            _, merged = self._merge_tool_call(tool_calls, event.tool_call)
                        if key in completed_tool_calls:
                            continue
                        validate_complete_tool_call(merged, "Google Gemini")
                        completed_tool_calls.add(key)
                        event.tool_call = dict(merged)
                        yield event
                        continue
                    yield event
            if not terminal_seen:
                raise RuntimeError("Google Gemini 流在终止事件之前结束。")
            return
        result = self.complete(request)
        if result.text:
            yield StreamEvent(
                type="text_delta", text=result.text, response_id=result.response_id, raw=result.raw
            )
        if result.reasoning:
            yield StreamEvent(type="reasoning_delta", reasoning=result.reasoning, raw=result.raw)
        if result.refusal:
            yield StreamEvent(
                type="refusal_delta",
                refusal=result.refusal,
                response_id=result.response_id,
                raw=result.raw,
            )
        for call in result.tool_calls:
            yield StreamEvent(type="tool_call_delta", tool_call=call, raw=result.raw)
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

    async def astream_events(
        self, request: CompletionRequest | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        request = coerce_completion_request(request, kwargs, "Google astream_events")
        request = request.copy_with(stream=True)
        self._validate_request(request)
        effective_system = extract_system_prompt(request.messages) or request.system_prompt
        config = self._build_generation_config(
            effective_system,
            request.tools,
            request.temperature if request.temperature_is_explicit else None,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            candidate_count=request.n,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            modalities=request.modalities,
            stop=request.stop,
            response_format=request.response_format,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
            extra_headers=request.extra_headers,
            extra_query=request.extra_query,
            thinking=request.thinking,
            service_tier=request.service_tier,
            timeout=request.timeout,
        )
        client = self._get_client()
        vertexai = self._effective_vertex_mode(client)
        contents = self._contents(
            request.messages,
            request.prompt,
            request.system_prompt,
            vertex_mode=vertexai,
        )
        if request.stream:
            response = retry_async_stream(
                lambda: client.aio.models.generate_content_stream(
                    model=self._model_name, contents=contents, config=config
                ),
                first_event_validator=lambda event: raise_for_stream_error_event(
                    event, "Google Gemini"
                ),
            )
            tool_calls: dict[Any, dict[str, Any]] = {}
            completed_tool_calls: set[Any] = set()
            terminal_seen = False
            async for chunk in iterate_async(response):
                for event in self._stream_events_from_chunk(chunk):
                    if event.type == "finish":
                        terminal_seen = True
                        for completed_event in self._complete_stream_tool_calls(
                            tool_calls,
                            completed_tool_calls,
                            event.response_id,
                            event.raw,
                        ):
                            yield completed_event
                        yield event
                        continue
                    if event.type == "tool_call_delta" and event.tool_call:
                        key, merged = self._merge_tool_call(tool_calls, event.tool_call)
                        event.tool_call = dict(merged)
                        yield event
                        continue
                    if event.type == "tool_call_completed" and event.tool_call:
                        previous_entries = tuple(tool_calls.values())
                        key, merged = self._merge_tool_call(
                            tool_calls,
                            event.tool_call,
                            merge_arguments=False,
                        )
                        if not any(merged is entry for entry in previous_entries):
                            # No matching delta was accumulated.  Keep the
                            # normal merge path so a completion-only event
                            # still carries its arguments exactly once.
                            _, merged = self._merge_tool_call(tool_calls, event.tool_call)
                        if key in completed_tool_calls:
                            continue
                        validate_complete_tool_call(merged, "Google Gemini")
                        completed_tool_calls.add(key)
                        event.tool_call = dict(merged)
                        yield event
                        continue
                    yield event
            if not terminal_seen:
                raise RuntimeError("Google Gemini 异步流在终止事件之前结束。")
            return
        result = await self.acomplete(request)
        if result.text:
            yield StreamEvent(
                type="text_delta", text=result.text, response_id=result.response_id, raw=result.raw
            )
        if result.reasoning:
            yield StreamEvent(type="reasoning_delta", reasoning=result.reasoning, raw=result.raw)
        if result.refusal:
            yield StreamEvent(
                type="refusal_delta",
                refusal=result.refusal,
                response_id=result.response_id,
                raw=result.raw,
            )
        for call in result.tool_calls:
            yield StreamEvent(type="tool_call_delta", tool_call=call, raw=result.raw)
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

    def create_cache(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().caches.create(
            model=kwargs.pop("model", self._model_name),
            **self._resource_config_kwargs(types.CreateCachedContentConfig, config, kwargs),
        )

    def get_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().caches.get(
            name=name, **self._resource_config_kwargs(types.GetCachedContentConfig, config, kwargs)
        )

    def list_caches(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().caches.list(
            **self._resource_config_kwargs(types.ListCachedContentsConfig, config, kwargs)
        )

    def update_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().caches.update(
            name=name,
            **self._resource_config_kwargs(types.UpdateCachedContentConfig, config, kwargs),
        )

    def delete_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().caches.delete(
            name=name,
            **self._resource_config_kwargs(types.DeleteCachedContentConfig, config, kwargs),
        )

    async def async_create_cache(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.caches.create(
                model=kwargs.pop("model", self._model_name),
                **self._resource_config_kwargs(types.CreateCachedContentConfig, config, kwargs),
            )
        )

    async def async_get_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.caches.get(
                name=name,
                **self._resource_config_kwargs(types.GetCachedContentConfig, config, kwargs),
            )
        )

    async def async_list_caches(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.caches.list(
                **self._resource_config_kwargs(types.ListCachedContentsConfig, config, kwargs)
            )
        )

    async def async_update_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.caches.update(
                name=name,
                **self._resource_config_kwargs(types.UpdateCachedContentConfig, config, kwargs),
            )
        )

    async def async_delete_cache(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.caches.delete(
                name=name,
                **self._resource_config_kwargs(types.DeleteCachedContentConfig, config, kwargs),
            )
        )

    def create_batch(self, src: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.create(
            model=kwargs.pop("model", self._model_name),
            src=src,
            **self._resource_config_kwargs(types.CreateBatchJobConfig, config, kwargs),
        )

    def create_embedding_batch(self, src: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.create_embeddings(
            model=kwargs.pop("model", self._model_name),
            src=src,
            **self._resource_config_kwargs(types.CreateEmbeddingsBatchJobConfig, config, kwargs),
        )

    def get_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.get(
            name=name, **self._resource_config_kwargs(types.GetBatchJobConfig, config, kwargs)
        )

    def list_batches(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.list(
            **self._resource_config_kwargs(types.ListBatchJobsConfig, config, kwargs)
        )

    def cancel_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.cancel(
            name=name, **self._resource_config_kwargs(types.CancelBatchJobConfig, config, kwargs)
        )

    def delete_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().batches.delete(
            name=name, **self._resource_config_kwargs(types.DeleteBatchJobConfig, config, kwargs)
        )

    async def async_create_batch(self, src: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.create(
                model=kwargs.pop("model", self._model_name),
                src=src,
                **self._resource_config_kwargs(types.CreateBatchJobConfig, config, kwargs),
            )
        )

    async def async_create_embedding_batch(
        self, src: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.create_embeddings(
                model=kwargs.pop("model", self._model_name),
                src=src,
                **self._resource_config_kwargs(
                    types.CreateEmbeddingsBatchJobConfig, config, kwargs
                ),
            )
        )

    async def async_get_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.get(
                name=name, **self._resource_config_kwargs(types.GetBatchJobConfig, config, kwargs)
            )
        )

    async def async_list_batches(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.list(
                **self._resource_config_kwargs(types.ListBatchJobsConfig, config, kwargs)
            )
        )

    async def async_cancel_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.cancel(
                name=name,
                **self._resource_config_kwargs(types.CancelBatchJobConfig, config, kwargs),
            )
        )

    async def async_delete_batch(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.batches.delete(
                name=name,
                **self._resource_config_kwargs(types.DeleteBatchJobConfig, config, kwargs),
            )
        )

    def list_models(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().models.list(
            **self._resource_config_kwargs(types.ListModelsConfig, config, kwargs)
        )

    def get_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().models.get(
            model=model, **self._resource_config_kwargs(types.GetModelConfig, config, kwargs)
        )

    def delete_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().models.delete(
            model=model, **self._resource_config_kwargs(types.DeleteModelConfig, config, kwargs)
        )

    def update_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().models.update(
            model=model, **self._resource_config_kwargs(types.UpdateModelConfig, config, kwargs)
        )

    def tune(
        self, base_model: str, training_dataset: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        """创建 Google GenAI 调优任务。"""
        tunings = self._require_resource(self._get_client(), "tunings", "调优")
        return tunings.tune(
            base_model=base_model,
            training_dataset=training_dataset,
            **self._resource_config_kwargs(types.CreateTuningJobConfig, config, kwargs),
        )

    def get_tuning(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client(), "tunings", "调优")
        return tunings.get(
            name=name, **self._resource_config_kwargs(types.GetTuningJobConfig, config, kwargs)
        )

    def list_tunings(self, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client(), "tunings", "调优")
        return tunings.list(
            **self._resource_config_kwargs(types.ListTuningJobsConfig, config, kwargs)
        )

    def cancel_tuning(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client(), "tunings", "调优")
        return tunings.cancel(
            name=name, **self._resource_config_kwargs(types.CancelTuningJobConfig, config, kwargs)
        )

    def validate_tuning_reward(
        self,
        parent: str,
        sample_response: Any,
        example: Any,
        single_reward_config: Any = None,
        composite_reward_config: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        tunings = self._require_resource(self._get_client(), "tunings", "调优")
        if single_reward_config is not None and composite_reward_config is not None:
            raise ValueError(
                "Google validate_reward 的 single_reward_config 与 "
                "composite_reward_config 不能同时传入。"
            )
        reward_kwargs: dict[str, Any] = {}
        if single_reward_config is not None:
            reward_kwargs["single_reward_config"] = single_reward_config
        if composite_reward_config is not None:
            reward_kwargs["composite_reward_config"] = composite_reward_config
        reward_kwargs.update(
            self._resource_config_kwargs(types.ValidateRewardConfig, config, kwargs)
        )
        return tunings.validate_reward(
            parent=parent,
            sample_response=sample_response,
            example=example,
            **reward_kwargs,
        )

    def recontext_image(
        self, source: Any, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return self._get_client().models.recontext_image(
            model=model or self._model_name,
            source=source,
            **self._resource_config_kwargs(types.RecontextImageConfig, config, kwargs),
        )

    async def async_list_models(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.list(
                **self._resource_config_kwargs(types.ListModelsConfig, config, kwargs)
            )
        )

    async def async_get_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.get(
                model=model, **self._resource_config_kwargs(types.GetModelConfig, config, kwargs)
            )
        )

    async def async_delete_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.delete(
                model=model, **self._resource_config_kwargs(types.DeleteModelConfig, config, kwargs)
            )
        )

    async def async_update_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.update(
                model=model, **self._resource_config_kwargs(types.UpdateModelConfig, config, kwargs)
            )
        )

    async def async_tune(
        self, base_model: str, training_dataset: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        tunings = self._require_resource(self._get_client().aio, "tunings", "异步调优")
        return await self._resolve_async_result(
            tunings.tune(
                base_model=base_model,
                training_dataset=training_dataset,
                **self._resource_config_kwargs(types.CreateTuningJobConfig, config, kwargs),
            )
        )

    async def async_get_tuning(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client().aio, "tunings", "异步调优")
        return await self._resolve_async_result(
            tunings.get(
                name=name, **self._resource_config_kwargs(types.GetTuningJobConfig, config, kwargs)
            )
        )

    async def async_list_tunings(self, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client().aio, "tunings", "异步调优")
        return await self._resolve_async_result(
            tunings.list(**self._resource_config_kwargs(types.ListTuningJobsConfig, config, kwargs))
        )

    async def async_cancel_tuning(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        tunings = self._require_resource(self._get_client().aio, "tunings", "异步调优")
        return await self._resolve_async_result(
            tunings.cancel(
                name=name,
                **self._resource_config_kwargs(types.CancelTuningJobConfig, config, kwargs),
            )
        )

    async def async_validate_tuning_reward(
        self,
        parent: str,
        sample_response: Any,
        example: Any,
        single_reward_config: Any = None,
        composite_reward_config: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        tunings = self._require_resource(self._get_client().aio, "tunings", "异步调优")
        if single_reward_config is not None and composite_reward_config is not None:
            raise ValueError(
                "Google validate_reward 的 single_reward_config 与 "
                "composite_reward_config 不能同时传入。"
            )
        reward_kwargs: dict[str, Any] = {}
        if single_reward_config is not None:
            reward_kwargs["single_reward_config"] = single_reward_config
        if composite_reward_config is not None:
            reward_kwargs["composite_reward_config"] = composite_reward_config
        reward_kwargs.update(
            self._resource_config_kwargs(types.ValidateRewardConfig, config, kwargs)
        )
        return await self._resolve_async_result(
            tunings.validate_reward(
                parent=parent,
                sample_response=sample_response,
                example=example,
                **reward_kwargs,
            )
        )

    async def async_recontext_image(
        self, source: Any, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.recontext_image(
                model=model or self._model_name,
                source=source,
                **self._resource_config_kwargs(types.RecontextImageConfig, config, kwargs),
            )
        )

    def generate_images(
        self, prompt: str, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return self._get_client().models.generate_images(
            model=model or self._model_name,
            prompt=prompt,
            **self._resource_config_kwargs(types.GenerateImagesConfig, config, kwargs),
        )

    def edit_image(
        self,
        prompt: str,
        reference_images: list[Any],
        model: str | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._get_client().models.edit_image(
            model=model or self._model_name,
            prompt=prompt,
            reference_images=reference_images,
            **self._resource_config_kwargs(types.EditImageConfig, config, kwargs),
        )

    def upscale_image(
        self,
        image: Any,
        upscale_factor: str,
        model: str | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._get_client().models.upscale_image(
            model=model or self._model_name,
            image=image,
            upscale_factor=upscale_factor,
            **self._resource_config_kwargs(types.UpscaleImageConfig, config, kwargs),
        )

    def segment_image(
        self, source: Any, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return self._get_client().models.segment_image(
            model=model or self._model_name,
            source=source,
            **self._resource_config_kwargs(types.SegmentImageConfig, config, kwargs),
        )

    async def async_generate_images(
        self, prompt: str, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.generate_images(
                model=model or self._model_name,
                prompt=prompt,
                **self._resource_config_kwargs(types.GenerateImagesConfig, config, kwargs),
            )
        )

    async def async_edit_image(
        self,
        prompt: str,
        reference_images: list[Any],
        model: str | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.edit_image(
                model=model or self._model_name,
                prompt=prompt,
                reference_images=reference_images,
                **self._resource_config_kwargs(types.EditImageConfig, config, kwargs),
            )
        )

    async def async_upscale_image(
        self,
        image: Any,
        upscale_factor: str,
        model: str | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.upscale_image(
                model=model or self._model_name,
                image=image,
                upscale_factor=upscale_factor,
                **self._resource_config_kwargs(types.UpscaleImageConfig, config, kwargs),
            )
        )

    async def async_segment_image(
        self, source: Any, model: str | None = None, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.segment_image(
                model=model or self._model_name,
                source=source,
                **self._resource_config_kwargs(types.SegmentImageConfig, config, kwargs),
            )
        )

    def generate_videos(
        self,
        prompt: str | None = None,
        model: str | None = None,
        image: Any = None,
        video: Any = None,
        source: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._get_client().models.generate_videos(
            model=model or self._model_name,
            prompt=prompt,
            image=image,
            video=video,
            source=source,
            **self._resource_config_kwargs(types.GenerateVideosConfig, config, kwargs),
        )

    def get_operation(self, operation: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().operations.get(
            operation, **self._resource_config_kwargs(types.GetOperationConfig, config, kwargs)
        )

    async def async_generate_videos(
        self,
        prompt: str | None = None,
        model: str | None = None,
        image: Any = None,
        video: Any = None,
        source: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.models.generate_videos(
                model=model or self._model_name,
                prompt=prompt,
                image=image,
                video=video,
                source=source,
                **self._resource_config_kwargs(types.GenerateVideosConfig, config, kwargs),
            )
        )

    async def async_get_operation(self, operation: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.operations.get(
                operation, **self._resource_config_kwargs(types.GetOperationConfig, config, kwargs)
            )
        )

    def create_chat(
        self,
        history: list[Any] | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._get_client().chats.create(
            model=kwargs.pop("model", self._model_name),
            history=history,
            **self._resource_config_kwargs(types.GenerateContentConfig, config, kwargs),
        )

    async def async_create_chat(
        self,
        history: list[Any] | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        # google-genai returns an AsyncChat object synchronously; subsequent
        # send_message calls on that object are awaitable.
        return await self._resolve_async_result(
            self._get_client().aio.chats.create(
                model=kwargs.pop("model", self._model_name),
                history=history,
                **self._resource_config_kwargs(types.GenerateContentConfig, config, kwargs),
            )
        )

    def create_file_search_store(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.create(
            **self._resource_config_kwargs(types.CreateFileSearchStoreConfig, config, kwargs)
        )

    def create_auth_token(self, config: Any = None, **kwargs: Any) -> Any:
        """创建 Google GenAI Auth Token。"""
        self._require_provider_resource("create_auth_token")
        return self._get_client().auth_tokens.create(
            **self._resource_config_kwargs(types.CreateAuthTokenConfig, config, kwargs)
        )

    # The next-generation Google resources are intentionally exposed as
    # provider-specific methods. Their request/response types are experimental
    # and should not be forced into the shared completion contract.
    def create_interaction(self, **kwargs: Any) -> Any:
        self._require_provider_resource("create_interaction")
        return self._get_client().interactions.create(**kwargs)

    def get_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("get_interaction")
        return self._get_client().interactions.get(id=interaction_id, **kwargs)

    def cancel_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("cancel_interaction")
        return self._get_client().interactions.cancel(id=interaction_id, **kwargs)

    def delete_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("delete_interaction")
        return self._get_client().interactions.delete(id=interaction_id, **kwargs)

    def create_agent(self, **kwargs: Any) -> Any:
        self._require_provider_resource("create_agent")
        return self._get_client().agents.create(**kwargs)

    def get_agent(self, agent_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("get_agent")
        return self._get_client().agents.get(agent_id, **kwargs)

    def list_agents(self, **kwargs: Any) -> Any:
        self._require_provider_resource("list_agents")
        return self._get_client().agents.list(**kwargs)

    def delete_agent(self, agent_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("delete_agent")
        return self._get_client().agents.delete(agent_id, **kwargs)

    def create_webhook(self, **kwargs: Any) -> Any:
        self._require_provider_resource("create_webhook")
        return self._get_client().webhooks.create(**kwargs)

    def get_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("get_webhook")
        return self._get_client().webhooks.get(webhook_id, **kwargs)

    def list_webhooks(self, **kwargs: Any) -> Any:
        self._require_provider_resource("list_webhooks")
        return self._get_client().webhooks.list(**kwargs)

    def update_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("update_webhook")
        return self._get_client().webhooks.update(webhook_id, **kwargs)

    def delete_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("delete_webhook")
        return self._get_client().webhooks.delete(webhook_id, **kwargs)

    def ping_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("ping_webhook")
        return self._get_client().webhooks.ping(webhook_id, **kwargs)

    def rotate_webhook_signing_secret(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("rotate_webhook_signing_secret")
        return self._get_client().webhooks.rotate_signing_secret(webhook_id, **kwargs)

    def create_environment(self, **kwargs: Any) -> Any:
        self._require_provider_resource("create_environment")
        return self._get_client().environments.create_environment(**kwargs)

    def get_environment(self, environment_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("get_environment")
        return self._get_client().environments.get_environment(id=environment_id, **kwargs)

    def list_environments(self, **kwargs: Any) -> Any:
        self._require_provider_resource("list_environments")
        return self._get_client().environments.list_environments(**kwargs)

    def delete_environment(self, environment_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("delete_environment")
        return self._get_client().environments.delete_environment(id=environment_id, **kwargs)

    def get_environment_files(self, environment_id: str, path: str, **kwargs: Any) -> Any:
        """列出环境快照中的文件或目录内容。"""
        self._require_provider_resource("get_environment_files")
        return self._get_client().environments.files.list(environment_id, path, **kwargs)

    def create_trigger(self, **kwargs: Any) -> Any:
        self._require_provider_resource("create_trigger")
        return self._get_client().triggers.create(**kwargs)

    def get_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("get_trigger")
        return self._get_client().triggers.get(trigger_id, **kwargs)

    def list_triggers(self, **kwargs: Any) -> Any:
        self._require_provider_resource("list_triggers")
        return self._get_client().triggers.list(**kwargs)

    def update_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("update_trigger")
        return self._get_client().triggers.update(trigger_id, **kwargs)

    def delete_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("delete_trigger")
        return self._get_client().triggers.delete(trigger_id, **kwargs)

    def run_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("run_trigger")
        return self._get_client().triggers.run(trigger_id, **kwargs)

    def list_trigger_executions(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("list_trigger_executions")
        return self._get_client().triggers.list_executions(trigger_id, **kwargs)

    def connect_live(self, *, model: str | None = None, config: Any = None) -> Any:
        """返回 Google Live API 的异步上下文管理器。

        google-genai 只在 ``client.aio.live`` 提供 Live 连接，不能把它伪装成
        普通的 ``await`` 方法；调用方应使用 ``async with`` 管理会话生命周期。
        """
        self._validate_resource_config(config)
        if not model:
            model = self._model_name
        client = self._get_client()
        try:
            aio = client.aio
        except (AttributeError, NotImplementedError) as exc:
            raise NotImplementedError("当前 google-genai SDK 未提供异步 Live 资源。") from exc
        live = self._require_resource(aio, "live", "Live")
        connect = self._require_resource(live, "connect", "Live 连接")
        if not callable(connect):
            raise NotImplementedError("当前 google-genai SDK 未提供可调用的 Live 连接资源。")
        return connect(model=model, config=config)

    def async_connect_live(self, *, model: str | None = None, config: Any = None) -> Any:
        """返回异步 Live API 上下文管理器，调用方使用 `async with`。"""
        self._validate_resource_config(config)
        return self.connect_live(model=model, config=config)

    def get_file_search_store(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.get(
            name=name,
            **self._resource_config_kwargs(types.GetFileSearchStoreConfig, config, kwargs),
        )

    def list_file_search_stores(self, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.list(
            **self._resource_config_kwargs(types.ListFileSearchStoresConfig, config, kwargs)
        )

    def delete_file_search_store(self, name: str, config: Any = None, **kwargs: Any) -> Any:
        return self._get_client().file_search_stores.delete(
            name=name,
            **self._resource_config_kwargs(types.DeleteFileSearchStoreConfig, config, kwargs),
        )

    def import_file_to_file_search_store(
        self,
        file_search_store_name: str,
        file_name: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._get_client().file_search_stores.import_file(
            file_search_store_name=file_search_store_name,
            file_name=file_name,
            **self._resource_config_kwargs(types.ImportFileConfig, config, kwargs),
        )

    def upload_to_file_search_store(
        self, file_search_store_name: str, file: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        return self._get_client().file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store_name,
            file=file,
            **self._resource_config_kwargs(types.UploadToFileSearchStoreConfig, config, kwargs),
        )

    def download_file_search_media(self, media_id: str, config: Any = None, **kwargs: Any) -> bytes:
        return self._get_client().file_search_stores.download_media(
            media_id=media_id,
            **self._resource_config_kwargs(types.DownloadMediaConfig, config, kwargs),
        )

    async def async_create_file_search_store(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.create(
                **self._resource_config_kwargs(types.CreateFileSearchStoreConfig, config, kwargs)
            )
        )

    async def async_create_auth_token(self, config: Any = None, **kwargs: Any) -> Any:
        """异步创建 Google GenAI Auth Token。"""
        self._require_provider_resource("async_create_auth_token")
        return await self._resolve_async_result(
            self._get_client().aio.auth_tokens.create(
                **self._resource_config_kwargs(types.CreateAuthTokenConfig, config, kwargs)
            )
        )

    async def async_create_interaction(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_create_interaction")
        return await self._resolve_async_result(
            self._get_client().aio.interactions.create(**kwargs)
        )

    async def async_get_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_get_interaction")
        return await self._resolve_async_result(
            self._get_client().aio.interactions.get(id=interaction_id, **kwargs)
        )

    async def async_cancel_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_cancel_interaction")
        return await self._resolve_async_result(
            self._get_client().aio.interactions.cancel(id=interaction_id, **kwargs)
        )

    async def async_delete_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_delete_interaction")
        return await self._resolve_async_result(
            self._get_client().aio.interactions.delete(id=interaction_id, **kwargs)
        )

    async def async_create_agent(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_create_agent")
        return await self._resolve_async_result(self._get_client().aio.agents.create(**kwargs))

    async def async_get_agent(self, agent_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_get_agent")
        return await self._resolve_async_result(
            self._get_client().aio.agents.get(agent_id, **kwargs)
        )

    async def async_list_agents(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_list_agents")
        return await self._resolve_async_result(self._get_client().aio.agents.list(**kwargs))

    async def async_delete_agent(self, agent_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_delete_agent")
        return await self._resolve_async_result(
            self._get_client().aio.agents.delete(agent_id, **kwargs)
        )

    async def async_create_webhook(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_create_webhook")
        return await self._resolve_async_result(self._get_client().aio.webhooks.create(**kwargs))

    async def async_get_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_get_webhook")
        return await self._resolve_async_result(
            self._get_client().aio.webhooks.get(webhook_id, **kwargs)
        )

    async def async_list_webhooks(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_list_webhooks")
        return await self._resolve_async_result(self._get_client().aio.webhooks.list(**kwargs))

    async def async_update_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_update_webhook")
        return await self._resolve_async_result(
            self._get_client().aio.webhooks.update(webhook_id, **kwargs)
        )

    async def async_delete_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_delete_webhook")
        return await self._resolve_async_result(
            self._get_client().aio.webhooks.delete(webhook_id, **kwargs)
        )

    async def async_ping_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_ping_webhook")
        return await self._resolve_async_result(
            self._get_client().aio.webhooks.ping(webhook_id, **kwargs)
        )

    async def async_rotate_webhook_signing_secret(self, webhook_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_rotate_webhook_signing_secret")
        return await self._resolve_async_result(
            self._get_client().aio.webhooks.rotate_signing_secret(webhook_id, **kwargs)
        )

    async def async_create_environment(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_create_environment")
        return await self._resolve_async_result(
            self._get_client().aio.environments.create_environment(**kwargs)
        )

    async def async_get_environment(self, environment_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_get_environment")
        return await self._resolve_async_result(
            self._get_client().aio.environments.get_environment(id=environment_id, **kwargs)
        )

    async def async_list_environments(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_list_environments")
        return await self._resolve_async_result(
            self._get_client().aio.environments.list_environments(**kwargs)
        )

    async def async_delete_environment(self, environment_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_delete_environment")
        return await self._resolve_async_result(
            self._get_client().aio.environments.delete_environment(id=environment_id, **kwargs)
        )

    async def async_get_environment_files(
        self, environment_id: str, path: str, **kwargs: Any
    ) -> Any:
        """异步列出环境快照中的文件或目录内容。"""
        self._require_provider_resource("async_get_environment_files")
        return await self._resolve_async_result(
            self._get_client().aio.environments.files.list(environment_id, path, **kwargs)
        )

    async def async_create_trigger(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_create_trigger")
        return await self._resolve_async_result(self._get_client().aio.triggers.create(**kwargs))

    async def async_get_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_get_trigger")
        return await self._resolve_async_result(
            self._get_client().aio.triggers.get(trigger_id, **kwargs)
        )

    async def async_list_triggers(self, **kwargs: Any) -> Any:
        self._require_provider_resource("async_list_triggers")
        return await self._resolve_async_result(self._get_client().aio.triggers.list(**kwargs))

    async def async_update_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_update_trigger")
        return await self._resolve_async_result(
            self._get_client().aio.triggers.update(trigger_id, **kwargs)
        )

    async def async_delete_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_delete_trigger")
        return await self._resolve_async_result(
            self._get_client().aio.triggers.delete(trigger_id, **kwargs)
        )

    async def async_run_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_run_trigger")
        return await self._resolve_async_result(
            self._get_client().aio.triggers.run(trigger_id, **kwargs)
        )

    async def async_list_trigger_executions(self, trigger_id: str, **kwargs: Any) -> Any:
        self._require_provider_resource("async_list_trigger_executions")
        return await self._resolve_async_result(
            self._get_client().aio.triggers.list_executions(trigger_id, **kwargs)
        )

    async def async_get_file_search_store(
        self, name: str, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.get(
                name=name,
                **self._resource_config_kwargs(types.GetFileSearchStoreConfig, config, kwargs),
            )
        )

    async def async_list_file_search_stores(self, config: Any = None, **kwargs: Any) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.list(
                **self._resource_config_kwargs(types.ListFileSearchStoresConfig, config, kwargs)
            )
        )

    async def async_delete_file_search_store(
        self, name: str, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.delete(
                name=name,
                **self._resource_config_kwargs(types.DeleteFileSearchStoreConfig, config, kwargs),
            )
        )

    async def async_import_file_to_file_search_store(
        self,
        file_search_store_name: str,
        file_name: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.import_file(
                file_search_store_name=file_search_store_name,
                file_name=file_name,
                **self._resource_config_kwargs(types.ImportFileConfig, config, kwargs),
            )
        )

    async def async_upload_to_file_search_store(
        self, file_search_store_name: str, file: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.upload_to_file_search_store(
                file_search_store_name=file_search_store_name,
                file=file,
                **self._resource_config_kwargs(types.UploadToFileSearchStoreConfig, config, kwargs),
            )
        )

    async def async_download_file_search_media(
        self, media_id: str, config: Any = None, **kwargs: Any
    ) -> bytes:
        return await self._resolve_async_result(
            self._get_client().aio.file_search_stores.download_media(
                media_id=media_id,
                **self._resource_config_kwargs(types.DownloadMediaConfig, config, kwargs),
            )
        )

    def invoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = _UNSET_TEMPERATURE,  # type: ignore[assignment]
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n: int | None = None,
        seed: int | None = None,
        modalities: list[str] | None = None,
        stop: list[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
    ) -> Generator[str, None, None]:
        """同步调用 Google LLM (CSE Sensor)。"""
        temperature = None if temperature is _UNSET_TEMPERATURE else temperature
        logger.info(f"调用 Google LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        effective_system_prompt = extract_system_prompt(messages) or system_prompt
        config = self._build_generation_config(
            effective_system_prompt,
            tools,
            temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            candidate_count=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            modalities=modalities,
            stop=stop,
            response_format=response_format,
            tool_choice=tool_choice,
            extra_body=extra_body,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            thinking=thinking,
            service_tier=service_tier,
        )
        contents = self._contents(
            messages,
            prompt,
            system_prompt,
            vertex_mode=self._effective_vertex_mode(client),
        )

        try:
            if stream:
                terminal_seen = False
                for chunk in retry_sync_stream(
                    lambda: client.models.generate_content_stream(
                        model=self._model_name, contents=contents, config=config
                    ),
                    first_event_validator=lambda event: raise_for_stream_error_event(
                        event, "Google Gemini"
                    ),
                ):
                    raise_for_stream_error_event(chunk, "Google Gemini")
                    terminal_seen = terminal_seen or self._stream_chunk_has_terminal_event(chunk)
                    refusal = self._extract_refusal(chunk)
                    if refusal:
                        raise RuntimeError(f"Google 请求被拒绝: {refusal}")
                    text = self._response_text(chunk)
                    if text:
                        yield text
                if not terminal_seen:
                    raise RuntimeError("Google Gemini 流在终止事件之前结束。")
            else:
                response = retry_sync_call(
                    lambda: client.models.generate_content(
                        model=self._model_name, contents=contents, config=config
                    )
                )
                refusal = self._extract_refusal(response)
                if refusal:
                    raise RuntimeError(f"Google 请求被拒绝: {refusal}")
                text = self._response_text(response)
                if text:
                    yield text

            duration = time.perf_counter() - start_time
            logger.info(f"Google LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Google LLM (%s) 出错: %s",
                self._model_name,
                error_text,
            )
            raise

    async def ainvoke(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float | None = _UNSET_TEMPERATURE,  # type: ignore[assignment]
        messages: Any = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n: int | None = None,
        seed: int | None = None,
        modalities: list[str] | None = None,
        stop: list[str] | str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_query: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """异步调用 Google LLM (CSE Sensor)。"""
        temperature = None if temperature is _UNSET_TEMPERATURE else temperature
        logger.info(f"异步调用 Google LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        effective_system_prompt = extract_system_prompt(messages) or system_prompt
        config = self._build_generation_config(
            effective_system_prompt,
            tools,
            temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            candidate_count=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            modalities=modalities,
            stop=stop,
            response_format=response_format,
            tool_choice=tool_choice,
            extra_body=extra_body,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            thinking=thinking,
            service_tier=service_tier,
        )
        contents = self._contents(
            messages,
            prompt,
            system_prompt,
            vertex_mode=self._effective_vertex_mode(client),
        )

        try:
            if stream:
                terminal_seen = False
                async for chunk in retry_async_stream(
                    lambda: client.aio.models.generate_content_stream(
                        model=self._model_name, contents=contents, config=config
                    ),
                    first_event_validator=lambda event: raise_for_stream_error_event(
                        event, "Google Gemini"
                    ),
                ):
                    raise_for_stream_error_event(chunk, "Google Gemini")
                    terminal_seen = terminal_seen or self._stream_chunk_has_terminal_event(chunk)
                    refusal = self._extract_refusal(chunk)
                    if refusal:
                        raise RuntimeError(f"Google 请求被拒绝: {refusal}")
                    text = self._response_text(chunk)
                    if text:
                        yield text
                if not terminal_seen:
                    raise RuntimeError("Google Gemini 异步流在终止事件之前结束。")
            else:
                response = await retry_async_call(
                    lambda: client.aio.models.generate_content(
                        model=self._model_name, contents=contents, config=config
                    )
                )
                refusal = self._extract_refusal(response)
                if refusal:
                    raise RuntimeError(f"Google 请求被拒绝: {refusal}")
                text = self._response_text(response)
                if text:
                    yield text

            duration = time.perf_counter() - start_time
            logger.info(f"Google LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Google LLM (%s) 异步出错: %s",
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
        logger.info(f"调用 Google Embedding ({self._model_name})，数量: {len(texts)}")
        start_time = time.perf_counter()

        try:
            config_values = self._embedding_options()
            overlap = sorted({"model", "contents", "config"}.intersection(kwargs))
            if overlap:
                raise ValueError(f"Google Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
            request_options = {
                key: kwargs.pop(key, None)
                for key in ("extra_headers", "extra_body", "extra_query", "timeout")
            }
            self._validate_embedding_kwargs(kwargs)
            config_values.update(kwargs)
            config_values.setdefault("task_type", "RETRIEVAL_DOCUMENT")
            http_options = self._http_options_for_request(**request_options)
            if http_options is not None:
                config_values["http_options"] = http_options
            config = types.EmbedContentConfig(**config_values)
            # Validate and construct the complete request before initializing
            # the SDK client, so invalid request extensions fail closed.
            client = self._get_client()
            vertexai = self._effective_vertex_mode(client)
            if vertexai is True and self._vertex_embed_content_only():
                embeddings = []
                for text in texts:
                    response = client.models.embed_content(
                        model=self._model_name, contents=[text], config=config
                    )
                    values = list(field(response, "embeddings", []) or [])
                    if len(values) != 1:
                        raise RuntimeError(
                            "Google Vertex Embedding 单内容请求未返回唯一 embedding。"
                        )
                    embeddings.append(normalize_embedding_vector(field(values[0], "values")))
            else:
                response = client.models.embed_content(
                    model=self._model_name, contents=texts, config=config
                )
                values = list(field(response, "embeddings", []) or [])
                if len(values) != len(texts):
                    raise RuntimeError("Google Embedding 返回数量与输入文本数量不一致。")
                embeddings = [normalize_embedding_vector(field(item, "values")) for item in values]
            duration = time.perf_counter() - start_time
            logger.info(f"Google Embedding ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Google Embedding (%s) 出错: %s",
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
        logger.info(f"异步调用 Google Embedding ({self._model_name})，数量: {len(texts)}")
        start_time = time.perf_counter()

        try:
            config_values = self._embedding_options()
            overlap = sorted({"model", "contents", "config"}.intersection(kwargs))
            if overlap:
                raise ValueError(f"Google Embedding 不允许覆盖请求字段: {', '.join(overlap)}")
            request_options = {
                key: kwargs.pop(key, None)
                for key in ("extra_headers", "extra_body", "extra_query", "timeout")
            }
            self._validate_embedding_kwargs(kwargs)
            config_values.update(kwargs)
            config_values.setdefault("task_type", "RETRIEVAL_DOCUMENT")
            http_options = self._http_options_for_request(**request_options)
            if http_options is not None:
                config_values["http_options"] = http_options
            config = types.EmbedContentConfig(**config_values)
            # Keep async embedding subject to the same fail-closed ordering as
            # the sync entry point.
            client = self._get_client()
            vertexai = self._effective_vertex_mode(client)
            if vertexai is True and self._vertex_embed_content_only():
                embeddings = []
                for text in texts:
                    response = await self._resolve_async_result(
                        client.aio.models.embed_content(
                            model=self._model_name, contents=[text], config=config
                        )
                    )
                    values = list(field(response, "embeddings", []) or [])
                    if len(values) != 1:
                        raise RuntimeError(
                            "Google Vertex Embedding 单内容请求未返回唯一 embedding。"
                        )
                    embeddings.append(normalize_embedding_vector(field(values[0], "values")))
            else:
                response = await self._resolve_async_result(
                    client.aio.models.embed_content(
                        model=self._model_name, contents=texts, config=config
                    )
                )
                values = list(field(response, "embeddings", []) or [])
                if len(values) != len(texts):
                    raise RuntimeError("Google Embedding 返回数量与输入文本数量不一致。")
                embeddings = [normalize_embedding_vector(field(item, "values")) for item in values]
            duration = time.perf_counter() - start_time
            logger.info(f"Google Embedding ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            error_text = redact_sensitive_text(str(e))
            logger.exception(
                "Google Embedding (%s) 异步出错: %s",
                self._model_name,
                error_text,
            )
            raise

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        kwargs.setdefault("task_type", "RETRIEVAL_QUERY")
        return self.embed_documents([text], **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        kwargs.setdefault("task_type", "RETRIEVAL_QUERY")
        return (await self.aembed_documents([text], **kwargs))[0]

    def close(self) -> None:
        """释放同步 Client 及其关联的异步资源。

        google-genai 的 ``Client.close`` 不会关闭 ``Client.aio``；同步上下文
        使用生命周期桥接器等待异步关闭，活动事件循环中则保留 Client 并
        显式要求调用 ``aclose``。
        """
        client = self._client
        if client is None:
            return
        first_error: Exception | None = None
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - close all resources before reporting
            first_error = exc
        try:
            close_resource_sync(client.aio, "Google 异步客户端")
        except Exception as exc:  # noqa: BLE001 - close all resources before reporting
            first_error = first_error or exc
        else:
            self._client = None
        if first_error is not None:
            raise first_error

    async def aclose(self) -> None:
        """释放 google-genai 的异步 Client，并在异常后清除状态。"""
        client = self._client
        self._client = None
        if client is None:
            return
        first_error: Exception | None = None
        try:
            result = client.aio.aclose()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - close all resources before reporting
            first_error = exc
        try:
            result = client.close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - close all resources before reporting
            first_error = first_error or exc
        if first_error is not None:
            raise first_error
