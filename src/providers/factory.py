import importlib
import inspect
from collections.abc import Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, ClassVar

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    RerankModel,
    TextEmbeddingModel,
)
from src.utils.config import (  # 导入 get_settings 函数
    ModelDetail,
    ModelProtocol,
    get_settings,
    require_detail_attributes,
    require_model_name,
    require_provider_name,
    safe_repr,
)
from src.utils.log_manager import get_module_logger  # 导入日志管理器
from src.utils.security import safe_exception_text, validate_secret_free_options

logger = get_module_logger(__name__)  # 获取当前模块的日志器


def _normalize_protocol_text(value: Any, *, label: str) -> str:
    """把协议名收敛成小写、连字符转下划线的形态。

    ``protocol_status`` 与 ``capability_report`` 是公开的诊断入口（配置界面与
    排障工具会调），参数来自调用方而非 ``Settings``——``ModelDetail.protocol``
    有 validator，但这两个入口收的是**裸取值**，没有任何校验。此前直接写
    ``str(value).strip().lower()``，于是 ``10**4301`` 撞上 CPython 的整数位数
    上限，抛的是 ``ValueError: Exceeds the limit (4300 digits) for integer
    string conversion``——既没有「哪个参数错了」，也没有「可选值是什么」，
    用户拿到的是与协议诊断毫无关系的整数转换错误。

    非字符串一律显式拒绝：协议名只能是字符串，``None``／列表／整数都说明调用方
    传错了参数，而不是「某个协议没被支持」。
    """
    if not isinstance(value, str):
        raise ValueError(f"{label} 必须是字符串，但得到 {safe_repr(value)}。")
    return value.strip().lower().replace("-", "_")


def _normalize_role_text(value: Any) -> str:
    """把角色名收敛成小写、去空白的形态。

    与 ``_normalize_protocol_text`` 同一判据、同一理由：``str(10**4301)`` 自己就抛
    ``ValueError: Exceeds the limit (4300 digits)``，于是「角色名写错了」变成整数
    转换错误。三个公开诊断入口（``capabilities``／``capability_report``／
    ``protocol_status``）此前各自写 ``str(role).strip().lower()``，**早于**
    ``_provider_info_for_role`` 内部的守卫，因此那里单独加的判据覆盖不到它们。
    """
    if not isinstance(value, str):
        raise ValueError(f"模型角色必须是字符串，但得到 {safe_repr(value)}。")
    return value.strip().lower()


class ModelProviderFactory:
    """模型提供商工厂"""

    _roles = frozenset({"llm", "embedding", "rerank"})

    _role_aliases: ClassVar[dict[str, dict[str, str]]] = {
        "rerank": {
            # SiliconFlow 的聊天/嵌入和 Rerank 使用不同的协议实现。
            "siliconflow": "siliconflow_rerank",
        }
    }

    _provider_map: ClassVar[dict[str, dict[str, Any]]] = {
        # LLM & Embedding Providers
        "google": {
            "module": "src.providers.google",
            "class": "GoogleProvider",
            "sdk_package": "google-genai",
            "integration": "native_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {ModelProtocol.GENERATE_CONTENT.value},
        },
        "openai": {
            "module": "src.providers.openai",
            "class": "OpenAIProvider",
            "sdk_package": "openai",
            "integration": "native_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            # 这里不把 SDK 契约冒充成任意配置 Base URL 的服务端验收。
            # `protocol_status` 仅在显式登记真实渠道证据后返回 true。
            "server_verified_protocols": set(),
        },
        "anthropic": {
            "module": "src.providers.anthropic",
            "class": "AnthropicProvider",
            "sdk_package": "anthropic",
            "integration": "native_sdk",
            "capabilities": {"llm"},
            "protocols": {ModelProtocol.MESSAGES.value},
        },
        "qwen": {
            "module": "src.providers.qwen",
            "class": "QwenProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "volcengine": {
            "module": "src.providers.volcengine",
            "class": "VolcengineProvider",
            "sdk_package": "volcengine-python-sdk",
            "integration": "native_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.ARK.value,
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "siliconflow": {
            "module": "src.providers.siliconflow",
            "class": "SiliconflowProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "ollama": {
            "module": "src.providers.ollama",
            "class": "OllamaProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "lm-studio": {
            "module": "src.providers.lm_studio",
            "class": "LMStudioProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm", "embedding"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "deepseek": {
            "module": "src.providers.deepseek",
            "class": "DeepSeekProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "grok": {
            "module": "src.providers.grok",
            "class": "GrokProvider",
            "sdk_package": "openai",
            "integration": "openai_compatible_sdk",
            "capabilities": {"llm"},
            "protocols": {
                ModelProtocol.CHAT_COMPLETIONS.value,
                ModelProtocol.RESPONSES.value,
            },
            "server_verified_protocols": set(),
        },
        "local-hash": {
            "module": "src.providers.local_hash",
            "class": "LocalHashEmbeddingProvider",
            "integration": "local",
            "capabilities": {"embedding"},
        },
        # Rerank Providers
        "jina": {
            "module": "src.providers.jina",
            "class": "JinaProvider",
            "sdk_package": "httpx",
            "integration": "http_json",
            "capabilities": {"rerank"},
        },
        "siliconflow_rerank": {
            "module": "src.providers.siliconflow_rerank",
            "class": "SiliconflowRerankProvider",
            "sdk_package": "httpx",
            "integration": "http_json",
            "capabilities": {"rerank"},
        },
    }

    @staticmethod
    @cache
    def _get_provider_class(provider_name: str) -> type:
        """动态导入并缓存提供商类，避免每次创建实例都重复导入模块。"""
        if provider_name not in ModelProviderFactory._provider_map:
            logger.error(f"不支持的模型提供商: {provider_name}")
            raise ValueError(f"不支持的模型提供商: {provider_name}")

        provider_info = ModelProviderFactory._provider_map[provider_name]
        try:
            module = importlib.import_module(provider_info["module"])
            ProviderClass = getattr(module, provider_info["class"])
            logger.debug(
                f"成功加载提供商类: {provider_info['class']} from {provider_info['module']}"
            )
            return ProviderClass
        except ImportError as e:
            logger.error(
                "导入提供商模块失败: %s - %s",
                provider_info["module"],
                safe_exception_text(e),
            )
            raise ImportError(f"无法加载提供商模块: {provider_info['module']}") from e
        except AttributeError as e:
            logger.error(
                "在模块中找不到提供商类: %s in %s - %s",
                provider_info["class"],
                provider_info["module"],
                safe_exception_text(e),
            )
            raise AttributeError(f"无法找到提供商类: {provider_info['class']}") from e

    @staticmethod
    def _resolve_provider_name(provider_name: str, role: str) -> str:
        return ModelProviderFactory._role_aliases.get(role, {}).get(provider_name, provider_name)

    @classmethod
    def _provider_info_for_role(cls, provider_name: str, role: str) -> tuple[str, dict[str, Any]]:
        # 键形态必须在任何成员判断**之前**收敛，判据与同文件的
        # ``_get_model_detail`` 逐字一致。这里此前完全漏了，而本函数是三个**公开**
        # 诊断入口（``capabilities``／``capability_report``／``protocol_status``，
        # 源码注释写明「配置界面和诊断使用」）的共同路径。实测修复前：
        #   * ``["a"]``／``{"k": 1}`` —— 下面 ``_role_aliases.get(role, {}).get(...)``
        #     与 ``_provider_map.get(...)`` 的字典查找抛裸 ``TypeError:
        #     unhashable type: 'list'``；
        #   * ``10**4301`` —— ``f"不支持的模型提供商: {provider_name}"`` 抛
        #     ``ValueError: Exceeds the limit (4300 digits)``。
        # 两条都不是「渠道名写错了」，而同一形态走 ``get_llm_provider`` 得到的是
        # ``_get_model_detail`` 给出的带指引 ``ValueError``——同一个错误的收场
        # 不该取决于调用的是哪条入口。
        if not isinstance(provider_name, str):
            raise ValueError(
                f"provider 名称必须是字符串，但得到 {safe_repr(provider_name)}。"
                " 请检查配置里的 *provider 字段。"
            )
        # ``role`` 同样要收敛（详见 ``_normalize_role_text``）。
        normalized_role = _normalize_role_text(role)
        if normalized_role not in cls._roles:
            allowed = ", ".join(sorted(cls._roles))
            raise ValueError(f"不支持的模型角色: {role}。可选值: {allowed}")
        resolved = cls._resolve_provider_name(provider_name, normalized_role)
        info = cls._provider_map.get(resolved)
        if info is None:
            raise ValueError(f"不支持的模型提供商: {provider_name}")
        capabilities = info.get("capabilities", set())
        if normalized_role not in capabilities:
            raise TypeError(f"提供商 {provider_name} 不支持 {normalized_role} 能力。")
        return resolved, info

    @staticmethod
    def _create_provider(
        provider_name: str,
        model_name: str,
        role: str,
        expected_type: type,
        protocol: ModelProtocol | str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        # 形态先收敛：本方法把 ``provider_name`` 拿去 ``_role_aliases.get(...)``／
        # ``_provider_map.get(...)`` 做字典查找、``_get_provider_class`` 做成员判定，
        # 并在「不支持的模型提供商」消息里 f-string 插值——不可哈希取值抛裸
        # ``TypeError: unhashable type: 'list'``、位数超限的整数抛
        # ``ValueError: Exceeds the limit``，都不是本模块其余拒绝路径承诺的带指引
        # ``ValueError``。``model_name`` 原样透传给 provider 构造器（``local-hash``
        # 直接 ``.rsplit()``，非字符串抛裸 ``AttributeError``）。
        #
        # 判据只做了一半：诊断入口 ``_provider_info_for_role``（``832abf0``）与
        # 展示层／``RetrievalService`` 都已用 ``require_provider_name`` 收敛，
        # 唯独**真正建 provider** 的这里漏了。三个 getter 走 ``_get_model_detail``
        # 也会到达本方法，因此这一处同时覆盖「getter 路径」与「直接调
        # ``_create_provider`` 的嵌入方／测试」。
        provider_name = require_provider_name(provider_name, label="provider 名称")
        model_name = require_model_name(model_name)
        # ``ModelDetail`` normally normalizes protocol values before they reach
        # the factory, but this internal helper is also used directly by
        # integrations and tests. Normalize again at the boundary so a plain
        # string does not fail later with ``AttributeError: value``.
        protocol_value: str | None = None
        if protocol is not None:
            normalized_protocol = ModelDetail.normalize_protocol(protocol)
            if normalized_protocol is None:
                raise ValueError("provider protocol 不能为空。")
            protocol_value = normalized_protocol.value
        resolved_provider_name = ModelProviderFactory._resolve_provider_name(provider_name, role)
        provider_info = ModelProviderFactory._provider_map.get(resolved_provider_name)
        if provider_info is not None:
            capabilities = provider_info.get("capabilities")
            if capabilities is not None and role not in capabilities:
                raise TypeError(f"提供商 {provider_name} 不支持 {role} 能力。")
            if protocol_value is not None:
                supported_protocols = provider_info.get("protocols", set())
                if protocol_value not in supported_protocols:
                    raise ValueError(f"提供商 {provider_name} 不支持 {protocol_value} 协议。")
        provider_class = ModelProviderFactory._get_provider_class(resolved_provider_name)
        if not issubclass(provider_class, expected_type):
            raise TypeError(
                f"提供商 {provider_name} 不支持 {role} 能力，需要实现 {expected_type.__name__}。"
            )
        provider_options = dict(options or {})
        parameters = inspect.signature(provider_class).parameters
        accepts_options = "options" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {"model_name": model_name}
        if accepts_options:
            kwargs["options"] = provider_options
        elif provider_options:
            unsupported = ", ".join(sorted(str(key) for key in provider_options))
            raise ValueError(
                f"提供商 {provider_name} 的构造函数不支持 options，但配置提供了: {unsupported}"
            )
        if protocol_value is None:
            instance = provider_class(**kwargs)
            return instance

        # Google/Anthropic/Volcengine 的协议固定在各自 SDK 中，不接受
        # protocol 构造参数；OpenAI 兼容 provider 才需要显式选择协议。
        parameters = inspect.signature(provider_class).parameters
        if "protocol" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            if (
                protocol_value is not None
                and provider_info is not None
                and protocol_value not in provider_info.get("protocols", set())
            ):
                raise ValueError(
                    f"提供商 {provider_name} 的构造函数不支持 protocol={protocol_value}。"
                )
            instance = provider_class(**kwargs)
            return instance
        kwargs["protocol"] = protocol_value
        instance = provider_class(**kwargs)
        if protocol_value == ModelProtocol.RESPONSES.value:
            # OpenAI-compatible adapters need explicit remote capability evidence
            # before exposing Responses resources. Native providers (for example
            # Google and Anthropic) do not implement this guard.
            require = getattr(instance, "_require_responses_resource", None)
            if callable(require):
                require("factory")
        return instance

    @staticmethod
    def _runtime_verified_protocols(
        options: Mapping[str, Any] | None,
        supported_protocols: set[str] | frozenset[str] | None = None,
        provider_name: str | None = None,
    ) -> frozenset[str]:
        """读取诊断调用使用的实例级服务端协议登记。"""
        if options is None:
            return frozenset()
        # 容器形态必须先判，且要用与 ``validate_secret_free_options`` 相同的
        # ``ValueError``：``"http_options" in options`` 对**列表**是元素成员判断
        # （``["http_options"]`` 会命中 google 分支），对整数抛裸
        # ``TypeError: argument of type 'int' is not iterable``，命中后
        # ``options.items()`` 又抛裸 ``AttributeError``。``capability_report`` 与
        # ``protocol_status`` 是公开诊断入口，调用方按 ``ValueError`` 分类；
        # 非 google 渠道走 else 分支时本来就得到 ``ValueError``，两个分支的
        # 异常分类必须一致。
        if not isinstance(options, Mapping):
            raise ValueError("Provider options 必须是对象。")
        if str(provider_name or "").strip().lower() == "google" and "http_options" in options:
            # ``http_options`` is a supported google-genai SDK container. Keep
            # the general options boundary strict while allowing only this
            # explicitly supported container and its safe trace headers.
            remaining = {key: value for key, value in options.items() if key != "http_options"}
            validated = validate_secret_free_options(remaining, "Provider")
            validated["http_options"] = validate_secret_free_options(
                {"http_options": options["http_options"]},
                "Provider",
                allowed_containers={"http_options", "headers"},
            )["http_options"]
        else:
            validated = validate_secret_free_options(options, "Provider")
        configured = validated.get("server_verified_protocols", ())
        if isinstance(configured, str):
            configured = (configured,)
        if not isinstance(configured, (list, tuple, set, frozenset)):
            raise ValueError("server_verified_protocols 必须是字符串或协议序列。")
        aliases = {
            "chat": ModelProtocol.CHAT_COMPLETIONS.value,
            "completion": ModelProtocol.CHAT_COMPLETIONS.value,
            "chat_completion": ModelProtocol.CHAT_COMPLETIONS.value,
            "response": ModelProtocol.RESPONSES.value,
            "anthropic": ModelProtocol.MESSAGES.value,
            "anthropic_messages": ModelProtocol.MESSAGES.value,
            "gemini": ModelProtocol.GENERATE_CONTENT.value,
            "generate": ModelProtocol.GENERATE_CONTENT.value,
        }
        normalized: set[str] = set()
        for value in configured:
            protocol = _normalize_protocol_text(value, label="server_verified_protocols")
            protocol = aliases.get(protocol, protocol)
            try:
                normalized_protocol = ModelProtocol(protocol).value
            except ValueError as exc:
                supported = ", ".join(item.value for item in ModelProtocol)
                raise ValueError(
                    f"不支持的 server_verified_protocols 协议: {value}。可选值: {supported}"
                ) from exc
            if supported_protocols is not None and normalized_protocol not in supported_protocols:
                supported = ", ".join(sorted(supported_protocols)) or "无"
                raise ValueError(
                    f"server_verified_protocols 中的协议 {value} 不适用于当前提供商；"
                    f"支持: {supported}"
                )
            normalized.add(normalized_protocol)
        return frozenset(normalized)

    @classmethod
    def capabilities(cls, provider_name: str, role: str = "llm") -> frozenset[str]:
        """返回指定角色的渠道能力集合，供 UI/配置校验使用。"""
        normalized_role = _normalize_role_text(role)
        resolved, _info = cls._provider_info_for_role(provider_name, normalized_role)
        provider_class = cls._get_provider_class(resolved)
        if normalized_role in {"embedding", "rerank"}:
            return frozenset({normalized_role})
        declared: Any = getattr(provider_class, "capabilities", frozenset())
        # The factory's role map is the authoritative source for which model
        # kinds can be instantiated.  Do not leak a shared adapter's embedding
        # or rerank methods into an LLM capability report.
        return (frozenset(declared) - {"embedding", "rerank"}) | {normalized_role}

    @classmethod
    def capability_report(
        cls,
        provider_name: str,
        role: str = "llm",
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """返回可供配置界面和诊断使用的机器可读能力报告。

        ``capabilities`` 表示本地适配器已经实现的能力；协议的
        ``server_verified`` 仍只表示目标服务端有明确验证证据，不会由 SDK
        的本地资源树自动推导。
        """
        normalized_role = _normalize_role_text(role)
        resolved, info = cls._provider_info_for_role(provider_name, normalized_role)
        provider_class = cls._get_provider_class(resolved)
        sdk_package = info.get("sdk_package")
        try:
            sdk_version = package_version(sdk_package) if sdk_package else None
        except PackageNotFoundError:
            sdk_version = None
        # `protocols` describes LLM wire protocols. Embedding and Rerank use
        # their own endpoint contracts and must not inherit chat protocol names.
        protocols = sorted(info.get("protocols", set())) if normalized_role == "llm" else []
        # 与 protocol_status 对齐：非 llm 角色的协议不适用，其 server_verified
        # 登记既不能被报告，也不能被 _runtime_verified_protocols 接纳。
        verified = (
            set(info.get("server_verified_protocols", set())) if normalized_role == "llm" else set()
        )
        verified.update(
            cls._runtime_verified_protocols(
                options,
                set(info.get("protocols", set())) if normalized_role == "llm" else set(),
                resolved,
            )
        )
        verified_list = sorted(verified)
        return {
            "provider": provider_name,
            "resolved_provider": resolved,
            "role": normalized_role,
            "provider_class": f"{provider_class.__module__}.{provider_class.__name__}",
            "integration": info.get("integration", "unknown"),
            "sdk": {
                "package": sdk_package,
                "version": sdk_version,
            },
            "capabilities": sorted(cls.capabilities(provider_name, normalized_role)),
            "protocols": protocols,
            "server_verified_protocols": verified_list,
            "protocol_status": {
                protocol: {
                    "adapter_supported": True,
                    "server_verified": protocol in verified,
                }
                for protocol in protocols
            },
        }

    @classmethod
    def protocol_status(
        cls,
        provider_name: str,
        protocol: str,
        options: Mapping[str, Any] | None = None,
        role: str | None = None,
    ) -> dict[str, bool]:
        """区分本地适配器支持与目标服务端已验证的协议。

        LLM 是默认角色；未显式指定时，对 embedding/rerank-only provider
        自动选择其唯一角色，以便诊断工具可以统一查询所有渠道。
        """
        # 形态判据必须在**最前**：下面 ``role is None`` 的推断分支会先对
        # ``provider_name`` 做两次字典查找（``_resolve_provider_name`` 与
        # ``_provider_map.get``），不可哈希的取值在那里就抛裸
        # ``TypeError: unhashable type``——早于 ``_provider_info_for_role`` 内部的
        # 守卫，因此那里单独加的判据覆盖不到本入口。实测修复前
        # ``protocol_status(["a"], "chat_completions")`` 复现。
        if not isinstance(provider_name, str):
            raise ValueError(
                f"provider 名称必须是字符串，但得到 {safe_repr(provider_name)}。"
                " 请检查配置里的 *provider 字段。"
            )
        if role is None:
            candidates = ("llm", "embedding", "rerank")
            normalized_role = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate
                    in cls._provider_map.get(
                        cls._resolve_provider_name(provider_name, candidate), {}
                    ).get("capabilities", set())
                ),
                None,
            )
            if normalized_role is None:
                raise ValueError(f"无法为提供商 {provider_name} 推断模型角色，请显式传入 role。")
        else:
            normalized_role = _normalize_role_text(role)
        resolved, info = cls._provider_info_for_role(provider_name, normalized_role)
        normalized = _normalize_protocol_text(protocol, label="protocol")
        normalized = {
            "chat": ModelProtocol.CHAT_COMPLETIONS.value,
            "completion": ModelProtocol.CHAT_COMPLETIONS.value,
            "chat_completion": ModelProtocol.CHAT_COMPLETIONS.value,
            "chat_completions": ModelProtocol.CHAT_COMPLETIONS.value,
            "response": ModelProtocol.RESPONSES.value,
            "responses": ModelProtocol.RESPONSES.value,
            "anthropic": ModelProtocol.MESSAGES.value,
            "anthropic_messages": ModelProtocol.MESSAGES.value,
            "gemini": ModelProtocol.GENERATE_CONTENT.value,
            "generate": ModelProtocol.GENERATE_CONTENT.value,
            "generate_content": ModelProtocol.GENERATE_CONTENT.value,
        }.get(normalized, normalized)
        adapter_supported = normalized in info.get("protocols", set())
        verified = set(info.get("server_verified_protocols", set()))
        verified.update(
            cls._runtime_verified_protocols(
                options,
                set(info.get("protocols", set())) if normalized_role == "llm" else set(),
                resolved,
            )
        )
        server_verified = normalized in verified
        return {"adapter_supported": adapter_supported, "server_verified": server_verified}

    @staticmethod
    def _get_model_detail(
        provider_key: str,
        configurations: Mapping[str, ModelDetail] | None,
        settings_field: str,
        role_label: str,
    ) -> ModelDetail:
        if configurations is None:
            configurations = getattr(get_settings(), settings_field)
        # 容器形态要**先**判：``provider_key not in configurations`` 对不可迭代的
        # 取值抛裸 ``TypeError: argument of type 'int' is not iterable``，而字符串
        # 容器会走**子串**判定（``"k" in "k"`` 为真），随后
        # ``"k"["k"]`` 抛 ``TypeError: string indices must be integers``。两条都
        # 不是 ``ValueError``。取值来自可直写的 ``SessionConfig.llm_configurations``，
        # 经 ``Chatbot.reload_llm`` 原样传到这里。
        if not isinstance(configurations, Mapping):
            raise ValueError(
                f"{settings_field} 必须是配置映射，但得到 {type(configurations).__name__}。"
                f" 请检查 {role_label} 的当前配置项。"
            )
        # 键形态同样要在成员判断**之前**收敛：``key not in dict`` 对不可哈希的取值抛
        # 裸 ``TypeError: unhashable type: 'list'``（不是 ``ValueError``，穿透
        # ``Chatbot.reload_llm`` 的 ``except Exception`` 分类与上游的
        # ``except ValueError``），而消息里的 f-string 对位数超限的整数抛
        # ``ValueError: Exceeds the limit``——类型对，内容与字段无关。
        # ``key`` 来源是 ``SessionConfig.active_llm_configuration`` 等可直写属性。
        # 同一判据在 ``src/runtime/contracts.py`` 的 ``_require_known_field`` 里
        # 已认领过（同样的 ``key not in ...`` + f-string 组合），这里对齐。
        if not isinstance(provider_key, str):
            raise ValueError(
                f"{settings_field} 的 key 必须是字符串，但得到 {safe_repr(provider_key)}。"
                f" 请检查 {role_label} 的当前配置项。"
            )
        if provider_key not in configurations:
            # 消息与 ``display_utils._require_model_detail`` 的同名守卫对齐：
            # 带可选值与指引——同一判据两套消息会让用户经 ``reload_llm`` 看到
            # 残缺版（只有「未找到 key」），无从知道有哪些可填的键。
            available = "、".join(sorted(str(k) for k in configurations)) or "无"
            logger.error(f"在{role_label}配置中未找到key: {provider_key}。可选值: {available}。")
            raise ValueError(
                f"在{role_label}配置中未找到key: {safe_repr(provider_key)}。"
                f"可选值: {available}。"
                " 请检查 config.toml 里的 default_*_provider 与 *_configurations 是否对应。"
            )
        # 取值的形态也要在这里收敛：``key in mapping`` 只证明**键**在，不证明值
        # 可用。``{"k": None}`` 的键确实存在，于是不会走上面「未找到key」的分支，
        # 而是把 ``None`` 原样返回给调用方——三个入口紧接着都读 ``config.provider``
        # 等属性，在那里抛裸 ``AttributeError``（不是 ``ValueError``，穿透
        # ``Chatbot.reload_llm`` 的 ``except ValueError`` 分类）。
        # 值来源是 ``SessionConfig.llm_configurations`` 等可直写映射。
        detail = configurations[provider_key]
        require_detail_attributes(
            detail,
            ("provider", "model_name", "protocol", "options"),
            field=settings_field,
            key=provider_key,
            role_label=role_label,
        )
        # 属性**存在**不代表取值可用：``ModelDetail`` 的构造器会拒绝非字符串
        # ``provider``／``model_name``，但属性赋值没有 ``__setattr__`` 守卫（实测
        # ``detail.provider = ["a"]`` 直接成功），``SimpleNamespace`` 替身同样能提供
        # 这些属性。三个 getter 紧接着读 ``config.provider``／``config.model_name``
        # 并写进 ``logger.info`` 的 f-string，随后交给 ``_create_provider``——前者对
        # 位数超限的整数抛 ``ValueError: Exceeds the limit``，后者对不可哈希取值抛
        # 裸 ``TypeError``。诊断入口 ``_provider_info_for_role`` 早已按同一判据收敛，
        # 这里对齐（``_create_provider`` 也各自判一次，覆盖直接调用它的嵌入方）。
        require_provider_name(detail.provider, label=f"{role_label}配置 {provider_key} 的 provider")
        require_model_name(
            detail.model_name, label=f"{role_label}配置 {provider_key} 的 model_name"
        )
        return detail

    @staticmethod
    def get_llm_provider(
        provider_key: str,
        configurations: Mapping[str, ModelDetail] | None = None,
    ) -> LargeLanguageModel:
        """获取一个语言模型提供商实例"""
        config = ModelProviderFactory._get_model_detail(
            provider_key, configurations, "llm_configurations", "LLM"
        )
        provider_name = config.provider
        model_name = config.model_name
        logger.info(f"正在获取LLM提供商: {provider_name}, 模型: {model_name}")

        try:
            instance = ModelProviderFactory._create_provider(
                provider_name,
                model_name,
                "llm",
                LargeLanguageModel,
                config.protocol,
                config.options,
            )
            logger.info(f"成功获取LLM提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(
                "获取LLM提供商实例失败: %s (%s) - %s",
                provider_name,
                model_name,
                safe_exception_text(e),
            )
            raise

    @staticmethod
    def get_embedding_provider(
        provider_key: str,
        configurations: Mapping[str, ModelDetail] | None = None,
    ) -> TextEmbeddingModel:
        """获取一个文本向量化模型提供商实例"""
        config = ModelProviderFactory._get_model_detail(
            provider_key, configurations, "embedding_configurations", "Embedding"
        )
        provider_name = config.provider
        model_name = config.model_name
        logger.info(f"正在获取Embedding提供商: {provider_name}, 模型: {model_name}")

        try:
            if config.protocol is not None:
                raise ValueError(
                    "Embedding 配置不支持 protocol；协议只能配置在 llm_configurations 中。"
                )
            instance = ModelProviderFactory._create_provider(
                provider_name, model_name, "embedding", TextEmbeddingModel, options=config.options
            )
            logger.info(f"成功获取Embedding提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(
                "获取Embedding提供商实例失败: %s (%s) - %s",
                provider_name,
                model_name,
                safe_exception_text(e),
            )
            raise

    @staticmethod
    def get_rerank_provider(
        provider_key: str,
        configurations: Mapping[str, ModelDetail] | None = None,
    ) -> RerankModel:
        """获取一个Rerank模型提供商实例"""
        config = ModelProviderFactory._get_model_detail(
            provider_key, configurations, "rerank_configurations", "Rerank"
        )
        provider_name = config.provider
        model_name = config.model_name
        logger.info(f"正在获取Rerank提供商: {provider_name}, 模型: {model_name}")

        try:
            if config.protocol is not None:
                raise ValueError(
                    "Rerank 配置不支持 protocol；协议只能配置在 llm_configurations 中。"
                )
            instance = ModelProviderFactory._create_provider(
                provider_name, model_name, "rerank", RerankModel, options=config.options
            )
            logger.info(f"成功获取Rerank提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(
                "获取Rerank提供商实例失败: %s (%s) - %s",
                provider_name,
                model_name,
                safe_exception_text(e),
            )
            raise
