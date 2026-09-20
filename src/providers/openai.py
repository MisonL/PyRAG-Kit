from typing import Any

from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.resources import AsyncOpenAIResources, OpenAIResources
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI 官方渠道，复用 OpenAI 兼容协议适配器。"""

    capabilities = OpenAICompatibleProvider.capabilities | frozenset(
        {
            "files", "batches", "vector_stores", "responses", "moderation",
            "audio", "images", "videos", "uploads", "parse", "conversations",
            "containers", "fine_tuning", "evals", "skills", "realtime",
            "webhooks", "admin", "content_provenance_checks",
        }
    )
    _resource_capabilities = frozenset(
        {
            "files", "batches", "vector_stores", "models", "moderation",
            "images", "audio", "videos", "uploads", "conversations",
            "containers", "fine_tuning", "evals",
        }
    )

    @property
    def resources(self) -> OpenAIResources:
        return OpenAIResources(self)

    @property
    def async_resources(self) -> AsyncOpenAIResources:
        return AsyncOpenAIResources(self)

    def __init__(self, model_name: str, protocol: str = "chat_completions", options: dict[str, Any] | None = None):
        # 保留本模块的配置入口，方便调用方在测试或运行时注入 settings。
        settings = get_settings()
        if not settings.openai_api_key:
            logger.error("OpenAI配置不完整：缺少 OPENAI_API_KEY。")
            raise ValueError("OpenAI配置不完整：缺少 OPENAI_API_KEY。")
        super().__init__(model_name=model_name, provider="openai", settings=settings, protocol=protocol, options=options)
