from typing import Any

from src.providers.openai_compatible import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    """Grok 模型提供商。"""

    def __init__(self, model_name: str, protocol: str = "chat_completions", options: dict[str, Any] | None = None):
        super().__init__(model_name=model_name, provider="grok", protocol=protocol, options=options)
