from typing import Any

from src.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """
    深度求索模型提供商。
    通过继承OpenAICompatibleProvider来复用与OpenAI API兼容的逻辑。
    """

    def __init__(self, model_name: str, protocol: str = "chat_completions", options: dict[str, Any] | None = None):
        super().__init__(model_name=model_name, provider="deepseek", protocol=protocol, options=options)

    def _chat_max_tokens_key(self) -> str:
        """DeepSeek Chat Completions 仍使用旧版 max_tokens 字段。"""
        return "max_tokens"
