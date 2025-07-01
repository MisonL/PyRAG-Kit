# -*- coding: utf-8 -*-
from src.providers.openai_compatible import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """
    阿里云通义千问模型提供商。
    通过继承OpenAICompatibleProvider来复用与OpenAI API兼容的逻辑。
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, provider="qwen")
