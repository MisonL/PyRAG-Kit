# -*- coding: utf-8 -*-
from src.providers.openai_compatible import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    """Grok 模型提供商。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, provider="grok")
