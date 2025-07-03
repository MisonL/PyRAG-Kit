# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import openai

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import settings


class OpenAICompatibleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    处理所有与OpenAI API格式兼容的提供商的通用逻辑。
    """

    def __init__(self, model_name: str, provider: str):
        self._model_name = model_name
        self._provider = provider

        # 从 settings 对象动态获取配置
        api_key_name = f"{provider.lower()}_api_key"
        base_url_name = f"{provider.lower()}_base_url"

        api_key = getattr(settings, api_key_name, None)
        base_url = getattr(settings, base_url_name, None)

        # 特殊处理ollama和lm-studio，它们通常不需要key
        if provider in ["ollama", "lm-studio"] and not api_key:
            api_key = "no-key-required"

        if not api_key:
            raise ValueError(f"{provider} 配置不完整：缺少 {api_key_name}。")
        if not base_url and provider not in ["openai"]: # OpenAI官方库可以不填base_url
            raise ValueError(f"{provider} 配置不完整：缺少 {base_url_name}。")

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用与OpenAI兼容的LLM。
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        
        try:
            request_params = {
                "model": self._model_name,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": temperature,
                "stream": stream,
            }
            if tools:
                request_params["tools"] = tools

            stream_response = self._client.chat.completions.create(**request_params)
            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            error_message = f"OpenAI兼容LLM ({self._provider}/{self._model_name}) 生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用与OpenAI兼容的Embedding模型。
        """
        try:
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"OpenAI兼容嵌入 ({self._provider}/{self._model_name}) 时出错: {e}")
            return [[] for _ in texts]
