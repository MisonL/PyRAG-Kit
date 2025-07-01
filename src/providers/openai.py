# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import openai

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import API_CONFIG


class OpenAIProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    OpenAI模型提供商，统一处理GPT LLM和Embedding。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        api_key = API_CONFIG.get("OPENAI_API_KEY")
        base_url = API_CONFIG.get("OPENAI_API_BASE")

        if not api_key:
            raise ValueError("OpenAI配置不完整：缺少 OPENAI_API_KEY。")

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
        调用OpenAI GPT模型。
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        
        try:
            # 构造请求参数，只有在tools不为None时才加入
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
            error_message = f"OpenAI LLM ({self._model_name}) 生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用OpenAI Embedding模型将文档列表向量化。
        """
        try:
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"OpenAI嵌入文档时出错 ({self._model_name}): {e}")
            return [[] for _ in texts]
