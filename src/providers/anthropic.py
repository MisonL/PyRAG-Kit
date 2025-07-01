# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import anthropic

from src.providers.__base__.model_provider import LargeLanguageModel
from src.utils.config import API_CONFIG


class AnthropicProvider(LargeLanguageModel):
    """
    Anthropic模型提供商，处理Claude系列模型。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        api_key = API_CONFIG.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic配置不完整：缺少 ANTHROPIC_API_KEY。")
        self._client = anthropic.Anthropic(api_key=api_key)

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用Anthropic Claude模型。
        """
        # Anthropic API v3 使用 system 参数
        try:
            request_params = {
                "model": self._model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                # "tools": tools, # Anthropic的工具使用方式不同，需要单独适配
            }
            if system_prompt:
                request_params["system"] = system_prompt

            with self._client.messages.stream(**request_params) as stream_response:
                for text in stream_response.text_stream:
                    yield text
        except Exception as e:
            error_message = f"Anthropic LLM ({self._model_name}) 生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"
