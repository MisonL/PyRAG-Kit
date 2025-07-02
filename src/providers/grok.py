# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import requests

from src.providers.__base__.model_provider import LargeLanguageModel
from src.utils.config import settings


class GrokProvider(LargeLanguageModel):
    """
    Grok模型提供商。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._api_key = settings.grok_api_key
        if not self._api_key:
            raise ValueError("Grok配置不完整：缺少 GROK_API_KEY。")
        self._base_url = str(settings.grok_base_url)

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用Grok模型。
        Grok API当前不支持流式响应，因此我们返回一个包含完整结果的生成器。
        """
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        
        # Grok API 不直接支持 system_prompt，但可以将其作为第一条消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self._model_name, 
            "messages": messages,
            "temperature": temperature,
            # "tools": tools, # Grok API 可能不支持或有不同的工具格式
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            yield content.strip()
        except Exception as e:
            error_message = f"Grok LLM ({self._model_name}) 生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"
