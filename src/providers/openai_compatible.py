# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import openai

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

class OpenAICompatibleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    处理所有与OpenAI API格式兼容的提供商的通用逻辑。
    """

    def __init__(self, model_name: str, provider: str):
        self._model_name = model_name
        self._provider = provider
        logger.info(f"初始化 OpenAICompatibleProvider，提供商: {provider}, 模型: {model_name}")

        # 从 get_settings() 获取配置
        current_settings = get_settings()
        api_key_name = f"{provider.lower()}_api_key"
        base_url_name = f"{provider.lower()}_base_url"

        api_key = getattr(current_settings, api_key_name, None)
        base_url = getattr(current_settings, base_url_name, None)
        logger.debug(f"获取配置：API Key Name: {api_key_name}, Base URL Name: {base_url_name}")

        # 特殊处理ollama和lm-studio，它们通常不需要key
        if provider in ["ollama", "lm-studio"] and not api_key:
            api_key = "no-key-required"
            logger.info(f"提供商 '{provider}' 不需要 API Key，使用默认值。")

        if not api_key:
            logger.error(f"{provider} 配置不完整：缺少 {api_key_name}。")
            raise ValueError(f"{provider} 配置不完整：缺少 {api_key_name}。")
        if not base_url and provider not in ["openai"]: # OpenAI官方库可以不填base_url
            logger.error(f"{provider} 配置不完整：缺少 {base_url_name}。")
            raise ValueError(f"{provider} 配置不完整：缺少 {base_url_name}。")

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAI兼容客户端初始化成功，Base URL: {base_url}")

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
        logger.info(f"调用 OpenAI兼容LLM ({self._provider}/{self._model_name})，流式: {stream}, 温度: {temperature}")
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
                logger.debug(f"LLM 调用包含 {len(tools)} 个工具。")

            stream_response = self._client.chat.completions.create(**request_params)
            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            logger.info("OpenAI兼容LLM 内容生成完成。")
        except Exception as e:
            error_message = f"OpenAI兼容LLM ({self._provider}/{self._model_name}) 生成内容时出错: {e}"
            logger.error(error_message, exc_info=True) # 记录详细异常信息
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用与OpenAI兼容的Embedding模型。
        """
        logger.info(f"调用 OpenAI兼容嵌入 ({self._provider}/{self._model_name})，文档数量: {len(texts)}")
        try:
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            embeddings = [item.embedding for item in response.data]
            logger.info(f"OpenAI兼容嵌入完成，生成 {len(embeddings)} 个嵌入。")
            return embeddings
        except Exception as e:
            error_message = f"OpenAI兼容嵌入 ({self._provider}/{self._model_name}) 时出错: {e}"
            logger.error(error_message, exc_info=True) # 记录详细异常信息
            return [[] for _ in texts]
