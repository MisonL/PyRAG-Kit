# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, Generator, List, AsyncGenerator

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class OpenAICompatibleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    处理所有与OpenAI API格式兼容的提供商的通用逻辑。
    已注入 CSE 性能传感器与 tenacity 重试机制。
    """

    def __init__(self, model_name: str, provider: str):
        self._model_name = model_name
        self._provider = provider
        settings = get_settings()
        
        api_key_name = f"{provider.lower()}_api_key"
        base_url_name = f"{provider.lower()}_base_url"
        
        self._api_key = getattr(settings, api_key_name, None)
        self._base_url = getattr(settings, base_url_name, None)
        
        if provider in ["ollama", "lm-studio"] and not self._api_key:
            self._api_key = "no-key-required"
            
        if not self._api_key:
            logger.error(f"{provider} API Key 未设置。")
            raise ValueError(f"{api_key_name} is required for {provider}")
        
        self._client: Optional[openai.OpenAI] = None
        self._aclient: Optional[openai.AsyncOpenAI] = None
        logger.info(f"初始化 OpenAICompatibleProvider ({provider})，模型: {model_name}")

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _get_aclient(self) -> openai.AsyncOpenAI:
        if self._aclient is None:
            self._aclient = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._aclient

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """同步调用 OpenAI 兼容 LLM (CSE Sensor)。"""
        logger.info(f"调用 {self._provider} LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            
            if stream:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                if response.choices and response.choices[0].message.content:
                    yield response.choices[0].message.content
            
            duration = time.perf_counter() - start_time
            logger.info(f"{self._provider} LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"{self._provider} LLM ({self._model_name}) 出错: {e}", exc_info=True)
            yield f"抱歉，提供商 {self._provider} 遇到错误: {str(e)}"

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """异步调用 OpenAI 兼容 LLM (CSE Sensor)。"""
        logger.info(f"异步调用 {self._provider} LLM ({self._model_name})，流式: {stream}")
        aclient = self._get_aclient()
        start_time = time.perf_counter()
        
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            response = await aclient.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            
            if stream:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                if response.choices and response.choices[0].message.content:
                    yield response.choices[0].message.content
            
            duration = time.perf_counter() - start_time
            logger.info(f"{self._provider} LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"{self._provider} LLM ({self._model_name}) 异步出错: {e}", exc_info=True)
            yield f"抱歉，提供商 {self._provider} 异步处理遇到错误: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步向量化文档。"""
        logger.info(f"调用 {self._provider} 嵌入 ({self._model_name})，数量: {len(texts)}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            response = client.embeddings.create(input=texts, model=self._model_name)
            embeddings = [item.embedding for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(f"{self._provider} 嵌入 ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"{self._provider} 嵌入 ({self._model_name}) 出错: {e}", exc_info=True)
            return [[] for _ in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步向量化文档。"""
        logger.info(f"异步调用 {self._provider} 嵌入 ({self._model_name})，数量: {len(texts)}")
        aclient = self._get_aclient()
        start_time = time.perf_counter()
        
        try:
            response = await aclient.embeddings.create(input=texts, model=self._model_name)
            embeddings = [item.embedding for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(f"{self._provider} 嵌入 ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"{self._provider} 嵌入 ({self._model_name}) 异步出错: {e}", exc_info=True)
            return [[] for _ in texts]
