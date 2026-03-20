# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, Generator, List, AsyncGenerator

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.__base__.model_provider import LargeLanguageModel
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class AnthropicProvider(LargeLanguageModel):
    """
    Anthropic模型提供商，处理Claude系列模型。
    已增加 tenacity 重试机制与性能传感器。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        settings = get_settings()
        self._api_key = settings.anthropic_api_key
        
        if not self._api_key:
            logger.error("Anthropic API Key 未设置。")
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        
        self._client: Optional[anthropic.Anthropic] = None
        self._aclient: Optional[anthropic.AsyncAnthropic] = None
        logger.info(f"初始化 AnthropicProvider，模型: {model_name}")

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _get_aclient(self) -> anthropic.AsyncAnthropic:
        if self._aclient is None:
            self._aclient = anthropic.AsyncAnthropic(api_key=self._api_key)
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
        """同步调用 Anthropic Claude LLM (CSE Sensor)。"""
        logger.info(f"调用 Anthropic LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            params = {
                "model": self._model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if system_prompt:
                params["system"] = system_prompt

            if stream:
                with client.messages.stream(**params) as response:
                    for text in response.text_stream:
                        yield text
            else:
                response = client.messages.create(**params)
                if response.content:
                    yield response.content[0].text
            
            duration = time.perf_counter() - start_time
            logger.info(f"Anthropic LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Anthropic LLM ({self._model_name}) 出错: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """异步调用 Anthropic Claude LLM (CSE Sensor)。"""
        logger.info(f"异步调用 Anthropic LLM ({self._model_name})，流式: {stream}")
        aclient = self._get_aclient()
        start_time = time.perf_counter()
        
        try:
            params = {
                "model": self._model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if system_prompt:
                params["system"] = system_prompt

            if stream:
                async with aclient.messages.stream(**params) as response:
                    async for text in response.text_stream:
                        yield text
            else:
                response = await aclient.messages.create(**params)
                if response.content:
                    yield response.content[0].text
            
            duration = time.perf_counter() - start_time
            logger.info(f"Anthropic LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Anthropic LLM ({self._model_name}) 异步出错: {e}", exc_info=True)
            raise
