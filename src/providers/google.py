# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, Generator, List, cast, AsyncGenerator

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class GoogleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    Google模型提供商，统一处理Gemini LLM和Embedding。
    已升级为使用最新的 google-genai SDK (v1.0+)。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        logger.info(f"初始化 GoogleProvider (google-genai)，模型: {model_name}")
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """延迟初始化并返回 genai.Client 实例。"""
        if not self._client:
            settings = get_settings()
            if not settings.google_api_key:
                logger.error("Google API Key 未设置。")
                raise ValueError("GOOGLE_API_KEY is required for GoogleProvider")
            
            logger.debug("正在初始化 google-genai Client")
            self._client = genai.Client(api_key=settings.google_api_key)
            logger.info("google-genai Client 初始化成功。")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """同步调用 Google LLM (CSE Sensor)。"""
        logger.info(f"调用 Google LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature)

        try:
            if stream:
                response = client.models.generate_content_stream(model=self._model_name, contents=prompt, config=config)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                response = client.models.generate_content(model=self._model_name, contents=prompt, config=config)
                if response.text:
                    yield response.text
            
            duration = time.perf_counter() - start_time
            logger.info(f"Google LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Google LLM ({self._model_name}) 出错: {e}", exc_info=True)
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
        system_prompt: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """异步调用 Google LLM (CSE Sensor)。"""
        logger.info(f"异步调用 Google LLM ({self._model_name})，流式: {stream}")
        client = self._get_client()
        start_time = time.perf_counter()
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature)

        try:
            if stream:
                response = await client.aio.models.generate_content_stream(model=self._model_name, contents=prompt, config=config)
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                response = await client.aio.models.generate_content(model=self._model_name, contents=prompt, config=config)
                if response.text:
                    yield response.text
            
            duration = time.perf_counter() - start_time
            logger.info(f"Google LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Google LLM ({self._model_name}) 异步出错: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步向量化文档。"""
        logger.info(f"调用 Google Embedding ({self._model_name})，数量: {len(texts)}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            response = client.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings = [list(e.values) for e in response.embeddings]
            duration = time.perf_counter() - start_time
            logger.info(f"Google Embedding ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"Google Embedding ({self._model_name}) 出错: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步向量化文档。"""
        logger.info(f"异步调用 Google Embedding ({self._model_name})，数量: {len(texts)}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            response = await client.aio.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings = [list(e.values) for e in response.embeddings]
            duration = time.perf_counter() - start_time
            logger.info(f"Google Embedding ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"Google Embedding ({self._model_name}) 异步出错: {e}", exc_info=True)
            raise
