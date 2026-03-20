import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from volcengine.ark import Ark, AsyncArk
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class VolcengineProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    火山引擎模型提供商 (豆包 Ark SDK)。
    已注入 CSE 性能传感器与 tenacity 重试机制。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        settings = get_settings()
        self._api_key = settings.volc_access_key # 映射为 ak/sk 或直接使用 api_key (Ark 支持 API Key)
        # 注意：Ark SDK 推荐使用 API Key 模式，对应 volc_access_key
        if not self._api_key:
            logger.error("火山引擎 API Key (VOLC_ACCESS_KEY) 未设置。")
            raise ValueError("VOLC_ACCESS_KEY is required for VolcengineProvider")
        
        self._client: Optional[Ark] = None
        self._aclient: Optional[AsyncArk] = None
        logger.info(f"初始化 VolcengineProvider (Ark)，模型: {model_name}")

    def _get_client(self) -> Ark:
        if self._client is None:
            self._client = Ark(api_key=self._api_key)
        return self._client

    def _get_aclient(self) -> AsyncArk:
        if self._aclient is None:
            self._aclient = AsyncArk(api_key=self._api_key)
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
        """同步调用火山引擎 LLM (CSE Sensor)。"""
        logger.info(f"调用火山引擎 LLM ({self._model_name})，流式: {stream}")
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
            logger.info(f"火山引擎 LLM ({self._model_name}) 调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"火山引擎 LLM ({self._model_name}) 出错: {e}", exc_info=True)
            yield f"抱歉，火山引擎遇到错误: {str(e)}"

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
        """异步调用火山引擎 LLM (CSE Sensor)。"""
        logger.info(f"异步调用 火山引擎 LLM ({self._model_name})，流式: {stream}")
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
            logger.info(f"火山引擎 LLM ({self._model_name}) 异步调用完成，耗时: {duration:.2f}s")
        except Exception as e:
            logger.error(f"火山引擎 LLM ({self._model_name}) 异步出错: {e}", exc_info=True)
            yield f"抱歉，火山引擎异步处理遇到错误: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步向量化文档。"""
        logger.info(f"调用火山引擎嵌入 ({self._model_name})，数量: {len(texts)}")
        client = self._get_client()
        start_time = time.perf_counter()
        
        try:
            # Ark SDK 嵌入调用
            response = client.embeddings.create(input=texts, model=self._model_name)
            embeddings = [item.embedding for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(f"火山引擎嵌入 ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"火山引擎嵌入 ({self._model_name}) 出错: {e}", exc_info=True)
            return [[] for _ in texts]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步向量化文档。"""
        logger.info(f"异步调用 火山引擎嵌入 ({self._model_name})，数量: {len(texts)}")
        aclient = self._get_aclient()
        start_time = time.perf_counter()
        
        try:
            response = await aclient.embeddings.create(input=texts, model=self._model_name)
            embeddings = [item.embedding for item in response.data]
            duration = time.perf_counter() - start_time
            logger.info(f"火山引擎嵌入 ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"火山引擎嵌入 ({self._model_name}) 异步出错: {e}", exc_info=True)
            return [[] for _ in texts]
