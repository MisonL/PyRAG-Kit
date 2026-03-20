# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.__base__.model_provider import RerankModel
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class JinaProvider(RerankModel):
    """
    Jina Rerank模型提供商。
    已注入 CSE 性能传感器与 tenacity 重试机制。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        settings = get_settings()
        self._api_key = settings.jina_api_key
        if not self._api_key:
            logger.error("Jina API Key 未设置。")
            raise ValueError("JINA_API_KEY is required for JinaProvider")
        self._base_url = "https://api.jina.ai/v1/rerank"
        logger.info(f"初始化 JinaProvider，模型: {model_name}")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _prepare_payload(self, query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        return {
            "query": query,
            "documents": documents,
            "model": self._model_name,
            "top_n": top_n
        }

    def _parse_response(self, results: List[Dict], documents: List[str]) -> Tuple[List[int], List[float]]:
        # 创建内容到索引的映射
        content_to_index = {content: i for i, content in enumerate(documents)}
        indices = []
        scores = []
        for res in results:
            doc_content = res.get("document", {}).get("text") if isinstance(res.get("document"), dict) else res.get("document")
            if doc_content in content_to_index:
                indices.append(content_to_index[doc_content])
                scores.append(res.get("relevance_score", 0.0))
        return indices, scores

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def rerank(self, query: str, documents: List[str], top_n: int) -> Tuple[List[int], List[float]]:
        """同步 Rerank (CSE Sensor)。"""
        logger.info(f"调用 Jina Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx
        start_time = time.perf_counter()
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self._base_url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n)
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                
            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"Jina Rerank ({self._model_name}) 完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            logger.error(f"Jina Rerank ({self._model_name}) 出错: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def arerank(self, query: str, documents: List[str], top_n: int) -> Tuple[List[int], List[float]]:
        """异步 Rerank (CSE Sensor)。"""
        logger.info(f"异步调用 Jina Rerank ({self._model_name})，文档数: {len(documents)}")
        import httpx
        start_time = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as aclient:
                response = await aclient.post(
                    self._base_url,
                    headers=self._get_headers(),
                    json=self._prepare_payload(query, documents, top_n)
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                
            indices, scores = self._parse_response(results, documents)
            duration = time.perf_counter() - start_time
            logger.info(f"Jina Rerank ({self._model_name}) 异步完成，耗时: {duration:.2f}s")
            return indices, scores
        except Exception as e:
            logger.error(f"Jina Rerank ({self._model_name}) 异步出错: {e}", exc_info=True)
            raise
