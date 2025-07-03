# -*- coding: utf-8 -*-
from typing import List, cast

import requests

from src.providers.__base__.model_provider import RerankModel
from src.utils.config import settings


class SiliconflowRerankProvider(RerankModel):
    """
    SiliconFlow Rerank模型提供商。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._api_key = settings.siliconflow_api_key
        self._base_url = settings.siliconflow_base_url
        
        if not self._api_key:
            raise ValueError("错误：SiliconFlow Rerank 提供商需要 API 密钥。")
        if not self._base_url:
            raise ValueError("错误：SiliconFlow Rerank 提供商需要 Base URL。")

    def rerank(self, query: str, documents: List[str], top_n: int) -> tuple[list[int], list[float]]:
        """
        使用SiliconFlow Rerank API对文档进行重排序。
        返回一个元组，包含 (排序后的原始文档索引列表, 对应的相关度分数列表)。
        """
        base_url = cast(str, self._base_url) # 强制类型转换以解决Pylance问题
        url = f"{base_url.rstrip('/')}/rerank"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        
        payload = {
            "query": query,
            "documents": documents,
            "model": self._model_name,
            "top_n": top_n,
            "return_documents": True # 确保返回文档内容以进行匹配
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            rerank_results = response.json().get("results", [])
            
            # 创建一个从内容到原始索引的映射
            content_to_index_map = {content: i for i, content in enumerate(documents)}
            
            reranked_indices = []
            reranked_scores = []
            for res in rerank_results:
                doc_content = res.get("document", {}).get("text")
                relevance_score = res.get("relevance_score")
                
                if doc_content in content_to_index_map and relevance_score is not None:
                    reranked_indices.append(content_to_index_map[doc_content])
                    reranked_scores.append(relevance_score)
            
            return reranked_indices, reranked_scores
        except Exception as e:
            print(f"SiliconFlow Rerank出错: {e}")
            # 出错时，返回原始顺序的索引和0分
            return list(range(len(documents))), [0.0] * len(documents)
