# -*- coding: utf-8 -*-
from typing import List, cast

import requests

from src.providers.__base__.model_provider import RerankModel
from src.utils.config import API_CONFIG


class SiliconflowRerankProvider(RerankModel):
    """
    SiliconFlow Rerank模型提供商。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._api_key = API_CONFIG.get("SILICONFLOW_API_KEY")
        self._base_url = API_CONFIG.get("SILICONFLOW_BASE_URL")
        
        if not self._api_key:
            raise ValueError("错误：SiliconFlow Rerank 提供商需要 API 密钥。")
        if not self._base_url:
            raise ValueError("错误：SiliconFlow Rerank 提供商需要 Base URL。")

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[int]:
        """
        使用SiliconFlow Rerank API对文档进行重排序。
        返回排序后的原始文档索引列表。
        """
        base_url = cast(str, self._base_url) # 强制类型转换以解决Pylance问题
        url = f"{base_url.rstrip('/')}/rerank"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        
        payload = {
            "query": query,
            "documents": documents,
            "model": self._model_name,
            "top_n": top_n
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            rerank_results = response.json().get("results", [])
            
            # 创建一个从内容到原始索引的映射
            content_to_index_map = {content: i for i, content in enumerate(documents)}
            
            # 根据rerank结果的内容，找到其原始索引
            reranked_indices = []
            for res in rerank_results:
                doc_content = res.get("document")
                if doc_content in content_to_index_map:
                    reranked_indices.append(content_to_index_map[doc_content])
            
            return reranked_indices
        except Exception as e:
            print(f"SiliconFlow Rerank出错: {e}")
            # 出错时，返回原始顺序的索引
            return list(range(len(documents)))
