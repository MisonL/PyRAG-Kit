# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import numpy as np
import jieba
import copy
from typing import Any, Dict, List, Optional
from rich.console import Console

# 使用相对导入来引用同一 src 目录下的模块
from ..utils.config import RetrievalMethod, settings # 移除 CHAT_CONFIG, KB_CONFIG
from ..providers.factory import ModelProviderFactory
from .vdb.base import VectorStoreBase # 导入 VectorStoreBase

# =================================================================
# 2. 检索逻辑 (RETRIEVAL LOGIC)
# =================================================================
class HybridReranker:
    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        # 归一化BM25分数
        keyword_scores = [doc.get("keyword_score", 0) for doc in documents]
        max_keyword_score = max(keyword_scores) if keyword_scores else 1
        normalized_keyword_scores = [s / max_keyword_score for s in keyword_scores]
        # 归一化语义分数
        semantic_scores = [doc.get("semantic_score", 0) for doc in documents]
        max_semantic_score = max(semantic_scores) if semantic_scores else 1
        normalized_semantic_scores = [s / max_semantic_score for s in semantic_scores]
        
        for i, doc in enumerate(documents):
            doc["score"] = (settings.chat_vector_weight * normalized_semantic_scores[i] + # 使用 settings
                            settings.chat_keyword_weight * normalized_keyword_scores[i]) # 使用 settings
        return sorted(documents, key=lambda x: x["score"], reverse=True)

def retrieve_documents(query: str, vector_store: VectorStoreBase, console: Console) -> List[Dict]: # 更改类型提示
    retrieval_method = settings.chat_retrieval_method
    top_k = settings.chat_top_k
    
    # 直接通过 vector_store.search 获取结果，它会处理混合检索逻辑
    ranked_results = vector_store.search(query, top_k)

    # 使用Reranker（如果启用）
    if settings.chat_rerank_enabled:
        active_rerank_key = settings.default_rerank_provider
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_key)
        if rerank_provider and ranked_results:
            console.print(f"[dim]正在使用 '{active_rerank_key}' 进行重排...[/dim]")
            # 1. 提取文档内容用于重排
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            
            # 2. 调用 rerank 方法获取重排后的索引
            reranked_indices = rerank_provider.rerank(query, docs_to_rerank, top_n=top_k)
            
            # 3. 根据索引重新排序原始文档列表
            reranked_docs = [ranked_results[i] for i in reranked_indices]
            
            return reranked_docs[:top_k]
    
    return ranked_results[:top_k]