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
    def __init__(self, vector_weight: float, keyword_weight: float):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        
        # 为每个文档分配语义和关键字分数
        for doc in documents:
            doc['semantic_score'] = doc.get('semantic_score', 0)
            doc['keyword_score'] = doc.get('keyword_score', 0)

        keyword_scores = [doc['keyword_score'] for doc in documents]
        semantic_scores = [doc['semantic_score'] for doc in documents]

        # 归一化BM25分数
        max_keyword_score = max(keyword_scores) if keyword_scores else 0
        normalized_keyword_scores = [s / max_keyword_score if max_keyword_score != 0 else 0 for s in keyword_scores]
        
        # 归一化语义分数
        max_semantic_score = max(semantic_scores) if semantic_scores else 0
        normalized_semantic_scores = [s / max_semantic_score if max_semantic_score != 0 else 0 for s in semantic_scores]
        
        for i, doc in enumerate(documents):
            doc["score"] = (self.vector_weight * normalized_semantic_scores[i] +
                            self.keyword_weight * normalized_keyword_scores[i])
        return sorted(documents, key=lambda x: x["score"], reverse=True)

def retrieve_documents(
    query: str,
    vector_store: VectorStoreBase,
    console: Console,
    retrieval_method: RetrievalMethod,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    rerank_enabled: bool,
    active_rerank_configuration: str,
    score_threshold: float,
) -> List[Dict]:
    
    if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
        semantic_results = vector_store.search(query, top_k, search_type="semantic")
        keyword_results = vector_store.search(query, top_k, search_type="keyword")
        
        # 合并并去重
        all_docs_map = {doc['metadata']['source'] + str(doc['metadata'].get('page', '')): doc for doc in semantic_results}
        for doc in keyword_results:
            key = doc['metadata']['source'] + str(doc['metadata'].get('page', ''))
            if key in all_docs_map:
                all_docs_map[key]['keyword_score'] = doc.get('score', 0)
            else:
                all_docs_map[key] = doc

        # 为文档分配分数
        for doc in all_docs_map.values():
            doc['semantic_score'] = doc.get('score', 0) if 'semantic_score' not in doc else doc['semantic_score']
            doc['keyword_score'] = doc.get('score', 0) if 'keyword_score' not in doc else doc['keyword_score']

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(list(all_docs_map.values()))

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = vector_store.search(query, top_k, search_type="semantic")
    else: # FULL_TEXT_SEARCH
        ranked_results = vector_store.search(query, top_k, search_type="keyword")

    # 应用分数阈值过滤
    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]

    # 使用外部Reranker（如果启用）
    if rerank_enabled:
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_configuration)
        if rerank_provider and ranked_results:
            console.print(f"[dim]正在使用 '{active_rerank_configuration}' 进行重排...[/dim]")
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            
            try:
                reranked_indices, reranked_scores = rerank_provider.rerank(query, docs_to_rerank, top_n=top_k)
                
                # 根据索引和分数重新构建文档列表
                reranked_docs = []
                for i, score in zip(reranked_indices, reranked_scores):
                    if i < len(ranked_results):
                        doc = ranked_results[i]
                        doc['score'] = score # 更新为 reranker 的分数
                        reranked_docs.append(doc)
                
                # 按新的分数排序
                ranked_results = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)

            except Exception as e:
                console.print(f"[bold red]Rerank提供商出错: {e}[/bold red]")

    return ranked_results[:top_k]