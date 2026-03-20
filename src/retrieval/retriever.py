# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

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
from ..utils.config import RetrievalMethod, get_settings # 导入 get_settings 函数
from ..providers.factory import ModelProviderFactory
from .vdb.base import VectorStoreBase
from ..utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

# =================================================================
# 2. 检索逻辑 (RETRIEVAL LOGIC)
# =================================================================
class HybridReranker:
    def __init__(self, vector_weight: float, keyword_weight: float):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.debug(f"初始化混合重排器，向量权重: {vector_weight}, 关键词权重: {keyword_weight}")

    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            logger.info("没有文档可供重排，返回空列表。")
            return []
        
        logger.debug(f"开始混合重排，文档数量: {len(documents)}")
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
        
        ranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
        logger.debug(f"混合重排完成，返回 {len(ranked_docs)} 个文档。")
        return ranked_docs

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
    logger.info(f"开始同步检索: '{query[:20]}...', 方法: {retrieval_method.value}")
    
    ranked_results: List[Dict] = []
    
    if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
        semantic_results = vector_store.search(query, top_k, search_type="semantic")
        keyword_results = vector_store.search(query, top_k, search_type="keyword")
        
        all_docs_map = {doc['metadata']['source'] + str(doc['metadata'].get('page', '')): doc for doc in semantic_results}
        for doc in keyword_results:
            key = doc['metadata']['source'] + str(doc['metadata'].get('page', ''))
            if key in all_docs_map:
                all_docs_map[key]['keyword_score'] = doc.get('score', 0)
            else:
                all_docs_map[key] = doc

        for doc in all_docs_map.values():
            doc['semantic_score'] = doc.get('score', 0) if 'semantic_score' not in doc else doc['semantic_score']
            doc['keyword_score'] = doc.get('score', 0) if 'keyword_score' not in doc else doc['keyword_score']

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(list(all_docs_map.values()))

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = vector_store.search(query, top_k, search_type="semantic")
    else: 
        ranked_results = vector_store.search(query, top_k, search_type="keyword")

    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]

    if rerank_enabled and ranked_results:
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_configuration)
        if rerank_provider:
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            try:
                reranked_indices, reranked_scores = rerank_provider.rerank(query, docs_to_rerank, top_n=top_k)
                reranked_docs = []
                for i, score in zip(reranked_indices, reranked_scores):
                    if i < len(ranked_results):
                        doc = ranked_results[i]
                        doc['score'] = score
                        reranked_docs.append(doc)
                ranked_results = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)
            except Exception as e:
                logger.error(f"同步 Rerank 出错: {e}")

    return ranked_results[:top_k]

async def aretrieve_documents(
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
    """异步检索文档 (CSE Sensor)。"""
    import asyncio
    logger.info(f"开始异步检索: '{query[:20]}...', 方法: {retrieval_method.value}")
    
    ranked_results: List[Dict] = []
    
    if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
        semantic_task = vector_store.asearch(query, top_k, search_type="semantic")
        keyword_task = vector_store.asearch(query, top_k, search_type="keyword")
        semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
        
        all_docs_map = {doc['metadata']['source'] + str(doc['metadata'].get('page', '')): doc for doc in semantic_results}
        for doc in keyword_results:
            key = doc['metadata']['source'] + str(doc['metadata'].get('page', ''))
            if key in all_docs_map:
                all_docs_map[key]['keyword_score'] = doc.get('score', 0)
            else:
                all_docs_map[key] = doc

        for doc in all_docs_map.values():
            doc['semantic_score'] = doc.get('score', 0) if 'semantic_score' not in doc else doc['semantic_score']
            doc['keyword_score'] = doc.get('score', 0) if 'keyword_score' not in doc else doc['keyword_score']

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(list(all_docs_map.values()))

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = await vector_store.asearch(query, top_k, search_type="semantic")
    else:
        ranked_results = await vector_store.asearch(query, top_k, search_type="keyword")

    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]

    if rerank_enabled and ranked_results:
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_configuration)
        if rerank_provider:
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            try:
                reranked_indices, reranked_scores = await rerank_provider.arerank(query, docs_to_rerank, top_n=top_k)
                reranked_docs = []
                for i, score in zip(reranked_indices, reranked_scores):
                    if i < len(ranked_results):
                        doc = ranked_results[i]
                        doc['score'] = score
                        reranked_docs.append(doc)
                ranked_results = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)
                logger.info(f"外部异步重排完成，获取到 {len(ranked_results)} 个结果。")
            except Exception as e:
                logger.error(f"异步 Rerank 出错 (降级到原始评分): {e}")

    final_results = ranked_results[:top_k]
    logger.info(f"异步检索流程结束，返回 {len(final_results)} 条记录。")
    return final_results