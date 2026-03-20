# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import copy
from typing import Any, Dict, List

import jieba
import numpy as np
from rich.console import Console

# 使用相对导入来引用同一 src 目录下的模块
from ..utils.config import RetrievalMethod
from ..providers.factory import ModelProviderFactory
from .vdb.base import VectorStoreBase
from ..utils.log_manager import get_module_logger  # 导入日志管理器

logger = get_module_logger(__name__)  # 获取当前模块的日志器

# =================================================================
# 2. 检索逻辑 (RETRIEVAL LOGIC)
# =================================================================
class HybridReranker:
    def __init__(self, vector_weight: float, keyword_weight: float):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.debug(f"初始化混合重排器，向量权重: {vector_weight}, 关键词权重: {keyword_weight}")

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []

        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]

        score_range = max_score - min_score
        return [max(0.0, min(1.0, (score - min_score) / score_range)) for score in scores]

    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            logger.info("没有文档可供重排，返回空列表。")
            return []

        logger.debug(f"开始混合重排，文档数量: {len(documents)}")

        total_weight = self.vector_weight + self.keyword_weight
        if total_weight <= 0:
            normalized_vector_weight = 0.0
            normalized_keyword_weight = 0.0
        else:
            normalized_vector_weight = self.vector_weight / total_weight
            normalized_keyword_weight = self.keyword_weight / total_weight

        scored_documents = [copy.deepcopy(doc) for doc in documents]
        keyword_scores = [float(doc.get("keyword_score", 0) or 0) for doc in scored_documents]
        semantic_scores = [float(doc.get("semantic_score", 0) or 0) for doc in scored_documents]
        normalized_keyword_scores = self._normalize_scores(keyword_scores)
        normalized_semantic_scores = self._normalize_scores(semantic_scores)

        for index, doc in enumerate(scored_documents):
            doc["semantic_score"] = semantic_scores[index]
            doc["keyword_score"] = keyword_scores[index]
            doc["score"] = (
                normalized_vector_weight * normalized_semantic_scores[index]
                + normalized_keyword_weight * normalized_keyword_scores[index]
            )

        ranked_docs = sorted(scored_documents, key=lambda x: x["score"], reverse=True)
        logger.debug(f"混合重排完成，返回 {len(ranked_docs)} 个文档。")
        return ranked_docs


def _document_key(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id:
        return str(chunk_id)

    source = str(metadata.get("source", ""))
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{chunk_index}"

    page = metadata.get("page", "")
    page_content = document.get("page_content", "")
    return f"{source}:{page}:{page_content[:64]}"


def _merge_hybrid_results(semantic_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_results: Dict[str, Dict[str, Any]] = {}

    for doc in semantic_results:
        merged_doc = copy.deepcopy(doc)
        merged_doc["semantic_score"] = float(merged_doc.get("score", 0) or 0)
        merged_doc["keyword_score"] = float(merged_doc.get("keyword_score", 0) or 0)
        merged_results[_document_key(merged_doc)] = merged_doc

    for doc in keyword_results:
        key = _document_key(doc)
        keyword_score = float(doc.get("score", 0) or 0)
        if key in merged_results:
            merged_results[key]["keyword_score"] = keyword_score
        else:
            merged_doc = copy.deepcopy(doc)
            merged_doc["semantic_score"] = float(merged_doc.get("semantic_score", 0) or 0)
            merged_doc["keyword_score"] = keyword_score
            merged_results[key] = merged_doc

    for doc in merged_results.values():
        doc["semantic_score"] = float(doc.get("semantic_score", 0) or 0)
        doc["keyword_score"] = float(doc.get("keyword_score", 0) or 0)

    return list(merged_results.values())


def _promote_parent_context(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    promoted_documents: List[Dict[str, Any]] = []
    for doc in documents:
        promoted_doc = copy.deepcopy(doc)
        metadata = promoted_doc.get("metadata") or {}
        parent_content = metadata.get("parent_content")
        if parent_content:
            metadata.setdefault("matched_chunk_content", promoted_doc.get("page_content", ""))
            promoted_doc["page_content"] = parent_content
            promoted_doc["metadata"] = metadata
        promoted_documents.append(promoted_doc)
    return promoted_documents


def _deduplicate_parent_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: Dict[str, Dict[str, Any]] = {}

    for doc in documents:
        metadata = doc.get("metadata") or {}
        parent_key = metadata.get("parent_id")
        key = str(parent_key) if parent_key else _document_key(doc)

        existing_doc = deduplicated.get(key)
        if existing_doc is None or float(doc.get("score", 0) or 0) > float(existing_doc.get("score", 0) or 0):
            deduplicated[key] = copy.deepcopy(doc)

    return list(deduplicated.values())


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

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(_merge_hybrid_results(semantic_results, keyword_results))

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = vector_store.search(query, top_k, search_type="semantic")
    else: 
        ranked_results = vector_store.search(query, top_k, search_type="keyword")

    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]
    ranked_results = _promote_parent_context(ranked_results)
    ranked_results = _deduplicate_parent_documents(ranked_results)

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

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(_merge_hybrid_results(semantic_results, keyword_results))

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = await vector_store.asearch(query, top_k, search_type="semantic")
    else:
        ranked_results = await vector_store.asearch(query, top_k, search_type="keyword")

    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]
    ranked_results = _promote_parent_context(ranked_results)
    ranked_results = _deduplicate_parent_documents(ranked_results)

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
