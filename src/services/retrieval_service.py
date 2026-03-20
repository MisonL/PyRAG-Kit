# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console

from src.providers.factory import ModelProviderFactory
from src.retrieval.vdb.base import VectorStoreBase
from src.runtime.contracts import SessionConfig
from src.services.embedding_service import EmbeddingService
from src.utils.config import RetrievalMethod
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)
DEFAULT_RRF_K = 60.0


class HybridReranker:
    def __init__(self, vector_weight: float, keyword_weight: float, fusion_strategy: str = "rrf", rrf_k: float = DEFAULT_RRF_K):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.fusion_strategy = fusion_strategy.strip().lower()
        self.rrf_k = rrf_k
        if self.fusion_strategy not in {"rrf", "weighted"}:
            raise ValueError(f"不支持的混合检索融合策略: {fusion_strategy}")

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

    def rerank(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.fusion_strategy == "weighted":
            return self._weighted_rerank(documents)
        return self._rrf_rerank(documents)

    def _weight_tuple(self) -> tuple[float, float]:
        total = self.vector_weight + self.keyword_weight
        if total <= 0:
            return 0.0, 0.0
        return self.vector_weight / total, self.keyword_weight / total

    def _weighted_rerank(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        keyword_scores = [float(doc.get("keyword_score", 0) or 0) for doc in scored_documents]
        semantic_scores = [float(doc.get("semantic_score", 0) or 0) for doc in scored_documents]
        normalized_keyword = self._normalize_scores(keyword_scores)
        normalized_semantic = self._normalize_scores(semantic_scores)
        for index, doc in enumerate(scored_documents):
            doc["score"] = vector_weight * normalized_semantic[index] + keyword_weight * normalized_keyword[index]
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)

    def _rrf_rerank(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        for doc in scored_documents:
            semantic_rank = int(doc.get("semantic_rank") or 0)
            keyword_rank = int(doc.get("keyword_rank") or 0)
            semantic_rrf = 1.0 / (self.rrf_k + semantic_rank) if semantic_rank > 0 else 0.0
            keyword_rrf = 1.0 / (self.rrf_k + keyword_rank) if keyword_rank > 0 else 0.0
            doc["score"] = vector_weight * semantic_rrf + keyword_weight * keyword_rrf
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)


def _document_key(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id:
        return str(chunk_id)
    source = str(metadata.get("source", ""))
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{chunk_index}"
    return f"{source}:{document.get('page_content', '')[:64]}"


def _merge_hybrid_results(semantic_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for rank, document in enumerate(semantic_results, start=1):
        merged_doc = copy.deepcopy(document)
        merged_doc["semantic_score"] = float(merged_doc.get("score", 0) or 0)
        merged_doc["semantic_rank"] = rank
        merged_doc["keyword_score"] = float(merged_doc.get("keyword_score", 0) or 0)
        merged[_document_key(merged_doc)] = merged_doc
    for rank, document in enumerate(keyword_results, start=1):
        key = _document_key(document)
        if key in merged:
            merged[key]["keyword_score"] = float(document.get("score", 0) or 0)
            merged[key]["keyword_rank"] = rank
        else:
            merged_doc = copy.deepcopy(document)
            merged_doc["semantic_score"] = float(merged_doc.get("semantic_score", 0) or 0)
            merged_doc["keyword_score"] = float(document.get("score", 0) or 0)
            merged_doc["keyword_rank"] = rank
            merged[key] = merged_doc
    return list(merged.values())


def _promote_parent_context(documents: List[Dict[str, Any]], resolver: Optional[Callable[[str | None], Optional[str]]]) -> List[Dict[str, Any]]:
    promoted_documents: List[Dict[str, Any]] = []
    for document in documents:
        promoted_doc = copy.deepcopy(document)
        metadata = promoted_doc.get("metadata") or {}
        parent_content = resolver(metadata.get("parent_id")) if resolver else None
        if parent_content:
            metadata.setdefault("matched_chunk_content", promoted_doc.get("page_content", ""))
            promoted_doc["page_content"] = parent_content
        promoted_doc["metadata"] = metadata
        promoted_documents.append(promoted_doc)
    return promoted_documents


def _deduplicate_parent_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: Dict[str, Dict[str, Any]] = {}
    for document in documents:
        metadata = document.get("metadata") or {}
        parent_key = metadata.get("parent_id")
        key = str(parent_key) if parent_key else _document_key(document)
        existing = deduplicated.get(key)
        if existing is None or float(document.get("score", 0) or 0) > float(existing.get("score", 0) or 0):
            deduplicated[key] = copy.deepcopy(document)
    return list(deduplicated.values())


class RetrievalService:
    def __init__(self, vector_store: VectorStoreBase, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def retrieve(self, query: str, session_config: SessionConfig, console: Optional[Console] = None) -> List[Dict[str, Any]]:
        retrieval_method = session_config.retrieval_method
        effective_top_k = max(1, session_config.top_k) * max(1, session_config.retrieval_candidate_multiplier)
        parent_resolver = getattr(self.vector_store, "resolve_parent_content", None)

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
            semantic_results, keyword_results = await self._gather_hybrid_results(query, effective_top_k)
            reranker = HybridReranker(
                session_config.vector_weight,
                session_config.keyword_weight,
                fusion_strategy=session_config.hybrid_fusion_strategy,
            )
            ranked_results = reranker.rerank(_merge_hybrid_results(semantic_results, keyword_results))
        elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
            ranked_results = await self._semantic_retrieve(query, effective_top_k)
        else:
            ranked_results = await self._keyword_retrieve(query, effective_top_k)

        if self._should_apply_score_threshold(session_config):
            ranked_results = [doc for doc in ranked_results if float(doc.get("score", 0) or 0) >= session_config.score_threshold]

        ranked_results = _promote_parent_context(ranked_results, parent_resolver)
        ranked_results = _deduplicate_parent_documents(ranked_results)
        ranked_results = await self._rerank_if_needed(query, ranked_results, session_config)
        return ranked_results[: session_config.top_k]

    async def _gather_hybrid_results(self, query: str, top_k: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        semantic_results = await self._semantic_retrieve(query, top_k)
        keyword_results = await self._keyword_retrieve(query, top_k)
        return semantic_results, keyword_results

    async def _semantic_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self.embedding_service.embed_query(query)
            return await self._call_vector_store("semantic_search", query_embedding, top_k)
        except (AttributeError, NotImplementedError):
            return await self._legacy_search(query, top_k, "semantic")

    async def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            return await self._call_vector_store("keyword_search", query, top_k)
        except (AttributeError, NotImplementedError):
            return await self._legacy_search(query, top_k, "keyword")

    async def _call_vector_store(self, method_name: str, *args) -> List[Dict[str, Any]]:
        import asyncio

        method = getattr(self.vector_store, method_name)
        return await asyncio.to_thread(method, *args)

    async def _legacy_search(self, query: str, top_k: int, search_type: str) -> List[Dict[str, Any]]:
        import asyncio

        if hasattr(self.vector_store, "asearch"):
            return await self.vector_store.asearch(query, top_k, search_type=search_type)
        return await asyncio.to_thread(self.vector_store.search, query, top_k, search_type)

    async def _rerank_if_needed(self, query: str, ranked_results: List[Dict[str, Any]], session_config: SessionConfig) -> List[Dict[str, Any]]:
        if not session_config.rerank_enabled or not ranked_results:
            return ranked_results

        rerank_provider = ModelProviderFactory.get_rerank_provider(session_config.active_rerank_configuration)
        docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
        reranked_indices, reranked_scores = await rerank_provider.arerank(query, docs_to_rerank, top_n=session_config.top_k)
        reranked_docs: List[Dict[str, Any]] = []
        for index, score in zip(reranked_indices, reranked_scores):
            if index < len(ranked_results):
                document = copy.deepcopy(ranked_results[index])
                document["score"] = score
                reranked_docs.append(document)
        return sorted(reranked_docs, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _should_apply_score_threshold(session_config: SessionConfig) -> bool:
        if session_config.retrieval_method == RetrievalMethod.HYBRID_SEARCH and session_config.hybrid_fusion_strategy == "rrf":
            return False
        return True
