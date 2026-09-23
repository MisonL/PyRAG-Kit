from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from rich.console import Console

from src.providers.__base__.model_provider import RerankModel, close_resource_sync
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
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]
        score_range = max_score - min_score
        return [max(0.0, min(1.0, (score - min_score) / score_range)) for score in scores]

    def rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fusion_strategy == "weighted":
            return self._weighted_rerank(documents)
        return self._rrf_rerank(documents)

    def _weight_tuple(self) -> tuple[float, float]:
        total = self.vector_weight + self.keyword_weight
        if total <= 0:
            return 0.0, 0.0
        return self.vector_weight / total, self.keyword_weight / total

    def _weighted_rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        keyword_scores = [float(doc.get("keyword_score", 0) or 0) for doc in scored_documents]
        semantic_scores = [float(doc.get("semantic_score", 0) or 0) for doc in scored_documents]
        normalized_keyword = self._normalize_scores(keyword_scores)
        normalized_semantic = self._normalize_scores(semantic_scores)
        for index, doc in enumerate(scored_documents):
            doc["score"] = vector_weight * normalized_semantic[index] + keyword_weight * normalized_keyword[index]
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)

    def _rrf_rerank(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vector_weight, keyword_weight = self._weight_tuple()
        scored_documents = [copy.deepcopy(doc) for doc in documents]
        for doc in scored_documents:
            semantic_rank = int(doc.get("semantic_rank") or 0)
            keyword_rank = int(doc.get("keyword_rank") or 0)
            semantic_rrf = 1.0 / (self.rrf_k + semantic_rank) if semantic_rank > 0 else 0.0
            keyword_rrf = 1.0 / (self.rrf_k + keyword_rank) if keyword_rank > 0 else 0.0
            doc["score"] = vector_weight * semantic_rrf + keyword_weight * keyword_rrf
        return sorted(scored_documents, key=lambda item: item["score"], reverse=True)


def _document_key(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
    if chunk_id:
        return str(chunk_id)
    source = str(metadata.get("source", ""))
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{chunk_index}"
    return f"{source}:{document.get('page_content', '')[:64]}"


def _merge_hybrid_results(semantic_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
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


def _promote_parent_context(documents: list[dict[str, Any]], resolver: Callable[[str | None], str | None] | None) -> list[dict[str, Any]]:
    promoted_documents: list[dict[str, Any]] = []
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


def _deduplicate_parent_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
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
        self._rerank_providers: dict[tuple[str, str, str, str], RerankModel] = {}

    async def retrieve(self, query: str, session_config: SessionConfig, console: Console | None = None) -> list[dict[str, Any]]:
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
        return ranked_results[: max(1, session_config.top_k)]

    async def _gather_hybrid_results(self, query: str, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        semantic_results, keyword_results = await asyncio.gather(
            self._semantic_retrieve(query, top_k),
            self._keyword_retrieve(query, top_k),
        )
        return semantic_results, keyword_results

    async def _semantic_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        # Embedding failures must propagate. Only use the legacy path when the
        # vector store truly does not expose the newer semantic_search method;
        # an AttributeError/NotImplementedError raised inside a real provider is
        # a defect or an upstream failure, not a compatibility signal.
        if self._get_vector_store_method("semantic_search") is None:
            return await self._legacy_search(query, top_k, "semantic")
        query_embedding = await self.embedding_service.embed_query(query)
        return await self._call_vector_store("semantic_search", query_embedding, top_k)

    async def _keyword_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._get_vector_store_method("keyword_search") is None:
            return await self._legacy_search(query, top_k, "keyword")
        return await self._call_vector_store("keyword_search", query, top_k)

    def _get_vector_store_method(self, method_name: str) -> Callable[..., Any] | None:
        """Return an implemented vector-store method, excluding base stubs."""
        method = getattr(self.vector_store, method_name, None)
        if not callable(method):
            return None
        implementation = getattr(type(self.vector_store), method_name, None)
        base_implementation = getattr(VectorStoreBase, method_name, None)
        if implementation is base_implementation:
            return None
        return method

    async def _call_vector_store(self, method_name: str, *args) -> list[dict[str, Any]]:
        method = self._get_vector_store_method(method_name)
        if method is None:
            raise NotImplementedError(f"向量存储未实现 {method_name} 接口。")
        return await asyncio.to_thread(method, *args)

    async def _legacy_search(self, query: str, top_k: int, search_type: str) -> list[dict[str, Any]]:
        async_search = getattr(self.vector_store, "asearch", None)
        if callable(async_search):
            return await async_search(query, top_k, search_type=search_type)
        sync_search = getattr(self.vector_store, "search", None)
        if not callable(sync_search):
            raise NotImplementedError("向量存储未实现兼容 search 接口。")
        return await asyncio.to_thread(sync_search, query, top_k, search_type)

    def _get_rerank_provider(
        self,
        provider_key: str,
        configurations: Mapping[str, Any] | None = None,
    ) -> RerankModel:
        detail = configurations.get(provider_key) if configurations is not None else None
        options = getattr(detail, "options", {}) if detail is not None else {}
        try:
            options_fingerprint = json.dumps(
                options or {}, sort_keys=True, default=str, separators=(",", ":")
            )
        except (TypeError, ValueError):
            options_fingerprint = repr(options)
        cache_key = (
            provider_key,
            getattr(detail, "provider", ""),
            getattr(detail, "model_name", ""),
            options_fingerprint,
        )
        provider = self._rerank_providers.get(cache_key)
        if provider is None:
            # A configuration reload can keep the same key while changing the
            # endpoint, model or options. Drop the old instance before creating
            # its replacement so stale transports and credentials are not kept
            # alive indefinitely.
            stale_keys = [
                key for key in self._rerank_providers
                if key[0] == provider_key and key != cache_key
            ]
            for stale_key in stale_keys:
                stale_provider = self._rerank_providers.pop(stale_key)
                close_resource_sync(stale_provider, "Rerank Provider")
            if configurations is None:
                provider = ModelProviderFactory.get_rerank_provider(provider_key)
            else:
                provider = ModelProviderFactory.get_rerank_provider(provider_key, configurations)
            self._rerank_providers[cache_key] = provider
        return provider

    def clear_rerank_cache(self) -> None:
        """关闭并清空已缓存的 Rerank Provider。"""
        providers = list(self._rerank_providers.values())
        self._rerank_providers.clear()
        closed: set[int] = set()
        for provider in providers:
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            close_resource_sync(provider, "Rerank Provider")

    def close(self) -> None:
        """释放重排缓存及 EmbeddingService 持有的 Provider。"""
        self.clear_rerank_cache()
        close_embedding = getattr(self.embedding_service, "close", None)
        if callable(close_embedding):
            close_embedding()

    async def aclose(self) -> None:
        """异步释放重排缓存及 EmbeddingService 持有的 Provider。"""
        providers = list(self._rerank_providers.values())
        self._rerank_providers.clear()
        closed: set[int] = set()
        for provider in providers:
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            close = getattr(provider, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
                continue
            close = getattr(provider, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        close_embedding = getattr(self.embedding_service, "aclose", None)
        if not callable(close_embedding):
            close_embedding = getattr(self.embedding_service, "close", None)
        if callable(close_embedding):
            result = close_embedding()
            if inspect.isawaitable(result):
                await result

    async def _rerank_if_needed(self, query: str, ranked_results: list[dict[str, Any]], session_config: SessionConfig) -> list[dict[str, Any]]:
        if not session_config.rerank_enabled or not ranked_results:
            return ranked_results

        rerank_provider = self._get_rerank_provider(
            session_config.active_rerank_configuration,
            session_config.rerank_configurations,
        )
        docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
        # Keep the requested output bounded by the available candidates after
        # thresholding and parent de-duplication.
        rerank_top_n = max(1, min(session_config.top_k, len(docs_to_rerank)))
        reranked_indices, reranked_scores = await rerank_provider.arerank(
            query, docs_to_rerank, top_n=rerank_top_n
        )
        reranked_indices, reranked_scores = self._validate_rerank_output(
            reranked_indices,
            reranked_scores,
            document_count=len(ranked_results),
        )
        reranked_docs: list[dict[str, Any]] = []
        for index, score in zip(reranked_indices, reranked_scores):
            document = copy.deepcopy(ranked_results[index])
            document["score"] = score
            reranked_docs.append(document)
        return sorted(reranked_docs, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _validate_rerank_output(
        indices: Any,
        scores: Any,
        document_count: int,
    ) -> tuple[list[int], list[float]]:
        """Validate a provider result before it can alter retrieval ordering."""
        try:
            normalized_indices = list(indices)
            normalized_scores = list(scores)
        except TypeError as exc:
            raise RuntimeError("Rerank provider 返回值必须是可迭代的索引和分数。") from exc

        if len(normalized_indices) != len(normalized_scores):
            raise RuntimeError("Rerank provider 返回的索引和分数数量不一致。")
        if not normalized_indices:
            raise RuntimeError("Rerank provider 未返回可用结果。")

        seen_indices: set[int] = set()
        validated_scores: list[float] = []
        for position, (index, score) in enumerate(zip(normalized_indices, normalized_scores)):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < document_count
            ):
                raise RuntimeError(f"Rerank provider 返回的 index 无效: position={position}。")
            if index in seen_indices:
                raise RuntimeError(f"Rerank provider 返回重复 index: {index}。")
            seen_indices.add(index)
            if isinstance(score, bool):
                raise RuntimeError(f"Rerank provider 返回的 score 无效: index={index}。")
            try:
                normalized_score = float(score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Rerank provider 返回的 score 无效: index={index}。") from exc
            if not math.isfinite(normalized_score):
                raise RuntimeError(f"Rerank provider 返回的 score 必须是有限数值: index={index}。")
            validated_scores.append(normalized_score)

        return normalized_indices, validated_scores

    @staticmethod
    def _should_apply_score_threshold(session_config: SessionConfig) -> bool:
        return not (
            session_config.retrieval_method == RetrievalMethod.HYBRID_SEARCH
            and session_config.hybrid_fusion_strategy == "rrf"
        )
