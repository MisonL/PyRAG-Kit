import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import retrieve_documents
from src.services.retrieval_service import RetrievalService
from src.utils.config import RetrievalMethod


class MockVectorStore:
    def __init__(self, semantic_results, keyword_results, parent_documents=None):
        self.semantic_results = semantic_results
        self.keyword_results = keyword_results
        self.parent_documents = parent_documents or {}
        self.search_calls = []

    def search(self, query, top_k=5, search_type="semantic"):
        self.search_calls.append((search_type, top_k))
        if search_type == "semantic":
            return self.semantic_results
        if search_type == "keyword":
            return self.keyword_results
        raise AssertionError(f"unexpected search_type: {search_type}")

    def resolve_parent_content(self, parent_id):
        parent_document = self.parent_documents.get(parent_id)
        if not parent_document:
            return None
        return parent_document.get("content")


def test_hybrid_retrieval_starts_semantic_and_keyword_paths_in_parallel():
    service = object.__new__(RetrievalService)
    started = []
    release = asyncio.Event()

    async def fake_semantic(_query, _top_k):
        started.append("semantic")
        await release.wait()
        return ["semantic"]

    async def fake_keyword(_query, _top_k):
        started.append("keyword")
        await release.wait()
        return ["keyword"]

    service._semantic_retrieve = fake_semantic
    service._keyword_retrieve = fake_keyword

    async def run():
        task = asyncio.create_task(service._gather_hybrid_results("query", 3))
        async def wait_for_both_starts():
            while len(started) < 2:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_both_starts(), timeout=5)
        release.set()
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(run()) == (["semantic"], ["keyword"])
    assert set(started) == {"semantic", "keyword"}


def test_rerank_provider_is_cached_per_configuration(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        return [0], [float(top_n)]

    provider.arerank.side_effect = fake_arerank
    factory = MagicMock(return_value=provider)
    monkeypatch.setattr("src.services.retrieval_service.ModelProviderFactory.get_rerank_provider", factory)
    session_config = SimpleNamespace(
        rerank_enabled=True,
        active_rerank_configuration="siliconflow",
        top_k=1,
        rerank_configurations={"siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank")},
    )
    documents = [{"page_content": "doc", "score": 0.5, "metadata": {}}]

    async def run():
        first = await service._rerank_if_needed("query", documents, session_config)
        second = await service._rerank_if_needed("query", documents, session_config)
        return first, second

    first, second = asyncio.run(run())

    assert first[0]["score"] == 1.0
    assert second[0]["score"] == 1.0
    factory.assert_called_once_with("siliconflow", session_config.rerank_configurations)
    assert provider.arerank.call_count == 2


def test_rerank_provider_cache_refreshes_when_options_change(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    first_provider = MagicMock()
    second_provider = MagicMock()
    factory = MagicMock(side_effect=[first_provider, second_provider])
    monkeypatch.setattr("src.services.retrieval_service.ModelProviderFactory.get_rerank_provider", factory)
    first_config = SimpleNamespace(
        provider="siliconflow",
        model_name="rerank",
        options={"timeout": 10},
    )
    second_config = SimpleNamespace(
        provider="siliconflow",
        model_name="rerank",
        options={"timeout": 20},
    )

    assert service._get_rerank_provider("siliconflow", {"siliconflow": first_config}) is first_provider
    assert service._get_rerank_provider("siliconflow", {"siliconflow": second_config}) is second_provider
    assert factory.call_count == 2


def test_retrieval_service_close_disposes_cached_rerank_and_embedding_models():
    rerank = MagicMock()
    embedding = MagicMock()
    service = RetrievalService(vector_store=object(), embedding_service=embedding)
    service._rerank_providers = {("key", "provider", "model"): rerank}

    service.close()

    rerank.close.assert_called_once_with()
    embedding.close.assert_called_once_with()
    assert service._rerank_providers == {}


def test_retrieval_service_aclose_falls_back_to_sync_embedding_close():
    class SyncOnlyEmbedding:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    embedding = SyncOnlyEmbedding()
    service = RetrievalService(vector_store=object(), embedding_service=embedding)

    asyncio.run(service.aclose())

    assert embedding.closed


def test_rerank_top_n_is_clamped_after_candidate_reduction(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        assert top_n == 1
        return [0], [0.9]

    provider.arerank.side_effect = fake_arerank
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider",
        MagicMock(return_value=provider),
    )
    session_config = SimpleNamespace(
        rerank_enabled=True,
        active_rerank_configuration="siliconflow",
        top_k=5,
        rerank_configurations={
            "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank")
        },
    )

    result = asyncio.run(
        service._rerank_if_needed(
            "query", [{"page_content": "only", "score": 0.1, "metadata": {}}], session_config
        )
    )

    assert result[0]["score"] == 0.9


def test_semantic_retrieve_uses_legacy_search_when_method_is_missing():
    class LegacyStore:
        def search(self, query, top_k, search_type="semantic"):
            return [{"page_content": query, "score": top_k, "metadata": {"search_type": search_type}}]

    class UnusedEmbedding:
        async def embed_query(self, _query):
            raise AssertionError("legacy semantic search should not require embeddings")

    embedding = UnusedEmbedding()
    service = RetrievalService(LegacyStore(), embedding)

    result = asyncio.run(service._semantic_retrieve("query", 3))

    assert result == [{"page_content": "query", "score": 3, "metadata": {"search_type": "semantic"}}]


def test_semantic_retrieve_does_not_hide_embedding_attribute_errors():
    class SemanticStore:
        def semantic_search(self, _embedding, _top_k):
            return []

    class FailingEmbedding:
        async def embed_query(self, _query):
            raise AttributeError("embedding provider bug")

    service = RetrievalService(SemanticStore(), FailingEmbedding())

    with pytest.raises(AttributeError, match="embedding provider bug"):
        asyncio.run(service._semantic_retrieve("query", 3))


def test_semantic_retrieve_does_not_hide_vector_store_attribute_errors():
    class BrokenStore:
        def semantic_search(self, _embedding, _top_k):
            raise AttributeError("vector store bug")

    class Embedding:
        async def embed_query(self, _query):
            return [0.1]

    service = RetrievalService(BrokenStore(), Embedding())

    with pytest.raises(AttributeError, match="vector store bug"):
        asyncio.run(service._semantic_retrieve("query", 3))


@pytest.mark.parametrize(
    ("indices", "scores", "message"),
    [
        ([], [], "未返回可用结果"),
        ([0], [], "数量不一致"),
        ([2], [0.9], "index 无效"),
        ([0, 0], [0.9, 0.8], "重复 index"),
        ([0], [float("nan")], "必须是有限数值"),
    ],
)
def test_rerank_output_contract_failures_are_explicit(indices, scores, message):
    with pytest.raises(RuntimeError, match=message):
        RetrievalService._validate_rerank_output(indices, scores, document_count=2)


def test_retrieve_documents_uses_parent_sidecar_and_overfetches_candidates(monkeypatch):
    semantic_results = [
        {
            "page_content": "child content alpha",
            "score": 0.9,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-1",
                "parent_id": "parent-1",
            },
        },
        {
            "page_content": "child content beta",
            "score": 0.8,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-2",
                "parent_id": "parent-1",
            },
        },
        {
            "page_content": "child content gamma",
            "score": 0.7,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-beta-1",
                "parent_id": "parent-2",
            },
        },
    ]
    keyword_results = [
        {
            "page_content": "child content alpha",
            "score": 6.0,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-1",
                "parent_id": "parent-1",
            },
        }
    ]
    store = MockVectorStore(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        parent_documents={
            "parent-1": {"content": "parent content alpha"},
            "parent-2": {"content": "parent content beta"},
        },
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        top_k=2,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.0,
        fusion_strategy="rrf",
        candidate_multiplier=3,
    )

    assert sorted(store.search_calls) == [("keyword", 6), ("semantic", 6)]
    assert len(results) == 2
    assert {doc["metadata"]["parent_id"] for doc in results} == {"parent-1", "parent-2"}
    assert all(doc["page_content"].startswith("parent content") for doc in results)
    assert all(doc["metadata"]["matched_chunk_content"].startswith("child content") for doc in results)


def test_retrieve_documents_collapses_multiple_children_from_same_parent(monkeypatch):
    store = MockVectorStore(
        semantic_results=[
            {
                "page_content": "child content high",
                "score": 0.9,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            },
            {
                "page_content": "child content low",
                "score": 0.7,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-low",
                    "parent_id": "parent-1",
                },
            },
        ],
        keyword_results=[],
        parent_documents={"parent-1": {"content": "parent content"}},
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.0,
        candidate_multiplier=3,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["parent_id"] == "parent-1"
    assert results[0]["page_content"] == "parent content"
    assert results[0]["metadata"]["matched_chunk_content"] == "child content high"


def test_retrieve_documents_rrf_keeps_results_under_default_threshold(monkeypatch):
    store = MockVectorStore(
        semantic_results=[
            {
                "page_content": "child content high",
                "score": 0.9,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            }
        ],
        keyword_results=[
            {
                "page_content": "child content high",
                "score": 4.0,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            }
        ],
        parent_documents={"parent-1": {"content": "parent content"}},
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.4,
        fusion_strategy="rrf",
        candidate_multiplier=3,
    )

    assert results
    assert results[0]["page_content"] == "parent content"
    assert results[0]["metadata"]["parent_id"] == "parent-1"
