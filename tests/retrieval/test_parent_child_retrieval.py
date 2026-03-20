from unittest.mock import MagicMock

from src.retrieval.retriever import retrieve_documents
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

    assert store.search_calls == [("semantic", 6), ("keyword", 6)]
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
