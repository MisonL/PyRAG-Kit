from unittest.mock import MagicMock

from src.retrieval.retriever import retrieve_documents
from src.utils.config import RetrievalMethod


class MockVectorStore:
    def __init__(self, semantic_results, keyword_results):
        self.semantic_results = semantic_results
        self.keyword_results = keyword_results

    def search(self, query, top_k=5, search_type="semantic"):
        if search_type == "semantic":
            return self.semantic_results
        if search_type == "keyword":
            return self.keyword_results
        raise AssertionError(f"unexpected search_type: {search_type}")


def test_retrieve_documents_promotes_parent_content_and_preserves_chunk_identity(monkeypatch):
    semantic_results = [
        {
            "page_content": "child content alpha",
            "score": 0.9,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha",
                "parent_content": "parent content alpha",
            },
        },
        {
            "page_content": "child content beta",
            "score": 0.8,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-beta",
                "parent_content": "parent content beta",
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
                "chunk_id": "chunk-alpha",
                "parent_content": "parent content alpha",
            },
        }
    ]
    store = MockVectorStore(semantic_results, keyword_results)
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
        score_threshold=0.0,
    )

    assert len(results) == 2
    assert {doc["metadata"]["chunk_id"] for doc in results} == {"chunk-alpha", "chunk-beta"}
    assert results[0]["page_content"] in {"parent content alpha", "parent content beta"}
    assert all(doc["page_content"] == doc["metadata"]["parent_content"] for doc in results)
    assert all(doc["metadata"]["matched_chunk_content"].startswith("child content") for doc in results)


def test_retrieve_documents_filters_by_threshold_after_merge(monkeypatch):
    store = MockVectorStore(
        semantic_results=[
            {
                "page_content": "child content",
                "score": 0.1,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-low",
                    "parent_content": "parent content",
                },
            }
        ],
        keyword_results=[],
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
        score_threshold=0.5,
    )

    assert results == []


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
                    "parent_content": "parent content",
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
                    "parent_content": "parent content",
                },
            },
        ],
        keyword_results=[],
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
    )

    assert len(results) == 1
    assert results[0]["metadata"]["parent_id"] == "parent-1"
    assert results[0]["page_content"] == "parent content"
    assert results[0]["metadata"]["matched_chunk_content"] == "child content high"
