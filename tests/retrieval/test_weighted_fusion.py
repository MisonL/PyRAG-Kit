from src.retrieval.retriever import HybridReranker, _merge_hybrid_results


def test_hybrid_reranker_normalizes_mixed_score_scales():
    reranker = HybridReranker(vector_weight=0.8, keyword_weight=0.2)
    documents = [
        {"page_content": "doc-a", "semantic_score": 0.9, "keyword_score": 10.0},
        {"page_content": "doc-b", "semantic_score": 0.1, "keyword_score": 5.0},
    ]

    ranked = reranker.rerank(documents)

    assert [doc["page_content"] for doc in ranked] == ["doc-a", "doc-b"]
    assert ranked[0]["score"] == 1.0
    assert ranked[1]["score"] == 0.0


def test_hybrid_reranker_handles_uniform_positive_scores():
    reranker = HybridReranker(vector_weight=0.5, keyword_weight=0.5)
    documents = [
        {"page_content": "doc-a", "semantic_score": 3.0, "keyword_score": 4.0},
        {"page_content": "doc-b", "semantic_score": 3.0, "keyword_score": 4.0},
    ]

    ranked = reranker.rerank(documents)

    assert all(doc["score"] == 1.0 for doc in ranked)


def test_hybrid_reranker_handles_all_zero_scores():
    reranker = HybridReranker(vector_weight=0.5, keyword_weight=0.5)
    documents = [
        {"page_content": "doc-a", "semantic_score": 0.0, "keyword_score": 0.0},
        {"page_content": "doc-b", "semantic_score": 0.0, "keyword_score": 0.0},
    ]

    ranked = reranker.rerank(documents)

    assert all(doc["score"] == 0.0 for doc in ranked)


def test_merge_hybrid_results_keeps_distinct_child_chunks_on_same_page():
    semantic_results = [
        {
            "page_content": "child-a",
            "score": 0.9,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-a"},
        },
        {
            "page_content": "child-b",
            "score": 0.8,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-b"},
        },
    ]
    keyword_results = [
        {
            "page_content": "child-a",
            "score": 4.0,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-a"},
        }
    ]

    merged = _merge_hybrid_results(semantic_results, keyword_results)

    assert len(merged) == 2
    assert {doc["metadata"]["chunk_id"] for doc in merged} == {"chunk-a", "chunk-b"}
    merged_a = next(doc for doc in merged if doc["metadata"]["chunk_id"] == "chunk-a")
    assert merged_a["semantic_score"] == 0.9
    assert merged_a["keyword_score"] == 4.0
