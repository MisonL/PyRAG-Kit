import asyncio

from src.providers.local_hash import LocalHashEmbeddingProvider


def test_local_hash_embedding_is_deterministic():
    provider = LocalHashEmbeddingProvider("local-hash-32")

    first = provider.embed_documents(["测试文本", "another text"])
    second = provider.embed_documents(["测试文本", "another text"])

    assert first == second
    assert len(first[0]) == 32
    assert len(first[1]) == 32
    assert any(value != 0.0 for value in first[0])


def test_local_hash_embedding_async_matches_sync():
    provider = LocalHashEmbeddingProvider("local-hash-16")

    sync_vectors = provider.embed_documents(["异步一致性"])
    async_vectors = asyncio.run(provider.aembed_documents(["异步一致性"]))

    assert async_vectors == sync_vectors
