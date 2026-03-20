from types import SimpleNamespace

import numpy as np

from src.retrieval.vdb.faiss_store import FaissStore


class DummyEmbeddingModel:
    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]

    async def aembed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]


def test_faiss_store_persists_parent_sidecar_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.vdb.faiss_store.get_settings",
        lambda: SimpleNamespace(default_embedding_provider="dummy"),
    )
    monkeypatch.setattr(
        "src.retrieval.vdb.faiss_store.ModelProviderFactory.get_embedding_provider",
        lambda *args, **kwargs: DummyEmbeddingModel(),
    )

    store_path = tmp_path / "faiss_store.pkl"
    store = FaissStore(file_path=None)
    store.documents = [
        {
            "page_content": "child content",
            "metadata": {
                "source": "kb.md",
                "chunk_id": "chunk-1",
                "parent_id": "parent-1",
            },
        }
    ]
    store.embeddings = np.array([[0.0, 0.0]], dtype=np.float32)
    store.register_parent_documents(
        {
            "parent-1": {
                "content": "parent content",
                "metadata": {"source": "kb.md"},
            }
        }
    )

    store.save(str(store_path))

    reloaded = FaissStore(file_path=str(store_path))

    assert reloaded.resolve_parent_content("parent-1") == "parent content"
    assert reloaded.parent_documents["parent-1"]["metadata"]["source"] == "kb.md"
    assert reloaded.documents[0]["metadata"]["parent_id"] == "parent-1"
    assert "parent_content" not in reloaded.documents[0]["metadata"]
