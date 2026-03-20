import numpy as np

from src.retrieval.vdb.faiss_store import FaissStore


def test_faiss_store_persists_parent_sidecar_snapshot_roundtrip(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
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
    store._rebuild_indices()
    store.save_snapshot(str(snapshot_dir))

    reloaded = FaissStore(file_path=None)
    reloaded.load_snapshot(str(snapshot_dir))

    assert reloaded.resolve_parent_content("parent-1") == "parent content"
    assert reloaded.parent_documents["parent-1"]["metadata"]["source"] == "kb.md"
    assert reloaded.documents[0]["metadata"]["parent_id"] == "parent-1"
