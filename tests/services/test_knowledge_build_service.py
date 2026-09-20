import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.snapshot_repository import SnapshotRepository
from src.services.knowledge_build_service import KnowledgeBuildService


class _RunConfig:
    def __init__(self, knowledge_base_path: Path, snapshot_root: Path):
        self.knowledge_base_path = knowledge_base_path
        self.snapshot_root = snapshot_root
        self.default_vector_store = "faiss"

    @property
    def active_snapshot_marker(self) -> Path:
        return self.snapshot_root / "ACTIVE_SNAPSHOT"


class _IncompleteVectorStore:
    def save_snapshot(self, snapshot_dir: str) -> None:
        (Path(snapshot_dir) / "chunks.pkl").write_bytes(b"partial")

    def upsert_embeddings(self, documents, embeddings) -> None:
        raise AssertionError("空文档不应写入 embedding")


class _EmptyPipeline:
    def process(self, document):
        return []


def test_build_validates_temp_snapshot_before_activation(tmp_path, monkeypatch):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "article.md").write_text("内容", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    previous_snapshot = snapshot_root / "previous"
    previous_snapshot.mkdir()
    (previous_snapshot / "keep.txt").write_text("保留", encoding="utf-8")
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text("previous", encoding="utf-8")

    run_config = _RunConfig(knowledge_base, snapshot_root)
    repository = SnapshotRepository(run_config)
    monkeypatch.setattr(repository, "generate_snapshot_id", lambda: "kb-new")
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )

    embedding_service = SimpleNamespace(
        embedding_provider_key="local-hash",
        embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        embedding_service,
        repository,
    )

    with pytest.raises(FileNotFoundError, match="知识快照不完整"):
        asyncio.run(service.build("standard"))

    assert (snapshot_root / "ACTIVE_SNAPSHOT").read_text(encoding="utf-8") == "previous"
    assert (previous_snapshot / "keep.txt").read_text(encoding="utf-8") == "保留"
    assert not (snapshot_root / "kb-new").exists()
    assert not (snapshot_root / ".tmp-kb-new").exists()
