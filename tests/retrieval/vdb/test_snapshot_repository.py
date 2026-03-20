from src.retrieval.snapshot_repository import SnapshotRepository
from src.runtime.contracts import KnowledgeSnapshotManifest, build_run_config
from src.utils.config import get_settings


def test_snapshot_repository_writes_and_loads_manifest(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    monkeypatch.setattr("src.utils.config.CONFIG_TOML_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text('snapshot_root = "data/kb"\n', encoding="utf-8")

    run_config = build_run_config(get_settings())
    repository = SnapshotRepository(run_config)
    snapshot_id = repository.generate_snapshot_id(prefix="test")
    temp_dir = repository.create_temp_snapshot_dir(snapshot_id)
    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id=snapshot_id,
        store_type="faiss",
        embedding_provider="local-hash",
        embedding_model="local-hash-256",
        chunk_mode="standard",
        source_digest="abc123",
        document_count=2,
        chunk_count=10,
    )

    (temp_dir / "chunks.pkl").write_bytes(b"stub")
    (temp_dir / "parents.pkl").write_bytes(b"stub")
    (temp_dir / "semantic.index").write_bytes(b"stub")
    (temp_dir / "embeddings.npy").write_bytes(b"stub")
    (temp_dir / "stats.json").write_text("{}", encoding="utf-8")
    repository.write_manifest(temp_dir, manifest)
    final_dir = repository.finalize_snapshot(temp_dir, snapshot_id)

    loaded_manifest = repository.load_manifest(final_dir)

    assert repository.get_active_snapshot_id() == snapshot_id
    assert loaded_manifest.snapshot_id == snapshot_id
    assert loaded_manifest.embedding_model == "local-hash-256"
    get_settings.cache_clear()
