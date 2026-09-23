from pathlib import Path

import pytest

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


def test_snapshot_repository_cleans_only_requested_temp_dir(tmp_path):
    run_config = type("RunConfig", (), {"snapshot_root": tmp_path})()
    repository = SnapshotRepository(run_config)
    snapshot_id = "kb-cleanup"
    temp_dir = repository.create_temp_snapshot_dir(snapshot_id)
    (temp_dir / "partial.bin").write_bytes(b"partial")
    final_dir = tmp_path / "kb-existing"
    final_dir.mkdir()
    (final_dir / "keep.bin").write_bytes(b"keep")

    repository.cleanup_temp_snapshot_dir(snapshot_id)

    assert not temp_dir.exists()
    assert (final_dir / "keep.bin").exists()


# ── 回归：快照行数自洽（原先只查文件存在性）──


def _write_row_counts_snapshot(
    root, snapshot_id, chunk_count, embedding_rows, declared_chunk_count=None
):
    """按真实结构写一个快照，chunk/embedding 行数可独立指定。"""
    import json
    import pickle

    import numpy as np

    from src.runtime.contracts import KnowledgeSnapshotManifest

    snapshot_dir = Path(root) / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id=snapshot_id,
        store_type="faiss",
        embedding_provider="local-hash",
        embedding_model="local-hash-256",
        chunk_mode="standard",
        source_digest="test-digest",
        document_count=chunk_count,
        chunk_count=chunk_count,
    )
    (snapshot_dir / "manifest.toml").write_text(manifest.to_toml(), encoding="utf-8")
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(
            [
                {"page_content": f"chunk {index}", "metadata": {"source": f"s{index}.md"}}
                for index in range(chunk_count)
            ],
            file,
        )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((embedding_rows, 4), dtype=np.float32))
    (snapshot_dir / "semantic.index").write_bytes(b"")
    (snapshot_dir / "stats.json").write_text(
        json.dumps(
            {
                "chunk_count": declared_chunk_count
                if declared_chunk_count is not None
                else chunk_count
            }
        ),
        encoding="utf-8",
    )
    return snapshot_dir


def test_validate_snapshot_dir_rejects_chunk_embedding_row_mismatch(tmp_path):
    """分块数与向量数不一致时必须拒绝。

    修复前实测：``documents=3`` 而 ``embeddings=(2,4)``（``faiss_index.ntotal=2``）
    的 store，``save_snapshot`` 正常落盘、``validate_snapshot_dir`` 通过、加载后
    也不报错——不一致被完整持久化，之后 ``semantic_search`` 拿 ``indices`` 去索引
    ``self.documents`` 会越界，而报错现场离真正的原因很远。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-mismatch-rows", chunk_count=3, embedding_rows=2
    )

    with pytest.raises(ValueError, match="不自洽"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_rejects_stats_chunk_count_mismatch(tmp_path):
    """``stats.json`` 声明的 chunk_count 与实际分块数不符也必须拒绝。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-bad-stats", chunk_count=3, embedding_rows=3, declared_chunk_count=99
    )

    with pytest.raises(ValueError, match="stats.json"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_accepts_self_consistent_snapshot(tmp_path):
    """行数自洽的快照必须通过——收紧不能把正常快照一起拒掉。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-good", chunk_count=3, embedding_rows=3
    )

    repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_reports_unparseable_chunks_as_value_error(tmp_path):
    """截断的文件应报「快照不自洽」的 ``ValueError``，而不是裸 ``EOFError``。

    加载入口抛与真实原因无关的异常类型会让用户误诊，测试里就复现过占位空文件
    导致的 ``EOFError: Ran out of input``。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-truncated", chunk_count=3, embedding_rows=3
    )
    (snapshot_dir / "chunks.pkl").write_bytes(b"")

    with pytest.raises(ValueError, match="无法解析"):
        repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：manifest 的 schema 版本准入 ──


def test_load_manifest_rejects_unknown_schema_version(tmp_path, monkeypatch):
    """未知 schema 版本必须拒绝，而不是照着当前代码去解释。

    ``from_mapping`` 是纯解析（``str(data["schema_version"])``），不做版本判断；
    版本准入放在 ``load_manifest``。旧/新 schema 的字段语义可能已经变了
    （本项目 v1→v2 调整过分块与 embedding 的落盘结构），静默接受未知版本会得到
    错误结果而不是报错。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    snapshot_dir = snapshot_root / "kb-old-schema"
    snapshot_dir.mkdir()
    # 故意写一个当前程序不认识的 schema 版本
    (snapshot_dir / "manifest.toml").write_text(
        'schema_version = "999"\n'
        'snapshot_id = "kb-old-schema"\n'
        'created_at = "2020-01-01T00:00:00+00:00"\n'
        'store_type = "faiss"\n'
        'embedding_provider = "local-hash"\n'
        'embedding_model = "local-hash-256"\n'
        'chunk_mode = "standard"\n'
        'source_digest = "d"\n'
        "document_count = 1\n"
        "chunk_count = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema") as excinfo:
        repository.load_manifest(snapshot_dir)

    message = str(excinfo.value)
    assert "999" in message
    assert "2" in message


def test_load_manifest_accepts_current_schema_version(tmp_path, monkeypatch):
    """收紧不能误伤：当前版本的 manifest 必须能读回。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    snapshot_dir = snapshot_root / "kb-current"
    snapshot_dir.mkdir()
    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id="kb-current",
        store_type="faiss",
        embedding_provider="local-hash",
        embedding_model="local-hash-256",
        chunk_mode="standard",
        source_digest="d",
        document_count=1,
        chunk_count=1,
    )
    repository.write_manifest(snapshot_dir, manifest)

    loaded = repository.load_manifest(snapshot_dir)

    assert loaded.schema_version == manifest.schema_version
    assert loaded.snapshot_id == "kb-current"


def test_validate_snapshot_dir_rejects_directory_outside_root(tmp_path, monkeypatch):
    """快照目录必须在受信根内：快照内的 pickle 反序列化即执行代码。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    outside = tmp_path / "elsewhere"
    outside.mkdir()

    with pytest.raises(ValueError, match="受信根"):
        repository.validate_snapshot_dir(outside)


def test_validate_snapshot_dir_rejects_symlinked_snapshot_dir(tmp_path, monkeypatch):
    """快照目录本身是符号链接时必须拒绝，即使目标在根内。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    real = repository.root / "kb-real"
    real.mkdir()
    link = repository.root / "kb-link"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="符号链接"):
        repository.validate_snapshot_dir(link)


def test_validate_snapshot_dir_rejects_nested_snapshot_dir(tmp_path, monkeypatch):
    """快照目录必须是根的直接子目录，不能嵌在更深层级。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    nested = repository.root / "outer" / "kb-inner"
    nested.mkdir(parents=True)

    with pytest.raises(ValueError, match="直接位于快照根目录之下"):
        repository.validate_snapshot_dir(nested)
