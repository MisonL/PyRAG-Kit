import numpy as np
import pytest

from src.retrieval.vdb.faiss_store import FaissStore


def test_faiss_store_rejects_snapshot_without_embeddings_before_writing_npy(tmp_path):
    store = FaissStore(file_path=None)

    with pytest.raises(RuntimeError, match="语义索引为空"):
        store.save_snapshot(str(tmp_path / "snapshot"))

    assert not (tmp_path / "snapshot" / "embeddings.npy").exists()


def _activate_snapshot(snapshot_root, snapshot_id: str) -> None:
    """写入 ACTIVE_SNAPSHOT 标记，让快照目录成为「本应用管理的活动快照」。

    ``load_snapshot`` 现在会校验归属与激活状态（快照内的文件是 pickle，
    路径本身不能证明来源），所以测试必须构造出与真实运行一致的目录结构。
    """
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text(snapshot_id, encoding="utf-8")


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
    _activate_snapshot(tmp_path, "snapshot")

    reloaded = FaissStore(file_path=None)
    reloaded.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)

    assert reloaded.resolve_parent_content("parent-1") == "parent content"
    assert reloaded.parent_documents["parent-1"]["metadata"]["source"] == "kb.md"
    assert reloaded.documents[0]["metadata"]["parent_id"] == "parent-1"


def test_faiss_store_rejects_query_dimension_mismatch():
    store = FaissStore(file_path=None)
    store.documents = [{"page_content": "doc", "metadata": {}}]
    store.embeddings = np.array([[0.0, 0.0]], dtype=np.float32)
    store._rebuild_indices()

    with pytest.raises(ValueError, match="维度"):
        store.semantic_search([0.0, 0.0, 0.0])


# ── 回归：pickle 加载入口的信任边界 ──


def _make_valid_snapshot(snapshot_root, snapshot_id="kb-ok", *, activate=True):
    """造一个最小但自洽的快照目录（1 个分块 + 1 行向量）。"""
    import pickle

    snapshot_dir = snapshot_root / snapshot_id
    snapshot_dir.mkdir(parents=True)
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "内容", "metadata": {"source": "s.md"}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    # lexical.index 也要是合法 pickle：空 bytes 会让 pickle.load 抛 EOFError，
    # 那是「文件被截断」而不是本节要测的信任边界。
    with (snapshot_dir / "lexical.index").open("wb") as file:
        pickle.dump([["内容"]], file)
    if activate:
        (snapshot_root / "ACTIVE_SNAPSHOT").write_text(snapshot_id, encoding="utf-8")
    return snapshot_dir


def test_load_snapshot_requires_trust_root(tmp_path):
    """不声明 snapshot_root 时必须拒绝。

    快照内的 chunks.pkl / parents.pkl / lexical.index 都是 pickle，反序列化即
    执行代码。没有信任声明时不能退化成「任意目录都可加载」，否则这个边界等于
    不存在——那正是 AGENTS.md 要求「不导入不可信来源的 .pkl」被架空的形态。
    """
    snapshot_dir = _make_valid_snapshot(tmp_path / "snapshots")

    with pytest.raises(ValueError, match="缺少 snapshot_root"):
        FaissStore(file_path=None).load_snapshot(str(snapshot_dir))


def test_load_snapshot_rejects_directory_outside_trust_root(tmp_path):
    """信任根之外的目录必须拒绝，即使目录结构看起来完全合法。"""
    outside = tmp_path / "outside"
    snapshot_dir = _make_valid_snapshot(outside)
    trust_root = tmp_path / "snapshots"
    trust_root.mkdir()
    (trust_root / "ACTIVE_SNAPSHOT").write_text("kb-ok", encoding="utf-8")

    with pytest.raises(ValueError, match="必须位于受信根目录内"):
        FaissStore(file_path=None).load_snapshot(str(snapshot_dir), snapshot_root=trust_root)


def test_load_snapshot_rejects_symlink_escape(tmp_path):
    """信任根内的符号链接指向根外时必须拒绝（resolve 后再比归属）。"""
    real_root = tmp_path / "real"
    snapshot_dir = _make_valid_snapshot(real_root)
    trust_root = tmp_path / "snapshots"
    trust_root.mkdir()
    escape = trust_root / "kb-link"
    escape.symlink_to(snapshot_dir)
    (trust_root / "ACTIVE_SNAPSHOT").write_text("kb-link", encoding="utf-8")

    with pytest.raises(ValueError, match="必须位于受信根目录内"):
        FaissStore(file_path=None).load_snapshot(str(escape), snapshot_root=trust_root)


def test_load_snapshot_rejects_inactive_snapshot(tmp_path):
    """在信任根之下但未被 ACTIVE_SNAPSHOT 激活的目录必须拒绝。

    仅「在 root 之下」不足以证明目录可用：未激活或半成品目录同样在 root 之下。
    """
    snapshot_root = tmp_path / "snapshots"
    snapshot_dir = _make_valid_snapshot(snapshot_root, snapshot_id="kb-inactive", activate=False)

    with pytest.raises(ValueError, match="未被激活"):
        FaissStore(file_path=None).load_snapshot(str(snapshot_dir), snapshot_root=snapshot_root)


def test_load_snapshot_rejects_when_marker_points_elsewhere(tmp_path):
    """ACTIVE_SNAPSHOT 指向另一个快照时，加载非活动快照必须拒绝。"""
    snapshot_root = tmp_path / "snapshots"
    # 先造两个目录都不激活，再一次把标记写到 kb-active：
    # helper 默认激活，连调两次会让第二次调用把标记覆盖成 kb-other，
    # 那样被加载的目录反而成了活动快照，测不到「非活动」这条分支。
    other = _make_valid_snapshot(snapshot_root, snapshot_id="kb-other", activate=False)
    _make_valid_snapshot(snapshot_root, snapshot_id="kb-active")

    with pytest.raises(ValueError, match="不是当前活动快照"):
        FaissStore(file_path=None).load_snapshot(str(other), snapshot_root=snapshot_root)


def test_load_snapshot_accepts_activated_snapshot(tmp_path):
    """收紧不能误伤：合法活动快照必须能加载并读出内容。"""
    snapshot_root = tmp_path / "snapshots"
    snapshot_dir = _make_valid_snapshot(snapshot_root)

    store = FaissStore(file_path=None)
    store.load_snapshot(str(snapshot_dir), snapshot_root=snapshot_root)

    assert [doc["page_content"] for doc in store.documents] == ["内容"]


def test_load_requires_trusted_paths(tmp_path):
    """legacy pickle 加载同样必须声明来源。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    with pytest.raises(ValueError, match="受信来源声明"):
        FaissStore(file_path=None).load(str(legacy))


def test_load_rejects_untrusted_path(tmp_path):
    """不在任何受信根内的 legacy 文件必须拒绝。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)
    trust_root = tmp_path / "trusted"
    trust_root.mkdir()

    with pytest.raises(ValueError, match="不在任何受信路径内"):
        FaissStore(file_path=None).load(str(legacy), trusted_paths=[trust_root])


def test_load_accepts_trusted_path(tmp_path):
    """收紧不能误伤：声明来源后必须能加载。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump(
            {"documents": [{"page_content": "旧数据", "metadata": {}}], "embeddings": None},
            file,
        )

    store = FaissStore(file_path=None)
    store.load(str(legacy), trusted_paths=[tmp_path])

    assert [doc["page_content"] for doc in store.documents] == ["旧数据"]


def test_init_with_file_path_requires_trusted_paths(tmp_path):
    """``__init__(file_path=...)`` 不再隐式加载，必须先声明来源。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    with pytest.raises(ValueError, match="受信来源声明"):
        FaissStore(file_path=str(legacy))

    store = FaissStore(file_path=str(legacy), trusted_paths=[tmp_path])
    assert store.documents == []
