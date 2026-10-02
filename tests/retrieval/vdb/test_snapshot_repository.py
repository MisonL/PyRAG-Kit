import pickle
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
    root,
    snapshot_id,
    chunk_count,
    embedding_rows,
    declared_chunk_count=None,
    index_rows=None,
):
    """按真实结构写一个快照，chunk/embedding/索引行数可独立指定。"""
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
    # 写一个**可解析**的 FAISS 索引：占位字节会让 ``faiss.read_index`` 抛
    # RuntimeError，而真实加载路径（``FaissStore.load_snapshot``）同样会读它，
    # 所以占位文件构造出的快照本来就不可能被加载。
    import faiss

    index = faiss.IndexFlatL2(4)
    index.add(
        np.zeros((index_rows if index_rows is not None else embedding_rows, 4), dtype=np.float32)
    )
    faiss.write_index(index, str(snapshot_dir / "semantic.index"))
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


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    """构造一个快照根目录落在 ``tmp_path`` 下的 RunConfig。"""
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    monkeypatch.setattr("src.utils.config.CONFIG_TOML_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text('snapshot_root = "data/kb"\n', encoding="utf-8")
    return build_run_config(get_settings())


# ── 回归：标记文件的内容会被拼进路径，必须是合法的快照 id ──


@pytest.mark.parametrize(
    "bad_root",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(None, id="none"),
    ],
)
def test_snapshot_repository_rejects_non_path_snapshot_root(run_config, bad_root):
    """``snapshot_root`` 非 ``Path`` 时必须给 ``ValueError``，不是裸 ``AttributeError``。

    ``SnapshotRepository.__init__`` 直接对 ``run_config.snapshot_root`` 调
    ``.mkdir()``，而 ``RunConfig`` 是 frozen dataclass、字段没有校验器（同仓库多处
    已认领这条可达性），嵌入方可以自拼一个 ``snapshot_root=5`` 的配置。实测抛裸
    ``AttributeError: 'int' object has no attribute 'mkdir'``——不是本类其余拒绝
    路径承诺的带指引 ``ValueError``。``Path`` 之外的形态一律拒绝，不做静默强转：
    ``snapshot_root`` 的契约就是 ``Path``（``RunConfig`` 的标注）。
    """
    from dataclasses import replace

    with pytest.raises(ValueError, match="snapshot_root"):
        SnapshotRepository(replace(run_config, snapshot_root=bad_root))


@pytest.mark.parametrize(
    "payload",
    [
        "../../../tmp/elsewhere",  # 相对逃逸
        "/tmp/elsewhere",  # 绝对路径会顶掉 root
        "kb-abc/../../../etc",  # 含分隔符
        "sub/dir",
        "..",
    ],
)
def test_active_snapshot_id_rejects_path_like_values(run_config, payload):
    """标记文件是磁盘上的任意文本，不能直接当路径分量用。

    ``Path("/root") / "/tmp/x"`` 得 ``/tmp/x``——绝对路径会**顶掉** root，
    ``../`` 则向上逃逸。两种都会让「活动快照」指到快照根目录之外。
    """
    # 先构造仓库（它会建出快照根目录），再写标记文件——与真实流程一致。
    repository = SnapshotRepository(run_config)
    run_config.active_snapshot_marker.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.get_active_snapshot_id()


def test_active_snapshot_id_accepts_generated_ids(run_config):
    """收紧不能误伤：本应用自己生成的 id 形态必须照常接受。"""
    repository = SnapshotRepository(run_config)
    snapshot_id = repository.generate_snapshot_id()
    run_config.active_snapshot_marker.write_text(snapshot_id, encoding="utf-8")

    assert repository.get_active_snapshot_id() == snapshot_id


@pytest.mark.parametrize(
    "hostile",
    ["../outside", "../../etc", "/tmp/elsewhere", "a/../..", ".."],
)
def test_snapshot_id_is_rejected_by_every_path_building_method(run_config, hostile):
    """四个把 snapshot_id 拼进路径的方法都必须先校验。

    ``create_temp_snapshot_dir`` / ``cleanup_temp_snapshot_dir`` /
    ``finalize_snapshot`` 都对拼出的路径执行 ``rmtree``。实测
    ``finalize_snapshot(root/".tmp-x", "../../<victim>")`` 会把快照根目录**之外**
    的目录整个删掉——逃逸即等于删掉不属于自己的东西。
    """
    repository = SnapshotRepository(run_config)

    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.create_temp_snapshot_dir(hostile)
    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.cleanup_temp_snapshot_dir(hostile)
    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.finalize_snapshot(run_config.snapshot_root / ".tmp-x", hostile)


def test_finalize_snapshot_does_not_delete_outside_the_root(run_config, tmp_path):
    """端到端：恶意 id 不得让 root 外的目录被删掉。

    ``snapshot_root`` 是 ``tmp_path/data/kb``，因此 ``tmp_path`` 本身就在
    root 之外——用 ``../victim-dir`` 就能构造出「删除会波及 root 外」的场景，
    且目标目录随用例唯一，不会被其他用例的残留污染。
    """
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    (victim / "important.txt").write_text("用户数据", encoding="utf-8")

    repository = SnapshotRepository(run_config)
    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.finalize_snapshot(run_config.snapshot_root / ".tmp-x", f"../{victim.name}")

    assert victim.exists(), "快照根目录之外的目录被删掉了"
    assert (victim / "important.txt").is_file()


@pytest.mark.parametrize(
    "hostile_id",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(10**4301, id="beyond-int-str-digits"),
        pytest.param(b"abc", id="bytes"),
    ],
)
def test_snapshot_id_guard_rejects_non_path_forms(run_config, hostile_id):
    """非路径形态的 ``snapshot_id`` 必须收敛成 ``ValueError``，不是裸 ``TypeError``。

    ``_require_snapshot_id_within_root`` 第一行就写 ``Path(snapshot_id)``，而
    ``Path(5)``／``Path(["a"])`` 抛裸 ``TypeError: expected str, bytes or
    os.PathLike object, not ...``——不是本类承诺的带指引 ``ValueError``，也没有
    「快照 id 不合法」的语义。位数超限的整数更隐蔽：``Path(10**4301)`` 抛的是
    ``ValueError: Exceeds the limit``，类型对、内容却与快照 id 无关。

    三个 ``*_temp_snapshot_dir``／``finalize_snapshot`` 方法都对拼出的路径做
    ``rmtree``，而快照 id 的来源之一是磁盘上的标记文件；形态判据必须与同仓库的
    ``resolve_within``／``ensure_trusted_source`` 一致（构造 ``Path`` 之前先判
    形态）。
    """
    repository = SnapshotRepository(run_config)

    with pytest.raises(ValueError, match="快照 id"):
        repository.create_temp_snapshot_dir(hostile_id)
    with pytest.raises(ValueError, match="快照 id"):
        repository.cleanup_temp_snapshot_dir(hostile_id)
    with pytest.raises(ValueError, match="快照 id"):
        repository.finalize_snapshot(run_config.snapshot_root / ".tmp-x", hostile_id)


def test_snapshot_id_guard_does_not_break_the_normal_flow(run_config):
    """收紧不能误伤：生成 id 的完整流程必须照常走通。"""
    repository = SnapshotRepository(run_config)
    snapshot_id = repository.generate_snapshot_id()
    temp_dir = repository.create_temp_snapshot_dir(snapshot_id)
    (temp_dir / "chunks.pkl").write_bytes(b"x")

    final_dir = repository.finalize_snapshot(temp_dir, snapshot_id)
    assert final_dir.exists()
    assert final_dir.parent == Path(run_config.snapshot_root).resolve()

    repository.cleanup_temp_snapshot_dir(snapshot_id)


def test_validate_snapshot_dir_rejects_index_row_mismatch(tmp_path):
    """向量索引条数与分块数不一致时必须拒绝。

    ``FaissStore.load_snapshot`` 在 ``lexical.index`` 存在时**不会**重建 FAISS
    索引，直接使用盘上的 ``semantic.index``。实测 ntotal 比 chunks 多
    （3 vs 5）时本方法曾放行，加载后 ``semantic_search`` 拿 ``indices`` 去索引
    ``self.documents`` 抛 ``IndexError: list index out of range``——正是本方法
    要防的那类失败。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-index-mismatch", chunk_count=3, embedding_rows=3, index_rows=5
    )

    with pytest.raises(ValueError, match="向量索引条数与分块数不一致"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_accepts_matching_index_rows(tmp_path):
    """收紧不能误伤：索引条数与分块数一致时必须通过。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-index-ok", chunk_count=3, embedding_rows=3, index_rows=3
    )

    repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_rejects_lexical_row_mismatch(tmp_path):
    """关键词索引语料条数与分块数不一致时必须拒绝。

    ``keyword_search`` 拿 ``np.argsort(doc_scores)`` 的下标去索引
    ``self.documents``；语料比分块多时，命中越界位置的文档会让下标超出
    ``documents`` 而抛 ``IndexError``（实测语料 5 条、分块 3 条时复现）。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-lexical-mismatch", chunk_count=3, embedding_rows=3
    )
    import pickle

    (snapshot_dir / "lexical.index").write_bytes(pickle.dumps([["c"]] * 5))

    with pytest.raises(ValueError, match="关键词索引语料条数与分块数不一致"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_accepts_matching_lexical_rows(tmp_path):
    """收紧不能误伤：语料条数与分块数一致时必须通过。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-lexical-ok", chunk_count=3, embedding_rows=3
    )
    import pickle

    (snapshot_dir / "lexical.index").write_bytes(pickle.dumps([["c"]] * 3))

    repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：parents.pkl 与另外两个 pickle 一样必须可解析、形态正确 ──
#
# ``validate_snapshot_dir`` 把 ``chunks.pkl`` 与 ``lexical.index`` 的解析失败都
# 转成了「快照不自洽」的 ValueError，却漏了 ``parents.pkl``——它是同一批
# ``pickle.load`` 的目标之一，同样是 required file。实测 ``parents.pkl`` 写坏时
# 校验照常通过，直到 ``FaissStore.load_snapshot`` 抛裸的 ``UnpicklingError``：
# 用户看到的是 ``UnpicklingError: invalid load key, 'n'.``，既没有「快照坏了」
# 的语义，也没有「请重建」的指引；形态不对（``None``／``int``）则抛
# ``TypeError: 'NoneType' object is not iterable``。同一份快照里
# ``chunks.pkl`` 坏掉时给出的却是有指引的 ValueError，两者不一致。


def test_validate_snapshot_dir_rejects_unparseable_parents():
    """``parents.pkl`` 无法解析时必须报「快照不自洽」而不是漏到加载期。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-parentsbad01", chunk_count=3, embedding_rows=3
    )
    (snapshot_dir / "parents.pkl").write_bytes(b"not a pickle at all")

    with pytest.raises(ValueError, match="parents.pkl"):
        repository.validate_snapshot_dir(snapshot_dir)


@pytest.mark.parametrize("payload", [None, 42, ["p0", "p1"], "p0"])
def test_validate_snapshot_dir_rejects_non_mapping_parents(payload):
    """``parents.pkl`` 的顶层必须是映射。

    非映射在加载期会以 ``TypeError: 'NoneType' object is not iterable`` 这类
    与快照无关的形态抛出（``dict(self.parent_documents)``）。
    """
    import pickle

    from src.retrieval.snapshot_repository import SnapshotRepository

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-parentsshp1", chunk_count=3, embedding_rows=3
    )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump(payload, file)

    with pytest.raises(ValueError, match="parents.pkl"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_rejects_parent_entries_without_content():
    """父条目不是 dict／str 时无法取出内容，必须在加载前拒绝。"""
    import pickle

    from src.retrieval.snapshot_repository import SnapshotRepository

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-parentsent1", chunk_count=3, embedding_rows=3
    )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({"p0": ["not", "a", "document"]}, file)

    with pytest.raises(ValueError, match="parents.pkl"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_snapshot_dir_accepts_well_formed_parents():
    """收紧不能误伤：正常形态的 ``parents.pkl`` 必须照常通过。"""
    import pickle

    from src.retrieval.snapshot_repository import SnapshotRepository

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-parentsok01", chunk_count=3, embedding_rows=3
    )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump(
            {
                "p0": {"content": "父内容", "metadata": {"source": "s0.md"}},
                "p1": "字符串形态的父内容",
            },
            file,
        )

    repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：manifest.toml 解析失败要给出「快照坏了」的语义 ──
#
# ``validate_snapshot_dir`` 把 chunks.pkl / parents.pkl / stats.json /
# semantic.index / lexical.index 的解析失败都转成了「快照不自洽」的 ValueError
# 并附带重建指引，唯独 **manifest.toml** 没走这一步：它只查文件是否存在，不查
# 能否解析。实测把 manifest 写成 `a = "unterminated` 后
# ``validate_snapshot_dir`` 照常通过，紧接着 ``load_manifest`` 抛裸的
# ``TOMLDecodeError: Unterminated string``——而 ``VectorStoreFactory`` 在启动时
# 正是「先 validate 再 load_manifest」，用户看到的是一个与「快照坏了」无关的
# 异常类型，也没有重建指引。同类失败不该有两种收场方式。


def test_load_manifest_reports_malformed_toml_as_inconsistent_snapshot(tmp_path, monkeypatch):
    """manifest 语法错要给「快照不自洽 + 重建指引」，而不是裸 TOMLDecodeError。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    snapshot_dir = snapshot_root / "kb-broken-manifest"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.toml").write_text('a = "unterminated', encoding="utf-8")

    with pytest.raises(ValueError, match="重建") as excinfo:
        repository.load_manifest(snapshot_dir)

    # 原始异常类型保留在消息里，便于定位（与其它快照文件的处理一致）
    assert "TOMLDecodeError" in str(excinfo.value)


def test_validate_snapshot_dir_rejects_unparsable_manifest(tmp_path, monkeypatch):
    """校验入口就要拦下坏 manifest——否则「先 validate 再 load」的调用方白跑一趟。"""
    import pickle

    import numpy as np

    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    snapshot_dir = snapshot_root / "kb-broken-manifest"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.toml").write_text('a = "unterminated', encoding="utf-8")
    # 其余文件保持自洽，确保拒绝的原因就是 manifest 本身
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "x", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    (snapshot_dir / "stats.json").write_text('{"chunk_count": 1}', encoding="utf-8")

    import faiss

    index = faiss.IndexFlatL2(4)
    index.add(np.zeros((1, 4), dtype=np.float32))
    faiss.write_index(index, str(snapshot_dir / "semantic.index"))

    with pytest.raises(ValueError, match="manifest"):
        repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：快照 id 必须是**单一目录名** ──
#
# 原先的判定是「解析后的父目录等于 root」，它拦不住 ``"x/../victim"``：该值
# 解析成 ``root/victim``，父目录**正是** root，于是放行。而三个方法实际操作的
# 路径是 ``self.root / f".tmp-{snapshot_id}"``——前缀 ``.tmp-`` 会**改变** ``..``
# 的消解结果（``.tmp-x`` 是一个真实路径分量），``".tmp-x/../victim"`` 最终落在
# ``root/victim``，即一个**正式快照**。实测
# ``cleanup_temp_snapshot_dir("x/../victim")`` 对 ``root/victim`` 执行了 rmtree：
# 「只清理任务临时目录」的意图被绕过，删的是正式快照。


@pytest.mark.parametrize(
    "snapshot_id",
    ["x/../victim", "../../victim", "..", ".", "/absolute", "a/b", ""],
)
def test_snapshot_id_must_be_a_single_path_component(tmp_path, monkeypatch, snapshot_id):
    """含分隔符、``.``／``..``、绝对路径或空的快照 id 一律拒绝。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    with pytest.raises(ValueError, match="快照 id 不合法"):
        repository.cleanup_temp_snapshot_dir(snapshot_id)


def test_snapshot_id_accepts_ids_produced_by_the_generator(tmp_path, monkeypatch):
    """收紧不能误伤：``generate_snapshot_id()`` 的产物必须照常可用。"""
    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    for prefix in ("kb", "legacy"):
        snapshot_id = repository.generate_snapshot_id(prefix=prefix)
        temp_dir = repository.create_temp_snapshot_dir(snapshot_id)

        assert temp_dir.exists()
        repository.cleanup_temp_snapshot_dir(snapshot_id)


# ── 回归：chunks.pkl 的顶层与向量维度 ──


def test_validate_rejects_chunks_pickle_with_none_top_level(tmp_path, monkeypatch):
    """``chunks.pkl=None`` 原先被当作 0 个分块放行，加载时才抛裸 ``TypeError``。"""
    import pickle

    import faiss
    import numpy as np

    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_dir = _build_minimal_snapshot(tmp_path, monkeypatch)
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(None, file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((0, 4), dtype=np.float32))
    faiss.write_index(faiss.IndexFlatL2(4), str(snapshot_dir / "semantic.index"))
    (snapshot_dir / "stats.json").write_text('{"chunk_count": 0}', encoding="utf-8")
    (snapshot_dir / "lexical.index").unlink(missing_ok=True)

    repository = SnapshotRepository(type("RunConfig", (), {"snapshot_root": tmp_path})())
    with pytest.raises(ValueError, match="chunks.pkl 的顶层"):
        repository.validate_snapshot_dir(snapshot_dir)


def test_validate_rejects_embedding_index_dimension_mismatch(tmp_path, monkeypatch):
    """行数一致但**维度**不一致时也必须拒绝。

    实测分块数／embedding 行数／ntotal 都是 1，而 ``embeddings.npy`` 是 4 维、
    ``semantic.index.d`` 是 3 时校验放行；加载后 ``semantic_search`` 把 4 维查询
    向量送进 3 维索引，FAISS 抛的是**空消息的 AssertionError**。
    """
    import pickle

    import faiss
    import numpy as np

    from src.retrieval.snapshot_repository import SnapshotRepository

    snapshot_dir = _build_minimal_snapshot(tmp_path, monkeypatch)
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "x", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    index = faiss.IndexFlatL2(3)
    index.add(np.zeros((1, 3), dtype=np.float32))
    faiss.write_index(index, str(snapshot_dir / "semantic.index"))
    (snapshot_dir / "stats.json").write_text('{"chunk_count": 1}', encoding="utf-8")
    (snapshot_dir / "lexical.index").unlink(missing_ok=True)

    repository = SnapshotRepository(type("RunConfig", (), {"snapshot_root": tmp_path})())
    with pytest.raises(ValueError, match="维度"):
        repository.validate_snapshot_dir(snapshot_dir)


def _build_minimal_snapshot(tmp_path, monkeypatch):
    """造一个受信根目录下的空快照目录，供上面两个用例填充。

    信任根用 ``RunConfig`` 替身声明（与 ``test_snapshot_repository_cleans_only_
    requested_temp_dir`` 同一约定），而不是改 ``config.toml``：``build_run_config``
    解析出的根与这里必须一致，否则 ``validate_snapshot_dir`` 会先报「不在受信
    根目录内」，掩盖本用例要验的内容校验。
    """
    snapshot_dir = tmp_path / "kb-probe"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.toml").write_text(
        KnowledgeSnapshotManifest.create(
            snapshot_id="kb-probe",
            store_type="faiss",
            embedding_provider="local-hash",
            embedding_model="local-hash-256",
            chunk_mode="standard",
            source_digest="d",
            document_count=0,
            chunk_count=0,
        ).to_toml(),
        encoding="utf-8",
    )
    return snapshot_dir


# ── 回归：manifest 与 stats.json 的字段解析失败也要有重建指引 ──
#
# ``load_manifest`` 与 ``validate_snapshot_dir`` 此前只包住了 ``tomllib.load``
# 的失败，**解析之后的字段取值**没有包：``from_mapping`` 里 ``int(data[...])``
# 与 ``data[...]`` 直接取值，于是
#   * ``chunk_count = "abc"`` -> 裸 ``ValueError: invalid literal for int()``
#   * ``chunk_count = "9"*5000`` -> 裸 ``ValueError: Exceeds the limit (4300 digits)``
#   * 字段缺失 -> 裸 ``KeyError: 'chunk_count'``
# 三者都没有「快照不自洽」的语义与重建指引，而同一份文件语法错时给的正是那种
# 消息——同类失败不该有两种收场方式。``stats.json`` 的 ``int(declared)`` 同理
# （``"abc"``／列表／超长数字串分别抛 ValueError／TypeError／ValueError）。


def _write_manifest_snapshot(root, snapshot_id, manifest_text):
    """写一个只有 manifest.toml 内容可控的快照目录。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = Path(root) / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "manifest.toml").write_text(manifest_text, encoding="utf-8")
    return repository, snapshot_dir


def _manifest_toml():
    from src.runtime.contracts import KnowledgeSnapshotManifest

    return KnowledgeSnapshotManifest.create(
        snapshot_id="kb-manifest",
        store_type="faiss",
        embedding_provider="local-hash",
        embedding_model="local-hash-256",
        chunk_mode="standard",
        source_digest="test-digest",
        document_count=3,
        chunk_count=3,
    ).to_toml()


def test_load_manifest_reports_unparseable_counts_as_inconsistent_snapshot():
    """``chunk_count`` 写成非数值时必须是带重建指引的 ``ValueError``。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    _, snapshot_dir = _write_manifest_snapshot(
        repository.root,
        "kb-bad-count",
        _manifest_toml().replace("chunk_count = 3", 'chunk_count = "abc"'),
    )

    with pytest.raises(ValueError, match="重建") as excinfo:
        repository.load_manifest(snapshot_dir)

    assert "不自洽" in str(excinfo.value)
    # 原始异常类型保留在消息里，便于定位
    assert "ValueError" in str(excinfo.value)


def test_load_manifest_reports_missing_fields_as_inconsistent_snapshot():
    """字段缺失时不能抛裸 ``KeyError``。

    ``KeyError`` **不继承** ``ValueError``，所以它既逃出本方法的收场约定，也
    逃出 ``VectorStoreFactory`` 启动路径上任何 ``except ValueError``。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    manifest_without_count = "\n".join(
        line for line in _manifest_toml().splitlines() if not line.startswith("chunk_count")
    )
    _, snapshot_dir = _write_manifest_snapshot(
        repository.root, "kb-missing-count", manifest_without_count
    )

    with pytest.raises(ValueError, match="重建") as excinfo:
        repository.load_manifest(snapshot_dir)

    assert "KeyError" in str(excinfo.value)


def test_load_manifest_reports_oversized_count_literal_as_inconsistent_snapshot():
    """位数超限的数字串同样要走「快照不自洽」，而不是裸的位数超限错误。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    _, snapshot_dir = _write_manifest_snapshot(
        repository.root,
        "kb-huge-count",
        _manifest_toml().replace("chunk_count = 3", 'chunk_count = "' + "9" * 5000 + '"'),
    )

    with pytest.raises(ValueError, match="重建") as excinfo:
        repository.load_manifest(snapshot_dir)

    assert "不自洽" in str(excinfo.value)


def test_load_manifest_still_accepts_a_valid_manifest():
    """收紧不能误伤：合法 manifest 照常通过。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    _, snapshot_dir = _write_manifest_snapshot(repository.root, "kb-ok", _manifest_toml())

    manifest = repository.load_manifest(snapshot_dir)

    assert manifest.chunk_count == 3


@pytest.mark.parametrize(
    "declared,label",
    [
        ("abc", "非数值字符串"),
        ("1.5", "浮点字符串"),
        ([1], "列表"),
        ("9" * 5000, "位数超限的数字串"),
    ],
)
def test_validate_snapshot_dir_reports_unparseable_stats_count(declared, label):
    """``stats.json`` 的 ``chunk_count`` 形态异常时必须给重建指引。

    ``int(declared)`` 直接调用会抛裸的 ``ValueError``／``TypeError``——都没有
    「快照不自洽」的语义，而紧邻的「数值不等」分支给的正是那种消息。
    """
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-bad-stats-shape", chunk_count=3, embedding_rows=3
    )
    import json

    (snapshot_dir / "stats.json").write_text(
        json.dumps({"chunk_count": declared}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="重建") as excinfo:
        repository.validate_snapshot_dir(snapshot_dir)

    assert "stats.json" in str(excinfo.value), label


# ── 回归：快照 id 的长度上界 ──
#
# ``_require_snapshot_id_within_root`` 原判据对超长名字**全部为真**（单分量、
# 相对、非 ``.``），但拼成路径后 ``.exists()`` 抛
# ``OSError: [Errno 63] File name too long``。而
# ``VectorStoreFactory._load_existing_state`` 的第一步就是
# ``get_active_snapshot_dir()``，因此一个被外部工具写坏的标记文件会让**启动**
# 以 ``OSError`` 失败，而本类的契约是带指引的 ``ValueError``。


def test_active_snapshot_id_rejects_overlong_value(run_config):
    """超长 id 必须在拼接路径之前被拒，不能等到 ``stat`` 抛 ``OSError``。"""
    repository = SnapshotRepository(run_config)
    run_config.active_snapshot_marker.write_text("z" * 300, encoding="utf-8")

    with pytest.raises(ValueError, match="过长"):
        repository.get_active_snapshot_dir()


def test_active_snapshot_id_length_bound_does_not_reject_generated_ids(run_config):
    """上界不能误伤本应用自己生成的 id。"""
    repository = SnapshotRepository(run_config)
    snapshot_id = repository.generate_snapshot_id(prefix="legacy")
    run_config.active_snapshot_marker.write_text(snapshot_id, encoding="utf-8")

    assert repository.get_active_snapshot_id() == snapshot_id


# ── 回归：``parents.pkl`` 的错误消息不得自己抛 ──
#
# ``invalid_parents`` 用 ``str(key)`` 拼装，而键来自磁盘上的 parents.pkl
# （pickle 是二进制格式，不受 CPython 十进制位数限制）。位数超限时，**正要报
# 「父条目形态不对」**的那一步自己抛 ``ValueError: Exceeds the limit``——用户
# 拿到与快照无关的整数转换错误。


def test_validate_snapshot_dir_reports_huge_integer_parent_key(tmp_path):
    """父条目非法且键是超大整数时，必须是带重建指引的 ``ValueError``。"""
    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.utils.config import get_settings

    repository = SnapshotRepository(build_run_config(get_settings()))
    snapshot_dir = _write_row_counts_snapshot(
        repository.root, "kb-huge-parent-key", chunk_count=1, embedding_rows=1
    )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({10**4301: [1, 2]}, file)

    with pytest.raises(ValueError, match="父条目"):
        repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：符号链接环必须收敛成 ``ValueError`` ──
#
# ``_require_snapshot_id_within_root`` 里 ``Path(self.root / snapshot_id)
# .resolve()`` 对**符号链接环**抛 ``RuntimeError: Symlink loop from ...``。
# 它**不是** ``ValueError``，穿透本类的契约（见上方
# ``test_active_snapshot_id_rejects_overlong_value`` 的注释：「本类的契约是带
# 指引的 ``ValueError``」）。同项目的两个兄弟判据早已为同一情况收敛：
#
#   * ``utils/security.resolve_within`` 明确把 ``(OSError, RuntimeError)``
#     收成 ``ValueError``，注释写明「一条『拒绝加载』的消息变成 pathlib 的
#     内部错误」；
#   * ``utils/cleanup._reject_unsafe_cache_dir`` 同样收敛，理由同源。
#
# 可达性：``ACTIVE_SNAPSHOT`` 是磁盘上的任意文本，标记内容直接拼进路径。用户
# 手工摆一个环（或外部工具写坏）即触发，而 ``VectorStoreFactory
# ._load_existing_state`` 的第一步就是 ``get_active_snapshot_dir()``——启动
# 会以 pathlib 的内部错误收场，而不是「标记文件坏了，请重建快照」。


def test_active_snapshot_id_rejects_symlink_loop(tmp_path, monkeypatch):
    """标记内容指向符号链接环时，必须抛 ``ValueError`` 而不是 ``RuntimeError``。"""
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))

    loop = repository.root / "loop"
    loop.symlink_to(loop)
    repository.run_config.active_snapshot_marker.write_text("loop", encoding="utf-8")

    with pytest.raises(ValueError):
        repository.get_active_snapshot_id()


def test_active_snapshot_id_rejects_a_directory_marker(tmp_path, monkeypatch):
    """标记路径是**目录**时必须给 ``ValueError``，不是裸 ``IsADirectoryError``。

    ``get_active_snapshot_id`` 只判 ``marker.exists()``（目录也为真）就直接
    ``read_text``，``IsADirectoryError``（``OSError`` 子类）原样冒到调用方。
    本类的契约是带指引的 ``ValueError``，而 ``VectorStoreFactory
    ._load_existing_state`` 的第一步就是本方法——启动会以裸 ``OSError`` 收场。
    """
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))
    repository.run_config.active_snapshot_marker.mkdir()

    with pytest.raises(ValueError):
        repository.get_active_snapshot_id()


def test_active_snapshot_id_rejects_a_non_utf8_marker(tmp_path, monkeypatch):
    """标记文件不是 UTF-8 时必须给 ``ValueError``，不是裸 ``UnicodeDecodeError``。

    ``UnicodeDecodeError`` 继承 ``ValueError``，但它**不是**本类其余拒绝路径
    给出的那种带重建指引的消息；调用方按消息内容分类/展示时接不住。标记文件
    是磁盘上的任意字节，外部工具写坏即触发。
    """
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    monkeypatch.setattr("src.utils.config.ROOT_DIR", tmp_path)
    (tmp_path / "config.toml").write_text('snapshot_root = "snapshots"\n', encoding="utf-8")
    get_settings.cache_clear()
    repository = SnapshotRepository(build_run_config(get_settings()))
    repository.run_config.active_snapshot_marker.write_bytes(b"\xff\xfe bad")

    with pytest.raises(ValueError) as excinfo:
        repository.get_active_snapshot_id()

    message = str(excinfo.value)
    assert "ACTIVE_SNAPSHOT" in message or "标记" in message, message


def test_load_snapshot_rejects_a_directory_marker(tmp_path):
    """``FaissStore.load_snapshot`` 对目录形态的标记文件给 ``ValueError``。

    与 ``SnapshotRepository.get_active_snapshot_id`` 同一处、同一判据：两者都
    在 ``VectorStoreFactory._load_existing_state`` 的启动路径上，标记路径是目录
    时 ``read_text`` 抛裸 ``IsADirectoryError``。
    """
    from src.retrieval.vdb.faiss_store import FaissStore

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    snapshot = snapshot_root / "kb-x"
    snapshot.mkdir()
    (snapshot_root / "ACTIVE_SNAPSHOT").mkdir()

    store = FaissStore()
    with pytest.raises(ValueError) as excinfo:
        store.load_snapshot(str(snapshot), snapshot_root=snapshot_root)

    message = str(excinfo.value)
    assert "ACTIVE_SNAPSHOT" in message, message


def test_load_snapshot_rejects_a_non_utf8_marker(tmp_path):
    """标记文件不是 UTF-8 时 ``FaissStore.load_snapshot`` 给 ``ValueError``。"""
    from src.retrieval.vdb.faiss_store import FaissStore

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    snapshot = snapshot_root / "kb-x"
    snapshot.mkdir()
    (snapshot_root / "ACTIVE_SNAPSHOT").write_bytes(b"\xff\xfe bad")

    store = FaissStore()
    with pytest.raises(ValueError) as excinfo:
        store.load_snapshot(str(snapshot), snapshot_root=snapshot_root)

    message = str(excinfo.value)
    assert "ACTIVE_SNAPSHOT" in message, message
