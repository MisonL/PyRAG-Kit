from pathlib import Path

import faiss
import numpy as np
import pytest

from src.retrieval.vdb.faiss_store import FaissStore


def test_faiss_store_rejects_snapshot_without_embeddings_before_writing_npy(tmp_path):
    store = FaissStore(file_path=None)

    with pytest.raises(RuntimeError, match="语义索引为空"):
        store.save_snapshot(str(tmp_path / "snapshot"))

    assert not (tmp_path / "snapshot" / "embeddings.npy").exists()


@pytest.mark.parametrize(
    "bad_path",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(10**4301, id="beyond-int-str-digits"),
    ],
)
@pytest.mark.parametrize("method", ["save", "save_snapshot"])
def test_faiss_store_write_entries_reject_non_path_arguments(bad_path, method):
    """写入侧入口的非路径入参必须收敛成 ``ValueError``，不是裸 ``TypeError``。

    ``save``／``save_snapshot`` 是 ``load``／``load_snapshot`` 的写入侧兄弟
    （``VectorStoreBase`` 的契约方法，同一份公开接口），读取侧早已按同一判据
    校验入参形态，写入侧这一半漏了：``Path(5)``／``Path(["a"])`` 抛裸
    ``TypeError: expected str, bytes or os.PathLike object, not ...``，位数超限
    的整数抛 ``ValueError: Exceeds the limit``——类型对、内容却与目标路径无关。
    两条都不在本类其余拒绝路径承诺的带指引 ``ValueError`` 契约内。
    """
    store = FaissStore(file_path=None)
    store.documents = []
    store.embeddings = np.zeros((1, 4), dtype=np.float32)
    store.faiss_index = faiss.IndexFlatL2(4)

    with pytest.raises(ValueError, match="必须是字符串或路径"):
        getattr(store, method)(bad_path)


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


# ``load_snapshot`` 在 ``lexical.index`` 存在时**直接使用盘上的语料**，不重建；
# 而 ``keyword_search`` 拿 ``np.argsort(doc_scores)`` 的下标去索引
# ``self.documents``。语料条数与分块数不等时，下标就对不上：
#   语料 3 条、文档 2 条 -> ``IndexError: list index out of range``
#   语料 1 条、文档 2 条 -> 静默丢结果（用户看到「搜不到」而不是「快照坏了」）
# 顶层不是序列时更早收场，且抛的是裸 ``TypeError``／``AttributeError``，
# 与「快照不自洽」无关、也没有重建指引。
#
# 生产路径已由 ``SnapshotRepository.validate_snapshot_dir`` 挡住这些形态；
# ``load_snapshot`` 是 ``VectorStoreBase`` 的契约方法，直接调它的嵌入方不经过
# 那道校验（与 ``_validate_document_shape`` 同一取舍），因此这里补最小不变量。
@pytest.mark.parametrize(
    ("lexical", "pattern"),
    [
        ([["内容"]], "语料条数与分块数不一致"),  # 1 条语料 vs 2 条文档
        ([["a"], ["b"], ["c"]], "语料条数与分块数不一致"),
        ([], "语料条数与分块数不一致"),
        (None, "语料条数与分块数不一致"),
        ("ab", "顶层不是列表"),  # len("ab") == 2 会静默通过，必须按类型拦
        ({"a": 1}, "顶层不是列表"),
        (3, "顶层不是列表"),  # 原先抛裸 TypeError: object of type 'int' has no len()
    ],
    ids=["少一条", "多一条", "空列表", "None", "字符串", "dict", "int"],
)
def test_load_snapshot_rejects_lexical_index_row_mismatch(tmp_path, lexical, pattern):
    import pickle

    snapshot_dir = tmp_path / "kb-lex"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(
            [
                {"page_content": "alpha", "metadata": {}},
                {"page_content": "beta", "metadata": {}},
            ],
            file,
        )
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))
    with (snapshot_dir / "lexical.index").open("wb") as file:
        pickle.dump(lexical, file)
    _activate_snapshot(tmp_path, "kb-lex")

    store = FaissStore(file_path=None)
    with pytest.raises(ValueError, match=pattern):
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)


def test_load_snapshot_accepts_matching_lexical_index(tmp_path):
    """收紧不能误伤：条数一致时照常加载，且关键词检索可用。"""
    snapshot_dir = _make_valid_snapshot(tmp_path, "kb-lex-ok")
    # ``_make_valid_snapshot`` 只写 1 个分块，这里再补一个，保持与语料一致。
    import pickle

    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(
            [
                {"page_content": "内容", "metadata": {"source": "s.md"}},
                {"page_content": "另一个分块", "metadata": {"source": "s.md"}},
            ],
            file,
        )
    with (snapshot_dir / "lexical.index").open("wb") as file:
        pickle.dump([["内容"], ["另一个分块"]], file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))

    store = FaissStore(file_path=None)
    store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)

    assert len(store.documents) == 2
    assert len(store._tokenized_docs_cache) == 2
    assert store.bm25_index is not None


# ── 回归：lexical.index 的**条目**形态也要拦 ──
#
# 上面那组用例只覆盖了 ``lexical.index`` 的**顶层**形态。条目本身不是序列时，
# 顶层「是列表、条数也对」——两道守卫全部放行，随后 ``BM25Okapi`` 在
# ``_initialize`` 里对每条语料调 ``len(document)``，抛裸
# ``TypeError: object of type 'int' has no len()``。
#
# 这条路径在**生产**上可达：``SnapshotRepository.validate_snapshot_dir`` 只查
# 顶层与条数，``FaissStore.load_snapshot`` 拿到盘上的语料后直接建 BM25。实测
# 语料 ``[7]``、``[10**4301]``、``[1.5]`` 三种形态都能穿过 validate 并在
# load_snapshot 抛 ``TypeError``——异常类型与「快照坏了」无关，也没有重建指引，
# 而同一个文件顶层写坏时给的正是有指引的 ``ValueError``。同类失败不该有两种
# 收场方式。
@pytest.mark.parametrize(
    ("lexical", "label"),
    [
        ([7], "int"),
        ([10**4301], "huge-int"),
        ([1.5], "float"),
        ([None], "None"),
        ([7, ["ok"]], "mixed"),
    ],
    ids=["int", "huge-int", "float", "None", "mixed"],
)
def test_load_snapshot_rejects_non_sequence_lexical_entries(tmp_path, lexical, label):
    """语料条目不是序列时必须给「快照不自洽」的 ``ValueError``，不是裸 ``TypeError``。"""
    import pickle

    snapshot_dir = tmp_path / f"kb-lex-entry-{label}"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "alpha", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    with (snapshot_dir / "lexical.index").open("wb") as file:
        pickle.dump(lexical, file)
    _activate_snapshot(tmp_path, snapshot_dir.name)

    store = FaissStore(file_path=None)
    with pytest.raises(ValueError, match="lexical.index"):
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)


def test_validate_snapshot_dir_rejects_non_sequence_lexical_entries(tmp_path):
    """生产路径的第一道守卫也要拦住，否则坏快照会被 finalize 成活动快照。"""
    import pickle

    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.runtime.contracts import KnowledgeSnapshotManifest

    repository = SnapshotRepository(
        type(
            "RunConfig",
            (),
            {"snapshot_root": tmp_path, "active_snapshot_marker": tmp_path / "ACTIVE_SNAPSHOT"},
        )()
    )
    snapshot_dir = tmp_path / "kb-lex-entry-validate"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "alpha", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    (snapshot_dir / "stats.json").write_text('{"chunk_count": 1}', encoding="utf-8")
    import faiss

    index = faiss.IndexFlatIP(4)
    index.add(np.zeros((1, 4), dtype=np.float32))
    faiss.write_index(index, str(snapshot_dir / "semantic.index"))
    repository.write_manifest(
        snapshot_dir,
        KnowledgeSnapshotManifest.create(
            snapshot_id=snapshot_dir.name,
            store_type="faiss",
            embedding_provider="local-hash",
            embedding_model="local-hash-256",
            chunk_mode="standard",
            source_digest="d",
            document_count=1,
            chunk_count=1,
        ),
    )
    with (snapshot_dir / "lexical.index").open("wb") as file:
        pickle.dump([7], file)

    with pytest.raises(ValueError, match="lexical.index"):
        repository.validate_snapshot_dir(snapshot_dir)


# ── 回归：载入失败必须让 store 保持原状 ──
#
# 两个载入入口原先都是「边解析边写 self」：``load`` 先把 documents／
# embeddings／parent_documents 赋进 self 再调归一化，``load_snapshot`` 每读一个
# 文件就写一次。中途任何一步失败都留下**半更新**的 store——实测一次被拒的
# ``load`` 之后 documents 与 embeddings 是新值，而 faiss_index 与
# _tokenized_docs_cache 还是旧的（索引 2 条对应 1 条文档），随后
# ``semantic_search`` 抛 ``TypeError: 'str' object does not support item
# assignment``：报错现场与真正的原因（载入被拒）毫无关联。
#
# 调用方看到异常后本可以继续用旧数据，半更新状态让这个选项也消失了。
def _legacy_file(tmp_path, payload):
    import pickle

    path = tmp_path / "legacy.pkl"
    with path.open("wb") as file:
        pickle.dump(payload, file)
    return path


def test_load_rejection_leaves_store_untouched(tmp_path):
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {}}, {"page_content": "beta", "metadata": {}}],
        np.zeros((2, 4), dtype=np.float32),
    )
    before = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
        list(store._tokenized_docs_cache),
    )

    path = _legacy_file(tmp_path, {"documents": ["bad orphan"], "parent_documents": {}})
    with pytest.raises(ValueError, match="不是映射"):
        store.load(str(path), trusted_paths=[tmp_path])

    after = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
        list(store._tokenized_docs_cache),
    )
    assert before == after
    # 半更新的直接症状：索引与文档下标错配，检索崩在与原因无关的位置。
    assert [hit["page_content"] for hit in store.semantic_search(np.zeros(4), top_k=5)] == [
        "alpha",
        "beta",
    ]


def test_load_snapshot_rejection_leaves_store_untouched(tmp_path):
    snapshot_dir = tmp_path / "kb-bad"
    snapshot_dir.mkdir()
    import pickle

    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump(["bad orphan"], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    _activate_snapshot(tmp_path, "kb-bad")

    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {}}], np.zeros((1, 4), dtype=np.float32)
    )
    before = (list(store.documents), store.embeddings.shape, store.faiss_index.ntotal)

    with pytest.raises(ValueError, match="不是映射"):
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)

    after = (list(store.documents), store.embeddings.shape, store.faiss_index.ntotal)
    assert before == after
    assert store.faiss_index.ntotal == len(store.documents)


# ── 回归：embeddings 的形状要在**赋值之前**判 ──
#
# 写入侧 ``upsert_embeddings`` 早已断言 ``ndim == 2``，两个载入入口都没有：
# ``load`` 把 ``data.get("embeddings")`` 直接赋给 ``self``，``load_snapshot``
# 直接 ``np.load`` 之后同样赋值，随后 ``_rebuild_indices`` 才去读
# ``len(self.embeddings)`` 与 ``self.embeddings.shape[1]``。实测：
#   0 维（``np.float32(1.0)``）-> ``TypeError: len() of unsized object``
#   1 维（``np.zeros(4)``）    -> ``IndexError: tuple index out of range``
#   ``list``（legacy pkl）     -> ``AttributeError: 'list' object has no attribute 'shape'``
# 三者都不是 ``ValueError``，穿透调用方全部 ``except ValueError`` 分类。
#
# 更严重的是它**同时破坏了两个载入入口承诺的原子性**：形状检查发生在
# ``self.documents``／``self.embeddings`` 已经写进去之后，于是失败留下的正是
# 那组测试要防的半更新状态。实测一次被拒的 ``load`` 之后 docs=2 而 faiss 索引
# 仍是旧的 1 条——「失败必须让 store 保持原状」这条契约被绕过了。
@pytest.mark.parametrize(
    ("embeddings", "label"),
    [
        (np.float32(1.0), "0d"),
        (np.zeros(4, dtype=np.float32), "1d"),
        (np.zeros((1, 4, 1), dtype=np.float32), "3d"),
        ([[0.0, 0.0]], "list"),
        (5, "int"),
    ],
    ids=["0d", "1d", "3d", "list", "int"],
)
def test_load_rejects_non_matrix_embeddings(tmp_path, embeddings, label):
    """legacy 载入的形状错误必须是 ``ValueError``，且 store 保持原状。"""
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {}}], np.zeros((1, 4), dtype=np.float32)
    )
    before = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
    )

    path = _legacy_file(
        tmp_path,
        {
            "documents": [{"page_content": "a", "metadata": {}}],
            "embeddings": embeddings,
        },
    )
    with pytest.raises(ValueError, match="embeddings"):
        store.load(str(path), trusted_paths=[tmp_path])

    after = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
    )
    assert before == after, label


@pytest.mark.parametrize(
    ("array", "label"),
    [
        (np.float32(1.0), "0d"),
        (np.zeros(4, dtype=np.float32), "1d"),
        (np.zeros((1, 4, 1), dtype=np.float32), "3d"),
    ],
    ids=["0d", "1d", "3d"],
)
def test_load_snapshot_rejects_non_matrix_embeddings(tmp_path, array, label):
    """快照载入的形状错误同样必须是 ``ValueError``。"""
    import pickle

    snapshot_dir = tmp_path / f"kb-emb-{label}"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "alpha", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", array)
    _activate_snapshot(tmp_path, snapshot_dir.name)

    store = FaissStore(file_path=None)
    with pytest.raises(ValueError, match="embeddings"):
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)


# ── 回归：查询向量的形态校验要排在转换**之前** ──
#
# ``semantic_search`` 的守卫写的是 ``query_vector = np.asarray(...)`` 之后才判
# ``ndim``，而 ``np.asarray(10**4301, dtype=np.float32)`` **自己就抛**
# ``OverflowError: int too large to convert to float``——``ArithmeticError``
# 子类而**不是** ``ValueError``，穿透调用方全部 ``except ValueError`` 分类。
# 本函数已按同一判据修过 ``_require_top_k``（那里拦住的是 FAISS 的裸
# ``TypeError``／``AssertionError``），查询向量这条漏了：同一个「参数形态非法」
# 的处置，两处收场不同。
#
# 同类的还有 ``keyword_search`` 的 ``query_text``：非字符串直接进
# ``jieba.cut``，抛 ``AttributeError: 'int' object has no attribute 'decode'``。
@pytest.mark.parametrize(
    "bad_query",
    [10**4301, -(10**4301), 10**400],
    ids=["4301digits", "negative-4301digits", "400digits"],
)
def test_semantic_search_rejects_unconvertible_query_with_value_error(bad_query):
    """超出 double 范围的查询向量必须是 ``ValueError``，不是 ``OverflowError``。"""
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {}}], np.zeros((1, 4), dtype=np.float32)
    )

    with pytest.raises(ValueError, match="查询向量"):
        store.semantic_search(bad_query, top_k=1)


@pytest.mark.parametrize(
    "bad_query",
    [5, None, ["a"], 10**4301],
    ids=["int", "None", "list", "huge-int"],
)
def test_keyword_search_rejects_non_string_query_with_value_error(bad_query):
    """非字符串查询必须是 ``ValueError``，不是 jieba 的裸 ``AttributeError``。"""
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha beta", "metadata": {}}], np.zeros((1, 4), dtype=np.float32)
    )

    with pytest.raises(ValueError, match="查询"):
        store.keyword_search(bad_query, top_k=1)


# ``documents`` / ``parents`` 的**顶层**形态同样要有守卫。生产路径由
# ``SnapshotRepository.validate_snapshot_dir`` 覆盖，但两个载入方法是
# ``VectorStoreBase`` 的契约方法，直接调它们的嵌入方不经过那道校验。实测原先的
# 收场方式都是与「快照不自洽」无关的裸异常：
#   分块列表 int／None -> TypeError: 'int' object is not iterable
#   分块列表 dict／str -> 逐键/逐字符当成文档，报「第 0 条文档不是映射」（误导）
#   父块索引 int／None／list -> AttributeError: 'int' object has no attribute 'items'
@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (3, "分块列表的顶层不是序列"),
        (None, "分块列表的顶层不是序列"),
        ("ab", "分块列表的顶层不是序列"),  # len("ab") == 2 曾让它逐字符当成文档
        ({"a": 1}, "分块列表的顶层不是序列"),
    ],
    ids=["int", "None", "str", "dict"],
)
def test_load_rejects_malformed_documents_top_level(tmp_path, payload, pattern):
    store = FaissStore(file_path=None)
    path = _legacy_file(tmp_path, {"documents": payload, "parent_documents": {}})

    with pytest.raises(ValueError, match=pattern):
        store.load(str(path), trusted_paths=[tmp_path])


@pytest.mark.parametrize("payload", [3, None, [1, 2]], ids=["int", "None", "list"])
def test_load_rejects_malformed_parent_documents_top_level(tmp_path, payload):
    store = FaissStore(file_path=None)
    path = _legacy_file(
        tmp_path,
        {"documents": [{"page_content": "a", "metadata": {}}], "parent_documents": payload},
    )

    with pytest.raises(ValueError, match="父块索引的顶层不是映射"):
        store.load(str(path), trusted_paths=[tmp_path])


def test_load_rejects_non_mapping_payload(tmp_path):
    """legacy pkl 的顶层不是映射时，原先抛裸 ``AttributeError``。"""
    store = FaissStore(file_path=None)
    path = _legacy_file(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="顶层不是映射"):
        store.load(str(path), trusted_paths=[tmp_path])


def test_load_legacy_does_not_mutate_the_input_payload(tmp_path):
    """归一化必须返回新对象：调用方持有的那份数据不该被 ``pop`` 掉父块内容。"""

    payload = {
        "documents": [
            {
                "page_content": "child",
                "metadata": {"parent_id": "p1", "parent_content": "父块正文"},
            }
        ],
        "parent_documents": {},
    }
    path = _legacy_file(tmp_path, payload)

    store = FaissStore(file_path=None)
    store.load(str(path), trusted_paths=[tmp_path])

    assert store.resolve_parent_content("p1") == "父块正文"
    assert payload["documents"][0]["metadata"]["parent_content"] == "父块正文"


def test_load_snapshot_requires_trust_root(tmp_path):
    """不声明 snapshot_root 时必须拒绝。

    快照内的 chunks.pkl / parents.pkl / lexical.index 都是 pickle，反序列化即
    执行代码。没有信任声明时不能退化成「任意目录都可加载」，否则这个边界等于
    不存在——那正是 AGENTS.md 要求「不导入不可信来源的 .pkl」被架空的形态。
    """
    snapshot_dir = _make_valid_snapshot(tmp_path / "snapshots")

    with pytest.raises(ValueError, match="缺少 snapshot_root"):
        FaissStore(file_path=None).load_snapshot(str(snapshot_dir))


@pytest.mark.parametrize(
    "bad_argument",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(10**4301, id="beyond-int-str-digits"),
    ],
)
@pytest.mark.parametrize("which", ["snapshot_dir", "snapshot_root"])
def test_load_snapshot_rejects_non_path_arguments(tmp_path, which, bad_argument):
    """``load_snapshot`` 的非路径入参必须收敛成 ``ValueError``，不是裸 ``TypeError``。

    ``load``（legacy 入口）早已按同一判据修过，而 ``load_snapshot`` 这一半漏了：
    它在调 ``resolve_within`` **之前**先写 ``Path(snapshot_dir)`` / ``Path(snapshot_root)``，
    非路径取值在那里抛裸 ``TypeError: expected str, bytes or os.PathLike object,
    not ...``；位数超限的整数抛 ``ValueError: Exceeds the limit``——类型对、内容
    却与快照路径无关。两条都不在本方法其余拒绝路径的 ``ValueError`` 契约内，
    调用方 ``VectorStoreFactory._load_existing_state`` 按它分类。
    """
    snapshot_dir = _make_valid_snapshot(tmp_path / "snapshots")
    kwargs = {"snapshot_dir": str(snapshot_dir), "snapshot_root": str(tmp_path)}
    kwargs[which] = bad_argument

    with pytest.raises(ValueError, match="必须是字符串或路径"):
        FaissStore(file_path=None).load_snapshot(
            kwargs["snapshot_dir"], snapshot_root=kwargs["snapshot_root"]
        )


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


# ── 回归：upsert_embeddings 失败时不得留下半更新的 store ──


def test_upsert_embeddings_rejects_dimension_mismatch_without_mutating():
    """维度不一致时必须拒绝，且 store 保持自洽。

    原实现先 ``self.documents.extend(...)`` 再判维度：维度不一致时抛异常，
    但 documents 已经多出一批没有对应向量的条目。实测拒绝之后
    ``documents=2`` 而 ``embeddings=(1,4)``，store 已不自洽，而
    ``save_snapshot`` 会照常把这份不自洽落盘。失败必须不留痕。
    """
    store = FaissStore(file_path=None)
    store.upsert_embeddings([{"page_content": "a", "metadata": {}}], np.zeros((1, 4), np.float32))

    with pytest.raises(ValueError, match="维度与现有索引不一致"):
        store.upsert_embeddings(
            [{"page_content": "b", "metadata": {}}], np.zeros((1, 8), np.float32)
        )

    assert len(store.documents) == 1
    assert store.embeddings.shape == (1, 4)
    assert store.faiss_index.ntotal == len(store.documents)


def test_upsert_embeddings_rejects_count_mismatch_without_mutating():
    """文档数与向量数不一致时同样不得改动 store。"""
    store = FaissStore(file_path=None)

    with pytest.raises(ValueError, match="数量不一致"):
        store.upsert_embeddings(
            [{"page_content": "a", "metadata": {}}, {"page_content": "b", "metadata": {}}],
            np.zeros((1, 4), np.float32),
        )

    assert store.documents == []
    assert store.embeddings is None


# ``upsert_embeddings`` 原先只校验 embeddings，文档形态放任 ``extend`` 之后由
# ``_rebuild_indices`` 在派生索引时撞出来。异常看似「拒绝写入」，实际上
# ``self.documents`` 已经多出这条孤儿——实测拒绝之后 documents=1 而
# ``_build_index_text`` 在下次重建索引时对同一条再次抛出，于是 store **永久
# 卡死**：连续 3 次喂完全合法的文档全部报同一个 ``AttributeError``，
# documents 停在 5 条不再增长，向量库事实上报废。且 ``save_snapshot`` 会把这份
# 不自洽落盘（实测盘上 chunks.pkl 有 2 条、embeddings 只有 1 条）。
#
# 判据与载入边界 ``_validate_document_shape`` 共用同一份实现，因此这里按同一组
# 形态参数化，确认写入侧也逐条拒绝、且**不留痕**。
@pytest.mark.parametrize(
    ("document", "label"),
    [
        ("just a string", "条目不是映射(str)"),
        (None, "条目不是映射(None)"),
        (["page_content"], "条目不是映射(list)"),
        ({"page_content": None, "metadata": {}}, "page_content=None"),
        ({"page_content": ["a"], "metadata": {}}, "page_content=list"),
        ({"page_content": 3, "metadata": {}}, "page_content=int"),
        ({"page_content": "ok", "metadata": [1, 2]}, "metadata=list"),
        ({"page_content": "ok", "metadata": "nope"}, "metadata=str"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_upsert_embeddings_rejects_malformed_documents_without_mutating(document, label):
    """形态非法的条目必须在写入前拒绝，且 store 保持自洽。"""
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "good", "metadata": {"source": "a"}}], np.zeros((1, 4), np.float32)
    )

    with pytest.raises(ValueError, match="Document 契约"):
        store.upsert_embeddings([document], np.zeros((1, 4), np.float32))

    assert len(store.documents) == 1, f"{label} 被拒后仍留下了孤儿条目"
    assert store.embeddings.shape == (1, 4)
    assert store.faiss_index.ntotal == 1
    assert len(store._tokenized_docs_cache) == len(store.documents)


def test_upsert_embeddings_rejection_does_not_wedge_the_store():
    """被拒一次之后，store 必须仍能接受合法写入。

    这是上一条的**后果**断言，单独成例是因为它才是真正致命的那个：孤儿条目留在
    ``self.documents`` 里时，``_rebuild_indices`` 每次都会对同一条重新抛异常，
    于是任何后续写入——包括完全合法的——都失败，store 再也无法增长。
    """
    store = FaissStore(file_path=None)
    store.upsert_embeddings([{"page_content": "a", "metadata": {}}], np.zeros((1, 4), np.float32))

    with pytest.raises(ValueError, match="Document 契约"):
        store.upsert_embeddings(["orphan"], np.zeros((1, 4), np.float32))

    for index in range(3):
        store.upsert_embeddings(
            [{"page_content": f"ok{index}", "metadata": {}}], np.zeros((1, 4), np.float32)
        )

    assert len(store.documents) == 4
    assert store.embeddings.shape == (4, 4)
    assert store.faiss_index.ntotal == 4
    assert store.bm25_index is not None
    assert store.bm25_index.corpus_size == 4


def test_upsert_embeddings_rejection_message_points_at_the_writer():
    """写入侧的消息必须给出「修正来源后重试」，而不是「重建快照」。

    两个调用点共用同一份形态校验，但处置指引不同：载入侧遇到的是盘上已有的坏
    数据（重建快照），写入侧遇到的是正在写入的坏数据（修正来源）。写死「请重建
    快照」会让写入侧的用户去重建一个还没写坏的旧快照。
    """
    store = FaissStore(file_path=None)

    with pytest.raises(ValueError) as excinfo:
        store.upsert_embeddings(["orphan"], np.zeros((1, 4), np.float32))

    message = str(excinfo.value)
    assert "已拒绝写入" in message
    assert "重建快照" not in message


def test_rebuild_indices_tolerates_orphan_documents():
    """索引重建是派生缓存，不得因 ``documents`` 里的历史脏数据而抛异常。

    ``documents`` 可能带着写入侧校验**之前**留下的形态（``load_snapshot`` 从
    pickle 读回的任意内容）。派生期一抛，整个 store 就卡死；这里要求它降级：
    脏条目缺席检索，其余条目照常可用。
    """
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {"source": "a"}}], np.zeros((1, 4), np.float32)
    )
    # 绕过写入侧校验，模拟历史遗留形态。
    store.documents.append("legacy orphan")
    store.documents.append({"page_content": ["not", "text"], "metadata": "nope"})

    store._rebuild_indices()

    assert len(store._tokenized_docs_cache) == len(store.documents)
    assert store._tokenized_docs_cache[1] == []
    assert store._tokenized_docs_cache[2] == []
    # 合法条目照常可检索。
    hits = store.keyword_search("alpha", top_k=5)
    assert [hit["page_content"] for hit in hits] == ["alpha"]


def test_upsert_embeddings_appends_consistently():
    """收紧不能误伤：维度一致的追加必须照常工作，且索引与文档保持同步。"""
    store = FaissStore(file_path=None)

    store.upsert_embeddings([{"page_content": "a", "metadata": {}}], np.zeros((1, 4), np.float32))
    store.upsert_embeddings(
        [{"page_content": "b", "metadata": {}}, {"page_content": "c", "metadata": {}}],
        np.zeros((2, 4), np.float32),
    )

    assert len(store.documents) == 3
    assert store.embeddings.shape == (3, 4)
    assert store.faiss_index.ntotal == 3
    assert store.bm25_index is not None
    assert store.bm25_index.corpus_size == 3


# ── 回归：载入的文档必须符合 Document 契约 ──
#
# ``FaissStore`` 载入时会把快照／legacy pkl 里的条目直接当成 ``Document`` 用，
# 而 ``Document`` 的契约是 ``content: str`` + ``metadata: dict``
# （见 ``src/models/document.py``）。原实现只做 ``metadata or {}`` 的兜底，其余
# 形态一律放行：实测 ``page_content=None`` 的快照能通过 ``validate_snapshot_dir``、
# 加载成功，直到 ``display_results`` 里 ``.replace()`` 才抛
# ``AttributeError: 'NoneType' object has no attribute 'replace'``——
# 报错现场离真正的原因（快照里这一条不是字符串）很远。``metadata`` 非 dict 与
# 条目本身不是 dict 同理。
#
# 在载入边界拒绝，理由与 ``chunks.pkl`` 的解析检查一致：这是快照内容不自洽，
# 不是显示层的问题。


def _write_legacy(tmp_path, documents):
    """写一个最小可用的 legacy pkl 并返回路径。"""
    import pickle

    import numpy as np

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump(
            {
                "documents": documents,
                "embeddings": np.zeros((len(documents), 4), dtype=np.float32),
                "parent_documents": {},
            },
            file,
        )
    return legacy


@pytest.mark.parametrize(
    "document,label",
    [
        ({"page_content": None, "metadata": {"source": "a.md"}}, "page_content 为 None"),
        ({"page_content": 42, "metadata": {"source": "a.md"}}, "page_content 为整数"),
        ({"page_content": ["x"], "metadata": {"source": "a.md"}}, "page_content 为列表"),
        ({"page_content": "x", "metadata": ["a"]}, "metadata 为列表"),
        ({"page_content": "x", "metadata": "not-a-dict"}, "metadata 为字符串"),
        ({"page_content": "x", "metadata": 42}, "metadata 为整数"),
    ],
)
def test_load_rejects_documents_violating_the_document_contract(tmp_path, document, label):
    """字段形态不符合 ``Document`` 契约时必须在载入期拒绝。"""
    legacy = _write_legacy(tmp_path, [document])
    store = FaissStore()

    with pytest.raises(ValueError, match="文档"):
        store.load(str(legacy), trusted_paths=[tmp_path])


def test_load_rejects_non_mapping_document_entries(tmp_path):
    """条目本身不是映射时同样要拒绝，而不是抛 ``'str' object has no attribute 'get'``。"""
    legacy = _write_legacy(tmp_path, ["这不是一个文档"])
    store = FaissStore()

    with pytest.raises(ValueError, match="文档"):
        store.load(str(legacy), trusted_paths=[tmp_path])


def test_load_accepts_well_formed_documents(tmp_path):
    """收紧不能误伤：合规文档照常载入，``metadata`` 缺省仍补成空 dict。"""
    legacy = _write_legacy(
        tmp_path,
        [
            {"page_content": "正文", "metadata": {"source": "a.md", "chunk_id": "c1"}},
            {"page_content": "只有正文"},
        ],
    )
    store = FaissStore()

    store.load(str(legacy), trusted_paths=[tmp_path])

    assert store.documents[0]["page_content"] == "正文"
    assert store.documents[0]["metadata"]["chunk_id"] == "c1"
    assert store.documents[1]["metadata"] == {}


# ── 回归：``metadata`` 的 None 值不得让 save_snapshot 崩掉 ──
#
# ``doc.get("metadata", {})`` 的默认值只在**键缺失**时生效；键存在而值为
# ``None`` 时它返回 ``None``，随后的 ``.get`` 抛
# ``AttributeError: 'NoneType' object has no attribute 'get'``。同文件另外三处
# （``_build_index_text``、``_normalize_loaded_documents``、
# ``_normalize_parent_document``）用的都是 ``or {}``，只有 stats 计算这一处用了
# ``.get(k, {})``。实测 ``upsert_embeddings`` 接受 ``metadata=None`` 的条目并
# 正常建好索引（``_rebuild_indices`` 走的是 ``or {}``），到 ``save_snapshot``
# 才崩——同一份数据两条路径的判据不一致，而崩溃发生在知识库构建的最后一步。


def test_save_snapshot_survives_documents_with_none_metadata(tmp_path):
    """``metadata=None`` 的条目必须能存快照，与索引构建路径的判据一致。"""
    import json

    import numpy as np

    from src.retrieval.vdb.faiss_store import FaissStore

    store = FaissStore()
    store.upsert_embeddings(
        [
            {"page_content": "a", "metadata": {"source": "x.md"}},
            {"page_content": "b", "metadata": None},
        ],
        np.zeros((2, 4), dtype=np.float32),
    )

    snapshot_dir = tmp_path / "snap"
    store.save_snapshot(str(snapshot_dir))

    stats = json.loads((snapshot_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["chunk_count"] == 2
    # 断言与「metadata 缺键」这条既有路径**同值**，而不是断言某个具体数字：
    # ``document_count`` 的口径是「distinct source」，没有 source 的条目会以
    # ``None`` 占一个桶。要紧的是 ``metadata=None`` 与 ``metadata={}`` 得到同一
    # 结果——此前前者崩、后者正常。
    missing_key_store = FaissStore()
    missing_key_store.upsert_embeddings(
        [
            {"page_content": "a", "metadata": {"source": "x.md"}},
            {"page_content": "b", "metadata": {}},
        ],
        np.zeros((2, 4), dtype=np.float32),
    )
    other_dir = tmp_path / "snap-missing-key"
    missing_key_store.save_snapshot(str(other_dir))
    other_stats = json.loads((other_dir / "stats.json").read_text(encoding="utf-8"))

    assert stats["document_count"] == other_stats["document_count"]


def test_save_snapshot_still_counts_documents_normally(tmp_path):
    """收紧不能误伤：正常 metadata 的来源计数不得改变。"""
    import json

    import numpy as np

    from src.retrieval.vdb.faiss_store import FaissStore

    store = FaissStore()
    store.upsert_embeddings(
        [
            {"page_content": "a", "metadata": {"source": "x.md"}},
            {"page_content": "b", "metadata": {"source": "x.md"}},
            {"page_content": "c", "metadata": {"source": "y.md"}},
        ],
        np.zeros((3, 4), dtype=np.float32),
    )

    snapshot_dir = tmp_path / "snap"
    store.save_snapshot(str(snapshot_dir))

    stats = json.loads((snapshot_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["document_count"] == 2
    assert stats["chunk_count"] == 3


# ── 回归：BM25 对「所有 token 都为空」不得除零 ──
#
# ``rank_bm25`` 的 ``_calc_idf`` 计算 ``idf_sum / len(self.idf)``，**所有 token
# 都为空**时该分母为 0。``jieba.cut("")`` 返回 ``[]``，因此一条
# ``page_content=""`` 的文档就足以构造出 ``[[]]`` 这个形态——而空
# ``page_content`` 在快照里是合法的（``_validate_document_shape`` 只要求它是
# ``str``）。原守卫只判外层 ``not tokenized_docs``，挡不住它。


@pytest.mark.parametrize(
    "tokenized",
    [
        pytest.param([[]], id="single-empty"),
        pytest.param([[], []], id="all-empty"),
    ],
)
def test_bm25_index_handles_all_empty_tokens(tokenized):
    assert FaissStore._build_bm25_index(tokenized) is None


def test_bm25_index_still_builds_with_usable_corpus():
    assert FaissStore._build_bm25_index([["词"]]) is not None


def test_bm25_index_handles_mixed_empty_and_nonempty():
    """只有部分条目为空时仍应建出索引。"""
    assert FaissStore._build_bm25_index([["词"], []]) is not None


# ── 回归：``parent_id`` 是位数超限的整数时不得崩 ──
#
# ``parent_id`` 来自知识快照的元数据，而本项目加载 pickle 格式的快照——pickle
# 是二进制格式、不受 CPython 十进制位数限制。四处都用裸 ``str()``，位数超限时
# 抛 ``ValueError``：在 ``load_snapshot`` 里它让**加载宣布成功之后**的每一次
# 检索都崩，在 ``resolve_parent_content`` 里它让父块召回整体不可用。


def test_parent_id_survives_huge_integer_metadata(tmp_path):
    """超长 parent_id 必须能正常往返，而不是让加载与检索崩掉。"""
    huge_parent_id = 10**4301
    snapshot_dir = tmp_path / "snapshot"
    store = FaissStore(file_path=None)
    store.documents = [
        {
            "page_content": "child",
            "metadata": {"source": "kb.md", "chunk_id": "c1", "parent_id": huge_parent_id},
        }
    ]
    store.embeddings = np.array([[0.0, 0.0]], dtype=np.float32)
    store.register_parent_documents(
        {huge_parent_id: {"content": "父块正文", "metadata": {"source": "kb.md"}}}
    )
    store._rebuild_indices()
    store.save_snapshot(str(snapshot_dir))
    _activate_snapshot(tmp_path, "snapshot")

    reloaded = FaissStore(file_path=None)
    reloaded.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)

    assert reloaded.resolve_parent_content(huge_parent_id) == "父块正文"


# ── 回归：载入侧必须与写入侧一样拒绝非有限向量 ──
#
# 写入侧 ``EmbeddingService._as_float32_matrix`` 明确拒绝 NaN／inf，理由写在
# 那里：FAISS 把 NaN 向量照单收下（``ntotal`` 正常增加），既不报错也不跳过，
# 直到查询时那个下标干脆不出现在结果里——用户看到的是「搜到了」而不是「这份
# 快照坏了」。而两个载入入口的 ``_validate_embeddings_shape`` 此前只判
# ``ndim``，于是同一份内容写入侧拒绝、载入侧放行。
#
# 判据必须落在 ``astype(np.float32)`` **之后**：``1e300`` 在 float64 里是有限
# 的，cast 成 float32 才溢出成 ``inf``。只对原数组调 ``isfinite`` 会在这里再次
# 与写入侧分叉，因此四个方向都要有用例。


@pytest.mark.parametrize(
    ("embeddings", "label"),
    [
        (np.full((2, 4), np.nan), "NaN"),
        (np.full((2, 4), np.inf), "正无穷"),
        (np.full((2, 4), -np.inf), "负无穷"),
        (np.array([[1e300, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]), "float64 有限但 cast 后溢出"),
    ],
    ids=["nan", "inf", "neg-inf", "post-cast-overflow"],
)
def test_load_rejects_non_finite_embeddings(tmp_path, embeddings, label):
    """legacy 载入的非有限向量必须是 ``ValueError``，且 store 保持原状。"""
    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": "alpha", "metadata": {}}], np.zeros((1, 4), dtype=np.float32)
    )
    before = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
    )

    path = _legacy_file(
        tmp_path,
        {
            "documents": [{"page_content": "a", "metadata": {}}],
            "embeddings": embeddings,
        },
    )
    with pytest.raises(ValueError, match="非有限值"):
        store.load(str(path), trusted_paths=[tmp_path])

    after = (
        [doc["page_content"] for doc in store.documents],
        store.embeddings.shape,
        store.faiss_index.ntotal,
    )
    assert before == after, label


@pytest.mark.parametrize(
    ("array", "label"),
    [
        (np.full((1, 4), np.nan), "nan"),
        (np.full((1, 4), np.inf), "inf"),
        (np.array([[1e300, 0.0, 0.0, 0.0]]), "post-cast-overflow"),
    ],
    ids=["nan", "inf", "post-cast-overflow"],
)
def test_load_snapshot_rejects_non_finite_embeddings(tmp_path, array, label):
    """快照载入的非有限向量同样必须是 ``ValueError``。"""
    import pickle

    snapshot_dir = tmp_path / f"kb-finite-{label}"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "alpha", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", array)
    _activate_snapshot(tmp_path, snapshot_dir.name)

    store = FaissStore(file_path=None)
    with pytest.raises(ValueError, match="非有限值"):
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)


def test_load_accepts_finite_embeddings_without_leaking_a_warning(tmp_path):
    """收紧不能误伤：正常向量照常载入，且不泄漏无谓的 RuntimeWarning。

    cast 那份副本对正常取值不会溢出，因此这里不该有任何 ``RuntimeWarning``——
    守卫用 ``catch_warnings`` 包住 cast 是为了压掉**预期内**的溢出警告，不是
    把整类警告静默掉。
    """
    import pickle
    import warnings

    snapshot_dir = tmp_path / "kb-finite-ok"
    snapshot_dir.mkdir()
    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "alpha", "metadata": {}}], file)
    with (snapshot_dir / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float64))
    _activate_snapshot(tmp_path, snapshot_dir.name)

    store = FaissStore(file_path=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.load_snapshot(str(snapshot_dir), snapshot_root=tmp_path)

    assert store.embeddings.shape == (1, 4)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# ── 回归：父块索引的顶层形态两个入口收场一致 ──
#
# 载入侧 ``_normalize_documents`` 早就拒绝非映射的父块索引，给的是带指引的
# ``ValueError``；而写入侧 ``register_parent_documents`` 直接在 ``.items()`` 上
# 抛裸 ``AttributeError: 'int' object has no attribute 'items'``。两者都是
# ``VectorStoreBase`` 的公开契约方法，`_normalize_documents` 的注释也已写明
# 「直接调它们的嵌入方不经过快照校验」——同一取舍本应两侧都执行。
#
# 本仓自带流程到不了写入侧这条分支（``knowledge_build_service`` 传的是 dict，
# 且前面已用 isinstance 过滤），因此这是一致性修复而非崩溃修复。


@pytest.mark.parametrize(
    "bad_index", [5, None, [1, 2], "x", 10**4301], ids=["int", "None", "list", "str", "huge-int"]
)
def test_register_parent_documents_rejects_non_mapping(bad_index):
    """写入侧非映射必须是 ``ValueError``，不是裸 ``AttributeError``。"""
    store = FaissStore(file_path=None)

    with pytest.raises(ValueError, match="父块索引"):
        store.register_parent_documents(bad_index)

    assert store.parent_documents == {}


def test_register_and_load_agree_on_non_mapping_parent_index():
    """同一形态在写入侧与载入侧必须同为 ``ValueError``（收场一致）。"""
    store = FaissStore(file_path=None)
    with pytest.raises(ValueError) as write_error:
        store.register_parent_documents(5)

    with pytest.raises(ValueError) as load_error:
        FaissStore._normalize_documents([], 5)

    assert "父块索引的顶层不是映射" in str(write_error.value)
    assert "父块索引的顶层不是映射" in str(load_error.value)


def test_register_parent_documents_still_accepts_a_mapping():
    """收紧不能误伤：正常映射照常注册，空 content 仍按原语义丢弃。"""
    store = FaissStore(file_path=None)

    store.register_parent_documents(
        {
            "p-ok": {"content": "父块正文", "metadata": {"k": 1}},
            "p-empty": {"content": "", "metadata": {"k": 2}},
        }
    )

    assert set(store.parent_documents) == {"p-ok"}
    assert store.resolve_parent_content("p-ok") == "父块正文"


# ── 回归：trusted_paths 传成字符串会被逐字符拆开，边界静默失效 ──


def test_load_rejects_string_trusted_paths_instead_of_splitting_it(tmp_path):
    """``trusted_paths`` 是**可迭代**，传裸字符串会被逐字符拆成信任根。

    ``ensure_trusted_source`` 对 ``trusted_roots`` 直接 ``for root in ...``，
    而 ``str`` 是合法的可迭代——实测 ``trusted_paths="/tmp/trusted"`` 会拆成
    ``"/"``、``"t"``、``"m"``… 一堆单字符根，其中 ``"/"`` 解析为文件系统根，
    于是**任意绝对路径都被接受**：``/anywhere/evil.pkl`` 在 ``trusted_paths=
    "/etc"`` 下也返回成功。这正是「不导入不可信来源的 .pkl」那条边界要挡的形态，
    却因为参数写错而静默失效。同一文件对 ``Sequence`` 参数的判据早已显式排除
    ``str``／``bytes``／``bytearray``（见 ``_require_documents``／
    ``normalize_messages``），这里必须对齐。
    """
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    with pytest.raises(ValueError, match="受信"):
        FaissStore(file_path=None).load(str(legacy), trusted_paths=str(tmp_path))


# ── 回归：``ensure_trusted_source`` 的 ``trusted_roots`` 非可迭代时要给可读异常 ──
#
# 该函数已显式排除 ``str``／``bytes``／``bytearray``（它们是可迭代但会被逐字符
# 拆开），却漏了**其它不可迭代**的形态：``for root in trusted_roots`` 对
# ``5``／``True``／``object()`` 抛裸 ``TypeError: 'int' object is not iterable``
# ——不是本函数承诺的带 label 的 ``ValueError``，而调用方 ``FaissStore.load``
# 按 ``ValueError`` 分类。可达路径：``load`` 是 ``VectorStoreBase`` 的契约方法，
# 嵌入方直接调用时把 ``trusted_paths`` 写错形态即命中。


@pytest.mark.parametrize(
    "trusted_paths",
    [
        pytest.param(5, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(object(), id="object"),
        pytest.param(10**4301, id="beyond-int-str-digits"),
    ],
)
def test_load_rejects_non_iterable_trusted_paths_readably(tmp_path, trusted_paths):
    """``trusted_paths`` 不可迭代时必须给带指引的 ``ValueError``，不得裸抛。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    with pytest.raises(ValueError, match="受信"):
        FaissStore(file_path=None).load(str(legacy), trusted_paths=trusted_paths)


def test_load_still_accepts_a_path_list_as_trusted_paths(tmp_path):
    """收紧不能误伤：合法的路径列表照常加载。"""
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    store = FaissStore(file_path=None)
    store.load(str(legacy), trusted_paths=[tmp_path])

    assert store.documents == []


def test_load_rejects_non_path_trusted_roots_readably(tmp_path):
    """信任根不是路径时给可读的 ``ValueError``，而不是裸 ``TypeError``。

    ``Path(root)`` 对 ``None``／``int``／``list`` 抛的是裸
    ``TypeError: expected str, bytes or os.PathLike object, not ...``——它不是
    ``ValueError``，穿透 ``load`` 调用方按 ``ValueError`` 建立的错误分类，且
    消息里没有任何字段名。同一模块的 ``resolve_within`` 对**候选路径**的同类
    形态给出的是带 label 的 ``ValueError``，两条判据必须一致。
    """
    import pickle

    legacy = tmp_path / "legacy.pkl"
    with legacy.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    with pytest.raises(ValueError, match="受信"):
        FaissStore(file_path=None).load(str(legacy), trusted_paths=[None])
    with pytest.raises(ValueError, match="受信"):
        FaissStore(file_path=None).load(str(legacy), trusted_paths=[5])


def test_load_rejects_non_path_candidate_readably(tmp_path):
    """候选路径不是路径时给可读的 ``ValueError``，而不是裸 ``TypeError``。

    ``FaissStore.load`` 把路径交给 ``ensure_trusted_source``。候选路径形态非法
    时，``resolve_within`` 的守卫抛 ``ValueError``，但 ``ensure_trusted_source``
    的逐根试探会把它吞掉（``except ValueError: continue``），最后落到兜底消息里
    的 ``Path(candidate)``——那里对 ``None``／``int``／``list`` 抛裸
    ``TypeError``。本函数的契约是带 label 的 ``ValueError``。
    """
    with pytest.raises(ValueError, match="路径必须是字符串或路径"):
        FaissStore(file_path=None).load(None, trusted_paths=[tmp_path])
    with pytest.raises(ValueError, match="路径必须是字符串或路径"):
        FaissStore(file_path=None).load(5, trusted_paths=[tmp_path])


def test_resolve_within_rejects_non_path_arguments_readably(tmp_path):
    """``resolve_within`` 对非路径的候选／根给带 label 的 ``ValueError``。

    同模块的 ``(OSError, RuntimeError)`` 收敛判据早已认领「pathlib 的内部错误
    不该穿透 ``ValueError`` 契约」，``Path(...)`` 对非路径取值抛的 ``TypeError``
    是同一类：不是 ``ValueError``、消息里没有字段名，而两个调用方都按
    ``ValueError`` 分类。
    """
    from src.utils.security import resolve_within

    with pytest.raises(ValueError, match="路径必须是字符串或路径"):
        resolve_within(None, tmp_path, label="知识快照目录")
    with pytest.raises(ValueError, match="受信根必须是字符串或路径"):
        resolve_within(tmp_path, 5, label="知识快照目录")


# ── 回归：``FaissStore.__init__`` 的 ``file_path`` 也要判形态 ──
#
# 构造器第一件事就是 ``os.path.exists(file_path)``，而它早于 ``load`` 里那道
# 形态守卫——本类的 ``save``／``load``／``save_snapshot``／``load_snapshot``
# 都已按同一判据收紧，构造器这条公开入口漏了。实测非路径取值的收场是裸异常，
# 都不在本类的 ``ValueError`` 契约内：
#   * ``["a"]``／``{"a": 1}``／``object()`` -> 裸
#     ``TypeError: stat: path should be string, bytes, os.PathLike or integer,
#     not ...``；
#   * ``10**4301``（位数超限的整数）-> 裸 ``OverflowError: fd is greater than
#     maximum``——它是 ``ArithmeticError`` 子类而**不是** ``ValueError``，穿透
#     调用方全部 ``except ValueError`` 分类；
#   * ``5`` 更隐蔽：``os.path.exists(5)`` 把整数当成**文件描述符**探测（实测
#     返回 ``False`` 而**不报错**），于是构造静默成功、``file_path`` 留着这个
#     整数——与 ``save(5)`` 把字节写进 fd 5 是同一类「整数被当成 fd」的问题。
# 可达路径：``FaissStore`` 是公开类（测试与库调用方直接构造），且
# ``VectorStoreFactory.get_vector_store(store_type, file_path)`` 把 ``file_path``
# 原样透传到这里。


@pytest.mark.parametrize(
    "bad_path",
    [
        pytest.param(["a"], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(object(), id="object"),
        pytest.param(10**4301, id="beyond-double"),
        pytest.param(5, id="int-as-fd"),
        pytest.param(b"x", id="bytes"),
    ],
)
def test_init_rejects_non_path_file_path(bad_path):
    """``file_path`` 非路径时必须给带指引的 ``ValueError``，不得裸抛或静默接受。"""
    with pytest.raises(ValueError, match="file_path|路径"):
        FaissStore(file_path=bad_path, trusted_paths=[Path("/tmp")])


def test_init_still_accepts_a_missing_path(tmp_path):
    """收紧不能误伤：指向不存在文件的合法路径照常构造出空 store。"""
    store = FaissStore(file_path=str(tmp_path / "missing.pkl"), trusted_paths=[tmp_path])

    assert store.documents == []
    assert store.embeddings is None


# ── 回归：upsert_embeddings 的 documents 顶层容器缺形态判据 ──
#
# 上面那一组参数化覆盖的是**条目**形态（``_validate_document_shape`` 逐条判），
# 顶层容器这一半漏了：``upsert_embeddings`` 的第一行就是 ``if not documents:
# return``，随后 ``len(documents)``。实测：
#   * ``documents=5``  -> 裸 ``TypeError: object of type 'int' has no len()``，
#     不是本方法其余拒绝路径承诺的 ``ValueError``（载入侧
#     ``_normalize_documents`` 对同一形态给的是带指引的 ``ValueError``）；
#   * ``documents=None`` -> **静默返回**，调用方以为写入成功，而 store 一个字
#     都没变——写入路径上的静默无操作比报错更难查。
# 判据与载入侧 ``_normalize_documents`` 的同名守卫一致（拒绝 str／bytes／映射，
# 要求有 ``__len__``）。


@pytest.mark.parametrize(
    "documents",
    [
        pytest.param(5, id="int"),
        pytest.param(None, id="none"),
        pytest.param("abc", id="str"),
        pytest.param({"a": 1}, id="dict"),
    ],
)
def test_upsert_embeddings_rejects_non_sequence_documents(documents):
    """非序列的 ``documents`` 必须给出带指引的 ``ValueError``，不得静默返回。"""
    store = FaissStore(file_path=None)

    with pytest.raises(ValueError, match="文档列表"):
        store.upsert_embeddings(documents, np.zeros((1, 4), np.float32))

    assert store.documents == []


def test_upsert_embeddings_still_accepts_document_sequences():
    """收紧不能误伤：正常文档列表照常写入。"""
    store = FaissStore(file_path=None)

    store.upsert_embeddings(
        [{"page_content": "good", "metadata": {"source": "a"}}], np.zeros((1, 4), np.float32)
    )

    assert len(store.documents) == 1
    assert store.faiss_index.ntotal == 1


# ── 回归：upsert_embeddings 的 embeddings 入参缺形态判据 ──
#
# 上面那组覆盖的是 ``documents``，而同一方法的**兄弟入参** ``embeddings``
# 此前完全没有守卫：``np.array(..., dtype=np.float32)`` 的三类失败全部裸抛，
# 而 ``documents`` 那一半已经给出带指引的 ``ValueError``——同一方法的两个入参，
# 收场一个可读一个裸抛。实测：
#   * ``10**4301`` -> ``OverflowError: int too large to convert to float``
#     （``ArithmeticError`` 子类，**不是** ``ValueError``，穿透调用方全部
#     ``except ValueError`` 分类）；
#   * ``{"a": 1}`` -> ``TypeError: float() argument must be a string or a real
#     number, not 'dict'``；
#   * ``"abc"`` -> numpy 的 ``ValueError: could not convert string to float``。
# 判据与同文件 ``semantic_search`` 的查询向量转换逐字相同（那里早已按同一分类
# 收敛），两者必须一致。


@pytest.mark.parametrize(
    "embeddings",
    [
        pytest.param(10**4301, id="int-beyond-double"),
        pytest.param("abc", id="str"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(object(), id="object"),
    ],
)
def test_upsert_embeddings_rejects_unconvertible_embeddings(embeddings):
    """形态无法转成 float32 矩阵时，收场必须是带指引的 ``ValueError``。"""
    store = FaissStore(file_path=None)

    with pytest.raises(ValueError, match="embeddings"):
        store.upsert_embeddings([{"page_content": "a", "metadata": {}}], embeddings)

    assert store.documents == []
    assert store.embeddings is None
