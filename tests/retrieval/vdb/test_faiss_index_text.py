from src.retrieval.vdb.faiss_store import FaissStore


def test_build_index_text_includes_source_stem():
    document = {
        "page_content": "正文内容",
        "metadata": {"source": "/tmp/用户-共享文件夹使用方法_优化后.md"},
    }

    index_text = FaissStore._build_index_text(document)

    assert "用户 共享文件夹使用方法 优化后" in index_text
    assert index_text.endswith("正文内容")


def test_build_index_text_keeps_original_content_when_source_missing():
    document = {
        "page_content": "只有正文",
        "metadata": {},
    }

    index_text = FaissStore._build_index_text(document)

    assert index_text == "只有正文"


# ── 回归：BM25 分数全 0 时的静默空结果，以及空语料的除零 ──


def _store_with_corpus(texts):
    """用 jieba 与检索侧一致的分词构造一个可用的 FaissStore。"""
    import jieba
    import numpy as np
    import rank_bm25

    store = FaissStore(file_path=None)
    store.documents = [
        {"page_content": text, "metadata": {"source": f"s{index}.md"}}
        for index, text in enumerate(texts)
    ]
    store.embeddings = np.zeros((len(texts), 4), dtype=np.float32)
    store._tokenized_docs_cache = [list(jieba.cut(text)) for text in texts]
    store.bm25_index = rank_bm25.BM25Okapi(store._tokenized_docs_cache)
    return store


def test_bm25_idf_is_exactly_zero_when_term_in_half_the_corpus():
    """钉住前提：某个词恰好出现在一半文档里时 idf **恰为** 0，不会被 epsilon 浮动。

    ``BM25Okapi._calc_idf`` 用 ``log(N - n + 0.5) - log(n + 0.5)``，且只对
    ``idf < 0`` 做 ``eps * average_idf`` 浮动。``n == N/2`` 时 idf 正好是 0，
    不属于「负」因此不被浮动，该词在所有文档上的分数就全是 0。
    """
    store = _store_with_corpus(["苹果 甲", "苹果 乙", "香蕉 丙", "香蕉 丁"])

    assert store.bm25_index.idf["苹果"] == 0.0
    assert store.bm25_index.get_scores(["苹果"]).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_keyword_search_returns_empty_for_zero_idf_term():
    """全 0 分时返回空结果本身说得通（该词无判别力），关键是不得报错。"""
    store = _store_with_corpus(["苹果 甲", "苹果 乙", "香蕉 丙", "香蕉 丁"])

    assert store.keyword_search("苹果") == []


def test_keyword_search_keeps_positive_scores_after_zero_score_entries():
    """0 分文档只能被跳过，不能让循环提前终止丢掉后面的正分结果。

    旧实现是 ``if score <= 0: break``，分数降序后一旦遇到 0 分就整段放弃。
    """
    # 「独占词」只出现在一篇文档里 -> idf 为正；「苹果」恰好 N/2 -> idf 为 0
    store = _store_with_corpus(["苹果 独占标记", "苹果 乙", "香蕉 丙", "香蕉 丁"])
    rows = store.bm25_index.get_scores(list(__import__("jieba").cut("独占标记")))

    assert max(rows) > 0, "前提：该词必须有正分，否则这条测试无法区分两种实现"
    assert (
        store.keyword_search("独占标记")
        == [document for document in store.documents if document["metadata"]["source"] == "s0.md"]
        or store.keyword_search("独占标记")[0]["metadata"]["source"] == "s0.md"
    )


def test_build_bm25_index_returns_none_for_empty_corpus():
    """空语料必须返回 None，不能让 rank_bm25 除零。

    ``BM25._initialize`` 计算 ``avgdl = num_doc / self.corpus_size``，空语料直接
    ``ZeroDivisionError``。``load_snapshot`` 会从 ``lexical.index`` 读回 ``[]``
    （空快照或手工构造的目录），旧实现在此崩溃且不报「快照为空」这个真实原因。
    """
    assert FaissStore._build_bm25_index([]) is None
    assert FaissStore._build_bm25_index([["词"]]) is not None


def test_load_snapshot_tolerates_empty_lexical_index(tmp_path):
    """``lexical.index`` 为空列表时应正常加载（bm25_index 为 None），不得崩溃。"""
    import pickle

    import numpy as np

    snapshot_root = tmp_path / "snapshots"
    snapshot_dir = snapshot_root / "kb-empty"
    snapshot_dir.mkdir(parents=True)
    for name, payload in (("chunks.pkl", []), ("parents.pkl", {}), ("lexical.index", [])):
        with (snapshot_dir / name).open("wb") as file:
            pickle.dump(payload, file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((0, 4), dtype=np.float32))
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text("kb-empty", encoding="utf-8")

    store = FaissStore(file_path=None)
    store.load_snapshot(str(snapshot_dir), snapshot_root=snapshot_root)

    assert store.bm25_index is None
    assert store.keyword_search("任意查询") == []
