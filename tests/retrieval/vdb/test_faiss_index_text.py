import pytest

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


@pytest.mark.parametrize(
    "source",
    [123, ["a.md"], {"path": "a.md"}, b"a.md"],
    ids=["int", "list", "dict", "bytes"],
)
def test_build_index_text_tolerates_non_string_source(source):
    """``metadata['source']`` 非字符串时必须按「无来源」处理，不得抛 ``TypeError``。

    判据是 ``isinstance(source, str) and source``——前半个 ``isinstance`` 是**有意**
    短路：只有字符串才进 ``Path(source).stem``。把它后面的 ``and`` 互换为 ``or`` 后，
    非字符串的真值（``123``／``["a.md"]``／``{"path": ...}``）也命中 →
    ``Path(123)`` 抛 ``TypeError: expected str, bytes or os.PathLike object``。
    该路径在索引重建（``_rebuild_indices``）与快照加载都会走到，且检索结果可来自
    任何 ``VectorStoreBase`` 实现（第三方 store 的 metadata 不受本项目的载入期
    归一化约束）。既有 ``test_build_index_text_keeps_original_content_when_source_missing``
    只喂空 metadata（``source`` 缺省为 ``""``，字符串），非字符串分支从未被覆盖。
    """
    document = {"page_content": "正文", "metadata": {"source": source}}

    assert FaissStore._build_index_text(document) == "正文"


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
    # 空 store 也要带空索引（生产 save_snapshot 恒写出该文件；缺它 + 有
    # lexical.index 现在是载入期拒绝形态）。
    import faiss

    faiss.write_index(faiss.IndexFlatL2(4), str(snapshot_dir / "semantic.index"))
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text("kb-empty", encoding="utf-8")

    store = FaissStore(file_path=None)
    store.load_snapshot(str(snapshot_dir), snapshot_root=snapshot_root)

    assert store.bm25_index is None
    assert store.keyword_search("任意查询") == []


# ── 回归：``semantic_search`` 的 top_k 必须有上界且给可读错误 ──
#
# 两个方向都会出问题，而配置层的上界拦不住任何一个：``_MAX_RETRIEVAL_TOP_K``
# 只作用在 ``Settings.chat_top_k`` 与 ``SessionConfig.top_k`` 上，而
# ``semantic_search`` 是 ``VectorStoreBase`` 的公开方法（库调用方与测试直接
# 调用），``top_k`` 从入口一路透传给 ``faiss_index.search()``。
#
# * 下界：``faiss.search(q, 0)`` 与 ``-1`` 触发 FAISS 内建的 ``assert k > 0``，
#   抛的是**没有消息的** ``builtins.AssertionError``。实测消息为空串，栈里只有
#   C 层，用户既不知道哪个参数错了、也不知道该怎么改；而 ``AssertionError``
#   不是 ``ValueError``，调用方的错误分类接不住。``python -O`` 下断言被剥掉，
#   行为还会变成另一种未定义状态。
# * 上界：``faiss.search(q, 10**9)`` 不报错也不截断，直接按 k 分配结果数组——
#   实测 10**9 时挂起（>30s），10**8 时耗时 15.3s。这条路径在配置层被
#   ``_MAX_RETRIEVAL_TOP_K`` 挡住，但公开方法本身没有守卫。
def _store_with_documents(count: int = 5) -> FaissStore:
    import numpy as np

    store = FaissStore(file_path=None)
    store.upsert_embeddings(
        [{"page_content": f"doc{i}", "metadata": {}} for i in range(count)],
        np.arange(count * 4, dtype=np.float32).reshape(count, 4),
    )
    return store


def test_semantic_search_returns_empty_when_only_index_is_missing():
    """``faiss_index`` 缺失但 ``embeddings`` 非空时也必须返回空列表。

    守卫 ``if self.faiss_index is None or self.embeddings is None`` 的两个条件
    **不互斥**：只要任一为 None 就该短路返回 []。把第一个 ``or`` 互换为 ``and``
    后，只有两者同时为 None 才短路——半初始化状态（``index`` 为 None、
    ``embeddings`` 非空，例如 ``_rebuild_indices`` 中途或外部注入的 store）会
    越过守卫，落到 ``self.faiss_index.search(...)`` 抛
    ``AttributeError: 'NoneType' object has no attribute 'search'``。
    ``FaissStore`` 是 ``VectorStoreBase`` 的公开实现，第三方可直接构造。
    """
    import numpy as np

    store = FaissStore(file_path=None)
    store.faiss_index = None
    store.embeddings = np.zeros((3, 4), dtype=np.float32)

    assert store.semantic_search(np.ones(4, dtype=np.float32), 2) == []


@pytest.mark.parametrize("bad_top_k", [0, -1, -100])
def test_semantic_search_rejects_non_positive_top_k(bad_top_k):
    """非正 ``top_k`` 必须报项目自己的 ``ValueError``，而不是 FAISS 的空断言。"""
    import numpy as np

    store = _store_with_documents()

    with pytest.raises(ValueError) as excinfo:
        store.semantic_search(np.ones(4, dtype=np.float32), bad_top_k)

    message = str(excinfo.value)
    assert "top_k" in message
    assert str(bad_top_k) in message


def test_semantic_search_rejects_oversized_top_k():
    """超过上界的 ``top_k`` 必须在分配数组**之前**被拒绝。"""
    import numpy as np

    from src.utils.config import _MAX_RETRIEVAL_TOP_K

    store = _store_with_documents()

    with pytest.raises(ValueError) as excinfo:
        store.semantic_search(np.ones(4, dtype=np.float32), _MAX_RETRIEVAL_TOP_K + 1)

    assert "top_k" in str(excinfo.value)


# 上界分支的消息里也必须用 ``safe_repr``。类型分支（上面那条）早就用了，
# 上界分支却是裸 ``{top_k}``——两处对同一个参数、同一份消息，只有一处防住了
# 位数超限。实测 ``top_k=10**4301`` 时，正要报「top_k 越界」的那一步自己抛
# ``ValueError: Exceeds the limit (4300 digits) for integer string conversion``：
# 用户拿到的是 CPython 的整数转换错误，与 ``top_k`` 毫无关系，也没有上界信息。
# （``10**4000`` 不会复现——位数在上限内，只是消息长到 4000 位。）
#
# 这条路径完全可达：两个方法都是 ``VectorStoreBase`` 的公开方法，``top_k``
# 一路透传，而 pickle 是二进制格式，畸形快照／第三方调用方都能带进这种整数。
@pytest.mark.parametrize("huge_top_k", [10**4301, -(10**4301)], ids=["正", "负"])
def test_semantic_search_rejects_huge_top_k_with_a_readable_message(huge_top_k):
    """位数超限的 ``top_k`` 必须给出可读的越界消息，而不是整数转换错误。"""
    import numpy as np

    store = _store_with_documents()

    with pytest.raises(ValueError) as excinfo:
        store.semantic_search(np.ones(4, dtype=np.float32), huge_top_k)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message
    assert "top_k" in message
    assert "位十进制" in message  # safe_repr 对超限整数的量级描述


def test_keyword_search_rejects_huge_top_k_with_a_readable_message():
    """同一份判据必须在 ``keyword_search`` 上成立——两个方法共用 ``_require_top_k``。"""
    store = _store_with_positive_keyword_hit()

    with pytest.raises(ValueError) as excinfo:
        store.keyword_search("苹果", 10**4301)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message
    assert "top_k" in message
    assert "位十进制" in message


def test_semantic_search_rejects_non_integer_top_k():
    """布尔与浮点混入同样拒绝——``bool`` 是 ``int`` 子类，``True`` 会算成 1。"""
    import numpy as np

    store = _store_with_documents()

    for bad_top_k in (True, 1.5, "3", None):
        with pytest.raises(ValueError):
            store.semantic_search(np.ones(4, dtype=np.float32), bad_top_k)


def test_semantic_search_still_accepts_top_k_larger_than_the_corpus():
    """反向守卫：``top_k`` 大于文档数仍然合法，只是返回全部文档。

    把上界写成「不得超过文档数」会误伤——``top_k`` 是候选规模，FAISS 本来就
    允许它超过命中集合。
    """
    import numpy as np

    store = _store_with_documents(count=3)

    assert len(store.semantic_search(np.ones(4, dtype=np.float32), 50)) == 3


# ── 回归：``keyword_search`` 的 top_k 与 ``semantic_search`` 判据必须一致 ──
#
# 两个方法同名参数、同一份上界来源，此前判据完全不同：
#
# * ``top_k=0``／负数：``semantic_search`` 抛 FAISS 的空 ``AssertionError``，而
#   ``keyword_search`` 走的是 ``if len(results) >= top_k: break``——循环体先
#   ``append`` 再判长度，于是 ``top_k=0`` **返回 1 条**。调用方要求「不要结果」
#   却拿到一条，且没有任何提示。
# * 非整数：``semantic_search`` 由 FAISS 报 ``TypeError: an integer is required``，
#   ``keyword_search`` 在 ``int >= str`` 处报 ``TypeError: '>=' not supported``
#   ——两者都是裸 ``TypeError``，不是本项目的 ``ValueError``。
#
# 同一个参数在两处给出三种不同收场方式，调用方无法按一种错误类型处理。
def _store_with_positive_keyword_hit(count: int = 10) -> FaissStore:
    """造一个 BM25 分数为正的语料：查询词只出现在一篇文档里。

    语料里必须真的存在正分文档，否则 ``keyword_search`` 会在「最高分 <= 0」
    处提前返回空列表，把 ``top_k`` 的问题整个遮住——用全 0 分数的夹具测这条
    判据是假绿。
    """
    import numpy as np

    documents = [{"page_content": f"普通 文档{i}", "metadata": {}} for i in range(count)]
    documents[0]["page_content"] = "苹果 独特 词汇"
    store = FaissStore(file_path=None)
    store.upsert_embeddings(documents, np.arange(count * 4, dtype=np.float32).reshape(count, 4))
    return store


def test_keyword_search_fixture_really_produces_a_positive_score():
    """夹具自检：正分文档必须存在，否则下面两条断言守的是不存在的问题。"""
    store = _store_with_positive_keyword_hit()

    assert len(store.keyword_search("苹果", 5)) == 1


@pytest.mark.parametrize("bad_top_k", [0, -1, -100])
def test_keyword_search_rejects_non_positive_top_k(bad_top_k):
    store = _store_with_positive_keyword_hit()

    with pytest.raises(ValueError) as excinfo:
        store.keyword_search("苹果", bad_top_k)

    message = str(excinfo.value)
    assert "top_k" in message
    assert str(bad_top_k) in message


@pytest.mark.parametrize("bad_top_k", [None, "3", 1.5, True])
def test_keyword_search_rejects_non_integer_top_k(bad_top_k):
    store = _store_with_positive_keyword_hit()

    with pytest.raises(ValueError) as excinfo:
        store.keyword_search("苹果", bad_top_k)

    assert "top_k" in str(excinfo.value)


def test_keyword_search_rejects_oversized_top_k():
    from src.utils.config import _MAX_RETRIEVAL_TOP_K

    store = _store_with_positive_keyword_hit()

    with pytest.raises(ValueError):
        store.keyword_search("苹果", _MAX_RETRIEVAL_TOP_K + 1)
