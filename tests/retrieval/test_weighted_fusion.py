import pytest

from src.retrieval.retriever import HybridReranker, _merge_hybrid_results
from src.services.retrieval_service import (
    _document_key,
    _promote_parent_context,
    _stable_fingerprint,
    _unpack_rerank_result,
)


def test_hybrid_reranker_normalizes_mixed_score_scales():
    reranker = HybridReranker(vector_weight=0.8, keyword_weight=0.2, fusion_strategy="weighted")
    documents = [
        {"page_content": "doc-a", "semantic_score": 0.9, "keyword_score": 10.0},
        {"page_content": "doc-b", "semantic_score": 0.1, "keyword_score": 5.0},
    ]

    ranked = reranker.rerank(documents)

    assert [doc["page_content"] for doc in ranked] == ["doc-a", "doc-b"]
    assert ranked[0]["score"] == 1.0
    assert ranked[1]["score"] == 0.0


def test_hybrid_reranker_handles_uniform_positive_scores():
    reranker = HybridReranker(vector_weight=0.5, keyword_weight=0.5, fusion_strategy="weighted")
    documents = [
        {"page_content": "doc-a", "semantic_score": 3.0, "keyword_score": 4.0},
        {"page_content": "doc-b", "semantic_score": 3.0, "keyword_score": 4.0},
    ]

    ranked = reranker.rerank(documents)

    assert all(doc["score"] == 1.0 for doc in ranked)


def test_hybrid_reranker_handles_all_zero_scores():
    reranker = HybridReranker(vector_weight=0.5, keyword_weight=0.5, fusion_strategy="weighted")
    documents = [
        {"page_content": "doc-a", "semantic_score": 0.0, "keyword_score": 0.0},
        {"page_content": "doc-b", "semantic_score": 0.0, "keyword_score": 0.0},
    ]

    ranked = reranker.rerank(documents)

    assert all(doc["score"] == 0.0 for doc in ranked)


def test_merge_hybrid_results_keeps_distinct_child_chunks_on_same_page():
    semantic_results = [
        {
            "page_content": "child-a",
            "score": 0.9,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-a"},
        },
        {
            "page_content": "child-b",
            "score": 0.8,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-b"},
        },
    ]
    keyword_results = [
        {
            "page_content": "child-a",
            "score": 4.0,
            "metadata": {"source": "kb.md", "page": 1, "chunk_id": "chunk-a"},
        }
    ]

    merged = _merge_hybrid_results(semantic_results, keyword_results)

    assert len(merged) == 2
    assert {doc["metadata"]["chunk_id"] for doc in merged} == {"chunk-a", "chunk-b"}
    merged_a = next(doc for doc in merged if doc["metadata"]["chunk_id"] == "chunk-a")
    assert merged_a["semantic_score"] == 0.9
    assert merged_a["keyword_score"] == 4.0


def test_rrf_reranker_uses_rank_positions_not_score_magnitudes():
    reranker = HybridReranker(vector_weight=0.8, keyword_weight=0.2, fusion_strategy="rrf")
    documents = [
        {"page_content": "doc-a", "semantic_rank": 1, "keyword_rank": 3},
        {"page_content": "doc-b", "semantic_rank": 2, "keyword_rank": 1},
    ]

    ranked = reranker.rerank(documents)

    assert [doc["page_content"] for doc in ranked] == ["doc-a", "doc-b"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_merge_hybrid_results_keeps_documents_sharing_a_long_prefix():
    """缺少 ``chunk_id``／``chunk_index`` 时，键不能靠截断内容生成。

    原实现取 ``page_content[:64]`` 作键。两份来源相同、前 64 字符相同而正文
    不同的文档因此碰撞：语义结果里的那一篇被关键词结果覆盖，
    ``_merge_hybrid_results`` 只返回一条，用户少拿一篇且没有任何提示。
    """
    prefix = "本文档描述了一个非常长的技术说明" * 6  # 远超 64 字符
    semantic_results = [
        {"page_content": prefix + "AAAA 第一部分", "score": 0.9, "metadata": {"source": "kb.md"}},
    ]
    keyword_results = [
        {"page_content": prefix + "BBBB 第二部分", "score": 4.0, "metadata": {"source": "kb.md"}},
    ]

    merged = _merge_hybrid_results(semantic_results, keyword_results)

    assert len(merged) == 2, "前缀相同的两篇不同文档被并成了一条"
    contents = {doc["page_content"] for doc in merged}
    assert prefix + "AAAA 第一部分" in contents
    assert prefix + "BBBB 第二部分" in contents


def test_merge_hybrid_results_still_merges_identical_documents_without_ids():
    """收紧不能误伤：同一篇文档在两路结果里出现时仍须合并计分。"""
    document = {"page_content": "同一篇内容", "metadata": {"source": "kb.md"}}
    semantic_results = [{**document, "score": 0.9}]
    keyword_results = [{**document, "score": 4.0}]

    merged = _merge_hybrid_results(semantic_results, keyword_results)

    assert len(merged) == 1
    assert merged[0]["semantic_score"] == 0.9
    assert merged[0]["keyword_score"] == 4.0


@pytest.mark.parametrize(
    "vector_weight,keyword_weight",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
    ],
)
def test_hybrid_reranker_rejects_non_finite_or_non_positive_weights(vector_weight, keyword_weight):
    """权重之和必须是正的有限值，否则得分是 ``nan`` 或全零。

    ``nan`` 尤其隐蔽：``nan <= 0`` 为 False，只判「和是否为正」会放过去，
    随后每个文档得分都是 ``nan``——排序结果不确定，且 ``nan`` 与分数阈值的
    比较恒为 False，整批结果被静默丢弃。
    """
    reranker = HybridReranker(
        vector_weight=vector_weight, keyword_weight=keyword_weight, fusion_strategy="weighted"
    )

    with pytest.raises(ValueError, match="有限数值|不能同时为 0"):
        reranker.rerank([{"page_content": "a", "semantic_score": 1.0, "keyword_score": 1.0}])


def test_hybrid_reranker_accepts_valid_weights():
    """收紧不能误伤：合法权重照常算出得分。"""
    reranker = HybridReranker(vector_weight=0.7, keyword_weight=0.3, fusion_strategy="weighted")

    ranked = reranker.rerank([{"page_content": "a", "semantic_score": 1.0, "keyword_score": 1.0}])

    assert ranked[0]["score"] == 1.0


# ── 回归：融合链路对分数字段的形态异常必须容忍 ──
#
# 展示层已有 ``document_score`` 收敛形态（``bool``／``nan``／非数字一律退回
# 默认值），但**融合链路**没有：它直接 ``float(doc.get("score") or 0)``。检索结果
# 可以来自任何 ``VectorStoreBase`` 实现——``FaissStore`` 归一化，第三方 store
# 不会，而抽象基类正是为替换实现而存在的。实测异常形态会分两种方式失败：
#
# 1. **崩溃**：``score="abc"`` 抛 ``ValueError``、``score=[1]`` 抛 ``TypeError``，
#    异常从 ``_merge_hybrid_results`` 一路冒到 chat 主循环之外。
# 2. **静默错序**（更隐蔽）：``_normalize_scores`` 里 ``max()``/``min()`` 一旦
#    遇到 ``nan`` 就双双返回 ``nan``，``score_range`` 也是 ``nan``，每个文档的
#    归一化结果都是 ``nan``——排序时 ``nan`` 的比较恒为 False，该文档会**留在
#    首位**（Python 的 ``sorted`` 保持原相对顺序），于是一条分数不可用的文档被
#    排到了正常文档前面，且没有任何提示。
#
# 判据用「不抛异常 + 得分是有限数值」，而不是「等于某个具体值」：后者会把这套
# 兜底规则写死在测试里，而这里要守的是「异常形态不会破坏排序与过滤」。


def _score_of(doc):
    return doc.get("score")


@pytest.mark.parametrize(
    "malformed",
    [
        {"page_content": "a", "score": None},
        {"page_content": "a", "score": "0.5"},
        {"page_content": "a", "score": "abc"},
        {"page_content": "a", "score": [1]},
        {"page_content": "a", "score": {"v": 1}},
        {"page_content": "a", "score": True},
        {"page_content": "a", "score": float("nan")},
        {"page_content": "a", "score": float("inf")},
        {"page_content": "a"},
    ],
)
def test_merge_hybrid_results_tolerates_malformed_scores(malformed):
    """分数字段形态异常时不得抛异常，且得分必须是有限数值。"""
    import math

    merged = _merge_hybrid_results([malformed], [malformed])

    assert len(merged) == 1
    assert math.isfinite(float(merged[0]["score"]))


@pytest.mark.parametrize(
    "malformed",
    [
        {"page_content": "a", "semantic_score": "abc", "keyword_score": 0.5},
        {"page_content": "a", "semantic_score": [1], "keyword_score": 0.5},
        {"page_content": "a", "keyword_score": {"v": 1}},
        {"page_content": "a", "semantic_score": float("nan"), "keyword_score": 0.5},
        {"page_content": "a", "semantic_score": float("inf"), "keyword_score": 0.5},
    ],
)
@pytest.mark.parametrize("strategy", ["weighted", "rrf"])
def test_hybrid_reranker_tolerates_malformed_scores(malformed, strategy):
    """融合打分不得因分数字段形态异常而抛错，结果也必须是有限数值。"""
    import math

    reranker = HybridReranker(0.5, 0.5, fusion_strategy=strategy)

    ranked = reranker.rerank([malformed])

    assert len(ranked) == 1
    assert math.isfinite(float(ranked[0]["score"]))


def test_weighted_rerank_does_not_promote_a_nan_scored_document():
    """一条 ``nan`` 得分的文档不得因为比较恒为 False 而留在首位。

    这是本组问题里最隐蔽的一支：``sorted`` 对 ``nan`` 的比较恒为 False，该文档
    保持原相对顺序、即留在第一位，用户看到一条分数不可用的结果排在正常结果
    之前。
    """
    import math

    reranker = HybridReranker(0.5, 0.5, fusion_strategy="weighted")
    documents = [
        {"page_content": "坏分数", "semantic_score": float("nan"), "keyword_score": 0.0},
        {"page_content": "好分数", "semantic_score": 1.0, "keyword_score": 1.0},
    ]

    ranked = reranker.rerank(documents)

    assert all(math.isfinite(float(doc["score"])) for doc in ranked)
    assert ranked[0]["page_content"] == "好分数"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.5", 0.5),
        (" 0.25 ", 0.25),
        (3, 3.0),
        (2.5, 2.5),
    ],
)
def test_coerce_score_still_accepts_convertible_values(value, expected):
    """收紧不能误伤：能转换的值必须照旧转换。

    旧实现是 ``float(x or 0)``，它接受 ``"0.5"`` 这类数字字符串。若改成「非数字
    类型一律退回默认值」，一个返回字符串得分的第三方 store 会被静默降为 0.0，
    再被分数阈值全部过滤——从「能用」变成「静默返回空结果」。
    """
    from src.ui.display_utils import coerce_score

    assert coerce_score(value) == expected


@pytest.mark.parametrize(
    "value",
    ["abc", "", None, True, False, [1], {"v": 1}, float("nan"), float("inf"), float("-inf")],
)
def test_coerce_score_falls_back_for_unusable_values(value):
    """真正不可用的值退回默认值，不抛异常、也不产出非有限数。"""
    from src.ui.display_utils import coerce_score

    result = coerce_score(value)

    assert result == 0.0


def test_retrieve_normalizes_score_at_the_exit_boundary():
    """``retrieve`` 出口的 ``score`` 必须是有限浮点，三条路径一致。

    单路检索（语义／关键词）不经过 ``HybridReranker``，因此不会重算 score——
    第三方 store 返回的 ``"0.8"`` 会原样流到出口。结果是同一份结果出现
    「阈值按 0.8 通过筛选、界面显示 0.0000」，而排序、去重与展示又各自按不同
    规则解释它。这里让出口只有一种形态。
    """
    import asyncio
    import math

    from src.retrieval.vdb.base import VectorStoreBase
    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, get_settings

    class _StringScoreStore(VectorStoreBase):
        def add_documents(self, documents):
            raise NotImplementedError

        async def aadd_documents(self, documents):
            raise NotImplementedError

        def search(self, query, top_k=5, search_type="semantic"):
            raise NotImplementedError

        async def asearch(self, query, top_k=5, search_type="semantic"):
            raise NotImplementedError

        def save(self, path):
            raise NotImplementedError

        def load(self, path, *, trusted_paths=None):
            raise NotImplementedError

        def get_embedding_model(self):
            raise NotImplementedError

        def semantic_search(self, query_embedding, top_k=5):
            return [{"page_content": "命中文档", "score": "0.8", "metadata": {}}]

        def resolve_parent_content(self, parent_id):
            return None

    class _StubEmbedding:
        async def embed_query(self, query):
            return [0.0] * 4

    config = build_session_config(get_settings())
    config["retrieval_method"] = RetrievalMethod.SEMANTIC_SEARCH
    config["score_threshold"] = 0.5

    service = RetrievalService(vector_store=_StringScoreStore(), embedding_service=_StubEmbedding())
    results = asyncio.run(service.retrieve("查询", config))

    assert results, "字符串得分通过了阈值，结果不该为空"
    for document in results:
        assert isinstance(document["score"], float), "出口 score 必须是 float"
        assert math.isfinite(document["score"])
    assert results[0]["score"] == 0.8


# ── 回归：融合策略与文档键的非字符串/超大取值 ──
#
# ``HybridReranker`` 原先直接调 ``fusion_strategy.strip()``，非字符串抛
# ``AttributeError``，而契约是 ``ValueError``；``_document_key`` 对元数据做
# ``str()``，而元数据来自知识快照（含 pickle 来源，二进制格式不受十进制位数
# 限制），位数超限时 ``str()`` 抛 ``ValueError``。


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(["rrf"], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(123, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_hybrid_reranker_rejects_non_string_fusion_strategy(bad_value):
    with pytest.raises(ValueError, match="融合策略"):
        HybridReranker(0.5, 0.5, fusion_strategy=bad_value)


@pytest.mark.parametrize("metadata_key", ["chunk_id", "doc_id", "chunk_index", "source"])
def test_document_key_survives_huge_integer_metadata(metadata_key):
    key = _document_key({"metadata": {metadata_key: 10**4301}, "page_content": "内容"})
    assert isinstance(key, str) and key


def test_document_key_stays_stable_for_equal_values():
    """同一取值必须得到同一键，否则去重会失效。"""
    first = _document_key({"metadata": {"chunk_id": 10**4301}, "page_content": "a"})
    second = _document_key({"metadata": {"chunk_id": 10**4301}, "page_content": "b"})
    assert first == second


# ── 回归：rerank provider 返回值必须先验形态再解包 ──
#
# 直接写 ``a, b = await provider.arerank(...)`` 时，``None``／单值抛裸
# ``TypeError``、三元组抛裸 ``ValueError``——都在 ``_validate_rerank_output``
# （正是「校验 provider 返回值」的边界）之前，用户拿到的是「无法解包」而不是
# 「provider 返回值不合法」。


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(None, id="none"),
        pytest.param(5, id="single-value"),
        pytest.param(([0], [0.5], "extra"), id="three-elements"),
        pytest.param("ab", id="string"),
    ],
)
def test_rerank_result_must_be_a_pair(bad_value):
    with pytest.raises(RuntimeError, match="二元组"):
        _unpack_rerank_result(bad_value)


def test_rerank_result_accepts_valid_pairs():
    assert _unpack_rerank_result(([0], [0.5])) == ([0], [0.5])
    assert _unpack_rerank_result([[0], [0.5]]) == ([0], [0.5])


# ── 回归：权重之和与 RRF 分母都是**派生值**，必须自己判可转换性 ──
#
# 两个权重各自都在 double 范围内（构造期已判），但它们的**和**可以超出；
# ``rrf_k + rank`` 同理。``math.isfinite`` 对超出范围的 int 自己抛
# ``OverflowError``——``ArithmeticError`` 子类，**不是** ``ValueError``，穿透
# 本类调用方的全部 ``except ValueError`` 分类。


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(10**308, id="just-over-double-sum"),
        pytest.param(10**305, id="large-but-summable"),
    ],
)
def test_weight_sum_overflow_is_a_value_error(weight):
    """两个权重各自合法但和超出范围时，收场必须是 ``ValueError``。"""
    reranker = HybridReranker(weight, weight)

    # 10**305 * 2 仍在 double 范围内，因此那条路径应正常返回；
    # 10**308 * 2 超出，必须在构造期或 rerank 期抛 ValueError。
    try:
        reranker.rerank([{"metadata": {}, "page_content": "x", "score": 0.5}])
    except ValueError:
        pass
    except ArithmeticError as exc:  # pragma: no cover - 回归失败时触发
        pytest.fail(f"溢出应以 ValueError 收场，实际是 {type(exc).__name__}: {exc}")


@pytest.mark.parametrize(
    "rank",
    [
        pytest.param(10**400, id="400digits"),
        pytest.param(10**4301, id="4301digits"),
        pytest.param(-5, id="negative"),
        pytest.param("abc", id="non-numeric-string"),
    ],
)
def test_rrf_rank_out_of_range_is_treated_as_absent(rank):
    """非法的 rank 视为「未参与该路检索」，不得抛 ``OverflowError``。"""
    reranker = HybridReranker(0.5, 0.5)

    result = reranker.rerank([{"metadata": {}, "page_content": "x", "semantic_rank": rank}])

    assert result[0]["score"] == 0.0


def test_rrf_rank_in_range_still_counts():
    reranker = HybridReranker(0.5, 0.5)

    result = reranker.rerank([{"metadata": {}, "page_content": "x", "semantic_rank": 1}])

    assert result[0]["score"] > 0.0


# ── 回归：rerank 缓存键必须能区分含超大整数的 options ──
#
# ``json.dumps(default=str)`` 对 int 无效（``default=`` 只对未知类型生效），
# ``repr`` 兜底抛同一个 ``ValueError``；而 ``safe_repr`` 对容器只退回 ``<dict>``，
# 会让两份不同的 options 撞到同一个 provider 实例。


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"x": 10**4301}, {"x": 10**4302}),
        ({"x": 10**4301}, {"y": 10**4301}),
        ({"x": 10**4301}, {"x": 10**4301 + 1}),
    ],
)
def test_stable_fingerprint_distinguishes_huge_integers(left, right):
    assert _stable_fingerprint(left) != _stable_fingerprint(right)


def test_stable_fingerprint_does_not_raise_on_huge_integers():
    assert _stable_fingerprint({"nested": [{"deep": 10**4301}]})


# ── 回归：``_stable_fingerprint`` 的两条承诺在集合与自引用容器上都不成立 ──
#
# 该函数的 docstring 写了两条承诺，实测都只在 dict/list/tuple 上成立：
#
# 1.「递归到叶子」——集合走的是 ``repr()`` 兜底，**没有递归**。集合里含位数
#    超限的整数时 ``repr()`` 抛 ``ValueError``，于是整个集合塌成 ``<set>``，
#    区分度归零。``ModelDetail.options`` 是 ``dict[str, Any]``，
#    ``validate_secret_free_options`` 只查敏感键名、不做递归类型校验，实测
#    ``options={"x": {10**4301}}`` 与 ``{"x": {10**4302}}`` 都被接受，随后两者
#    得到同一个缓存键——两份不同的 rerank 配置静默复用同一个 provider 实例，
#    比崩溃更糟（错误配置的请求打到错误的端点上，且没有任何报错）。
# 2.「绝不抛异常」——自引用容器让它无限递归，抛 ``RecursionError``，而
#    ``RecursionError`` 继承 ``RuntimeError``，不在调用方
#    ``except (TypeError, ValueError)`` 的范围内，会直接穿透成未脱敏的 traceback。
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"x": {10**4301}}, {"x": {10**4302}}),
        ({"x": frozenset({10**4301})}, {"x": frozenset({10**4302})}),
        ({"x": {10**4301}}, {"x": frozenset({10**4301})}),
        ({"x": {1, 2}}, {"x": {3, 4}}),
    ],
)
def test_stable_fingerprint_distinguishes_set_members(left, right):
    assert _stable_fingerprint(left) != _stable_fingerprint(right)


def test_stable_fingerprint_survives_self_referencing_containers():
    """自引用容器不得让指纹函数抛异常（docstring 承诺「绝不抛」）。"""
    cycle_list: list = []
    cycle_list.append(cycle_list)
    cycle_dict: dict = {}
    cycle_dict["self"] = cycle_dict

    assert _stable_fingerprint(cycle_list)
    assert _stable_fingerprint(cycle_dict)
    # 两条自引用路径必须可区分，否则「不抛」是靠塌成同一个常量换来的。
    assert _stable_fingerprint(cycle_list) != _stable_fingerprint(cycle_dict)


# ── 回归：父块内容必须收敛成字符串，不能把 resolver 的返回值原样写进正文 ──
#
# ``_promote_parent_context`` 把 ``resolver(parent_id)`` 直接赋给
# ``promoted_doc["page_content"]``，只判了真值（``if parent_content:``）、没判类型。
# 而 ``page_content: str`` 是 ``Document`` 的契约（``src/models/document.py``），
# ``FaissStore._validate_document_shape`` 在快照载入期也按这条校验。
#
# 可达性：``parent_resolver`` 取自 ``getattr(self.vector_store,
# "resolve_parent_content", None)``，而 ``VectorStoreBase`` 正是为替换实现而存在
# 的——``resolve_parent_content`` 的签名是 ``-> str | None``，但第三方实现返回
# 别的形态不会被任何地方拦下。实测一个返回列表的 store 让
# ``page_content`` 变成 ``['a', 'b']``，返回整数变成 ``123``：展示层
# ``document_content`` 把它收敛成 ``"N/A"``（用户看到「无内容」而不是报错），
# 而 prompt 构造对正文做 ``.lower()`` 时抛 ``AttributeError``——检索整体不可用，
# 报错现场离真正的原因（父块解析器返回了非字符串）很远。
#
# 位数超限的整数是同一根因的另一面：``10**4301`` 作为 ``page_content`` 能通过
# 本函数，直到下游任何一次 ``str()``／``repr()``／``.lower()`` 才抛
# ``ValueError: Exceeds the limit``（pickle 是二进制格式，不受十进制位数限制，
# 畸形快照能把这种值带进来）。
def _promote(document: dict, resolver) -> dict:
    return _promote_parent_context([document], resolver)[0]


def test_promote_parent_context_coerces_non_string_parent_content():
    """非字符串父内容必须收敛成字符串，且正文里能看到它。"""
    document = {"page_content": "子块正文", "metadata": {"parent_id": "p1"}, "score": 0.5}

    promoted = _promote(document, lambda pid: ["a", "b"])

    assert isinstance(promoted["page_content"], str)
    # 只断言「是 str」不够：一个把非字符串一律换成空串的实现也能通过，
    # 而用户会看到空正文。父块内容必须仍然可见。
    assert "a" in promoted["page_content"] and "b" in promoted["page_content"]


def test_promote_parent_context_coerces_huge_integer_parent_content():
    """位数超限的整数父内容不得让下游的 ``str()`` 崩。"""
    document = {"page_content": "子块正文", "metadata": {"parent_id": "p1"}, "score": 0.5}

    promoted = _promote(document, lambda pid: 10**4301)

    content = promoted["page_content"]
    assert isinstance(content, str)
    # 下游会做的事：prompt 构造与展示都会拿它当字符串处理。
    assert content.lower() == content.lower()


def test_promote_parent_context_keeps_string_parent_content_intact():
    """反向守卫：正常的字符串父内容必须原样保留，不能被改写。"""
    document = {"page_content": "子块正文", "metadata": {"parent_id": "p1"}, "score": 0.5}

    promoted = _promote(document, lambda pid: "父块正文")

    assert promoted["page_content"] == "父块正文"
    assert promoted["metadata"]["matched_chunk_content"] == "子块正文"


@pytest.mark.parametrize("empty_value", [None, "", 0, False, [], {}])
def test_promote_parent_context_leaves_content_alone_when_nothing_to_promote(empty_value):
    """空值仍表示「没有父块」，正文保持子块内容不变。"""
    document = {"page_content": "子块正文", "metadata": {"parent_id": "p1"}, "score": 0.5}

    promoted = _promote(document, lambda pid: empty_value)

    assert promoted["page_content"] == "子块正文"
    assert "matched_chunk_content" not in promoted["metadata"]
