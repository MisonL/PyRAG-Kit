import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval import retriever
from src.retrieval.retriever import retrieve_documents
from src.services.retrieval_service import RetrievalService
from src.utils.config import RetrievalMethod


class MockVectorStore:
    def __init__(self, semantic_results, keyword_results, parent_documents=None):
        self.semantic_results = semantic_results
        self.keyword_results = keyword_results
        self.parent_documents = parent_documents or {}
        self.search_calls = []

    def search(self, query, top_k=5, search_type="semantic"):
        self.search_calls.append((search_type, top_k))
        if search_type == "semantic":
            return self.semantic_results
        if search_type == "keyword":
            return self.keyword_results
        raise AssertionError(f"unexpected search_type: {search_type}")

    def resolve_parent_content(self, parent_id):
        parent_document = self.parent_documents.get(parent_id)
        if not parent_document:
            return None
        return parent_document.get("content")


def test_hybrid_retrieval_starts_semantic_and_keyword_paths_in_parallel():
    service = object.__new__(RetrievalService)
    started = []
    release = asyncio.Event()

    async def fake_semantic(_query, _top_k):
        started.append("semantic")
        await release.wait()
        return ["semantic"]

    async def fake_keyword(_query, _top_k):
        started.append("keyword")
        await release.wait()
        return ["keyword"]

    service._semantic_retrieve = fake_semantic
    service._keyword_retrieve = fake_keyword

    async def run():
        task = asyncio.create_task(service._gather_hybrid_results("query", 3))

        async def wait_for_both_starts():
            while len(started) < 2:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_both_starts(), timeout=5)
        release.set()
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(run()) == (["semantic"], ["keyword"])
    assert set(started) == {"semantic", "keyword"}


def test_rerank_provider_is_cached_per_configuration(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        return [0], [float(top_n)]

    provider.arerank.side_effect = fake_arerank
    factory = MagicMock(return_value=provider)
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider", factory
    )
    session_config = SimpleNamespace(
        rerank_enabled=True,
        active_rerank_configuration="siliconflow",
        top_k=1,
        rerank_configurations={
            "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank")
        },
    )
    documents = [{"page_content": "doc", "score": 0.5, "metadata": {}}]

    async def run():
        first = await service._rerank_if_needed("query", documents, session_config)
        second = await service._rerank_if_needed("query", documents, session_config)
        return first, second

    first, second = asyncio.run(run())

    assert first[0]["score"] == 1.0
    assert second[0]["score"] == 1.0
    factory.assert_called_once_with("siliconflow", session_config.rerank_configurations)
    assert provider.arerank.call_count == 2


def test_rerank_provider_cache_refreshes_when_options_change(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    first_provider = MagicMock()
    second_provider = MagicMock()
    factory = MagicMock(side_effect=[first_provider, second_provider])
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider", factory
    )
    first_config = SimpleNamespace(
        provider="siliconflow",
        model_name="rerank",
        options={"timeout": 10},
    )
    second_config = SimpleNamespace(
        provider="siliconflow",
        model_name="rerank",
        options={"timeout": 20},
    )

    assert (
        service._get_rerank_provider("siliconflow", {"siliconflow": first_config}) is first_provider
    )
    assert (
        service._get_rerank_provider("siliconflow", {"siliconflow": second_config})
        is second_provider
    )
    assert factory.call_count == 2


def test_retrieval_service_close_disposes_cached_rerank_and_embedding_models():
    rerank = MagicMock()
    embedding = MagicMock()
    service = RetrievalService(vector_store=object(), embedding_service=embedding)
    service._rerank_providers = {("key", "provider", "model"): rerank}

    service.close()

    rerank.close.assert_called_once_with()
    embedding.close.assert_called_once_with()
    assert service._rerank_providers == {}


def test_retrieval_service_aclose_falls_back_to_sync_embedding_close():
    class SyncOnlyEmbedding:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    embedding = SyncOnlyEmbedding()
    service = RetrievalService(vector_store=object(), embedding_service=embedding)

    asyncio.run(service.aclose())

    assert embedding.closed


def test_rerank_top_n_is_clamped_after_candidate_reduction(monkeypatch):
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        assert top_n == 1
        return [0], [0.9]

    provider.arerank.side_effect = fake_arerank
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider",
        MagicMock(return_value=provider),
    )
    session_config = SimpleNamespace(
        rerank_enabled=True,
        active_rerank_configuration="siliconflow",
        top_k=5,
        rerank_configurations={
            "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank")
        },
    )

    result = asyncio.run(
        service._rerank_if_needed(
            "query", [{"page_content": "only", "score": 0.1, "metadata": {}}], session_config
        )
    )

    assert result[0]["score"] == 0.9


def test_semantic_retrieve_uses_legacy_search_when_method_is_missing():
    class LegacyStore:
        def search(self, query, top_k, search_type="semantic"):
            return [
                {"page_content": query, "score": top_k, "metadata": {"search_type": search_type}}
            ]

    class UnusedEmbedding:
        async def embed_query(self, _query):
            raise AssertionError("legacy semantic search should not require embeddings")

    embedding = UnusedEmbedding()
    service = RetrievalService(LegacyStore(), embedding)

    result = asyncio.run(service._semantic_retrieve("query", 3))

    assert result == [
        {"page_content": "query", "score": 3, "metadata": {"search_type": "semantic"}}
    ]


def test_semantic_retrieve_does_not_hide_embedding_attribute_errors():
    class SemanticStore:
        def semantic_search(self, _embedding, _top_k):
            return []

    class FailingEmbedding:
        async def embed_query(self, _query):
            raise AttributeError("embedding provider bug")

    service = RetrievalService(SemanticStore(), FailingEmbedding())

    with pytest.raises(AttributeError, match="embedding provider bug"):
        asyncio.run(service._semantic_retrieve("query", 3))


def test_semantic_retrieve_does_not_hide_vector_store_attribute_errors():
    class BrokenStore:
        def semantic_search(self, _embedding, _top_k):
            raise AttributeError("vector store bug")

    class Embedding:
        async def embed_query(self, _query):
            return [0.1]

    service = RetrievalService(BrokenStore(), Embedding())

    with pytest.raises(AttributeError, match="vector store bug"):
        asyncio.run(service._semantic_retrieve("query", 3))


@pytest.mark.parametrize(
    ("indices", "scores", "message"),
    [
        ([], [], "未返回可用结果"),
        ([0], [], "数量不一致"),
        ([2], [0.9], "index 无效"),
        ([0, 0], [0.9, 0.8], "重复 index"),
        ([0], [float("nan")], "必须是有限数值"),
    ],
)
def test_rerank_output_contract_failures_are_explicit(indices, scores, message):
    with pytest.raises(RuntimeError, match=message):
        RetrievalService._validate_rerank_output(indices, scores, document_count=2)


def test_retrieve_documents_uses_parent_sidecar_and_overfetches_candidates(monkeypatch):
    semantic_results = [
        {
            "page_content": "child content alpha",
            "score": 0.9,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-1",
                "parent_id": "parent-1",
            },
        },
        {
            "page_content": "child content beta",
            "score": 0.8,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-2",
                "parent_id": "parent-1",
            },
        },
        {
            "page_content": "child content gamma",
            "score": 0.7,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-beta-1",
                "parent_id": "parent-2",
            },
        },
    ]
    keyword_results = [
        {
            "page_content": "child content alpha",
            "score": 6.0,
            "metadata": {
                "source": "knowledge_base/sample.md",
                "page": 1,
                "chunk_id": "chunk-alpha-1",
                "parent_id": "parent-1",
            },
        }
    ]
    store = MockVectorStore(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        parent_documents={
            "parent-1": {"content": "parent content alpha"},
            "parent-2": {"content": "parent content beta"},
        },
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        top_k=2,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.0,
        fusion_strategy="rrf",
        candidate_multiplier=3,
    )

    assert sorted(store.search_calls) == [("keyword", 6), ("semantic", 6)]
    assert len(results) == 2
    assert {doc["metadata"]["parent_id"] for doc in results} == {"parent-1", "parent-2"}
    assert all(doc["page_content"].startswith("parent content") for doc in results)
    assert all(
        doc["metadata"]["matched_chunk_content"].startswith("child content") for doc in results
    )


def test_retrieve_documents_collapses_multiple_children_from_same_parent(monkeypatch):
    store = MockVectorStore(
        semantic_results=[
            {
                "page_content": "child content high",
                "score": 0.9,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            },
            {
                "page_content": "child content low",
                "score": 0.7,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-low",
                    "parent_id": "parent-1",
                },
            },
        ],
        keyword_results=[],
        parent_documents={"parent-1": {"content": "parent content"}},
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.0,
        candidate_multiplier=3,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["parent_id"] == "parent-1"
    assert results[0]["page_content"] == "parent content"
    assert results[0]["metadata"]["matched_chunk_content"] == "child content high"


def test_retrieve_documents_rrf_keeps_results_under_default_threshold(monkeypatch):
    store = MockVectorStore(
        semantic_results=[
            {
                "page_content": "child content high",
                "score": 0.9,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            }
        ],
        keyword_results=[
            {
                "page_content": "child content high",
                "score": 4.0,
                "metadata": {
                    "source": "knowledge_base/sample.md",
                    "page": 1,
                    "chunk_id": "chunk-high",
                    "parent_id": "parent-1",
                },
            }
        ],
        parent_documents={"parent-1": {"content": "parent content"}},
    )
    console = MagicMock()

    monkeypatch.setattr(
        "src.retrieval.retriever.ModelProviderFactory.get_rerank_provider",
        lambda *args, **kwargs: None,
    )

    results = retrieve_documents(
        query="sample query",
        vector_store=store,
        console=console,
        retrieval_method=RetrievalMethod.HYBRID_SEARCH,
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        active_rerank_configuration="siliconflow",
        score_threshold=0.4,
        fusion_strategy="rrf",
        candidate_multiplier=3,
    )

    assert results
    assert results[0]["page_content"] == "parent content"
    assert results[0]["metadata"]["parent_id"] == "parent-1"


def _rerank_session_config(top_k):
    return SimpleNamespace(
        rerank_enabled=True,
        active_rerank_configuration="siliconflow",
        top_k=top_k,
        rerank_configurations={
            "siliconflow": SimpleNamespace(provider="siliconflow", model_name="rerank")
        },
    )


def test_rerank_reorders_documents_by_returned_index_and_score(monkeypatch):
    """provider 返回乱序 index 时，结果必须按 index 重排并按分数降序。

    既有用例的替身恒返回 ``[0]``（恒等置换），重排与 ``sorted(reverse=True)``
    从未被执行，因此这条路径长期无守卫。
    """
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        assert top_n == 3
        return [2, 0, 1], [0.9, 0.5, 0.3]

    provider.arerank.side_effect = fake_arerank
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider",
        MagicMock(return_value=provider),
    )
    documents = [
        {"page_content": "doc-a", "score": 0.1, "metadata": {}},
        {"page_content": "doc-b", "score": 0.2, "metadata": {}},
        {"page_content": "doc-c", "score": 0.3, "metadata": {}},
    ]

    reranked = asyncio.run(
        service._rerank_if_needed("query", documents, _rerank_session_config(top_k=3))
    )

    assert [doc["page_content"] for doc in reranked] == ["doc-c", "doc-a", "doc-b"]
    assert [doc["score"] for doc in reranked] == [0.9, 0.5, 0.3]
    # 原列表不得被就地修改
    assert [doc["page_content"] for doc in documents] == ["doc-a", "doc-b", "doc-c"]


def test_rerank_output_order_follows_scores_not_provider_order(monkeypatch):
    """provider 若按 index 升序返回但分数递减，最终顺序仍应由分数决定。"""
    service = RetrievalService(vector_store=object(), embedding_service=object())
    provider = MagicMock()

    async def fake_arerank(_query, _documents, top_n):
        return [0, 1, 2], [0.1, 0.9, 0.4]

    provider.arerank.side_effect = fake_arerank
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider",
        MagicMock(return_value=provider),
    )
    documents = [{"page_content": f"doc-{name}", "score": 0.0, "metadata": {}} for name in "abc"]

    reranked = asyncio.run(
        service._rerank_if_needed("query", documents, _rerank_session_config(top_k=3))
    )

    assert [doc["page_content"] for doc in reranked] == ["doc-b", "doc-c", "doc-a"]


@pytest.mark.parametrize(
    ("indices", "scores", "match"),
    [
        ([True], [0.9], "index 无效"),
        ([0], [True], "score 无效"),
        ([0, 0], [0.9, 0.8], "重复 index"),
        ([3], [0.9], "index 无效"),
        ([-1], [0.9], "index 无效"),
        ([0], [float("inf")], "有限数值"),
    ],
)
def test_validate_rerank_output_rejects_invalid_provider_results(indices, scores, match):
    """契约失败必须是显式异常，而不是让错误顺序静默进入检索结果。"""
    with pytest.raises(RuntimeError, match=match):
        RetrievalService._validate_rerank_output(indices, scores, document_count=3)


# ---------------------------------------------------------------------------
# 超出双精度范围的整数
# ---------------------------------------------------------------------------
#
# provider 的 score 来自网络 JSON，里面出现一个超大整数字面量就足以让
# ``float(score)`` 抛 ``OverflowError``——它不是 ``ValueError``，原实现里的
# ``except (TypeError, ValueError)`` 接不住，于是本该是 ``RuntimeError``
# 的契约失败变成了裸异常。
#
# 参数 id 必须显式给出：``10**4301`` 超过 ``sys.get_int_max_str_digits()``，
# pytest 自动生成 id 时调 ``str(val)`` 会自己抛 ``ValueError``。
_BEYOND_DOUBLE_INTS = [
    pytest.param(10**400, id="400digits"),
    pytest.param(-(10**400), id="negative-400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
]


@pytest.mark.parametrize("score", _BEYOND_DOUBLE_INTS)
def test_validate_rerank_output_rejects_scores_beyond_double_range(score):
    """超出双精度范围的 score 必须报成 ``RuntimeError``，不是 ``OverflowError``。"""
    with pytest.raises(RuntimeError, match="超出双精度范围"):
        RetrievalService._validate_rerank_output([0], [score], document_count=2)


# ── 向量存储接口缺失与 rerank 返回值形态的覆盖缺口 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3725 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 742 ``_call_vector_store`` 的「store 未实现该接口」→ 裸
#   ``TypeError: 'NoneType' object is not callable``（拿 ``None`` 当方法调），
#   而调用方按 ``NotImplementedError`` 分类；
# * 753 ``_legacy_search`` 的「store 既没有 asearch 也没有 search」→ 同样
#   退化成裸 ``TypeError``；
# * 1015 ``_validate_rerank_output`` 的「provider 返回不可迭代」→ 裸
#   ``UnboundLocalError: normalized_indices``（变量未绑定）；
# * 1045 同一函数的 ``float(score)`` 失败分支 → ``None`` 这类取值裸
#   ``UnboundLocalError: normalized_score``。
#
# 既有的 ``test_validate_rerank_output_rejects_invalid_provider_results`` 覆盖
# 的是 ``bool``／越界／重复 index 等**取值**判据，两条**形态**判据从未被反向覆盖。
def test_call_vector_store_reports_missing_interface_readably():
    """store 未实现该接口时必须抛 ``NotImplementedError``，不是裸 ``TypeError``。"""

    class BareStore:
        pass

    service = object.__new__(RetrievalService)
    service.vector_store = BareStore()

    with pytest.raises(NotImplementedError, match="未实现 semantic_search 接口"):
        asyncio.run(service._call_vector_store("semantic_search", None))


def test_legacy_search_reports_missing_interface_readably():
    """store 既无 asearch 也无 search 时必须抛 ``NotImplementedError``。"""

    class BareStore:
        pass

    service = object.__new__(RetrievalService)
    service.vector_store = BareStore()

    with pytest.raises(NotImplementedError, match="未实现兼容 search 接口"):
        asyncio.run(service._legacy_search("query", 1, "semantic"))


def test_validate_rerank_output_rejects_non_iterable_result():
    """provider 返回不可迭代的取值时必须抛 ``RuntimeError``，不是裸 ``UnboundLocalError``。"""
    with pytest.raises(RuntimeError, match="必须是可迭代的索引和分数"):
        RetrievalService._validate_rerank_output(5, 5, document_count=3)


@pytest.mark.parametrize(
    "bad_score",
    [
        pytest.param(None, id="none"),
        pytest.param([0.5], id="list"),
    ],
)
def test_validate_rerank_output_rejects_unconvertible_score(bad_score):
    """``float(score)`` 转换失败的取值必须抛 ``RuntimeError``，不是裸 ``UnboundLocalError``。"""
    with pytest.raises(RuntimeError, match="score 无效"):
        RetrievalService._validate_rerank_output([0], [bad_score], document_count=3)


class _CloseFailingService:
    """检索成功但 ``aclose`` 抛错的服务替身。"""

    def __init__(self, results=None, close_error=None):
        self._results = results or []
        self._close_error = close_error

    async def retrieve(self, **_kwargs):
        return list(self._results)

    async def aclose(self):
        if self._close_error is not None:
            raise self._close_error


def _aretrieve_with_service(monkeypatch, service):
    """用替身服务跑一次 ``aretrieve_documents``，绕开真实 embedding/工厂。"""
    monkeypatch.setattr("src.retrieval.retriever.RetrievalService", lambda **_kwargs: service)
    monkeypatch.setattr("src.retrieval.retriever.EmbeddingService", lambda *_a, **_k: object())
    return asyncio.run(
        retriever.aretrieve_documents(
            query="q",
            vector_store=object(),
            console=MagicMock(),
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            top_k=3,
            vector_weight=0.7,
            keyword_weight=0.3,
            rerank_enabled=False,
            active_rerank_configuration="",
            score_threshold=0.0,
        )
    )


def test_aretrieve_returns_results_when_service_close_fails(monkeypatch, caplog):
    """关闭失败不能顶替掉一次成功的检索结果。

    ``finally`` 里裸 ``await aclose()`` 时，检索已返回的命中结果被丢弃，
    调用方只看到与检索无关的关闭异常（实测返回 1 条命中的检索整体抛
    ``RuntimeError: 关闭失败``）。契约与 ``retrieval_test.core`` 的
    ``_aclose_quietly``、``Chatbot.close`` 一致：释放失败只记警告。
    """
    import logging

    service = _CloseFailingService(
        results=[{"page_content": "命中", "score": 0.9, "metadata": {}}],
        close_error=RuntimeError("关闭失败"),
    )
    with caplog.at_level(logging.WARNING, logger="src.retrieval.retriever"):
        results = _aretrieve_with_service(monkeypatch, service)
    assert results == [{"page_content": "命中", "score": 0.9, "metadata": {}}]
    assert "释放检索服务失败" in caplog.text


def test_aretrieve_keeps_original_error_when_both_retrieve_and_close_fail(monkeypatch, caplog):
    """检索与关闭都失败时，保留**原始**异常，不被关闭异常顶替。

    ``finally`` 里的异常会替换掉 ``try`` 中正在上抛的那个——真正的失败原因
    从栈上消失，排查时看到的是与检索无关的关闭错误。
    """
    import logging

    service = _CloseFailingService(close_error=RuntimeError("关闭也失败"))

    async def _retrieve(**_kwargs):
        raise ValueError("检索本身失败")

    service.retrieve = _retrieve
    with (
        caplog.at_level(logging.WARNING, logger="src.retrieval.retriever"),
        pytest.raises(ValueError, match="检索本身失败"),
    ):
        _aretrieve_with_service(monkeypatch, service)
    assert "释放检索服务失败" in caplog.text


def test_aretrieve_close_failure_is_redacted_in_log(monkeypatch, caplog):
    """关闭异常的日志必须脱敏——``aclose`` 的异常文本可能带凭证。"""
    import logging

    service = _CloseFailingService(
        results=[],
        close_error=RuntimeError("api_key=sk-abcdefghijklmnopqrstuvwxyz123456"),
    )
    with caplog.at_level(logging.WARNING, logger="src.retrieval.retriever"):
        _aretrieve_with_service(monkeypatch, service)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in caplog.text
