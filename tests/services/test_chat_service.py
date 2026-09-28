import asyncio

import pytest

from src.services.chat_service import ChatService


class _FailingModel:
    async def ainvoke(self, **_kwargs):
        raise AttributeError("provider response bug")
        yield ""


class _EmptyModel:
    async def ainvoke(self, **_kwargs):
        if False:
            yield ""


class _ModernAnthropicModel:
    def __init__(self):
        self.calls = []

    def supports_sampling_option(self, name):
        assert name == "temperature"
        return False

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        yield "检索"


class _ResponsesReasoningModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        yield "检索"


class _SamplingModel(_ResponsesReasoningModel):
    def supports_sampling_option(self, name):
        assert name == "temperature"
        return True


def test_identify_intent_propagates_provider_failure():
    service = ChatService(_FailingModel(), retrieval_service=object())

    with pytest.raises(RuntimeError, match="意图识别失败"):
        asyncio.run(service.identify_intent("问题"))


def test_identify_intent_rejects_empty_provider_response():
    service = ChatService(_EmptyModel(), retrieval_service=object())

    with pytest.raises(RuntimeError, match="返回空结果"):
        asyncio.run(service.identify_intent("问题"))


def test_identify_intent_omits_sampling_for_models_that_reject_temperature():
    model = _ModernAnthropicModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert model.calls == [{"prompt": model.calls[0]["prompt"], "stream": False}]


def test_identify_intent_omits_sampling_without_capability_declaration():
    model = _ResponsesReasoningModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert "temperature" not in model.calls[0]


def test_generate_reply_omits_sampling_without_capability_declaration():
    model = _ResponsesReasoningModel()
    service = ChatService(model, retrieval_service=object())

    async def consume():
        return [
            chunk
            async for chunk in service.generate_reply(
                "问题",
                "意图",
                [],
                type("SessionConfig", (), {"chat_temperature": 0.7})(),
            )
        ]

    assert asyncio.run(consume()) == ["检索"]
    assert "temperature" not in model.calls[0]


def test_chat_service_passes_sampling_when_capability_is_declared():
    model = _SamplingModel()
    service = ChatService(model, retrieval_service=object())

    assert asyncio.run(service.identify_intent("问题")) == "检索"
    assert model.calls[0]["temperature"] == 0.1

    async def consume():
        return [
            chunk
            async for chunk in service.generate_reply(
                "问题",
                "意图",
                [],
                type("SessionConfig", (), {"chat_temperature": 0.4})(),
            )
        ]

    assert asyncio.run(consume()) == ["检索"]
    assert model.calls[1]["temperature"] == 0.4


def test_build_prompt_handles_documents_without_source_metadata():
    prompt = ChatService._build_prompt(
        "问题",
        "意图",
        [{"page_content": "知识内容"}],
    )

    assert "来源: 未知来源" in prompt
    assert "内容: 知识内容" in prompt


# ── 回归：prompt 构造不得被形态异常的文档打断，也不得把 None 当正文喂给 LLM ──
#
# ``_build_prompt`` 直接对检索结果做 ``doc.get('metadata') or {}`` 与
# ``doc.get('page_content', '')``。检索结果可以来自任何 ``VectorStoreBase``
# 实现——``FaissStore`` 会在载入期校验并归一化，第三方 store 不会。实测
# ``metadata`` 为 list、条目本身不是映射时抛 ``AttributeError``，而更隐蔽的一类
# 是 ``page_content=None``：它不抛错，而是把字面量 ``内容: None`` 当作知识喂给
# LLM——模型会把「None」当成一条真实的知识内容来回答。


@pytest.mark.parametrize(
    "document,label",
    [
        ({"page_content": "正文", "metadata": ["a"]}, "metadata 为列表"),
        ({"page_content": "正文", "metadata": "not-a-dict"}, "metadata 为字符串"),
        ({"page_content": "正文", "metadata": 42}, "metadata 为整数"),
        ("不是映射", "条目本身不是映射"),
        (None, "条目为 None"),
        (42, "条目为整数"),
    ],
)
def test_build_prompt_survives_malformed_documents(document, label):
    """形态异常时必须照常产出 prompt，而不是抛 ``AttributeError``。

    断言分两层，缺一不可：

    1. 不抛异常，且产出了 prompt（``"问题"`` 在文本里）。
    2. **有可用正文时，正文必须真的进入 prompt**。

    第 2 条是后补的：原先只断言 ``"问题" in prompt``，而那个字符串在「没有找到
    相关知识」那一支里也有。实测把 ``_build_prompt`` 换成一个「丢弃全部知识、
    只回显问题」的实现，这条用例照样通过——它能发现崩溃，却**发现不了知识正文
    被静默丢弃**（那正是本项目最该防的失败：检索退化被伪装成模型答不出来）。

    ``metadata`` 异常的那几组正文都是 ``"正文"``，因此必须出现在 prompt 里；
    条目本身不是映射／是 ``None`` ／是整数的那几组没有正文可取，走无知识分支。
    """
    prompt = ChatService._build_prompt("问题", "意图", [document])

    assert "问题" in prompt
    has_usable_content = (
        isinstance(document, dict)
        and isinstance(document.get("page_content"), str)
        and document["page_content"]
    )
    if has_usable_content:
        assert document["page_content"] in prompt, f"{label}: 可用正文被丢弃了"


@pytest.mark.parametrize(
    "document",
    [
        {"page_content": None, "metadata": {"source": "a.md"}},
        {"page_content": ["x"], "metadata": {"source": "a.md"}},
        {"page_content": 42, "metadata": {"source": "a.md"}},
        {"metadata": {"source": "a.md"}},
    ],
)
def test_build_prompt_does_not_feed_malformed_content_to_the_model(document):
    """非字符串内容不得以 ``内容: None`` 这类字面量进入 prompt。

    这类形态不抛错，但会把「None」当作一条真实知识交给模型——比崩溃更难发现。
    """
    prompt = ChatService._build_prompt("问题", "意图", [document])

    assert "内容: None" not in prompt
    assert "内容: ['x']" not in prompt
    assert "内容: 42" not in prompt


def test_build_prompt_still_includes_well_formed_content():
    """收紧不能误伤：正常文档仍要完整进入 prompt。"""
    prompt = ChatService._build_prompt(
        "问题", "意图", [{"page_content": "知识内容", "metadata": {"source": "kb.md"}}]
    )

    assert "来源: kb.md" in prompt
    assert "内容: 知识内容" in prompt


# ── 回归：检索服务对 store 返回值的形态要求 ──
#
# ``RetrievalService`` 是三个消费者（聊天 prompt、召回测试表格、聊天详情表格）
# 的共同上游。检索结果可以来自任何 ``VectorStoreBase`` 实现，而抽象基类正是为
# 替换实现而存在的——实测返回 ``None`` 或 ``list[str]`` 的 store 会在这里抛
# ``TypeError``／``AttributeError``，报错现场离「你的 store 返回了什么」很远。


def _retrieval_service_with_store(store):
    import numpy as np

    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, Settings

    class _Embedding:
        async def embed_query(self, _query):
            return np.zeros(4, dtype=np.float32)

    config = build_session_config(Settings(_env_file=None))
    config["retrieval_method"] = RetrievalMethod.SEMANTIC_SEARCH
    config["score_threshold"] = 0.0
    return RetrievalService(store, _Embedding()), config


@pytest.mark.parametrize(
    "payload,label",
    [
        (None, "返回 None"),
        ({"a": 1}, "返回映射"),
        (["x"], "返回字符串列表"),
        ([None], "列表元素为 None"),
    ],
)
def test_retrieve_rejects_malformed_store_results(payload, label):
    """store 返回非文档列表时必须给出可读的错误，而不是 ``TypeError``。"""
    import asyncio
    import io

    from rich.console import Console

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return payload

    service, config = _retrieval_service_with_store(_Store())

    with pytest.raises(TypeError, match="文档"):
        asyncio.run(service.retrieve("查询", config, console=Console(file=io.StringIO())))


def test_retrieve_accepts_well_formed_store_results():
    """收紧不能误伤：正常结果照常返回。"""
    import asyncio
    import io

    from rich.console import Console

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "正文", "metadata": {"source": "a.md"}, "score": 0.5}]

    service, config = _retrieval_service_with_store(_Store())

    documents = asyncio.run(service.retrieve("查询", config, console=Console(file=io.StringIO())))

    assert len(documents) == 1
    assert documents[0]["page_content"] == "正文"


@pytest.mark.parametrize(
    "metadata,label",
    [
        (["a"], "元数据为列表"),
        ("not-a-dict", "元数据为字符串"),
        (42, "元数据为整数"),
        ({"source": "a.md"}, "正常元数据"),
        (None, "元数据为 None"),
    ],
)
def test_retrieve_survives_malformed_metadata(metadata, label):
    """``metadata`` 非映射时检索必须照常完成。

    ``document.get("metadata") or {}`` 只挡得住 ``None`` 与空值：**非空的非映射**
    会原样留下，随后 ``_deduplicate_parent_documents`` 里的 ``.get("parent_id")``
    抛 ``AttributeError: 'list' object has no attribute 'get'``——而这是检索管线
    内部，报错现场离「你的 store 返回了什么」很远。
    """
    import asyncio
    import io

    from rich.console import Console

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "正文", "metadata": metadata, "score": 0.5}]

    service, config = _retrieval_service_with_store(_Store())

    documents = asyncio.run(service.retrieve("查询", config, console=Console(file=io.StringIO())))

    assert len(documents) == 1
    assert isinstance(documents[0]["metadata"], dict)


@pytest.mark.parametrize(
    "page_content,expected,label",
    [
        ("正文", "正文", "正常内容"),
        (None, "", "内容为 None"),
        (["x"], "", "内容为列表"),
        (42, "", "内容为整数"),
        ("", "", "内容为空串"),
    ],
)
def test_rerank_receives_only_strings(page_content, expected, label):
    """送给 Rerank provider 的必须是字符串列表。

    ``docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]``
    在内容非字符串时会把 ``None``／``42`` 这类值直接交给 provider，由 provider 的
    ``_validate_inputs`` 抛「documents 必须全部是字符串」——错误信息指向 provider
    的参数，而真正的原因是上游 store 返回了形态异常的文档。与展示层同理，
    在这里收敛成字符串更靠近原因。
    """
    import asyncio
    import io

    from rich.console import Console

    received: list[list[str]] = []

    class _Rerank:
        async def arerank(self, _query, documents, *, top_n):
            received.append(list(documents))
            return list(range(len(documents))), [0.5] * len(documents)

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": page_content, "metadata": {"source": "a.md"}, "score": 0.5}]

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    service._get_rerank_provider = lambda *_args, **_kwargs: _Rerank()

    asyncio.run(service.retrieve("查询", config, console=Console(file=io.StringIO())))

    assert received == [[expected]], label


# ── 回归：hybrid 检索要在**合并之前**校验两路结果的形态 ──
#
# ``_require_document_list`` 原先只在合并与 rerank **之后**运行，而
# ``_merge_hybrid_results`` 已经对每个条目调了 ``.get(...)``——store 返回
# ``[None]`` 时抛的是 ``AttributeError: 'NoneType' object has no attribute 'get'``，
# 而不是该校验承诺的可定位 ``TypeError``（附带是哪一路、第几条出的问题）。
# 三条检索路径的收场方式因此不一致：semantic／keyword 单路给可读错误，
# hybrid 给裸 AttributeError。


@pytest.mark.parametrize(
    ("label", "semantic", "keyword"),
    [
        ("语义路返回 [None]", [None], []),
        ("关键词路返回 [None]", [], [None]),
        ("语义路返回 None", None, []),
        ("关键词路返回字符串", [], "不是列表"),
    ],
)
def test_hybrid_retrieval_reports_which_side_violated_the_contract(label, semantic, keyword):
    """两路结果的形态问题必须报成可定位的 TypeError，且指明是哪一路。"""
    import asyncio

    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, get_settings

    class _Store:
        def semantic_search(self, query_embedding, top_k=5):
            return semantic

        def keyword_search(self, query_text, top_k=5):
            return keyword

        def resolve_parent_content(self, parent_id):
            return None

    class _Embedding:
        async def embed_query(self, query):
            return [0.0] * 4

    config = build_session_config(get_settings())
    config["retrieval_method"] = RetrievalMethod.HYBRID_SEARCH
    service = RetrievalService(vector_store=_Store(), embedding_service=_Embedding())

    with pytest.raises(TypeError, match="检索"):
        asyncio.run(service.retrieve("查询", config))


# ── 回归：`retrieve` 对 top_k／候选倍率的形态要求 ──
#
# `retrieve` 直接算 `max(1, session_config.top_k) * max(1, ...)`，而
# `SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫（见
# `_require_finite_number` 的 docstring）。嵌入方、或携带超大整数的 pickle
# 快照，可以把这两个字段直写成任何形态，此时 `max()` 抛裸
# `TypeError: '>' not supported between instances of 'NoneType' and 'int'`。
#
# 这是**生产入口**：`Chatbot.chat_async` 每次提问都调它，而异常落在
# `_require_document_list` 等本文件既有的错误分类之外——`retrieve` 的契约是
# 返回文档列表或给出可读错误，不是把比较运算的 `TypeError` 漏给调用方。
# 同文件的 `_coerce_rank`／`_require_finite_number` 早已按同一判据收敛兄弟字段。


@pytest.mark.parametrize(
    "bad_value",
    [None, "3", [3], {"a": 1}, 10**4301, float("nan"), float("inf")],
    ids=["None", "str", "list", "dict", "huge-int", "nan", "inf"],
)
@pytest.mark.parametrize("field_name", ["top_k", "retrieval_candidate_multiplier"])
def test_retrieve_rejects_malformed_top_k_fields(field_name, bad_value):
    """`top_k`／候选倍率形态非法时必须给 ``ValueError``，不是裸 ``TypeError``。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return []

    service, config = _retrieval_service_with_store(_Store())
    object.__setattr__(config, field_name, bad_value)

    with pytest.raises(ValueError, match=field_name):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize(
    "bad_value", [-5, 0, -(10**4301)], ids=["negative", "zero", "huge-negative"]
)
@pytest.mark.parametrize("field_name", ["top_k", "retrieval_candidate_multiplier"])
def test_retrieve_clamps_non_positive_scale_fields(field_name, bad_value):
    """非正数沿用既有的 ``max(1, ...)`` 语义：收敛成 1，而不是报错。

    这不是放宽：原实现就是 ``max(1, session_config.top_k)``，负数与 0 一直被
    收敛成 1。收敛语义必须保持不变，否则「直接写属性」这条路径会从「静默取
    一条」变成「抛错」，是行为变更而不是修缺陷。超长负整数同理——它的绝对值
    再大也不影响 ``max(1, ...)`` 的结果。
    """

    class _Store:
        def semantic_search(self, _query_embedding, top_k):
            # store 拿到的是 ``top_k × multiplier`` 的乘积：被写的那个字段收敛
            # 成 1，另一个保持默认值（``top_k=5``、``multiplier=3``），因此乘积
            # 是 5 或 3，取决于被写的是哪一个。
            assert top_k == (3 if field_name == "top_k" else 5), top_k
            return []

    service, config = _retrieval_service_with_store(_Store())
    object.__setattr__(config, field_name, bad_value)

    assert asyncio.run(service.retrieve("查询", config)) == []


def test_retrieve_accepts_legal_top_k_fields():
    """收紧不能误伤：合法取值照常检索。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    config["top_k"] = 5
    config["retrieval_candidate_multiplier"] = 3

    results = asyncio.run(service.retrieve("查询", config))

    assert [doc["page_content"] for doc in results] == ["内容"]


# ── 回归：`retrieve` 对 score_threshold／rerank_enabled／融合策略的形态要求 ──
#
# 同一类缺陷的**其余字段**：`retrieve` 还直接读
#   * `session_config.score_threshold`——`coerce_score(...) >= threshold`，非数值
#     抛裸 `TypeError: '>=' not supported between instances of 'float' and 'NoneType'`；
#   * `session_config.rerank_enabled`——`if not ... or not ...`，真值语义掩盖形态；
#   * `session_config.hybrid_fusion_strategy`——与 `"rrf"` 比较、并传给
#     `HybridReranker`（后者自己会拒绝非字符串，但 `_should_apply_score_threshold`
#     先按它判分支）。
# 直写属性是既有前提（见 `_require_finite_number` 的 docstring）。上面那组用例只
# 覆盖了 `top_k`／候选倍率两个字段，同一函数里的其余字段没有覆盖——同类失败不该
# 只修一部分。


@pytest.mark.parametrize(
    "bad_value",
    [None, "0.5", [0.5], {"a": 0.5}, 10**4301, float("nan"), float("inf")],
    ids=["None", "str", "list", "dict", "huge-int", "nan", "inf"],
)
def test_retrieve_rejects_malformed_score_threshold(bad_value):
    """阈值形态非法时必须给 ``ValueError``，不是裸 ``TypeError``。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    object.__setattr__(config, "score_threshold", bad_value)

    with pytest.raises(ValueError, match="score_threshold"):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize(
    "bad_value", [None, "yes", 1, 0, [], {}], ids=["None", "str", "int", "zero", "list", "dict"]
)
def test_retrieve_rejects_malformed_rerank_enabled(bad_value):
    """``rerank_enabled`` 必须是真正的布尔值，不能靠真值语义掩盖形态。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    object.__setattr__(config, "rerank_enabled", bad_value)

    with pytest.raises(ValueError, match="rerank_enabled"):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize(
    "bad_value", [None, 1, ["rrf"], {"rrf": 1}], ids=["None", "int", "list", "dict"]
)
def test_retrieve_rejects_malformed_fusion_strategy(bad_value):
    """融合策略形态非法时必须给带字段名的 ``ValueError``。"""
    from src.utils.config import RetrievalMethod

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return []

        def keyword_search(self, _query_text, _top_k):
            return []

    service, config = _retrieval_service_with_store(_Store())
    config["retrieval_method"] = RetrievalMethod.HYBRID_SEARCH
    object.__setattr__(config, "hybrid_fusion_strategy", bad_value)

    with pytest.raises(ValueError, match="hybrid_fusion_strategy"):
        asyncio.run(service.retrieve("查询", config))


# ── 回归：rerank provider key 的形态校验 ──
#
# `_get_rerank_provider` 直接写 `configurations.get(provider_key)`，而
# `provider_key` 来自可直写的 `session_config.active_rerank_configuration`。
# 不可哈希的取值在 `.get()` 处就抛裸 `TypeError: unhashable type: 'list'`——
# 早于 `ModelProviderFactory` 的守卫，因此工厂那一侧的修复覆盖不到这条路径。
# 这是 `retrieve` 的 rerank 分支，`rerank_enabled=True` 时每次检索都会走到。


@pytest.mark.parametrize("bad_key", [[1], {1: 2}, {1, 2}], ids=["list", "dict", "set"])
def test_rerank_provider_lookup_rejects_unhashable_key(bad_key):
    """不可哈希的 rerank key 必须给 ``ValueError``，不是裸 ``TypeError``。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "active_rerank_configuration", bad_key)

    with pytest.raises(ValueError, match="active_rerank_configuration"):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize(
    "bad_configurations",
    [5, [1], "x", True, 10**4301],
    ids=["int", "list", "str", "bool", "huge-int"],
)
def test_rerank_provider_lookup_rejects_non_mapping_configurations(bad_configurations):
    """`rerank_configurations` 非映射时必须给 ``ValueError``，不是裸 ``AttributeError``。

    `_get_rerank_provider` 直接对它调 `.get()`，而该字段来自可直写的
    `session_config.rerank_configurations`。`AttributeError` 落在 `retrieve`
    既有的错误分类之外，报错现场离「配置项形态错了」很远。
    """

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", bad_configurations)

    with pytest.raises(ValueError, match="rerank_configurations"):
        asyncio.run(service.retrieve("查询", config))


# ── 回归：rerank 配置**取值**的形态校验 ──
#
# `_get_rerank_provider` 取 `configurations[key]` 后：
#   * `getattr(detail, "provider", "")` 只是有默认值，**不校验类型**——该值会进
#     `cache_key` 元组，非可哈希形态在 `self._rerank_providers.get(cache_key)`
#     处抛裸 `TypeError: unhashable type: 'list'`；
#   * detail 本身不是 `ModelDetail`（dict／None／int）时，
#     `ModelProviderFactory.get_rerank_provider` 里的 `config.provider` 抛裸
#     `AttributeError`。
# 两者来源都是可直写的 `session_config.rerank_configurations`（映射的**值**），
# 而 `rerank_enabled=True` 时每次检索都会走到。
@pytest.mark.parametrize("bad_provider", [["p"], {"p": 1}, {1, 2}], ids=["list", "dict", "set"])
def test_rerank_provider_lookup_rejects_unhashable_provider_field(bad_provider):
    """`detail.provider` 不可哈希时必须给 ``ValueError``，不是裸 ``TypeError``。

    `ModelDetail` 是 pydantic 模型，**构造**时会拒绝非字符串的 provider，但
    普通属性没有赋值守卫（`validate_assignment` 未开）——实测
    `detail.provider = ["p"]` 直接成功。因此这条路径是可达的。
    """
    from src.utils.config import ModelDetail

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    detail = ModelDetail(provider="siliconflow", model_name="m")
    detail.provider = bad_provider

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", {"siliconflow": detail})

    with pytest.raises(ValueError, match="provider"):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize(
    "bad_detail", [{"provider": "siliconflow"}, None, 5, "x"], ids=["dict", "None", "int", "str"]
)
def test_rerank_provider_lookup_rejects_non_detail_value(bad_detail):
    """配置值不是 `ModelDetail` 时必须给 ``ValueError``，不是裸 ``AttributeError``。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", {"siliconflow": bad_detail})

    with pytest.raises(ValueError, match="rerank_configurations"):
        asyncio.run(service.retrieve("查询", config))


@pytest.mark.parametrize("bad_model_name", [["m"], {"m": 1}, {1, 2}], ids=["list", "dict", "set"])
def test_rerank_provider_lookup_rejects_unhashable_model_name(bad_model_name):
    """`detail.model_name` 不可哈希时必须给 ``ValueError``，不是裸 ``TypeError``。

    它与 `provider` 同进 `cache_key` 元组，同样只有 `getattr` 的默认值、没有
    类型判据——实测 `detail.model_name = ["m"]` 直接成功（`ModelDetail` 未开
    `validate_assignment`）。
    """
    from src.utils.config import ModelDetail

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    detail = ModelDetail(provider="siliconflow", model_name="m")
    detail.model_name = bad_model_name

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", {"siliconflow": detail})

    with pytest.raises(ValueError, match="model_name"):
        asyncio.run(service.retrieve("查询", config))


def test_rerank_provider_lookup_rejects_deleted_provider_attribute():
    """`del detail.provider` 后必须给 ``ValueError``。

    `getattr(detail, "provider", "")` 只兜「属性缺失」，兜出来的空串会把「属性
    没了」伪装成「provider 名为空」，一路带到工厂再抛裸 ``AttributeError``。
    """
    from src.utils.config import ModelDetail

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    detail = ModelDetail(provider="siliconflow", model_name="m")
    del detail.provider

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", {"siliconflow": detail})

    with pytest.raises(ValueError, match="provider"):
        asyncio.run(service.retrieve("查询", config))


def test_rerank_provider_lookup_still_reports_missing_key():
    """收紧不能误伤：key 真的不存在时保持原有「未找到 key」的 ``ValueError``。"""

    class _Store:
        def semantic_search(self, _query_embedding, _top_k):
            return [{"page_content": "内容", "metadata": {"source": "a.md"}, "score": 0.9}]

    service, config = _retrieval_service_with_store(_Store())
    config["rerank_enabled"] = True
    object.__setattr__(config, "rerank_configurations", {})

    with pytest.raises(ValueError, match="未找到key"):
        asyncio.run(service.retrieve("查询", config))
