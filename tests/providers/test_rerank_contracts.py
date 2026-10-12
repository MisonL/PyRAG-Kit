from types import SimpleNamespace

import pytest

from src.providers.jina import JinaProvider
from src.providers.siliconflow_rerank import SiliconflowRerankProvider


def test_jina_duplicate_documents_keep_distinct_indices():
    provider = object.__new__(JinaProvider)
    indices, scores = provider._parse_response(
        [
            {"document": {"text": "same"}, "relevance_score": 0.9},
            {"document": {"text": "same"}, "relevance_score": 0.8},
        ],
        ["same", "same"],
    )
    assert indices == [0, 1]
    assert scores == [0.9, 0.8]


def test_siliconflow_duplicate_documents_keep_distinct_indices():
    provider = object.__new__(SiliconflowRerankProvider)
    indices, scores = provider._parse_response(
        [
            {"document": {"text": "same"}, "relevance_score": 0.9},
            {"document": {"text": "same"}, "relevance_score": 0.8},
        ],
        ["same", "same"],
    )
    assert indices == [0, 1]
    assert scores == [0.9, 0.8]


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_accepts_top_n_larger_than_document_count(provider_type):
    """上游 rerank API 自行截断，top_n 超过文档数属合法输入。"""
    provider = object.__new__(provider_type)

    assert provider._validate_inputs("query", ["document"], 2) is None


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
@pytest.mark.parametrize("top_n", [0, -1, True, 1.5, "1", None])
def test_rerank_rejects_invalid_top_n(provider_type, top_n):
    """top_n 必须是 >= 1 的整数；bool 是 int 的子类，须显式排除。"""
    provider = object.__new__(provider_type)

    with pytest.raises(ValueError, match="top_n"):
        provider._validate_inputs("query", ["document"], top_n)


# `_validate_inputs` 有四个判据：query 非空字符串、documents 非空列表、
# documents 元素全为字符串、top_n 范围。上面两条只钉住了最后一条——前三条的
# 用例要么喂合法输入（`"query"`／`["document"]`），要么根本不经过这里，于是
# 把前三道 `raise` 换成 `pass` 后全量套件仍然全绿。删除后行为确实改变（实测）：
# `query=5`／`"  "`、`documents=[]`、`documents=["a", 5]` 全部**静默接受**并原样
# 进入请求体发给上游；`documents=5` 从带指引的 `ValueError` 变成裸
# `TypeError: 'int' object is not iterable`（`any(...)` 迭代失败）。
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
@pytest.mark.parametrize("query", [5, None, "  ", ""], ids=["int", "none", "blank", "empty"])
def test_rerank_rejects_invalid_query(provider_type, query):
    """query 必须是非空字符串，不能静默放行非字符串或空白串。"""
    provider = object.__new__(provider_type)

    with pytest.raises(ValueError, match="query 必须是非空字符串"):
        provider._validate_inputs(query, ["document"], 1)


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
@pytest.mark.parametrize("documents", [5, None, [], "abc"], ids=["int", "none", "empty", "str"])
def test_rerank_rejects_invalid_documents_container(provider_type, documents):
    """documents 必须是非空列表；非列表取值要给 ``ValueError`` 而不是裸 ``TypeError``。"""
    provider = object.__new__(provider_type)

    with pytest.raises(ValueError, match="documents 必须是非空列表"):
        provider._validate_inputs("query", documents, 1)


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
@pytest.mark.parametrize("documents", [["a", 5], [None], ["ok", {"t": 1}]])
def test_rerank_rejects_non_string_document_entries(provider_type, documents):
    """documents 的元素必须全是字符串，否则会在请求体里被静默序列化。"""
    provider = object.__new__(provider_type)

    with pytest.raises(ValueError, match="documents 必须全部是字符串"):
        provider._validate_inputs("query", documents, 1)


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_validate_inputs_accepts_well_formed_arguments(provider_type):
    """收紧不能误伤：合法的 query／documents／top_n 照常通过。"""
    provider = object.__new__(provider_type)

    assert provider._validate_inputs("query", ["document"], 1) is None


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_invalid_index_is_explicit(provider_type):
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="index 无效"):
        provider._parse_response(
            [{"index": 4, "relevance_score": 0.9}],
            ["one", "two"],
        )


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_accepts_positive_index_within_range(provider_type):
    """区间内的**正**下标必须被接受并映射到对应文档。

    范围判据 ``not 0 <= direct_index < len(documents)`` 是链式比较，``<=`` 反向
    成 ``>=`` 后整条判据变成「index > 0 或 index >= len 即拒绝」——除 0 以外的
    合法下标全被拒（实测 index=1 报「index 无效」），而**负**下标反而被放行，
    落到 ``positions[...].remove(-1)`` 抛裸 ``ValueError``，穿透本方法承诺的
    ``RuntimeError``。既有用例只喂过 index=0 与越界的 index=4，因此这一半从未
    被触达。
    """
    provider = object.__new__(provider_type)

    indices, scores = provider._parse_response(
        [{"index": 1, "relevance_score": 0.9}],
        ["one", "two"],
    )

    assert indices == [1]
    assert scores == [0.9]


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_rejects_negative_index_with_runtime_error(provider_type):
    """负下标必须报 ``index 无效``，不得穿透成 ``list.remove`` 的裸 ``ValueError``。"""
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="index 无效"):
        provider._parse_response(
            [{"index": -1, "relevance_score": 0.9}],
            ["one", "two"],
        )


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_unknown_document_is_explicit(provider_type):
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="无法映射"):
        provider._parse_response(
            [{"document": {"text": "unknown"}, "relevance_score": 0.9}],
            ["one", "two"],
        )


@pytest.mark.parametrize("score", ["not-a-number", float("nan"), float("inf")])
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_invalid_score_is_explicit(provider_type, score):
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="relevance_score"):
        provider._parse_response(
            [{"index": 0, "relevance_score": score}],
            ["one"],
        )


@pytest.mark.parametrize("score", [True, False], ids=["true", "false"])
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_bool_score_is_not_reported_as_out_of_range(provider_type, score):
    """``relevance_score`` 为 bool 时不得报「超出双精度范围」。

    超范围判据是 ``isinstance(raw_score, int) and not isinstance(raw_score, bool)
    and not is_double_convertible(raw_score)``：中间那个 ``not isinstance(...,
    bool)`` 是**有意**排除 bool 的（bool 是 int 子类，``float(True) == 1.0`` 会
    静默变 1.0，走的不是「超范围」这条）。把它前面的 ``and`` 互换为 ``or`` 后，
    ``raw_score=True`` 会命中 ``(isinstance(int)=True or ...) and not
    is_double_convertible(True)=True`` → 误报「超出双精度范围」。既有
    ``test_rerank_invalid_score_is_explicit`` 只喂字符串／nan／inf（都带
    ``relevance_score`` 键走 float() 路径），``test_rerank_response_score_beyond
    _double_range_is_explicit`` 只喂超大整数（两种变体都 raise，区分不了），
     因此 bool 这条独立分支从未被覆盖。
    """
    provider = object.__new__(provider_type)

    indices, scores = provider._parse_response(
        [{"index": 0, "relevance_score": score}],
        ["one"],
    )

    assert indices == [0]
    assert scores == [1.0 if score else 0.0]


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1])
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_non_finite_or_non_positive_timeout_is_explicit(provider_type, timeout):
    provider = object.__new__(provider_type)
    provider._options = {"timeout": timeout}

    with pytest.raises(ValueError, match="timeout.*正数"):
        provider._request_timeout()


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_duplicate_index_is_explicit(provider_type):
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="重复 index"):
        provider._parse_response(
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            ["one"],
        )


def test_siliconflow_rerank_rejects_return_documents_override():
    provider = object.__new__(SiliconflowRerankProvider)
    provider._model_name = "rerank-model"
    provider._options = {"return_documents": False}

    with pytest.raises(ValueError, match="return_documents"):
        provider._prepare_payload("query", ["document"], 1)


def test_jina_rerank_rejects_return_documents_override():
    """Jina 与 SiliconFlow 一样把 return_documents 固定为 True，不允许覆盖。"""
    provider = object.__new__(JinaProvider)
    provider._model_name = "rerank-model"
    provider._options = {"return_documents": False}

    with pytest.raises(ValueError, match="return_documents"):
        provider._prepare_payload("query", ["document"], 1)


def test_jina_rerank_requests_documents_by_default():
    """未显式配置时也要请求返回文档，与 SiliconFlow 行为一致。"""
    provider = object.__new__(JinaProvider)
    provider._model_name = "rerank-model"
    provider._options = {}

    payload = provider._prepare_payload("query", ["document"], 1)

    assert payload["return_documents"] is True


def test_siliconflow_rerank_requests_documents_by_default():
    """SiliconFlow 侧的 sibling：默认请求也必须带 ``return_documents=True``。

    反向成 ``False`` 后响应里没有文档文本，调用方按「返回文档」写的
    ``_parse_response`` 拿到空内容——而 Jina 侧早有这条默认值断言，
    SiliconFlow 侧只断言了「不允许覆盖」，默认值本身无人看守。
    """
    provider = object.__new__(SiliconflowRerankProvider)
    provider._model_name = "rerank-model"
    provider._options = {}

    payload = provider._prepare_payload("query", ["document"], 1)

    assert payload["return_documents"] is True


# ---------------------------------------------------------------------------
# 超出双精度范围的整数
# ---------------------------------------------------------------------------
#
# ``math.isfinite(x)`` 与 ``float(x)`` 对**任意大的 Python int** 都会先尝试
# 转成 C double，超出 ``DBL_MAX``（约 1.8e308）时抛
# ``OverflowError: int too large to convert to float``。
#
# 这不是学术边界：``config.toml`` 的整数是**无上界**的 TOML 整数，``tomllib``
# 原样解析成 Python int；``options`` 又是文档指定的扩展入口。实测
# ``[rerank_configurations.x.options] timeout = <401 位十进制整数>`` 让
# ``JinaProvider._request_timeout`` 抛裸 ``OverflowError``。
#
# 后果比「报错难看」严重：这些方法承诺抛 ``ValueError``，而调用方
# （``retrieval_service`` 的校验链、``get_settings`` 的 ``except ValueError``
# 兜底、``questionary`` 的 validator 契约）都只接 ``ValueError``。
# ``OverflowError`` 是 ``ArithmeticError`` 的子类，**不是** ``ValueError``，
# 于是它穿透所有既有守卫，冒到解释器默认 handler。
#
# 参数 id 必须显式给出：``10**4301`` 超过 ``sys.get_int_max_str_digits()``，
# pytest 为它自动生成 id 时调 ``str(val)`` 会自己抛 ``ValueError``，测试在
# **收集阶段**就崩掉。
_BEYOND_DOUBLE_INTS = [
    pytest.param(10**400, id="400digits"),
    pytest.param(-(10**400), id="negative-400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
]


@pytest.mark.parametrize("timeout", _BEYOND_DOUBLE_INTS)
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_timeout_beyond_double_range_is_explicit(provider_type, timeout):
    """超出双精度范围的 ``timeout`` 必须报成「必须是正数」，而不是 ``OverflowError``。"""
    provider = object.__new__(provider_type)
    provider._options = {"timeout": timeout}

    with pytest.raises(ValueError, match="timeout.*正数"):
        provider._request_timeout()


@pytest.mark.parametrize("score", _BEYOND_DOUBLE_INTS)
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_response_score_beyond_double_range_is_explicit(provider_type, score):
    """响应里的超大整数得分必须报成 ``RuntimeError``，而不是 ``OverflowError``。

    ``_parse_response`` 的契约是「形态异常一律 ``RuntimeError``」，原实现的
    ``except (KeyError, TypeError, ValueError)`` 接不住 ``OverflowError``。
    """
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="relevance_score"):
        provider._parse_response([{"index": 0, "relevance_score": score}], ["one"])


# ``_parse_response`` 的契约是「形态异常一律 ``RuntimeError``」。原实现在下面
# 这个位置上用了 ``res["relevance_score"]``——它**不在**那个
# ``except (KeyError, TypeError, ValueError)`` 的覆盖范围内（那段只包
# ``float(...)``），于是缺字段的响应项让契约漏成裸
# ``KeyError: 'relevance_score'``。调用方的错误分类与日志脱敏都按
# ``RuntimeError`` 写，接不住 ``KeyError``。
#
# 上面的 ``test_rerank_invalid_score_is_explicit`` 覆盖不到这条：它的探针
# （``"not-a-number"``／``nan``／``inf``）**都带** ``relevance_score`` 键，
# 走的正是 ``float()`` 那条被 ``try`` 包住的路径。缺字段是另一条分支。
@pytest.mark.parametrize(
    "item",
    [
        {"index": 0},
        {"index": 0, "document": {"text": "one"}},
        {"document": {"text": "one"}},
    ],
    ids=["index-only", "index-and-document", "document-only"],
)
@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_missing_score_is_explicit(provider_type, item):
    """缺少 ``relevance_score`` 的结果项必须报成 ``RuntimeError``。

    响应来自网络 JSON，任何服务器或网关都可能省略这个字段——实测修复前抛的是
    裸 ``KeyError``，而本方法的文档承诺是 ``RuntimeError``。
    """
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="relevance_score"):
        provider._parse_response([item], ["one"])


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_parse_response_never_leaks_a_bare_key_error(provider_type):
    """推广：任意畸形响应都不得漏出 ``KeyError``。

    上面那条钉住已知的那个字段；这条用一批形态各异的条目断言「绝不漏出
    ``KeyError``」——那类异常只可能来自本函数自己的字典下标，不可能来自契约。
    """
    provider = object.__new__(provider_type)
    probes = [
        [{}],
        [{"index": 0}],
        [{"document": {"text": "one"}}],
        [{"index": 0, "document": {"text": "one"}}],
        [{"index": 1}],
        [{"document": "one"}],
        [{"relevance_score": 0.5}],
    ]

    for probe in probes:
        try:
            provider._parse_response(probe, ["one"])
        except RuntimeError:
            pass
        except BaseException as exc:
            raise AssertionError(f"{probe!r} 抛出了 {type(exc).__name__}: {exc}") from exc


# ── 响应解析的形态判据 ──
#
# ``_parse_response`` 的四条形态判据（results 非列表、结果项非 Mapping、
# index/document 关联、document 不可映射）在整套测试里从未执行：既有用例
# ``test_rerank_invalid_index_is_explicit``／``test_rerank_unknown_document_is_explicit``
# 命中的都是「index 值越界」与「document 映射不到输入」两条**取值**判据。
# 行覆盖证实四条从未执行；把它们的 ``raise`` 换成 ``pass`` 后全量 3341 条仍全绿。
#
# ``rerank``／``arerank`` 里「响应必须是 JSON 对象」那两条同样是缺口：
# 它们在 ``payload.get("results")`` 之前，非 Mapping 取值会先撞裸
# ``AttributeError: 'list' object has no attribute 'get'``。删掉后实测正是如此。


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    payload: object = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse(self.payload)


class _FakeAsyncClient(_FakeClient):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse(self.payload)


class _FakeHttpx:
    Client = _FakeClient
    AsyncClient = _FakeAsyncClient


def _fake_rerank_providers(provider_type):
    """构造绕过网络与配置读取的 provider 实例。"""
    provider = object.__new__(provider_type)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "https://example.invalid/v1/rerank"
    provider._api_key = "k"
    return [provider]


def _fake_rerank_http(monkeypatch, payload):
    """把两个 provider 内部 ``import httpx`` 拿到的客户端换成假实现。"""
    import sys

    _FakeClient.payload = payload
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx())


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
@pytest.mark.parametrize("bad_item", [5, "text", ["x"]], ids=["int", "str", "list"])
def test_rerank_rejects_non_mapping_result_item(provider_type, bad_item):
    """结果项必须是 Mapping。

    删除守卫后 ``"index" in res`` 对整数取值抛裸 ``TypeError: argument of type
    'int' is not iterable``——而本模块承诺的解析契约是 ``RuntimeError``，
    调用方的错误分类与日志脱敏都按 ``RuntimeError`` 写，接不住 ``TypeError``。
    """
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="包含无效结果项"):
        provider._parse_response([bad_item], ["one"])


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_rejects_index_document_mismatch(provider_type):
    """响应声明的 ``index`` 与随附的 ``document`` 文本必须指向同一个输入文档。

    上游返回的 index 与 document 不一致时，删除守卫会**静默接受**：实测返回
    ``([0], [1.0])``，把「命中的是另一篇文档」当成有效排序交给检索层。
    """
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="index 与 document 不匹配"):
        provider._parse_response(
            [{"index": 0, "document": {"text": "other"}, "relevance_score": 0.9}],
            ["one", "other"],
        )


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_rejects_non_mapping_result_entries(provider_type):
    """``_parse_response`` 声明 ``results`` 是列表，传入非列表必须显式报错。

    这条与 ``rerank``／``arerank`` 里那条同消息判据构成前后两道闸；删除后本
    函数会在 ``for res in results`` 处以形态各异的异常崩掉。
    """
    provider = object.__new__(provider_type)

    with pytest.raises(RuntimeError, match="缺少有效的 results 列表"):
        provider._parse_response("not-a-list", ["one"])


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_response_must_be_json_object(provider_type, monkeypatch):
    """上游响应必须是 JSON 对象：同步与异步两条入口都要在 ``.get`` 之前报错。

    走公开 ``rerank``／``arerank`` 入口并用假 ``httpx`` 拦网，验证删除守卫后
    收场从 ``RuntimeError`` 退化成裸 ``AttributeError``。
    """
    import asyncio

    monkeypatch.setattr(
        "src.providers.jina.get_settings",
        lambda: SimpleNamespace(jina_api_key="k"),
    )
    monkeypatch.setattr(
        "src.providers.siliconflow_rerank.get_settings",
        lambda: SimpleNamespace(siliconflow_api_key="k", siliconflow_base_url="https://x/v1"),
    )

    for provider in _fake_rerank_providers(provider_type):
        _fake_rerank_http(monkeypatch, ["not", "a", "map"])
        for run in (
            lambda p: p.rerank(query="q", documents=["a", "b"], top_n=2),
            lambda p: asyncio.run(p.arerank(query="q", documents=["a", "b"], top_n=2)),
        ):
            with pytest.raises(RuntimeError, match="响应必须是 JSON 对象"):
                run(provider)


# ── 构造期与 payload 构造的覆盖缺口（Jina / SiliconFlow 兄弟对） ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3698 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 61／51 ``options`` 含未知字段 → **静默构造**出带未知字段的 provider，
#   非法配置一路带进请求体；
# * 66／58 缺 ``JINA_API_KEY``／``SILICONFLOW_API_KEY`` → **静默构造**出无
#   凭证的 provider，请求发出时才 401；
# * 61（SiliconFlow）缺 ``SILICONFLOW_BASE_URL`` → **静默构造**，``_base_url``
#   落成 ``None``；
# * 96／90 ``options`` 与 ``options.extra_body`` 重复 → **静默产出** payload，
#   其中一个取值被无声丢弃（``truncate`` 被 extra_body 覆盖）；
# * 101／96 ``options.extra_body`` 覆盖请求字段 → **静默产出**
#   ``payload["model"]="hijack"``，把请求的模型整个换掉。
#
# 构造期判据必须经真实 ``__init__`` 才能覆盖，而 ``tests/providers/test_factory.py``
# 会从 ``sys.modules`` 删除这两个模块再重新导入——字符串形式的 ``monkeypatch``
# 会打到**新**模块，本文件顶层绑定的仍是**旧**类。因此与
# ``test_sdk_capabilities.py`` 的 anthropic 助手同法：把 ``get_settings`` 打到
# 类自己的 ``__init__.__globals__`` 上。
def _rerank_settings_patch(provider_type, monkeypatch, **settings):
    monkeypatch.setitem(
        provider_type.__init__.__globals__, "get_settings", lambda: SimpleNamespace(**settings)
    )


_RERANK_SETTINGS = {
    JinaProvider: {"jina_api_key": "k"},
    SiliconflowRerankProvider: {"siliconflow_api_key": "k", "siliconflow_base_url": "https://x/v1"},
}


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_constructor_rejects_unknown_option_field(provider_type, monkeypatch):
    """``options`` 里的未知字段必须在构造期拒绝，不能静默带进请求体。"""
    _rerank_settings_patch(provider_type, monkeypatch, **_RERANK_SETTINGS[provider_type])

    with pytest.raises(ValueError, match="包含不支持的字段: bogus_field"):
        provider_type("m", {"bogus_field": 1})


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_constructor_requires_api_key(provider_type, monkeypatch):
    """缺 API Key 必须在构造期拒绝，不能静默构造出无凭证的 provider。"""
    settings = dict(_RERANK_SETTINGS[provider_type])
    settings["jina_api_key" if provider_type is JinaProvider else "siliconflow_api_key"] = None
    _rerank_settings_patch(provider_type, monkeypatch, **settings)

    with pytest.raises(ValueError, match="API_KEY is required"):
        provider_type("m")


def test_siliconflow_constructor_requires_base_url(monkeypatch):
    """缺 ``SILICONFLOW_BASE_URL`` 必须在构造期拒绝（Jina 的 base_url 硬编码，无此判据）。"""
    _rerank_settings_patch(
        SiliconflowRerankProvider,
        monkeypatch,
        siliconflow_api_key="k",
        siliconflow_base_url=None,
    )

    with pytest.raises(ValueError, match="SILICONFLOW_BASE_URL is required"):
        SiliconflowRerankProvider("m")


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_rejects_option_duplicated_in_extra_body(provider_type):
    """``options`` 与 ``options.extra_body`` 重复时必须报错，不能静默丢弃其中一个。"""
    provider = object.__new__(provider_type)
    provider._model_name = "m"
    provider._options = {"truncate": True, "extra_body": {"truncate": False}}

    with pytest.raises(ValueError, match="与 options.extra_body 重复: truncate"):
        provider._prepare_payload("q", ["d"], 1)


@pytest.mark.parametrize("provider_type", [JinaProvider, SiliconflowRerankProvider])
def test_rerank_rejects_extra_body_override_of_request_field(provider_type):
    """``options.extra_body`` 覆盖请求字段时必须报错，不能静默换掉 ``model``。"""
    provider = object.__new__(provider_type)
    provider._model_name = "m"
    provider._options = {"extra_body": {"model": "hijack"}}

    with pytest.raises(ValueError, match="extra_body 不允许覆盖请求字段: model"):
        provider._prepare_payload("q", ["d"], 1)


# ── 回归：Rerank 的重试耗尽必须抛**原始异常**（boolc 扫描存活点）──
#
# ``@retry(..., reraise=True)`` 是 jina／siliconflow 两条 rerank 路径共用的
# 守卫（``True`` ↔ ``False`` 互换后全量测试仍全绿）。反向成 ``False`` 后
# tenacity 把最后一次异常包成 ``RetryError`` 抛出——它既不是 ``ValueError``
# 也不是 ``RuntimeError``／``ConnectionError``，调用方按后三者写的错误分类
# 全部接不住：一次「上游不可达」被显示成与错误内容无关的 tenacity 内部类型。
#
# 同文件 ``test_rerank_*`` 都靠 monkeypatch 掉 httpx，重试守卫本身从未执行。


@pytest.mark.parametrize(
    "provider_cls",
    [JinaProvider, SiliconflowRerankProvider],
    ids=["jina", "siliconflow"],
)
def test_rerank_reraise_original_error_after_retries_exhausted(monkeypatch, provider_cls):
    """重试耗尽后必须抛出**原始** ``ConnectionError``，不是 ``RetryError``。"""
    provider = object.__new__(provider_cls)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "http://example.invalid"
    provider._api_key = "k"
    provider._get_headers = dict
    provider._request_timeout = lambda: 1.0

    class _BoomClient:
        def __init__(self, *_args, **_kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            raise ConnectionError("上游不可达")

    monkeypatch.setattr("httpx.Client", _BoomClient)
    # 不真的等 2s + 4s 退避：直接掏掉 wait 调用（tenacity 的 sleep 是实例属性）。
    monkeypatch.setattr(provider_cls.rerank.retry, "sleep", lambda _seconds: None)

    with pytest.raises(ConnectionError):
        provider.rerank("q", ["d1"], 1)


@pytest.mark.parametrize(
    "provider_cls",
    [JinaProvider, SiliconflowRerankProvider],
    ids=["jina", "siliconflow"],
)
def test_arerank_reraise_original_error_after_retries_exhausted(monkeypatch, provider_cls):
    """异步入口是**另一处装饰器**（同文件 256／248），守卫必须逐个看守。"""
    import asyncio

    provider = object.__new__(provider_cls)
    provider._model_name = "m"
    provider._options = {}
    provider._base_url = "http://example.invalid"
    provider._api_key = "k"
    provider._get_headers = dict
    provider._request_timeout = lambda: 1.0

    class _BoomAsyncClient:
        def __init__(self, *_args, **_kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            raise ConnectionError("上游不可达")

    monkeypatch.setattr("httpx.AsyncClient", _BoomAsyncClient)

    # 异步重试的 ``sleep`` 是被 ``await`` 的，必须给协程函数——给同步 lambda
    # 会抛 ``TypeError: object NoneType can't be used in 'await' expression``，
    # 那样测试在**未变异**的代码上也会失败，红验证就成了假绿。
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider_cls.arerank.retry, "sleep", _no_sleep)

    with pytest.raises(ConnectionError):
        asyncio.run(provider.arerank("q", ["d1"], 1))
