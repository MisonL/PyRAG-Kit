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
