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
