"""``SessionConfig`` 是运行期可变的独立配置入口，必须自带字段校验。

``Settings`` 只在启动时校验一次；UI（``src/ui/config_menu.py``）与库调用方拿到的是
``SessionConfig``，它们经 ``__setitem__`` 写入的值此前完全不校验。修复前实测
``chat_config["top_k"] = 10**9`` 能一路走到 ``faiss_index.search()``：FAISS 既不报错
也不截断，按 10 亿条分配（实测 10**6 时就已分配满槽位并耗时 1.56s）。
"""

import pytest

from src.runtime.contracts import SessionConfig, build_session_config
from src.utils.config import (
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    RetrievalMethod,
    Settings,
)


@pytest.fixture
def session_config() -> SessionConfig:
    return build_session_config(Settings(_env_file=None))


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("top_k", 10**9),
        ("top_k", 0),
        ("top_k", -1),
        ("top_k", 99999999999999999999),
        ("top_k", "5"),
        ("top_k", True),
        ("top_k", 5.5),
        ("retrieval_candidate_multiplier", 10**6),
        ("retrieval_candidate_multiplier", 0),
        ("vector_weight", -5),
        ("vector_weight", 1.5),
        ("vector_weight", float("nan")),
        ("keyword_weight", float("inf")),
        ("score_threshold", -999),
        ("score_threshold", 2.0),
        ("retrieval_method", "totally-invalid"),
        ("hybrid_fusion_strategy", "bogus"),
        ("rerank_enabled", "yes"),
        ("chat_temperature", -1.0),
        ("chat_temperature", float("nan")),
    ],
)
def test_session_config_rejects_illegal_values(session_config, field_name, bad_value):
    """各字段的非法值必须在写入时拒绝，不能拖到检索期。"""
    with pytest.raises(ValueError, match=field_name):
        session_config[field_name] = bad_value


def test_session_config_rejects_unknown_field(session_config):
    """拼错字段名应报错，而不是静默新增一个属性。"""
    with pytest.raises(KeyError, match="没有字段"):
        session_config["不存在的字段"] = 1


@pytest.mark.parametrize(
    "field_name,good_value,expected",
    [
        ("top_k", 20, 20),
        ("top_k", _MAX_RETRIEVAL_TOP_K, _MAX_RETRIEVAL_TOP_K),
        ("retrieval_candidate_multiplier", _MAX_RETRIEVAL_MULTIPLIER, _MAX_RETRIEVAL_MULTIPLIER),
        ("vector_weight", 0.3, 0.3),
        ("score_threshold", 0.0, 0.0),
        ("score_threshold", 1.0, 1.0),
    ],
)
def test_session_config_accepts_legal_boundary_values(
    session_config, field_name, good_value, expected
):
    """上界与下界本身必须可用——收紧不能把合法边界一起拒掉。"""
    session_config[field_name] = good_value
    assert session_config[field_name] == expected


def test_session_config_normalizes_retrieval_method_string(session_config):
    """UI 传的是枚举值的中文 value，应被归一化成枚举成员。"""
    session_config["retrieval_method"] = "全文检索"
    assert session_config["retrieval_method"] is RetrievalMethod.FULL_TEXT_SEARCH

    session_config["retrieval_method"] = RetrievalMethod.HYBRID_SEARCH
    assert session_config["retrieval_method"] is RetrievalMethod.HYBRID_SEARCH


def test_session_config_validator_registry_covers_scalar_fields():
    """注册表必须覆盖所有标量字段。

    漏登记会让该字段回落到「原样写入」，也就是本次修复前的无校验状态——
    这类静默回归不会让任何行为测试变红，只能靠结构断言盯住。
    """
    scalar_fields = {
        "retrieval_method",
        "vector_weight",
        "keyword_weight",
        "hybrid_fusion_strategy",
        "retrieval_candidate_multiplier",
        "rerank_enabled",
        "top_k",
        "score_threshold",
        "chat_temperature",
    }
    registered = set(SessionConfig._VALIDATORS)
    assert scalar_fields <= registered, f"以下标量字段没有登记校验器: {scalar_fields - registered}"
    # 反方向：注册表里的字段必须真实存在，否则是拼错名字的死登记
    known_fields = set(build_session_config(Settings(_env_file=None)).to_dict())
    assert registered <= known_fields, f"注册表里有不存在的字段: {registered - known_fields}"


def test_session_config_to_dict_still_reflects_writes(session_config):
    """写入后 ``to_dict`` 必须能看到新值（否则 UI 显示会与实际不符）。"""
    session_config["top_k"] = 42
    assert session_config.to_dict()["top_k"] == 42
