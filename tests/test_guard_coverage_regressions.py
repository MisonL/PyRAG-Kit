"""补齐「删守卫后全量测试仍绿」的非 raise 守卫覆盖缺口（工具/服务/检索批）。

与 ``tests/providers/test_guard_coverage_regressions.py`` 同一来源：变异扫描
（把守卫的 ``if`` 整体替换成 ``pass``）后全量 3804 条测试仍绿的存活守卫。
逐条用真实入口确认删除后行为改变，再按守卫存在时的行为写断言。
"""

import asyncio

import numpy as np

from src.providers.local_hash import _normalize as normalize_vector
from src.retrieval.vdb.faiss_store import FaissStore
from src.services.retrieval_service import (
    HybridReranker,
    RetrievalService,
    _document_key,
    _document_metadata,
)
from src.utils.config import Settings, _coerce_number
from src.utils.security import (
    _model_dump_mapping,
    redact_sensitive_text,
    validate_secret_free_resource_args,
    validate_secret_free_resource_kwargs,
)

# --- utils.security --------------------------------------------------------


def test_redact_masked_credential_keeps_key_name_for_followup_redaction():
    """掩码凭证后跟 ``键: 值`` 时，键名要保留、值要被吃掉。

    删除守卫后键名一并被替换成 ``[REDACTED]``，后续的键值脱敏规则失去锚点，
    明文值直接落进日志（实测 ``sk-abc***token: hunter2`` → ``[REDACTED]: hunter2``）。
    """
    redacted = redact_sensitive_text("sk-abc***token: hunter2")
    assert "hunter2" not in redacted
    assert redacted == "[REDACTED]token: [REDACTED]"
    assert redact_sensitive_text("sk-abc***token=hunter2") == "[REDACTED]token=[REDACTED]"


def test_secret_free_validators_accept_none_as_absent():
    """``None`` 表示「没有传参数」，返回空容器。

    删除守卫后 None 落到「必须是对象/元组」的 ``ValueError``，未传参数的
    常规调用形态被拒。
    """
    assert validate_secret_free_resource_kwargs(None, "P") == {}
    assert validate_secret_free_resource_args(None, "P") == ()


def test_model_dump_mapping_rejects_non_mapping_dump_result():
    """``model_dump`` 返回非 Mapping（或自身）时返回 None。

    删除守卫后原样返回列表或对象自身，调用方按 Mapping 处理时行为不可预期
    （本函数契约是「读不到就不检查」的 fail-open，返回 None 是契约的一部分）。
    """

    class ListDump:
        def model_dump(self, **kwargs):
            return ["not", "mapping"]

    class SelfDump:
        def model_dump(self, **kwargs):
            return self

    assert _model_dump_mapping(ListDump()) is None
    assert _model_dump_mapping(SelfDump()) is None


# --- utils.config ----------------------------------------------------------


def test_split_separators_none_falls_back_to_field_default():
    """``kb_splitter_separators=None`` 回退到字段默认值而不是返回 None。

    删除守卫后 None 原样返回，pydantic 在后续校验中拿到 None，默认分片
    分隔符配置失效。
    """
    default = Settings.model_fields["kb_splitter_separators"].default_factory()
    assert Settings.split_separators(None) == default
    assert Settings.split_separators("") == default


def test_coerce_number_returns_int_for_decimal_literals():
    """十进制整数字面量返回 ``int`` 而不是 ``float``。

    删除守卫后 ``"123"`` 走 float 分支返回 ``123.0``——类型变化会影响下游
    以 ``isinstance(x, int)`` 为判据的校验。
    """
    result = _coerce_number("123", "chat_top_k")
    assert result == 123
    assert isinstance(result, int)


# --- retrieval -------------------------------------------------------------


def test_faiss_normalize_parent_document_keeps_string_content():
    """字符串形态的 parent_document 转成 ``{"content": ...}`` 而不是丢弃。

    删除守卫后字符串内容被替换成空串——父块内容静默消失。
    """
    assert FaissStore._normalize_parent_document("正文") == {"content": "正文", "metadata": {}}
    assert FaissStore._normalize_parent_document(None) == {"content": "", "metadata": {}}


def test_hybrid_reranker_normalize_scores_empty_returns_empty():
    """空分数列表返回空列表。

    删除守卫后 ``max([])`` 抛裸 ``ValueError: max() arg is an empty sequence``。
    """
    assert HybridReranker._normalize_scores([]) == []


def test_document_metadata_non_dict_returns_empty():
    """非 dict 文档的元数据返回空 dict。

    删除守卫后 ``"x".get`` 抛裸 ``AttributeError``——检索结果可以来自任何
    VectorStore 实现，展示/合并路径不能因此崩溃。
    """
    assert _document_metadata("x") == {}


def test_document_key_uses_chunk_index_when_present():
    """有 ``chunk_index`` 时键取 ``source:index`` 而不是内容摘要。

    删除守卫后键变成整段内容的 sha256——同一分块的键随内容改写而漂移，
    混合合并会把同一块当成两块。
    """
    document = {"page_content": "abc", "metadata": {"chunk_index": 3, "source": "s"}}
    assert _document_key(document) == "s:3"


def test_clear_rerank_cache_closes_duplicate_provider_once():
    """同一 provider 对象出现在多个缓存槽位时只关闭一次。

    删除守卫后按槽位逐个关闭，同一对象被 ``close`` 两次——对非幂等的
    资源释放（计数、置空后的二次释放）是重复释放。
    """

    class CountingProvider:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    service = object.__new__(RetrievalService)
    provider = CountingProvider()
    service._rerank_providers = {"a": provider, "b": provider}
    service.clear_rerank_cache()
    assert provider.closed == 1


# --- services --------------------------------------------------------------


def test_embed_in_batches_empty_returns_empty_matrix():
    """空输入返回 ``(0, 0)`` 矩阵而不是让 ``np.vstack`` 抛错。

    删除守卫后 ``np.vstack([])`` 抛裸 ``ValueError: need at least one array
    to concatenate``。
    """
    from src.services.embedding_service import EmbeddingService

    service = object.__new__(EmbeddingService)
    service.run_config = type("C", (), {"kb_embedding_batch_size": 8})()

    async def fake_embed(texts):
        return np.zeros((len(texts), 4), dtype="float32")

    service.embed_texts = fake_embed
    assert asyncio.run(service.embed_in_batches([])).shape == (0, 0)
    assert asyncio.run(service.embed_in_batches(["a", "b"])).shape == (2, 4)


def test_probe_dimension_handles_all_supported_shapes():
    """维度探针：ndarray 读 shape、空列表返回 None、首元素 ndarray 读其长度。

    删除 224 后 ndarray 返回 None（维度未知，上界校验被跳过）；删除 227 后
    空列表 ``[0]`` 抛裸 ``IndexError``；删除 230 后 list-of-ndarray 返回 None。
    """
    from src.services.embedding_service import EmbeddingService

    probe = EmbeddingService._probe_dimension
    assert probe(np.zeros((2, 4), dtype="float32")) == 4
    assert probe([]) is None
    assert probe([[1.0, 2.0]]) == 2
    assert probe([np.zeros(3, dtype="float32")]) == 3


def test_build_prompt_skips_documents_without_content():
    """空 ``page_content`` 的文档不产生「来源: x / 内容: 」空条目。

    删除守卫后空文档也进上下文，提示词里出现无内容的知识条目。
    """
    from src.services.chat_service import ChatService

    documents = [
        {"page_content": "", "metadata": {"source": "a"}},
        {"page_content": "有内容", "metadata": {"source": "b"}},
    ]
    prompt = ChatService._build_prompt("问题", "qa", documents)
    assert "来源: a" not in prompt
    assert "来源: b" in prompt


# --- retrieval_test（Excel 日志）--------------------------------------------


def test_excel_cell_value_preserves_scalar_types():
    """单元格值保持原生类型：None/bool/int 不被字符串化。

    删除 197 后 None 变成字符串 ``'None'``；删除 199 后 bool 变成 ``'True'``；
    删除 203 后 int 变成 ``'5'``——Excel 里数字列变成文本列。
    """
    from src.retrieval_test.excel_logger import ExcelLogger

    assert ExcelLogger._cell_value(None) is None
    assert ExcelLogger._cell_value(True) is True
    assert ExcelLogger._cell_value(5) == 5
    assert isinstance(ExcelLogger._cell_value(5), int)


def test_excel_json_safe_preserves_none_and_strings():
    """``_json_safe`` 对 None/str 原样返回。

    删除守卫后 None 变成 ``'None'``、``"s"`` 变成 ``"'s'"``——JSON 结构里
    混入 repr 字符串。
    """
    from src.retrieval_test.excel_logger import ExcelLogger

    assert ExcelLogger._json_safe(None) is None
    assert ExcelLogger._json_safe("s") == "s"


# --- providers.resources ---------------------------------------------------


def test_wrap_native_resource_value_keeps_plain_containers_and_modules():
    """dict 等普通容器、模块与类不被包成 ``_NativeResourceProxy``。

    删除 137 后 dict 被包装（属性访问被代理拦截，类型判断变化）；删除 139
    后模块/类被包装——``getattr`` 走代理而不是真实模块。
    """
    from src.providers import resources as resources_module

    class FakeProvider:
        pass

    payload = {"a": 1}
    assert resources_module._wrap_native_resource_value(payload, FakeProvider(), "p.x") is payload
    assert resources_module._wrap_native_resource_value(np, FakeProvider(), "p.x") is np
    assert (
        resources_module._wrap_native_resource_value(FakeProvider, FakeProvider(), "p.x")
        is FakeProvider
    )


# --- ETL -------------------------------------------------------------------


def test_fallback_separator_not_duplicated_when_already_present():
    """separators 已含空串时不再追加第二个空串。

    删除守卫后 ``["a", ""]`` 变成 ``["a", "", ""]``——重复的分隔符进入
    分片器配置。
    """
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    assert RecursiveTextSplitter._with_fallback_separator(["a", ""]) == ["a", ""]
    assert RecursiveTextSplitter._with_fallback_separator(["a"]) == ["a", ""]


def test_local_hash_normalize_zero_vector_is_returned_unchanged():
    """零向量归一化时原样返回而不是除零。

    删除守卫后 ``norm == 0`` 时执行 ``value / norm`` 抛裸 ``ZeroDivisionError``。
    """
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    assert normalize_vector([3.0, 4.0]) == [0.6, 0.8]
