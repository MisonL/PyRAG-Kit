"""补齐「删守卫后全量测试仍绿」的非 raise 守卫覆盖缺口（工具/服务/检索批）。

与 ``tests/providers/test_guard_coverage_regressions.py`` 同一来源：变异扫描
（把守卫的 ``if`` 整体替换成 ``pass``）后全量 3804 条测试仍绿的存活守卫。
逐条用真实入口确认删除后行为改变，再按守卫存在时的行为写断言。
"""

import asyncio
import pathlib

import numpy as np
import pytest

from src.providers.local_hash import _normalize as normalize_vector
from src.retrieval.vdb.faiss_store import FaissStore
from src.runtime.contracts import RetrievalMethod, build_session_config
from src.services.retrieval_service import (
    HybridReranker,
    RetrievalService,
    _document_key,
    _document_metadata,
)
from src.utils.config import Settings, _coerce_number, get_settings
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


# --- 收口批：检索、配置校验器、快照与 UI -----------------------------------


def test_faiss_resolve_parent_content_tolerates_none_valued_entries():
    """``parent_documents`` 中的 None 值条目返回 None 而不是崩溃。

    删除守卫后 ``None.get("content")`` 抛裸 ``AttributeError``——载入期
    归一化只保证 dict 形态，外部写入或旧快照可能留下 None 值。
    """
    store = object.__new__(FaissStore)
    store.parent_documents = {"k1": {"content": "父块正文"}, "k2": None}
    assert store.resolve_parent_content("k1") == "父块正文"
    assert store.resolve_parent_content("k2") is None
    assert store.resolve_parent_content("missing") is None
    assert store.resolve_parent_content("") is None


def test_resource_proxy_children_cache_short_circuits_getattr():
    """``_children`` 中已缓存的子资源直接返回，不落到被包装对象上。

    删除守卫后 ``__getattr__`` 绕过缓存走 ``getattr(self._value, name)``——
    缓存项（如占位包装）被真实属性覆盖，甚至在被包装对象没有该属性时抛
    ``AttributeError``。
    """
    from src.providers import resources as resources_module

    class Inner:
        pass

    class FakeProvider:
        pass

    proxy = resources_module._NativeResourceProxy(Inner(), FakeProvider(), "p")
    proxy._children["x"] = "cached"
    assert proxy.x == "cached"


def test_choice_default_returns_active_value_when_present():
    """激活项在候选集合里时返回它本身。

    删除守卫后始终返回 None——questionary 的默认选中项丢失，菜单回落到
    未作答状态。
    """
    from src.ui.config_menu import _choice_default

    class Choice:
        def __init__(self, value):
            self.value = value

    assert _choice_default([Choice("a"), Choice("b")], "b") == "b"
    assert _choice_default([Choice("a")], "z") is None


def test_model_detail_validate_options_none_returns_empty_mapping():
    """``options=None`` 归一成空字典而不是抛错。

    删除守卫后 None 落到「必须是 TOML 表或字典」的 ``ValueError``——未配置
    options 的模型条目无法构造。
    """
    import types

    from src.utils.config import ModelDetail

    info = types.SimpleNamespace(data={"provider": "openai"}, field_name="options")
    validator = ModelDetail.__pydantic_decorators__.field_validators["validate_options"].func
    assert validator(None, info) == {}
    assert validator({"temperature": 1}, info) == {"temperature": 1}


def test_validate_retrieval_method_maps_member_names_and_values_to_enum():
    """成员名（``HYBRID_SEARCH``）与取值（``混合检索``）都归一成枚举。

    删除 1161 的守卫后成员名形态原样返回字符串；删除 1165 的守卫后取值
    形态原样返回字符串——下游按枚举比较时静默失配。
    """
    from src.utils.config import RetrievalMethod, Settings

    validator = Settings.__pydantic_decorators__.field_validators["validate_retrieval_method"].func
    assert validator("HYBRID_SEARCH") is RetrievalMethod.HYBRID_SEARCH
    assert validator("混合检索") is RetrievalMethod.HYBRID_SEARCH
    assert validator(RetrievalMethod.HYBRID_SEARCH) is RetrievalMethod.HYBRID_SEARCH


def test_snapshot_repository_empty_marker_returns_none():
    """``ACTIVE_SNAPSHOT`` 为空时返回 None（未激活），而不是抛校验错误。

    删除守卫后空标记落到「快照 id 不合法」的 ``ValueError``——首次构建前的
    空标记文件让启动流程以校验错误收场。
    """
    import types

    from src.retrieval.snapshot_repository import SnapshotRepository

    root = tmp_path_factory_dir()
    marker = root / "ACTIVE_SNAPSHOT"
    repository = object.__new__(SnapshotRepository)
    repository.run_config = types.SimpleNamespace(active_snapshot_marker=marker)
    marker.write_text("")
    assert repository.get_active_snapshot_id() is None


def tmp_path_factory_dir():
    import pathlib
    import tempfile

    return pathlib.Path(tempfile.mkdtemp())


def test_get_vector_store_method_excludes_base_stub():
    """只有基类桩实现的方法返回 None（未实现），真实覆写返回方法。

    删除守卫后基类桩也被当成已实现，调用方拿到 ``NotImplementedError`` 桩
    而不是走回退路径。
    """
    from src.retrieval.vdb.base import VectorStoreBase
    from src.services.retrieval_service import RetrievalService

    class Full(VectorStoreBase):
        def add_documents(self, *args, **kwargs):
            return None

        def aadd_documents(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

        async def asearch(self, *args, **kwargs):
            return []

        def get_embedding_model(self, *args, **kwargs):
            return None

        def load(self, *args, **kwargs):
            return None

        def save(self, *args, **kwargs):
            return None

    service = object.__new__(RetrievalService)
    stub = Full()
    Full.semantic_search = VectorStoreBase.semantic_search
    service.vector_store = stub
    assert service._get_vector_store_method("semantic_search") is None

    class Impl:
        def semantic_search(self, query, top_k):
            return []

    service.vector_store = Impl()
    assert service._get_vector_store_method("semantic_search") is not None


def test_legacy_search_prefers_async_search_implementation():
    """向量存储提供 ``asearch`` 时走异步实现，不回退到同步 ``search``。

    删除守卫后异步实现被跳过，直接找 ``search``——只有 ``asearch`` 的
    存储抛 ``NotImplementedError``。
    """
    from src.services.retrieval_service import RetrievalService

    class AsyncSearcher:
        async def asearch(self, query, top_k, search_type=None):
            return ["async"]

    service = object.__new__(RetrievalService)
    service.vector_store = AsyncSearcher()
    assert asyncio.run(service._legacy_search("q", 1, "semantic")) == ["async"]


def test_aclose_closes_duplicate_provider_once():
    """异步关闭时同一 provider 对象在多个槽位中只关一次。

    删除守卫后按槽位逐个 ``aclose``，同一对象被关闭两次。
    """

    class AsyncProvider:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    service = object.__new__(RetrievalService)
    provider = AsyncProvider()
    service._rerank_providers = {"a": provider, "b": provider}
    service.embedding_service = None
    asyncio.run(service.aclose())
    assert provider.closed == 1


def test_run_retrieval_test_skips_empty_query(monkeypatch):
    """空查询被跳过，不进入检索。

    删除守卫后空串会传给 ``retrieve``——交互循环里回车即触发一次无意义
    检索（或错误）。
    """
    from src.retrieval_test import core as retrieval_core

    calls = []

    class FakePromptSession:
        def __init__(self, responses):
            self._responses = iter(responses)

        async def prompt_async(self, *_args, **_kwargs):
            return next(self._responses)

    class RecordingService:
        def __init__(self, *_args, **_kwargs):
            return None

        async def retrieve(self, query, *_args, **_kwargs):
            calls.append(query)
            return []

    monkeypatch.setattr(retrieval_core, "PromptSession", lambda: FakePromptSession(["", "/quit"]))
    monkeypatch.setattr(retrieval_core, "RetrievalService", RecordingService)
    monkeypatch.setattr(retrieval_core, "EmbeddingService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        retrieval_core.VectorStoreFactory, "get_default_vector_store", lambda: object()
    )
    monkeypatch.setattr(retrieval_core, "ExcelLogger", lambda: None)
    asyncio.run(retrieval_core.run_retrieval_test_async())
    assert calls == []


def test_aclose_quietly_returns_without_closable_service(caplog):
    """服务没有 ``aclose``/``close`` 时安静返回，不产生警告。

    删除守卫后 ``close()`` 在 None 上抛 ``TypeError``，虽被外层 except 吞掉，
    但会多打一条「释放检索服务失败」警告——「没有可关闭资源」不是失败，
    警告会误导排查。
    """
    import logging

    from src.retrieval_test.core import _aclose_quietly

    with caplog.at_level(logging.WARNING, logger="src.retrieval_test.core"):
        asyncio.run(_aclose_quietly(object()))
    assert "释放检索服务失败" not in caplog.text


# --- 收口批：吞掉型 except 的静默降级契约 ----------------------------------


def test_copy_nested_falls_back_when_model_copy_fails():
    """``model_copy`` 抛错时回退到 deepcopy，而不是把异常放出去。

    ``_copy_nested`` 的契约是「拿到一份副本，绝不因复制失败中断调用方」。
    删除两处吞掉后，``model_copy``/``deepcopy`` 的异常都会冒泡。
    """

    from src.utils.security import _copy_nested

    class BadModelCopy:
        def __init__(self):
            self.x = 1

        def model_copy(self, **kwargs):
            raise RuntimeError("model_copy 失败")

    copied = _copy_nested(BadModelCopy())
    assert copied is not None

    class BadDeepcopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy 失败")

    obj = BadDeepcopy()
    assert _copy_nested(obj) is obj


def test_model_dump_mapping_retries_without_kwargs_and_fails_open():
    """``model_dump`` 的三种失败形态都收敛为「读不到」。

    * 不接受 ``exclude_none`` 关键字 → 无参重试（实测 SDK 旧版本）；
    * 无参重试也抛错 → 返回 None（fail-open，见 docstring）；
    * 首次调用抛非 ``TypeError`` → 同样返回 None。
    """

    from src.utils.security import _model_dump_mapping

    class NoKwargs:
        def model_dump(self, **kwargs):
            if kwargs:
                raise TypeError("不接受参数")
            return {"a": 1}

    assert _model_dump_mapping(NoKwargs()) == {"a": 1}

    class AlwaysBoom:
        def model_dump(self, **kwargs):
            if kwargs:
                raise TypeError("不接受参数")
            raise RuntimeError("无参也崩")

    assert _model_dump_mapping(AlwaysBoom()) is None

    class Boom:
        def model_dump(self, **kwargs):
            raise RuntimeError("dump 崩")

    assert _model_dump_mapping(Boom()) is None


def test_require_retrieval_method_reports_guidance_for_broken_value_attribute():
    """取值属性抛错的输入落到带指引的 ``ValueError``，而不是把原异常放出去。

    删除吞掉后 ``Weird.value`` 的 ``RuntimeError`` 直接冒泡——调用方拿到
    的是与配置校验无关的异常。
    """

    from src.utils.config import require_retrieval_method

    class Weird:
        @property
        def value(self):
            raise RuntimeError("取值失败")

    with pytest.raises(ValueError, match="retrieval_method 必须是"):
        require_retrieval_method(Weird())


def test_get_relative_path_returns_input_for_empty_string():
    """空字符串路径原样返回而不是抛 ``ValueError``。

    删除吞掉后 ``Path("").relative_to`` 的 ``ValueError: no path specified``
    冒泡——展示层因一条空路径整表渲染失败。
    """

    from src.ui.display_utils import get_relative_path

    assert get_relative_path("") == ""


def test_get_chat_logger_degrades_to_console_when_log_dir_unusable(caplog):
    """日志目录不可写时降级为仅控制台并告警一次，不中断启动。

    删除吞掉后 ``_build_rotating_handler`` 的 ``OSError`` 冒泡——日志目录
    指向一个已存在的普通文件时，查询直接崩在日志初始化。
    """
    import logging

    from src.utils import log_manager

    blocked_dir = tmp_path_factory_dir() / "blocked"
    blocked_dir.write_text("我是文件不是目录")

    class FakeSettings:
        log_path = str(blocked_dir)
        log_level = "INFO"

    original = log_manager.get_settings
    log_manager.get_settings = lambda: FakeSettings()
    # chat_logger 是模块级单例：别的用例可能已经给它配置过 handlers，
    # 而配置分支只在「还没有 handler」时执行。先清空再断言。
    chat_logger = logging.getLogger("chat_logger")
    saved_handlers = chat_logger.handlers
    chat_logger.handlers = []
    try:
        with caplog.at_level(logging.WARNING, logger="src.utils.log_manager"):
            logger = log_manager.get_chat_logger()
        assert logger.name == "chat_logger"
        assert "无法写入日志目录" in caplog.text
    finally:
        for handler in logger.handlers:
            if handler not in saved_handlers:
                handler.close()
        chat_logger.handlers = saved_handlers
        log_manager.get_settings = original


def test_cleanup_old_logs_continues_when_one_file_cannot_be_removed(caplog):
    """单个旧日志删除失败只记日志、继续清理，不让异常打断退出钩子。

    删除吞掉后 ``os.remove`` 的 ``PermissionError`` 冒泡——退出路径以
    裸 traceback 终止。
    """
    import logging
    import os

    from src.utils import log_manager

    log_dir = tmp_path_factory_dir()
    old = log_dir / "app_log_2020-01-01.log"
    old.write_text("x")
    fresh = log_dir / "app_log_2026-10-07.log"
    fresh.write_text("y")
    os.chmod(old, 0o500)
    os.chmod(log_dir, 0o500)

    class FakeSettings:
        log_path = str(log_dir)
        log_retention_days = 1
        log_level = "INFO"

    original = log_manager.get_settings
    log_manager.get_settings = lambda: FakeSettings()
    try:
        with caplog.at_level(logging.ERROR, logger="src.utils.log_manager"):
            log_manager.cleanup_old_logs()
        assert "清理日志文件" in caplog.text
    finally:
        log_manager.get_settings = original
        os.chmod(log_dir, 0o700)


def test_reject_unsafe_cache_dir_skips_unresolvable_protected_paths():
    """``protected_paths`` 里的链接环被跳过，其余项仍参与判定。

    删除吞掉后 ``Path.resolve()`` 的 ``RuntimeError: Symlink loop`` 冒泡——
    而本函数在 atexit 里被调用，异常让清理完全没做。
    """
    import pathlib

    from src.utils.cleanup import _reject_unsafe_cache_dir

    root = tmp_path_factory_dir()
    loop = root / "loop"
    loop.symlink_to(loop)
    cache_dir = root / "cache"
    cache_dir.mkdir()
    reason = _reject_unsafe_cache_dir(str(cache_dir), [loop, pathlib.Path(cache_dir)])
    assert reason is not None and "覆盖了应用自身路径" in reason


def test_rerank_provider_fingerprint_falls_back_for_unserializable_options(monkeypatch):
    """``options`` 含超大整数时回退到 ``_stable_fingerprint`` 而不是崩溃。

    ``json.dumps`` 对 ``10**4301`` 抛 ``ValueError``（整数位数上限）；
    删除吞掉后每次检索都在构造缓存键处崩。
    """
    from types import SimpleNamespace

    from src.services.retrieval_service import RetrievalService

    service = object.__new__(RetrievalService)
    service._rerank_providers = {}
    sentinel = object()
    monkeypatch.setattr(
        "src.services.retrieval_service.ModelProviderFactory.get_rerank_provider",
        lambda *_args, **_kwargs: sentinel,
    )
    detail = SimpleNamespace(provider="jina", model_name="m", options={"x": 10**4301})
    provider = service._get_rerank_provider("k", {"k": detail})
    assert provider is sentinel


def test_run_retrieval_test_reports_excel_logger_failure_and_continues(monkeypatch):
    """ExcelLogger 初始化失败时打印提示并继续运行。

    删除吞掉后初始化异常冒泡——召回测试（唯一产出是 Excel 日志）直接退出，
    用户不知道记录没有落盘。
    """
    from src.retrieval_test import core as retrieval_core

    printed = []

    class FakePromptSession:
        def __init__(self, responses):
            self._responses = iter(responses)

        async def prompt_async(self, *_args, **_kwargs):
            return next(self._responses)

    class BoomLogger:
        def __init__(self):
            raise RuntimeError("初始化失败")

    monkeypatch.setattr(retrieval_core, "PromptSession", lambda: FakePromptSession(["/quit"]))
    monkeypatch.setattr(retrieval_core, "ExcelLogger", BoomLogger)
    monkeypatch.setattr(retrieval_core, "RetrievalService", lambda *_a, **_k: object())
    monkeypatch.setattr(retrieval_core, "EmbeddingService", lambda *_a, **_k: object())
    monkeypatch.setattr(
        retrieval_core.VectorStoreFactory, "get_default_vector_store", lambda: object()
    )
    monkeypatch.setattr(retrieval_core.console, "print", lambda *a, **k: printed.append(str(a)))
    asyncio.run(retrieval_core.run_retrieval_test_async())
    assert any("Excel 日志初始化失败" in message for message in printed)


def test_run_retrieval_test_reports_database_load_failure(monkeypatch):
    """数据库加载失败时打印提示并返回，不抛异常。

    删除吞掉后初始化异常冒泡——用户拿到裸 traceback 而不是可读的失败原因。
    """
    from src.retrieval_test import core as retrieval_core

    printed = []

    def boom():
        raise RuntimeError("加载失败")

    monkeypatch.setattr(retrieval_core, "PromptSession", lambda: object())
    monkeypatch.setattr(retrieval_core, "ExcelLogger", lambda: None)
    monkeypatch.setattr(retrieval_core.VectorStoreFactory, "get_default_vector_store", boom)
    monkeypatch.setattr(retrieval_core.console, "print", lambda *a, **k: printed.append(str(a)))
    asyncio.run(retrieval_core.run_retrieval_test_async())
    assert any("数据库加载失败" in message for message in printed)


def test_run_retrieval_test_breaks_on_keyboard_interrupt(monkeypatch):
    """交互循环里 Ctrl-C 干净退出，不作为异常抛出。

    删除吞掉后 ``KeyboardInterrupt`` 冒泡——退出交互模式变成一次崩溃。
    """
    from src.retrieval_test import core as retrieval_core

    class InterruptingSession:
        async def prompt_async(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(retrieval_core, "PromptSession", InterruptingSession)
    monkeypatch.setattr(retrieval_core, "ExcelLogger", lambda: None)
    monkeypatch.setattr(retrieval_core, "RetrievalService", lambda *_a, **_k: object())
    monkeypatch.setattr(retrieval_core, "EmbeddingService", lambda *_a, **_k: object())
    monkeypatch.setattr(
        retrieval_core.VectorStoreFactory, "get_default_vector_store", lambda: object()
    )
    asyncio.run(retrieval_core.run_retrieval_test_async())


def test_excel_logger_reports_original_error_when_placeholder_cleanup_fails(caplog, monkeypatch):
    """构造失败时：占位文件回收失败只记日志，抛出的仍是**原始**失败。

    删除吞掉后回收占位文件的 ``OSError`` 顶替原始失败——用户看到的失败
    原因与实际不符。
    """
    import logging

    from src.retrieval_test import excel_logger

    class BoomWorkbook:
        def __init__(self):
            raise RuntimeError("工作簿构造失败")

    log_dir = tmp_path_factory_dir()
    placeholder_dir = log_dir / "placeholder.xlsx"
    placeholder_dir.mkdir()

    monkeypatch.setattr(
        excel_logger.ExcelLogger,
        "_reserve_log_path",
        staticmethod(lambda *_a, **_k: placeholder_dir),
    )
    monkeypatch.setattr(excel_logger, "Workbook", BoomWorkbook)
    with (
        caplog.at_level(logging.DEBUG, logger="src.retrieval_test.excel_logger"),
        pytest.raises(RuntimeError, match="工作簿构造失败"),
    ):
        excel_logger.ExcelLogger(log_dir=str(log_dir))
    assert "回收占位日志文件失败" in caplog.text


def test_reject_unsafe_cache_dir_treats_unresolvable_home_as_absent():
    """``Path.home()`` 不可得时视为「没有家目录」而不是让函数整体崩溃。

    删除吞掉后 ``RuntimeError`` 冒泡——本函数在 atexit 里被调用，异常让
    清理完全没做。
    """
    import pathlib

    from src.utils.cleanup import _reject_unsafe_cache_dir

    cache_dir = tmp_path_factory_dir() / "cache"
    cache_dir.mkdir()

    original_home = pathlib.Path.home

    def boom():
        raise RuntimeError("无家目录")

    pathlib.Path.home = staticmethod(boom)
    try:
        assert _reject_unsafe_cache_dir(str(cache_dir)) is None
    finally:
        pathlib.Path.home = original_home


def test_aclose_quietly_logs_and_swallows_close_failure(caplog):
    """关闭失败只记警告，不让异常打断退出流程。

    删除吞掉后 ``aclose`` 的异常冒泡——退出清理路径上的失败变成了崩溃。
    """
    import logging

    from src.retrieval_test.core import _aclose_quietly

    class Bad:
        async def aclose(self):
            raise RuntimeError("关闭失败")

    with caplog.at_level(logging.WARNING, logger="src.retrieval_test.core"):
        asyncio.run(_aclose_quietly(Bad()))
    assert "释放检索服务失败" in caplog.text


def test_excel_logger_log_results_swallows_save_failure(caplog):
    """写 Excel 失败只记日志，不打断召回循环。

    删除吞掉后 ``log_results`` 抛 ``FileNotFoundError``——一次写盘失败让
    整个召回测试退出，已展示的结果全部丢失。
    """
    import logging

    from src.retrieval_test import excel_logger

    log_dir = tmp_path_factory_dir()
    logger = excel_logger.ExcelLogger(str(log_dir))
    logger.filepath = log_dir / "no-such-dir" / "x.xlsx"
    with caplog.at_level(logging.ERROR, logger="src.retrieval_test.excel_logger"):
        logger.log_results("q", [{"content": "c", "source": "s", "score": 1.0}])
    assert "保存 Excel 日志文件失败" in caplog.text


def test_capability_report_returns_none_for_missing_sdk_package(monkeypatch):
    """SDK 包未安装时 version 报 None，报告本身照常生成。

    删除吞掉后 ``PackageNotFoundError`` 冒泡——能力报告对任何未安装可选
    依赖的渠道直接失败。
    """
    from importlib import metadata

    from src.providers.factory import ModelProviderFactory

    def boom(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("src.providers.factory.package_version", boom)
    report = ModelProviderFactory.capability_report("google", "llm")
    assert report["sdk"]["version"] is None


# --- except 元组收窄：声称捕获的第二类型必须有测试触发 ----------------------
#
# 来源：把 ``except (A, B, ...)`` 收窄成 ``except A`` 后全量测试仍绿的存活点。
# 逐条用探针确认「命中被丢弃类型时行为确实改变」，再按守卫存在时的行为写断言。


def test_safe_repr_reports_custom_repr_failure_as_magnitude():
    """``repr()`` 抛 ``RuntimeError`` 时退回量级描述而不是冒泡。

    收窄成 ``except ValueError`` 后，自定义 ``__repr__`` 失败的对象会让校验
    消息拼接崩掉——而 ``safe_repr`` 的契约是「绝不抛」。
    """
    from src.utils.config import safe_repr

    class _BadRepr:
        def __repr__(self):
            raise RuntimeError("repr 失败")

    assert safe_repr(_BadRepr()) == "<_BadRepr>"


def test_security_display_repr_reports_custom_repr_failure_as_magnitude():
    """``_display_repr`` 对 ``repr`` 抛错的对象退回类型名而不是冒泡。"""
    from src.utils.security import _display_repr

    class _BadRepr:
        def __repr__(self):
            raise RuntimeError("repr 失败")

    assert _display_repr(_BadRepr()) == "<_BadRepr>"


def test_redact_sensitive_text_reports_custom_str_failure_as_magnitude():
    """``redact_sensitive_text`` 对 ``str()`` 抛错的对象退回类型名而不是冒泡。

    收窄成 ``except ValueError`` 后，日志脱敏在遇到自定义 ``__str__`` 失败时
    抛出——脱敏本应是「绝不阻断日志」的边界。
    """
    from src.utils.security import redact_sensitive_text as redact

    class _BadStr:
        def __str__(self):
            raise RuntimeError("str 失败")

    assert redact(_BadStr()) == "<_BadStr>"


def test_display_render_value_falls_back_on_custom_str_failure():
    """``_render_value`` 对 ``str()`` 抛错的对象退回 ``safe_repr``。"""
    from src.ui.display_utils import _render_value

    class _BadStr:
        def __str__(self):
            raise RuntimeError("str 失败")

        def __repr__(self):
            # ``_render_value`` 降级走 ``safe_repr``，其内部再用 ``repr()``；
            # 两者都要失败才会落到类型名兜底。
            raise RuntimeError("repr 失败")

    assert _render_value(_BadStr()) == "<_BadStr>"


def test_display_path_text_falls_back_on_custom_str_failure():
    """``_path_text`` 对 ``str()`` 抛错的对象退回 ``safe_repr``。"""
    from src.ui.display_utils import _path_text

    class _BadStr:
        def __str__(self):
            raise RuntimeError("str 失败")

        def __repr__(self):
            raise RuntimeError("repr 失败")

    assert _path_text(_BadStr()) == "<_BadStr>"


def test_fingerprint_falls_back_on_custom_repr_failure():
    """``_fingerprint`` 对 ``repr`` 抛错的对象退回类型名而不是冒泡。

    收窄成 ``except ValueError`` 后，``options`` 里一个自定义 ``__repr__``
    失败的对象会让缓存键计算崩掉。
    """
    from src.services.retrieval_service import _fingerprint

    class _BadRepr:
        def __repr__(self):
            raise RuntimeError("repr 失败")

    assert _fingerprint(_BadRepr(), set()) == "<_BadRepr>"


def test_resolve_within_converts_symlink_loop_to_valueerror(tmp_path):
    """``Path.resolve()`` 对符号链接环抛 ``RuntimeError``，收敛成 ``ValueError``。

    收窄成 ``except OSError`` 后，``RuntimeError: Symlink loop`` 穿透本函数的
    ``ValueError`` 契约，调用方（``ensure_trusted_source`` 只接 ``ValueError``）
    拿到的是 pathlib 内部错误。
    """
    from src.utils.security import resolve_within

    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="无法解析"):
        resolve_within(tmp_path / "a" / "x", tmp_path, label="快照")


def test_require_finite_number_rejects_numeric_string():
    """数值字符串必须被拒，而不是经 ``to_finite_float`` 静默转成浮点。

    删除类型守卫后，``"0.5"`` 会被 ``to_finite_float`` 接受并返回 ``0.5``——
    ``SessionConfig`` 的兄弟字段本应只接受数值。
    """
    from src.runtime.contracts import SessionConfig

    with pytest.raises(ValueError, match="必须是数值"):
        SessionConfig._require_finite_number("0.5", "权重")


def test_require_chat_temperature_rejects_numeric_string_with_numeric_message():
    """``require_chat_temperature`` 对非数值给出「必须是数值」，而不是范围消息。

    删除类型守卫后，``"0.5"`` 落到范围检查并报「必须在 0.0 到 1.0 之间」——
    把「类型不对」说成了「范围不对」。
    """
    from src.utils.config import require_chat_temperature

    with pytest.raises(ValueError, match="必须是数值"):
        require_chat_temperature("0.5")


def test_validate_rerank_output_reports_non_numeric_score_readably():
    """非数值 score（如 ``"abc"``）收敛成本方法的 ``RuntimeError``。

    收窄成 ``except TypeError`` 后，``float("abc")`` 的 ``ValueError`` 冒泡，
    调用方拿到的是裸转换错误而不是「provider 返回的 score 无效」。
    """
    with pytest.raises(RuntimeError, match="score 无效"):
        RetrievalService._validate_rerank_output([0], ["abc"], 5)


def test_key_text_falls_back_to_digest_for_int_subclass_repr_failure():
    """数值子类的 ``repr`` 抛错时退回摘要键，而不是冒泡。

    ``_key_text`` 的 ``except (ValueError, RuntimeError, RecursionError)`` 收窄成
    ``except ValueError`` 后，``int`` 派生类覆写 ``__repr__`` 抛 ``RuntimeError``
    会让去重键计算崩掉——这正是「合并检索结果」的中途。
    """

    class _BadReprInt(int):
        def __repr__(self):
            raise RuntimeError("repr 失败")

    from src.services.retrieval_service import _key_text

    assert _key_text(_BadReprInt(7)).startswith("<整数，")


# --- 边界比较符互换：恰好等于阈值的情形必须有测试 --------------------------
#
# 来源：把 ``<``↔``<=``、``>``↔``>=`` 互换后全量仍绿的存活点。存活即说明
# 「取值恰好等于阈值」从未被测试覆盖（off-by-one 盲区）。逐条用探针确认
# 边界值处的行为改变后，按守卫存在时的行为写断言。


def test_separator_table_accepts_exactly_the_maximum():
    """分隔符表恰好等于上界（64）时接受，65 项才拒绝。

    ``>`` 改 ``>=`` 后 64 项被误拒——真实配置可能正好用满上界。
    """
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    limit = RecursiveTextSplitter._MAX_SEPARATORS
    assert RecursiveTextSplitter._require_separators([f"s{i}" for i in range(limit)])
    with pytest.raises(ValueError, match="分隔符表最多"):
        RecursiveTextSplitter._require_separators([f"s{i}" for i in range(limit + 1)])


def test_client_timeout_accepts_exactly_the_maximum():
    """客户端超时恰好等于上界时接受，超过才拒绝。"""
    from src.providers.__base__.model_provider import (
        _MAX_CLIENT_TIMEOUT_SECONDS,
        normalize_client_timeout,
    )

    assert normalize_client_timeout(_MAX_CLIENT_TIMEOUT_SECONDS, "X", "timeout") == (
        _MAX_CLIENT_TIMEOUT_SECONDS
    )
    with pytest.raises(ValueError, match="不能超过"):
        normalize_client_timeout(_MAX_CLIENT_TIMEOUT_SECONDS + 1, "X", "timeout")


@pytest.mark.parametrize("status", [500, "500"])
def test_retryable_error_includes_status_500_itself(status):
    """500 本身可重试（``>= 500`` 不是 ``> 500``）。"""
    from types import SimpleNamespace

    from src.providers.__base__.model_provider import is_retryable_error

    assert is_retryable_error(SimpleNamespace(status_code=status)) is True
    assert is_retryable_error(SimpleNamespace(status_code=499)) is False


def test_google_http_options_timeout_accepts_exactly_the_millisecond_maximum():
    """``http_options.timeout`` 恰好等于毫秒上界时接受，多 1 毫秒才拒绝。"""
    from src.providers.__base__.model_provider import _MAX_CLIENT_TIMEOUT_SECONDS
    from src.providers.google import GoogleProvider

    boundary_ms = int(_MAX_CLIENT_TIMEOUT_SECONDS * 1000)
    GoogleProvider._validate_http_options({"timeout": boundary_ms})
    with pytest.raises(ValueError, match="正整数毫秒"):
        GoogleProvider._validate_http_options({"timeout": boundary_ms + 1})


def test_google_partial_args_accepts_list_index_zero():
    """``json_path`` 的列表索引 0 是合法写入位置（``< 0`` 不是 ``<= 0``）。"""
    from src.providers.google import GoogleProvider

    result = GoogleProvider._partial_args_to_mapping(
        [{"json_path": "$.a[0]", "string_value": "hello"}]
    )
    assert result == {"a": ["hello"]}


def test_google_merge_tool_arguments_extends_list_at_boundary():
    """列表恰好需要扩容一个位置（``index == len``）时必须扩容，否则 IndexError。"""
    from src.providers.google import GoogleProvider

    assert GoogleProvider._merge_tool_arguments([1], [1, 2]) == [1, 2]


def test_local_hash_dimension_zero_falls_back_to_default():
    """``model_name`` 解析出维度 0 时退回默认，而不是产生 0 维向量。"""
    from src.providers.local_hash import DEFAULT_DIMENSION, _parse_dimension

    assert _parse_dimension("local-hash-0") == DEFAULT_DIMENSION
    assert _parse_dimension("local-hash-1") == 1


@pytest.mark.parametrize("count", [2**63, 2**64 - 1])
def test_manifest_count_accepts_up_to_64_bits(count):
    """位数恰好到 64 位（``2**64 - 1``）时接受，第 65 位才拒绝。"""
    from src.runtime.contracts import _manifest_count

    assert _manifest_count(count, "document_count") == count
    with pytest.raises(ValueError, match="超出合理范围"):
        _manifest_count(2**64, "document_count")


def test_toml_string_does_not_escape_plain_space():
    """普通空格不需要转义（``< ' '`` 不是 ``<= ' '``），控制字符才转义。"""
    from src.runtime.contracts import KnowledgeSnapshotManifest

    assert KnowledgeSnapshotManifest._toml_string("a b") == '"a b"'
    assert KnowledgeSnapshotManifest._toml_string("a\x01b") == '"a\\u0001b"'
    assert KnowledgeSnapshotManifest._toml_string("a\x7fb") == '"a\\u007Fb"'


def test_safe_repr_truncates_only_beyond_limit():
    """``repr`` 长度恰好等于 ``limit`` 时不截断，超过才截断。"""
    from src.utils.config import safe_repr

    exact = "x" * 78  # repr 后恰好 80 字符
    assert len(repr(exact)) == 80
    assert safe_repr(exact) == repr(exact)
    over = "x" * 79
    assert safe_repr(over).endswith("...")


def test_snapshot_id_accepts_exactly_the_maximum_length():
    """快照 id 恰好等于长度上界（64）时接受，65 才拒绝。"""
    import tempfile

    from src.retrieval.snapshot_repository import (
        _MAX_SNAPSHOT_ID_LENGTH,
        SnapshotRepository,
    )

    repo = SnapshotRepository.__new__(SnapshotRepository)
    repo.root = pathlib.Path(tempfile.mkdtemp())
    repo._require_snapshot_id_within_root("a" * _MAX_SNAPSHOT_ID_LENGTH, source="test")
    with pytest.raises(ValueError, match="快照 id 过长"):
        repo._require_snapshot_id_within_root("a" * (_MAX_SNAPSHOT_ID_LENGTH + 1), source="test")


def test_rebuild_indices_skips_faiss_for_empty_embeddings():
    """空嵌入矩阵不建 FAISS 索引（``len > 0`` 不是 ``len >= 0``）。"""
    import numpy as np

    store = object.__new__(FaissStore)
    store.documents = []
    store.embeddings = np.zeros((0, 4), dtype=np.float32)
    store._tokenized_docs_cache = None
    store.bm25_index = None
    store.faiss_index = None
    store._rebuild_indices()
    assert store.faiss_index is None


def test_deduplicate_parent_documents_keeps_first_on_equal_scores():
    """分数相同时保留先出现的文档（``>`` 不是 ``>=``）。"""
    from src.services.retrieval_service import _deduplicate_parent_documents

    documents = [
        {"page_content": "first", "metadata": {"parent_id": "p1"}, "score": 0.5},
        {"page_content": "second", "metadata": {"parent_id": "p1"}, "score": 0.5},
    ]
    assert [d["page_content"] for d in _deduplicate_parent_documents(documents)] == ["first"]


def test_keyword_search_stops_exactly_at_top_k():
    """结果数恰好达到 ``top_k`` 时停止（``>=`` 不是 ``>``），不多取一条。"""
    import jieba
    import numpy as np
    import rank_bm25

    store = FaissStore(file_path=None)
    # 每篇各含一个只出现一次的独有词 -> 四篇都是正分；查询四词时四条都合格。
    texts = ["独一 甲", "独二 乙", "独三 丙", "独四 丁"]
    store.documents = [
        {"page_content": text, "metadata": {"source": f"s{index}.md"}}
        for index, text in enumerate(texts)
    ]
    store.embeddings = np.zeros((len(texts), 4), dtype=np.float32)
    store._tokenized_docs_cache = [list(jieba.cut(text)) for text in texts]
    store.bm25_index = rank_bm25.BM25Okapi(store._tokenized_docs_cache)

    query = "独一 独二 独三 独四"
    scores = store.bm25_index.get_scores(list(jieba.cut(query)))
    assert (scores > 0).sum() == 4, "前提：四篇都必须正分"
    assert len(store.keyword_search(query, top_k=2)) == 2
    assert len(store.keyword_search(query, top_k=1)) == 1


def test_retrieve_accepts_candidate_product_exactly_at_the_maximum():
    """``top_k × multiplier`` 恰好等于上界时接受，越界才拒绝。"""
    from src.services.retrieval_service import _MAX_RETRIEVAL_CANDIDATES, RetrievalService

    class _Store:
        def semantic_search(self, embedding, top_k):
            return []

        def resolve_parent_content(self, parent_id):
            return None

    class _Embedding:
        async def embed_query(self, query):
            return [0.0] * 4

    config = build_session_config(get_settings())
    config.retrieval_method = RetrievalMethod.SEMANTIC_SEARCH
    config.rerank_enabled = False
    config.top_k = _MAX_RETRIEVAL_CANDIDATES
    config.retrieval_candidate_multiplier = 1
    # 恰好等于上界：接受（不抛）。
    asyncio.run(RetrievalService(_Store(), _Embedding()).retrieve("q", config))


def test_retrieve_keeps_document_scoring_exactly_at_the_threshold():
    """分数恰好等于阈值时保留（``>=`` 不是 ``>``）。"""
    from src.services.retrieval_service import RetrievalService

    class _Store:
        def semantic_search(self, embedding, top_k):
            return [
                {"page_content": "exact", "metadata": {"source": "a.md"}, "score": 0.5},
                {"page_content": "below", "metadata": {"source": "b.md"}, "score": 0.4},
            ]

        def resolve_parent_content(self, parent_id):
            return None

    class _Embedding:
        async def embed_query(self, query):
            return [0.0] * 4

    config = build_session_config(get_settings())
    config.retrieval_method = RetrievalMethod.SEMANTIC_SEARCH
    config.rerank_enabled = False
    config.score_threshold = 0.5
    results = asyncio.run(RetrievalService(_Store(), _Embedding()).retrieve("q", config))
    assert [d["page_content"] for d in results] == ["exact"]


def test_overlap_accepts_one_below_the_maximum():
    """overlap 恰好等于上界减一（999999）时接受，等于上界（1000000）才拒绝。"""
    assert Settings.validate_non_negative_overlap(999999) == 999999
    with pytest.raises(ValueError, match="必须小于"):
        Settings.validate_non_negative_overlap(1000000)


def test_save_snapshot_rejects_when_only_faiss_index_is_missing():
    """``faiss_index is None``（embeddings 非空）时也要拒绝保存（``or`` 不是 ``and``）。

    ``or`` 改 ``and`` 后只有两者**同时**为空才拒绝，单缺索引时静默写出不完整
    快照——重载后语义检索不可用。
    """
    import tempfile

    store = object.__new__(FaissStore)
    store.faiss_index = None
    store.embeddings = np.zeros((2, 4), dtype=np.float32)
    store.documents = []
    store.parent_documents = {}
    store.bm25_index = None
    store._tokenized_docs_cache = None
    with pytest.raises(RuntimeError, match="语义索引为空"):
        store.save_snapshot(tempfile.mkdtemp())


def test_session_config_accepts_one_zero_weight():
    """只有一个权重为 0 时接受（``and`` 不是 ``or``）。"""
    config = build_session_config(get_settings())
    config.set_many({"vector_weight": 0.5, "keyword_weight": 0.0})
    assert config.keyword_weight == 0.0
    with pytest.raises(ValueError, match="不能同时为 0"):
        config.set_many({"vector_weight": 0.0, "keyword_weight": 0.0})


def test_unit_weight_rejects_bool_with_numeric_message():
    """``bool`` 权重报「必须是数值」，而不是「必须是有限数值」。

    ``or`` 改 ``and`` 后 ``True`` 落到有限性判据，消息把「类型不对」说成
    「取值不对」。
    """
    from src.runtime.contracts import SessionConfig

    with pytest.raises(ValueError, match="必须是数值"):
        SessionConfig._validate_unit_weight(True, "权重")
