"""本轮边界加固的回归测试。

每一条都对应一个**已实测复现**的缺陷：先确认它在未修源码时变红，再确认修复后
变绿。命名沿用缺陷类别（异常分类泄漏、读写不对称、静默语义破坏），便于日后按
类别排查同类问题。
"""

import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import httpx
import numpy as np
import pytest
from openai._models import construct_type
from openai.types.chat import ChatCompletionChunk

import src.utils.config as config_module
from src.providers.__base__.model_provider import merge_tool_call_fragment
from src.providers.factory import ModelProviderFactory
from src.providers.google import GoogleProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.retrieval.vdb.faiss_store import FaissStore
from src.runtime.contracts import RetrievalMethod, SessionConfig
from src.services.retrieval_service import _coerce_fusion_strategy, _key_text
from src.utils.config import Settings, get_settings, safe_repr
from src.utils.log_manager import cleanup_old_logs

# pytest 自动生成 id 时会对参数调 ``str()``，位数超限的整数会让收集阶段自己崩掉。
_HUGE_INT = 10**4301


def _patch_provider_settings(monkeypatch, module_name: str, **values):
    """替换 provider 模块里的 ``get_settings``，避免依赖真实凭证。

    provider 构造器在缺 key 时直接拒绝，而 ``get_settings`` 带 ``lru_cache``——
    设置环境变量不会生效（缓存里已有实例）。仓库既有用例同样打这个模块级名字。
    """
    monkeypatch.setattr(f"{module_name}.get_settings", lambda: SimpleNamespace(**values))


def _log_settings(log_path: str, *, retention_days: int):
    """构造 ``cleanup_old_logs`` 需要的 settings 替身。

    ``get_module_logger`` 在清理过程中会被调用，它要读 ``log_level``——只给
    ``log_path``／``log_retention_days`` 会让清理路径本身因属性缺失而失败。
    """
    return SimpleNamespace(
        log_path=log_path,
        log_retention_days=retention_days,
        log_level="INFO",
    )


def _session_config(**overrides):
    """构造一个合法 SessionConfig，按需覆盖字段。"""
    values = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 2,
        "rerank_enabled": False,
        "top_k": 5,
        "score_threshold": 0.0,
        "active_llm_configuration": "x",
        "active_rerank_configuration": "y",
        "llm_configurations": {},
        "rerank_configurations": {},
        "chat_temperature": 0.7,
    }
    values.update(overrides)
    return SessionConfig(**values)


# ---------------------------------------------------------------------------
# 一、异常分类泄漏：守卫抛出的异常必须在调用方的 except ValueError/RuntimeError 之内
# ---------------------------------------------------------------------------


def _chunk_with_tool_call_index(index):
    """用真实 SDK 解析构造一个 chunk（``construct_type`` 跳过已声明字段的校验）。"""
    return construct_type(
        type_=ChatCompletionChunk,
        value={
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": index, "id": "a", "function": {"name": "f", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
    )


@pytest.mark.parametrize("index", [[1, 2], {"a": 1}, {1}, ["a"]])
def test_merge_tool_call_fragment_tolerates_unhashable_index(index):
    """网关把 ``tool_calls[].index`` 写成数组/对象时不得抛裸 ``TypeError``。

    ``openai`` SDK 的流式解析走 ``construct_type``，**跳过已声明字段的
    Literal/int 校验**，因此 ``index=[1, 2]``（声明类型是 ``int``）会原样到达
    provider。修复前 ``accumulator.setdefault(("index", [1, 2]), ...)`` 抛
    ``TypeError: unhashable type``——调用方按 ``ValueError``／``RuntimeError``
    分类，接不住。
    """
    chunk = _chunk_with_tool_call_index(index)
    tool_call = chunk.choices[0].delta.tool_calls[0]
    key, merged = merge_tool_call_fragment({}, {"index": tool_call.index, "id": tool_call.id})
    # 不可哈希的 index 等价于「没有 index」，按既有优先级退回 id 键。
    assert key == ("id", "a")
    assert merged["id"] == "a"
    # 没有 id 可退时才落到位置键——两条都不得抛异常。
    key, _ = merge_tool_call_fragment({}, {"index": tool_call.index})
    assert key == ("position", 0)


def test_merge_tool_call_fragment_keeps_hashable_index_verbatim():
    """收紧不能误伤：可哈希的 index 仍然作为稳定键。"""
    key, _ = merge_tool_call_fragment({}, {"index": 0})
    assert key == ("index", 0)
    key, _ = merge_tool_call_fragment({}, {"index": "s"})
    assert key == ("index", "s")


def test_merge_response_tool_call_tolerates_unhashable_identity():
    """Responses 路径的 ``_merge_response_tool_call`` 与同文件既有守卫对齐。"""
    for tool_call in ({"index": [1, 2]}, {"call_id": ["x"]}, {"index": {}}):
        key, _ = OpenAICompatibleProvider._merge_response_tool_call({}, tool_call)
        assert key == ("position", 0)


@pytest.mark.parametrize("choices", [{"0": {"index": 0}}, 3, "abc"])
def test_chat_response_error_rejects_non_sequence_choices(choices):
    """``choices`` 非序列时必须报成 ``RuntimeError``，而不是裸 ``KeyError``/``TypeError``。"""
    response = {"choices": choices}
    with pytest.raises(RuntimeError, match="choices 必须是列表"):
        OpenAICompatibleProvider._raise_for_chat_response_error(response)


def test_chat_response_error_still_allows_usage_chunk_without_choices():
    """收紧不能误伤：流式的 usage chunk 不带 choices，仍按原语义放行。"""
    OpenAICompatibleProvider._raise_for_chat_response_error({}, allow_empty_choices=True)


@pytest.mark.parametrize("timeout", [1e306, "abc", float("inf"), 0, -1])
def test_request_timeout_is_validated_before_reaching_the_sdk(timeout):
    """请求级 ``timeout`` 与构造期 ``options.timeout`` 用同一判据。

    修复前非法值直达 httpx，SDK 把形态各异的异常统一包成
    ``APIConnectionError('Connection error.')``，而 ``is_retryable_error`` 按类名
    匹配到 ``connection`` 判定可重试——同一个注定失败的请求被重试 3 次。
    """
    provider = OpenAICompatibleProvider(
        "m",
        "openai",
        settings=SimpleNamespace(openai_api_key="k", openai_base_url="http://127.0.0.1:1/v1"),
    )
    with pytest.raises(ValueError):
        provider._build_chat_request(prompt="hi", timeout=timeout)


def test_request_timeout_accepts_httpx_timeout_object():
    """收紧不能误伤：``httpx.Timeout`` 对象原样保留（SDK 接受，TOML 无法表达）。"""
    provider = OpenAICompatibleProvider(
        "m",
        "openai",
        settings=SimpleNamespace(openai_api_key="k", openai_base_url="http://127.0.0.1:1/v1"),
    )
    timeout = httpx.Timeout(5.0)
    assert provider._build_chat_request(prompt="hi", timeout=timeout)["timeout"] is timeout


@pytest.mark.parametrize(
    "tools",
    [
        [{"function": "not-an-object"}],
        [{"name": "f", "function": None}],
        [{"type": ["google_search"]}],
        [{"type": {"a": 1}}],
        ["oops"],
    ],
)
def test_google_convert_tools_rejects_malformed_definitions(tools):
    """Google 的畸形工具定义必须报成 ``ValueError``（与 Anthropic 同形态一致）。"""
    with pytest.raises(ValueError, match=r"Google 工具定义\[0\]"):
        GoogleProvider._convert_tools(tools)


def test_google_convert_tools_still_accepts_valid_definitions():
    """收紧不能误伤：合法的 function schema 与原生工具照旧通过。"""
    converted = GoogleProvider._convert_tools(
        [{"function": {"name": "lookup", "parameters": []}}, {"type": "google_search"}]
    )
    assert len(converted) == 2


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
        pytest.param(["RRF"], id="list"),
        pytest.param(None, id="none"),
        pytest.param(123, id="int"),
    ],
)
def test_display_and_menu_render_fusion_strategy_beyond_conversion_limit(value):
    """融合策略的显示路径不得因非字符串或位数超限而崩。

    ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫，嵌入方或携带
    超大整数的 pickle 快照可以直写该字段。修复前裸 ``.upper()`` 抛
    ``AttributeError``、裸 ``str()`` 抛 ``ValueError: Exceeds the limit``。
    """
    from src.ui.display_utils import _render_value

    config = _session_config()
    config.hybrid_fusion_strategy = value
    assert _render_value(config.hybrid_fusion_strategy).upper()


def test_render_value_keeps_normal_strings_verbatim():
    """收紧不能误伤：正常字符串的显示形态不变（不加引号）。"""
    from src.ui.display_utils import _render_value

    assert _render_value("rrf").upper() == "RRF"


@pytest.mark.parametrize("key,value", [("top_k", 5), ("retrieval_candidate_multiplier", 2)])
def test_setitem_rejects_non_integer_sibling_before_comparison(key, value):
    """``__setitem__`` 读兄弟字段时也要收敛，与 ``set_many`` 对齐。

    修复前 ``getattr`` 拿到 list/str 后 ``top_k * multiplier`` 与随后的 ``>``
    比较抛裸 ``TypeError``，穿透调用方按 ``ValueError`` 建立的分类。
    """
    sibling = "retrieval_candidate_multiplier" if key == "top_k" else "top_k"
    config = _session_config()
    setattr(config, sibling, [1, 2])
    with pytest.raises(ValueError, match="必须是整数"):
        config[key] = value


def test_setitem_rejects_nan_sibling_weight():
    """``nan == 0.0`` 恒为 False，不收敛兄弟字段会静默放行两个权重同时失效。"""
    config = _session_config()
    config.keyword_weight = float("nan")
    with pytest.raises(ValueError, match="必须是有限数值"):
        config["vector_weight"] = 0.0


def test_setitem_still_accepts_legal_values():
    """收紧不能误伤：合法写入照旧生效。"""
    config = _session_config()
    config["top_k"] = 7
    assert config.top_k == 7
    config["vector_weight"] = 0.4
    assert config.vector_weight == 0.4


@pytest.mark.parametrize("options", [["http_options"], [], "http_options", 5])
def test_capability_report_rejects_non_mapping_options(options):
    """公开诊断入口对非 Mapping 的 options 必须与其它渠道同分类（``ValueError``）。

    修复前 google 分支的 ``"http_options" in options`` 对整数抛裸 ``TypeError``、
    命中后 ``options.items()`` 抛裸 ``AttributeError``；非 google 渠道走 else
    分支本来就得到 ``ValueError``。
    """
    with pytest.raises(ValueError, match="options 必须是对象"):
        ModelProviderFactory.capability_report("google", "llm", options)


def test_capability_report_still_accepts_mapping_options():
    """收紧不能误伤：合法的 options 映射照旧通过。"""
    report = ModelProviderFactory.capability_report("google", "llm", {})
    assert report["provider"] == "google"


def test_deeply_nested_toml_reports_redacted_value_error(tmp_path):
    """深层嵌套的 TOML 必须收敛成脱敏 ``ValueError``，而不是穿透的 ``RecursionError``。

    ``RecursionError`` 继承 ``RuntimeError`` 而**不是** ``ValueError``，修复前
    ``get_settings`` 的四个 handler 一个都接不住——``import main`` 直接以数千帧
    裸 traceback 终止。
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text("a = " + "[" * 500 + "]" * 500, encoding="utf-8")
    # ``get_settings`` 带 ``lru_cache``：不清缓存的话这次调用直接返回上一个用例
    # 缓存的 Settings，根本不会去读这份坏配置。
    config_module.get_settings.cache_clear()
    try:
        with (
            patch.object(config_module, "CONFIG_TOML_PATH", config_file),
            pytest.raises(ValueError, match="配置解析失败"),
        ):
            config_module.get_settings()
    finally:
        config_module.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 二、读写 / 入口校验不对称
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arguments", [{"a": 1}, [1, 2], 7])
def test_streaming_tool_arguments_match_the_non_streaming_form(arguments):
    """流式 ``arguments`` 要与 ``_extract_tool_calls`` 一样归一成 JSON 字符串。

    修复前同一个输入在两条路径上给出不同类型：流式交付 dict/list/int，非流式
    交付 JSON 文本，调用方按字符串 ``json.loads`` 时拿到 dict 并抛 ``TypeError``。
    """
    chunk = construct_type(
        type_=ChatCompletionChunk,
        value={
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "x",
                                "function": {"name": "f", "arguments": arguments},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
    )
    events = OpenAICompatibleProvider._chat_stream_events(chunk)
    accumulator: dict = {}
    for event in events:
        if event.type == "tool_call_delta":
            merge_tool_call_fragment(accumulator, event.tool_call)
    delivered = next(iter(accumulator.values()))["arguments"]
    assert isinstance(delivered, str)


def test_streaming_tool_argument_fragments_still_concatenate():
    """收紧不能误伤：字符串分片的拼接语义不变。"""
    accumulator: dict = {}
    for fragment in ('{"a":', "1}"):
        chunk = construct_type(
            type_=ChatCompletionChunk,
            value={
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "x",
                                    "function": {"name": "f", "arguments": fragment},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
        )
        for event in OpenAICompatibleProvider._chat_stream_events(chunk):
            if event.type == "tool_call_delta":
                merge_tool_call_fragment(accumulator, event.tool_call)
    assert next(iter(accumulator.values()))["arguments"] == '{"a":1}'


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param(True, id="bool"),
        pytest.param(10**400, id="beyond-double"),
    ],
)
def test_google_options_max_tokens_is_validated(value):
    """``options.max_tokens`` 与 ``max_output_tokens`` 必须同一判据。

    修复前这条赋值绕过了 ``_validate_generation_count``，非法值原样进入
    ``GenerateContentConfig``（pydantic 对 ``-1``／``True``／超大整数都不报错）。
    """
    provider = GoogleProvider("gemini-2.0-flash", {"max_tokens": value})
    with pytest.raises(ValueError, match="max_tokens"):
        provider._build_generation_config(None, None, None)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(0, id="zero"),
        pytest.param("abc", id="string"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(10**400, id="beyond-double"),
    ],
)
def test_google_embedding_dimensionality_is_validated(value):
    """``output_dimensionality`` 此前只过键名白名单，取值原样透传。"""
    provider = GoogleProvider("text-embedding-004", {"output_dimensionality": value})
    with pytest.raises(ValueError, match="output_dimensionality"):
        provider._embedding_options()


def test_google_embedding_dimensionality_accepts_legal_value():
    """收紧不能误伤：合法维度照旧通过，未设该键时不受影响。"""
    provider = GoogleProvider("text-embedding-004", {"output_dimensionality": 768})
    assert provider._embedding_options()["output_dimensionality"] == 768
    assert (
        "output_dimensionality" not in GoogleProvider("m", {"task_type": "x"})._embedding_options()
    )


def test_parent_splitter_rejects_overlap_equal_to_size():
    """父分片器的 ``overlap < size`` 与子分片器同一判据，两种结构模式都要过。

    修复前 ``_validate_hierarchical_sizes`` 只校验**子**分片；``Settings`` 的
    ``overlap < size`` 拦不住直接构造 ``RecursiveTextSplitter`` 的调用方，而
    LangChain 对 ``overlap == size`` **静默接受**（实测 size=100/ov=100 通过、
    ov=101 抛 ValueError）——于是 standard 分支既不报错也不推进。
    """
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(
        parent_chunk_size=1500, parent_chunk_overlap=1500, structure_mode="standard"
    )
    with pytest.raises(ValueError, match="kb_chunk_overlap 必须小于 kb_chunk_size"):
        splitter._validate_parent_sizes()


def test_parent_splitter_accepts_legal_overlap():
    """收紧不能误伤：默认比例照旧通过。"""
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    RecursiveTextSplitter(
        parent_chunk_size=1500, parent_chunk_overlap=150, structure_mode="standard"
    )._validate_parent_sizes()


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([[float("nan"), 0.1]], dtype=np.float32),
        np.array([[float("inf"), 0.1]], dtype=np.float32),
        np.array([[1e300, 0.1]], dtype=np.float64),
    ],
)
def test_upsert_embeddings_rejects_non_finite_vectors(embeddings):
    """写入侧要与载入侧同一判据。

    修复前只有载入侧 ``_validate_embeddings_shape`` 校验，写入侧放行——FAISS 把
    NaN/inf 照单收下（``ntotal`` 正常增加），那个下标在查询时静默消失，而
    ``save_snapshot`` 会把它落盘。
    """
    store = FaissStore()
    with pytest.raises(ValueError, match="非有限值"):
        store.upsert_embeddings([{"page_content": "x", "metadata": {}}], embeddings)
    # 失败必须不留痕。
    assert store.documents == []
    assert store.embeddings is None


def test_upsert_embeddings_accepts_finite_vectors():
    """收紧不能误伤：有限向量照旧写入。"""
    store = FaissStore()
    store.upsert_embeddings(
        [{"page_content": "x", "metadata": {}}], np.array([[0.1, 0.2]], dtype=np.float32)
    )
    assert len(store.documents) == 1


def _write_snapshot(root: Path, snapshot_id: str, document_count: int, index_rows: int) -> Path:
    """造一个已激活的快照目录，``semantic.index`` 的行数可控。"""
    snapshot = root / snapshot_id
    snapshot.mkdir()
    documents = [
        {"page_content": f"d{i}", "metadata": {"chunk_id": f"c{i}"}} for i in range(document_count)
    ]
    embeddings = np.arange(document_count * 2, dtype=np.float32).reshape(document_count, 2)
    index = faiss.IndexFlatL2(2)
    if index_rows:
        index.add(np.arange(index_rows * 2, dtype=np.float32).reshape(index_rows, 2))
    with (snapshot / "chunks.pkl").open("wb") as file:
        pickle.dump(documents, file)
    with (snapshot / "parents.pkl").open("wb") as file:
        pickle.dump({}, file)
    np.save(snapshot / "embeddings.npy", embeddings)
    faiss.write_index(index, str(snapshot / "semantic.index"))
    with (snapshot / "lexical.index").open("wb") as file:
        pickle.dump([[f"d{i}"] for i in range(document_count)], file)
    (root / "ACTIVE_SNAPSHOT").write_text(snapshot_id, encoding="utf-8")
    return snapshot


@pytest.mark.parametrize("index_rows", [1, 3])
def test_load_snapshot_rejects_semantic_index_row_mismatch(tmp_path, index_rows):
    """``semantic.index`` 的行数是第三条必须与 ``documents`` 对齐的量。

    修复前只校验 ``lexical.index`` 的条数。索引少一条时第 2 篇文档永远搜不到且
    无任何提示；多一条时 ``self.documents[doc_index]`` 抛 ``IndexError``。
    """
    snapshot = _write_snapshot(
        tmp_path, f"kb-{index_rows}", document_count=2, index_rows=index_rows
    )
    with pytest.raises(ValueError, match="语义索引条数与分块数不一致"):
        FaissStore().load_snapshot(str(snapshot), snapshot_root=str(tmp_path))


def test_load_snapshot_accepts_self_consistent_snapshot(tmp_path):
    """收紧不能误伤：自洽的快照照旧加载。"""
    snapshot = _write_snapshot(tmp_path, "kb-ok", document_count=2, index_rows=2)
    store = FaissStore()
    store.load_snapshot(str(snapshot), snapshot_root=str(tmp_path))
    assert len(store.documents) == 2
    assert store.faiss_index.ntotal == 2


# ---------------------------------------------------------------------------
# 三、静默语义破坏 / 数据丢失
# ---------------------------------------------------------------------------


def test_vertex_partial_args_fragments_merge_without_stable_identity():
    """只有 ``will_continue``、无 id/index 的分片必须能归并。

    这是 Vertex ``stream_function_call_arguments`` 模式的真实形状（google-genai
    自带 fixture 记录）：分片1 带 ``will_continue=True``，分片2 带 ``partial_args``
    与 ``will_continue=False``，两者都没有 id/index。修复前 ``_stream_events_from_chunk``
    构造 delta 字典时漏写 ``will_continue``，读侧三项判据全假，抛
    ``RuntimeError: 缺少稳定 id/index``——该模式下**每一个**多分片调用都失败。
    """
    from google.genai import types

    chunks = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="get_current_weather", will_continue=True
                                )
                            )
                        ],
                    )
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="get_current_weather",
                                    partial_args=[
                                        types.PartialArg(
                                            json_path="$.country", null_value="NULL_VALUE"
                                        )
                                    ],
                                    will_continue=False,
                                )
                            )
                        ],
                    )
                )
            ]
        ),
    ]
    accumulator: dict = {}
    for chunk in chunks:
        for event in GoogleProvider._stream_events_from_chunk(chunk):
            if event.type == "tool_call_delta":
                GoogleProvider._merge_tool_call(accumulator, event.tool_call)
    assert len(accumulator) == 1
    assert next(iter(accumulator.values()))["arguments"] == {"country": None}


@pytest.mark.parametrize("index", [100_001, 20_000_000])
def test_partial_args_json_path_index_is_bounded(index):
    """``json_path`` 的数组下标必须有上界。

    写入循环用 ``while len(current) <= token: current.append(None)`` 填充到该下标，
    因此这个值直接决定一次分片要分配多少个元素：实测 ``$.items[20000000]`` 分配
    2000 万个 ``None``（RSS 峰值 +76MB、耗时 6.5 秒）。
    """
    with pytest.raises(ValueError, match="索引过大"):
        GoogleProvider._partial_args_to_mapping(
            [{"json_path": f"$.items[{index}]", "string_value": "x"}]
        )


@pytest.mark.parametrize("index", [1, 100_000])
def test_partial_args_json_path_accepts_legal_index(index):
    """收紧不能误伤：上界之内的下标照旧写入。"""
    result = GoogleProvider._partial_args_to_mapping(
        [{"json_path": f"$.items[{index}]", "string_value": "x"}]
    )
    assert len(result["items"]) == index + 1


def test_toml_options_payload_keys_are_not_lowercased(tmp_path):
    """``options`` 下的用户载荷键名必须原样保留。

    修复前每一层嵌套字典的键都被小写化，``required = ["OrderId"]`` 与
    ``properties = { OrderId = ... }`` 被拆成 ``["OrderId"]`` 与 ``["orderid"]``，
    schema 的 required 与 properties 不再对应，结构化输出静默失效。
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[llm_configurations.openai]
model_name = "gpt-4o"
[llm_configurations.openai.options]
MAX_TOKENS = 100
response_format = { type = "json_schema", json_schema = { schema = { required = ["OrderId"], properties = { OrderId = { type = "string" } } } } }
extra_body = { Nested = { DeepKey = true } }
""",
        encoding="utf-8",
    )
    with patch.object(config_module, "CONFIG_TOML_PATH", config_file):
        raw = config_module.load_toml_config()
    options = raw["llm_configurations"]["openai"]["options"]
    # 选项名是项目/SDK 词汇，照旧归一（大小写不敏感）。
    assert "max_tokens" in options
    schema = options["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["OrderId"]
    assert list(schema["properties"]) == ["OrderId"]
    assert all(key in schema["properties"] for key in schema["required"])
    # 取值载荷的嵌套键同样保留。
    assert options["extra_body"] == {"Nested": {"DeepKey": True}}


def test_toml_entry_names_are_still_preserved(tmp_path):
    """收紧不能误伤：``llm_configurations`` 的条目名仍按用户命名保留。"""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[llm_configurations.MyEntry]\nprovider = "openai"\nmodel_name = "m"\n',
        encoding="utf-8",
    )
    with patch.object(config_module, "CONFIG_TOML_PATH", config_file):
        raw = config_module.load_toml_config()
    assert "MyEntry" in raw["llm_configurations"]


@pytest.mark.parametrize("value", [1.9, "1_0", "+7", "2.5"])
def test_candidate_multiplier_matches_chat_top_k_parsing(value):
    """``retrieval_candidate_multiplier`` 与 ``chat_top_k`` 用同一套解析。

    修复前这里直接 ``int(v)``：``1.9`` 被静默截断成 ``1``，``"1_0"`` 被接受为 10、
    ``"+7"`` 被接受为 7——而 ``parse_decimal_int`` 的 docstring 明确写着这两类
    「安静的错值」要明确拒绝。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(retrieval_candidate_multiplier=value)


def test_candidate_multiplier_accepts_legal_values():
    """收紧不能误伤：纯十进制字面量与整数照旧通过。"""
    assert Settings(retrieval_candidate_multiplier=7).retrieval_candidate_multiplier == 7
    assert Settings(retrieval_candidate_multiplier="7").retrieval_candidate_multiplier == 7
    assert Settings(retrieval_candidate_multiplier=2.0).retrieval_candidate_multiplier == 2


def test_key_text_distinguishes_integers_beyond_conversion_limit():
    """去重键不得把不同的超限整数塌成同一个字符串。

    修复前 ``safe_repr`` 的降级描述只保留量级（``<整数，约 4302 位十进制>``），
    ``10**4301`` 与 ``10**4301 + 1`` 得到同一个键，``_merge_hybrid_results`` 于是
    把两条不同的文档并成一条，其中一条静默消失。
    """
    first = _key_text(_HUGE_INT)
    second = _key_text(_HUGE_INT + 1)
    assert first != second
    # 同一取值必须稳定，否则去重本身失效。
    assert _key_text(_HUGE_INT) == first
    # 符号要编进去，``n`` 与 ``-n`` 不得碰撞。
    assert _key_text(-_HUGE_INT) != first


def test_key_text_keeps_normal_values_verbatim():
    """收紧不能误伤：正常取值的键形态不变。"""
    assert _key_text("s") == "s"
    assert _key_text(5) == "5"
    assert _key_text(10**400) == safe_repr(10**400)


@pytest.mark.parametrize("raw", ["RRF", " rrf ", "Rrf"])
def test_fusion_strategy_normalizes_before_the_rrf_check(raw):
    """``"RRF"`` 必须与 ``"rrf"`` 得到同一结论。

    修复前 ``HybridReranker`` 会 ``.strip().lower()``，而
    ``_should_apply_score_threshold`` 用**原始**取值与字面量 ``"rrf"`` 比较：
    ``"RRF"`` 走 RRF 融合（分数是排名倒数之和）却不被识别为 RRF，阈值过滤照常
    套用——两种量纲混在一起，结果被静默清空。
    """
    assert _coerce_fusion_strategy(raw) == "rrf"
    config = _session_config()
    config.hybrid_fusion_strategy = raw
    normalized = _coerce_fusion_strategy(config.hybrid_fusion_strategy)
    assert _SessionConfigThresholdCheck(config, normalized) is True


class _SessionConfigThresholdCheck:
    """调用 ``RetrievalService._should_apply_score_threshold`` 的可读包装。"""

    def __new__(cls, config, fusion_strategy):
        from src.services.retrieval_service import RetrievalService

        return not RetrievalService._should_apply_score_threshold(
            config, fusion_strategy=fusion_strategy
        )


def test_fusion_strategy_normalization_keeps_weighted():
    """收紧不能误伤：``weighted`` 仍照常套用阈值过滤。"""
    from src.services.retrieval_service import RetrievalService

    assert _coerce_fusion_strategy("WEIGHTED") == "weighted"
    config = _session_config()
    assert (
        RetrievalService._should_apply_score_threshold(config, fusion_strategy="weighted") is True
    )


def test_cleanup_old_logs_tolerates_log_path_being_a_file(tmp_path):
    """``log_path`` 是已存在的普通文件时不得抛 ``NotADirectoryError``。

    写入侧对此的处置是降级为仅控制台，而清理侧由退出钩子调用——异常会让程序在
    退出路径上以裸 traceback 终止。
    """
    log_path = tmp_path / "not_a_dir.txt"
    log_path.write_text("x", encoding="utf-8")
    settings = _log_settings(str(log_path), retention_days=7)

    with patch("src.utils.log_manager.get_settings", return_value=settings):
        cleanup_old_logs()  # 不应抛异常


def test_cleanup_old_logs_still_removes_expired_files(tmp_path):
    """收紧不能误伤：过期的本应用日志照旧被删。"""
    from datetime import datetime, timedelta

    old_date = (datetime.now().astimezone() - timedelta(days=30)).strftime("%Y-%m-%d")
    expired = tmp_path / f"app_log_{old_date}.log"
    expired.write_text("x", encoding="utf-8")

    settings = _log_settings(str(tmp_path), retention_days=7)

    with patch("src.utils.log_manager.get_settings", return_value=settings):
        cleanup_old_logs()
    assert not expired.exists()


def test_get_settings_still_returns_settings():
    """收紧不能误伤：正常配置加载路径不受影响。"""
    assert get_settings() is not None


# ---------------------------------------------------------------------------
# 四、volcengine / anthropic 渠道的边界缺陷（第二轮审查）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_type",
    [
        "web_search_20250305",
        "code_execution_20250522",
        "text_editor_20250728",
        "memory_20250818",
        "web_fetch_20250910",
        "browser_toolset_20260801",
        "tool_search_tool_bm25",
        "computer_toolset_20260801",
        # 已从当前 SDK 联合类型移除、但仍在服务端生效的旧字面量
        "computer_20250124",
        "bash_20250124",
        "text_editor_20250124",
        "server_custom",
        "custom",
    ],
)
def test_anthropic_server_tools_pass_through_verbatim(tool_type):
    """服务端工具必须原样透传，不得被降级成空 schema 的普通函数工具。

    白名单原先只认 ``server_`` 前缀与三个 2025-01-24 时代的字面量，而 SDK 里
    真实存在二十种服务端工具类型——它们被降级后 ``max_uses`` 等专属字段被丢弃，
    服务端收到自定义函数工具而非服务端检索工具，用户看到 web_search 静默失效。
    """
    from src.providers.anthropic import AnthropicProvider

    tool = {"type": tool_type, "name": "n", "max_uses": 5}
    assert AnthropicProvider._convert_tools([tool])[0] == tool


def test_anthropic_server_tool_types_track_the_sdk():
    """推导集合必须跟随 SDK，而不是写死的白名单。

    SDK 升级新增工具时集合自动跟随——写死白名单的失效模式正是「SDK 加了新工具、
    本地静默降级」。
    """
    import typing

    import anthropic

    from src.providers.anthropic import _SERVER_TOOL_TYPES

    derived: set[str] = set()
    for member in typing.get_args(anthropic.types.ToolUnionParam):
        if member is anthropic.types.ToolParam:
            continue
        for literal in typing.get_args(typing.get_type_hints(member).get("type")):
            if isinstance(literal, str):
                derived.add(literal)
    assert derived, "SDK 联合类型里应当能推导出服务端工具类型"
    assert derived <= _SERVER_TOOL_TYPES


def test_anthropic_function_tools_still_become_schemas():
    """收紧不能误伤：普通函数工具仍走 schema 转换分支。"""
    from src.providers.anthropic import AnthropicProvider

    converted = AnthropicProvider._convert_tools(
        [{"function": {"name": "lookup", "parameters": {"type": "object"}}}]
    )
    assert converted[0]["name"] == "lookup"
    assert "input_schema" in converted[0]


@pytest.mark.parametrize("value", ["json_object", 42, ["a"]])
def test_anthropic_response_format_rejects_non_mapping(value):
    """``response_format`` 的两个来源必须同判据。

    ``options.response_format`` 那一侧早有守卫，但**请求参数**来源直接进
    ``_response_format_config``，非 Mapping 时 ``.get()`` 抛裸 ``AttributeError``。
    """
    from src.providers.anthropic import AnthropicProvider

    with pytest.raises(ValueError, match="response_format 必须是对象"):
        AnthropicProvider._response_format_config(value)


@pytest.mark.parametrize("value", ["json_object", 42, ["a"]])
def test_ark_response_format_rejects_non_mapping(value):
    """Ark Responses 同一语义的同一判据（与 google.py 对齐）。"""
    from src.providers.volcengine import VolcengineProvider

    with pytest.raises(ValueError, match="response_format 必须是对象"):
        VolcengineProvider._convert_responses_format(value)


def test_response_format_still_converts_legal_values():
    """收紧不能误伤：合法的 ``json_object``／``json_schema`` 照旧转换。"""
    from src.providers.anthropic import AnthropicProvider
    from src.providers.volcengine import VolcengineProvider

    assert AnthropicProvider._response_format_config({"type": "json_object"}) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
    assert VolcengineProvider._convert_responses_format({"type": "json_object"}) == {
        "format": {"type": "json_object"}
    }


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(10**400, id="beyond-double"),
        pytest.param("abc", id="string"),
        pytest.param([1], id="list"),
        pytest.param(1.5, id="float"),
        pytest.param(-5, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(0, id="zero"),
    ],
)
def test_anthropic_request_max_tokens_is_validated(value, monkeypatch):
    """请求级 ``max_tokens`` 与 ``options.max_tokens`` 必须同一判据。

    修复前请求级取值原样进 params，由 SDK 深处的
    ``_calculate_nonstreaming_timeout`` 执行 ``maximum_time * max_tokens / 128_000``
    时才炸：``10**400`` 抛 ``OverflowError``（``ArithmeticError`` 子类，既不落
    ``except ValueError`` 也不落 ``except RuntimeError``），``"abc"`` 抛
    ``TypeError``，而 ``1.5``／``-5``／``True`` 静默通过、原样发给服务端。
    """
    _patch_provider_settings(monkeypatch, "src.providers.anthropic", anthropic_api_key="test-key")
    from src.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-sonnet-4-5")
    with pytest.raises(ValueError, match="必须是正整数"):
        provider._build_message_params(prompt="hi", max_tokens=value, stream=False)


def test_anthropic_request_max_tokens_accepts_positive_integers(monkeypatch):
    """收紧不能误伤：合法取值照旧通过；未提供时用模型默认。"""
    _patch_provider_settings(monkeypatch, "src.providers.anthropic", anthropic_api_key="test-key")
    from src.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-sonnet-4-5")
    assert (
        provider._build_message_params(prompt="hi", max_tokens=4096, stream=False)["max_tokens"]
        == 4096
    )
    assert provider._build_message_params(prompt="hi", stream=False)["max_tokens"] > 0


@pytest.mark.parametrize(
    "block",
    [
        pytest.param({"type": "refusal", "refusal": "被拒绝"}, id="refusal-key"),
        pytest.param({"type": "refusal", "reason": "被拒绝"}, id="reason-key"),
        pytest.param({"type": "refusal", "text": "被拒绝"}, id="text-key"),
        pytest.param(
            {"type": "refusal", "refusal": "A", "text": "B", "reason": "C"}, id="priority"
        ),
    ],
)
def test_anthropic_refusal_block_matches_between_streaming_and_complete(block):
    """非流式与流式对同一 refusal 块必须给出同一结果。

    非流式原先只读 ``reason``／``text``，漏了 SDK ``RefusalBlock`` 的正式字段名
    ``refusal``——于是 ``[{"type": "refusal", "refusal": "被拒绝"}]`` 在非流式得到
    ``refusal=None``、``text=''``、``finish_reason='end_turn'``，``invoke()`` 输出
    空列表：一次**安全拦截**变成「空成功」。
    """
    from anthropic._models import construct_type
    from anthropic.types import Message

    from src.providers.anthropic import AnthropicProvider

    response = construct_type(
        type_=Message,
        value={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [block],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    non_streaming = AnthropicProvider._extract_result(response).refusal
    event = AnthropicProvider._stream_event(
        {"type": "content_block_start", "index": 0, "content_block": block}, "m"
    )
    assert non_streaming == (event.refusal if event else None)
    assert non_streaming == "被拒绝" or non_streaming == "A"


def test_anthropic_non_refusal_block_is_unaffected():
    """收紧不能误伤：普通 text 块不产生 refusal。"""
    from anthropic._models import construct_type
    from anthropic.types import Message

    from src.providers.anthropic import AnthropicProvider

    response = construct_type(
        type_=Message,
        value={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    result = AnthropicProvider._extract_result(response)
    assert result.text == "hello"
    assert result.refusal is None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-args"),
        pytest.param({"store": True}, id="store-only"),
        pytest.param({"text": {"format": {"type": "json_object"}}}, id="text-only"),
    ],
)
def test_ark_dynamic_create_validates_without_trigger_fields(kwargs, monkeypatch):
    """Ark 动态创建路径在「一个触发字段都没有」时也必须校验。

    修复前校验被整体跳过，请求直达 SDK 撞上 ``_utils/_utils.py`` 的
    ``assert "model" in kwargs``，抛 **``AssertionError``**——既不落
    ``except ValueError`` 也不落 ``except RuntimeError``。
    """
    _patch_provider_settings(
        monkeypatch,
        "src.providers.volcengine",
        ark_api_key="test-key",
        volc_access_key=None,
        volc_secret_key=None,
        volc_base_url="https://ark.invalid/v3",
    )
    from src.providers.volcengine import VolcengineProvider

    provider = VolcengineProvider(
        "m", options={"server_verified_protocols": ["responses"]}, protocol="responses"
    )
    with pytest.raises(ValueError, match="必须显式提供 model"):
        provider._require_provider_resource("responses.create", kwargs)


def test_ark_dynamic_non_create_calls_are_not_forced_to_supply_model(monkeypatch):
    """收紧不能误伤：retrieve／delete／list 不得被要求提供 ``model``。"""
    _patch_provider_settings(
        monkeypatch,
        "src.providers.volcengine",
        ark_api_key="test-key",
        volc_access_key=None,
        volc_secret_key=None,
        volc_base_url="https://ark.invalid/v3",
    )
    from src.providers.volcengine import VolcengineProvider

    provider = VolcengineProvider(
        "m", options={"server_verified_protocols": ["responses"]}, protocol="responses"
    )
    for method in ("responses.retrieve", "responses.delete", "responses.list"):
        provider._require_provider_resource(method, {})


def test_ark_dynamic_create_with_full_args_passes(monkeypatch):
    """收紧不能误伤：完整参数的创建调用照旧通过。"""
    _patch_provider_settings(
        monkeypatch,
        "src.providers.volcengine",
        ark_api_key="test-key",
        volc_access_key=None,
        volc_secret_key=None,
        volc_base_url="https://ark.invalid/v3",
    )
    from src.providers.volcengine import VolcengineProvider

    provider = VolcengineProvider(
        "m", options={"server_verified_protocols": ["responses"]}, protocol="responses"
    )
    provider._require_provider_resource("responses.create", {"model": "m", "input": "x"})


# ---------------------------------------------------------------------------
# 五、分片重叠的比例上界（第 12 条）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlap", [751, 1490, 1499])
def test_settings_rejects_overlap_beyond_half_of_size(overlap):
    """overlap 超过 size 的一半时必须拒绝。

    ``overlap < size`` 拦不住「接近但不相等」这一类，而它同样静默：实测一篇
    15000 字符的文档在 size=1500 时，overlap=1490（99%）产出 1651 个分片
    （137.6x）、overlap=1499 产出 13751 个（1145.9x）——全部进入 embedding 与
    FAISS 写入。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="不能超过 kb_chunk_size 的一半"):
        Settings(kb_chunk_size=1500, kb_chunk_overlap=overlap)


def test_settings_accepts_overlap_at_exactly_half(settings_factory=None):
    """收紧不能误伤：正好一半是合法上界。"""
    assert Settings(kb_chunk_size=1500, kb_chunk_overlap=750).kb_chunk_overlap == 750
    assert Settings(kb_chunk_size=100, kb_chunk_overlap=50).kb_chunk_overlap == 50


def test_splitter_rejects_overlap_beyond_half_of_size():
    """分片器层同判据——它是公开类，调用方可以绕过 ``Settings`` 直接构造。"""
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    for mode in ("standard", "hierarchical"):
        splitter = RecursiveTextSplitter(
            parent_chunk_size=1500, parent_chunk_overlap=1499, structure_mode=mode
        )
        with pytest.raises(ValueError, match="不能超过 kb_chunk_size 的一半"):
            if mode == "hierarchical":
                splitter._validate_hierarchical_sizes()
            else:
                splitter._validate_parent_sizes()


def test_splitter_accepts_overlap_within_the_bound():
    """收紧不能误伤：默认比例与正好一半都照旧可用。"""
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
    from src.models.document import Document

    for overlap in (150, 750):
        splitter = RecursiveTextSplitter(
            parent_chunk_size=1500, parent_chunk_overlap=overlap, structure_mode="standard"
        )
        splitter._validate_parent_sizes()
        assert splitter.split([Document(content="句子内容。" * 3000)])
