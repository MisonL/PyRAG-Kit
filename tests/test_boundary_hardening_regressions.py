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


# ---------------------------------------------------------------------------
# 六、请求头取值与 embedding 配置取值的校验（第三轮审查）
# ---------------------------------------------------------------------------


def _capture_anthropic_headers(provider, *, beta: bool) -> dict[str, object]:
    """替换 provider 的 ``count_tokens`` 客户端，捕获它实际收到的参数。

    客户端是懒加载的（构造期 ``_client`` 为 ``None``），因此要整体替换
    ``_get_client``／``_get_aclient`` 而不是给 ``provider._client`` 打属性——
    后者对 ``None`` 抛 ``AttributeError: 'NoneType' object has no attribute
    'messages'``。仓库既有的协议适配测试用的是同一种注入方式。
    """
    captured: dict[str, object] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(input_tokens=3)

    async def _acapture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(input_tokens=3)

    # ``create`` 与 ``count_tokens`` 共用同一个捕获函数：测试关心的是**发给 SDK
    # 的参数**，入口不同不改变判据。
    client = SimpleNamespace(
        messages=SimpleNamespace(count_tokens=_capture, create=_capture),
        beta=SimpleNamespace(messages=SimpleNamespace(count_tokens=_capture, create=_capture)),
    )
    aclient = SimpleNamespace(
        messages=SimpleNamespace(count_tokens=_acapture, create=_acapture),
        beta=SimpleNamespace(messages=SimpleNamespace(count_tokens=_acapture, create=_acapture)),
    )
    provider._get_client = lambda: client
    provider._get_aclient = lambda: aclient
    return captured


def _anthropic_provider(monkeypatch):
    """走真实构造器拿到 provider，避免手工拼内部属性时漏字段。"""
    _patch_provider_settings(monkeypatch, "src.providers.anthropic", anthropic_api_key="test-key")
    from src.providers.anthropic import AnthropicProvider

    return AnthropicProvider("claude-sonnet-4-5")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(42, id="int"),
        pytest.param(b"x", id="bytes"),
        pytest.param(["a"], id="list"),
        pytest.param({"a": 1}, id="dict"),
    ],
)
def test_anthropic_rejects_non_string_header_values(monkeypatch, value):
    """``extra_headers`` 的**取值**必须校验，不能只校验容器。

    ``validate_secret_free_request_overrides`` 只确认 ``extra_headers`` 是 Mapping，
    值原样保留。非字符串值的两种收场都不在渠道的错误契约内：交给 httpx 时抛
    ``TypeError: Header value must be str or bytes``；而本模块对 ``anthropic-beta``
    做 ``.split(",")`` 时抛 ``AttributeError: 'int' object has no attribute 'split'``
    ——两者都不落 ``except ValueError``／``except RuntimeError``。
    """
    provider = _anthropic_provider(monkeypatch)
    with pytest.raises(ValueError, match="extra_headers"):
        provider.count_tokens(prompt="hi", user="u1", extra_headers={"anthropic-beta": value})


def test_anthropic_drops_none_header_values(monkeypatch):
    """``None`` 头值等价于「该头未设置」，必须剔除而不是透传。

    httpx 对 ``None`` 头值抛 ``TypeError``，SDK 更早一步抛
    ``AttributeError: 'NoneType' object has no attribute 'encode'``。
    """
    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=False)

    provider.count_tokens(
        prompt="hi", user="u1", extra_headers={"x-trace": None, "anthropic-beta": None}
    )
    headers = captured["extra_headers"]
    assert "x-trace" not in headers
    assert "None" not in headers.get("anthropic-beta", "")


def test_anthropic_user_without_extra_headers_still_works(monkeypatch):
    """``user`` 已设置而 ``extra_headers`` 为 ``None`` 时必须能合并出 beta 头。

    修复前这里对 ``None`` 做 ``headers["anthropic-beta"] = ...``，抛裸
    ``TypeError: 'NoneType' object does not support item assignment``。而
    ``extra_headers`` 不传恰恰是最常见的调用形态。
    """
    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=False)

    provider.count_tokens(prompt="hi", user="u1")
    assert captured["extra_headers"]["anthropic-beta"] == "user-profiles-2026-08-18"


@pytest.mark.parametrize("async_path", [False, True])
def test_anthropic_beta_count_tokens_uses_validated_headers(monkeypatch, async_path):
    """beta 计数入口必须用**校验后**的头，不能回退到原始的 ``request.extra_headers``。

    同文件 ``beta=False`` 的计数入口用的是 ``profile_headers``，而 ``beta=True``
    的两个入口把校验结果丢进 ``_``，转头写回原始字典——被剔除的 ``None`` 值于是
    原样发给 httpx。
    """
    import asyncio

    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=True)

    call = (
        provider.async_beta_count_tokens(prompt="hi", user="u1", extra_headers={"x-trace": None})
        if async_path
        else provider.beta_count_tokens(prompt="hi", user="u1", extra_headers={"x-trace": None})
    )
    if async_path:
        asyncio.run(call)

    assert "x-trace" not in captured["extra_headers"]


def test_anthropic_header_validation_does_not_touch_legal_headers(monkeypatch):
    """收紧不能误伤：合法字符串头（含既有 anthropic-beta 值）照旧保留并合并。"""
    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=False)

    provider.count_tokens(
        prompt="hi", user="u1", extra_headers={"x-trace": "t", "anthropic-beta": "a,b"}
    )
    headers = captured["extra_headers"]
    assert headers["x-trace"] == "t"
    assert headers["anthropic-beta"] == "a,b,user-profiles-2026-08-18"


# ── embedding 配置的取值校验 ──
#
# ``EmbeddingService.embedding_model_detail`` 原先只判**键**在不在，不判**值**
# 可用。``{"local-hash": None}`` 的键确实存在，于是 ``None`` 被原样返回，两个
# 调用点紧接着读 ``.provider``／``.model_name``，在那里抛裸 ``AttributeError``。
# 同一份配置走 ``ModelProviderFactory.get_embedding_provider`` 得到的是
# ``ValueError``——两条路径的报错契约本该一致。


def _embedding_service(configurations):
    from src.services.embedding_service import EmbeddingService

    return EmbeddingService(
        SimpleNamespace(
            default_embedding_provider="local-hash",
            embedding_configurations=configurations,
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param("local-hash-32", id="string"),
        pytest.param(5, id="int"),
        pytest.param(SimpleNamespace(provider="local-hash"), id="missing-model-name"),
        pytest.param(SimpleNamespace(model_name="m"), id="missing-provider"),
    ],
)
def test_embedding_model_detail_rejects_unusable_values(value):
    """取值不提供 ``provider``／``model_name`` 时必须抛 ``ValueError``。

    判据是**属性存在性**而非 ``isinstance(..., ModelDetail)``：使用方只要求取值
    具备这些属性，仓库测试就用 ``SimpleNamespace`` 构造等价配置。
    """
    with pytest.raises(ValueError, match="必须是提供"):
        _ = _embedding_service({"local-hash": value}).embedding_model_detail


def test_embedding_model_detail_accepts_attribute_bearing_values():
    """收紧不能误伤：``ModelDetail`` 与具备同名属性的替身都照旧可用。"""
    from src.utils.config import ModelDetail

    detail = ModelDetail(provider="local-hash", model_name="local-hash-32")
    assert _embedding_service({"local-hash": detail}).embedding_model_detail is detail

    stand_in = SimpleNamespace(provider="local-hash", model_name="m")
    assert _embedding_service({"local-hash": stand_in}).embedding_model_detail is stand_in


def test_knowledge_build_surfaces_unusable_embedding_detail_as_value_error():
    """建库主路径上同判据——错误分类必须是 ``ValueError``，不是裸 ``AttributeError``。

    零分块的 markdown 会跳过 embedding 调用，直接走到写清单那一步读
    ``model_name``；这是最晚才暴露、也最难从消息反推原因的路径。
    """
    from src.services.knowledge_build_service import KnowledgeBuildService

    service = KnowledgeBuildService.__new__(KnowledgeBuildService)
    service.embedding_service = _embedding_service({"local-hash": None})

    with pytest.raises(ValueError, match="必须是提供"):
        _ = service.embedding_service.embedding_model_detail.model_name


# ── user / betas 的取值校验 ──
#
# ``CompletionRequest`` 是普通 dataclass、``__post_init__`` 不校验这两个字段
# （声明类型分别是 ``str | None`` 与 ``Sequence[str] | None``）。它们都进 HTTP
# 请求：``user`` 转成 ``user_profile_id`` 头，``betas`` 拼成 ``anthropic-beta``
# 头。走真实 SDK（``httpx2.MockTransport`` 拦网）实测的修复前收场：
#
#   invoke(user=5)         -> AttributeError: 'int' object has no attribute 'encode'
#   beta_create(betas=5)   -> TypeError: 'int' object is not iterable
#   beta_create(betas=b"x")-> 发出 anthropic-beta: 120
#   beta_create(betas=[1, 2]) -> 发出 anthropic-beta: 1,2
#   beta_create(betas={"a": 1}) -> 发出 anthropic-beta: a（静默取键）
#
# 前者不是本渠道 ``except ValueError``／``except RuntimeError`` 分类内的类型；
# 后三者不报错，但服务端收到一个不存在的 beta 名——要么报一个与调用方写法
# 毫无关系的错，要么静默忽略掉本想启用的特性。


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(b"x", id="bytes"),
        pytest.param({"a": 1}, id="dict"),
    ],
)
def test_anthropic_rejects_non_string_user(monkeypatch, value):
    """``user`` 非字符串必须抛 ``ValueError``，不是裸 ``AttributeError``。"""
    provider = _anthropic_provider(monkeypatch)
    _capture_anthropic_headers(provider, beta=False)

    with pytest.raises(ValueError, match="user 必须是字符串"):
        provider.count_tokens(prompt="hi", user=value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(b"x", id="bytes"),
        pytest.param("s", id="bare-string"),
        pytest.param({"a": 1}, id="mapping"),
    ],
)
def test_anthropic_rejects_non_list_betas(monkeypatch, value):
    """``betas`` 不是字符串列表时必须拒绝。

    ``bytes``／``str``／``Mapping`` 都是**可迭代**的，``list()`` 不会报错而是
    静默给出错误结果：``list(b"x")`` 得 ``[120]``、``list("ab")`` 把单个 beta
    名拆成两个、``list({"a": 1})`` 取键得 ``["a"]``。
    """
    provider = _anthropic_provider(monkeypatch)
    _capture_anthropic_headers(provider, beta=True)

    with pytest.raises(ValueError, match="betas"):
        provider.beta_create(prompt="hi", betas=value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, 2], id="ints"),
        pytest.param(["a", 1], id="mixed"),
        pytest.param([None], id="none-element"),
    ],
)
def test_anthropic_rejects_non_string_beta_elements(monkeypatch, value):
    """``betas`` 的元素非字符串必须拒绝——它们会被拼进 HTTP 头。"""
    provider = _anthropic_provider(monkeypatch)
    _capture_anthropic_headers(provider, beta=True)

    with pytest.raises(ValueError, match="元素必须是字符串"):
        provider.beta_create(prompt="hi", betas=value)


def test_anthropic_betas_validated_without_user(monkeypatch):
    """``betas`` 必须在 ``user`` 为 ``None`` 时也校验。

    ``_user_profile_fields`` 只在 ``user is not None`` 时被调用，而 ``betas``
    走 ``beta_values`` 直接进 params——把校验只放在那个函数里会漏掉这条路径，
    实测 ``beta_create(prompt="hi", betas=5)`` 仍抛裸 ``TypeError``。
    """
    provider = _anthropic_provider(monkeypatch)
    _capture_anthropic_headers(provider, beta=True)

    with pytest.raises(ValueError, match="betas"):
        provider.beta_create(prompt="hi", betas=5)


def test_anthropic_user_and_betas_still_accept_legal_values(monkeypatch):
    """收紧不能误伤：合法 ``user`` 与 ``betas`` 照旧合并进 ``anthropic-beta``。"""
    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=True)

    provider.beta_create(prompt="hi", user="u1", betas=["a", "b"])
    assert captured["betas"] == ["a", "b", "user-profiles-2026-08-18"]
    assert captured["user_profile_id"] == "u1"


def test_anthropic_betas_alone_reach_params_unchanged(monkeypatch):
    """``user`` 为 ``None`` 时 ``betas`` 原样进 params（不注入 profile 头）。"""
    provider = _anthropic_provider(monkeypatch)
    captured = _capture_anthropic_headers(provider, beta=True)

    provider.beta_create(prompt="hi", betas=["a"])
    assert captured["betas"] == ["a"]
    assert "user_profile_id" not in captured


# ── 工具定义与消息列表的容器形态 ──
#
# 四个 provider 的 ``_convert_tools`` 都写 ``for index, tool in enumerate(tools)``，
# 而同函数对**可迭代但元素非法**的形态（``"s"``／``{"a": 1}``／``b"x"``）早已给
# 带序号的 ``ValueError``——只有「不可迭代」这一半漏了，``enumerate`` 抛裸
# ``TypeError: 'int' object is not iterable``。同一判据的两半必须都在，否则
# 「工具定义写错」的收场取决于它是标量还是序列。
#
# ``extract_system_prompt`` 是更典型的一处：它在多数 provider 里**先于**
# ``normalize_messages``（后者有完整判据）被调用，因此非法取值在这里就炸。
# 实测 ``messages=5`` 抛裸 ``TypeError``、``messages="s"`` 抛裸
# ``AttributeError: 'str' object has no attribute 'get'``——字符串是合法的
# ``Sequence``，只判可迭代拦不住。


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param(object(), id="opaque-object"),
    ],
)
def test_google_rejects_non_sequence_tools(value):
    """``tools`` 不可迭代时必须抛 ``ValueError``，不是裸 ``TypeError``。"""
    from src.providers.google import GoogleProvider

    with pytest.raises(ValueError, match="工具定义必须是列表"):
        GoogleProvider._convert_tools(value)


@pytest.mark.parametrize(
    "value",
    [pytest.param(5, id="int"), pytest.param(1.5, id="float")],
)
def test_anthropic_rejects_non_sequence_tools(value):
    """Anthropic 同判据——``_build_message_params(prompt="hi", tools=5)`` 实测抛裸 ``TypeError``。"""
    from src.providers.anthropic import AnthropicProvider

    with pytest.raises(ValueError, match="工具定义必须是列表"):
        AnthropicProvider._convert_tools(value)


@pytest.mark.parametrize(
    "value",
    [pytest.param(5, id="int"), pytest.param(1.5, id="float")],
)
def test_responses_converters_reject_non_sequence_tools(value):
    """Ark 与 OpenAI 兼容的 Responses 转换器同判据。"""
    from src.providers.openai_compatible import OpenAICompatibleProvider
    from src.providers.volcengine import VolcengineProvider

    with pytest.raises(ValueError, match="工具定义必须是列表"):
        OpenAICompatibleProvider._convert_responses_tools(value)
    with pytest.raises(ValueError, match="工具定义必须是列表"):
        VolcengineProvider._convert_responses_tools(value)


def test_tool_converters_still_accept_legal_sequences():
    """收紧不能误伤：``None`` 与空列表照旧得到 ``None``，合法工具照旧转换。"""
    from src.providers.anthropic import AnthropicProvider
    from src.providers.google import GoogleProvider
    from src.providers.openai_compatible import OpenAICompatibleProvider
    from src.providers.volcengine import VolcengineProvider

    for converter in (
        GoogleProvider._convert_tools,
        AnthropicProvider._convert_tools,
        OpenAICompatibleProvider._convert_responses_tools,
        VolcengineProvider._convert_responses_tools,
    ):
        assert converter(None) is None
        assert converter([]) is None

    converted = GoogleProvider._convert_tools(
        [{"function": {"name": "lookup", "parameters": {"type": "object"}}}]
    )
    assert converted and converted[0] is not None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("s", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param({"a": 1}, id="mapping"),
    ],
)
def test_extract_system_prompt_rejects_non_message_containers(value):
    """``messages`` 非法形态必须抛 ``ValueError``。

    字符串与 ``bytes`` 都是合法的 ``Sequence``，只判「可迭代」拦不住——
    前者迭代出单字符、后者迭代出整数，随后 ``.get`` 抛裸 ``AttributeError``。
    """
    from src.providers.__base__.model_provider import extract_system_prompt

    with pytest.raises(ValueError, match="messages 必须是消息对象序列"):
        extract_system_prompt(value)


@pytest.mark.parametrize(
    "value",
    [pytest.param(5, id="int"), pytest.param("s", id="bare-string")],
)
def test_extract_system_prompt_rejects_non_mapping_entries(value):
    """条目非 Mapping 必须报错而不是跳过。

    静默跳过会让一条写错的消息**悄悄消失**，system 提示随之丢失——比报错
    更难查。
    """
    from src.providers.__base__.model_provider import extract_system_prompt

    with pytest.raises(ValueError, match=r"messages\[0\] 必须是消息对象"):
        extract_system_prompt([value])


def test_extract_system_prompt_still_extracts_system_content():
    """收紧不能误伤：正常提取照旧，空列表与 ``None`` 返回 ``None``。"""
    from src.providers.__base__.model_provider import extract_system_prompt

    assert extract_system_prompt(None) is None
    assert extract_system_prompt([]) is None
    assert extract_system_prompt([{"role": "user", "content": "hi"}]) is None
    assert (
        extract_system_prompt(
            [{"role": "system", "content": "A"}, {"role": "user", "content": "hi"}]
        )
        == "A"
    )


def test_google_invoke_surfaces_bad_messages_as_value_error(monkeypatch):
    """走真实入口：``messages=5`` 必须得到 ``ValueError``。

    必须**消费** ``invoke`` 的返回值——它是生成器，函数体在首次 ``next`` 之前
    不执行，只调不迭代等于没验。
    """
    _patch_provider_settings(monkeypatch, "src.providers.google", google_api_key="test-key")
    from src.providers.google import GoogleProvider

    provider = GoogleProvider("gemini-2.0-flash")
    with pytest.raises(ValueError, match="messages 必须是消息对象序列"):
        list(provider.invoke(prompt="hi", stream=False, messages=5))


# ── rerank 与 embedding 入口的入参形态 ──
#
# 两个 rerank provider 的 ``rerank``／``arerank`` 都把 ``len(documents)`` 写在
# ``_validate_inputs`` **之前**的日志行里：非序列取值在那里抛裸
# ``TypeError: object of type 'int' has no len()``，而紧随其后的校验对同一取值
# 给的是带指引的 ``ValueError``。同一判据的两半顺序反了。
#
# embedding 侧更彻底：``TextEmbeddingModel`` 的六个实现没有一处判 ``texts``
# 的形态。``texts=5`` 抛裸 ``TypeError``；``texts="abc"`` 是合法的
# ``Sequence``，被**静默**当成三条单字符文本向量化（local_hash 实测返回 3 条）。


def _rerank_providers():
    from src.providers.jina import JinaProvider
    from src.providers.siliconflow_rerank import SiliconflowRerankProvider

    return [
        ("Jina", JinaProvider("m"), JinaProvider._validate_inputs),
        ("SiliconFlow", SiliconflowRerankProvider("m"), SiliconflowRerankProvider._validate_inputs),
    ]


@pytest.mark.parametrize("value", [pytest.param(5, id="int"), pytest.param(1.5, id="float")])
def test_rerank_rejects_non_sequence_documents(value, monkeypatch):
    """``documents`` 非序列必须抛 ``ValueError``，不是裸 ``TypeError``。

    四个入口（两个 provider × 同步/异步）的日志行都在校验之前求值
    ``len(documents)``。
    """
    monkeypatch.setattr(
        "src.providers.jina.get_settings",
        lambda: SimpleNamespace(jina_api_key="k"),
    )
    monkeypatch.setattr(
        "src.providers.siliconflow_rerank.get_settings",
        lambda: SimpleNamespace(siliconflow_api_key="k", siliconflow_base_url="https://x/v1"),
    )
    import asyncio

    for name, provider, _validate in _rerank_providers():
        with pytest.raises(ValueError, match="必须是非空列表"):
            provider.rerank(query="q", documents=value, top_n=3)
        with pytest.raises(ValueError, match="必须是非空列表"):
            asyncio.run(provider.arerank(query="q", documents=value, top_n=3))


def test_rerank_still_accepts_legal_documents(monkeypatch):
    """收紧不能误伤：合法 ``documents`` 照旧通过校验。"""
    monkeypatch.setattr(
        "src.providers.jina.get_settings", lambda: SimpleNamespace(jina_api_key="k")
    )
    monkeypatch.setattr(
        "src.providers.siliconflow_rerank.get_settings",
        lambda: SimpleNamespace(siliconflow_api_key="k", siliconflow_base_url="https://x/v1"),
    )
    for name, _provider, validate in _rerank_providers():
        validate("q", ["a", "b"], 2)


def _embedding_providers():
    from src.providers.google import GoogleProvider
    from src.providers.local_hash import LocalHashEmbeddingProvider
    from src.providers.openai_compatible import OpenAICompatibleProvider
    from src.providers.volcengine import VolcengineProvider

    openai_settings = SimpleNamespace(
        openai_api_key="k", openai_api_base="https://api.openai.com/v1"
    )
    return [
        ("LocalHash", LocalHashEmbeddingProvider("local-hash-32")),
        ("Google", GoogleProvider("gemini-2.0-flash")),
        (
            "OpenAI",
            OpenAICompatibleProvider("text-embedding-3-large", "openai", settings=openai_settings),
        ),
        ("Ark", VolcengineProvider("m")),
    ]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param([b"x"], id="bytes-element"),
        pytest.param([1], id="int-element"),
    ],
)
def test_embedding_rejects_non_string_list_texts(value, monkeypatch):
    """``texts`` 非法形态必须抛 ``ValueError``。

    ``"abc"`` 尤其要紧：它是合法的 ``Sequence``，此前被静默拆成三条单字符文本
    向量化——调用方以为传了一条文本，实际入库三条垃圾向量。
    """
    _patch_provider_settings(monkeypatch, "src.providers.google", google_api_key="k")
    _patch_provider_settings(
        monkeypatch,
        "src.providers.volcengine",
        volc_access_key="a",
        volc_secret_key="s",
        volc_base_url="https://ark.cn-beijing.volces.com/api/v3",
        ark_api_key="k",
    )
    import asyncio

    for name, provider in _embedding_providers():
        with pytest.raises(ValueError, match="texts"):
            provider.embed_documents(value)
        with pytest.raises(ValueError, match="texts"):
            asyncio.run(provider.aembed_documents(value))


def test_embedding_still_accepts_legal_texts(monkeypatch):
    """收紧不能误伤：空列表、列表与元组照旧可用。

    空列表**合法**——语义是「没有文档要向量化」，各家返回空列表，不能收紧成
    「必须非空」（``require_text_list`` 与 rerank 的 ``_validate_inputs`` 在
    这一点上有意不同）。
    """
    _patch_provider_settings(monkeypatch, "src.providers.google", google_api_key="k")
    from src.providers.__base__.model_provider import require_text_list
    from src.providers.local_hash import LocalHashEmbeddingProvider

    provider = LocalHashEmbeddingProvider("local-hash-32")
    assert provider.embed_documents([]) == []
    assert len(provider.embed_documents(["a", "b"])) == 2
    assert len(provider.embed_documents(("a", "b"))) == 2
    assert require_text_list([], "X") == []


def test_require_text_list_reports_the_offending_index():
    """元素非法时消息里要有下标，否则用户不知道是哪一条。"""
    from src.providers.__base__.model_provider import require_text_list

    with pytest.raises(ValueError, match=r"texts\[2\]"):
        require_text_list(["a", "b", 3], "X")


# ── ETL 三个阶段的入参形态 ──
#
# ``Pipeline.process`` → ``Cleaner.clean`` → ``Splitter.split`` 是建库主路径上
# 相邻的三步（``KnowledgeBuildService._process_file`` 对知识库里的每个文件调
# 一次），此前三处都不判入参形态。实测：
#
#   process(5)   -> AttributeError: 'int' object has no attribute 'metadata'
#   clean(5)     -> TypeError: 'int' object is not iterable
#   clean("abc") -> AttributeError: 'str' object has no attribute 'content'
#   split(5)     -> TypeError: object of type 'int' has no len()
#   split("abc") -> AttributeError: 'str' object has no attribute 'metadata'
#
# 三处都不在调用方按 ``ValueError`` 建立的分类内，且 ``"abc"`` 这类字符串是
# 合法 ``Sequence``——只判可迭代拦不住。


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param(None, id="none"),
    ],
)
def test_pipeline_process_rejects_non_document(value):
    """``Pipeline.process`` 的入参必须是 ``Document``。"""
    from src.etl.pipeline import Pipeline

    with pytest.raises(ValueError, match="必须是 Document"):
        Pipeline.__new__(Pipeline).process(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param({"a": 1}, id="mapping"),
    ],
)
def test_cleaner_rejects_non_sequence_documents(value):
    """``BasicCleaner.clean`` 的入参必须是文档列表。"""
    from src.etl.cleaners.basic_cleaner import BasicCleaner

    with pytest.raises(ValueError, match="清洗器 documents"):
        BasicCleaner().clean(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param(None, id="none"),
    ],
)
def test_cleaner_rejects_non_document_entries(value):
    """条目非 ``Document`` 必须报错并带下标。"""
    from src.etl.cleaners.basic_cleaner import BasicCleaner

    with pytest.raises(ValueError, match=r"清洗器 documents\[0\] 必须是 Document"):
        BasicCleaner().clean([value])


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param({"a": 1}, id="mapping"),
    ],
)
def test_splitter_rejects_non_sequence_documents(value):
    """``RecursiveTextSplitter.split`` 的入参必须是文档列表。"""
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(parent_chunk_size=300, parent_chunk_overlap=50)
    with pytest.raises(ValueError, match="分片器 documents"):
        splitter.split(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param("abc", id="bare-string"),
        pytest.param(None, id="none"),
    ],
)
def test_splitter_rejects_non_document_entries(value):
    """条目非 ``Document`` 必须报错并带下标。"""
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(parent_chunk_size=300, parent_chunk_overlap=50)
    with pytest.raises(ValueError, match=r"分片器 documents\[0\] 必须是 Document"):
        splitter.split([value])


def test_split_hierarchical_shares_the_same_guard():
    """``split_hierarchical`` 是公开方法，必须与 ``split`` 同判据。

    只守卫 ``split`` 会留下「直接调 ``split_hierarchical``」的口子——实测
    修复前它抛 ``TypeError: 'int' object is not iterable``。
    """
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(
        parent_chunk_size=1500,
        parent_chunk_overlap=150,
        child_chunk_size=300,
        child_chunk_overlap=50,
        structure_mode="hierarchical",
    )
    with pytest.raises(ValueError, match="分片器 documents"):
        splitter.split_hierarchical(5)
    with pytest.raises(ValueError, match=r"分片器 documents\[0\] 必须是 Document"):
        splitter.split_hierarchical([5])


def test_etl_stages_still_accept_legal_documents():
    """收紧不能误伤：三个阶段的合法路径照旧工作，且结果与修复前一致。"""
    from src.etl.cleaners.basic_cleaner import BasicCleaner
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
    from src.models.document import Document

    cleaner = BasicCleaner()
    assert cleaner.clean([]) == []

    cleaned = cleaner.clean([Document(content="a\r\n\r\n\r\nb")])
    assert len(cleaned) == 1 and "a" in cleaned[0].content

    splitter = RecursiveTextSplitter(parent_chunk_size=300, parent_chunk_overlap=50)
    assert splitter.split([]) == []
    assert len(splitter.split([Document(content="句子内容。" * 200)])) > 1

    # ``Pipeline.process`` 需要一个完整装配的实例，这里只断言它的守卫放行
    # 合法 ``Document``（走真实的 ``isinstance`` 判据，而不是重复构造）。
    assert isinstance(Document(content="x"), Document)


# ── 配置写入入口的容器形态 ──
#
# ``SessionConfig.set_many`` 与 ``Chatbot.chat_config`` setter 是同一条写入路径
# 的两端（后者优先转调前者），此前都不判入参容器形态。实测 ``5`` 与 ``None``
# 在 ``for key in values`` 处抛裸 ``TypeError: 'int' object is not iterable``，
# 而紧随其后的 ``_require_known_field`` 对可迭代但字段名非法的形态给的是带
# 指引的 ``ValueError``——同一个「配置写错」的收场不该取决于它是标量还是映射。
#
# ``"abc"`` 更隐蔽：字符串是合法的 ``Sequence``，被拆成 ``a``／``b``／``c``
# 三个字段名，报出与调用方写法毫无关系的 ``KeyError``。


def _session_config_for_writes():
    from src.runtime.contracts import build_session_config
    from src.utils.config import get_settings

    return build_session_config(get_settings())


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(None, id="none"),
        pytest.param("abc", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param([1], id="list"),
    ],
)
def test_set_many_rejects_non_mapping(value):
    """``set_many`` 的入参必须是映射。"""
    with pytest.raises(ValueError, match="批量写入必须是字段映射"):
        _session_config_for_writes().set_many(value)


def test_set_many_still_accepts_mappings():
    """收紧不能误伤：空映射与合法映射照旧工作。"""
    config = _session_config_for_writes()
    config.set_many({})
    config.set_many({"top_k": 3})
    assert config.top_k == 3


def test_set_many_still_reports_unknown_fields():
    """字段名非法时的消息形态不变（带字段名与可选值）。"""
    config = _session_config_for_writes()
    with pytest.raises(ValueError, match="retrieval_method"):
        config.set_many({"retrieval_method": "不存在"})


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(None, id="none"),
        pytest.param("abc", id="bare-string"),
        pytest.param(b"x", id="bytes"),
        pytest.param([1], id="list"),
    ],
)
def test_chat_config_setter_rejects_non_mapping(value):
    """``Chatbot.chat_config`` setter 的入参必须是 ``SessionConfig`` 或映射。"""
    from src.chat.core import Chatbot

    bot = Chatbot.__new__(Chatbot)
    bot.session_config = _session_config_for_writes()
    with pytest.raises(ValueError, match="必须是 SessionConfig 或字段映射"):
        bot.chat_config = value


def test_chat_config_setter_still_accepts_both_forms():
    """收紧不能误伤：``SessionConfig`` 实例与映射两条路径照旧工作。"""
    from src.chat.core import Chatbot
    from src.runtime.contracts import SessionConfig

    bot = Chatbot.__new__(Chatbot)
    bot.session_config = _session_config_for_writes()

    replacement = _session_config_for_writes()
    assert isinstance(replacement, SessionConfig)
    bot.chat_config = replacement
    assert bot.chat_config is replacement

    bot.chat_config = {"top_k": 7}
    assert bot.session_config.top_k == 7


def test_chat_config_setter_rejection_is_atomic():
    """被拒时不得留下改了一半的状态。

    这条守着既有的原子性契约（``deepcopy`` 副本上写入），容器守卫加在它之前
    不能破坏它。
    """
    from src.chat.core import Chatbot

    bot = Chatbot.__new__(Chatbot)
    bot.session_config = _session_config_for_writes()
    before = bot.session_config.top_k

    with pytest.raises(ValueError):
        bot.chat_config = {"top_k": 42, "retrieval_method": "不存在"}
    assert bot.session_config.top_k == before


# ── retrieval_method 的取值收敛（三个取用点） ──
#
# ``retrieval_method`` 声明为 ``RetrievalMethod``，但 ``SessionConfig`` 是公开类、
# 普通属性没有 ``__setattr__`` 守卫，嵌入方（或携带异常取值的 pickle 快照）可以
# 直写成任何形态。修复前实测三个取用点各自的收场：
#
#   * ``RetrievalService.retrieve``：``==`` 判分支，而它**没有**形态收敛——是该
#     函数里唯一没有 ``_coerce_*`` 助手的字段（top_k／候选倍率／score_threshold／
#     rerank_enabled／hybrid_fusion_strategy 都有）。``"hybrid"``／``None``／``5``
#     静默落进 else 分支，**退化成纯关键词检索**（实测只调了 keyword_search）。
#     用户看到的是「配好了混合检索但只召回关键词命中」，不是一条配置错误。
#   * ``display_chat_config``：读 ``.value``，在 ``start_chat_session_async`` 里于
#     任何提示符**之前**被调用，直写 ``5``／``None`` 抛裸
#     ``AttributeError: 'int' object has no attribute 'value'``，会话在启动横幅
#     这一步就终止。
#   * ``config_menu.edit_retrieval_params``：读同一个 ``.value``，菜单渲染不出来。


def _retrieval_service_with_recording():
    """返回 (service, calls)：记录实际走到了哪条检索路径。"""
    from src.retrieval.vdb.faiss_store import FaissStore
    from src.runtime.contracts import build_run_config
    from src.services.embedding_service import EmbeddingService
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import get_settings

    service = RetrievalService(FaissStore(), EmbeddingService(build_run_config(get_settings())))
    calls: list[str] = []
    original_semantic = service._semantic_retrieve
    original_keyword = service._keyword_retrieve

    async def recording_semantic(query, top_k):
        calls.append("semantic")
        return await original_semantic(query, top_k)

    async def recording_keyword(query, top_k):
        calls.append("keyword")
        return await original_keyword(query, top_k)

    service._semantic_retrieve = recording_semantic
    service._keyword_retrieve = recording_keyword
    return service, calls


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("hybrid", id="english-alias"),
        pytest.param("语义", id="abbreviated"),
        pytest.param(None, id="none"),
        pytest.param(5, id="int"),
        pytest.param(["向量检索"], id="list"),
        # id 必须显式给：pytest 生成参数 id 时会对取值调 ``str()``，位数超限的
        # 整数会让**收集阶段**自己崩掉（实测）。
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
    ],
)
def test_retrieve_rejects_unusable_retrieval_method(bad_value):
    """无法识别的检索模式必须报错，而不是静默走错分支。

    ``"hybrid"`` 这条尤其要紧：它是 ``"混合检索"`` 的英文写法，用户很容易写，
    而修复前它悄悄变成纯关键词检索。
    """
    import asyncio

    from src.runtime.contracts import build_session_config
    from src.utils.config import get_settings

    service, calls = _retrieval_service_with_recording()
    config = build_session_config(get_settings())
    config.retrieval_method = bad_value

    with pytest.raises(ValueError, match="retrieval_method 必须是"):
        asyncio.run(service.retrieve("测试", config))

    # 关键断言：被拒之前**没有**发起任何一路检索。只断言「抛了 ValueError」
    # 会让「先检索再校验」的实现也通过。
    assert calls == []


@pytest.mark.parametrize(
    ("value", "expected_calls"),
    [
        (RetrievalMethod.HYBRID_SEARCH, ["semantic", "keyword"]),
        ("混合检索", ["semantic", "keyword"]),
        (RetrievalMethod.SEMANTIC_SEARCH, ["semantic"]),
        ("向量检索", ["semantic"]),
        (RetrievalMethod.FULL_TEXT_SEARCH, ["keyword"]),
        ("全文检索", ["keyword"]),
    ],
)
def test_retrieve_keeps_working_for_valid_retrieval_methods(value, expected_calls):
    """收紧不能误伤：枚举成员与它的字符串取值（config.toml 的既有契约）都照旧。"""
    import asyncio

    from src.runtime.contracts import build_session_config
    from src.utils.config import get_settings

    service, calls = _retrieval_service_with_recording()
    config = build_session_config(get_settings())
    config.retrieval_method = value

    asyncio.run(service.retrieve("测试", config))

    assert calls == expected_calls


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("hybrid", id="english-alias"),
        pytest.param(5, id="int"),
        pytest.param(None, id="none"),
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
    ],
)
def test_display_chat_config_rejects_unusable_retrieval_method(bad_value):
    """启动横幅不得因可直写的取值而抛裸 ``AttributeError``。

    本函数在 ``start_chat_session_async`` 里于任何提示符**之前**被调用，修复前
    用户看到的是 ``'int' object has no attribute 'value'``——会话直接终止。
    """
    from rich.console import Console

    from src.runtime.contracts import build_session_config
    from src.ui.display_utils import display_chat_config
    from src.utils.config import get_settings

    config = build_session_config(get_settings())
    config.retrieval_method = bad_value

    with pytest.raises(ValueError, match="retrieval_method 必须是"):
        display_chat_config(Console(quiet=True), config)


def test_display_chat_config_keeps_working_for_valid_retrieval_method():
    """收紧不能误伤：合法的枚举成员与字符串取值照旧渲染。

    配置**自造**而不是 ``build_session_config(get_settings())``：后者读开发机上的
    ``config.toml``，而本机实测 ``active_llm_configuration='google'`` 并不在
    ``llm_configurations`` 的键里——那条与本用例无关的既有不一致会让本测试以
    ``ValueError`` 失败，把「retrieval_method 的收紧误伤了合法取值」这条断言
    变成对开发者本地配置的断言。
    """
    from rich.console import Console

    from src.ui.display_utils import display_chat_config
    from src.utils.config import RetrievalMethod

    for value in (RetrievalMethod.HYBRID_SEARCH, "混合检索"):
        config = _session_config(
            retrieval_method=value,
            active_llm_configuration="demo",
            llm_configurations={"demo": SimpleNamespace(provider="openai", model_name="demo")},
        )
        display_chat_config(Console(quiet=True), config)


def test_session_config_validator_and_runtime_reader_agree():
    """配置边界与运行期读取侧对同一取值必须给出同一种收场。

    两边此前各写一份判据（``SessionConfig._validate_retrieval_method`` 与
    ``retrieve`` 的裸 ``==``），现在共用 ``require_retrieval_method``。这条测试
    盯住「只修一边」——只改读取侧会让「写得进、读不出」，只改写入侧则相反。
    """
    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import require_retrieval_method
    from src.utils.config import get_settings

    config = build_session_config(get_settings())
    for bad_value in ("hybrid", None, 5):
        with pytest.raises(ValueError):
            config["retrieval_method"] = bad_value
        with pytest.raises(ValueError):
            require_retrieval_method(bad_value)


def test_require_retrieval_method_accepts_enum_and_string_forms():
    """共享助手接受既有形态，并对无法识别的取值一律拒绝。"""
    from src.services.retrieval_service import require_retrieval_method
    from src.utils.config import RetrievalMethod

    assert require_retrieval_method("混合检索") is RetrievalMethod.HYBRID_SEARCH
    assert require_retrieval_method(RetrievalMethod.HYBRID_SEARCH) is RetrievalMethod.HYBRID_SEARCH
    for bad_value in ("hybrid", "HYBRID_SEARCH", "", " 混合检索", 0, True, None, ["向量检索"]):
        with pytest.raises(ValueError):
            require_retrieval_method(bad_value)


def test_require_retrieval_method_accepts_duck_typed_substitutes():
    """判据是**属性存在性**，不是具体类型——仓库的既有约定。

    ``require_detail_attributes`` 的 docstring 写明「使用方只要求取值提供这几个
    属性，任何具备它们的对象都能正常工作（仓库里的测试就用 ``SimpleNamespace``
    构造等价配置）」。``display_utils``／``chat/core`` 的既有测试都用
    ``SimpleNamespace(value="混合检索")`` 作 ``retrieval_method`` 的替身，按
    ``isinstance(value, RetrievalMethod)`` 收窄会把它们一并拒掉——实测如此，
    4 个既有用例直接 error。

    这条盯住「收紧不能把等价替身一起拒掉」：它是对**判据方向**的断言，而不是
    对某个具体替身类型的断言。
    """
    from src.services.retrieval_service import require_retrieval_method
    from src.utils.config import RetrievalMethod

    assert (
        require_retrieval_method(SimpleNamespace(value="混合检索")) is RetrievalMethod.HYBRID_SEARCH
    )
    # ``.value`` 本身是枚举成员也要接受（替身套替身）。
    assert (
        require_retrieval_method(SimpleNamespace(value=RetrievalMethod.SEMANTIC_SEARCH))
        is RetrievalMethod.SEMANTIC_SEARCH
    )
    # 取值形态不对时仍要拒绝，且消息里带的是**原取值**（便于定位是哪个配置项）。
    with pytest.raises(ValueError, match="retrieval_method 必须是"):
        require_retrieval_method(SimpleNamespace(value="hybrid"))
    with pytest.raises(ValueError, match="retrieval_method 必须是"):
        require_retrieval_method(SimpleNamespace())


# ── chat_temperature 的取值收敛（两个读取点） ──
#
# ``chat_temperature`` 有**完整**的校验器——``Settings.validate_chat_temperature``
# 与 ``SessionConfig._validate_chat_temperature`` 各自一份，判据一致（拒 bool、
# 非数值、非有限、负数、超出双精度的整数）。但两个**读取点**都直读原属性，而
# ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫。实测修复前
# ``"abc"``／``None``／``[0.5]``／``True``／``-1`` 全部**静默**通过并原样成为
# SDK 的请求 kwargs——而它们正是 ``session_config["chat_temperature"] = ...``
# 会明确拒绝的取值。同一个字段「写得进就合法、直写就非法」两种口径。


class _RecordingSamplingModel:
    """支持 sampling 的模型替身，记录收到的 kwargs。"""

    def __init__(self):
        self.calls: list[dict] = []

    def supports_sampling_option(self, name):
        return name == "temperature"

    async def ainvoke(self, **kwargs):
        self.calls.append(kwargs)
        yield "回复"


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("abc", id="str"),
        pytest.param(None, id="none"),
        pytest.param([0.5], id="list"),
        pytest.param(True, id="bool"),
        pytest.param(-1, id="negative"),
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
    ],
)
def test_generate_reply_rejects_unusable_chat_temperature(bad_value):
    """不可用的 ``chat_temperature`` 必须在本地被拒，而不是进请求 kwargs。

    修复前实测：这些取值全部**静默**通过，``"abc"`` 与 ``[0.5]`` 原样成为 SDK
    参数，``-1`` 被网关按默认值接受。用户看到的是服务端的参数错误，而不是一条
    本地配置错误。
    """
    import asyncio

    from src.runtime.contracts import build_session_config
    from src.services.chat_service import ChatService
    from src.utils.config import get_settings

    model = _RecordingSamplingModel()
    service = ChatService(model, None)
    config = build_session_config(get_settings())
    config.chat_temperature = bad_value

    async def consume():
        return [chunk async for chunk in service.generate_reply("问题", "意图", [], config)]

    with pytest.raises(ValueError, match="chat_temperature"):
        asyncio.run(consume())

    # 关键断言：被拒之前**没有**发起请求。只断言「抛了 ValueError」会让
    # 「先发请求再校验」的实现也通过。
    assert model.calls == []


@pytest.mark.parametrize("value", [0.0, 0.7, 1.0, 2, 0])
def test_generate_reply_keeps_working_for_valid_chat_temperature(value):
    """收紧不能误伤：合法的非负数值照旧传给模型。"""
    import asyncio

    from src.runtime.contracts import build_session_config
    from src.services.chat_service import ChatService
    from src.utils.config import get_settings

    model = _RecordingSamplingModel()
    service = ChatService(model, None)
    config = build_session_config(get_settings())
    config.chat_temperature = value

    async def consume():
        return [chunk async for chunk in service.generate_reply("问题", "意图", [], config)]

    assert asyncio.run(consume()) == ["回复"]
    assert model.calls[0]["temperature"] == float(value)


def test_chat_temperature_reader_and_validator_agree():
    """配置边界与运行期读取侧对同一取值必须给出同一种收场。

    两边此前各写一份判据（``SessionConfig._validate_chat_temperature`` 与两个读取
    点的裸属性访问），现在共用 ``require_chat_temperature``。这条盯住「只修一边」。
    """
    from src.runtime.contracts import build_session_config
    from src.utils.config import get_settings, require_chat_temperature

    config = build_session_config(get_settings())
    for bad_value in ("abc", None, [0.5], True, -1, _HUGE_INT):
        with pytest.raises(ValueError, match="chat_temperature"):
            config["chat_temperature"] = bad_value
        with pytest.raises(ValueError, match="chat_temperature"):
            require_chat_temperature(bad_value)
    # 合法取值两侧都要接受，且收敛成同一形态（float）。
    for good_value in (0.0, 0.7, 2):
        config["chat_temperature"] = good_value
        assert config.chat_temperature == require_chat_temperature(good_value) == float(good_value)


# ── 阈值过滤的融合策略判据（库调用方的默认分支） ──
#
# ``_should_apply_score_threshold`` 的 docstring 写着「``fusion_strategy`` 由调用方
# 传入**已收敛**的取值：``retrieve`` 已经校验过，这里再读一次原属性会绕过那道
# 校验」，但它的**兜底分支**恰恰就在读原属性。判据是 ``strategy == "rrf"``，而
# ``"RRF"``／``" rrf "`` 走的是 RRF 融合却不等于这个字面量——阈值过滤会被错误地
# 套用在排名倒数的量纲上，正是 ``_coerce_fusion_strategy`` 记录的那个缺陷。生产
# 路径（``retrieve``）传的是收敛后的值，因此这条只在库调用方直接调本方法时可达。


@pytest.mark.parametrize("value", ["rrf", "RRF", " rrf ", "Rrf"])
def test_score_threshold_skipped_for_rrf_regardless_of_case(value):
    """RRF 融合（任何大小写／空白形态）都必须跳过阈值过滤。"""
    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, get_settings

    config = build_session_config(get_settings())
    config.retrieval_method = RetrievalMethod.HYBRID_SEARCH
    config.hybrid_fusion_strategy = value

    assert RetrievalService._should_apply_score_threshold(config) is False


def test_score_threshold_applied_for_weighted_and_other_methods():
    """反向守卫：weighted 融合与非混合检索仍要套用阈值。"""
    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, get_settings

    config = build_session_config(get_settings())
    config.retrieval_method = RetrievalMethod.HYBRID_SEARCH
    config.hybrid_fusion_strategy = "weighted"
    assert RetrievalService._should_apply_score_threshold(config) is True

    config.hybrid_fusion_strategy = "rrf"
    config.retrieval_method = RetrievalMethod.SEMANTIC_SEARCH
    assert RetrievalService._should_apply_score_threshold(config) is True


def test_score_threshold_rejects_unusable_fusion_strategy_in_default_branch():
    """兜底分支不得把无法识别的策略静默当成「非 RRF」。

    静默判定会让阈值过滤套用在排名倒数的量纲上（结果被清空），而正确收场是一条
    配置错误。
    """
    from src.runtime.contracts import build_session_config
    from src.services.retrieval_service import RetrievalService
    from src.utils.config import RetrievalMethod, get_settings

    config = build_session_config(get_settings())
    config.retrieval_method = RetrievalMethod.HYBRID_SEARCH
    for bad_value in (None, 5, ["rrf"]):
        config.hybrid_fusion_strategy = bad_value
        with pytest.raises(ValueError, match="hybrid_fusion_strategy"):
            RetrievalService._should_apply_score_threshold(config)


# ── 模型配置的展示路径（provider 形态 / 裸插值） ──
#
# ``ModelDetail.provider`` 声明为 ``str``，但 ``SessionConfig.llm_configurations``
# ／``rerank_configurations`` 的 ``_VALIDATORS`` 明确不登记（「按原样写入，由使用方
# 负责其内部结构」），嵌入方可以往映射里放任意取值。``display_chat_config`` 拿它拼
# ``f"{provider.lower()}_api_key"``——实测修复前：
#
#   * ``provider=["openai"]`` -> 更早一步，在 ``active_providers = {…provider}``
#     这个**集合**里抛 ``TypeError: unhashable type: 'list'``；
#   * ``provider=None``／``5`` -> ``AttributeError: 'NoneType' object has no
#     attribute 'lower'``；
#   * ``active_*_configuration``／``model_name`` 是位数超限整数时，同一张表的裸
#     f-string 抛 ``ValueError: Exceeds the limit``——而**同一个函数**的
#     ``top_k``／``score_threshold``／权重早已走 ``_render_value``（注释写明
#     「本函数在任何提示符之前被调用，会话在启动横幅这一步就终止」）。
#   * ``logger.debug(f"…{active_llm}")`` 的裸插值在 DEBUG 关闭时也**已经求值**，
#     同一个取值照样抛。
#
# 本函数在 ``start_chat_session_async`` 里于任何提示符**之前**被调用，因此以上
# 每一条都让会话在「知识快照加载成功」之后立刻终止。


def _model_config(**overrides):
    """构造一个 ``display_chat_config`` 能完整渲染的 ``SessionConfig``。"""
    values = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 2,
        "rerank_enabled": False,
        "top_k": 5,
        "score_threshold": 0.0,
        "active_llm_configuration": "demo",
        "active_rerank_configuration": "demo",
        "llm_configurations": {"demo": SimpleNamespace(provider="openai", model_name="gpt-5")},
        "rerank_configurations": {
            "demo": SimpleNamespace(provider="siliconflow", model_name="bge-reranker-v2-m3")
        },
        "chat_temperature": 0.7,
    }
    values.update(overrides)
    return SessionConfig(**values)


@pytest.mark.parametrize(
    "bad_provider",
    [
        pytest.param(["openai"], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(None, id="none"),
        pytest.param(5, id="int"),
    ],
)
def test_display_chat_config_rejects_unusable_provider(bad_provider):
    """非字符串 provider 必须收敛成 ``ValueError``，而不是裸 ``AttributeError``／``TypeError``。

    ``["openai"]`` 这条走的是**集合**（``{provider}``）而不是 ``.lower()``，因此
    它抛的是 ``TypeError: unhashable type``——两处判据必须都收敛。
    """
    from rich.console import Console

    from src.ui.display_utils import display_chat_config

    config = _model_config(
        llm_configurations={"demo": SimpleNamespace(provider=bad_provider, model_name="gpt-5")}
    )

    with pytest.raises(ValueError, match="provider 必须是字符串"):
        display_chat_config(Console(quiet=True), config)


def test_display_chat_config_rejects_unusable_rerank_provider():
    """Rerank 侧同一判据：``rerank_enabled=True`` 时才读它。"""
    from rich.console import Console

    from src.ui.display_utils import display_chat_config

    config = _model_config(
        rerank_enabled=True,
        rerank_configurations={"demo": SimpleNamespace(provider=5, model_name="m")},
    )

    with pytest.raises(ValueError, match="provider 必须是字符串"):
        display_chat_config(Console(quiet=True), config)


@pytest.mark.parametrize(
    "field",
    ["model_name", "active_llm_configuration", "active_rerank_configuration"],
)
def test_display_chat_config_renders_values_beyond_the_conversion_limit(field):
    """位数超限的整数必须**可渲染**（降级成量级描述），而不是让会话终止。

    同一张表的 ``top_k``／``score_threshold``／权重早已按这条处理；这条盯住
    「只修了一部分单元格」。
    """
    from rich.console import Console

    from src.ui.display_utils import display_chat_config

    if field == "model_name":
        config = _model_config(
            llm_configurations={"demo": SimpleNamespace(provider="openai", model_name=_HUGE_INT)}
        )
    elif field == "active_llm_configuration":
        # 激活名与配置键同源（都来自 config.toml 或快照），因此要让键也存在，
        # 否则先撞上「未找到key」那条（那是另一条判据，不是本条要测的渲染）。
        config = _model_config(
            active_llm_configuration=_HUGE_INT,
            llm_configurations={_HUGE_INT: SimpleNamespace(provider="openai", model_name="gpt-5")},
        )
    else:
        config = _model_config(
            rerank_enabled=True,
            active_rerank_configuration=_HUGE_INT,
            rerank_configurations={
                _HUGE_INT: SimpleNamespace(provider="siliconflow", model_name="m")
            },
        )

    # 不得抛异常——这正是修复点。
    display_chat_config(Console(quiet=True), config)


def test_display_chat_config_keeps_working_for_valid_model_config():
    """收紧不能误伤：合法配置照旧渲染。"""
    from rich.console import Console

    from src.ui.display_utils import display_chat_config

    display_chat_config(Console(quiet=True), _model_config(rerank_enabled=True))


def test_require_provider_name_accepts_any_string_and_rejects_the_rest():
    """共享助手只要求「是字符串」，不查 provider 白名单。

    白名单是**工厂**的职责（``_get_provider_class`` 会给「不支持的模型提供商」）。
    在展示层提前套白名单会让「配置里有个尚未支持的 provider」从「启动时列出配置」
    变成「启动时崩溃」——那是更差的收场。
    """
    from src.utils.config import require_provider_name

    assert require_provider_name("openai") == "openai"
    assert require_provider_name("not-yet-supported") == "not-yet-supported"
    for bad_value in (None, 5, ["openai"], {"a": 1}, True, _HUGE_INT):
        with pytest.raises(ValueError, match="必须是字符串"):
            require_provider_name(bad_value)


# ── 配置编辑器的动态上界（读可直写字段） ──
#
# ``config_menu._top_k_ceiling`` / ``_multiplier_ceiling`` 读的是**另一个**规模
# 字段来算动态上界（``min(_MAX_RETRIEVAL_TOP_K, _MAX_RETRIEVAL_CANDIDATES //
# max(1, m))``），而那两个字段都可被直写。实测修复前「倍率被直写成 ``"abc"``、
# 用户选『调整 Top K』」在 ``max()`` 处抛裸
# ``TypeError: '>' not supported between instances of 'str' and 'int'``。
#
# 菜单在 chat 主循环的 ``try`` **之外**：内层只接 ``(KeyboardInterrupt,
# EOFError)``、外层只有 ``finally``，因此这会以裸 traceback 终止整个会话，用户
# 丢失对话。``retrieve`` 对同样两个字段早有 ``_coerce_scale_field``，现在判据
# 上移为 ``require_scale_field``（``src/utils/config.py``，``_MAX_RETRIEVAL_*``
# 的归属地），服务层与 UI 共用——UI 不反向依赖服务层私有函数。


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("abc", id="str"),
        pytest.param(None, id="none"),
        pytest.param([2], id="list"),
        pytest.param(True, id="bool"),
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
    ],
)
def test_top_k_ceiling_rejects_unusable_multiplier(bad_value):
    """倍率字段形态不可用时，``_top_k_ceiling`` 必须报错而不是抛裸 ``TypeError``。"""
    from src.ui.config_menu import _top_k_ceiling

    with pytest.raises(ValueError, match="retrieval_candidate_multiplier"):
        _top_k_ceiling(bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("abc", id="str"),
        pytest.param(None, id="none"),
        pytest.param([2], id="list"),
        pytest.param(True, id="bool"),
    ],
)
def test_multiplier_ceiling_rejects_unusable_top_k(bad_value):
    """对称方向：``top_k`` 形态不可用时 ``_multiplier_ceiling`` 同样报错。"""
    from src.ui.config_menu import _multiplier_ceiling

    with pytest.raises(ValueError, match="top_k"):
        _multiplier_ceiling(bad_value)


@pytest.mark.parametrize(
    ("multiplier", "top_k"),
    [(1, 2000), (3, 666), (20, 100), (100, 20), (2, 5)],
)
def test_ceilings_keep_working_for_valid_scale_fields(multiplier, top_k):
    """收紧不能误伤：合法取值照旧算出上界。"""
    from src.ui.config_menu import _multiplier_ceiling, _top_k_ceiling

    assert _top_k_ceiling(multiplier) >= 1
    assert _multiplier_ceiling(top_k) >= 1


def test_edit_retrieval_params_survives_broken_scale_field():
    """端到端：字段被直写坏掉时，菜单必须给出可读错误而不是终止会话。

    这条走的是真实入口（``edit_retrieval_params`` 的选择分支），而不是只调
    ``_top_k_ceiling``——「助手修好了但调用点没接上」是另一类漏网。
    """
    from unittest.mock import patch

    from src.runtime.contracts import build_session_config
    from src.ui import config_menu
    from src.utils.config import get_settings

    config = build_session_config(get_settings()).to_dict()
    config["retrieval_candidate_multiplier"] = "abc"
    # 第一次选目标项，第二次退出循环（否则 ``while True`` 不结束）。
    choices = iter(["top_k", "back"])

    with (
        patch.object(config_menu.questionary, "select") as select,
        patch.object(config_menu.questionary, "text") as text,
    ):
        select.return_value.ask.side_effect = lambda: next(choices, "back")
        text.return_value.ask.return_value = None

        with pytest.raises(ValueError, match="retrieval_candidate_multiplier"):
            config_menu.edit_retrieval_params(config)


def test_require_scale_field_matches_the_service_layer_semantics():
    """共享助手的语义与 ``retrieve`` 的 ``max(1, ...)`` 一致。

    非正数收敛成 1（读取已有状态时的容错），``bool`` 显式拒绝（它几乎必然是
    上游把标志位写进了规模字段），超上界拒绝。
    """
    from src.utils.config import _MAX_RETRIEVAL_CANDIDATES, require_scale_field

    assert require_scale_field(5, "top_k") == 5
    assert require_scale_field(0, "top_k") == 1
    assert require_scale_field(-3, "top_k") == 1
    for bad_value in (True, False, "5", None, [5], 1.5, _HUGE_INT):
        with pytest.raises(ValueError):
            require_scale_field(bad_value, "top_k")
    with pytest.raises(ValueError, match="超出合理范围"):
        require_scale_field(_MAX_RETRIEVAL_CANDIDATES + 1, "top_k")


# ── markup_text 的模板形态 ──
#
# ``markup_text`` 对**值**参数早已用 ``_render_value`` 完整收敛（``10**4301``
# 降级成量级描述），对占位符数量不符与 NUL 也都抛 ``ValueError``；唯独**模板**
# 参数没有形态判据，而函数体第一行就对它做成员测试。实测修复前非字符串模板抛
# 裸 ``TypeError``，且消息随形态各不相同：
#
#   * ``5``／``None`` -> ``argument of type 'int' is not iterable``
#   * ``['a']``／``('a',)``／``{'a': 1}`` -> ``expected string or bytes-like object``
#   * ``b'x'`` -> ``a bytes-like object is required, not 'str'``
#
# 三种消息都不指向「模板写错了」，也不是 ``ValueError``。同一个「入参形态非法」
# 的处置，同一个函数里两种收场——判据只做了一半。
#
# 可达性：44 个调用点里 43 个模板是字符串字面量，剩下 1 个
# （``scripts/embed_knowledge_base.py`` 的 ``mask_api_key(api_key_value)``）两个
# 分支都返回字面量。因此本仓自带流程走不到——修的是**公开函数**的契约一致性。


@pytest.mark.parametrize(
    "bad_template",
    [
        pytest.param(5, id="int"),
        pytest.param(None, id="none"),
        pytest.param(["a"], id="list"),
        pytest.param(("a",), id="tuple"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(b"x", id="bytes"),
        pytest.param(_HUGE_INT, id="beyond-int-max-str-digits"),
    ],
)
def test_markup_text_rejects_unusable_template(bad_template):
    """非字符串模板必须收敛成 ``ValueError``，而不是形态各异的裸 ``TypeError``。"""
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="标记模板必须是字符串"):
        markup_text(bad_template)


@pytest.mark.parametrize(
    ("template", "values", "expected"),
    [
        ("纯文本", (), "纯文本"),
        ("[bold]{}[/bold]", ("x",), "x"),
        ("[dim]本地模型，无需设置[/dim]", (), "本地模型，无需设置"),
        ("[green]a: {} / b: {}[/green]", (1, 2), "a: 1 / b: 2"),
    ],
)
def test_markup_text_keeps_working_for_valid_templates(template, values, expected):
    """收紧不能误伤：字符串模板（含零占位符与多占位符）照旧渲染。"""
    from src.ui.display_utils import markup_text

    assert markup_text(template, *values).plain == expected


def test_markup_text_placeholder_errors_still_win_over_shape_guard():
    """形态守卫不能改变既有的占位符判据。

    模板是合法字符串但占位符数量不符时，仍须给出原来那条消息——新加的守卫放在
    最前面，不能把它盖掉。
    """
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="占位符少于实参数量"):
        markup_text("无占位符", "多余的值")
    with pytest.raises(ValueError, match="占位符多于实参数量"):
        markup_text("{} 与 {}", "只有一个")
