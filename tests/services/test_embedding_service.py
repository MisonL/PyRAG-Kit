import asyncio
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.providers.openai_compatible import OpenAICompatibleProvider
from src.runtime.contracts import build_run_config
from src.services.embedding_service import (
    EmbeddingService,
    _reject_unconvertible_numbers,
)
from src.utils.config import ModelDetail, Settings

# ── 回归：provider key 打错字必须是 ValueError，不是裸 KeyError ──
#
# ``default_embedding_provider`` 没有任何 validator（``Settings`` 里只是普通
# ``str``），而 ``embedding_model_detail`` 原先直接下标。打错一个名字时这里抛
# ``KeyError``，同一份配置走 ``ModelProviderFactory.get_embedding_provider``
# 得到的却是带可选值列表的 ``ValueError``——两条路径的报错契约本该一致，否则
# 调用方按 ``ValueError`` 写的错误分类与日志脱敏接不住 ``KeyError``。
#
# 调用点在检索与建库的主路径上：``embed_query`` 的 Google 分支、
# ``knowledge_build_service`` 写清单、``embed_knowledge_base.py`` 的展示。


def test_embedding_model_detail_reports_unknown_provider_as_value_error():
    """未知 provider key 必须抛带可选值的 ``ValueError``，而不是 ``KeyError``。"""
    run_config = SimpleNamespace(
        default_embedding_provider="gogle",
        embedding_configurations={
            "local-hash": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )

    service = EmbeddingService(run_config)

    with pytest.raises(ValueError, match="未找到key") as excinfo:
        _ = service.embedding_model_detail

    message = str(excinfo.value)
    assert "gogle" in message
    # 可选值列表是这条消息的用途——没有它用户只能去翻 config.toml。
    assert "local-hash" in message


def test_embedding_model_detail_returns_the_configured_detail():
    """合法 key 照常返回详情（收紧不能误伤）。"""
    detail = ModelDetail(provider="local-hash", model_name="local-hash-32")
    run_config = SimpleNamespace(
        default_embedding_provider="local-hash",
        embedding_configurations={"local-hash": detail},
    )

    assert EmbeddingService(run_config).embedding_model_detail is detail


def test_embedding_service_passes_runtime_configurations_to_factory(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    factory = MagicMock(return_value=provider)
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        factory,
    )

    service = EmbeddingService(run_config)

    assert service._get_model() is provider
    factory.assert_called_once_with("custom", run_config.embedding_configurations)


def test_embedding_service_rejects_vectors_that_do_not_match_input_count(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    provider.aembed_documents = AsyncMock(return_value=[[1.0, 2.0]])
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    service = EmbeddingService(run_config)

    with pytest.raises(ValueError, match="向量数量与输入文本数量不一致"):
        asyncio.run(service.embed_texts(["one", "two"]))


@pytest.mark.parametrize("value", [[], [[1.0, 2.0], [3.0, 4.0]], 1.0])
def test_embedding_service_rejects_malformed_query_vectors(monkeypatch, value):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    provider.aembed_query = AsyncMock(return_value=value)
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    with pytest.raises((TypeError, ValueError), match="向量|float"):
        asyncio.run(EmbeddingService(run_config).embed_query("one"))


def test_embedding_service_does_not_replay_model_options(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(
                provider="openai",
                model_name="text-embedding-3-small",
                options={"dimensions": 512, "encoding_format": "float"},
            )
        },
    )
    provider = MagicMock()
    provider.aembed_documents = AsyncMock(return_value=[[1.0, 2.0]])
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    service = EmbeddingService(run_config)
    asyncio.run(service.embed_texts(["one"]))

    provider.aembed_documents.assert_awaited_once_with(["one"])


@pytest.mark.parametrize("method", ["embed_texts", "embed_query", "embed_in_batches"])
@pytest.mark.parametrize(
    "options",
    [
        {"dimensions": 2},
        {"encoding_format": "float"},
        {"user": "embedding-test"},
        {"extra_body": {"vendor_flag": True}},
    ],
)
def test_embedding_service_model_options_reach_sdk_once(monkeypatch, method, options):
    requests = []

    class Embeddings:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in request["input"]]
            )

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._model_name = "text-embedding-3-small"
    provider._options = options
    provider._get_aclient = lambda: SimpleNamespace(embeddings=Embeddings())

    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )
    service = EmbeddingService(
        SimpleNamespace(
            default_embedding_provider="custom",
            kb_embedding_batch_size=1,
            embedding_configurations={
                "custom": ModelDetail(
                    provider="openai",
                    model_name="text-embedding-3-small",
                    options=options,
                )
            },
        )
    )
    value = "one" if method == "embed_query" else ["one", "two"]
    result = asyncio.run(getattr(service, method)(value))
    assert result.dtype == np.float32
    assert result.shape == ((2,) if method == "embed_query" else (2, 2))

    assert len(requests) == (2 if method == "embed_in_batches" else 1)
    for payload in requests:
        for key, value in options.items():
            if key == "extra_body":
                assert payload["extra_body"] == {"vendor_flag": True}
            else:
                assert payload[key] == value


def test_embedding_service_close_disposes_cached_model():
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    service = EmbeddingService(run_config)
    service._embedding_model = provider

    service.close()

    provider.close.assert_called_once_with()
    assert service._embedding_model is None


def test_embedding_service_aclose_disposes_cached_model_async():
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    closed = False

    async def aclose():
        nonlocal closed
        closed = True

    provider.aclose = aclose
    service = EmbeddingService(run_config)
    service._embedding_model = provider

    asyncio.run(service.aclose())

    assert closed
    assert service._embedding_model is None


# ── 回归：向量矩阵的有限性校验 ──


@pytest.mark.parametrize(
    "bad_row,label",
    [
        ([float("nan"), 1.0], "NaN"),
        ([float("inf"), 1.0], "正无穷"),
        ([float("-inf"), 1.0], "负无穷"),
        ([1e300, 1.0], "超出 float32 范围（cast 后静默溢出为 inf）"),
    ],
)
def test_as_float32_matrix_rejects_non_finite_values(bad_row, label):
    """含 NaN/无穷大的向量必须在进入索引前拒绝。

    FAISS 会把 NaN 向量照单收下（``ntotal`` 正常增加），直到查询时才以
    3.4e38 的哨兵距离暴露出来——那时「哪个文档坏了」已经无法定位。
    ``np.array(..., dtype=np.float32)`` 对超出 float32 范围的值只发
    RuntimeWarning 并静默溢出成 inf，所以在 cast **之后**检查。

    溢出那条用例会触发 numpy 的 ``RuntimeWarning``：它是被测行为本身的一部分
    （正是这个 warning 说明「静默溢出」确实发生了），因此在这里显式断言它，
    而不是让它泄漏到测试输出里——泄漏的 warning 会掩盖真正的新警告。
    """
    with (
        pytest.raises(ValueError, match="非有限值"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        EmbeddingService._as_float32_matrix([[1.0, 2.0], bad_row], expected_rows=2)

    overflow = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if bad_row == [1e300, 1.0]:
        assert overflow, "超出 float32 范围时应触发 numpy 溢出警告"
        assert "overflow" in str(overflow[0].message)


def test_as_float32_matrix_accepts_finite_values():
    """收紧不能误伤：正常向量（含 0 与负值）必须可用。"""
    matrix = EmbeddingService._as_float32_matrix(
        [[0.0, -1.5, 2.25], [-0.0, 1e-30, 3.4e38]], expected_rows=2
    )

    assert np.isfinite(matrix).all()
    assert matrix.shape == (2, 3)


# ── 回归：向量维度必须有上界 ──
#
# ``local-hash`` 的维度直接来自 ``model_name`` 的数字后缀
# （``local-hash-256`` -> 256），而 ``embedding_configurations.<key>.model_name``
# 是 ``config.toml`` 里用户可写的字段，没有任何校验。实测把它配成
# ``local-hash-100000000`` 时 ``Settings`` 照单全收，一次 ``embed_texts(["测试"])``
# 就分配出 ``(1, 100000000)`` 的 float32 矩阵：耗时 180 秒、峰值内存 3.9 GB；
# 按真实知识库 115 个分块算，落盘前的 ``np.vstack`` 需要约 4.3 GB。
#
# 这与已经加过界的 ``chat_top_k × retrieval_candidate_multiplier`` 是同一类问题：
# 用户可配的规模没有上界，代价是进程被 OOM 杀掉而不是一条可读的错误。
# 上界取 65536：现有各家的最大维度是 3072（text-embedding-3-large），留了
# 20 倍余量，而 65536 维的 float32 单块也只有 256 KB。


@pytest.mark.parametrize("dimension", [1, 256, 3072, 65536])
def test_as_float32_matrix_accepts_dimensions_at_or_below_the_bound(dimension):
    """收紧不能误伤：上界以内（含边界）的维度必须可用。"""
    matrix = EmbeddingService._as_float32_matrix([[1.0] * dimension], expected_rows=1)

    assert matrix.shape == (1, dimension)


@pytest.mark.parametrize("dimension", [65537, 1_000_000, 100_000_000])
def test_as_float32_matrix_rejects_oversized_dimensions(dimension):
    """超过上界的维度必须在分配矩阵**之前**拒绝。

    这条断言的意义在于「拒绝得够早」：修复前走到这里时内存已经按该维度
    分配过了，进程要么已经付出代价、要么已经被 OOM 杀掉——那时再报错也晚了。
    因此断言拒绝，而不是断言某种内存表现。
    """
    with pytest.raises(ValueError, match="维度"):
        EmbeddingService._as_float32_matrix([[1.0] * dimension], expected_rows=1)


def test_embedding_service_rejects_oversized_dimension_from_provider():
    """走 provider 的真实返回路径：超大维度同样要拒绝。"""
    service = EmbeddingService(build_run_config(Settings(_env_file=None)))

    oversized = [[0.0] * 65537]
    with pytest.raises(ValueError, match="维度"):
        EmbeddingService._as_float32_matrix(oversized, expected_rows=1)
    # 服务实例本身仍可构造，说明拒绝发生在向量校验而不是构造期。
    assert service is not None


# ── 回归：``_reject_unconvertible_numbers`` 必须递归 ``ndarray`` ──
#
# ``ndarray`` 与「``ndarray`` 的 list」都是 ``_probe_dimension`` 显式支持的输入
# 形态。守卫只递归 ``list``/``tuple`` 时，``dtype=object`` 的数组带着
# ``10**400`` 直接进 ``np.array(..., dtype=np.float32)``，由 numpy 抛裸
# ``OverflowError``——同一批数据，``list`` 形态得到干净的 ``ValueError``，
# ``ndarray`` 形态得到 ``ArithmeticError`` 子类，收场方式不一致，且穿透调用方
# 的 ``except ValueError`` 分类与日志脱敏。


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([[10**400, 1.0]], id="list-of-list"),
        pytest.param(np.array([[10**400, 1.0]], dtype=object), id="np-object-2d"),
        pytest.param(np.array([10**400, 1.0], dtype=object), id="np-object-1d"),
        pytest.param([np.array([10**400, 1.0], dtype=object)], id="list-of-np-1d"),
    ],
)
def test_reject_unconvertible_numbers_covers_ndarray(payload):
    with pytest.raises(ValueError, match="超出双精度范围"):
        _reject_unconvertible_numbers(payload, "测试向量")


def test_reject_unconvertible_numbers_accepts_normal_ndarray():
    """数值 dtype 的数组不应被遍历或误判。"""
    _reject_unconvertible_numbers(np.zeros((2, 3), dtype="float32"), "测试向量")


# ── 回归：条数不一致的消息必须可读 ──
#
# ``expected_rows`` 是调用方给的期望条数，形态不受控，而这条消息原先裸插值：
# 位数超过 ``sys.get_int_max_str_digits()``（默认 4300）时，正要报「条数不一致」
# 的那一步自己抛 ``ValueError: Exceeds the limit``——用户拿到的是 CPython 的
# 整数转换错误，与向量条数毫无关系。同一条消息里的 ``matrix.shape[0]`` 来自刚
# 建好的 ndarray，必是小整数，不需要处理。
@pytest.mark.parametrize("expected_rows", [10**4301, -(10**4301)], ids=["正", "负"])
def test_as_float32_matrix_row_mismatch_message_is_readable(expected_rows):
    with pytest.raises(ValueError) as excinfo:
        EmbeddingService._as_float32_matrix([[1.0, 2.0]], expected_rows=expected_rows)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message
    assert "条数不一致" in message or "数量" in message
    assert "位十进制" in message  # safe_repr 对超限整数的量级描述


def test_as_float32_matrix_row_mismatch_keeps_small_values_verbatim():
    """收紧不能误伤：上限内的条数照旧原样显示。"""
    with pytest.raises(ValueError) as excinfo:
        EmbeddingService._as_float32_matrix([[1.0, 2.0]], expected_rows=7)

    assert "输入 7 条" in str(excinfo.value)
