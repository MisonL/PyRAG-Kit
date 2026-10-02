import asyncio

import pytest

from src.providers.local_hash import LocalHashEmbeddingProvider


def test_local_hash_embedding_is_deterministic():
    provider = LocalHashEmbeddingProvider("local-hash-32")

    first = provider.embed_documents(["测试文本", "another text"])
    second = provider.embed_documents(["测试文本", "another text"])

    assert first == second
    assert len(first[0]) == 32
    assert len(first[1]) == 32
    assert any(value != 0.0 for value in first[0])


def test_local_hash_embedding_async_matches_sync():
    provider = LocalHashEmbeddingProvider("local-hash-16")

    sync_vectors = provider.embed_documents(["异步一致性"])
    async_vectors = asyncio.run(provider.aembed_documents(["异步一致性"]))

    assert async_vectors == sync_vectors


# ── 回归：维度必须在分配向量之前受上界约束 ──
#
# ``_parse_dimension`` 从 ``model_name`` 的数字后缀取维度，而
# ``embedding_configurations.<key>.model_name`` 是 ``config.toml`` 里用户可写的
# 字段。``_embed_text`` 会**先按该维度分配出完整向量**再返回：实测
# ``local-hash-100000000`` 单次调用耗时 136 秒、峰值内存 3.9 GB，按真实知识库
# 115 个分块算 ``np.vstack`` 需要约 4.3 GB。等调用方拿到结果再判已经晚了——
# 代价已经付出，或者进程已经被 OOM 杀掉。


@pytest.mark.parametrize("model_name", ["local-hash-1", "local-hash-256", "local-hash-65536"])
def test_local_hash_accepts_dimensions_at_or_below_the_bound(model_name):
    """收紧不能误伤：上界以内（含边界）的 model_name 必须照常可用。"""
    provider = LocalHashEmbeddingProvider(model_name)

    assert len(provider.embed_documents(["文本"])[0]) == int(model_name.rsplit("-", 1)[-1])


@pytest.mark.parametrize(
    "model_name", ["local-hash-65537", "local-hash-1000000", "local-hash-100000000"]
)
def test_local_hash_rejects_oversized_dimensions_at_construction(model_name):
    """超过上界的维度必须在**构造期**拒绝，不能等到分配向量。

    构造期就拒绝意味着没有任何分配发生过——这正是修复的要点，也是断言
    「异常发生在构造而不是 embed」的原因：若只在 embed 里判，向量已经建好了。
    """
    with pytest.raises(ValueError, match="上界"):
        LocalHashEmbeddingProvider(model_name)


def test_local_hash_keeps_default_dimension_for_unnamed_suffixes():
    """没有数字后缀时仍回落到默认维度，不受上界影响。"""
    provider = LocalHashEmbeddingProvider("local-hash")

    assert len(provider.embed_documents(["文本"])[0]) == 256


# ── 回归：``isdigit()`` 为真不代表 ``int()`` 能解析 ──
#
# ``_parse_dimension`` 原先用 ``suffix.isdigit()`` 做判据，但 ``isdigit()`` 对
# 128 个 Unicode 码位为真（``²`` ``³`` ``①`` ``፩`` ``₅`` …）而 ``int()`` 对它
# 们一律抛 ``ValueError``。``model_name`` 是 ``config.toml`` 里用户可写的字段，
# 于是 ``model_name = "local-hash-²"`` 让构造期抛出一条裸解析错误——消息只有
# ``invalid literal for int() with base 10: '²'``，既不提 ``model_name`` 也不提
# 配置，用户无从知道该改哪里。
#
# 全码位穷举：``isdigit()`` 为真的 788 个码位里 128 个 ``int()`` 抛错；
# ``isdecimal()`` 为真的 660 个码位 ``int()`` **全部**可解析。因此 ``isdecimal()``
# 才是「这串字符是十进制整数」的准确判据。


@pytest.mark.parametrize(
    "suffix",
    ["²", "①", "፹", "₅", "¹", "⑴"],
)
def test_local_hash_falls_back_to_default_for_unicode_digits_int_cannot_parse(suffix):
    """``int()`` 解析不了的「数字」必须走默认维度，而不是抛裸解析错误。"""
    provider = LocalHashEmbeddingProvider(f"local-hash-{suffix}")

    assert len(provider.embed_documents(["文本"])[0]) == 256


@pytest.mark.parametrize("suffix", ["٣", "１２", "۷"])
def test_local_hash_still_parses_unicode_digits_int_can_parse(suffix):
    """收紧不能误伤：``int()`` 真能解析的 Unicode 十进制数字照常生效。

    ``٣``（阿拉伯-印度数字 3）、``１２``（全角 12）、``۷``（波斯数字 7）都
    是 ``isdecimal()`` 为真且 ``int()`` 可解析的合法十进制数字。
    """
    provider = LocalHashEmbeddingProvider(f"local-hash-{suffix}")

    assert len(provider.embed_documents(["文本"])[0]) == int(suffix)


def test_local_hash_falls_back_to_default_for_digit_strings_beyond_the_limit():
    """超长数字后缀必须回落默认维度，而不是抛 ``ValueError``。

    ``"9" * 4301`` 是合法十进制整数、``isdecimal()`` 也为真，但位数超过
    ``sys.get_int_max_str_digits()``（默认 4300），``int()`` 照样抛错。
    """
    import sys

    suffix = "9" * (sys.get_int_max_str_digits() + 1)
    provider = LocalHashEmbeddingProvider(f"local-hash-{suffix}")

    assert len(provider.embed_documents(["文本"])[0]) == 256


# ── 回归：非字符串 model_name 在构造期抛裸 AttributeError ──
#
# ``_parse_dimension`` 第一行就是 ``model_name.rsplit("-", maxsplit=1)``，而
# ``LocalHashEmbeddingProvider`` 是**公开类**，嵌入方可以直接构造（本仓测试
# 就是这么用 ``LocalHashEmbeddingProvider("local-hash-32")`` 的）。实测
# ``10**4301``／``["m"]``／``None``／``5`` 全部抛裸
# ``AttributeError: 'int' object has no attribute 'rsplit'``——不在本模块其余
# 拒绝路径承诺的 ``ValueError`` 契约内（维度超上界给的是带指引的 ``ValueError``）。
# ``model_name`` 的来源与 ``config.toml`` 的 ``model_name`` 同源；工厂路径已用
# ``require_model_name`` 收敛，但直接构造这条路径漏了。


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param(10**4301, id="huge-int"),
        pytest.param(["m"], id="list"),
        pytest.param(None, id="none"),
        pytest.param(5, id="int"),
    ],
)
def test_local_hash_rejects_non_string_model_name_readably(model_name):
    """非字符串 ``model_name`` 必须给出带字段名的 ``ValueError``。"""
    with pytest.raises(ValueError) as excinfo:
        LocalHashEmbeddingProvider(model_name)

    assert "model_name" in str(excinfo.value)


def test_local_hash_still_accepts_string_model_names():
    """收紧不能误伤：字符串形态照常解析。"""
    assert len(LocalHashEmbeddingProvider("local-hash-32").embed_documents(["文本"])[0]) == 32
    assert len(LocalHashEmbeddingProvider("local-hash").embed_documents(["文本"])[0]) == 256
