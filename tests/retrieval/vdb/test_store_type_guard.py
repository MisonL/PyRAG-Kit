"""``VectorStoreFactory.get_vector_store`` 的 ``store_type`` 类型守卫。

单独成文件而不是放进 ``test_factory.py``：那个文件有一个 ``autouse`` 的
``patch_settings`` fixture，会把模块级 ``VectorStoreFactory`` 全局替换成 mock
版本（其中仍是旧的 ``store_type.lower()`` 实现），因此在那里断言「非字符串抛
``ValueError``」测到的是 mock，不是被测代码。
"""

import pytest

from src.retrieval.vdb.base import VectorStoreBase
from src.retrieval.vdb.factory import VectorStoreFactory


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(["faiss"], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(123, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_get_vector_store_rejects_non_string_store_type(bad_value):
    """非字符串必须是 ``ValueError``，而不是 ``AttributeError``。

    原实现直接调 ``store_type.lower()``。``Settings.default_vector_store`` 能拦
    住配置来源的非字符串，但本方法是公开入口，测试与库调用方会直接调用。
    """
    with pytest.raises(ValueError, match="向量存储类型"):
        VectorStoreFactory.get_vector_store(bad_value)


def test_get_vector_store_still_accepts_faiss():
    """正常取值不受守卫影响。"""
    store = VectorStoreFactory.get_vector_store("faiss")
    assert store is not None


# ── VectorStoreBase 的四个默认实现必须报「未实现」而不是静默返回 None ──
#
# 这四个默认方法在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的
# ``raise`` 换成 ``pass`` 后全量 3725 条仍全绿）。删除后行为确实改变（实测，
# 经一个只实现抽象方法的子类）：``upsert_embeddings``／``semantic_search``／
# ``save_snapshot``／``load_snapshot`` 全部**静默返回 None**。
#
# 后果按调用方分两类：``RetrievalService._get_vector_store_method`` 会把
# 「与基类同名的实现」判为未实现（靠的是 ``__func__ is`` 比对，不看返回值），
# 所以那条路径不受影响；但**第三方 store**（本仓的 ``VectorStoreBase`` 是公开
# 扩展点）直接实现子类、忘掉某个方法时，拿到的 ``None`` 会被当成「没有结果」
# 继续往下走——一次本该显式报错的接口缺失变成静默空结果。
class _MinimalStore(VectorStoreBase):
    """只实现抽象方法的最小 store，用于探测四个默认实现。"""

    def save(self, path):
        return "saved"

    def load(self, path, *, trusted_paths=None):
        return "loaded"

    def get_embedding_model(self):
        return None

    def add_documents(self, documents):
        return None

    def aadd_documents(self, documents):
        return None

    def search(self, query, top_k=5, search_type="semantic"):
        return []

    def asearch(self, query, top_k=5, search_type="semantic"):
        return []


@pytest.mark.parametrize(
    "method_name,args",
    [
        pytest.param("upsert_embeddings", ([], None), id="upsert-embeddings"),
        pytest.param("semantic_search", (None,), id="semantic-search"),
        pytest.param("save_snapshot", ("/tmp/x",), id="save-snapshot"),
        pytest.param("load_snapshot", ("/tmp/x",), id="load-snapshot"),
    ],
)
def test_vector_store_base_defaults_report_not_implemented(method_name, args):
    """未实现的接口必须抛 ``NotImplementedError``，不能静默返回 None。"""
    store = _MinimalStore()

    with pytest.raises(NotImplementedError, match="未实现"):
        getattr(store, method_name)(*args)


def test_get_vector_store_rejects_unsupported_string_store_type():
    """字符串但不支持的 store 类型必须抛 ``ValueError``，不能静默返回 ``None``。

    删除守卫后 ``get_vector_store("unsupported")`` 返回 ``None``——调用方拿到
    ``None`` 当向量存储用，崩在更远处。``test_factory.py`` 里那条同名用例断言
    的是 ``autouse`` fixture 换上的 **mock**（mock 里仍有旧的 ``raise``），
    真代码的这条守卫从未被反向覆盖。
    """
    with pytest.raises(ValueError, match="不支持的向量存储类型: unsupported"):
        VectorStoreFactory.get_vector_store("unsupported")


def test_get_vector_store_still_accepts_faiss_with_uppercase():
    """反向守卫：``FaissStore`` 的大小写归一化照常生效。"""
    store = VectorStoreFactory.get_vector_store("FAISS")

    assert store is not None
