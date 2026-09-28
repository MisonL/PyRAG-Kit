"""``VectorStoreFactory.get_vector_store`` 的 ``store_type`` 类型守卫。

单独成文件而不是放进 ``test_factory.py``：那个文件有一个 ``autouse`` 的
``patch_settings`` fixture，会把模块级 ``VectorStoreFactory`` 全局替换成 mock
版本（其中仍是旧的 ``store_type.lower()`` 实现），因此在那里断言「非字符串抛
``ValueError``」测到的是 mock，不是被测代码。
"""

import pytest

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
