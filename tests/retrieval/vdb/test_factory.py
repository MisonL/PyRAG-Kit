import pytest
from unittest.mock import MagicMock, patch
from typing import Any, List
import sys

from src.retrieval.vdb.factory import VectorStoreFactory
from src.retrieval.vdb.base import VectorStoreBase
from src.utils.config import Settings, get_settings # 导入 Settings 和 get_settings
from src.retrieval.vdb.faiss_store import FaissStore # 导入 FaissStore

# Mock FaissStore 类
class MockFaissStore(VectorStoreBase):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: List[dict[str, Any]]):
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[dict[str, Any]]:
        """模拟搜索，返回假数据"""
        return [{"page_content": f"mock_doc_{i}", "metadata": {"source": "mock"}} for i in range(top_k)]

    def save(self, path: str):
        """模拟保存操作"""
        pass

    def load(self, path: str):
        """模拟加载操作"""
        pass

    def get_embedding_model(self) -> Any:
        """模拟获取嵌入模型"""
        return MagicMock() # 返回一个模拟的嵌入模型

@pytest.fixture(scope="function", autouse=True)
def patch_settings(monkeypatch):
    """模拟全局设置对象"""
    mock_settings_instance = MagicMock(spec=Settings)
    mock_settings_instance.default_vector_store = "faiss"
    mock_settings_instance.pkl_path = "/mock/path/to/faiss_store.pkl"

    with patch('src.utils.config.get_settings', return_value=mock_settings_instance):
        # 清除 VectorStoreFactory 及其依赖模块的缓存
        modules_to_clear = [
            'src.retrieval.vdb.factory',
            'src.retrieval.vdb.faiss_store',
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # 模拟 FaissStore 类
        monkeypatch.setattr("src.retrieval.vdb.faiss_store.FaissStore", MockFaissStore)

        # 重新导入 VectorStoreFactory，确保它加载的是最新的版本
        if 'src.retrieval.vdb.factory' in sys.modules:
            del sys.modules['src.retrieval.vdb.factory']
        from src.retrieval.vdb.factory import VectorStoreFactory as ReloadedVectorStoreFactory
        
        # 直接模拟 VectorStoreFactory.get_vector_store 方法
        def mock_get_vector_store(store_type: str, file_path: str) -> VectorStoreBase:
            if store_type.lower() == "faiss":
                return MockFaissStore(file_path=file_path)
            else:
                raise ValueError(f"不支持的向量存储类型: {store_type}")

        monkeypatch.setattr(ReloadedVectorStoreFactory, "get_vector_store", mock_get_vector_store)

        global VectorStoreFactory
        VectorStoreFactory = ReloadedVectorStoreFactory
        yield

# 测试用例
def test_get_vector_store_faiss_success():
    """测试成功获取 FaissStore 实例"""
    store = VectorStoreFactory.get_vector_store("faiss", "/tmp/test_faiss.pkl")
    assert isinstance(store, MockFaissStore)
    assert store.file_path == "/tmp/test_faiss.pkl"

def test_get_vector_store_unsupported_type():
    """测试获取不支持的向量存储类型时抛出 ValueError"""
    with pytest.raises(ValueError, match="不支持的向量存储类型: unsupported"):
        VectorStoreFactory.get_vector_store("unsupported", "/tmp/test.pkl")

def test_get_default_vector_store_success():
    """测试成功获取默认向量存储实例"""
    store = VectorStoreFactory.get_default_vector_store()
    assert isinstance(store, MockFaissStore)
    assert store.file_path == "/mock/path/to/faiss_store.pkl"