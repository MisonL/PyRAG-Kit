# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStoreBase(ABC):
    """
    向量存储的抽象基类。
    定义了所有向量存储实现必须遵循的标准接口。
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """同步添加文档。"""
        pass

    @abstractmethod
    async def aadd_documents(self, documents: List[Dict[str, Any]]):
        """异步添加文档。"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """同步搜索文档。"""
        pass

    @abstractmethod
    async def asearch(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """异步搜索文档。"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存向量存储。"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载向量存储。"""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Any:
        """获取嵌入模型。"""
        pass