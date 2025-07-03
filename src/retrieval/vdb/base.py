from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStoreBase(ABC):
    """
    向量存储的抽象基类。
    定义了所有向量存储实现必须遵循的标准接口。
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向向量存储中添加文档。
        
        Args:
            documents: 包含文档内容和元数据的字典列表。
                       每个字典应至少包含 'page_content' 键。
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        在向量存储中搜索与查询最相关的文档。

        Args:
            query: 搜索查询字符串。
            top_k: 返回最相关文档的数量。
            search_type: 搜索类型，可以是 'semantic'（向量检索）或 'keyword'（全文检索）。

        Returns:
            包含相关文档内容和元数据的字典列表。
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        将向量存储保存到指定路径。
        
        Args:
            path: 保存向量存储的路径。
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        从指定路径加载向量存储。
        
        Args:
            path: 加载向量存储的路径。
        """
        pass

    @abstractmethod
    def get_embedding_model(self) -> Any:
        """
        获取当前向量存储使用的嵌入模型实例。
        
        Returns:
            嵌入模型实例。
        """
        pass