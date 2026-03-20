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

    def upsert_embeddings(self, documents: List[Dict[str, Any]], embeddings: Any):
        """根据外部已生成的向量写入文档。默认未实现。"""
        raise NotImplementedError("当前向量存储未实现 upsert_embeddings。")

    def semantic_search(self, query_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """使用查询向量执行语义检索。默认未实现。"""
        raise NotImplementedError("当前向量存储未实现 semantic_search。")

    def keyword_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """使用原始文本执行关键词检索。默认退回 search。"""
        return self.search(query_text, top_k=top_k, search_type="keyword")

    def save_snapshot(self, snapshot_dir: str):
        """将当前状态保存为快照。默认未实现。"""
        raise NotImplementedError("当前向量存储未实现 save_snapshot。")

    def load_snapshot(self, snapshot_dir: str):
        """从快照目录加载状态。默认未实现。"""
        raise NotImplementedError("当前向量存储未实现 load_snapshot。")

    def register_parent_documents(self, parent_documents: Dict[str, Dict[str, Any]]):
        """注册父分段侧车数据。"""
        return None

    def resolve_parent_content(self, parent_id: str | None) -> str | None:
        """根据父分段 ID 解析父内容。"""
        return None
