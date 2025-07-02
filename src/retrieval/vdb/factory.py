from typing import Any
from .base import VectorStoreBase
from .faiss_store import FaissStore
from ...utils.config import settings

class VectorStoreFactory:
    """
    向量存储工厂类。
    根据配置返回一个具体的 VectorStore 实例。
    """

    @staticmethod
    def get_vector_store(store_type: str, file_path: str) -> VectorStoreBase:
        """
        根据指定的存储类型获取向量存储实例。
        
        Args:
            store_type: 向量存储的类型（例如 "faiss"）。
            file_path: 向量存储文件的路径。
            
        Returns:
            VectorStoreBase 的实例。
            
        Raises:
            ValueError: 如果指定的存储类型不支持。
        """
        if store_type.lower() == "faiss":
            return FaissStore(file_path=file_path)
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")

    @staticmethod
    def get_default_vector_store() -> VectorStoreBase:
        """
        获取默认配置的向量存储实例。
        
        Returns:
            VectorStoreBase 的实例。
        """
        default_store_type = settings.default_vector_store
        default_file_path = settings.pkl_path
        return VectorStoreFactory.get_vector_store(default_store_type, default_file_path)