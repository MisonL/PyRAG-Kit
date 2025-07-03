# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from typing import Any
from .base import VectorStoreBase
from .faiss_store import FaissStore
from ...utils.config import get_settings # 导入 get_settings 函数

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
        current_settings = get_settings() # 获取当前配置
        default_store_type = current_settings.default_vector_store
        default_file_path = current_settings.pkl_path
        return VectorStoreFactory.get_vector_store(default_store_type, default_file_path)