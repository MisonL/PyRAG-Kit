# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from abc import ABC, abstractmethod
from typing import List
from src.models.document import Document

class BaseExtractor(ABC):
    """
    抽象基类：文档内容抽取器。
    定义了从一个代表源的 Document 对象中抽取内容的接口。
    """

    @abstractmethod
    def extract(self, document: Document, **kwargs) -> List[Document]:
        """
        从给定的 Document 对象中抽取内容。
        对于某些类型（如纯文本），可能只是原样返回。
        对于其他类型（如PDF），则会执行实际的文本提取逻辑。

        Args:
            document (Document): 包含原始内容和元数据的文档对象。
            **kwargs: 额外的参数，用于控制抽取行为。

        Returns:
            List[Document]: 抽取出的一个或多个文档对象列表。
        """
        pass
