# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from abc import ABC, abstractmethod
from typing import List
from src.models.document import Document

class BaseSplitter(ABC):
    """
    抽象基类：文档内容分割器。
    定义了将长文本分割成更小、更易于处理的块（chunks）的接口。
    """

    @abstractmethod
    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        将文档列表中的文本内容分割成更小的块。

        Args:
            documents (List[Document]): 待分割的文档对象列表。
            **kwargs: 额外的参数，用于控制分割行为，如 chunk_size, chunk_overlap, separators等。

        Returns:
            List[Document]: 分割后的文档块列表。
        """
        pass
