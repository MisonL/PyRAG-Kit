# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from abc import ABC, abstractmethod
from typing import List
from src.models.document import Document

class BaseCleaner(ABC):
    """
    抽象基类：文档内容清洗器。
    定义了对原始文本内容进行清洗（如去除多余空格、特殊字符、HTML标签等）的接口。
    """

    @abstractmethod
    def clean(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        对文档列表中的文本内容进行清洗。

        Args:
            documents (List[Document]): 待清洗的文档对象列表。
            **kwargs: 额外的参数，用于控制清洗行为。

        Returns:
            List[Document]: 清洗后的文档对象列表。
        """
        pass
