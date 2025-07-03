from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseExtractor(ABC):
    """
    抽象基类：文档内容抽取器。
    定义了从不同来源（如文件、URL等）抽取原始文本内容的接口。
    """

    @abstractmethod
    def extract(self, source: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        从给定源中抽取文本内容。

        Args:
            source (Any): 待抽取内容的源，可以是文件路径、URL、字节流等。
            **kwargs: 额外的参数，用于控制抽取行为。

        Returns:
            List[Dict[str, Any]]: 抽取出的文档列表，每个文档是一个字典，
                                  至少包含 'content' 键（原始文本内容）。
                                  可以包含 'metadata' 键（文档元数据）。
        """
        pass
