from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSplitter(ABC):
    """
    抽象基类：文档内容分割器。
    定义了将长文本分割成更小、更易于处理的块（chunks）的接口。
    """

    @abstractmethod
    def split(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        将文档列表中的文本内容分割成更小的块。

        Args:
            documents (List[Dict[str, Any]]): 待分割的文档列表，每个文档字典至少包含 'content' 键。
            **kwargs: 额外的参数，用于控制分割行为，如 chunk_size, chunk_overlap, separators等。

        Returns:
            List[Dict[str, Any]]: 分割后的文本块列表，每个块是一个字典，
                                  至少包含 'content' 键（文本块内容）。
                                  可以包含 'metadata' 键（文本块元数据，如来源、页码等）。
        """
        pass
