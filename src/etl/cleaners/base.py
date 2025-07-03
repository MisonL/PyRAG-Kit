from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseCleaner(ABC):
    """
    抽象基类：文档内容清洗器。
    定义了对原始文本内容进行清洗（如去除多余空格、特殊字符、HTML标签等）的接口。
    """

    @abstractmethod
    def clean(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        对文档列表中的文本内容进行清洗。

        Args:
            documents (List[Dict[str, Any]]): 待清洗的文档列表，每个文档字典至少包含 'content' 键。
            **kwargs: 额外的参数，用于控制清洗行为。

        Returns:
            List[Dict[str, Any]]: 清洗后的文档列表，每个文档字典至少包含 'content' 键。
        """
        pass
