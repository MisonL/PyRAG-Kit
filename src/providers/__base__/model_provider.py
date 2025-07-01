# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List

from typing import Any, Dict, Generator, List

class LargeLanguageModel(ABC):
    """语言模型抽象基类"""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用语言模型。

        :param prompt: 用户输入。
        :param system_prompt: 系统提示。
        :param tools: 工具列表。
        :param stream: 是否流式输出。
        :param temperature: 温度参数。
        :return: 一个生成器，用于流式输出结果。
        """
        pass

class TextEmbeddingModel(ABC):
    """文本向量化模型抽象基类"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将文档列表向量化。

        :param texts: 文本列表。
        :return: 向量列表。
        """
        pass

class RerankModel(ABC):
    """Rerank模型抽象基类"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int) -> List[int]:
        """
        对文档列表进行重排序。

        :param query: 查询语句。
        :param documents: 待排序的文档列表。
        :param top_n: 需要返回的重排后文档数量。
        :return: 排序后的文档索引列表。
        """
        pass