# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Tuple

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
        """同步调用语言模型。"""
        pass

    @abstractmethod
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """异步调用语言模型。"""
        pass

class TextEmbeddingModel(ABC):
    """文本向量化模型抽象基类"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步将文档列表向量化。"""
        pass

    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步将文档列表向量化。"""
        pass

class RerankModel(ABC):
    """Rerank模型抽象基类"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int) -> Tuple[List[int], List[float]]:
        """同步对文档列表进行重排序。"""
        pass

    @abstractmethod
    async def arerank(self, query: str, documents: List[str], top_n: int) -> Tuple[List[int], List[float]]:
        """异步对文档列表进行重排序。"""
        pass