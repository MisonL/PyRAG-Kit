from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from src.providers.__base__.model_provider import close_resource_sync
from src.providers.factory import ModelProviderFactory
from src.runtime.contracts import RunConfig


class EmbeddingService:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self._embedding_model = None

    @property
    def embedding_provider_key(self) -> str:
        return self.run_config.default_embedding_provider

    @property
    def embedding_model_detail(self):
        return self.run_config.embedding_configurations[self.embedding_provider_key]

    def _get_model(self):
        if self._embedding_model is None:
            self._embedding_model = ModelProviderFactory.get_embedding_provider(
                self.embedding_provider_key,
                self.run_config.embedding_configurations,
            )
        return self._embedding_model

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = await model.aembed_documents(texts)
        return self._as_float32_matrix(embeddings, expected_rows=len(texts))

    async def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        options: dict[str, Any] = {}
        if self.embedding_model_detail.provider == "google":
            options.setdefault("task_type", "RETRIEVAL_QUERY")
        vector = await model.aembed_query(query, **options)
        matrix = self._as_float32_matrix([vector], expected_rows=1)
        return matrix[0]

    async def embed_in_batches(self, texts: list[str]) -> np.ndarray:
        batch_size = max(1, self.run_config.kb_embedding_batch_size)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(await self.embed_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches)

    def close(self) -> None:
        """释放按需创建的 Embedding Provider。"""
        model = self._embedding_model
        self._embedding_model = None
        if model is None:
            return
        close_resource_sync(model, "Embedding Provider")

    async def aclose(self) -> None:
        """异步释放按需创建的 Embedding Provider。"""
        model = self._embedding_model
        self._embedding_model = None
        if model is None:
            return
        close = getattr(model, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return
        close = getattr(model, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _as_float32_matrix(embeddings: Any, expected_rows: int | None = None) -> np.ndarray:
        matrix = np.array(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("EmbeddingService 期望二维向量矩阵。")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("EmbeddingService 收到空向量矩阵。")
        if expected_rows is not None and matrix.shape[0] != expected_rows:
            raise ValueError(
                "EmbeddingService 收到的向量数量与输入文本数量不一致。"
                f" 输入 {expected_rows} 条，返回 {matrix.shape[0]} 条。"
            )
        return matrix
