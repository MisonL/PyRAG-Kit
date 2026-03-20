# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Any, List

import numpy as np

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
            self._embedding_model = ModelProviderFactory.get_embedding_provider(self.embedding_provider_key)
        return self._embedding_model

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = await model.aembed_documents(texts)
        return self._as_float32_matrix(embeddings)

    async def embed_query(self, query: str) -> np.ndarray:
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    async def embed_in_batches(self, texts: List[str]) -> np.ndarray:
        batch_size = max(1, self.run_config.kb_embedding_batch_size)
        batches: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(await self.embed_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches)

    @staticmethod
    def _as_float32_matrix(embeddings: Any) -> np.ndarray:
        matrix = np.array(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("EmbeddingService 期望二维向量矩阵。")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("EmbeddingService 收到空向量矩阵。")
        return matrix
