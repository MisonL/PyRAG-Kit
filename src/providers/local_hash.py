# -*- coding: utf-8 -*-
import asyncio
import hashlib
import math
import re
from typing import List

import jieba

from src.providers.__base__.model_provider import TextEmbeddingModel

DEFAULT_DIMENSION = 256
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

jieba.setLogLevel(jieba.logging.ERROR)


def _parse_dimension(model_name: str) -> int:
    suffix = model_name.rsplit("-", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() and int(suffix) > 0 else DEFAULT_DIMENSION


def _tokenize(text: str) -> List[str]:
    tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]
    return tokens if tokens else TOKEN_PATTERN.findall(text)


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class LocalHashEmbeddingProvider(TextEmbeddingModel):
    """离线可复现的本地哈希向量化模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = _parse_dimension(model_name)

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            primary_index = int.from_bytes(digest[:8], "big") % self.dimension
            secondary_index = int.from_bytes(digest[8:], "big") % self.dimension
            primary_sign = 1.0 if digest[0] % 2 == 0 else -1.0
            secondary_sign = 1.0 if digest[15] % 2 == 0 else -1.0
            vector[primary_index] += primary_sign
            vector[secondary_index] += secondary_sign
        return _normalize(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
