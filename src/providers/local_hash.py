import asyncio
import hashlib
import math
import re

import jieba  # type: ignore[import-untyped]

from src.providers.__base__.model_provider import TextEmbeddingModel, require_text_list
from src.utils.config import _MAX_EMBEDDING_DIMENSION, parse_decimal_int, safe_repr

DEFAULT_DIMENSION = 256
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

jieba.setLogLevel(jieba.logging.ERROR)


def _parse_dimension(model_name: str) -> int:
    """从 ``model_name`` 的数字后缀取维度，超过上界时直接拒绝。

    校验必须在这里而不是调用方：``_embed_text`` 会**先按该维度分配出完整向量**
    再返回，等调用方拿到结果再判已经晚了——实测 ``local-hash-100000000`` 单次
    调用耗时 136 秒、峰值内存 3.9 GB。维度是用户可写的 ``model_name`` 决定的，
    与 ``chat_top_k`` 那类规模参数同理，都需要上界。
    """
    # 形态先收敛：下一行对 ``model_name`` 调 ``.rsplit()``，非字符串在那里抛裸
    # ``AttributeError: 'int' object has no attribute 'rsplit'``——不是本模块其余
    # 拒绝路径承诺的 ``ValueError``（维度超上界给的是带指引的 ``ValueError``）。
    # ``LocalHashEmbeddingProvider`` 是公开类，嵌入方可以绕过工厂直接构造；工厂
    # 路径已用 ``require_model_name`` 收敛，这条路径同样要有判据。判据与
    # ``require_model_name`` 一致：只接受字符串。
    if not isinstance(model_name, str):
        raise ValueError(f"local-hash 的 model_name 必须是字符串，但得到 {safe_repr(model_name)}。")
    suffix = model_name.rsplit("-", maxsplit=1)[-1]
    # 解析走 ``parse_decimal_int``：它同时挡住「``isdigit()`` 为真但 ``int()``
    # 抛错」（``²`` ``①`` ``፩`` …）与「位数超过 ``sys.get_int_max_str_digits()``」
    # （``"9" * 4301`` 是合法整数却照样抛错）两类。任何一种漏掉都会变成一条
    # 不带 ``model_name`` 的裸解析错误（实测消息只有 "invalid literal for int()
    # with base 10: '²'"），用户无从知道是配置里的哪个字段写坏了。
    parsed = parse_decimal_int(suffix)
    dimension = parsed if parsed is not None and parsed > 0 else DEFAULT_DIMENSION
    if dimension > _MAX_EMBEDDING_DIMENSION:
        raise ValueError(
            f"local-hash 的维度超过上界：model_name={model_name!r} 解析出 {dimension}，"
            f"上界 {_MAX_EMBEDDING_DIMENSION}。请检查配置里的 model_name。"
        )
    return dimension


def _tokenize(text: str) -> list[str]:
    tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]
    return tokens if tokens else TOKEN_PATTERN.findall(text)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class LocalHashEmbeddingProvider(TextEmbeddingModel):
    """离线可复现的本地哈希向量化模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = _parse_dimension(model_name)

    def _embed_text(self, text: str) -> list[float]:
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

    @staticmethod
    def _validate_kwargs(kwargs: dict) -> None:
        if kwargs:
            unsupported = ", ".join(sorted(str(key) for key in kwargs))
            raise ValueError("LocalHash Embedding 不支持请求参数: " + unsupported)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        self._validate_kwargs(kwargs)
        texts = require_text_list(texts, "LocalHash")
        return [self._embed_text(text) for text in texts]

    async def aembed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        self._validate_kwargs(kwargs)
        return await asyncio.to_thread(self.embed_documents, texts)
