from __future__ import annotations

import math
from collections.abc import Callable, Iterator, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from src.utils.config import (
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    ModelDetail,
    RetrievalMethod,
    Settings,
)

SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class RunConfig:
    knowledge_base_path: Path
    legacy_pkl_path: Path
    snapshot_root: Path
    cache_path: Path
    log_path: Path
    default_vector_store: str
    default_llm_provider: str
    default_embedding_provider: str
    default_rerank_provider: str
    llm_configurations: dict[str, ModelDetail]
    embedding_configurations: dict[str, ModelDetail]
    rerank_configurations: dict[str, ModelDetail]
    chat_temperature: float
    kb_embedding_batch_size: int
    kb_chunk_size: int
    kb_chunk_overlap: int
    kb_child_chunk_size: int
    kb_child_chunk_overlap: int

    @property
    def active_snapshot_marker(self) -> Path:
        return self.snapshot_root / "ACTIVE_SNAPSHOT"


class SessionConfig(MutableMapping[str, Any]):
    def __init__(
        self,
        retrieval_method: RetrievalMethod,
        vector_weight: float,
        keyword_weight: float,
        hybrid_fusion_strategy: str,
        retrieval_candidate_multiplier: int,
        rerank_enabled: bool,
        top_k: int,
        score_threshold: float,
        active_llm_configuration: str,
        active_rerank_configuration: str,
        llm_configurations: dict[str, ModelDetail],
        rerank_configurations: dict[str, ModelDetail],
        chat_temperature: float,
    ):
        self.retrieval_method = retrieval_method
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.hybrid_fusion_strategy = hybrid_fusion_strategy
        self.retrieval_candidate_multiplier = retrieval_candidate_multiplier
        self.rerank_enabled = rerank_enabled
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.active_llm_configuration = active_llm_configuration
        self.active_rerank_configuration = active_rerank_configuration
        self.llm_configurations = llm_configurations
        self.rerank_configurations = rerank_configurations
        self.chat_temperature = chat_temperature

    # 可经 ``__setitem__`` 写入的字段及其校验器。``__init__`` 信任调用方
    # （``build_session_config`` 从已校验的 ``Settings`` 取值，合法），
    # 但 UI 与库调用方走 ``__setitem__``，必须逐字段校验。
    #
    # 这里不是重复 `Settings` 的工作：`Settings` 只在启动时校验一次，而
    # `SessionConfig` 是运行期可变的独立入口。修 `top_k` 上界之前实测
    # `chat_config["top_k"] = 10**9` 能一路走到 `faiss_index.search()`，
    # FAISS 既不报错也不截断，按 10 亿条分配。注册表在类体外填充（类体内直接
    # 引用 `cls._validate_x` 会在定义期拿不到绑定方法）。
    _VALIDATORS: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.to_dict():
            raise KeyError(f"SessionConfig 没有字段 {key!r}。")
        validator = self._VALIDATORS.get(key)
        setattr(self, key, validator(value) if validator is not None else value)

    @staticmethod
    def _require_bounded_int(value: Any, label: str, upper: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} 必须是整数，但得到 {value!r}。")
        if not 1 <= value <= upper:
            raise ValueError(f"{label} 必须是 1 到 {upper} 之间的整数，但得到 {value}。")
        return value

    @classmethod
    def _validate_top_k(cls, value: Any) -> int:
        return cls._require_bounded_int(value, "top_k", _MAX_RETRIEVAL_TOP_K)

    @classmethod
    def _validate_candidate_multiplier(cls, value: Any) -> int:
        return cls._require_bounded_int(
            value, "retrieval_candidate_multiplier", _MAX_RETRIEVAL_MULTIPLIER
        )

    @staticmethod
    def _validate_unit_weight(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} 必须是数值，但得到 {value!r}。")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} 必须是有限数值，但得到 {value!r}。")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label} 必须在 0.0 到 1.0 之间，但得到 {value!r}。")
        return number

    @classmethod
    def _validate_vector_weight(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "vector_weight")

    @classmethod
    def _validate_keyword_weight(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "keyword_weight")

    @classmethod
    def _validate_score_threshold(cls, value: Any) -> float:
        return cls._validate_unit_weight(value, "score_threshold")

    @classmethod
    def _validate_retrieval_method(cls, value: Any) -> RetrievalMethod:
        if isinstance(value, RetrievalMethod):
            return value
        try:
            return RetrievalMethod(value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in RetrievalMethod)
            raise ValueError(f"retrieval_method 必须是 {allowed} 之一，但得到 {value!r}。") from exc

    @staticmethod
    def _validate_fusion_strategy(value: Any) -> str:
        if value not in {"rrf", "weighted"}:
            raise ValueError(
                f"hybrid_fusion_strategy 必须是 'rrf' 或 'weighted'，但得到 {value!r}。"
            )
        return str(value)

    @staticmethod
    def _validate_rerank_enabled(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"rerank_enabled 必须是布尔值，但得到 {value!r}。")
        return value

    @staticmethod
    def _validate_chat_temperature(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"chat_temperature 必须是数值，但得到 {value!r}。")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"chat_temperature 必须是非负有限数值，但得到 {value!r}。")
        return number

    def __delitem__(self, key: str) -> None:
        raise TypeError("SessionConfig 不支持删除字段。")

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_method": self.retrieval_method,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "hybrid_fusion_strategy": self.hybrid_fusion_strategy,
            "retrieval_candidate_multiplier": self.retrieval_candidate_multiplier,
            "rerank_enabled": self.rerank_enabled,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "active_llm_configuration": self.active_llm_configuration,
            "active_rerank_configuration": self.active_rerank_configuration,
            "llm_configurations": self.llm_configurations,
            "rerank_configurations": self.rerank_configurations,
            "chat_temperature": self.chat_temperature,
        }


# 每个可经 ``__setitem__`` 写入的字段都要在这里登记；未登记的字段（模型配置字典
# 等）按原样写入，由使用方负责其内部结构。
SessionConfig._VALIDATORS = {
    "retrieval_method": SessionConfig._validate_retrieval_method,
    "vector_weight": SessionConfig._validate_vector_weight,
    "keyword_weight": SessionConfig._validate_keyword_weight,
    "hybrid_fusion_strategy": SessionConfig._validate_fusion_strategy,
    "retrieval_candidate_multiplier": SessionConfig._validate_candidate_multiplier,
    "rerank_enabled": SessionConfig._validate_rerank_enabled,
    "top_k": SessionConfig._validate_top_k,
    "score_threshold": SessionConfig._validate_score_threshold,
    "chat_temperature": SessionConfig._validate_chat_temperature,
}


@dataclass(frozen=True)
class KnowledgeSnapshotManifest:
    schema_version: str
    snapshot_id: str
    created_at: str
    store_type: str
    embedding_provider: str
    embedding_model: str
    chunk_mode: str
    source_digest: str
    document_count: int
    chunk_count: int

    @classmethod
    def create(
        cls,
        snapshot_id: str,
        store_type: str,
        embedding_provider: str,
        embedding_model: str,
        chunk_mode: str,
        source_digest: str,
        document_count: int,
        chunk_count: int,
    ) -> KnowledgeSnapshotManifest:
        return cls(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            created_at=datetime.now(UTC).isoformat(),
            store_type=store_type,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_mode=chunk_mode,
            source_digest=source_digest,
            document_count=document_count,
            chunk_count=chunk_count,
        )

    def to_toml(self) -> str:
        return "\n".join(
            [
                f'schema_version = "{self.schema_version}"',
                f'snapshot_id = "{self.snapshot_id}"',
                f'created_at = "{self.created_at}"',
                f'store_type = "{self.store_type}"',
                f'embedding_provider = "{self.embedding_provider}"',
                f'embedding_model = "{self.embedding_model}"',
                f'chunk_mode = "{self.chunk_mode}"',
                f'source_digest = "{self.source_digest}"',
                f"document_count = {self.document_count}",
                f"chunk_count = {self.chunk_count}",
                "",
            ]
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> KnowledgeSnapshotManifest:
        return cls(
            schema_version=str(data["schema_version"]),
            snapshot_id=str(data["snapshot_id"]),
            created_at=str(data["created_at"]),
            store_type=str(data["store_type"]),
            embedding_provider=str(data["embedding_provider"]),
            embedding_model=str(data["embedding_model"]),
            chunk_mode=str(data["chunk_mode"]),
            source_digest=str(data["source_digest"]),
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
        )


def build_run_config(settings: Settings) -> RunConfig:
    return RunConfig(
        knowledge_base_path=Path(settings.knowledge_base_path),
        legacy_pkl_path=Path(settings.pkl_path),
        snapshot_root=Path(settings.snapshot_root),
        cache_path=Path(settings.cache_path),
        log_path=Path(settings.log_path),
        default_vector_store=settings.default_vector_store,
        default_llm_provider=settings.default_llm_provider,
        default_embedding_provider=settings.default_embedding_provider,
        default_rerank_provider=settings.default_rerank_provider,
        llm_configurations=deepcopy(settings.llm_configurations),
        embedding_configurations=deepcopy(settings.embedding_configurations),
        rerank_configurations=deepcopy(settings.rerank_configurations),
        chat_temperature=settings.chat_temperature,
        kb_embedding_batch_size=settings.kb_embedding_batch_size,
        kb_chunk_size=settings.kb_chunk_size,
        kb_chunk_overlap=settings.kb_chunk_overlap,
        kb_child_chunk_size=settings.kb_child_chunk_size,
        kb_child_chunk_overlap=settings.kb_child_chunk_overlap,
    )


def build_session_config(settings: Settings) -> SessionConfig:
    return SessionConfig(
        retrieval_method=settings.chat_retrieval_method,
        vector_weight=settings.chat_vector_weight,
        keyword_weight=settings.chat_keyword_weight,
        hybrid_fusion_strategy=settings.hybrid_fusion_strategy,
        retrieval_candidate_multiplier=settings.retrieval_candidate_multiplier,
        rerank_enabled=settings.chat_rerank_enabled,
        top_k=settings.chat_top_k,
        score_threshold=settings.chat_score_threshold,
        active_llm_configuration=settings.default_llm_provider,
        active_rerank_configuration=settings.default_rerank_provider,
        llm_configurations=deepcopy(settings.llm_configurations),
        rerank_configurations=deepcopy(settings.rerank_configurations),
        chat_temperature=settings.chat_temperature,
    )
