# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, MutableMapping

from src.utils.config import ModelDetail, RetrievalMethod, Settings


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
    llm_configurations: Dict[str, ModelDetail]
    embedding_configurations: Dict[str, ModelDetail]
    rerank_configurations: Dict[str, ModelDetail]
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
        llm_configurations: Dict[str, ModelDetail],
        rerank_configurations: Dict[str, ModelDetail],
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

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("SessionConfig 不支持删除字段。")

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
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
    ) -> "KnowledgeSnapshotManifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
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
    def from_mapping(cls, data: Dict[str, Any]) -> "KnowledgeSnapshotManifest":
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
