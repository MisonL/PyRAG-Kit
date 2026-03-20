# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

from src.etl.pipeline import Pipeline
from src.models.document import Document
from src.retrieval.snapshot_repository import SnapshotRepository
from src.retrieval.vdb.base import VectorStoreBase
from src.runtime.contracts import KnowledgeSnapshotManifest, RunConfig
from src.services.embedding_service import EmbeddingService
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)


class KnowledgeBuildService:
    def __init__(
        self,
        run_config: RunConfig,
        vector_store: VectorStoreBase,
        embedding_service: EmbeddingService,
        snapshot_repository: SnapshotRepository,
    ):
        self.run_config = run_config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.snapshot_repository = snapshot_repository

    async def build(self, splitter_structure_mode: str) -> Dict[str, object]:
        markdown_files = sorted(self.run_config.knowledge_base_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"知识库目录 '{self.run_config.knowledge_base_path}' 中未找到 Markdown 文件。")

        snapshot_id = self.snapshot_repository.generate_snapshot_id()
        temp_dir = self.snapshot_repository.create_temp_snapshot_dir(snapshot_id)
        pipeline = Pipeline.from_file_path(markdown_files[0], splitter_structure_mode=splitter_structure_mode)

        chunk_count = 0
        for file_path in markdown_files:
            chunks = self._process_file(pipeline, file_path, splitter_structure_mode)
            await self._persist_chunks(chunks)
            chunk_count += len(chunks)

        manifest = KnowledgeSnapshotManifest.create(
            snapshot_id=snapshot_id,
            store_type=self.run_config.default_vector_store,
            embedding_provider=self.embedding_service.embedding_provider_key,
            embedding_model=self.embedding_service.embedding_model_detail.model_name,
            chunk_mode=splitter_structure_mode,
            source_digest=self._compute_source_digest(markdown_files),
            document_count=len(markdown_files),
            chunk_count=chunk_count,
        )
        self.vector_store.save_snapshot(str(temp_dir))
        self.snapshot_repository.write_manifest(temp_dir, manifest)
        self.snapshot_repository.finalize_snapshot(temp_dir, snapshot_id)
        final_dir = self.snapshot_repository.get_active_snapshot_dir()
        if final_dir is None:
            raise RuntimeError("活动知识快照切换失败。")
        self.snapshot_repository.validate_snapshot_dir(final_dir)
        return {
            "snapshot_id": snapshot_id,
            "snapshot_dir": str(final_dir),
            "document_count": len(markdown_files),
            "chunk_count": chunk_count,
            "embedding_provider": self.embedding_service.embedding_provider_key,
            "embedding_model": self.embedding_service.embedding_model_detail.model_name,
            "chunk_mode": splitter_structure_mode,
        }

    def _process_file(self, pipeline: Pipeline, file_path: Path, splitter_structure_mode: str) -> List[Document]:
        logger.info("正在处理文件: %s", file_path)
        content = file_path.read_text(encoding="utf-8")
        document = Document(content=content, metadata={"source": str(file_path)})
        chunks = pipeline.process(document)
        if splitter_structure_mode == "hierarchical" and hasattr(pipeline.splitter, "parent_documents"):
            self.vector_store.register_parent_documents(getattr(pipeline.splitter, "parent_documents", {}))
        return chunks

    async def _persist_chunks(self, chunks: List[Document]) -> None:
        if not chunks:
            return
        documents = [{"page_content": chunk.content, "metadata": chunk.metadata} for chunk in chunks]
        texts = [doc["page_content"] for doc in documents]
        embeddings = await self.embedding_service.embed_in_batches(texts)
        self.vector_store.upsert_embeddings(documents, embeddings)

    @staticmethod
    def _compute_source_digest(markdown_files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for file_path in markdown_files:
            digest.update(str(file_path).encode("utf-8"))
            digest.update(file_path.read_bytes())
        return digest.hexdigest()
