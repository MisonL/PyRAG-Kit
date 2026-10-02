from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

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

    async def build(self, splitter_structure_mode: str) -> dict[str, object]:
        # 形态先收敛：下一行直接对 ``knowledge_base_path`` 调 ``.glob()``，非
        # ``Path`` 取值在那里抛裸 ``AttributeError: 'int' object has no attribute
        # 'glob'``——不是本服务其余拒绝路径承诺的带指引 ``ValueError``。
        # ``RunConfig`` 是 frozen dataclass、字段没有校验器，嵌入方可以自拼
        # ``knowledge_base_path=5`` 的配置。只接受 ``Path``（``RunConfig`` 的标注
        # 就是它），不做静默强转。
        knowledge_base_path = self.run_config.knowledge_base_path
        if not isinstance(knowledge_base_path, Path):
            raise ValueError(
                "knowledge_base_path 必须是 Path，"
                f"但得到 {type(knowledge_base_path).__name__}。"
                " 请检查 config.toml 里的 knowledge_base_path 配置项。"
            )
        markdown_files = sorted(knowledge_base_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"知识库目录 '{knowledge_base_path}' 中未找到 Markdown 文件。")

        snapshot_id = self.snapshot_repository.generate_snapshot_id()
        temp_dir = self.snapshot_repository.create_temp_snapshot_dir(snapshot_id)
        finalized = False
        try:
            pipeline = Pipeline.from_file_path(
                markdown_files[0], splitter_structure_mode=splitter_structure_mode
            )

            chunk_count = 0
            pending_documents: list[dict[str, Any]] = []
            pending_embeddings: list[np.ndarray] = []
            pending_parent_documents: dict[str, dict[str, Any]] = {}
            for file_path in markdown_files:
                chunks = self._process_file(pipeline, file_path)
                if splitter_structure_mode == "hierarchical" and hasattr(
                    pipeline.splitter, "parent_documents"
                ):
                    parent_documents = getattr(pipeline.splitter, "parent_documents", {})
                    if isinstance(parent_documents, dict):
                        pending_parent_documents.update(parent_documents)
                if chunks:
                    documents = [
                        {"page_content": chunk.content, "metadata": chunk.metadata}
                        for chunk in chunks
                    ]
                    texts = [chunk.content for chunk in chunks]
                    pending_documents.extend(documents)
                    pending_embeddings.append(await self.embedding_service.embed_in_batches(texts))
                chunk_count += len(chunks)

            if pending_documents:
                embeddings = np.vstack(pending_embeddings)
                if len(pending_documents) != len(embeddings):
                    raise ValueError("知识快照中的文档数量与 embedding 数量不一致。")
                self.vector_store.upsert_embeddings(pending_documents, embeddings)
            if pending_parent_documents:
                self.vector_store.register_parent_documents(pending_parent_documents)

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
            self.snapshot_repository.validate_snapshot_dir(temp_dir)
            self.snapshot_repository.finalize_snapshot(temp_dir, snapshot_id)
            final_dir = self.snapshot_repository.get_active_snapshot_dir()
            if final_dir is None:
                raise RuntimeError("活动知识快照切换失败。")
            self.snapshot_repository.validate_snapshot_dir(final_dir)
            finalized = True
            return {
                "snapshot_id": snapshot_id,
                "snapshot_dir": str(final_dir),
                "document_count": len(markdown_files),
                "chunk_count": chunk_count,
                "embedding_provider": self.embedding_service.embedding_provider_key,
                "embedding_model": self.embedding_service.embedding_model_detail.model_name,
                "chunk_mode": splitter_structure_mode,
            }
        finally:
            if not finalized:
                try:
                    self.snapshot_repository.cleanup_temp_snapshot_dir(snapshot_id)
                except Exception:
                    logger.exception("清理临时知识快照失败: %s", temp_dir)

    def _process_file(self, pipeline: Pipeline, file_path: Path) -> list[Document]:
        logger.info("正在处理文件: %s", file_path)
        # 读取失败要把**文件名**带进错误里。知识库目录通常有几十个文件，而
        # ``UnicodeDecodeError`` 的消息只有字节偏移（``'utf-8' codec can't
        # decode byte 0xff in position 0``），不说是哪个文件——用户拿到的是
        # 「程序运行期间发生错误」加一串偏移量，无从下手。实测 39 个文件的
        # 知识库里混入一个非 UTF-8 文件即是这种收场。
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"知识库文件不是 UTF-8 编码，无法读取: {file_path}。"
                f" 请把该文件转为 UTF-8 后重试（{type(exc).__name__}: {exc}）。"
            ) from exc
        document = Document(content=content, metadata={"source": str(file_path)})
        chunks = pipeline.process(document)
        return chunks

    @staticmethod
    def _compute_source_digest(markdown_files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for file_path in markdown_files:
            digest.update(str(file_path).encode("utf-8"))
            digest.update(file_path.read_bytes())
        return digest.hexdigest()
