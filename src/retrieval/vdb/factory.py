# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from pathlib import Path

from .base import VectorStoreBase
from .faiss_store import FaissStore
from ..snapshot_repository import SnapshotRepository
from ...runtime.contracts import KnowledgeSnapshotManifest, build_run_config
from ...utils.config import get_settings


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(store_type: str, file_path: str | None = None) -> VectorStoreBase:
        if store_type.lower() == "faiss":
            return FaissStore(file_path=file_path)
        raise ValueError(f"不支持的向量存储类型: {store_type}")

    @staticmethod
    def get_default_vector_store(load_existing: bool = True) -> VectorStoreBase:
        settings = get_settings()
        run_config = build_run_config(settings)
        store = VectorStoreFactory.get_vector_store(run_config.default_vector_store, None)
        if load_existing:
            VectorStoreFactory._load_existing_state(store, run_config)
        return store

    @staticmethod
    def _load_existing_state(store: VectorStoreBase, run_config) -> None:
        snapshot_repository = SnapshotRepository(run_config)
        active_snapshot_dir = snapshot_repository.get_active_snapshot_dir()
        if active_snapshot_dir is not None:
            snapshot_repository.validate_snapshot_dir(active_snapshot_dir)
            store.load_snapshot(str(active_snapshot_dir))
            return

        legacy_path = Path(run_config.legacy_pkl_path)
        if not legacy_path.exists():
            return

        store.load(str(legacy_path))
        snapshot_id = snapshot_repository.generate_snapshot_id(prefix="legacy")
        temp_dir = snapshot_repository.create_temp_snapshot_dir(snapshot_id)
        store.save_snapshot(str(temp_dir))
        manifest = KnowledgeSnapshotManifest.create(
            snapshot_id=snapshot_id,
            store_type=run_config.default_vector_store,
            embedding_provider=run_config.default_embedding_provider,
            embedding_model=run_config.embedding_configurations[run_config.default_embedding_provider].model_name,
            chunk_mode="legacy-import",
            source_digest="legacy-import",
            document_count=len({doc.get("metadata", {}).get("source") for doc in getattr(store, "documents", [])}),
            chunk_count=len(getattr(store, "documents", [])),
        )
        snapshot_repository.write_manifest(temp_dir, manifest)
        final_dir = snapshot_repository.finalize_snapshot(temp_dir, snapshot_id)
        snapshot_repository.validate_snapshot_dir(final_dir)
