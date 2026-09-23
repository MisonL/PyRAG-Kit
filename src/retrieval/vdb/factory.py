# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from collections.abc import Iterable
from pathlib import Path

from ...runtime.contracts import KnowledgeSnapshotManifest, build_run_config
from ...utils.config import get_settings
from ...utils.log_manager import get_module_logger
from ..snapshot_repository import SnapshotRepository
from .base import VectorStoreBase
from .faiss_store import FaissStore

logger = get_module_logger(__name__)


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(
        store_type: str,
        file_path: str | None = None,
        *,
        trusted_paths: Iterable[str | Path] | None = None,
    ) -> VectorStoreBase:
        if store_type.lower() == "faiss":
            return FaissStore(file_path=file_path, trusted_paths=trusted_paths)
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
            manifest = snapshot_repository.load_manifest(active_snapshot_dir)
            embedding_detail = run_config.embedding_configurations.get(
                run_config.default_embedding_provider
            )
            if embedding_detail is None:
                raise ValueError(
                    f"当前 embedding 配置缺少活动提供商: {run_config.default_embedding_provider}"
                )
            if (
                manifest.embedding_provider != run_config.default_embedding_provider
                or manifest.embedding_model != embedding_detail.model_name
            ):
                raise RuntimeError(
                    "活动知识快照的 embedding 配置与当前运行配置不一致: "
                    f"快照={manifest.embedding_provider}/{manifest.embedding_model}，"
                    f"当前={run_config.default_embedding_provider}/{embedding_detail.model_name}。"
                    "请重建知识快照或切换回原 embedding 配置。"
                )
            store.load_snapshot(str(active_snapshot_dir), snapshot_root=run_config.snapshot_root)
            return

        legacy_path = Path(run_config.legacy_pkl_path)
        if not legacy_path.exists():
            return

        # 只信任配置里声明的 legacy 文件所在目录：这条路径要 pickle.load，
        # 而路径本身不能证明来源。
        store.load(str(legacy_path), trusted_paths=[legacy_path.parent])
        snapshot_id = snapshot_repository.generate_snapshot_id(prefix="legacy")
        temp_dir = snapshot_repository.create_temp_snapshot_dir(snapshot_id)
        finalized = False
        try:
            store.save_snapshot(str(temp_dir))
            manifest = KnowledgeSnapshotManifest.create(
                snapshot_id=snapshot_id,
                store_type=run_config.default_vector_store,
                embedding_provider=run_config.default_embedding_provider,
                embedding_model=run_config.embedding_configurations[
                    run_config.default_embedding_provider
                ].model_name,
                chunk_mode="legacy-import",
                source_digest="legacy-import",
                document_count=len(
                    {
                        doc.get("metadata", {}).get("source")
                        for doc in getattr(store, "documents", [])
                    }
                ),
                chunk_count=len(getattr(store, "documents", [])),
            )
            snapshot_repository.write_manifest(temp_dir, manifest)
            # 先在临时目录里校验再 finalize：finalize 会更新 ACTIVE_SNAPSHOT，
            # 校验放到它后面就只剩「坏快照已经被激活」这一种收场方式。
            snapshot_repository.validate_snapshot_dir(temp_dir)
            final_dir = snapshot_repository.finalize_snapshot(temp_dir, snapshot_id)
            snapshot_repository.validate_snapshot_dir(final_dir)
            finalized = True
        finally:
            if not finalized:
                # 与 KnowledgeBuildService.build 一致：清理失败只记日志，
                # 不能掩盖真正的失败原因。
                try:
                    snapshot_repository.cleanup_temp_snapshot_dir(snapshot_id)
                except Exception:
                    logger.exception("清理临时知识快照失败: %s", temp_dir)
