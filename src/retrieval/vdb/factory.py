# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ...runtime.contracts import KnowledgeSnapshotManifest, build_run_config
from ...ui.display_utils import document_source
from ...utils.config import get_settings, require_detail_attributes, safe_repr
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
        # 先判类型再调 ``.lower()``：非字符串取值抛 ``AttributeError``，而本方法
        # 的契约是 ``ValueError``。``Settings.default_vector_store`` 已能拦住
        # 配置来源的非字符串，但本方法是公开入口（测试与库调用方直接调用），
        # 与 ``RecursiveTextSplitter.structure_mode``、``SessionConfig`` 的融合
        # 策略同一判据。
        if not isinstance(store_type, str):
            raise ValueError(f"不支持的向量存储类型: {safe_repr(store_type)}")
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
    def _require_embedding_detail(run_config) -> Any:
        """按 ``default_embedding_provider`` 取活动 embedding 配置详情。

        ``_load_existing_state`` 的两条分支（活动快照与 legacy 导入）都要用同一
        份配置去构造 manifest。此前两条分支的判据不一致：活动快照分支显式判过
        「键不在映射里」，给出带字段名与指引的 ``ValueError``；legacy 分支直接
        下标取值，配置里把 provider 名打错（或漏写整段
        ``embedding_configurations``）时抛的是裸 ``KeyError: 'local-hash-typo'``
        ——没有字段名、没有可选值，也不是调用方按 ``ValueError`` 建立的错误
        分类。同一个「embedding 配置缺失」的错误，收场不该取决于磁盘上恰好有
        没有活动快照。

        容器形态也要判：``RunConfig`` 是 frozen dataclass、字段无校验器，嵌入方
        可以自拼一个 ``embedding_configurations`` 非映射的配置送进来，那时
        ``.get`` 抛裸 ``AttributeError``。判据与
        ``EmbeddingService.embedding_model_detail`` 一致（同一字段、同一来源）。

        键在**不**代表值可用（同 ``EmbeddingService.embedding_model_detail``）：
        ``{"local-hash": None}`` 的键确实存在，于是不会走「缺少活动提供商」的
        分支，而调用方紧接着读 ``.model_name``——抛的是裸 ``AttributeError``，
        穿透调用方按 ``ValueError`` 建立的错误分类。这里一并收敛。
        """
        configurations = run_config.embedding_configurations
        key = run_config.default_embedding_provider
        # 键形态必须在成员判断**之前**收敛，判据与措辞都与
        # ``EmbeddingService.embedding_model_detail`` 的同名守卫一致（两处是同一段
        # 逻辑的副本，同一字段、同一来源）：``key not in configurations`` 对不可
        # 哈希的取值抛裸 ``TypeError: unhashable type: 'list'``，而本函数的契约是
        # 带指引的 ``ValueError``。
        if not isinstance(key, str):
            raise ValueError(
                "default_embedding_provider 必须是字符串，"
                f"但得到 {safe_repr(key)}。"
                " 请检查 config.toml 里的 default_embedding_provider 与"
                " embedding_configurations 是否对应。"
            )
        if not isinstance(configurations, Mapping) or key not in configurations:
            raise ValueError(
                f"当前 embedding 配置缺少活动提供商: {safe_repr(key)}。"
                " 请检查 config.toml 里的 default_embedding_provider 与"
                " embedding_configurations 是否对应。"
            )
        detail = configurations[key]
        require_detail_attributes(
            detail,
            ("provider", "model_name"),
            field="embedding_configurations",
            key=key,
            role_label="Embedding",
        )
        return detail

    @staticmethod
    def _load_existing_state(store: VectorStoreBase, run_config) -> None:
        snapshot_repository = SnapshotRepository(run_config)
        active_snapshot_dir = snapshot_repository.get_active_snapshot_dir()
        if active_snapshot_dir is not None:
            snapshot_repository.validate_snapshot_dir(active_snapshot_dir)
            manifest = snapshot_repository.load_manifest(active_snapshot_dir)
            embedding_detail = VectorStoreFactory._require_embedding_detail(run_config)
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

        # 形态先收敛：``Path(5)``／``Path(["a"])`` 抛裸 ``TypeError: expected str,
        # bytes or os.PathLike object, not ...``，位数超限的整数抛 ``ValueError:
        # Exceeds the limit``——两条都不是本模块其余拒绝路径承诺的带指引
        # ``ValueError``。``RunConfig`` 是 frozen dataclass、字段没有校验器（同文件
        # ``_require_embedding_detail`` 已认领这条可达性），而本方法由
        # ``get_default_vector_store`` 在 ``Chatbot`` 初始化时调用。判据与
        # ``FaissStore.load``／``resolve_within`` 的同类守卫一致。
        legacy_value = run_config.legacy_pkl_path
        if not isinstance(legacy_value, (str, os.PathLike)):
            raise ValueError(
                "legacy_pkl_path 必须是字符串或路径，"
                f"但得到 {type(legacy_value).__name__}。"
                " 请检查 config.toml 里的 pkl_path 配置项。"
            )
        legacy_path = Path(legacy_value)
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
            # 与活动快照分支同一处取值、同一判据（见 ``_require_embedding_detail``）：
            # 此前这里直接下标，缺配置时抛裸 ``KeyError``。
            embedding_detail = VectorStoreFactory._require_embedding_detail(run_config)
            manifest = KnowledgeSnapshotManifest.create(
                snapshot_id=snapshot_id,
                store_type=run_config.default_vector_store,
                embedding_provider=run_config.default_embedding_provider,
                embedding_model=embedding_detail.model_name,
                chunk_mode="legacy-import",
                source_digest="legacy-import",
                # ``metadata`` 的形态由助手收敛：``FaissStore`` 会在载入期归一化，
                # 但本分支正是为**第三方 store** 准备的（``get_default_vector_store``
                # 可以返回任何 ``VectorStoreBase`` 实现），它的 ``documents`` 里
                # ``metadata=None``／列表、甚至条目本身不是映射都会让这里的
                # ``.get(...).get(...)`` 抛 ``AttributeError``。
                document_count=len(
                    {document_source(doc, "") for doc in getattr(store, "documents", [])}
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
