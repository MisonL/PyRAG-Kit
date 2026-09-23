from __future__ import annotations

import json
import pickle  # nosec B403 - 仅用于读取应用自管的本地快照文件
import shutil
import tomllib
import uuid
from pathlib import Path

from src.runtime.contracts import SCHEMA_VERSION, KnowledgeSnapshotManifest, RunConfig
from src.utils.security import resolve_within


class SnapshotRepository:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.root = run_config.snapshot_root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_active_snapshot_id(self) -> str | None:
        marker = self.run_config.active_snapshot_marker
        if not marker.exists():
            return None
        snapshot_id = marker.read_text(encoding="utf-8").strip()
        return snapshot_id or None

    def get_active_snapshot_dir(self) -> Path | None:
        snapshot_id = self.get_active_snapshot_id()
        if not snapshot_id:
            return None
        snapshot_dir = self.root / snapshot_id
        return snapshot_dir if snapshot_dir.exists() else None

    def create_temp_snapshot_dir(self, snapshot_id: str) -> Path:
        temp_dir = self.root / f".tmp-{snapshot_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def cleanup_temp_snapshot_dir(self, snapshot_id: str) -> None:
        """删除指定构建任务的临时目录，不触碰正式快照。"""
        temp_dir = self.root / f".tmp-{snapshot_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def finalize_snapshot(self, temp_dir: Path, snapshot_id: str) -> Path:
        final_dir = self.root / snapshot_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        self.activate_snapshot(snapshot_id)
        return final_dir

    def activate_snapshot(self, snapshot_id: str) -> None:
        marker = self.run_config.active_snapshot_marker
        marker.parent.mkdir(parents=True, exist_ok=True)
        temp_marker = marker.with_suffix(".tmp")
        temp_marker.write_text(snapshot_id, encoding="utf-8")
        temp_marker.replace(marker)

    def generate_snapshot_id(self, prefix: str = "kb") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def write_manifest(self, snapshot_dir: Path, manifest: KnowledgeSnapshotManifest) -> None:
        (snapshot_dir / "manifest.toml").write_text(manifest.to_toml(), encoding="utf-8")

    def load_manifest(self, snapshot_dir: Path) -> KnowledgeSnapshotManifest:
        """读取 manifest，并校验 ``schema_version`` 与当前实现一致。

        版本放在这里判而不是 ``from_mapping`` 里：``from_mapping`` 是纯解析，
        塞进版本策略会让「解析」和「准入」两件事在同一处各判一次。未知版本必须
        拒绝——旧/新 schema 的字段语义可能已经变了（本项目 v1 到 v2 就调整过
        分块与 embedding 的落盘结构），照着当前代码去解释未知版本会得到错误结果
        而不是报错。
        """
        with (snapshot_dir / "manifest.toml").open("rb") as file:
            data = tomllib.load(file)
        manifest = KnowledgeSnapshotManifest.from_mapping(data)
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "知识快照的 schema 版本与当前程序不兼容。"
                f" 快照={manifest.schema_version}，当前程序={SCHEMA_VERSION}。"
                " 请重建知识快照；如需保留旧快照，请先用生成它的版本导出数据。"
            )
        return manifest

    def validate_snapshot_dir(self, snapshot_dir: Path) -> None:
        """校验快照文件齐全**且三个数据源的行数自洽**。

        只查文件存在性是不够的：实测手工构造 ``documents=3`` 而
        ``embeddings=(2,4)``（``faiss_index.ntotal=2``）的 store，
        ``save_snapshot`` 正常落盘、本方法通过、加载后也不报错——不一致被
        完整持久化。之后 ``semantic_search`` 拿 ``indices`` 去索引
        ``self.documents`` 会越界，而报错现场离真正的原因（写入时就不一致）
        很远。这里在加载/切换快照的入口就把三者对齐。
        """
        # 先确认目录归属，再谈内容：快照目录里的 chunks.pkl/parents.pkl/
        # lexical.index 都是 pickle，反序列化即执行代码，而「路径长在 root 附近」
        # 不构成来源证明。resolve_within 会把符号链接逃逸（root/link -> /etc）
        # 一并拦下。
        resolved = resolve_within(snapshot_dir, self.root, label="知识快照目录")
        if snapshot_dir.is_symlink():
            raise ValueError(f"知识快照目录不能是符号链接: {snapshot_dir}")
        if resolved.parent != Path(self.root).resolve():
            raise ValueError(
                f"知识快照目录必须直接位于快照根目录之下: {resolved}（根={self.root}）。"
            )
        snapshot_dir = resolved

        required_files = [
            snapshot_dir / "manifest.toml",
            snapshot_dir / "chunks.pkl",
            snapshot_dir / "parents.pkl",
            snapshot_dir / "semantic.index",
            snapshot_dir / "embeddings.npy",
            snapshot_dir / "stats.json",
        ]
        missing_files = [str(path.name) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"知识快照不完整，缺少文件: {', '.join(missing_files)}")

        self._validate_snapshot_row_counts(snapshot_dir)

    @staticmethod
    def _validate_snapshot_row_counts(snapshot_dir: Path) -> None:
        """确认 chunks / embeddings / 向量索引三者的行数一致。

        读取全部走本地受信快照目录（与 ``FaissStore.load_snapshot`` 同一
        前提），且只取形状不取内容语义。
        """
        import numpy as np

        # 解析失败要转成「快照不自洽」的 ValueError：这里是加载入口，抛裸的
        # EOFError/UnpicklingError 会让调用方看到与真实原因无关的异常类型
        # （测试里就复现过：占位空文件 chunks.pkl -> EOFError）。
        try:
            with (snapshot_dir / "chunks.pkl").open("rb") as file:
                chunks = pickle.load(file)  # nosec B301 - 应用自管的快照目录
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：chunks.pkl 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc

        try:
            embeddings = np.load(snapshot_dir / "embeddings.npy")
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：embeddings.npy 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc

        chunk_count = len(chunks) if chunks is not None else 0
        if embeddings.ndim != 2:
            raise ValueError(f"知识快照的 embeddings.npy 不是二维数组: ndim={embeddings.ndim}。")
        embedding_rows = int(embeddings.shape[0])

        if chunk_count != embedding_rows:
            raise ValueError(
                "知识快照不自洽：分块数与向量数不一致。"
                f" chunks.pkl={chunk_count}，embeddings.npy={embedding_rows}。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            )

        stats_path = snapshot_dir / "stats.json"
        try:
            with stats_path.open("rb") as file:
                stats = json.load(file)
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：stats.json 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        if not isinstance(stats, dict):
            raise ValueError("知识快照不自洽：stats.json 的顶层不是对象。")
        declared = stats.get("chunk_count")
        if declared is not None and int(declared) != chunk_count:
            raise ValueError(
                "知识快照不自洽：stats.json 的 chunk_count 与实际分块数不一致。"
                f" stats.json={declared}，chunks.pkl={chunk_count}。"
            )
