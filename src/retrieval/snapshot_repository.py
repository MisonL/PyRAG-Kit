# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from src.runtime.contracts import KnowledgeSnapshotManifest, RunConfig


class SnapshotRepository:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.root = run_config.snapshot_root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_active_snapshot_id(self) -> Optional[str]:
        marker = self.run_config.active_snapshot_marker
        if not marker.exists():
            return None
        snapshot_id = marker.read_text(encoding="utf-8").strip()
        return snapshot_id or None

    def get_active_snapshot_dir(self) -> Optional[Path]:
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
        with (snapshot_dir / "manifest.toml").open("rb") as file:
            data = tomllib.load(file)
        return KnowledgeSnapshotManifest.from_mapping(data)

    def write_stats(self, snapshot_dir: Path, stats: Dict[str, Any]) -> None:
        (snapshot_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_stats(self, snapshot_dir: Path) -> Dict[str, Any]:
        stats_path = snapshot_dir / "stats.json"
        if not stats_path.exists():
            return {}
        return json.loads(stats_path.read_text(encoding="utf-8"))

    def validate_snapshot_dir(self, snapshot_dir: Path) -> None:
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

    def load_active_manifest(self) -> Optional[KnowledgeSnapshotManifest]:
        snapshot_dir = self.get_active_snapshot_dir()
        if snapshot_dir is None:
            return None
        self.validate_snapshot_dir(snapshot_dir)
        return self.load_manifest(snapshot_dir)
