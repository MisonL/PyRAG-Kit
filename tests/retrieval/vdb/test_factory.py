import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.vdb.base import VectorStoreBase
from src.retrieval.vdb.factory import VectorStoreFactory
from src.utils.config import Settings


# Mock FaissStore 类
class MockFaissStore(VectorStoreBase):
    def __init__(self, file_path: str | None):
        self.file_path = file_path
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[dict[str, Any]]):
        self.documents.extend(documents)

    def search(
        self, query: str, top_k: int = 5, search_type: str = "semantic"
    ) -> list[dict[str, Any]]:
        return [
            {"page_content": f"mock_doc_{i}", "metadata": {"source": "mock"}} for i in range(top_k)
        ]

    async def aadd_documents(self, documents: list[dict[str, Any]]):
        self.documents.extend(documents)

    async def asearch(
        self, query: str, top_k: int = 5, search_type: str = "semantic"
    ) -> list[dict[str, Any]]:
        return [
            {"page_content": f"mock_doc_{i}", "metadata": {"source": "mock"}} for i in range(top_k)
        ]

    def save(self, path: str):
        """模拟保存操作"""

    def load(self, path: str, *, trusted_paths=None):
        """模拟加载操作（签名与 FaissStore 一致，含来源校验参数）。"""
        self.file_path = path

    def load_snapshot(self, snapshot_dir: str, *, snapshot_root=None):
        """模拟从快照目录加载，记录路径供断言。"""
        if not snapshot_root:
            raise ValueError("缺少 snapshot_root，已拒绝加载快照。")
        self.file_path = snapshot_dir

    def get_embedding_model(self) -> Any:
        """模拟获取嵌入模型"""
        return MagicMock()  # 返回一个模拟的嵌入模型


@pytest.fixture(scope="function", autouse=True)
def patch_settings(monkeypatch, tmp_path):
    """模拟全局设置对象"""
    mock_settings_instance = MagicMock(spec=Settings)
    mock_settings_instance.default_vector_store = "faiss"
    mock_settings_instance.pkl_path = str(tmp_path / "faiss_store.pkl")
    mock_settings_instance.snapshot_root = str(tmp_path / "kb")
    mock_settings_instance.knowledge_base_path = str(tmp_path / "knowledge_base")
    mock_settings_instance.cache_path = str(tmp_path / "cache")
    mock_settings_instance.log_path = str(tmp_path / "logs")
    mock_settings_instance.default_llm_provider = "google"
    mock_settings_instance.default_embedding_provider = "local-hash"
    mock_settings_instance.default_rerank_provider = "siliconflow"
    mock_settings_instance.llm_configurations = {}
    mock_settings_instance.embedding_configurations = {
        "local-hash": MagicMock(model_name="local-hash-256"),
    }
    mock_settings_instance.rerank_configurations = {}
    mock_settings_instance.chat_temperature = 0.7
    mock_settings_instance.kb_embedding_batch_size = 32
    mock_settings_instance.kb_chunk_size = 1500
    mock_settings_instance.kb_chunk_overlap = 150
    mock_settings_instance.kb_child_chunk_size = 300
    mock_settings_instance.kb_child_chunk_overlap = 30

    with patch("src.utils.config.get_settings", return_value=mock_settings_instance):
        # 清除 VectorStoreFactory 及其依赖模块的缓存
        modules_to_clear = [
            "src.retrieval.vdb.factory",
            "src.retrieval.vdb.faiss_store",
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # 模拟 FaissStore 类
        monkeypatch.setattr("src.retrieval.vdb.faiss_store.FaissStore", MockFaissStore)

        # 重新导入 VectorStoreFactory，确保它加载的是最新的版本
        if "src.retrieval.vdb.factory" in sys.modules:
            del sys.modules["src.retrieval.vdb.factory"]
        from src.retrieval.vdb.factory import (
            VectorStoreFactory as ReloadedVectorStoreFactory,
        )

        # 直接模拟 VectorStoreFactory.get_vector_store 方法
        def mock_get_vector_store(
            store_type: str, file_path: str | None = None, *, trusted_paths=None
        ) -> VectorStoreBase:
            if store_type.lower() == "faiss":
                return MockFaissStore(file_path=file_path)
            else:
                raise ValueError(f"不支持的向量存储类型: {store_type}")

        monkeypatch.setattr(ReloadedVectorStoreFactory, "get_vector_store", mock_get_vector_store)

        global VectorStoreFactory
        VectorStoreFactory = ReloadedVectorStoreFactory
        yield


# 测试用例
def test_get_vector_store_faiss_success():
    """测试成功获取 FaissStore 实例"""
    store = VectorStoreFactory.get_vector_store("faiss", "/tmp/test_faiss.pkl")
    assert isinstance(store, MockFaissStore)
    assert store.file_path == "/tmp/test_faiss.pkl"


def test_get_vector_store_unsupported_type():
    """测试获取不支持的向量存储类型时抛出 ValueError"""
    with pytest.raises(ValueError, match="不支持的向量存储类型: unsupported"):
        VectorStoreFactory.get_vector_store("unsupported", "/tmp/test.pkl")


def test_get_default_vector_store_success():
    """测试成功获取默认向量存储实例"""
    store = VectorStoreFactory.get_default_vector_store()
    assert isinstance(store, MockFaissStore)
    assert store.file_path is None


def test_get_default_vector_store_without_loading_existing():
    """测试获取空白向量存储实例，不加载现有索引文件。"""
    store = VectorStoreFactory.get_default_vector_store(load_existing=False)
    assert isinstance(store, MockFaissStore)
    assert store.file_path is None


def _write_snapshot(root, snapshot_id, provider, model):
    """按真实快照目录结构写入一个活动快照，供兼容性检测使用。"""
    from pathlib import Path

    from src.runtime.contracts import KnowledgeSnapshotManifest

    snapshot_dir = Path(root) / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id=snapshot_id,
        store_type="faiss",
        embedding_provider=provider,
        embedding_model=model,
        chunk_mode="standard",
        source_digest="test-digest",
        document_count=1,
        chunk_count=1,
    )
    (snapshot_dir / "manifest.toml").write_text(manifest.to_toml(), encoding="utf-8")
    # 快照目录校验除了「文件齐全」还要求 chunks.pkl 与 embeddings.npy 的行数自洽
    # （见 SnapshotRepository._validate_snapshot_row_counts），因此这里写最小但
    # 可解析的内容：1 个分块 + 1 行向量。兼容性检测本身发生在 load_snapshot 之前，
    # 这些内容不参与断言。
    import pickle

    import numpy as np

    with (snapshot_dir / "chunks.pkl").open("wb") as file:
        pickle.dump([{"page_content": "占位分块", "metadata": {"source": "placeholder.md"}}], file)
    np.save(snapshot_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
    (snapshot_dir / "parents.pkl").write_bytes(b"")
    (snapshot_dir / "semantic.index").write_bytes(b"")
    (snapshot_dir / "stats.json").write_text(
        json.dumps({"chunk_count": 1}, ensure_ascii=False), encoding="utf-8"
    )
    (Path(root) / "ACTIVE_SNAPSHOT").write_text(snapshot_id, encoding="utf-8")
    return snapshot_dir


def test_default_vector_store_rejects_mismatched_embedding_provider(tmp_path):
    """活动快照的 embedding provider 与当前运行配置不同时必须显式失败。

    向量空间不兼容时静默加载会产出错误的检索结果，因此这里要求抛出
    带快照值与当前值的 RuntimeError，而不是继续使用旧索引。
    """
    from src.utils.config import get_settings

    settings = get_settings()
    _write_snapshot(settings.snapshot_root, "kb-mismatch", "openai", "text-embedding-3-large")

    with pytest.raises(RuntimeError, match="embedding 配置与当前运行配置不一致"):
        VectorStoreFactory.get_default_vector_store()


def test_default_vector_store_rejects_mismatched_embedding_model(tmp_path):
    """provider 相同但模型名不同同样不兼容，也必须拦截。"""
    from src.utils.config import get_settings

    settings = get_settings()
    _write_snapshot(settings.snapshot_root, "kb-model-mismatch", "local-hash", "local-hash-512")

    with pytest.raises(RuntimeError, match="embedding 配置与当前运行配置不一致"):
        VectorStoreFactory.get_default_vector_store()


def test_default_vector_store_accepts_matching_embedding(tmp_path):
    """配置一致时正常加载，不应触发兼容性错误。"""
    from src.utils.config import get_settings

    settings = get_settings()
    _write_snapshot(settings.snapshot_root, "kb-match", "local-hash", "local-hash-256")

    store = VectorStoreFactory.get_default_vector_store()
    assert isinstance(store, MockFaissStore)


# ── 回归：legacy 导入失败必须清理 .tmp-*，且不污染已有快照 ──


class _PartialWriterStore:
    """``save_snapshot`` 只落一个残缺文件，模拟构建中途失败。"""

    def __init__(self):
        self.documents = [{"page_content": "旧数据", "metadata": {"source": "old.md"}}]

    def load(self, path: str, *, trusted_paths=None):
        return

    def save_snapshot(self, snapshot_dir: str) -> None:
        from pathlib import Path

        (Path(snapshot_dir) / "chunks.pkl").write_bytes(b"partial")


def test_legacy_import_cleans_temp_dir_on_failure(tmp_path):
    """legacy 导入在 ``create_temp_snapshot_dir`` 之后失败时必须清掉 ``.tmp-*``。

    ``factory.py`` 的 legacy 分支原先没有 ``knowledge_build_service`` 那样的
    ``finalized``/``finally`` 结构，任何中途失败都会在快照根目录留下
    ``.tmp-legacy-*``，并随时间累积。
    """
    import pickle
    from pathlib import Path

    from src.retrieval.snapshot_repository import SnapshotRepository
    from src.retrieval.vdb.factory import VectorStoreFactory
    from src.runtime.contracts import build_run_config
    from src.utils.config import get_settings

    settings = get_settings()
    snapshot_root = Path(settings.snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    # 已有正式快照 + 活动标记，断言它们不被 legacy 导入的失败牵连
    previous = snapshot_root / "previous"
    previous.mkdir(exist_ok=True)
    (previous / "keep.txt").write_text("保留", encoding="utf-8")
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text("previous", encoding="utf-8")

    legacy_path = Path(settings.pkl_path)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    with legacy_path.open("wb") as file:
        pickle.dump({"documents": [], "embeddings": None}, file)

    run_config = build_run_config(settings)
    repository = SnapshotRepository(run_config)
    monkeypatch_generated = "legacy-fixed-id"

    with (
        patch.object(repository, "generate_snapshot_id", return_value=monkeypatch_generated),
        patch("src.retrieval.vdb.factory.SnapshotRepository", return_value=repository),
        patch(
            "src.retrieval.vdb.factory.VectorStoreFactory.get_vector_store",
            return_value=_PartialWriterStore(),
        ),
        pytest.raises(FileNotFoundError, match="知识快照不完整"),
    ):
        VectorStoreFactory._load_existing_state(object(), run_config)

    assert not (snapshot_root / f".tmp-{monkeypatch_generated}").exists(), (
        "失败的 legacy 导入留下了临时快照目录"
    )
    assert (snapshot_root / "ACTIVE_SNAPSHOT").read_text(encoding="utf-8") == "previous"
    assert (previous / "keep.txt").read_text(encoding="utf-8") == "保留"
