import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.snapshot_repository import SnapshotRepository
from src.services.knowledge_build_service import KnowledgeBuildService


class _RunConfig:
    def __init__(self, knowledge_base_path: Path, snapshot_root: Path):
        self.knowledge_base_path = knowledge_base_path
        self.snapshot_root = snapshot_root
        self.default_vector_store = "faiss"

    @property
    def active_snapshot_marker(self) -> Path:
        return self.snapshot_root / "ACTIVE_SNAPSHOT"


class _IncompleteVectorStore:
    def save_snapshot(self, snapshot_dir: str) -> None:
        (Path(snapshot_dir) / "chunks.pkl").write_bytes(b"partial")

    def upsert_embeddings(self, documents, embeddings) -> None:
        raise AssertionError("空文档不应写入 embedding")


class _EmptyPipeline:
    def process(self, document):
        return []


def test_build_validates_temp_snapshot_before_activation(tmp_path, monkeypatch):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "article.md").write_text("内容", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    previous_snapshot = snapshot_root / "previous"
    previous_snapshot.mkdir()
    (previous_snapshot / "keep.txt").write_text("保留", encoding="utf-8")
    (snapshot_root / "ACTIVE_SNAPSHOT").write_text("previous", encoding="utf-8")

    run_config = _RunConfig(knowledge_base, snapshot_root)
    repository = SnapshotRepository(run_config)
    monkeypatch.setattr(repository, "generate_snapshot_id", lambda: "kb-new")
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )

    embedding_service = SimpleNamespace(
        embedding_provider_key="local-hash",
        embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        embedding_service,
        repository,
    )

    with pytest.raises(FileNotFoundError, match="知识快照不完整"):
        asyncio.run(service.build("standard"))

    assert (snapshot_root / "ACTIVE_SNAPSHOT").read_text(encoding="utf-8") == "previous"
    assert (previous_snapshot / "keep.txt").read_text(encoding="utf-8") == "保留"
    assert not (snapshot_root / "kb-new").exists()
    assert not (snapshot_root / ".tmp-kb-new").exists()


def test_build_reports_which_file_is_not_utf8(tmp_path, monkeypatch):
    """非 UTF-8 文件的错误消息必须带上**是哪个文件**。

    ``UnicodeDecodeError`` 自带的消息只有字节偏移，而知识库目录通常有几十个
    文件；用户面对「程序运行期间发生错误」加一串偏移量无从下手。这里断言消息
    里出现了出问题的那个文件名，并且临时快照目录不残留。
    """
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "正常文档.md").write_text("# 正常\n\n正文", encoding="utf-8")
    (knowledge_base / "损坏的文档.md").write_bytes(b"\xff\xfe invalid \xc3\x28")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )
    embedding_service = SimpleNamespace(
        embedding_provider_key="local-hash",
        embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        embedding_service,
        SnapshotRepository(run_config),
    )

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(service.build("standard"))

    message = str(excinfo.value)
    assert "损坏的文档.md" in message, f"错误消息里没有出问题的文件名: {message}"
    assert "UTF-8" in message
    assert not list(snapshot_root.glob(".tmp-*")), "失败后临时快照目录必须被清理"


def test_build_still_accepts_utf8_knowledge_base(tmp_path, monkeypatch):
    """收紧不能误伤：全是 UTF-8 的知识库照常构建。"""
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "文档.md").write_text("# 标题\n\n正文", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )
    embedding_service = SimpleNamespace(
        embedding_provider_key="local-hash",
        embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        embedding_service,
        SnapshotRepository(run_config),
    )

    # 空分片会被 ``_IncompleteVectorStore`` 判为不完整快照，说明文件读取这一步过了。
    with pytest.raises(FileNotFoundError, match="知识快照不完整"):
        asyncio.run(service.build("standard"))


@pytest.mark.parametrize(
    "bad_path",
    [
        pytest.param(5, id="int"),
        pytest.param(["a"], id="list"),
        pytest.param(None, id="none"),
    ],
)
def test_build_rejects_non_path_knowledge_base_path(tmp_path, bad_path):
    """``knowledge_base_path`` 非 ``Path`` 时必须给 ``ValueError``。

    ``build`` 第一行就写 ``self.run_config.knowledge_base_path.glob("*.md")``，而
    ``RunConfig`` 是 frozen dataclass、字段没有校验器（同仓库多处已认领这条
    可达性）。非 ``Path`` 取值在那里抛裸
    ``AttributeError: 'int' object has no attribute 'glob'``——不是本服务其余拒绝
    路径承诺的带指引 ``ValueError``，而 ``build`` 由 CLI 与 ``main.py`` 的知识库
    向量化入口直接调用。
    """
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    run_config = _RunConfig(bad_path, snapshot_root)
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        SimpleNamespace(
            embedding_provider_key="local-hash",
            embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
        ),
        SnapshotRepository(run_config),
    )

    with pytest.raises(ValueError, match="knowledge_base_path"):
        asyncio.run(service.build("standard"))
