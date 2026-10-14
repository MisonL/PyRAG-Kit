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


def test_build_does_not_crash_on_a_directory_named_like_markdown(tmp_path, monkeypatch):
    """``*.md`` 会匹配**目录**，``build`` 不得因此抛裸 ``OSError``。

    ``Path.glob("*.md")`` 只按名字匹配、不看类型，因此一个名为 ``notes.md`` 的
    **目录**（或 FIFO、符号链接）会被当成知识库文件。修复前它一路进
    ``_process_file`` 的 ``read_text``，抛 ``IsADirectoryError``（``OSError``
    子类）——不是 ``build`` 其余拒绝路径承诺的带指引消息，而它由 CLI 与
    ``main.py`` 的知识库向量化入口直接调用。修复后非文件条目被跳过，正常文档
    照常构建（这里走到「快照不完整」，说明读取这一步已过）。
    """
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "正常文档.md").write_text("# 正常\n\n正文", encoding="utf-8")
    (knowledge_base / "同名目录.md").mkdir()

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        SimpleNamespace(
            embedding_provider_key="local-hash",
            embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
        ),
        SnapshotRepository(run_config),
    )

    with pytest.raises(FileNotFoundError, match="知识快照不完整"):
        asyncio.run(service.build("standard"))


def test_build_reports_when_only_non_file_entries_match(tmp_path, monkeypatch):
    """知识库里**只有**非文件的 ``*.md`` 条目时，应给「未找到 Markdown 文件」。

    跳过非文件条目之后，若一个可读文件都不剩，仍是原来的 ``FileNotFoundError``
    ——不能因为「glob 有结果」就以为有文档可构建。
    """
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "只是目录.md").mkdir()

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        SimpleNamespace(
            embedding_provider_key="local-hash",
            embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
        ),
        SnapshotRepository(run_config),
    )

    with pytest.raises(FileNotFoundError, match="未找到 Markdown 文件"):
        asyncio.run(service.build("standard"))
    assert not list(snapshot_root.glob(".tmp-*")), "失败后临时快照目录必须被清理"


def test_build_reports_which_file_is_unreadable(tmp_path, monkeypatch):
    """读取失败（``OSError``）的消息必须带上**文件名**，与 UTF-8 那条同判据。

    ``read_text`` 还会因权限不足抛 ``PermissionError``、因条目是特殊文件抛
    ``IsADirectoryError``——都是 ``OSError``，消息里只有系统错误码，没有
    「知识库目录里的哪个文件」这一层定位。这里用权限位构造该失败（POSIX 下
    生效；root 运行时会跳过）。
    """
    import os

    if os.geteuid() == 0:
        pytest.skip("以 root 运行时权限位不生效")

    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "正常文档.md").write_text("# 正常\n\n正文", encoding="utf-8")
    unreadable = knowledge_base / "无权限文档.md"
    unreadable.write_text("# 秘密\n\n正文", encoding="utf-8")
    unreadable.chmod(0o000)

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )
    service = KnowledgeBuildService(
        run_config,
        _IncompleteVectorStore(),
        SimpleNamespace(
            embedding_provider_key="local-hash",
            embedding_model_detail=SimpleNamespace(model_name="local-hash-32"),
        ),
        SnapshotRepository(run_config),
    )

    try:
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(service.build("standard"))
        message = str(excinfo.value)
        assert "无权限文档.md" in message, f"错误消息里没有出问题的文件名: {message}"
        assert not list(snapshot_root.glob(".tmp-*")), "失败后临时快照目录必须被清理"
    finally:
        unreadable.chmod(0o644)


# ── 文档数与 embedding 数不一致、活动快照切换失败的覆盖缺口 ──
#
# 两条判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3725 条仍全绿）。删除后行为确实改变（实测）：
# * 92 文档数与 embedding 行数不一致 → ``np.vstack`` 后的错位数据被
#   **静默写入** store（或由 store 以形态无关的错误拒绝），而不是在
#   ``build`` 里以「知识快照中的文档数量与 embedding 数量不一致」报出；
# * 113 ``finalize_snapshot`` 后拿不到活动快照目录 → 报错缺失，
#   ``build`` 会继续走到 ``validate_snapshot_dir(None)`` 的裸异常。
#
# 既有用例里 ``_IncompleteVectorStore`` 只覆盖了「临时快照不自洽」这条
# **先于**两条判据的路径，两条判据本身从未被反向覆盖。
class _MismatchedEmbeddingService:
    """embed 返回的行数与输入文本数不一致，用于触发 92 行判据。"""

    embedding_provider_key = "local-hash"
    embedding_model_detail = SimpleNamespace(model_name="local-hash-32")

    async def embed_in_batches(self, texts):
        import numpy as np

        return np.zeros((len(texts) + 1, 4), dtype=np.float32)


class _CountingPipeline:
    """每个文件产出一个分块。"""

    class _Splitter:
        def __init__(self):
            self.parent_documents: dict = {}

    splitter = _Splitter()

    def process(self, document):
        from src.models.document import Document

        return [Document(content="chunk", metadata={"source": "a.md"})]


def test_build_rejects_document_embedding_count_mismatch(tmp_path, monkeypatch):
    """embedding 行数与文档数不一致时必须在写入 store 之前报错。"""
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "article.md").write_text("内容", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    repository = SnapshotRepository(run_config)
    monkeypatch.setattr(repository, "generate_snapshot_id", lambda: "kb-mismatch")

    class _Store:
        def save_snapshot(self, snapshot_dir):
            raise AssertionError("不一致的数据不应走到 save_snapshot")

    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _CountingPipeline(),
    )
    service = KnowledgeBuildService(run_config, _Store(), _MismatchedEmbeddingService(), repository)

    with pytest.raises(ValueError, match="文档数量与 embedding 数量不一致"):
        asyncio.run(service.build("standard"))

    assert not (snapshot_root / "kb-mismatch").exists()
    assert not (snapshot_root / ".tmp-kb-mismatch").exists()


def test_build_reports_missing_active_snapshot_after_finalize(tmp_path, monkeypatch):
    """``finalize_snapshot`` 之后拿不到活动快照目录时必须显式报错。

    删除守卫后 ``build`` 会继续用 ``None`` 调 ``validate_snapshot_dir``，
    崩在路径处理内部，消息与真实原因（快照切换失败）无关。
    """
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "article.md").write_text("内容", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    repository = SnapshotRepository(run_config)
    monkeypatch.setattr(repository, "generate_snapshot_id", lambda: "kb-lost")
    monkeypatch.setattr(repository, "get_active_snapshot_dir", lambda: None)

    class _Store:
        def upsert_embeddings(self, documents, embeddings):
            return None

        def save_snapshot(self, snapshot_dir):
            # 写出通过 ``validate_snapshot_dir`` 所需的最小文件集
            import json
            import pickle

            import faiss
            import numpy as np

            target = Path(snapshot_dir)
            with (target / "chunks.pkl").open("wb") as file:
                pickle.dump([{"page_content": "c", "metadata": {"source": "a.md"}}], file)
            with (target / "parents.pkl").open("wb") as file:
                pickle.dump({}, file)
            np.save(target / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
            index = faiss.IndexFlatL2(4)
            index.add(np.zeros((1, 4), dtype=np.float32))
            faiss.write_index(index, str(target / "semantic.index"))
            (target / "stats.json").write_text(json.dumps({"chunk_count": 1}), encoding="utf-8")
            with (target / "lexical.index").open("wb") as file:
                pickle.dump([{"t": 1}], file)

    class _Embedding:
        embedding_provider_key = "local-hash"
        embedding_model_detail = SimpleNamespace(model_name="local-hash-32")

        async def embed_in_batches(self, texts):
            import numpy as np

            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _CountingPipeline(),
    )
    service = KnowledgeBuildService(run_config, _Store(), _Embedding(), repository)

    with pytest.raises(RuntimeError, match="活动知识快照切换失败"):
        asyncio.run(service.build("standard"))


def test_build_reports_original_error_when_temp_cleanup_fails(tmp_path, monkeypatch, caplog):
    """临时快照清理失败时，抛出的仍是**原始**失败原因，且清理失败被记录。

    删除吞掉后 ``cleanup_temp_snapshot_dir`` 的异常顶替原始失败——用户
    看到的失败与实际根因不符。
    """
    import logging

    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "article.md").write_text("内容", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()

    run_config = _RunConfig(knowledge_base, snapshot_root)
    repository = SnapshotRepository(run_config)
    monkeypatch.setattr(repository, "generate_snapshot_id", lambda: "kb-cleanup-fail")
    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _EmptyPipeline(),
    )

    def failing_cleanup(_snapshot_id):
        raise RuntimeError("清理本身失败")

    monkeypatch.setattr(repository, "cleanup_temp_snapshot_dir", failing_cleanup)

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

    with (
        caplog.at_level(logging.ERROR, logger="src.services.knowledge_build_service"),
        pytest.raises(FileNotFoundError, match="知识快照不完整"),
    ):
        asyncio.run(service.build("standard"))
    assert "清理临时知识快照失败" in caplog.text


# ── 源码摘要的第二次读取同样要带文件名的指引 ──
#
# ``_compute_source_digest`` 是 ``*.md`` 清单的**第二处**读取点（``06f6a67``
# 只修了 ``_process_file`` 那半边），且它跑在全部 embedding 完成**之后**：
# 此时最贵的一步已经做完，失败收场却是一条没有定位、没有「重试什么」指引的
# 裸 ``OSError``。


def test_compute_source_digest_reports_filename_on_permission_error(tmp_path):
    """读取失败必须给出与 ``_process_file`` 同一契约的带文件名 ``ValueError``。"""
    import os

    if os.geteuid() == 0:
        pytest.skip("以 root 运行时权限位不生效")

    document = tmp_path / "文档.md"
    document.write_text("# 正常", encoding="utf-8")
    files = sorted(tmp_path.glob("*.md"))
    document.chmod(0o000)
    try:
        with pytest.raises(ValueError) as excinfo:
            KnowledgeBuildService._compute_source_digest(files)
        message = str(excinfo.value)
        assert "知识库文件无法读取" in message, message
        assert "文档.md" in message, f"消息里没有出问题的文件名: {message}"
    finally:
        document.chmod(0o644)


def test_compute_source_digest_reports_filename_when_file_disappears(tmp_path):
    """``glob`` 之后文件被删（TOCTOU）同样要给带文件名的 ``ValueError``。"""
    document = tmp_path / "文档.md"
    document.write_text("# 正常", encoding="utf-8")
    files = sorted(tmp_path.glob("*.md"))
    document.unlink()

    with pytest.raises(ValueError) as excinfo:
        KnowledgeBuildService._compute_source_digest(files)

    message = str(excinfo.value)
    assert "知识库文件无法读取" in message, message
    assert "文档.md" in message, f"消息里没有出问题的文件名: {message}"


def test_compute_source_digest_normal_path_is_hex_digest(tmp_path):
    """正常路径不受影响：仍是同一份内容的十六进制 SHA-256。"""
    (tmp_path / "a.md").write_text("内容A", encoding="utf-8")
    (tmp_path / "b.md").write_text("内容B", encoding="utf-8")

    digest = KnowledgeBuildService._compute_source_digest(sorted(tmp_path.glob("*.md")))

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_build_digest_failure_after_embedding_still_names_the_file(tmp_path, monkeypatch):
    """embedding 之后才变成不可读时，失败消息仍要带文件名。

    这是该缺陷的真实时序：``_process_file`` 读成功 → embedding 跑完 →
    文件被删/失去权限 → ``_compute_source_digest`` 读第二次。原实现抛裸
    ``PermissionError``（``build`` 的其余拒绝路径不承诺这种收场），而此时
    embedding 已经全部完成。
    """
    import os

    if os.geteuid() == 0:
        pytest.skip("以 root 运行时权限位不生效")

    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    victim = knowledge_base / "文档.md"
    victim.write_text("# 正常\n\n正文", encoding="utf-8")

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    run_config = _RunConfig(knowledge_base, snapshot_root)

    monkeypatch.setattr(
        "src.services.knowledge_build_service.Pipeline.from_file_path",
        lambda *args, **kwargs: _CountingPipeline(),
    )

    class _RevokingEmbedding:
        """模拟「embedding 期间文件被删」的时序。"""

        embedding_provider_key = "local-hash"
        embedding_model_detail = SimpleNamespace(model_name="local-hash-32")

        async def embed_in_batches(self, texts):
            victim.unlink()
            import numpy as np

            return np.zeros((len(texts), 4), dtype=np.float32)

    class _TolerantStore:
        """upsert 容忍写入；digest 失败必须发生在 save_snapshot 之前。"""

        def upsert_embeddings(self, documents, embeddings) -> None:
            pass

        def save_snapshot(self, snapshot_dir: str) -> None:
            raise AssertionError("digest 失败时不应走到 save_snapshot")

    service = KnowledgeBuildService(
        run_config,
        _TolerantStore(),
        _RevokingEmbedding(),
        SnapshotRepository(run_config),
    )

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(service.build("standard"))

    message = str(excinfo.value)
    assert "知识库文件无法读取" in message, message
    assert "文档.md" in message, f"消息里没有出问题的文件名: {message}"
    assert not list(snapshot_root.glob(".tmp-*")), "失败后临时快照目录必须被清理"
