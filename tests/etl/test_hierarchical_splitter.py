from types import SimpleNamespace

from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document


def test_hierarchical_splitter_builds_parent_child_metadata(monkeypatch):
    settings = SimpleNamespace(
        kb_chunk_size=120,
        kb_chunk_overlap=0,
        kb_splitter_separators=["\n\n", "\n", " "],
    )
    monkeypatch.setattr(
        "src.etl.splitters.recursive_text_splitter.get_settings",
        lambda: settings,
    )

    splitter = RecursiveTextSplitter(
        mode="char",
        structure_mode="hierarchical",
        parent_chunk_size=120,
        parent_chunk_overlap=0,
        child_chunk_size=40,
        child_chunk_overlap=0,
    )
    document = Document(
        content="Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 8,
        metadata={"source": "knowledge_base/sample.md"},
    )

    chunks = splitter.split([document])

    assert len(chunks) > 2
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert len({chunk.metadata["parent_id"] for chunk in chunks}) < len(chunks)
    assert all(chunk.metadata["source"] == "knowledge_base/sample.md" for chunk in chunks)
    assert all("parent_content" not in chunk.metadata for chunk in chunks)
    assert all("parent_chunk_index" in chunk.metadata for chunk in chunks)
    assert splitter.parent_documents
    parent_content = next(iter(splitter.parent_documents.values()))["content"]
    assert isinstance(parent_content, str) and parent_content
    assert any(chunk.content != parent_content for chunk in chunks)


# ── 回归：chunk_size 在「分隔符耗尽」的退化输入下也必须生效 ──


def _splitter_with(mode="token", structure_mode="standard", **overrides):
    """构造分片器并覆盖指定设置字段，返回 (splitter, settings, 原值快照)。"""
    from src.utils.config import get_settings

    settings = get_settings()
    original = {key: getattr(settings, key) for key in overrides}
    for key, value in overrides.items():
        setattr(settings, key, value)
    splitter = RecursiveTextSplitter(mode=mode, structure_mode=structure_mode)
    return splitter, settings, original


def _restore(settings, original):
    for key, value in original.items():
        setattr(settings, key, value)


def test_splitter_enforces_chunk_size_without_matching_separator(monkeypatch):
    """默认分隔符 ``["###"]`` 下，不含 ``###`` 的文档不得整篇变成一个块。

    ``RecursiveCharacterTextSplitter._split_text`` 在 ``not new_separators`` 时走
    ``final_chunks.append(s)``——**原样追加**，不再按 ``chunk_size`` 切。默认配置
    ``kb_splitter_separators = ["###"]`` 恰好命中：``###`` 是唯一匹配项，
    ``new_separators`` 直接为空。实测修复前 ``chunk_size=1500`` 时 3600 token 的
    纯列表文本产出单个 3599 token 块。
    """
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    text = "- 项目说明文字\n" * 600
    assert len(encoder.encode(text)) > 1500, "前提：输入必须远超 chunk_size"

    splitter, settings, original = _splitter_with(
        kb_chunk_size=1500, kb_chunk_overlap=0, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "list.md"})])
    finally:
        _restore(settings, original)

    assert len(chunks) > 1, "分隔符耗尽时仍应被硬切分，而不是整篇一个块"
    oversize = [
        len(encoder.encode(chunk.content))
        for chunk in chunks
        if len(encoder.encode(chunk.content)) > 1500
    ]
    assert not oversize, f"以下分片超过 chunk_size=1500: {oversize}"


def test_splitter_output_is_unchanged_when_separator_matches(monkeypatch):
    """兜底分隔符只在「前面都不匹配」时生效，不得改变正常文档的切分结果。

    含 ``###`` 的文档在 ``["###"]`` 与 ``["###", ""]`` 下块数/最大块应完全一致
    （实测 3 / 252）。
    """
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    text = "\n\n".join(f"### 小节{index}\n" + "内容。" * 60 for index in range(6))

    splitter, settings, original = _splitter_with(
        kb_chunk_size=300, kb_chunk_overlap=30, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "doc.md"})])
    finally:
        _restore(settings, original)

    assert len(chunks) == 3
    assert max(len(encoder.encode(chunk.content)) for chunk in chunks) == 252


def test_strip_leading_punctuation_does_not_delete_content():
    """开头标点只能被规范化，不能被删除。

    旧实现 ``re.sub(r"^[\\s.。]+", "", text).strip()`` 直接删掉开头连续的句号；
    当分隔符本身就是句号时，LangChain 的 ``keep_separator=True`` 会把上一句的句号
    留在下一个分片开头，于是那个句号被永久丢弃。
    """
    splitter = RecursiveTextSplitter(mode="char")

    assert splitter._strip_leading_punctuation("。第7句内容") == "第7句内容"
    assert splitter._strip_leading_punctuation("。。开头两句") == "开头两句"
    assert splitter._strip_leading_punctuation("   \n 正常文本") == "正常文本"
    assert splitter._strip_leading_punctuation("开头就是正文") == "开头就是正文"


def test_splitter_keeps_all_content_when_punctuation_is_the_separator(monkeypatch):
    """分隔符是句号时不得丢掉任何正文，也不得产出 0 个块。

    实测修复前 ``chunk_size=40, separators=["。"]``：原始 843 字符分成 15 块后只剩
    829 字符，丢失 1.66%；整篇以句号开头的输入更会被判空并 ``continue``，产出 0 块。
    """
    body = "。".join(f"第{index}句内容" for index in range(1, 61))

    splitter, settings, original = _splitter_with(
        kb_chunk_size=40, kb_chunk_overlap=0, kb_splitter_separators=["。"]
    )
    try:
        chunks = splitter.split([Document(content=body, metadata={"source": "body.md"})])
    finally:
        _restore(settings, original)

    assert chunks, "不得产出 0 个块"
    covered = "".join(chunk.content for chunk in chunks)
    for char in body:
        if char == "。":
            continue
        assert char in covered, f"字符 {char!r} 在所有分片中都找不到"


def test_splitter_keeps_punctuation_only_document(monkeypatch):
    """整篇以句号开头/结尾时仍必须产出分片（旧实现全部判空跳过）。"""
    text = "。" * 50 + "正文" + "。" * 50

    splitter, settings, original = _splitter_with(
        kb_chunk_size=40, kb_chunk_overlap=0, kb_splitter_separators=["。"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "dots.md"})])
    finally:
        _restore(settings, original)

    assert chunks
    assert any("正文" in chunk.content for chunk in chunks)
