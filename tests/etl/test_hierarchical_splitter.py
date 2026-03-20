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
