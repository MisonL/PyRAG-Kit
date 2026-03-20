from src.retrieval.vdb.faiss_store import FaissStore


def test_build_index_text_includes_source_stem():
    document = {
        "page_content": "正文内容",
        "metadata": {"source": "/tmp/用户-共享文件夹使用方法_优化后.md"},
    }

    index_text = FaissStore._build_index_text(document)

    assert "用户 共享文件夹使用方法 优化后" in index_text
    assert index_text.endswith("正文内容")


def test_build_index_text_keeps_original_content_when_source_missing():
    document = {
        "page_content": "只有正文",
        "metadata": {},
    }

    index_text = FaissStore._build_index_text(document)

    assert index_text == "只有正文"
