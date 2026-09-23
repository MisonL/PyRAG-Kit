from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.pipeline import Pipeline
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import get_settings  # 导入 get_settings 函数

# 用于测试的 Markdown 示例内容
SAMPLE_MARKDOWN = """
# Markdown 测试文档

## 简介

这是一个用于测试 ETL 流水线的 **Markdown** 文件。
它包含多种元素，例如：

- 列表项 1
- 列表项 2

以及一些需要被清洗的 `多余空格`  和

连续的换行符。


## 结论

测试应该能正确处理这些内容。
"""


@pytest.fixture(scope="module")
def mock_markdown_document():
    """提供一个模拟的 Markdown Document 对象用于测试"""
    mock_path = MagicMock(spec=Path)
    mock_path.name = "test.md"
    mock_path.suffix = ".md"
    return Document(content=SAMPLE_MARKDOWN, metadata={"source": str(mock_path)})


def test_full_etl_pipeline_for_markdown(mock_markdown_document):
    """
    测试完整的 ETL 流水线是否能正确处理 Markdown 文档。
    这个测试验证了从文件类型判断、处理器选择到最终切分的整个流程。
    """
    # 备份并临时修改全局配置以适应测试场景
    current_settings = get_settings()  # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators  # 备份分隔符

    # 临时设置较小的块大小，确保文档被分割
    current_settings.kb_chunk_size = 50
    current_settings.kb_chunk_overlap = 10
    current_settings.kb_splitter_separators = ["\n\n", "\n", " ", ""]  # 确保分隔符设置

    try:
        # 从模拟的文件路径初始化 Pipeline
        pipeline = Pipeline.from_file_path(Path(mock_markdown_document.metadata["source"]))
        # 强制设置 splitter 模式为 'char' 以支持旧测试逻辑 (50字符)
        pipeline.splitter = RecursiveTextSplitter(mode="char")

        # 执行处理流程
        processed_docs = pipeline.process(mock_markdown_document)

        # 断言结果
        assert isinstance(processed_docs, list), "处理结果应该是一个列表"
        assert len(processed_docs) > 1, "文档应该被切分成多个部分"

        # 验证清洗效果：不应再有多余的两个以上连续空格或三个以上连续换行符
        for doc in processed_docs:
            assert "  " not in doc.content, "不应存在连续的两个空格"  # 直接检查清洗后的内容
            assert "\n\n\n" not in doc.content, "不应存在连续的三个换行符"

        # 验证元数据是否被正确继承
        assert processed_docs[0].metadata["source"] == mock_markdown_document.metadata["source"]

        # 验证切分内容
        assert "Markdown 测试文档" in processed_docs[0].content
        assert "ETL 流水线" in processed_docs[1].content

    finally:
        # 恢复原始配置，避免影响其他测试
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators  # 恢复分隔符


def test_markdown_extractor(mock_markdown_document):
    """单独测试 MarkdownExtractor 的功能"""
    extractor = MarkdownExtractor()
    # extract 方法返回 List[Document]，所以需要取第一个元素
    extracted_docs = extractor.extract(mock_markdown_document)
    assert isinstance(extracted_docs, list)
    assert len(extracted_docs) == 1
    # Markdown 提取器应该保留原始内容
    assert extracted_docs[0].content == SAMPLE_MARKDOWN


def test_basic_cleaner():
    """单独测试 BasicCleaner 的文本清洗功能"""
    cleaner = BasicCleaner()
    dirty_text = "你好  世界 \n\n\n  再见.  "
    doc = Document(content=dirty_text, metadata={})
    # clean 方法期望 List[Document] 作为输入
    cleaned_docs = cleaner.clean([doc])
    assert isinstance(cleaned_docs, list)
    assert len(cleaned_docs) == 1
    # 验证多余空格、换行符和末尾空格是否被处理
    assert cleaned_docs[0].content == "你好 世界 \n\n 再见."


def test_recursive_text_splitter():
    """单独测试 RecursiveTextSplitter 的文本分割功能"""
    # 备份并临时修改全局配置
    current_settings = get_settings()  # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators

    # 直接在测试中设置适合分割的参数
    test_chunk_size = 20
    test_chunk_overlap = 5
    test_separators = ["\n\n", "\n", " "]  # 明确分隔符

    try:
        # 实例化 RecursiveTextSplitter 时，它会从 settings 读取配置
        current_settings.kb_chunk_size = test_chunk_size
        current_settings.kb_chunk_overlap = test_chunk_overlap
        current_settings.kb_splitter_separators = test_separators

        splitter = RecursiveTextSplitter(mode="char")  # 实例化时强制用 char 模式

        # 使用一个更长的文本来测试分割和重叠
        # 注意断言的是「每个块都不超过 chunk_size」而不是某个固定的块数：
        # 旧断言要求恰好 2 块，但第 1 块长 22 字符、已经超过 chunk_size=20——
        # 也就是说这条断言把「chunk_size 失效」当成了期望行为。分隔符表补上 ""
        # 兜底后长句会被真正切开，块数随之变化。
        long_text = "这是一个非常长的句子，需要被正确地切分开来。\n\n这是第二部分。"
        doc = Document(content=long_text, metadata={"source": "test.txt"})

        # split 方法期望 List[Document] 作为输入
        split_docs = splitter.split([doc])

        assert split_docs, "分割结果不应为空"
        for chunk in split_docs:
            assert len(chunk.content) <= test_chunk_size, (
                f"分片长度 {len(chunk.content)} 超过 chunk_size={test_chunk_size}: "
                f"{chunk.content!r}"
            )
        # 内容不得丢失：所有分片的并集必须覆盖原文的每一个非空白字符。
        # 不能用「拼接后等于原文」——chunk_overlap=5 的设计就是让相邻块共享
        # 5 个字符，拼接必然重复，那是正确行为而不是丢失。
        covered = "".join(chunk.content for chunk in split_docs)
        for char in long_text.replace("\n", "").replace(" ", ""):
            assert char in covered, f"字符 {char!r} 在所有分片中都找不到"
        assert split_docs[0].content.startswith("这是一个非常长的句子"), (
            f"首块应保留原文开头，实际为 {split_docs[0].content!r}"
        )
        assert split_docs[-1].content.endswith("这是第二部分。"), (
            f"末块应保留原文结尾，实际为 {split_docs[-1].content!r}"
        )
        # 验证元数据
        assert all(chunk.metadata["source"] == "test.txt" for chunk in split_docs)

    finally:
        # 恢复原始配置
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators
