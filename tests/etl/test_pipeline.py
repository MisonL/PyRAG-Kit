import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.etl.pipeline import Pipeline
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import get_settings # 导入 get_settings 函数

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
    return Document(content=SAMPLE_MARKDOWN, metadata={'source': str(mock_path)})

def test_full_etl_pipeline_for_markdown(mock_markdown_document):
    """
    测试完整的 ETL 流水线是否能正确处理 Markdown 文档。
    这个测试验证了从文件类型判断、处理器选择到最终切分的整个流程。
    """
    # 备份并临时修改全局配置以适应测试场景
    current_settings = get_settings() # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators # 备份分隔符
    
    # 临时设置较小的块大小，确保文档被分割
    current_settings.kb_chunk_size = 50
    current_settings.kb_chunk_overlap = 10
    current_settings.kb_splitter_separators = ["\n\n", "\n", " ", ""] # 确保分隔符设置

    try:
        # 从模拟的文件路径初始化 Pipeline
        pipeline = Pipeline.from_file_path(Path(mock_markdown_document.metadata['source']))

        # 执行处理流程
        processed_docs = pipeline.process(mock_markdown_document)

        # 断言结果
        assert isinstance(processed_docs, list), "处理结果应该是一个列表"
        assert len(processed_docs) > 1, "文档应该被切分成多个部分"
        
        # 验证清洗效果：不应再有多余的两个以上连续空格或三个以上连续换行符
        for doc in processed_docs:
            assert "  " not in doc.content, "不应存在连续的两个空格" # 直接检查清洗后的内容
            assert "\n\n\n" not in doc.content, "不应存在连续的三个换行符"

        # 验证元数据是否被正确继承
        assert processed_docs[0].metadata['source'] == mock_markdown_document.metadata['source']

        # 验证切分内容
        assert "Markdown 测试文档" in processed_docs[0].content
        assert "ETL 流水线" in processed_docs[1].content

    finally:
        # 恢复原始配置，避免影响其他测试
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators # 恢复分隔符

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
    current_settings = get_settings() # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators
    
    # 直接在测试中设置适合分割的参数
    test_chunk_size = 20
    test_chunk_overlap = 5
    test_separators = ["\n\n", "\n", " "] # 明确分隔符

    try:
        # 实例化 RecursiveTextSplitter 时，它会从 settings 读取配置
        current_settings.kb_chunk_size = test_chunk_size
        current_settings.kb_chunk_overlap = test_chunk_overlap
        current_settings.kb_splitter_separators = test_separators

        splitter = RecursiveTextSplitter() # 实例化时会读取 settings
        
        # 使用一个更长的文本来测试分割和重叠
        # 调整文本内容，使其在 chunk_size=20, chunk_overlap=5 的情况下能被分割成两部分
        long_text = "这是一个非常长的句子，需要被正确地切分开来。\n\n这是第二部分。"
        doc = Document(content=long_text, metadata={'source': 'test.txt'})

        # split 方法期望 List[Document] 作为输入
        split_docs = splitter.split([doc])

        assert len(split_docs) == 2, "文本应该被分割成两部分"
        # 验证第一部分的内容
        assert split_docs[0].content == "这是一个非常长的句子，需要被正确地切分开来。"
        # 验证第二部分的内容
        assert split_docs[1].content == "这是第二部分。"
        # 验证元数据
        assert split_docs[0].metadata['source'] == 'test.txt'

    finally:
        # 恢复原始配置
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators
