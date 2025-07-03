# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from typing import List, Type
from pathlib import Path # 导入 Path
from src.etl.extractors.base import BaseExtractor
from src.etl.cleaners.base import BaseCleaner
from src.etl.splitters.base import BaseSplitter
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

class Pipeline:
    """
    文档处理流水线。
    根据配置动态组合抽取器、清洗器和分割器来处理文档。
    """

    def __init__(self,
                 extractor: BaseExtractor,
                 cleaner: BaseCleaner,
                 splitter: BaseSplitter):
        """
        初始化 Pipeline。
        Args:
            extractor (BaseExtractor): 文档抽取器实例。
            cleaner (BaseCleaner): 文档清洗器实例。
            splitter (BaseSplitter): 文档分割器实例。
        """
        self.extractor = extractor
        self.cleaner = cleaner
        self.splitter = splitter
        logger.info("ETL Pipeline 初始化完成。")

    @classmethod
    def from_file_path(cls, file_path: Path) -> "Pipeline":
        """
        根据文件路径创建并初始化 Pipeline 实例。
        此方法用于根据文件类型选择合适的处理器。
        """
        logger.info(f"正在从文件路径 '{file_path}' 创建 ETL Pipeline。")
        # 根据文件类型选择抽取器 (目前只有 MarkdownExtractor)
        # 未来这里可以有更复杂的逻辑，例如根据 file_path.suffix 选择不同的 Extractor
        extractor_instance: BaseExtractor = MarkdownExtractor()
        logger.debug(f"选择抽取器: {extractor_instance.__class__.__name__}")
        
        # 清洗器和分割器是固定的
        cleaner_instance: BaseCleaner = BasicCleaner()
        logger.debug(f"选择清洗器: {cleaner_instance.__class__.__name__}")
        # RecursiveTextSplitter 现在从 settings 读取配置，无需传递参数
        splitter_instance: BaseSplitter = RecursiveTextSplitter()
        logger.debug(f"选择分割器: {splitter_instance.__class__.__name__}")
        
        # 实例化并返回 Pipeline
        pipeline_instance = cls(
            extractor=extractor_instance,
            cleaner=cleaner_instance,
            splitter=splitter_instance
        )
        logger.info(f"ETL Pipeline 从文件路径 '{file_path}' 创建成功。")
        return pipeline_instance

    def process(self, document: Document) -> List[Document]: # 方法名和签名修改
        """
        处理单个文档，执行抽取、清洗和分割操作。

        Args:
            document (Document): 待处理的文档对象。

        Returns:
            List[Document]: 处理后的文本块列表。
        """
        logger.info(f"开始处理文档: {document.metadata.get('source', '未知来源')}")
        # 抽取 (Extractor 现在接受 Document 对象)
        extracted_docs = self.extractor.extract(document)
        logger.debug(f"抽取完成，得到 {len(extracted_docs)} 个文档。")
        
        # 清洗
        cleaned_docs = self.cleaner.clean(extracted_docs)
        logger.debug(f"清洗完成，得到 {len(cleaned_docs)} 个文档。")
        
        # 分割
        final_chunks = self.splitter.split(cleaned_docs)
        logger.info(f"分割完成，得到 {len(final_chunks)} 个文本块。")
        
        return final_chunks
