# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import time
from typing import List, Type
from pathlib import Path
from src.etl.extractors.base import BaseExtractor
from src.etl.cleaners.base import BaseCleaner
from src.etl.splitters.base import BaseSplitter
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class Pipeline:
    """
    文档处理流水线。
    根据配置动态组合抽取器、清洗器和分割器来处理文档。
    已注入 CSE 性能传感器。
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
    def from_file_path(cls, file_path: Path, splitter_structure_mode: str = "standard") -> "Pipeline":
        """
        根据文件路径创建并初始化 Pipeline 实例。
        """
        logger.info(f"正在从文件路径 '{file_path}' 创建 ETL Pipeline。")
        extractor_instance: BaseExtractor = MarkdownExtractor()
        cleaner_instance: BaseCleaner = BasicCleaner()
        splitter_instance: BaseSplitter = RecursiveTextSplitter(structure_mode=splitter_structure_mode)
        
        return cls(
            extractor=extractor_instance,
            cleaner=cleaner_instance,
            splitter=splitter_instance
        )

    def process(self, document: Document) -> List[Document]:
        """
        处理单个文档，并监控每一步的耗时 (CSE Sensor)。
        """
        source = document.metadata.get('source', '未知来源')
        logger.info(f"开始处理文档: {source}")
        
        start_total = time.perf_counter()
        
        # 抽取
        start_step = time.perf_counter()
        extracted_docs = self.extractor.extract(document)
        logger.info(f"抽取完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s")
        
        # 清洗
        start_step = time.perf_counter()
        cleaned_docs = self.cleaner.clean(extracted_docs)
        logger.info(f"清洗完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s")
        
        # 分割
        start_step = time.perf_counter()
        final_chunks = self.splitter.split(cleaned_docs)
        logger.info(f"分割完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s, 分片数: {len(final_chunks)}")
        
        total_duration = time.perf_counter() - start_total
        logger.info(f"文档总处理时长: {source}, 总计: {total_duration:.2f}s")
        
        return final_chunks
