import os
from typing import List, Dict, Any, Type
from src.etl.extractors.base import BaseExtractor
from src.etl.cleaners.base import BaseCleaner
from src.etl.splitters.base import BaseSplitter
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.utils.config import settings # 导入全局配置

class PipelineManager:
    """
    文档处理流水线管理器。
    根据配置动态组合抽取器、清洗器和分割器来处理文档。
    """

    def __init__(self):
        # 根据配置初始化处理器
        # 这里可以根据 settings.etl_config 等配置来动态选择和实例化
        # 暂时硬编码使用已实现的处理器
        self.extractor: BaseExtractor = MarkdownExtractor()
        self.cleaner: BaseCleaner = BasicCleaner()
        
        # 从全局配置中获取分割器参数
        chunk_size = settings.kb_chunk_size
        chunk_overlap = settings.kb_chunk_overlap
        separators = settings.kb_splitter_separators
        
        self.splitter: BaseSplitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def process_documents(self, document_paths: List[str]) -> List[Dict[str, Any]]:
        """
        处理给定路径的文档列表，执行抽取、清洗和分割操作。

        Args:
            document_paths (List[str]): 待处理的文档文件路径列表。

        Returns:
            List[Dict[str, Any]]: 处理后的文本块列表。
        """
        all_extracted_docs = []
        for path in document_paths:
            # 抽取
            extracted_docs = self.extractor.extract(path)
            all_extracted_docs.extend(extracted_docs)
        
        # 清洗
        cleaned_docs = self.cleaner.clean(all_extracted_docs)
        
        # 分割
        final_chunks = self.splitter.split(cleaned_docs)
        
        return final_chunks
