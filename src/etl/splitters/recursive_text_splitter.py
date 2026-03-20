# -*- coding: utf-8 -*-
from typing import List, Optional, Callable
import tiktoken

from .base import BaseSplitter
from src.models.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class RecursiveTextSplitter(BaseSplitter):
    """
    递归文本分割器。
    使用 LangChain 的 RecursiveCharacterTextSplitter。
    支持基于字符长度或基于 Token 数量的分割。
    """

    def __init__(self, mode: str = "token", encoding_name: str = "cl100k_base"):
        """
        初始化 RecursiveTextSplitter。
        
        Args:
            mode (str): 分割模式，可选 "char" 或 "token"。
            encoding_name (str): tiktoken 编码名称。
        """
        self.mode = mode
        self._encoder = tiktoken.get_encoding(encoding_name)
        logger.info(f"初始化 RecursiveTextSplitter，模式: {mode}, 编码: {encoding_name}")
        self._init_splitter()

    def _get_length_function(self) -> Callable[[str], int]:
        """根据模式返回长度计算函数。"""
        if self.mode == "token":
            return lambda x: len(self._encoder.encode(x))
        return len

    def _init_splitter(self):
        """根据当前配置初始化内部分割器。"""
        current_settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=current_settings.kb_chunk_size,
            chunk_overlap=current_settings.kb_chunk_overlap,
            separators=current_settings.kb_splitter_separators,
            length_function=self._get_length_function(),
            is_separator_regex=False
        )
        logger.info(f"文本分割器已就绪: chunk_size={current_settings.kb_chunk_size}, mode={self.mode}")

    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        将文档列表中的文本内容分割成更小的块。
        """
        logger.info(f"开始分割 {len(documents)} 个文档。")
        
        # 实时同步最新配置
        self._init_splitter()
        
        all_chunks: List[Document] = []
        for doc in documents:
            logger.debug(f"正在分割文档: {doc.metadata.get('source', '未知来源')}")
            
            # 使用 LangChain 分割器
            langchain_chunks = self.text_splitter.create_documents(
                [doc.content],
                metadatas=[doc.metadata]
            )
            
            for i, chunk in enumerate(langchain_chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_index"] = i
                # 计算并记录本块的 token 数量
                chunk_metadata["token_count"] = len(self._encoder.encode(chunk.page_content))
                
                all_chunks.append(Document(
                    content=chunk.page_content,
                    metadata=chunk_metadata
                ))
            
            logger.debug(f"文档 '{doc.metadata.get('source', '未知来源')}' 分割完成，生成 {len(langchain_chunks)} 个块。")
        
        logger.info(f"分割任务完成，总计生成 {len(all_chunks)} 个文本块 (模式: {self.mode})。")
        return all_chunks
