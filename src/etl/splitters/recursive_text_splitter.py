# -*- coding: utf-8 -*-
import re
import uuid
from typing import Callable, List, Optional

import tiktoken

from .base import BaseSplitter
from src.models.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHILD_CHUNK_OVERLAP = 30

class RecursiveTextSplitter(BaseSplitter):
    """
    递归文本分割器。
    使用 LangChain 的 RecursiveCharacterTextSplitter。
    支持基于字符长度或基于 Token 数量的分割。
    """

    def __init__(
        self,
        mode: str = "token",
        encoding_name: str = "cl100k_base",
        structure_mode: str = "standard",
        parent_chunk_size: Optional[int] = None,
        parent_chunk_overlap: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        child_chunk_overlap: Optional[int] = None,
    ):
        """
        初始化 RecursiveTextSplitter。
        
        Args:
            mode (str): 分割模式，可选 "char" 或 "token"。
            encoding_name (str): tiktoken 编码名称。
        """
        if structure_mode not in {"standard", "hierarchical"}:
            raise ValueError(f"不支持的结构模式: {structure_mode}")
        self.mode = mode
        self.structure_mode = structure_mode
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_documents: dict[str, dict[str, object]] = {}
        self._encoder = tiktoken.get_encoding(encoding_name)
        logger.info(
            "初始化 RecursiveTextSplitter，模式: %s, 结构模式: %s, 编码: %s",
            mode,
            structure_mode,
            encoding_name,
        )
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
            chunk_size=self.parent_chunk_size or current_settings.kb_chunk_size,
            chunk_overlap=self.parent_chunk_overlap if self.parent_chunk_overlap is not None else current_settings.kb_chunk_overlap,
            separators=current_settings.kb_splitter_separators,
            length_function=self._get_length_function(),
            is_separator_regex=False
        )
        logger.info(
            "文本分割器已就绪: chunk_size=%s, mode=%s, structure=%s",
            self.parent_chunk_size or current_settings.kb_chunk_size,
            self.mode,
            self.structure_mode,
        )

    def _build_child_splitter(self) -> RecursiveCharacterTextSplitter:
        """构建子分片器。"""
        current_settings = get_settings()
        return RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size if self.child_chunk_size is not None else current_settings.kb_child_chunk_size,
            chunk_overlap=(
                self.child_chunk_overlap
                if self.child_chunk_overlap is not None
                else current_settings.kb_child_chunk_overlap
            ),
            separators=current_settings.kb_splitter_separators,
            length_function=self._get_length_function(),
            is_separator_regex=False,
        )

    @staticmethod
    def _strip_leading_punctuation(text: str) -> str:
        """去掉分片开头的句号类标点。"""
        return re.sub(r"^[\s.。]+", "", text).strip()

    def _split_standard_documents(self, documents: List[Document]) -> List[Document]:
        """标准单层分片。"""
        all_chunks: List[Document] = []
        for doc in documents:
            logger.debug(f"正在分割文档: {doc.metadata.get('source', '未知来源')}")

            langchain_chunks = self.text_splitter.create_documents([doc.content], metadatas=[doc.metadata])
            for i, chunk in enumerate(langchain_chunks):
                chunk_content = self._strip_leading_punctuation(chunk.page_content)
                if not chunk_content:
                    continue

                chunk_metadata = chunk.metadata.copy()
                chunk_id = uuid.uuid4().hex
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["doc_id"] = chunk_id
                chunk_metadata["chunk_index"] = i
                chunk_metadata["token_count"] = len(self._encoder.encode(chunk_content))
                all_chunks.append(Document(content=chunk_content, metadata=chunk_metadata))

            logger.debug(
                "文档 '%s' 分割完成，生成 %s 个块。",
                doc.metadata.get("source", "未知来源"),
                len(langchain_chunks),
            )
        return all_chunks

    def split_hierarchical(self, documents: List[Document]) -> List[Document]:
        """层级分片：先切父块，再切子块。"""
        all_chunks: List[Document] = []
        child_splitter = self._build_child_splitter()
        self.parent_documents = {}

        for doc in documents:
            logger.debug(f"正在层级分割文档: {doc.metadata.get('source', '未知来源')}")
            parent_documents = self.text_splitter.create_documents([doc.content], metadatas=[doc.metadata])
            doc_chunk_count = 0

            for parent_index, parent_doc in enumerate(parent_documents):
                parent_content = self._strip_leading_punctuation(parent_doc.page_content)
                if not parent_content:
                    continue

                parent_id = uuid.uuid4().hex
                self.parent_documents[parent_id] = {
                    "content": parent_content,
                    "metadata": {
                        **parent_doc.metadata.copy(),
                        "parent_chunk_index": parent_index,
                    },
                }
                child_documents = child_splitter.create_documents([parent_content], metadatas=[parent_doc.metadata])

                for child_index, child_doc in enumerate(child_documents):
                    child_content = self._strip_leading_punctuation(child_doc.page_content)
                    if not child_content:
                        continue

                    child_metadata = child_doc.metadata.copy()
                    chunk_id = uuid.uuid4().hex
                    child_metadata["chunk_id"] = chunk_id
                    child_metadata["doc_id"] = chunk_id
                    child_metadata["chunk_index"] = child_index
                    child_metadata["parent_id"] = parent_id
                    child_metadata["parent_chunk_index"] = parent_index
                    child_metadata["token_count"] = len(self._encoder.encode(child_content))
                    all_chunks.append(Document(content=child_content, metadata=child_metadata))
                    doc_chunk_count += 1

            logger.debug(
                "文档 '%s' 层级分割完成，生成 %s 个块。",
                doc.metadata.get("source", "未知来源"),
                doc_chunk_count,
            )

        return all_chunks

    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        将文档列表中的文本内容分割成更小的块。
        """
        logger.info(f"开始分割 {len(documents)} 个文档。")

        # 实时同步最新配置
        self._init_splitter()
        self.parent_documents = {}

        if self.structure_mode == "hierarchical":
            all_chunks = self.split_hierarchical(documents)
        else:
            all_chunks = self._split_standard_documents(documents)

        logger.info(
            "分割任务完成，总计生成 %s 个文本块 (模式: %s, 结构: %s)。",
            len(all_chunks),
            self.mode,
            self.structure_mode,
        )
        return all_chunks
