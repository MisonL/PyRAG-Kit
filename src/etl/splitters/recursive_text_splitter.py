# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import re
import uuid
from collections.abc import Callable

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.document import Document
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

from .base import BaseSplitter

logger = get_module_logger(__name__)


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
        parent_chunk_size: int | None = None,
        parent_chunk_overlap: int | None = None,
        child_chunk_size: int | None = None,
        child_chunk_overlap: int | None = None,
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

    @staticmethod
    def _with_fallback_separator(separators: list[str]) -> list[str]:
        """确保分隔符表以 ``""`` 结尾，让 ``chunk_size`` 在退化输入下仍然生效。

        ``RecursiveCharacterTextSplitter._split_text`` 在递归到最后一个分隔符时
        走的是 ``if not new_separators: final_chunks.append(s)`` —— **原样追加**，
        不再按 ``chunk_size`` 切。因此分隔符表里只要没排到 ``""``（唯一能逐字符
        硬切的那一项），「分隔符耗尽」的文本就会整段留下。

        默认配置 ``kb_splitter_separators = ["###"]`` 恰好命中这一点：
        ``###`` 是唯一的匹配项，`new_separators` 直接为空，于是**任何不含
        ``###`` 的文档整篇变成一个块**。实测 ``chunk_size=1500`` 时，3600 token
        的纯列表文本产出单个 3599 token 的块；对真实知识库 39 个文件扫描，34 个
        文件产出超限块（最坏：子块上限 300，实测 4595）。

        兜底只在「前面所有分隔符都不匹配」时才被用到：实测含 ``###`` 的文档在
        ``["###"]`` 与 ``["###", ""]`` 下块数/最大块完全一致（3 / 252）。
        """
        if "" in separators:
            return list(separators)
        return [*separators, ""]

    def _init_splitter(self):
        """根据当前配置初始化内部分割器。"""
        current_settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size or current_settings.kb_chunk_size,
            chunk_overlap=self.parent_chunk_overlap
            if self.parent_chunk_overlap is not None
            else current_settings.kb_chunk_overlap,
            separators=self._with_fallback_separator(current_settings.kb_splitter_separators),
            length_function=self._get_length_function(),
            is_separator_regex=False,
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
            chunk_size=self.child_chunk_size
            if self.child_chunk_size is not None
            else current_settings.kb_child_chunk_size,
            chunk_overlap=(
                self.child_chunk_overlap
                if self.child_chunk_overlap is not None
                else current_settings.kb_child_chunk_overlap
            ),
            separators=self._with_fallback_separator(current_settings.kb_splitter_separators),
            length_function=self._get_length_function(),
            is_separator_regex=False,
        )

    @staticmethod
    def _strip_leading_punctuation(text: str) -> str:
        r"""去掉分片开头因切分残留的句号类标点，但不丢弃信息。

        旧实现是 ``re.sub(r"^[\s.。]+", "", text).strip()`` —— 直接**删除**
        开头连续的空白与句号。当分隔符本身就是句号（``kb_splitter_separators``
        配成 ``。``）时，LangChain 的 ``keep_separator=True`` 会把
        上一句的句号留在下一个分片开头，于是那个句号被永久丢弃：实测
        ``chunk_size=40, separators=["。"]`` 下原始 843 字符分成 15 块后只剩
        829 字符，丢失 1.66%。最坏情况整篇都是句号分隔符时，全部分片都可能被判为空
        并 ``continue`` 掉，产出 0 个块。

        现在只规范化边界空白：把开头的句号替换为等长空格再 strip，句中信息
        一字不改，也不会再产生「内容全被删掉因此该块被跳过」的路径。
        """
        text = re.sub(r"^[\s.。]+", lambda match: " " * len(match.group()), text)
        return text.strip()

    def _split_standard_documents(self, documents: list[Document]) -> list[Document]:
        """标准单层分片。"""
        all_chunks: list[Document] = []
        for doc in documents:
            logger.debug(f"正在分割文档: {doc.metadata.get('source', '未知来源')}")

            langchain_chunks = self.text_splitter.create_documents(
                [doc.content], metadatas=[doc.metadata]
            )
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

    def split_hierarchical(self, documents: list[Document]) -> list[Document]:
        """层级分片：先切父块，再切子块。"""
        all_chunks: list[Document] = []
        child_splitter = self._build_child_splitter()
        self.parent_documents = {}

        for doc in documents:
            logger.debug(f"正在层级分割文档: {doc.metadata.get('source', '未知来源')}")
            parent_documents = self.text_splitter.create_documents(
                [doc.content], metadatas=[doc.metadata]
            )
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
                child_documents = child_splitter.create_documents(
                    [parent_content], metadatas=[parent_doc.metadata]
                )

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

    def split(self, documents: list[Document], **kwargs) -> list[Document]:
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
