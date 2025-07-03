from typing import List, Dict, Any, Optional
from .base import BaseSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveTextSplitter(BaseSplitter):
    """
    递归文本分割器。
    使用 LangChain 的 RecursiveCharacterTextSplitter 将文本分割成块。
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150, separators: Optional[List[str]] = None):
        """
        初始化 RecursiveTextSplitter。

        Args:
            chunk_size (int): 每个文本块的最大字符数。
            chunk_overlap (int): 文本块之间的重叠字符数。
            separators (Optional[List[str]]): 用于分割文本的字符串列表，按优先级从高到低排列。
                                    如果为 None，则使用默认分隔符。
        """
        if separators is None:
            # 默认分隔符，与 LangChain 默认行为一致
            self.separators = ["\n\n", "\n", " ", ""]
        else:
            self.separators = separators

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len, # 使用 len() 作为长度函数
            is_separator_regex=False # 默认不使用正则表达式
        )

    def split(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        将文档列表中的文本内容分割成更小的块。

        Args:
            documents (List[Dict[str, Any]]): 待分割的文档列表，每个文档字典至少包含 'content' 键。
            **kwargs: 额外的参数，可以覆盖初始化时的 chunk_size, chunk_overlap, separators。

        Returns:
            List[Dict[str, Any]]: 分割后的文本块列表。
        """
        all_chunks = []
        
        # 允许在运行时覆盖初始化参数
        current_chunk_size = kwargs.get("chunk_size", self.text_splitter._chunk_size)
        current_chunk_overlap = kwargs.get("chunk_overlap", self.text_splitter._chunk_overlap)
        current_separators = kwargs.get("separators", self.text_splitter._separators)

        # 如果运行时参数与初始化参数不同，则重新创建 text_splitter
        if (current_chunk_size != self.text_splitter._chunk_size or
            current_chunk_overlap != self.text_splitter._chunk_overlap or
            current_separators != self.text_splitter._separators):
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=current_chunk_size,
                chunk_overlap=current_chunk_overlap,
                separators=current_separators,
                length_function=len,
                is_separator_regex=False
            )

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # 使用 LangChain 的分割器进行分割
            chunks = self.text_splitter.create_documents([content], metadatas=[metadata])
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_index"] = i # 添加块索引
                all_chunks.append({
                    "content": chunk.page_content,
                    "metadata": chunk_metadata
                })
        return all_chunks
