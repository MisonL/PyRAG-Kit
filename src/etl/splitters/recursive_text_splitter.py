# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from typing import List, Optional
from .base import BaseSplitter
from src.models.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

class RecursiveTextSplitter(BaseSplitter):
    """
    递归文本分割器。
    使用 LangChain 的 RecursiveCharacterTextSplitter 将文本分割成块。
    """

    def __init__(self):
        """
        初始化 RecursiveTextSplitter。
        参数从全局配置 get_settings() 中读取。
        """
        logger.info("初始化 RecursiveTextSplitter。")
        current_settings = get_settings() # 获取当前配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=current_settings.kb_chunk_size,
            chunk_overlap=current_settings.kb_chunk_overlap,
            separators=current_settings.kb_splitter_separators,
            length_function=len,
            is_separator_regex=False
        )
        logger.info(f"文本分割器配置: chunk_size={current_settings.kb_chunk_size}, chunk_overlap={current_settings.kb_chunk_overlap}, separators={current_settings.kb_splitter_separators}")

    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        将文档列表中的文本内容分割成更小的块。

        Args:
            documents (List[Document]): 待分割的文档对象列表。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Document]: 分割后的文档块列表。
        """
        logger.info(f"开始分割 {len(documents)} 个文档。")
        all_chunks: List[Document] = []
        
        # 获取最新的配置并重新初始化 text_splitter
        current_settings = get_settings() # 获取当前配置
        current_chunk_size = current_settings.kb_chunk_size
        current_chunk_overlap = current_settings.kb_chunk_overlap
        current_separators = current_settings.kb_splitter_separators

        # 每次都重新初始化分割器，以确保使用最新的配置
        logger.info("正在根据最新配置重新初始化文本分割器。")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=current_chunk_size,
            chunk_overlap=current_chunk_overlap,
            separators=current_separators,
            length_function=len,
            is_separator_regex=False
        )
        logger.info(f"文本分割器重新配置: chunk_size={current_chunk_size}, chunk_overlap={current_chunk_overlap}, separators={current_separators}")

        for doc in documents:
            logger.debug(f"正在分割文档: {doc.metadata.get('source', '未知来源')}")
            # 使用 LangChain 的分割器进行分割
            # langchain 的 create_documents 返回它自己的 Document 类型
            langchain_chunks = self.text_splitter.create_documents(
                [doc.content],
                metadatas=[doc.metadata]
            )
            
            for i, chunk in enumerate(langchain_chunks):
                # 将 langchain 的 Document 转换回我们自己的 Document 模型
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_index"] = i # 添加块索引
                
                all_chunks.append(Document(
                    content=chunk.page_content,
                    metadata=chunk_metadata
                ))
            logger.debug(f"文档 '{doc.metadata.get('source', '未知来源')}' 分割完成，生成 {len(langchain_chunks)} 个块。")
        
        logger.info(f"所有文档分割完成，总共生成 {len(all_chunks)} 个文本块。")
        return all_chunks
