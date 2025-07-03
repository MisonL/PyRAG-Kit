import os
from typing import List, Dict, Any
from .base import BaseExtractor

class MarkdownExtractor(BaseExtractor):
    """
    Markdown 文档内容抽取器。
    从 Markdown 文件中抽取原始文本内容。
    """

    def extract(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """
        从 Markdown 文件中抽取文本内容。

        Args:
            source (str): Markdown 文件的路径。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Dict[str, Any]]: 包含抽取出的文档内容的列表。
                                  每个文档字典包含 'content' 和 'metadata'。
        Raises:
            FileNotFoundError: 如果文件路径不存在。
            IOError: 如果读取文件时发生错误。
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"文件未找到: {source}")

        try:
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取文件名作为元数据
            file_name = os.path.basename(source)
            
            return [{
                "content": content,
                "metadata": {"source": file_name, "file_path": source}
            }]
        except Exception as e:
            raise IOError(f"读取文件 {source} 时发生错误: {e}")
