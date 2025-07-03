import re
from typing import List, Dict, Any
from .base import BaseCleaner

class BasicCleaner(BaseCleaner):
    """
    基础文本内容清洗器。
    执行常见的文本清洗操作，如去除多余空格、换行符等。
    """

    def clean(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        对文档列表中的文本内容进行基础清洗。

        Args:
            documents (List[Dict[str, Any]]): 待清洗的文档列表，每个文档字典至少包含 'content' 键。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Dict[str, Any]]: 清洗后的文档列表。
        """
        cleaned_documents = []
        for doc in documents:
            content = doc.get("content", "")
            
            # 移除多余的空白字符（包括换行符、制表符等），替换为单个空格
            content = re.sub(r'\s+', ' ', content).strip()
            
            # 可以根据需要添加更多清洗规则，例如：
            # content = re.sub(r'\[.*?\]\(.*?\)','', content) # 移除Markdown链接
            # content = re.sub(r'#+','', content) # 移除Markdown标题符号
            
            cleaned_documents.append({
                "content": content,
                "metadata": doc.get("metadata", {})
            })
        return cleaned_documents
