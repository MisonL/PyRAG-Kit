# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import re
from typing import List
from .base import BaseCleaner
from src.models.document import Document

class BasicCleaner(BaseCleaner):
    """
    基础文本内容清洗器。
    执行常见的文本清洗操作，如去除多余空格、换行符等。
    """

    def clean(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        对文档列表中的文本内容进行基础清洗。

        Args:
            documents (List[Document]): 待清洗的文档对象列表。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Document]: 清洗后的文档对象列表。
        """
        cleaned_documents = []
        for doc in documents:
            # 直接访问 Document 对象的属性
            content = doc.content
            
            # 1. 将所有回车换行符统一为单个换行符
            content = content.replace('\r\n', '\n')
            
            # 2. 临时替换双换行符，以防止它们在后续的空白字符处理中被破坏
            double_newline_placeholder = "__DOUBLE_NEWLINE_PLACEHOLDER__"
            content = content.replace('\n\n', double_newline_placeholder)
            
            # 3. 将所有连续的空白字符（包括单个换行符、制表符、多个空格）替换为单个空格
            content = re.sub(r'\s+', ' ', content)
            
            # 4. 将占位符替换回双换行符
            content = content.replace(double_newline_placeholder, '\n\n')
            
            # 5. 移除字符串首尾的空白字符
            content = content.strip()
            
            cleaned_documents.append(Document(
                content=content,
                metadata=doc.metadata
            ))
        return cleaned_documents
