# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from pydantic import BaseModel
from typing import Dict, Any

class Document(BaseModel):
    """
    表示一个文档块的Pydantic模型。
    用于在ETL流水线和检索过程中传递文本内容及其相关元数据。
    """
    content: str
    metadata: Dict[str, Any] = {}

    def __str__(self):
        """返回文档内容的字符串表示，方便调试。"""
        return f"Document(content='{self.content[:50]}...', metadata={self.metadata})"

    def __repr__(self):
        """返回文档的官方字符串表示。"""
        return self.__str__()