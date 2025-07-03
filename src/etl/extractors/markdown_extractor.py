# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from typing import List
from .base import BaseExtractor
from src.models.document import Document

class MarkdownExtractor(BaseExtractor):
    """
    Markdown 文档内容抽取器。
    在ETL流水线中，此阶段主要起验证和传递作用，
    因为内容已在创建Document对象时从文件中读取。
    """

    def extract(self, document: Document, **kwargs) -> List[Document]:
        """
        从代表源文件的 Document 对象中“抽取”内容。
        对于Markdown，主要是将单个文档对象放入列表中，以符合流水线格式。

        Args:
            document (Document): 包含源文件内容的文档对象。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Document]: 包含单个文档对象的列表。
        """
        # 由于内容已在外部读取并封装到 Document 对象中，
        # 此处的 "extract" 只是按约定返回一个列表。
        return [document]
