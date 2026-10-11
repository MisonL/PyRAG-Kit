# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

from typing import Any

from pydantic import BaseModel, Field

from src.utils.config import safe_repr


class Document(BaseModel):
    """
    表示一个文档块的Pydantic模型。
    用于在ETL流水线和检索过程中传递文本内容及其相关元数据。
    """

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __str__(self):
        """返回文档内容的字符串表示，方便调试。

        ``metadata`` 声明为 ``dict[str, Any]``，取值不受控；``repr`` 里一个位数
        超过 ``sys.get_int_max_str_digits()``（默认 4300）的整数就让整条 f-string
        抛 ``ValueError: Exceeds the limit``。这比「显示不好看」严重：``__repr__``
        是 pytest 失败报告、日志与调试器的**兜底渲染**，它一抛，报错现场就什么都
        看不到（实测 pytest 把失败对象显示成
        ``<[ValueError(...) raised in repr()] Document object at 0x...>``）。

        与 ``ui.display_utils._render_value``／``_path_text`` 同一取舍：只在渲染
        **真的失败**时才降级到 ``safe_repr``，正常路径逐字不变（``safe_repr`` 会在
        80 字符处截断，真实分片 metadata 的 ``repr`` 长约 127 字符，无条件套用会
        改变所有正常文档的调试输出）。
        """
        try:
            return f"Document(content='{self.content[:50]}...', metadata={self.metadata})"
        except (ValueError, RuntimeError, RecursionError):
            return (
                f"Document(content='{self.content[:50]}...', metadata={safe_repr(self.metadata)})"
            )

    def __repr__(self):
        """返回文档的官方字符串表示。"""
        return self.__str__()
