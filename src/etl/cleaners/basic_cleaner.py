# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import re
from collections.abc import Sequence

from src.models.document import Document
from src.utils.config import safe_repr

from .base import BaseCleaner

# 与第 3 步折叠空白用的是同一条判据（``\s`` 的 Unicode 语义）。哨兵候选必须
# 用它筛掉，否则哨兵会被 ``re.sub(r"\s+", " ")`` 一起折叠掉、双换行无法还原。
_WHITESPACE_RE = re.compile(r"\s")


class BasicCleaner(BaseCleaner):
    """
    基础文本内容清洗器。
    执行常见的文本清洗操作，如去除多余空格、换行符等。
    """

    def clean(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        对文档列表中的文本内容进行基础清洗。

        Args:
            documents (List[Document]): 待清洗的文档对象列表。
            **kwargs: 额外的参数（目前未使用）。

        Returns:
            List[Document]: 清洗后的文档对象列表。
        """
        # 形态校验先做：``documents=5`` 抛裸
        # ``TypeError: 'int' object is not iterable``，``"abc"``／``[5]``／
        # ``[None]`` 抛裸 ``AttributeError: 'str' object has no attribute
        # 'content'``（下面的 ``doc.content``）——两者都不在调用方按
        # ``ValueError`` 建立的分类内。判据与同一层的
        # ``RecursiveTextSplitter._require_documents`` 一致（两者在
        # ``Pipeline.process`` 里前后相邻调用，同一份 ``Document`` 列表）。
        if isinstance(documents, (str, bytes, bytearray)) or not isinstance(documents, Sequence):
            raise ValueError(f"清洗器 documents 必须是文档列表，但得到 {safe_repr(documents)}。")
        cleaned_documents = []
        for index, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"清洗器 documents[{index}] 必须是 Document，但得到 {safe_repr(doc)}。"
                )
            # 直接访问 Document 对象的属性
            content = doc.content

            # 1. 将所有回车换行符统一为单个换行符
            content = content.replace("\r\n", "\n")

            # 2. 临时替换双换行符，以防止它们在后续的空白字符处理中被破坏。
            #
            # 哨兵按「内容里不存在的字符」现算，而不是用固定的可读字面量：
            # 原实现硬编码 ``"__DOUBLE_NEWLINE_PLACEHOLDER__"``，它与用户内容
            # 碰撞——文档里出现同样的字符串就会被改写成双换行，而整段由它
            # 构成的段落会被抹成空串（实测 ``"__DOUBLE_NEWLINE_PLACEHOLDER__"
            # * 2`` 清洗后得到空串）。内容是用户数据，不该被实现细节改写。
            #
            # 哨兵取**单个**「内容里没有、且不是空白」的字符，有三条硬要求：
            #
            # a. 不在内容里。否则替换回双换行时会把用户自己的字符也改掉。
            # b. 不是空白字符。第 3 步的 ``re.sub(r"\s+", " ")`` 会把它一起
            #    折叠掉，哨兵丢失、双换行无法还原。
            # c. **单字符**，且不与内容的同类字符相邻。用「NUL 游程 + 1」这种
            #    多字符哨兵看似更省事，实则在边界上歧义：内容
            #    ``"a\x00\x00\x00\n\nb"`` 换出 4 个 NUL 后与原有 3 个 NUL
            #    连成 7 个，``str.replace`` 取最左匹配、只吃掉前 4 个，剩下 3 个
            #    NUL 被留在了双换行**之后**——实测得到
            #    ``'a\n\n\x00\x00\x00b'``，用户内容被静默改写。单个且内容中
            #    不存在的字符不可能有这种歧义。
            #
            # 复杂度：``set(content)`` 一次扫描（O(n)），随后的候选扫描只在
            # 「缺失的非空白字符」上迭代，正常文本第一个候选就命中（NUL）。
            # 旧写法 ``while placeholder in content: placeholder += "\x00"`` 每次
            # ``in`` 都是一次全串扫描、每次 ``+=`` 又让哨兵长一位，于是 N 个 NUL
            # 要扫 N 次——实测 20 万个 NUL 耗时 19.8 秒，100 万 NUL 按平方外推约
            # 1300 秒。触发输入很现实：零填充的 md（写入中断／预分配／误拷的
            # 二进制残片）或 Windows 用户存成 UTF-16LE 的纯 ASCII 文档，
            # ``read_text(encoding="utf-8")`` 对两者**都不报错**，因此
            # ``Pipeline._process_file`` 的解码失败分支拦不住，用户只看到
            # 「向量化处理」卡死数分钟且无任何提示。
            present = set(content)
            placeholder = "\x00"
            for code in range(0x110000):
                if 0xD800 <= code <= 0xDFFF:
                    continue  # 孤立代理项，不能出现在合法文本里
                candidate = chr(code)
                if candidate in present or _WHITESPACE_RE.match(candidate):
                    continue
                placeholder = candidate
                break
            content = content.replace("\n\n", placeholder)

            # 3. 将所有连续的空白字符（包括单个换行符、制表符、多个空格）替换为单个空格
            content = re.sub(r"\s+", " ", content)

            # 4. 将占位符替换回双换行符
            content = content.replace(placeholder, "\n\n")

            # 5. 移除字符串首尾的空白字符
            content = content.strip()

            cleaned_documents.append(Document(content=content, metadata=doc.metadata))
        return cleaned_documents
