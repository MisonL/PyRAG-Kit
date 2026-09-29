# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import re
import uuid
from collections.abc import Callable
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.document import Document
from src.utils.config import get_settings, safe_repr
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
        # 两条守卫缺一不可：
        # * 集合成员判断对不可哈希的值抛 ``TypeError: unhashable type``——调用方
        #   可以传列表/字典，异常类型与「模式不支持」无关。
        # * 消息里插值 ``{structure_mode}`` 自己会抛：``str(10**4301)`` 撞上
        #   CPython 的整数位数上限，用户看到 ``Exceeds the limit (4300 digits)``。
        if not isinstance(structure_mode, str) or structure_mode not in {
            "standard",
            "hierarchical",
        }:
            raise ValueError(f"不支持的结构模式: {safe_repr(structure_mode)}")
        # ``mode`` 原先完全没有校验，直接赋值后在 ``_get_length_function`` 里只判
        # ``== "token"``：任何拼错/大小写不符/非字符串的取值都**静默退回 char
        # 模式**。实测 ``"Token"``／``"tokens"``／``None``／``["token"]`` 全部得到
        # char 分片——``chunk_size=1500`` 于是意味着 1500 **字符**而不是 1500
        # token，中文下每 token 约一字，分片规模与配置意图不符且无任何提示。
        # 这违反 AGENTS.md「不用静默回退或占位结果掩盖失败」。
        if not isinstance(mode, str) or mode not in {"token", "char"}:
            raise ValueError(f"不支持的分割模式: {safe_repr(mode)}，只接受 'token' 或 'char'。")
        self.mode = mode
        self.structure_mode = structure_mode
        # 尺寸参数的形态守卫。原先只做比较（``_validate_hierarchical_sizes``），
        # 于是 ``child_chunk_size='300'`` 构造成功、到 split 时才抛
        # ``TypeError: '>=' not supported between instances of 'int' and 'str'``；
        # ``child_chunk_overlap=['x']`` 同理。契约是同文件的 ``ValueError``，
        # 而 ``TypeError`` 不是——调用方的错误分类接不住它。
        self.parent_chunk_size = self._require_optional_size(parent_chunk_size, "parent_chunk_size")
        self.parent_chunk_overlap = self._require_optional_size(
            parent_chunk_overlap, "parent_chunk_overlap", minimum=0
        )
        self.child_chunk_size = self._require_optional_size(child_chunk_size, "child_chunk_size")
        self.child_chunk_overlap = self._require_optional_size(
            child_chunk_overlap, "child_chunk_overlap", minimum=0
        )
        self.parent_documents: dict[str, dict[str, object]] = {}
        self._encoder = tiktoken.get_encoding(encoding_name)
        logger.info(
            "初始化 RecursiveTextSplitter，模式: %s, 结构模式: %s, 编码: %s",
            mode,
            structure_mode,
            encoding_name,
        )
        self._init_splitter()

    @staticmethod
    def _require_optional_size(value: Any, label: str, *, minimum: int = 1) -> int | None:
        """把可选的尺寸参数收敛成 ``int``；非法形态抛 ``ValueError``。

        ``None`` 表示「用 settings 的值」，照常放行。``bool`` 是 ``int`` 子类，
        ``True`` 会静默变成 ``1``——与 ``_coerce_number`` 同源的判据。
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} 必须是整数，但得到 {safe_repr(value)}。")
        if value < minimum:
            # 与上面那条同为「值本身非法」，消息里的取值同样要经 ``safe_repr``。
            # 裸 ``{value}`` 在**负的**超大整数上抛
            # ``ValueError: Exceeds the limit``：能走到这个分支的正是
            # ``value < minimum`` 成立的取值，而位数超限的整数里只有负的才满足
            # 它（``10**4301 >= 1`` 不成立，直接返回）。实测
            # ``parent_chunk_size=-(10**4301)`` 时用户看到的是整数转换错误，而不是
            # 「必须大于等于 1」。
            raise ValueError(f"{label} 必须大于等于 {minimum}，但得到 {safe_repr(value)}。")
        return value

    def _get_length_function(self) -> Callable[[str], int]:
        """根据模式返回长度计算函数。"""
        if self.mode == "token":
            return lambda x: len(self._encoder.encode(x))
        return len

    # 分隔符表的长度上界。LangChain 的 ``RecursiveCharacterTextSplitter._split_text``
    # 按分隔符表**递归**：每一层为「剩余分隔符」再开一次调用。实测 1500 个分隔符
    # 抛 ``RecursionError: maximum recursion depth exceeded``——它不是
    # ``ValueError``，穿透调用方的错误分类与脱敏。表来自用户可写的
    # ``kb_splitter_separators``（TOML 数组），``Settings`` 的 validator 只做
    # ``str -> list`` 拆分、没有长度上界。
    #
    # 上界取 64：真实配置是个位数（默认只有 ``["###"]``），而 LangChain 每层只
    # 消耗一个分隔符，64 层远在递归上限之内，同时挡住「配置写错成超长数组」。
    _MAX_SEPARATORS = 64

    @staticmethod
    def _require_separators(separators: Any) -> list[str]:
        """校验分隔符表的形态与长度，返回原列表。

        ``None`` 交给调用方按 settings 兜底；这里只管非 ``None`` 的取值。
        """
        if not isinstance(separators, list):
            raise ValueError(f"分隔符表必须是列表，但得到 {safe_repr(separators)}。")
        for item in separators:
            if not isinstance(item, str):
                raise ValueError(f"分隔符表的每一项都必须是字符串，但得到 {safe_repr(item)}。")
        if len(separators) > RecursiveTextSplitter._MAX_SEPARATORS:
            raise ValueError(
                f"分隔符表最多 {RecursiveTextSplitter._MAX_SEPARATORS} 项，"
                f"但得到 {len(separators)} 项。分隔符表按项递归切分，过长的表会"
                "在切分时抛 RecursionError。"
            )
        return separators

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
            # ``is not None`` 而不是 ``or``：``or`` 把 ``0`` 当假值兜底成 settings
            # 的值（实际用 1500），而 ``_validate_hierarchical_sizes`` 用的是
            # ``is not None``、会保留 0 参与比较——于是「校验的值」与「用的值」
            # 不是同一个，实测 ``parent_chunk_size=0, child_chunk_size=300`` 报出
            # 「parent=0」，而真正生效的是 1500。上面的形态守卫现在会先拒掉 0，
            # 这里同步对齐语义，避免两处判据再次分叉。
            chunk_size=self.parent_chunk_size
            if self.parent_chunk_size is not None
            else current_settings.kb_chunk_size,
            chunk_overlap=self.parent_chunk_overlap
            if self.parent_chunk_overlap is not None
            else current_settings.kb_chunk_overlap,
            separators=self._with_fallback_separator(
                self._require_separators(current_settings.kb_splitter_separators)
            ),
            length_function=self._get_length_function(),
            is_separator_regex=False,
        )
        logger.info(
            "文本分割器已就绪: chunk_size=%s, mode=%s, structure=%s",
            self.parent_chunk_size
            if self.parent_chunk_size is not None
            else current_settings.kb_chunk_size,
            self.mode,
            self.structure_mode,
        )

    def _parent_sizes(self) -> tuple[int, int]:
        """返回当前生效的父分片 ``(size, overlap)``。"""
        settings = get_settings()
        size = (
            self.parent_chunk_size if self.parent_chunk_size is not None else settings.kb_chunk_size
        )
        overlap = (
            self.parent_chunk_overlap
            if self.parent_chunk_overlap is not None
            else settings.kb_chunk_overlap
        )
        return size, overlap

    def _validate_parent_sizes(self) -> None:
        """校验父分片器的 ``overlap < size`` 与 ``overlap <= size / 2``。

        为什么必须在这里判：``RecursiveTextSplitter`` 是公开类，调用方可以直接
        构造它而不经过 ``Settings``，而 LangChain 的
        ``RecursiveCharacterTextSplitter`` 只在 ``overlap > size`` 时报错，
        **相等时静默接受**（实测 size=100/ov=100 通过、ov=101 抛 ValueError）——
        每个分片只前进 1 个字符。

        比例上界（``overlap <= size / 2``）拦的是「接近但不相等」这一类，
        它同样静默：实测一篇 15000 字符的文档在 size=1500 时，overlap=150（10%）
        产出 14 个分片（1.1x），而 overlap=1490（99%）产出 1651 个（137.6x）、
        overlap=1499 产出 13751 个（1145.9x）——全部进入 embedding 与 FAISS
        写入。判据与 ``Settings.validate_overlap_smaller_than_size`` 同源，两个
        入口必须一致（``Settings`` 拦配置来源、这里拦直接构造）。

        ``_validate_hierarchical_sizes`` 对**子**分片早有同一判据（注释写明「相等
        会让子分片器无法推进」），父分片这条同一语义的判据此前缺失；而父分片器在
        standard 与 hierarchical 两种模式下都会构造（见 ``_init_splitter``），
        因此这条校验不能只挂在层级分支上。
        """
        size, overlap = self._parent_sizes()
        if overlap >= size:
            raise ValueError(
                "kb_chunk_overlap 必须小于 kb_chunk_size，"
                f"但得到 overlap={safe_repr(overlap)}、size={safe_repr(size)}。"
                " 相等会让父分片器几乎无法推进，产出成百上千倍的冗余分片。"
            )
        if overlap * 2 > size:
            raise ValueError(
                "kb_chunk_overlap 不能超过 kb_chunk_size 的一半，"
                f"但得到 overlap={safe_repr(overlap)}、size={safe_repr(size)}。"
                " 重叠比例过高会让父分片器几乎无法推进，产出成百上千倍的冗余分片。"
            )

    def _validate_hierarchical_sizes(self) -> None:
        """层级模式下校验子分片与父分片的尺寸关系。

        这两条约束原先写在 ``Settings`` 的 model_validator 里，但**模式不在
        Settings 里**——它是分片时的运行时参数（``build(splitter_structure_mode)``
        或 CLI 的 ``--mode``）。无条件校验的后果是一组「standard 下完全合法」的
        取值连配置都加载不了：实测 ``kb_chunk_size=100, kb_child_chunk_size=300``
        被 ``ValidationError`` 拒绝，而 standard 模式只调用普通分片器
        （``_split_standard_documents``），子分片器的参数根本不参与。

        放在这里判定同时保证了「校验的正是真正会用到的参数」：层级分支一定会
        构造子分片器，非层级分支一定不会。

        * ``child_size >= chunk_size``：层级结构静默退化为一层。实测
          ``kb_chunk_size=300, kb_child_chunk_size=1500`` 被接受后父块数从 2 涨到
          10（每个父块只产出 1 个子块）；相等同理（parent=child=300 时不同父块数
          == 分块数，15/15）。
        * ``child_overlap >= child_size``：子分片器无法推进（得到空分片或死循环）。
        """
        self._validate_parent_sizes()
        settings = get_settings()
        child_size = (
            self.child_chunk_size
            if self.child_chunk_size is not None
            else settings.kb_child_chunk_size
        )
        child_overlap = (
            self.child_chunk_overlap
            if self.child_chunk_overlap is not None
            else settings.kb_child_chunk_overlap
        )
        parent_size = self._parent_sizes()[0]
        if child_overlap >= child_size:
            raise ValueError(
                "kb_child_chunk_overlap 必须小于 kb_child_chunk_size，"
                f"但得到 overlap={safe_repr(child_overlap)}、size={safe_repr(child_size)}。"
                " 相等会让子分片器无法推进。"
            )
        if child_size >= parent_size:
            raise ValueError(
                "kb_child_chunk_size 必须小于 kb_chunk_size（子分片不得大于或等于父分片），"
                f"但得到 child={safe_repr(child_size)}、parent={safe_repr(parent_size)}。"
                " 否则层级结构静默退化为一层。"
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
            separators=self._with_fallback_separator(
                self._require_separators(current_settings.kb_splitter_separators)
            ),
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

        现在只规范化边界空白：把开头的**空白**替换为等长空格再 strip，标点一字
        不动。句中信息一字不改，也不会再产生「内容全被删掉因此该块被跳过」的路径。

        上一版写的是 ``re.sub(r"^[\s.。]+", lambda m: " " * len(m.group()), text)``
        ——看似「替换而不是删除」，实则与旧实现**逐字符等价**：被替换的字符集
        （空白 + ``.`` + ``。``）恰好等于随后 ``strip()`` 会删掉的字符集，替换成
        空格后立刻又被 strip 掉。实测 3 万条随机输入零差异，``"。" * 200`` 依然
        产出 0 块、``separators=["。"]`` 下依然丢 2.9% 的字符。因此那条 CHANGELOG
        声称的「句中信息一字不丢」当时并不成立。

        现在**只**匹配空白。句号属于正文——当分隔符本身就是 ``。`` 时，
        ``keep_separator=True`` 会把上一句的句号留在下一个分片开头，删掉它
        就是丢掉用户内容；而 ``。`` 本身是有意义的分隔信息，保留它才能让分片
        拼接回原文。
        """
        text = re.sub(r"^\s+", lambda match: " " * len(match.group()), text)
        return text.strip()

    def _split_standard_documents(self, documents: list[Document]) -> list[Document]:
        """标准单层分片。"""
        all_chunks: list[Document] = []
        for doc in documents:
            logger.debug(f"正在分割文档: {doc.metadata.get('source', '未知来源')}")

            langchain_chunks = self.text_splitter.create_documents(
                [doc.content], metadatas=[doc.metadata]
            )
            chunks_before = len(all_chunks)
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

            # 日志必须报**实际收录**的块数：原先写的是 ``len(langchain_chunks)``
            # （过滤前的数量），一份被整段判空的文档会显示「生成 8 个块」而实际
            # 贡献 0 个——日志反过来掩盖了问题。
            self._report_document_chunks(doc, len(all_chunks) - chunks_before)
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

            self._report_document_chunks(doc, doc_chunk_count, hierarchical=True)

        return all_chunks

    @staticmethod
    def _report_document_chunks(
        doc: Document, produced: int, *, hierarchical: bool = False
    ) -> None:
        r"""记录单篇文档**实际收录**的块数。

        原先标准分支记的是 ``len(langchain_chunks)``——过滤**前**的数量。一份
        被整段判空的文档会显示「生成 8 个块」而实际贡献 0 个，日志反过来掩盖了
        问题。

        ``produced == 0`` 只可能出现在原文（按 ``\s`` 语义）为空时：过滤条件是
        「该块去掉首尾空白后为空」，若所有块都被滤掉，则所有块都是空白，即原文
        只有空白。实测对 16 种输入 × 5 组分隔符 × 3 组尺寸 × 2 种结构组合做笛卡尔
        扫描，没有出现「原文有非空白内容却产出 0 块」的情形——因此这里不为它写
        警告分支（写不到的死代码只会让人误以为有防护）。
        """
        source = doc.metadata.get("source", "未知来源")
        if produced:
            stage = "层级分割" if hierarchical else "分割"
            logger.debug("文档 '%s' %s完成，生成 %s 个块。", source, stage, produced)
        else:
            logger.debug("文档 '%s' 清洗后为空，跳过。", source)

    def split(self, documents: list[Document], **kwargs) -> list[Document]:
        """
        将文档列表中的文本内容分割成更小的块。
        """
        logger.info(f"开始分割 {len(documents)} 个文档。")

        # 实时同步最新配置
        self._init_splitter()
        self.parent_documents = {}

        if self.structure_mode == "hierarchical":
            self._validate_hierarchical_sizes()
            all_chunks = self.split_hierarchical(documents)
        else:
            # 父分片器的 ``overlap < size`` 与结构模式无关：standard 分支同样会
            # 构造并使用它，这条校验不能只挂在层级分支上。
            self._validate_parent_sizes()
            all_chunks = self._split_standard_documents(documents)

        logger.info(
            "分割任务完成，总计生成 %s 个文本块 (模式: %s, 结构: %s)。",
            len(all_chunks),
            self.mode,
            self.structure_mode,
        )
        return all_chunks
