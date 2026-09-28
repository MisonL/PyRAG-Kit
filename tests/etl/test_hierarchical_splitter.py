import logging
from types import SimpleNamespace

import pytest

from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document


def test_hierarchical_splitter_builds_parent_child_metadata(monkeypatch):
    settings = SimpleNamespace(
        kb_chunk_size=120,
        kb_chunk_overlap=0,
        kb_splitter_separators=["\n\n", "\n", " "],
    )
    monkeypatch.setattr(
        "src.etl.splitters.recursive_text_splitter.get_settings",
        lambda: settings,
    )

    splitter = RecursiveTextSplitter(
        mode="char",
        structure_mode="hierarchical",
        parent_chunk_size=120,
        parent_chunk_overlap=0,
        child_chunk_size=40,
        child_chunk_overlap=0,
    )
    document = Document(
        content="Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 8,
        metadata={"source": "knowledge_base/sample.md"},
    )

    chunks = splitter.split([document])

    assert len(chunks) > 2
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert len({chunk.metadata["parent_id"] for chunk in chunks}) < len(chunks)
    assert all(chunk.metadata["source"] == "knowledge_base/sample.md" for chunk in chunks)
    assert all("parent_content" not in chunk.metadata for chunk in chunks)
    assert all("parent_chunk_index" in chunk.metadata for chunk in chunks)
    assert splitter.parent_documents
    parent_content = next(iter(splitter.parent_documents.values()))["content"]
    assert isinstance(parent_content, str) and parent_content
    assert any(chunk.content != parent_content for chunk in chunks)


# ── 回归：chunk_size 在「分隔符耗尽」的退化输入下也必须生效 ──


def _splitter_with(mode="token", structure_mode="standard", **overrides):
    """构造分片器并覆盖指定设置字段，返回 (splitter, settings, 原值快照)。"""
    from src.utils.config import get_settings

    settings = get_settings()
    original = {key: getattr(settings, key) for key in overrides}
    for key, value in overrides.items():
        setattr(settings, key, value)
    splitter = RecursiveTextSplitter(mode=mode, structure_mode=structure_mode)
    return splitter, settings, original


def _restore(settings, original):
    for key, value in original.items():
        setattr(settings, key, value)


def test_splitter_enforces_chunk_size_without_matching_separator(monkeypatch):
    """默认分隔符 ``["###"]`` 下，不含 ``###`` 的文档不得整篇变成一个块。

    ``RecursiveCharacterTextSplitter._split_text`` 在 ``not new_separators`` 时走
    ``final_chunks.append(s)``——**原样追加**，不再按 ``chunk_size`` 切。默认配置
    ``kb_splitter_separators = ["###"]`` 恰好命中：``###`` 是唯一匹配项，
    ``new_separators`` 直接为空。实测修复前 ``chunk_size=1500`` 时 3600 token 的
    纯列表文本产出单个 3599 token 块。
    """
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    text = "- 项目说明文字\n" * 600
    assert len(encoder.encode(text)) > 1500, "前提：输入必须远超 chunk_size"

    splitter, settings, original = _splitter_with(
        kb_chunk_size=1500, kb_chunk_overlap=0, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "list.md"})])
    finally:
        _restore(settings, original)

    assert len(chunks) > 1, "分隔符耗尽时仍应被硬切分，而不是整篇一个块"
    oversize = [
        len(encoder.encode(chunk.content))
        for chunk in chunks
        if len(encoder.encode(chunk.content)) > 1500
    ]
    assert not oversize, f"以下分片超过 chunk_size=1500: {oversize}"


def test_splitter_output_is_unchanged_when_separator_matches(monkeypatch):
    """兜底分隔符只在「前面都不匹配」时生效，不得改变正常文档的切分结果。

    含 ``###`` 的文档在 ``["###"]`` 与 ``["###", ""]`` 下块数/最大块应完全一致
    （实测 3 / 252）。
    """
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    text = "\n\n".join(f"### 小节{index}\n" + "内容。" * 60 for index in range(6))

    splitter, settings, original = _splitter_with(
        kb_chunk_size=300, kb_chunk_overlap=30, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "doc.md"})])
    finally:
        _restore(settings, original)

    assert len(chunks) == 3
    assert max(len(encoder.encode(chunk.content)) for chunk in chunks) == 252


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("。第7句内容", "。第7句内容", id="leading-ideographic-period"),
        pytest.param("。。开头两句", "。。开头两句", id="two-leading-periods"),
        pytest.param("...文本", "...文本", id="leading-ascii-dots"),
        pytest.param("   \n 正常文本", "正常文本", id="leading-whitespace"),
        pytest.param("开头就是正文", "开头就是正文", id="no-leading-noise"),
    ],
)
def test_strip_leading_punctuation_keeps_every_punctuation_character(text, expected):
    r"""开头标点必须**原样保留**，只有空白被规范化。

    两版实现都没做到这件事：最初是 ``re.sub(r"^[\s.。]+", "", text).strip()``
    直接删除；随后改成「替换为等长空格」看似修好了，实则与旧实现**逐字符等价**
    ——被替换的字符集恰好等于紧随其后的 ``strip()`` 会删掉的字符集，替换出来的
    空格立刻又被删掉（实测 3 万条随机输入零差异）。而 ``。`` 属于正文：当分隔符
    本身就是 ``。`` 时，LangChain 的 ``keep_separator=True`` 会把它留在下一个
    分片开头，删掉它就是丢掉用户内容。

    这条断言把 ``。`` 也算进期望值——旧版本刻意跳过了这个字符（
    ``if char == "。": continue``），于是断言对被测缺陷零区分度。
    """
    splitter = RecursiveTextSplitter(mode="char")

    assert splitter._strip_leading_punctuation(text) == expected


def test_splitter_keeps_all_content_when_punctuation_is_the_separator(monkeypatch):
    """分隔符是句号时不得丢掉任何正文，也不得产出 0 个块。

    实测修复前 ``chunk_size=40, separators=["。"]``：原始 843 字符分成 15 块后只剩
    829 字符，丢失 1.66%；整篇以句号开头的输入更会被判空并 ``continue``，产出 0 块。
    """
    body = "。".join(f"第{index}句内容" for index in range(1, 61))

    splitter, settings, original = _splitter_with(
        kb_chunk_size=40, kb_chunk_overlap=0, kb_splitter_separators=["。"]
    )
    try:
        chunks = splitter.split([Document(content=body, metadata={"source": "body.md"})])
    finally:
        _restore(settings, original)

    assert chunks, "不得产出 0 个块"
    # 断言**逐字相等**而不是「每个字符都能在某个分片里找到」：后者是
    # 子序列判定，丢掉重复的句号照样通过——旧版本甚至刻意写
    # ``if char == "。": continue`` 把唯一会丢的那个字符排除在外，对被测缺陷
    # 零区分度。分片按顺序拼接必须还原原文（本用例 overlap=0）。
    covered = "".join(chunk.content for chunk in chunks)
    assert covered == body, f"正文被改写：原文 {len(body)} 字符，分片合计 {len(covered)} 字符"


# ── 回归：单篇文档实际产出 0 块时必须留下痕迹 ──
#
# 标准分支原先记的是 ``len(langchain_chunks)``（**过滤前**的数量）：一份被整段
# 判空的文档会显示「生成 8 个块」而实际贡献 0 个——日志反过来掩盖了问题。而
# 「原文非空却产出 0 块」这种静默丢数据此前连一行日志都没有，用户会把「检索不到」
# 误当成检索质量问题。


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    def messages(self, level: str) -> list[str]:
        return [message for name, message in self.records if name == level]


@pytest.fixture
def splitter_log_records():
    """捕获分片器 logger 的记录。"""
    import logging as logging_module

    import src.etl.splitters.recursive_text_splitter as module

    handler = _RecordHandler()
    module.logger.addHandler(handler)
    previous_level = module.logger.level
    module.logger.setLevel(logging_module.DEBUG)
    try:
        yield handler
    finally:
        module.logger.removeHandler(handler)
        module.logger.setLevel(previous_level)


def test_splitter_logs_actual_chunk_count_not_pre_filter_count(monkeypatch, splitter_log_records):
    """日志必须报**实际收录**的块数，而不是过滤前的数量。

    ``chunk_size=1`` 时 LangChain 逐字符切分，一段连续的空白会产出多个「纯空白」
    的块，它们随后被 ``if not chunk_content: continue`` 滤掉。实测
    ``"a" + "\t"*5 + "b"`` 在 ``chunk_size=1`` 下过滤前 7 个、实际收录 2 个——
    日志若退回 ``len(langchain_chunks)`` 就会报「生成 7 个块」。
    """
    splitter, settings, original = _splitter_with(
        kb_chunk_size=1, kb_chunk_overlap=0, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split(
            [Document(content="a" + "\t" * 5 + "b", metadata={"source": "count.md"})]
        )
    finally:
        _restore(settings, original)

    assert len(chunks) == 2, f"夹具前提变了：实际收录 {len(chunks)} 个块"

    debug_messages = [m for m in splitter_log_records.messages("DEBUG") if "count.md" in m]
    assert debug_messages, "单篇文档完成时必须留下日志"

    # 日志里的数字必须等于该文档实际贡献的块数。
    import re as re_module

    reported = int(re_module.search(r"生成 (\d+) 个块", debug_messages[-1]).group(1))
    assert reported == len(chunks)


def test_splitter_reports_an_empty_document_as_empty(monkeypatch, splitter_log_records):
    """清洗后为空的文档必须记成「跳过」而不是「生成 0 个块」。

    空文件与纯空白文档是正常输入（清洗器会把它们折叠成空串）。日志要能让用户
    一眼看出「这份文档没内容」，而不是给一个会让人去找分片器 bug 的「0 个块」。
    """
    splitter, settings, original = _splitter_with(
        kb_chunk_size=120, kb_chunk_overlap=0, kb_splitter_separators=["###"]
    )
    try:
        chunks = splitter.split([Document(content="   \n\t  ", metadata={"source": "empty.md"})])
    finally:
        _restore(settings, original)

    assert chunks == []
    messages = [m for m in splitter_log_records.messages("DEBUG") if "empty.md" in m]
    assert messages, "空文档也要留下日志"
    assert "清洗后为空" in messages[-1]
    assert not [m for m in splitter_log_records.messages("WARNING") if "empty.md" in m]


def test_splitter_keeps_punctuation_only_document(monkeypatch):
    """整篇以句号开头/结尾时仍必须产出分片（旧实现全部判空跳过）。"""
    text = "。" * 50 + "正文" + "。" * 50

    splitter, settings, original = _splitter_with(
        kb_chunk_size=40, kb_chunk_overlap=0, kb_splitter_separators=["。"]
    )
    try:
        chunks = splitter.split([Document(content=text, metadata={"source": "dots.md"})])
    finally:
        _restore(settings, original)

    assert chunks
    assert any("正文" in chunk.content for chunk in chunks)


# ── 回归：尺寸参数的越界消息必须可读 ──
#
# ``_require_optional_size`` 的**类型**分支早就用 ``safe_repr``，**范围**分支
# 却是裸 ``{value}``——同一参数、同一份消息，只有一处防住了位数超限。
# 实测 ``parent_chunk_size=-(10**4301)`` 时，正要报「必须大于等于 1」的那一步
# 自己抛 ``ValueError: Exceeds the limit (4300 digits) for integer string
# conversion``：用户拿到的是 CPython 的整数转换错误，与尺寸参数毫无关系。
#
# 只有**负的**超限整数会走到范围分支（``value < minimum`` 成立），因此两个方向
# 都要断言——只测正数会让这条守卫永远绿着。
@pytest.mark.parametrize(
    ("value", "label"),
    [
        (-(10**4301), "parent_chunk_size"),
        (-(10**4301), "child_chunk_overlap"),
        (0, "parent_chunk_size"),
        (-1, "child_chunk_size"),
    ],
    ids=["超限负-parent", "超限负-child", "零", "负一"],
)
def test_splitter_size_rejection_message_is_readable(value, label):
    with pytest.raises(ValueError) as excinfo:
        RecursiveTextSplitter(**{label: value})

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message
    assert label in message
    assert "大于等于" in message


def test_splitter_size_rejection_keeps_small_values_verbatim():
    """收紧不能误伤：上限内的非法取值照旧原样显示，不要被量级描述替换。"""
    with pytest.raises(ValueError) as excinfo:
        RecursiveTextSplitter(parent_chunk_size=-1)

    assert "但得到 -1" in str(excinfo.value)


def test_splitter_accepts_the_largest_legal_size():
    """反向守卫：合法的大取值（位数远在上限内）照常接受。"""
    splitter = RecursiveTextSplitter(parent_chunk_size=10**6, parent_chunk_overlap=0)
    assert splitter.parent_chunk_size == 10**6
