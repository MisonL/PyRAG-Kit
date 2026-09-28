from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.pipeline import Pipeline
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import get_settings  # 导入 get_settings 函数

# 用于测试的 Markdown 示例内容
SAMPLE_MARKDOWN = """
# Markdown 测试文档

## 简介

这是一个用于测试 ETL 流水线的 **Markdown** 文件。
它包含多种元素，例如：

- 列表项 1
- 列表项 2

以及一些需要被清洗的 `多余空格`  和

连续的换行符。


## 结论

测试应该能正确处理这些内容。
"""


@pytest.fixture(scope="module")
def mock_markdown_document():
    """提供一个模拟的 Markdown Document 对象用于测试"""
    mock_path = MagicMock(spec=Path)
    mock_path.name = "test.md"
    mock_path.suffix = ".md"
    return Document(content=SAMPLE_MARKDOWN, metadata={"source": str(mock_path)})


def test_full_etl_pipeline_for_markdown(mock_markdown_document):
    """
    测试完整的 ETL 流水线是否能正确处理 Markdown 文档。
    这个测试验证了从文件类型判断、处理器选择到最终切分的整个流程。
    """
    # 备份并临时修改全局配置以适应测试场景
    current_settings = get_settings()  # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators  # 备份分隔符

    # 临时设置较小的块大小，确保文档被分割
    current_settings.kb_chunk_size = 50
    current_settings.kb_chunk_overlap = 10
    current_settings.kb_splitter_separators = ["\n\n", "\n", " ", ""]  # 确保分隔符设置

    try:
        # 从模拟的文件路径初始化 Pipeline
        pipeline = Pipeline.from_file_path(Path(mock_markdown_document.metadata["source"]))
        # 强制设置 splitter 模式为 'char' 以支持旧测试逻辑 (50字符)
        pipeline.splitter = RecursiveTextSplitter(mode="char")

        # 执行处理流程
        processed_docs = pipeline.process(mock_markdown_document)

        # 断言结果
        assert isinstance(processed_docs, list), "处理结果应该是一个列表"
        assert len(processed_docs) > 1, "文档应该被切分成多个部分"

        # 验证清洗效果：不应再有多余的两个以上连续空格或三个以上连续换行符
        for doc in processed_docs:
            assert "  " not in doc.content, "不应存在连续的两个空格"  # 直接检查清洗后的内容
            assert "\n\n\n" not in doc.content, "不应存在连续的三个换行符"

        # 验证元数据是否被正确继承
        assert processed_docs[0].metadata["source"] == mock_markdown_document.metadata["source"]

        # 验证切分内容
        assert "Markdown 测试文档" in processed_docs[0].content
        assert "ETL 流水线" in processed_docs[1].content

    finally:
        # 恢复原始配置，避免影响其他测试
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators  # 恢复分隔符


def test_markdown_extractor(mock_markdown_document):
    """单独测试 MarkdownExtractor 的功能"""
    extractor = MarkdownExtractor()
    # extract 方法返回 List[Document]，所以需要取第一个元素
    extracted_docs = extractor.extract(mock_markdown_document)
    assert isinstance(extracted_docs, list)
    assert len(extracted_docs) == 1
    # Markdown 提取器应该保留原始内容
    assert extracted_docs[0].content == SAMPLE_MARKDOWN


def test_basic_cleaner():
    """单独测试 BasicCleaner 的文本清洗功能"""
    cleaner = BasicCleaner()
    dirty_text = "你好  世界 \n\n\n  再见.  "
    doc = Document(content=dirty_text, metadata={})
    # clean 方法期望 List[Document] 作为输入
    cleaned_docs = cleaner.clean([doc])
    assert isinstance(cleaned_docs, list)
    assert len(cleaned_docs) == 1
    # 验证多余空格、换行符和末尾空格是否被处理
    assert cleaned_docs[0].content == "你好 世界 \n\n 再见."


def test_recursive_text_splitter():
    """单独测试 RecursiveTextSplitter 的文本分割功能"""
    # 备份并临时修改全局配置
    current_settings = get_settings()  # 获取当前配置
    original_chunk_size = current_settings.kb_chunk_size
    original_chunk_overlap = current_settings.kb_chunk_overlap
    original_separators = current_settings.kb_splitter_separators

    # 直接在测试中设置适合分割的参数
    test_chunk_size = 20
    test_chunk_overlap = 5
    test_separators = ["\n\n", "\n", " "]  # 明确分隔符

    try:
        # 实例化 RecursiveTextSplitter 时，它会从 settings 读取配置
        current_settings.kb_chunk_size = test_chunk_size
        current_settings.kb_chunk_overlap = test_chunk_overlap
        current_settings.kb_splitter_separators = test_separators

        splitter = RecursiveTextSplitter(mode="char")  # 实例化时强制用 char 模式

        # 使用一个更长的文本来测试分割和重叠
        # 注意断言的是「每个块都不超过 chunk_size」而不是某个固定的块数：
        # 旧断言要求恰好 2 块，但第 1 块长 22 字符、已经超过 chunk_size=20——
        # 也就是说这条断言把「chunk_size 失效」当成了期望行为。分隔符表补上 ""
        # 兜底后长句会被真正切开，块数随之变化。
        long_text = "这是一个非常长的句子，需要被正确地切分开来。\n\n这是第二部分。"
        doc = Document(content=long_text, metadata={"source": "test.txt"})

        # split 方法期望 List[Document] 作为输入
        split_docs = splitter.split([doc])

        assert split_docs, "分割结果不应为空"
        for chunk in split_docs:
            assert len(chunk.content) <= test_chunk_size, (
                f"分片长度 {len(chunk.content)} 超过 chunk_size={test_chunk_size}: "
                f"{chunk.content!r}"
            )
        # 内容不得丢失：所有分片的并集必须覆盖原文的每一个非空白字符。
        # 不能用「拼接后等于原文」——chunk_overlap=5 的设计就是让相邻块共享
        # 5 个字符，拼接必然重复，那是正确行为而不是丢失。
        covered = "".join(chunk.content for chunk in split_docs)
        for char in long_text.replace("\n", "").replace(" ", ""):
            assert char in covered, f"字符 {char!r} 在所有分片中都找不到"
        assert split_docs[0].content.startswith("这是一个非常长的句子"), (
            f"首块应保留原文开头，实际为 {split_docs[0].content!r}"
        )
        assert split_docs[-1].content.endswith("这是第二部分。"), (
            f"末块应保留原文结尾，实际为 {split_docs[-1].content!r}"
        )
        # 验证元数据
        assert all(chunk.metadata["source"] == "test.txt" for chunk in split_docs)

    finally:
        # 恢复原始配置
        current_settings.kb_chunk_size = original_chunk_size
        current_settings.kb_chunk_overlap = original_chunk_overlap
        current_settings.kb_splitter_separators = original_separators


# ── 回归：清洗器不得用会被用户内容命中的哨兵 ──


@pytest.mark.parametrize(
    "text",
    [
        "本文档说明 __DOUBLE_NEWLINE_PLACEHOLDER__ 的用途。",
        "A\n\nB __DOUBLE_NEWLINE_PLACEHOLDER__ C\n\nD",
        "__DOUBLE_NEWLINE_PLACEHOLDER____DOUBLE_NEWLINE_PLACEHOLDER__",
    ],
)
def test_basic_cleaner_does_not_rewrite_content_that_looks_like_its_sentinel(text):
    """用户内容里出现旧哨兵字面量时不得被改写。

    原实现硬编码 ``"__DOUBLE_NEWLINE_PLACEHOLDER__"`` 作为双换行的临时替身，
    与用户内容碰撞：文档里出现同样的字符串会被改写成双换行，而整段由它构成的
    内容会被抹成空串（实测两个该字面量相连 -> 清洗后得到 ``""``）。内容是用户
    数据，不该被实现细节改写。
    """
    cleaner = BasicCleaner()

    cleaned = cleaner.clean([Document(content=text, metadata={})])

    assert cleaned[0].content == text.strip()


# 复杂度判据的规模与阈值。基准取 20k 而不是 2k：2k 的耗时约 0.1ms，已接近
# ``time.perf_counter()`` 的可靠量级，机器负载的抖动会让比值从 ~50 跳到 ~900
# （实测同一台机器上 8 次采样出现 967、598 两个越界值），是一条会间歇误报的
# 断言。20k 的基准约 0.7ms，采样稳定得多。
_SENTINEL_BASE_COUNT = 20_000
_SENTINEL_LARGE_COUNT = 200_000
# 每档取多次采样的**最小值**：最小值逼近「无干扰」耗时，而最大值会把 GC 与
# 调度抖动算进去。
_SENTINEL_REPEATS = 5
# 规模放大 10 倍：线性实现实测比值 7.8~11.5，平方实现实测 700。取 60 作为判据
# ——给线性留约 5 倍余量，同时把平方增长稳稳判红（离阈值还有 12 倍）。
_SENTINEL_MAX_RATIO = 60


def test_basic_cleaner_sentinel_length_is_computed_in_one_pass():
    """哨兵长度必须一次算够——逐字符增长是 O(n²)，能把建库挂死。

    原写法 ``while placeholder in content: placeholder += "\x00"`` 每次 ``in``
    都是一次全串扫描、每次 ``+=`` 又让哨兵长一位，于是 N 个 NUL 要扫 N 次。
    实测 20 万个 NUL 耗时 19.8 秒（旧的字面量哨兵是 O(1)），100 万 NUL 按平方
    外推约 1300 秒。

    触发输入很现实：零填充的 md（写入中断／预分配／误拷的二进制残片）或
    Windows 用户存成 UTF-16LE 的纯 ASCII 文档——``read_text(encoding="utf-8")``
    对两者**都不报错**，因此 ``Pipeline._process_file`` 的解码失败分支拦不住，
    用户只看到「向量化处理」卡死数分钟且无任何提示。

    这里断言的是**复杂度**而不是绝对耗时：单纯卡一个秒数阈值会在慢机器上误报，
    而平方增长（10 倍规模 → 100 倍耗时）无论如何都超出线性量级。
    """
    import time

    cleaner = BasicCleaner()

    def clean_duration(count: int) -> float:
        content = "\x00" * count
        start = time.perf_counter()
        cleaner.clean([Document(content=content, metadata={})])
        return time.perf_counter() - start

    def best_duration(count: int) -> float:
        # 预热一次，避免把首次调用的 import/编译开销算进基准。
        clean_duration(1_000)
        return max(min(clean_duration(count) for _ in range(_SENTINEL_REPEATS)), 1e-5)

    small = best_duration(_SENTINEL_BASE_COUNT)
    large = best_duration(_SENTINEL_LARGE_COUNT)

    assert large / small < _SENTINEL_MAX_RATIO, (
        f"{_SENTINEL_LARGE_COUNT} 个 NUL 耗时 {large:.3f}s，"
        f"而 {_SENTINEL_BASE_COUNT} 个耗时 {small:.3f}s"
        f"（比值 {large / small:.0f}，阈值 {_SENTINEL_MAX_RATIO}）"
        "——哨兵长度可能又退回逐字符增长。"
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a\x00\x00\x00\n\nb", id="nul-run-then-paragraph-break"),
        pytest.param("a\n\n\x00\x00\x00b", id="paragraph-break-then-nul-run"),
        pytest.param("a\x00\n\n\x00b", id="nul-runs-on-both-sides"),
        pytest.param("a\x00\n\nb", id="single-nul"),
        pytest.param("\x00\x00\n\n", id="nul-run-and-break-only"),
    ],
)
def test_basic_cleaner_keeps_paragraph_breaks_adjacent_to_nul_runs(text):
    """双换行紧邻 NUL 游程时，正文必须逐字保留、双换行仍还原成 ``\n\n``。

    哨兵用「NUL 游程 + 1」这种多字符形态时，边界上是歧义的：内容
    ``"a\x00\x00\x00\n\nb"`` 换出 4 个 NUL 后与原有 3 个 NUL 连成 7 个，
    ``str.replace`` 取最左匹配、只吃掉前 4 个，剩下 3 个 NUL 被留在双换行
    **之后**——实测得到 ``'a\n\n\x00\x00\x00b'``，用户内容被静默改写。
    单个且内容中不存在的字符不会有这种歧义。
    """
    cleaner = BasicCleaner()

    cleaned = cleaner.clean([Document(content=text, metadata={})])

    assert cleaned[0].content == text.strip()


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a\x01\n\nb", id="soh"),
        pytest.param("a\x02\n\nb", id="stx"),
        pytest.param("a\x00\x01\n\nb", id="soh-after-nul"),
    ],
)
def test_basic_cleaner_sentinel_survives_hostile_control_characters(text):
    """内容里已有候选哨兵字符时，必须另选一个不在内容里的。

    ``\x00``／``\x01`` 都是「非空白且可能被用作哨兵」的字符。内容里已经出现
    它们时，若仍拿它当哨兵，替换回双换行那一步会把用户自己的字符也改成
    ``\n\n``——正是「哨兵必须不在内容里」这条要求要挡住的情形。
    """
    cleaner = BasicCleaner()

    cleaned = cleaner.clean([Document(content=text, metadata={})])

    assert cleaned[0].content == text.strip()


def test_basic_cleaner_sentinel_is_not_a_whitespace_character():
    r"""哨兵不能是空白字符，否则会被 ``re.sub(r"\s+", " ")`` 一起折叠掉。

    内容里塞满各种空白（含 Unicode 空白）时，第一个非空白候选必然是控制字符，
    双换行必须仍然还原。若哨兵退化成空格/制表符之类，双换行会被折叠成单个
    空格，这条会变红。
    """
    cleaner = BasicCleaner()

    text = "\t\n\r \u00a0\u3000\u200b\n\n尾"
    cleaned = cleaner.clean([Document(content=text, metadata={})])

    assert "\n\n" in cleaned[0].content


def test_basic_cleaner_preserves_paragraph_breaks_and_collapses_whitespace():
    """收紧不能误伤：双换行保留、连续空白折叠、首尾去空白的既有行为不变。"""
    cleaner = BasicCleaner()

    cleaned = cleaner.clean([Document(content="你好  世界 \n\n\n  再见.  ", metadata={})])

    assert cleaned[0].content == "你好 世界 \n\n 再见."


# ── 回归：子分片尺寸关系只在**层级模式**下校验 ──
#
# 这两条约束原先写在 ``Settings`` 的 model_validator 里，但**模式不在 Settings
# 里**——它是分片时的运行时参数（``build(splitter_structure_mode)``／CLI 的
# ``--mode``）。无条件校验的后果是一组「standard 下完全合法」的取值连配置都
# 加载不了：实测 ``kb_chunk_size=100, kb_child_chunk_size=300`` 被
# ``ValidationError`` 拒绝，而 standard 模式只调用普通分片器
# （``_split_standard_documents``），子分片器的参数根本不参与。
#
# 判定现在放在 ``RecursiveTextSplitter._validate_hierarchical_sizes()``：层级
# 分支一定会构造子分片器，非层级分支一定不会，因此校验的正是真正会用到的参数。


def _settings_with(monkeypatch, **overrides):
    """返回一份按 overrides 覆盖过的 settings 替身，供分片器读取。"""
    real = get_settings()
    # 用 ``model_fields`` 而不是 ``dir()``：后者会连带访问 ``model_computed_fields``
    # 等 pydantic 内部属性，触发 PydanticDeprecatedSince211 警告。
    values = {name: getattr(real, name) for name in type(real).model_fields}
    values.update(overrides)
    stub = type("_StubSettings", (), values)()
    monkeypatch.setattr("src.etl.splitters.recursive_text_splitter.get_settings", lambda: stub)
    return stub


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("子分片大于父分片", {"kb_chunk_size": 300, "kb_child_chunk_size": 1500}),
        ("子分片等于父分片", {"kb_chunk_size": 300, "kb_child_chunk_size": 300}),
        (
            "子分片 overlap 不小于 size",
            {"kb_chunk_size": 300, "kb_child_chunk_size": 100, "kb_child_chunk_overlap": 100},
        ),
    ],
)
def test_hierarchical_mode_rejects_degenerate_child_sizes(monkeypatch, label, overrides):
    """层级模式下必须拒绝会让层级静默退化（或无法推进）的子分片尺寸。"""
    _settings_with(monkeypatch, **overrides)
    splitter = RecursiveTextSplitter(structure_mode="hierarchical")

    with pytest.raises(ValueError, match="kb_child"):
        splitter.split([Document(content="正文" * 300, metadata={"source": "a.md"})])


@pytest.mark.parametrize(
    ("label", "bad_value"),
    [
        ("整数", 10**4301),
        ("列表", ["standard"]),
        ("字典", {"mode": "standard"}),
        ("集合", {"standard"}),
    ],
    ids=["huge-int", "list", "dict", "set"],
)
def test_splitter_rejects_illegal_structure_mode_with_value_error(label, bad_value):
    """非法 ``structure_mode`` 必须报 ``ValueError``，不能漏出 ``TypeError``／位数上限。

    两条独立缺陷，同一个入口：集合成员判断对不可哈希的值抛
    ``TypeError: unhashable type``；消息里插值 ``{structure_mode}`` 对位数超限的
    整数抛 ``ValueError: Exceeds the limit``（``str(10**4301)``）。后者尤其误导
    ——用户看到的是整数转换问题，而真实原因是模式名写错了。

    这个入口的用户可达路径是 ``embed_knowledge_base.py`` 的 ``--mode``（argparse
    有 ``choices`` 拦着）与直接调用分片器的库使用者，因此按库契约要求
    ``ValueError``。
    """
    with pytest.raises(ValueError) as excinfo:
        RecursiveTextSplitter(structure_mode=bad_value)

    message = str(excinfo.value)
    assert "不支持的结构模式" in message, f"{label}: {message}"
    assert "Exceeds the limit" not in message, f"{label}: {message}"


# ── 回归：分割模式与尺寸参数的形态必须报 ValueError ──
#
# ``mode`` 原先完全没有校验，直接赋值后在 ``_get_length_function`` 里只判
# ``== "token"``：任何拼错／大小写不符／非字符串的取值都**静默退回 char 模式**
# （实测 ``"Token"``／``"tokens"``／``None``／``["token"]`` 全部得到 char 分片）。
# ``chunk_size=1500`` 于是意味着 1500 **字符**而不是 1500 token，中文下每 token
# 约一字，分片规模与配置意图不符且无任何提示——违反 AGENTS.md「不用静默回退或
# 占位结果掩盖失败」。
#
# 尺寸参数原先只做比较，不做类型检查：``child_chunk_size='300'`` 构造成功、到
# split 时才抛 ``TypeError: '>=' not supported between instances of 'int' and
# 'str'``，而契约是同文件的 ``ValueError``。


@pytest.mark.parametrize(
    ("label", "bad_value"),
    [
        ("大小写不符", "Token"),
        ("拼写错误", "tokens"),
        ("None", None),
        ("列表", ["token"]),
        ("整数", 123),
        ("位数超限整数", 10**4301),
    ],
    ids=["capitalized", "plural", "none", "list", "int", "huge-int"],
)
def test_splitter_rejects_illegal_mode_with_value_error(label, bad_value):
    """非法 ``mode`` 必须报 ``ValueError``，不能静默退回 char 模式。"""
    with pytest.raises(ValueError) as excinfo:
        RecursiveTextSplitter(mode=bad_value)

    message = str(excinfo.value)
    assert "不支持的分割模式" in message, f"{label}: {message}"
    assert "Exceeds the limit" not in message, f"{label}: {message}"


def test_splitter_accepts_valid_modes():
    """收紧不能误伤：两个合法模式照常构造，且分片行为确实不同。"""
    text = "这是一段中文测试文本，用于观察分片行为。" * 60

    # overlap 必须一并给出：settings 的默认 overlap（150）大于这里的
    # parent_chunk_size，LangChain 会在构造期就拒绝。
    def chunks_for(mode: str) -> list:
        splitter = RecursiveTextSplitter(mode=mode, parent_chunk_size=100, parent_chunk_overlap=10)
        return splitter.split([Document(content=text, metadata={"source": "a.md"})])

    token_chunks = chunks_for("token")
    char_chunks = chunks_for("char")

    assert token_chunks and char_chunks
    # 模式确实生效（若两者相同，说明 mode 又被忽略了）。
    assert len(token_chunks) != len(char_chunks)


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("size 是数字字符串", {"child_chunk_size": "300"}),
        ("overlap 是列表", {"child_chunk_overlap": ["x"]}),
        ("parent size 是数字字符串", {"parent_chunk_size": "1500"}),
        ("parent overlap 是布尔值", {"parent_chunk_overlap": True}),
        ("size 是 None 之外的假值", {"child_chunk_size": 0}),
    ],
    ids=["size-str", "overlap-list", "parent-str", "overlap-bool", "size-zero"],
)
def test_splitter_rejects_malformed_size_arguments(label, kwargs):
    """尺寸参数形态非法时必须在**构造期**报 ``ValueError``。

    原先这些值被静默接受，直到 ``split()`` 里做比较时才抛 ``TypeError``——
    错误现场离写错的调用很远，且 ``TypeError`` 不是本模块的契约类型。
    """
    with pytest.raises(ValueError) as excinfo:
        RecursiveTextSplitter(structure_mode="hierarchical", **kwargs)

    assert "必须是整数" in str(excinfo.value) or "必须大于等于" in str(excinfo.value), label


def test_splitter_accepts_valid_size_arguments():
    """收紧不能误伤：合法尺寸（含 None 表示用 settings）照常构造。"""
    splitter = RecursiveTextSplitter(
        structure_mode="hierarchical",
        parent_chunk_size=500,
        parent_chunk_overlap=50,
        child_chunk_size=100,
        child_chunk_overlap=10,
    )

    assert splitter.parent_chunk_size == 500
    assert splitter.child_chunk_overlap == 10

    defaults = RecursiveTextSplitter(structure_mode="hierarchical")
    assert defaults.parent_chunk_size is None


def test_splitter_rejects_oversized_separator_table(monkeypatch):
    """分隔符表过长必须在构造期报 ``ValueError``，而不是到切分时抛 ``RecursionError``。

    LangChain 的 ``_split_text`` 按分隔符表**递归**，每层消耗一项。实测 1500 项
    抛 ``RecursionError: maximum recursion depth exceeded``——它不是
    ``ValueError``，穿透调用方的错误分类与脱敏。表来自用户可写的
    ``kb_splitter_separators``，``Settings`` 的 validator 只做 ``str -> list``
    拆分、没有长度上界。
    """
    _settings_with(monkeypatch, kb_splitter_separators=[f"sep{i}" for i in range(1500)])

    with pytest.raises(ValueError, match="分隔符表最多"):
        RecursiveTextSplitter(structure_mode="standard")


def test_splitter_accepts_a_reasonable_separator_table(monkeypatch):
    """收紧不能误伤：真实规模的表照常工作。"""
    _settings_with(monkeypatch, kb_splitter_separators=["###", "\n", " ", ""])

    chunks = RecursiveTextSplitter(structure_mode="standard").split(
        [Document(content="a" * 2000, metadata={"source": "a.md"})]
    )

    assert chunks


def test_splitter_accepts_valid_structure_modes():
    """收紧不能误伤：两个合法模式照常构造。"""
    for mode in ("standard", "hierarchical"):
        assert RecursiveTextSplitter(structure_mode=mode).structure_mode == mode


def test_hierarchical_overlap_message_survives_an_unprintable_integer():
    """越界消息本身不得抛异常——它是唯一会插值 ``child_overlap`` 的地方。

    构造器参数是 ``Settings`` 之外的第二条入口：``RecursiveTextSplitter``
    的 ``child_chunk_overlap`` 直接来自调用方，``Settings`` 的上界拦不住它。
    实测修复前 ``child_chunk_overlap=10**4301`` 时，本该报「overlap 必须小于
    size」的消息自己插值这个整数，用户看到的是
    ``ValueError: Exceeds the limit (4300 digits)``——与重叠参数毫无关系，
    而且 ``Exceeds the limit`` 会让人以为是整数转换的 bug。

    ``repr(10**4301)`` 抛 ``ValueError`` 是因为 ``sys.get_int_max_str_digits()``
    默认 4300 位；判据用 ``safe_repr`` 兜住。
    """
    splitter = RecursiveTextSplitter(structure_mode="hierarchical", child_chunk_overlap=10**4301)

    with pytest.raises(ValueError) as excinfo:
        splitter.split([Document(content="正文" * 300, metadata={"source": "a.md"})])

    message = str(excinfo.value)
    assert "kb_child_chunk_overlap" in message, message
    assert "Exceeds the limit" not in message, message


def test_standard_mode_ignores_unused_child_sizes(monkeypatch):
    """收紧不能误伤：standard 模式不使用子分片器，那组取值必须照常可用。

    这是本组问题的**用户可见**后果：只想用 standard 的用户，原先必须先改掉
    用不到的 ``kb_child_*`` 字段才能加载配置。
    """
    _settings_with(monkeypatch, kb_chunk_size=300, kb_child_chunk_size=1500)
    splitter = RecursiveTextSplitter(structure_mode="standard")

    chunks = splitter.split([Document(content="正文" * 300, metadata={"source": "a.md"})])

    assert chunks, "standard 模式必须照常产出分片"


def test_hierarchical_mode_accepts_valid_child_sizes(monkeypatch):
    """收紧不能误伤：合法的层级尺寸照常工作。"""
    _settings_with(
        monkeypatch, kb_chunk_size=300, kb_child_chunk_size=100, kb_child_chunk_overlap=10
    )
    splitter = RecursiveTextSplitter(structure_mode="hierarchical")

    chunks = splitter.split([Document(content="正文" * 300, metadata={"source": "a.md"})])

    assert chunks
    assert splitter.parent_documents, "层级模式必须产出父块"
