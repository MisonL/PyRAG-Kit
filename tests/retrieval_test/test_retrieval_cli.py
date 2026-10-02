import asyncio
import os

import pytest

from src.retrieval_test.core import run_retrieval_test_async


class FakePromptSession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def prompt_async(self, *_args, **_kwargs):
        return next(self._responses)


def test_run_retrieval_test_async_can_exit(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.VectorStoreFactory.get_default_vector_store",
        lambda: object(),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)

    asyncio.run(run_retrieval_test_async())


def test_run_retrieval_test_async_prints_error_when_retrieval_fails(monkeypatch):
    printed_messages = []

    class FailingRetrievalService:
        def __init__(self, *_args, **_kwargs):
            return None

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["query", "/quit"]),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)
    monkeypatch.setattr("src.retrieval_test.core.RetrievalService", FailingRetrievalService)
    monkeypatch.setattr(
        "src.retrieval_test.core.EmbeddingService", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.VectorStoreFactory.get_default_vector_store", lambda: object()
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.console.print",
        lambda message, *args, **kwargs: printed_messages.append(str(message)),
    )

    asyncio.run(run_retrieval_test_async())

    assert any("召回测试出错" in message for message in printed_messages)


def test_retrieval_failure_output_redacts_credentials(monkeypatch):
    """召回测试的错误出口曾把裸异常交给 console/logger，凭证会原样落到
    终端与日志；``chat`` 路径已做脱敏，两者必须一致。

    断言行为而不是源码文本：读源码做子串匹配既拦不住回归，也无法覆盖
    拼接、``str(exc)`` 等其它写法。
    """
    secret = "sk-proj-FAKE0000FAKE0000FAKE0000FAKE0000"
    printed_messages = []
    logged_messages = []

    class FailingRetrievalService:
        def __init__(self, *_args, **_kwargs):
            return None

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError(f"服务端返回错误密钥为{secret}。")

    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["query", "/quit"]),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)
    monkeypatch.setattr("src.retrieval_test.core.RetrievalService", FailingRetrievalService)
    monkeypatch.setattr(
        "src.retrieval_test.core.EmbeddingService", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.VectorStoreFactory.get_default_vector_store", lambda: object()
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.console.print",
        lambda message, *args, **kwargs: printed_messages.append(str(message)),
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.logger.error",
        lambda message, *args, **kwargs: logged_messages.append(
            message % args if args else message
        ),
    )

    asyncio.run(run_retrieval_test_async())

    combined = " ".join(printed_messages + logged_messages)
    assert secret not in combined, combined
    assert "REDACTED" in combined, combined


# ── 回归：文档正文与用户输入里的标记形态不得打断召回测试 ──
#
# rich 把 ``console.print(str)``／``Table.add_row(str)`` 的字符串按标记解析。
# 知识库文档正文里出现字面量 ``[/]``（例如「匹配斜杠用 [/] 表示」）时，
# ``display_results`` 抛 ``MarkupError``；而异常处理本身走的是同一个通道——
# ``safe_exception_text()`` 拼出的消息里带着 ``[/]``，于是内层 except 抛、
# ``main()`` 的兜底 except 也抛、``run_cli`` 的兜底 except 再抛，三层全废，
# 程序以裸 traceback 退出。修复前实测确认了这条完整链路。


def _console_buffer(monkeypatch):
    from io import StringIO

    from rich.console import Console

    from src.retrieval_test import core

    buffer = StringIO()
    monkeypatch.setattr(
        core, "console", Console(file=buffer, width=200, force_terminal=False, no_color=True)
    )
    return buffer


MARKUP_LIKE_DOCUMENT = "配置说明：匹配斜杠用 [/] 表示。"


def test_display_results_survives_markup_like_document_content(monkeypatch):
    """文档正文含 ``[/]`` 时表格必须照常渲染，且内容原样显示。"""
    from src.retrieval_test.core import display_results

    buffer = _console_buffer(monkeypatch)

    display_results(
        "斜杠怎么写",
        [
            {
                "page_content": MARKUP_LIKE_DOCUMENT,
                "metadata": {"source": "/kb/[/]说明.md"},
                "score": 0.9,
            }
        ],
    )

    rendered = buffer.getvalue()
    assert "[/]" in rendered


def test_display_results_survives_markup_like_query(monkeypatch):
    """用户自己输入的查询含 ``[/]`` 时，无结果提示面板也必须照常渲染。"""
    from src.retrieval_test.core import display_results

    buffer = _console_buffer(monkeypatch)

    display_results("正则里 [/] 是什么意思", [])

    assert "正则里 [/] 是什么意思" in buffer.getvalue()


def test_retrieval_error_path_survives_markup_like_exception(monkeypatch):
    """异常消息含标记形态时，错误提示必须能打出来。

    这条覆盖的是级联失效：修复前 ``safe_exception_text()`` 的结果再次进入
    rich 标记解析并抛错，把「报告错误」这一步本身也变成崩溃。
    """
    printed = []

    class MarkupLikeFailure:
        def __init__(self, *_args, **_kwargs):
            return None

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError("bad tag [/] in message")

    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["query", "/quit"]),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)
    monkeypatch.setattr("src.retrieval_test.core.RetrievalService", MarkupLikeFailure)
    monkeypatch.setattr(
        "src.retrieval_test.core.EmbeddingService", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.VectorStoreFactory.get_default_vector_store", lambda: object()
    )

    from io import StringIO

    from rich.console import Console

    from src.retrieval_test import core

    buffer = StringIO()
    monkeypatch.setattr(
        core, "console", Console(file=buffer, width=200, force_terminal=False, no_color=True)
    )
    monkeypatch.setattr(core.logger, "error", lambda message, *a, **k: printed.append(message))

    asyncio.run(run_retrieval_test_async())

    assert "召回测试出错" in buffer.getvalue()


# ── 回归：Excel 导出不得把文档内容当公式写入 ──
#
# ``openpyxl`` 按值猜类型：以 ``=`` 开头的字符串会被存成**公式**
# （``data_type='f'``），Excel 打开日志时求值。实测文档内容 ``=1+1`` 落盘后
# 是公式而非文本，``=cmd|'/c calc'!A0`` 这类 DDE 载荷同样照单全收。
# 两条后果：日志不再忠实记录「检索到了什么」（内容列显示的是计算结果），
# 以及在打开日志的机器上执行文档里的内容。


def test_excel_log_keeps_formula_like_content_as_text(tmp_path):
    """以 ``=`` 开头的内容必须按文本落盘，不能被当作公式。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "=1+1",
        [
            {"page_content": "=1+1", "metadata": {"source": "=a.md"}, "score": 0.5},
            {"page_content": "=cmd|'/c calc'!A0", "metadata": {"source": "b.md"}, "score": 0.4},
        ],
    )

    worksheet = load_workbook(logger.filepath).active
    content_column = [
        row[6] for row in worksheet.iter_rows(min_row=2, max_col=8) if row[6].value is not None
    ]
    assert content_column, "日志里没有任何内容行"
    for cell in content_column:
        assert cell.data_type == "s", f"{cell.value!r} 被存成了 {cell.data_type!r}（非文本）"

    query_cell = worksheet.cell(row=2, column=2)
    assert query_cell.data_type == "s", f"查询 {query_cell.value!r} 被存成了公式"
    source_cell = worksheet.cell(row=2, column=5)
    assert source_cell.data_type == "s", f"来源 {source_cell.value!r} 被存成了公式"


# ── 回归：同一秒内的日志文件名必须互不覆盖 ──
#
# 文件名原先只有到**秒**的时间戳，同一秒内构造的第二个实例会拿到同一个路径并
# 覆盖前一个文件——实测两个实例的 ``filepath`` 完全相同。召回测试的**唯一产出**
# 就是这份日志，覆盖等于悄悄丢掉一次测试的结果。可达路径：从召回测试菜单返回
# 主菜单后再次进入（``main.py`` 的菜单循环），或嵌入方在同一秒内构造两个 logger。


def test_excel_loggers_in_the_same_second_get_distinct_paths(tmp_path):
    """同一秒内的两个实例必须落在不同文件上，且互不覆盖对方的内容。"""
    from src.retrieval_test.excel_logger import ExcelLogger

    first = ExcelLogger(log_dir=str(tmp_path))
    first.log_results(
        "第一个实例的查询",
        [{"page_content": "第一条", "metadata": {"source": "a.md"}, "score": 0.5}],
    )
    second = ExcelLogger(log_dir=str(tmp_path))
    second.log_results(
        "第二个实例的查询",
        [{"page_content": "第二条", "metadata": {"source": "b.md"}, "score": 0.6}],
    )

    assert first.filepath != second.filepath
    assert os.path.exists(first.filepath)
    assert os.path.exists(second.filepath)

    # 两边的内容都必须还在——覆盖的话第一个文件里会只剩第二个实例的行。
    from openpyxl import load_workbook

    first_rows = list(load_workbook(first.filepath).active.iter_rows(values_only=True))
    second_rows = list(load_workbook(second.filepath).active.iter_rows(values_only=True))

    assert any("第一个实例的查询" in str(cell) for row in first_rows for cell in row if cell)
    assert not any("第二个实例的查询" in str(cell) for row in first_rows for cell in row if cell)
    assert any("第二个实例的查询" in str(cell) for row in second_rows for cell in row if cell)


def test_excel_logger_keeps_the_plain_name_when_nothing_collides(tmp_path):
    """反向守卫：无冲突时文件名保持原形态，不加多余后缀。"""
    import re

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))

    assert re.fullmatch(r"recall_test_log_\d{8}_\d{6}\.xlsx", os.path.basename(logger.filepath))


def test_excel_logger_skips_names_taken_by_other_files(tmp_path):
    """已存在的同名文件（例如上一次运行留下的）必须被跳过，而不是被覆盖。

    判据是 ``O_CREAT | O_EXCL`` 而不是 ``os.path.exists``：后者是「先查后写」，
    两个进程同时构造时会双双认为路径可用、然后互相覆盖。
    """
    import re
    from datetime import datetime

    from src.retrieval_test.excel_logger import ExcelLogger

    # 预置一个与本次同秒同名的文件，模拟「上一次运行在同一秒留下的日志」。
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    occupied = tmp_path / f"recall_test_log_{timestamp}.xlsx"
    occupied.write_bytes(b"PREEXISTING")

    logger = ExcelLogger(log_dir=str(tmp_path))

    assert logger.filepath != str(occupied)
    assert re.fullmatch(
        rf"recall_test_log_{timestamp}_\d+\.xlsx", os.path.basename(logger.filepath)
    )
    # 已有文件的内容不得被动过。
    assert occupied.read_bytes() == b"PREEXISTING"


def test_excel_logger_cleans_up_the_placeholder_when_construction_fails(tmp_path, monkeypatch):
    """构造中途失败必须回收已独占创建的占位文件。

    ``_reserve_log_path`` 先把文件建出来才能独占，若随后建工作簿/写表头失败，
    日志目录里会留下一个 0 字节的 ``.xlsx``——看起来像一次「没有结果的召回测试」。
    """
    from src.retrieval_test.excel_logger import ExcelLogger

    def boom(self):
        raise RuntimeError("写表头失败")

    monkeypatch.setattr(ExcelLogger, "_write_header", boom)

    with pytest.raises(RuntimeError, match="写表头失败"):
        ExcelLogger(log_dir=str(tmp_path))

    assert list(tmp_path.iterdir()) == [], "失败后不得留下占位文件"


def test_excel_log_keeps_all_content_verbatim(tmp_path):
    """收紧不能误伤：内容必须逐字保留，不加前缀也不改写。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    payloads = ["=1+1", "+1+1", "-1+1", "@SUM(1,1)", "普通文本", "含 [/] 的正文"]
    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "查询",
        [
            {"page_content": text, "metadata": {"source": f"s{i}.md"}, "score": 0.5}
            for i, text in enumerate(payloads)
        ],
    )

    worksheet = load_workbook(logger.filepath).active
    stored = [
        row[6].value
        for row in worksheet.iter_rows(min_row=2, max_col=8)
        if row[6].value is not None
    ]
    assert stored == payloads


# ── 回归：日志文件必须能被读回（写入成功 ≠ 可读） ──
#
# xlsx 是 XML，两处各自拒绝不同的码位：``openpyxl`` 的 ``check_string()`` 在
# ``append()`` 期拒绝 C0 控制符；XML 1.0 不允许的码位在 ``save()`` 期被写成非法
# 字符引用——**写入成功、读回失败**。只处理前者时 ``load_workbook`` 抛
# ``ParseError``，而 ``log_results`` 的 ``except Exception`` 把它吞成一行日志，
# 用户看到「召回测试完成」而文件打不开。


@pytest.mark.parametrize(
    "codepoint,label",
    [
        pytest.param(0xD800, "孤立高代理项起", id="surrogate-high-start"),
        pytest.param(0xDC80, "孤立低代理项", id="surrogate-low"),
        pytest.param(0xDFFF, "孤立高代理项止", id="surrogate-high-end"),
        pytest.param(0xFFFE, "非字符 FFFE", id="noncharacter-fffe"),
        pytest.param(0xFFFF, "非字符 FFFF", id="noncharacter-ffff"),
    ],
)
def test_excel_log_workbook_stays_readable_for_xml_hostile_codepoints(tmp_path, codepoint, label):
    """XML 不接受的码位必须被替换掉，否则日志文件打不开。

    代理项经 legacy pickle 路径可达（pickle 是二进制格式，不受 UTF-8 约束，
    实测 ``pickle.loads(pickle.dumps('caf\\udce9'))`` 正常往返）；``U+FFFE``／
    ``U+FFFF`` 是**合法 UTF-8 码位**，``read_text(encoding="utf-8")`` 与
    ``BasicCleaner`` 的 ``re.sub(r"\\s+", " ")`` 都放行。断言落在
    ``load_workbook`` 能读回上——只断言 ``save()`` 没抛是假绿，因为原实现正是
    「写入成功、读回失败」。
    """
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    hostile = chr(codepoint)
    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        f"查询{hostile}",
        [
            {
                "page_content": f"正文{hostile}内容",
                "metadata": {"source": f"a{hostile}.md"},
                "score": 0.5,
            }
        ],
    )

    worksheet = load_workbook(logger.filepath).active
    stored = [cell.value for row in worksheet.iter_rows() for cell in row if cell.value is not None]

    # 读得回是硬要求；同时断言内容没被整段吃掉，只有那一个码位被替换。
    assert any("正文" in str(value) and "内容" in str(value) for value in stored)
    assert not any(hostile in str(value) for value in stored)


def test_excel_log_character_filter_does_not_touch_legal_characters(tmp_path):
    """收紧不能误伤：合法码位（含 emoji、CJK、``U+D7FF``／``U+E000`` 边界、
    合法空白与 ``U+FFFD``）必须逐字保留。

    载荷刻意不含 ``\r``：XML 规范要求解析器把行尾规范化为 ``\n``，那是 XML 的
    性质而非本过滤器所为（``\r`` 单列在下面断言）。
    """
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    legal = "\t\n 中文 \U0001f600 \ufffd\ud7ff\ue000"
    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "查询",
        [{"page_content": legal, "metadata": {"source": "a.md"}, "score": 0.5}],
    )

    worksheet = load_workbook(logger.filepath).active
    stored = [cell.value for row in worksheet.iter_rows() for cell in row if cell.value is not None]

    assert legal in stored


def test_excel_log_carriage_return_is_normalized_by_xml_not_replaced(tmp_path):
    """``\r`` 被 XML 规范化成 ``\n``，**不是**被本过滤器替换成 ``\ufffd``。

    两者后果不同：规范化只是行尾形态变化（内容仍在），而替换会引入一个用户
    从未写过的字符。这条断言把两种情形区分开，避免把 XML 的固有行为误当成
    过滤器的误伤去「修」。
    """
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "查询",
        [{"page_content": "a\rb", "metadata": {"source": "a.md"}, "score": 0.5}],
    )

    worksheet = load_workbook(logger.filepath).active
    stored = [cell.value for row in worksheet.iter_rows() for cell in row if cell.value is not None]

    assert "a\nb" in stored
    assert not any("\ufffd" in str(value) for value in stored)


@pytest.mark.parametrize(
    "document,label",
    [
        ({"page_content": None, "metadata": {"source": "a.md"}, "score": 0.5}, "内容为 None"),
        ({"page_content": ["x"], "metadata": {"source": "a.md"}, "score": 0.5}, "内容为列表"),
        ({"page_content": "x", "metadata": None, "score": 0.5}, "元数据为 None"),
        ({"page_content": "x", "metadata": ["a"], "score": 0.5}, "元数据为列表"),
        ({"page_content": "x", "metadata": {"source": ["a"]}, "score": 0.5}, "来源为列表"),
        ({"page_content": "x", "metadata": {"source": "a.md"}, "score": None}, "得分为 None"),
        ({"page_content": "x", "metadata": {"source": "a.md"}, "score": "0.5"}, "得分为字符串"),
        ("不是映射", "条目本身不是映射"),
    ],
)
def test_display_results_survives_malformed_documents(monkeypatch, document, label):
    """字段形态异常时展示必须照常完成。

    检索结果可以来自任何 ``VectorStoreBase`` 实现——``FaissStore`` 会在载入期
    校验并归一化，但第三方 store 不会，而抽象基类正是为替换实现而存在的。
    展示是错误处理之后的最后一环，它本身不该成为新的失败点。
    """
    from src.retrieval_test.core import display_results

    buffer = _console_buffer(monkeypatch)

    display_results("查询", [document])

    assert buffer.getvalue(), label


# ── 回归：Excel 日志必须容忍形态异常的检索结果 ──
#
# ``log_results`` 直接对每个条目做 ``doc.get(...)``／``metadata.get(...)`` 并把
# 值原样交给 ``openpyxl`` 与 ``json.dumps``。检索结果可以来自任何
# ``VectorStoreBase`` 实现——``FaissStore`` 归一化，第三方 store 不会，而抽象
# 基类正是为替换实现而存在的。实测六类形态会抛异常，且都发生在**写日志**这一步：
#
# * ``metadata`` 为 ``None``／列表 → ``AttributeError: 'NoneType' object has no
#   attribute 'get'``；
# * 条目本身不是映射 → 同样的 ``AttributeError``；
# * ``page`` 为映射／列表 → ``openpyxl`` 抛 ``ValueError: Cannot convert {...}
#   to Excel``；
# * ``semantic_score`` 为 ``set`` → ``json.dumps`` 抛 ``TypeError: Object of
#   type set is not JSON serializable``；
# * ``score`` 为映射／列表 → 同样进不了 Excel。
#
# 日志是**旁路**：它失败不该让一次成功的召回测试以异常收场，也不该丢掉其余
# 正常条目的记录。判据因此是「不抛异常且正常条目仍被写入」，而不是「异常字段
# 被写成某个特定值」。


@pytest.mark.parametrize(
    "document,label",
    [
        ({"page_content": "c", "metadata": None, "score": 0.1}, "元数据为 None"),
        ({"page_content": "c", "metadata": ["a"], "score": 0.1}, "元数据为列表"),
        (
            {"page_content": "c", "metadata": {"source": "s", "page": {"v": 1}}, "score": 0.1},
            "页码为映射",
        ),
        (
            {"page_content": "c", "metadata": {"source": "s", "page": [1]}, "score": 0.1},
            "页码为列表",
        ),
        ({"page_content": "c", "metadata": {"source": "s"}, "score": {"v": 1}}, "得分为映射"),
        ({"page_content": "c", "metadata": {"source": "s"}, "score": [1]}, "得分为列表"),
        ({"page_content": "c", "metadata": {"source": "s"}, "score": float("nan")}, "得分为 nan"),
        ({"page_content": "c", "metadata": {"source": "s"}, "score": float("inf")}, "得分为 inf"),
        (
            {
                "page_content": "c",
                "metadata": {"source": "s"},
                "score": 0.1,
                "semantic_score": {1, 2},
            },
            "语义得分为集合",
        ),
        # 超大整数是**另一类**失败：形态上完全合法（就是 int），值却过不了下游。
        # 两条下游路径的容忍度不同，因此两个量级都要测：
        # * ``10**400``（401 位）——``json.dumps`` **能**序列化，但 ``openpyxl``
        #   写单元格前会调 ``isnan(value)``，那里抛 ``OverflowError: int too
        #   large to convert to float``（双精度上界约 1.8e308，即 309 位）。
        #   所以这一条只有 ``page``（走 openpyxl）会红，``semantic_score``（走
        #   json）不会——保留它正是为了钉住「json 侧不该被误收紧」。
        # * ``10**4301``——连 ``str()`` 都抛 ``ValueError: Exceeds the limit
        #   (4300 digits)``（CPython 的 ``sys.get_int_max_str_digits()``），
        #   两条路径都会红。
        #
        # 实测修复前这两种量级都让**整个工作簿一个字都没落盘**：异常在
        # ``workbook.save()`` 处被 ``except Exception: logger.exception(...)``
        # 吞掉，用户只看到日志里一行「保存 Excel 日志文件失败」，整批召回测试
        # 记录静默丢失。
        (
            {"page_content": "c", "metadata": {"source": "s", "page": 10**400}, "score": 0.1},
            "页码超出双精度范围",
        ),
        (
            {"page_content": "c", "metadata": {"source": "s", "page": 10**4301}, "score": 0.1},
            "页码超出整数位数上限",
        ),
        (
            {
                "page_content": "c",
                "metadata": {"source": "s"},
                "score": 0.1,
                "semantic_score": 10**400,
            },
            "语义得分超出双精度范围",
        ),
        (
            {
                "page_content": "c",
                "metadata": {"source": "s"},
                "score": 0.1,
                "keyword_score": 10**4301,
            },
            "关键词得分超出整数位数上限",
        ),
        ("不是映射", "条目本身不是映射"),
    ],
)
def test_excel_log_survives_malformed_documents(tmp_path, document, label):
    """形态异常的条目不得让写日志抛异常，且日志必须**真的**落盘。

    判据是「能读回且数据行在里面」，不是「文件存在」——后者是假绿：``save()``
    失败时 ``openpyxl`` 已经写下了部分 ZIP 内容，文件存在且非空，但
    ``load_workbook`` 抛 ``KeyError: There is no item named '[Content_Types].xml'
    in the archive``。实测这条路径确实会留下这样一个 2290 字节的坏文件，因此
    「文件存在」在**未修复**的代码上也成立。
    """
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))

    logger.log_results("查询", [document])

    # 表头 + 一条数据行。只查「文件存在」会在 save() 失败时假绿。
    worksheet = load_workbook(logger.filepath).active
    assert worksheet.max_row >= 2, f"{label}: 数据行未写入"


def test_excel_log_still_records_good_rows_next_to_malformed_ones(tmp_path):
    """一条坏条目不得带走整批记录——其余条目照旧写入。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "查询",
        [
            {"page_content": "第一条", "metadata": {"source": "a.md"}, "score": 0.5},
            {"page_content": "坏条目", "metadata": None, "score": 0.4},
            {"page_content": "第三条", "metadata": {"source": "c.md"}, "score": 0.3},
        ],
    )

    worksheet = load_workbook(logger.filepath).active
    stored = [
        row[6].value
        for row in worksheet.iter_rows(min_row=2, max_col=8)
        if row[6].value is not None
    ]
    assert "第一条" in stored
    assert "第三条" in stored


# ── 回归：``log_results`` 的 ``query`` 列必须与其余列同判据 ──
#
# ``query`` 是 ``log_results`` 里唯一没走收敛的列：其余列分别经
# ``document_*``／``_cell_value``／``_json_safe``，而 ``query`` 直接进 ``row``。
# 实测 ``query=10**4301``（位数超限的整数）时 ``openpyxl`` 在 ``save()`` 期对
# 每个数值调 ``safe_string`` -> ``isnan(value)``，抛
# ``OverflowError: int too large to convert to float``。后果正是 ``_cell_value``
# 的 docstring 描述的那一种：``save`` 外的 ``except Exception`` 把它吞成一行
# 日志，整个工作簿**一个字都没落盘**——同一次查询里完全正常的结果一并丢失，
# 用户看到「召回测试完成」而日志文件是坏的。


def test_excel_log_survives_huge_integer_query(tmp_path):
    """``query`` 是位数超限的整数时，整批结果仍必须落盘。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        10**4301,
        [
            {"page_content": "正文1", "metadata": {"source": "s1"}, "score": 0.9},
            {"page_content": "正文2", "metadata": {"source": "s2"}, "score": 0.8},
            {"page_content": "正文3", "metadata": {"source": "s3"}, "score": 0.7},
        ],
    )

    worksheet = load_workbook(logger.filepath).active
    # 表头 + 3 条结果。修复前 save 中途失败，文件连 [Content_Types].xml 都没有，
    # load_workbook 会直接抛 KeyError。
    assert worksheet.max_row == 4, f"期望表头 + 3 条结果，实际 {worksheet.max_row} 行"
    # query 列降级成量级描述，而不是抛异常。
    assert worksheet.cell(row=2, column=2).value


def test_excel_log_keeps_string_query_verbatim(tmp_path):
    """收紧不能误伤：正常字符串 query 逐字落盘。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "如何清除浏览器缓存？",
        [{"page_content": "正文", "metadata": {"source": "s"}, "score": 0.5}],
    )

    worksheet = load_workbook(logger.filepath).active
    assert worksheet.cell(row=2, column=2).value == "如何清除浏览器缓存？"


# ── 回归：log_results 对非列表的 results 静默写出垃圾行 ──
#
# ``log_results(query, results)`` 声明 ``results: list[dict]``，但它是**公开
# 方法**，且本模块的既有契约是「日志是旁路、失败不该丢记录」（见 ``_cell_value``
# 与 ``query`` 的收敛说明）。实测非列表取值有两种收场，都不在契约内：
#
#   * ``results = "abc"`` —— 字符串是合法可迭代对象，``for i, doc in
#     enumerate(results)`` 逐**字符**迭代，每个字符经
#     ``document_*`` 助手收敛成默认值，于是写出 3 条内容与来源都是占位符的
#     「结果」行。用户看到日志里写着 3 条命中，而实际一次检索返回的是字符串。
#   * ``results = 5`` / ``True`` / ``1.5`` —— 抛裸 ``TypeError: 'int' object is
#     not iterable``，而 ``log_results`` 的调用点位于 ``try`` **之外**
#     （见 ``_append_text_row`` 的说明），异常直接冒到
#     ``run_retrieval_test_async`` 的 ``except Exception``，一次**成功**的召回被
#    显示成「召回测试出错」。
#
# 形态判据与 ``display_results`` 一致（那里 ``if not documents`` 后直接迭代）
# ——但那里只判空、不判类型，因此这里按「非列表一律视为无结果」收敛：写入一行
# 「无结果」比写出三条占位垃圾诚实。


@pytest.mark.parametrize(
    "results",
    [
        pytest.param("abc", id="str"),
        pytest.param(5, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
    ],
)
def test_excel_logger_treats_non_list_results_as_empty(tmp_path, results):
    """非列表 ``results`` 不得逐字符迭代出垃圾行，也不得抛裸 ``TypeError``。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results("查询", results)

    rows = list(load_workbook(logger.filepath).active.iter_rows(values_only=True))
    # 表头 + 一行「无结果」，不得出现逐字符生成的伪结果行。
    assert len(rows) == 2, rows
    assert rows[1][4] == "无结果"


def test_excel_logger_still_logs_real_result_lists(tmp_path):
    """收紧不能误伤：正常的文档列表照常逐条落盘。"""
    from openpyxl import load_workbook

    from src.retrieval_test.excel_logger import ExcelLogger

    logger = ExcelLogger(log_dir=str(tmp_path))
    logger.log_results(
        "查询",
        [
            {"page_content": "第一条", "metadata": {"source": "a.md"}, "score": 0.5},
            {"page_content": "第二条", "metadata": {"source": "b.md"}, "score": 0.6},
        ],
    )

    rows = list(load_workbook(logger.filepath).active.iter_rows(values_only=True))
    assert len(rows) == 3
    assert {row[6] for row in rows[1:]} == {"第一条", "第二条"}
