"""``display_chat_config`` 打到终端的内容不得含凭证。

``api_key`` 一直有掩码，但 ``base_url`` 是原样输出的：自建网关常把凭证写在
URL 里（``https://user:token@gateway/v1``），日志侧有 ``RedactingFormatter``
兜底，终端这一路没有——凭证会直接出现在屏幕上、终端回滚里，以及用户随手
截的图里。
"""

from __future__ import annotations

import ast
import os
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from src.ui import display_utils
from src.ui.display_utils import coerce_score, display_chat_config, get_relative_path

SECRET_IN_KEY = "sk-super-secret-key"
SECRET_IN_URL = "sk-real-secret"


@pytest.fixture
def rendered(monkeypatch) -> str:
    """渲染一次配置面板，返回纯文本输出。"""
    settings = SimpleNamespace(
        qwen_api_key=SECRET_IN_KEY,
        qwen_base_url=f"https://user:{SECRET_IN_URL}@gateway.example.com/v1",
    )
    monkeypatch.setattr(display_utils, "get_settings", lambda: settings)

    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False, no_color=True)
    display_chat_config(console, _chat_config())
    return buffer.getvalue()


def test_display_chat_config_redacts_api_key(rendered):
    """API Key 只显示「已设置」，不回显任何字符。"""
    assert SECRET_IN_KEY not in rendered
    assert "已设置（值已隐藏）" in rendered


def test_display_chat_config_redacts_credentials_embedded_in_base_url(rendered):
    """内嵌在 Base URL 里的凭证必须脱敏。"""
    assert SECRET_IN_URL not in rendered, "Base URL 中的凭证被原样打印到终端"
    assert "REDACTED" in rendered


def test_display_chat_config_keeps_base_url_host_visible(rendered):
    """脱敏不能把整个 URL 抹掉——用户需要靠主机名确认连的是哪个网关。"""
    assert "gateway.example.com" in rendered


def _chat_config() -> dict:
    """构造 display_chat_config 需要的最小配置。"""
    return {
        "retrieval_method": SimpleNamespace(value="混合检索"),
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "hybrid_fusion_strategy": "weighted",
        "retrieval_candidate_multiplier": 2,
        "rerank_enabled": True,
        "active_rerank_configuration": "siliconflow",
        "rerank_configurations": {
            "siliconflow": SimpleNamespace(provider="siliconflow", model_name="bge-reranker-v2-m3")
        },
        "top_k": 5,
        "score_threshold": 0.5,
        "active_llm_configuration": "qwen3-8-max",
        "llm_configurations": {
            "qwen3-8-max": SimpleNamespace(provider="qwen", model_name="qwen3-8b")
        },
    }


# ── 回归：配置面板里的数值字段必须可渲染 ──
#
# ``display_chat_config`` 的五个数值字段原先走裸 ``str()``／f-string 插值。
# ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫（见
# ``_require_finite_number`` 的 docstring），嵌入方或携带超大整数的 pickle
# 快照可以直写这些字段，于是 ``str(10**4301)`` 撞上 CPython 的
# ``sys.get_int_max_str_digits()``（默认 4300）抛 ``ValueError: Exceeds the
# limit``。
#
# 后果比「显示不好看」严重得多：本函数在 ``start_chat_session_async`` 里于
# **任何提示符之前**被调用，用户刚看到「知识快照加载成功」就收到一条整数转换
# 错误，会话随即终止。
@pytest.mark.parametrize(
    "field",
    [
        "vector_weight",
        "keyword_weight",
        "retrieval_candidate_multiplier",
        "top_k",
        "score_threshold",
    ],
)
def test_display_chat_config_renders_values_beyond_the_conversion_limit(monkeypatch, field):
    monkeypatch.setattr(
        display_utils,
        "get_settings",
        lambda: SimpleNamespace(qwen_api_key=None, qwen_base_url=None),
    )
    config = _chat_config()
    config[field] = 10**4301

    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False, no_color=True)
    display_chat_config(console, config)

    output = buffer.getvalue()
    assert "Exceeds the limit" not in output
    assert "位十进制" in output


def test_display_chat_config_keeps_small_numbers_verbatim(rendered):
    """收紧不能误伤：正常数值照旧原样显示，不加引号、不截断。"""
    assert "0.7 / 0.3" in rendered
    assert "top_k):" in rendered and " 5" in rendered


# ── 回归：不可信文本不得被当作 rich 标记解析 ──
#
# rich 的 ``console.print(str)``／``Table.add_row(str)``／``Panel(str)`` 都会把
# 字符串按**标记**解析（``Console(markup=True)`` 是默认值），而本项目的这些位置
# 里插着来自知识库文档、用户输入与异常消息的文本。后果有两类：
#
# 1. 崩溃。文本里出现字面量 ``[/]``、``[/color]``、``[/if]`` 这类形态时
#    ``rich.markup.render`` 抛 ``MarkupError``。实测知识库文档里
#    「匹配斜杠用 [/] 表示」这一行就能触发。更糟的是**异常处理本身也在用同一
#    个通道**：``safe_exception_text()`` 拼出的消息里带着 ``[/]``，
#    于是内层 ``except`` 抛、``main()`` 的兜底 ``except`` 也抛、``run_cli``
#    的兜底 ``except`` 再抛——三层全废，程序以裸 traceback 退出。
# 2. 静默改写。``[b]粗体[/b]`` 被吃掉标记，markdown 链接
#    ``[https://x](https://x)`` 丢掉方括号部分。实测真实知识库 3038 行里有
#    12 行在渲染后被改写，用户看到的内容与检索到的内容不一致。


def test_markup_text_keeps_literal_markup_in_values():
    """值里的标记形态必须原样显示，而不是被解析或抛错。"""
    from src.ui.display_utils import markup_text

    buffer = StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, no_color=True)

    for value in [
        "匹配斜杠用 [/] 表示",
        "<p>[/color] 结尾</p>",
        "字段 {{name}} 与 [/if] 标记",
        "value [/bold]",
        "请参考 [b]重要说明[/b] 一节。",
        "[https://docs.conda.io/x.html](https://docs.conda.io/x.html)",
        "[url=https://x]点这里[/url]",
        "C:\\Users\\",
        "结尾反斜杠 \\",
        "a\\[b]",
    ]:
        console.print(markup_text("[bold red]启动失败:[/bold red] {}", value), end="")
        assert buffer.getvalue() == f"启动失败: {value}", value
        buffer.seek(0)
        buffer.truncate()


def test_markup_text_keeps_trusted_styling():
    """收紧不能误伤：模板自身的标记仍要生效，值也要继承所在位置的样式。"""
    from rich.text import Text

    from src.ui.display_utils import markup_text

    text = markup_text("[red]出错: {}[/red]", "值")
    assert isinstance(text, Text)
    assert text.plain == "出错: 值"
    assert [(span.start, span.end, str(span.style)) for span in text.spans] == [
        (0, 4, "red"),
        (4, 5, "red"),
    ]


def test_markup_text_places_multiple_values_in_order():
    """模板里可以放多个占位符，值按顺序落位。"""
    from src.ui.display_utils import markup_text

    text = markup_text("[red]{}[/red] 与 {} 结束", "第一个", "第二个")

    assert text.plain == "第一个 与 第二个 结束"


# ── 回归：``markup_text`` 的值必须可渲染 ──
#
# ``values`` 的类型标注是 ``object``，调用点既有字符串（``safe_exception_text``）
# 也有 int／float（``config_menu`` 的权重与倍率、``chat/core`` 的分块数）。
# 裸 ``str(value)`` 对位数超过 ``sys.get_int_max_str_digits()``（默认 4300）的
# 整数抛 ``ValueError: Exceeds the limit``——正要显示「倍率已更新为 …」的那一步
# 自己抛错，用户看到的是整数转换错误，与配置项毫无关系。
#
# ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫，嵌入方或携带
# 超大整数的 pickle 快照可以直写这些字段（见 ``_require_finite_number`` 的
# docstring），因此非字符串取值是真实形态而不是假想输入。
@pytest.mark.parametrize("huge", [10**4301, -(10**4301)], ids=["正", "负"])
def test_markup_text_renders_values_beyond_the_conversion_limit(huge):
    from src.ui.display_utils import markup_text

    text = markup_text("[green]倍率已更新为: {}[/green]", huge)

    assert "Exceeds the limit" not in text.plain
    assert "位十进制" in text.plain


def test_markup_text_keeps_small_non_string_values_verbatim():
    """收紧不能误伤：正常取值的显示形态不变（``3`` 不是 ``'3'``）。"""
    from src.ui.display_utils import markup_text

    assert markup_text("倍率: {}", 3).plain == "倍率: 3"
    assert markup_text("权重: {}", 0.5).plain == "权重: 0.5"
    assert markup_text("启用: {}", True).plain == "启用: True"


def test_markup_text_value_renderer_is_not_the_naive_str():
    """反向守卫：裸 ``str()`` 确实会抛——否则上面那条断言证明不了什么。"""
    with pytest.raises(ValueError, match="Exceeds the limit"):
        str(10**4301)


def test_markup_text_rejects_placeholder_mismatch():
    """模板与实参数量不符是编码错误，必须显式报错而不是错位拼接。"""
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="占位符"):
        markup_text("[red]{}[/red]")
    with pytest.raises(ValueError, match="占位符"):
        markup_text("没有占位符", "值")
    with pytest.raises(ValueError, match="占位符"):
        markup_text("{} 和 {}", "只有一个")


def test_display_chat_config_keeps_markup_like_base_url_intact(monkeypatch):
    """Base URL 里的标记形态不得被解析——IPv6 字面量本身就带方括号。

    该值来自 ``config.toml``／``.env``，未经任何校验。修复前
    ``https://gw.example.com/v1[/]`` 会让配置面板渲染抛 ``MarkupError``，
    而该异常落在 chat 主循环之外，会以裸 traceback 终止整个会话。
    """
    settings = SimpleNamespace(
        qwen_api_key=SECRET_IN_KEY,
        qwen_base_url="https://gw.example.com/v1[/]",
    )
    monkeypatch.setattr(display_utils, "get_settings", lambda: settings)

    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False, no_color=True)
    display_chat_config(console, _chat_config())

    assert "https://gw.example.com/v1[/]" in buffer.getvalue()


def test_markup_text_rejects_placeholder_inside_tag_parameters():
    """占位符落在标记参数内部是编码错误，必须报错。

    ``[link={}]点这里[/link]`` 不会抛错，而是渲染成 ``点这里https://x``——
    链接文本与 URL 粘连、样式丢失，属于静默错位；``[bold {}]`` 则抛
    ``MarkupError``。值只能出现在标记的文本位置。
    """
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="标记参数内部"):
        markup_text("[link={}]点这里[/link]", "https://x")
    with pytest.raises(ValueError, match="标记参数内部"):
        markup_text("[bold {}]x[/bold]", "red")
    # 文本位置的占位符照常接受。
    assert markup_text("[bold]{}[/bold]", "值").plain == "值"


# ── 回归：配置面板里**由 config.toml 派生**的单元格同样要按字面呈现 ──
#
# 上一轮只把「融合策略」与 Base URL 两个单元格包成了 ``Text``，其余仍是裸
# f-string／``str()``，而这些取值同样来自 ``config.toml``，同样没有白名单：
#
# * ``llm_configurations.<key>.model_name`` —— 工厂只校验 ``provider``
#   （``_provider_map`` 白名单），``model_name`` 原样透传。实测
#   ``model_name = "gemini[/]"`` 时 ``reload_llm()`` 返回 True（模型真的建起来了），
#   紧接着的 ``display_chat_config`` 抛 ``MarkupError``——聊天会话在**启动横幅**
#   这一步就崩，用户看不到任何配置。
# * ``active_llm_configuration`` / ``active_rerank_configuration`` 是配置表的
#   **键名**，任意字符串即可，同样是工厂放行、面板崩溃。
# * ``rerank_configurations.<key>.model_name`` 与 Rerank 行同理。
#
# 判据用「渲染结果里原样含该取值」，而不是「不抛异常」：标记被吃掉时同样不抛，
# 而面板会显示一个与配置不符的名字。


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "llm model_name",
            lambda config: config["llm_configurations"]["qwen3-8-max"].__setattr__(
                "model_name", "gemini[/]"
            ),
        ),
        (
            "rerank model_name",
            lambda config: config["rerank_configurations"]["siliconflow"].__setattr__(
                "model_name", "bge[/]"
            ),
        ),
        (
            "active_llm 键名",
            lambda config: (
                config["llm_configurations"].__setitem__(
                    "a[/]b", config["llm_configurations"].pop("qwen3-8-max")
                ),
                config.__setitem__("active_llm_configuration", "a[/]b"),
            ),
        ),
        (
            "active_rerank 键名",
            lambda config: (
                config["rerank_configurations"].__setitem__(
                    "a[/]b", config["rerank_configurations"].pop("siliconflow")
                ),
                config.__setitem__("active_rerank_configuration", "a[/]b"),
            ),
        ),
    ],
)
def test_display_chat_config_keeps_markup_like_model_fields_intact(monkeypatch, label, mutate):
    """模型名与配置键名里的标记形态必须原样显示，不能崩、也不能被吃掉。"""
    settings = SimpleNamespace(qwen_api_key=None, qwen_base_url=None)
    monkeypatch.setattr(display_utils, "get_settings", lambda: settings)
    config = _chat_config()
    mutate(config)

    buffer = StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, no_color=True)
    display_chat_config(console, config)

    rendered = buffer.getvalue()
    assert "[/]" in rendered, f"{label}: 标记形态被吞掉了"


# ── 回归：值必须继承「占位符所在位置」的样式 ──
#
# 修复前 ``markup_text`` 用 ``Text.from_markup(...).split(占位符)`` 切段，再从
# **切出来的相邻段**上取样式。而占位符本身是被切掉的那一段，样式恰恰只挂在它
# 身上——``[bold cyan]\x00[/bold cyan]`` 的 span 是 ``(0,1,bold cyan)``，
# ``split`` 后右侧段是空串、无样式。于是「值独占标记内容」这一**最常用**的写法
# （仓库里 13 处 ``[bold magenta]{}[/bold magenta]`` 都是这个形状）渲染出来完全
# 没有样式，且不报错——静默丢失。
#
# 反向的错位同样存在：``[bold]x[/bold]{}`` 里占位符在标记**之后**，旧实现却把
# 已拼接结果的末字符样式（``bold``）赋给了值，等于把值染上了它不该有的样式。
#
# 判据用「逐字符样式集合」而不是 span 边界：span 边界随值的长度变化，逐字符
# 比对才能既断言样式内容、又断言样式覆盖的范围。


def _styles_per_char(text):
    """返回 ``text`` 每个字符上叠加的样式名集合。"""
    return [
        frozenset(str(span.style) for span in text.spans if span.start <= index < span.end)
        for index in range(len(text.plain))
    ]


@pytest.mark.parametrize(
    ("template", "values", "expected_plain"),
    [
        ("[bold]{}[/bold]", ("值",), "值"),
        ("[bold cyan]{}[/bold cyan]", ("值",), "值"),
        ("[bold][red]{}[/red][/bold]", ("值",), "值"),
        ("[red]x{}[/red]", ("值",), "x值"),
        ("[red]{}x[/red]", ("值",), "值x"),
        ("[red]x{}y[/red]", ("值",), "x值y"),
        ("[red]{}[/red] 与 {} 结束", ("第一个", "第二个"), "第一个 与 第二个 结束"),
        ("[green]快照目录: [cyan]{}[/cyan][/green]", ("/tmp/x",), "快照目录: /tmp/x"),
        ("[bold]x[/bold]{}", ("值",), "x值"),
        ("[red]{}{}[/red]", ("a", "b"), "ab"),
        ("{}", ("值",), "值"),
        ("[red]{}[/red]", (None,), "None"),
        ("[red]{}[/red]", (42,), "42"),
    ],
)
def test_markup_text_applies_the_style_of_the_placeholder_position(
    template, values, expected_plain
):
    """值要带上「占位符原本所在位置」的样式，范围与数量都要正确。

    参考实现把值经 ``rich.markup.escape`` 转义后填回模板再 ``from_markup``：
    对不含标记的值 escape 是恒等变换，因此这是忠实的期望。
    """
    from rich.markup import escape
    from rich.text import Text

    from src.ui.display_utils import markup_text

    got = markup_text(template, *values)

    reference_markup = template
    for value in values:
        reference_markup = reference_markup.replace("{}", escape(str(value)), 1)
    reference = Text.from_markup(reference_markup)

    assert got.plain == expected_plain
    assert _styles_per_char(got) == _styles_per_char(reference)


def test_markup_text_does_not_leak_a_closed_tag_style_onto_the_value():
    """标记闭合之后的值不得继承该标记的样式。

    旧实现取「已拼接结果末字符」的样式兜底，``[bold]x[/bold]{}`` 因此把值染成
    粗体——值在标记**之外**，这个样式是错的。
    """
    from src.ui.display_utils import markup_text

    text = markup_text("[bold]x[/bold]{}", "值")

    assert text.plain == "x值"
    assert _styles_per_char(text)[1] == frozenset()


def test_markup_text_styles_a_value_spanning_multiple_characters():
    """多字符（含多行）的值要整体染色，而不是只有首字符。"""
    from src.ui.display_utils import markup_text

    text = markup_text("[red]{}[/red]", "多字符的值")

    assert text.plain == "多字符的值"
    assert _styles_per_char(text) == [frozenset({"red"})] * len("多字符的值")


# ── 回归：展示层的字段提取必须容忍形态异常的文档 ──
#
# ``display_results``（召回测试）与 ``Chatbot._display_retrieved_docs``（聊天）都
# 直接对检索结果做 ``.get(...).replace(...)`` 与 ``f"{score:.4f}"``。检索结果可以
# 来自任何 ``VectorStoreBase`` 实现——``FaissStore`` 会在载入期归一化，但
# **第三方 store 不会**，而抽象基类正是为替换实现而存在的。实测一个返回
# ``{"page_content": None, "metadata": None, "score": None}`` 的 store 能让
# 两个展示函数分别抛 ``AttributeError`` 与 ``TypeError``。
#
# 提取助手把这些字段统一收敛成可渲染的值：形态不对就退回默认显示，而不是
# 让「显示检索结果」这一步本身崩掉——它已经是错误处理之后的最后一环了。


@pytest.mark.parametrize(
    "document,expected",
    [
        ({"page_content": "正文"}, "正文"),
        ({"page_content": None}, "N/A"),
        ({"page_content": ["x"]}, "N/A"),
        ({"page_content": 42}, "N/A"),
        ({}, "N/A"),
        ("不是映射", "N/A"),
        (None, "N/A"),
    ],
)
def test_document_content_falls_back_for_malformed_values(document, expected):
    """内容字段的异常形态退回 ``N/A``，不抛错。"""
    from src.ui.display_utils import document_content

    assert document_content(document) == expected


@pytest.mark.parametrize(
    "document,expected",
    [
        ({"metadata": {"source": "a.md"}}, "a.md"),
        ({"metadata": None}, "N/A"),
        ({"metadata": ["a"]}, "N/A"),
        ({"metadata": {"source": None}}, "N/A"),
        ({"metadata": {"source": ["a"]}}, "N/A"),
        ({}, "N/A"),
        ("不是映射", "N/A"),
    ],
)
def test_document_source_falls_back_for_malformed_values(document, expected):
    """来源字段的异常形态退回 ``N/A``，不抛错。"""
    from src.ui.display_utils import document_source

    assert document_source(document) == expected


@pytest.mark.parametrize(
    "document,expected",
    [
        ({"score": 0.5}, 0.5),
        ({"score": 0}, 0.0),
        ({"score": None}, 0.0),
        # 数字字符串照旧转换——旧实现是 ``float(x or 0)``，它接受 ``"0.5"``。
        # 一律拒绝会让返回字符串得分的 store 被静默降为 0，再被分数阈值全部
        # 过滤：从「能用」变成「静默返回空结果」。
        ({"score": "0.5"}, 0.5),
        ({"score": "abc"}, 0.0),
        ({"score": True}, 0.0),
        ({"score": float("nan")}, 0.0),
        ({}, 0.0),
        ("不是映射", 0.0),
    ],
)
def test_document_score_falls_back_for_malformed_values(document, expected):
    """分数字段的异常形态退回 ``0.0``，不抛错。

    ``bool`` 也要退：``True`` 在格式化时显示成 ``1.0000``，看起来像一个真实
    得分，而它几乎必然是上游把标志位写进了分数字段。
    """
    from src.ui.display_utils import document_score

    assert document_score(document) == expected


# ── 结构性回归：把「rich 解析不可信值」这一类问题钉死 ──
#
# 上面每个测试都是针对**某个具体调用点**的回归。这类问题的共同点是「新写的代码
# 又把裸字符串交给了 rich」，逐个补测试挡不住下一次。这里用 AST 直接检查源码：
# 凡是会被 rich 解析标记的入口（``console.print``、``Table.add_row``、``Panel``），
# 其参数必须要么是**字面量**（作者显式写的模板，标记是有意的），要么已经被
# ``Text(...)`` / ``markup_text(...)`` 包装。变量、f-string、下标、拼接一律拒绝。
#
# 例外是那些**定义上就产出安全值**的调用，逐个登记并说明理由——新增例外必须
# 同样给出理由，否则测试会逼着作者去包装。

# ``add_column`` 的 header 同样按标记解析（实测 header="a[/]b" 抛 MarkupError），
# 因此一并纳入。仓库里现有的 header 都是字面量。
_RICH_MARKUP_ENTRY_POINTS = {"add_row", "add_column", "Panel", "Table"}

# 返回 ``Text`` 或**有意的**标记字面量的函数/构造器：调用它们就等于已经包装好了。
_SAFE_CALLS = {
    "Text",  # 显式包装，值不再经过标记解析
    "markup_text",  # 本模块的安全拼接
    "create_gradient",  # main.py：返回 Text
    "mask_api_key",  # 返回有意的标记字面量（"[dim]未设置[/dim]"）
    "Group",  # rich 容器，承载 Table 对象
}

# 构造器里同样会被解析标记的关键字：``Table(title=...)`` 与 ``Panel(title=...)``
# 实测 ``Table(title="a[/]b")`` 抛 MarkupError。这两个类不能整体放进 _SAFE_CALLS
# ——那样连它们的 ``title`` 一起豁免了（``Table`` 承载的是 Table 对象，安全的是
# 它作为**实参**出现时，不是它的 title）。
_MARKUP_PARSING_CONSTRUCTORS = {"Table", "Panel"}


# 逐条登记的例外：这些实参**定义上**不含外部输入，因此不必包装。
# 键是 ``相对路径:行号``，值是理由。行号变了条目就失效——测试会报出来，
# 逼着作者重新确认而不是让豁免静默漂移。
# 值里带一段**该行源码的原文**（去掉行号前缀后的内容）。只按行号登记是不够的：
# 上面增删几行，行号就会漂到另一处 rich 调用上——而邻近的行几乎都是 rich 调用，
# 于是豁免会静默地覆盖一个真正的违规。原文一比对，漂移立刻暴露。
_UNTRUSTED_FREE_ARGUMENTS = {
    "main.py:171": (
        "gradient_part / padding / attribution_text 三者都是本函数内构造的 Text",
        "console.print(gradient_part + padding + attribution_text)",
    ),
    "main.py:194": (
        "welcome_text 是本函数内逐段 append 出来的 Text",
        'console.print(Panel(welcome_text, border_style="green", width=banner_width))',
    ),
    "main.py:207": (
        "menu_content 是本函数内的字面量拼接（上面的测试已断言它是纯字面量）",
        "Panel(",
    ),
    "scripts/embed_knowledge_base.py:124": (
        "table 是 Table 对象，不是字符串",
        "console.print(table)",
    ),
    "src/chat/core.py:383": (
        "f'{document_score(doc):.4f}'：document_score 只返回有限浮点，格式化结果不含 [",
        'f"{document_score(doc):.4f}"',
    ),
    "src/chat/core.py:384": ("table 是 Table 对象，不是字符串", "self.console.print(table)"),
    "src/retrieval_test/core.py:70": (
        "str(index+1) 与 f'{document_score(doc):.4f}' 均为数字，不含 [",
        'f"{document_score(doc):.4f}"',
    ),
    "src/retrieval_test/core.py:71": ("table 是 Table 对象，不是字符串", "console.print(table)"),
}


def _is_literal_or_wrapped(node):
    """判断一个实参是否会原样呈现（而不是被 rich 当标记解析）。"""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in _SAFE_CALLS:
            return True
        # ``Table(title=...)``／``Panel(title=...)`` 的 title 会被解析标记，其余
        # 参数（Table 对象、border_style 等）不会。
        if name in _MARKUP_PARSING_CONSTRUCTORS:
            titles = [kw.value for kw in node.keywords if kw.arg == "title"]
            return all(_is_literal_or_wrapped(title) for title in titles)
        return all(_is_literal_or_wrapped(arg) for arg in node.args) and all(
            _is_literal_or_wrapped(kw.value) for kw in node.keywords if kw.arg
        )
    return not isinstance(node, (ast.JoinedStr, ast.Name, ast.Attribute, ast.Subscript, ast.BinOp))


def test_rich_markup_entry_points_never_receive_raw_variables():
    """凡是会被 rich 解析标记的入口，参数必须是字面量或已包装的值。

    这条断言覆盖 ``main.py``／``src/``／``scripts/`` 的全部源码。它挡的是一整类
    问题（f-string 里塞知识库正文、异常消息、``config.toml`` 取值），而不只是
    已发现的那些调用点。
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted(repo_root.glob("**/*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if relative.startswith((".venv/", "build/", "dist/", "release_artifacts/", "tests/")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "print":
                # 只关心 ``console.print`` / ``self.console.print``，放过 list/dict 的
                # print。取属性链的**根名字**来判断——只看一层 ``node.func.value``
                # 会漏掉 ``self.console.print``（它的 value 是 Attribute 而不是 Name），
                # 而那正是聊天主循环里打印表格的写法。
                base = node.func
                while isinstance(base, ast.Attribute):
                    base = base.value
                if not (isinstance(base, ast.Name) and base.id in {"console", "self"}):
                    continue
            elif name not in _RICH_MARKUP_ENTRY_POINTS:
                continue
            # 逐个入口确定「哪些实参会被解析标记」：
            # * ``add_column`` 的 header 是第一个位置参数，样式参数
            #   （style=、width=）不参与解析；
            # * ``Table`` 只有 ``title`` 参与解析（它的位置参数不是字符串）；
            # * ``Panel`` 的首个位置参数（renderable）与 ``title`` 都参与解析。
            titles = [keyword.value for keyword in node.keywords if keyword.arg == "title"]
            if name == "add_column":
                candidates = list(node.args[:1]) + titles
            elif name == "Table":
                candidates = titles
            else:
                candidates = list(node.args) + titles
            for argument in candidates:
                if _is_literal_or_wrapped(argument):
                    continue
                location = f"{relative}:{node.lineno}"
                if location in _UNTRUSTED_FREE_ARGUMENTS:
                    continue
                offenders.append(f"{location} {name}(... {ast.unparse(argument)})")

    assert not offenders, (
        "以下位置把未经包装的值交给了 rich 的标记解析入口，请用 Text(...) 或 "
        "markup_text(...) 包装；确属外部输入不可达的，在 "
        "_UNTRUSTED_FREE_ARGUMENTS 里登记并写明理由：\n" + "\n".join(offenders)
    )


def test_untrusted_free_argument_allowlist_has_no_stale_entries():
    """登记表不能有失效条目——否则豁免会静默漂移到别的代码上。

    逐条确认「该文件该行确实是一个被豁免的 rich 调用」。行号一动或代码被删，
    条目就对不上，测试报错。
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    stale = []
    for location, (reason, expected_fragment) in _UNTRUSTED_FREE_ARGUMENTS.items():
        relative, _, line_text = location.rpartition(":")
        path = repo_root / relative
        if not path.exists():
            stale.append(f"{location}（文件不存在）")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        line_number = int(line_text)
        if line_number > len(lines):
            stale.append(f"{location}（超出文件行数）")
            continue
        line = lines[line_number - 1]
        if not any(entry in line for entry in ("print(", "add_row(", "Panel(", "Table(")):
            stale.append(f"{location}（该行不再是 rich 调用: {line.strip()!r}）")
            continue
        if not reason:
            stale.append(f"{location}（缺少理由）")
        # 原文必须仍然对得上。只查「这行像不像 rich 调用」不够——邻近的行几乎
        # 都是 rich 调用，行号漂移后豁免会静默覆盖一个真正的违规。
        if expected_fragment and expected_fragment not in line:
            stale.append(
                f"{location}（行号已漂移：该行现在是 {line.strip()!r}，"
                f"但豁免登记的是 {expected_fragment!r}）"
            )

    assert not stale, "以下豁免条目已失效，请更新或删除：\n" + "\n".join(stale)


# ── 回归：构建脚本的配置总览表同样要按字面呈现 provider ──
#
# ``scripts/embed_knowledge_base.py::display_config_and_confirm`` 在**任何校验
# 之前**运行（``main_async`` 先调它、再建服务），它渲染的表格行标签由
# ``provider`` 拼出。修复前 ``provider = "a[/]b"`` 会让表格渲染抛
# ``MarkupError``——「是否使用以上配置继续？」这一步直接变成崩溃，用户连配置
# 都看不到。这个函数此前没有任何测试覆盖。


def _run_embed_config_display(provider: str, model_name: str):
    """调用构建脚本的配置总览，返回（终端输出, 异常）。"""
    from pathlib import Path
    from unittest.mock import patch

    from scripts import embed_knowledge_base

    class _StubRunConfig:
        """只提供 display_config_and_confirm 读取到的字段。"""

        def __init__(self):
            self.knowledge_base_path = Path("/tmp/kb")
            self.snapshot_root = Path("/tmp/snap")
            self.legacy_pkl_path = Path("/tmp/legacy.pkl")
            self.kb_chunk_size = 500
            self.kb_chunk_overlap = 50
            self.kb_child_chunk_size = 200
            self.kb_child_chunk_overlap = 20
            self.kb_embedding_batch_size = 32
            self.default_embedding_provider = "key"
            self.embedding_configurations = {
                "key": SimpleNamespace(provider=provider, model_name=model_name)
            }

    buffer = StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, no_color=True)
    settings = SimpleNamespace()
    with (
        patch.object(embed_knowledge_base, "Console", return_value=console),
        patch.object(embed_knowledge_base, "get_settings", return_value=settings),
        patch.object(embed_knowledge_base, "build_run_config", return_value=_StubRunConfig()),
        patch.object(console, "input", return_value="n"),
    ):
        try:
            embed_knowledge_base.display_config_and_confirm("standard")
        except SystemExit:
            pass  # 用户选择「n」时脚本以 exit(0) 结束，这是正常路径
        except Exception as exc:  # noqa: BLE001 - 测试要断言的就是「不该抛」
            return buffer.getvalue(), exc
    return buffer.getvalue(), None


@pytest.mark.parametrize("provider", ["a[/]b", "a[red]b"])
def test_embed_script_config_table_keeps_markup_like_provider_intact(provider):
    """provider 里的标记形态必须原样显示，不能崩、也不能被吞掉。"""
    rendered, error = _run_embed_config_display(provider, "bge-large")

    assert error is None, f"配置总览渲染抛出了 {type(error).__name__}: {error}"
    assert provider in rendered, "provider 的标记形态被吞掉了"


# ── 回归：转义的 ``\[`` 是字面量，不是标记 ──
#
# rich 的 ``_parse`` 按「前导反斜杠的奇偶」决定 ``[`` 是转义字面量还是真标记
# （``divmod(len(escapes), 2)[1]``）。助手原先只看正则是否匹配到 ``[``，于是
# ``r"\[example {}]"`` 这类**合法**模板被误判为「占位符在标记参数内部」并拒绝，
# 而 rich 自己会把它解析成 ``[example <值>]``。


def test_markup_text_accepts_escaped_bracket_before_the_placeholder():
    """转义的 ``\\[`` 是字面量：占位符不在标记参数内，必须接受。"""
    from rich.text import Text

    from src.ui.display_utils import markup_text

    text = markup_text(r"\[example {}]", "literal")

    assert text.plain == Text.from_markup(r"\[example literal]").plain == "[example literal]"


def test_markup_text_still_rejects_a_real_tag_after_a_literal_backslash():
    """收紧不能误伤：``\\\\[``（双反斜杠）里的 ``[`` 仍是**真标记**，照旧拒绝。"""
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="标记参数内部"):
        markup_text(r"\\[example {}]", "literal")


# ---------------------------------------------------------------------------
# 超出双精度范围的整数
# ---------------------------------------------------------------------------
#
# ``coerce_score`` 的契约是「绝不抛异常，形态异常时退回 ``default``」，而
# ``float(10**400)`` 抛 ``OverflowError``——它不是 ``ValueError``，原实现里
# 两个 ``except`` 都接不住。这个函数是展示层与检索层共用的唯一分数收敛规则，
# 裸异常会让「形态异常就退回默认值」这条契约整体失效。
_BEYOND_DOUBLE_INTS = [
    pytest.param(10**400, id="400digits"),
    pytest.param(-(10**400), id="negative-400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
]


@pytest.mark.parametrize("value", _BEYOND_DOUBLE_INTS)
def test_coerce_score_falls_back_for_ints_beyond_double_range(value):
    """超出双精度范围的整数必须退回默认值，而不是抛 ``OverflowError``。"""
    assert coerce_score(value) == 0.0
    assert coerce_score(value, default=-1.0) == -1.0


# ── 回归：``get_relative_path`` 的兜底分支不得自己抛 ──
#
# 原实现第 122 行的 ``str(absolute_path)`` 抛 ``ValueError: Exceeds the limit``，
# ``except ValueError`` 接住后第 127 行的 f-string **用同一个值再插值一次**，
# 把刚接住的异常原样重抛。兜底分支存在的唯一理由就是不再抛。实际来源是
# ``document_source(document)``，而文档元数据来自磁盘上的快照／legacy pkl。


def test_get_relative_path_survives_huge_integer():
    """位数超限的整数必须降级为量级描述，而不是抛异常。"""
    result = get_relative_path(10**4301)

    assert isinstance(result, str)
    assert result


def test_get_relative_path_actually_resolves_paths_under_root():
    """本函数的**设计意图**是换算相对路径，断言必须盯住这个属性。

    只断言「返回非空字符串」是假绿：一个把入参原样返回的实现也能通过，而
    「原样返回」正是回归时的实际表现。这里用 ROOT_DIR 下的真实路径断言结果
    确实是相对形态。
    """
    from src.utils.config import ROOT_DIR

    absolute = str(ROOT_DIR / "knowledge_base" / "doc.md")

    result = get_relative_path(absolute)

    assert result == os.path.join("knowledge_base", "doc.md")
    assert not os.path.isabs(result)
    assert str(ROOT_DIR) not in result


def test_get_relative_path_does_not_truncate_or_quote_long_paths():
    """超长路径不得被截断，也不得被加上引号。

    此前实现对所有取值都套 ``safe_repr``，它会加引号并在 80 字符处截断：
    带引号的前缀匹配不到 ``ROOT_DIR``（相对路径功能失效），截断则让用户看不到
    文件名——而文件名恰恰是展示路径的用途。
    """
    from src.utils.config import ROOT_DIR

    long_name = "a" * 200 + ".md"
    absolute = str(ROOT_DIR / "knowledge_base" / long_name)

    result = get_relative_path(absolute)

    assert result == os.path.join("knowledge_base", long_name)
    assert long_name in result
    assert not result.startswith("'")


def test_get_relative_path_still_resolves_normal_paths(tmp_path):
    """正常路径不受影响。"""
    result = get_relative_path(tmp_path / "kb" / "doc.md")

    assert isinstance(result, str)
    assert result


# ── 回归：激活的 provider key 缺失时必须报 ValueError ──
#
# ``default_llm_provider``／``default_rerank_provider``／
# ``active_*_configuration`` 没有任何 validator，而原实现直接
# ``chat_config["llm_configurations"][active_llm]``。实测打错一个 provider 名时，
# 调用点 ``chat/core.py`` 的 ``start_chat_session_async`` 在任何提示符之前抛裸
# ``KeyError``——用户刚看到「知识快照加载成功」，紧接着会话终止。


def _chat_config_for(active_llm, active_rerank=None, rerank_enabled=False):
    from src.utils.config import ModelDetail

    detail = ModelDetail(provider="google", model_name="gemini-test")
    from src.utils.config import RetrievalMethod

    return {
        "retrieval_method": RetrievalMethod.SEMANTIC_SEARCH,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 4,
        "rerank_enabled": rerank_enabled,
        "active_rerank_configuration": active_rerank,
        "rerank_configurations": {"siliconflow": detail},
        "top_k": 5,
        "score_threshold": 0.0,
        "active_llm_configuration": active_llm,
        "llm_configurations": {"google": detail},
    }


def test_display_chat_config_reports_unknown_llm_key_as_value_error(capsys):
    console = Console(file=StringIO(), width=200)

    with pytest.raises(ValueError, match="未找到key"):
        display_chat_config(console, _chat_config_for("not-a-provider"))


def test_display_chat_config_reports_unknown_rerank_key_as_value_error():
    console = Console(file=StringIO(), width=200)
    config = _chat_config_for("google", active_rerank="openai", rerank_enabled=True)

    with pytest.raises(ValueError, match="未找到key"):
        display_chat_config(console, config)


def test_display_chat_config_renders_known_keys():
    """正常配置不受影响。"""
    console = Console(file=StringIO(), width=200)

    display_chat_config(console, _chat_config_for("google"))


# ── 回归：被激活的配置**取值**不可用时也必须报 ValueError ──
#
# ``_require_model_detail`` 只做 ``key not in configurations`` 的成员判定，然后
# 把 ``configurations[key]`` 原样返回。键在、值不可用（``None``／dict／int）时
# 不会走「未找到key」分支，``None`` 一路带到三个调用点的
# ``..._model_details.provider``，在那里抛裸 ``AttributeError``。
# 值来源与键同源，都是可直写的配置映射（``SessionConfig`` 的
# ``_VALIDATORS`` 明确不登记 ``*_configurations``）。


@pytest.mark.parametrize(
    "bad_detail", [None, 5, "x", {"provider": "google"}], ids=["None", "int", "str", "dict"]
)
def test_display_chat_config_rejects_unusable_active_llm_detail(bad_detail):
    console = Console(file=StringIO(), width=200)
    config = _chat_config_for("google")
    config["llm_configurations"] = {"google": bad_detail}

    with pytest.raises(ValueError, match="LLM配置"):
        display_chat_config(console, config)


@pytest.mark.parametrize(
    "bad_detail", [None, 5, "x", {"provider": "google"}], ids=["None", "int", "str", "dict"]
)
def test_display_chat_config_rejects_unusable_active_rerank_detail(bad_detail):
    console = Console(file=StringIO(), width=200)
    config = _chat_config_for("google", active_rerank="siliconflow", rerank_enabled=True)
    config["rerank_configurations"] = {"siliconflow": bad_detail}

    with pytest.raises(ValueError, match="Rerank配置"):
        display_chat_config(console, config)


def test_display_chat_config_still_renders_valid_details():
    """收紧不能误伤：合法的 ``ModelDetail`` 照常渲染。"""
    console = Console(file=StringIO(), width=200)
    config = _chat_config_for("google", active_rerank="siliconflow", rerank_enabled=True)

    display_chat_config(console, config)


# ── markup_text 的 NUL 模板与 _require_model_detail 的非映射形态 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3725 条仍全绿）。删除后行为确实改变（实测）：
# * 76／78 附近的 NUL 模板判据：``markup_text("a\x00b", "v")`` 删掉后不再报
#   「模板不得包含 NUL」，而是落到「占位符少于实参数量」——用户被指向占位符，
#   真正的问题（模板里混进了控制字符）没人说；
# * 297 ``_require_model_detail`` 的「configurations 非映射」判据：传 ``None``
#   时删掉后落到下游的「key 必须是可哈希的取值」，把「配置映射本身不是映射」
#   说成「key 形态非法」，指向错误的排查方向。
#
# NUL 那条与既有的「占位符数量不符」必须分开断言：两者消息都是 ``ValueError``，
# 只看异常类型无法区分。
def test_markup_text_rejects_nul_in_template():
    """模板里混入 NUL 字符必须点名 NUL，不能报成占位符数量不符。"""
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="NUL"):
        markup_text("a\x00b", "value")


def test_markup_text_still_rejects_placeholder_mismatch_with_its_own_message():
    """反向守卫：占位符数量不符仍是「占位符」消息，与 NUL 那条可区分。"""
    from src.ui.display_utils import markup_text

    with pytest.raises(ValueError, match="占位符") as excinfo:
        markup_text("没有占位符", "值")

    assert "NUL" not in str(excinfo.value)


def test_require_model_detail_rejects_non_mapping_configurations():
    """``configurations`` 不是映射时必须点名这一半，不能报成 key 形态非法。"""
    from src.ui.display_utils import _require_model_detail

    with pytest.raises(ValueError) as excinfo:
        _require_model_detail(None, "demo", "Rerank")

    message = str(excinfo.value)
    assert "未找到key" in message
    assert "必须是可哈希的取值" not in message


def test_require_model_detail_rejects_non_mapping_configurations_with_list():
    """列表也是非映射；``_require_model_detail`` 收空列表与 ``None`` 都要覆盖。"""
    from src.ui.display_utils import _require_model_detail

    with pytest.raises(ValueError, match="未找到key"):
        _require_model_detail(["not", "a", "mapping"], "demo", "LLM")


# ── 回归：配置面板的两张表都必须关掉列头（boolc 扫描存活点）──
#
# ``Table(..., show_header=False)`` 反向成 ``True`` 后 rich 会插入一行**空白
# 列头**——两张表的列都没有名字（只有样式），于是那一行只剩两边框之间的空白。
# 用户看到的是「聊天机器人配置」与第一条数据之间、以及「API密钥与URL配置」
# 与 ``QWEN_API_KEY`` 之间各多出一个空行。既有的用例只断言了脱敏与数值渲染，
# 从不检查表格结构，两处守卫因此同时无人看守。


def test_display_chat_config_tables_render_without_a_header_row(rendered):
    """两张配置表都不得渲染列头行（列没有名字，只会多出一行空白）。

    ``show_header`` 反向成 ``True`` 后 rich 在标题与第一条数据之间插入一行只有
    边框的空白行。断言「标题行的下一行就是第一条数据」，两张表各判一次。
    """
    lines = [line for line in rendered.splitlines() if line.strip()]
    chat_index = next(i for i, line in enumerate(lines) if "聊天机器人配置" in line)
    assert "检索策略:" in lines[chat_index + 1], (
        f"聊天配置表在标题后多出非数据行：{lines[chat_index + 1]!r}"
    )

    api_index = next(i for i, line in enumerate(lines) if "API密钥与URL配置" in line)
    assert "QWEN_API_KEY" in lines[api_index + 1], (
        f"API 配置表在标题后多出非数据行：{lines[api_index + 1]!r}"
    )
