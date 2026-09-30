import math
import os
import re
from collections.abc import Mapping
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils.config import (  # 导入 get_settings 函数
    ROOT_DIR,
    get_settings,
    is_double_convertible,
    require_detail_attributes,
    require_provider_name,
    require_rerank_enabled,
    require_retrieval_method,
    safe_repr,
)
from ..utils.log_manager import get_module_logger  # 导入日志管理器
from ..utils.security import redact_sensitive_text

logger = get_module_logger(__name__)  # 获取当前模块的日志器

# 全局UI宽度定义
CONSOLE_WIDTH = 120

# 模板里的占位符。用 NUL 而不是 ``{}`` 直接进 ``Text.from_markup``：NUL 不可能
# 出现在任何正常的模板字面量里，替换后也不会被 rich 的标记语法解释。
_MARKUP_PLACEHOLDER = "\x00"

# 与 ``rich.markup`` 的标记识别保持一致（``((\\*)\[([a-z#/@][^[]*?)])``），
# 用于判断占位符是否落在标记参数内部。
_MARKUP_TAG_RE = re.compile(r"((\\*)\[([a-z#/@][^[]*?)])")


def markup_text(template: str, *values: object) -> Text:
    """把不可信的值填进带标记的模板，得到一段安全的 ``Text``。

    ``console.print(str)``、``Table.add_row(str)``、``Panel(str)`` 都会把字符串
    按 rich **标记**解析（``Console(markup=True)`` 是默认值），而本项目的这些
    位置里插着知识库文档正文、用户输入与异常消息。直接用 f-string 拼有两类后果：

    1. **崩溃**。正文里出现字面量 ``[/]``、``[/color]``、``[/if]`` 时
       ``rich.markup.render`` 抛 ``MarkupError``。实测知识库文档里
       「匹配斜杠用 [/] 表示」这一行即可触发。真正致命的是异常处理走的也是同一
       个通道：``safe_exception_text()`` 拼出的消息里带着 ``[/]``，于是内层
       ``except`` 抛、``main()`` 的兜底 ``except`` 也抛、``run_cli`` 的兜底
       ``except`` 再抛——三层兜底全部失效，程序以裸 traceback 退出。
    2. **静默改写**。``[b]粗体[/b]`` 的标记被吃掉，markdown 链接
       ``[https://x](https://x)`` 丢掉方括号部分。实测真实知识库 3038 行里有
       12 行渲染后被改写——用户看到的与检索到的不一致。

    这里让**模板**继续走标记解析（样式照旧生效），而**值**通过 ``Text.append``
    原样追加，不经过标记解析。值会继承它在模板里所处位置的样式。

    ``{}`` 是占位符；模板与实参数量不符是编码错误，显式报错而不是错位拼接。

    Args:
        template: 含 ``{}`` 占位符的 rich 标记模板。
        *values: 按顺序填入占位符的不可信值。

    Returns:
        Text: 可直接交给 ``console.print`` / ``Table.add_row`` 的文本。
    """
    # 模板形态先判：本函数是**公开**的，而下面第一行就对 ``template`` 做成员测试。
    # 非字符串取值在那里抛的是裸 ``TypeError: argument of type 'int' is not
    # iterable``（``None``／``int``）、``expected string or bytes-like object``
    # （list／tuple／dict）或 ``a bytes-like object is required``（bytes）——三种
    # 消息都不指向「模板写错了」，也不是 ``ValueError``。而本函数对**值**参数早已
    # 用 ``_render_value`` 完整收敛、对占位符数量不符与 NUL 也都抛 ``ValueError``：
    # 同一个「入参形态非法」的处置，模板这一半此前漏了。
    if not isinstance(template, str):
        raise ValueError(f"标记模板必须是字符串，但得到 {safe_repr(template)}。")
    if _MARKUP_PLACEHOLDER in template:
        raise ValueError("标记模板不得包含 NUL 字符。")
    # 占位符落在标记**参数**内部（``[link={}]``）时，值会被并进标记语法本身：
    # ``[link=https://x]点这里[/link]`` 渲染成 ``点这里https://x``（链接文本与
    # URL 粘连、样式也丢了），``[bold {}]`` 更是直接抛 MarkupError。这类模板
    # 是编码错误——值要出现在**文本**位置，不能出现在标记参数里。
    for _match in _MARKUP_TAG_RE.finditer(template):
        # 奇数的前导反斜杠表示 rich 把 ``[`` 当作**字面量**（``\[example {}]``
        # 解析成 ``[example <值>]``），此时占位符不在任何标记参数内部，照常接受。
        # 正则的 ``(\\*)`` 捕获组拿到的正是这段反斜杠——按 rich ``_parse`` 的
        # 规则（``divmod(len(escapes), 2)[1]``）判奇偶。
        if len(_match.group(2)) % 2 == 1:
            continue
        if "{}" in _match.group(3):
            raise ValueError(
                f"标记模板的占位符不能出现在标记参数内部: {template!r}。 值只能放在标记的文本位置。"
            )

    filled = template
    for _ in values:
        if "{}" not in filled:
            raise ValueError(f"模板占位符少于实参数量: {template!r}")
        filled = filled.replace("{}", _MARKUP_PLACEHOLDER, 1)
    if "{}" in filled:
        raise ValueError(f"模板占位符多于实参数量: {template!r}")

    # 先把整段模板交给 rich 解析（标记照常生效），占位符作为 NUL 留在解析结果里，
    # 再按 NUL 的位置把值插回去。
    #
    # 不能切成若干段分别处理：``split`` 会把占位符那一个字符**丢掉**，而标记解析
    # 出的样式恰恰只挂在它身上（``[bold cyan]\x00[/bold cyan]`` 的 span 是
    # ``(0, 1, bold cyan)``）。切段后右侧是空串、没有样式，于是
    # ``[bold magenta]{}[/bold magenta]``——仓库里 13 处都是这个形状——渲染出来
    # 完全没有样式，且不报错。反向的错位同样会出：靠「已拼接结果末字符」兜底会
    # 让 ``[bold]x[/bold]{}`` 把值染成粗体，而值在标记之外。
    parsed = Text.from_markup(filled)
    placeholder_offsets = [
        index for index, character in enumerate(parsed.plain) if character == _MARKUP_PLACEHOLDER
    ]

    result = Text()
    cursor = 0
    for position, value in zip(placeholder_offsets, values):
        if position > cursor:
            # 切片会带上这一段的样式（``Text.divide`` 按 span 重新落位）。
            result.append_text(parsed[cursor:position])
        # 样式取自占位符**自身**覆盖的 span；``Text.stylize`` 只接受 str/Style，
        # 这里原样透传，不拼接、不合并——与 ``from_markup`` 的叠加方式一致。
        styles = [span.style for span in parsed.spans if span.start <= position < span.end]
        start = len(result)
        # 值经 ``_render_value`` 而不是裸 ``str()``：本函数的 ``values`` 是
        # ``object``，调用点既有字符串（``safe_exception_text``）也有 int／float
        # （``config_menu`` 的权重与倍率），而 ``str(10**4301)`` 撞上 CPython 的
        # ``sys.get_int_max_str_digits()``（默认 4300）抛
        # ``ValueError: Exceeds the limit``。后果与 ``_path_text`` 那处同源：
        # 正要显示「倍率已更新为 …」的那一步自己抛错，用户看到的是整数转换
        # 错误。``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫，
        # 嵌入方或携带超大整数的 pickle 快照可以直写这些字段（见
        # ``_require_finite_number`` 的 docstring）。
        result.append(_render_value(value))
        for style in styles:
            result.stylize(style, start, len(result))
        cursor = position + 1
    if cursor < len(parsed.plain):
        result.append_text(parsed[cursor:])
    return result


def _render_value(value: Any) -> str:
    """把任意取值渲染成可显示的字符串，位数超限时退回 ``safe_repr`` 的量级描述。

    ``str()`` 对位数超过 ``sys.get_int_max_str_digits()``（默认 4300）的整数抛
    ``ValueError``。本函数的用途是**显示**一个值，不是拿它做算术或路径运算，
    因此降级成量级描述是可接受的——比抛异常好，也比整段数字刷屏好。

    与 ``_path_text`` 的区别只在取舍：那里**不能**对正常取值用 ``safe_repr``
    （会加引号、会在 80 字符处截断，破坏路径换算），所以只在 ``str()`` 真的
    失败时才降级；这里同样只在失败时降级，保持正常取值的显示形态不变
    （``3`` 显示成 ``3`` 而不是 ``'3'``）。
    """
    try:
        return str(value)
    except (ValueError, RuntimeError, RecursionError):
        return safe_repr(value)


def _path_text(value: Any) -> str:
    """把路径取值收敛成 ``str``，位数超限时退回 ``safe_repr`` 的量级描述。

    ``str()`` 对位数超过 ``sys.get_int_max_str_digits()``（默认 4300）的整数抛
    ``ValueError``，而 ``get_relative_path`` 的兜底分支存在的唯一理由就是不再抛。

    **不能**直接对正常路径用 ``safe_repr``：它会加一对引号、并在 80 字符处截断，
    而本函数的用途是把绝对路径换算成相对路径——带引号的前缀永远匹配不到
    ``ROOT_DIR``，``os.path.relpath`` 于是原样返回那条绝对路径（相对路径功能
    完全失效）；截断更会让用户看不到文件名。因此只在 ``str()`` 真的失败时才
    降级，正常路径走原样的 ``str()``。
    """
    try:
        return str(value)
    except (ValueError, RuntimeError, RecursionError):
        return safe_repr(value)


def get_relative_path(absolute_path: str) -> str:
    """将绝对路径转换为相对于项目根目录的路径，方便显示。"""
    # ``absolute_path`` 的实际来源是 ``document_source(document)``，而文档元数据
    # 来自磁盘上的快照／legacy pkl（pickle 是二进制格式，不受 CPython 十进制位数
    # 限制）。实测传入 4301 位整数时 ``str()`` 抛 ``ValueError``，而下面
    # ``except ValueError`` 里的 f-string **用同一个值再插值一次**，把刚接住的
    # 异常原样重抛——兜底分支抛在展示检索结果的最后一环，会把一次成功的检索变成
    # 「LLM 异步生成出错」。因此取值、日志插值与返回值都经 ``_path_text``。
    text = _path_text(absolute_path)
    try:
        relative_path = os.path.relpath(text, str(ROOT_DIR))
        logger.debug(f"将绝对路径 '{text}' 转换为相对路径: '{relative_path}'")
        return relative_path
    except ValueError:
        # 如果路径不在同一驱动器上（例如Windows），则返回原始路径
        logger.warning(f"无法将路径 '{text}' 转换为相对路径，返回原始路径。")
        return text


def document_content(document: Any, default: str = "N/A") -> str:
    """取出文档内容，形态异常时退回 ``default``。

    ``display_results`` 与 ``Chatbot._display_retrieved_docs`` 都直接对检索结果做
    ``.replace(...)``／切片，而检索结果可以来自任何 ``VectorStoreBase`` 实现——
    ``FaissStore`` 会在载入期校验并归一化，但**第三方 store 不会**，而抽象基类
    正是为替换实现而存在的。实测一个返回 ``{"page_content": None}`` 的 store 能让
    两个展示函数分别抛 ``AttributeError`` 与 ``TypeError``。

    「显示检索结果」是错误处理之后的最后一环，它本身不该成为新的失败点。
    """
    if not isinstance(document, Mapping):
        return default
    content = document.get("page_content")
    return content if isinstance(content, str) else default


def document_source(document: Any, default: str = "N/A") -> str:
    """取出文档来源，形态异常时退回 ``default``。理由同 ``document_content``。"""
    if not isinstance(document, Mapping):
        return default
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return default
    source = metadata.get("source")
    return source if isinstance(source, str) else default


def coerce_score(value: Any, default: float = 0.0) -> float:
    """把分数字段收敛成**有限**浮点，形态异常时退回 ``default``。

    这是全项目唯一的分数收敛规则，展示层（``document_score``）与检索层
    （``retrieval_service``）共用。两处曾各自实现，判据还不一致：
    展示层拒绝数字字符串、检索层接受它，于是一份 ``score="0.8"`` 的第三方结果
    会被检索层按 0.8 通过分数阈值、却由界面显示成 ``0.0000``——用户看到
    「0.0000 排在最前面」，而它恰恰是通过了 0.5 阈值的那些。

    逐条规则与理由：

    * ``bool`` 退回默认值。``True`` 会算成 ``1.0``，看起来像一个真实得分，而它
      几乎必然是上游把标志位写进了分数字段。
    * 数字字符串**照旧转换**。旧实现是 ``float(x or 0)``，接受 ``"0.5"``；一律
      拒绝会让返回字符串得分的 store 被静默降为 0，再被分数阈值全部过滤——
      从「能用」变成「静默返回空结果」。
    * ``nan``／``inf`` 退回默认值。``nan`` 不抛错却会静默破坏排序：融合里
      ``max()``/``min()`` 遇到它双双返回 ``nan``，每个文档的归一化结果都成了
      ``nan``，而 ``sorted`` 对 ``nan`` 的比较恒为 False，该文档保持原相对顺序、
      **留在首位**。``inf`` 同理，``max()`` 取到它之后其余文档全被压成 0。
    * 映射、列表等其它类型退回默认值——它们转换不了，且在展示层格式化时会抛错。
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        # ``float(10**400)`` 抛 ``OverflowError``（**不是** ``ValueError``），
        # 而这个函数承诺「绝不抛异常、退回默认值」。它是展示层与检索层共用的
        # 唯一分数收敛规则，裸异常会让「形态异常就退回默认值」这条契约失效。
        if not is_double_convertible(value):
            return default
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return default
    else:
        return default
    return number if math.isfinite(number) else default


def document_score(document: Any, default: float = 0.0) -> float:
    """取出文档得分，形态异常时退回 ``default``（规则见 ``coerce_score``）。"""
    if not isinstance(document, Mapping):
        return default
    return coerce_score(document.get("score"), default)


def _require_model_detail(configurations: Any, key: Any, role_label: str) -> Any:
    """按 key 取模型详情，缺失或取值不可用时抛带指引的 ``ValueError``。

    原先直接写 ``configurations[key]``，而 ``default_llm_provider``／
    ``default_rerank_provider``／``active_*_configuration`` **没有任何
    validator**（``Settings`` 里只是普通 ``str``，``SessionConfig`` 的
    ``_VALIDATORS`` 也没登记它们）。实测 ``default_llm_provider =
    "not-a-provider"``（打错字）时，调用点 ``chat/core.py`` 的
    ``start_chat_session_async`` 在任何提示符之前就抛裸 ``KeyError``——用户刚
    看到「知识快照加载成功」，紧接着会话终止。同一个 key 走
    ``ModelProviderFactory._get_model_detail`` 得到的是
    ``ValueError("在…配置中未找到key: …")``，错误分类本就该一致。

    成员判定只证明**键**在，不证明**值**可用：``{"k": None}`` 的键确实存在，
    于是不会走上面那条分支，``None`` 被原样返回给三个调用点，而它们紧接着都读
    ``..._model_details.provider``／``.model_name``——实测
    ``chat_config["rerank_configurations"] = {"siliconflow": None}`` 时在此抛裸
    ``AttributeError``（不是 ``ValueError``）。取值来源与键同源，都是可直写的
    配置映射，因此在这里一并收敛。
    """
    if not isinstance(configurations, Mapping) or key not in configurations:
        available = (
            ", ".join(sorted(configurations)) if isinstance(configurations, Mapping) else "无"
        )
        raise ValueError(
            f"在{role_label}配置中未找到key: {safe_repr(key)}。可选值: {available}。"
            " 请检查 config.toml 里的 default_*_provider 与 *_configurations 是否对应。"
        )
    detail = configurations[key]
    require_detail_attributes(
        detail,
        ("provider", "model_name"),
        field=f"{role_label}配置",
        key=key,
        role_label=role_label,
    )
    return detail


def display_chat_config(console: Console, chat_config: dict[str, Any]):
    """显示从传入的chat_config字典中获取的聊天机器人配置。"""
    logger.info("正在显示聊天机器人配置。")

    def mask_api_key(key: str | None) -> str:
        if not key:
            logger.debug("API Key 未设置，显示为 '[dim]未设置[/dim]'。")
            return "[dim]未设置[/dim]"
        logger.debug("API Key 已设置，隐藏全部值。")
        return "[dim]已设置（值已隐藏）[/dim]"

    # --- 聊天配置 ---
    chat_table = Table(
        title="[bold green]聊天机器人配置[/bold green]", show_header=False, box=None, padding=(0, 1)
    )
    chat_table.add_column(style="cyan")
    chat_table.add_column(style="bold white")
    # 这一整张表的**单元格**都来自 ``config.toml``，而表格单元格与
    # ``console.print`` 一样按标记解析。其中 ``model_name`` 与配置键名
    # （``active_llm_configuration`` 等）都没有白名单：工厂只校验 ``provider``
    # （``_provider_map`` 白名单），其余原样透传。实测 ``model_name = "gemini[/]"``
    # 时 ``reload_llm()`` 返回 True（模型确实建起来了），紧接着本函数抛
    # ``MarkupError``——聊天会话在启动横幅这一步就崩。
    #
    # 值一律经 ``Text()`` 原样呈现，样式由 ``add_column(style=...)`` 提供。
    # 直读 ``.value`` 会在可直写的非枚举取值上抛裸 ``AttributeError``，而本函数在
    # ``start_chat_session_async`` 里于任何提示符**之前**被调用——会话在启动横幅
    # 这一步就终止。与同一张表的其它单元格同因（普通属性没有 ``__setattr__`` 守卫）。
    chat_table.add_row(
        "检索策略:",
        Text(require_retrieval_method(chat_config["retrieval_method"]).value),
    )
    chat_table.add_row(
        "混合搜索权重 (向量/关键词):",
        Text(
            f"{_render_value(chat_config['vector_weight'])}"
            f" / {_render_value(chat_config['keyword_weight'])}"
        ),
    )
    # 与同一张表的其它单元格一致走 ``_render_value``：裸 ``str()`` 对位数超限的
    # 整数抛 ``ValueError: Exceeds the limit``，聊天会话在启动横幅这一步就终止。
    chat_table.add_row(
        "混合融合策略:", Text(_render_value(chat_config["hybrid_fusion_strategy"]).upper())
    )
    chat_table.add_row(
        "候选过量招募倍率:", Text(_render_value(chat_config["retrieval_candidate_multiplier"]))
    )
    # ``rerank_enabled`` 与同一张表的 ``retrieval_method`` 同因：``SessionConfig``
    # 是公开类、普通属性没有 ``__setattr__`` 守卫，直写 ``"false"``／``"no"``／``0``
    # 之后真值判断会给出与服务层**相反**的结论——实测直写 ``"false"`` 时这张表显示
    # 「启用」并渲染 Rerank 模型段落，而 ``RetrievalService._coerce_rerank_enabled``
    # 对同一个值直接报错。写入侧（``_validate_rerank_enabled``）只认 ``bool``，
    # 读取侧必须同判据。
    rerank_enabled = require_rerank_enabled(chat_config["rerank_enabled"])
    chat_table.add_row(
        "Rerank重排:",
        markup_text("[bold green]启用[/bold green]")
        if rerank_enabled
        else markup_text("[bold red]禁用[/bold red]"),
    )
    if rerank_enabled:
        active_rerank = chat_config["active_rerank_configuration"]
        rerank_model_details = _require_model_detail(
            chat_config["rerank_configurations"], active_rerank, "Rerank"
        )
        # 这一格与同一张表的权重／top_k 同因（见下方注释）：``active_*`` 与
        # ``model_name`` 都可被直写，裸 f-string 对位数超限的整数抛
        # ``ValueError: Exceeds the limit``；``provider`` 还可能是非字符串，
        # 在 ``provider.lower()`` 处抛裸 ``AttributeError``。全部经
        # ``_render_value`` / ``require_provider_name`` 收敛。
        rerank_provider = require_provider_name(
            rerank_model_details.provider, label="Rerank provider"
        )
        chat_table.add_row(
            "激活的Rerank模型:",
            Text(
                f"{_render_value(active_rerank)} "
                f"({rerank_provider}: {_render_value(rerank_model_details.model_name)})"
            ),
        )
        # 惰性 ``%s`` 而不是 f-string：后者在 DEBUG 关闭时也**已经求值**，
        # 裸插值对位数超限的整数照样抛 ``ValueError``。
        logger.debug("Rerank模型已启用，激活模型: %s", active_rerank)
    else:
        logger.debug("Rerank模型未启用。")
    # 这三个字段与上面的权重同理：``SessionConfig`` 是公开类、普通属性没有
    # ``__setattr__`` 守卫，嵌入方或携带超大整数的 pickle 快照可以直写，
    # 而裸 ``str()`` 对位数超限的整数抛 ``ValueError: Exceeds the limit``。
    # 本函数在 ``start_chat_session_async`` 里于任何提示符**之前**被调用，
    # 于是会话在「知识快照加载成功」之后立刻终止，用户看到的是整数转换错误。
    chat_table.add_row("返回文档数 (top_k):", Text(_render_value(chat_config["top_k"])))
    chat_table.add_row("语义搜索阈值:", Text(_render_value(chat_config["score_threshold"])))
    active_llm = chat_config["active_llm_configuration"]
    llm_model_details = _require_model_detail(chat_config["llm_configurations"], active_llm, "LLM")
    llm_provider = require_provider_name(llm_model_details.provider, label="LLM provider")
    chat_table.add_row(
        "激活的LLM:",
        Text(
            f"{_render_value(active_llm)} "
            f"({llm_provider}: {_render_value(llm_model_details.model_name)})"
        ),
    )
    logger.debug("激活的LLM: %s", active_llm)

    # --- API与URL配置 ---
    # API密钥和URL仍然从全局的、不可变的settings对象中读取，因为它们不应该在运行时被修改。
    api_table = Table(
        title="[bold green]API密钥与URL配置[/bold green]",
        show_header=False,
        box=None,
        padding=(0, 1),
    )
    api_table.add_column(style="cyan")
    api_table.add_column(style="bold white")
    # 仅显示与激活的LLM和Rerank相关的配置
    active_providers = {llm_model_details.provider}
    if rerank_enabled:
        # 与上面那处同源：这里是同一个裸索引的第二份拷贝，上面改了这里也要改，
        # 否则「rerank 配置缺失」只在表格渲染阶段被拦住、到了 API Key 段落又
        # 抛一次裸 ``KeyError``。
        active_providers.add(
            _require_model_detail(
                chat_config["rerank_configurations"],
                chat_config["active_rerank_configuration"],
                "Rerank",
            ).provider
        )

    current_settings = get_settings()  # 获取当前配置
    for provider_name in active_providers:
        key_name = f"{provider_name.lower()}_api_key"
        api_key = getattr(current_settings, key_name, None)
        if api_key:
            # 行标签由 provider 名拼出。provider 有 ``_provider_map`` 白名单，
            # 但仍按字面呈现——标签里的 ``[`` 会被解析。
            api_table.add_row(Text(f"{key_name.upper()}:"), mask_api_key(api_key))
            logger.debug("显示提供商 '%s' 的 API Key。", provider_name)
        else:
            logger.debug("提供商 '%s' 的 API Key 未设置。", provider_name)

        url_key_name = f"{provider_name.lower()}_base_url"
        api_url = getattr(current_settings, url_key_name, None)
        if api_url:
            # Base URL 可以内嵌凭证（``https://user:token@gateway/v1``）——
            # 这是自建网关的常见写法，而开发者文档也建议把 key 放进 URL。
            # 日志侧由 RedactingFormatter 兜底，终端这一路没有，之前会把
            # 完整凭证打到屏幕上（也会留在终端回滚与截图里）。这里先脱敏再显示。
            # Base URL 来自 config.toml／.env，未经校验——IPv6 字面量本身就带
            # 方括号（``http://[::1]:8080/v1``），含 ``[/]`` 形态的取值会让表格
            # 渲染抛 MarkupError。脱敏解决的是凭证泄漏，不解决标记解析。
            api_table.add_row(
                Text(f"{url_key_name.upper()}:"), Text(redact_sensitive_text(str(api_url)))
            )
            logger.debug("显示提供商 '%s' 的 Base URL。", provider_name)
        else:
            logger.debug("提供商 '%s' 的 Base URL 未设置。", provider_name)

    console.print(
        Panel(
            Group(chat_table, api_table),
            title="[bold yellow]当前聊天会话配置[/bold yellow]",
            border_style="blue",
            width=CONSOLE_WIDTH,
        )
    )
    logger.info("聊天机器人配置显示完成。")
