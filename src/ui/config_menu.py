import copy
from collections.abc import Callable, Mapping
from typing import Any

import questionary
from rich.panel import Panel

from ..utils.config import (
    _MAX_RETRIEVAL_CANDIDATES,
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    RetrievalMethod,
    parse_decimal_int,
    require_rerank_enabled,
    require_retrieval_method,
    require_scale_field,
    safe_repr,
)
from ..utils.log_manager import get_module_logger  # 导入日志管理器
from .display_utils import _render_value, create_console, display_chat_config, markup_text
from .terminal_theme import questionary_style_for_theme

logger = get_module_logger(__name__)  # 获取当前模块的日志器
console = create_console()


def _rerank_menu_label(chat_config: dict[str, Any]) -> str:
    """返回菜单里的 Rerank 状态文本；取值非法时显示「未知」。

    读取必须与写入侧同判据（``_validate_rerank_enabled`` 只认 ``bool``）：真值
    判断会把直写的 ``"false"`` 显示成「启用」，与服务层的
    ``_coerce_rerank_enabled`` 给出相反结论。

    但这里**不能**抛 ``ValueError``——``edit_retrieval_params`` 由
    ``launch_config_editor`` 经 ``asyncio.to_thread`` 调用，调用点只接
    ``(KeyboardInterrupt, EOFError)``，异常会终止整个会话。而菜单是用户唯一
    还能改配置的入口（``display_chat_config`` 在上游会直接拒绝）。非法取值
    显示「未知」，用户仍能进菜单把它改回布尔。
    """
    try:
        enabled = require_rerank_enabled(chat_config["rerank_enabled"])
    except ValueError:
        return "未知"
    return "启用" if enabled else "禁用"


def _rerank_default(chat_config: dict[str, Any]) -> bool | None:
    """读 ``rerank_enabled`` 作为 ``confirm`` 的默认项；取值非法时返回 ``None``。

    ``questionary.confirm`` 在用户直接回车时把 ``default`` **原样**返回
    （``prompt_toolkit`` 的 ``set_answer`` 分支），因此 ``default="false"`` 会让
    回车得到字符串 ``"false"``——写回配置后服务层报错，而界面上明明显示「是」。
    ``None`` 是 questionary 的合法默认（不预选），菜单照常可用。
    """
    try:
        return require_rerank_enabled(chat_config["rerank_enabled"])
    except ValueError:
        return None


def _top_k_ceiling(multiplier: int) -> int:
    """返回当前倍率下 ``top_k`` 可用的上界；当前倍率非法时回落到单字段上界。

    真正送进 ``faiss_index.search()`` 的是 ``top_k × multiplier``（FAISS 既不
    报错也不截断），因此单字段上界不够——UI 放行一个组合越界的值，会在应用
    配置时抛 ``ValueError`` 并终止会话。
    """
    # 收敛后再算：``multiplier`` 是可直写字段，非法取值让 ``require_scale_field``
    # 抛 ``ValueError``，而菜单在主循环的 ``try`` 之外——内层只接
    # ``(KeyboardInterrupt, EOFError)``、外层只有 ``finally``，抛出去会以裸
    # traceback 终止整个会话（实测「倍率被直写成 ``"abc"``、用户选『调整
    # Top K』」即复现）。回落到单字段上界：上界放得比「组合判据」宽只会把
    # 校验推给应用配置时的 ``SessionConfig`` 守卫（那里会拒绝并提示），菜单
    # 仍可用；直接炸掉则用户连改回合法值的机会都没有。
    try:
        multiplier = require_scale_field(multiplier, "retrieval_candidate_multiplier")
    except ValueError:
        return _MAX_RETRIEVAL_TOP_K
    return min(_MAX_RETRIEVAL_TOP_K, _MAX_RETRIEVAL_CANDIDATES // max(1, multiplier))


def _multiplier_ceiling(top_k: int) -> int:
    """返回当前 ``top_k`` 下候选倍率可用的上界；当前 ``top_k`` 非法时同上回落。"""
    # 与 ``_top_k_ceiling`` 同因：``top_k`` 同样可直写。
    try:
        top_k = require_scale_field(top_k, "top_k")
    except ValueError:
        return _MAX_RETRIEVAL_MULTIPLIER
    return min(_MAX_RETRIEVAL_MULTIPLIER, _MAX_RETRIEVAL_CANDIDATES // max(1, top_k))


def _bounded_int_validator(ceiling: int) -> Callable[[str], bool]:
    """构造 questionary 的整数输入校验器，接受 ``1`` 到 ``ceiling``。

    解析一律走 ``parse_decimal_int``，**校验器本身不抛任何异常**。这一点是硬要求：
    questionary 的 ``build_validator`` 只把 ``ValidationError`` 转成「输入无效」
    提示，其余异常会一路冒到 prompt_toolkit 的事件循环之外——用户按下回车后看到
    ``Unhandled exception in event loop`` 加一段 traceback，输入框卡死。实测两种
    输入都会造成这种收场：``²``（``isdigit()`` 为真但 ``int()`` 抛错）与
    ``"9" * 4301``（合法的十进制整数，但位数超过 ``sys.get_int_max_str_digits()``）。
    """

    def validate(text: str) -> bool:
        parsed = parse_decimal_int(text)
        return parsed is not None and 1 <= parsed <= ceiling

    return validate


def edit_retrieval_params(chat_config: dict[str, Any]) -> None:
    """编辑检索相关参数。"""
    logger.info("进入检索参数编辑菜单。")
    while True:
        # 与 ``display_chat_config`` 同因：直读 ``.value`` 对可直写的非枚举取值抛裸
        # ``AttributeError``，菜单渲染不出来。收敛后 ``default=`` 也拿到规范取值。
        # ``require_retrieval_method`` 对**非法取值**抛 ``ValueError``（在
        # ``KeyboardInterrupt/EOFError`` 之外），会让整个编辑器会话终止——与
        # ``_rerank_menu_label`` 同一契约：菜单是用户唯一还能改回合法值的入口，
        # 不能把它炸掉。非法取值显示「未知」，questionary 拿不到 ``default``
        # 时传 ``None``（不预选，仍是合法参数）。
        try:
            current_method = require_retrieval_method(chat_config["retrieval_method"]).value
        except ValueError:
            current_method = None
        # ``None`` 直接进 f-string 会渲染成 ``当前: None``——与下一项
        # ``_rerank_menu_label`` 对同一类非法取值刻意显示的「未知」不一致。
        # 菜单文案统一为「未知」，用户仍能看出这一项有问题并把它改回来。
        if current_method is None:
            current_method = "未知"

        current_top_k = chat_config["top_k"]
        # 与 ``display_chat_config`` 同因：真值判断对可直写的取值会给出与服务层
        # 相反的结论（``"false"`` 显示成「启用」）。菜单侧不抛错，非法取值显示
        # 「未知」——见 ``_rerank_menu_label``。
        current_rerank = _rerank_menu_label(chat_config)
        # 权重同样要经 ``safe_repr``：``SessionConfig`` 是公开类、普通属性没有
        # ``__setattr__`` 守卫，嵌入方或携带超大整数的 pickle 快照可以直写
        # ``cfg.vector_weight = 10**4301``，随后这里与下面三处的裸插值都会抛
        # ``ValueError: Exceeds the limit``——菜单与提示串根本渲染不出来。
        current_weights = (
            f"向量: {safe_repr(chat_config['vector_weight'])}"
            f" / 关键词: {safe_repr(chat_config['keyword_weight'])}"
        )
        # 这里**不能**用 ``safe_repr``：它给字符串加引号（``'rrf'`` 而非 ``rrf``），
        # 会改掉正常取值的显示形态。``_render_value`` 才是同一张表在
        # ``display_utils`` 里用的渲染器——先 ``str()``，只在撞上整数位数上限时
        # 降级成量级描述。``SessionConfig`` 的普通属性没有 ``__setattr__`` 守卫，
        # 嵌入方或携带超大整数的 pickle 快照可以直写
        # ``cfg.hybrid_fusion_strategy = 10**4301``（或 ``["RRF"]``／``None``）：
        # 裸 ``.upper()`` 对非字符串抛 ``AttributeError``，冲穿聊天主循环的
        # ``except ValueError`` 分类；``str(10**4301)`` 则抛 ``ValueError:
        # Exceeds the limit``，菜单根本渲染不出来。
        raw_fusion_strategy = _render_value(chat_config["hybrid_fusion_strategy"])
        # 非法取值（直写 ``None``/非字符串渲染成量级描述）与 method/rerank 项
        # 统一为「未知」——``NONE`` 这类大写形态对用户没有指向性。合法融合
        # 策略是 ``rrf``／``weighted``（判据与 ``SessionConfig._validate_fusion_strategy``
        # 及 questionary Choice value 一致——此前误写 ``weight``，合法默认
        # ``weighted`` 被显示成「未知」，实测复现），二者照常显示。
        current_fusion_strategy = (
            raw_fusion_strategy if raw_fusion_strategy.lower() in ("rrf", "weighted") else "未知"
        )
        current_candidate_multiplier = chat_config["retrieval_candidate_multiplier"]

        choice = questionary.select(
            "选择要调整的检索参数:",
            choices=[
                questionary.Choice(f"1. 检索模式 (当前: {current_method})", value="method"),
                questionary.Choice(
                    f"2. 检索数量 Top K (当前: {safe_repr(current_top_k)})", value="top_k"
                ),
                questionary.Choice(f"3. Rerank重排 (当前: {current_rerank})", value="rerank"),
                questionary.Choice(f"4. 混合搜索权重 (当前: {current_weights})", value="weights"),
                questionary.Choice(
                    f"5. 混合融合策略 (当前: {current_fusion_strategy})", value="fusion"
                ),
                questionary.Choice(
                    f"6. 候选过量招募倍率 (当前: {safe_repr(current_candidate_multiplier)})",
                    value="candidate_multiplier",
                ),
                questionary.Separator(),
                questionary.Choice("返回主菜单", value="back"),
            ],
            style=questionary_style_for_theme(),
        ).ask()

        if choice == "back" or choice is None:
            logger.info("退出检索参数编辑菜单。")
            break

        if choice == "method":
            # ``default`` 必须是候选集合内的值：非法 ``retrieval_method`` 时
            # ``current_method`` 是「未知」占位串，直接传给 questionary 会在
            # **构造期**抛 ``ValueError: Invalid 'default' value``（实测），
            # 整个编辑器退出——method 分支是唯一能写 ``retrieval_method`` 的
            # 入口，恰是该字段唯一还能被修回合法值的路径。与
            # ``_normalized_fusion_default`` 同构：不在候选集合时传 ``None``
            # （不预选，仍是合法参数）。
            method_choices = [
                questionary.Choice("全文检索", value=RetrievalMethod.FULL_TEXT_SEARCH.value),
                questionary.Choice("向量检索", value=RetrievalMethod.SEMANTIC_SEARCH.value),
                questionary.Choice(
                    "混合检索 (全文+向量)", value=RetrievalMethod.HYBRID_SEARCH.value
                ),
            ]
            method_values = {item.value for item in method_choices}
            new_method = questionary.select(
                "选择新的检索模式:",
                choices=method_choices,
                default=current_method if current_method in method_values else None,
                style=questionary_style_for_theme(),
            ).ask()
            if new_method:
                chat_config["retrieval_method"] = RetrievalMethod(new_method)
                console.print(markup_text("[green]检索模式已更新为: {}[/green]", new_method))
                logger.info(f"检索模式已更新为: {new_method}")

        elif choice == "top_k":
            # 上界必须与 SessionConfig/Settings 的守卫一致，且要连**组合**一起判：
            # 真正送进 faiss_index.search() 的是 top_k × multiplier（FAISS 既不
            # 报错也不截断）。这里只按单字段上界放行的话，用户设成 2000 会在
            # 应用配置时抛 ValueError——而那个异常落在 chat 主循环之外，会以裸
            # traceback 终止整个会话。所以校验里把当前倍率算进去。
            top_k_ceiling = _top_k_ceiling(chat_config["retrieval_candidate_multiplier"])
            new_top_k = questionary.text(
                f"输入新的Top K值 (1-{safe_repr(top_k_ceiling)}, 当前: {safe_repr(current_top_k)}):",
                validate=_bounded_int_validator(top_k_ceiling),
                # ``str`` 而不是裸取值：位数超限的整数连 ``str()`` 都抛，而
                # ``default=`` 的取值同样要经过渲染。
                default=safe_repr(current_top_k),
            ).ask()
            if new_top_k:
                chat_config["top_k"] = parse_decimal_int(new_top_k)
                console.print(markup_text("[green]Top K 已更新为: {}[/green]", new_top_k))
                logger.info(f"Top K 已更新为: {new_top_k}")

        elif choice == "rerank":
            # 同一收敛也用于 ``default=``，见 ``_rerank_default``。
            #
            # ``default=None`` 是 questionary 的**合法**取值，只是它的类型标注
            # （``default: bool = True``）没写出来：``prompt_toolkit`` 的
            # ``set_answer`` 分支是 ``if status["answer"] is None: status["answer"]
            # = default``——``None`` 正是它内部的「尚未作答」哨兵，用户直接回车
            # 时返回 ``None``，下面的 ``if new_rerank is not None`` 因而跳过写回。
            # 实测 ``questionary.confirm("x", default=None)`` 正常构造。标注与行为
            # 不符，故在此显式忽略。
            new_rerank = questionary.confirm(
                "是否启用Rerank重排?",
                default=_rerank_default(chat_config),  # type: ignore[arg-type]
            ).ask()
            if new_rerank is not None:
                chat_config["rerank_enabled"] = new_rerank
                console.print(
                    markup_text(
                        "[green]Rerank已更新为: {}[/green]", "启用" if new_rerank else "禁用"
                    )
                )
                logger.info(f"Rerank已更新为: {'启用' if new_rerank else '禁用'}")

        elif choice == "weights":
            if chat_config["retrieval_method"] != RetrievalMethod.HYBRID_SEARCH:
                console.print("[yellow]警告: 权重调整仅在 '混合检索' 模式下生效。[/yellow]")
                logger.warning("尝试调整权重，但当前检索模式不是混合检索。")

            def is_float_between_0_and_1(text):
                try:
                    val = float(text)
                    return 0.0 <= val <= 1.0
                except ValueError:
                    return False

            new_vector_weight_str = questionary.text(
                f"输入新的向量权重 (0.0-1.0, 当前: {safe_repr(chat_config['vector_weight'])}):",
                validate=is_float_between_0_and_1,
                default=safe_repr(chat_config["vector_weight"]),
            ).ask()

            if new_vector_weight_str:
                new_vector_weight = float(new_vector_weight_str)
                new_keyword_weight = round(1.0 - new_vector_weight, 2)
                # 两条写回有**顺序陷阱**：``SessionConfig``/编辑器共享的组合守卫
                # 「vector_weight 与 keyword_weight 不能同时为 0」按中间态判，
                # 固定先写 vector 会拒绝合法终态——默认 (1.0, 0.0) 下用户把
                # 向量权重改成 0（切纯关键词检索）实测报「不能同时为 0」，而
                # 反序写通过。先写**非零侧**使中间态永远不落入双 0：
                # (v=0, k=1) 先写 keyword、(v=1, k=0) 先写 vector，其余两侧
                # 均非 0 顺序无关。
                if new_vector_weight == 0.0:
                    chat_config["keyword_weight"] = new_keyword_weight
                    chat_config["vector_weight"] = new_vector_weight
                else:
                    chat_config["vector_weight"] = new_vector_weight
                    chat_config["keyword_weight"] = new_keyword_weight
                console.print(
                    markup_text(
                        "[green]混合搜索权重已更新为 -> 向量: {}, 关键词: {}[/green]",
                        chat_config["vector_weight"],
                        chat_config["keyword_weight"],
                    )
                )
                logger.info(
                    "混合搜索权重已更新为 -> 向量: %s, 关键词: %s",
                    safe_repr(chat_config["vector_weight"]),
                    safe_repr(chat_config["keyword_weight"]),
                )

        elif choice == "fusion":
            if chat_config["retrieval_method"] != RetrievalMethod.HYBRID_SEARCH:
                console.print("[yellow]警告: 融合策略仅在 '混合检索' 模式下生效。[/yellow]")
                logger.warning("尝试调整融合策略，但当前检索模式不是混合检索。")

            new_strategy = questionary.select(
                "选择新的混合融合策略:",
                choices=[
                    questionary.Choice("RRF (倒数排名融合)", value="rrf"),
                    questionary.Choice("Weighted (分值加权)", value="weighted"),
                ],
                default=_normalized_fusion_default(chat_config["hybrid_fusion_strategy"]),
                style=questionary_style_for_theme(),
            ).ask()
            if new_strategy:
                chat_config["hybrid_fusion_strategy"] = new_strategy
                console.print(
                    markup_text("[green]混合融合策略已更新为: {}[/green]", new_strategy.upper())
                )
                logger.info(f"混合融合策略已更新为: {new_strategy.upper()}")

        elif choice == "candidate_multiplier":
            # 同 top_k：上界要连当前 top_k 一起算，否则「两个字段各自合法、
            # 乘积越界」会在应用配置时抛错并终止会话。
            multiplier_ceiling = _multiplier_ceiling(chat_config["top_k"])
            new_multiplier = questionary.text(
                f"输入候选过量招募倍率 (1-{safe_repr(multiplier_ceiling)}, 当前: {safe_repr(current_candidate_multiplier)}):",
                validate=_bounded_int_validator(multiplier_ceiling),
                # 与同函数 top_k 分支的 ``default=safe_repr(current_top_k)`` 保持
                # 一致：``str()`` 对位数超限的整数抛 ``ValueError: Exceeds the
                # limit``，菜单在这一步直接报错；``safe_repr`` 退回量级描述，
                # 校验器随后会把它作为非法输入拒掉，用户仍能改。
                default=safe_repr(current_candidate_multiplier),
            ).ask()
            if new_multiplier:
                chat_config["retrieval_candidate_multiplier"] = parse_decimal_int(new_multiplier)
                console.print(
                    markup_text("[green]候选过量招募倍率已更新为: {}[/green]", new_multiplier)
                )
                logger.info(f"候选过量招募倍率已更新为: {new_multiplier}")
    # 检索参数的更改不需要重载任何模型


def _configuration_keys(configurations: Any, role_label: str) -> list[Any]:
    """取配置映射的键列表；取值不是映射时给出带指引的 ``ValueError``。

    ``SessionConfig`` 的 ``_VALIDATORS`` 明确不登记 ``*_configurations``
    （「按原样写入，由使用方负责其内部结构」），嵌入方可以把
    ``chat_config["llm_configurations"]`` 直写成 ``5``。原先直接写
    ``list(chat_config["llm_configurations"].keys())``，在那里抛裸
    ``AttributeError: 'int' object has no attribute 'keys'``——异常类型与
    「配置项形态错了」无关，且不是 ``ValueError``，会以裸 traceback 终止会话。
    判据与 ``display_utils._require_model_detail`` 一致。

    **只取键、不校验值**：菜单要把「值形态坏了」的配置项也列出来，否则用户
    没有入口去切换掉它（`display_chat_config` 在上游读值时会直接拒绝，菜单
    是唯一还能改配置的地方）。值的形态由使用方在真正读取时判。
    """
    if not isinstance(configurations, Mapping):
        raise ValueError(
            f"{role_label}配置必须是映射，但得到 {type(configurations).__name__}。"
            " 请检查 config.toml 里的 *_configurations 配置项。"
        )
    return list(configurations.keys())


def _configuration_choices(configurations: Any, role_label: str) -> list[questionary.Choice]:
    """把配置映射的键列表包成 questionary 候选。

    与 ``_configuration_keys`` 同一处取值、同一取舍（只判容器、不判取值），
    但**必须**包成 ``Choice``：``questionary.select`` 对每个候选调
    ``choice.get(...)``，只有 ``Choice`` 对象有该方法，原始字符串之所以能工作
    是因为 questionary 内部对 ``str`` 有特例——非字符串键（``int``／``tuple``／
    ``None``）一律抛裸 ``AttributeError: 'int' object has no attribute 'get'``。
    而 ``SessionConfig._VALIDATORS`` 明确不登记 ``*_configurations``，嵌入方或
    携带异常取值的 pickle 快照可以放进任意可哈希键。

    标题走 ``_render_value``：位数超限的整数键连 ``str()`` 都抛
    ``ValueError: Exceeds the limit``，菜单根本渲染不出来。``value`` 保留**原始
    键**——切换时写回的必须是它，``default=`` 也要拿原始键去候选集合里对。

    候选为空时 questionary 自己抛 ``ValueError: A list of choices needs to be
    provided.``——那是「配置里一个模型都没有」的真实情况，交给它报出比在这里
    编一个假选项更诚实。
    """
    return [
        questionary.Choice(title=_render_value(key), value=key)
        for key in _configuration_keys(configurations, role_label)
    ]


def _choice_default(choices: list[questionary.Choice], active: Any) -> Any:
    """返回可交给 ``questionary.select(default=...)`` 的取值。

    ``default`` 必须**存在于候选集合里**，否则 questionary 抛
    ``ValueError: Invalid `default` value passed``——激活项被删掉、或它本身是
    位数超限的整数时都会命中，而菜单是用户唯一还能改配置的入口。不在候选里就
    退回 ``None``：那是 questionary 内部的「尚未作答」哨兵（见
    ``edit_retrieval_params`` 里 ``default=None`` 的说明），合法且不抛。
    """
    for choice in choices:
        if choice.value == active:
            return active
    return None


def _normalized_fusion_default(value: Any) -> str | None:
    """把融合策略收敛成候选集合里的字面量，收敛不了时返回 ``None``。

    候选只有 ``"rrf"``／``"weighted"`` 两个字面量，而 ``default=`` 拿的是
    ``chat_config["hybrid_fusion_strategy"]`` 的**原始**取值——``SessionConfig``
    的普通属性没有 ``__setattr__`` 守卫，嵌入方或携带异常取值的 pickle 快照可以
    直写该字段。实测三种收场都不在契约内：``"RRF"``／``" rrf "``（大小写／空白
    未归一）与 ``1`` 让 questionary 抛 ``ValueError: Invalid 'default' value
    passed``，位数超限的整数则抛 ``ValueError: Exceeds the limit``。

    归一化口径必须与运行期一致，取 ``_coerce_fusion_strategy``（它同时服务
    ``RetrievalService`` 的读取路径，把 ``"RRF"``／``" rrf "`` 都收敛成
    ``"rrf"``）。收敛后仍不在候选集合里（例如 ``"bogus"``）就退回 ``None``：
    questionary 的「未作答」哨兵，合法且不抛，用户仍能选一个合法值。
    """
    from ..services.retrieval_service import _coerce_fusion_strategy

    try:
        normalized = _coerce_fusion_strategy(value)
    except ValueError:
        return None
    return normalized if normalized in {"rrf", "weighted"} else None


def edit_model_params(chat_config: dict[str, Any]) -> bool:
    """编辑模型相关参数。"""
    logger.info("进入模型参数编辑菜单。")
    llm_changed = False
    while True:
        # 与 ``edit_retrieval_params`` 同因：``active_*_configuration`` 可被直写，
        # 裸插值对位数超限的整数抛 ``ValueError: Exceeds the limit``，菜单渲染不出来。
        active_llm = chat_config["active_llm_configuration"]
        active_rerank = chat_config["active_rerank_configuration"]
        current_llm = _render_value(active_llm)
        current_rerank = _render_value(active_rerank)

        choice = questionary.select(
            "选择要切换的模型:",
            choices=[
                questionary.Choice(f"1. 语言模型 (LLM) (当前: {current_llm})", value="llm"),
                questionary.Choice(
                    f"2. 重排模型 (Rerank) (当前: {current_rerank})", value="rerank"
                ),
                questionary.Separator(),
                questionary.Choice("返回主菜单", value="back"),
            ],
            style=questionary_style_for_theme(),
        ).ask()

        if choice == "back" or choice is None:
            logger.info("退出模型参数编辑菜单。")
            break

        if choice == "llm":
            llm_options = _configuration_choices(chat_config["llm_configurations"], "LLM")
            new_llm = questionary.select(
                "选择新的LLM配置:",
                choices=llm_options,
                default=_choice_default(llm_options, active_llm),
                style=questionary_style_for_theme(),
            ).ask()
            # 与**原始**键比较，不与渲染后的 ``current_llm`` 比较：非字符串键
            # （或位数超限的整数键）渲染成量级描述后永不等于自身，会让「选中
            # 同一个配置」被误判成切换、白白触发一次 LLM 重载。
            if new_llm is not None and new_llm != active_llm:
                chat_config["active_llm_configuration"] = new_llm
                console.print(markup_text("[green]LLM配置已切换为: {}[/green]", new_llm))
                logger.info("LLM配置已切换为: %s", new_llm)
                llm_changed = True
            continue

        if choice == "rerank":
            rerank_options = _configuration_choices(chat_config["rerank_configurations"], "Rerank")
            new_rerank = questionary.select(
                "选择新的Rerank配置:",
                choices=rerank_options,
                default=_choice_default(rerank_options, active_rerank),
                style=questionary_style_for_theme(),
            ).ask()
            if new_rerank is not None:
                chat_config["active_rerank_configuration"] = new_rerank
                console.print(markup_text("[green]Rerank已切换为: {}[/green]", new_rerank))
                logger.info("Rerank已切换为: %s", new_rerank)
            continue

    return llm_changed


def launch_config_editor(chat_config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """
    启动交互式配置编辑器。
    返回一个元组 (llm_needs_reload, updated_config)
    """
    logger.info("启动配置编辑器。")
    console.print(
        Panel("进入配置模式...", title="[yellow]配置编辑器[/yellow]", border_style="yellow")
    )

    state = {"llm_needs_reload": False}

    # 编辑器全程**原地**写 ``chat_config``（``editable_config`` 是 ``/config``
    # 入口浅拷贝出来、随后会被 ``apply_config_update`` 应用到 bot 的同一份
    # 字典）。EOF 收敛承诺「按原样保存返回」，但 EOF 前已确认写入的每一项
    # （Top K、切换的 LLM 等）都留在这份字典里并随返回生效——实测「Top K
    # 输入 7 → EOF」返回 top_k=7 且提示「按原样」。因此入口先快照，EOF 时
    # 还原；这同时解决 reload 标志问题：EOF 前若切换过 LLM，标志尚未置位
    # （只在 ``edit_model_params`` 正常走完才置位），还原后残留的 LM 键不再
    # 以「已切换但未重载」的形态继续服务。
    snapshot = copy.deepcopy(chat_config)

    try:
        _run_config_editor_loop(chat_config, state)
    except EOFError:
        # ``questionary`` 的 ``.ask()`` 只捕 ``KeyboardInterrupt``（Ctrl+C 返回
        # ``None`` → 各层循环退一层）；**Ctrl+D/EOF 抛 ``EOFError`` 直接穿透**
        # （实测 /dev/null 与 \x04 均复现）。它经 ``asyncio.to_thread`` 到达
        # 聊天主循环的 ``except (KeyboardInterrupt, EOFError): break``——
        # 在最深的 Top K 输入框按一次 Ctrl+D 会**终止整个聊天会话**，与
        # Ctrl+C 的「退一层」语义不对称。EOF 中断按「放弃全部编辑、保存返回」
        # 收敛：还原快照使承诺与写回事实一致。
        chat_config.clear()
        chat_config.update(snapshot)
        state["llm_needs_reload"] = False
        console.print(markup_text("[yellow]输入已结束（EOF），配置编辑器按原样保存返回。[/yellow]"))
        logger.info("配置编辑器因 EOF 提前退出，已还原 EOF 前的全部修改。")

    return state["llm_needs_reload"], chat_config


def _run_config_editor_loop(chat_config: dict[str, Any], state: dict[str, bool]) -> None:
    """配置编辑器主循环（``launch_config_editor`` 的主体）。

    独立成函数是为了让 ``launch_config_editor`` 在外层统一收敛 ``EOFError``：
    各层 ``while True`` 只判 ``choice is None``（Ctrl+C 语义），EOF 从任何一层
    的 ``.ask()`` 抛出都会沿调用栈到达这里。
    """
    while True:
        choice = questionary.select(
            "主菜单: 请选择要执行的操作",
            choices=[
                "1. 调整检索参数",
                "2. 切换模型",
                "3. 查看当前完整配置",
                questionary.Separator(),
                questionary.Choice("4. 保存并返回聊天", value="exit"),
            ],
            style=questionary_style_for_theme(),
        ).ask()

        if choice == "exit" or choice is None:
            console.print(
                Panel(
                    "配置完成，返回聊天。",
                    title="[yellow]配置编辑器[/yellow]",
                    border_style="yellow",
                )
            )
            logger.info("配置编辑器退出，保存并返回聊天。")
            break

        if "1." in choice:
            logger.debug("用户选择调整检索参数。")
            # 检索参数的更改不会触发重载
            edit_retrieval_params(chat_config)
        elif "2." in choice:
            logger.debug("用户选择切换模型。")
            # 如果LLM发生变化，标记需要重载
            if edit_model_params(chat_config):
                state["llm_needs_reload"] = True
        elif "3." in choice:
            logger.debug("用户选择查看当前完整配置。")
            display_chat_config(console, chat_config)
            questionary.press_any_key_to_continue("按任意键返回主菜单...").ask()
            logger.debug("用户已查看配置并返回。")

    logger.info(f"配置编辑器完成，LLM需要重载: {state['llm_needs_reload']}")
