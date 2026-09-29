from collections.abc import Callable, Mapping
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel

from ..utils.config import (
    _MAX_RETRIEVAL_CANDIDATES,
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    RetrievalMethod,
    parse_decimal_int,
    safe_repr,
)
from ..utils.log_manager import get_module_logger  # 导入日志管理器
from .display_utils import _render_value, display_chat_config, markup_text

logger = get_module_logger(__name__)  # 获取当前模块的日志器
console = Console()


def _top_k_ceiling(multiplier: int) -> int:
    """返回当前倍率下 ``top_k`` 可用的上界。

    真正送进 ``faiss_index.search()`` 的是 ``top_k × multiplier``（FAISS 既不
    报错也不截断），因此单字段上界不够——UI 放行一个组合越界的值，会在应用
    配置时抛 ``ValueError`` 并终止会话。
    """
    return min(_MAX_RETRIEVAL_TOP_K, _MAX_RETRIEVAL_CANDIDATES // max(1, multiplier))


def _multiplier_ceiling(top_k: int) -> int:
    """返回当前 ``top_k`` 下候选倍率可用的上界。"""
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
        current_method = chat_config["retrieval_method"].value
        current_top_k = chat_config["top_k"]
        current_rerank = "启用" if chat_config["rerank_enabled"] else "禁用"
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
        current_fusion_strategy = _render_value(chat_config["hybrid_fusion_strategy"]).upper()
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
            style=questionary.Style([("pointer", "bold fg:yellow")]),
        ).ask()

        if choice == "back" or choice is None:
            logger.info("退出检索参数编辑菜单。")
            break

        if choice == "method":
            new_method = questionary.select(
                "选择新的检索模式:",
                choices=[
                    questionary.Choice("全文检索", value=RetrievalMethod.FULL_TEXT_SEARCH.value),
                    questionary.Choice("向量检索", value=RetrievalMethod.SEMANTIC_SEARCH.value),
                    questionary.Choice(
                        "混合检索 (全文+向量)", value=RetrievalMethod.HYBRID_SEARCH.value
                    ),
                ],
                default=current_method,
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
            new_rerank = questionary.confirm(
                "是否启用Rerank重排?", default=chat_config["rerank_enabled"]
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
                new_keyword_weight = 1.0 - new_vector_weight
                chat_config["vector_weight"] = new_vector_weight
                chat_config["keyword_weight"] = round(new_keyword_weight, 2)
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
                default=chat_config["hybrid_fusion_strategy"],
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


def edit_model_params(chat_config: dict[str, Any]) -> bool:
    """编辑模型相关参数。"""
    logger.info("进入模型参数编辑菜单。")
    llm_changed = False
    while True:
        current_llm = chat_config["active_llm_configuration"]
        current_rerank = chat_config["active_rerank_configuration"]

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
            style=questionary.Style([("pointer", "bold fg:yellow")]),
        ).ask()

        if choice == "back" or choice is None:
            logger.info("退出模型参数编辑菜单。")
            break

        if choice == "llm":
            llm_options = _configuration_keys(chat_config["llm_configurations"], "LLM")
            new_llm = questionary.select(
                "选择新的LLM配置:", choices=llm_options, default=current_llm
            ).ask()
            if new_llm and new_llm != current_llm:
                chat_config["active_llm_configuration"] = new_llm
                console.print(markup_text("[green]LLM配置已切换为: {}[/green]", new_llm))
                logger.info("LLM配置已切换为: %s", new_llm)
                llm_changed = True
            continue

        if choice == "rerank":
            rerank_options = _configuration_keys(chat_config["rerank_configurations"], "Rerank")
            new_rerank = questionary.select(
                "选择新的Rerank配置:", choices=rerank_options, default=current_rerank
            ).ask()
            if new_rerank:
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

    llm_needs_reload = False

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
            style=questionary.Style([("pointer", "bold fg:cyan")]),
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
                llm_needs_reload = True
        elif "3." in choice:
            logger.debug("用户选择查看当前完整配置。")
            display_chat_config(console, chat_config)
            questionary.press_any_key_to_continue("按任意键返回主菜单...").ask()
            logger.debug("用户已查看配置并返回。")

    logger.info(f"配置编辑器完成，LLM需要重载: {llm_needs_reload}")
    return llm_needs_reload, chat_config
