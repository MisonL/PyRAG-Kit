# -*- coding: utf-8 -*-
import questionary
from rich.console import Console
from rich.panel import Panel
from typing import Dict, Any, Tuple

from ..utils.config import settings, RetrievalMethod
from .display_utils import display_chat_config

console = Console()

def edit_retrieval_params(chat_config: Dict[str, Any]) -> None:
    """编辑检索相关参数。"""
    while True:
        current_method = chat_config['retrieval_method'].value
        current_top_k = chat_config['top_k']
        current_rerank = "启用" if chat_config['rerank_enabled'] else "禁用"
        current_weights = f"向量: {chat_config['vector_weight']} / 关键词: {chat_config['keyword_weight']}"

        choice = questionary.select(
            "选择要调整的检索参数:",
            choices=[
                questionary.Choice(f"1. 检索模式 (当前: {current_method})", value="method"),
                questionary.Choice(f"2. 检索数量 Top K (当前: {current_top_k})", value="top_k"),
                questionary.Choice(f"3. Rerank重排 (当前: {current_rerank})", value="rerank"),
                questionary.Choice(f"4. 混合搜索权重 (当前: {current_weights})", value="weights"),
                questionary.Separator(),
                questionary.Choice("返回主菜单", value="back")
            ],
            style=questionary.Style([('pointer', 'bold fg:yellow')]),
        ).ask()

        if choice == "back" or choice is None:
            break

        if choice == "method":
            new_method = questionary.select(
                "选择新的检索模式:",
                choices=[
                    questionary.Choice("全文检索", value=RetrievalMethod.FULL_TEXT_SEARCH.value),
                    questionary.Choice("向量检索", value=RetrievalMethod.SEMANTIC_SEARCH.value),
                    questionary.Choice("混合检索 (全文+向量)", value=RetrievalMethod.HYBRID_SEARCH.value)
                ],
                default=current_method
            ).ask()
            if new_method:
                chat_config['retrieval_method'] = RetrievalMethod(new_method)
                console.print(f"[green]检索模式已更新为: {new_method}[/green]")

        elif choice == "top_k":
            new_top_k = questionary.text(
                f"输入新的Top K值 (当前: {current_top_k}):",
                validate=lambda text: text.isdigit() and int(text) > 0,
                default=str(current_top_k)
            ).ask()
            if new_top_k:
                chat_config['top_k'] = int(new_top_k)
                console.print(f"[green]Top K 已更新为: {new_top_k}[/green]")

        elif choice == "rerank":
            new_rerank = questionary.confirm(
                "是否启用Rerank重排?",
                default=chat_config['rerank_enabled']
            ).ask()
            if new_rerank is not None:
                chat_config['rerank_enabled'] = new_rerank
                console.print(f"[green]Rerank已更新为: {'启用' if new_rerank else '禁用'}[/green]")
        
        elif choice == "weights":
            if chat_config['retrieval_method'] != RetrievalMethod.HYBRID_SEARCH:
                console.print("[yellow]警告: 权重调整仅在 '混合检索' 模式下生效。[/yellow]")
            
            def is_float_between_0_and_1(text):
                try:
                    val = float(text)
                    return 0.0 <= val <= 1.0
                except ValueError:
                    return False

            new_vector_weight_str = questionary.text(
                f"输入新的向量权重 (0.0-1.0, 当前: {chat_config['vector_weight']}):",
                validate=is_float_between_0_and_1,
                default=str(chat_config['vector_weight'])
            ).ask()

            if new_vector_weight_str:
                new_vector_weight = float(new_vector_weight_str)
                new_keyword_weight = 1.0 - new_vector_weight
                chat_config['vector_weight'] = new_vector_weight
                chat_config['keyword_weight'] = round(new_keyword_weight, 2)
                console.print(f"[green]混合搜索权重已更新为 -> 向量: {chat_config['vector_weight']}, 关键词: {chat_config['keyword_weight']}[/green]")
    # 检索参数的更改不需要重载任何模型
    return None

def edit_model_params(chat_config: Dict[str, Any]) -> bool:
    """编辑模型相关参数。"""
    llm_changed = False
    while True:
        current_llm = chat_config['active_llm_configuration']
        current_rerank = chat_config['active_rerank_configuration']

        choice = questionary.select(
            "选择要切换的模型:",
            choices=[
                questionary.Choice(f"1. 语言模型 (LLM) (当前: {current_llm})", value="llm"),
                questionary.Choice(f"2. 重排模型 (Rerank) (当前: {current_rerank})", value="rerank"),
                questionary.Separator(),
                questionary.Choice("返回主菜单", value="back")
            ],
            style=questionary.Style([('pointer', 'bold fg:yellow')]),
        ).ask()

        if choice == "back" or choice is None:
            break

        if choice == "llm":
            llm_options = list(chat_config['llm_configurations'].keys())
            new_llm = questionary.select(
                "选择新的LLM配置:",
                choices=llm_options,
                default=current_llm
            ).ask()
            if new_llm and new_llm != current_llm:
                chat_config['active_llm_configuration'] = new_llm
                console.print(f"[green]LLM配置已切换为: {new_llm}[/green]")
                llm_changed = True

        elif choice == "rerank":
            rerank_options = list(chat_config['rerank_configurations'].keys())
            new_rerank = questionary.select(
                "选择新的Rerank配置:",
                choices=rerank_options,
                default=current_rerank
            ).ask()
            if new_rerank:
                chat_config['active_rerank_configuration'] = new_rerank
                console.print(f"[green]Rerank已切换为: {new_rerank}[/green]")
                # Rerank模型是按需加载的，所以不需要标记状态

    return llm_changed

def launch_config_editor(chat_config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    启动交互式配置编辑器。
    返回一个元组 (llm_needs_reload, updated_config)
    """
    console.print(Panel("进入配置模式...", title="[yellow]配置编辑器[/yellow]", border_style="yellow"))
    
    llm_needs_reload = False

    while True:
        choice = questionary.select(
            "主菜单: 请选择要执行的操作",
            choices=[
                "1. 调整检索参数",
                "2. 切换模型",
                "3. 查看当前完整配置",
                questionary.Separator(),
                questionary.Choice("4. 保存并返回聊天", value="exit")
            ],
            style=questionary.Style([('pointer', 'bold fg:cyan')]),
        ).ask()

        if choice == "exit" or choice is None:
            console.print(Panel("配置完成，返回聊天。", title="[yellow]配置编辑器[/yellow]", border_style="yellow"))
            break
        
        if "1." in choice:
            # 检索参数的更改不会触发重载
            edit_retrieval_params(chat_config)
        elif "2." in choice:
            # 如果LLM发生变化，标记需要重载
            if edit_model_params(chat_config):
                llm_needs_reload = True
        elif "3." in choice:
            display_chat_config(console, chat_config)
            questionary.press_any_key_to_continue("按任意键返回主菜单...").ask()

    return llm_needs_reload, chat_config