# -*- coding: utf-8 -*-
import os
from typing import Optional, Dict, Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from ..utils.config import settings, ROOT_DIR

# 全局UI宽度定义
CONSOLE_WIDTH = 120

def get_relative_path(absolute_path: str) -> str:
    """将绝对路径转换为相对于项目根目录的路径，方便显示。"""
    try:
        # 使用 os.path.relpath 计算相对路径
        return os.path.relpath(str(absolute_path), str(ROOT_DIR))
    except ValueError:
        # 如果路径不在同一驱动器上（例如Windows），则返回原始路径
        return str(absolute_path)

def display_chat_config(console: Console, chat_config: Dict[str, Any]):
    """显示从传入的chat_config字典中获取的聊天机器人配置。"""
    def mask_api_key(key: Optional[str]) -> str:
        if not key: return "[dim]未设置[/dim]"
        return f"{key[:10]}..."

    # --- 聊天配置 ---
    chat_table = Table(title="[bold green]聊天机器人配置[/bold green]", show_header=False, box=None, padding=(0, 1))
    chat_table.add_column(style="cyan")
    chat_table.add_column(style="bold white")
    chat_table.add_row("检索策略:", chat_config['retrieval_method'].value)
    chat_table.add_row("混合搜索权重 (向量/关键词):", f"{chat_config['vector_weight']} / {chat_config['keyword_weight']}")
    chat_table.add_row("Rerank重排:", "[bold green]启用[/bold green]" if chat_config['rerank_enabled'] else "[bold red]禁用[/bold red]")
    if chat_config['rerank_enabled']:
        active_rerank = chat_config['active_rerank_configuration']
        rerank_model_details = chat_config['rerank_configurations'][active_rerank]
        chat_table.add_row("激活的Rerank模型:", f"{active_rerank} ({rerank_model_details.provider}: {rerank_model_details.model_name})")
    chat_table.add_row("返回文档数 (top_k):", str(chat_config['top_k']))
    chat_table.add_row("语义搜索阈值:", str(chat_config['score_threshold']))
    active_llm = chat_config['active_llm_configuration']
    llm_model_details = chat_config['llm_configurations'][active_llm]
    chat_table.add_row("激活的LLM:", f"{active_llm} ({llm_model_details.provider}: {llm_model_details.model_name})")

    # --- API与URL配置 ---
    # API密钥和URL仍然从全局的、不可变的settings对象中读取，因为它们不应该在运行时被修改。
    api_table = Table(title="[bold green]API密钥与URL配置[/bold green]", show_header=False, box=None, padding=(0, 1))
    api_table.add_column(style="cyan")
    api_table.add_column(style="bold white")
    # 仅显示与激活的LLM和Rerank相关的配置
    active_providers = {llm_model_details.provider}
    if chat_config['rerank_enabled']:
        active_providers.add(chat_config['rerank_configurations'][chat_config['active_rerank_configuration']].provider)
    
    for provider_name in active_providers:
        key_name = f"{provider_name.lower()}_api_key"
        api_key = getattr(settings, key_name, None)
        if api_key:
            api_table.add_row(f"{key_name.upper()}:", mask_api_key(api_key))
        
        url_key_name = f"{provider_name.lower()}_base_url"
        api_url = getattr(settings, url_key_name, None)
        if api_url:
             api_table.add_row(f"{url_key_name.upper()}:", str(api_url) or "[dim]未设置[/dim]")


    console.print(Panel(Group(chat_table, api_table), title="[bold yellow]当前聊天会话配置[/bold yellow]", border_style="blue", width=CONSOLE_WIDTH))