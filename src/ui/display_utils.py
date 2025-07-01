# -*- coding: utf-8 -*-
from typing import Optional
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from ..utils.config import CHAT_CONFIG, API_CONFIG

def display_chat_config(console: Console):
    """显示聊天机器人相关的全面配置信息。"""
    def mask_api_key(key: Optional[str]) -> str:
        if not key: return "[dim]未设置[/dim]"
        return f"{key[:10]}..."

    # --- 聊天配置 ---
    chat_table = Table(title="[bold green]聊天机器人配置 (CHAT_CONFIG)[/bold green]", show_header=False, box=None, padding=(0, 1))
    chat_table.add_column(style="cyan")
    chat_table.add_column(style="bold white")
    chat_table.add_row("检索策略:", CHAT_CONFIG['retrieval_method'].value)
    chat_table.add_row("混合搜索权重 (向量/关键词):", f"{CHAT_CONFIG['vector_weight']} / {CHAT_CONFIG['keyword_weight']}")
    chat_table.add_row("Rerank重排:", "[bold green]启用[/bold green]" if CHAT_CONFIG['rerank_enabled'] else "[bold red]禁用[/bold red]")
    if CHAT_CONFIG['rerank_enabled']:
        active_rerank = CHAT_CONFIG['active_rerank_configuration']
        rerank_model_details = CHAT_CONFIG['rerank_configurations'][active_rerank]
        chat_table.add_row("激活的Rerank模型:", f"{active_rerank} ({rerank_model_details['provider']}: {rerank_model_details['model_name']})")
    chat_table.add_row("返回文档数 (top_k):", str(CHAT_CONFIG['top_k']))
    chat_table.add_row("语义搜索阈值:", str(CHAT_CONFIG['score_threshold']))
    active_llm = CHAT_CONFIG['active_llm_configuration']
    llm_model_details = CHAT_CONFIG['llm_configurations'][active_llm]
    chat_table.add_row("激活的LLM:", f"{active_llm} ({llm_model_details['provider']}: {llm_model_details['model_name']})")

    # --- API与URL配置 ---
    api_table = Table(title="[bold green]API密钥与URL配置 (API_CONFIG)[/bold green]", show_header=False, box=None, padding=(0, 1))
    api_table.add_column(style="cyan")
    api_table.add_column(style="bold white")
    # 仅显示与激活的LLM和Rerank相关的配置
    active_providers = {llm_model_details['provider']}
    if CHAT_CONFIG['rerank_enabled']:
        active_providers.add(CHAT_CONFIG['rerank_configurations'][CHAT_CONFIG['active_rerank_configuration']]['provider'])
    
    for provider_name in active_providers:
        key_name = f"{provider_name.upper()}_API_KEY"
        if key_name in API_CONFIG:
            api_table.add_row(f"{key_name}:", mask_api_key(API_CONFIG[key_name]))
        
        url_key_name = f"{provider_name.upper()}_BASE_URL"
        if url_key_name in API_CONFIG:
             api_table.add_row(f"{url_key_name}:", API_CONFIG[url_key_name] or "[dim]未设置[/dim]")


    console.print(Panel(Group(chat_table, api_table), title="[bold yellow]当前聊天会话配置[/bold yellow]", border_style="blue"))