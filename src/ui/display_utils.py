# -*- coding: utf-8 -*-
import os
from typing import Optional, Dict, Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from ..utils.config import get_settings, ROOT_DIR # 导入 get_settings 函数
from ..utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

# 全局UI宽度定义
CONSOLE_WIDTH = 120

def get_relative_path(absolute_path: str) -> str:
    """将绝对路径转换为相对于项目根目录的路径，方便显示。"""
    try:
        # 使用 os.path.relpath 计算相对路径
        relative_path = os.path.relpath(str(absolute_path), str(ROOT_DIR))
        logger.debug(f"将绝对路径 '{absolute_path}' 转换为相对路径: '{relative_path}'")
        return relative_path
    except ValueError:
        # 如果路径不在同一驱动器上（例如Windows），则返回原始路径
        logger.warning(f"无法将路径 '{absolute_path}' 转换为相对路径，返回原始路径。")
        return str(absolute_path)

def display_chat_config(console: Console, chat_config: Dict[str, Any]):
    """显示从传入的chat_config字典中获取的聊天机器人配置。"""
    logger.info("正在显示聊天机器人配置。")
    def mask_api_key(key: Optional[str]) -> str:
        if not key:
            logger.debug("API Key 未设置，显示为 '[dim]未设置[/dim]'。")
            return "[dim]未设置[/dim]"
        logger.debug("API Key 已设置，显示部分掩码。")
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
        logger.debug(f"Rerank模型已启用，激活模型: {active_rerank}")
    else:
        logger.debug("Rerank模型未启用。")
    chat_table.add_row("返回文档数 (top_k):", str(chat_config['top_k']))
    chat_table.add_row("语义搜索阈值:", str(chat_config['score_threshold']))
    active_llm = chat_config['active_llm_configuration']
    llm_model_details = chat_config['llm_configurations'][active_llm]
    chat_table.add_row("激活的LLM:", f"{active_llm} ({llm_model_details.provider}: {llm_model_details.model_name})")
    logger.debug(f"激活的LLM: {active_llm}")

    # --- API与URL配置 ---
    # API密钥和URL仍然从全局的、不可变的settings对象中读取，因为它们不应该在运行时被修改。
    api_table = Table(title="[bold green]API密钥与URL配置[/bold green]", show_header=False, box=None, padding=(0, 1))
    api_table.add_column(style="cyan")
    api_table.add_column(style="bold white")
    # 仅显示与激活的LLM和Rerank相关的配置
    active_providers = {llm_model_details.provider}
    if chat_config['rerank_enabled']:
        active_providers.add(chat_config['rerank_configurations'][chat_config['active_rerank_configuration']].provider)
    
    current_settings = get_settings() # 获取当前配置
    for provider_name in active_providers:
        key_name = f"{provider_name.lower()}_api_key"
        api_key = getattr(current_settings, key_name, None)
        if api_key:
            api_table.add_row(f"{key_name.upper()}:", mask_api_key(api_key))
            logger.debug(f"显示提供商 '{provider_name}' 的 API Key。")
        else:
            logger.debug(f"提供商 '{provider_name}' 的 API Key 未设置。")
        
        url_key_name = f"{provider_name.lower()}_base_url"
        api_url = getattr(current_settings, url_key_name, None)
        if api_url:
             api_table.add_row(f"{url_key_name.upper()}:", str(api_url) or "[dim]未设置[/dim]")
             logger.debug(f"显示提供商 '{provider_name}' 的 Base URL: {api_url}")
        else:
            logger.debug(f"提供商 '{provider_name}' 的 Base URL 未设置。")


    console.print(Panel(Group(chat_table, api_table), title="[bold yellow]当前聊天会话配置[/bold yellow]", border_style="blue", width=CONSOLE_WIDTH))
    logger.info("聊天机器人配置显示完成。")