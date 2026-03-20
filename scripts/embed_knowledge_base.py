# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import argparse
import asyncio
import os
import sys
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# 从 src 导入重构后的模块
from src.utils.config import get_settings
from src.providers.factory import ModelProviderFactory
from src.retrieval.vdb.base import VectorStoreBase
from src.retrieval.vdb.factory import VectorStoreFactory
from src.ui.display_utils import CONSOLE_WIDTH, get_relative_path
from src.etl.pipeline import Pipeline
from src.models.document import Document
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

# =================================================================
# 2. 知识库核心 (KNOWLEDGE BASE CORE)
# =================================================================
async def process_documents_async(vector_store: VectorStoreBase, splitter_structure_mode: str):
    console = Console()
    logger.info("开始异步处理知识库文档。")
    current_settings = get_settings()
    kb_dir = current_settings.knowledge_base_path
    
    # 获取所有Markdown文件路径
    markdown_files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir) if f.endswith(".md")]

    if not markdown_files:
        error_message = f"错误: 知识库目录 '{kb_dir}' 中未找到Markdown文档。"
        console.print(f"[bold red]{error_message}[/bold red]")
        logger.error(error_message)
        sys.exit(1)

    console.print("[bold]开始通过 ETL 流水线处理文档...[/bold]")
    logger.info(f"找到 {len(markdown_files)} 个Markdown文档。")
    
    # 实例化 Pipeline
    pipeline = Pipeline.from_file_path(Path(markdown_files[0]), splitter_structure_mode=splitter_structure_mode)
    
    final_chunks = []
    # 这里处理文档内容读取可以保持同步，或者使用 aiofiles
    for file_path in markdown_files:
        logger.info(f"正在处理文件: {get_relative_path(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = Document(content=content, metadata={'source': file_path})
        chunks = pipeline.process(doc)
        if splitter_structure_mode == "hierarchical" and hasattr(pipeline.splitter, "parent_documents"):
            vector_store.register_parent_documents(getattr(pipeline.splitter, "parent_documents", {}))
        final_chunks.extend(chunks)
        logger.debug(f"文件 '{get_relative_path(file_path)}' 生成 {len(chunks)} 个文本块。")
 
    console.print(f"  [green]总共生成 [bold]{len(final_chunks)}[/bold] 个文本块。[/green]")
    
    # 将文档内容和元数据存入 VectorStore 实例
    documents_to_embed = [{"page_content": chunk.content, "metadata": chunk.metadata} for chunk in final_chunks]
    logger.info(f"正在异步将 {len(documents_to_embed)} 个文本块添加到向量存储。")
    
    # 异步添加文档（包含异步嵌入生成和线程池索引更新）
    await vector_store.aadd_documents(documents_to_embed)
    
    # 保存到文件
    logger.info(f"正在保存向量存储到: {current_settings.pkl_path}")
    vector_store.save(current_settings.pkl_path)

    console.print(f"\n[bold green]知识库已成功处理并保存到 '[cyan]{current_settings.pkl_path}[/cyan]'。[/bold green]")
    logger.info(f"知识库处理完成。路径: {current_settings.pkl_path}")

# =================================================================
# 4. 辅助与主函数 (HELPERS & MAIN)
# =================================================================
def display_config_and_confirm(splitter_structure_mode: str):
    """以美观的表格形式显示全面的配置信息，并请求用户确认。"""
    console = Console()
    logger.info("正在显示配置显示。")
    current_settings = get_settings()

    def mask_api_key(key: Optional[str]) -> str:
        if not key or key == "no-key-required":
            return "[dim]未设置或无需设置[/dim]"
        if len(key) > 12:
            return f"[white]{key[:6]}...{key[-4:]}[/white]"
        return "[white]已设置[/white]"

    table = Table(
        box=box.ROUNDED,
        padding=(0, 2),
        title="[bold yellow]异步向量化脚本配置总览[/bold yellow]",
        show_header=False,
        width=CONSOLE_WIDTH
    )
    table.add_column(justify="right", style="cyan", no_wrap=True, width=28)
    table.add_column(style="bright_white")

    table.add_row("[bold green]知识库配置[/bold green]", "")
    table.add_row("知识库目录", f"[bold cyan]{get_relative_path(current_settings.knowledge_base_path)}[/bold cyan]")
    table.add_row("输出文件路径", f"[bold cyan]{get_relative_path(current_settings.pkl_path)}[/bold cyan]")
    table.add_row("索引模式", f"[bold magenta]{splitter_structure_mode}[/bold magenta]")
    table.add_row("文本切分块大小", f"[bold magenta]{current_settings.kb_chunk_size}[/bold magenta]")
    table.add_row("切分重叠量", f"[bold magenta]{current_settings.kb_chunk_overlap}[/bold magenta]")
    table.add_row("子分段块大小", f"[bold magenta]{current_settings.kb_child_chunk_size}[/bold magenta]")
    table.add_row("子分段重叠量", f"[bold magenta]{current_settings.kb_child_chunk_overlap}[/bold magenta]")
    
    table.add_section()
    table.add_row("[bold green]模型与API配置[/bold green]", "")
    active_embedding_key = current_settings.default_embedding_provider
    embedding_model_details = current_settings.embedding_configurations[active_embedding_key]
    provider = embedding_model_details.provider
    
    table.add_row("激活嵌入提供商", f"[bold green]{active_embedding_key}[/bold green] ([dim]{provider}[/dim])")
    table.add_row("模型名称", f"[bold bright_white]{embedding_model_details.model_name}[/bold bright_white]")

    key_field_name = f"{provider.lower()}_api_key"
    api_key_value = getattr(current_settings, key_field_name, None)
    table.add_row(f"API Key ({key_field_name.upper()})", mask_api_key(api_key_value))

    console.print(table)
    console.print("[yellow]配置信息来源于 config.ini, .env 或环境变量。[/yellow]")
    
    choice = console.input("是否使用以上配置继续？ (y/n): ").lower()
    if choice not in ['y', 'yes']:
        console.print("[bold red]操作取消。[/bold red]")
        sys.exit(0)

async def main_async(splitter_structure_mode: str):
    console = Console()
    logger.info("脚本开始执行 (Async)。")
    display_config_and_confirm(splitter_structure_mode)
    
    vector_store = VectorStoreFactory.get_default_vector_store()
    await process_documents_async(vector_store, splitter_structure_mode)
    console.print("\n[bold green]知识库异步嵌入完成。[/bold green]")
    logger.info("知识库嵌入脚本执行完成。")

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将知识库文档向量化并保存到本地存储。")
    parser.add_argument(
        "--mode",
        choices=["standard", "hierarchical"],
        default="standard",
        help="索引模式。standard 为单层分片，hierarchical 为父子分片。",
    )
    return parser.parse_args()

def main():
    """同步入口包装。"""
    try:
        args = parse_args()
        asyncio.run(main_async(args.mode))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
