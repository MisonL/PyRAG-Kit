# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import os
import sys
import pickle
from typing import Any, Optional, List, Dict
from pathlib import Path # 导入 Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# 从 src 导入重构后的模块
from src.utils.config import get_settings # 导入 get_settings 函数
from src.providers.factory import ModelProviderFactory
from src.retrieval.vdb.base import VectorStoreBase
from src.retrieval.vdb.factory import VectorStoreFactory
from src.ui.display_utils import CONSOLE_WIDTH, get_relative_path
from src.etl.pipeline import Pipeline # 导入 Pipeline
from src.models.document import Document # 导入 Document 模型
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

# =================================================================
# 2. 知识库核心 (KNOWLEDGE BASE CORE)
# =================================================================
def process_documents(vector_store: VectorStoreBase):
    console = Console()
    logger.info("开始处理知识库文档。")
    current_settings = get_settings() # 获取当前配置
    kb_dir = current_settings.knowledge_base_path
    
    # 获取所有Markdown文件路径
    markdown_files = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".md"):
            markdown_files.append(os.path.join(kb_dir, filename))

    if not markdown_files:
        error_message = f"错误: 知识库目录 '{kb_dir}' 中未找到Markdown文档。"
        console.print(f"[bold red]{error_message}[/bold red]")
        logger.error(error_message)
        sys.exit(1)

    console.print("[bold]开始通过ETL流水线处理文档...[/bold]")
    logger.info(f"找到 {len(markdown_files)} 个Markdown文档。")
    # 实例化 Pipeline，它会根据文件路径自动选择处理器
    # 假设所有文件类型相同，取第一个文件路径来初始化 Pipeline
    pipeline = Pipeline.from_file_path(Path(markdown_files[0]))
    
    final_chunks = []
    for file_path in markdown_files:
        logger.info(f"正在处理文件: {get_relative_path(file_path)}")
        # 从文件路径创建 Document 对象
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = Document(content=content, metadata={'source': file_path})
        chunks = pipeline.process(doc)
        final_chunks.extend(chunks)
        logger.debug(f"文件 '{get_relative_path(file_path)}' 生成 {len(chunks)} 个文本块。")
 
    console.print(f"  [green]总共生成 [bold]{len(final_chunks)}[/bold] 个文本块。[/green]")
    logger.info(f"所有文档处理完成，总共生成 {len(final_chunks)} 个文本块。")
    
    # 将文档内容和元数据存入 VectorStore 实例
    documents_to_embed = [{"page_content": chunk.content, "metadata": chunk.metadata} for chunk in final_chunks] # 确保使用 chunk.content 和 chunk.metadata
    logger.info(f"正在将 {len(documents_to_embed)} 个文本块添加到向量存储。")
    vector_store.add_documents(documents_to_embed)
    
    console.print("  [bold]开始生成文本嵌入...[/bold]")
    logger.info("文本嵌入生成已在 FaissStore.add_documents 中处理。")
    # 嵌入生成逻辑已移至 FaissStore.add_documents
    
    # 保存到文件
    logger.info(f"正在保存向量存储到: {current_settings.pkl_path}")
    vector_store.save(current_settings.pkl_path)

    console.print(f"\n[bold green]知识库已成功处理并保存到 '[cyan]{current_settings.pkl_path}[/cyan]'。[/bold green]")
    logger.info(f"知识库处理和保存完成。路径: {current_settings.pkl_path}")

# =================================================================
# 4. 辅助与主函数 (HELPERS & MAIN)
# =================================================================
def display_config_and_confirm():
    """以美观的表格形式显示全面的配置信息，并请求用户确认。"""
    console = Console()
    logger.info("正在显示配置信息并请求用户确认。")
    current_settings = get_settings() # 获取当前配置

    def mask_api_key(key: Optional[str]) -> str:
        """对API密钥进行脱敏处理，使其更安全地显示。"""
        if not key or key == "no-key-required": # 适配 lm-studio 和 ollama 的 "no-key-required"
            logger.debug("API Key 未设置或无需设置。")
            return "[dim]未设置或无需设置[/dim]"
        if len(key) > 12:
            logger.debug("API Key 已设置，显示部分掩码。")
            return f"[white]{key[:6]}...{key[-4:]}[/white]"
        logger.debug("API Key 已设置，显示为 '已设置'。")
        return "[white]已设置[/white]"

    table = Table(
        box=box.ROUNDED,
        padding=(0, 2),
        title="[bold yellow]向量化脚本配置总览[/bold yellow]",
        show_header=False,
        width=CONSOLE_WIDTH
    )
    # 参数名列: 蓝色
    table.add_column(justify="right", style="cyan", no_wrap=True, width=28)
    # 参数值列: 亮白色
    table.add_column(style="bright_white")

    # --- 知识库配置 ---
    table.add_row("[bold green]知识库配置[/bold green]", "")
    table.add_row("知识库目录", f"[bold cyan]{get_relative_path(current_settings.knowledge_base_path)}[/bold cyan]")
    table.add_row("输出文件路径", f"[bold cyan]{get_relative_path(current_settings.pkl_path)}[/bold cyan]")
    table.add_row("文本切分块大小 (Chunk Size)", f"[bold magenta]{current_settings.kb_chunk_size}[/bold magenta]")
    table.add_row("文本切分重叠量 (Overlap)", f"[bold magenta]{current_settings.kb_chunk_overlap}[/bold magenta]")
    # 将分隔符列表转换为更易读的字符串
    separators_str = ", ".join([f"'{s}'" for s in current_settings.kb_splitter_separators])
    table.add_row("切分分隔符 (Separators)", f"[bold bright_white]{separators_str}[/bold bright_white]")
    logger.debug("知识库配置信息已添加到表格。")
    table.add_section()

    # --- 模型与API配置 ---
    table.add_row("[bold green]模型与API配置[/bold green]", "")
    active_embedding_key = current_settings.default_embedding_provider
    embedding_model_details = current_settings.embedding_configurations[active_embedding_key]
    provider = embedding_model_details.provider
    
    table.add_row("激活的嵌入模型 (Provider)", f"[bold green]{active_embedding_key}[/bold green] ([dim]{provider}[/dim])")
    table.add_row("模型名称 (Model Name)", f"[bold bright_white]{embedding_model_details.model_name}[/bold bright_white]")
    logger.debug(f"激活的嵌入模型: {active_embedding_key} ({provider}: {embedding_model_details.model_name})")

    # 动态获取API Key和Base URL的字段名
    key_field_name = f"{provider.lower()}_api_key"
    
    # 处理不一致的URL字段名
    url_field_name = ""
    if provider == "openai":
        url_field_name = "openai_api_base"
    elif hasattr(current_settings, f"{provider.lower()}_base_url"):
        url_field_name = f"{provider.lower()}_base_url"

    api_key_value = getattr(current_settings, key_field_name, None)
    # mask_api_key 函数已内置样式，无需额外添加
    table.add_row(f"对应的 API Key ({key_field_name.upper()})", mask_api_key(api_key_value))
    logger.debug(f"显示提供商 '{provider}' 的 API Key。")

    if url_field_name:
        base_url_value = getattr(current_settings, url_field_name, None)
        if base_url_value:
            table.add_row(f"对应的 Base URL ({url_field_name.upper()})", f"[bold cyan]{base_url_value}[/bold cyan]")
            logger.debug(f"显示提供商 '{provider}' 的 Base URL: {base_url_value}")
        else:
            logger.debug(f"提供商 '{provider}' 的 Base URL 未设置。")

    console.print(table)
    console.print("[yellow]配置信息来源于 [bold]config.ini[/bold], [bold].env[/bold] 文件或 [bold]环境变量[/bold]。[/yellow]")
    logger.info("配置信息显示完成。")
    
    choice = console.input("是否使用以上配置继续处理？ ([bold green]y[/bold green]/[bold red]n[/bold red]): ").lower()
    if choice not in ['y', 'yes']:
        console.print("[bold red]操作已取消。[/bold red]")
        logger.info("用户取消操作。")
        sys.exit(0)
    logger.info("用户确认继续处理。")

def main():
    console = Console()
    logger.info("脚本开始执行。")
    display_config_and_confirm()
    
    # 知识库目录检查已移至 process_documents 内部
    # kb_dir = settings.knowledge_base_path 
    # if not os.path.exists(kb_dir) or not any(f.endswith('.md') for f in os.listdir(kb_dir)):
    #      console.print(f"[bold red]错误: 知识库目录 '{kb_dir}' 不存在或为空。请先添加Markdown文档。[/bold red]")
    #      sys.exit(1)

    vector_store = VectorStoreFactory.get_default_vector_store() # 通过工厂获取实例
    process_documents(vector_store)
    console.print("\n[bold green]知识库嵌入完成。现在可以启动聊天机器人会话进行测试。[/bold green]")
    logger.info("知识库嵌入脚本执行完成。")

if __name__ == "__main__":
    main()
