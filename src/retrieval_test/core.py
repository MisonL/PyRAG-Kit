# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

from ..utils.config import get_settings
from ..retrieval.vdb.factory import VectorStoreFactory
from ..retrieval.retriever import retrieve_documents
from ..utils.log_manager import get_module_logger
from .excel_logger import ExcelLogger

# =================================================================
# 2. 全局变量和初始化
# =================================================================
console = Console()
logger = get_module_logger(__name__)

# =================================================================
# 3. 核心功能
# =================================================================
def display_results(query: str, documents: List[Dict]):
    """使用 rich.Table 格式化并显示检索结果。"""
    if not documents:
        console.print(Panel(f"对查询 “[bold yellow]{query}[/bold yellow]” 没有找到相关结果。", title="[bold red]无结果[/bold red]", border_style="red"))
        return

    table = Table(title=f"对查询 “[bold green]{query}[/bold green]” 的召回测试结果", show_header=True, header_style="bold magenta")
    table.add_column("排名", style="dim", width=4)
    table.add_column("内容", style="cyan")
    table.add_column("来源", style="yellow", width=30)
    table.add_column("分数", style="green", width=10)

    for i, doc in enumerate(documents):
        content = doc.get('page_content', 'N/A').replace('\n', ' ').strip()
        source = doc.get('metadata', {}).get('source', 'N/A')
        page = doc.get('metadata', {}).get('page')
        score = doc.get('score', 0.0)
        
        # 确保内容不会过长
        if len(content) > 150:
            content = content[:150] + "..."
            
        # 格式化来源
        source_display = f"{source}"
        if page is not None:
            source_display += f" (页码: {page})"

        table.add_row(
            str(i + 1),
            content,
            source_display,
            f"{score:.4f}"
        )

    console.print(table)

def run_retrieval_test():
    """
    运行召回测试的用户交互界面。
    提示用户输入查询，调用检索逻辑，并显示结果。
    """
    console.print(Panel("进入召回测试模式。输入 '/quit' 退出。", title="[bold yellow]召回测试[/bold yellow]", border_style="yellow"))

    # 初始化 Excel 日志记录器
    try:
        excel_logger = ExcelLogger()
        console.print(f"[green]测试结果将记录到: {excel_logger.filepath}[/green]")
    except Exception as e:
        console.print(f"[bold red]初始化 Excel 日志记录器失败: {e}[/bold red]")
        excel_logger = None

    # 在循环外加载一次配置和向量存储，避免重复加载
    try:
        settings = get_settings()
        vector_store = VectorStoreFactory.get_default_vector_store()
    except Exception as e:
        logger.error(f"加载配置或向量存储时出错: {e}", exc_info=True)
        console.print(Panel(f"[bold red]错误:[/bold red] 无法加载向量数据库，请先确保已成功生成知识库文件。", title="[bold red]初始化失败[/bold red]", border_style="red"))
        return

    while True:
        try:
            query = prompt(HTML('<deepskyblue><b>请输入测试查询: </b></deepskyblue>'))
            if query.lower() == '/quit':
                break
            
            if not query:
                continue

            logger.info(f"开始对查询进行召回测试: '{query}'")
            
            # 执行检索
            with console.status("[bold green]正在检索文档...[/bold green]"):
                documents = retrieve_documents(
                    query=query,
                    vector_store=vector_store,
                    console=console,
                    retrieval_method=settings.chat_retrieval_method,
                    top_k=settings.chat_top_k,
                    vector_weight=settings.chat_vector_weight,
                    keyword_weight=settings.chat_keyword_weight,
                    rerank_enabled=settings.chat_rerank_enabled,
                    active_rerank_configuration=settings.default_rerank_provider,
                    score_threshold=settings.chat_score_threshold
                )
            
            # 显示结果
            display_results(query, documents)

            # 记录到 Excel
            if excel_logger:
                excel_logger.log_results(query, documents)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]检测到中断信号，退出召回测试模式。[/bold yellow]")
            break
        except Exception as e:
            logger.error(f"召回测试期间发生错误: {e}", exc_info=True)
            console.print(Panel(f"[bold red]发生错误:[/bold red] {e}", title="[bold red]错误[/bold red]", border_style="red"))
