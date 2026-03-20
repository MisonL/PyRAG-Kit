import asyncio
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

from ..utils.config import get_settings
from ..retrieval.vdb.factory import VectorStoreFactory
from ..retrieval.retriever import aretrieve_documents  # 使用异步检索
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
    """显示结果。"""
    if not documents:
        console.print(Panel(f"对查询 “[bold yellow]{query}[/bold yellow]” 无结果。", border_style="red"))
        return

    table = Table(title=f"“{query}” 召回测试", show_header=True)
    table.add_column("排名", width=4)
    table.add_column("内容")
    table.add_column("来源", width=30)
    table.add_column("分数", width=10)

    for i, doc in enumerate(documents):
        content = doc.get('page_content', 'N/A').replace('\n', ' ')[:150] + "..."
        source = doc.get('metadata', {}).get('source', 'N/A')
        score = doc.get('score', 0.0)
        table.add_row(str(i + 1), content, source, f"{score:.4f}")
    console.print(table)

async def run_retrieval_test_async():
    """异步执行召回测试。"""
    console.print(Panel("进入异步召回测试模式。输入 '/quit' 退出。", border_style="yellow"))
    
    excel_logger = None
    try:
        excel_logger = ExcelLogger()
    except Exception as e:
        logger.error(f"ExcelLogger 初始化失败: {e}")

    try:
        settings = get_settings()
        vector_store = VectorStoreFactory.get_default_vector_store()
    except Exception as e:
        console.print(f"[red]数据库加载失败: {e}[/red]")
        return

    while True:
        try:
            query = prompt(HTML('<deepskyblue><b>测试查询: </b></deepskyblue>'))
            if query.lower() == '/quit':
                break
            if not query: continue

            logger.info(f"开启异步检索测试: '{query}'")
            
            with console.status("[bold green]正在异步检索...[/bold green]"):
                # 调用异步检索
                documents = await aretrieve_documents(
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
            
            display_results(query, documents)

            if excel_logger:
                excel_logger.log_results(query, documents)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            logger.error(f"召回测试出错: {e}")

def run_retrieval_test():
    """同步入口。"""
    asyncio.run(run_retrieval_test_async())
