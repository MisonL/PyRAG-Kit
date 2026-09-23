import asyncio
import inspect
import os
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

from ..retrieval.vdb.factory import VectorStoreFactory
from ..runtime.contracts import build_run_config, build_session_config
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..ui.display_utils import get_relative_path
from ..utils.config import get_settings
from ..utils.log_manager import get_module_logger
from ..utils.security import safe_exception_text
from .excel_logger import ExcelLogger

console = Console()
logger = get_module_logger(__name__)


def display_results(query: str, documents: list[dict]):
    if not documents:
        console.print(Panel(f"对查询 “[bold yellow]{query}[/bold yellow]” 无结果。", border_style="red"))
        return

    table = Table(title=f"“{query}” 召回测试", show_header=True)
    table.add_column("排名", width=4)
    table.add_column("内容")
    table.add_column("来源", width=30)
    table.add_column("分数", width=10)

    for index, doc in enumerate(documents):
        content = doc.get("page_content", "N/A").replace("\n", " ")[:150] + "..."
        source = get_relative_path(doc.get("metadata", {}).get("source", "N/A"))
        score = doc.get("score", 0.0)
        table.add_row(str(index + 1), content, source, f"{score:.4f}")
    console.print(table)



async def _aclose_quietly(service: Any) -> None:
    """释放服务持有的资源，失败只记录警告，不打断退出流程。"""
    close = getattr(service, "aclose", None) or getattr(service, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the main flow
        logger.warning("释放检索服务失败: %s", safe_exception_text(exc))


async def run_retrieval_test_async():
    console.print(Panel("进入异步召回测试模式。输入 '/quit' 退出。", border_style="yellow"))
    session = PromptSession()

    excel_logger = None
    try:
        excel_logger = ExcelLogger()
    except Exception as exc:  # noqa: BLE001 - interactive setup reports all failures
        logger.error("ExcelLogger 初始化失败: %s", safe_exception_text(exc))

    try:
        run_config = build_run_config(get_settings())
        session_config = build_session_config(get_settings())
        vector_store = VectorStoreFactory.get_default_vector_store()
        retrieval_service = RetrievalService(vector_store, EmbeddingService(run_config))
    except Exception as exc:  # noqa: BLE001 - interactive setup reports all failures
        console.print(f"[red]数据库加载失败: {safe_exception_text(exc)}[/red]")
        return

    try:
        while True:
            try:
                query = await session.prompt_async(HTML('<deepskyblue><b>测试查询: </b></deepskyblue>'))
                if query.lower() == "/quit":
                    break
                if not query:
                    continue

                with console.status("[bold green]正在异步检索...[/bold green]"):
                    documents = await retrieval_service.retrieve(query, session_config, console=console)
                display_results(query, documents)
                if excel_logger:
                    excel_logger.log_results(query, documents)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as exc:  # noqa: BLE001 - interactive loop remains usable
                console.print(f"[red]召回测试出错: {safe_exception_text(exc)}[/red]")
                logger.error("召回测试出错: %s", safe_exception_text(exc))
    finally:
        # 与 chat 路径保持一致：Embedding provider 与向量存储持有 SDK 客户端和
        # 连接池，退出时不释放会留下未关闭的资源。释放失败不掩盖主流程结果，
        # 只记录警告。
        await _aclose_quietly(retrieval_service)


def run_retrieval_test():
    asyncio.run(run_retrieval_test_async())
