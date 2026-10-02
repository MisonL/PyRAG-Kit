import asyncio
import inspect
import os
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

from ..retrieval.vdb.factory import VectorStoreFactory
from ..runtime.contracts import build_run_config, build_session_config
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..ui.display_utils import (
    document_content,
    document_score,
    document_source,
    get_relative_path,
    markup_text,
)
from ..utils.config import get_settings
from ..utils.log_manager import get_module_logger
from ..utils.security import safe_exception_text
from .excel_logger import ExcelLogger

console = Console()
logger = get_module_logger(__name__)


def display_results(query: str, documents: list[dict]):
    # 形态先收敛，与同一主循环里的 ``ExcelLogger.log_results`` 同判据（两者拿的是
    # 同一个 ``documents``）。实测非列表取值两种收场都不在契约内：
    # ``"abc"`` 是合法可迭代对象，逐**字符**迭代后每个字符经 ``document_*`` 助手
    # 收敛成占位符，渲染出 3 行内容与来源都是 ``N/A`` 的伪结果表格（用户看到
    # 「召回了 3 条」，而实际返回的是字符串）；``5`` 抛裸
    # ``TypeError: 'int' object is not iterable``。按「非列表视为无结果」收敛。
    if not isinstance(documents, list):
        documents = []
    if not documents:
        console.print(
            Panel(
                markup_text("对查询 “[bold yellow]{}[/bold yellow]” 无结果。", query),
                border_style="red",
            )
        )
        return

    # 查询、来源与内容都来自用户输入或知识库文档，可能含字面量 ``[/]`` 这类
    # 标记形态；直接交给 rich 会被当作标记解析（轻则改写内容，重则抛
    # MarkupError，而该异常在异常处理通道里会二次抛出）。
    table = Table(title=markup_text("“{}” 召回测试", query), show_header=True)
    table.add_column("排名", width=4)
    table.add_column("内容")
    table.add_column("来源", width=30)
    table.add_column("分数", width=10)

    for index, doc in enumerate(documents):
        # 字段形态由 document_* 助手收敛：检索结果可以来自任何 VectorStoreBase
        # 实现，第三方 store 不会像 FaissStore 那样在载入期归一化。展示是错误
        # 处理之后的最后一环，不该成为新的失败点。
        content = document_content(doc).replace("\n", " ")[:150] + "..."
        source = get_relative_path(document_source(doc))
        table.add_row(str(index + 1), Text(content), Text(source), f"{document_score(doc):.4f}")
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
        # 必须同时告知用户：召回测试的**唯一产出**就是 Excel 日志，而这里失败
        # 之后每轮查询都走 ``if excel_logger:`` 静默跳过——用户会照常看到
        # 召回结果，以为测试记录下来了，实际一个文件都没有。本文件其它失败
        # 路径（数据库加载、服务初始化）都用 ``console.print`` 显式提示，这里
        # 此前只写日志，而日志级别由 ``log_level`` 决定，配成 ``ERROR`` 以上
        # 时连那行都看不到。
        logger.error("ExcelLogger 初始化失败: %s", safe_exception_text(exc))
        console.print(
            markup_text(
                "[yellow]Excel 日志初始化失败，本次召回测试不会记录结果: {}[/yellow]",
                safe_exception_text(exc),
            )
        )

    try:
        run_config = build_run_config(get_settings())
        session_config = build_session_config(get_settings())
        vector_store = VectorStoreFactory.get_default_vector_store()
        retrieval_service = RetrievalService(vector_store, EmbeddingService(run_config))
    except Exception as exc:  # noqa: BLE001 - interactive setup reports all failures
        console.print(markup_text("[red]数据库加载失败: {}[/red]", safe_exception_text(exc)))
        return

    try:
        while True:
            try:
                query = await session.prompt_async(
                    HTML("<deepskyblue><b>测试查询: </b></deepskyblue>")
                )
                if query.lower() == "/quit":
                    break
                if not query:
                    continue

                with console.status("[bold green]正在异步检索...[/bold green]"):
                    documents = await retrieval_service.retrieve(
                        query, session_config, console=console
                    )
                display_results(query, documents)
                if excel_logger:
                    excel_logger.log_results(query, documents)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as exc:  # noqa: BLE001 - interactive loop remains usable
                console.print(markup_text("[red]召回测试出错: {}[/red]", safe_exception_text(exc)))
                logger.error("召回测试出错: %s", safe_exception_text(exc))
    finally:
        # 与 chat 路径保持一致：Embedding provider 与向量存储持有 SDK 客户端和
        # 连接池，退出时不释放会留下未关闭的资源。释放失败不掩盖主流程结果，
        # 只记录警告。
        await _aclose_quietly(retrieval_service)


def run_retrieval_test():
    asyncio.run(run_retrieval_test_async())
