import argparse
import asyncio
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.snapshot_repository import SnapshotRepository
from src.retrieval.vdb.factory import VectorStoreFactory
from src.runtime.contracts import build_run_config
from src.services.embedding_service import EmbeddingService
from src.services.knowledge_build_service import KnowledgeBuildService
from src.ui.display_utils import (
    CONSOLE_WIDTH,
    _require_model_detail,
    get_relative_path,
    markup_text,
)
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger
from src.utils.security import safe_exception_text

logger = get_module_logger(__name__)


def display_config_and_confirm(splitter_structure_mode: str):
    console = Console()
    settings = get_settings()
    run_config = build_run_config(settings)

    def mask_api_key(key: str | None) -> str:
        if not key or key == "no-key-required":
            return "[dim]未设置或无需设置[/dim]"
        return "[dim]已设置（值已隐藏）[/dim]"

    table = Table(
        box=box.ROUNDED,
        padding=(0, 2),
        title="[bold yellow]知识库快照构建配置总览[/bold yellow]",
        show_header=False,
        width=CONSOLE_WIDTH,
    )
    table.add_column(justify="right", style="cyan", no_wrap=True, width=28)
    table.add_column(style="bright_white")
    table.add_row("[bold green]知识库配置[/bold green]", "")
    table.add_row(
        "知识库目录",
        markup_text(
            "[bold cyan]{}[/bold cyan]",
            get_relative_path(str(run_config.knowledge_base_path)),
        ),
    )
    table.add_row(
        "快照根目录",
        markup_text("[bold cyan]{}[/bold cyan]", get_relative_path(str(run_config.snapshot_root))),
    )
    table.add_row(
        "旧版 pkl 路径",
        markup_text(
            "[bold cyan]{}[/bold cyan]",
            get_relative_path(str(run_config.legacy_pkl_path)),
        ),
    )
    table.add_row(
        "索引模式", markup_text("[bold magenta]{}[/bold magenta]", splitter_structure_mode)
    )
    table.add_row(
        "文本切分块大小",
        markup_text("[bold magenta]{}[/bold magenta]", run_config.kb_chunk_size),
    )
    table.add_row(
        "切分重叠量", markup_text("[bold magenta]{}[/bold magenta]", run_config.kb_chunk_overlap)
    )
    table.add_row(
        "子分段块大小",
        markup_text("[bold magenta]{}[/bold magenta]", run_config.kb_child_chunk_size),
    )
    table.add_row(
        "子分段重叠量",
        markup_text("[bold magenta]{}[/bold magenta]", run_config.kb_child_chunk_overlap),
    )
    table.add_row(
        "嵌入批大小",
        markup_text("[bold magenta]{}[/bold magenta]", run_config.kb_embedding_batch_size),
    )

    embedding_key = run_config.default_embedding_provider
    # 走共用助手而不是直接下标：``default_embedding_provider`` 没有 validator，
    # 打错字时裸下标抛 ``KeyError``，而 ``main()`` 只接 ``KeyboardInterrupt``
    # ——用户看到 traceback，报错里连可选值都没有。chat 路径已改用同一契约。
    embedding_detail = _require_model_detail(
        run_config.embedding_configurations, embedding_key, "Embedding"
    )
    provider = embedding_detail.provider
    table.add_section()
    table.add_row("[bold green]模型与 API 配置[/bold green]", "")
    table.add_row(
        "激活嵌入提供商",
        markup_text("[bold green]{}[/bold green] ([dim]{}[/dim])", embedding_key, provider),
    )
    table.add_row(
        "模型名称",
        markup_text("[bold bright_white]{}[/bold bright_white]", embedding_detail.model_name),
    )
    if provider == "local-hash":
        table.add_row("API Key", markup_text("[dim]本地模型，无需设置[/dim]"))
    else:
        # 行标签由 ``provider`` 拼出，而该值来自 ``config.toml``，本函数在
        # **任何校验之前**运行（``main_async`` 先调它、再建服务），因此标签里的
        # ``[/]`` 会让表格渲染抛 ``MarkupError``，把「确认配置」这一步变成崩溃。
        key_field_name = f"{provider.lower()}_api_key"
        api_key_value = getattr(settings, key_field_name, None)
        table.add_row(
            Text(f"API Key ({key_field_name.upper()})"), markup_text(mask_api_key(api_key_value))
        )

    console.print(table)
    console.print("[yellow]配置信息来源于 config.toml、.env 或环境变量。[/yellow]")
    choice = console.input("是否使用以上配置继续？ (y/n): ").lower()
    if choice not in ["y", "yes"]:
        console.print("[bold red]操作取消。[/bold red]")
        sys.exit(0)


async def main_async(splitter_structure_mode: str):
    console = Console()
    settings = get_settings()
    run_config = build_run_config(settings)
    display_config_and_confirm(splitter_structure_mode)

    vector_store = VectorStoreFactory.get_default_vector_store(load_existing=False)
    snapshot_repository = SnapshotRepository(run_config)
    embedding_service = EmbeddingService(run_config)
    build_service = KnowledgeBuildService(
        run_config=run_config,
        vector_store=vector_store,
        embedding_service=embedding_service,
        snapshot_repository=snapshot_repository,
    )
    result = None
    try:
        result = await build_service.build(splitter_structure_mode)
    finally:
        # 构建成功与失败（建库失败是常态路径）都要释放 provider：``build`` 内
        # ``embed_in_batches`` 懒加载出 provider（openai/google/ark 等渠道各持有
        # httpx AsyncClient 连接池），而菜单循环里每选一次「向量化处理」都新建
        # 一次 ``EmbeddingService``——不释放则连接池随菜单循环逐次累积。失败
        # 收场只记日志，不顶替构建本身的异常。判据与
        # ``retrieval_test.core._aclose_quietly`` 一致。
        try:
            await embedding_service.aclose()
        except Exception as exc:  # noqa: BLE001 - teardown 不得顶替主流程异常
            logger.warning("释放 Embedding 服务失败: %s", safe_exception_text(exc))

    console.print(
        markup_text(
            "\n[bold green]知识库构建完成。活动快照: [cyan]{}[/cyan][/bold green]",
            result["snapshot_id"],
        )
    )
    console.print(markup_text("[green]快照目录: [cyan]{}[/cyan][/green]", result["snapshot_dir"]))
    logger.info("知识库快照构建完成: %s", result)


def parse_args():
    parser = argparse.ArgumentParser(description="将知识库文档构建为本地知识快照。")
    parser.add_argument(
        "--mode",
        choices=["standard", "hierarchical"],
        default="standard",
        help="索引模式。standard 为单层分片，hierarchical 为父子分片。",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        asyncio.run(main_async(args.mode))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
