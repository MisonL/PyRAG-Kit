# -*- coding: utf-8 -*-
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.snapshot_repository import SnapshotRepository
from src.retrieval.vdb.factory import VectorStoreFactory
from src.runtime.contracts import build_run_config
from src.services.embedding_service import EmbeddingService
from src.services.knowledge_build_service import KnowledgeBuildService
from src.ui.display_utils import CONSOLE_WIDTH, get_relative_path
from src.utils.config import get_settings
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)


def display_config_and_confirm(splitter_structure_mode: str):
    console = Console()
    settings = get_settings()
    run_config = build_run_config(settings)

    def mask_api_key(key: Optional[str]) -> str:
        if not key or key == "no-key-required":
            return "[dim]未设置或无需设置[/dim]"
        if len(key) > 12:
            return f"[white]{key[:6]}...{key[-4:]}[/white]"
        return "[white]已设置[/white]"

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
    table.add_row("知识库目录", f"[bold cyan]{get_relative_path(str(run_config.knowledge_base_path))}[/bold cyan]")
    table.add_row("快照根目录", f"[bold cyan]{get_relative_path(str(run_config.snapshot_root))}[/bold cyan]")
    table.add_row("旧版 pkl 路径", f"[bold cyan]{get_relative_path(str(run_config.legacy_pkl_path))}[/bold cyan]")
    table.add_row("索引模式", f"[bold magenta]{splitter_structure_mode}[/bold magenta]")
    table.add_row("文本切分块大小", f"[bold magenta]{run_config.kb_chunk_size}[/bold magenta]")
    table.add_row("切分重叠量", f"[bold magenta]{run_config.kb_chunk_overlap}[/bold magenta]")
    table.add_row("子分段块大小", f"[bold magenta]{run_config.kb_child_chunk_size}[/bold magenta]")
    table.add_row("子分段重叠量", f"[bold magenta]{run_config.kb_child_chunk_overlap}[/bold magenta]")
    table.add_row("嵌入批大小", f"[bold magenta]{run_config.kb_embedding_batch_size}[/bold magenta]")

    embedding_key = run_config.default_embedding_provider
    embedding_detail = run_config.embedding_configurations[embedding_key]
    provider = embedding_detail.provider
    table.add_section()
    table.add_row("[bold green]模型与 API 配置[/bold green]", "")
    table.add_row("激活嵌入提供商", f"[bold green]{embedding_key}[/bold green] ([dim]{provider}[/dim])")
    table.add_row("模型名称", f"[bold bright_white]{embedding_detail.model_name}[/bold bright_white]")
    if provider == "local-hash":
        table.add_row("API Key", "[dim]本地模型，无需设置[/dim]")
    else:
        key_field_name = f"{provider.lower()}_api_key"
        api_key_value = getattr(settings, key_field_name, None)
        table.add_row(f"API Key ({key_field_name.upper()})", mask_api_key(api_key_value))

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
    result = await build_service.build(splitter_structure_mode)

    console.print(
        f"\n[bold green]知识库构建完成。活动快照: [cyan]{result['snapshot_id']}[/cyan][/bold green]"
    )
    console.print(f"[green]快照目录: [cyan]{result['snapshot_dir']}[/cyan][/green]")
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
