# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import os
import re
import sys
import numpy as np
import pickle
from typing import Any, Optional, List, Dict

# --- 动态路径调整 ---
# 将项目根目录（即'scripts'目录的父目录）添加到sys.path
# 这样我们就可以从 src 目录导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# 从 src 导入重构后的模块
from src.utils.config import settings # 移除 KB_CONFIG, API_CONFIG
from src.providers.factory import ModelProviderFactory
from src.retrieval.vdb.base import VectorStoreBase # 导入 VectorStoreBase
from src.retrieval.vdb.factory import VectorStoreFactory # 导入 VectorStoreFactory
from src.ui.display_utils import CONSOLE_WIDTH, get_relative_path

# =================================================================
# 2. 文本处理 (TEXT PROCESSING)
# =================================================================
class Document(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)

class TextSplitter(ABC):
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, **kwargs: Any):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

class EnhanceRecursiveCharacterTextSplitter(TextSplitter):
    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        final_chunks = []
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if re.search(_s, text):
                separator = _s
                break
        
        splits = re.split(f'({separator})', text)
        _good_splits = [s for s in splits if s]
        
        _merger = ""
        for s in _good_splits:
            if len(_merger) + len(s) < self._chunk_size:
                _merger += s
            else:
                final_chunks.append(_merger)
                _merger = s
        final_chunks.append(_merger)
        return [chunk for chunk in final_chunks if chunk and chunk.strip()]

# =================================================================
# 3. 知识库核心 (KNOWLEDGE BASE CORE)
# =================================================================
def process_documents(vector_store: VectorStoreBase): # 更改类型提示
    console = Console()
    kb_dir = settings.knowledge_base_path # 使用 settings
    all_texts = []
    all_metadatas = []
    console.print("[bold]开始处理文档...[/bold]")
    for filename in os.listdir(kb_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(kb_dir, filename)
            console.print(f"  [cyan]正在读取文件:[/cyan] {filename}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                text_splitter = EnhanceRecursiveCharacterTextSplitter(
                    chunk_size=settings.kb_chunk_size, # 使用 settings
                    chunk_overlap=settings.kb_chunk_overlap, # 使用 settings
                    separators=settings.kb_splitter_separators # 使用 settings
                )
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({"source": filename})

    console.print(f"  [green]总共生成 [bold]{len(all_texts)}[/bold] 个文本块。[/green]")
    
    # 将文档内容和元数据存入 VectorStore 实例
    documents_to_embed = [{"page_content": text, "metadata": meta} for text, meta in zip(all_texts, all_metadatas)]
    vector_store.add_documents(documents_to_embed) # 调用 add_documents 方法

    console.print("  [bold]开始生成文本嵌入...[/bold]")
    # 嵌入生成逻辑已移至 FaissStore.add_documents
    
    # 保存到文件
    vector_store.save(settings.pkl_path) # 使用 vector_store.save 和 settings.pkl_path

    console.print(f"\n[bold green]知识库已成功处理并保存到 '[cyan]{settings.pkl_path}[/cyan]'。[/bold green]") # 使用 settings.pkl_path

# =================================================================
# 4. 辅助与主函数 (HELPERS & MAIN)
# =================================================================
def display_config_and_confirm():
    """以美观的表格形式显示全面的配置信息，并请求用户确认。"""
    console = Console()

    def mask_api_key(key: Optional[str]) -> str:
        """对API密钥进行脱敏处理，使其更安全地显示。"""
        if not key or key == "lm-studio":
            return "[dim]未设置或无需设置[/dim]"
        if len(key) > 12:
            return f"[white]{key[:6]}...{key[-4:]}[/white]"
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
    table.add_row("知识库目录", f"[bold cyan]{get_relative_path(settings.knowledge_base_path)}[/bold cyan]")
    table.add_row("输出文件路径", f"[bold cyan]{get_relative_path(settings.pkl_path)}[/bold cyan]")
    table.add_row("文本切分块大小 (Chunk Size)", f"[bold magenta]{settings.kb_chunk_size}[/bold magenta]")
    table.add_row("文本切分重叠量 (Overlap)", f"[bold magenta]{settings.kb_chunk_overlap}[/bold magenta]")
    table.add_row("切分分隔符 (Separators)", f"[bold bright_white]{settings.kb_splitter_separators}[/bold bright_white]")
    table.add_section()

    # --- 模型与API配置 ---
    table.add_row("[bold green]模型与API配置[/bold green]", "")
    active_embedding_key = settings.default_embedding_provider
    embedding_model_details = settings.embedding_configurations[active_embedding_key]
    provider = embedding_model_details.provider
    
    table.add_row("激活的嵌入模型 (Provider)", f"[bold green]{active_embedding_key}[/bold green] ([dim]{provider}[/dim])")
    table.add_row("模型名称 (Model Name)", f"[bold bright_white]{embedding_model_details.model_name}[/bold bright_white]")

    # 动态获取API Key和Base URL的字段名
    key_field_name = f"{provider.lower()}_api_key"
    
    # 处理不一致的URL字段名
    url_field_name = ""
    if provider == "openai":
        url_field_name = "openai_api_base"
    elif hasattr(settings, f"{provider.lower()}_base_url"):
        url_field_name = f"{provider.lower()}_base_url"

    api_key_value = getattr(settings, key_field_name, None)
    # mask_api_key 函数已内置样式，无需额外添加
    table.add_row(f"对应的 API Key ({key_field_name.upper()})", mask_api_key(api_key_value))

    if url_field_name:
        base_url_value = getattr(settings, url_field_name, None)
        if base_url_value:
            table.add_row(f"对应的 Base URL ({url_field_name.upper()})", f"[bold cyan]{base_url_value}[/bold cyan]")

    console.print(table)
    console.print("[yellow]配置信息来源于 [bold]config.ini[/bold], [bold].env[/bold] 文件或 [bold]环境变量[/bold]。[/yellow]")
    
    choice = console.input("是否使用以上配置继续处理？ ([bold green]y[/bold green]/[bold red]n[/bold red]): ").lower()
    if choice not in ['y', 'yes']:
        console.print("[bold red]操作已取消。[/bold red]")
        sys.exit(0)

def main():
    console = Console()
    display_config_and_confirm()
    
    kb_dir = settings.knowledge_base_path # 使用 settings
    if not os.path.exists(kb_dir) or not any(f.endswith('.md') for f in os.listdir(kb_dir)):
         console.print(f"[bold red]错误: 知识库目录 '{kb_dir}' 不存在或为空。请先添加Markdown文档。[/bold red]")
         sys.exit(1)

    vector_store = VectorStoreFactory.get_default_vector_store() # 通过工厂获取实例
    process_documents(vector_store)
    console.print("\n[bold green]知识库嵌入完成。现在可以启动聊天机器人会话进行测试。[/bold green]")

if __name__ == "__main__":
    main()
