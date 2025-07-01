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
from rich.console import Group

# 从 src 导入重构后的模块
from src.utils.config import KB_CONFIG, CHAT_CONFIG, API_CONFIG
from src.providers.model_providers import ModelFactory
from src.retrieval.retriever import VectorStore # VectorStore现在从这里导入

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
def process_documents(vector_store: VectorStore):
    console = Console()
    kb_dir = KB_CONFIG["kb_dir"]
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
                    chunk_size=KB_CONFIG["chunk_size"],
                    chunk_overlap=KB_CONFIG["chunk_overlap"],
                    separators=KB_CONFIG["text_splitter_separators"]
                )
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({"source": filename})

    console.print(f"  [green]总共生成 [bold]{len(all_texts)}[/bold] 个文本块。[/green]")
    
    # 将文档内容和元数据存入 VectorStore 实例
    documents_to_embed = [{"page_content": text, "metadata": meta} for text, meta in zip(all_texts, all_metadatas)]
    vector_store.documents = documents_to_embed
    
    console.print("  [bold]开始生成文本嵌入...[/bold]")
    embedding_provider = ModelFactory.get_model_provider("embedding")
    batch_size = KB_CONFIG["embedding_batch_size"]
    all_embeddings = []
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        batch_embeddings = embedding_provider.embed_documents(texts=batch_texts)
        all_embeddings.extend(batch_embeddings)
        console.print(f"    [dim]已处理 {min(i + batch_size, len(all_texts))} / {len(all_texts)} 个文本块。[/dim]")

    vector_store.embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # 保存到文件
    with open(vector_store.file_path, "wb") as f:
        pickle.dump({"documents": vector_store.documents, "embeddings": vector_store.embeddings}, f)

    console.print(f"\n[bold green]知识库已成功处理并保存到 '[cyan]{vector_store.file_path}[/cyan]'。[/bold green]")

# =================================================================
# 4. 辅助与主函数 (HELPERS & MAIN)
# =================================================================
def display_config_and_confirm():
    """显示全面的配置信息并请求用户确认。"""
    console = Console()

    def mask_api_key(key: Optional[str]) -> str:
        if not key: return "[dim]未设置[/dim]"
        return f"{key[:10]}..."

    kb_table = Table(title="[bold green]知识库构建配置 (KB_CONFIG)[/bold green]", show_header=False, box=None, padding=(0, 1))
    kb_table.add_column(style="cyan")
    kb_table.add_column(style="bold white")
    active_embedding = KB_CONFIG['active_embedding_configuration']
    embedding_model_details = KB_CONFIG['embedding_configurations'][active_embedding]
    kb_table.add_row("激活的嵌入模型:", f"{active_embedding} ({embedding_model_details['provider']}: {embedding_model_details['model_name']})")
    kb_table.add_row("知识库目录:", KB_CONFIG['kb_dir'])
    kb_table.add_row("输出文件:", KB_CONFIG['output_file'])
    
    api_table = Table(title="[bold green]相关API配置[/bold green]", show_header=False, box=None, padding=(0, 1))
    api_table.add_column(style="cyan")
    api_table.add_column(style="bold white")
    provider = embedding_model_details['provider']
    key_name = f"{provider.upper()}_API_KEY"
    url_name = f"{provider.upper()}_BASE_URL"
    if key_name in API_CONFIG:
        api_table.add_row(f"{key_name}:", mask_api_key(API_CONFIG.get(key_name)))
    if url_name in API_CONFIG and API_CONFIG.get(url_name):
        api_table.add_row(f"{url_name}:", API_CONFIG.get(url_name))

    console.print(Panel(Group(kb_table, api_table), title="[bold yellow]向量化脚本配置总览[/bold yellow]", border_style="blue"))
    console.print("[yellow]配置信息来源于 [bold]src/utils/config.py[/bold] 和 [bold].env[/bold] 文件。[/yellow]")
    
    choice = console.input("是否使用以上配置继续处理？ ([bold green]y[/bold green]/[bold red]n[/bold red]): ").lower()
    if choice not in ['y', 'yes']:
        console.print("[bold red]操作已取消。[/bold red]")
        sys.exit(0)

def main():
    console = Console()
    display_config_and_confirm()
    
    kb_dir = KB_CONFIG["kb_dir"]
    if not os.path.exists(kb_dir) or not any(f.endswith('.md') for f in os.listdir(kb_dir)):
         console.print(f"[bold red]错误: 知识库目录 '{kb_dir}' 不存在或为空。请先添加Markdown文档。[/bold red]")
         sys.exit(1)

    vector_store = VectorStore(KB_CONFIG["output_file"])
    process_documents(vector_store)
    console.print("\n[bold green]知识库嵌入完成。现在可以启动聊天机器人会话进行测试。[/bold green]")

if __name__ == "__main__":
    main()
