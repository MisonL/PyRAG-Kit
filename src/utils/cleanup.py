# -*- coding: utf-8 -*-
import os
import glob
import atexit
import shutil
from rich.console import Console
from .config import CACHE_PATH

console = Console()

def cleanup_temp_files():
    """
    在程序退出时清理由本程序创建的 .cache 目录。
    """
    console.print("\n[dim]执行退出前清理任务...[/dim]")
    
    cache_dir = CACHE_PATH
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            console.print(f"[green]已成功删除缓存目录: {cache_dir}[/green]")
        except OSError as e:
            console.print(f"[red]删除缓存目录 {cache_dir} 时出错: {e}[/red]")
    else:
        console.print("[dim]未找到 .cache 目录，无需清理。[/dim]")

# 注册函数，使其在程序正常退出时被调用
atexit.register(cleanup_temp_files)