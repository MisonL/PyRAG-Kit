# -*- coding: utf-8 -*-
import sys
import os
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from rich.console import Console
from rich.panel import Panel

# --- 绝对导入 ---
from src.utils import cleanup # 导入此模块以注册atexit钩子
from src.utils.config import CACHE_PATH

console = Console()

# =================================================================
# 应用程序预热 (APP WARM-UP)
# =================================================================
def initialize_dependencies():
    """
    在程序启动时预热耗时较长的库，并抑制其输出，提升用户体验。
    同时，主动管理缓存文件的位置。
    """
    console.print("[dim]正在初始化依赖项...[/dim]")
    
    # 1. 使用从配置中导入的缓存目录
    cache_dir = CACHE_PATH
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 2. 预热jieba并完全抑制其所有启动日志
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        import jieba
        import jieba.posseg as pseg
        
        jieba.setLogLevel(jieba.logging.ERROR)
        setattr(jieba.dt, 'tmp_dir', str(cache_dir))
        list(pseg.cut(""))

    console.print("[dim]依赖项初始化完成。[/dim]")

def display_menu():
    """使用rich库显示美化的交互式菜单。"""
    menu_content = (
        "请选择要执行的操作:\n"
        "[bold cyan]1.[/bold cyan] 知识库文档向量化处理\n"
        "[bold cyan]2.[/bold cyan] 启动聊天机器人会话\n"
        "[bold cyan]3.[/bold cyan] 退出程序"
    )
    console.print(Panel(menu_content, title="[bold yellow]主菜单[/bold yellow]", border_style="green", expand=False, highlight=True))

def main():
    """程序主入口，提供交互式菜单并实现延迟加载。"""
    initialize_dependencies()
    while True:
        display_menu()
        try:
            choice = console.input("[bold]请输入选项 (1-3):[/bold] ")
            if choice == '1':
                console.print(Panel("[bold green]开始执行知识库文档向量化处理...[/bold green]", border_style="green"))
                # 延迟加载和执行
                from scripts.embed_knowledge_base import main as run_embedding_process
                run_embedding_process()
                console.print(Panel("[bold green]向量化处理完成。[/bold green]", border_style="green"))
            elif choice == '2':
                console.print(Panel("[bold green]启动聊天机器人会话...[/bold green]", border_style="green"))
                # 延迟加载和执行
                from src.chat.core import start_chat_session
                start_chat_session()
                console.print(Panel("[bold green]聊天会话结束。[/bold green]", border_style="green"))
            elif choice == '3':
                console.print("[bold]正在退出程序... 再见！[/bold]")
                sys.exit(0)
            else:
                console.print("[bold red]无效的选项，请输入 1, 2, 或 3。[/bold red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]检测到中断信号，正在退出程序... 再见！[/bold yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[bold red]程序运行期间发生错误:[/bold red] {e}")
            console.print("[bold red]请检查错误信息并重试。[/bold red]")

if __name__ == "__main__":
    main()