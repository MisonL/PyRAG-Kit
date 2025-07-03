# -*- coding: utf-8 -*-
import sys
import os
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import pyfiglet
from typing import Tuple
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# --- 绝对导入 ---
# 导入 cleanup 模块以注册 atexit 钩子
import src.utils.cleanup
# 从新的配置模块导入 settings 实例
from src.utils.config import get_settings
# 导入日志管理器，以便调用其清理函数
import src.utils.log_manager
# 导入UI工具
from src.ui.display_utils import CONSOLE_WIDTH

console = Console()
VERSION = "1.2.0" # 程序版本

# =================================================================
# 应用程序界面 (APP UI)
# =================================================================
def create_gradient(text: str, start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]) -> Text:
    """为文本创建从左到右的水平颜色渐变效果。"""
    text_obj = Text()
    total_length = len(text)
    for i, char in enumerate(text):
        r = int(start_color[0] + (end_color[0] - start_color[0]) * (i / max(1, total_length - 1)))
        g = int(start_color[1] + (end_color[1] - start_color[1]) * (i / max(1, total_length - 1)))
        b = int(start_color[2] + (end_color[2] - start_color[2]) * (i / max(1, total_length - 1)))
        text_obj.append(char, style=f"rgb({r},{g},{b})")
    return text_obj

def display_banner():
    """显示程序的启动横幅。"""
    # 使用 'big' 字体，它是 standard 的加粗和放大版本，清晰且有冲击力
    fig = pyfiglet.Figlet(font='big')
    banner_text = fig.renderText('PyRAG-Kit')
    
    # 定义渐变色 (左蓝右红)
    blue_rgb = (0, 0, 255)
    red_rgb = (255, 0, 0)

    lines = banner_text.splitlines()
    banner_width = max(len(l) for l in lines) if lines else CONSOLE_WIDTH

    # 准备署名文本，使用Dify官方的黑(白)蓝配色并增加括号
    attribution_text = Text("inspired by [", style="white")
    attribution_text.append("d", style="bold white")
    attribution_text.append("if", style="bold bright_blue")
    attribution_text.append("y", style="bold white")
    attribution_text.append("]", style="white")

    # 逐行打印，以便对特定行进行特殊处理
    for i, line in enumerate(lines):
        # 如果是倒数第二行，添加署名
        if i == len(lines) - 2:
            line_content = line.rstrip()
            gradient_part = create_gradient(line_content, blue_rgb, red_rgb)
            
            # 计算填充，确保署名在右下角对齐
            padding_size = banner_width - len(line_content) - len(attribution_text)
            if padding_size < 1:
                padding_size = 1
            
            padding = Text(" " * padding_size)
            
            # 组合并打印该行
            console.print(gradient_part + padding + attribution_text)
        else:
            # 其他行正常打印渐变效果
            console.print(create_gradient(line, blue_rgb, red_rgb))
    
    # 构建包含丰富链接和信息的欢迎面板
    welcome_text = Text(justify="center")
    welcome_text.append(f"欢迎使用 PyRAG-Kit - 版本 {VERSION}\n", style="bold cyan")
    welcome_text.append("一个 Dify 核心逻辑的 Python 实现，用于本地验证其知识库向量化、分段及检索流程。\n\n", style="dim")
    welcome_text.append("作者: ", style="bold")
    welcome_text.append("Mison", style="default")
    welcome_text.append("  ·  邮箱: ", style="bold")
    welcome_text.append("1360962086@qq.com", style="default")
    welcome_text.append("\n") # 换行
    welcome_text.append("GitHub: ", style="bold")
    # 使用正确的 GitHub 仓库地址
    github_url = "https://github.com/MisonL/PyRAG-Kit"
    welcome_text.append(github_url, style=f"link {github_url}")

    # 设置面板宽度与 banner 宽度一致
    console.print(Panel(welcome_text, border_style="green", width=banner_width))

def display_menu():
    """使用rich库显示美化的交互式菜单。"""
    menu_content = (
        "请选择要执行的操作:\n"
        "[bold cyan]1.[/bold cyan] 知识库文档向量化处理\n"
        "[bold cyan]2.[/bold cyan] 召回测试\n"
        "[bold cyan]3.[/bold cyan] 启动聊天机器人会话\n"
        "[bold cyan]4.[/bold cyan] 退出程序"
    )
    console.print(Panel(menu_content, title="[bold yellow]主菜单[/bold yellow]", border_style="green", expand=False, highlight=True))

# =================================================================
# 应用程序预热 (APP WARM-UP)
# =================================================================
def initialize_dependencies():
    """
    在程序启动时预热耗时较长的库，并抑制其输出，提升用户体验。
    同时，主动管理缓存文件的位置。
    """
    console.print("[dim]正在初始化依赖项...[/dim]")
    
    # 1. 执行日志清理
    src.utils.log_manager.cleanup_old_logs()
    
    # 2. 使用从 settings 实例获取的缓存目录
    cache_dir = get_settings().cache_path
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

def main():
    """程序主入口，提供交互式菜单并实现延迟加载。"""
    display_banner()
    initialize_dependencies()
    while True:
        display_menu()
        try:
            # 使用 prompt_toolkit 替代 console.input，并优化样式
            choice = prompt(HTML('<skyblue><b>请输入选项 (1-4): </b></skyblue>'))
            if choice == '1':
                console.print(Panel("[bold green]开始执行知识库文档向量化处理...[/bold green]", border_style="green", width=CONSOLE_WIDTH))
                # 延迟加载和执行
                from scripts.embed_knowledge_base import main as run_embedding_process
                run_embedding_process()
                console.print(Panel("[bold green]向量化处理完成。[/bold green]", border_style="green", width=CONSOLE_WIDTH))
            elif choice == '2':
                console.print(Panel("[bold green]开始执行召回测试...[/bold green]", border_style="green", width=CONSOLE_WIDTH))
                # 延迟加载和执行
                from src.retrieval_test.core import run_retrieval_test
                run_retrieval_test()
                console.print(Panel("[bold green]召回测试完成。[/bold green]", border_style="green", width=CONSOLE_WIDTH))
            elif choice == '3':
                console.print(Panel("[bold green]启动聊天机器人会话...[/bold green]", border_style="green", width=CONSOLE_WIDTH))
                # 延迟加载和执行
                from src.chat.core import start_chat_session
                start_chat_session()
                console.print(Panel("[bold green]聊天会话结束。[/bold green]", border_style="green", width=CONSOLE_WIDTH))
            elif choice == '4':
                console.print("[bold]正在退出程序... 再见！[/bold]")
                sys.exit(0)
            else:
                console.print("[bold red]无效的选项，请输入 1, 2, 3, 或 4。[/bold red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]检测到中断信号，正在退出程序... 再见！[/bold yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[bold red]程序运行期间发生错误:[/bold red] {e}")
            console.print("[bold red]请检查错误信息并重试。[/bold red]")

if __name__ == "__main__":
    main()