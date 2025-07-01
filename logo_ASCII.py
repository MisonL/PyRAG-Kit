# -*- coding: utf-8 -*-
from rich.console import Console
from rich.text import Text
import pyfiglet
from typing import Tuple

console = Console()

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
    # 使用 'big' 字体
    fig = pyfiglet.Figlet(font='big')
    banner_text = fig.renderText('PyRAG-Kit')
    
    # 定义渐变色 (左蓝右红)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    
    gradient_banner = create_gradient(banner_text, blue, red)
    console.print(gradient_banner)

if __name__ == "__main__":
    display_banner()