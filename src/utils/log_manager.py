# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime, timedelta
from rich.console import Console

# 使用相对导入从配置模块获取必要的设置
from .config import settings
from ..ui.display_utils import get_relative_path

console = Console()

def cleanup_old_logs():
    """
    自动清理并删除超过指定保留天数的旧日志文件。
    该函数会扫描日志目录，并根据文件名中编码的日期来判断日志是否过期。
    """
    # 确保日志目录存在，如果不存在则无需执行任何操作
    log_path_str = str(settings.log_path)
    if not os.path.exists(log_path_str):
        console.print(f"[dim]日志目录 '{get_relative_path(log_path_str)}' 不存在，跳过清理。[/dim]")
        return

    console.print(f"[dim]开始扫描并清理日志目录 '{get_relative_path(log_path_str)}'... (保留天数: {settings.log_retention_days})[/dim]")
    
    # 定义用于从文件名中提取日期的正则表达式
    # 匹配 'chat_log_YYYY-MM-DD.xlsx' 格式
    log_file_pattern = re.compile(r"chat_log_(\d{4}-\d{2}-\d{2})\.xlsx")
    
    deleted_count = 0
    checked_count = 0
    
    # 获取当前时间，用于计算过期阈值
    now = datetime.now()
    # 计算允许的最早日期，早于此日期的日志将被删除
    retention_delta = timedelta(days=settings.log_retention_days)
    cutoff_date = now - retention_delta

    # 遍历日志目录中的所有文件
    for filename in os.listdir(settings.log_path):
        checked_count += 1
        match = log_file_pattern.match(filename)
        
        # 如果文件名匹配日志格式
        if match:
            # 从文件名中提取日期字符串
            date_str = match.group(1)
            try:
                # 将日期字符串转换为datetime对象
                log_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # 如果日志文件的日期早于截止日期，则删除它
                if log_date < cutoff_date:
                    file_path = os.path.join(settings.log_path, filename)
                    os.remove(file_path)
                    deleted_count += 1
                    console.print(f"[yellow]已删除过期日志文件: {filename}[/yellow]")
            except ValueError:
                # 如果日期格式不正确，则忽略该文件
                console.print(f"[dim]文件名 '{filename}' 中的日期格式不正确，已跳过。[/dim]")

    if deleted_count > 0:
        console.print(f"[green]日志清理完成。共检查 {checked_count} 个文件，删除了 {deleted_count} 个过期日志。[/green]")
    else:
        console.print(f"[dim]日志清理完成。未发现需要删除的过期日志。[/dim]")
