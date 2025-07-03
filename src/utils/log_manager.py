# -*- coding: utf-8 -*-
import logging
import os
import re # 导入 re 模块
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.utils.config import get_settings

def get_chat_logger() -> logging.Logger:
    """
    获取并配置聊天日志记录器。
    日志将输出到控制台和文件。
    """
    logger = logging.getLogger("chat_logger")
    current_settings = get_settings()
    logger.setLevel(current_settings.log_level) # 从配置中获取日志级别

    # 避免重复添加处理器
    if not logger.handlers:
        current_settings = get_settings()
        log_dir = current_settings.log_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        log_file_name = f"chat_log_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_path = os.path.join(log_dir, log_file_name)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=1 * 1024 * 1024, # 1 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_module_logger(name: str) -> logging.Logger:
    """
    获取并配置模块日志记录器。
    日志将输出到控制台和文件。
    """
    logger = logging.getLogger(name)
    current_settings = get_settings()
    logger.setLevel(current_settings.log_level) # 从配置中获取日志级别

    # 避免重复添加处理器
    if not logger.handlers:
        current_settings = get_settings()
        log_dir = current_settings.log_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        log_file_name = f"app_log_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_path = os.path.join(log_dir, log_file_name)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=1 * 1024 * 1024, # 1 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def cleanup_old_logs():
    """
    根据配置的保留天数清理旧的日志文件。
    """
    current_settings = get_settings()
    log_dir = current_settings.log_path
    log_retention_days = current_settings.log_retention_days
    
    if not os.path.exists(log_dir):
        return

    now = datetime.now()
    
    # 获取所有日志文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    
    for filename in log_files:
        file_path = os.path.join(log_dir, filename)
        try:
            # 从文件名中解析日期，例如 "chat_log_2023-10-26.log" 或 "app_log_2023-10-26.log"
            match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if match:
                file_date_str = match.group(1)
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                
                if (now - file_date).days > log_retention_days:
                    os.remove(file_path)
                    get_module_logger(__name__).info(f"已删除旧日志文件: {filename}")
            else:
                # 如果文件名不符合日期模式，也记录一下，但不删除
                get_module_logger(__name__).warning(f"日志文件名不符合日期模式，跳过清理: {filename}")
        except Exception as e:
            get_module_logger(__name__).error(f"清理日志文件 {filename} 时出错: {e}", exc_info=True)

# 示例用法 (可选，用于测试)
if __name__ == "__main__":
    chat_logger = get_chat_logger()
    chat_logger.info("这是一条聊天信息。")
    chat_logger.warning("这是一条聊天警告。")

    module_logger = get_module_logger(__name__)
    module_logger.info("这是一条模块信息。")
    module_logger.error("这是一条模块错误！")
    
    # 测试日志清理
    # 为了测试，可以临时修改 log_retention_days 为一个很小的值，并创建一些旧文件
    # cleanup_old_logs()
