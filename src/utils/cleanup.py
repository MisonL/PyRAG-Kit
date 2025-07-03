# -*- coding: utf-8 -*-
import os
import glob
import atexit
import shutil
from .config import get_settings # 导入 get_settings 函数
from .log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

def cleanup_temp_files():
    """
    在程序退出时清理由本程序创建的 .cache 目录。
    """
    logger.info("执行退出前清理任务...")
    
    # 从 get_settings() 获取缓存路径
    current_settings = get_settings()
    cache_dir = current_settings.cache_path
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"已成功删除缓存目录: {cache_dir}")
        except OSError as e:
            logger.error(f"删除缓存目录 {cache_dir} 时出错: {e}", exc_info=True)
    else:
        logger.info("未找到 .cache 目录，无需清理。")

# 注册函数，使其在程序正常退出时被调用
atexit.register(cleanup_temp_files)