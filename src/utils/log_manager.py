import logging
import os
import re  # 导入 re 模块
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.utils.config import get_settings
from src.utils.security import redact_sensitive_text


class RedactingFormatter(logging.Formatter):
    """在格式化完整 traceback 后脱敏，避免异常链泄漏凭证。

    ``logging.Formatter.format`` 会把格式化后的 traceback 缓存在
    ``record.exc_text`` 上，后续 handler 直接复用该缓存。如果本 formatter 只是
    在 ``super().format()`` 之后脱敏，缓存里留下的仍是未脱敏文本，随后执行的
    非脱敏 handler（例如宿主应用自行挂在 root logger 上的 handler）就会拿到原始
    凭证。因此这里在格式化前清空缓存、在脱敏后写回，使缓存内容本身也是脱敏的。

    边界：Python logging 的 handler 顺序决定，只有在脱敏 handler 先于其它
    handler 执行时才能保护它们。项目自身的 logger 满足该前提（子 logger 的
    handler 先执行，再传播到 root）；若宿主在同一个 logger 上把普通 handler
    排在脱敏 handler 之前，那个 handler 已先输出了原始内容，无法追回。
    """

    def format(self, record: logging.LogRecord) -> str:
        record.exc_text = None
        formatted = super().format(record)
        # Formatter 会把 traceback 写回 ``record.exc_text``；用脱敏结果覆盖，
        # 让后续 handler 复用到的缓存也是脱敏后的文本。
        if record.exc_text is not None:
            record.exc_text = redact_sensitive_text(record.exc_text)
        return redact_sensitive_text(formatted)


def get_chat_logger() -> logging.Logger:
    """
    获取并配置聊天日志记录器。
    日志将输出到控制台和文件。
    """
    logger = logging.getLogger("chat_logger")
    current_settings = get_settings()
    logger.setLevel(current_settings.log_level)  # 从配置中获取日志级别

    # 避免重复添加处理器
    if not logger.handlers:
        current_settings = get_settings()
        log_dir = current_settings.log_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = RedactingFormatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        log_file_name = f"chat_log_{datetime.now().astimezone().strftime('%Y-%m-%d')}.log"
        file_path = os.path.join(log_dir, log_file_name)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=1 * 1024 * 1024,  # 1 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = RedactingFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    logger.setLevel(current_settings.log_level)  # 从配置中获取日志级别

    # 避免重复添加处理器
    if not logger.handlers:
        current_settings = get_settings()
        log_dir = current_settings.log_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = RedactingFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        log_file_name = f"app_log_{datetime.now().astimezone().strftime('%Y-%m-%d')}.log"
        file_path = os.path.join(log_dir, log_file_name)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=1 * 1024 * 1024,  # 1 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = RedactingFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
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

    now = datetime.now().astimezone()

    # 获取所有日志文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]

    for filename in log_files:
        file_path = os.path.join(log_dir, filename)
        try:
            # 从文件名中解析日期，例如 "chat_log_2023-10-26.log" 或 "app_log_2023-10-26.log"
            match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
            if match:
                file_date_str = match.group(1)
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(tzinfo=now.tzinfo)

                if (now - file_date).days > log_retention_days:
                    os.remove(file_path)
                    get_module_logger(__name__).info(f"已删除旧日志文件: {filename}")
            else:
                # 如果文件名不符合日期模式，也记录一下，但不删除
                get_module_logger(__name__).warning(
                    f"日志文件名不符合日期模式，跳过清理: {filename}"
                )
        except Exception:  # noqa: BLE001 - cleanup continues independently per file
            get_module_logger(__name__).exception("清理日志文件 %s 时出错", filename)


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
