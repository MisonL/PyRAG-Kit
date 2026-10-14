import logging
import os
import re  # 导入 re 模块
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.utils.config import get_settings
from src.utils.security import redact_sensitive_text, safe_exception_text

# 日志目录不可写时，降级 warning 只打一次。
#
# ``get_module_logger`` 被 20 多个模块在 import 期调用，每个模块都会走一遍
# 文件 handler 的创建。若各自打一条，只读安装下用户会看到一长串重复的
# 「无法写入日志目录」，把真正的问题埋掉。
_DEGRADED_WARNING_EMITTED = False

# 本应用写出的日志文件名前缀。写入（``_build_rotating_handler``）与清理
# （``cleanup_old_logs``）共用同一份定义，避免清理规则与命名规则各自演化。
APP_LOG_PREFIX = "app_log"
CHAT_LOG_PREFIX = "chat_log"
# 清理认 ``<前缀>_<YYYY-MM-DD>.log`` 以及它经 ``RotatingFileHandler`` 轮转出来的
# 备份 ``<前缀>_<YYYY-MM-DD>.log.<序号>``。
#
# 备份后缀是必需的，不是锦上添花：写入侧用 ``RotatingFileHandler(maxBytes=1MB,
# backupCount=5)``，单个文件超过 1 MB 就轮转出 ``.log.1`` ~ ``.log.5``。旧模式
# 以 ``$`` 结尾，那些备份**一个都不匹配**，于是「清理旧日志」只删掉主文件，
# 把绝大部分体积留在盘上——实测造一批 30 天前的日志（主文件 + 5 个备份），
# 清理后仍剩 83% 的字节。而 ``log_path`` 是用户可配置项，指向磁盘时这份残留
# 会一直累积。
#
# 后缀限定为 ``.`` + 数字：``RotatingFileHandler`` 的备份名就是
# ``baseFilename + "." + 序号``（``doRollover`` 里 ``dfn = self.rotation_filename(
# self.baseFilename + "." + str(i))``），不会产生其它形态。
_LOG_FILENAME_PATTERN = re.compile(
    rf"^(?:{APP_LOG_PREFIX}|{CHAT_LOG_PREFIX})_(\d{{4}}-\d{{2}}-\d{{2}})\.log(?:\.\d+)?$"
)


def _warn_once(logger: logging.Logger, message: str, *args: object) -> None:
    """只在首次调用时发出降级 warning。"""
    global _DEGRADED_WARNING_EMITTED
    if _DEGRADED_WARNING_EMITTED:
        return
    _DEGRADED_WARNING_EMITTED = True
    logger.warning(message, *args)


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

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = RedactingFormatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        # 与 ``get_module_logger`` 同理：日志目录不可写时降级为仅控制台，
        # 不因诊断能力缺失而中断查询。
        try:
            file_handler = _build_rotating_handler(current_settings.log_path, CHAT_LOG_PREFIX)
        except (OSError, ValueError) as exc:
            # ``ValueError`` 也要接：``os.makedirs`` 对含 NUL 字节的路径抛
            # ``ValueError: embedded null byte``（**不是** ``OSError``）。配置层的
            # ``resolve_path`` 现在已拒绝 NUL，这里是纵深防御——路径字段有六个，
            # 任何一条来源将来都可能再引入这个形态。
            _warn_once(logger, "无法写入日志目录，本次运行的日志仅输出到控制台: %s", exc)
        else:
            logger.addHandler(file_handler)

    return logger


def get_module_logger(name: str) -> logging.Logger:
    """
    获取并配置模块日志记录器。
    日志将输出到控制台和文件。
    """
    logger = logging.getLogger(name)
    try:
        current_settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - 日志配置在 import 期不能让进程崩
        # 本函数在模块导入期就会被调用（``src/utils/cleanup.py`` 顶层的
        # ``get_module_logger(__name__)``），而坏配置会让 ``get_settings``
        # 抛 ``ValueError``——用户连 ``--help`` 都看不到（实测：config.toml
        # 加一行语法错误后 ``main.py --help`` 以裸 traceback 收场）。日志
        # 级别取不到就退回默认值，主流程稍后调用 ``get_settings()`` 时
        # 仍会以脱敏的配置错误正常报错。
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.warning("配置尚未可用，日志按默认级别初始化: %s", safe_exception_text(exc))
        return logger
    logger.setLevel(current_settings.log_level)  # 从配置中获取日志级别

    # 避免重复添加处理器
    if not logger.handlers:
        current_settings = get_settings()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = RedactingFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器 (每天一个文件，最大 1MB，保留 5 个文件)
        #
        # 本函数在模块导入期就会被调用（``src/utils/cleanup.py`` 顶层的
        # ``get_module_logger(__name__)``），而默认的 ``log_path`` 解析到
        # 应用根目录之下。安装在 ``/opt``、``/Applications``、
        # ``C:\Program Files`` 这类只读位置时，``makedirs`` 会抛
        # ``PermissionError``，把整个进程挡在 import 阶段——连入口的脱敏
        # 边界都还没建立。日志属于可选的诊断能力，不可写时降级为仅控制台，
        # 而不是让程序无法启动。
        try:
            file_handler = _build_rotating_handler(
                current_settings.log_path, APP_LOG_PREFIX, with_location=True
            )
        except (OSError, ValueError) as exc:
            # ``ValueError`` 也要接：``os.makedirs`` 对含 NUL 字节的路径抛
            # ``ValueError: embedded null byte``（**不是** ``OSError``），因此
            # 只接 ``OSError`` 时这条降级路径会被绕过——而本函数在**模块导入期**
            # 就被 ``cleanup.py`` 调用，异常会让 ``import main`` 直接失败。
            # 配置层的 ``resolve_path`` 现在已拒绝 NUL，这里是纵深防御。
            _warn_once(logger, "无法写入日志目录，本次运行仅输出到控制台: %s", exc)
        else:
            logger.addHandler(file_handler)

    return logger


def _build_rotating_handler(log_dir: str, prefix: str, *, with_location: bool = False):
    """构建按天滚动的文件处理器；目录不可创建/不可写时抛 ``OSError``。

    含 NUL 字节的路径抛的是 ``ValueError``（不是 ``OSError``），调用方两者都接。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"{prefix}_{datetime.now().astimezone().strftime('%Y-%m-%d')}.log"
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file_name),
        maxBytes=1 * 1024 * 1024,  # 1 MB
        backupCount=5,
        encoding="utf-8",
    )
    if with_location:
        pattern = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    else:
        pattern = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler.setFormatter(RedactingFormatter(pattern))
    return file_handler


def cleanup_old_logs():
    """
    根据配置的保留天数清理旧的日志文件。
    """
    current_settings = get_settings()
    log_dir = current_settings.log_path
    log_retention_days = current_settings.log_retention_days

    # ``exists`` 对**文件**也返回 True，于是下面 ``os.listdir`` 抛
    # ``NotADirectoryError``。``log_path`` 是用户可配置项，指向一个已存在的
    # 普通文件完全可能（写入侧对此的处置是降级为仅控制台，见
    # ``_build_rotating_handler`` 的 ``except (OSError, ValueError)``），而本函数
    # 由退出钩子调用——异常会让程序在退出路径上以裸 traceback 终止。清理不是
    # 必须完成的能力，形态不对就跳过。
    if not os.path.isdir(log_dir):
        return

    now = datetime.now().astimezone()

    # 只清理本应用写出的日志。
    #
    # 这里原先匹配「目录下所有 *.log，且文件名含任意 YYYY-MM-DD」，而
    # ``log_path`` 是用户可配置项：把它指向共享目录（``/var/log``、某个
    # 挂载的日志盘）时，退出钩子会连带删掉别的程序与用户的日志——实测
    # ``nginx_access_2020-01-01.log``、``postgres_2020-01-01.log`` 乃至
    # ``backup_2020-01-01.log`` 都被删掉，且 ``rmtree`` 之外的这个删除同样
    # 不可逆。现在要求文件名严格等于本应用的命名形态
    # （``app_log_<日期>.log`` / ``chat_log_<日期>.log``），与写入侧同源。
    for filename in os.listdir(log_dir):
        match = _LOG_FILENAME_PATTERN.match(filename)
        if not match:
            # 别的程序或用户的文件：不是本应用的清理对象，也不值得每个都告警。
            continue
        file_path = os.path.join(log_dir, filename)
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            if (now - file_date).days > log_retention_days:
                os.remove(file_path)
                get_module_logger(__name__).info(f"已删除旧日志文件: {filename}")
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
