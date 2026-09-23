import atexit
import os
import shutil
from pathlib import Path

from .config import get_settings  # 导入 get_settings 函数
from .log_manager import get_module_logger  # 导入日志管理器
from .security import redact_sensitive_text

logger = get_module_logger(__name__)  # 获取当前模块的日志器


def _reject_unsafe_cache_dir(cache_dir: str) -> str | None:
    """判断 cache 目录是否可安全删除；不可删时返回原因，可删返回 None。

    ``cache_path`` 是用户可配置项，而 ``shutil.rmtree`` 的后果不可逆：实测
    ``Settings(cache_path="/")``、``"/etc"``、``"~"``、``"../.."`` 全部被接受，
    随后 ``cleanup_temp_files()`` 会把整个目录删掉——它已在 atexit 注册，每次
    程序正常退出都会执行。

    这里只做最低限度的常识判断，不试图猜用户意图：根目录、家目录、以及常见的
    系统目录一律不删，并把判定结果写进日志。
    """
    if not cache_dir:
        return "cache_path 为空"
    resolved = Path(cache_dir).resolve()
    if resolved == Path(resolved.anchor):
        return f"{resolved} 是文件系统根目录"
    home = Path.home().resolve()
    if resolved == home:
        return f"{resolved} 是当前用户的家目录"
    protected_roots = [
        Path("/etc"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/var"),
        Path("/System"),
        Path("/Library"),
        Path("/Applications"),
    ]
    for protected in protected_roots:
        if resolved == protected.resolve():
            return f"{resolved} 是系统目录"
    # 目录名不是 .cache 时只警告不阻断：用户可能有意把缓存放到别处
    # （例如 config.toml 里写 cache_path = "data/cache"）。
    return None


def cleanup_temp_files():
    """在程序退出时清理由本程序创建的缓存目录。

    删除前先做安全判断：``cache_path`` 可被用户配置成任意路径，而这里的
    ``rmtree`` 在 atexit 里执行、后果不可逆，不能无条件照做。
    """
    logger.info("执行退出前清理任务...")

    current_settings = get_settings()
    cache_dir = current_settings.cache_path

    unsafe_reason = _reject_unsafe_cache_dir(cache_dir)
    if unsafe_reason is not None:
        logger.error(
            "拒绝清理缓存目录，因其不是安全的删除目标: %s（%s）。 请检查配置项 cache_path。",
            cache_dir,
            unsafe_reason,
        )
        return

    if not os.path.exists(cache_dir):
        logger.info("未找到 .cache 目录，无需清理。")
        return

    if not os.path.isdir(cache_dir):
        logger.warning("缓存路径不是目录，跳过清理: %s", cache_dir)
        return

    try:
        shutil.rmtree(cache_dir)
        logger.info(f"已成功删除缓存目录: {cache_dir}")
    except OSError as e:
        error_text = redact_sensitive_text(str(e))
        logger.exception(
            "删除缓存目录 %s 时出错: %s",
            cache_dir,
            error_text,
        )


# 注册函数，使其在程序正常退出时被调用
atexit.register(cleanup_temp_files)
