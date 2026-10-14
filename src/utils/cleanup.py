import atexit
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from .config import (  # 导入 get_settings 函数
    CONFIG_TOML_PATH,
    ENV_FILE_PATH,
    ROOT_DIR,
    get_settings,
)
from .log_manager import get_module_logger  # 导入日志管理器
from .security import safe_exception_text

logger = get_module_logger(__name__)  # 获取当前模块的日志器


def _protected_app_paths() -> list[Path]:
    """返回应用自身依赖的、绝不能被当作缓存删掉的路径。

    只挡住 ``/``、家目录与系统目录是不够的：``cache_path`` 也可以指向应用
    自己的目录。实测冻结产物中把 ``cache_path`` 配成 ``"."``（解析为应用根
    目录），退出钩子会把整个安装目录删掉——``.env``（内含 API Key）、
    ``knowledge_base/``（用户放进去的源文档）与 ``config.toml`` 一并消失，
    且无法恢复。``"data"``、``"knowledge_base"``、``"data/logs"`` 同理：
    它们分别覆盖快照目录、源文档与日志目录。

    因此判定不依赖目录名，而是拿应用真实使用的路径来比对。
    """
    candidates = [ROOT_DIR, CONFIG_TOML_PATH, ENV_FILE_PATH]
    settings = get_settings()
    # ``pkl_path`` 是 legacy 知识库文件（用户投入的原始数据），此前漏在清单外：
    # 把 ``cache_path`` 指向 ``data``、同时让 knowledge_base_path／snapshot_root／
    # log_path 都落在 ``data`` 之外时，``data`` 不覆盖任何受保护路径，
    # ``_reject_unsafe_cache_dir`` 放行，退出清理随即 ``rmtree("data")`` ——
    # 连带删掉 ``data/employee_kb.pkl``（实测复现）。它与 ``knowledge_base_path``
    # 同属「用户数据」，必须一并保护。
    for name in ("knowledge_base_path", "pkl_path", "snapshot_root", "log_path"):
        value = getattr(settings, name, None)
        if value:
            candidates.append(Path(value))
    return candidates


def _reject_unsafe_cache_dir(
    cache_dir: str,
    protected_paths: Iterable[Path] | None = None,
) -> str | None:
    """判断 cache 目录是否可安全删除；不可删时返回原因，可删返回 None。

    ``cache_path`` 是用户可配置项，而 ``shutil.rmtree`` 的后果不可逆：实测
    ``Settings(cache_path="/")``、``"/etc"``、``"~"``、``"../.."`` 全部被接受，
    随后 ``cleanup_temp_files()`` 会把整个目录删掉——它已在 atexit 注册，每次
    程序正常退出都会执行。

    这里只做最低限度的常识判断，不试图猜用户意图：根目录、家目录、常见系统
    目录，以及**应用自身依赖的路径**一律不删，并把判定结果写进日志。
    """
    if not cache_dir:
        return "cache_path 为空"
    try:
        resolved = Path(cache_dir).resolve()
    except (OSError, RuntimeError):
        # ``resolve()`` 对符号链接环抛 ``RuntimeError: Symlink loop from ...``。
        # 这一行此前没有守卫（下面遍历 ``protected_paths`` 的那轮有），于是
        # ``cache_path`` 指向链接环时本函数整体抛异常，而它在 atexit 里被调用
        # ——用户看到 ``Exception ignored in atexit callback`` 加 traceback，
        # **清理完全没做**。链接环不是可安全删除的目标（它甚至不是目录），
        # 因此与「不可删」同一收场，返回原因而不是抛。
        return f"{cache_dir} 无法解析（符号链接环或权限问题）"
    if resolved == Path(resolved.anchor):
        return f"{resolved} 是文件系统根目录"
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    if home is not None and resolved == home:
        return f"{resolved} 是当前用户的家目录"
    # 清单原先只有 ``/etc``／``/usr``／``/bin``／``/sbin``／``/var`` 与三个
    # macOS 目录，**漏掉了同样属于系统级、且删除后果不可逆的一批**。实测放行的
    # 包括：
    #
    #   * ``/tmp``（在 macOS 上解析成 ``/private/tmp``）——整个系统临时目录；
    #   * ``/Users``——macOS 上**全部用户家目录的父目录**，``rmtree`` 会删掉
    #     机器上每个账户的家目录；``/home`` 在 Linux 上同理；
    #   * ``/opt``——本文件的 docstring 自己就把「安装在 ``/opt``」列为场景，
    #     而它的父目录同样不该被删；
    #   * ``/dev``（设备节点）、``/proc``、``/sys``（内核虚拟文件系统）。
    #
    # 判据与上面几项一致：**相等**才算命中，因此 ``/Volumes/MyDrive/cache``、
    # ``/opt/myapp/cache`` 这类正常缓存位置不受影响（本函数从不过度拦截用户
    # 有意指定的缓存目录，见下方「目录名不是 .cache 时只警告不阻断」）。
    protected_roots = [
        Path("/etc"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/var"),
        Path("/System"),
        Path("/Library"),
        Path("/Applications"),
        # 系统级目录（此前遗漏）
        # ``nosec B108``：bandit 的 hardcoded_tmp_directory 规则按
        # ``context.string_val.startswith("/tmp")`` 匹配字面量，认不出这是
        # 「禁止删除的目录清单」而不是「用固定路径建临时文件」——本行既不
        # 创建也不写入任何文件，语义与 CWE-377 恰好相反。它是清单里唯一以
        # 临时目录前缀开头的项，因此只有这一处需要标注。
        Path("/tmp"),  # nosec B108 - 禁止删除清单项，非临时文件用法
        Path("/Users"),
        Path("/home"),
        Path("/opt"),
        Path("/boot"),
        Path("/dev"),
        Path("/proc"),
        Path("/sys"),
        Path("/run"),
        Path("/srv"),
        Path("/mnt"),
        Path("/media"),
        Path("/root"),
        # macOS 的外接卷挂载点；只挡挂载点本身，卷内的目录照常可用。
        Path("/Volumes"),
        # macOS 上 ``/etc``／``/var``／``/tmp`` 都指向它下面，挡一层总入口。
        Path("/private"),
    ]
    for protected in protected_roots:
        if resolved == protected.resolve():
            return f"{resolved} 是系统目录"

    # 应用自身的路径：相等（cache_path = "."）或位于其下（cache_path = "data"）
    # 都意味着删除会波及配置、源文档或已建好的知识快照。
    if protected_paths is None:
        protected_paths = _protected_app_paths()
    for protected in protected_paths:
        try:
            protected_resolved = Path(protected).resolve()
        except (OSError, RuntimeError):
            # ``RuntimeError`` 也要接：``Path.resolve()`` 对符号链接环抛
            # ``RuntimeError: Symlink loop from ...``，而 ``except OSError``
            # 接不住它——实测一条配置里的链接环让本函数整体抛异常，退出钩子
            # 以「Exception ignored in atexit callback」收场，清理**完全没做**。
            # 本注释原写「跳过该项，不阻断判定」，只有接住才成立。
            continue
        if protected_resolved == resolved or protected_resolved.is_relative_to(resolved):
            return f"{resolved} 覆盖了应用自身路径 {protected_resolved}"

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
        error_text = safe_exception_text(e)
        logger.exception(
            "删除缓存目录 %s 时出错: %s",
            cache_dir,
            error_text,
        )


# 注册函数，使其在程序正常退出时被调用
atexit.register(cleanup_temp_files)
