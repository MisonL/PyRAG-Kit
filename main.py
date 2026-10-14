import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pyfiglet
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

# 冻结运行时把 tiktoken 的缓存目录指向包内预置的 BPE 文件：tiktoken 在缓存
# 未命中时会联网拉取编码文件（tiktoken/load.py），离线环境下首次构建知识库
# 必然失败。构建脚本已把 cl100k_base 放进 ``<app_root>/tiktoken_cache``。
#
# 这里用赋值而非 ``setdefault``：tiktoken 的 ``read_file_cached`` 只看
# ``TIKTOKEN_CACHE_DIR`` **是否存在**，不存在「先查该目录、再回落别处」的
# 语义。用 setdefault 的话，环境里一旦已有该变量（CI、容器、用户显式设置），
# 包内预置的 BPE 就被完全旁路，离线运行退回联网并失败——预置形同虚设。
#
# 覆盖是安全的：全项目只有 ``cl100k_base`` 一种编码（见
# ``RecursiveTextSplitter.__init__`` 的默认值，无配置入口），而包内文件
# 就是它的 BPE；命中缓存时 tiktoken 只读不写，因此只读安装同样可用。
# 目录缺失时保持不设置，让 tiktoken 走自己的默认路径而不是被锁死在空目录；
# 但这是降级而非正常状态——包内文件是发布包唯一的离线兜底，缺了它就只能
# 联网，而 tiktoken 的 ``requests.get(blobpath)`` **没有超时**（见
# ``tiktoken/load.py`` 的 ``read_file``），网络黑洞下会长时间卡住而不是
# 快速失败。因此这里显式告知，避免用户面对一个没有线索的挂起。
if getattr(sys, "frozen", False):
    _bundled_tiktoken_cache = Path(sys.executable).resolve().parent / "tiktoken_cache"
    if _bundled_tiktoken_cache.is_dir():
        os.environ["TIKTOKEN_CACHE_DIR"] = str(_bundled_tiktoken_cache)
    else:
        print(
            f"警告: 未找到包内的分词编码缓存 {_bundled_tiktoken_cache}，"
            "首次分片将尝试联网获取 cl100k_base；"
            "若本机无法访问外网，请重新解压完整的发布包。",
            file=sys.stderr,
        )

# --- 绝对导入 ---
# 导入 cleanup 模块以注册 atexit 钩子
import src.utils.cleanup

# 导入日志管理器，以便调用其清理函数
import src.utils.log_manager

# 导入UI工具
from src.models.document import Document
from src.ui.display_utils import CONSOLE_WIDTH, markup_text

# 从新的配置模块导入 settings 实例
from src.utils.config import get_settings, resolve_app_root
from src.utils.security import safe_exception_text

console = Console()


def _resolve_version() -> str:
    """返回程序版本号，优先取构建期写入的 ``VERSION.txt``。

    版本号有两处消费方：发布包的**文件名**来自 ``pyproject.toml``（构建脚本
    读取），而横幅与自检输出来自本模块。两处各写一份时，只改一处会让包名与
    自报版本不一致，而 ``--validate`` 只 grep ``smoke test ok``、不带版本号，
    构建照样通过。构建脚本因此把版本写进 ``VERSION.txt``，此处优先读取，
    使产物自报的版本与包名同源；源码运行时退回读 ``pyproject.toml``。
    """
    # 用 resolve_app_root() 而不是 __file__ 的父目录：冻结后 main.py 位于
    # ``_internal/``，而 VERSION.txt 与 pyproject.toml 都在应用根目录
    # （可执行文件旁），按 __file__ 找会两者皆空、静默退回 "0.0.0"。
    app_root = resolve_app_root()

    bundled = app_root / "VERSION.txt"
    if bundled.is_file():
        # 读取必须兜住：本函数在**模块导入期**执行，早于 ``main()`` 的任何
        # ``try``，异常会直接终止进程。实测 VERSION.txt 编码损坏（例如被某个
        # 编辑器写成 UTF-16 或被截断）时 ``read_text`` 抛 ``UnicodeDecodeError``，
        # 程序连横幅都打不出来就退出，exit code 1、消息是裸 traceback。
        #
        # 这与本函数的设计意图相反：下面两级兜底（pyproject.toml → "0.0.0"）
        # 存在的意义正是「读不到就降级」，而读不到的原因不止「文件不存在」。
        try:
            text = bundled.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            text = ""
        if text:
            return text

    pyproject = app_root / "pyproject.toml"
    if pyproject.is_file():
        import tomllib

        try:
            with pyproject.open("rb") as handle:
                version = tomllib.load(handle)["project"]["version"]
        except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
            # ``TypeError`` 与 ``ValueError`` 都要接：``["project"]`` 与
            # ``["version"]`` 在下标形态不对时抛 ``TypeError``
            # （``project = "x"`` -> ``string indices must be integers``、
            # ``project = 5`` -> ``'int' object is not subscriptable``），
            # 不是 ``KeyError``。本函数在**模块导入期**执行，漏出的异常会让
            # 进程以裸 traceback 退出，而两级兜底存在的意义正是「读不到就降级」。
            # ``ValueError`` 覆盖 ``tomllib`` 对超长整数等取值的拒绝。
            pass
        else:
            if isinstance(version, str) and version:
                return version

    return "0.0.0"  # 兜底：仅在既无 VERSION.txt 也无 pyproject.toml 时出现


VERSION = _resolve_version()


# =================================================================
# 应用程序界面 (APP UI)
# =================================================================
def create_gradient(
    text: str, start_color: tuple[int, int, int], end_color: tuple[int, int, int]
) -> Text:
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
    fig = pyfiglet.Figlet(font="big")
    banner_text = fig.renderText("PyRAG-Kit")

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
            padding_size = max(padding_size, 1)

            padding = Text(" " * padding_size)

            # 组合并打印该行
            console.print(gradient_part + padding + attribution_text)
        else:
            # 其他行正常打印渐变效果
            console.print(create_gradient(line, blue_rgb, red_rgb))

    # 构建包含丰富链接和信息的欢迎面板
    welcome_text = Text(justify="center")
    welcome_text.append(f"欢迎使用 PyRAG-Kit - 版本 {VERSION}\n", style="bold cyan")
    welcome_text.append(
        "一个 Dify 核心逻辑的 Python 实现，用于本地验证其知识库向量化、分段及检索流程。\n\n",
        style="dim",
    )
    welcome_text.append("作者: ", style="bold")
    welcome_text.append("Mison", style="default")
    welcome_text.append("  ·  邮箱: ", style="bold")
    welcome_text.append("1360962086@qq.com", style="default")
    welcome_text.append("\n")  # 换行
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
    console.print(
        Panel(
            menu_content,
            title="[bold yellow]主菜单[/bold yellow]",
            border_style="green",
            expand=False,
            highlight=True,
        )
    )


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
    #
    # 该目录是 ``jieba`` 的临时目录，属于可选优化：不可写时回落到系统临时
    # 目录即可，不该让程序无法启动。默认的 ``cache_path``（``.cache``）解析
    # 到应用根目录之下，安装在 ``/opt``、``/Applications``、
    # ``C:\Program Files`` 这类只读位置时 ``makedirs`` 会抛
    # ``PermissionError``——而本函数在 ``main()`` 的 ``try`` 之外，崩溃表现为
    # 裸 traceback。与 ``log_manager`` 的日志降级同理。
    cache_dir = _prepare_cache_dir(get_settings().cache_path)

    # 2. 预热jieba并完全抑制其所有启动日志
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        import jieba
        import jieba.posseg as pseg

        jieba.setLogLevel(jieba.logging.ERROR)
        jieba.dt.tmp_dir = str(cache_dir)
        list(pseg.cut(""))

    console.print("[dim]依赖项初始化完成。[/dim]")


def _prepare_cache_dir(configured_path: str) -> Path:
    """返回可写的缓存目录，不可写时回落到系统临时目录。

    只影响 ``jieba`` 的临时文件位置；``settings.cache_path`` 本身不改写，
    以免影响退出时 ``cleanup_temp_files()`` 对用户配置路径的判断。
    """
    import tempfile

    candidate = Path(configured_path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        if os.access(candidate, os.W_OK):
            return candidate
        reason = "目录不可写"
    except OSError as exc:
        reason = str(exc)

    fallback = Path(tempfile.gettempdir()) / "pyrag-kit-cache"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # 与全仓「面向用户的异常输出一律经 safe_exception_text」一致（脱敏 +
        # 240 截断），不直接把异常对象交给 markup_text。
        console.print(
            markup_text(
                "[yellow]缓存目录不可用，且无法创建临时目录：{}[/yellow]",
                safe_exception_text(exc),
            )
        )
    else:
        console.print(
            markup_text(
                "[yellow]缓存目录不可用（{}：{}），本次运行改用临时目录 {}。[/yellow]",
                configured_path,
                reason,
                fallback,
            )
        )
    return fallback


def run_smoke_test() -> int:
    """执行非交互式启动自检。

    自检必须真正触达发布包的关键路径，否则「构建通过、包却是坏的」会一路
    漏到用户手里：PyInstaller 的静态分析看不见 ``pkgutil.iter_modules``
    对 ``tiktoken_ext`` 命名空间包的扫描，漏收该插件时
    ``get_encoding("cl100k_base")`` 直到用户点击「知识库文档向量化」才会
    抛 ``Unknown encoding``，而只打印路径的自检会照常通过。这里替换成
    真实调用分片器的编码加载，把该类缺陷拦在构建阶段。

    同时执行 ``initialize_dependencies()``：它建缓存目录、预热 jieba，且
    位于 ``main()`` 的 ``try`` 之外。先前只打印路径的自检走不到那里，于是
    「只读安装下 ``cache_path`` 不可写导致启动崩溃」在自检里完全不可见——
    验证入口绕开了故障路径，得到的是假通过。
    """
    settings = get_settings()
    _assert_splitter_encoding_available()
    _assert_provider_modules_importable()
    initialize_dependencies()
    # 路径与版本都可能含方括号（``log_path`` 是用户可配置项，安装目录名也
    # 由用户决定），与上面几处同理：走标记解析会被改写或抛错。
    console.print(markup_text("PyRAG-Kit {} smoke test ok", VERSION))
    console.print(markup_text("app_root={}", resolve_app_root()))
    console.print(markup_text("log_path={}", settings.log_path))
    return 0


def _assert_provider_modules_importable() -> None:
    """确认每个已注册的 provider 模块都能导入。

    十三个 provider 模块**没有任何静态导入路径**——它们只经
    ``importlib.import_module(provider_info["module"])`` 按运行时字符串加载
    （``src/providers/factory.py``）。PyInstaller 的静态分析看不见这一层，
    因此 ``scripts/build_binary_release.py`` 的 ``HIDDEN_IMPORTS`` 是它们在
    冻结包里能否被找到的唯一依据。

    构建脚本的清单与实际注册表分处两个文件，各自演化；此前没有任何守卫把两者
    绑在一起，也没有任何构建期检查会碰这些模块——自检只走分片器与日志路径，
    不实例化任何 provider。漏收一个模块的后果是：PyInstaller 不报错、
    ``--validate`` 通过，用户在那个渠道上才拿到 ``ModuleNotFoundError``。
    这与 ``tiktoken_ext.openai_public`` 当初漏收是同一个失效模式。

    这里逐个 import（只导入模块，不实例化 provider，因此不需要任何凭证）。
    """
    import importlib

    from src.providers.factory import ModelProviderFactory

    for provider_name, info in sorted(ModelProviderFactory._provider_map.items()):
        module_name = info["module"]
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"provider 模块无法导入: {provider_name} -> {module_name}。"
                " 若这是在发布包自检中出现的，请检查构建脚本的 HIDDEN_IMPORTS "
                "是否覆盖了该模块（PyInstaller 的静态分析看不见 importlib 加载）。"
            ) from exc


def _assert_splitter_encoding_available() -> None:
    """确认分片器依赖的 tiktoken 编码在本次运行环境中确实可用。

    走 ``RecursiveTextSplitter`` 的真实入口，而不是直接调 ``tiktoken``，
    这样插件发现、编码加载与分片器的构造顺序都会被覆盖到。
    """
    from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter

    splitter = RecursiveTextSplitter()
    chunks = splitter.split([Document(content="smoke 自检文本。" * 20)])
    if not chunks:
        raise RuntimeError("分片器自检未产出任何分片。")


def run_cli(argv: list[str] | None = None) -> int:
    """解析启动参数并执行对应入口。

    这里包住整个启动阶段：``display_banner`` 与 ``initialize_dependencies``
    在 ``main()`` 的 ``try`` 之前执行，配置错误会以裸 traceback 落到终端。
    import 期的失败由 ``get_settings`` 的边界负责（此处尚未执行）。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        # 交互式程序被静默吞参数的体验很差：新用户照惯例敲 ``--help`` 时
        # 直接进了交互菜单（实测），只能 Ctrl+C 退出重来。给出简明用法
        # 后退出，不初始化依赖（``--help`` 必须瞬时返回）。
        console.print(markup_text("PyRAG-Kit {}", VERSION))
        console.print("用法: python main.py [选项]")
        console.print("  --smoke-test   非交互式启动自检（构建验证用）")
        console.print("  -h, --help     显示本帮助并退出")
        console.print("  无参数          进入交互式菜单（向量化 / 召回测试 / 聊天）")
        return 0
    try:
        if "--smoke-test" in args:
            return run_smoke_test()
        main()
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary reports failures
        console.print(markup_text("[bold red]启动失败:[/bold red] {}", safe_exception_text(exc)))
        return 1
    return 0


def main():
    """程序主入口，提供交互式菜单并实现延迟加载。"""
    display_banner()
    initialize_dependencies()
    while True:
        display_menu()
        try:
            # 使用 prompt_toolkit 替代 console.input，并优化样式
            choice = prompt(HTML("<skyblue><b>请输入选项 (1-4): </b></skyblue>"))
            if choice == "1":
                console.print(
                    Panel(
                        "[bold green]开始执行知识库文档向量化处理...[/bold green]",
                        border_style="green",
                        width=CONSOLE_WIDTH,
                    )
                )
                # 延迟加载和执行。
                # ``SystemExit`` 不能终止整个程序：``embed_knowledge_base`` 在用户
                # 取消确认时 ``sys.exit(0)``，而这里的 ``except Exception`` 接不住
                # 它（``SystemExit`` 继承 ``BaseException``）。捕获后回到主菜单，
                # 让用户能继续别的选项——「取消」不等于「退出程序」。
                from scripts.embed_knowledge_base import main as run_embedding_process

                try:
                    run_embedding_process()
                except SystemExit as exc:
                    if exc.code not in (0, None):
                        raise
                    console.print("[bold yellow]操作已取消。[/bold yellow]")
                else:
                    console.print(
                        Panel(
                            "[bold green]向量化处理完成。[/bold green]",
                            border_style="green",
                            width=CONSOLE_WIDTH,
                        )
                    )
            elif choice == "2":
                console.print(
                    Panel(
                        "[bold green]开始执行召回测试...[/bold green]",
                        border_style="green",
                        width=CONSOLE_WIDTH,
                    )
                )
                # 延迟加载和执行
                from src.retrieval_test.core import run_retrieval_test

                run_retrieval_test()
                console.print(
                    Panel(
                        "[bold green]召回测试完成。[/bold green]",
                        border_style="green",
                        width=CONSOLE_WIDTH,
                    )
                )
            elif choice == "3":
                console.print(
                    Panel(
                        "[bold green]启动聊天机器人会话...[/bold green]",
                        border_style="green",
                        width=CONSOLE_WIDTH,
                    )
                )
                # 延迟加载和执行
                from src.chat.core import start_chat_session

                start_chat_session()
                console.print(
                    Panel(
                        "[bold green]聊天会话结束。[/bold green]",
                        border_style="green",
                        width=CONSOLE_WIDTH,
                    )
                )
            elif choice == "4":
                console.print("[bold]正在退出程序... 再见！[/bold]")
                sys.exit(0)
            else:
                console.print("[bold red]无效的选项，请输入 1, 2, 3, 或 4。[/bold red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]检测到中断信号，正在退出程序... 再见！[/bold yellow]")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001 - top-level CLI boundary reports failures
            console.print(
                markup_text(
                    "\n[bold red]程序运行期间发生错误:[/bold red] {}", safe_exception_text(e)
                )
            )
            console.print("[bold red]请检查错误信息并重试。[/bold red]")


if __name__ == "__main__":
    sys.exit(run_cli())
