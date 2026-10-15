#!/usr/bin/env python3
import argparse
import hashlib
import os
import platform
import shutil

# The release command is assembled from fixed local build arguments.
import subprocess  # nosec B404
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = PROJECT_ROOT / "build"
DIST_ROOT = PROJECT_ROOT / "dist"
ARTIFACT_ROOT = PROJECT_ROOT / "release_artifacts"
APP_NAME = "PyRAG-Kit"
TIKTOKEN_CACHE_DIRNAME = "tiktoken_cache"
CACHE_DIRNAME = ".cache"
SPEC_PATH = PROJECT_ROOT / f"{APP_NAME}.spec"
PACKAGE_FILES = [
    "README.md",
    "LICENSE",
    "DIFY_LICENSE",
    # Dify 衍生代码遵循修改后的 Apache 2.0，该许可证第 4(a) 条要求随分发
    # 提供许可证副本；DIFY_LICENSE 只是引用它的摘要，不能替代全文。
    "licenses/APACHE-2.0.txt",
]
# 必须与 app 读取配置的位置一致：``config.toml`` 与 ``.env`` 都由
# ``ROOT_DIR`` 解析，冻结时即 ``bundle_root / APP_NAME``。放在包根会被静默
# 忽略，用户复制示例后仍拿不到预期配置。
APP_ROOT_FILES = [
    "config.toml.example",
    ".env.example",
]
HIDDEN_IMPORTS = [
    # tiktoken 用 pkgutil.iter_modules 在运行时扫描 tiktoken_ext 命名空间包来
    # 发现编码插件（tiktoken/registry.py:19-33），PyInstaller 的静态分析看不到
    # 这一步。漏掉它的后果是冻结包内 pkgutil.iter_modules 返回空列表，
    # get_encoding("cl100k_base") 抛 "Unknown encoding"，ETL 分片第一步即失败。
    "tiktoken_ext.openai_public",
    "src.providers.google",
    "src.providers.openai",
    "src.providers.anthropic",
    "src.providers.qwen",
    "src.providers.volcengine",
    "src.providers.siliconflow",
    "src.providers.ollama",
    "src.providers.lm_studio",
    "src.providers.deepseek",
    "src.providers.grok",
    "src.providers.local_hash",
    "src.providers.jina",
    "src.providers.siliconflow_rerank",
]
EXCLUDED_MODULES = [
    "pandas",
    "scipy",
    "sklearn",
    "pytest",
    "_pytest",
]

_TARGET_ENVIRONMENTS = {
    "windows-x64": ("windows", "x86_64"),
    "macos-x64": ("darwin", "x86_64"),
    "macos-arm64": ("darwin", "arm64"),
    "linux-x64": ("linux", "x86_64"),
    "linux-arm64": ("linux", "arm64"),
}


def _normalize_machine(machine: str) -> str:
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "aarch64": "arm64",
    }
    normalized = machine.strip().lower().replace("-", "_")
    return aliases.get(normalized, normalized)


def validate_target_environment(
    target: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> None:
    """Reject targets that the current PyInstaller host cannot produce.

    PyInstaller builds for the interpreter's host platform and architecture;
    the release script does not perform cross-compilation. Failing before the
    output cleanup prevents an incorrectly named artifact from replacing a
    previously valid one.
    """
    expected = _TARGET_ENVIRONMENTS.get(target)
    if expected is None:
        raise ValueError(f"不支持的发布目标: {target}")

    actual_system = (system or platform.system()).strip().lower()
    actual_machine = _normalize_machine(machine or platform.machine())
    expected_system, expected_machine = expected
    if actual_system != expected_system or actual_machine != expected_machine:
        detected = f"{actual_system}-{actual_machine}"
        required = f"{expected_system}-{expected_machine}"
        raise RuntimeError(
            f"发布目标 {target} 需要 {required} 构建环境，当前检测到 {detected}。"
            " PyInstaller 不执行跨平台交叉编译，请在匹配的 runner 或主机上构建。"
        )


def load_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return pyproject["project"]["version"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建跨平台二进制发布包。")
    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "windows-x64",
            "macos-x64",
            "macos-arm64",
            "linux-x64",
            "linux-arm64",
        ],
        help="目标平台标识。",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="构建完成后执行最小烟测。",
    )
    return parser.parse_args()


def clean_output_dirs() -> None:
    for directory in (BUILD_ROOT, DIST_ROOT, ARTIFACT_ROOT):
        if directory.exists():
            shutil.rmtree(directory)
    if SPEC_PATH.exists():
        SPEC_PATH.unlink()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def run_pyinstaller() -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        APP_NAME,
        "--collect-data",
        "pyfiglet",
    ]
    for hidden_import in HIDDEN_IMPORTS:
        command.extend(["--hidden-import", hidden_import])
    for excluded_module in EXCLUDED_MODULES:
        command.extend(["--exclude-module", excluded_module])
    command.append("main.py")
    # shell=False and fixed command arguments keep this invocation bounded.
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # nosec B603


def stage_bundle(target: str, version: str) -> Path:
    bundle_name = f"{APP_NAME}-{version}-{target}"
    bundle_root = ARTIFACT_ROOT / bundle_name
    app_source = DIST_ROOT / APP_NAME
    app_target = bundle_root / APP_NAME

    shutil.copytree(app_source, app_target)
    for relative_file in PACKAGE_FILES:
        # 不要复用 ``target``：它是本函数的参数（平台标识，如 "macos-arm64"），
        # 被循环变量遮蔽后，循环之后再用它会拿到 Path 而不是平台名。
        destination = bundle_root / relative_file
        # PACKAGE_FILES 含子目录路径（如 licenses/APACHE-2.0.txt），
        # copy2 不会自动创建父目录。
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_file, destination)
    for relative_file in APP_ROOT_FILES:
        destination = app_target / relative_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_file, destination)
    prepare_runtime_layout(bundle_root)
    # BPE 预置单独一步：它需要联网（缓存缺失时），与「建目录」是两件事，
    # 拆开后布局逻辑的单元测试不必依赖网络或宿主缓存。
    stage_tiktoken_cache(app_target)
    # 把版本写进包内：``main.py`` 优先读它，使产物自报的版本与包名
    # （来自 pyproject.toml）同源，消除「改了 pyproject 忘了改 main.py」
    # 导致包名与横幅不一致、而 --validate 照常通过的漂移。
    (app_target / "VERSION.txt").write_text(version + "\n", encoding="utf-8")

    return bundle_root


def prepare_runtime_layout(bundle_root: Path) -> None:
    """在冻结应用自己的根目录下准备可写运行时目录。

    ``resolve_app_root()`` 在冻结时返回可执行文件所在目录
    （``src/utils/config.py``），也就是 ``bundle_root / APP_NAME``，而所有
    默认相对路径（``knowledge_base``、``data/kb``、``data/logs``）都经
    ``resolve_path`` 解析到该目录之下。把这些目录建在 ``bundle_root`` 上会
    留下一个应用永远不会读的死目录，用户按 README 把文档放进去只会拿到
    ``FileNotFoundError``。此处统一以 ``APP_NAME`` 子目录为根。
    """
    app_root = bundle_root / APP_NAME
    knowledge_base_dir = app_root / "knowledge_base"
    data_kb_dir = app_root / "data" / "kb"
    data_logs_dir = app_root / "data" / "logs"

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    data_kb_dir.mkdir(parents=True, exist_ok=True)
    data_logs_dir.mkdir(parents=True, exist_ok=True)

    placeholder = knowledge_base_dir / "README.md"
    placeholder.write_text(
        "# 知识库目录\n\n请将您的 Markdown 知识库文档放入当前目录，然后再执行知识库构建。\n",
        encoding="utf-8",
    )


# cl100k_base 的 BPE blob 与其 sha256。哈希取自 tiktoken 的
# ``tiktoken_ext/openai_public.py``（``cl100k_base`` 构造器里的
# ``expected_hash``），构建期用它校验预置进包内的文件——发布包里的缓存是
# 只读的，用户离线时无法触发 tiktoken 自身的重取与校验，坏文件只会在
# 分片时表现为难以归因的解析错误。
TIKTOKEN_BLOB_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TIKTOKEN_BLOB_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def stage_tiktoken_cache(app_root: Path) -> None:
    """把 cl100k_base 的 BPE 文件预置进包内，供离线首次运行使用。

    ``tiktoken`` 在缓存未命中时会 ``requests.get`` 远程 blob
    （``tiktoken/load.py``），因此未预置的包在离线环境首次构建知识库必然
    失败。键名是 ``sha1(blobpath)``，与 ``read_file_cached`` 的查找逻辑一致；
    运行时由 ``main.py`` 把 ``TIKTOKEN_CACHE_DIR`` 指向该目录。

    缓存缺失时**由本函数自己联网获取**，而不是要求构建者先手工跑一次知识库
    构建：CI runner 是全新机器，``$TMPDIR`` 里没有任何缓存，把「开发者本机
    恰好构建过一次」当成构建前提会让五个发布目标全部失败。构建机本来就在
    联网拉取 PyInstaller 依赖，多取一个文件不改变信任模型；取回后按
    ``TIKTOKEN_BLOB_SHA256`` 校验，坏数据不会进包。
    """
    cache_dir = app_root / TIKTOKEN_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / _tiktoken_cache_key()

    source = Path(tiktoken_cache_lookup_path(_tiktoken_cache_key()))
    if source.is_file():
        payload = source.read_bytes()
    else:
        payload = _download_tiktoken_blob()

    actual = hashlib.sha256(payload).hexdigest()
    if actual != TIKTOKEN_BLOB_SHA256:
        raise RuntimeError(
            f"cl100k_base 的 BPE 文件校验失败：期望 sha256 {TIKTOKEN_BLOB_SHA256}，"
            f"实际 {actual}。\n"
            "该文件可能已损坏或过期；请清理缓存后重试，"
            "若 tiktoken 升级导致 blob 变更，需同步更新 TIKTOKEN_BLOB_SHA256。"
        )
    destination.write_bytes(payload)


def _tiktoken_cache_key() -> str:
    """返回 tiktoken 为 cl100k_base 使用的缓存键（``sha1(blobpath)``）。"""
    # 非安全用途：这只是 tiktoken 自己的缓存文件名算法，必须与之逐字一致。
    return hashlib.sha1(TIKTOKEN_BLOB_URL.encode(), usedforsecurity=False).hexdigest()


def _download_tiktoken_blob() -> bytes:
    """联网取回 cl100k_base 的 BPE 数据。

    失败时换成带上下文的错误：裸 ``URLError`` 只说明某次连接失败，看不出
    「这是发布包的离线兜底依赖」，也无从知道该往哪个方向修。
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(TIKTOKEN_BLOB_URL, timeout=60) as response:  # nosec B310
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"无法获取 cl100k_base 的 BPE 数据: {exc}\n"
            f"来源: {TIKTOKEN_BLOB_URL}\n"
            "该文件是发布包离线运行的分词兜底依赖，构建期必须取得。"
            "若构建环境无外网，请预先把缓存放到 "
            "``$TIKTOKEN_CACHE_DIR`` 或 ``$TMPDIR/data-gym-cache`` 下，"
            "键名为 sha1(blob 路径)。"
        ) from exc


def tiktoken_cache_lookup_path(cache_key: str) -> str:
    """复刻 tiktoken 的缓存查找顺序，返回当前主机的缓存文件路径。

    逐字对齐 ``tiktoken.load.read_file_cached``：变量是否**存在**决定是否
    采用（空串表示禁用缓存，tiktoken 会直接走网络，此处返回空串让调用方
    据此判断），而不是按真值判断。
    """
    for variable in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        if variable in os.environ:
            override = os.environ[variable]
            if override == "":
                return ""
            return str(Path(override) / cache_key)
    import tempfile

    return str(Path(tempfile.gettempdir()) / "data-gym-cache" / cache_key)


def find_dangling_symlinks(bundle_root: Path) -> list[Path]:
    """列出 bundle 内目标不存在的符号链接。

    ``ZipFile.write`` 会解引用符号链接去取元数据，目标缺失时抛的是裸
    ``FileNotFoundError``（不带任何「这是发布包符号链接」的线索），而
    ``tarfile.add`` 会把链接本身写进归档、不报错。两个分支行为不一致，
    且在 Windows 这个唯一的 zip 目标上表现为一个难以归因的构建失败。
    归档前先显式找出它们，让失败信息说清是什么、有几条。
    """
    return sorted(
        entry for entry in bundle_root.rglob("*") if entry.is_symlink() and not entry.exists()
    )


def archive_bundle(bundle_root: Path, target: str) -> Path:
    if target == "windows-x64":
        dangling = find_dangling_symlinks(bundle_root)
        if dangling:
            listed = "\n  ".join(str(path.relative_to(bundle_root)) for path in dangling)
            raise RuntimeError(
                f"发布包内含 {len(dangling)} 个目标不存在的符号链接，无法打包为 ZIP："
                f"\n  {listed}\n"
                " ZIP 格式无法保留符号链接，``ZipFile.write`` 又会解引用它们，"
                "请检查构建产物；若这些链接是有意为之，请改用 tar.gz 目标。"
            )
        archive_path = bundle_root.parent / f"{bundle_root.name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(bundle_root.rglob("*")):
                archive.write(file_path, file_path.relative_to(bundle_root.parent))
        return archive_path

    archive_path = bundle_root.parent / f"{bundle_root.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(bundle_root, arcname=bundle_root.name)
    return archive_path


def executable_path(bundle_root: Path) -> Path:
    executable_name = APP_NAME + (".exe" if os.name == "nt" else "")
    return bundle_root / APP_NAME / executable_name


def validate_bundle(bundle_root: Path) -> None:
    executable = executable_path(bundle_root)
    # The executable is the bundle produced immediately before validation.
    result = subprocess.run(  # nosec B603
        [str(executable), "--smoke-test"],
        cwd=bundle_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env={
            **os.environ,
            # Windows 下 exe 的 stdout 是管道（capture_output=True），rich 的
            # legacy console renderer 会按 cp1252 编码中文输出，抛
            # ``UnicodeEncodeError``（v1.5.0 首次发版实测，stderr 带出后才
            # 可见）。强制子进程走 UTF-8 模式：真实用户在终端交互运行时
            # stdout 是控制台而非管道，不走这条路径，不受影响。
            "PYTHONUTF8": "1",
        },
    )
    if result.returncode != 0:
        # ``check=True`` 抛出的 ``CalledProcessError`` 不携带子进程输出——
        # Windows exe 自检 exit 1 时（v1.5.0 首次发版实测）日志里只有一行
        # 退出码，真实故障原因完全不可见。这里把 stdout/stderr 原样带出来。
        raise RuntimeError(
            f"发布包自检失败（退出码 {result.returncode}）。\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if "smoke test ok" not in result.stdout:
        raise RuntimeError("发布包自检输出不符合预期。")

    # 自检输出必须带上与包名一致的版本号：只 grep "smoke test ok" 的话，
    # 产物横幅写着旧版本、文件名却是新版本也能通过。
    expected_version = load_version()
    if f"PyRAG-Kit {expected_version} smoke test ok" not in result.stdout:
        raise RuntimeError(
            f"发布包自报版本与 pyproject.toml 的 {expected_version} 不一致。\n"
            f"自检输出:\n{result.stdout}"
        )


def main() -> None:
    args = parse_args()
    version = load_version()

    validate_target_environment(args.target)
    clean_output_dirs()
    run_pyinstaller()
    bundle_root = stage_bundle(args.target, version)

    # 先校验再归档：反过来的话，校验失败时归档已经落盘，且内容是校验前的
    # 状态——「校验通过」与「归档内容」之间没有任何约束关系。
    if args.validate:
        validate_bundle(bundle_root)
        # 自检会真实初始化日志与缓存目录，在 bundle 内留下空日志与 .cache。
        # 它们是构建期残留，必须在校验之后、归档之前清掉，否则会被打进发布包
        # （并且暴露构建日期）。
        remove_runtime_residue(bundle_root)

    archive_path = archive_bundle(bundle_root, args.target)
    if SPEC_PATH.exists():
        SPEC_PATH.unlink()

    print(archive_path)


def remove_runtime_residue(bundle_root: Path) -> None:
    """删除自检在 bundle 内产生的运行时残留。

    只删运行产物（日志、缓存），保留 ``prepare_runtime_layout`` 建的占位
    目录本身——用户解压后应当看到空的 ``knowledge_base/`` 与 ``data/``。
    """
    app_root = bundle_root / APP_NAME
    logs_dir = app_root / "data" / "logs"
    if logs_dir.is_dir():
        for entry in logs_dir.iterdir():
            if entry.is_file():
                entry.unlink()
    cache_dir = app_root / CACHE_DIRNAME
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
