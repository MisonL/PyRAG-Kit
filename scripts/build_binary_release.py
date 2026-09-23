#!/usr/bin/env python3
import argparse
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
SPEC_PATH = PROJECT_ROOT / f"{APP_NAME}.spec"
PACKAGE_FILES = [
    "README.md",
    "LICENSE",
    "DIFY_LICENSE",
    # Dify 衍生代码遵循修改后的 Apache 2.0，该许可证第 4(a) 条要求随分发
    # 提供许可证副本；DIFY_LICENSE 只是引用它的摘要，不能替代全文。
    "licenses/APACHE-2.0.txt",
    "config.toml.example",
    ".env.example",
]
HIDDEN_IMPORTS = [
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
    prepare_runtime_layout(bundle_root)

    return bundle_root


def prepare_runtime_layout(bundle_root: Path) -> None:
    knowledge_base_dir = bundle_root / "knowledge_base"
    data_kb_dir = bundle_root / "data" / "kb"
    data_logs_dir = bundle_root / "data" / "logs"

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    data_kb_dir.mkdir(parents=True, exist_ok=True)
    data_logs_dir.mkdir(parents=True, exist_ok=True)

    placeholder = knowledge_base_dir / "README.md"
    placeholder.write_text(
        "# 知识库目录\n\n请将您的 Markdown 知识库文档放入当前目录，然后再执行知识库构建。\n",
        encoding="utf-8",
    )


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
        check=True,
        timeout=30,
    )
    if "smoke test ok" not in result.stdout:
        raise RuntimeError("发布包自检输出不符合预期。")


def main() -> None:
    args = parse_args()
    version = load_version()

    validate_target_environment(args.target)
    clean_output_dirs()
    run_pyinstaller()
    bundle_root = stage_bundle(args.target, version)
    archive_path = archive_bundle(bundle_root, args.target)
    if SPEC_PATH.exists():
        SPEC_PATH.unlink()

    if args.validate:
        validate_bundle(bundle_root)

    print(archive_path)


if __name__ == "__main__":
    main()
