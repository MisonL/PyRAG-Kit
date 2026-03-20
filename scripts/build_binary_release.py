#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import subprocess
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
    command.append("main.py")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def stage_bundle(target: str, version: str) -> Path:
    bundle_name = f"{APP_NAME}-{version}-{target}"
    bundle_root = ARTIFACT_ROOT / bundle_name
    app_source = DIST_ROOT / APP_NAME
    app_target = bundle_root / APP_NAME

    shutil.copytree(app_source, app_target)
    for relative_file in PACKAGE_FILES:
        shutil.copy2(PROJECT_ROOT / relative_file, bundle_root / relative_file)
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
        "# 知识库目录\n\n"
        "请将您的 Markdown 知识库文档放入当前目录，然后再执行知识库构建。\n",
        encoding="utf-8",
    )


def archive_bundle(bundle_root: Path, target: str) -> Path:
    if target == "windows-x64":
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
    subprocess.run(
        [str(executable)],
        cwd=bundle_root,
        input="4\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )


def main() -> None:
    args = parse_args()
    version = load_version()

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
