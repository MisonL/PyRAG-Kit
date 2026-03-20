#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 CHANGELOG.md 提取指定版本的发布说明。")
    parser.add_argument("--version", required=True, help="版本号，例如 1.3.0。")
    parser.add_argument("--output", required=True, help="输出文件路径。")
    return parser.parse_args()


def extract_section(version: str) -> str:
    lines = CHANGELOG_PATH.read_text(encoding="utf-8").splitlines()
    start_marker = f"## [{version}]"
    start_index = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        if line.startswith(start_marker):
            start_index = index
            continue
        if start_index is not None and line.startswith("## ["):
            end_index = index
            break

    if start_index is None:
        raise ValueError(f"在 CHANGELOG.md 中未找到版本 {version}。")

    return "\n".join(lines[start_index:end_index]).strip() + "\n"


def main() -> None:
    args = parse_args()
    content = extract_section(args.version)
    Path(args.output).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
