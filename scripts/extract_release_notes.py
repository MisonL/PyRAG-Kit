#!/usr/bin/env python3
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 CHANGELOG.md 提取指定版本的发布说明。")
    parser.add_argument("--version", required=True, help="版本号，例如 1.3.0。")
    parser.add_argument("--output", required=True, help="输出文件路径。")
    return parser.parse_args()


# GitHub Release notes 的 API 上限是 125000 字符（HTTP 422 body is too long，
# v1.5.0 发版实测：264861 字符被拒）。留出余量。
RELEASE_NOTES_MAX_CHARS = 120_000


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

    # Release notes 只面向用户：遇到维护者向的存档节标题即止（完整记录仍在
    # CHANGELOG）。当前项目的存档节以「内部存档」命名。
    cut_index = end_index
    for index in range(start_index, end_index):
        if "内部存档" in lines[index] and lines[index].startswith("#"):
            cut_index = index
            break

    content = "\n".join(lines[start_index:cut_index]).strip() + "\n"
    if len(content) <= RELEASE_NOTES_MAX_CHARS:
        return content
    # 超长时按整行边界截断，绝不在截断处伪装成完整记录。
    truncated_lines = []
    current = 0
    for line in content.splitlines():
        if current + len(line) + 1 > RELEASE_NOTES_MAX_CHARS:
            break
        truncated_lines.append(line)
        current += len(line) + 1
    return (
        "\n".join(truncated_lines).rstrip()
        + f"\n\n> 本版本变更条目较多，已超出 GitHub Release 说明的长度上限，"
        f"以上为前段；完整记录见仓库 CHANGELOG.md 的 [{version}] 节。\n"
    )


def main() -> None:
    args = parse_args()
    content = extract_section(args.version)
    Path(args.output).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
