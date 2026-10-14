"""枚举源码里的变异候选位点，输出 verify_sites.py 可消费的站点文件。

用法::

    python list_sites.py [--root .] [--kinds drop,notin,isnot,eqneq] [路径 ...]

输出每行 ``target|line|col|kind``：``target`` 相对 ``--root``（默认当前
目录），``col`` 为 CPython AST 的 ``col_offset``（UTF-8 字节偏移）。
路径参数缺省为 ``src``；``.py`` 文件与目录都可传入。
"""

import argparse
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mutate import KINDS, iter_sites


def iter_python_files(root: pathlib.Path, paths: list[str]):
    for raw in paths:
        path = (root / raw).resolve()
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="枚举变异候选位点")
    parser.add_argument("--root", default=".", help="输出路径的相对根（默认当前目录）")
    parser.add_argument(
        "--kinds",
        default=",".join(KINDS),
        help=f"逗号分隔的 kind 列表（默认全部：{','.join(KINDS)}）",
    )
    parser.add_argument("paths", nargs="*", default=["src"], help="要扫描的文件或目录（默认 src）")
    args = parser.parse_args(argv[1:])

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = sorted(set(kinds).difference(KINDS))
    if unknown:
        print(f"未知 kind: {', '.join(unknown)}；可用: {', '.join(KINDS)}", file=sys.stderr)
        return 2

    root = pathlib.Path(args.root).resolve()
    rows: list[tuple[str, int, int, str]] = []
    for path in iter_python_files(root, args.paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"跳过无法解析的文件 {path}: {exc}", file=sys.stderr)
            continue
        target = path.relative_to(root).as_posix()
        for kind in kinds:
            for lineno, col, _ in iter_sites(tree, kind):
                rows.append((target, lineno, col, kind))
    for target, lineno, col, kind in sorted(rows):
        print(f"{target}|{lineno}|{col}|{kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
