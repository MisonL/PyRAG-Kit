"""守卫变异器：对 ``(行号, 列偏移)`` 唯一确定的 AST 节点施加语义相邻变异。

支持的 kind：

* ``drop``：``not X`` → ``X``，删除 ``not`` 关键字本身（只删三个字符，
  不替换整个节点——``not (a and b)`` 的操作数跨度不含括号，整段替换会
  产出语法错误）。
* ``notin``：``x not in y`` → ``x in y``，只替换运算符区段。
* ``isnot``：``x is not y`` → ``x is y``，只替换运算符区段。
* ``eqneq``：``x == y`` → ``x != y``（反之亦然），只替换运算符区段；
  多操作数比较只翻转第一个 ``Eq``/``NotEq``。
* ``cmpdir``：``x < y`` → ``x > y``、``x <= y`` → ``x >= y``（反之亦然），
  方向反转；多操作数比较只翻转第一个 ``Lt``/``LtE``/``Gt``/``GtE``。

坐标约定：``line`` 为 1 起始行号，``col`` 为 CPython AST 的 ``col_offset``
（**UTF-8 字节**偏移）。旧版 ``mut_cmp.py`` 把 ``col_offset`` 当字符偏移
用，含中文的行会错位——本模块统一走 ``_char_offset`` 换算。

用法::

    python mutate.py <相对路径> <行号> <列偏移> <kind>

退出码：0=已施加；2=未找到位点；3=变异体语法错误。
"""

import ast
import pathlib
import re
import sys

KINDS = ("drop", "notin", "isnot", "eqneq", "cmpdir")

_OP_KINDS = {
    "notin": (ast.NotIn,),
    "isnot": (ast.IsNot,),
    "eqneq": (ast.Eq, ast.NotEq),
    "cmpdir": (ast.Lt, ast.LtE, ast.Gt, ast.GtE),
}


class SiteNotFound(Exception):
    """目标坐标上没有恰好一个符合 kind 的节点。"""


class MutantSyntaxError(Exception):
    """变异后的源码无法通过 ``compile()``。"""


def _char_offset(lines: list[str], lineno: int, byte_col: int) -> int:
    """把 ``(行号, UTF-8 字节列)`` 换算成全文**字符**偏移。"""
    line_start = sum(len(x) for x in lines[: lineno - 1])
    line_text = lines[lineno - 1]
    prefix_bytes = line_text.encode("utf-8")[:byte_col]
    return line_start + len(prefix_bytes.decode("utf-8"))


def iter_sites(tree: ast.AST, kind: str):
    """产出 ``(lineno, col_offset, node)``：全树中所有该 kind 的位点。"""
    if kind == "drop":
        for node in ast.walk(tree):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                yield node.lineno, node.col_offset, node
        return
    wanted = _OP_KINDS[kind]
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, wanted) for op in node.ops):
            yield node.lineno, node.col_offset, node


def find_site(tree: ast.AST, line: int, col: int, kind: str) -> ast.AST:
    """精确定位唯一节点；未命中或重复命中抛 ``SiteNotFound``。"""
    matches = [
        node
        for lineno, col_offset, node in iter_sites(tree, kind)
        if (lineno, col_offset) == (line, col)
    ]
    if len(matches) != 1:
        raise SiteNotFound(f"{kind} 位点在 {line}:{col} 找到 {len(matches)} 个，期望 1")
    return matches[0]


def find_nearest_site(
    tree: ast.AST, line: int, col: int, kind: str, window: int
) -> tuple[int, int] | None:
    """漂移定位：``±window`` 行内找同 kind 节点，**唯一最近**才返回坐标。

    平局时用列偏移消解：同距候选中若恰有一个 ``col_offset`` 与原列一致，
    取它（行号漂移而列未变是最常见的形态）。仍无法唯一确定则返回
    ``None``——宁可 MUTFAIL 也不能把变异施加到别的守卫上。
    """
    candidates = sorted(
        (lineno, col_offset)
        for lineno, col_offset, _ in iter_sites(tree, kind)
        if abs(lineno - line) <= window
    )
    if not candidates:
        return None
    best_distance = min(abs(lineno - line) for lineno, _ in candidates)
    nearest = [(lineno, c) for lineno, c in candidates if abs(lineno - line) == best_distance]
    if len(nearest) == 1:
        return nearest[0]
    col_matches = [(lineno, c) for lineno, c in nearest if c == col]
    if len(col_matches) == 1:
        return col_matches[0]
    return None


def mutate_source(src: str, node: ast.AST, kind: str) -> str:
    """返回对 ``node`` 施加 ``kind`` 变异后的全文（不写盘、不校验）。"""
    lines = src.splitlines(keepends=True)
    if kind == "drop":
        start = _char_offset(lines, node.lineno, node.col_offset)
        if src[start : start + 3] != "not":
            raise SiteNotFound(f"drop 定位不符: {src[start : start + 3]!r}")
        end = start + 3
        # 后面若是空格则一并删掉一个，避免留下双空格。
        if end < len(src) and src[end] == " ":
            end += 1
        return src[:start] + src[end:]

    if kind not in _OP_KINDS:
        raise ValueError(f"未知 kind: {kind}")
    wanted = _OP_KINDS[kind]
    index = next(i for i, op in enumerate(node.ops) if isinstance(op, wanted))
    prev = node.left if index == 0 else node.comparators[index - 1]
    nxt = node.comparators[index]
    seg_start = _char_offset(lines, prev.end_lineno, prev.end_col_offset)
    seg_end = _char_offset(lines, nxt.lineno, nxt.col_offset)
    segment = src[seg_start:seg_end]
    if kind == "notin":
        pattern, replacement = r"\bnot\s+in\b", "in"
    elif kind == "isnot":
        pattern, replacement = r"\bis\s+not\b", "is"
    elif kind == "eqneq":
        pattern, replacement = r"==|!=", "!=" if isinstance(node.ops[index], ast.Eq) else "=="
    else:
        # 运算符节点（ast.Lt 等）不带位置信息，只能靠区段正则；按最长优先
        # 排列，避免 ``<`` 先于 ``<=`` 命中而在 ``<=`` 里切出 ``> =``。
        pattern = r"<=|>=|<|>"

        def replacement(match: re.Match[str]) -> str:
            return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[match.group(0)]

    new_segment, count = re.subn(pattern, replacement, segment, count=1)
    if count != 1:
        raise SiteNotFound(f"运算符区段内未找到 {pattern!r}: {segment!r}")
    return src[:seg_start] + new_segment + src[seg_end:]


def apply_mutation(path: str | pathlib.Path, line: int, col: int, kind: str) -> str:
    """定位、变异、语法校验并写盘；返回 kind。"""
    file_path = pathlib.Path(path)
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = find_site(tree, line, col, kind)
    new_src = mutate_source(src, node, kind)
    # 变异必须产出合法 Python：语法错误会让 pytest 在收集期就失败，
    # 被验证器记成「已杀死」——那是假阴性，会把真缺口藏起来。
    try:
        compile(new_src, str(file_path), "exec")
    except SyntaxError as exc:
        raise MutantSyntaxError(str(exc)) from exc
    file_path.write_text(new_src, encoding="utf-8")
    return kind


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(f"用法: {argv[0]} <相对路径> <行号> <列偏移> <kind>", file=sys.stderr)
        return 2
    path, line, col, kind = argv[1], int(argv[2]), int(argv[3]), argv[4]
    try:
        print(apply_mutation(path, line, col, kind))
    except SiteNotFound as exc:
        print(f"SITE-NOT-FOUND {exc}", file=sys.stderr)
        return 2
    except MutantSyntaxError as exc:
        print(f"MUTANT-SYNTAX {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
