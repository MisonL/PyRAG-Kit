"""变异验证驱动：注入变异 → 跑 pytest → 按退出码分类判定。

站点文件每行一条，``#`` 注释与空行忽略，字段以 ``|`` 分隔::

    target|line|col|kind[|selector[|expected]]

* ``target``：相对仓库根的源码路径。
* ``selector``：``文件::测试名``，缺省时跑全量套件。
* ``expected``：``KILLED``/``SURVIVED``/``MUTFAIL``/``COLLECT_ERROR``，
  提供后进入**回归比对模式**：判定与 expected 不符即失败。

判定（只有 pytest 退出码 0/1 是判定，其余一律非判定）::

    pytest 0    → SURVIVED（变异存活，覆盖缺口）
    pytest 1    → KILLED（变异被测试杀死）
    pytest 4/5  → COLLECT_ERROR（选择器失效等，必须人工看）
    pytest 2/3  → ERROR
    超时        → TIMEOUT
    位点未找到且漂移定位失败 → MUTFAIL

防呆保证（每条对应一次实测翻车）：

1. 启动时校验每个 worktree 已注册且干净，并 ``checkout --detach`` 到主仓
   HEAD——陈旧提交/残留变异会让整轮判定失真。
2. 位点未命中时在 ``--drift-window`` 行内做**唯一最近**漂移定位，命中则
   在备注里打印 ``DRIFTED``；平局或无匹配 → MUTFAIL，绝不施加到别的守卫。
3. 选择器先在 worktree 内预检文件与测试名存在——pytest 对不存在的测试名
   返回退出码 4，历史上被 ``returncode != 0`` 误判成「已杀死」。
4. 每站点 try/finally 还原源码；结束复查全部 worktree 清洁。
5. 无 selector 的扫描默认 ``pytest -q -x``（被杀变异秒退）；全量套件是
   唯一判据，没有「子集先行」的两级语义。

用法::

    python verify_sites.py --worktrees /path/wt1,/path/wt2 sites.txt
    python verify_sites.py --selftest
"""

import argparse
import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mutate import (
    KINDS,
    MutantSyntaxError,
    SiteNotFound,
    apply_mutation,
    find_nearest_site,
    find_site,
)

PYTEST_TIMEOUT = 600
DEFAULT_REPO = pathlib.Path(__file__).resolve().parents[2]

VERDICT_KILLED = "KILLED"
VERDICT_SURVIVED = "SURVIVED"
VERDICT_MUTFAIL = "MUTFAIL"
VERDICT_COLLECT_ERROR = "COLLECT_ERROR"
VERDICT_ERROR = "ERROR"
VERDICT_TIMEOUT = "TIMEOUT"

#: 可以写进站点文件 expected 列的判定；ERROR/TIMEOUT 是基础设施故障，
#: 不允许作为期望值。
EXPECTED_VERDICTS = (VERDICT_KILLED, VERDICT_SURVIVED, VERDICT_MUTFAIL, VERDICT_COLLECT_ERROR)


def classify_pytest_exit(returncode: int) -> str:
    """把 pytest 退出码映射成判定；只有 0/1 是判定。"""
    if returncode == 0:
        return VERDICT_SURVIVED
    if returncode == 1:
        return VERDICT_KILLED
    if returncode in (4, 5):
        return VERDICT_COLLECT_ERROR
    return VERDICT_ERROR


def check_selector(worktree: pathlib.Path, selector: str) -> str | None:
    """在 worktree 内预检 ``文件::测试名``；返回错误原因或 ``None``。"""
    if "::" not in selector:
        return f"选择器缺少 ``::``: {selector!r}"
    file_part, _, test_name = selector.partition("::")
    path = worktree / file_part
    if not path.is_file():
        return f"选择器文件不存在: {file_part}"
    source = path.read_text(encoding="utf-8")
    if not re.search(rf"^def {re.escape(test_name)}\(", source, re.MULTILINE):
        return f"选择器测试不存在: {file_part} 中未找到 def {test_name}("
    return None


def parse_sites(text: str) -> list[dict]:
    """解析站点文件；格式错误直接抛 ``ValueError``。"""
    sites = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) not in (4, 5, 6):
            raise ValueError(f"第 {number} 行字段数应为 4/5/6，实际 {len(fields)}: {raw!r}")
        target, lineno, col, kind = fields[0], int(fields[1]), int(fields[2]), fields[3]
        if kind not in KINDS:
            raise ValueError(f"第 {number} 行未知 kind {kind!r}；可用: {', '.join(KINDS)}")
        selector = fields[4] if len(fields) >= 5 else ""
        expected = fields[5] if len(fields) == 6 else ""
        if expected and expected not in EXPECTED_VERDICTS:
            raise ValueError(
                f"第 {number} 行未知 expected {expected!r}；可用: {', '.join(EXPECTED_VERDICTS)}"
            )
        sites.append(
            {
                "target": target,
                "line": lineno,
                "col": col,
                "kind": kind,
                "selector": selector,
                "expected": expected,
            }
        )
    return sites


def prepare_worktree(repo: pathlib.Path, name: str, head: str) -> pathlib.Path:
    """校验 worktree 已注册、不是主仓本身且干净，然后 detach 到 ``head``。"""
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    registered = [
        pathlib.Path(line[len("worktree ") :]).resolve()
        for line in listing.splitlines()
        if line.startswith("worktree ")
    ]
    worktree = pathlib.Path(name).resolve()
    if worktree not in registered:
        raise RuntimeError(f"worktree 未注册于 {repo}: {name}")
    if worktree == registered[0]:
        raise RuntimeError(f"不能把主仓本身当 worktree: {name}")
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=worktree, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"worktree 不干净，先处理残留改动:\n{dirty}")
    subprocess.run(
        ["git", "checkout", "--detach", head],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=True,
    )
    return worktree


def run_pytest(
    worktree: pathlib.Path, selector: str, python: str, first_failure: bool
) -> tuple[int, str]:
    """跑 pytest；返回 ``(退出码, 输出末行)``。超时抛 ``subprocess.TimeoutExpired``。

    字节码缓存隔离：``==``↔``!=`` 这类变异不改变文件长度，若还原与再变异发生在
    与上一次编译**同一秒**内，CPython 的 (mtime, size) 校验会把上一轮的
    ``__pycache__`` 当成有效缓存——pytest 实际执行的是上一轮的字节码，判定与
    当前位点无关。每次运行把 pyc 写到独立前缀目录并清除，强制按源码重新编译。
    """
    args = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if first_failure:
        args.append("-x")
    if selector:
        args.append(selector)
    pycache_prefix = worktree / ".pycache-audit"
    shutil.rmtree(pycache_prefix, ignore_errors=True)
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(pycache_prefix)}
    try:
        proc = subprocess.run(
            args,
            cwd=worktree,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PYTEST_TIMEOUT,
            check=False,
            env=env,
        )
    finally:
        shutil.rmtree(pycache_prefix, ignore_errors=True)
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode, lines[-1] if lines else ""


def verify_site(
    worktree: pathlib.Path,
    site: dict,
    python: str,
    drift_window: int,
    first_failure: bool,
) -> tuple[str, str]:
    """在单个 worktree 上验证一个位点；返回 ``(判定, 备注)``。"""
    target = site["target"]
    if site["selector"]:
        selector_error = check_selector(worktree, site["selector"])
        if selector_error:
            return VERDICT_COLLECT_ERROR, selector_error
    path = worktree / target
    note = ""
    try:
        subprocess.run(["git", "checkout", "--", target], cwd=worktree, check=True)
        try:
            apply_mutation(path, site["line"], site["col"], site["kind"])
        except SiteNotFound as exc:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            drifted = find_nearest_site(tree, site["line"], site["col"], site["kind"], drift_window)
            if drifted is None:
                return VERDICT_MUTFAIL, f"{exc}；±{drift_window} 行内无唯一最近同 kind 节点"
            note = f"DRIFTED {site['line']}:{site['col']}→{drifted[0]}:{drifted[1]}"
            site["line"], site["col"] = drifted
            apply_mutation(path, drifted[0], drifted[1], site["kind"])
        try:
            returncode, tail = run_pytest(worktree, site["selector"], python, first_failure)
        except subprocess.TimeoutExpired:
            return VERDICT_TIMEOUT, f"pytest 超过 {PYTEST_TIMEOUT}s"
        verdict = classify_pytest_exit(returncode)
        detail = " | ".join(part for part in (note, tail) if part)
        return verdict, detail
    except (SiteNotFound, MutantSyntaxError) as exc:
        return VERDICT_MUTFAIL, str(exc)
    finally:
        subprocess.run(["git", "checkout", "--", target], cwd=worktree, check=True)


def _catch(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


def selftest() -> int:
    """不触碰 worktree 的防呆自检。"""
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  ok   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name} {detail}")

    print("退出码分类:")
    check("pytest 0 → SURVIVED", classify_pytest_exit(0) == VERDICT_SURVIVED)
    check("pytest 1 → KILLED", classify_pytest_exit(1) == VERDICT_KILLED)
    check("pytest 4 → COLLECT_ERROR", classify_pytest_exit(4) == VERDICT_COLLECT_ERROR)
    check("pytest 5 → COLLECT_ERROR", classify_pytest_exit(5) == VERDICT_COLLECT_ERROR)
    check("pytest 2 → ERROR", classify_pytest_exit(2) == VERDICT_ERROR)
    check("pytest 3 → ERROR", classify_pytest_exit(3) == VERDICT_ERROR)

    print("选择器预检:")
    good = (
        "tests/providers/test_security_boundaries.py"
        "::test_secret_free_options_nested_non_string_key_reports_path"
    )
    check("有效选择器通过", check_selector(DEFAULT_REPO, good) is None)
    check("缺 :: 被拒", check_selector(DEFAULT_REPO, "tests/test_cleanup.py") is not None)
    check("缺文件被拒", check_selector(DEFAULT_REPO, "tests/nope.py::test_x") is not None)
    check(
        "缺测试名被拒",
        check_selector(DEFAULT_REPO, "tests/test_cleanup.py::test_no_such_test") is not None,
    )

    print("站点文件解析:")
    sites = parse_sites(
        "# 注释\n\nsrc/a.py|10|4|drop\nsrc/b.py|20|8|eqneq|tests/t.py::test_x|KILLED\n"
    )
    check("注释与空行被忽略", len(sites) == 2)
    check(
        "四字段行无 selector/expected",
        sites[0]["selector"] == "" and sites[0]["expected"] == "",
    )
    check(
        "六字段行解析完整",
        sites[1]["expected"] == "KILLED" and sites[1]["selector"] == "tests/t.py::test_x",
    )
    check(
        "未知 kind 被拒", isinstance(_catch(lambda: parse_sites("src/a.py|10|4|bogus")), ValueError)
    )
    check(
        "未知 expected 被拒",
        isinstance(_catch(lambda: parse_sites("src/a.py|10|4|drop||NOPE")), ValueError),
    )

    print("漂移定位:")
    tree = ast.parse("x = 1\nif a == b:\n    pass\nif c != d:\n    pass\n")
    check("精确命中", find_site(tree, 2, 3, "eqneq") is not None)
    check("未命中抛错", isinstance(_catch(lambda: find_site(tree, 99, 3, "eqneq")), SiteNotFound))
    check("唯一最近漂移", find_nearest_site(tree, 1, 3, "eqneq", 10) == (2, 3))
    check("平局返回 None", find_nearest_site(tree, 3, 3, "eqneq", 10) is None)
    check("窗口外返回 None", find_nearest_site(tree, 1, 3, "eqneq", 0) is None)
    tree_tie = ast.parse("x = 1\nif a == b:\n    pass\n    if c == d:\n        pass\n")
    check("同距同列消解平局", find_nearest_site(tree_tie, 3, 3, "eqneq", 10) == (2, 3))
    check("同距异列返回 None", find_nearest_site(tree_tie, 3, 9, "eqneq", 10) is None)

    print(f"\nselftest 失败: {len(failures)}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="变异验证驱动")
    parser.add_argument("sites", nargs="?", help="站点文件路径")
    parser.add_argument("--repo", default=str(DEFAULT_REPO), help="主仓路径（默认脚本所在仓库）")
    parser.add_argument("--worktrees", default="", help="逗号分隔的 worktree 路径（并行分片）")
    parser.add_argument("--python", default=sys.executable, help="跑 pytest 的解释器")
    parser.add_argument("--drift-window", type=int, default=3, help="漂移定位的最大行距")
    parser.add_argument("--no-first-failure", action="store_true", help="扫描时不加 -x")
    parser.add_argument("--selftest", action="store_true", help="运行防呆自检后退出")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()
    if not args.sites:
        parser.error("缺少站点文件（或用 --selftest）")

    repo = pathlib.Path(args.repo).resolve()
    sites = parse_sites(pathlib.Path(args.sites).read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    worktrees = [
        prepare_worktree(repo, name.strip(), head)
        for name in args.worktrees.split(",")
        if name.strip()
    ]
    if not worktrees:
        raise RuntimeError("至少需要一个 --worktrees")

    lock = threading.Lock()
    results: list[tuple[dict, str, str]] = []
    problems: list[str] = []

    def worker(worktree: pathlib.Path, jobs: list[dict]) -> None:
        for site in jobs:
            tag = f"{site['target']}:{site['line']}:{site['col']}:{site['kind']}"
            try:
                verdict, note = verify_site(
                    worktree, site, args.python, args.drift_window, not args.no_first_failure
                )
            except Exception as exc:  # noqa: BLE001
                verdict, note = VERDICT_ERROR, f"{type(exc).__name__}: {exc}"
            with lock:
                results.append((site, verdict, note))
                line = f"RESULT {verdict} {tag}"
                if site["selector"]:
                    line += f" selector={site['selector']}"
                if note:
                    line += f" | {note}"
                print(line, flush=True)
                if site["expected"]:
                    if verdict != site["expected"]:
                        problems.append(f"{tag}: expected {site['expected']}，实际 {verdict}")
                elif verdict not in (VERDICT_KILLED, VERDICT_SURVIVED):
                    problems.append(f"{tag}: 非判定结果 {verdict} | {note}")

    buckets: list[list[dict]] = [[] for _ in worktrees]
    for index, site in enumerate(sites):
        buckets[index % len(worktrees)].append(site)
    threads = [
        threading.Thread(target=worker, args=(worktree, jobs), daemon=True)
        for worktree, jobs in zip(worktrees, buckets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for worktree in worktrees:
        dirty = subprocess.run(
            ["git", "status", "--short"], cwd=worktree, capture_output=True, text=True, check=True
        ).stdout.strip()
        if dirty:
            problems.append(f"worktree 残留改动 {worktree}:\n{dirty}")

    counts: dict[str, int] = {}
    for _, verdict, _ in results:
        counts[verdict] = counts.get(verdict, 0) + 1
    print(
        f"\n汇总: {len(results)} 个位点 — "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )
    if problems:
        print(f"\n需要处理 {len(problems)} 条:")
        for item in problems:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
