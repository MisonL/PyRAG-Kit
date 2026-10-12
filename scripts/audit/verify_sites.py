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
2. **基线对照**：先跑一次未变异的 pytest。基线不绿（有预先存在的失败）时，
   任何位点都不得进 KILLED/SURVIVED 判定——否则红套件会把真存活点判成
   KILLED（假绿），把真缺口藏起来。
3. 位点未命中时在 ``--drift-window`` 行内做**唯一最近**漂移定位，命中则
   在备注里打印 ``DRIFTED``；平局或无匹配 → MUTFAIL，绝不施加到别的守卫。
4. 选择器先在 worktree 内预检文件与测试名存在——pytest 对不存在的测试名
   返回退出码 4，历史上被 ``returncode != 0`` 误判成「已杀死」。
5. 每站点 try/finally 还原源码；结束复查全部 worktree 清洁。
6. 无 selector 的扫描默认 ``pytest -q -x``（被杀变异秒退）；全量套件是
   唯一判据，没有「子集先行」的两级语义。

用法::

    python verify_sites.py --worktrees /path/wt1,/path/wt2 sites.txt
    python verify_sites.py --selftest
"""

import argparse
import ast
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
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

#: 读**工作区源码文本**（而非导入后行为）的测试节点。变异扫描期间源码处于变异
#: 态，这类测试会因为「源码长什么样变了」而红，与被测行为无关——把它们算进
#: 「有没有测试失败」会让假 KILLED。实测：green39 夹具里 4 个 SURVIVED 位点
#: （``anthropic.py:972``、``factory.py:356``、``model_provider.py:1202``／
#: ``336``）落在 ``test_green39_fixture_sites_resolve_at_head`` 引用的节点上，
#: 该元测试在变异下必红，于是 4 个存活点全部被误报 KILLED；去掉它后全量 4420
#: passed（真·存活）。逐个剔除前要**实测**（同 deselect 跑一次，只剩该测试红才算
#: 收进来）：盲目的 ``--deselect`` 对不存在的 node id 是静默通过的，不构成保护。
_SOURCE_INSPECTING_TESTS = (
    # 夹具坐标与源码 AST 的逐位点比对：变异点任一落进夹具引用的节点即必红。
    "tests/scripts/test_audit_tooling.py::test_green39_fixture_sites_resolve_at_head",
    # 读全仓源码做 AST 级断言（重复装饰符、markup 入口、raw %-logging、
    # provider 的 importlib 隐藏导入、chat 的恢复分支形态）：断言的是源码文本
    # 本身，任何变异都可能让它们变红，与变异点的实际被测行为无关。
    "tests/test_boundary_hardening_regressions.py::test_no_duplicate_decorators_in_source_tree",
    "tests/test_boundary_hardening_regressions.py::test_config_branch_missing_update_method_raises_a_caught_type",
    "tests/test_display_utils.py::test_rich_markup_entry_points_never_receive_raw_variables",
    "tests/test_display_utils.py::test_untrusted_free_argument_allowlist_has_no_stale_entries",
    "tests/test_release_scripts.py::test_registered_provider_modules_have_no_static_import_path",
    "tests/providers/test_security_boundaries.py::test_user_facing_error_paths_do_not_print_raw_exceptions",
)


def classify_pytest_exit(returncode: int) -> str:
    """把 pytest 退出码映射成判定；只有 0/1 是判定。"""
    if returncode == 0:
        return VERDICT_SURVIVED
    if returncode == 1:
        return VERDICT_KILLED
    if returncode in (4, 5):
        return VERDICT_COLLECT_ERROR
    return VERDICT_ERROR


# pytest 因「解释器/依赖不可用」而非「测试失败」退出时，返回码往往是 1（与
# KILLED 同码），只有输出能区分。没有这层嗅探时，``--python`` 指到一个没装
# pytest 的解释器会让**整轮**位点全部误报 KILLED。
_PYTEST_UNAVAILABLE_MARKERS = (
    "No module named pytest",
    "No module named 'pytest'",
    "ERROR: usage:",
    "INTERNALERROR",
    "attempted relative import",
)


def pytest_output_is_infrastructure_failure(output: str) -> bool:
    """pytest 输出是否表明「基础设施不可用」而非「测试失败」。

    这类收场与「变异被杀死」退出码相同（1），必须先排除，否则整轮判定失真。
    """
    return any(marker in output for marker in _PYTEST_UNAVAILABLE_MARKERS)


def check_selector(worktree: pathlib.Path, selector: str) -> str | None:
    """在 worktree 内预检 ``文件::测试名``；返回错误原因或 ``None``。"""
    if "::" not in selector:
        return f"选择器缺少 ``::``: {selector!r}"
    file_part, _, test_name = selector.partition("::")
    path = worktree / file_part
    if not path.is_file():
        return f"选择器文件不存在: {file_part}"
    source = path.read_text(encoding="utf-8")
    # 不能用 ``^def name(``：该正则只匹配**顶格**函数，类内方法、任何缩进 def、
    # async def 全被误判「测试不存在」→ 假 COLLECT_ERROR，而 pytest 本可收集。
    # 按 AST 遍历 ``FunctionDef``/``AsyncFunctionDef`` 取名字，与 pytest 的
    # 实际收集方式一致。
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"选择器文件语法错误: {file_part}: {exc}"
    test_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if test_name not in test_names:
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
        if not target:
            raise ValueError(f"第 {number} 行 target 为空: {raw!r}")
        if lineno < 1:
            raise ValueError(f"第 {number} 行行号必须 ≥1，实际 {lineno}: {raw!r}")
        if col < 0:
            raise ValueError(f"第 {number} 行列偏移必须 ≥0，实际 {col}: {raw!r}")
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

    **自指测试必须排除**：套件里有读**工作区源码文本**的元测试（校验夹具坐标
    是否命中当前源码）。扫描时工作区正处于变异态，变异点一旦落在元测试引用的
    节点上，元测试必红——它测的不是被测行为，而是「源码本身长什么样」。把它算进
    「有没有测试失败」的判据，会让**任何**指向夹具所引用节点的存活变异被误报
    KILLED（实测 green39 夹具里 4 个 SURVIVED 位点因此全部假报 KILLED）。
    这些测试对工具自身仍有价值，但只在**未变异**的源码上跑，见
    ``_SOURCE_INSPECTING_TESTS``。
    """
    args = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if first_failure:
        args.append("-x")
    for node in _SOURCE_INSPECTING_TESTS:
        args += ["--deselect", node]
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
    output = proc.stdout + proc.stderr
    if pytest_output_is_infrastructure_failure(output):
        # 退出码 1 在这里不代表「变异被杀死」——解释器/依赖不可用也是 1。
        # 返回一个非判定码（2）让上层归 ENVIRONMENT 而非 KILLED。
        return 2, (lines[-1] if lines else "pytest 基础设施不可用")
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


def _probe_selector_ast() -> str | None:
    """把一段含类方法与 async def 的源码写进临时文件，验证 ``check_selector``
    的 AST 版本能同时识别这三类声明（旧的正则只认顶格 def）。"""
    source = (
        "def test_plain():\n"
        "    pass\n"
        "class TestX:\n"
        "    def test_method_inside_class(self):\n"
        "        pass\n"
        "async def test_async_top_level():\n"
        "    pass\n"
    )
    worktree = pathlib.Path(tempfile.mkdtemp())
    (worktree / "t.py").write_text(source, encoding="utf-8")
    try:
        return check_selector(worktree, "t.py::test_method_inside_class") or (
            check_selector(worktree, "t.py::test_async_top_level")
            or check_selector(worktree, "t.py::test_plain")
        )
    finally:
        shutil.rmtree(worktree, ignore_errors=True)


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
    # 旧实现用 ``^def name(`` 正则，只认顶格函数——类方法与 async def 被误判
    # 「测试不存在」。AST 遍历必须能识别这两类。
    check(
        "类方法/缩进声明被识别",
        _probe_selector_ast() is None,
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
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="跳过基线对照（自担风险：红套件会把存活点误判成 KILLED）",
    )
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

    # 基线对照：先跑一次**未变异**的套件。基线不绿时任何位点都不能进
    # KILLED/SURVIVED 判定——红套件会把真存活点判成 KILLED（假绿），
    # 把真缺口藏起来。这是与「退出码分类」同等重要的前置门。
    if not args.no_baseline:
        print("基线对照中（未变异全量套件）...", flush=True)
        for worktree in worktrees:
            try:
                rc, tail = run_pytest(worktree, "", args.python, first_failure=False)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"基线 pytest 超时 {PYTEST_TIMEOUT}s：{worktree}") from None
            if rc != 0:
                raise RuntimeError(
                    f"基线套件不绿（退出码 {rc}）于 {worktree}：{tail}\n"
                    " 预先存在的失败会让红套件把存活点误判成 KILLED。"
                    " 请先在干净 HEAD 上跑全量套件并修复，或加 --no-baseline 跳过（自担风险）。"
                )
        print("基线绿。", flush=True)

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
