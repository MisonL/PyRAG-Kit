"""scripts/audit 变异验证工具链的回归测试。

覆盖三类防呆保护：退出码分类（pytest 4/5 不得被当成「已杀死」）、选择器
预检（不存在的测试名必须报 COLLECT_ERROR 而不是假红）、漂移定位（唯一
最近才采用，平局/无匹配必须 MUTFAIL）。
"""

import ast
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import mutate
from mutate import (
    MutantSyntaxError,
    SiteNotFound,
    apply_mutation,
    find_nearest_site,
    find_site,
)
from verify_sites import (
    VERDICT_COLLECT_ERROR,
    VERDICT_KILLED,
    VERDICT_SURVIVED,
    classify_pytest_exit,
    parse_sites,
    prepare_worktree,
)

VERIFY_SITES = REPO_ROOT / "scripts" / "audit" / "verify_sites.py"


def test_classify_pytest_exit_only_zero_and_one_are_verdicts():
    assert classify_pytest_exit(0) == VERDICT_SURVIVED
    assert classify_pytest_exit(1) == VERDICT_KILLED
    # 退出码 4/5 是「没有收集到测试」——历史上被 returncode != 0 误判成红。
    assert classify_pytest_exit(4) == VERDICT_COLLECT_ERROR
    assert classify_pytest_exit(5) == VERDICT_COLLECT_ERROR
    assert classify_pytest_exit(2) not in (VERDICT_KILLED, VERDICT_SURVIVED)
    assert classify_pytest_exit(3) not in (VERDICT_KILLED, VERDICT_SURVIVED)


def test_parse_sites_fields_and_validation():
    sites = parse_sites(
        "# 注释\n\nsrc/a.py|10|4|drop\nsrc/b.py|20|8|eqneq|tests/t.py::test_x|KILLED\n"
    )
    assert len(sites) == 2
    assert sites[0]["selector"] == "" and sites[0]["expected"] == ""
    assert sites[1]["selector"] == "tests/t.py::test_x"
    assert sites[1]["expected"] == "KILLED"
    with pytest.raises(ValueError):
        parse_sites("src/a.py|10|4|bogus")
    with pytest.raises(ValueError):
        parse_sites("src/a.py|10|4|drop||NOPE")


def test_mutate_drop_removes_not(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text('def f(flag):\n    if not flag:\n        return "off"\n    return "on"\n')
    apply_mutation(target, 2, 7, "drop")
    assert "if flag:" in target.read_text(encoding="utf-8")


def test_mutate_eqneq_both_directions(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("def f(a, b):\n    if a == b:\n        return 1\n    return 0\n")
    apply_mutation(target, 2, 7, "eqneq")
    assert "a != b" in target.read_text(encoding="utf-8")
    target.write_text("def f(a, b):\n    if a != b:\n        return 1\n    return 0\n")
    apply_mutation(target, 2, 7, "eqneq")
    assert "a == b" in target.read_text(encoding="utf-8")


def test_mutate_cmpdir_reverses_direction_in_both_shapes(tmp_path):
    """``<``↔``>`` 与 ``<=``↔``>=`` 各自反向；``<=`` 不能被切出 ``> =``。"""
    target = tmp_path / "mod.py"
    target.write_text("def f(a, b):\n    return a <= b\n")
    apply_mutation(target, 2, 11, "cmpdir")
    assert "a >= b" in target.read_text(encoding="utf-8")

    target.write_text("def f(a, b):\n    return a >= b\n")
    apply_mutation(target, 2, 11, "cmpdir")
    assert "a <= b" in target.read_text(encoding="utf-8")

    target.write_text("def f(a, b):\n    return a < b\n")
    apply_mutation(target, 2, 11, "cmpdir")
    assert "a > b" in target.read_text(encoding="utf-8")

    target.write_text("def f(a, b):\n    return a > b\n")
    apply_mutation(target, 2, 11, "cmpdir")
    assert "a < b" in target.read_text(encoding="utf-8")


def test_mutate_cmpdir_flips_only_first_operator_of_chain(tmp_path):
    """链式比较 ``0 <= x < 10`` 只翻转第一个运算符（与 eqneq 同约定）。"""
    target = tmp_path / "mod.py"
    target.write_text("def f(x):\n    return 0 <= x < 10\n")
    apply_mutation(target, 2, 11, "cmpdir")
    assert "0 >= x < 10" in target.read_text(encoding="utf-8")


def test_mutate_boolc_flips_literal_both_directions(tmp_path):
    """``True``↔``False`` 两条方向都要翻转。"""
    target = tmp_path / "mod.py"
    target.write_text("def f(flag=True):\n    return flag\n")
    apply_mutation(target, 1, 11, "boolc")
    assert "flag=False" in target.read_text(encoding="utf-8")

    target.write_text("def f(flag=False):\n    return flag\n")
    apply_mutation(target, 1, 11, "boolc")
    assert "flag=True" in target.read_text(encoding="utf-8")


def test_mutate_boolc_ignores_non_boolean_constants():
    """``1``／``0``／``"True"``／``None`` 取值不同型，不是 boolc 的位点。

    ``bool`` 是 ``int`` 的子类，判据必须先按取值类型筛（``isinstance(value,
    bool)``）再收位点；只按 ``Constant`` 收会把 ``1``／``0`` 也当成布尔字面量，
    变异落到数字上、``boolc`` 的结论随之失真。
    """
    tree = ast.parse('x = 1\ny = "True"\nz = None\nw = True\nv = False\n')
    assert [(ln, col) for ln, col, _ in mutate.iter_sites(tree, "boolc")] == [(4, 4), (5, 4)]


def test_mutate_boolop_swaps_and_or_both_directions(tmp_path):
    """``and``→``or``、``or``→``and`` 两个方向都可施加。"""
    target = tmp_path / "mod.py"
    target.write_text("ok = a and b\n")
    # 站点键是第一个值 ``a`` 的结尾 (1, 6)。
    apply_mutation(target, 1, 6, "boolop")
    assert "ok = a or b" in target.read_text(encoding="utf-8")

    target.write_text("ok = a or b\n")
    apply_mutation(target, 1, 6, "boolop")
    assert "ok = a and b" in target.read_text(encoding="utf-8")


def test_mutate_boolop_uses_first_value_end_as_site_key():
    """链式 ``a and b or c`` 的两层 BoolOp 起点相同，故键取第一个值的结尾。"""
    tree = ast.parse("x = a and b or c\n")
    assert sorted((ln, col) for ln, col, _ in mutate.iter_sites(tree, "boolop")) == [
        (1, 5),
        (1, 11),
    ]


def test_mutate_boolop_flips_only_first_operator(tmp_path):
    """``a and b and c`` 只翻最内层第一个运算符，后面的保持不变。"""
    target = tmp_path / "mod.py"
    target.write_text("ok = a and b and c\n")
    apply_mutation(target, 1, 6, "boolop")
    assert "ok = a or b and c" in target.read_text(encoding="utf-8")


def test_mutate_boolop_comment_mentioning_keyword_is_not_flipped(tmp_path):
    """夹缝里有注释且注释含 and/or 文本时，必须翻真运算符而非注释词。

    ``src/`` 生产代码不这么写，但正则按关键字找会先命中注释（实测形态
    ``a  # keep and note`` 后换行 ``and b``），变异落进注释即无声无效。
    """
    target = tmp_path / "mod.py"
    target.write_text("ok = (\n    a  # keep and note\n    and b\n)\n")
    apply_mutation(target, 2, 5, "boolop")
    source = target.read_text(encoding="utf-8")
    assert "# keep and note" in source  # 注释原样保留
    assert "    or b\n" in source  # 真运算符被翻转


def test_mutate_arith_swaps_add_sub_and_mul_div(tmp_path):
    """加减互换、乘除互换各一条，且各自只翻转到对面运算符。"""
    target = tmp_path / "mod.py"
    target.write_text("total = a + b\navg = total / n\n")
    # ``total = a + b``：站点键是左操作数 ``a`` 的结尾 (1, 9)。
    apply_mutation(target, 1, 9, "arith")
    assert "total = a - b" in target.read_text(encoding="utf-8")

    target.write_text("total = a - b\navg = total / n\n")
    apply_mutation(target, 1, 9, "arith")
    assert "total = a + b" in target.read_text(encoding="utf-8")

    target.write_text("avg = total * n\n")
    apply_mutation(target, 1, 11, "arith")
    assert "avg = total / n" in target.read_text(encoding="utf-8")


def test_mutate_arith_uses_left_operand_end_as_site_key():
    """链式 ``a + b + c`` 的两层 BinOp 起点相同，故键取左操作数结尾。

    若按 BinOp 的 ``col_offset`` 作键，两层都指向第一个 ``a``，``find_site``
    会命中两个节点而判 MUTFAIL，链式位点整片漏掉。
    """
    tree = ast.parse("r = a + b + c\n")
    assert sorted((ln, col) for ln, col, _ in mutate.iter_sites(tree, "arith")) == [
        (1, 5),
        (1, 9),
    ]


def test_mutate_arith_ignores_string_concatenation():
    """字符串字面量参与的 ``+`` 是拼接不是算术，不进 arith 位点。

    其变异体（``"a" - "b"``）在运行期抛 ``TypeError`` 而非改变计算，属噪声。
    """
    tree = ast.parse('x = "a" + "b"\ny = a + b\nz = a + "s"\n')
    assert [(ln, col) for ln, col, _ in mutate.iter_sites(tree, "arith")] == [(2, 5)]


def test_mutate_arith_flips_only_first_operator_of_chain(tmp_path):
    """``x / y / z`` 的两个 ``/`` 各是一个位点，一次只翻一个。"""
    target = tmp_path / "mod.py"
    target.write_text("w = x / y / z\n")
    apply_mutation(target, 1, 5, "arith")
    assert "w = x * y / z" in target.read_text(encoding="utf-8")

    target.write_text("w = x / y / z\n")
    apply_mutation(target, 1, 9, "arith")
    assert "w = x / y * z" in target.read_text(encoding="utf-8")


def test_mutate_column_is_utf8_byte_offset(tmp_path):
    """含中文的行：col_offset 是 UTF-8 字节偏移，不是字符偏移。"""
    target = tmp_path / "mod.py"
    target.write_text('x = "中文" if not flag else "b"\n')
    tree = ast.parse(target.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.UnaryOp))
    apply_mutation(target, node.lineno, node.col_offset, "drop")
    assert target.read_text(encoding="utf-8") == 'x = "中文" if flag else "b"\n'


def test_mutate_unknown_site_raises(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n")
    with pytest.raises(SiteNotFound):
        apply_mutation(target, 99, 0, "drop")


def test_apply_mutation_rejects_syntax_error_without_writing(tmp_path, monkeypatch):
    """变异体语法错误必须报 MutantSyntaxError 且不写盘。

    语法错误的变异体会让 pytest 在收集期失败，历史上被扫描器记成
    「已杀死」——那是假阴性，会把真缺口藏起来。
    """
    target = tmp_path / "mod.py"
    target.write_text("if a == b:\n    pass\n")
    monkeypatch.setattr(mutate, "mutate_source", lambda src, node, kind: "if (a == b:\n")
    with pytest.raises(MutantSyntaxError):
        apply_mutation(target, 1, 3, "eqneq")
    assert target.read_text(encoding="utf-8") == "if a == b:\n    pass\n"


def test_find_nearest_site_unique_and_tie_break():
    tree = ast.parse("x = 1\nif a == b:\n    pass\nif c != d:\n    pass\n")
    assert find_nearest_site(tree, 1, 3, "eqneq", 10) == (2, 3)
    assert find_nearest_site(tree, 3, 3, "eqneq", 10) is None  # 平局
    assert find_nearest_site(tree, 1, 3, "eqneq", 0) is None  # 窗口外
    tie_tree = ast.parse("x = 1\nif a == b:\n    pass\n    if c == d:\n        pass\n")
    assert find_nearest_site(tie_tree, 3, 3, "eqneq", 10) == (2, 3)  # 同距同列消解
    assert find_nearest_site(tie_tree, 3, 9, "eqneq", 10) is None  # 同距异列


def test_find_site_requires_exactly_one_match():
    tree = ast.parse("if a == b:\n    pass\n")
    assert find_site(tree, 1, 3, "eqneq") is not None
    with pytest.raises(SiteNotFound):
        find_site(tree, 99, 3, "eqneq")


def test_source_inspecting_tests_are_excluded_from_verdicts():
    """读工作区源码文本的测试必须被排除在判定之外。

    实测：green39 夹具里 4 个 SURVIVED 位点恰好落进
    ``test_green39_fixture_sites_resolve_at_head`` 引用的节点，元测试在变异下必红
    ——扫描于是把 4 个存活点全部误报 KILLED。这些测试在**未变异**的源码上仍然
    有价值，所以是排除而非删除；但排除列表必须是**存在**的，``--deselect`` 对
    不存在的 node id 静默通过，写错名字不会报错。
    """
    from verify_sites import _SOURCE_INSPECTING_TESTS

    assert _SOURCE_INSPECTING_TESTS, "排除列表不得为空——空列表意味着自指测试又被算进判定"
    for node in _SOURCE_INSPECTING_TESTS:
        file_part, _, test_name = node.partition("::")
        assert file_part.endswith(".py") and test_name, node
        path = REPO_ROOT / file_part
        assert path.exists(), f"排除项指向不存在的文件: {file_part}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = any(
            isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == test_name
            for top in tree.body
            for fn in _iter_functions(top)
        )
        assert found, f"排除项指向不存在的测试: {node}"


def test_source_inspecting_registry_is_complete():
    """排除清单必须覆盖探测器发现的**每一个**自指测试。

    上面的用例只验证「清单里的条目存在」——一个新增的、读源码的测试不会自己
    进清单，扫描便再次把它算进判定（假 KILLED 静默复现）。这里用静态探测器
    做完整性核对：凡「锚定仓库源码 + 读文件/解析/导入」的测试都必须已登记。
    """
    from verify_sites import _SOURCE_INSPECTING_TESTS, detect_source_inspecting_tests

    detected = set(detect_source_inspecting_tests(REPO_ROOT))
    registered = set(_SOURCE_INSPECTING_TESTS)
    # 探测器先自证非空，否则「⊆」在空集上恒真、这条用例形同虚设。
    assert len(detected) >= 5, f"探测器只发现 {len(detected)} 个，锚点匹配可能已失效"
    assert detected <= registered, (
        "以下测试读仓库源码但未登记进 _SOURCE_INSPECTING_TESTS，"
        f"变异扫描会把它们算进判定并制造假 KILLED：{sorted(detected - registered)}"
    )


def test_detect_source_inspecting_distinguishes_source_from_artifacts(tmp_path):
    """探测器要抓读源码的、放过只读临时产物的——后者是变异的合法判据。"""
    from verify_sites import detect_source_inspecting_tests

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_probe.py").write_text(
        "import ast\n"
        "import pathlib\n"
        "\n"
        "REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]\n"
        "\n"
        "\n"
        "def test_reads_source_via_literal():\n"
        "    return pathlib.Path('src/providers/factory.py').read_text()\n"
        "\n"
        "\n"
        "def test_reads_source_via_module_global():\n"
        "    return ast.parse((REPO_ROOT / 'scripts' / 'audit' / 'mutate.py').read_text())\n"
        "\n"
        "\n"
        "def test_reads_only_temp_artifacts(tmp_path):\n"
        "    return (tmp_path / 'chunks.pkl').read_bytes()\n"
        "\n"
        "\n"
        "def test_docstring_mention_is_not_a_read(tmp_path):\n"
        '    """这个测试从不真的 read_text，只在文档里提到 src/ 而已。"""\n'
        "    return tmp_path.name\n",
        encoding="utf-8",
    )
    detected = set(detect_source_inspecting_tests(tmp_path))
    assert "tests/test_probe.py::test_reads_source_via_literal" in detected
    assert "tests/test_probe.py::test_reads_source_via_module_global" in detected
    assert "tests/test_probe.py::test_reads_only_temp_artifacts" not in detected
    assert "tests/test_probe.py::test_docstring_mention_is_not_a_read" not in detected


def _iter_functions(top):
    """遍历模块级 ClassDef 内的函数（含嵌套类里的方法）。"""
    if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield top
    elif isinstance(top, ast.ClassDef):
        for stmt in top.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield stmt
            elif isinstance(stmt, ast.ClassDef):
                yield from _iter_functions(stmt)


def test_verify_sites_selftest_passes():
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SITES), "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "selftest 失败: 0" in proc.stdout


def test_green39_fixture_parses_with_expected_verdicts():
    fixture = REPO_ROOT / "scripts" / "audit" / "regressions" / "green39.txt"
    sites = parse_sites(fixture.read_text(encoding="utf-8"))
    assert len(sites) == 39
    counts: dict[str, int] = {}
    for site in sites:
        counts[site["expected"]] = counts.get(site["expected"], 0) + 1
    assert counts == {"KILLED": 35, "SURVIVED": 4}


def test_green39_fixture_sites_resolve_at_head():
    """夹具坐标必须在当前 HEAD 上精确命中，否则它在静默失效。

    历史翻车：夹具记录在扫描当轮的坐标上，源码后续演化（行号漂移、守卫随修复
    移动、死代码删除整段移位）后，39 个位点里 10 个已无法定位、10 个需漂移。
    夹具却只按「字段可解析」放行，于是「已知答案」失去锚定意义。这条断言把
    「坐标命中当前 HEAD」钉死：凡报 SiteNotFound 即夹具需重新基线化。
    """
    fixture = REPO_ROOT / "scripts" / "audit" / "regressions" / "green39.txt"
    sites = parse_sites(fixture.read_text(encoding="utf-8"))
    stale = []
    for site in sites:
        target = REPO_ROOT / site["target"]
        tree = ast.parse(target.read_text(encoding="utf-8"))
        try:
            find_site(tree, site["line"], site["col"], site["kind"])
        except SiteNotFound:
            stale.append(f"{site['target']}:{site['line']}:{site['col']}:{site['kind']}")
    assert not stale, "夹具坐标已在 HEAD 上失效，需重新基线化：\n" + "\n".join(stale)


def test_pytest_infrastructure_failure_is_not_killed():
    """解释器/依赖不可用（退出码也是 1）不得被当成 KILLED。

    ``--python`` 指到没装 pytest 的解释器时，整轮位点会全部误报 KILLED；
    这层嗅探把「基础设施不可用」与「测试失败」区分开。
    """
    from verify_sites import pytest_output_is_infrastructure_failure

    assert pytest_output_is_infrastructure_failure("/usr/bin/python: No module named pytest")
    assert pytest_output_is_infrastructure_failure("INTERNALERROR> ...")
    assert not pytest_output_is_infrastructure_failure("1 failed, 10 passed in 2.0s")


def test_selector_precheck_recognizes_class_methods_and_async(tmp_path):
    """选择器预检必须用 AST 而非 ``^def`` 正则：类方法与 async def 也不能误判。"""
    from verify_sites import check_selector

    source = (
        "def test_plain():\n"
        "    pass\n"
        "class TestX:\n"
        "    def test_method_inside_class(self):\n"
        "        pass\n"
        "async def test_async_top_level():\n"
        "    pass\n"
    )
    (tmp_path / "t.py").write_text(source, encoding="utf-8")
    assert check_selector(tmp_path, "t.py::test_plain") is None
    assert check_selector(tmp_path, "t.py::test_method_inside_class") is None
    assert check_selector(tmp_path, "t.py::test_async_top_level") is None
    assert check_selector(tmp_path, "t.py::test_missing") is not None


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def audit_repo(tmp_path):
    """最小 git 仓库 + 一个 worktree：守卫、能杀死它的测试、无覆盖的守卫。"""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / ".gitignore").write_text("__pycache__/\n")
    # 夹具仓库必须与 pytest 走同一套导入路径，否则 ``python -m pytest`` 在
    # worktree 里收集不到 ``src``。
    (repo / "conftest.py").write_text(
        "import pathlib, sys\n"
        "ROOT = pathlib.Path(__file__).resolve().parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n",
        encoding="utf-8",
    )
    (repo / "src" / "guard.py").write_text(
        "def check(value):\n"
        '    if value == "expected":\n'
        '        return "ok"\n'
        '    return "bad"\n'
        "\n"
        "def other(value):\n"
        '    return value == "x"\n'
    )
    (repo / "tests" / "test_guard.py").write_text(
        'from src.guard import check\n\n\ndef test_check():\n    assert check("expected") == "ok"\n'
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "audit@example.com")
    _git(repo, "config", "user.name", "audit")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "--detach", str(worktree))
    return repo, worktree


def test_prepare_worktree_refuses_main_repo(audit_repo):
    repo, _ = audit_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(RuntimeError, match="主仓"):
        prepare_worktree(repo, str(repo), head)


def test_prepare_worktree_refuses_dirty(audit_repo):
    repo, worktree = audit_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    (worktree / "src" / "guard.py").write_text("# 残留改动\n")
    with pytest.raises(RuntimeError, match="不干净"):
        prepare_worktree(repo, str(worktree), head)


def test_verify_sites_end_to_end(audit_repo, tmp_path):
    """KILLED / SURVIVED / COLLECT_ERROR 三种判定各一，退出码 0，worktree 无残留。"""
    repo, worktree = audit_repo
    sites_file = tmp_path / "sites.txt"
    sites_file.write_text(
        "# 端到端\n"
        "src/guard.py|2|7|eqneq|tests/test_guard.py::test_check|KILLED\n"
        "src/guard.py|7|11|eqneq||SURVIVED\n"
        "src/guard.py|2|7|eqneq|tests/test_guard.py::test_missing|COLLECT_ERROR\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SITES),
            "--repo",
            str(repo),
            "--worktrees",
            str(worktree),
            str(sites_file),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT KILLED src/guard.py:2:7:eqneq" in proc.stdout
    assert "RESULT SURVIVED src/guard.py:7:11:eqneq" in proc.stdout
    assert "RESULT COLLECT_ERROR src/guard.py:2:7:eqneq" in proc.stdout
    status = subprocess.run(
        ["git", "status", "--short"], cwd=worktree, capture_output=True, text=True, check=False
    ).stdout
    assert status.strip() == ""


def test_verify_sites_baseline_guard_rejects_red_suite(tmp_path):
    """基线不绿时整轮必须中止，不得把存活点误判成 KILLED。

    这是「无基线对照」那条历史翻车的回归：红套件会让真存活点报 KILLED，
    把真缺口藏起来。夹具仓库里直接提交一个常红测试（不能是 worktree 未跟踪
    文件——``prepare_worktree`` 会拒绝脏 worktree），verify_sites 必须以非零
    退出并在输出里说明基线不绿，而不是照常输出 KILLED。
    """
    repo = tmp_path / "red-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import pathlib, sys\n"
        "ROOT = pathlib.Path(__file__).resolve().parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n",
        encoding="utf-8",
    )
    (repo / "src" / "guard.py").write_text(
        'def check(value):\n    if value == "expected":\n        return "ok"\n    return "bad"\n',
        encoding="utf-8",
    )
    (repo / "tests" / "test_always_red.py").write_text(
        "def test_always_red():\n    assert False, 'baseline is red'\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "audit@example.com")
    _git(repo, "config", "user.name", "audit")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    worktree = tmp_path / "red-wt"
    _git(repo, "worktree", "add", "--detach", str(worktree))

    sites_file = tmp_path / "sites.txt"
    sites_file.write_text("src/guard.py|2|7|eqneq||SURVIVED\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SITES),
            "--repo",
            str(repo),
            "--worktrees",
            str(worktree),
            str(sites_file),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode != 0, "基线不绿时 verify_sites 必须非零退出"
    assert "基线" in proc.stdout + proc.stderr
    assert "RESULT KILLED" not in proc.stdout
