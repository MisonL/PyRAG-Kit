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
    assert counts == {"KILLED": 33, "SURVIVED": 5, "MUTFAIL": 1}


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
