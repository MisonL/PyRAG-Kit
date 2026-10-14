import os
import subprocess
import sys
from pathlib import Path

import pytest

import main

REPO_ROOT = Path(__file__).resolve().parents[1]

# 冻结分支（main.py 顶层）只在 import 期跑一次，无法用 monkeypatch 触发，
# 因此用子进程伪造 ``sys.frozen`` / ``sys.executable`` 后重新 import。
_FROZEN_PROBE = """
import os, sys

sys.frozen = True
sys.executable = {executable!r}
import main  # noqa: F401

print("TIKTOKEN_CACHE_DIR=" + os.environ.get("TIKTOKEN_CACHE_DIR", "<unset>"))
"""


def _run_frozen_probe(executable: str, cache_dir: str | None = None) -> subprocess.CompletedProcess:
    """在伪造的冻结环境里 import main，返回子进程结果。

    ``cache_dir`` 为 ``None`` 时清除 ``TIKTOKEN_CACHE_DIR``；给值则预置为它——
    预置那一路是区分「赋值」与 ``setdefault`` 的唯一办法：``setdefault`` 在变量
    已存在时会保留用户那个值，而用户若把该变量设成空串，tiktoken 的
    ``read_file_cached`` 会直接关闭缓存，随包分发的 BPE 文件就被整个绕开。
    """
    env = dict(os.environ)
    if cache_dir is None:
        env.pop("TIKTOKEN_CACHE_DIR", None)
    else:
        env["TIKTOKEN_CACHE_DIR"] = cache_dir
    return subprocess.run(
        [sys.executable, "-c", _FROZEN_PROBE.format(executable=executable)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


# ``_resolve_version`` 在**模块导入期**执行，早于 ``main()`` 的任何 ``try``，
# 异常会直接终止进程。它自带两级兜底（VERSION.txt → pyproject.toml → "0.0.0"），
# 但原实现只覆盖了「文件不存在」——读取本身也会失败。
#
# 实测 VERSION.txt 编码损坏（被编辑器写成 UTF-16、或被截断）时 ``read_text``
# 抛 ``UnicodeDecodeError``，程序连横幅都打不出来就退出（exit code 1，消息是
# 裸 traceback）。这与设计意图相反：兜底链的意义正是「读不到就降级」，而读不到
# 的原因不止「不存在」。
#
# 用子进程测：``VERSION = _resolve_version()`` 在 import 期求值，本进程里
# 已经定值，monkeypatch 改不到。``resolve_app_root`` 取的是可执行文件所在目录
# （源码运行时是仓库根），因此把仓库根当作 app_root、在其中放一个坏的
# VERSION.txt 即可复现——但那会污染工作树，所以改为在子进程里替换
# ``resolve_app_root`` 的返回值。
_BAD_VERSION_PROBE = """
import pathlib, tempfile

root = pathlib.Path(tempfile.mkdtemp())
(root / "VERSION.txt").write_bytes({payload!r})
(root / "pyproject.toml").write_text(
    '[project]' + chr(10) + 'name = "pyrag-kit"' + chr(10) + 'version = "9.9.9"' + chr(10),
    encoding="utf-8",
)

import main
# ``_resolve_version`` 读的是模块全局里的 ``resolve_app_root``，改它即可把
# 兜底路径指向临时目录，避免污染工作树。
main._resolve_version.__globals__["resolve_app_root"] = lambda: root
print("VERSION=" + main._resolve_version())
"""


# ``pyproject.toml`` 形态异常时同样必须降级。``["project"]`` 与
# ``["version"]`` 在下标形态不对时抛 ``TypeError``（``project = "x"`` ->
# ``string indices must be integers``；``project = 5`` -> ``'int' object is not
# subscriptable``），**不是** ``KeyError``——原实现的 ``except`` 元组里没有它，
# 而本函数在模块导入期执行，进程会以裸 traceback 退出。
_MALFORMED_PYPROJECT_PROBE = """
import pathlib, tempfile

root = pathlib.Path(tempfile.mkdtemp())
(root / "pyproject.toml").write_text({content!r}, encoding="utf-8")

import main
main._resolve_version.__globals__["resolve_app_root"] = lambda: root
print("VERSION=" + main._resolve_version())
"""


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("project 是字符串", 'project = "string"'),
        ("project 是列表", "project = [1, 2]"),
        ("project 是整数", "project = 5"),
        ("version 是列表", '[project]\nversion = ["1.4.0"]'),
    ],
    ids=["project-string", "project-list", "project-int", "version-list"],
)
def test_resolve_version_survives_malformed_pyproject(label, content):
    """``pyproject.toml`` 形态异常时必须降级到 ``0.0.0``，不能让导入期抛异常。"""
    result = subprocess.run(
        [sys.executable, "-c", _MALFORMED_PYPROJECT_PROBE.format(content=content)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"{label}: 进程退出码 {result.returncode}\n{result.stderr[-400:]}"
    )
    assert "VERSION=0.0.0" in result.stdout, f"{label}: 未降级到 0.0.0\n{result.stdout}"


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("UTF-16 BOM 残留", b"\xff\xfe\x00bad"),
        ("非法起始字节", b"\xffbad"),
        ("截断的多字节序列", b"1.4.0\xe4\xb8"),
    ],
    ids=["utf16", "invalid-start", "truncated"],
)
def test_resolve_version_survives_undecodable_version_file(label, payload):
    """VERSION.txt 无法解码时必须降级，不能让程序在导入期就退出。"""
    result = subprocess.run(
        [sys.executable, "-c", _BAD_VERSION_PROBE.format(payload=payload)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"{label}: 进程退出码 {result.returncode}\n{result.stderr[-400:]}"
    )
    assert "VERSION=9.9.9" in result.stdout, f"{label}: 未降级到 pyproject.toml\n{result.stdout}"


def test_run_cli_uses_smoke_test(monkeypatch):
    called = {"count": 0}

    def fake_run_smoke_test():
        called["count"] += 1
        return 0

    monkeypatch.setattr(main, "run_smoke_test", fake_run_smoke_test)

    result = main.run_cli(["--smoke-test"])

    assert result == 0
    assert called["count"] == 1


def test_frozen_with_bundled_tiktoken_cache_sets_env(tmp_path):
    """包内缓存存在时应把它设为 ``TIKTOKEN_CACHE_DIR``。

    必须是赋值而不是 ``setdefault``：否则用户环境里一个空的
    ``TIKTOKEN_CACHE_DIR``（等于关闭缓存）会把随包分发的 BPE 文件整个绕开。
    """
    app_root = tmp_path / "PyRAG-Kit"
    (app_root / "tiktoken_cache").mkdir(parents=True)

    result = _run_frozen_probe(str(app_root / "PyRAG-Kit"))

    assert result.returncode == 0, result.stderr
    assert f"TIKTOKEN_CACHE_DIR={app_root / 'tiktoken_cache'}" in result.stdout
    assert "未找到包内的分词编码缓存" not in result.stderr


def test_frozen_overrides_preexisting_empty_tiktoken_cache_dir(tmp_path):
    """用户环境里已有的 ``TIKTOKEN_CACHE_DIR`` 不得把包内缓存挤掉。"""
    app_root = tmp_path / "PyRAG-Kit"
    (app_root / "tiktoken_cache").mkdir(parents=True)

    result = _run_frozen_probe(str(app_root / "PyRAG-Kit"), cache_dir="")

    assert result.returncode == 0, result.stderr
    assert f"TIKTOKEN_CACHE_DIR={app_root / 'tiktoken_cache'}" in result.stdout


def test_frozen_without_bundled_tiktoken_cache_warns(tmp_path):
    """冻结运行时若包内没有 BPE 缓存，必须显式警告。

    包内文件是发布包唯一的离线兜底；缺失时 tiktoken 会走
    ``requests.get(blobpath)``，而该调用**没有超时**，网络黑洞下会长时间挂起。
    给出提示总好过让用户面对一个没有线索的卡顿。
    """
    result = _run_frozen_probe(str(tmp_path / "PyRAG-Kit"))

    assert result.returncode == 0, result.stderr
    assert "未找到包内的分词编码缓存" in result.stderr
    assert "TIKTOKEN_CACHE_DIR=<unset>" in result.stdout


# ── 回归：顶层错误边界不得被异常消息里的标记形态打断 ──
#
# ``console.print(str)`` 的字符串会被 rich 按标记解析。修复前
# ``safe_exception_text()`` 的结果再次进入这个通道：异常消息里带 ``[/]`` 时
# 连「报告错误」本身都抛 ``MarkupError``，三层兜底（内层 except、``main()``
# 的 except、``run_cli`` 的 except）全部失效，程序以裸 traceback 退出。
# 这条链路由一条含 ``[/]`` 的知识库文档触发，已端到端复现。


def test_run_cli_survives_markup_like_exception(monkeypatch, capsys):
    """异常消息含标记形态时，启动失败提示必须能打出来且退出码为 1。"""

    def failing_main():
        raise RuntimeError("bad tag [/] in message")

    monkeypatch.setattr(main, "main", failing_main)

    result = main.run_cli([])

    captured = capsys.readouterr()
    assert result == 1
    assert "bad tag [/] in message" in captured.out


def test_run_cli_survives_markup_like_config_error(monkeypatch, capsys):
    """异常消息形如 rich 的闭合标记（``[/red]``）时同样不得二次抛错。"""

    def failing_main():
        raise ValueError("closing tag [/red] at position 3")

    monkeypatch.setattr(main, "main", failing_main)

    result = main.run_cli([])

    captured = capsys.readouterr()
    assert result == 1
    assert "closing tag [/red]" in captured.out


def test_smoke_test_survives_markup_like_path(monkeypatch, capsys):
    """``log_path`` 含方括号时自检必须照常输出。

    该值来自 ``config.toml``，是用户可配置项；安装目录名同样由用户决定。
    修复前它直接进 ``console.print(f"...")``，会被当作 rich 标记解析——
    ``[/]`` 抛 ``MarkupError``，而自检正是构建脚本用来判断产物好坏的入口。
    """
    from pathlib import Path

    class FakeSettings:
        log_path = "/tmp/[kb]/logs"

    monkeypatch.setattr(main, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(main, "initialize_dependencies", lambda: None)
    monkeypatch.setattr(main, "_assert_splitter_encoding_available", lambda: None)
    monkeypatch.setattr(main, "resolve_app_root", lambda: Path("/tmp/[app]"))

    result = main.run_smoke_test()

    captured = capsys.readouterr()
    assert result == 0
    assert "smoke test ok" in captured.out
    assert "/tmp/[kb]/logs" in captured.out


# ── 回归：自检必须触达每个 provider 模块 ──
#
# 十三个 provider 模块只经 ``importlib.import_module(运行时字符串)`` 加载
# （``src/providers/factory.py``），PyInstaller 的静态分析看不见它们，因此
# ``HIDDEN_IMPORTS`` 是冻结包里能否找到它们的唯一依据。此前自检只走分片器与
# 日志路径，**不实例化也不导入任何 provider**——漏收一个模块时 PyInstaller
# 不报错、``--validate`` 通过，用户在那个渠道上才拿到 ``ModuleNotFoundError``。
# 这与 ``tiktoken_ext.openai_public`` 当初漏收是同一个失效模式。


def test_smoke_test_asserts_every_provider_module_is_importable():
    """自检必须逐个导入注册表里的 provider 模块。"""
    import main
    from src.providers.factory import ModelProviderFactory

    # 正常环境下应当通过
    main._assert_provider_modules_importable()

    # 模块导入失败时必须报成带指引的 RuntimeError，而不是让 ImportError 穿出去
    import importlib

    real_import = importlib.import_module

    def failing_import(name, *args, **kwargs):
        if name == "src.providers.local_hash":
            raise ImportError("No module named 'src.providers.local_hash'")
        return real_import(name, *args, **kwargs)

    original = importlib.import_module
    importlib.import_module = failing_import
    try:
        with pytest.raises(RuntimeError, match="HIDDEN_IMPORTS"):
            main._assert_provider_modules_importable()
    finally:
        importlib.import_module = original

    # 断言这条守卫确实覆盖了注册表里的每个模块，而不是只挑了几个
    assert len(ModelProviderFactory._provider_map) >= 13


# ── 回归：分片器自检必须校验产出非空 ──
#
# ``_assert_splitter_encoding_available`` 是 ``--smoke-test`` 用来确认 tiktoken
# 编码在冻结环境里可用的守卫。原实现只调 ``splitter.split(...)`` 并丢弃返回值：
# 编码缺失导致分片器静默产出空列表时，自检照样报 "smoke test ok"，构建脚本据
# 此放行发布包。删除 ``if not chunks: raise`` 后全量 3790 条仍全绿（变异扫描
# 确认），因此补这条反向用例。


def test_splitter_self_check_rejects_empty_chunks(monkeypatch):
    """分片器产出空列表时自检必须报错，而不是静默通过。"""
    import src.etl.splitters.recursive_text_splitter as splitter_module

    class EmptySplitter:
        def split(self, documents):
            return []

    monkeypatch.setattr(splitter_module, "RecursiveTextSplitter", EmptySplitter)

    with pytest.raises(RuntimeError, match="分片器自检未产出任何分片"):
        main._assert_splitter_encoding_available()


def test_splitter_self_check_passes_in_real_environment():
    """正常环境下真实分片器必须产出分片（守卫的绿色一侧）。"""
    main._assert_splitter_encoding_available()


def test_run_cli_help_exits_before_interactive_menu(monkeypatch, capsys):
    """``-h``/``--help`` 打印用法并瞬时退出，不进交互菜单。

    旧实现除 ``--smoke-test`` 外静默忽略一切参数（实测 ``--help`` 直接进
    交互菜单），新用户只能 Ctrl+C 退出重来。
    """
    called = {"main": 0, "smoke": 0, "init": 0}

    monkeypatch.setattr(main, "main", lambda: called.__setitem__("main", 1))
    monkeypatch.setattr(main, "run_smoke_test", lambda: called.__setitem__("smoke", 1))
    monkeypatch.setattr(main, "initialize_dependencies", lambda: called.__setitem__("init", 1))
    # atexit 注销路径也要可注入：cleanup 模块的 import 在 main.py 顶层已完成
    monkeypatch.setattr(
        "atexit.unregister",
        lambda callback: called.__setitem__("unregister", callback),
    )

    for flag in ("-h", "--help"):
        called.update(main=0, smoke=0, init=0, unregister=None)
        result = main.run_cli([flag])

        assert result == 0
        assert called["main"] == 0  # 未进交互菜单
        assert called["smoke"] == 0  # 未误触发自检
        assert called["init"] == 0  # 未初始化依赖（--help 必须瞬时返回）
        assert called["unregister"] is main.src.utils.cleanup.cleanup_temp_files  # 已注销退出清理
        output = capsys.readouterr().out
        assert "用法" in output
        assert "--smoke-test" in output


def test_run_cli_help_does_not_delete_cache(monkeypatch):
    """``--help`` 退出不得触发退出清理（rmtree 缓存目录）。

    ``import src.utils.cleanup`` 的模块级 ``atexit.register`` 在 ``--help``
    分支之前已生效，不注销的话 ``--help`` 退出会把上一次会话留下的缓存目录
    整个删掉（实测复现）。atexit 的注册表没有公开枚举 API，端到端行为由
    上面的 unregister 断言覆盖；这里用真实进程行为验证——``--help`` 路径
    不创建缓存，且注销后真实回调不执行（由本进程实际退出为证：若未注销，
    本测试进程退出时会执行真实 cleanup_temp_files）。
    """
    # 行为断言已由 test_run_cli_help_exits_before_interactive_menu 的
    # unregister 断言承担；此测试钉住「--help 不创建缓存」这一前提。
    monkeypatch.setattr(main, "main", lambda: None)
    result = main.run_cli(["--help"])
    assert result == 0
