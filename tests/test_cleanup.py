"""``cleanup_temp_files`` 是 atexit 钩子，删除前必须挡住危险路径。

``cache_path`` 是用户可配置项，而 ``shutil.rmtree`` 的后果不可逆：实测
``Settings(cache_path="/")``、``"/etc"``、``"~"``、``"../.."`` 全部被 pydantic
接受，随后退出钩子会把整个目录删掉。这里守的是「删错目录」这一类无法回滚的失败。
"""

from pathlib import Path
from sys import platform
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.utils.cleanup import _reject_unsafe_cache_dir, cleanup_temp_files


@pytest.mark.parametrize(
    "dangerous",
    ["/", "/etc", "/usr", "/var", "/System", "/Library", "/bin", "/sbin"],
)
def test_reject_unsafe_cache_dir_rejects_system_paths(dangerous):
    """文件系统根目录与常见系统目录一律拒绝。"""
    assert _reject_unsafe_cache_dir(dangerous) is not None


def test_reject_unsafe_cache_dir_rejects_home_directory():
    """家目录本身拒绝（但不连带拒绝家目录下的子目录）。"""
    assert _reject_unsafe_cache_dir(str(Path.home())) is not None


def test_reject_unsafe_cache_dir_rejects_empty_value():
    assert _reject_unsafe_cache_dir("") is not None


def test_reject_unsafe_cache_dir_allows_ordinary_cache_dir(tmp_path):
    """收紧不能误伤：非默认名的普通缓存目录仍可清理。

    用户可能有意把缓存放到别处（``config.toml`` 里写
    ``cache_path = "data/cache"``），判定不能只看目录名是不是 ``.cache``。
    """
    ordinary = tmp_path / "data" / "cache"
    ordinary.mkdir(parents=True)
    assert _reject_unsafe_cache_dir(str(ordinary)) is None


def test_cleanup_temp_files_removes_ordinary_cache_dir(tmp_path):
    """正常路径必须照常清理，功能不能被安全判断挡住。"""
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "entry.txt").write_text("c", encoding="utf-8")

    with patch("src.utils.cleanup.get_settings", return_value=_settings_with(str(cache))):
        cleanup_temp_files()

    assert not cache.exists()


def test_cleanup_temp_files_skips_missing_dir(tmp_path):
    """目录不存在时是正常路径，不应报错。"""
    missing = tmp_path / "absent"

    with patch("src.utils.cleanup.get_settings", return_value=_settings_with(str(missing))):
        cleanup_temp_files()

    assert not missing.exists()


def test_cleanup_temp_files_skips_non_directory(tmp_path):
    """cache_path 指向文件时跳过，不能把它当目录删。"""
    target = tmp_path / "not_a_dir.txt"
    target.write_text("data", encoding="utf-8")

    with patch("src.utils.cleanup.get_settings", return_value=_settings_with(str(target))):
        cleanup_temp_files()

    assert target.read_text(encoding="utf-8") == "data"


def _settings_with(cache_path: str):
    """构造只带 cache_path 的最小 settings 替身。"""
    from types import SimpleNamespace

    return SimpleNamespace(cache_path=cache_path)


def test_cleanup_temp_files_handles_oserror_gracefully(tmp_path):
    """删除失败只记日志、不抛异常：这是退出路径，抛错会打断解释器收尾。"""
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "x.txt").write_text("x", encoding="utf-8")

    with (
        patch("src.utils.cleanup.get_settings", return_value=_settings_with(str(cache))),
        patch("src.utils.cleanup.shutil.rmtree", side_effect=OSError("权限不足")),
    ):
        cleanup_temp_files()

    assert cache.exists()


def _patch_app_paths(monkeypatch, app_root: Path, **extra_paths: str) -> None:
    """把 cleanup 模块看到的「应用自身路径」指向 ``app_root``。

    ``cleanup`` 通过模块级名字使用 ``ROOT_DIR`` / ``CONFIG_TOML_PATH`` /
    ``ENV_FILE_PATH``，因此必须 patch 到该模块的命名空间上，而不是
    ``src.utils.config``——后者改了也不影响已经绑定的名字。
    """
    monkeypatch.setattr("src.utils.cleanup.ROOT_DIR", app_root)
    monkeypatch.setattr("src.utils.cleanup.CONFIG_TOML_PATH", app_root / "config.toml")
    monkeypatch.setattr("src.utils.cleanup.ENV_FILE_PATH", app_root / ".env")
    monkeypatch.setattr(
        "src.utils.cleanup.get_settings",
        lambda: SimpleNamespace(
            cache_path=str(app_root / ".cache"),
            knowledge_base_path=str(app_root / "knowledge_base"),
            snapshot_root=str(app_root / "data" / "kb"),
            log_path=str(app_root / "data" / "logs"),
            **extra_paths,
        ),
    )


# ── 回归：cache_path 指向应用自身目录时会删掉安装目录 ──
#
# 原有的保护只挡 ``/``、家目录与一份硬编码的 POSIX 系统目录清单，拦不住
# ``cache_path = "."``：它解析为应用根目录，退出钩子会把整个安装目录删掉。
# 实测冻结产物中 ``.env``（内含 API Key）、``knowledge_base/``（用户放进去的
# 源文档）与 ``config.toml`` 一并消失，且无法恢复。


@pytest.mark.parametrize(
    "dangerous",
    [".", "./", "data/..", ".."],
)
def test_reject_unsafe_cache_dir_rejects_app_root(dangerous, tmp_path, monkeypatch):
    """解析后落在应用根目录或其上层的一律拒绝。"""
    app_root = tmp_path / "PyRAG-Kit"
    app_root.mkdir()
    _patch_app_paths(monkeypatch, app_root)

    resolved = str((app_root / dangerous).resolve())

    assert _reject_unsafe_cache_dir(resolved) is not None


@pytest.mark.parametrize(
    "dangerous",
    ["data", "knowledge_base", "data/logs", "data/kb"],
)
def test_reject_unsafe_cache_dir_rejects_app_runtime_dirs(dangerous, tmp_path, monkeypatch):
    """覆盖应用运行时数据目录的 cache_path 同样拒绝。

    ``cache_path = "data"`` 会连带删掉 ``data/kb``（已建好的知识快照）与
    ``data/logs``；``"knowledge_base"`` 会删掉用户的源文档。
    """
    app_root = tmp_path / "PyRAG-Kit"
    (app_root / "data" / "kb").mkdir(parents=True)
    (app_root / "knowledge_base").mkdir()
    _patch_app_paths(monkeypatch, app_root)

    resolved = str((app_root / dangerous).resolve())

    assert _reject_unsafe_cache_dir(resolved) is not None


def test_cleanup_temp_files_does_not_delete_app_root(tmp_path, monkeypatch):
    """端到端：即使 cache_path 指向应用根目录，退出钩子也必须拒绝删除。

    这是本组用例的落点——``_reject_unsafe_cache_dir`` 返回原因后，
    ``cleanup_temp_files`` 必须据此提前返回，而不是继续 ``rmtree``。
    """
    app_root = tmp_path / "PyRAG-Kit"
    (app_root / "knowledge_base").mkdir(parents=True)
    (app_root / "knowledge_base" / "doc.md").write_text("用户文档", encoding="utf-8")
    (app_root / ".env").write_text("SECRET=x\n", encoding="utf-8")
    (app_root / "config.toml").write_text("cache_path = '.'\n", encoding="utf-8")

    monkeypatch.setattr("src.utils.cleanup.ROOT_DIR", app_root)
    monkeypatch.setattr("src.utils.cleanup.CONFIG_TOML_PATH", app_root / "config.toml")
    monkeypatch.setattr("src.utils.cleanup.ENV_FILE_PATH", app_root / ".env")
    monkeypatch.setattr(
        "src.utils.cleanup.get_settings", lambda: SimpleNamespace(cache_path=str(app_root))
    )

    cleanup_temp_files()

    assert app_root.exists(), "应用根目录被退出钩子删掉了"
    assert (app_root / ".env").is_file()
    assert (app_root / "knowledge_base" / "doc.md").is_file()


def test_reject_unsafe_cache_dir_still_allows_ordinary_cache_dirs(tmp_path, monkeypatch):
    """收紧不能误伤：应用根目录之下的独立缓存目录仍可清理。"""
    app_root = tmp_path / "PyRAG-Kit"
    (app_root / "data" / "kb").mkdir(parents=True)
    _patch_app_paths(monkeypatch, app_root)

    for allowed in [".cache", "data/cache", "tmp/cache"]:
        resolved = str((app_root / allowed).resolve())
        assert _reject_unsafe_cache_dir(resolved) is None, f"{allowed} 被误拒"


def test_reject_unsafe_cache_dir_accepts_explicit_protected_paths(tmp_path):
    """``protected_paths`` 可显式传入，便于调用方与测试脱离全局配置。"""
    app_root = tmp_path / "app"
    app_root.mkdir()

    assert _reject_unsafe_cache_dir(str(app_root), protected_paths=[app_root]) is not None
    assert _reject_unsafe_cache_dir(str(tmp_path / "elsewhere"), protected_paths=[app_root]) is None


# ── 回归：``cache_path`` 是符号链接环时不得让 atexit 钩子整体失效 ──
#
# ``_reject_unsafe_cache_dir`` 首行就 ``Path(cache_dir).resolve()``，而
# ``resolve()`` 对链接环抛 ``RuntimeError: Symlink loop from ...``。该行此前没有
# 守卫（遍历 ``protected_paths`` 的那轮有），于是本函数整体抛异常——而它在
# atexit 里被调用，用户看到 ``Exception ignored in atexit callback`` 加
# traceback，**清理完全没做**。


def test_symlink_loop_cache_dir_is_rejected_without_raising(tmp_path):
    """链接环不是可安全删除的目标，应返回原因而不是抛。"""
    link_a = tmp_path / "a"
    link_b = tmp_path / "b"
    link_a.symlink_to(link_b)
    link_b.symlink_to(link_a)

    reason = _reject_unsafe_cache_dir(str(link_a))

    assert reason is not None
    assert "无法解析" in reason


def test_symlink_loop_does_not_break_cleanup_temp_files(tmp_path, monkeypatch):
    """钩子本身必须跑完，不得抛异常。"""
    link_a = tmp_path / "a"
    link_b = tmp_path / "b"
    link_a.symlink_to(link_b)
    link_b.symlink_to(link_a)

    class FakeSettings:
        cache_path = str(link_a)

    monkeypatch.setattr("src.utils.cleanup.get_settings", lambda: FakeSettings())

    cleanup_temp_files()  # 不抛即通过


# ── 回归：系统级目录清单必须完整 ──
#
# 清单原先只有 ``/etc``／``/usr``／``/bin``／``/sbin``／``/var`` 与三个 macOS
# 目录，漏掉了同样属于系统级、且 ``rmtree`` 后果不可逆的一批。实测以下路径
# **全部被放行**，退出钩子会把它们整个删掉：
#
#   * ``/tmp``（macOS 上解析成 ``/private/tmp``）——整个系统临时目录；
#   * ``/Users``——macOS 上**全部用户家目录的父目录**（``/home`` 在 Linux 上
#     同理），删除等于抹掉机器上每个账户的家目录；
#   * ``/opt``——本模块 docstring 自己就把「安装在 ``/opt``」列为场景；
#   * ``/dev``／``/proc``／``/sys`` 等。
#
# 判据是**相等**，因此只挡挂载点与系统根本身，卷内／目录内的正常缓存位置不受
# 影响（见下方「不误伤」用例）。


@pytest.mark.parametrize(
    "dangerous",
    [
        "/tmp",
        "/Users",
        "/home",
        "/opt",
        "/dev",
        "/proc",
        "/sys",
        "/boot",
        "/run",
        "/srv",
        "/mnt",
        "/media",
        "/root",
        "/Volumes",
        "/private",
    ],
)
def test_reject_unsafe_cache_dir_rejects_system_level_roots(dangerous):
    """系统级目录（含用户家目录的父目录）一律拒绝。"""
    assert _reject_unsafe_cache_dir(dangerous) is not None


# ``/private/tmp`` 只在 macOS 上是 ``/tmp`` 的符号链接目标（守卫经由 ``/tmp``
# 的解析结果命中它）。Linux 上 ``/private`` 不是系统目录、该路径不存在，守卫
# 按相等判据放行才是正确行为（``cleanup_temp_files`` 在 rmtree 前有 exists
# 检查，放行无害）——quality workflow 只跑 ubuntu，此断言不能进公共参数表。
@pytest.mark.skipif(
    platform != "darwin",
    reason="/private/tmp 是 macOS 上 /tmp 的解析形态，Linux 上不是系统目录",
)
def test_reject_unsafe_cache_dir_rejects_macos_tmp_resolution_target():
    assert _reject_unsafe_cache_dir("/private/tmp") is not None


@pytest.mark.parametrize(
    "allowed",
    [
        "/tmp/myapp/cache",
        "/opt/myapp/cache",
        "/Volumes/MyDrive/cache",
        "/Users/someone/Library/Caches/app",
    ],
)
def test_reject_unsafe_cache_dir_still_allows_paths_inside_system_roots(allowed):
    """收紧不能误伤：系统根**内部**的缓存目录照常可用。

    判据是「解析后相等」，因此 ``/tmp/myapp/cache`` 这类显式指定的缓存位置
    必须继续放行——本函数从不过度拦截用户有意配置的缓存目录。
    """
    assert _reject_unsafe_cache_dir(allowed) is None
