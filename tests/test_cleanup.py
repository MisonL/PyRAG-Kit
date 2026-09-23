"""``cleanup_temp_files`` 是 atexit 钩子，删除前必须挡住危险路径。

``cache_path`` 是用户可配置项，而 ``shutil.rmtree`` 的后果不可逆：实测
``Settings(cache_path="/")``、``"/etc"``、``"~"``、``"../.."`` 全部被 pydantic
接受，随后退出钩子会把整个目录删掉。这里守的是「删错目录」这一类无法回滚的失败。
"""

from pathlib import Path
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
