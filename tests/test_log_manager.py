"""`src/utils/log_manager.py` 的日志目录降级行为。"""

import logging

import pytest

from src.utils import log_manager


@pytest.fixture(autouse=True)
def _reset_degraded_flag(monkeypatch):
    """每个用例都从「尚未降级」开始，避免用例间互相影响。"""
    monkeypatch.setattr(log_manager, "_DEGRADED_WARNING_EMITTED", False)
    yield


def test_degradation_warning_is_emitted_once(monkeypatch, caplog):
    """降级 warning 只打一次，不随模块数量线性增长。

    ``get_module_logger`` 被 20 多个模块在 import 期调用，各自打一条会让
    只读安装下的输出被重复信息淹没。
    """
    monkeypatch.setattr(log_manager, "get_settings", lambda: _FakeSettings())

    with caplog.at_level(logging.WARNING):
        for name in ("mod.a", "mod.b", "mod.c"):
            log_manager.get_module_logger(name)

    degradations = [r for r in caplog.records if "无法写入日志目录" in r.getMessage()]
    assert len(degradations) == 1


def test_module_logger_still_has_console_handler_when_degraded(monkeypatch):
    """降级后仍要有控制台 handler，否则日志彻底丢失。"""
    monkeypatch.setattr(log_manager, "get_settings", lambda: _FakeSettings())

    logger = log_manager.get_module_logger("test.degraded.handlers")

    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


class _FakeSettings:
    """日志目录指向一个不存在且无法创建的位置，触发降级。"""

    log_level = "WARNING"
    log_path = "/proc/self/nonexistent-log-dir"


# ── 回归：日志清理只应动本应用写出的文件 ──
#
# ``log_path`` 是用户可配置项。原实现删除「目录下所有 *.log 且文件名含任意
# YYYY-MM-DD」，把 ``log_path`` 指向共享目录（``/var/log``、挂载的日志盘）时
# 会连带删掉别的程序与用户的日志，且不可逆。


def _cleanup_with(monkeypatch, log_dir, retention_days=15):
    from types import SimpleNamespace

    settings = SimpleNamespace(
        log_path=str(log_dir), log_retention_days=retention_days, log_level="WARNING"
    )
    monkeypatch.setattr(log_manager, "get_settings", lambda: settings)
    log_manager.cleanup_old_logs()


def test_cleanup_old_logs_removes_app_owned_logs(tmp_path, monkeypatch):
    """本应用自己的过期日志照常清理，功能不能被收紧挡住。"""
    (tmp_path / "app_log_2020-01-01.log").write_text("a", encoding="utf-8")
    (tmp_path / "chat_log_2020-01-01.log").write_text("c", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert not (tmp_path / "app_log_2020-01-01.log").exists()
    assert not (tmp_path / "chat_log_2020-01-01.log").exists()


def test_cleanup_old_logs_keeps_recent_app_logs(tmp_path, monkeypatch):
    """未过期的本应用日志必须保留。"""
    from datetime import datetime

    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    recent = tmp_path / f"app_log_{today}.log"
    recent.write_text("今天的日志", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert recent.exists()


@pytest.mark.parametrize(
    "foreign",
    [
        "nginx_access_2020-01-01.log",
        "postgres_2020-01-01.log",
        "backup_2020-01-01.log",
        "app_logger_2020-01-01.log",
        "my_app_log_2020-01-01.log",
        "app_log_2020-01-01.log.bak",
    ],
)
def test_cleanup_old_logs_leaves_other_programs_logs_alone(tmp_path, monkeypatch, foreign):
    """别的程序或用户的文件不得被删除。

    ``log_path`` 可以指向共享目录；删掉 ``nginx_access_2020-01-01.log`` 这种
    与本应用无关的文件是不可逆的越界行为。判定按完整文件名形态而非「含日期
    的子串」，所以 ``my_app_log_...`` 这类前缀也不匹配。
    """
    victim = tmp_path / foreign
    victim.write_text("别人的数据", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert victim.exists(), f"{foreign} 被误删"


def test_cleanup_old_logs_ignores_non_log_files(tmp_path, monkeypatch):
    """非 ``.log`` 文件从不参与清理。"""
    keep = tmp_path / "notes.txt"
    keep.write_text("x", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert keep.exists()


# ── 回归：``RotatingFileHandler`` 的轮转备份也必须被清理 ──
#
# 写入侧是 ``RotatingFileHandler(maxBytes=1MB, backupCount=5)``，单个文件超过
# 1 MB 就轮转出 ``.log.1`` ~ ``.log.5``。旧模式以 ``$`` 结尾，那些备份**一个都
# 不匹配**，于是「清理旧日志」只删掉主文件，把绝大部分体积留在盘上——实测造
# 一批 30 天前的日志（主文件 + 5 个备份），清理后仍剩 83% 的字节。而
# ``log_path`` 是用户可配置项，指向磁盘时这份残留会一直累积。


def test_cleanup_old_logs_removes_rotation_backups(tmp_path, monkeypatch):
    """轮转备份属于本应用的日志，过期后必须一并清理。"""
    for suffix in (".log.1", ".log.2", ".log.5"):
        (tmp_path / f"app_log_2020-01-01{suffix}").write_text("x", encoding="utf-8")
    (tmp_path / "chat_log_2020-01-01.log.3").write_text("x", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert not list(tmp_path.glob("app_log_2020-01-01.log.*")), "app 轮转备份未被清理"
    assert not list(tmp_path.glob("chat_log_2020-01-01.log.*")), "chat 轮转备份未被清理"


def test_cleanup_old_logs_keeps_recent_rotation_backups(tmp_path, monkeypatch):
    """收紧不能误伤：未过期的轮转备份必须保留。"""
    from datetime import datetime

    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    recent = tmp_path / f"app_log_{today}.log.1"
    recent.write_text("今天的备份", encoding="utf-8")

    _cleanup_with(monkeypatch, tmp_path)

    assert recent.exists()


def test_cleanup_old_logs_actually_rotates_through_a_real_handler(tmp_path, monkeypatch):
    """走真实 ``RotatingFileHandler`` 产生备份，再断言清理覆盖它们。

    直接写 ``.log.1`` 这样的文件名等于假设了备份名的形态；这里让
    ``RotatingFileHandler`` 自己轮转，断言的是**真实产生出来的**文件名能被清理
    规则匹配。handler 的备份名由 ``doRollover`` 的 ``baseFilename + "." + 序号``
    决定，若将来改变形态，这条用例会失败。
    """
    import logging.handlers

    handler = logging.handlers.RotatingFileHandler(
        str(tmp_path / "app_log_2020-01-01.log"), maxBytes=200, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.rotation.real")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        for _ in range(80):
            logger.info("x" * 60)
    finally:
        handler.close()

    produced = sorted(p.name for p in tmp_path.iterdir() if ".log." in p.name)
    assert produced, "RotatingFileHandler 未产生备份，用例前提不成立"

    _cleanup_with(monkeypatch, tmp_path)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [], f"真实轮转产生的文件未被清理干净: {remaining}"
