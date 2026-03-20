# -*- coding: utf-8 -*-
import main


def test_run_cli_uses_smoke_test(monkeypatch):
    called = {"count": 0}

    def fake_run_smoke_test():
        called["count"] += 1
        return 0

    monkeypatch.setattr(main, "run_smoke_test", fake_run_smoke_test)

    result = main.run_cli(["--smoke-test"])

    assert result == 0
    assert called["count"] == 1
