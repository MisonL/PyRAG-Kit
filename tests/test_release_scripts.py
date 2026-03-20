# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

from scripts.build_binary_release import prepare_runtime_layout, validate_bundle
from scripts.extract_release_notes import extract_section


def test_prepare_runtime_layout_creates_placeholder_directories(tmp_path):
    prepare_runtime_layout(tmp_path)

    assert (tmp_path / "data" / "kb").is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
    placeholder = tmp_path / "knowledge_base" / "README.md"
    assert placeholder.is_file()
    assert "Markdown" in placeholder.read_text(encoding="utf-8")


def test_extract_section_returns_version_content():
    section = extract_section("1.3.0")

    assert section.startswith("## [1.3.0]")
    assert "运行与配置" in section


def test_validate_bundle_uses_smoke_test(monkeypatch, tmp_path):
    bundle_root = tmp_path / "bundle"
    app_dir = bundle_root / "PyRAG-Kit"
    app_dir.mkdir(parents=True)
    executable = app_dir / "PyRAG-Kit"
    executable.write_text("", encoding="utf-8")

    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="PyRAG-Kit 1.3.0 smoke test ok\n", stderr="")

    monkeypatch.setattr("scripts.build_binary_release.subprocess.run", fake_run)

    validate_bundle(bundle_root)

    assert recorded["command"][-1] == "--smoke-test"
    assert recorded["kwargs"]["check"] is True
