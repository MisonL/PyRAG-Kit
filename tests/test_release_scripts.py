# -*- coding: utf-8 -*-
from pathlib import Path

from scripts.build_binary_release import prepare_runtime_layout
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
