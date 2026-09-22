import subprocess

import pytest

from scripts.build_binary_release import (
    PACKAGE_FILES,
    prepare_runtime_layout,
    stage_bundle,
    validate_bundle,
    validate_target_environment,
)
from scripts.extract_release_notes import extract_section


def test_prepare_runtime_layout_creates_placeholder_directories(tmp_path):
    prepare_runtime_layout(tmp_path)

    assert (tmp_path / "data" / "kb").is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
    placeholder = tmp_path / "knowledge_base" / "README.md"
    assert placeholder.is_file()
    assert "Markdown" in placeholder.read_text(encoding="utf-8")


def test_extract_section_returns_version_content():
    section = extract_section("1.4.0")

    assert section.startswith("## [1.4.0]")
    assert "Provider 与协议" in section


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
        return subprocess.CompletedProcess(
            command, 0, stdout="PyRAG-Kit 1.4.0 smoke test ok\n", stderr=""
        )

    monkeypatch.setattr("scripts.build_binary_release.subprocess.run", fake_run)

    validate_bundle(bundle_root)

    assert recorded["command"][-1] == "--smoke-test"
    assert recorded["kwargs"]["check"] is True


@pytest.mark.parametrize(
    ("target", "system", "machine"),
    [
        ("macos-x64", "Darwin", "x86_64"),
        ("macos-arm64", "Darwin", "arm64"),
        ("linux-x64", "Linux", "amd64"),
        ("linux-arm64", "Linux", "aarch64"),
        ("windows-x64", "Windows", "AMD64"),
    ],
)
def test_validate_target_environment_accepts_matching_host(target, system, machine):
    validate_target_environment(target, system=system, machine=machine)


def test_validate_target_environment_rejects_cross_architecture():
    with pytest.raises(RuntimeError, match="不执行跨平台交叉编译"):
        validate_target_environment("macos-arm64", system="Darwin", machine="x86_64")


def test_stage_bundle_copies_package_files_including_subdirectories(monkeypatch, tmp_path):
    """PACKAGE_FILES 含子目录路径（licenses/APACHE-2.0.txt）。

    ``shutil.copy2`` 不会创建父目录，缺了 mkdir 会在打包时抛
    FileNotFoundError；而这一步只在发布时才跑到，本地测试很容易漏掉。
    """
    dist_root = tmp_path / "dist" / "PyRAG-Kit"
    dist_root.mkdir(parents=True)
    (dist_root / "PyRAG-Kit").write_bytes(b"binary")
    artifact_root = tmp_path / "release_artifacts"

    monkeypatch.setattr("scripts.build_binary_release.DIST_ROOT", tmp_path / "dist")
    monkeypatch.setattr("scripts.build_binary_release.ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr("scripts.build_binary_release.PROJECT_ROOT", tmp_path)
    for relative in PACKAGE_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("content", encoding="utf-8")

    bundle_root = stage_bundle("linux-x64", "1.4.0")

    assert bundle_root.name == "PyRAG-Kit-1.4.0-linux-x64"
    for relative in PACKAGE_FILES:
        assert (bundle_root / relative).is_file(), relative
    # 子目录路径必须真的落在子目录里，而不是被拍平。
    assert (bundle_root / "licenses" / "APACHE-2.0.txt").parent.name == "licenses"
