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


# ── 回归：ZIP 分支对悬空符号链接的确定行为 ──


def test_archive_bundle_zip_rejects_dangling_symlink(tmp_path, monkeypatch):
    """ZIP 分支遇到悬空符号链接必须给出可归因的 RuntimeError。

    实测修复前 ``ZipFile.write`` 抛的是裸 ``FileNotFoundError``，消息里只有
    一个路径，看不出「这是发布包里的符号链接」，而 tar.gz 分支对同一目录完全
    正常——两个分支行为不一致，且在 Windows（唯一的 zip 目标）上表现为一个难
    以归因的构建失败。
    """
    from scripts.build_binary_release import archive_bundle

    monkeypatch.setattr("scripts.build_binary_release.ARTIFACT_ROOT", tmp_path)
    bundle_root = tmp_path / "PyRAG-Kit-1.4.0-windows-x64"
    bundle_root.mkdir()
    (bundle_root / "real.txt").write_text("hello", encoding="utf-8")
    (bundle_root / "dangling").symlink_to("nowhere.txt")

    with pytest.raises(RuntimeError, match="符号链接") as excinfo:
        archive_bundle(bundle_root, "windows-x64")

    assert "dangling" in str(excinfo.value)
    assert not (tmp_path / f"{bundle_root.name}.zip").exists()


def test_archive_bundle_zip_succeeds_without_symlinks(tmp_path, monkeypatch):
    """没有悬空链接时 ZIP 必须正常生成，且能被读回。"""
    import zipfile

    from scripts.build_binary_release import archive_bundle

    monkeypatch.setattr("scripts.build_binary_release.ARTIFACT_ROOT", tmp_path)
    bundle_root = tmp_path / "PyRAG-Kit-1.4.0-windows-x64"
    bundle_root.mkdir()
    (bundle_root / "real.txt").write_text("hello", encoding="utf-8")
    (bundle_root / "sub").mkdir()
    (bundle_root / "sub" / "nested.txt").write_text("nested", encoding="utf-8")

    archive_path = archive_bundle(bundle_root, "windows-x64")

    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert f"{bundle_root.name}/real.txt" in names
    assert f"{bundle_root.name}/sub/nested.txt" in names


def test_archive_bundle_targz_keeps_dangling_symlink(tmp_path, monkeypatch):
    """tar.gz 分支必须保持既有行为：悬空链接照原样写入归档，不报错。"""
    import tarfile

    from scripts.build_binary_release import archive_bundle

    monkeypatch.setattr("scripts.build_binary_release.ARTIFACT_ROOT", tmp_path)
    bundle_root = tmp_path / "PyRAG-Kit-1.4.0-linux-x64"
    bundle_root.mkdir()
    (bundle_root / "real.txt").write_text("hello", encoding="utf-8")
    (bundle_root / "dangling").symlink_to("nowhere.txt")

    archive_path = archive_bundle(bundle_root, "linux-x64")

    assert archive_path.exists()
    with tarfile.open(archive_path) as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert f"{bundle_root.name}/dangling" in members
    assert members[f"{bundle_root.name}/dangling"].issym()


def test_find_dangling_symlinks_ignores_valid_links_and_files(tmp_path):
    """只有「是符号链接且目标不存在」才算悬空。"""
    from scripts.build_binary_release import find_dangling_symlinks

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "real.txt").write_text("hello", encoding="utf-8")
    (bundle_root / "valid").symlink_to("real.txt")
    (bundle_root / "dangling").symlink_to("nowhere.txt")

    assert [path.name for path in find_dangling_symlinks(bundle_root)] == ["dangling"]
