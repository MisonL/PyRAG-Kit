"""发布工作流的 shell 逻辑回归测试。

工作流里的校验步骤跑在 CI 上，本地看不见；一旦写错（例如 glob 匹配不到任何
文件），循环体不执行、步骤照样退出 0，坏包会静默流到用户手里。这里把步骤的
``run`` 脚本从 YAML 里取出来，用合成归档在本地真跑一遍。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

# 这些用例把工作流里的 shell 脚本真跑一遍。GitHub 的 Windows runner 上
# ``bash``/``tar``/``find`` 由 Git for Windows 提供，行为与 POSIX 不完全一致，
# 而本机无法验证——因此非 POSIX 平台直接跳过，宁可不覆盖也不要让一次发布
# 因为测试环境的差异而失败。
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="工作流脚本的本地复跑依赖 POSIX shell 与 coreutils，Windows 上无法验证",
)

_HASH_TOOL = shutil.which("sha256sum") or shutil.which("shasum")


def _extract_run_step(job: str, step_name: str) -> str:
    """从工作流 YAML 中取出某个步骤的 ``run: |`` 脚本文本。

    刻意不引入 YAML 解析依赖：只用缩进定位，取不到就报错——解析失败必须是
    响亮的失败，否则测试会以为自己在校验一段它根本没读到的脚本。
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    job_header = f"  {job}:"
    try:
        start = lines.index(job_header)
    except ValueError:  # pragma: no cover - 工作流结构被改动时的响亮失败
        raise AssertionError(f"未找到 job {job!r}（是否重命名了？）") from None

    step_header = f"      - name: {step_name}"
    offset = next(
        (i for i, line in enumerate(lines[start:], start) if line == step_header),
        None,
    )
    assert offset is not None, f"job {job!r} 中未找到步骤 {step_name!r}"

    run_at = next(
        (i for i, line in enumerate(lines[offset:], offset) if line.strip() == "run: |"),
        None,
    )
    assert run_at is not None, f"步骤 {step_name!r} 没有块状 run 脚本"

    indent = len(lines[run_at]) - len(lines[run_at].lstrip())
    body_indent = indent + 2
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) < body_indent:
            break
        body.append(line[body_indent:] if len(line) > body_indent else "")
    assert body, f"步骤 {step_name!r} 的 run 脚本为空"
    return "\n".join(body) + "\n"


@pytest.fixture(scope="module")
def verify_script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("wf") / "verify.sh"
    path.write_text(_extract_run_step("publish", "Verify archive contents"), encoding="utf-8")
    return path


@pytest.fixture
def bash_env(tmp_path: Path) -> dict[str, str]:
    """提供 ``python`` 可执行文件——CI 由 setup-python 保证，本地不一定有。"""
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run_verify(script: Path, case_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """在 ``case_dir`` 下运行校验脚本（与 CI 一样，cwd 即工作区根）。"""
    return subprocess.run(
        ["bash", str(script)],
        cwd=case_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _make_tar(case_dir: Path, name: str, *, with_env: bool, with_cache: bool) -> Path:
    """按 download-artifact 的嵌套布局造一个归档：release_artifacts/<target>/。"""
    staging = case_dir / "staging" / name
    (staging / "PyRAG-Kit" / "tiktoken_cache").mkdir(parents=True)
    (staging / "PyRAG-Kit" / "tiktoken_cache" / "blob").write_text("bpe", encoding="utf-8")
    (staging / "PyRAG-Kit" / "VERSION.txt").write_text("1.4.0\n", encoding="utf-8")
    if with_env:
        (staging / ".env").write_text("SECRET=x\n", encoding="utf-8")

    target_dir = case_dir / "release_artifacts" / "macos-x64"
    target_dir.mkdir(parents=True)
    archive = target_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        if with_cache:
            tar.add(staging, arcname=name)
        else:
            # 只收 VERSION.txt：模拟构建脚本漏掉 BPE 预置的产物。
            tar.add(staging / "PyRAG-Kit" / "VERSION.txt", arcname=f"{name}/PyRAG-Kit/VERSION.txt")
    return archive


def _make_zip(case_dir: Path, name: str, *, with_env: bool) -> Path:
    target_dir = case_dir / "release_artifacts" / "windows-x64"
    target_dir.mkdir(parents=True)
    archive = target_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{name}/PyRAG-Kit/tiktoken_cache/blob", "bpe")
        if with_env:
            zf.writestr(f"{name}/.env", "SECRET=x\n")
    return archive


def test_verify_step_accepts_a_clean_archive(tmp_path, verify_script, bash_env):
    _make_tar(tmp_path, "PyRAG-Kit-1.4.0-macos-x64", with_env=False, with_cache=True)

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 0, result.stderr
    assert "归档校验通过" in result.stdout


def test_verify_step_rejects_archives_that_contain_credentials(tmp_path, verify_script, bash_env):
    _make_tar(tmp_path, "PyRAG-Kit-1.4.0-macos-x64", with_env=True, with_cache=True)

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 1
    assert "归档内含凭证文件" in result.stderr
    assert ".env" in result.stderr


def test_verify_step_rejects_archives_without_the_offline_cache(tmp_path, verify_script, bash_env):
    _make_tar(tmp_path, "PyRAG-Kit-1.4.0-macos-x64", with_env=False, with_cache=False)

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 1
    assert "缺少 tiktoken_cache" in result.stderr


def test_verify_step_detects_credentials_in_a_large_archive(tmp_path, verify_script, bash_env):
    """归档列表超过管道缓冲区时，凭证检查必须仍然命中。

    原实现写的是 `printf ... | grep -q ...`：`grep -q` 命中即退出，上游
    `printf` 收到 SIGPIPE 后以 141 退出，`set -o pipefail` 让整条管道返回 141，
    而 `if` 只把 0 当成功——于是**命中被判成未命中**，含 `.env` 的包被放行发布。

    触发条件是列表超过管道缓冲区（Linux 上 64 KiB）。真实 v1.4.0 产物的列表
    已经是 75–79 KB，因此这条检查自发布起就一直失效。这里的归档刻意造出足够
    多的条目：小归档（其它用例）永远测不到这条路径。
    """
    staging = tmp_path / "staging" / "big"
    cache_dir = staging / "PyRAG-Kit" / "tiktoken_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "blob").write_text("bpe", encoding="utf-8")
    (staging / "PyRAG-Kit" / "VERSION.txt").write_text("1.4.0\n", encoding="utf-8")
    (staging / ".env").write_text("SECRET=x\n", encoding="utf-8")
    # 让 namelist 超过 64 KiB：每条目约 40 字节，6000 条约 240 KB。
    filler = staging / "PyRAG-Kit" / "filler"
    filler.mkdir()
    for index in range(6000):
        (filler / f"entry_{index:05d}.txt").write_text("x", encoding="utf-8")

    target_dir = tmp_path / "release_artifacts" / "macos-x64"
    target_dir.mkdir(parents=True)
    archive = target_dir / "PyRAG-Kit-1.4.0-macos-x64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname="big")

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "归档内含凭证文件" in result.stderr


def test_verify_step_accepts_a_large_archive_that_has_the_cache(tmp_path, verify_script, bash_env):
    """同一形态的干净大归档必须通过——否则「凭证检查」可能是靠误报变红的。

    这条是上一条的反向守卫：去掉 `-q` 之后，`tiktoken_cache` 检查若仍用管道，
    会因 SIGPIPE 误报缺失，把正常包拦下。两条一起才能钉住「读完整输入」。
    """
    staging = tmp_path / "staging" / "big"
    cache_dir = staging / "PyRAG-Kit" / "tiktoken_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "blob").write_text("bpe", encoding="utf-8")
    (staging / "PyRAG-Kit" / "VERSION.txt").write_text("1.4.0\n", encoding="utf-8")
    filler = staging / "PyRAG-Kit" / "filler"
    filler.mkdir()
    for index in range(6000):
        (filler / f"entry_{index:05d}.txt").write_text("x", encoding="utf-8")

    target_dir = tmp_path / "release_artifacts" / "macos-x64"
    target_dir.mkdir(parents=True)
    archive = target_dir / "PyRAG-Kit-1.4.0-macos-x64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname="big")

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "归档校验通过" in result.stdout


def test_verify_step_fails_when_no_archive_was_downloaded(tmp_path, verify_script, bash_env):
    """下载步骤失败时不能静默通过——空集必须报错，否则整步等于没跑。"""
    (tmp_path / "release_artifacts").mkdir()

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 1
    assert "未找到任何归档" in result.stderr


def test_verify_step_reads_the_nested_download_layout(tmp_path, verify_script, bash_env):
    """归档在 ``release_artifacts/<target>/`` 下（download-artifact 的布局）。

    用 ``release_artifacts/*.tar.gz`` 这样的单层 glob 会一个都匹配不到，
    循环体不执行、fail 保持 0，整步静默通过——这正是本用例要钉住的行为。
    """
    archive = _make_tar(tmp_path, "PyRAG-Kit-1.4.0-macos-x64", with_env=False, with_cache=True)
    assert archive.parent.parent.name == "release_artifacts"
    assert not list((tmp_path / "release_artifacts").glob("*.tar.gz")), (
        "单层 glob 本就匹配不到，正是旧实现静默通过的原因"
    )

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 0, result.stderr
    assert "归档校验通过（1 个）" in result.stdout


def test_verify_step_checks_zip_archives_too(tmp_path, verify_script, bash_env):
    _make_zip(tmp_path, "PyRAG-Kit-1.4.0-windows-x64", with_env=True)

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 1
    assert "归档内含凭证文件" in result.stderr


def test_verify_step_accepts_a_clean_zip(tmp_path, verify_script, bash_env):
    _make_zip(tmp_path, "PyRAG-Kit-1.4.0-windows-x64", with_env=False)

    result = _run_verify(verify_script, tmp_path, bash_env)

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(_HASH_TOOL is None, reason="本机既无 sha256sum 也无 shasum")
def test_checksums_land_where_publish_looks_and_use_basenames(tmp_path, bash_env):
    """校验和必须写进 ``release_artifacts/``，且条目只带文件名。

    两点都会让用户白拿一份对不上号的 SHA256SUMS：写到工作区根目录时
    publish 步骤按前缀查找、根本不会上传它；条目带 ``<target>/`` 前缀时，
    用户把单个归档下载下来后 ``shasum -c`` 找不到文件。
    """
    archive = _make_tar(tmp_path, "PyRAG-Kit-1.4.0-macos-x64", with_env=False, with_cache=True)
    script = tmp_path / "sums.sh"
    script.write_text(_extract_run_step("publish", "Generate checksums"), encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=bash_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    sums = tmp_path / "release_artifacts" / "SHA256SUMS.txt"
    assert sums.is_file(), "校验和必须落在 publish 步骤查找的目录内"

    lines = [line for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    digest, _, filename = lines[0].partition("  ")
    assert filename == archive.name
    assert "/" not in filename

    # 摘要必须与归档实际内容一致——文件名对了但摘要错了同样是坏产物。
    import hashlib

    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()

    # 用户视角：把归档与校验和放在同一目录，`shasum -c` 必须通过。
    user_dir = tmp_path / "download"
    user_dir.mkdir()
    shutil.copy(archive, user_dir / archive.name)
    shutil.copy(sums, user_dir / "SHA256SUMS.txt")
    check = subprocess.run(
        [_HASH_TOOL, "-c", "SHA256SUMS.txt"],
        cwd=user_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "OK" in check.stdout
