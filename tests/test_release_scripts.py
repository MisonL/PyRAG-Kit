import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_binary_release import (
    APP_NAME,
    APP_ROOT_FILES,
    PACKAGE_FILES,
    load_version,
    prepare_runtime_layout,
    stage_bundle,
    validate_bundle,
    validate_target_environment,
)
from scripts.extract_release_notes import extract_section


@pytest.fixture
def bpe_source(monkeypatch, tmp_path):
    """给 BPE 预置一个自给的源文件，使测试不依赖宿主机缓存也不联网。

    ``stage_tiktoken_cache`` 的源路径由 ``TIKTOKEN_CACHE_DIR`` 决定，缺失时
    会联网下载——两者都会让单元测试依赖外部状态（宿主机恰好有缓存时绿、
    CI 上无缓存时红）。这里预置一份内容可辨的夹具文件并指向它。
    """
    cache_dir = tmp_path / "_bpe_source"
    cache_dir.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "scripts.build_binary_release._download_tiktoken_blob",
        lambda: pytest.fail("本测试不应联网下载 BPE"),
    )
    return cache_dir


def test_prepare_runtime_layout_creates_placeholder_directories(tmp_path):
    """运行时目录必须建在冻结应用自己的根目录下。

    冻结时 ``resolve_app_root()`` 返回可执行文件所在目录，也就是
    ``bundle_root / APP_NAME``；默认相对路径（``knowledge_base``、
    ``data/kb``、``data/logs``）都解析到那里。建在 ``bundle_root`` 上会留下
    一个应用永远不读的死目录，用户按 README 放文档只会拿到
    ``FileNotFoundError``。
    """
    prepare_runtime_layout(tmp_path)

    app_root = tmp_path / APP_NAME
    assert (app_root / "data" / "kb").is_dir()
    assert (app_root / "data" / "logs").is_dir()
    placeholder = app_root / "knowledge_base" / "README.md"
    assert placeholder.is_file()
    assert "Markdown" in placeholder.read_text(encoding="utf-8")
    # 包根不能残留同名死目录，否则用户会继续把文档放错位置。
    assert not (tmp_path / "knowledge_base").exists()


def _tiktoken_cache_key() -> str:
    import hashlib

    from scripts.build_binary_release import TIKTOKEN_BLOB_URL

    return hashlib.sha1(TIKTOKEN_BLOB_URL.encode(), usedforsecurity=False).hexdigest()


def test_stage_tiktoken_cache_copies_verified_blob(tmp_path, monkeypatch, bpe_source):
    """预置进包的 BPE 必须通过 sha256 校验，且落在 tiktoken 的查找键名下。"""
    import hashlib

    from scripts.build_binary_release import (
        TIKTOKEN_BLOB_SHA256,
        TIKTOKEN_CACHE_DIRNAME,
        stage_tiktoken_cache,
    )

    payload = b"x" * 32
    monkeypatch.setattr(
        "scripts.build_binary_release.TIKTOKEN_BLOB_SHA256", hashlib.sha256(payload).hexdigest()
    )
    (bpe_source / _tiktoken_cache_key()).write_bytes(payload)

    app_root = tmp_path / "app"
    stage_tiktoken_cache(app_root)

    staged = app_root / TIKTOKEN_CACHE_DIRNAME / _tiktoken_cache_key()
    assert staged.read_bytes() == payload
    # 键名必须与 tiktoken 的查找算法一致，否则运行时命中不到。
    assert staged.name == _tiktoken_cache_key()
    assert TIKTOKEN_BLOB_SHA256  # 常量存在且非空


def test_stage_tiktoken_cache_rejects_corrupted_blob(tmp_path, monkeypatch, bpe_source):
    """内容与声明哈希不符时必须失败，坏数据不能进发布包。"""
    from scripts.build_binary_release import stage_tiktoken_cache

    (bpe_source / _tiktoken_cache_key()).write_bytes(b"corrupted")

    with pytest.raises(RuntimeError, match="校验失败"):
        stage_tiktoken_cache(tmp_path / "app")


def test_remove_runtime_residue_clears_logs_and_cache(tmp_path):
    """自检产生的日志与缓存必须被清掉，占位目录本身要保留。"""
    from scripts.build_binary_release import APP_NAME, CACHE_DIRNAME, remove_runtime_residue

    bundle_root = tmp_path / "PyRAG-Kit-1.4.0-linux-x64"
    app_root = bundle_root / APP_NAME
    logs = app_root / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "app_log_2026-09-24.log").write_text("build-time residue", encoding="utf-8")
    cache = app_root / CACHE_DIRNAME
    cache.mkdir()
    (cache / "tmpfile").write_text("x", encoding="utf-8")

    remove_runtime_residue(bundle_root)

    assert list(logs.iterdir()) == []
    assert not cache.exists()
    # 占位目录本身要保留，用户解压后应看到空的 knowledge_base/ 与 data/。
    assert logs.is_dir()


def test_bundle_version_matches_pyproject(tmp_path, monkeypatch):
    """产物自报版本必须与 pyproject.toml 一致。

    版本号有两个消费方：发布包**文件名**来自 ``pyproject.toml``（构建脚本
    读取），而横幅与自检输出来自 ``main.py``。构建脚本把版本写进包内
    ``VERSION.txt``，``main.py`` 优先读它——若读的位置不对（例如按
    ``__file__`` 找，冻结后会落到 ``_internal/``），就会静默退回兜底值，
    包名与横幅不一致而自检只 grep "smoke test ok" 时照样通过。
    """
    from scripts.build_binary_release import APP_NAME, load_version

    app_root = tmp_path / APP_NAME
    app_root.mkdir(parents=True)
    (app_root / "VERSION.txt").write_text("9.9.9\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parent.parent
    # 冻结时 resolve_app_root() 取可执行文件所在目录；源码模式取 __file__ 的
    # 祖父目录。这里在导入 main 之前把该函数打桩成目标目录，等价于冻结分支。
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "import src.utils.config as cfg\n"
        "cfg.resolve_app_root = lambda: Path(sys.argv[2])\n"
        "import main\n"
        "print(main.VERSION)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(project_root), str(app_root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "9.9.9"
    assert load_version() != "9.9.9"


def test_extract_section_returns_version_content():
    section = extract_section("1.4.0")

    assert section.startswith("## [1.4.0]")
    assert "升级说明" in section


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
        # 版本必须与 ``load_version()`` 动态同源：硬编码会在每次升版本时
        # 触发「发布包自报版本与 pyproject.toml 不一致」的守卫（发 1.5.0
        # 时实测）。这正是本守卫要拦的漂移形态，夹具不能自己 reproduces 它。
        version = load_version()
        return subprocess.CompletedProcess(
            command, 0, stdout=f"PyRAG-Kit {version} smoke test ok\n", stderr=""
        )

    monkeypatch.setattr("scripts.build_binary_release.subprocess.run", fake_run)
    monkeypatch.setattr(
        "scripts.build_binary_release.load_version",
        lambda: load_version(),
    )

    validate_bundle(bundle_root)

    assert recorded["command"][-1] == "--smoke-test"
    # 退出码由 validate_bundle 自行检查（失败时携带 stdout/stderr），不经 check=True 抛出
    assert recorded["kwargs"]["check"] is False
    assert recorded["kwargs"]["capture_output"] is True
    # Windows 管道下 rich legacy renderer 按 cp1252 编码中文会崩（v1.5.0 发版实测），
    # 自检子进程必须强制 UTF-8 模式
    assert recorded["kwargs"]["env"]["PYTHONUTF8"] == "1"


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


def test_stage_bundle_copies_package_files_including_subdirectories(
    monkeypatch, tmp_path, bpe_source
):
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

    for relative in APP_ROOT_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("content", encoding="utf-8")

    import hashlib

    payload = b"stub-bpe"
    monkeypatch.setattr(
        "scripts.build_binary_release.TIKTOKEN_BLOB_SHA256", hashlib.sha256(payload).hexdigest()
    )
    (bpe_source / _tiktoken_cache_key()).write_bytes(payload)

    bundle_root = stage_bundle("linux-x64", "1.4.0")

    assert bundle_root.name == "PyRAG-Kit-1.4.0-linux-x64"
    for relative in PACKAGE_FILES:
        assert (bundle_root / relative).is_file(), relative
    # 子目录路径必须真的落在子目录里，而不是被拍平。
    assert (bundle_root / "licenses" / "APACHE-2.0.txt").parent.name == "licenses"
    # 配置示例必须落在 app 真正读取的位置（ROOT_DIR == bundle_root/APP_NAME），
    # 放在包根会被静默忽略。
    for relative in APP_ROOT_FILES:
        assert (bundle_root / APP_NAME / relative).is_file(), relative
        assert not (bundle_root / relative).exists(), relative


# ── 打包数据裁剪（pyfiglet 字体 / jieba 附加模型） ──


def test_prune_bundle_extras_keeps_only_used_fonts(tmp_path):
    """横幅只用 font="big"，pyfiglet 缺省字体是 standard；其余字体
    （打包后约 7.7 MB）运行时永远读不到，staging 时必须裁掉。"""
    from scripts.build_binary_release import prune_bundle_extras

    fonts = tmp_path / "_internal" / "pyfiglet" / "fonts"
    fonts.mkdir(parents=True)
    for name in ("big.flf", "standard.flf", "defleppard.flf", "bigmono12.tlf"):
        (fonts / name).write_bytes(b"font-data")

    prune_bundle_extras(tmp_path)

    assert sorted(p.name for p in fonts.iterdir()) == ["big.flf", "standard.flf"]


def test_prune_bundle_extras_removes_jieba_unused_subpackages(tmp_path):
    """hook-jieba 把 posseg/analyse/lac_small 的数据模型原样复制进包
    （约 24 MB）；项目只用 jieba.cut/lcut，三个子目录必须删掉，
    词典 dict.txt 与 finalseg 保留。"""
    from scripts.build_binary_release import prune_bundle_extras

    jieba = tmp_path / "_internal" / "jieba"
    (jieba / "posseg").mkdir(parents=True)
    (jieba / "analyse").mkdir()
    (jieba / "lac_small" / "model_baseline").mkdir(parents=True)
    (jieba / "finalseg").mkdir()
    (jieba / "dict.txt").write_bytes(b"dict")
    (jieba / "posseg" / "prob_emit.p").write_bytes(b"p")

    prune_bundle_extras(tmp_path)

    assert not (jieba / "posseg").exists()
    assert not (jieba / "analyse").exists()
    assert not (jieba / "lac_small").exists()
    assert (jieba / "dict.txt").is_file()
    assert (jieba / "finalseg").is_dir()


def test_prune_bundle_extras_silent_when_directories_missing(tmp_path):
    """pyfiglet / jieba 任一未被打进包时（防御分支）不得抛错。"""
    from scripts.build_binary_release import prune_bundle_extras

    prune_bundle_extras(tmp_path)  # 不应抛 FileNotFoundError


def test_stage_bundle_prunes_bundle_extras(monkeypatch, tmp_path, bpe_source):
    """staging 流程整体接入字体裁剪：真实 PyInstaller 产物里的字体目录
    在 stage_bundle 之后只应剩两个字体文件。"""
    dist_root = tmp_path / "dist" / "PyRAG-Kit"
    app_dist = dist_root / APP_NAME
    fonts_dir = app_dist / "_internal" / "pyfiglet" / "fonts"
    fonts_dir.mkdir(parents=True)
    for name in ("big.flf", "standard.flf", "defleppard.flf"):
        (fonts_dir / name).write_bytes(b"font-data")
    jieba_dir = app_dist / "_internal" / "jieba" / "posseg"
    jieba_dir.mkdir(parents=True)
    (jieba_dir / "prob_emit.p").write_bytes(b"p")

    artifact_root = tmp_path / "release_artifacts"
    monkeypatch.setattr("scripts.build_binary_release.DIST_ROOT", dist_root)
    monkeypatch.setattr("scripts.build_binary_release.ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr("scripts.build_binary_release.PROJECT_ROOT", tmp_path)
    for relative in PACKAGE_FILES + APP_ROOT_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("content", encoding="utf-8")

    import hashlib

    payload = b"stub-bpe"
    monkeypatch.setattr(
        "scripts.build_binary_release.TIKTOKEN_BLOB_SHA256", hashlib.sha256(payload).hexdigest()
    )
    (bpe_source / _tiktoken_cache_key()).write_bytes(payload)

    bundle_root = stage_bundle("linux-x64", "1.4.0")

    staged_fonts = bundle_root / APP_NAME / "_internal" / "pyfiglet" / "fonts"
    assert sorted(p.name for p in staged_fonts.iterdir()) == ["big.flf", "standard.flf"]
    assert not (bundle_root / APP_NAME / "_internal" / "jieba" / "posseg").exists()


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


# ── 回归：``HIDDEN_IMPORTS`` 必须覆盖 provider 注册表里的每个模块 ──
#
# 十三个 provider 模块**没有任何静态导入路径**——它们只经
# ``importlib.import_module(provider_info["module"])`` 按运行时字符串加载
# （``src/providers/factory.py``）。PyInstaller 的静态分析看不见这一层，
# 因此 ``HIDDEN_IMPORTS`` 是它们在冻结包里能否被找到的**唯一**依据。
#
# 两个清单分处两个文件，各自演化：往注册表里加一个 provider 而忘了同步
# ``HIDDEN_IMPORTS``，构建照常通过（PyInstaller 不报错）、``--validate`` 照常
# 通过（自检不实例化任何 provider），用户在那个渠道上拿到的是
# ``ModuleNotFoundError``。这与 ``tiktoken_ext.openai_public`` 当初漏收是同一个
# 失效模式——只是那次被发现了，这次没有守卫。


def test_hidden_imports_cover_every_registered_provider_module():
    """注册表里的每个模块都必须出现在 ``HIDDEN_IMPORTS`` 里。"""
    from scripts.build_binary_release import HIDDEN_IMPORTS
    from src.providers.factory import ModelProviderFactory

    registered = {info["module"] for info in ModelProviderFactory._provider_map.values()}
    missing = sorted(registered - set(HIDDEN_IMPORTS))

    assert not missing, (
        f"以下 provider 模块已注册但未列入 HIDDEN_IMPORTS，冻结包中将无法导入: {missing}"
    )


def test_hidden_imports_has_no_stale_provider_entries():
    """反向守卫：``HIDDEN_IMPORTS`` 不得残留已不存在的模块名。

    只查一个方向的话，注册表删掉一个 provider 后清单里的旧名字会一直留着，
    而 PyInstaller 对不存在的 hidden import 只是警告——清单会慢慢失真，等到
    真需要它覆盖某个模块时已经没人信它了。
    """
    from scripts.build_binary_release import HIDDEN_IMPORTS
    from src.providers.factory import ModelProviderFactory

    registered = {info["module"] for info in ModelProviderFactory._provider_map.values()}
    stale = sorted(
        name
        for name in HIDDEN_IMPORTS
        if name.startswith("src.providers.") and name not in registered
    )

    assert not stale, f"HIDDEN_IMPORTS 里的 provider 模块已不在注册表中: {stale}"


def test_registered_provider_modules_have_no_static_import_path():
    """确认这些模块确实只靠 ``importlib`` 加载——这正是需要 HIDDEN_IMPORTS 的原因。

    若哪天某个 provider 被静态导入了，这条用例会失败，提示该模块已不再依赖
    ``HIDDEN_IMPORTS``（清单可以相应精简）。这是对上面两条守卫前提的校验：
    前提不成立时它们守的是一个不存在的问题。
    """
    import re
    import subprocess

    from src.providers.factory import ModelProviderFactory

    for module in sorted({info["module"] for info in ModelProviderFactory._provider_map.values()}):
        # ``from X import Y`` 与 ``from .pkg import X`` 都算静态导入，与
        # ``import X`` 同罪——只 grep ``import {module}`` 会漏掉最常见的
        # from 形态，前提校验在它声称覆盖的路径上放空。
        patterns = [
            f"import {module}",
            f"from {module} import",
            f"from {module.rsplit('.', 1)[0]} import",
        ]
        result = subprocess.run(
            ["grep", "-rnE", "|".join(patterns), "--include=*.py", "src", "scripts", "main.py"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
        # factory.py 里的是 ``"module": "src.providers.x"`` 字符串，不是 import 语句
        hits = [line for line in result.stdout.splitlines() if "factory.py" not in line]
        assert not hits, f"{module} 出现了静态导入，HIDDEN_IMPORTS 不再是它唯一的发现途径: {hits}"
    assert re.search(r"importlib\.import_module", Path("src/providers/factory.py").read_text())


# ── 发布脚本四条判据的覆盖缺口 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3790 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 102 未知发布目标 → ``expected is None`` 后解包，抛裸
#   ``TypeError: cannot unpack non-iterable NoneType object``；
# * 295 联网下载 BPE 失败 → **静默返回 None**，随后 ``sha256(None)`` 在更远处
#   抛 ``TypeError``，与「构建环境无外网」这个真实原因无关；
# * 377 自检输出不含 "smoke test ok" → 消息退化成「自报版本与 pyproject.toml
#   的 1.4.0 不一致」——把「自检根本没跑通」说成「版本号写错了」，指向完全
#   错误的排查方向；
# * 383 自检输出带**错误版本号** → **静默通过**，旧版本产物冒充新版本发布。
def test_validate_target_environment_rejects_unknown_target():
    """未知发布目标必须报 ``ValueError``，不是解包 None 的裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="不支持的发布目标"):
        validate_target_environment("bogus-target")


def test_download_tiktoken_blob_reports_network_failure(monkeypatch):
    """联网下载失败必须报带上下文的 ``RuntimeError``，不能静默返回 None。"""
    import urllib.error
    import urllib.request

    from scripts.build_binary_release import _download_tiktoken_blob

    def failing_urlopen(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)

    with pytest.raises(RuntimeError, match="无法获取 cl100k_base"):
        _download_tiktoken_blob()


def test_validate_bundle_reports_missing_smoke_marker(monkeypatch, tmp_path):
    """自检输出不含 "smoke test ok" 时必须点名这一条，不能报成版本不一致。"""
    bundle_root = tmp_path / "bundle"
    app_dir = bundle_root / "PyRAG-Kit"
    app_dir.mkdir(parents=True)
    executable = app_dir / "PyRAG-Kit"
    executable.write_text("", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="some other output\n", stderr="")

    monkeypatch.setattr("scripts.build_binary_release.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="自检输出不符合预期"):
        validate_bundle(bundle_root)


def test_validate_bundle_rejects_mismatched_version(monkeypatch, tmp_path):
    """自检输出带错误版本号时必须拒绝，不能静默放行。"""
    bundle_root = tmp_path / "bundle"
    app_dir = bundle_root / "PyRAG-Kit"
    app_dir.mkdir(parents=True)
    executable = app_dir / "PyRAG-Kit"
    executable.write_text("", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="PyRAG-Kit 0.0.1-wrong smoke test ok\n", stderr=""
        )

    monkeypatch.setattr("scripts.build_binary_release.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="自报版本与 pyproject.toml"):
        validate_bundle(bundle_root)


def test_extract_section_reports_missing_version():
    """请求的版本不在 CHANGELOG 中时必须报 ``ValueError``。

    删除守卫后 ``start_index`` 保持 ``None``，切片 ``lines[None:end]`` 会
    **静默返回整份 CHANGELOG 的开头**——发布说明里出现「更新日志」总标题与
    Unreleased 段，看起来像一份正常的文件。
    """
    with pytest.raises(ValueError, match="未找到版本"):
        extract_section("99.99.99")


def test_extract_release_notes_truncates_to_github_limit(tmp_path, monkeypatch):
    """超长版本的 notes 必须截断——GitHub Release API 上限 125000 字符。

    v1.5.0 发版实测：1.5.0 节 264861 字符，``gh release create`` 以
    ``body is too long (maximum is 125000 characters)`` 拒绝，publish job
    失败（五平台产物全部构建成功却发不出去）。截断按整行边界，尾部附
    完整记录指引，绝不伪装成完整记录。
    """
    from scripts.extract_release_notes import extract_section

    changelog = tmp_path / "CHANGELOG.md"
    bulk = "\n".join(f"- 条目 {index}：" + "x" * 200 for index in range(1000))
    changelog.write_text(
        f"## [9.9.9] - 2026-01-01\n\n### 修复\n\n{bulk}\n\n## [9.8.0]\n\n旧版。\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.extract_release_notes.CHANGELOG_PATH", changelog)
    notes = extract_section("9.9.9")

    assert len(notes) <= 120_000
    assert "超出 GitHub Release 说明的长度上限" in notes
    assert notes.rstrip().endswith("。")


def test_extract_release_notes_stops_at_internal_archive_section(tmp_path, monkeypatch):
    """维护者向的存档节不进 Release notes。

    存档节是审查过程记录（十万字符级），对用户是噪声；完整内容仍在
    CHANGELOG。截断保险与存档排除是两条独立防线，各自都要有测试。
    """
    from scripts.extract_release_notes import extract_section

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "## [9.9.9] - 2026-01-01\n\n"
        "### 新增功能\n\n- 用户可见改动。\n\n"
        "### 内部存档：审查与加固过程\n\n以下为过程记录：\n\n"
        "- **第 N 轮 agents 复查**：大量维护者细节。\n\n"
        "## [9.8.0]\n\n旧版。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.extract_release_notes.CHANGELOG_PATH", changelog)
    notes = extract_section("9.9.9")

    assert "用户可见改动" in notes
    assert "内部存档" not in notes
    assert "agents 复查" not in notes
    assert notes.rstrip().endswith("用户可见改动。")
