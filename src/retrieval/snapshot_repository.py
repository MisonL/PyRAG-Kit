from __future__ import annotations

import json
import pickle  # nosec B403 - 仅用于读取应用自管的本地快照文件
import shutil
import tomllib
import uuid
from pathlib import Path

from src.runtime.contracts import SCHEMA_VERSION, KnowledgeSnapshotManifest, RunConfig
from src.utils.config import safe_repr
from src.utils.security import resolve_within

# 快照 id 的长度上界。本项目自己生成的 id 形如 ``<prefix>-<12 hex>``（几十个
# 字符），legacy 前缀也不过十几个字符；这个界留了极大余量，只可能被畸形数据
# 触及。它的作用是把「名字太长导致 stat 失败」这类**文件系统层**错误收敛成
# 带指引的 ``ValueError``——大多数文件系统的单个路径分量上限是 255 字节，而
# 中文字符在 UTF-8 下占 3 字节，因此这里按字符数取一个保守值。
_MAX_SNAPSHOT_ID_LENGTH = 64


class SnapshotRepository:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        root = run_config.snapshot_root
        # 形态先收敛：下一行直接对 ``root`` 调 ``.mkdir()``，非 ``Path`` 取值在那里
        # 抛裸 ``AttributeError: 'int' object has no attribute 'mkdir'``——不是本类
        # 其余拒绝路径承诺的带指引 ``ValueError``。``RunConfig`` 是 frozen
        # dataclass、字段没有校验器，嵌入方可以自拼 ``snapshot_root=5`` 的配置。
        # 只接受 ``Path``（``RunConfig`` 的标注就是它），不做静默强转。
        if not isinstance(root, Path):
            raise ValueError(
                f"snapshot_root 必须是 Path，但得到 {type(root).__name__}。"
                " 请检查 config.toml 里的 snapshot_root 配置项。"
            )
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_active_snapshot_id(self) -> str | None:
        marker = self.run_config.active_snapshot_marker
        if not marker.exists():
            return None
        # 读取本身要收敛：``exists()`` 对**目录**也为真，``read_text`` 于是抛
        # ``IsADirectoryError``；标记文件不是 UTF-8 时抛 ``UnicodeDecodeError``。
        # 两者都不是本类其余拒绝路径承诺的带指引 ``ValueError``，而
        # ``VectorStoreFactory._load_existing_state`` 的第一步就是本方法——
        # 启动会以裸 ``OSError``／``UnicodeDecodeError`` 收场。标记文件是磁盘上的
        # 任意字节，外部工具写坏即触发。
        try:
            snapshot_id = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"知识快照的 ACTIVE_SNAPSHOT 标记无法读取: {marker}"
                f"（{type(exc).__name__}: {exc}）。"
                " 该路径可能是一个目录，或文件内容不是 UTF-8；"
                "请检查该文件，或重建知识快照。"
            ) from exc
        if not snapshot_id:
            return None
        self._require_snapshot_id_within_root(snapshot_id, source=str(marker))
        return snapshot_id

    def _require_snapshot_id_within_root(self, snapshot_id: str, *, source: str) -> None:
        """确认 ``snapshot_id`` 拼进路径后仍直接位于快照根目录之下。

        快照 id 有两个来源：标记文件（磁盘上的任意文本）与调用方传参。两者都
        会被拼成 ``self.root / snapshot_id``，而该表达式对绝对路径会**顶掉**
        root（``Path("/a") / "/b"`` 得 ``/b``），对 ``..`` 与 ``a/../..`` 则会
        向上逃逸。本类的 ``create_temp_snapshot_dir`` / ``cleanup_temp_snapshot_dir``
        / ``finalize_snapshot`` 都对这个路径做 ``rmtree``，所以逃逸出去就是
        「删掉快照根目录之外的东西」——实测 ``finalize_snapshot(root/".tmp-x",
        "../../<victim>")`` 会把 root 外的目录整个删掉。

        判定用「解析后父目录必须等于 root」，而不是检查字符串里有没有分隔符：
        后者漏掉 ``..``（不含分隔符，却能跳出 root）。
        """
        # 判定用「必须是一个单一路径分量」，而不是「解析后的父目录等于 root」。
        # 后者拦不住 ``"x/../victim"``：它解析成 ``root/victim``，父目录**正是**
        # root，于是放行。而三个方法实际操作的路径是
        # ``self.root / f".tmp-{snapshot_id}"``——前缀 ``.tmp-`` 会**改变** ``..``
        # 的消解结果（``.tmp-x`` 是一个真实路径分量），``".tmp-x/../victim"``
        # 最终落在 ``root/victim``，即一个**正式快照**。实测
        # ``cleanup_temp_snapshot_dir("x/../victim")`` 对 ``root/victim`` 执行了
        # rmtree——「只清理任务临时目录」的意图被绕过，删的是正式快照。
        #
        # 快照 id 由本项目的 ``generate_snapshot_id()`` 产出（``<prefix>-<12 hex>``），
        # 或来自标记文件，两者都不含分隔符，因此「单分量」不会误伤合法值。
        # 判定放在 ``root`` 解析之前，对 ``Path`` 的规范化行为不敏感。
        # 形态先收敛：下面第一行就对 ``snapshot_id`` 构造 ``Path``，而
        # ``Path(5)``／``Path(["a"])``／``Path(b"abc")`` 抛裸 ``TypeError:
        # expected str, bytes or os.PathLike object, not ...``——不是本类承诺的
        # 带指引 ``ValueError``。位数超限的整数更隐蔽：``Path(10**4301)`` 抛
        # ``ValueError: Exceeds the limit``，类型对、内容却与快照 id 无关。
        # 快照 id 的来源之一是磁盘上的标记文件（``get_active_snapshot_id``），
        # 而三个 ``*_temp_snapshot_dir``／``finalize_snapshot`` 都对拼出的路径做
        # ``rmtree``，形态判据必须与 ``resolve_within``／``ensure_trusted_source``
        # 一致：构造 ``Path`` 之前先判形态。``str`` 之外的 ``bytes`` 也不接受
        # ——本项目所有快照路径都是文本路径，``os.fspath`` 对 ``bytes`` 会给出
        # 另一种报错，与其下游行为不一致。
        if not isinstance(snapshot_id, str):
            raise ValueError(
                f"快照 id 不合法: {safe_repr(snapshot_id)}（来源: {source}）。"
                " 它必须是字符串形式的单一目录名。请检查该值，或重建知识快照。"
            )
        if (
            not snapshot_id
            or Path(snapshot_id).is_absolute()
            or Path(snapshot_id).name != snapshot_id
            or snapshot_id in {".", ".."}
        ):
            raise ValueError(
                f"快照 id 不合法: {snapshot_id!r}（来源: {source}）。"
                " 它必须是单一目录名（不含路径分隔符，也不是 ``.``／``..``）。"
                " 请检查该值，或重建知识快照。"
            )
        # 长度也要判：上面的判据对超长名字**全部为真**（单个分量、相对、非
        # ``.``），但拼成路径后 ``.exists()`` 抛
        # ``OSError: [Errno 63] File name too long``——而
        # ``VectorStoreFactory._load_existing_state`` 的第一步就是
        # ``get_active_snapshot_dir()``，因此一个被外部工具写坏的标记文件会让
        # **启动**以 ``OSError`` 失败，而本类的契约是带指引的 ``ValueError``。
        # 上界取 ``_MAX_SNAPSHOT_ID_LENGTH``，见该常量说明。
        if len(snapshot_id) > _MAX_SNAPSHOT_ID_LENGTH:
            raise ValueError(
                f"快照 id 过长: {len(snapshot_id)} 个字符（来源: {source}），"
                f"上界 {_MAX_SNAPSHOT_ID_LENGTH}。"
                " 请检查该值，或重建知识快照。"
            )
        # ``resolve()`` 对符号链接环抛 ``RuntimeError: Symlink loop from ...``，
        # 它**不是** ``ValueError``，会穿透本类的契约（本类的其余判据一律抛带
        # 指引的 ``ValueError``）。标记文件是磁盘上的任意文本，用户手工摆一个
        # 环即触发；而 ``VectorStoreFactory._load_existing_state`` 的第一步就是
        # ``get_active_snapshot_dir()``，于是**启动**以 pathlib 的内部错误收场。
        # 判据与 ``utils.security.resolve_within``／``utils.cleanup
        # ._reject_unsafe_cache_dir`` 一致：把 ``(OSError, RuntimeError)`` 统一
        # 收敛成 ``ValueError``。``strict=False`` 不覆盖这种情况——环是**解析期**
        # 的失败，与目标是否存在无关。
        try:
            root = Path(self.root).resolve()
            candidate = (self.root / snapshot_id).resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"快照 id 无法解析（来源: {source}，id={snapshot_id!r}）："
                f"{type(exc).__name__}: {exc}。"
                " 快照根目录或其下的路径可能包含符号链接环或过长的间接层，"
                "请检查该值，或重建知识快照。"
            ) from exc
        if candidate.parent != root:
            raise ValueError(
                f"快照 id 不合法: {snapshot_id!r}（来源: {source}）。"
                f" 它解析到 {candidate}，不在快照根目录 {root} 之下。"
                " 请检查该值，或重建知识快照。"
            )

    def get_active_snapshot_dir(self) -> Path | None:
        snapshot_id = self.get_active_snapshot_id()
        if not snapshot_id:
            return None
        snapshot_dir = self.root / snapshot_id
        return snapshot_dir if snapshot_dir.exists() else None

    def create_temp_snapshot_dir(self, snapshot_id: str) -> Path:
        self._require_snapshot_id_within_root(snapshot_id, source="create_temp_snapshot_dir")
        temp_dir = self.root / f".tmp-{snapshot_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def cleanup_temp_snapshot_dir(self, snapshot_id: str) -> None:
        """删除指定构建任务的临时目录，不触碰正式快照。"""
        self._require_snapshot_id_within_root(snapshot_id, source="cleanup_temp_snapshot_dir")
        temp_dir = self.root / f".tmp-{snapshot_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def finalize_snapshot(self, temp_dir: Path, snapshot_id: str) -> Path:
        self._require_snapshot_id_within_root(snapshot_id, source="finalize_snapshot")
        # 兄弟入参 ``temp_dir`` 同样要判形态：本方法的两个参数，此前只有
        # ``snapshot_id`` 有判据。``temp_dir.replace(...)`` 是 ``Path`` 的方法，
        # 非 ``Path`` 取值在这里抛的是与「快照目录写错了」无关的裸异常——实测
        # ``5``／``None``／``["a"]`` 抛 ``AttributeError: ... object has no
        # attribute 'replace'``，而 ``str`` 更隐蔽：字符串**有** ``replace``，
        # 只是语义完全不同，抛的是 ``TypeError: replace expected at least 2
        # arguments, got 1``。本类其余拒绝路径一律给带指引的 ``ValueError``。
        #
        # 字符串形态尤其现实：同项目里目录参数普遍标注 ``str | os.PathLike``
        # （``FaissStore.save_snapshot(snapshot_dir: str)``、本类的
        # ``save_snapshot`` 等），调用方按那个惯例传字符串就会命中这里。
        # 判据与 ``__init__`` 对 ``snapshot_root`` 的一致：只接受 ``Path``
        # （本方法的标注就是它），不做静默强转。
        if not isinstance(temp_dir, Path):
            raise ValueError(
                "finalize_snapshot 的 temp_dir 必须是 Path，"
                f"但得到 {type(temp_dir).__name__}。"
                " 请传入 create_temp_snapshot_dir() 的返回值。"
            )
        final_dir = self.root / snapshot_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        self.activate_snapshot(snapshot_id)
        return final_dir

    def activate_snapshot(self, snapshot_id: str) -> None:
        # 与另外三个接收 ``snapshot_id`` 的方法（``create_temp_snapshot_dir``／
        # ``cleanup_temp_snapshot_dir``／``finalize_snapshot``）共用同一判据。
        # 本方法此前是**唯一**漏掉它的：它把 ``snapshot_id`` 直接写进标记文件，
        # 而标记文件正是 ``get_active_snapshot_id``／``get_active_snapshot_dir``
        # 的输入——两个读取侧都会用 ``_require_snapshot_id_within_root`` 校验。
        # 于是同一个值在**写**侧被接受、在**读**侧被拒绝，收场是「激活成功」之后
        # 每一次启动都在 ``VectorStoreFactory._load_existing_state`` 的第一步失败。
        # 实测 ``activate_snapshot("../evil")`` 正常返回并把 ``../evil`` 写进
        # ``ACTIVE_SNAPSHOT``，随后 ``get_active_snapshot_id()`` 抛
        # 「快照 id 不合法: '../evil'」；非字符串（``5``／``None``）与位数超限的
        # 整数则在这一行抛裸 ``TypeError: data must be str``／``ValueError:
        # Exceeds the limit``——而本类的契约是带指引的 ``ValueError``。
        # 判据在写入之前，异常时标记文件保持不变。
        self._require_snapshot_id_within_root(snapshot_id, source="activate_snapshot")
        marker = self.run_config.active_snapshot_marker
        marker.parent.mkdir(parents=True, exist_ok=True)
        temp_marker = marker.with_suffix(".tmp")
        temp_marker.write_text(snapshot_id, encoding="utf-8")
        temp_marker.replace(marker)

    def generate_snapshot_id(self, prefix: str = "kb") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def write_manifest(self, snapshot_dir: Path, manifest: KnowledgeSnapshotManifest) -> None:
        # 形态判据与同类的 ``__init__``（``snapshot_root``）、``finalize_snapshot``
        # （``temp_dir``）一致：只接受 ``Path``，不静默强转。此前这一处漏了，非
        # ``Path`` 取值在 ``snapshot_dir / "manifest.toml"`` 处抛裸
        # ``TypeError: unsupported operand type(s) for /: 'str' and 'str'``——没有
        # 字段名、也没有「该传什么」的指引，而本类的其余拒绝路径一律给带指引的
        # ``ValueError``。
        #
        # 字符串形态尤其现实：同一个 ``temp_dir`` 在
        # ``KnowledgeBuildService.build`` 里紧邻的两行分别以 ``str``（
        # ``vector_store.save_snapshot(str(temp_dir))``）与 ``Path``
        # （``write_manifest(temp_dir, ...)``）传给两个 API，调用方按前者的惯例
        # 传字符串就会命中这里。
        if not isinstance(snapshot_dir, Path):
            raise ValueError(
                "write_manifest 的 snapshot_dir 必须是 Path，"
                f"但得到 {type(snapshot_dir).__name__}。"
                " 请传入 create_temp_snapshot_dir() 的返回值。"
            )
        (snapshot_dir / "manifest.toml").write_text(manifest.to_toml(), encoding="utf-8")

    def load_manifest(self, snapshot_dir: Path) -> KnowledgeSnapshotManifest:
        """读取 manifest，并校验 ``schema_version`` 与当前实现一致。

        版本放在这里判而不是 ``from_mapping`` 里：``from_mapping`` 是纯解析，
        塞进版本策略会让「解析」和「准入」两件事在同一处各判一次。未知版本必须
        拒绝——旧/新 schema 的字段语义可能已经变了（本项目 v1 到 v2 就调整过
        分块与 embedding 的落盘结构），照着当前代码去解释未知版本会得到错误结果
        而不是报错。
        """
        # 解析失败与其它快照文件同样转成「快照不自洽」的 ValueError：本方法在
        # ``VectorStoreFactory`` 里紧跟在 ``validate_snapshot_dir`` 之后，是启动
        # 路径上的第二步。抛裸的 ``TOMLDecodeError``（它继承 ``ValueError``，但
        # 消息是 ``Unterminated string`` 这类纯语法描述）既没有「快照坏了」的
        # 语义，也没有重建指引，与 chunks.pkl／parents.pkl／stats.json 的收场
        # 方式不一致。
        try:
            with (snapshot_dir / "manifest.toml").open("rb") as file:
                data = tomllib.load(file)
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：manifest.toml 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        # ``from_mapping`` 是纯解析，字段缺失抛 ``KeyError``、计数字段不是整数
        # 时抛 ``int()`` 的裸 ``ValueError``（``"abc"`` -> invalid literal；
        # 位数超限 -> Exceeds the limit）。这些同样属于「快照写坏了」，却都没有
        # 「快照不自洽」的语义与重建指引——而同一份文件语法错时给的正是那种消息。
        # 校验放在这里而不是 ``from_mapping`` 里：那里保持纯解析，不塞策略。
        #
        # ``OverflowError`` 也接住：``int(inf)`` 抛它，而 TOML 有原生的 ``inf``
        # 浮点字面量——``document_count = inf`` 是合法的 manifest.toml，
        # ``tomllib`` 直接解析成 ``float('inf')``（实测）。它继承
        # ``ArithmeticError`` 而非 ``ValueError``，三个既有分支都接不住。
        try:
            manifest = KnowledgeSnapshotManifest.from_mapping(data)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"知识快照不自洽：manifest.toml 的字段无法解析（{type(exc).__name__}: {exc}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "知识快照的 schema 版本与当前程序不兼容。"
                f" 快照={manifest.schema_version}，当前程序={SCHEMA_VERSION}。"
                " 请重建知识快照；如需保留旧快照，请先用生成它的版本导出数据。"
            )
        return manifest

    def validate_snapshot_dir(self, snapshot_dir: Path) -> None:
        """校验快照文件齐全**且各数据源的行数自洽**。

        只查文件存在性是不够的：实测手工构造 ``documents=3`` 而
        ``embeddings=(2,4)`` 的 store，``save_snapshot`` 正常落盘、本方法通过、
        加载后也不报错——不一致被完整持久化。之后 ``semantic_search`` 拿
        ``indices`` 去索引 ``self.documents`` 会越界，而报错现场离真正的原因
        （写入时就不一致）很远。这里在加载/切换快照的入口就把它们对齐。
        """
        # 先确认目录归属，再谈内容：快照目录里的 chunks.pkl/parents.pkl/
        # lexical.index 都是 pickle，反序列化即执行代码，而「路径长在 root 附近」
        # 不构成来源证明。resolve_within 会把符号链接逃逸（root/link -> /etc）
        # 一并拦下。
        resolved = resolve_within(snapshot_dir, self.root, label="知识快照目录")
        # 符号链接判据必须看**原始**（未解析）取值：``resolve_within`` 已经把
        # 符号链接解析掉了，对 ``resolved`` 调 ``is_symlink()`` 恒为 False，等于
        # 没有这条判据。但原始取值的形态不受本方法的标注约束——``resolve_within``
        # 显式接受 ``str | os.PathLike``（见它的形态判据），因此调用方传字符串是
        # 合法输入，而 ``str`` 没有 ``is_symlink``：实测传字符串时本方法在
        # ``resolve_within`` 通过之后抛裸 ``AttributeError: 'str' object has no
        # attribute 'is_symlink'``——本类其余拒绝路径一律给带指引的 ``ValueError``。
        # 判据落在 ``resolve_within`` 之后，因此这里只需包一层 ``Path``：
        # ``int``／``None``／``list`` 已被它拒掉，剩下的必然是 ``str`` 或
        # ``os.PathLike``。
        if Path(snapshot_dir).is_symlink():
            raise ValueError(f"知识快照目录不能是符号链接: {snapshot_dir}")
        if resolved.parent != Path(self.root).resolve():
            raise ValueError(
                f"知识快照目录必须直接位于快照根目录之下: {resolved}（根={self.root}）。"
            )
        snapshot_dir = resolved

        required_files = [
            snapshot_dir / "manifest.toml",
            snapshot_dir / "chunks.pkl",
            snapshot_dir / "parents.pkl",
            snapshot_dir / "semantic.index",
            snapshot_dir / "embeddings.npy",
            snapshot_dir / "stats.json",
        ]
        missing_files = [str(path.name) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"知识快照不完整，缺少文件: {', '.join(missing_files)}")

        # manifest 也在 required_files 里，但此前只查存在性、不查能否解析。
        # 结果是「先 validate 再 load_manifest」的调用方（``VectorStoreFactory``）
        # 在校验这一步拿到通过，到下一步才炸，且炸出的是裸 TOMLDecodeError。
        try:
            with (snapshot_dir / "manifest.toml").open("rb") as file:
                tomllib.load(file)
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：manifest.toml 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc

        self._validate_snapshot_row_counts(snapshot_dir)

    @staticmethod
    def _validate_snapshot_row_counts(snapshot_dir: Path) -> None:
        """确认 chunks / embeddings / 两个索引文件的行数一致。

        读取全部走本地受信快照目录（与 ``FaissStore.load_snapshot`` 同一
        前提），且只取形状不取内容语义。
        """
        import numpy as np

        # 解析失败要转成「快照不自洽」的 ValueError：这里是加载入口，抛裸的
        # EOFError/UnpicklingError 会让调用方看到与真实原因无关的异常类型
        # （测试里就复现过：占位空文件 chunks.pkl -> EOFError）。
        try:
            with (snapshot_dir / "chunks.pkl").open("rb") as file:
                chunks = pickle.load(file)  # nosec B301 - 应用自管的快照目录
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：chunks.pkl 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc

        try:
            embeddings = np.load(snapshot_dir / "embeddings.npy")
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：embeddings.npy 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc

        # 顶层必须是序列。``None`` 会被下面的 ``len()`` 当成 0 个分块而放行，
        # 但 ``FaissStore._normalize_loaded_documents`` 随后对 ``None`` 迭代，
        # 抛裸的 ``TypeError: 'NoneType' object is not iterable``——与「快照
        # 不自洽」无关的类型，也没有重建指引。
        if chunks is None:
            raise ValueError(
                "知识快照不自洽：chunks.pkl 的顶层是 None，无法作为分块列表使用。请重建快照。"
            )
        if isinstance(chunks, (str, bytes, dict)) or not hasattr(chunks, "__len__"):
            raise ValueError(
                "知识快照不自洽：chunks.pkl 的顶层不是列表"
                f"（{type(chunks).__name__}），无法作为分块列表使用。请重建快照。"
            )
        chunk_count = len(chunks)
        if embeddings.ndim != 2:
            raise ValueError(f"知识快照的 embeddings.npy 不是二维数组: ndim={embeddings.ndim}。")
        embedding_rows = int(embeddings.shape[0])

        if chunk_count != embedding_rows:
            raise ValueError(
                "知识快照不自洽：分块数与向量数不一致。"
                f" chunks.pkl={chunk_count}，embeddings.npy={embedding_rows}。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            )

        # ``parents.pkl`` 是三个 ``pickle.load`` 目标之一，同为 required file，但
        # 此前只有它没被检查：写坏时本方法照常通过，直到
        # ``FaissStore.load_snapshot`` 抛裸的 ``UnpicklingError: invalid load key``；
        # 顶层写成 ``None``／``int`` 则在 ``dict(self.parent_documents)`` 处抛
        # ``TypeError: 'NoneType' object is not iterable``。两者都没有「快照坏了」
        # 的语义，也没有重建指引——而同一份快照里 ``chunks.pkl`` 坏掉时给出的正是
        # 有指引的 ValueError，同类失败不该有两种收场方式。
        try:
            with (snapshot_dir / "parents.pkl").open("rb") as file:
                parents = pickle.load(file)  # nosec B301 - 应用自管的快照目录
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：parents.pkl 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        if not isinstance(parents, dict):
            raise ValueError(
                "知识快照不自洽：parents.pkl 的顶层不是映射"
                f"（{type(parents).__name__}），无法作为父块索引使用。请重建快照。"
            )
        # 父条目要么是 dict（取 ``content``），要么是 str（整体即内容）——
        # ``FaissStore._normalize_parent_document`` 的取值契约。其他类型会让
        # 加载期以 ``'list' object has no attribute 'get'`` 收场。
        # ``safe_repr`` 而不是 ``str``：这段代码的唯一用途是**拼错误消息**，
        # 而键来自磁盘上的 parents.pkl（pickle 是二进制格式，不受十进制位数
        # 限制）。用 ``str`` 时，正要报「父条目形态不对」的那一步自己抛
        # ``ValueError: Exceeds the limit``，用户拿到与快照无关的整数转换错误。
        invalid_parents = [
            safe_repr(key) for key, value in parents.items() if not isinstance(value, (dict, str))
        ]
        if invalid_parents:
            preview = "、".join(invalid_parents[:3])
            raise ValueError(
                "知识快照不自洽：parents.pkl 中的父条目既不是映射也不是字符串"
                f"（例如 {preview}），无法取出父块内容。请重建快照。"
            )

        stats_path = snapshot_dir / "stats.json"
        try:
            with stats_path.open("rb") as file:
                stats = json.load(file)
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：stats.json 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        if not isinstance(stats, dict):
            raise ValueError("知识快照不自洽：stats.json 的顶层不是对象。")
        # ``int(declared)`` 同样不能直接调：``"abc"`` 抛裸 ValueError、列表抛
        # TypeError、超长数字串抛位数超限的 ValueError，三者都没有「快照不自洽」
        # 的语义与重建指引。与上面 ``from_mapping`` 的处理同理。
        #
        # ``OverflowError`` 必须一并接住：``int(inf)`` 抛的是它，而它继承
        # ``ArithmeticError``、**不是** ``ValueError``。这不是假想输入——
        # ``json.load`` 把 JSON 的 ``Infinity``／``1e999`` 都解析成 ``inf``
        # （前者是 Python 扩展字面量，后者是**合法 JSON**），于是启动路径上
        # 用户看到的是 ``OverflowError: cannot convert float infinity to
        # integer``，既无「快照不自洽」语义也无重建指引。对照：``NaN`` 走
        # ``ValueError`` 分支，原本就能得到正确消息——只有 ``inf`` 这一支漏了。
        declared = stats.get("chunk_count")
        if declared is not None:
            try:
                declared_count = int(declared)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "知识快照不自洽：stats.json 的 chunk_count 不是整数"
                    f"（{type(declared).__name__}: {declared!r}，{type(exc).__name__}）。"
                    " 快照可能在写入或复制过程中被截断，请重建。"
                ) from exc
            if declared_count != chunk_count:
                raise ValueError(
                    "知识快照不自洽：stats.json 的 chunk_count 与实际分块数不一致。"
                    f" stats.json={declared}，chunks.pkl={chunk_count}。"
                )

        # 向量索引的条数也要对齐。只查 chunks 与 embeddings 是不够的：
        # ``FaissStore.load_snapshot`` 在 ``lexical.index`` 存在时**不会**重建
        # FAISS 索引，直接使用盘上的 ``semantic.index``。实测 ntotal 比
        # chunks 多（3 vs 5）时本方法放行，加载后 ``semantic_search`` 拿
        # ``indices`` 去索引 ``self.documents`` 抛
        # ``IndexError: list index out of range``——正是本方法要防的那类失败。
        import faiss

        try:
            index = faiss.read_index(str(snapshot_dir / "semantic.index"))
        except Exception as exc:
            raise ValueError(
                f"知识快照不自洽：semantic.index 无法解析（{type(exc).__name__}）。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            ) from exc
        index_total = int(index.ntotal)
        if index_total != chunk_count:
            raise ValueError(
                "知识快照不自洽：向量索引条数与分块数不一致。"
                f" semantic.index={index_total}，chunks.pkl={chunk_count}。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            )
        # 维度也要对齐。只比行数是不够的：实测分块数／embedding 行数／ntotal
        # 都是 1，但 ``embeddings.npy`` 是 4 维而 ``semantic.index.d`` 是 3 时
        # 校验放行，加载后 ``semantic_search`` 把 4 维查询向量送进 3 维索引，
        # FAISS 抛的是**空消息的 AssertionError**——没有「快照坏了」的语义，
        # 也没有重建指引。这类不一致同样只可能来自写入或复制损坏。
        embedding_dimension = int(embeddings.shape[1])
        index_dimension = int(index.d)
        if index_dimension != embedding_dimension:
            raise ValueError(
                "知识快照不自洽：向量索引维度与 embedding 维度不一致。"
                f" semantic.index.d={index_dimension}，embeddings.npy={embedding_dimension}。"
                " 快照可能在写入或复制过程中被截断，请重建。"
            )

        # 关键词索引的语料条数同理。``FaissStore.load_snapshot`` 直接用盘上的
        # ``lexical.index`` 建 BM25，而 ``keyword_search`` 拿
        # ``np.argsort(doc_scores)`` 的下标去索引 ``self.documents``——语料比
        # 分块多时，命中越界位置的文档会让下标超出 ``documents`` 而抛
        # ``IndexError``（实测语料 5 条、分块 3 条时复现）。
        lexical_path = snapshot_dir / "lexical.index"
        if lexical_path.exists():
            try:
                with lexical_path.open("rb") as file:
                    tokenized = pickle.load(file)  # nosec B301 - 应用自管的快照目录
            except Exception as exc:
                raise ValueError(
                    f"知识快照不自洽：lexical.index 无法解析（{type(exc).__name__}）。"
                    " 快照可能在写入或复制过程中被截断，请重建。"
                ) from exc
            # 形态守卫与上面 ``chunks.pkl`` 的一致：``is not None`` 只挡
            # ``None``，而 ``lexical.index`` 顶层是 ``int`` 时 ``len()`` 抛裸
            # ``TypeError: object of type 'int' has no len()``——无「快照不自洽」
            # 语义、无重建指引，与该文件其它三个 pickle 的收场方式不一致。
            # 字符串形态更隐蔽：``len("ab") == 2`` 会被当成 2 条语料静默通过。
            if isinstance(tokenized, (str, bytes, dict)) or (
                tokenized is not None and not hasattr(tokenized, "__len__")
            ):
                raise ValueError(
                    "知识快照不自洽：lexical.index 的顶层不是列表"
                    f"（{type(tokenized).__name__}），无法作为关键词索引语料使用。请重建快照。"
                )
            lexical_rows = len(tokenized) if tokenized is not None else 0
            if lexical_rows != chunk_count:
                raise ValueError(
                    "知识快照不自洽：关键词索引语料条数与分块数不一致。"
                    f" lexical.index={lexical_rows}，chunks.pkl={chunk_count}。"
                    " 快照可能在写入或复制过程中被截断，请重建。"
                )
            # **条目**形态：上面只覆盖顶层与条数，而 ``FaissStore.load_snapshot``
            # 拿到盘上的语料后直接交给 ``BM25Okapi``，它在 ``_initialize`` 里对
            # 每条语料调 ``len(document)``。实测 ``[7]``／``[1.5]``／``[None]``
            # 三种条目都能穿过本方法，随后在载入期抛裸 ``TypeError: object of
            # type 'int' has no len()``——而本方法的职责正是在**激活之前**拦住
            # 坏快照：放行的后果是坏快照被 finalize 成活动快照，此后每次启动都
            # 以与「快照坏了」无关的异常类型失败，且用户拿不到重建指引。
            #
            # 只判「是不是序列」而不要求 ``list[str]``：语料由本应用写出，形态
            # 确定；这里挡的是「根本不是序列」这一类。
            for position, entry in enumerate(tokenized if tokenized is not None else []):
                if isinstance(entry, (str, bytes)) or not hasattr(entry, "__len__"):
                    raise ValueError(
                        "知识快照不自洽：lexical.index 的第 "
                        f"{position} 条语料不是序列（{type(entry).__name__}），"
                        "无法作为关键词索引语料使用。请重建快照。"
                    )
