# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import json
import os

# Pickle is retained only for trusted local legacy snapshots.
import pickle  # nosec B403
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import jieba  # type: ignore[import-untyped]
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "FAISS is not installed. Please install it with `pip install faiss-cpu`."
    ) from exc

from ...ui.display_utils import document_source
from ...utils.config import _MAX_RETRIEVAL_TOP_K, safe_repr
from ...utils.log_manager import get_module_logger
from ...utils.security import ensure_trusted_source, resolve_within
from .base import VectorStoreBase

logger = get_module_logger(__name__)
jieba.setLogLevel(jieba.logging.ERROR)


def _parent_key(parent_id: Any) -> str:
    """把 ``parent_id`` 收敛成父文档字典的键。

    四处都用裸 ``str()``，而 ``parent_id`` 来自知识快照的元数据——本项目加载
    pickle 格式的快照，pickle 是二进制格式、不受 CPython 十进制位数限制
    （实测 ``pickle.loads(pickle.dumps(10**4301))`` 正常往返）。位数超限时
    ``str()`` 抛 ``ValueError: Exceeds the limit``：在 ``load_snapshot`` 里它让
    **加载宣布成功之后**的每一次检索都崩，在 ``resolve_parent_content`` 里它让
    父块召回整体不可用。``safe_repr`` 对位数超限的值给出量级描述，同一取值
    始终得到同一键，因此父块仍能被正确解析到。
    """
    if isinstance(parent_id, str):
        return parent_id
    return safe_repr(parent_id)


def _require_top_k(top_k: Any) -> int:
    """校验检索方法的 ``top_k``，返回原值。

    ``semantic_search`` 与 ``keyword_search`` 同名参数、同一份上界来源，判据必须
    一致——此前两处完全不同，同一取值给出三种收场：

    * ``0``／负数：``semantic_search`` 抛 FAISS 内建 ``assert k > 0`` 产生的
      **无消息** ``AssertionError``（``str(e)`` 是空串，栈里只有 C 层）；
      ``keyword_search`` 则因为循环体先 ``append`` 再判 ``len(results) >= top_k``，
      **返回 1 条**——调用方要求「不要结果」却拿到一条，且无任何提示。
    * 非整数：FAISS 报 ``TypeError: an integer is required``，
      ``keyword_search`` 在 ``int >= str`` 处报 ``TypeError: '>=' not supported``
      ——两者都是裸 ``TypeError``，不是本项目的 ``ValueError``，调用方按
      ``except ValueError`` 写的处理接不住。
    * 过大：FAISS 不报错也不截断，直接按 k 分配结果数组——实测 ``top_k=10**9``
      挂起（>30s），``10**8`` 耗时 15.3s。配置层的 ``_MAX_RETRIEVAL_TOP_K`` 拦不住
      这里：它只作用在 ``Settings.chat_top_k`` 与 ``SessionConfig.top_k`` 上，而
      这两个方法是 ``VectorStoreBase`` 的公开方法，参数一路透传到 FAISS。

    ``bool`` 显式排除：它是 ``int`` 子类，``True`` 会被当成 1 放行，而那几乎必然
    是上游把标志位写进了规模字段。
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k 必须是整数，但得到 {safe_repr(top_k)}。")
    if not 1 <= top_k <= _MAX_RETRIEVAL_TOP_K:
        # 这条消息里也必须用 ``safe_repr``，与上面那条一致。裸 ``{top_k}`` 对位数
        # 超过 ``sys.get_int_max_str_digits()``（默认 4300）的整数会抛
        # ``ValueError: Exceeds the limit``——正要报「top_k 越界」的那一步自己
        # 抛错，用户拿到的是整数转换错误，与 top_k 毫无关系。实测
        # ``top_k=10**4301`` 时两个方法都复现；``10**4000`` 则正常（位数在上限
        # 内，只是消息被 ``safe_repr`` 的 80 字符截断规则处理得更可读）。
        raise ValueError(
            f"top_k 必须是 1 到 {_MAX_RETRIEVAL_TOP_K} 之间的整数，但得到 {safe_repr(top_k)}。"
        )
    return top_k


class FaissStore(VectorStoreBase):
    def __init__(
        self,
        file_path: str | None = None,
        *,
        trusted_paths: Iterable[str | os.PathLike[str]] | None = None,
    ):
        """构造空 store，或从一个**显式受信**的 legacy pickle 载入。

        ``file_path`` 只在同时给出 ``trusted_paths`` 时才加载。pickle 反序列化
        等同执行代码，路径本身不能证明来源（``AGENTS.md`` 要求只加载本项目生成或
        明确受信的文件），所以没有受信声明时这里不加载、也不静默改用空索引——
        直接抛错，让调用方决定是补上信任声明还是走快照加载。
        """
        self.file_path = file_path
        self.documents: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.parent_documents: dict[str, dict[str, Any]] = {}
        self._tokenized_docs_cache: list[list[str]] = []
        self.bm25_index: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None

        if not file_path:
            return
        if not os.path.exists(file_path):
            return
        self.load(file_path, trusted_paths=trusted_paths)

    @staticmethod
    def _normalize_parent_document(parent_document: Any) -> dict[str, Any]:
        if isinstance(parent_document, dict):
            return {
                "content": parent_document.get("content", ""),
                "metadata": parent_document.get("metadata") or {},
            }
        if isinstance(parent_document, str):
            return {"content": parent_document, "metadata": {}}
        return {"content": "", "metadata": {}}

    @staticmethod
    def _index_text_of(document: Any) -> str:
        """从任意形态的条目里取出建索引用文本，非映射条目返回空串。

        ``_build_index_text`` 的契约是「拿到一个映射」，而 ``self.documents``
        的**历史**内容不受写入侧校验的约束：``load_snapshot`` 会从 pickle 读回
        任意形态（虽有 ``_validate_document_shape``，但那是后来才加的边界），而
        重建索引是**由 documents 派生的缓存**，不该在派生期抛异常——一抛就卡死
        整个 store（见 ``_rebuild_indices`` 的说明）。这里把非法形态降级成空串：
        该条目缺席检索，但其余条目照常可用，store 也不再卡死。
        """
        if not isinstance(document, Mapping):
            return ""
        # ``_build_index_text`` 的两处取值同样要求形态正确：``metadata`` 非映射时
        # ``.get`` 抛 ``AttributeError``，``page_content`` 非字符串时要么在
        # ``in`` 上抛 ``TypeError``，要么把列表交给 ``jieba.cut`` 抛
        # ``AttributeError: 'list' object has no attribute 'decode'``。这里先归一化
        # 成契约形态再交给它——非字符串内容没有可检索的文本表示，降级为空串。
        metadata = document.get("metadata")
        content = document.get("page_content", "")
        return FaissStore._build_index_text(
            {
                "page_content": content if isinstance(content, str) else "",
                "metadata": metadata if isinstance(metadata, Mapping) else {},
            }
        )

    @staticmethod
    def _build_index_text(document: dict[str, Any]) -> str:
        metadata = document.get("metadata") or {}
        source = metadata.get("source", "")
        source_hint = ""
        if isinstance(source, str) and source:
            source_hint = Path(source).stem.replace("_", " ").replace("-", " ")

        page_content = document.get("page_content", "") or ""
        if source_hint and source_hint not in page_content:
            return f"{source_hint}\n{page_content}"
        return page_content

    @staticmethod
    def _build_bm25_index(tokenized_docs: list[list[str]]) -> BM25Okapi | None:
        """构造 BM25 索引；空语料返回 None 而不是让 rank_bm25 除零。

        ``BM25._initialize`` 计算 ``avgdl = num_doc / self.corpus_size``，
        语料为空时直接 ``ZeroDivisionError``。``load_snapshot`` 会从
        ``lexical.index`` 读回 ``[]``（空快照或手工构造的目录），旧实现
        在此崩溃且不报「快照为空」这个真实原因。
        """
        # 两个条件缺一不可：``not tokenized_docs`` 只挡住**外层为空**，而
        # ``rank_bm25`` 的 ``_calc_idf`` 在**所有 token 都为空**时也会除零
        # （``idf_sum / len(self.idf)``，分母为 0）。实测 ``[[]]`` 与 ``[[], []]``
        # 都抛 ``ZeroDivisionError: division by zero``——``jieba.cut("")`` 返回
        # ``[]``，因此一条 ``page_content=""`` 的文档就足以构造出这个形态，而空
        # ``page_content`` 在快照里是合法的（``_validate_document_shape`` 只要求
        # 它是 ``str``）。两种形态的语义相同：没有可检索的语料，返回 ``None``。
        if not tokenized_docs or not any(tokenized_docs):
            return None
        return BM25Okapi(tokenized_docs)

    def _rebuild_indices(self) -> None:
        if self.documents:
            # 先整体算完再赋值。原先直接给 ``self._tokenized_docs_cache`` 赋一个
            # 列表推导式的结果——推导式在赋值**前**整体求值，中途抛异常时该属性
            # 保持旧值，而 ``self.documents`` 已经多出条目，于是
            # ``_tokenized_docs_cache[i]`` 与 ``self.documents[i]`` 不再一一对应。
            # 两个消费点都按下标取值：``keyword_search`` 拿
            # ``self._tokenized_docs_cache[idx]`` 打分却取 ``self.documents[idx]``
            # 作结果，``_promote_parent_context`` 同样按下标取文档。错配即张冠李戴。
            #
            # 更要紧的是它让 store **卡死**：``_build_index_text`` 对非映射条目抛
            # ``AttributeError``，因此只要 documents 里混进一条孤儿，此后**每一次**
            # ``upsert_embeddings`` 都会在重建索引时重复抛出——连喂完全合法的文档
            # 也失败。实测连续 3 次合法 upsert 全部报同一个错，documents 停在 5 条
            # 不再增长，向量库事实上报废。索引重建是「由 documents 派生的缓存」，
            # 必须容忍 documents 里的任何形态，把校验交给写入侧的入口。
            tokenized_docs = [list(jieba.cut(self._index_text_of(doc))) for doc in self.documents]
            self._tokenized_docs_cache = tokenized_docs
            self.bm25_index = self._build_bm25_index(tokenized_docs)
        else:
            self._tokenized_docs_cache = []
            self.bm25_index = None

        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = int(self.embeddings.shape[1])
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings.astype(np.float32))
        else:
            self.faiss_index = None

    @staticmethod
    def _validate_embeddings_shape(embeddings: Any, *, source: str) -> None:
        """确认向量矩阵是二维的，非法形态给出可读的 ``ValueError``。

        两个载入入口（``load``／``load_snapshot``）原先把取值直接赋给
        ``self.embeddings``，由 ``_rebuild_indices`` 事后读 ``len()`` 与
        ``shape[1]``——于是形状错误抛的是 ``TypeError: len() of unsized object``
        （0 维）、``IndexError: tuple index out of range``（1 维）或
        ``AttributeError: 'list' object has no attribute 'shape'``（legacy pkl 里
        是普通列表）。三者都不是 ``ValueError``，穿透调用方的错误分类；更要紧的
        是校验发生在赋值**之后**，因此失败会留下半更新的 store——而两个入口的
        契约都是「失败必须让 store 保持原状」。

        写入侧 ``upsert_embeddings`` 早已有同一断言（``ndim != 2``），这里补齐
        读取侧，判据与措辞对齐。

        ``None`` 是合法取值：表示「这份 legacy／快照没有向量」，后续
        ``_rebuild_indices`` 会把 ``faiss_index`` 置为 ``None``。
        """
        if embeddings is None:
            return
        ndim = getattr(embeddings, "ndim", None)
        if ndim != 2:
            raise ValueError(
                f"{source}的 embeddings 必须是二维数组"
                f"（得到 {type(embeddings).__name__}，ndim={ndim!r}）。"
                " 请重建知识快照或修正数据来源。"
            )

    @staticmethod
    def _validate_document_shape(document: Any, position: int, *, source: str) -> None:
        """确认条目符合 ``Document`` 的契约，``source`` 用于说明它来自哪里。

        ``Document`` 的字段是 ``content: str`` + ``metadata: dict``
        （见 ``src/models/document.py``），而载入路径原先把条目直接当成 ``Document``
        用，只做 ``metadata or {}`` 的兜底。实测 ``page_content=None`` 的快照能通过
        ``validate_snapshot_dir``、加载成功，直到 ``display_results`` 里
        ``.replace()`` 才抛 ``AttributeError: 'NoneType' object has no attribute
        'replace'``——报错现场离真正的原因（快照里这一条不是字符串）很远。
        ``metadata`` 非 dict 与条目本身不是 dict 同理。

        两个调用点的契约相同、处置不同，因此消息里的处置指引必须分开：
        ``load_snapshot`` 遇到的是**盘上已有的**坏数据，用户能做的是重建快照；
        ``upsert_embeddings`` 遇到的是**正在写入**的坏数据，用户能做的是修正来源
        并重试，让他「重建快照」会指向一个还没写坏的旧快照。原先写死「请重建
        快照」的措辞只在载入侧成立。
        """
        if not isinstance(document, dict):
            raise ValueError(
                f"{source}：第 {position} 条文档不是映射"
                f"（{type(document).__name__}），无法作为文档使用。"
            )
        # ``page_content`` 在 ``Document`` 里是必填的 ``str``（没有默认值），
        # 因此 ``None`` 与其它非字符串同样是违规——消费者里
        # ``display_results`` 调 ``.replace()``、``_display_retrieved_docs`` 切片，
        # 都会在 None 上崩。键**缺失**则不同：各消费者都写了默认值
        # （``.get("page_content", "")``），因此缺失照旧放行。
        content = document.get("page_content", "")
        if not isinstance(content, str):
            raise ValueError(
                f"{source}：第 {position} 条文档的 page_content 不是字符串"
                f"（{type(content).__name__}）。"
            )
        metadata = document.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(
                f"{source}：第 {position} 条文档的 metadata 不是映射（{type(metadata).__name__}）。"
            )

    @staticmethod
    def _normalize_documents(
        documents: Any, parent_documents: Any
    ) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        """校验并归一化载入的条目，返回新值，**不改动 ``self``**。

        盘上的父条目也要过一遍 ``_normalize_parent_document``，与
        ``register_parent_documents`` 同一条契约。

        快照校验器**显式放行** ``str`` 形态的父条目（它的注释写着「要么是 dict
        （取 ``content``），要么是 str（整体即内容）——``FaissStore._normalize_parent_document``
        的取值契约」），但载入路径原样赋值、只补不归一化，于是那个契约在载入
        侧从未被履行：``resolve_parent_content`` 拿 str 去 ``.get("content")``。

        实测后果：``parents.pkl`` 为 ``{"p1": "父块正文"}`` 时，
        ``validate_snapshot_dir``、``load_manifest``、``load_snapshot`` **全部**
        通过（快照被激活并宣布加载成功），而聊天与召回测试的每一次检索都在
        ``_promote_parent_context`` 抛
        ``AttributeError: 'str' object has no attribute 'get'``。这条形态不是
        假想输入：legacy 导入分支 ``store.load()`` → ``save_snapshot()`` 会把
        第三方 pkl 里的 str 值原样写进新快照。

        **不改动 ``self`` 是刻意的**：调用方在全部校验通过后才一次性赋值，失败
        时 store 保持原状。原先本方法是就地改 ``self.documents``／
        ``self.parent_documents`` 的，而两个调用方又已经先把新值赋了进去——中途
        失败就留下半更新的 store（见 ``load`` 的说明）。
        """
        # 顶层形态同样要有守卫。``SnapshotRepository.validate_snapshot_dir`` 已经
        # 覆盖这些形态，但 ``load``／``load_snapshot`` 是 ``VectorStoreBase`` 的
        # 契约方法，直接调它们的嵌入方不经过那道校验（与
        # ``_validate_document_shape`` 同一取舍）。实测后果都是与「快照不自洽」
        # 无关的裸异常：
        #   分块列表是 int／None -> ``TypeError: 'int' object is not iterable``
        #   分块列表是 dict／str -> 逐字符/逐键当成文档，报「第 0 条文档不是映射」
        #     （误导：真正的问题是顶层形态）
        #   父块索引是 int／None／list -> ``AttributeError: 'int' object has no
        #     attribute 'items'``
        # 判定与 ``validate_snapshot_dir`` 的同名守卫一致：拒绝 str／bytes／映射，
        # 并要求有 ``__len__``。``len("ab") == 2`` 这类会静默通过的形态正是靠
        # 类型判断拦下的。
        if isinstance(documents, (str, bytes, Mapping)) or not hasattr(documents, "__len__"):
            raise ValueError(
                "知识快照不自洽：分块列表的顶层不是序列"
                f"（{type(documents).__name__}），无法作为文档列表使用。请重建快照。"
            )
        if not isinstance(parent_documents, Mapping):
            raise ValueError(
                "知识快照不自洽：父块索引的顶层不是映射"
                f"（{type(parent_documents).__name__}），无法作为父块索引使用。请重建快照。"
            )

        normalized_parent_documents = {
            _parent_key(parent_id): FaissStore._normalize_parent_document(parent_document)
            for parent_id, parent_document in parent_documents.items()
        }
        normalized_documents: list[Any] = []
        for position, document in enumerate(documents):
            FaissStore._validate_document_shape(
                document,
                position,
                source="知识快照不自洽，请重建快照",
            )
            # 浅拷贝而不是原地改：``metadata.pop("parent_content", ...)`` 会写进
            # 调用方持有的那个字典，而「任何一条校验失败都不留痕」要求连输入也
            # 不被改写。归一化后的内容完全一致，只是换了对象。
            metadata = dict(document.get("metadata") or {})
            parent_id = metadata.get("parent_id")
            parent_content = metadata.pop("parent_content", None)
            if (
                parent_id
                and parent_content
                and _parent_key(parent_id) not in normalized_parent_documents
            ):
                normalized_parent_documents[_parent_key(parent_id)] = {
                    "content": parent_content,
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key not in {"chunk_id", "doc_id", "parent_id", "parent_chunk_index"}
                    },
                }
            normalized_documents.append({**document, "metadata": metadata})
        return normalized_documents, normalized_parent_documents

    def register_parent_documents(self, parent_documents: dict[str, dict[str, Any]]):
        for parent_id, parent_document in parent_documents.items():
            normalized = self._normalize_parent_document(parent_document)
            if normalized["content"]:
                self.parent_documents[_parent_key(parent_id)] = normalized

    def resolve_parent_content(self, parent_id: str | None) -> str | None:
        if not parent_id:
            return None
        parent_document = self.parent_documents.get(_parent_key(parent_id))
        if not parent_document:
            return None
        content = parent_document.get("content")
        return content if isinstance(content, str) and content else None

    def upsert_embeddings(self, documents: list[dict[str, Any]], embeddings: Any):
        if not documents:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)
        if new_embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维数组。")
        if len(documents) != len(new_embeddings):
            raise ValueError("documents 与 embeddings 的数量不一致。")

        # 维度检查必须在改动 ``self.documents`` **之前**：原先先 extend 再判维度，
        # 维度不一致时抛异常但 documents 已经多出一批没有对应向量的条目。实测
        # 拒绝之后 documents=2 而 embeddings=(1,4)，store 已不自洽，而
        # ``save_snapshot`` 会照常把这份不自洽落盘（之后的语义检索会拿
        # ``indices`` 去索引 ``documents``，越界或错配）。失败必须不留痕。
        if self.embeddings is not None and self.embeddings.shape[1] != new_embeddings.shape[1]:
            raise ValueError("新增向量维度与现有索引不一致。")

        # 条目形态同样必须在改动 ``self.documents`` **之前**确认，理由与上面那条
        # 维度检查一致：写入侧原先只校验 embeddings，文档形态则放任 ``extend``
        # 之后由 ``_rebuild_indices`` 在派生索引时撞出来。实测一条非映射条目
        # （第三方 store 的返回值、手工构造的 pkl）会让
        # ``_build_index_text`` 抛 ``AttributeError: 'str' object has no attribute
        # 'get'``——异常看似「拒绝写入」，实际上 documents 已经多出这条孤儿，
        # ``save_snapshot`` 会把它写进快照，而 store 在此之后**再也无法接受任何
        # 写入**。错误类型也从「快照不自洽」退化成裸 ``AttributeError``。
        #
        # 判据与载入边界 ``_validate_document_shape`` 完全一致（同一份
        # ``Document`` 契约），因此复用同一份实现；只是这里的下标从 0 起算，
        # 而载入路径的 position 用于描述「第几条」，两者语义相同。
        for position, document in enumerate(documents):
            self._validate_document_shape(
                document,
                position,
                source="待写入的文档不符合 Document 契约，已拒绝写入",
            )

        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        self._rebuild_indices()

    def add_documents(self, documents: list[dict[str, Any]]):
        raise RuntimeError(
            "FaissStore.add_documents 已废弃，请使用外部 EmbeddingService 后调用 upsert_embeddings。"
        )

    async def aadd_documents(self, documents: list[dict[str, Any]]):
        raise RuntimeError("FaissStore.aadd_documents 已废弃，请使用 KnowledgeBuildService。")

    def semantic_search(self, query_embedding: Any, top_k: int = 5) -> list[dict[str, Any]]:
        if self.faiss_index is None or self.embeddings is None:
            return []

        # 校验放在调 ``faiss_index.search()`` **之前**：FAISS 对非法 k 的收场
        # 不可用（详见 ``_require_top_k``）。
        _require_top_k(top_k)

        # 转换失败要收敛成 ``ValueError``。``np.asarray`` 的失败分三类，只有一类
        # 是本项目契约里的类型：
        #   ``OverflowError``——``np.asarray(10**4301, dtype=np.float32)``，超双精度
        #     范围的整数（容器里混入这类元素同样）。它是 ``ArithmeticError`` 子类
        #     而**不是** ``ValueError``，穿透调用方全部 ``except ValueError`` 分类。
        #   ``TypeError``——``object()``／集合／生成器，消息是 ``float() argument
        #     must be a string or a real number``。
        #   ``ValueError``——``"abc"`` 这类字符串，消息是 numpy 的
        #     ``could not convert string to float``，与本项目的措辞不同。
        # 本函数已按同一判据修过 ``_require_top_k``（那里拦住的是 FAISS 的裸
        # ``TypeError``／``AssertionError``），查询向量这条漏了：同一个「参数形态
        # 非法」的处置，两处收场不同。
        #
        # 不逐元素预检：``np.zeros(4, dtype=np.float32)`` 是 ``embed_query`` 的
        # 真实输出，逐元素判有限性会把合法的 numpy 标量一并拒掉（实测
        # ``to_finite_float(np.float32(0.1))`` 返回 ``None``——它只认 Python
        # 的 int／float）。非有限值不在这里拦：``ndim`` 守卫之后 FAISS 自己处理。
        try:
            query_vector = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "查询向量必须是一维数值序列，但得到 "
                f"{type(query_embedding).__name__}（{type(exc).__name__}: {exc}）。"
                " 请检查嵌入模型的输出。"
            ) from exc
        if query_vector.ndim != 1 or query_vector.size == 0:
            raise ValueError("查询向量必须是一维且非空。")
        expected_dimension = int(self.embeddings.shape[1])
        if query_vector.shape[0] != expected_dimension:
            raise ValueError(
                f"查询向量维度 {query_vector.shape[0]} 与索引维度 {expected_dimension} 不一致；"
                "请重建知识快照或切换回构建该快照的嵌入模型。"
            )
        query_vector = query_vector.reshape(1, expected_dimension)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        similarities = 1.0 / (1.0 + distances[0])

        results: list[dict[str, Any]] = []
        for index, doc_index in enumerate(indices[0]):
            if doc_index == -1:
                continue
            document = deepcopy(self.documents[doc_index])
            document["score"] = float(similarities[index])
            results.append(document)
        return results

    def keyword_search(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.bm25_index is None:
            return []

        # 与 ``semantic_search`` 同一判据（详见 ``_require_top_k``）。这里尤其
        # 重要：下面的收集循环是「先 append 再判长度」，``top_k=0`` 会返回 1 条。
        _require_top_k(top_k)

        # ``jieba.cut`` 对非字符串抛 ``AttributeError: 'int' object has no
        # attribute 'decode'``（它按字节处理输入）——不是本项目的 ``ValueError``，
        # 消息也指向 jieba 内部。查询文本只能来自用户输入或调用方，形态必须显式
        # 校验，与 ``_require_top_k``／查询向量同一判据。
        if not isinstance(query_text, str):
            raise ValueError(f"关键词检索的查询文本必须是字符串，但得到 {safe_repr(query_text)}。")
        tokenized_query = list(jieba.cut(query_text))
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1]

        # 分数可能整片为 0，而不是「最高分是 0」。rank_bm25 的 idf 是
        # ``log(N - n + 0.5) - log(n + 0.5)``，且只对 ``idf < 0`` 做 epsilon
        # 浮动——当某个词**恰好**出现在一半文档里（``n == N/2``）时 idf 恰为
        # 0，不属于「负」因此不被浮动，该词在所有文档上的分数就全是 0。
        # 查询词根本不在语料里时同样全 0。
        #
        # 此时旧实现（``if score <= 0: break``）静默返回空列表，且区分不了
        # 下面两种完全不同的情况：
        #   (a) 该词无判别力 / 语料里没有这个词 —— 空结果本身说得通；
        #   (b) 排名里混着正分文档，只是 0 分文档排在前面把循环提前 break ——
        #       这是丢结果。
        # 因此先看全局最高分：最高分 <= 0 直接返回空（并在 debug 里说明原因），
        # 否则按分数降序收集，遇 0 分只跳过该条、不终止。
        if doc_scores.size == 0 or float(doc_scores[top_indices[0]]) <= 0:
            logger.debug(
                "关键词检索无有效分数（查询词可能不在语料中，或恰好出现在一半文档里"
                " 导致 BM25 idf 为 0），返回空结果。"
            )
            return []

        results: list[dict[str, Any]] = []
        for index in top_indices:
            score = float(doc_scores[index])
            if score <= 0:
                continue
            document = deepcopy(self.documents[index])
            document["score"] = score
            results.append(document)
            if len(results) >= top_k:
                break
        return results

    def search(
        self, query: str, top_k: int = 5, search_type: str = "semantic"
    ) -> list[dict[str, Any]]:
        raise RuntimeError(
            "FaissStore.search 已废弃，请通过 RetrievalService 调用语义检索或关键词检索。"
        )

    async def asearch(
        self, query: str, top_k: int = 5, search_type: str = "semantic"
    ) -> list[dict[str, Any]]:
        raise RuntimeError(
            "FaissStore.asearch 已废弃，请通过 RetrievalService 调用语义检索或关键词检索。"
        )

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "parent_documents": self.parent_documents,
                },
                file,
            )

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        trusted_paths: Iterable[str | os.PathLike[str]] | None = None,
    ):
        """从明确受信的 legacy pickle 载入。

        ``trusted_paths`` 必须是调用方声明的信任根（生产路径下即
        ``RunConfig.legacy_pkl_path`` 的所在目录）。为 ``None`` 或空时拒绝加载：
        pickle 会执行任意代码，而「路径看起来像本项目的文件」不构成来源证明。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"向量存储文件未找到: {path}")
        ensure_trusted_source(path, trusted_paths, label="legacy 向量存储文件")
        with open(path, "rb") as file:
            data = pickle.load(file)  # nosec B301 - 上方已校验来源
        # 全部解析、校验到**局部变量**，最后一次性赋给 ``self``。原先先把三个
        # 字段赋进 ``self`` 再调 ``_normalize_loaded_documents``，中途失败就留下
        # 半更新的 store——实测一次被拒的载入之后：``documents`` 换成了新值、
        # ``embeddings`` 是新值，而 ``faiss_index`` 与 ``_tokenized_docs_cache``
        # 还是**旧**的（索引条数 2 对应 1 条文档），随后 ``semantic_search`` 抛
        # ``TypeError: 'str' object does not support item assignment``——报错现场
        # 与真正的原因（载入被拒）毫无关联。失败必须让 store 保持原状。
        if not isinstance(data, Mapping):
            raise ValueError(
                "legacy 向量存储文件不自洽：顶层不是映射"
                f"（{type(data).__name__}），无法取出 documents／embeddings。"
            )
        documents, parent_documents = self._normalize_documents(
            data.get("documents", []), data.get("parent_documents", {})
        )
        embeddings = data.get("embeddings")
        # 形状必须在赋值**之前**判：下面的 ``self.documents = documents`` 之后
        # ``_rebuild_indices`` 才读形状，那时失败已经留下半更新的 store。
        self._validate_embeddings_shape(embeddings, source="legacy 向量存储文件")

        self.documents = documents
        self.parent_documents = parent_documents
        self.embeddings = embeddings
        self._rebuild_indices()

    def save_snapshot(self, snapshot_dir: str):
        if self.faiss_index is None or self.embeddings is None:
            raise RuntimeError("当前语义索引为空，无法保存快照。")
        snapshot_path = Path(snapshot_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        with (snapshot_path / "chunks.pkl").open("wb") as file:
            pickle.dump(self.documents, file)
        with (snapshot_path / "parents.pkl").open("wb") as file:
            pickle.dump(self.parent_documents, file)
        with (snapshot_path / "lexical.index").open("wb") as file:
            pickle.dump(self._tokenized_docs_cache, file)
        np.save(snapshot_path / "embeddings.npy", self.embeddings)
        faiss.write_index(self.faiss_index, str(snapshot_path / "semantic.index"))
        # ``.get("metadata", {})`` 的默认值只在**键缺失**时生效；键存在而值为
        # ``None`` 时它返回 ``None``，随后的 ``.get`` 抛
        # ``AttributeError: 'NoneType' object has no attribute 'get'``。同文件另外
        # 三处（``_build_index_text``、``_normalize_loaded_documents``、
        # ``_normalize_parent_document``）用的都是 ``or {}``，这里与它们对齐。
        # 实测 ``upsert_embeddings`` 接受 ``metadata=None`` 的条目并正常建好索引
        # （``_rebuild_indices`` 里走的是 ``or {}``），到 ``save_snapshot`` 才崩——
        # 同一份数据两条路径的判据不一致。
        stats = {
            # 集合推导会对取值取哈希，而 ``source`` 来自磁盘上的快照／legacy
            # pkl，形态不受控：``source=["a"]`` 或 ``{"a": 1}`` 抛
            # ``TypeError: unhashable type``，而这条路径正是把文档写进快照的那
            # 一步——异常类型与「快照写入失败」毫无关联。``document_source``
            # 对非字符串安全返回 ``"N/A"``，与 ``factory.py`` 用的是同一个。
            "document_count": len({document_source(doc) for doc in self.documents}),
            "chunk_count": len(self.documents),
            "parent_count": len(self.parent_documents),
            "embedding_dimension": int(self.embeddings.shape[1])
            if self.embeddings is not None
            else 0,
        }
        (snapshot_path / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_snapshot(
        self,
        snapshot_dir: str | os.PathLike[str],
        *,
        snapshot_root: str | os.PathLike[str] | None = None,
    ):
        """从**本应用管理的活动快照**载入。

        快照目录里的三个文件都是 pickle，反序列化即执行代码。这里不信任传进来
        的路径本身，而是要求调用方给出 ``snapshot_root``（生产路径下即
        ``RunConfig.snapshot_root``），然后校验：

        1. 目录解析后位于 ``snapshot_root`` 之内（符号链接逃逸会被拦下）；
        2. ``ACTIVE_SNAPSHOT`` 标记存在，且它指向的目录就是 ``snapshot_dir``。

        第 2 条是关键：仅「在 root 之下」还不足以证明目录可用——未激活或半成品
        目录同样在 root 之下。保留 pickle 扩展名是为了兼容 Dify 衍生的既有快照。
        """
        if not snapshot_root:
            raise ValueError(
                "缺少 snapshot_root，已拒绝加载快照。"
                " 快照目录内的文件是 pickle，必须由调用方声明信任根（"
                "RunConfig.snapshot_root 或 SnapshotRepository.root）。"
            )
        snapshot_path = resolve_within(
            Path(snapshot_dir), Path(snapshot_root), label="知识快照目录"
        )
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"知识快照目录不存在: {snapshot_path}")
        marker = snapshot_path.parent / "ACTIVE_SNAPSHOT"
        if not marker.exists():
            raise ValueError(f"知识快照目录未被激活，缺少标记文件: {marker}")
        active_id = marker.read_text(encoding="utf-8").strip()
        active_dir = (marker.parent / active_id).resolve()
        if active_dir != snapshot_path:
            raise ValueError(
                "知识快照目录不是当前活动快照。"
                f" 传入={snapshot_path}，ACTIVE_SNAPSHOT 指向={active_dir}。"
            )
        # 与 ``load()`` 同一取舍：全部解析、校验到**局部变量**，最后一次性赋给
        # ``self``。原先每读一个文件就直接写 ``self``，中途任何一步失败都留下
        # 半更新的 store——实测一次被拒的 ``load_snapshot`` 之后 ``documents``
        # 是新值而 ``faiss_index`` 还是旧的，后续检索在错配的下标上崩出与
        # 「快照坏了」无关的异常。失败必须让 store 保持原状。
        with (snapshot_path / "chunks.pkl").open("rb") as file:
            raw_documents = pickle.load(file)  # nosec B301
        with (snapshot_path / "parents.pkl").open("rb") as file:
            raw_parent_documents = pickle.load(file)  # nosec B301
        embeddings_path = snapshot_path / "embeddings.npy"
        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        self._validate_embeddings_shape(embeddings, source="知识快照")
        faiss_index_path = snapshot_path / "semantic.index"
        faiss_index = faiss.read_index(str(faiss_index_path)) if faiss_index_path.exists() else None
        # 顺序与 ``load()`` 一致：先归一化文档，再建索引。此前这里是反的，
        # 于是「``lexical.index`` 是否存在」决定了用户看到哪条错误——实测同一
        # 份畸形 chunks（``page_content`` 是 ``int``）：
        #   有 lexical.index → ``ValueError: 知识快照不自洽：第 0 条文档的
        #     page_content 不是字符串（int）。请重建快照。``
        #   缺 lexical.index → ``AttributeError: 'int' object has no attribute
        #     'decode'``（``_rebuild_indices`` 先跑，在 ``_build_index_text`` 里崩）
        # 而 ``validate_snapshot_dir`` 的必需文件清单里没有 ``lexical.index``，
        # 缺它是合法状态（下面的 ``else`` 分支就是为它准备的）。``load()`` 早已
        # 是正确的顺序，两处对齐。
        documents, parent_documents = self._normalize_documents(raw_documents, raw_parent_documents)
        lexical_path = snapshot_path / "lexical.index"
        if lexical_path.exists():
            with lexical_path.open("rb") as file:
                tokenized_docs = pickle.load(file)  # nosec B301
            # 盘上的 ``lexical.index`` 与 ``self.documents`` 必须等长，否则
            # ``keyword_search`` 拿 ``argsort`` 的下标去索引 ``self.documents``
            # 会越界或错配（实测语料 3 条、文档 2 条 -> ``IndexError: list index
            # out of range``；语料 1 条、文档 2 条 -> 静默丢结果）。
            #
            # 生产路径**已经**由 ``SnapshotRepository.validate_snapshot_dir`` 挡住
            # 这些形态，本方法是库级公开 API（``VectorStoreBase`` 的契约方法），
            # 直接调它的嵌入方不会经过那道校验——``_validate_document_shape`` 的
            # 注释已说明同一取舍。这里只断言「顶层是序列且长度对齐」这条最小
            # 不变量，不逐条校验语料内容：条目本身是什么形态不影响下标对应关系。
            #
            # 判定必须在 ``_build_bm25_index`` **之前**，否则非序列形态会先在
            # ``any()`` 上抛裸 ``TypeError: 'int' object is not iterable``——与
            # 「快照不自洽」无关的类型，也没有重建指引。``None`` 同样要拦：
            # ``_build_bm25_index(None)`` 返回 ``None``，``keyword_search`` 于是
            # 对一份有 2 条文档的快照静默返回空结果，用户看到的是「搜不到」而不是
            # 「快照坏了」。
            if isinstance(tokenized_docs, (str, bytes, dict)) or (
                tokenized_docs is not None and not hasattr(tokenized_docs, "__len__")
            ):
                raise ValueError(
                    "知识快照不自洽：lexical.index 的顶层不是列表"
                    f"（{type(tokenized_docs).__name__}），无法作为关键词索引语料使用。请重建快照。"
                )
            # ``len()`` 必须排在上面的形态守卫**之后**：``int`` 这类没有
            # ``__len__`` 的形态会先在这里抛裸 ``TypeError: object of type 'int'
            # has no len()``，用户拿到的又是与快照无关的错误类型。
            tokenized_rows = 0 if tokenized_docs is None else len(tokenized_docs)
            if tokenized_rows != len(documents):
                raise ValueError(
                    "知识快照不自洽：关键词索引语料条数与分块数不一致。"
                    f" lexical.index={tokenized_rows}，"
                    f" chunks.pkl={len(documents)}。"
                    " 快照可能在写入或复制过程中被截断，请重建。"
                )
            # **条目**形态也要拦：上面两道守卫只覆盖顶层（是不是列表）与条数，
            # 而 ``BM25Okapi._initialize`` 会对每条语料调 ``len(document)``——
            # 实测语料 ``[7]``、``[10**4301]``、``[1.5]``、``[None]`` 全部穿过
            # 顶层守卫与条数校验，在 ``_build_bm25_index`` 里抛裸
            # ``TypeError: object of type 'int' has no len()``。这条路径在生产上
            # 可达：``SnapshotRepository.validate_snapshot_dir`` 同样只看顶层与
            # 条数，于是坏快照能被 finalize 成**活动快照**，此后每次启动都在
            # 载入期以与「快照坏了」无关的异常类型失败。
            #
            # 只判「有没有 ``__len__``」而不判「是不是 ``list[str]``」：语料是
            # 本应用写出的分词结果，形态确定；这里要挡的是「根本不是序列」，
            # 过严会把将来可能的迭代器形态一并拒掉。
            for position, entry in enumerate(tokenized_docs if tokenized_docs is not None else []):
                if isinstance(entry, (str, bytes)) or not hasattr(entry, "__len__"):
                    raise ValueError(
                        "知识快照不自洽：lexical.index 的第 "
                        f"{position} 条语料不是序列（{type(entry).__name__}），"
                        "无法作为关键词索引语料使用。请重建快照。"
                    )
            self._tokenized_docs_cache = tokenized_docs if tokenized_docs is not None else []
            self.bm25_index = self._build_bm25_index(self._tokenized_docs_cache)
        # 全部校验通过，到这里才改动 store。``_rebuild_indices`` 走 ``else`` 分支
        # 时会读 ``self.documents``，因此必须排在这些赋值之后。
        self.documents = documents
        self.parent_documents = parent_documents
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        if not lexical_path.exists():
            self._rebuild_indices()

    def import_legacy_snapshot(
        self,
        legacy_path: str,
        *,
        trusted_paths: Iterable[str | os.PathLike[str]] | None = None,
    ):
        """把旧版 pkl 读入内存（不落快照）。与 ``load`` 共用同一来源校验。"""
        start_time = time.perf_counter()
        self.load(legacy_path, trusted_paths=trusted_paths)
        logger.info("旧版 pkl 已导入内存，耗时: %.4fs", time.perf_counter() - start_time)

    def get_embedding_model(self) -> Any:
        raise RuntimeError("FaissStore 不再直接持有嵌入模型，请使用 EmbeddingService。")
