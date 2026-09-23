# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import json
import os

# Pickle is retained only for trusted local legacy snapshots.
import pickle  # nosec B403
import time
from collections.abc import Iterable
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

from ...utils.log_manager import get_module_logger
from ...utils.security import ensure_trusted_source, resolve_within
from .base import VectorStoreBase

logger = get_module_logger(__name__)
jieba.setLogLevel(jieba.logging.ERROR)


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
        if not tokenized_docs:
            return None
        return BM25Okapi(tokenized_docs)

    def _rebuild_indices(self) -> None:
        if self.documents:
            self._tokenized_docs_cache = [
                list(jieba.cut(self._build_index_text(doc))) for doc in self.documents
            ]
            self.bm25_index = self._build_bm25_index(self._tokenized_docs_cache)
        else:
            self._tokenized_docs_cache = []
            self.bm25_index = None

        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = int(self.embeddings.shape[1])
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings.astype(np.float32))
        else:
            self.faiss_index = None

    def _normalize_loaded_documents(self) -> None:
        normalized_parent_documents = dict(self.parent_documents)
        for document in self.documents:
            metadata = document.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            parent_content = metadata.pop("parent_content", None)
            if parent_id and parent_content and str(parent_id) not in normalized_parent_documents:
                normalized_parent_documents[str(parent_id)] = {
                    "content": parent_content,
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key not in {"chunk_id", "doc_id", "parent_id", "parent_chunk_index"}
                    },
                }
            document["metadata"] = metadata
        self.parent_documents = normalized_parent_documents

    def register_parent_documents(self, parent_documents: dict[str, dict[str, Any]]):
        for parent_id, parent_document in parent_documents.items():
            normalized = self._normalize_parent_document(parent_document)
            if normalized["content"]:
                self.parent_documents[str(parent_id)] = normalized

    def resolve_parent_content(self, parent_id: str | None) -> str | None:
        if not parent_id:
            return None
        parent_document = self.parent_documents.get(str(parent_id))
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

        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            if self.embeddings.shape[1] != new_embeddings.shape[1]:
                raise ValueError("新增向量维度与现有索引不一致。")
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

        query_vector = np.asarray(query_embedding, dtype=np.float32)
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
        self.documents = data.get("documents", [])
        self.embeddings = data.get("embeddings")
        self.parent_documents = data.get("parent_documents", {})
        self._normalize_loaded_documents()
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
        stats = {
            "document_count": len(
                {doc.get("metadata", {}).get("source") for doc in self.documents}
            ),
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
        with (snapshot_path / "chunks.pkl").open("rb") as file:
            self.documents = pickle.load(file)  # nosec B301
        with (snapshot_path / "parents.pkl").open("rb") as file:
            self.parent_documents = pickle.load(file)  # nosec B301
        embeddings_path = snapshot_path / "embeddings.npy"
        self.embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        faiss_index_path = snapshot_path / "semantic.index"
        self.faiss_index = (
            faiss.read_index(str(faiss_index_path)) if faiss_index_path.exists() else None
        )
        lexical_path = snapshot_path / "lexical.index"
        if lexical_path.exists():
            with lexical_path.open("rb") as file:
                self._tokenized_docs_cache = pickle.load(file)  # nosec B301
            self.bm25_index = self._build_bm25_index(self._tokenized_docs_cache)
        else:
            self._rebuild_indices()
        self._normalize_loaded_documents()

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
