# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "FAISS is not installed. Please install it with `pip install faiss-cpu`."
    ) from exc

from .base import VectorStoreBase
from ...utils.log_manager import get_module_logger

logger = get_module_logger(__name__)
jieba.setLogLevel(jieba.logging.ERROR)


class FaissStore(VectorStoreBase):
    def __init__(self, file_path: Optional[str] = None):
        import asyncio

        self.file_path = file_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.parent_documents: Dict[str, Dict[str, Any]] = {}
        self._tokenized_docs_cache: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.lock = asyncio.Lock()

        if self.file_path and os.path.exists(self.file_path):
            self.load(self.file_path)

    @staticmethod
    def _normalize_parent_document(parent_document: Any) -> Dict[str, Any]:
        if isinstance(parent_document, dict):
            return {
                "content": parent_document.get("content", ""),
                "metadata": parent_document.get("metadata") or {},
            }
        if isinstance(parent_document, str):
            return {"content": parent_document, "metadata": {}}
        return {"content": "", "metadata": {}}

    @staticmethod
    def _build_index_text(document: Dict[str, Any]) -> str:
        metadata = document.get("metadata") or {}
        source = metadata.get("source", "")
        source_hint = ""
        if isinstance(source, str) and source:
            source_hint = Path(source).stem.replace("_", " ").replace("-", " ")

        page_content = document.get("page_content", "") or ""
        if source_hint and source_hint not in page_content:
            return f"{source_hint}\n{page_content}"
        return page_content

    def _rebuild_indices(self) -> None:
        if self.documents:
            self._tokenized_docs_cache = [list(jieba.cut(self._build_index_text(doc))) for doc in self.documents]
            self.bm25_index = BM25Okapi(self._tokenized_docs_cache)
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

    def register_parent_documents(self, parent_documents: Dict[str, Dict[str, Any]]):
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

    def upsert_embeddings(self, documents: List[Dict[str, Any]], embeddings: Any):
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

    def add_documents(self, documents: List[Dict[str, Any]]):
        raise RuntimeError("FaissStore.add_documents 已废弃，请使用外部 EmbeddingService 后调用 upsert_embeddings。")

    async def aadd_documents(self, documents: List[Dict[str, Any]]):
        raise RuntimeError("FaissStore.aadd_documents 已废弃，请使用 KnowledgeBuildService。")

    def semantic_search(self, query_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.faiss_index is None or self.embeddings is None:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        similarities = 1.0 / (1.0 + distances[0])

        results: List[Dict[str, Any]] = []
        for index, doc_index in enumerate(indices[0]):
            if doc_index == -1:
                continue
            document = pickle.loads(pickle.dumps(self.documents[doc_index]))
            document["score"] = float(similarities[index])
            results.append(document)
        return results

    def keyword_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25_index is None:
            return []

        tokenized_query = list(jieba.cut(query_text))
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1]

        results: List[Dict[str, Any]] = []
        for index in top_indices:
            score = float(doc_scores[index])
            if score <= 0:
                break
            document = pickle.loads(pickle.dumps(self.documents[index]))
            document["score"] = score
            results.append(document)
            if len(results) >= top_k:
                break
        return results

    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        raise RuntimeError("FaissStore.search 已废弃，请通过 RetrievalService 调用语义检索或关键词检索。")

    async def asearch(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        raise RuntimeError("FaissStore.asearch 已废弃，请通过 RetrievalService 调用语义检索或关键词检索。")

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

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"向量存储文件未找到: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        self.documents = data.get("documents", [])
        self.embeddings = data.get("embeddings")
        self.parent_documents = data.get("parent_documents", {})
        self._normalize_loaded_documents()
        self._rebuild_indices()

    def save_snapshot(self, snapshot_dir: str):
        snapshot_path = Path(snapshot_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        with (snapshot_path / "chunks.pkl").open("wb") as file:
            pickle.dump(self.documents, file)
        with (snapshot_path / "parents.pkl").open("wb") as file:
            pickle.dump(self.parent_documents, file)
        with (snapshot_path / "lexical.index").open("wb") as file:
            pickle.dump(self._tokenized_docs_cache, file)
        np.save(snapshot_path / "embeddings.npy", self.embeddings)
        if self.faiss_index is None:
            raise RuntimeError("当前语义索引为空，无法保存快照。")
        faiss.write_index(self.faiss_index, str(snapshot_path / "semantic.index"))
        stats = {
            "document_count": len({doc.get("metadata", {}).get("source") for doc in self.documents}),
            "chunk_count": len(self.documents),
            "parent_count": len(self.parent_documents),
            "embedding_dimension": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
        }
        (snapshot_path / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_snapshot(self, snapshot_dir: str):
        snapshot_path = Path(snapshot_dir)
        with (snapshot_path / "chunks.pkl").open("rb") as file:
            self.documents = pickle.load(file)
        with (snapshot_path / "parents.pkl").open("rb") as file:
            self.parent_documents = pickle.load(file)
        embeddings_path = snapshot_path / "embeddings.npy"
        self.embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        faiss_index_path = snapshot_path / "semantic.index"
        self.faiss_index = faiss.read_index(str(faiss_index_path)) if faiss_index_path.exists() else None
        lexical_path = snapshot_path / "lexical.index"
        if lexical_path.exists():
            with lexical_path.open("rb") as file:
                self._tokenized_docs_cache = pickle.load(file)
            self.bm25_index = BM25Okapi(self._tokenized_docs_cache)
        else:
            self._rebuild_indices()
        self._normalize_loaded_documents()

    def import_legacy_snapshot(self, legacy_path: str):
        start_time = time.perf_counter()
        self.load(legacy_path)
        logger.info("旧版 pkl 已导入内存，耗时: %.4fs", time.perf_counter() - start_time)

    def get_embedding_model(self) -> Any:
        raise RuntimeError("FaissStore 不再直接持有嵌入模型，请使用 EmbeddingService。")
