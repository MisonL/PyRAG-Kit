# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import time
import numpy as np
import jieba
import copy
import pickle
import os
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

# 尝试导入 faiss，如果失败则提供提示
try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Please install it with `pip install faiss-cpu` "
        "or `pip install faiss-gpu` for GPU support."
    )

from .base import VectorStoreBase
from ...utils.config import get_settings, RetrievalMethod
from ...providers.factory import ModelProviderFactory
from ...utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class FaissStore(VectorStoreBase):
    """
    基于 FAISS 和 BM25 的向量存储实现。
    已注入 CSE 性能传感器。
    """
    def __init__(self, file_path: Optional[str] = None):
        import asyncio
        self.file_path = file_path
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.parent_documents: Dict[str, Dict[str, Any]] = {}
        self._tokenized_docs_cache: List[List[str]] = [] # 缓存分词结果以优化 BM25
        self.lock = asyncio.Lock() # 并发锁，确保写入安全
        
        # ... (rest of init same)
        try:
            current_settings = get_settings()
            active_embedding_key = current_settings.default_embedding_provider
            self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
        except Exception as e:
            logger.error(f"FaissStore 初始化获取模型失败: {e}")
            self.embedding_model = None

        if self.file_path and os.path.exists(self.file_path):
            self.load(self.file_path)

    def _update_indices(self, new_documents: List[Dict], new_embeddings: np.ndarray):
        """增量更新索引逻辑。"""
        # 1. 更新 BM25 缓存与索引
        new_tokens = [list(jieba.cut(doc.get("page_content", ""))) for doc in new_documents]
        self._tokenized_docs_cache.extend(new_tokens)
        self.bm25_index = BM25Okapi(self._tokenized_docs_cache)
        
        # 2. 更新 FAISS 索引
        if self.faiss_index is None:
            dimension = new_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
        
        self.faiss_index.add(new_embeddings)
        logger.debug(f"索引增量更新完成，当前总量: {len(self.documents)}")

    @staticmethod
    def _normalize_parent_document(parent_document: Any) -> Dict[str, Any]:
        if isinstance(parent_document, dict):
            parent_content = parent_document.get("content", "")
            parent_metadata = parent_document.get("metadata") or {}
            return {
                "content": parent_content,
                "metadata": parent_metadata,
            }
        if isinstance(parent_document, str):
            return {"content": parent_document, "metadata": {}}
        return {"content": "", "metadata": {}}

    def register_parent_documents(self, parent_documents: Dict[str, Dict[str, Any]]):
        """注册父分段侧车数据。"""
        if not parent_documents:
            return

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

    def add_documents(self, documents: List[Dict[str, Any]]):
        """同步添加文档。"""
        if not documents: return
        
        logger.info(f"正在同步添加 {len(documents)} 个文档到 FaissStore。")
        start_total = time.perf_counter()

        new_texts = [doc.get("page_content", "") for doc in documents]
        model = self.get_embedding_model()
        new_embeddings = np.array(model.embed_documents(texts=new_texts), dtype=np.float32)

        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))

        self._update_indices(documents, new_embeddings)
        
        logger.info(f"FaissStore 同步添加完成，耗时: {time.perf_counter() - start_total:.4f}s")

    async def aadd_documents(self, documents: List[Dict[str, Any]]):
        """异步添加文档 (并发安全/性能优化)。"""
        if not documents: return
        
        async with self.lock: # 关键：确保原子性
            logger.info(f"正在异步添加 {len(documents)} 个文档 (已加锁)。")
            start_total = time.perf_counter()

            new_texts = [doc.get("page_content", "") for doc in documents]
            model = self.get_embedding_model()
            embeddings_list = await model.aembed_documents(texts=new_texts)
            new_embeddings = np.array(embeddings_list, dtype=np.float32)

            self.documents.extend(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack((self.embeddings, new_embeddings))

            # 索引更新放入线程池，防止阻塞事件循环
            import asyncio
            await asyncio.to_thread(self._update_indices, documents, new_embeddings)
            
            logger.info(f"异步添加完成，耗时: {time.perf_counter() - start_total:.4f}s")

    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """同步搜索 (CSE Sensor)。"""
        start_time = time.perf_counter()
        logger.info(f"FaissStore 同步搜索: type={search_type}, query='{query[:20]}...'")
        
        try:
            results = []
            if search_type == "semantic":
                if self.embeddings is None or self.faiss_index is None:
                    return []
                
                model = self.get_embedding_model()
                query_embedding = np.array(model.embed_documents([query])[0], dtype=np.float32).reshape(1, -1)
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                similarities = 1.0 / (1.0 + distances[0])

                for i, idx in enumerate(indices[0]):
                    if idx == -1: continue
                    doc = copy.deepcopy(self.documents[idx])
                    doc["score"] = float(similarities[i])
                    results.append(doc)
            
            elif search_type == "keyword":
                if not self.bm25_index:
                    return []
                
                tokenized_query = list(jieba.cut(query))
                doc_scores = self.bm25_index.get_scores(tokenized_query)
                top_indices = np.argsort(doc_scores)[::-1]
                
                for i in top_indices:
                    if doc_scores[i] > 0:
                        doc = copy.deepcopy(self.documents[i])
                        doc["score"] = float(doc_scores[i])
                        results.append(doc)
                    else:
                        break
                results = results[:top_k]
            
            logger.info(f"FaissStore 同步搜索完成，耗时: {time.perf_counter() - start_time:.4f}s")
            return results
        except Exception as e:
            logger.error(f"FaissStore 同步搜索出错: {e}", exc_info=True)
            return []

    async def asearch(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """异步搜索 (CSE Sensor)。"""
        import asyncio
        start_time = time.perf_counter()
        logger.info(f"FaissStore 异步搜索: type={search_type}, query='{query[:20]}...'")
        
        try:
            results = []
            if search_type == "semantic":
                if self.embeddings is None or self.faiss_index is None:
                    return []
                
                model = self.get_embedding_model()
                # 异步获取查询嵌入
                emb_list = await model.aembed_documents([query])
                query_embedding = np.array(emb_list[0], dtype=np.float32).reshape(1, -1)
                
                # FAISS 搜索是 CPU 密集型，放入线程池
                distances, indices = await asyncio.to_thread(self.faiss_index.search, query_embedding, top_k)
                similarities = 1.0 / (1.0 + distances[0])

                for i, idx in enumerate(indices[0]):
                    if idx == -1: continue
                    doc = copy.deepcopy(self.documents[idx])
                    doc["score"] = float(similarities[i])
                    results.append(doc)
            
            elif search_type == "keyword":
                if not self.bm25_index:
                    return []
                
                # 分词和评分是 CPU 密集型
                tokenized_query = await asyncio.to_thread(list, jieba.cut(query))
                doc_scores = await asyncio.to_thread(self.bm25_index.get_scores, tokenized_query)
                top_indices = np.argsort(doc_scores)[::-1]
                
                for i in top_indices:
                    if doc_scores[i] > 0:
                        doc = copy.deepcopy(self.documents[i])
                        doc["score"] = float(doc_scores[i])
                        results.append(doc)
                    else:
                        break
                results = results[:top_k]
            
            logger.info(f"FaissStore 异步搜索完成，耗时: {time.perf_counter() - start_time:.4f}s")
            return results
        except Exception as e:
            logger.error(f"FaissStore 异步搜索出错: {e}", exc_info=True)
            return []

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "parent_documents": self.parent_documents,
                },
                f,
            )

    def _normalize_loaded_documents(self):
        normalized_parent_documents: Dict[str, Dict[str, Any]] = dict(self.parent_documents)

        for document in self.documents:
            metadata = document.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            parent_content = metadata.pop("parent_content", None)
            if parent_id and parent_content and parent_id not in normalized_parent_documents:
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

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"向量存储文件未找到: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.embeddings = data.get("embeddings", None)
            self.parent_documents = data.get("parent_documents", {})

        self._normalize_loaded_documents()
        
        # 重新初始化索引和缓存
        if self.documents:
            self._tokenized_docs_cache = [list(jieba.cut(doc.get("page_content", ""))) for doc in self.documents]
            self.bm25_index = BM25Okapi(self._tokenized_docs_cache)
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings)

    def get_embedding_model(self) -> Any:
        if self.embedding_model is None:
            current_settings = get_settings()
            active_embedding_key = current_settings.default_embedding_provider
            self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
        return self.embedding_model
