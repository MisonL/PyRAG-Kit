import numpy as np
import jieba
import copy
import pickle
import os
from typing import Any, Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
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
from ...utils.config import settings, RetrievalMethod # 导入 RetrievalMethod
from ...providers.factory import ModelProviderFactory

class FaissStore(VectorStoreBase):
    """
    基于 FAISS 和 BM25 的向量存储实现。
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.embedding_model = None # 在加载或添加文档时设置

    def _initialize_bm25(self):
        """在加载文档或添加新文档后初始化/更新BM25索引。"""
        if self.documents:
            tokenized_corpus = [list(jieba.cut(doc.get("page_content", ""))) for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)

    def _initialize_faiss_index(self):
        """根据当前嵌入初始化FAISS索引。"""
        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            # 确保 self.faiss_index 不为 None 且 self.embeddings 是有效的 numpy 数组
            if self.faiss_index is not None and isinstance(self.embeddings, np.ndarray):
                self.faiss_index.add(self.embeddings) # type: ignore

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向向量存储中添加文档。
        此方法会生成嵌入并更新FAISS和BM25索引。
        
        Args:
            documents: 包含文档内容和元数据的字典列表。
                       每个字典应至少包含 'page_content' 键。
        """
        if not documents:
            return

        # 获取嵌入模型
        if self.embedding_model is None:
            active_embedding_key = settings.default_embedding_provider
            self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)

        # 生成新文档的嵌入
        new_texts = [doc.get("page_content", "") for doc in documents]
        new_embeddings = np.array(self.embedding_model.embed_documents(texts=new_texts), dtype=np.float32)

        # 合并新旧文档和嵌入
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))

        # 更新BM25和FAISS索引
        self._initialize_bm25()
        self._initialize_faiss_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在向量存储中搜索与查询最相关的文档。
        结合语义搜索和BM25搜索，并根据配置进行混合或单一检索。
        
        Args:
            query: 搜索查询字符串。
            top_k: 返回最相关文档的数量。
            
        Returns:
            包含相关文档内容和元数据的字典列表。
        """
        retrieval_method = settings.chat_retrieval_method
        score_threshold = settings.chat_score_threshold
        
        semantic_results = []
        if self.embeddings is not None and len(self.embeddings) > 0 and self.faiss_index is not None:
            if self.embedding_model is None:
                active_embedding_key = settings.default_embedding_provider
                self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
            
            query_embedding = np.array(self.embedding_model.embed_documents([query])[0], dtype=np.float32).reshape(1, -1)
            
            # FAISS 搜索
            # 确保 self.faiss_index 不为 None 且 query_embedding 是有效的 numpy 数组
            if self.faiss_index is not None and isinstance(query_embedding, np.ndarray):
                distances, indices = self.faiss_index.search(query_embedding, top_k * 2) # type: ignore # 检索更多以供过滤
            else:
                distances, indices = np.array([[]]), np.array([[]]) # 如果条件不满足，返回空数组
            
            similarities = 1 - (distances[0] / np.max(distances[0])) if distances.size > 0 and np.max(distances[0]) != 0 else np.zeros_like(distances[0]) # 归一化距离为相似度
            
            semantic_results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue # 跳过无效索引
                if similarities.size > i and similarities[i] >= score_threshold: # 确保索引有效
                    doc = copy.deepcopy(self.documents[idx])
                    doc["semantic_score"] = similarities[i]
                    semantic_results.append(doc)
            semantic_results = sorted(semantic_results, key=lambda x: x.get("semantic_score", 0), reverse=True)[:top_k]

        full_text_results = []
        if self.bm25_index:
            tokenized_query = list(jieba.cut(query))
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-top_k:][::-1]
            full_text_results = [
                {**copy.deepcopy(self.documents[i]), "keyword_score": doc_scores[i]}
                for i in top_indices if doc_scores[i] > 0
            ][:top_k]

        # 合并和重排逻辑 (与 retriever.py 中的 HybridReranker 类似)
        if retrieval_method == RetrievalMethod.HYBRID_SEARCH: # 使用导入的 RetrievalMethod
            all_docs = {doc["metadata"]["source"] + doc["page_content"]: doc for doc in semantic_results}
            for doc in full_text_results:
                key = doc["metadata"]["source"] + doc["page_content"]
                if key in all_docs: all_docs[key].update(doc)
                else: all_docs[key] = doc
            
            # 归一化BM25分数
            documents_to_rerank = list(all_docs.values())
            keyword_scores = [doc.get("keyword_score", 0) for doc in documents_to_rerank]
            max_keyword_score = max(keyword_scores) if keyword_scores else 1
            normalized_keyword_scores = [s / max_keyword_score for s in keyword_scores]
            
            # 归一化语义分数
            semantic_scores = [doc.get("semantic_score", 0) for doc in documents_to_rerank]
            max_semantic_score = max(semantic_scores) if semantic_scores else 1
            normalized_semantic_scores = [s / max_semantic_score for s in semantic_scores]
            
            for i, doc in enumerate(documents_to_rerank):
                doc["score"] = (settings.chat_vector_weight * normalized_semantic_scores[i] +
                                settings.chat_keyword_weight * normalized_keyword_scores[i])
            ranked_results = sorted(documents_to_rerank, key=lambda x: x["score"], reverse=True)
        elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH: # 使用导入的 RetrievalMethod
            ranked_results = sorted(semantic_results, key=lambda x: x.get("semantic_score", 0), reverse=True)
        else: # FULL_TEXT_SEARCH
            ranked_results = sorted(full_text_results, key=lambda x: x.get("keyword_score", 0), reverse=True)

        return ranked_results[:top_k]

    def save(self, path: str):
        """
        将向量存储保存到指定路径。
        
        Args:
            path: 保存向量存储的路径。
        """
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    def load(self, path: str):
        """
        从指定路径加载向量存储。
        
        Args:
            path: 加载向量存储的路径。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"向量存储文件未找到: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.embeddings = data.get("embeddings", None)
        self._initialize_bm25()
        self._initialize_faiss_index()

    def get_embedding_model(self) -> Any:
        """
        获取当前向量存储使用的嵌入模型实例。
        
        Returns:
            嵌入模型实例。
        """
        if self.embedding_model is None:
            active_embedding_key = settings.default_embedding_provider
            self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
        return self.embedding_model