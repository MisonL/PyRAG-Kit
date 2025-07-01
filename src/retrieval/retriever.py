# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import numpy as np
import jieba
import copy
from typing import Any, Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from rich.console import Console

# 使用相对导入来引用同一 src 目录下的模块
from ..utils.config import CHAT_CONFIG, KB_CONFIG, RetrievalMethod
from ..providers.factory import ModelProviderFactory

# =================================================================
# 2. 向量存储与搜索 (VECTOR STORE & SEARCH)
# =================================================================
class VectorStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        """在加载文档后初始化BM25索引。"""
        if self.documents:
            tokenized_corpus = [list(jieba.cut(doc.get("page_content", ""))) for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)

    def semantic_search(self, query_embedding: np.ndarray, top_k: int, score_threshold: float) -> List[Dict]:
        if self.embeddings is None or len(self.embeddings) == 0: return []
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {**copy.deepcopy(self.documents[i]), "score": similarities[i]}
            for i in top_indices if similarities[i] >= score_threshold
        ]
        return results

    def bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """使用BM25算法进行全文搜索。"""
        if not self.bm25_index: return []
        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        results = [
            {**copy.deepcopy(self.documents[i]), "score": doc_scores[i]}
            for i in top_indices if doc_scores[i] > 0
        ]
        return results

# =================================================================
# 3. 检索逻辑 (RETRIEVAL LOGIC)
# =================================================================
class HybridReranker:
    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents: return []
        # 归一化BM25分数
        keyword_scores = [doc.get("keyword_score", 0) for doc in documents]
        max_keyword_score = max(keyword_scores) if keyword_scores else 1
        normalized_keyword_scores = [s / max_keyword_score for s in keyword_scores]
        # 归一化语义分数
        semantic_scores = [doc.get("semantic_score", 0) for doc in documents]
        max_semantic_score = max(semantic_scores) if semantic_scores else 1
        normalized_semantic_scores = [s / max_semantic_score for s in semantic_scores]
        
        for i, doc in enumerate(documents):
            doc["score"] = (CHAT_CONFIG["vector_weight"] * normalized_semantic_scores[i] +
                            CHAT_CONFIG["keyword_weight"] * normalized_keyword_scores[i])
        return sorted(documents, key=lambda x: x["score"], reverse=True)

def retrieve_documents(query: str, vector_store: VectorStore, console: Console) -> List[Dict]:
    retrieval_method = CHAT_CONFIG["retrieval_method"]
    top_k = CHAT_CONFIG["top_k"]
    score_threshold = CHAT_CONFIG["score_threshold"]
    
    # 语义搜索
    active_embedding_key = KB_CONFIG['active_embedding_configuration']
    embedding_provider = ModelProviderFactory.get_embedding_provider(active_embedding_key)
    query_embedding = np.array(embedding_provider.embed_documents([query])[0])
    semantic_results = vector_store.semantic_search(query_embedding, top_k, score_threshold)
    for doc in semantic_results: doc["semantic_score"] = doc.pop("score", 0)
    
    # 全文搜索 (BM25)
    full_text_results = vector_store.bm25_search(query, top_k)
    for doc in full_text_results: doc["keyword_score"] = doc.pop("score", 0)

    # 合并和重排
    if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
        all_docs = {doc["metadata"]["source"] + doc["page_content"]: doc for doc in semantic_results}
        for doc in full_text_results:
            key = doc["metadata"]["source"] + doc["page_content"]
            if key in all_docs: all_docs[key].update(doc)
            else: all_docs[key] = doc
        ranked_results = HybridReranker().rerank(list(all_docs.values()))
    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        ranked_results = sorted(semantic_results, key=lambda x: x.get("semantic_score", 0), reverse=True)
    else: # FULL_TEXT_SEARCH
        ranked_results = sorted(full_text_results, key=lambda x: x.get("keyword_score", 0), reverse=True)

    # 使用Reranker（如果启用）
    if CHAT_CONFIG["rerank_enabled"]:
        active_rerank_key = CHAT_CONFIG['active_rerank_configuration']
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_key)
        if rerank_provider and ranked_results:
            console.print(f"[dim]正在使用 '{active_rerank_key}' 进行重排...[/dim]")
            # 1. 提取文档内容用于重排
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            
            # 2. 调用 rerank 方法获取重排后的索引
            reranked_indices = rerank_provider.rerank(query, docs_to_rerank, top_n=top_k)
            
            # 3. 根据索引重新排序原始文档列表
            reranked_docs = [ranked_results[i] for i in reranked_indices]
            
            return reranked_docs[:top_k]
    
    return ranked_results[:top_k]