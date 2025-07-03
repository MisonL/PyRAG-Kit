# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

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
from ...utils.config import get_settings, RetrievalMethod # 导入 get_settings 函数和 RetrievalMethod
from ...providers.factory import ModelProviderFactory
from ...utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

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
        logger.info(f"FaissStore 初始化完成，文件路径: {file_path if file_path else '未指定'}")
    
    def _initialize_bm25(self):
        """在加载文档或添加新文档后初始化/更新BM25索引。"""
        if self.documents:
            logger.debug("正在初始化 BM25 索引...")
            tokenized_corpus = [list(jieba.cut(doc.get("page_content", ""))) for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 索引初始化完成。")
        else:
            logger.debug("没有文档，跳过 BM25 索引初始化。")

    def _initialize_faiss_index(self):
        """根据当前嵌入初始化FAISS索引。"""
        if self.embeddings is not None and len(self.embeddings) > 0:
            logger.debug("正在初始化 FAISS 索引...")
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            # 确保 self.faiss_index 不为 None 且 self.embeddings 是有效的 numpy 数组
            if self.faiss_index is not None and isinstance(self.embeddings, np.ndarray):
                self.faiss_index.add(self.embeddings) # type: ignore
                logger.info("FAISS 索引初始化完成。")
            else:
                logger.warning("FAISS 索引或嵌入数据无效，无法添加。")
        else:
            logger.debug("没有嵌入数据，跳过 FAISS 索引初始化。")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向向量存储中添加文档。
        此方法会生成嵌入并更新FAISS和BM25索引。
        
        Args:
            documents: 包含文档内容和元数据的字典列表。
                       每个字典应至少包含 'page_content' 键。
        """
        if not documents:
            logger.warning("尝试添加空文档列表，操作取消。")
            return
        
        logger.info(f"正在添加 {len(documents)} 个文档到 FaissStore。")

        # 获取嵌入模型
        if self.embedding_model is None:
            logger.debug("嵌入模型未初始化，正在获取默认嵌入提供商。")
            current_settings = get_settings() # 获取当前配置
            active_embedding_key = current_settings.default_embedding_provider
            try:
                self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
                logger.info(f"成功获取嵌入模型: {active_embedding_key}")
            except Exception as e:
                logger.error(f"获取嵌入模型失败: {e}", exc_info=True)
                raise

        # 生成新文档的嵌入
        new_texts = [doc.get("page_content", "") for doc in documents]
        logger.debug(f"正在为 {len(new_texts)} 个文档生成嵌入。")
        try:
            new_embeddings = np.array(self.embedding_model.embed_documents(texts=new_texts), dtype=np.float32)
            logger.debug("嵌入生成完成。")
        except Exception as e:
            logger.error(f"生成文档嵌入失败: {e}", exc_info=True)
            raise

        # 合并新旧文档和嵌入
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        logger.info(f"当前文档总数: {len(self.documents)}, 嵌入总数: {len(self.embeddings) if self.embeddings is not None else 0}")

        # 更新BM25和FAISS索引
        self._initialize_bm25()
        self._initialize_faiss_index()
        logger.info("文档添加和索引更新完成。")

    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        在向量存储中搜索与查询最相关的文档。
        根据 search_type 参数执行语义搜索或关键字搜索。

        Args:
            query: 搜索查询字符串。
            top_k: 返回最相关文档的数量。
            search_type: 搜索类型 ('semantic' 或 'keyword')。

        Returns:
            包含相关文档内容和元数据的字典列表。
        """
        if search_type == "semantic":
            if self.embeddings is None or len(self.embeddings) == 0 or self.faiss_index is None:
                return []
            
            if self.embedding_model is None:
                current_settings = get_settings() # 获取当前配置
                active_embedding_key = current_settings.default_embedding_provider
                self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
            
            query_embedding = np.array(self.embedding_model.embed_documents([query])[0], dtype=np.float32).reshape(1, -1)
            
            if self.faiss_index is None or not isinstance(query_embedding, np.ndarray):
                 return []

            distances, indices = self.faiss_index.search(query_embedding, top_k) # type: ignore
            
            # L2 距离转换为相似度分数 (0-1范围)
            # 一个简单的转换方法是 1 / (1 + distance)
            similarities = 1.0 / (1.0 + distances[0])

            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue
                doc = copy.deepcopy(self.documents[idx])
                doc["score"] = similarities[i] # 使用 'score' 作为通用分数键
                results.append(doc)
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        elif search_type == "keyword":
            if not self.bm25_index:
                return []
            
            tokenized_query = list(jieba.cut(query))
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            
            # 获取所有分数大于0的文档的索引和分数
            top_indices = np.argsort(doc_scores)[::-1]
            
            results = []
            for i in top_indices:
                if doc_scores[i] > 0:
                    doc = copy.deepcopy(self.documents[i])
                    doc["score"] = doc_scores[i] # 使用 'score' 作为通用分数键
                    results.append(doc)
                else:
                    # 由于分数是排序的，一旦遇到非正数就可以停止
                    break
            
            # 在返回之前按分数排序并取 top_k
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")

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
            current_settings = get_settings() # 获取当前配置
            active_embedding_key = current_settings.default_embedding_provider
            self.embedding_model = ModelProviderFactory.get_embedding_provider(active_embedding_key)
        return self.embedding_model