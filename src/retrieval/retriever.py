# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import numpy as np
import jieba
import copy
from typing import Any, Dict, List, Optional
from rich.console import Console

# 使用相对导入来引用同一 src 目录下的模块
from ..utils.config import RetrievalMethod, get_settings # 导入 get_settings 函数
from ..providers.factory import ModelProviderFactory
from .vdb.base import VectorStoreBase
from ..utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

# =================================================================
# 2. 检索逻辑 (RETRIEVAL LOGIC)
# =================================================================
class HybridReranker:
    def __init__(self, vector_weight: float, keyword_weight: float):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.debug(f"初始化混合重排器，向量权重: {vector_weight}, 关键词权重: {keyword_weight}")

    def rerank(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            logger.info("没有文档可供重排，返回空列表。")
            return []
        
        logger.debug(f"开始混合重排，文档数量: {len(documents)}")
        # 为每个文档分配语义和关键字分数
        for doc in documents:
            doc['semantic_score'] = doc.get('semantic_score', 0)
            doc['keyword_score'] = doc.get('keyword_score', 0)

        keyword_scores = [doc['keyword_score'] for doc in documents]
        semantic_scores = [doc['semantic_score'] for doc in documents]

        # 归一化BM25分数
        max_keyword_score = max(keyword_scores) if keyword_scores else 0
        normalized_keyword_scores = [s / max_keyword_score if max_keyword_score != 0 else 0 for s in keyword_scores]
        
        # 归一化语义分数
        max_semantic_score = max(semantic_scores) if semantic_scores else 0
        normalized_semantic_scores = [s / max_semantic_score if max_semantic_score != 0 else 0 for s in semantic_scores]
        
        for i, doc in enumerate(documents):
            doc["score"] = (self.vector_weight * normalized_semantic_scores[i] +
                            self.keyword_weight * normalized_keyword_scores[i])
        
        ranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
        logger.debug(f"混合重排完成，返回 {len(ranked_docs)} 个文档。")
        return ranked_docs

def retrieve_documents(
    query: str,
    vector_store: VectorStoreBase,
    console: Console,
    retrieval_method: RetrievalMethod,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    rerank_enabled: bool,
    active_rerank_configuration: str,
    score_threshold: float,
) -> List[Dict]:
    logger.info(f"开始检索文档，查询: '{query}', 检索方法: {retrieval_method.value}, top_k: {top_k}")
    
    ranked_results: List[Dict] = [] # 明确类型
    
    if retrieval_method == RetrievalMethod.HYBRID_SEARCH:
        logger.info("执行混合搜索。")
        semantic_results = vector_store.search(query, top_k, search_type="semantic")
        keyword_results = vector_store.search(query, top_k, search_type="keyword")
        
        logger.debug(f"语义搜索结果数量: {len(semantic_results)}, 关键词搜索结果数量: {len(keyword_results)}")

        # 合并并去重
        all_docs_map = {doc['metadata']['source'] + str(doc['metadata'].get('page', '')): doc for doc in semantic_results}
        for doc in keyword_results:
            key = doc['metadata']['source'] + str(doc['metadata'].get('page', ''))
            if key in all_docs_map:
                # 如果文档已存在，更新其关键词分数
                all_docs_map[key]['keyword_score'] = doc.get('score', 0)
                logger.debug(f"合并文档: {key}，更新关键词分数。")
            else:
                # 如果文档不存在，添加新文档
                all_docs_map[key] = doc
                logger.debug(f"合并文档: {key}，添加新文档。")

        # 为文档分配分数
        for doc in all_docs_map.values():
            doc['semantic_score'] = doc.get('score', 0) if 'semantic_score' not in doc else doc['semantic_score']
            doc['keyword_score'] = doc.get('score', 0) if 'keyword_score' not in doc else doc['keyword_score']

        reranker = HybridReranker(vector_weight, keyword_weight)
        ranked_results = reranker.rerank(list(all_docs_map.values()))
        logger.info(f"混合搜索完成，初始重排文档数量: {len(ranked_results)}")

    elif retrieval_method == RetrievalMethod.SEMANTIC_SEARCH:
        logger.info("执行语义搜索。")
        ranked_results = vector_store.search(query, top_k, search_type="semantic")
        logger.info(f"语义搜索完成，文档数量: {len(ranked_results)}")
    else: # FULL_TEXT_SEARCH
        logger.info("执行关键词搜索。")
        ranked_results = vector_store.search(query, top_k, search_type="keyword")
        logger.info(f"关键词搜索完成，文档数量: {len(ranked_results)}")

    # 应用分数阈值过滤
    initial_count = len(ranked_results)
    ranked_results = [doc for doc in ranked_results if doc.get("score", 0) >= score_threshold]
    if len(ranked_results) < initial_count:
        logger.info(f"应用分数阈值 {score_threshold} 后，过滤掉 {initial_count - len(ranked_results)} 个文档。")

    # 使用外部Reranker（如果启用）
    if rerank_enabled:
        logger.info(f"外部Rerank已启用，正在使用 '{active_rerank_configuration}' 进行重排。")
        rerank_provider = ModelProviderFactory.get_rerank_provider(active_rerank_configuration)
        if rerank_provider and ranked_results:
            console.print(f"[dim]正在使用 '{active_rerank_configuration}' 进行重排...[/dim]")
            docs_to_rerank = [doc.get("page_content", "") for doc in ranked_results]
            
            try:
                reranked_indices, reranked_scores = rerank_provider.rerank(query, docs_to_rerank, top_n=top_k)
                
                # 根据索引和分数重新构建文档列表
                reranked_docs = []
                for i, score in zip(reranked_indices, reranked_scores):
                    if i < len(ranked_results):
                        doc = ranked_results[i]
                        doc['score'] = score # 更新为 reranker 的分数
                        reranked_docs.append(doc)
                
                # 按新的分数排序
                ranked_results = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)
                logger.info(f"外部重排完成，返回 {len(ranked_results)} 个文档。")

            except Exception as e:
                logger.error(f"Rerank提供商出错: {e}", exc_info=True) # 记录详细异常信息
                console.print(f"[bold red]Rerank提供商出错: {e}[/bold red]")
    else:
        logger.info("外部Rerank未启用。")

    final_results = ranked_results[:top_k]
    logger.info(f"检索过程完成，最终返回 {len(final_results)} 个文档。")
    return final_results