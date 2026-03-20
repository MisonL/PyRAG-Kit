# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Dict, List

from rich.console import Console

from ..providers.factory import ModelProviderFactory
from ..runtime.contracts import SessionConfig, build_run_config
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import (
    HybridReranker,
    RetrievalService,
    _merge_hybrid_results,
)
from ..utils.config import RetrievalMethod, get_settings


def _build_session_config(
    retrieval_method: RetrievalMethod,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    rerank_enabled: bool,
    active_rerank_configuration: str,
    score_threshold: float,
    fusion_strategy: str,
    candidate_multiplier: int,
) -> SessionConfig:
    settings = get_settings()
    session_config = SessionConfig(
        retrieval_method=retrieval_method,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        hybrid_fusion_strategy=fusion_strategy,
        retrieval_candidate_multiplier=candidate_multiplier,
        rerank_enabled=rerank_enabled,
        top_k=top_k,
        score_threshold=score_threshold,
        active_llm_configuration=settings.default_llm_provider,
        active_rerank_configuration=active_rerank_configuration,
        llm_configurations=settings.llm_configurations,
        rerank_configurations=settings.rerank_configurations,
        chat_temperature=settings.chat_temperature,
    )
    return session_config


def retrieve_documents(
    query: str,
    vector_store,
    console: Console,
    retrieval_method: RetrievalMethod,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    rerank_enabled: bool,
    active_rerank_configuration: str,
    score_threshold: float,
    fusion_strategy: str = "rrf",
    candidate_multiplier: int = 3,
) -> List[Dict]:
    return asyncio.run(
        aretrieve_documents(
            query=query,
            vector_store=vector_store,
            console=console,
            retrieval_method=retrieval_method,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            rerank_enabled=rerank_enabled,
            active_rerank_configuration=active_rerank_configuration,
            score_threshold=score_threshold,
            fusion_strategy=fusion_strategy,
            candidate_multiplier=candidate_multiplier,
        )
    )


async def aretrieve_documents(
    query: str,
    vector_store,
    console: Console,
    retrieval_method: RetrievalMethod,
    top_k: int,
    vector_weight: float,
    keyword_weight: float,
    rerank_enabled: bool,
    active_rerank_configuration: str,
    score_threshold: float,
    fusion_strategy: str = "rrf",
    candidate_multiplier: int = 3,
) -> List[Dict]:
    run_config = build_run_config(get_settings())
    session_config = _build_session_config(
        retrieval_method=retrieval_method,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        rerank_enabled=rerank_enabled,
        active_rerank_configuration=active_rerank_configuration,
        score_threshold=score_threshold,
        fusion_strategy=fusion_strategy,
        candidate_multiplier=candidate_multiplier,
    )
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedding_service=EmbeddingService(run_config),
    )
    return await retrieval_service.retrieve(query=query, session_config=session_config, console=console)
