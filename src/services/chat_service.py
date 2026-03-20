# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List

from src.providers.__base__.model_provider import LargeLanguageModel
from src.runtime.contracts import SessionConfig
from src.services.retrieval_service import RetrievalService
from src.utils.log_manager import get_chat_logger


class ChatService:
    def __init__(self, llm_model: LargeLanguageModel, retrieval_service: RetrievalService):
        self.llm_model = llm_model
        self.retrieval_service = retrieval_service
        self.logger = get_chat_logger()

    async def identify_intent(self, user_query: str) -> str:
        prompt = f"你是一个意图识别助手。判断用户意图。\n用户问题: {user_query}\n直接输出意图简述。"
        response = ""
        try:
            async for chunk in self.llm_model.ainvoke(prompt=prompt, stream=False, temperature=0.1):
                response += chunk
        except Exception as exc:
            self.logger.error("异步意图识别失败: %s", exc)
            return user_query
        return response.strip() or user_query

    async def retrieve(self, user_input: str, session_config: SessionConfig, console: Any) -> tuple[str, List[Dict[str, Any]]]:
        intent = await self.identify_intent(user_input)
        documents = await self.retrieval_service.retrieve(intent, session_config, console=console)
        return intent, documents

    async def generate_reply(
        self,
        user_input: str,
        intent: str,
        documents: List[Dict[str, Any]],
        session_config: SessionConfig,
    ) -> AsyncGenerator[str, None]:
        prompt = self._build_prompt(user_input, intent, documents)
        async for chunk in self.llm_model.ainvoke(
            prompt=prompt,
            stream=True,
            temperature=session_config.chat_temperature,
        ):
            yield chunk

    @staticmethod
    def _build_prompt(user_input: str, intent: str, documents: List[Dict[str, Any]]) -> str:
        context = "\n".join(
            f"来源: {doc['metadata']['source']}\n内容: {doc.get('page_content', '')}"
            for doc in documents
        )
        if context:
            return (
                "你是一个智能客服。根据知识回答。\n"
                f"意图: {intent}\n"
                f"问题: {user_input}\n"
                f"知识:\n{context}\n"
                "回答要简洁。"
            )
        return (
            "你是一个智能客服。\n"
            f"用户问题: {user_input}\n"
            "没有找到相关知识。请礼貌告知。"
        )
