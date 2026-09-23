from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from src.providers.__base__.model_provider import LargeLanguageModel
from src.runtime.contracts import SessionConfig
from src.services.retrieval_service import RetrievalService
from src.utils.log_manager import get_chat_logger
from src.utils.security import redact_sensitive_text


class ChatService:
    def __init__(self, llm_model: LargeLanguageModel, retrieval_service: RetrievalService):
        self.llm_model = llm_model
        self.retrieval_service = retrieval_service
        self.logger = get_chat_logger()

    def _temperature_kwargs(self, temperature: float) -> dict[str, Any]:
        supports_sampling = getattr(self.llm_model, "supports_sampling_option", None)
        if callable(supports_sampling) and supports_sampling("temperature"):
            return {"temperature": temperature}
        return {}

    async def identify_intent(self, user_query: str) -> str:
        prompt = f"你是一个意图识别助手。判断用户意图。\n用户问题: {user_query}\n直接输出意图简述。"
        response = ""
        try:
            invoke_kwargs: dict[str, Any] = {"prompt": prompt, "stream": False}
            invoke_kwargs.update(self._temperature_kwargs(0.1))
            async for chunk in self.llm_model.ainvoke(**invoke_kwargs):
                response += chunk
        except Exception as exc:
            self.logger.error("异步意图识别失败: %s", redact_sensitive_text(str(exc)))
            raise RuntimeError("意图识别失败，无法继续检索。") from exc
        intent = response.strip()
        if not intent:
            raise RuntimeError("意图识别返回空结果，无法继续检索。")
        return intent

    async def retrieve(self, user_input: str, session_config: SessionConfig, console: Any) -> tuple[str, list[dict[str, Any]]]:
        intent = await self.identify_intent(user_input)
        documents = await self.retrieval_service.retrieve(intent, session_config, console=console)
        return intent, documents

    async def generate_reply(
        self,
        user_input: str,
        intent: str,
        documents: list[dict[str, Any]],
        session_config: SessionConfig,
    ) -> AsyncGenerator[str, None]:
        prompt = self._build_prompt(user_input, intent, documents)
        invoke_kwargs: dict[str, Any] = {"prompt": prompt, "stream": True}
        invoke_kwargs.update(self._temperature_kwargs(session_config.chat_temperature))
        async for chunk in self.llm_model.ainvoke(**invoke_kwargs):
            yield chunk

    @staticmethod
    def _build_prompt(user_input: str, intent: str, documents: list[dict[str, Any]]) -> str:
        context = "\n".join(
            f"来源: {(doc.get('metadata') or {}).get('source') or '未知来源'}\n"
            f"内容: {doc.get('page_content', '')}"
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
