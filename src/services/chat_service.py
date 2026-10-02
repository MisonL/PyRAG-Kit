from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from src.providers.__base__.model_provider import LargeLanguageModel
from src.runtime.contracts import SessionConfig
from src.services.retrieval_service import RetrievalService
from src.ui.display_utils import document_content, document_source
from src.utils.config import require_chat_temperature, safe_repr
from src.utils.log_manager import get_chat_logger
from src.utils.security import safe_exception_text


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
        # 与 ``_build_prompt`` 同判据：非字符串取值不得以字面量进 prompt。
        # 这里的 f-string 在 ``try`` **之外**，因此位数超限的整数抛的是裸
        # ``ValueError: Exceeds the limit``（而不是本类承诺的
        # ``RuntimeError: 意图识别失败``），``None``／``5`` 则静默生成
        # ``用户问题: None`` 交给模型。
        if not isinstance(user_query, str):
            raise ValueError(f"用户输入必须是字符串，但得到 {safe_repr(user_query)}。")
        prompt = f"你是一个意图识别助手。判断用户意图。\n用户问题: {user_query}\n直接输出意图简述。"
        response = ""
        try:
            invoke_kwargs: dict[str, Any] = {"prompt": prompt, "stream": False}
            invoke_kwargs.update(self._temperature_kwargs(0.1))
            async for chunk in self.llm_model.ainvoke(**invoke_kwargs):
                response += chunk
        except Exception as exc:
            self.logger.error("异步意图识别失败: %s", safe_exception_text(exc))
            raise RuntimeError("意图识别失败，无法继续检索。") from exc
        intent = response.strip()
        if not intent:
            raise RuntimeError("意图识别返回空结果，无法继续检索。")
        return intent

    async def retrieve(
        self, user_input: str, session_config: SessionConfig, console: Any
    ) -> tuple[str, list[dict[str, Any]]]:
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
        # 收敛后再用：``chat_temperature`` 的两个校验器都在**写入**侧，而
        # ``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫——直写
        # ``"abc"``／``None``／``[0.5]``／``True``／``-1`` 会原样进请求 kwargs
        # （实测），而它们正是 ``session_config["chat_temperature"] = ...`` 明确
        # 拒绝的取值。非法值到服务端才暴露，用户看到的是接口参数错误。
        invoke_kwargs.update(
            self._temperature_kwargs(require_chat_temperature(session_config.chat_temperature))
        )
        async for chunk in self.llm_model.ainvoke(**invoke_kwargs):
            yield chunk

    @staticmethod
    def _build_prompt(user_input: str, intent: str, documents: list[dict[str, Any]]) -> str:
        # 标量入参的形态先收敛，判据与下面文档处理**同一条**：非字符串值不得以
        # 字面量进 prompt。这两个参数此前完全没有守卫，实测
        # ``user_input=None``／``5``／``["a"]`` 静默生成
        # ``用户问题: None``／``用户问题: 5``／``用户问题: ['a']``——模型会把
        # 字面量 ``None`` 当成一个真实的用户问题去回答（与
        # ``test_build_prompt_does_not_feed_malformed_content_to_the_model``
        # 钉住的 ``内容: None`` 是同一形态，只是那一条盯的是文档字段）。
        # 位数超限的整数则更干脆：``f"{10**4301}"`` 抛裸
        # ``ValueError: Exceeds the limit``，消息与「问题写错了」毫无关系。
        #
        # 抛 ``ValueError`` 而不是静默替换成空串：本方法的两个调用点
        # （``ChatService.generate_reply`` 与 ``Chatbot.chat_async`` 的内联分支）
        # 都在按 ``ValueError`` 分类的边界内，而「用户输入不是文本」是调用方的
        # 编码错误，不该被掩盖成一次正常问答。
        for label, value in (("用户输入", user_input), ("意图", intent)):
            if not isinstance(value, str):
                raise ValueError(f"{label}必须是字符串，但得到 {safe_repr(value)}。")
        # ``documents`` 的容器形态同判据。上面两个标量参数有守卫而这一半此前
        # 漏了，实测两种收场都不在契约内：``5``／``None`` 抛裸
        # ``TypeError: 'int' object is not iterable``（不是 ``ValueError``，两个
        # 调用点的错误分类接不住）；``"abc"`` 更隐蔽——字符串是合法可迭代对象，
        # 逐**字符**迭代后每个字符都被下面的「内容为空则跳过」滤掉，于是**静默
        # 退化成「没有找到相关知识」**，一次带着知识调用的请求被无声降级，模型
        # 据此回答「我不知道」。
        if not isinstance(documents, list):
            raise ValueError(f"documents 必须是文档列表，但得到 {safe_repr(documents)}。")
        # 字段形态统一由 display_utils 的助手收敛。检索结果可以来自任何
        # ``VectorStoreBase`` 实现——``FaissStore`` 会在载入期校验并归一化，
        # 第三方 store 不会。两处后果：``metadata`` 非映射时抛
        # ``AttributeError``；更隐蔽的是 ``page_content=None`` 不抛错，而是把
        # 字面量 ``内容: None`` 当作一条真实知识喂给模型。
        #
        # 内容缺失时整条跳过而不是留一行空「内容:」：向模型提交一条没有正文的
        # 「知识」只会增加噪声，还可能让它顺着回答。
        entries = []
        for document in documents:
            content = document_content(document, "")
            if not content:
                continue
            entries.append(f"来源: {document_source(document, '未知来源')}\n内容: {content}")
        context = "\n".join(entries)
        if context:
            return (
                "你是一个智能客服。根据知识回答。\n"
                f"意图: {intent}\n"
                f"问题: {user_input}\n"
                f"知识:\n{context}\n"
                "回答要简洁。"
            )
        return f"你是一个智能客服。\n用户问题: {user_input}\n没有找到相关知识。请礼貌告知。"
