# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List

import anthropic

from src.providers.__base__.model_provider import LargeLanguageModel
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

class AnthropicProvider(LargeLanguageModel):
    """
    Anthropic模型提供商，处理Claude系列模型。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        logger.info(f"初始化 AnthropicProvider，模型: {model_name}")
        current_settings = get_settings() # 获取当前配置
        api_key = current_settings.anthropic_api_key
        if not api_key:
            logger.error("Anthropic配置不完整：缺少 ANTHROPIC_API_KEY。")
            raise ValueError("Anthropic配置不完整：缺少 ANTHROPIC_API_KEY。")
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic客户端初始化成功。")

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用Anthropic Claude模型。
        """
        logger.info(f"调用 Anthropic LLM ({self._model_name})，流式: {stream}, 温度: {temperature}")
        # Anthropic API v3 使用 system 参数
        try:
            request_params = {
                "model": self._model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                # "tools": tools, # Anthropic的工具使用方式不同，需要单独适配
            }
            if system_prompt:
                request_params["system"] = system_prompt
                logger.debug(f"使用系统提示: {system_prompt[:50]}...")

            with self._client.messages.stream(**request_params) as stream_response:
                for text in stream_response.text_stream:
                    yield text
            logger.info("Anthropic LLM 内容生成完成。")
        except Exception as e:
            error_message = f"Anthropic LLM ({self._model_name}) 生成内容时出错: {e}"
            logger.error(error_message, exc_info=True) # 记录详细异常信息
            yield "抱歉，我在生成回答时遇到了一些问题。"
