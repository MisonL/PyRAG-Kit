# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List, cast

from google.generativeai.client import configure
from google.generativeai.embedding import embed_content
from google.generativeai.generative_models import GenerativeModel

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

# 初始化Google API
# 在需要时才获取 settings，以确保配置是最新的
current_settings_for_api_key = get_settings()
if current_settings_for_api_key.google_api_key:
    configure(api_key=current_settings_for_api_key.google_api_key)
    logger.info("Google API 已配置。")
else:
    logger.warning("Google API Key 未设置，GoogleProvider 可能无法正常工作。")


class GoogleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    Google模型提供商，统一处理Gemini LLM和Embedding。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        logger.info(f"初始化 GoogleProvider，模型: {model_name}")
        # 仅在需要时（即调用invoke时）才初始化GenerativeModel，以节省资源
        self._model: GenerativeModel | None = None

    def _initialize_llm(self):
        """延迟初始化LLM模型。"""
        if not self._model:
            logger.debug(f"正在延迟初始化 Google GenerativeModel: {self._model_name}")
            self._model = GenerativeModel(self._model_name)
            logger.info(f"Google GenerativeModel ({self._model_name}) 初始化成功。")

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = None,  # Google API v1 不直接支持 system_prompt
        tools: List[Dict[str, Any]] | None = None, # Google API v1 不直接支持 tools
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用Google Gemini模型。
        注意：Google的Python SDK v1的stream实现是异步的，
        这里为了兼容性，我们使用非流式API并以流式方式返回单个结果。
        """
        logger.info(f"调用 Google LLM ({self._model_name})，流式: {stream}, 温度: {temperature}")
        self._initialize_llm()
        if not self._model:
            logger.error("Google GenerativeModel未能初始化。")
            yield "错误：Google GenerativeModel未能初始化。"
            return

        try:
            # 启用流式响应
            response_stream = self._model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
            logger.info("Google LLM 内容生成完成。")
        except Exception as e:
            error_message = f"Google LLM ({self._model_name}) 生成内容时出错: {e}"
            logger.error(error_message, exc_info=True) # 记录详细异常信息
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用Google Embedding模型将文档列表向量化。
        """
        logger.info(f"调用 Google Embedding ({self._model_name})，文档数量: {len(texts)}")
        try:
            embeddings = embed_content(
                model=self._model_name,
                content=texts,
                task_type="retrieval_document"
            )
            result = cast(List[List[float]], embeddings.get("embedding"))
            if result is not None:
                logger.info(f"Google Embedding 完成，生成 {len(result)} 个嵌入。")
                return result
            else:
                logger.warning("Google Embedding 返回空结果。")
                return [[] for _ in texts]
        except Exception as e:
            error_message = f"Google嵌入文档时出错 ({self._model_name}): {e}。将返回空嵌入。"
            logger.error(error_message, exc_info=True) # 记录详细异常信息
            return [[] for _ in texts]
