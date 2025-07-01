# -*- coding: utf-8 -*-
from typing import Any, Dict, Generator, List, cast

from google.generativeai.client import configure
from google.generativeai.embedding import embed_content
from google.generativeai.generative_models import GenerativeModel

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import API_CONFIG

# 初始化Google API
if API_CONFIG.get("GOOGLE_API_KEY"):
    configure(api_key=API_CONFIG["GOOGLE_API_KEY"])


class GoogleProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    Google模型提供商，统一处理Gemini LLM和Embedding。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        # 仅在需要时（即调用invoke时）才初始化GenerativeModel，以节省资源
        self._model: GenerativeModel | None = None

    def _initialize_llm(self):
        """延迟初始化LLM模型。"""
        if not self._model:
            self._model = GenerativeModel(self._model_name)

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
        self._initialize_llm()
        if not self._model:
            yield "错误：Google GenerativeModel未能初始化。"
            return

        try:
            # 启用流式响应
            response_stream = self._model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            error_message = f"Google LLM ({self._model_name}) 生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用Google Embedding模型将文档列表向量化。
        """
        try:
            embeddings = embed_content(
                model=self._model_name,
                content=texts,
                task_type="retrieval_document"
            )
            result = cast(List[List[float]], embeddings.get("embedding"))
            return result if result is not None else [[] for _ in texts]
        except Exception as e:
            print(f"Google嵌入文档时出错 ({self._model_name}): {e}。将返回空嵌入。")
            return [[] for _ in texts]
