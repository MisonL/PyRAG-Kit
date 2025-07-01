from __future__ import annotations
# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import os
import re
import numpy as np
import requests
import pickle
import time
import json
import copy
import logging
import uuid
import tiktoken
import openai
import anthropic
import hashlib
import hmac
from typing import Any, Optional, List, Dict, cast, Generator
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from threading import Lock
from google.generativeai.client import configure
from google.generativeai.embedding import embed_content
from google.generativeai.generative_models import GenerativeModel

from ..utils.config import KB_CONFIG, CHAT_CONFIG, API_CONFIG

# =================================================================
# 2. 抽象基类 (ABSTRACT BASE CLASSES)
# =================================================================

class BaseProviderModel(ABC):
    """所有模型的通用基类。"""
    def __init__(self, model_name: str):
        self._model_name = model_name

class TextEmbeddingModel(BaseProviderModel):
    """文本嵌入模型的接口。"""
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class LLM(BaseProviderModel):
    """大语言模型的接口。"""
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        """以流式方式生成内容。"""
        raise NotImplementedError

class RerankModel(BaseProviderModel):
    """重排模型的接口。"""
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        raise NotImplementedError

# =================================================================
# 3. 具体实现 (CONCRETE IMPLEMENTATIONS)
# =================================================================

# --- 非兼容API实现 ---

class GoogleEmbedding(TextEmbeddingModel):
    """Google Embedding模型实现"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = embed_content(model=self._model_name, content=texts, task_type="retrieval_document")
            result = cast(List[List[float]], embeddings.get("embedding"))
            return result if result is not None else [[] for _ in texts]
        except Exception as e:
            print(f"Google嵌入文档时出错: {e}。将返回空嵌入。")
            return [[] for _ in texts]

class GoogleLLM(LLM):
    """Google Gemini系列模型的封装"""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self._model = GenerativeModel(model_name)

    def generate_content(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Google LLM ({self._model_name}) 生成内容时出错: {e}")
            return "抱歉，我在生成回答时遇到了一些问题。"

    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        """对于不支持流式的API，直接yield完整的非流式结果。"""
        yield self.generate_content(prompt)

class VolcengineBase:
    """火山引擎API签名逻辑的基类"""
    def __init__(self):
        access_key = API_CONFIG.get("VOLC_ACCESS_KEY")
        if not access_key:
            raise ValueError("火山引擎配置不完整：缺少 VOLC_ACCESS_KEY。")
        self._access_key = access_key

        secret_key = API_CONFIG.get("VOLC_SECRET_KEY")
        if not secret_key:
            raise ValueError("火山引擎配置不完整：缺少 VOLC_SECRET_KEY。")
        self._secret_key = secret_key

        base_url = API_CONFIG.get("VOLC_BASE_URL")
        if not base_url:
            raise ValueError("火山引擎配置不完整：缺少 VOLC_BASE_URL。")
        self._base_url = base_url

    def _sign_request(self, method: str, path: str, query: str, headers: Dict, body: bytes) -> Dict:
        """为火山引擎API请求生成签名"""
        canonical_request = f"{method}\n{path}\n{query}\n"
        signed_headers = sorted([f"{k.lower()}:{v}" for k, v in headers.items()])
        canonical_request += "\n".join(signed_headers) + "\n"
        signed_headers_str = ";".join(sorted([k.lower() for k in headers.keys()]))
        canonical_request += signed_headers_str + "\n"
        payload_hash = hashlib.sha256(body).hexdigest()
        canonical_request += payload_hash

        current_time = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        credential_scope = f"{time.strftime('%Y%m%d', time.gmtime())}/cn-beijing/ml_maas/request"
        string_to_sign = f"HMAC-SHA256\n{current_time}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        k_date = hmac.new(self._secret_key.encode('utf-8'), time.strftime('%Y%m%d', time.gmtime()).encode('utf-8'), hashlib.sha256).digest()
        k_region = hmac.new(k_date, b'cn-beijing', hashlib.sha256).digest()
        k_service = hmac.new(k_region, b'ml_maas', hashlib.sha256).digest()
        k_signing = hmac.new(k_service, b'request', hashlib.sha256).digest()
        
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        auth_header = f"HMAC-SHA256 Credential={self._access_key}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        headers['Authorization'] = auth_header
        headers['X-Date'] = current_time
        return headers

class VolcengineEmbedding(VolcengineBase, TextEmbeddingModel):
    """火山引擎嵌入模型实现"""
    def __init__(self, model_name: str, **kwargs):
        TextEmbeddingModel.__init__(self, model_name)
        VolcengineBase.__init__(self)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self._base_url.rstrip('/')}/api/v1/embeddings"
        path = "/api/v1/embeddings"
        body = json.dumps({"model": self._model_name, "input": texts}).encode('utf-8')
        headers = {'Host': self._base_url.replace('https://', '').split('/')[0], 'Content-Type': 'application/json'}
        signed_headers = self._sign_request("POST", path, "", headers, body)
        
        try:
            response = requests.post(url, headers=signed_headers, data=body, timeout=30)
            response.raise_for_status()
            result = response.json()
            return [item['embedding'] for item in result.get('data', [])]
        except Exception as e:
            print(f"火山引擎嵌入时出错: {e}")
            return [[] for _ in texts]

class VolcengineLLM(VolcengineBase, LLM):
    """火山引擎大语言模型（如豆包）的封装"""
    def __init__(self, model_name: str, **kwargs):
        LLM.__init__(self, model_name)
        VolcengineBase.__init__(self)

    def generate_content(self, prompt: str) -> str:
        url = f"{self._base_url.rstrip('/')}/api/v1/chat/completions"
        path = "/api/v1/chat/completions"
        payload = {"model": self._model_name, "messages": [{"role": "user", "content": prompt}]}
        body = json.dumps(payload).encode('utf-8')
        headers = {'Host': self._base_url.replace('https://', '').split('/')[0], 'Content-Type': 'application/json'}
        signed_headers = self._sign_request("POST", path, "", headers, body)
        
        try:
            response = requests.post(url, headers=signed_headers, data=body, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            print(f"火山引擎 LLM ({self._model_name}) 生成内容时出错: {e}")
            return "抱歉，我在生成回答时遇到了一些问题。"

    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        """对于不支持流式的API，直接yield完整的非流式结果。"""
        yield self.generate_content(prompt)

class GrokLLM(LLM):
    """Grok API的LLM实现"""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self._api_key = API_CONFIG.get("GROK_API_KEY")
        if not self._api_key:
            raise ValueError("Grok配置不完整：缺少 GROK_API_KEY。")
        self._base_url = API_CONFIG.get("GROK_BASE_URL", "https://api.x.ai/v1")

    def generate_content(self, prompt: str) -> str:
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model_name, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            print(f"Grok LLM ({self._model_name}) 生成内容时出错: {e}")
            return "抱歉，我在生成回答时遇到了一些问题。"

    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        """对于不支持流式的API，直接yield完整的非流式结果。"""
        yield self.generate_content(prompt)

class AnthropicLLM(LLM):
    """Anthropic Claude系列模型的封装"""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self._api_key = API_CONFIG.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic配置不完整：缺少 ANTHROPIC_API_KEY。")
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate_content(self, prompt: str) -> str:
        # 对于流式优先的模型，非流式方法可以简单地聚合流式结果
        full_response = "".join(self.generate_content_stream(prompt))
        return full_response.strip()

    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        try:
            with self._client.messages.stream(
                model=self._model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            error_message = f"Anthropic LLM ({self._model_name}) 流式生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"

# --- OpenAI兼容API实现 ---

class OpenAICompatibleBase:
    """处理所有OpenAI兼容API的通用逻辑"""
    def __init__(self, provider: str):
        api_key_name = f"{provider.upper()}_API_KEY"
        base_url_name = f"{provider.upper()}_BASE_URL"
        
        api_key = API_CONFIG.get(api_key_name)
        base_url = API_CONFIG.get(base_url_name)

        # 特殊处理ollama和lm-studio，它们通常不需要key
        if provider in ["ollama", "lm-studio"] and not api_key:
            api_key = "no-key-required"

        if not api_key:
            raise ValueError(f"{provider} 配置不完整：缺少 {api_key_name}。")
        if not base_url and provider not in ["openai"]:
            print(f"警告: 提供商 '{provider}' 的 base_url 未配置。")

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

class OpenAICompatibleEmbedding(OpenAICompatibleBase, TextEmbeddingModel):
    """OpenAI兼容的嵌入模型"""
    def __init__(self, model_name: str, provider: str, **kwargs):
        TextEmbeddingModel.__init__(self, model_name)
        OpenAICompatibleBase.__init__(self, provider)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"OpenAI兼容嵌入 ({self._model_name}) 时出错: {e}")
            return [[] for _ in texts]

class OpenAICompatibleLLM(OpenAICompatibleBase, LLM):
    """OpenAI兼容的LLM"""
    def __init__(self, model_name: str, provider: str, **kwargs):
        LLM.__init__(self, model_name)
        OpenAICompatibleBase.__init__(self, provider)

    def generate_content(self, prompt: str) -> str:
        # 对于流式优先的模型，非流式方法可以简单地聚合流式结果
        full_response = "".join(self.generate_content_stream(prompt))
        return full_response.strip()

    def generate_content_stream(self, prompt: str) -> Generator[str, None, None]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            error_message = f"使用 {self._model_name} ({self._client.base_url}) 流式生成内容时出错: {e}"
            print(error_message)
            yield "抱歉，我在生成回答时遇到了一些问题。"

# --- Rerank实现 ---

class SiliconflowRerankProvider(RerankModel):
    """SiliconFlow Rerank模型实现"""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self._api_key = API_CONFIG.get("SILICONFLOW_API_KEY")
        self._base_url = API_CONFIG.get("SILICONFLOW_API_URL")
        if not self._api_key or not self._base_url:
            raise ValueError("错误：SiliconFlow Rerank 提供商需要 API 密钥和 URL。")

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        base_url = self._base_url
        if not base_url:
            print("错误：SiliconFlow Rerank的base_url未配置。")
            return documents
        url = f"{base_url.rstrip('/')}/rerank"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        doc_contents = [doc.get("page_content", "") for doc in documents]
        payload = {"query": query, "documents": doc_contents, "model": self._model_name, "top_n": CHAT_CONFIG["top_k"]}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            rerank_results = response.json().get("results", [])
            
            original_docs_map = {doc.get("page_content", ""): doc for doc in documents}
            reranked_docs = []
            for res in rerank_results:
                doc_content = res.get("document")
                if doc_content in original_docs_map:
                    original_doc = original_docs_map[doc_content]
                    original_doc["rerank_score"] = res.get("relevance_score")
                    reranked_docs.append(original_doc)
            return reranked_docs
        except Exception as e:
            print(f"SiliconFlow Rerank出错: {e}")
            return documents

class JinaRerankProvider(RerankModel):
    """Jina Rerank模型实现"""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self._api_key = API_CONFIG.get("JINA_API_KEY")
        if not self._api_key:
            raise ValueError("错误：Jina Rerank 提供商需要 API 密钥。")
        self._base_url = "https://api.jina.ai/v1/rerank"

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        doc_contents = [doc.get("page_content", "") for doc in documents]
        payload = {"query": query, "documents": doc_contents, "model": self._model_name, "top_n": CHAT_CONFIG["top_k"]}
        try:
            response = requests.post(self._base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            rerank_results = response.json().get("results", [])
            
            original_docs_map = {doc.get("page_content", ""): doc for doc in documents}
            reranked_docs = []
            for res in rerank_results:
                doc_content = res.get("document")
                if doc_content in original_docs_map:
                    original_doc = original_docs_map[doc_content]
                    original_doc["rerank_score"] = res.get("relevance_score")
                    reranked_docs.append(original_doc)
            return reranked_docs
        except Exception as e:
            print(f"Jina Rerank出错: {e}")
            return documents

# =================================================================
# 4. 统一模型工厂 (UNIFIED MODEL FACTORY)
# =================================================================

class ModelFactory:
    @staticmethod
    def get_model_provider(model_type: str) -> Any:
        """
        根据模型类型和配置文件，创建并返回相应的模型提供商实例。
        """
        if model_type == "embedding":
            config_name = KB_CONFIG["active_embedding_configuration"]
            config = KB_CONFIG["embedding_configurations"][config_name]
            provider = config["provider"]
            model_name = config["model_name"]
            
            if provider == "google":
                return GoogleEmbedding(model_name=model_name)
            elif provider == "volcengine":
                return VolcengineEmbedding(model_name=model_name)
            elif provider in ["openai", "siliconflow", "qwen", "ollama"]:
                return OpenAICompatibleEmbedding(model_name=model_name, provider=provider)
            else:
                raise ValueError(f"不支持的Embedding提供商: {provider}")
            
        elif model_type == "llm":
            config_name = CHAT_CONFIG["active_llm_configuration"]
            config = CHAT_CONFIG["llm_configurations"][config_name]
            provider = config["provider"]
            model_name = config["model_name"]

            if provider == "google":
                return GoogleLLM(model_name=model_name)
            elif provider == "volcengine":
                return VolcengineLLM(model_name=model_name)
            elif provider == "grok":
                return GrokLLM(model_name=model_name)
            elif provider == "anthropic":
                return AnthropicLLM(model_name=model_name)
            elif provider in ["openai", "siliconflow", "qwen", "ollama", "lm-studio", "deepseek"]:
                return OpenAICompatibleLLM(model_name=model_name, provider=provider)
            else:
                raise ValueError(f"不支持的LLM提供商: {provider}")

        elif model_type == "rerank":
            config_name = CHAT_CONFIG["active_rerank_configuration"]
            config = CHAT_CONFIG["rerank_configurations"][config_name]
            provider = config["provider"]
            model_name = config["model_name"]
            if provider == "siliconflow":
                return SiliconflowRerankProvider(model_name=model_name)
            elif provider == "jina":
                return JinaRerankProvider(model_name=model_name)
            else:
                raise ValueError(f"不支持的Rerank提供商: {provider}")
            
        raise ValueError(f"不支持的模型类型: {model_type}")

# =================================================================
# 5. 辅助工具 (UTILITIES)
# =================================================================
logger = logging.getLogger(__name__)
_tokenizer: Any = None
_lock = Lock()

class GPT2Tokenizer:
    @staticmethod
    def get_encoder() -> Any:
        global _tokenizer
        if _tokenizer is None:
            with _lock:
                if _tokenizer is None:
                    try:
                        _tokenizer = tiktoken.get_encoding("gpt2")
                    except Exception as e:
                        logger.warning(f"无法加载gpt2 tokenizer: {e}")
                        _tokenizer = "failed"
        if _tokenizer == "failed":
            return None
        return _tokenizer

    @staticmethod
    def get_num_tokens(text: str) -> int:
        encoder = GPT2Tokenizer.get_encoder()
        if encoder is None:
            return len(text) // 4
        return len(encoder.encode(text))