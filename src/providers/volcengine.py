# -*- coding: utf-8 -*-
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Generator, List, cast

import requests

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    TextEmbeddingModel,
)
from src.utils.config import API_CONFIG


class VolcengineProvider(LargeLanguageModel, TextEmbeddingModel):
    """
    火山引擎模型提供商，统一处理豆包LLM和Embedding。
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._access_key = API_CONFIG.get("VOLC_ACCESS_KEY")
        self._secret_key = API_CONFIG.get("VOLC_SECRET_KEY")
        self._base_url = API_CONFIG.get("VOLC_BASE_URL")

        if not all([self._access_key, self._secret_key, self._base_url]):
            raise ValueError("火山引擎配置不完整：缺少 VOLC_ACCESS_KEY, VOLC_SECRET_KEY, 或 VOLC_BASE_URL。")

    def _sign_request(self, method: str, path: str, query: str, headers: Dict, body: bytes) -> Dict:
        """为火山引擎API请求生成签名"""
        secret_key = cast(str, self._secret_key) # 强制类型转换

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
        
        k_date = hmac.new(secret_key.encode('utf-8'), time.strftime('%Y%m%d', time.gmtime()).encode('utf-8'), hashlib.sha256).digest()
        k_region = hmac.new(k_date, b'cn-beijing', hashlib.sha256).digest()
        k_service = hmac.new(k_region, b'ml_maas', hashlib.sha256).digest()
        k_signing = hmac.new(k_service, b'request', hashlib.sha256).digest()
        
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        auth_header = f"HMAC-SHA256 Credential={self._access_key}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        headers['Authorization'] = auth_header
        headers['X-Date'] = current_time
        return headers

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: List[Dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        调用火山引擎LLM（如豆包）。
        API不支持流式响应，因此返回一个包含完整结果的生成器。
        """
        base_url = cast(str, self._base_url) # 强制类型转换
        url = f"{base_url.rstrip('/')}/api/v1/chat/completions"
        path = "/api/v1/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        body = json.dumps(payload).encode('utf-8')
        headers = {'Host': base_url.replace('https://', '').split('/')[0], 'Content-Type': 'application/json'}
        signed_headers = self._sign_request("POST", path, "", headers, body)
        
        try:
            response = requests.post(url, headers=signed_headers, data=body, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            yield content.strip()
        except Exception as e:
            print(f"火山引擎 LLM ({self._model_name}) 生成内容时出错: {e}")
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用火山引擎Embedding模型。
        """
        base_url = cast(str, self._base_url) # 强制类型转换
        url = f"{base_url.rstrip('/')}/api/v1/embeddings"
        path = "/api/v1/embeddings"
        body = json.dumps({"model": self._model_name, "input": texts}).encode('utf-8')
        headers = {'Host': base_url.replace('https://', '').split('/')[0], 'Content-Type': 'application/json'}
        signed_headers = self._sign_request("POST", path, "", headers, body)
        
        try:
            response = requests.post(url, headers=signed_headers, data=body, timeout=30)
            response.raise_for_status()
            result = response.json()
            return [item['embedding'] for item in result.get('data', [])]
        except Exception as e:
            print(f"火山引擎嵌入时出错 ({self._model_name}): {e}")
            return [[] for _ in texts]
