# -*- coding: utf-8 -*-
import importlib
from typing import Any, Dict, Type

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    RerankModel,
    TextEmbeddingModel,
)
from src.utils.config import (
    CHAT_CONFIG,
    EMBEDDING_CONFIGS,
    LLM_CONFIGS,
    RERANK_CONFIGS,
)

class ModelProviderFactory:
    """模型提供商工厂"""

    _provider_map: Dict[str, Dict[str, Any]] = {
        # LLM & Embedding Providers
        "google": {"module": "src.providers.google", "class": "GoogleProvider"},
        "openai": {"module": "src.providers.openai", "class": "OpenAIProvider"},
        "anthropic": {"module": "src.providers.anthropic", "class": "AnthropicProvider"},
        "qwen": {"module": "src.providers.qwen", "class": "QwenProvider"},
        "volcengine": {"module": "src.providers.volcengine", "class": "VolcengineProvider"},
        "siliconflow": {"module": "src.providers.siliconflow", "class": "SiliconflowProvider"},
        "ollama": {"module": "src.providers.ollama", "class": "OllamaProvider"},
        "lm-studio": {"module": "src.providers.lm_studio", "class": "LMStudioProvider"},
        "deepseek": {"module": "src.providers.deepseek", "class": "DeepSeekProvider"},
        "grok": {"module": "src.providers.grok", "class": "GrokProvider"},
        
        # Rerank Providers
        "jina": {"module": "src.providers.jina", "class": "JinaProvider"},
        "siliconflow_rerank": {"module": "src.providers.siliconflow_rerank", "class": "SiliconflowRerankProvider"},
    }

    @staticmethod
    def _get_provider_class(provider_name: str) -> Type:
        """动态导入并返回提供商类"""
        if provider_name not in ModelProviderFactory._provider_map:
            raise ValueError(f"不支持的模型提供商: {provider_name}")

        provider_info = ModelProviderFactory._provider_map[provider_name]
        module = importlib.import_module(provider_info["module"])
        return getattr(module, provider_info["class"])

    @staticmethod
    def get_llm_provider(provider_key: str) -> LargeLanguageModel:
        """获取一个语言模型提供商实例"""
        if provider_key not in LLM_CONFIGS:
            raise ValueError(f"在LLM配置中未找到key: {provider_key}")

        config = LLM_CONFIGS[provider_key]
        provider_name = config["provider"]
        model_name = config["model_name"]

        ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
        return ProviderClass(model_name=model_name)

    @staticmethod
    def get_embedding_provider(provider_key: str) -> TextEmbeddingModel:
        """获取一个文本向量化模型提供商实例"""
        if provider_key not in EMBEDDING_CONFIGS:
            raise ValueError(f"在Embedding配置中未找到key: {provider_key}")

        config = EMBEDDING_CONFIGS[provider_key]
        provider_name = config["provider"]
        model_name = config["model_name"]

        ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
        return ProviderClass(model_name=model_name)

    @staticmethod
    def get_rerank_provider(provider_key: str) -> RerankModel:
        """获取一个Rerank模型提供商实例"""
        if provider_key not in RERANK_CONFIGS:
            raise ValueError(f"在Rerank配置中未找到key: {provider_key}")

        config = RERANK_CONFIGS[provider_key]
        # Rerank提供商的key可能与LLM/Embedding提供商的key冲突（如siliconflow）
        # 因此，我们在这里使用一个特殊的key，或者直接在配置中指定provider_map的key
        # 为了简单起见，我们假设rerank的provider name是唯一的
        provider_name = config["provider"]
        if provider_name == "siliconflow":
            provider_name = "siliconflow_rerank" # 映射到唯一的rerank provider

        model_name = config["model_name"]

        ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
        return ProviderClass(model_name=model_name)
