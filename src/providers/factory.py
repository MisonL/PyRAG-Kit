# -*- coding: utf-8 -*-
import importlib
from typing import Any, Dict, Type

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    RerankModel,
    TextEmbeddingModel,
)
from src.utils.config import get_settings # 导入 get_settings 函数
from src.utils.log_manager import get_module_logger # 导入日志管理器

logger = get_module_logger(__name__) # 获取当前模块的日志器

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
            logger.error(f"不支持的模型提供商: {provider_name}")
            raise ValueError(f"不支持的模型提供商: {provider_name}")

        provider_info = ModelProviderFactory._provider_map[provider_name]
        try:
            module = importlib.import_module(provider_info["module"])
            ProviderClass = getattr(module, provider_info["class"])
            logger.debug(f"成功加载提供商类: {provider_info['class']} from {provider_info['module']}")
            return ProviderClass
        except ImportError as e:
            logger.error(f"导入提供商模块失败: {provider_info['module']} - {e}")
            raise ImportError(f"无法加载提供商模块: {provider_info['module']}") from e
        except AttributeError as e:
            logger.error(f"在模块中找不到提供商类: {provider_info['class']} in {provider_info['module']} - {e}")
            raise AttributeError(f"无法找到提供商类: {provider_info['class']}") from e

    @staticmethod
    def get_llm_provider(provider_key: str) -> LargeLanguageModel:
        """获取一个语言模型提供商实例"""
        current_settings = get_settings() # 获取当前配置
        if provider_key not in current_settings.llm_configurations:
            logger.error(f"在LLM配置中未找到key: {provider_key}")
            raise ValueError(f"在LLM配置中未找到key: {provider_key}")

        config = current_settings.llm_configurations[provider_key]
        provider_name = config.provider
        model_name = config.model_name
        logger.info(f"正在获取LLM提供商: {provider_name}, 模型: {model_name}")

        try:
            ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
            instance = ProviderClass(model_name=model_name)
            logger.info(f"成功获取LLM提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(f"获取LLM提供商实例失败: {provider_name} ({model_name}) - {e}")
            raise

    @staticmethod
    def get_embedding_provider(provider_key: str) -> TextEmbeddingModel:
        """获取一个文本向量化模型提供商实例"""
        current_settings = get_settings() # 获取当前配置
        if provider_key not in current_settings.embedding_configurations:
            logger.error(f"在Embedding配置中未找到key: {provider_key}")
            raise ValueError(f"在Embedding配置中未找到key: {provider_key}")

        config = current_settings.embedding_configurations[provider_key]
        provider_name = config.provider
        model_name = config.model_name
        logger.info(f"正在获取Embedding提供商: {provider_name}, 模型: {model_name}")

        try:
            ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
            instance = ProviderClass(model_name=model_name)
            logger.info(f"成功获取Embedding提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(f"获取Embedding提供商实例失败: {provider_name} ({model_name}) - {e}")
            raise

    @staticmethod
    def get_rerank_provider(provider_key: str) -> RerankModel:
        """获取一个Rerank模型提供商实例"""
        current_settings = get_settings() # 获取当前配置
        if provider_key not in current_settings.rerank_configurations:
            logger.error(f"在Rerank配置中未找到key: {provider_key}")
            raise ValueError(f"在Rerank配置中未找到key: {provider_key}")

        config = current_settings.rerank_configurations[provider_key]
        # Rerank提供商的key可能与LLM/Embedding提供商的key冲突（如siliconflow）
        # 因此，我们在这里使用一个特殊的key，或者直接在配置中指定provider_map的key
        # 为了简单起见，我们假设rerank的provider name是唯一的
        provider_name = config.provider
        if provider_name == "siliconflow":
            provider_name = "siliconflow_rerank" # 映射到唯一的rerank provider
        
        model_name = config.model_name
        logger.info(f"正在获取Rerank提供商: {provider_name}, 模型: {model_name}")

        try:
            ProviderClass = ModelProviderFactory._get_provider_class(provider_name)
            instance = ProviderClass(model_name=model_name)
            logger.info(f"成功获取Rerank提供商实例: {provider_name} ({model_name})")
            return instance
        except Exception as e:
            logger.error(f"获取Rerank提供商实例失败: {provider_name} ({model_name}) - {e}")
            raise
