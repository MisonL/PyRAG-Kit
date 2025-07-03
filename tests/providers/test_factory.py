import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Generator, List
import sys # 导入 sys 模块

from src.utils.config import Settings, get_settings
from src.providers.__base__.model_provider import LargeLanguageModel, TextEmbeddingModel, RerankModel
from src.models.document import Document
from src.utils.config import ModelDetail
from src.providers.factory import ModelProviderFactory # 在这里导入 ModelProviderFactory

# Mock 类定义
class MockLLMProvider(LargeLanguageModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt: str, system_prompt: str | None = "You are a helpful assistant.", tools: List[dict[str, Any]] | None = None, stream: bool = True, temperature: float = 0.7) -> Generator[str, None, None]:
        """模拟 LLM 聊天响应"""
        if stream:
            yield f"Mock LLM stream response for {self.model_name}"
        else:
            yield f"Mock LLM response for {self.model_name}"

class MockEmbeddingProvider(TextEmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """模拟文档嵌入"""
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """模拟查询嵌入"""
        return [0.2] * 10

class MockRerankProvider(RerankModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """模拟重排，返回原始顺序和递减的分数"""
        indices = list(range(len(documents)))
        scores = [1.0 - i * 0.1 for i in range(len(documents))]
        return indices[:top_n], scores[:top_n]

@pytest.fixture(scope="module")
def mock_settings():
    """模拟全局设置对象"""
    mock_llm_config = ModelDetail(provider="mock_llm", model_name="mock-llm-model")
    mock_embedding_config = ModelDetail(provider="mock_embedding", model_name="mock-embedding-model")
    mock_rerank_config = ModelDetail(provider="mock_rerank", model_name="mock-rerank-model")
    mock_siliconflow_rerank_config = ModelDetail(provider="siliconflow", model_name="mock-siliconflow-rerank-model")

    mock_settings_instance = MagicMock(spec=Settings)
    mock_settings_instance.llm_configurations = {"default_llm": mock_llm_config}
    mock_settings_instance.embedding_configurations = {"default_embedding": mock_embedding_config}
    mock_settings_instance.rerank_configurations = {
        "default_rerank": mock_rerank_config,
        "siliconflow_rerank_key": mock_siliconflow_rerank_config
    }
    mock_settings_instance.siliconflow_api_key = "mock_siliconflow_api_key"
    mock_settings_instance.siliconflow_base_url = "http://mock-siliconflow-base-url.com"
    mock_settings_instance.google_api_key = "mock_google_api_key"

    return mock_settings_instance

@pytest.fixture(scope="function", autouse=True)
def patch_settings(mock_settings, monkeypatch):
    """在测试期间替换全局 get_settings 函数，使其返回模拟的 settings 对象，并修改 _provider_map"""
    with patch('src.utils.config.get_settings', return_value=mock_settings):
        # 清除 ModelProviderFactory 及其依赖模块的缓存
        # 这确保了 ModelProviderFactory 在每次测试时都使用最新的模拟配置
        modules_to_clear = [
            'src.providers.factory',
            'src.providers.google',
            'src.providers.openai',
            'src.providers.anthropic',
            'src.providers.qwen',
            'src.providers.volcengine',
            'src.providers.siliconflow',
            'src.providers.ollama',
            'src.providers.lm_studio',
            'src.providers.deepseek',
            'src.providers.grok',
            'src.providers.jina',
            'src.providers.siliconflow_rerank',
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # 重新导入 ModelProviderFactory，确保它加载的是最新的版本
        from src.providers.factory import ModelProviderFactory as ReloadedModelProviderFactory
        
        # 直接模拟 _get_provider_class 方法
        def mock_get_provider_class(provider_name: str):
            if provider_name == "mock_llm":
                return MockLLMProvider
            elif provider_name == "mock_embedding":
                return MockEmbeddingProvider
            elif provider_name == "mock_rerank":
                return MockRerankProvider
            elif provider_name == "siliconflow_rerank":
                return MockRerankProvider
            elif provider_name == "google":
                return MockLLMProvider
            else:
                raise ValueError(f"不支持的模型提供商: {provider_name}")

        monkeypatch.setattr(ReloadedModelProviderFactory, "_get_provider_class", mock_get_provider_class)

        # 将重新导入的 ModelProviderFactory 赋值给全局 ModelProviderFactory，以便测试函数使用
        global ModelProviderFactory
        ModelProviderFactory = ReloadedModelProviderFactory
        yield

# 测试用例
def test_get_llm_provider_success():
    """测试成功获取 LLM 提供商"""
    llm_provider = ModelProviderFactory.get_llm_provider("default_llm")
    assert isinstance(llm_provider, MockLLMProvider)
    assert llm_provider.model_name == "mock-llm-model"

def test_get_llm_provider_not_found():
    """测试获取不存在的 LLM 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在LLM配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_llm_provider("non_existent_key")

def test_get_embedding_provider_success():
    """测试成功获取 Embedding 提供商"""
    embedding_provider = ModelProviderFactory.get_embedding_provider("default_embedding")
    assert isinstance(embedding_provider, MockEmbeddingProvider)
    assert embedding_provider.model_name == "mock-embedding-model"

def test_get_embedding_provider_not_found():
    """测试获取不存在的 Embedding 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在Embedding配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_embedding_provider("non_existent_key")

def test_get_rerank_provider_success():
    """测试成功获取 Rerank 提供商"""
    rerank_provider = ModelProviderFactory.get_rerank_provider("default_rerank")
    assert isinstance(rerank_provider, MockRerankProvider)
    assert rerank_provider.model_name == "mock-rerank-model"

def test_get_rerank_provider_siliconflow_mapping():
    """测试 Siliconflow Rerank 提供商的特殊映射"""
    rerank_provider = ModelProviderFactory.get_rerank_provider("siliconflow_rerank_key")
    assert isinstance(rerank_provider, MockRerankProvider)
    assert rerank_provider.model_name == "mock-siliconflow-rerank-model"

def test_get_rerank_provider_not_found():
    """测试获取不存在的 Rerank 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在Rerank配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_rerank_provider("non_existent_key")

def test_unsupported_provider_type():
    """测试 _get_provider_class 方法处理不支持的提供商名称"""
    with pytest.raises(ValueError, match="不支持的模型提供商: unsupported_provider"):
        ModelProviderFactory._get_provider_class("unsupported_provider")
