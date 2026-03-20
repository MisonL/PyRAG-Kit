import pytest
from unittest.mock import MagicMock, patch
from src.providers.google import GoogleProvider

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """强制模拟 settings 并在环境中设置 key"""
    mock_settings_instance = MagicMock()
    mock_settings_instance.google_api_key = "fake_key"
    monkeypatch.setattr("src.providers.google.get_settings", lambda: mock_settings_instance)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    return mock_settings_instance

@pytest.fixture
def mock_genai_client():
    with patch('src.providers.google.genai.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client

def test_google_provider_init(mock_settings):
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    assert provider._model_name == "gemini-1.5-flash"
    assert provider._client is None

def test_google_provider_get_client(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    client = provider._get_client()
    assert client == mock_genai_client
    # Verify Client initialization
    from src.providers.google import genai
    genai.Client.assert_called_once_with(api_key="fake_key")

def test_google_provider_invoke_non_stream(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="gemini-1.5-flash")
    
    # 直接模拟 _get_client 以规避 tenacity 装饰器可能带来的环境隔离问题
    with patch.object(GoogleProvider, '_get_client', return_value=mock_genai_client):
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_genai_client.models.generate_content.return_value = mock_response
        
        result = list(provider.invoke("test prompt", stream=False))
        
        assert result == ["Hello world"]
        mock_genai_client.models.generate_content.assert_called_once()

def test_google_provider_embed_documents(mock_settings, mock_genai_client):
    provider = GoogleProvider(model_name="embedding-001")
    
    # 直接模拟 _get_client
    with patch.object(GoogleProvider, '_get_client', return_value=mock_genai_client):
        mock_emb_1 = MagicMock()
        mock_emb_1.values = [0.1, 0.2]
        mock_emb_2 = MagicMock()
        mock_emb_2.values = [0.3, 0.4]
        
        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb_1, mock_emb_2]
        mock_genai_client.models.embed_content.return_value = mock_response
        
        texts = ["text1", "text2"]
        result = provider.embed_documents(texts)
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_genai_client.models.embed_content.assert_called_once()
