import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.providers.openai_compatible import OpenAICompatibleProvider
from src.services.embedding_service import EmbeddingService
from src.utils.config import ModelDetail


def test_embedding_service_passes_runtime_configurations_to_factory(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    factory = MagicMock(return_value=provider)
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        factory,
    )

    service = EmbeddingService(run_config)

    assert service._get_model() is provider
    factory.assert_called_once_with("custom", run_config.embedding_configurations)


def test_embedding_service_rejects_vectors_that_do_not_match_input_count(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    provider.aembed_documents = AsyncMock(return_value=[[1.0, 2.0]])
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    service = EmbeddingService(run_config)

    with pytest.raises(ValueError, match="向量数量与输入文本数量不一致"):
        asyncio.run(service.embed_texts(["one", "two"]))


@pytest.mark.parametrize("value", [[], [[1.0, 2.0], [3.0, 4.0]], 1.0])
def test_embedding_service_rejects_malformed_query_vectors(monkeypatch, value):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    provider.aembed_query = AsyncMock(return_value=value)
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    with pytest.raises((TypeError, ValueError), match="向量|float"):
        asyncio.run(EmbeddingService(run_config).embed_query("one"))


def test_embedding_service_does_not_replay_model_options(monkeypatch):
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(
                provider="openai",
                model_name="text-embedding-3-small",
                options={"dimensions": 512, "encoding_format": "float"},
            )
        },
    )
    provider = MagicMock()
    provider.aembed_documents = AsyncMock(return_value=[[1.0, 2.0]])
    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )

    service = EmbeddingService(run_config)
    asyncio.run(service.embed_texts(["one"]))

    provider.aembed_documents.assert_awaited_once_with(["one"])


@pytest.mark.parametrize("method", ["embed_texts", "embed_query", "embed_in_batches"])
@pytest.mark.parametrize("options", [
    {"dimensions": 2},
    {"encoding_format": "float"},
    {"user": "embedding-test"},
    {"extra_body": {"vendor_flag": True}},
])
def test_embedding_service_model_options_reach_sdk_once(monkeypatch, method, options):
    requests = []

    class Embeddings:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in request["input"]]
            )

    provider = object.__new__(OpenAICompatibleProvider)
    provider._provider = "openai"
    provider._model_name = "text-embedding-3-small"
    provider._options = options
    provider._get_aclient = lambda: SimpleNamespace(embeddings=Embeddings())

    monkeypatch.setattr(
        "src.services.embedding_service.ModelProviderFactory.get_embedding_provider",
        lambda *_args: provider,
    )
    service = EmbeddingService(SimpleNamespace(
        default_embedding_provider="custom", kb_embedding_batch_size=1,
        embedding_configurations={"custom": ModelDetail(
            provider="openai", model_name="text-embedding-3-small", options=options,
        )},
    ))
    value = "one" if method == "embed_query" else ["one", "two"]
    result = asyncio.run(getattr(service, method)(value))
    assert result.dtype == np.float32
    assert result.shape == ((2,) if method == "embed_query" else (2, 2))

    assert len(requests) == (2 if method == "embed_in_batches" else 1)
    for payload in requests:
        for key, value in options.items():
            if key == "extra_body":
                assert payload["extra_body"] == {"vendor_flag": True}
            else:
                assert payload[key] == value


def test_embedding_service_close_disposes_cached_model():
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    service = EmbeddingService(run_config)
    service._embedding_model = provider

    service.close()

    provider.close.assert_called_once_with()
    assert service._embedding_model is None


def test_embedding_service_aclose_disposes_cached_model_async():
    run_config = SimpleNamespace(
        default_embedding_provider="custom",
        embedding_configurations={
            "custom": ModelDetail(provider="local-hash", model_name="local-hash-32")
        },
    )
    provider = MagicMock()
    closed = False

    async def aclose():
        nonlocal closed
        closed = True

    provider.aclose = aclose
    service = EmbeddingService(run_config)
    service._embedding_model = provider

    asyncio.run(service.aclose())

    assert closed
    assert service._embedding_model is None
