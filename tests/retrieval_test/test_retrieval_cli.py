import asyncio

from src.retrieval_test.core import run_retrieval_test_async


class FakePromptSession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def prompt_async(self, *_args, **_kwargs):
        return next(self._responses)


def test_run_retrieval_test_async_can_exit(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["/quit"]),
    )
    monkeypatch.setattr(
        "src.retrieval_test.core.VectorStoreFactory.get_default_vector_store",
        lambda: object(),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)

    asyncio.run(run_retrieval_test_async())


def test_run_retrieval_test_async_prints_error_when_retrieval_fails(monkeypatch):
    printed_messages = []

    class FailingRetrievalService:
        def __init__(self, *_args, **_kwargs):
            return None

        async def retrieve(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.retrieval_test.core.PromptSession",
        lambda: FakePromptSession(["query", "/quit"]),
    )
    monkeypatch.setattr("src.retrieval_test.core.ExcelLogger", lambda: None)
    monkeypatch.setattr("src.retrieval_test.core.RetrievalService", FailingRetrievalService)
    monkeypatch.setattr("src.retrieval_test.core.EmbeddingService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.retrieval_test.core.VectorStoreFactory.get_default_vector_store", lambda: object())
    monkeypatch.setattr("src.retrieval_test.core.console.print", lambda message, *args, **kwargs: printed_messages.append(str(message)))

    asyncio.run(run_retrieval_test_async())

    assert any("召回测试出错" in message for message in printed_messages)
