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
