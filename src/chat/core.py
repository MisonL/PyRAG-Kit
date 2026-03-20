# -*- coding: utf-8 -*-
import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

from ..providers.factory import ModelProviderFactory
from ..retrieval.vdb.base import VectorStoreBase
from ..retrieval.vdb.factory import VectorStoreFactory
from ..runtime.contracts import RunConfig, SessionConfig, build_run_config, build_session_config
from ..services.chat_service import ChatService
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..ui.config_menu import launch_config_editor
from ..ui.display_utils import display_chat_config, get_relative_path
from ..utils.config import get_settings
from ..utils.log_manager import get_chat_logger


class Chatbot:
    def __init__(self, console: Console):
        self.console = console
        self.run_config: RunConfig = build_run_config(get_settings())
        self.session_config: SessionConfig = build_session_config(get_settings())
        self.vector_store: Optional[VectorStoreBase] = None
        self.retrieval_service: Optional[RetrievalService] = None
        self.llm_model = None
        self.chat_service: Optional[ChatService] = None
        self.logger = get_chat_logger()

        self._initialize_vector_store()
        self.reload_llm()

    @property
    def chat_config(self) -> SessionConfig:
        return self.session_config

    @chat_config.setter
    def chat_config(self, value: SessionConfig | Dict[str, Any]):
        if isinstance(value, SessionConfig):
            self.session_config = value
            return
        if not hasattr(self, "session_config"):
            self.session_config = build_session_config(get_settings())
        for key, item in value.items():
            self.session_config[key] = item

    def _initialize_vector_store(self):
        self.vector_store = VectorStoreFactory.get_default_vector_store()
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=EmbeddingService(self.run_config),
        )
        self.console.print("[green]知识快照加载成功。[/green]")

    def reload_llm(self) -> bool:
        try:
            llm_key = self.session_config.active_llm_configuration
            self.llm_model = ModelProviderFactory.get_llm_provider(llm_key)
            if self.retrieval_service is None:
                raise RuntimeError("检索服务尚未初始化。")
            self.chat_service = ChatService(self.llm_model, self.retrieval_service)
            return True
        except Exception as exc:
            self.console.print(f"[bold red]重载 LLM 出错: {exc}[/bold red]")
            return False

    async def _identify_intent_async(self, user_query: str) -> str:
        if getattr(self, "chat_service", None) is None:
            return user_query
        return await self.chat_service.identify_intent(user_query)

    async def _retrieve_knowledge_async(self, retrieval_query: str) -> List[Dict[str, Any]]:
        if getattr(self, "retrieval_service", None) is None:
            return []
        return await self.retrieval_service.retrieve(retrieval_query, self.session_config, console=self.console)

    async def chat_async(self, user_input: str) -> AsyncGenerator[str, None]:
        intent = await self._identify_intent_async(user_input)
        retrieved_docs = await self._retrieve_knowledge_async(intent)
        self.console.print(f"  [bold]意图:[/bold] [yellow]{intent}[/yellow]")
        self._display_retrieved_docs(retrieved_docs)

        full_response = ""
        try:
            if getattr(self, "chat_service", None) is not None:
                async for chunk in self.chat_service.generate_reply(
                    user_input=user_input,
                    intent=intent,
                    documents=retrieved_docs,
                    session_config=self.session_config,
                ):
                    full_response += chunk
                    yield chunk
            else:
                prompt = (
                    f"你是一个智能客服。用户问题: {user_input}\n"
                    f"意图: {intent}\n"
                    f"知识: {retrieved_docs}\n回答要简洁。"
                )
                async for chunk in self.llm_model.ainvoke(
                    prompt=prompt,
                    stream=True,
                    temperature=self.session_config["chat_temperature"],
                ):
                    full_response += chunk
                    yield chunk
            self.logger.info(
                "Query: %s | Intent: %s | Docs: %s\nResponse: %s",
                user_input,
                intent,
                len(retrieved_docs),
                full_response,
            )
        except Exception as exc:
            self.console.print(f"[red]LLM 异步生成出错: {exc}[/red]")
            yield "抱歉，生成回复时遇到错误。"

    def _display_retrieved_docs(self, docs: List[Dict[str, Any]]):
        if not docs:
            self.console.print("[yellow]无相关文档。[/yellow]")
            return

        table = Table(title="[bold cyan]检索详情[/bold cyan]", show_header=True)
        table.add_column("来源", style="cyan")
        table.add_column("预览", style="white")
        table.add_column("得分", style="bold")
        for doc in docs:
            preview = doc.get("page_content", "")[:60].replace("\n", " ") + "..."
            source = get_relative_path(doc.get("metadata", {}).get("source", "N/A"))
            table.add_row(source, preview, f"{doc.get('score', 0):.4f}")
        self.console.print(table)


async def start_chat_session_async():
    console = Console()
    bot = Chatbot(console)
    session = PromptSession()

    if bot.llm_model:
        display_chat_config(console, bot.chat_config)
        console.print(f"客服已就绪 ([bold green]{bot.chat_config['active_llm_configuration']}[/bold green])")

        while True:
            try:
                user_query = await session.prompt_async(HTML('<skyblue><b>你: </b></skyblue>'))
                if not user_query.strip():
                    continue
                if user_query.lower() == "/quit":
                    break

                if user_query.lower() == "/config":
                    llm_needs_reload, updated_config = await asyncio.to_thread(
                        launch_config_editor,
                        bot.chat_config,
                    )
                    bot.chat_config = updated_config
                    if llm_needs_reload:
                        bot.reload_llm()
                    display_chat_config(console, bot.chat_config)
                    continue

                response_panel = Panel("", title="客服", border_style="green")
                full_response = ""
                with Live(response_panel, console=console, refresh_per_second=10) as live:
                    async for chunk in bot.chat_async(user_query):
                        full_response += chunk
                        live.update(Panel(Text(full_response), title="客服", border_style="green"))

            except (KeyboardInterrupt, EOFError):
                break
        console.print("[yellow]感谢使用，再见！[/yellow]")
    else:
        console.print("[red]模型初始化失败。[/red]")


def start_chat_session():
    asyncio.run(start_chat_session_async())


if __name__ == "__main__":
    start_chat_session()
