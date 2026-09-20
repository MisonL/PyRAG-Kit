import asyncio
import inspect
import os
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

from ..providers.__base__.model_provider import LargeLanguageModel, close_resource_sync
from ..providers.factory import ModelProviderFactory
from ..retrieval.vdb.base import VectorStoreBase
from ..retrieval.vdb.factory import VectorStoreFactory
from ..runtime.contracts import (
    RunConfig,
    SessionConfig,
    build_run_config,
    build_session_config,
)
from ..services.chat_service import ChatService
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..ui.config_menu import launch_config_editor
from ..ui.display_utils import display_chat_config, get_relative_path
from ..utils.config import get_settings
from ..utils.log_manager import get_chat_logger
from ..utils.security import redact_sensitive_text


def _safe_exception_text(exc: BaseException) -> str:
    """终端显示错误类型和上下文，同时遮蔽常见凭证格式。"""
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    message = redact_sensitive_text(message)
    return f"{type(exc).__name__}: {message[:240]}"


class Chatbot:
    def __init__(self, console: Console):
        self.console = console
        self.run_config: RunConfig = build_run_config(get_settings())
        self.session_config: SessionConfig = build_session_config(get_settings())
        self.vector_store: VectorStoreBase | None = None
        self.retrieval_service: RetrievalService | None = None
        self.llm_model: LargeLanguageModel | None = None
        self.chat_service: ChatService | None = None
        self.logger = get_chat_logger()

        self._initialize_vector_store()
        self.reload_llm()

    @property
    def chat_config(self) -> SessionConfig:
        return self.session_config

    @chat_config.setter
    def chat_config(self, value: SessionConfig | dict[str, Any]):
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
        previous_model = self.llm_model
        new_model: LargeLanguageModel | None = None
        try:
            llm_key = self.session_config.active_llm_configuration
            llm_configurations = getattr(self.session_config, "llm_configurations", None)
            if llm_configurations is None:
                new_model = ModelProviderFactory.get_llm_provider(llm_key)
            else:
                new_model = ModelProviderFactory.get_llm_provider(llm_key, llm_configurations)
            if self.retrieval_service is None:
                raise RuntimeError("检索服务尚未初始化。")
            chat_service = ChatService(new_model, self.retrieval_service)
            self.llm_model = new_model
            self.chat_service = chat_service
            if previous_model is not None and previous_model is not new_model:
                self._dispose_model(previous_model)
            return True
        except Exception as exc:  # noqa: BLE001 - reload boundary reports provider failures
            if new_model is not None and new_model is not self.llm_model:
                self._dispose_model(new_model)
            self.console.print(f"[bold red]重载 LLM 出错: {_safe_exception_text(exc)}[/bold red]")
            return False

    def _dispose_model(self, model: LargeLanguageModel) -> None:
        """关闭已替换或初始化失败的 Provider，避免连接池泄漏。"""
        try:
            close_resource_sync(model, "LLM Provider")
        except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
            self.logger.warning(
                "关闭 LLM Provider 失败: %s",
                redact_sensitive_text(str(exc)),
            )

    async def _adispose_model(self, model: LargeLanguageModel) -> None:
        """在异步会话中释放同步、异步 SDK 客户端。"""
        close = getattr(model, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
                return
            except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
                self.logger.warning(
                    "异步关闭 LLM Provider 失败: %s",
                    redact_sensitive_text(str(exc)),
                )
                return
        await asyncio.to_thread(self._dispose_model, model)

    def close(self) -> None:
        model = self.llm_model
        self.llm_model = None
        self.chat_service = None
        if model is not None:
            self._dispose_model(model)
        retrieval_service = self.retrieval_service
        self.retrieval_service = None
        close_retrieval = getattr(retrieval_service, "close", None)
        if callable(close_retrieval):
            try:
                close_retrieval()
            except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
                self.logger.warning(
                    "关闭检索服务失败: %s",
                    redact_sensitive_text(str(exc)),
                )

    async def aclose(self) -> None:
        model = self.llm_model
        self.llm_model = None
        self.chat_service = None
        if model is not None:
            await self._adispose_model(model)
        retrieval_service = self.retrieval_service
        self.retrieval_service = None
        close_retrieval = getattr(retrieval_service, "aclose", None)
        if callable(close_retrieval):
            try:
                result = close_retrieval()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
                self.logger.warning(
                    "异步关闭检索服务失败: %s",
                    redact_sensitive_text(str(exc)),
                )
        else:
            close_retrieval = getattr(retrieval_service, "close", None)
            if callable(close_retrieval):
                try:
                    result = close_retrieval()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
                    self.logger.warning(
                        "关闭检索服务失败: %s",
                        redact_sensitive_text(str(exc)),
                    )

    def apply_config_update(self, updated_config: SessionConfig | dict[str, Any], llm_needs_reload: bool) -> None:
        previous_config = deepcopy(self.session_config) if llm_needs_reload else None
        self.chat_config = updated_config
        if not llm_needs_reload:
            return
        if self.reload_llm():
            return
        if previous_config is not None:
            self.session_config = previous_config
        self.console.print("[bold yellow]LLM 切换失败，已保留当前模型配置。[/bold yellow]")

    async def apply_config_update_async(
        self,
        updated_config: SessionConfig | dict[str, Any],
        llm_needs_reload: bool,
    ) -> None:
        """异步会话中的配置更新，避免同步 close 遗漏异步连接池。"""
        previous_config = deepcopy(self.session_config) if llm_needs_reload else None
        self.chat_config = updated_config
        if not llm_needs_reload:
            return

        previous_model = self.llm_model
        new_model: LargeLanguageModel | None = None
        try:
            llm_key = self.session_config.active_llm_configuration
            configurations = getattr(self.session_config, "llm_configurations", None)
            new_model = ModelProviderFactory.get_llm_provider(
                llm_key, configurations
            ) if configurations is not None else ModelProviderFactory.get_llm_provider(llm_key)
            if self.retrieval_service is None:
                raise RuntimeError("检索服务尚未初始化。")
            chat_service = ChatService(new_model, self.retrieval_service)
            self.llm_model = new_model
            self.chat_service = chat_service
            if previous_model is not None and previous_model is not new_model:
                await self._adispose_model(previous_model)
            return
        except Exception as exc:  # noqa: BLE001 - reload boundary reports provider failures
            if new_model is not None and new_model is not self.llm_model:
                await self._adispose_model(new_model)
            if previous_config is not None:
                self.session_config = previous_config
            self.console.print(f"[bold red]重载 LLM 出错: {_safe_exception_text(exc)}[/bold red]")
            self.console.print("[bold yellow]LLM 切换失败，已保留当前模型配置。[/bold yellow]")

    async def _identify_intent_async(self, user_query: str) -> str:
        chat_service = getattr(self, "chat_service", None)
        if chat_service is None:
            return user_query
        return await chat_service.identify_intent(user_query)

    async def _retrieve_knowledge_async(self, retrieval_query: str) -> list[dict[str, Any]]:
        retrieval_service = getattr(self, "retrieval_service", None)
        if retrieval_service is None:
            return []
        return await retrieval_service.retrieve(retrieval_query, self.session_config, console=self.console)

    async def chat_async(self, user_input: str) -> AsyncGenerator[str, None]:
        intent = user_input
        retrieved_docs: list[dict[str, Any]] = []
        full_response = ""
        try:
            intent = await self._identify_intent_async(user_input)
            retrieved_docs = await self._retrieve_knowledge_async(intent)
            self.console.print(f"  [bold]意图:[/bold] [yellow]{intent}[/yellow]")
            self._display_retrieved_docs(retrieved_docs)

            chat_service = getattr(self, "chat_service", None)
            if chat_service is not None:
                async for chunk in chat_service.generate_reply(
                    user_input=user_input,
                    intent=intent,
                    documents=retrieved_docs,
                    session_config=self.session_config,
                ):
                    full_response += chunk
                    yield chunk
            elif (llm_model := getattr(self, "llm_model", None)) is not None:
                prompt = (
                    f"你是一个智能客服。用户问题: {user_input}\n"
                    f"意图: {intent}\n"
                    f"知识: {retrieved_docs}\n回答要简洁。"
                )
                invoke_kwargs: dict[str, Any] = {"prompt": prompt, "stream": True}
                supports_sampling = getattr(llm_model, "supports_sampling_option", None)
                if callable(supports_sampling) and supports_sampling("temperature"):
                    invoke_kwargs["temperature"] = self.session_config["chat_temperature"]
                async for chunk in llm_model.ainvoke(**invoke_kwargs):
                    full_response += chunk
                    yield chunk
            else:
                raise RuntimeError("聊天服务尚未初始化。")
            self.logger.info(
                "Query: %s | Intent: %s | Docs: %s\nResponse: %s",
                user_input,
                intent,
                len(retrieved_docs),
                full_response,
            )
        except Exception as exc:
            error_text = redact_sensitive_text(str(exc))
            self.logger.exception(
                "聊天请求处理失败: %s",
                error_text,
            )
            self.console.print(f"[red]LLM 异步生成出错: {_safe_exception_text(exc)}[/red]")
            # 已经输出部分内容时不要追加道歉文本，避免用户收到混合响应；
            # 具体异常已通过日志和控制台显式记录。
            if not full_response:
                yield "抱歉，处理请求时遇到错误。"

    def _display_retrieved_docs(self, docs: list[dict[str, Any]]):
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
    try:
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
                        editable_config = (
                            bot.chat_config.to_dict()
                            if isinstance(bot.chat_config, SessionConfig)
                            else dict(bot.chat_config)
                        )
                        llm_needs_reload, updated_config = await asyncio.to_thread(
                            launch_config_editor,
                            editable_config,
                        )
                        # 真实 Chatbot 使用异步重载以正确释放 SDK 连接池；保留
                        # 同步兼容入口，便于嵌入方或旧版替身实现配置编辑。
                        apply_async = getattr(bot, "apply_config_update_async", None)
                        if callable(apply_async):
                            result = apply_async(updated_config, llm_needs_reload)
                            if inspect.isawaitable(result):
                                await result
                        else:
                            apply_sync = getattr(bot, "apply_config_update", None)
                            if not callable(apply_sync):
                                raise AttributeError("Chatbot 缺少配置更新方法。")
                            apply_sync(updated_config, llm_needs_reload)
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
    finally:
        close = getattr(bot, "aclose", None)
        if close is not None:
            await close()


def start_chat_session():
    asyncio.run(start_chat_session_async())


if __name__ == "__main__":
    start_chat_session()
