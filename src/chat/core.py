import asyncio
import inspect
import os
from collections.abc import AsyncGenerator, Mapping
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
from ..ui.display_utils import (
    display_chat_config,
    document_content,
    document_score,
    document_source,
    get_relative_path,
    markup_text,
)
from ..utils.config import get_settings, require_chat_temperature, safe_repr
from ..utils.log_manager import get_chat_logger
from ..utils.security import safe_exception_text


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
        # 形态先判：标注是 ``SessionConfig | dict``，但运行时不受约束。非 Mapping
        # 取值在此抛裸 ``TypeError: 'int' object is not iterable``（下面两条路径
        # 的 ``.items()``／``set_many`` 迭代），而非法**字段名**给的是带指引的
        # ``ValueError``。同一个「配置写错」的收场不该取决于它是标量还是映射。
        if not isinstance(value, Mapping):
            raise ValueError(
                f"chat_config 必须是 SessionConfig 或字段映射，但得到 {safe_repr(value)}。"
            )
        if not hasattr(self, "session_config"):
            self.session_config = build_session_config(get_settings())
        # 先在副本上逐个写入，全部通过后再整体替换。
        #
        # 直接循环写 ``self.session_config`` 不是原子的：任一字段被拒时，
        # 前面的字段已经落进去了——实测 ``{"top_k": 42, "retrieval_method":
        # "不存在"}`` 会留下「top_k 已改成 42、检索模式还是旧的」这种改了一半
        # 的状态，而调用方只看到一个异常，以为整批都没生效。
        #
        # 用 ``deepcopy`` 而不是 ``SessionConfig(**to_dict())``：``session_config``
        # 的类型是运行时可替换的（嵌入方可以传入自己的映射实现），``__setitem__``
        # 是唯一被要求的接口，重建具体类型会把这类实现挡在外面。
        candidate = deepcopy(self.session_config)
        # 有 ``set_many`` 就用它：逐个 ``__setitem__`` 会拿**旧**的兄弟字段值
        # 做组合校验，于是「top_k × multiplier」这类上界约束会因为中间态瞬时
        # 越界而拒绝一个合法终态，且结论随 dict 键顺序变化。
        set_many = getattr(candidate, "set_many", None)
        if callable(set_many):
            set_many(value)
        else:
            for key, item in value.items():
                candidate[key] = item
        self.session_config = candidate

    def _initialize_vector_store(self):
        self.vector_store = VectorStoreFactory.get_default_vector_store()
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=EmbeddingService(self.run_config),
        )
        # 不能无条件报「加载成功」：没有任何快照/旧版 pkl 时 store 是空的
        # （documents=0、faiss_index=None），用户会先看到成功再看到永远是
        # 「无相关文档」，把构建缺失误诊成检索质量差。
        chunk_count = len(getattr(self.vector_store, "documents", []) or [])
        if chunk_count:
            self.console.print(
                markup_text("[green]知识快照加载成功，共 {} 个分块。[/green]", chunk_count)
            )
        else:
            self.console.print(
                "[bold yellow]未找到可用的知识快照，检索将始终返回空结果。[/bold yellow]"
            )
            self.console.print(
                "[yellow]请先在主菜单执行「知识库文档向量化处理」构建快照。[/yellow]"
            )

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
            self.console.print(
                markup_text("[bold red]重载 LLM 出错: {}[/bold red]", safe_exception_text(exc))
            )
            return False

    def _dispose_model(self, model: LargeLanguageModel) -> None:
        """关闭已替换或初始化失败的 Provider，避免连接池泄漏。"""
        try:
            close_resource_sync(model, "LLM Provider")
        except Exception as exc:  # noqa: BLE001 - teardown continues after SDK failures
            self.logger.warning(
                "关闭 LLM Provider 失败: %s",
                safe_exception_text(exc),
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
                    safe_exception_text(exc),
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
                    safe_exception_text(exc),
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
                    safe_exception_text(exc),
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
                        safe_exception_text(exc),
                    )

    def apply_config_update(
        self, updated_config: SessionConfig | dict[str, Any], llm_needs_reload: bool
    ) -> None:
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
            new_model = (
                ModelProviderFactory.get_llm_provider(llm_key, configurations)
                if configurations is not None
                else ModelProviderFactory.get_llm_provider(llm_key)
            )
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
            self.console.print(
                markup_text("[bold red]重载 LLM 出错: {}[/bold red]", safe_exception_text(exc))
            )
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
        return await retrieval_service.retrieve(
            retrieval_query, self.session_config, console=self.console
        )

    async def chat_async(self, user_input: str) -> AsyncGenerator[str, None]:
        intent = user_input
        retrieved_docs: list[dict[str, Any]] = []
        full_response = ""
        try:
            intent = await self._identify_intent_async(user_input)
            retrieved_docs = await self._retrieve_knowledge_async(intent)
            self.console.print(markup_text("  [bold]意图:[/bold] [yellow]{}[/yellow]", intent))
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
                    # 与 ``ChatService.generate_reply`` 同判据：这里读的是可直写的
                    # 字段，非法取值同样会静默进请求 kwargs。
                    invoke_kwargs["temperature"] = require_chat_temperature(
                        self.session_config["chat_temperature"]
                    )
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
            error_text = safe_exception_text(exc)
            self.logger.exception(
                "聊天请求处理失败: %s",
                error_text,
            )
            self.console.print(
                markup_text("[red]LLM 异步生成出错: {}[/red]", safe_exception_text(exc))
            )
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
            # 字段形态由 document_* 助手收敛（检索结果可以来自第三方 store，
            # 不经 FaissStore 的载入期归一化）；预览与来源还可能含字面量
            # ``[/]`` 之类的标记形态，因此用 Text 包装后再交给 add_row。
            preview = document_content(doc, "")[:60].replace("\n", " ") + "..."
            source = get_relative_path(document_source(doc))
            table.add_row(Text(source), Text(preview), f"{document_score(doc):.4f}")
        self.console.print(table)


async def start_chat_session_async():
    console = Console()
    bot = Chatbot(console)
    try:
        session = PromptSession()

        if bot.llm_model:
            display_chat_config(console, bot.chat_config)
            console.print(
                markup_text(
                    "客服已就绪 ([bold green]{}[/bold green])",
                    bot.chat_config["active_llm_configuration"],
                )
            )

            while True:
                try:
                    user_query = await session.prompt_async(HTML("<skyblue><b>你: </b></skyblue>"))
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
                        # 配置应用失败**不能**终止整个会话。``SessionConfig`` 的
                        # 校验器会拒绝非法取值（``ValueError``），而编辑器的动态
                        # 上界与它是两套独立实现，任一侧演化就会命中这条路径——
                        # 实测编辑器返回「非法字段 + 非法值」时，异常穿透内层只接
                        # ``(KeyboardInterrupt, EOFError)`` 的 ``try``，用户被丢回
                        # 主菜单并丢失会话。同文件的 ``chat_async`` 对异常是恢复的
                        # （打印后继续下一轮），这里与之对齐：报错、保留原配置、
                        # 回到提示符。
                        try:
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
                        except (ValueError, KeyError) as exc:
                            # 经 ``getattr`` 取 logger：``bot`` 的类型是运行时
                            # 可替换的（嵌入方可以传入自己的实现），同文件其它
                            # 位置也用 ``getattr`` 探测 ``apply_config_update``／
                            # ``aclose``。诊断日志取不到时不能反过来把恢复路径
                            # 变成新的失败点。
                            logger = getattr(bot, "logger", None)
                            if logger is not None:
                                logger.warning(
                                    "配置应用失败，保留原配置: %s", safe_exception_text(exc)
                                )
                            console.print(
                                markup_text(
                                    "[bold yellow]配置未生效: {}[/bold yellow] 已保留当前配置。",
                                    safe_exception_text(exc),
                                )
                            )
                            continue
                        display_chat_config(console, bot.chat_config)
                        continue

                    response_panel = Panel("", title="客服", border_style="green")
                    full_response = ""
                    with Live(response_panel, console=console, refresh_per_second=10) as live:
                        async for chunk in bot.chat_async(user_query):
                            full_response += chunk
                            live.update(
                                Panel(Text(full_response), title="客服", border_style="green")
                            )

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
