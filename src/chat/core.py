# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator
import copy # 导入 copy 模块

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.text import Text
from rich.live import Live
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

import asyncio
# 使用相对导入
from ..providers.factory import ModelProviderFactory
from ..providers.__base__.model_provider import LargeLanguageModel
from ..utils.config import get_settings, RetrievalMethod
from ..ui.config_menu import launch_config_editor
from ..ui.display_utils import display_chat_config
from ..retrieval.retriever import aretrieve_documents  # 使用异步检索
from ..retrieval.vdb.base import VectorStoreBase
from ..retrieval.vdb.factory import VectorStoreFactory
from ..utils.log_manager import get_chat_logger

# =================================================================
# 2. 聊天机器人核心 (CHATBOT CORE)
# =================================================================
class Chatbot:
    def __init__(self, console: Console):
        self.console = console
        self.vector_store: Optional[VectorStoreBase] = None
        self.llm_model: Optional[LargeLanguageModel] = None
        self.logger = get_chat_logger()
        
        current_settings = get_settings()
        self.chat_config = {
            "retrieval_method": current_settings.chat_retrieval_method,
            "vector_weight": current_settings.chat_vector_weight,
            "keyword_weight": current_settings.chat_keyword_weight,
            "hybrid_fusion_strategy": current_settings.hybrid_fusion_strategy,
            "retrieval_candidate_multiplier": current_settings.retrieval_candidate_multiplier,
            "rerank_enabled": current_settings.chat_rerank_enabled,
            "top_k": current_settings.chat_top_k,
            "score_threshold": current_settings.chat_score_threshold,
            "active_llm_configuration": current_settings.default_llm_provider,
            "active_rerank_configuration": current_settings.default_rerank_provider,
            "llm_configurations": copy.deepcopy(current_settings.llm_configurations),
            "rerank_configurations": copy.deepcopy(current_settings.rerank_configurations),
            "chat_temperature": current_settings.chat_temperature,
        }
        self._initialize_vector_store()
        self.reload_llm()

    def _initialize_vector_store(self):
        self.vector_store = VectorStoreFactory.get_default_vector_store()
        try:
            current_settings = get_settings()
            self.vector_store.load(current_settings.pkl_path)
            self.console.print(f"[green]知识库 '{os.path.basename(current_settings.pkl_path)}' 加载成功。[/green]")
        except FileNotFoundError:
            self.console.print(f"[bold red]警告：知识库未找到。请运行向量化脚本。[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]加载知识库出错: {e}[/bold red]")

    def reload_llm(self) -> bool:
        try:
            active_llm_key = self.chat_config["active_llm_configuration"]
            self.llm_model = ModelProviderFactory.get_llm_provider(active_llm_key)
            return True if self.llm_model else False
        except Exception as e:
            self.console.print(f"[bold red]重载LLM出错: {e}[/bold red]")
            return False

    async def _identify_intent_async(self, user_query: str) -> str:
        """异步意图识别。"""
        if not self.llm_model: return user_query
        prompt = f"你是一个意图识别助手。判断用户意图。\n用户问题: {user_query}\n直接输出意图简述。"
        try:
            # 异步调用 invoke
            response = ""
            async for chunk in self.llm_model.ainvoke(
                prompt=prompt,
                stream=False,
                temperature=0.1
            ):
                response += chunk
            return response.strip() or user_query
        except Exception as e:
            logger.error(f"异步意图识别失败: {e}")
            return user_query

    async def _retrieve_knowledge_async(self, query: str) -> List[Dict[str, Any]]:
        """异步知识检索。"""
        if not self.vector_store: return []
        return await aretrieve_documents(
            query=query,
            vector_store=self.vector_store,
            console=self.console,
            retrieval_method=self.chat_config['retrieval_method'],
            top_k=self.chat_config['top_k'],
            vector_weight=self.chat_config['vector_weight'],
            keyword_weight=self.chat_config['keyword_weight'],
            rerank_enabled=self.chat_config['rerank_enabled'],
            active_rerank_configuration=self.chat_config['active_rerank_configuration'],
            score_threshold=self.chat_config['score_threshold'],
            fusion_strategy=self.chat_config['hybrid_fusion_strategy'],
            candidate_multiplier=self.chat_config['retrieval_candidate_multiplier'],
        )

    async def chat_async(self, user_input: str) -> "AsyncGenerator[str, None]":
        """核心异步聊天流。"""
        # 1. 意图识别
        intent = await self._identify_intent_async(user_input)
        self.console.print(f"  [bold]意图:[/bold] [yellow]{intent}[/yellow]")
        
        # 2. 检索
        retrieved_docs = await self._retrieve_knowledge_async(user_input)
        self._display_retrieved_docs(retrieved_docs)
        
        # 3. 构造 Prompt
        context_str = "\n".join([f"来源: {doc['metadata']['source']}\n内容: {doc.get('page_content', '')}" for doc in retrieved_docs])
        if context_str:
            prompt = f"你是一个智能客服。根据知识回答。\n意图: {intent}\n问题: {user_input}\n知识:\n{context_str}\n回答要简洁。"
        else:
            prompt = f"你是一个智能客服。用户问题: {user_input}\n没找到相关知识。请礼貌告知。"

        # 4. 异步流式生成
        full_response = ""
        context_summary = "\n".join([f"[{doc['metadata']['source']}]" for doc in retrieved_docs])
        
        try:
            async for chunk in self.llm_model.ainvoke(
                prompt=prompt,
                stream=True,
                temperature=self.chat_config['chat_temperature']
            ):
                full_response += chunk
                yield chunk
            
            # 记录日志
            self.logger.info(f"Query: {user_input} | Intent: {intent} | Docs: {len(retrieved_docs)}\nResponse: {full_response}")
        except Exception as e:
            self.console.print(f"[red]LLM 异步生成出错: {e}[/red]")
            yield "抱歉，生成回复时遇到错误。"

    def _display_retrieved_docs(self, docs: List[Dict]):
        if not docs:
            self.console.print("[yellow]无相关文档。[/yellow]")
            return
        
        table = Table(title="[bold cyan]检索详情[/bold cyan]", show_header=True)
        table.add_column("来源", style="cyan")
        table.add_column("预览", style="white")
        table.add_column("得分", style="bold")
        for doc in docs:
            preview = doc.get("page_content", "")[:60].replace("\n", " ") + "..."
            table.add_row(doc["metadata"]["source"], preview, f"{doc.get('score', 0):.4f}")
        self.console.print(table)

# =================================================================
# 4. 主函数 (MAIN)
# =================================================================
async def start_chat_session_async():
    """启动异步交互式聊天。"""
    console = Console()
    bot = Chatbot(console)
    
    if bot.llm_model:
        display_chat_config(console, bot.chat_config)
        console.print(f"客服已就绪 ([bold green]{bot.chat_config['active_llm_configuration']}[/bold green])")
        
        while True:
            try:
                # prompt_toolkit prompt 是阻塞的，但在简单的 CLI 中通常可以接受
                # 如果追求极致，可以使用异步版本的 prompt
                user_query = prompt(HTML('<skyblue><b>你: </b></skyblue>'))
                if user_query.lower() == '/quit':
                    break
                
                if user_query.lower() == '/config':
                    from ..ui.config_menu import launch_config_editor
                    llm_needs_reload, updated_config = launch_config_editor(bot.chat_config)
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
    """入口包装。"""
    asyncio.run(start_chat_session_async())

if __name__ == '__main__':
    start_chat_session()
