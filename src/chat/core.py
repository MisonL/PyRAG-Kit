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

# 使用相对导入
from ..providers.factory import ModelProviderFactory
from ..providers.__base__.model_provider import LargeLanguageModel
from ..utils.config import get_settings, RetrievalMethod # 导入 get_settings 函数和 RetrievalMethod
from ..ui.config_menu import launch_config_editor
from ..ui.display_utils import display_chat_config
from ..retrieval.retriever import retrieve_documents
from ..retrieval.vdb.base import VectorStoreBase
from ..retrieval.vdb.factory import VectorStoreFactory
from ..utils.log_manager import get_chat_logger # 导入新的日志管理器

# =================================================================
# 2. 聊天机器人核心 (CHATBOT CORE)
# =================================================================
class Chatbot:
    def __init__(self, console: Console):
        self.console = console
        self.vector_store: Optional[VectorStoreBase] = None
        self.llm_model: Optional[LargeLanguageModel] = None
        self.logger = get_chat_logger() # 使用新的日志管理器
        
        # 在初始化时获取一次配置，并存储在实例变量中
        current_settings = get_settings()
        self.chat_config = {
            "retrieval_method": current_settings.chat_retrieval_method,
            "vector_weight": current_settings.chat_vector_weight,
            "keyword_weight": current_settings.chat_keyword_weight,
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
        """初始化向量存储，并尝试加载现有知识库。"""
        self.vector_store = VectorStoreFactory.get_default_vector_store()
        try:
            current_settings = get_settings() # 再次获取最新配置
            self.vector_store.load(current_settings.pkl_path)
            self.console.print(f"[green]知识库 '[cyan]{os.path.basename(current_settings.pkl_path)}[/cyan]' 已加载。[/green]")
        except FileNotFoundError:
            current_settings = get_settings() # 再次获取最新配置
            self.console.print(f"[bold red]警告：知识库文件 '{current_settings.pkl_path}' 未找到。请先运行向量化脚本。[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]加载知识库时出错: {e}[/bold red]")

    def reload_llm(self) -> bool:
        """重新加载或初始化LLM模型。"""
        try:
            active_llm_key = self.chat_config["active_llm_configuration"]
            self.console.print(f"[dim]正在加载LLM: [bold cyan]{active_llm_key}[/bold cyan]...[/dim]")
            self.llm_model = ModelProviderFactory.get_llm_provider(active_llm_key)
            if self.llm_model:
                self.console.print(f"[green]LLM [bold cyan]{active_llm_key}[/bold cyan] 加载成功。[/green]")
                return True
            return False
        except Exception as e:
            self.console.print(f"[bold red]初始化LLM时出错: {e}[/bold red]")
            self.llm_model = None
            return False

    def _identify_intent(self, user_query: str) -> str:
        if not self.llm_model: return user_query # 如果LLM未加载，直接返回原始查询
        prompt = f"你是一个意图识别助手。请根据用户的问题，判断其意图。\n用户问题: {user_query}\n请直接输出用户意图的简短描述。"
        try:
            # 使用invoke并设置stream=False来获取完整结果
            response_generator = self.llm_model.invoke(
                prompt=prompt,
                stream=False,
                temperature=self.chat_config['chat_temperature']
            )
            response = "".join(list(response_generator)).strip()
            # 如果模型返回空或非常短的响应，可能表示不确定，直接返回原始查询
            if not response or len(response) < 4:
                 return user_query
            return response
        except Exception as e:
            self.console.print(f"[red]意图识别时出错: {e}[/red]")
            return user_query # 出错时返回原始查询

    def _retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
        if not self.vector_store: return []
        self.console.print(f"[dim]正在使用 '[yellow]{self.chat_config['retrieval_method'].value}[/yellow]' 模式检索...[/dim]")
        return retrieve_documents(
            query=query,
            vector_store=self.vector_store,
            console=self.console,
            retrieval_method=self.chat_config['retrieval_method'],
            top_k=self.chat_config['top_k'],
            vector_weight=self.chat_config['vector_weight'],
            keyword_weight=self.chat_config['keyword_weight'],
            rerank_enabled=self.chat_config['rerank_enabled'],
            active_rerank_configuration=self.chat_config['active_rerank_configuration'],
            score_threshold=self.chat_config['score_threshold']
        )

    def _generate_answer_stream(self, user_query: str, intent: str, retrieved_docs: List[Dict[str, Any]]) -> "Generator[str, None, None]":
        if not self.llm_model:
            yield "抱歉，我现在无法回答问题。"
            return

        context_str = "\n".join([f"来源: {doc['metadata']['source']}\n内容: {doc.get('page_content', '')}" for doc in retrieved_docs])
        
        if context_str:
            prompt = f"你是一个智能客服助手。请根据以下信息回答用户的问题。\n用户意图: {intent}\n用户问题: {user_query}\n相关知识:\n---\n{context_str}\n---\n请根据提供的相关知识，简洁明了地回答用户的问题。如果相关知识不足以回答问题，请礼貌地告知用户。"
        else:
            prompt = f"你是一个智能客服助手。用户问题: {user_query}\n用户意图: {intent}\n没有找到相关知识。请礼貌地告知用户无法回答，并建议他们提供更多信息或尝试其他问题。"
        
        try:
            # 使用新的invoke方法并确保以流式传输
            yield from self.llm_model.invoke(
                prompt=prompt,
                stream=True,
                temperature=self.chat_config['chat_temperature']
            )
        except Exception as e:
            self.console.print(f"[red]LLM 生成最终回答时出错: {e}[/red]")
            yield "抱歉，我在生成回答时遇到了一些问题。"

    def chat(self, user_input: str) -> "Generator[str, None, None]":
        intent = self._identify_intent(user_input)
        self.console.print(f"  [bold]识别到的意图:[/bold] [yellow]{intent}[/yellow]")
        
        retrieved_docs = self._retrieve_knowledge(user_input)
        self._display_retrieved_docs(retrieved_docs)
        
        context_for_log = "\n".join([f"[{doc['metadata']['source']}]" for doc in retrieved_docs])
        
        def response_generator_with_logging():
            full_response = ""
            answer_stream = self._generate_answer_stream(user_input, intent, retrieved_docs)
            for chunk in answer_stream:
                full_response += chunk
                yield chunk
            
            self.logger.info(f"UserQuery: {user_input}\nDetectedIntent: {intent}\nRetrievedContext: {context_for_log}\nLLMResponse: {full_response}")

        return response_generator_with_logging()

    def _display_retrieved_docs(self, docs: List[Dict]):
        if not docs:
            self.console.print("[yellow]未检索到相关文档。[/yellow]")
            return
        
        table = Table(title="[bold cyan]检索到的相关文档[/bold cyan]", show_header=True, header_style="bold magenta")
        table.add_column("来源", style="cyan", no_wrap=True)
        table.add_column("内容预览", style="white")
        table.add_column("相关度", style="bold")

        max_score = max(doc.get("score", 0) for doc in docs) if docs else 1
        
        for doc in docs:
            score = doc.get("score", 0)
            normalized_score = score / max_score
            
            red = int(255 * (1 - normalized_score))
            blue = int(255 * normalized_score)
            color_style = Style(color=f"rgb({red},80,{blue})")
            
            preview = doc.get("page_content", "")[:100].replace("\n", " ") + "..."
            table.add_row(doc["metadata"]["source"], preview, Text(f"{score:.4f}", style=color_style))
        
        self.console.print(table)

# =================================================================
# 4. 主函数 (MAIN)
# =================================================================
def start_chat_session():
    """启动交互式聊天会话。"""
    console = Console()
    
    bot = Chatbot(console)
    
    if bot.llm_model:
        display_chat_config(console, bot.chat_config)
        console.print(f"我是你的智能客服（由 [bold green]{bot.chat_config['active_llm_configuration']}[/bold green] 支持），请输入问题（输入'[bold]/quit[/bold]'或'[bold]/config[/bold]'）：")
        
        while True:
            try:
                # 使用 prompt_toolkit 替代 console.input，并优化样式
                user_query = prompt(HTML('<skyblue><b>你: </b></skyblue>'))
                if user_query.lower() == '/quit':
                    console.print("[bold yellow]再见！[/bold yellow]")
                    break
                
                if user_query.lower() == '/config':
                    # 使用新的配置编辑器接口
                    from ..ui.config_menu import launch_config_editor
                    llm_needs_reload, updated_config = launch_config_editor(bot.chat_config)
                    bot.chat_config = updated_config  # 更新配置
                    if llm_needs_reload:
                        console.print("[yellow]检测到LLM配置变更，正在重载模型...[/yellow]")
                        bot.reload_llm()
                    display_chat_config(console, bot.chat_config)  # 使用新的配置显示
                    console.print(f"我是你的智能客服（由 [bold green]{bot.chat_config['active_llm_configuration']}[/bold green] 支持），请输入问题（输入'[bold]/quit[/bold]'或'[bold]/config[/bold]'）：")
                    continue

                answer_stream = bot.chat(user_query)
                
                response_panel = Panel("", title="[bold green]客服[/bold green]", border_style="green", highlight=True)
                full_response = ""

                with Live(response_panel, console=console, refresh_per_second=10, vertical_overflow="visible") as live:
                    for chunk in answer_stream:
                        full_response += chunk
                        live.update(Panel(Text(full_response), title="[bold green]客服[/bold green]", border_style="green", highlight=True))
            
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold yellow]再见！[/bold yellow]")
                break
    else:
        console.print("[bold red]错误：LLM模型未能成功初始化，程序无法启动。请检查配置和API密钥。[/bold red]")

if __name__ == '__main__':
    start_chat_session()