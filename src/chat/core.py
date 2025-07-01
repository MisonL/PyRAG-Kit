# -*- coding: utf-8 -*-
# =================================================================
# 1. 导入 (IMPORTS)
# =================================================================
import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator

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
from ..utils.config import CHAT_CONFIG, KB_CONFIG, LOG_PATH
from ..ui.config_menu import launch_config_editor
from ..ui.display_utils import display_chat_config
from ..retrieval.retriever import VectorStore, retrieve_documents

# =================================================================
# 2. 日志记录器 (LOGGER)
# =================================================================
class ChatLogger:
    def __init__(self):
        self.log_dir = LOG_PATH
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, f"chat_log_{datetime.now().strftime('%Y-%m-%d')}.xlsx")
        self.log_data = []

    def log(self, query: str, intent: str, context: str, response: str):
        self.log_data.append({
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "UserQuery": query,
            "DetectedIntent": intent,
            "RetrievedContext": context,
            "LLMResponse": response
        })
        df = pd.DataFrame(self.log_data)
        df.to_excel(self.log_file, index=False, engine='openpyxl')

# =================================================================
# 3. 聊天机器人核心 (CHATBOT CORE)
# =================================================================
class Chatbot:
    def __init__(self, console: Console):
        self.console = console
        self.vector_store = self._load_vector_store()
        self.llm_model: Optional[LargeLanguageModel] = None
        self.logger = ChatLogger()
        self.reload_llm() # 初始加载

    def _load_vector_store(self) -> Optional[VectorStore]:
        file_path = KB_CONFIG["output_file"]
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                vs = VectorStore(file_path)
                vs.documents = data.get("documents", [])
                vs.embeddings = data.get("embeddings")
                vs._initialize_bm25() # 确保BM25被初始化
                self.console.print(f"[green]知识库 '[cyan]{os.path.basename(file_path)}[/cyan]' 已加载。[/green]")
                return vs
        except FileNotFoundError:
            self.console.print(f"[bold red]警告：知识库文件 '{file_path}' 未找到。请先运行向量化脚本。[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]加载知识库时出错: {e}[/bold red]")
        return None

    def reload_llm(self) -> bool:
        """重新加载或初始化LLM模型。"""
        try:
            active_llm_key = CHAT_CONFIG['active_llm_configuration']
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
        if not self.llm_model: return "未知意图"
        prompt = f"你是一个意图识别助手。请根据用户的问题，判断其意图。\n用户问题: {user_query}\n请直接输出用户意图的简短描述。"
        try:
            # 使用invoke并设置stream=False来获取完整结果
            response_generator = self.llm_model.invoke(prompt=prompt, stream=False)
            return "".join(list(response_generator))
        except Exception as e:
            self.console.print(f"[red]意图识别时出错: {e}[/red]")
            return "未知意图"

    def _retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
        if not self.vector_store: return []
        self.console.print(f"[dim]正在使用 '[yellow]{CHAT_CONFIG['retrieval_method'].value}[/yellow]' 模式检索...[/dim]")
        return retrieve_documents(query, self.vector_store, self.console)

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
            yield from self.llm_model.invoke(prompt=prompt, stream=True)
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
            
            self.logger.log(user_input, intent, context_for_log, full_response)

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
        display_chat_config(console)
        console.print(f"我是你的智能客服（由 [bold green]{CHAT_CONFIG['active_llm_configuration']}[/bold green] 支持），请输入问题（输入'[bold]/quit[/bold]'或'[bold]/config[/bold]'）：")
        
        while True:
            try:
                # 使用 prompt_toolkit 替代 console.input，并优化样式
                user_query = prompt(HTML('<skyblue><b>你: </b></skyblue>'))
                if user_query.lower() == '/quit':
                    console.print("[bold yellow]再见！[/bold yellow]")
                    break
                
                if user_query.lower() == '/config':
                    llm_needs_reload, _ = launch_config_editor()
                    if llm_needs_reload:
                        console.print("[yellow]检测到LLM配置变更，正在重载模型...[/yellow]")
                        bot.reload_llm()
                    display_chat_config(console)
                    console.print(f"我是你的智能客服（由 [bold green]{CHAT_CONFIG['active_llm_configuration']}[/bold green] 支持），请输入问题（输入'[bold]/quit[/bold]'或'[bold]/config[/bold]'）：")
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