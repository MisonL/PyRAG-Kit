<div align="center">

# PyRAG-Kit

**Py**thon **R**etrieval-**A**ugmented **G**eneration **Kit**

</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: Mison](https://img.shields.io/badge/Author-Mison-brightgreen)](mailto:1360962086@qq.com)

</div>

**PyRAG-Kit** 是一个功能强大的本地化、模块化的Python RAG（检索增强生成）工具包。它能够读取您本地的文档，通过先进的检索技术和大型语言模型（LLM），为您提供基于私有知识的智能问答。

---

## ✨ 核心功能

- **🔌 多模型支持**: 无缝集成多种主流和本地大语言模型，包括 Google Gemini, OpenAI GPT, Anthropic Claude, 阿里云通义千问, 豆包, DeepSeek, Grok, 以及通过 Ollama 或 LM Studio 运行的本地模型。
- **🚀 高级检索策略**:
    - **向量检索**: 基于语义相似度进行搜索。
    - **全文检索**: 使用 BM25 算法进行关键词匹配。
    - **混合检索**: 智能结合向量与全文检索的优势，并通过权重调整优化排序。
- **🔄 Rerank 精排**: 支持接入 Jina, SiliconFlow 等重排模型，对检索结果进行二次精排，显著提升答案的相关性。
- **⚙️ 动态交互式配置**: 无需修改代码，在程序运行时通过美观的交互式菜单动态切换LLM、调整检索策略、修改权重等。
- **📄 流式响应**: 客服回答采用打字机流式输出，提升用户交互体验。
- **📊 Excel日志记录**: 自动将每一次对话的详细信息（时间、问题、意图、上下文、回答）记录到 Excel 文件中，便于审计和分析。
- **🧹 自动清理**: 程序退出时自动清理生成的缓存文件，保持项目目录整洁。

## 📂 项目结构

项目采用了现代化的目录结构，将源代码、数据、脚本和文档清晰地分离开来。

```
.
├── data/                # 生成的数据 (被 .gitignore 忽略)
│   ├── employee_kb.pkl  # 知识库向量文件
│   └── logs/            # 聊天日志
├── knowledge_base/      # 存放你的原始知识库 .md 文件
├── scripts/             # 独立脚本
│   └── embed_knowledge_base.py # 知识库向量化脚本
├── src/                 # 核心源代码
│   ├── chat/            # 聊天核心逻辑
│   ├── providers/       # 所有模型提供商的实现
│   ├── retrieval/       # 检索逻辑
│   ├── ui/              # 用户界面 (菜单、显示工具)
│   └── utils/           # 辅助工具 (配置、清理)
├── tests/               # (预留) 自动化测试
├── main.py              # 程序主入口
├── .env.example         # 环境变量配置模板
├── .gitignore           # Git忽略文件配置
├── README.md            # 就是你正在看的这个文件
└── requirements.txt     # Python依赖项
```

## 🚀 安装与运行

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. 安装依赖

项目使用 `requirements.txt` 管理依赖。

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`。

```bash
cp .env.example .env
```

然后，编辑 `.env` 文件，填入你需要的 API 密钥。例如：

```env
# .env

# --- 必填项 ---
# 至少需要配置一个你希望使用的模型的API Key
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."
# ... 其他API Key

# --- 可选项 ---
# 如果使用代理或第三方服务，可以在这里配置 Base URL
# OPENAI_API_BASE="https://my-proxy.com/v1"
```

你也可以在 `src/utils/config.py` 文件中修改默认的模型配置和程序行为。

### 4. 准备知识库

将你的 `.md` 格式的知识库文档放入 `knowledge_base` 文件夹中。

### 5. 运行程序

直接从项目根目录运行主程序文件即可。

```bash
python main.py
```

程序启动后，你将看到一个主菜单：

1.  **首次运行，请选择 `1. 知识库文档向量化处理`**。
    -   该脚本会读取 `knowledge_base` 目录下的所有文档，将其处理并生成 `data/employee_kb.pkl` 向量文件。
2.  **向量化处理完成后，选择 `2. 启动聊天机器人会话`**。
    -   现在你可以开始与你的本地知识库进行对话了！
    -   在聊天中，输入 `/config` 可以随时打开动态配置菜单。
    -   输入 `/quit` 可以退出聊天。

---

<div align="center">

*作者: Mison* · *联系邮箱: 1360962086@qq.com*

</div>
