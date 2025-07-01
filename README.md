<div align="center">

![Logo](imgs/logo.jpg)

</div>

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: Mison](https://img.shields.io/badge/Author-Mison-brightgreen)](mailto:1360962086@qq.com)

</div>

> **PyRAG-Kit** 是一个 **Dify 核心逻辑的 Python 实现**，旨在提供一个轻量级的本地化工具，用于快速验证和理解 Dify 的知识库核心工作流程，包括文档的**向量化、分段处理**以及**混合检索**策略。

---

## ✨ 核心功能

- **🔌 多模型支持**: 无缝集成多种主流和本地大语言模型，包括 Google Gemini, OpenAI GPT, Anthropic Claude, 阿里云通义千问, 豆包, DeepSeek, Grok, 以及通过 Ollama 或 LM Studio 运行的本地模型。
- **🚀 高级检索策略**: 支持向量检索、全文检索和混合检索，并通过 Rerank 模型二次精排，提升答案相关性。
- **⚙️ 动态交互配置**: 运行时通过 `/config` 命令打开交互式菜单，动态切换LLM、调整检索策略、修改权重等。
- **📄 流式响应**: 客服回答采用打字机流式输出，提升用户交互体验。
- **📊 Excel日志**: 自动将每一次对话的详细信息记录到 Excel 文件中，便于审计和分析。
- **🧹 智能缓存与清理**: 自动处理知识库向量化，并在程序退出时清理缓存，保持项目整洁。

## 📸 程序截图

![Main Screenshot](imgs/main.jpg)


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
├── config.ini.example   # 配置文件模板
├── .gitignore           # Git忽略文件配置
├── README.md            # 就是你正在看的这个文件
└── requirements.txt     # Python依赖项
```

## 🚀 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/MisonL/PyRAG-Kit.git
cd PyRAG-Kit
```

### 2. 安装依赖

项目使用 `requirements.txt` 管理依赖。

```bash
pip install -r requirements.txt
```

### 3. 进行配置

项目使用 `config.ini` 文件进行配置。请先复制模板文件：

```bash
cp config.ini.example config.ini
```

然后，编辑 `config.ini` 文件，填入你需要的 API 密钥和自定义设置。

```ini
# config.ini

[API_KEYS]
# 填入你希望使用的模型的API Key
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "AIzaSy..."
# ... 其他API Key

[BEHAVIOR]
# 默认使用的LLM提供商 (例如: google, openai, anthropic)
DEFAULT_LLM_PROVIDER = google

# ... 其他行为配置
```

> **注意**: 你也可以通过设置**环境变量**来覆盖 `config.ini` 中的 `API_KEYS`，例如 `export OPENAI_API_KEY="sk-..."`，这在服务器部署时非常有用。

### 4. 准备知识库

将你的 `.md` 格式的知识库文档放入 `knowledge_base` 文件夹中。

### 5. 运行程序

直接从项目根目录运行主程序文件即可。

```bash
python main.py
```

**程序会自动完成以下工作:**
1.  **检查知识库**: 如果 `knowledge_base` 目录中有新的或更新的文档，程序会自动进行向量化并更新 `data/employee_kb.pkl` 文件。
2.  **启动聊天**: 直接进入交互式聊天会话。

**常用命令:**
-   输入 `/config` 可以随时打开动态配置菜单。
-   输入 `/quit` 或 `exit` 可以退出聊天。

---

<div align="center">

*作者: Mison* · *联系邮箱: 1360962086@qq.com*

</div>
