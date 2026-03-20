# 快速上手

本指南将引导您完成 PyRAG-Kit 的环境准备、配置和首次运行。

## 1. 环境要求

*   Python 3.11 或更高版本
*   Git

## 2. 克隆项目

首先，使用 Git 克隆项目仓库到您的本地计算机。

```bash
git clone https://github.com/MisonL/PyRAG-Kit.git
cd PyRAG-Kit
```

## 3. 安装依赖

项目的所有 Python 依赖项都通过 `uv` 管理。我们建议您在 Python 虚拟环境中进行安装，以避免与系统库冲突。

**创建并激活虚拟环境 (可选但推荐):**

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**安装依赖:**

```bash
uv sync
```

## 4. 准备知识库

1.  找到项目根目录下的 `knowledge_base` 文件夹。
2.  将您自己的 `.md` (Markdown) 格式的知识库文件放入此文件夹中。
3.  您可以放入一个或多个文件。程序会自动读取该目录下的所有 `.md` 文件。

> **注意**: 如果该文件夹为空，程序在启动时会提示您添加知识库文件。

## 5. 准备配置文件

在运行程序之前，请先准备本地配置。详细说明请参阅 [配置指南](./configuration.md)。

最快捷的方式是编辑 `config.toml` 和 `.env` 文件：

1.  复制配置文件模板：
    ```bash
    cp config.toml.example config.toml
    cp .env.example .env
    ```
2.  `config.toml.example` 会纳入版本控制，`config.toml` 仅用于本地环境，默认不应提交。
3.  默认嵌入模型是 `local-hash`，知识库向量化不依赖外部 Embedding API。
4.  打开 `.env` 文件，填入您要使用的聊天模型 API Key。例如，如果您使用 OpenAI 兼容渠道：
    ```dotenv
    OPENAI_API_KEY="sk-..."
    OPENAI_API_BASE="https://apis.iflow.cn/v1"
    DEFAULT_LLM_PROVIDER="iflow-qwen3-max"
    ```
5.  如需调整默认路径、base url 或检索参数，请编辑 `config.toml`。

## 6. 运行程序

完成以上步骤后，您就可以启动程序了。

```bash
uv run main.py
```

程序启动后会进入主菜单，您可以按需执行：

*   **1. 知识库文档向量化处理**: 处理 `knowledge_base/` 目录中的 Markdown 文件，并生成本地向量缓存。
*   **2. 召回测试**: 针对当前知识库运行检索验证。
*   **3. 启动聊天机器人会话**: 加载本地向量缓存并进入对话界面。

如需直接重建知识库，也可以执行：

```bash
uv run python -m scripts.embed_knowledge_base --mode standard
```

默认输出文件位于 `data/employee_kb.pkl`。

**常用命令:**

*   在聊天界面输入 `/config` 可以随时打开动态配置菜单，切换模型或调整检索参数。
*   输入 `/quit` 或 `exit` 可以安全地退出程序。
