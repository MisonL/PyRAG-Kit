# 快速上手

本指南将引导您完成 PyRAG-Kit 的环境准备、安装和首次运行。

## 1. 环境要求

*   Python 3.9 或更高版本
*   Git

## 2. 克隆项目

首先，使用 Git 克隆项目仓库到您的本地计算机。

```bash
git clone https://github.com/MisonL/PyRAG-Kit.git
cd PyRAG-Kit
```

## 3. 安装依赖

项目的所有 Python 依赖项都记录在 `requirements.txt` 文件中。我们建议您在 Python 虚拟环境中进行安装，以避免与系统库冲突。

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
pip install -r requirements.txt
```

## 4. 准备知识库

1.  找到项目根目录下的 `knowledge_base` 文件夹。
2.  将您自己的 `.md` (Markdown) 格式的知识库文件放入此文件夹中。
3.  您可以放入一个或多个文件。程序会自动读取该目录下的所有 `.md` 文件。

> **注意**: 如果该文件夹为空，程序在启动时会提示您添加知识库文件。

## 5. 配置 API 密钥

在运行程序之前，您至少需要配置一个大语言模型 (LLM) 的 API 密钥。详细的配置方法请参阅 [配置指南](./configuration.md)。

最快捷的方式是编辑 `config.ini` 文件：

1.  复制配置文件模板：
    ```bash
    cp config.ini.example config.ini
    ```
2.  打开 `config.ini` 文件，找到 `[API_KEYS]` 部分，并填入您所拥有模型的 API Key。例如，如果您有 Google Gemini 的 Key：
    ```ini
    [API_KEYS]
    GOOGLE_API_KEY = "AIzaSy..."
    ```

## 6. 运行程序

完成以上步骤后，您就可以启动程序了。

```bash
python main.py
```

程序首次运行时，会自动执行以下操作：

*   **知识库向量化**: 检查 `knowledge_base` 目录中的文档，如果发现有新的或已修改的文档，它将处理这些文档，生成向量并保存在 `data/` 目录下。这个过程可能需要一些时间，具体取决于您的文档大小和计算机性能。
*   **启动聊天界面**: 向量化完成后，程序将进入交互式聊天模式，您可以开始提问。

**常用命令:**

*   在聊天界面输入 `/config` 可以随时打开动态配置菜单，切换模型或调整检索参数。
*   输入 `/quit` 或 `exit` 可以安全地退出程序。