# 配置指南

PyRAG-Kit 采用了一套灵活且分层的配置系统，允许您通过多种方式管理程序的行为和密钥，以适应不同的使用场景，如本地开发、团队协作和生产部署。

## 配置加载优先级

系统会按照以下顺序加载配置，排在前面的方式会覆盖排在后面的同名配置项：

1.  **环境变量 (Environment Variables)**: 优先级最高。非常适合在服务器或 Docker 容器中部署时使用，能够安全地管理敏感信息。
2.  **`.env` 文件**: 位于项目根目录。用于存放不希望提交到版本控制系统（如 Git）的个人配置或敏感数据。
3.  **`config.ini` 文件**: 位于项目根目录。项目的主要配置文件，用于设置大部分非敏感的默认参数。
4.  **代码中的默认值**: 优先级最低。作为备用选项，确保程序在没有任何外部配置时也能运行。

---

## 配置方式详解

### 方式一: `config.ini` 文件 (推荐)

这是最常用、最直观的配置方式，适合设置项目的基础参数。

1.  **创建文件**:
    将项目根目录下的 `config.ini.example` 复制一份并重命名为 `config.ini`。
    ```bash
    cp config.ini.example config.ini
    ```

2.  **编辑文件**:
    用文本编辑器打开 `config.ini`。文件内部按功能块（如 `[API_KEYS]`, `[CHAT]`）对配置项进行了分组，并附有详细的中文注释。

### 方式二: `.env` 文件

当您需要覆盖 `config.ini` 中的某些配置，特别是 API 密钥等敏感信息，或者希望为本地环境设置特定参数时，`.env` 文件是理想选择。

1.  在项目根目录创建一个名为 `.env` 的文件。
2.  在文件中以 `KEY=VALUE` 的格式添加配置。**注意**：这里的 `KEY` 必须与 `config.ini` 中的键名完全一致，但不需要段落名 `[SECTION]`。

    ```dotenv
    # .env 文件示例
    # 这里的值会覆盖 config.ini 中的同名值
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
    DEFAULT_LLM_PROVIDER="openai-gpt4o"
    LOG_LEVEL="DEBUG"
    ```

> **重要**: 为了安全起见，`.env` 文件通常应该被添加到 `.gitignore` 文件中，以避免将敏感信息泄露到代码仓库。

### 方式三: 环境变量

在生产环境或自动化脚本中，使用环境变量是业界标准做法。

-   **Linux / macOS**:
    ```bash
    export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
    export DEFAULT_LLM_PROVIDER="openai-gpt4o"
    python main.py
    ```
-   **Windows (PowerShell)**:
    ```powershell
    $env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
    $env:DEFAULT_LLM_PROVIDER="openai-gpt4o"
    python main.py
    ```

---

## `config.ini` 文件详解

以下是 `config.ini` 文件中主要配置部分的说明。

### `[API_KEYS]`
用于存放所有需要使用的第三方服务 API 密钥。
- `GOOGLE_API_KEY`, `OPENAI_API_KEY`, 等: 填入对应平台提供的密钥。

### `[BASE_URLS]`
如果您使用 API 代理、第三方中转服务（如 one-api）或本地部署的模型（如 Ollama, LM Studio），请在此处配置它们的访问地址（Base URL）。

### `[GENERAL]`
通用设置。
- `LOG_LEVEL`: 设置日志记录的详细程度 (`DEBUG`, `INFO`, `WARNING`, `ERROR`)。`DEBUG` 级别最详细。
- `LOG_PATH`: 日志文件的存放目录。

### `[PATHS]`
路径配置。
- `KNOWLEDGE_BASE_PATH`: 存放原始知识库 `.md` 文件的目录。
- `PKL_PATH`: 向量化后知识库的存储路径。

### `[KNOWLEDGE_BASE]`
知识库构建和文本处理相关的参数。
- `KB_CHUNK_SIZE`: 文本分块的最大长度。
- `KB_CHUNK_OVERLAP`: 相邻文本块之间的重叠字符数，有助于保持上下文连续性。
- `KB_EMBEDDING_BATCH_SIZE`: 向量化时每批处理的文本数量。

### `[BEHAVIOR]`
程序启动时的默认行为。
- `DEFAULT_LLM_PROVIDER`: 默认使用的聊天大语言模型。
- `DEFAULT_EMBEDDING_PROVIDER`: 默认使用的向量化模型。
- `DEFAULT_RERANK_PROVIDER`: 默认使用的 Rerank 精排模型。

### `[CHAT]`
聊天核心功能配置。
- `CHAT_RETRIEVAL_METHOD`: 检索方法，可选 `SEMANTIC_SEARCH` (向量检索), `FULL_TEXT_SEARCH` (全文检索), `HYBRID_SEARCH` (混合检索)。
- `CHAT_VECTOR_WEIGHT`, `CHAT_KEYWORD_WEIGHT`: 在混合检索中，两种检索方式的权重。
- `CHAT_RERANK_ENABLED`: 是否启用 Rerank 精排。
- `CHAT_TOP_K`: 检索返回的文档片段数量。
- `CHAT_SCORE_THRESHOLD`: 向量搜索的相似度得分阈值，低于此分数的将被过滤。
- `CHAT_TEMPERATURE`: 控制 LLM 回答的创造性和随机性。

### `[MODEL_CONFIGURATIONS]`
模型配置，采用 JSON 格式。这里定义了程序中所有可用的模型。
- `EMBEDDING_CONFIGURATIONS`: 定义可用的向量化模型。
- `RERANK_CONFIGURATIONS`: 定义可用的 Rerank 模型。
- `LLM_CONFIGURATIONS`: 定义可用的聊天大语言模型。

您可以自由地在这个 JSON 结构中添加、修改或删除模型条目，以支持新的模型或自定义现有模型。`key` 是您在程序中看到的名称，`provider` 对应 `src/providers` 下的具体实现，`model_name` 则是传递给 API 的实际模型标识符。