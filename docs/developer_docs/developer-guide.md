# 开发者指南

本指南面向希望深入理解 PyRAG-Kit 源代码、进行二次开发或贡献代码的开发者。

## 1. 项目架构概述

PyRAG-Kit 采用了清晰、模块化的项目结构，旨在实现高内聚、低耦合，方便开发者理解和扩展。

```
.
├── src/                 # 核心源代码
│   ├── chat/            # 聊天核心逻辑 (core.py)
│   ├── etl/             # 数据处理流水线 (提取、清洗、分割)
│   ├── providers/       # 所有模型提供商的实现
│   ├── retrieval/       # 检索逻辑 (retriever.py, vdb/)
│   ├── runtime/         # 运行期配置对象与快照契约
│   ├── services/        # 应用服务层 (构建、检索、聊天)
│   ├── ui/              # 用户界面 (config_menu.py, display_utils.py)
│   └── utils/           # 辅助工具 (config.py, log_manager.py)
├── tests/               # 单元测试
├── knowledge_base/      # 原始知识库文档
├── scripts/             # 独立脚本 (如知识库向量化)
├── main.py              # 程序主入口
├── config.toml.example  # 主配置模板
└── config.toml          # 本地主配置文件 (默认不纳入版本控制)
```

### 工作流程概览

1.  **启动 (`main.py`)**:
    *   渲染主菜单。
    *   根据用户选择进入向量化、召回测试或聊天会话。

2.  **Chatbot 初始化**:
    *   加载配置文件 (`src/utils/config.py`) 并构建 `RunConfig` / `SessionConfig`。
    *   根据活动知识快照加载向量数据库 (`src/retrieval/vdb/`)。
    *   初始化检索服务、聊天服务和默认的 LLM 提供商。

3.  **用户交互 (`src/chat/core.py`)**:
    *   接收用户输入。
    *   若输入为 `/config`，则调用 `ui` 模块 (`src/ui/config_menu.py`) 进入配置菜单。
    *   若为普通问题，则调用 `retrieval` 模块 (`src/retrieval/retriever.py`) 检索相关文档。
    *   将问题和检索到的文档构建成 Prompt，发送给当前加载的 LLM 提供商。
    *   流式接收并显示 LLM 的回答。

## 2. 核心模块详解

### `src/providers` - 模型提供商

这是扩展新模型的关键目录。

*   **`__base__/model_provider.py`**: 定义了所有模型提供商必须遵循的抽象基类 `LargeLanguageModel` 和 `RerankModel`。它们规定了 `invoke` 和 `rerank` 等核心接口。
*   **`factory.py`**: 实现了一个工厂模式，用于根据配置动态创建和获取指定的模型提供商实例。
*   **`google.py`, `openai.py`, etc.**: 每个文件都是一个具体模型提供商的实现，负责处理与该平台 API 的所有交互（如认证、请求构建、响应解析）。

**如何添加一个新的模型提供商？**

1.  在 `src/providers/` 目录下创建一个新的 `my_new_provider.py` 文件。
2.  在该文件中创建一个类，继承自 `LargeLanguageModel`。
3.  实现基类中定义的抽象方法，主要是 `invoke()`。
4.  在 `src/providers/factory.py` 中导入您的新类，并在 `ModelProviderFactory` 的 `get_llm_provider` 方法中添加一个 `elif` 分支来注册您的新提供商。
5.  在 `config.toml` 的 `llm_configurations`、`embedding_configurations` 或 `rerank_configurations` 中添加使用您的新提供商的配置。

**当前默认嵌入提供商**

项目当前默认使用 `local-hash` 作为本地嵌入模型，用于保证默认链路无需额外 Embedding API 即可运行。对应实现位于 `src/providers/local_hash.py`。

### `src/etl` - 数据处理流水线

ETL (Extract, Transform, Load) 模块负责将原始的 `.md` 文档转换成可供检索的向量化数据。

*   **`pipeline.py`**: 定义了完整的 ETL 流程。
*   **`extractors/`**: 负责从不同类型的文件中提取文本内容（当前主要是 Markdown）。
*   **`cleaners/`**: 负责对提取的文本进行清洗，如去除多余空格、URL 等。
*   **`splitters/`**: 负责将长文本分割成较小的、有意义的块 (Chunks)，以便进行向量化。

### `src/retrieval` - 检索模块

该模块负责从向量数据库中检索信息。

*   **`retriever.py`**: 兼容入口层，对外保留原有 `retrieve_documents` 接口，内部委派给服务层。
*   **`vdb/`**: 向量数据库 (Vector Database) 的实现。
    *   `base.py`: 定义了向量存储的抽象基类 `VectorStoreBase`。
    *   `faiss_store.py`: 使用 Facebook AI 的 `faiss` 库实现的本地向量存储，并保存到知识快照目录。
    *   `factory.py`: 用于创建向量存储实例的工厂。
    *   `snapshot_repository.py`: 管理 `ACTIVE_SNAPSHOT` 指针和快照目录结构。

### `src/services` - 应用服务层

*   **`knowledge_build_service.py`**: 负责知识库构建编排，执行 ETL、批量嵌入和快照落盘。
*   **`retrieval_service.py`**: 负责候选召回、融合、父子上下文提升和可选 rerank。
*   **`chat_service.py`**: 负责意图识别、Prompt 拼装和流式回复。
*   **`embedding_service.py`**: 负责 embedding provider 生命周期和批量向量生成。

## 3. 测试

项目在 `tests/` 目录下为核心功能编写了单元测试。在进行任何修改或添加新功能后，强烈建议您运行相关测试以确保代码的正确性和稳定性。

```bash
# 运行所有测试
uv run pytest

# 运行特定文件的测试
uv run pytest tests/providers/test_factory.py
```

通过遵循上述结构和实践，您可以更轻松地为 PyRAG-Kit 添加新功能、修复错误或将其集成到您自己的项目中。
