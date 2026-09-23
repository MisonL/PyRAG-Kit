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

*   **`__base__/model_provider.py`**: 定义 `CompletionRequest`、`CompletionResult`、`LargeLanguageModel`、`TextEmbeddingModel` 和 `RerankModel`。`invoke`/`ainvoke` 保留文本流兼容接口；需要多轮、多模态、工具调用、结构化输出和 usage 时使用 `complete`/`acomplete`。
*   **`factory.py`**: 实现了一个工厂模式，用于根据配置动态创建和获取指定的模型提供商实例。
*   **`google.py`, `openai.py`, etc.**: 每个文件都是一个具体模型提供商的实现，负责处理与该平台 API 的所有交互（如认证、请求构建、响应解析）。

**如何添加一个新的模型提供商？**

1.  在 `src/providers/` 目录下创建一个新的 `my_new_provider.py` 文件。
2.  在该文件中实现所需能力接口：`LargeLanguageModel`、`TextEmbeddingModel` 或 `RerankModel`。
3.  实现对应抽象方法，并在 `src/providers/factory.py` 的 `_provider_map` 中注册模块和类；不要在 `get_*_provider` 方法中添加分支。
4.  确保配置角色与实现能力匹配。工厂会在实例化时校验接口，并对需要不同协议实现的渠道使用角色别名。
5.  在 `config.toml` 的 `llm_configurations`、`embedding_configurations` 或 `rerank_configurations` 中添加配置，并为工厂和渠道请求补充测试。

Provider 的协议边界保持清晰：OpenAI、Qwen、SiliconFlow、DeepSeek、Grok、Ollama 和 LM Studio 复用 `openai` SDK 的 Chat/Responses 兼容接口；这些渠道的 LLM 配置默认使用 Chat Completions，也可显式选择 `responses`，但兼容服务端是否实现该路径必须单独验证。官方 OpenAI Base URL 可按 SDK 契约使用 Responses；代理、中转或自建 Base URL 必须在完成实际验证后通过 `options.server_verified_protocols = ["responses"]` 登记，否则 Provider 会在请求前显式拒绝。只有官方 OpenAI Provider 声明 Files、Batches、Vector Stores、Skills、Realtime、Webhooks、Admin 和 Content Provenance Checks 等 OpenAI 专属资源，并通过 `.native` 保留当前 SDK 的完整资源树。Google、Anthropic 和 Volcengine 使用各自 SDK，Volcengine 同时支持 Ark Chat、Responses 和 Classification；Ark Classification 通过 SDK 官方资源类显式挂载，SDK 不提供该资源时直接报错。Jina 与 SiliconFlow Rerank 使用 HTTP JSON 接口。新增渠道时先确认协议、服务端证据和能力，再选择复用适配器或实现独立 provider。

`ModelDetail.options` 是非敏感 SDK 扩展参数的唯一配置入口。工厂会将其传入 Provider，并拒绝凭证、请求头和 Base URL 字段；Provider 还会把客户端构造参数与请求参数分流，不支持的能力应在 Provider 边界显式报错。SDK 专属文件、批处理、缓存、token 计数等资源通过 Provider 的显式方法调用，不得塞进普通 `invoke()`。同步和异步 Facade 应保持 SDK 的真实调用约定：例如 Google `aio.models.generate_content_stream()` 需要等待一次后再异步迭代，而 `aio.chats.create()` 本身同步返回 `AsyncChat`。

Responses 适配位于 `src/providers/openai_compatible.py`，共享消息归一化位于 `src/providers/__base__/model_provider.py`。请求构造、Chat Completions 工具转换、`assistant.tool_calls`/`role=tool` 历史转换、`text`/`image_url`/`file` 多模态块转换、同步/异步文本提取和 SSE 事件解析集中在这些适配器中；Ark Responses 还要按目标 SDK 变体转换 `input_audio`、`input_video` 和 `image_pixel_limit`，当前 OpenAI SDK 不支持的音视频输入块必须在请求前显式拒绝。`response.failed`、`response.incomplete`、取消状态和错误事件必须显式转换为异常，不能静默结束流。扩展事件类型时应先补契约测试，再修改统一文本流接口。

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
