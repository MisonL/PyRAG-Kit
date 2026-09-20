# 大语言模型 (LLM) 支持

PyRAG-Kit 的一个核心优势是其高度的灵活性和可扩展性，尤其体现在对多种大语言模型 (LLM) 提供商的无缝支持上。您可以根据自己的需求和资源，轻松切换和配置不同的模型。

## 失败语义说明

- 当 Provider 的凭证缺失、SDK 导入失败或远端 API 调用失败时，系统会显式报错并记录日志。
- 聊天模型失败时，不再返回占位回答；Embedding 或 Rerank 失败时，也不再返回空向量或零分结果伪装为成功。
- 在聊天界面通过 `/config` 切换 LLM 时，如果新模型加载失败，系统会自动保留当前活动模型配置。

## 支持的模型提供商

本项目通过模块化的提供商 (Provider) 设计，内置了对以下主流和本地模型服务的支持：

### 云服务模型

*   **Google**: 支持 Gemini 系列模型。
*   **OpenAI**: 支持 GPT 系列模型，如 GPT-4o, GPT-3.5-Turbo。
*   **Anthropic**: 支持 Claude 系列模型，如 Claude 3.5 Sonnet。
*   **阿里云 (Qwen)**: 支持通义千问系列模型。
*   **火山引擎 (VolcEngine)**: 支持豆包 (Doubao) 系列模型。
*   **深度求索 (DeepSeek)**: 支持 DeepSeek 系列模型。
*   **xAI**: 支持 Grok 系列模型。
*   **SiliconFlow**: 一个集成了多种开源模型的平台，可通过其统一 API 调用。
*   **Jina AI**: 主要用于提供高质量的 Rerank 模型。

### 本地化模型

*   **Ollama**: 支持通过 Ollama 在本地运行的各种开源模型，如 Llama3, Gemma 等。
*   **LM Studio**: 支持通过 LM Studio 在本地运行的 `gguf` 格式模型。

## 配置方法

所有模型的默认配置放在 `config.toml`，密钥通过 `.env` 或环境变量覆盖。`config.toml` 为本地文件，请从 `config.toml.example` 复制生成；`.env` 请从 `.env.example` 复制生成。

### 1. 配置访问地址

如果您使用代理、第三方中转服务或本地模型，请在 `config.toml` 中配置对应的 API 访问地址。例如，Ollama 的默认地址是 `http://localhost:11434/v1`。

Google Vertex AI 的 Application Default Credentials（ADC）通过环境变量配置，不要把凭证对象或密钥内容写入模型条目的 `options`。先完成 ADC 登录，再设置：

```dotenv
GOOGLE_GENAI_USE_VERTEXAI="true"
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_CLOUD_LOCATION="us-central1"
# 可选：服务账号文件路径（不要提交该文件）
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/service-account.json"
```

`google-genai` 也支持 `GEMINI_API_KEY` 作为 API key 别名；同时设置时优先使用 `GOOGLE_API_KEY`。未设置 API key 时，显式的 Vertex 开关允许 SDK 使用 ADC；程序化集成也可向 `GoogleProvider` 传入 `credentials`，但该对象不能放进 `config.toml`。

### 2. 协议与 SDK

不同渠道不强制使用各供应商的官方 SDK，项目按协议复用适配器：

| 渠道 | 能力 | 实现方式 |
| --- | --- | --- |
| Google | LLM、Embedding | `google-genai` SDK |
| OpenAI | LLM、Embedding | `openai` SDK |
| Qwen、SiliconFlow、Ollama、LM Studio | LLM、Embedding | `openai` SDK 的 Chat/Responses 兼容接口；资源能力按目标服务端另行确认 |
| DeepSeek、Grok | LLM | `openai` SDK 的 Chat/Responses 兼容接口；资源能力按目标服务端另行确认 |
| Anthropic | LLM | `anthropic` SDK |
| Volcengine | LLM、Embedding、Ark Responses | Volcengine Ark SDK |
| Jina、SiliconFlow | Rerank | HTTP JSON 接口（`httpx`） |

Qwen 的 OpenAI 兼容地址默认使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`；SiliconFlow 的聊天/Embedding 和 Rerank 是两种不同接口，Rerank 会由工厂映射到专用适配器。

OpenAI 兼容的 LLM 配置默认调用 Chat Completions。需要调用 Responses 时，在单个 `llm_configurations` 条目中设置 `protocol = "responses"`。Ark 也可使用同样的协议字段：

```toml
[llm_configurations.openai-responses]
provider = "openai"
model_name = "gpt-5"
protocol = "responses"
```

如果 `provider = "openai"` 使用的是代理、中转或自建 Base URL，即使它实现了 Responses，也必须先完成该渠道的实际验证，再在条目的 `options` 中登记：

```toml
[llm_configurations.openai-proxy-responses]
provider = "openai"
model_name = "your-model"
protocol = "responses"
options = { server_verified_protocols = ["responses"] }
```

官方 `https://api.openai.com/v1` 会按 OpenAI SDK 契约使用 Responses；自定义 Base URL 不会因 provider 名称为 `openai` 而自动视为官方服务。未完成服务端验证时，系统会在请求前显式报错，避免把本地 SDK 资源误发到不支持的渠道。

Ark Responses 也遵循同一门禁。使用 Volcengine 的 Responses 前，必须确认目标 Ark Base URL 提供 `/responses`，并在配置中登记 `options = { server_verified_protocols = ["responses"] }`；未登记时，工厂、Provider 和资源 Facade 都会在请求前显式拒绝。

Responses 请求会使用 `input`、`instructions` 和 `tools` 字段，并把文本增量统一转换为现有聊天流。工具定义会从 Chat Completions 的 `function` 结构转换为 Responses Function Tool；多轮历史中的 `assistant.tool_calls` 与 `role=tool` 会分别转换为 `function_call` 和 `function_call_output`，消息中的 `text`、`image_url` 和 `file` 内容块也会转换为 Responses 的 `input_text`、`input_image` 和 `input_file`。Ark Responses 另外支持 `input_audio`、`input_video` 与 `image_pixel_limit`；OpenAI Responses 对这些 Ark 专属块会在请求前显式拒绝。缺少关联 ID、无效 JSON 或不受当前 SDK 支持的内容块也会在请求前显式报错。失败、取消或不完整响应也会显式报错。Embedding 仍使用 `embeddings.create`，不会因为 LLM 协议设置而改变。

统一的高级调用入口是 Provider 的 `complete()`/`acomplete()`，支持 `messages`（多轮和多模态内容）、`tools`、`tool_choice`、`response_format`、`max_tokens`、`top_p`、`stop`、`seed` 和 `extra_body`。结果包含 `text`、`tool_calls`、`usage`、`finish_reason` 和原始 SDK 响应。旧版 `invoke()`/`ainvoke()` 仍只输出文本流，以保持聊天界面兼容。需要 reasoning、thinking、拒答或流式 usage 时使用 `stream_events()`/`astream_events()`。

多轮工具调用的历史可以原样回填：请求边界会把扁平的 `{"id","type","name","arguments"}` 归一化为各协议需要的形状，并读取 Responses `custom_tool_call` 与 Anthropic `tool_use` 使用的 `input` 别名。各协议对关联 ID 的要求不同——Chat Completions 与 Responses 要求 `assistant.tool_calls` 带 `id`（缺少时在请求前显式报错），Google Gemini 不需要，其 `FunctionCall.id` 为可选且默认缺省，因此无 id 的 Google 工具调用可以正常回填。Provider 自己产出的 `tool_calls` 都带有协议所需的标识。

非敏感供应商参数放在模型条目的 `options` 表中，工厂会拒绝 `api_key`、`access_key`、`secret_key`、`token`、`headers` 和 `base_url` 等字段：

```toml
[llm_configurations.claude-thinking]
provider = "anthropic"
model_name = "claude-sonnet-4-6"
options = { max_tokens = 8192, thinking = { type = "adaptive" } }
```

Claude Sonnet 4.6 及更新的支持自适应 thinking；Sonnet 4.5 等仅支持手动模式时，应改用 `thinking = { type = "enabled", budget_tokens = 4096 }`。

Claude 4.5 及更新模型默认按官方契约移除 `temperature`、`top_p` 和 `top_k`。如果目标是明确支持这些字段的 Anthropic-compatible 代理，可在该模型的 `options` 中显式设置 `allow_deprecated_sampling = true`；官方端点不要开启此选项。

SDK 专属资源使用显式方法，不混入普通聊天请求：官方 OpenAI Provider 提供 `files`、`batches`、`vector_stores`、`audio`、`images`、`videos`、`uploads`、`conversations`、`containers`、`fine_tuning`、`evals`、`skills`、`realtime`、`webhooks`、`admin` 和 `content_provenance_checks` 等资源；这些资源也可通过 `provider.resources.native` / `provider.async_resources.async_native` 访问当前 SDK 的完整原生树。Google 提供 `files`、`caches`、`batches`、`file_search`、图像/视频、模型、调优、`auth_tokens` 和 NextGen 资源；Anthropic 提供 token count、Message Batches、Beta Files、Models，以及 `resources.beta_agents`、`resources.beta_sessions` 等 Beta 资源树；Ark 提供 Responses、Files、Batch、Tokenization、Classification、多模态 Embedding、Context、Content Generation 和 Images。兼容渠道只保证通用 Chat/Responses/Embedding 适配，默认只通过 `.native` 暴露底层客户端，不会把官方 OpenAI 的资源接口伪装成所有服务端都支持。这些方法只证明本地 SDK 调用链已适配，未替代真实账号、模型和渠道验收。

Embedding 文档和查询会区分任务类型：Google 使用 `RETRIEVAL_DOCUMENT` 与 `RETRIEVAL_QUERY`，OpenAI 兼容/Ark 可在 `options` 中设置 `dimensions`、`encoding_format` 等参数。批处理、文件、缓存和图像等资源必须显式调用，不会被普通 RAG 流程隐式触发。

这里的支持表示本地适配器能够发出 Responses 请求，不代表每个 Qwen、SiliconFlow、Ollama、LM Studio、DeepSeek 或 Grok 的兼容服务端都已实现 `/responses`。工厂的 `protocol_status()` 会分别返回 `adapter_supported` 与 `server_verified`；当前仓库未对这些外部 Base URL 做真实渠道验收，`server_verified` 默认是 `false`。启用前请确认目标 Base URL 的服务端文档和实际能力；不支持时应删除 `protocol` 或改回 `chat_completions`。

### 3. 定义模型实例

所有可供程序使用的模型都在 `config.toml` 中以 TOML 表定义，分为三类：

*   `llm_configurations`: 聊天模型。
*   `embedding_configurations`: 向量化模型。
*   `rerank_configurations`: Rerank 精排模型。

每个模型条目的结构如下：

```toml
[llm_configurations.google-pro]
provider = "google"
model_name = "gemini-1.5-pro-latest"
```

*   **`your-custom-name`**: 您为这个模型配置起的名字。这个名字会显示在 `/config` 菜单中供您选择。例如，`google-pro`。
*   **`provider`**: 指定使用哪个模型提供商的实现。这个值必须与 `src/providers/` 目录下的某个文件名（或工厂类中的标识符）相对应。例如，`google`。
*   **`model_name`**: 要调用的实际模型名称/ID。这个值会直接传递给对应服务商的 API。例如，`gemini-1.5-pro-latest`。
*   **`protocol`**（仅 LLM 可选）: 指定渠道协议。OpenAI 兼容渠道和 Volcengine 可使用 `chat_completions`/`responses`（Volcengine 省略时为 Ark Chat）；Google、Anthropic 使用各自固定协议。Embedding 与 Rerank 配置不应填写此字段。

## 当前默认推荐组合

对于本地验证场景，当前默认组合是：

```toml
default_embedding_provider = "local-hash"

[embedding_configurations.local-hash]
provider = "local-hash"
model_name = "local-hash-256"
```

这可以保证知识库向量化、召回测试和聊天主链在没有外部 Embedding API 的情况下仍然可用。

如果您使用 OpenAI 兼容渠道作为聊天模型，可以在 `.env` 中覆盖：

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://api.openai.com/v1"
DEFAULT_LLM_PROVIDER="openai"
```

并在 `config.toml` 中保留对应模型定义：

```toml
[llm_configurations.openai]
provider = "openai"
model_name = "gpt-4o"
```

## 如何添加新模型

假设您想添加一个通过 SiliconFlow 平台提供的 `Qwen/Qwen2-57B-A14B-Instruct` 模型，可以按以下步骤操作：

1.  **确保密钥已配置**: 在 `.env` 中填入 `SILICONFLOW_API_KEY`。
2.  **编辑 `llm_configurations`**: 在 `config.toml` 中添加一个新的表：

    ```toml
    [llm_configurations.sf-qwen2-57b]
    provider = "siliconflow"
    model_name = "Qwen/Qwen2-57B-A14B-Instruct"
    ```

3.  **重启程序**: 保存 `config.toml` 文件并重新启动 `uv run main.py`。

现在，您就可以在 `/config` 菜单的“切换模型” -> “语言模型 (LLM)” 选项中看到并选择 `sf-qwen2-57b` 了。
