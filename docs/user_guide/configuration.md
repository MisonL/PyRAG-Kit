# 配置指南

PyRAG-Kit 采用分层配置：环境变量 > `.env` > `config.toml` > 代码默认值。

## 配置原则

- `config.toml.example` 是版本库内模板文件。
- `config.toml` 只放非密钥配置，例如 base url、路径、检索参数和模型映射；该文件仅用于本地环境，默认不纳入版本控制。
- `.env.example` 是版本库内的密钥模板文件，可复制为本地 `.env`。
- `.env` 只放 API Key 和本机临时覆盖项。
- 环境变量优先级最高，适合容器和生产环境。

## 使用 `config.toml`

1. 复制模板文件：

```bash
cp config.toml.example config.toml
```

2. 编辑 `config.toml`。文件按功能分组，常用字段包括：

```toml
log_level = "WARNING"
knowledge_base_path = "knowledge_base"
default_embedding_provider = "local-hash"
chat_top_k = 5
chat_score_threshold = 0.4

[llm_configurations.google]
provider = "google"
model_name = "gemini-2.5-flash"
```

3. 修改后重启程序。
4. 默认情况下，知识库构建会在 `snapshot_root` 下生成一个新的快照目录，并更新 `ACTIVE_SNAPSHOT` 指针。`pkl_path` 仅保留为旧格式迁移入口。

## 使用 `.env`

`.env` 适合存放敏感信息，例如：

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY="your-openai-api-key"
GOOGLE_API_KEY="AIza..."
# 也可使用 google-genai 官方别名；同时设置时 GOOGLE_API_KEY 优先
# GEMINI_API_KEY="AIza..."
```

`.env` 中的同名键会覆盖 `config.toml`。

Google Vertex AI 使用 Application Default Credentials（ADC）时，不要把凭证对象或密钥内容写入 `config.toml`。
请在运行环境中完成 `gcloud auth application-default login`，或设置服务账号文件路径，并显式选择 Vertex 模式：

```dotenv
GOOGLE_GENAI_USE_VERTEXAI="true"
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_CLOUD_LOCATION="us-central1"
# 可选：改用服务账号文件；文件本身不要提交到仓库
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/service-account.json"
```

没有 `GOOGLE_API_KEY` 时，`GOOGLE_GENAI_USE_VERTEXAI=true` 会让 Provider 使用 Vertex ADC；`ModelDetail.options` 只接受非敏感参数，不能用来保存 `credentials`。

如果您使用 OpenAI 兼容渠道，可以这样配置：

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://api.openai.com/v1"
DEFAULT_LLM_PROVIDER="openai"
```

`.env` 中的 `OPENAI_API_BASE` 会覆盖 `config.toml` 的 `openai_api_base`，适合把流量指向代理、中转或自建网关。
注意 `provider = "openai"` 的条目都会走这个 Base URL，因此指向网关时，`model_name` 必须是该网关实际提供的模型 ID。

### 旧端点升级提示

配置加载时会对已知失效的旧端点发出显式警告，而不是让请求在运行期失败：

- `qwen_base_url` 为 `https://dashscope.aliyuncs.com/api/v1` 时会提示改为 `compatible-mode/v1`。
  DashScope 原生 `/api/v1` 本身仍在服务，但它的路径是
  `/services/aigc/text-generation/generation`，不提供本项目按 OpenAI 兼容协议请求的
  `{base_url}/chat/completions`。
- `volc_base_url` 为 `https://maas-api.ml-platform-cn-beijing.volces.com` 时会提示改为
  `https://ark.cn-beijing.volces.com/api/v3`；旧域名已停止服务。

自建或代理端点不受影响，警告只针对上述精确匹配的旧值。

## 主要字段

### Base URL

`openai_api_base`、`siliconflow_base_url`、`qwen_base_url`、`deepseek_base_url`、`ollama_base_url`、`lm_studio_base_url`、`volc_base_url`、`grok_base_url`

### 路径与日志

- `knowledge_base_path`
- `pkl_path`
- `snapshot_root`
- `log_path`
- `cache_path`
- `log_level`
- `log_retention_days`

### 知识库处理

- `kb_splitter_separators`
- `kb_chunk_size`
- `kb_chunk_overlap`
- `kb_child_chunk_size`
- `kb_child_chunk_overlap`
- `kb_embedding_batch_size`

### 聊天与检索

- `chat_retrieval_method`
- `chat_vector_weight`
- `chat_keyword_weight`
- `hybrid_fusion_strategy`
- `retrieval_candidate_multiplier`
- `chat_rerank_enabled`
- `chat_top_k`
- `chat_score_threshold`
- `chat_temperature`

### 模型配置

模型配置使用 TOML 表，不再使用 JSON 字符串：

```toml
[llm_configurations.demo]
provider = "openai"
model_name = "gpt-5.6-sol"
```

- `llm_configurations`: 聊天模型
- `embedding_configurations`: 向量化模型
- `rerank_configurations`: 精排模型

### 默认本地向量化

当前默认嵌入模型是 `local-hash`：

```toml
[embedding_configurations.local-hash]
provider = "local-hash"
model_name = "local-hash-256"
```

它用于本地可复现验证，不依赖外部 Embedding API。

### 切换 Embedding 需重建快照

活动快照的 `manifest.toml` 会记录构建时的 `embedding_provider` 与 `embedding_model`。
加载快照时框架会核对两者是否与当前运行配置一致，不一致会显式报错：

```
活动知识快照的 embedding 配置与当前运行配置不一致:
快照=<provider>/<model>，当前=<provider>/<model>。请重建知识快照或切换回原 embedding 配置。
```

这是刻意的硬性检查：不同模型的向量空间不兼容，静默复用旧索引会产出错误的检索结果。
切换 Embedding 模型（包括同一 provider 内换模型，例如 `text-embedding-004` → `gemini-embedding-2`）
后必须重建知识快照。
