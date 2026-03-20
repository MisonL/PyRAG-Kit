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
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY="AIza..."
```

`.env` 中的同名键会覆盖 `config.toml`。

如果您使用 OpenAI 兼容渠道，可以这样配置：

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://apis.iflow.cn/v1"
DEFAULT_LLM_PROVIDER="iflow-qwen3-max"
```

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
model_name = "gpt-4o"
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
