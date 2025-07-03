# 大语言模型 (LLM) 支持

PyRAG-Kit 的一个核心优势是其高度的灵活性和可扩展性，尤其体现在对多种大语言模型 (LLM) 提供商的无缝支持上。您可以根据自己的需求和资源，轻松切换和配置不同的模型。

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

所有模型的配置都集中在 `config.ini` 文件中（或通过 `.env` / 环境变量覆盖）。

### 1. 配置 API 密钥和访问地址

*   **`[API_KEYS]`**: 在此部分填入您从各云服务平台获取的 API 密钥。
*   **`[BASE_URLS]`**: 如果您使用代理、第三方中转服务或本地模型，请在此处配置对应的 API 访问地址 (Base URL)。例如，Ollama 的默认地址是 `http://localhost:11434/v1`。

### 2. 定义模型实例

所有可供程序使用的模型都在 `[MODEL_CONFIGURATIONS]` 部分以 JSON 格式定义。这里分为三类：

*   `LLM_CONFIGURATIONS`: 定义聊天模型。
*   `EMBEDDING_CONFIGURATIONS`: 定义向量化模型。
*   `RERANK_CONFIGURATIONS`: 定义 Rerank 精排模型。

每个模型条目的结构如下：

```json
"your-custom-name": {
    "provider": "provider_name_in_src",
    "model_name": "actual_model_id_for_api"
}
```

*   **`your-custom-name`**: 您为这个模型配置起的名字。这个名字会显示在 `/config` 菜单中供您选择。例如，`"google-pro"`。
*   **`provider`**: 指定使用哪个模型提供商的实现。这个值必须与 `src/providers/` 目录下的某个文件名（或工厂类中的标识符）相对应。例如，`"google"`。
*   **`model_name`**: 要调用的实际模型名称/ID。这个值会直接传递给对应服务商的 API。例如，`"gemini-1.5-pro-latest"`。

## 如何添加新模型

假设您想添加一个通过 SiliconFlow 平台提供的 `Qwen/Qwen2-57B-A14B-Instruct` 模型，可以按以下步骤操作：

1.  **确保密钥已配置**: 在 `config.ini` 的 `[API_KEYS]` 中填入 `SILICONFLOW_API_KEY`。
2.  **编辑 `LLM_CONFIGURATIONS`**: 在 `config.ini` 的 `LLM_CONFIGURATIONS` JSON 块中，添加一个新的键值对：

    ```json
    "sf-qwen2-57b": {"provider": "siliconflow", "model_name": "Qwen/Qwen2-57B-A14B-Instruct"},
    ```

3.  **重启程序**: 保存 `config.ini` 文件并重新启动 `main.py`。

现在，您就可以在 `/config` 菜单的“切换模型” -> “语言模型 (LLM)” 选项中看到并选择 `sf-qwen2-57b` 了。