# 更新日志

所有此项目的显著更改都将记录在此文件中。

## [1.4.0] - 2026-09-20

### Provider 与协议

- 建立统一 Provider 工厂与抽象请求模型，补齐 LLM、Embedding、Rerank 的能力声明、配置校验、缓存与同步/异步生命周期管理。
- 新增 `complete()`/`acomplete()` 高级调用入口，统一支持多轮消息、工具调用、结构化输出、多模态内容、usage 与结束原因；`invoke()`/`ainvoke()` 保留文本流兼容行为。
- 支持 Chat Completions、Responses、Google Generate Content 和 Anthropic Messages 四类线协议；OpenAI 兼容渠道与 Ark 可通过 `protocol = "responses"` 选择 Responses，Ark 的 `ark` 取值是 Chat Completions 的别名。
- Responses 对非官方端点要求 `options.server_verified_protocols = ["responses"]` 显式登记；未登记时工厂、Provider 和资源 Facade 均在请求前拒绝，避免把本地 SDK 资源误发到未实现 `/responses` 的服务端。
- 新增原生资源 Facade，显式暴露文件、批处理、缓存、向量库、token 计数、调优等 SDK 能力，不再把资源生命周期混入普通聊天请求。
- 深度适配 Google GenAI、Anthropic、Volcengine Ark、Jina 与 SiliconFlow Rerank；OpenAI 兼容渠道复用 `openai` SDK，Jina 与 SiliconFlow Rerank 使用 HTTP JSON。
- Embedding 区分文档与查询任务类型：Google 使用 `RETRIEVAL_DOCUMENT` 与 `RETRIEVAL_QUERY`。

### 安全

- 新增配置、请求与日志边界的凭证校验：模型级 `options` 与请求级扩展不得携带密钥、请求头、query 或 Base URL 覆盖，资源 Facade 的关键字与位置参数同样受检。
- 日志与错误信息统一脱敏常见凭证表示（Bearer、API key、URL userinfo 与 query）。
- 兼容渠道不再被当作官方 OpenAI 端点放行 SDK 专属资源；仅精确匹配 `https://api.openai.com/v1` 时按 SDK 契约放行。
- 收紧失败语义：凭证、SDK 或远端调用失败时显式报错，不再以空 embedding、零分 rerank 或占位回答掩盖失败。

### 移除与调整

- 移除 Dify 上游子模块；`src/etl/`、`src/retrieval/` 等移植代码继续遵守 `DIFY_LICENSE` 并保留上游版权声明。
- 移除 iFlow 渠道及其配置示例。
- 移除 qwen rerank 配置示例；该渠道不提供 rerank 能力，调用时会显式报错。
- 发布脚本新增目标环境校验，`--target` 与主机系统/架构不匹配时在清理构建产物前显式失败，不再生成错误架构的发布包。
- Qwen Base URL 改为 OpenAI 兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`；Volcengine Base URL 改为 Ark API `https://ark.cn-beijing.volces.com/api/v3`。

### 测试与文档

- 新增 Provider 协议适配、SDK 能力、凭证边界、Rerank 契约与失败语义回归测试；测试总数增至 760。
- 引入 Ruff、Bandit 与 MyPy 到开发依赖，并补齐对应配置。
- 更新 `README.md`、`AGENTS.md` 与 `docs/`，同步协议选择、`options` 边界、原生资源入口和 Vertex ADC 配置口径。

## [1.3.0] - 2026-03-20

### 运行与配置

- 将主配置迁移到 `config.toml`，并明确 `config.toml` 为本地文件，不再纳入版本控制。
- 新增 `local-hash` 本地嵌入提供方，默认知识库构建链路不再依赖外部 Embedding API。
- 新增 `iflow-qwen3-max` 的 OpenAI 兼容配置示例，并补齐 `config.toml.example` 与 `.env` 的文档口径。

### 稳定性与检索

- 修复聊天会话和召回测试中的异步输入问题，消除 `asyncio.run()` 与事件循环冲突。
- 收紧 Provider 失败契约：当凭证、SDK 或远端调用失败时，统一改为显式报错并保留日志，不再返回占位回答、空 embedding 或零分 rerank 结果。
- 修复聊天 `/config` 切换 LLM 时的状态回滚问题；当新模型加载失败时，活动模型配置会回退到旧值，避免界面状态与实际模型实例不一致。
- 修正 `grok` 与 `volcengine` Provider 的真实导入链路，并新增 Provider 导入烟测，避免“测试通过但模块无法导入”的假收敛。
- 优化知识库重建流程，向量化时不再加载旧索引追加写入。
- 增强本地检索质量：索引文本引入来源文件标题上下文，聊天检索使用意图文本而不是原始提示约束语。
- 完成父子分段检索、RRF 融合、sidecar 父文档存储等 RAG 核心能力的适配与校验。
- 将知识库持久化升级为快照目录，新增 `ACTIVE_SNAPSHOT` 指针、快照清单和按快照加载的运行时流程。
- 引入 `RunConfig`、`SessionConfig`、`KnowledgeBuildService`、`RetrievalService`、`ChatService` 等分层运行时服务。
- 修正真实交互场景中的展示与输入边界：来源路径改为相对路径，聊天空输入不再触发检索和模型调用。

### 测试与文档

- 增加多组回归测试，覆盖本地嵌入、异步交互、索引文本构建、RRF 行为、sidecar 持久化和配置默认值。
- 更新 `README.md`、`docs/` 和 `AGENTS.md`，同步当前启动方式、配置结构和运行口径。

## [1.2.0] - 2025-07-03

### 架构重构

- 完成第二阶段重构：解耦向量存储。
  - 定义了 `VectorStoreBase` 抽象基类，统一了向量存储接口。
  - 实现了 `FaissStore`，将 FAISS 逻辑封装其中。
  - 创建了 `VectorStoreFactory`，实现了向量存储的动态加载。
- 完成第三阶段重构：构建文档处理流水线。
  - 创建了 `etl` 模块，并为 `extractors`、`cleaners`、`splitters` 定义了抽象基类。
  - 实现了针对 Markdown 的抽取器、基础文本清洗器和递归文本分割器。
  - 创建了 `PipelineManager`，实现了数据处理的动态组合。
- 完成第四阶段重构：提升健壮性与开发者体验。
  - 为核心模块如模型提供商工厂、向量存储工厂、ETL 流水线编写了全面的单元测试。
  - 优化了日志记录，并完善了项目文档。

### 许可证合规性

- 检查了 Dify 的许可证，并为项目添加了 `DIFY_LICENSE` 文件。
- 为项目中非 Dify 移植代码的部分添加了 `LICENSE` 文件。
- 在所有移植自 Dify 的代码文件顶部添加了许可证声明。
- 在 `README.md` 中明确列出了所有移植自 Dify 的代码文件。

## [1.1.0] - 2025-07-02

### 架构重构

- 明确项目目标为 Dify 核心逻辑的本地验证器，通过轻量级 Python 实现对齐并验证 Dify 的核心工作流。
- 启动向配置、抽象基类和工厂模式的系统性重构。
- 完成第一阶段重构：模型管理系统。
  - 废除了在代码中硬编码模型实例的方式。
  - 建立了 `LargeLanguageModel`、`TextEmbeddingModel`、`RerankModel` 等模型抽象基类。
  - 实现了 `ModelProviderFactory`，用于根据配置动态加载和实例化不同的模型提供商。

### 新功能与优化

- 优化启动界面。
  - 新增由 `pyfiglet` 生成的 ASCII Art 启动横幅。
  - 为标题实现从左到右的蓝红颜色渐变效果。
  - 添加了包含版本、描述、作者和 GitHub 链接的欢迎面板，并确保其宽度与标题对齐。
- 重构配置系统。
  - 从 `.env` 和 `python-dotenv` 迁移到 `config.ini` 和内置的 `configparser`，以解决复杂 JSON 配置解析问题。
  - 实现环境变量优先的配置加载策略，当环境变量存在时会覆盖 `config.ini` 中的设置。
  - 更新 Google LLM 的默认模型为 `gemini-2.5-flash` 和 `gemini-2.5-pro`。
- 提升代码质量。
  - 解耦了 Rerank 提供商，通过依赖注入将 `top_n` 作为参数传递给 `rerank` 方法，移除了对全局配置的直接依赖。
