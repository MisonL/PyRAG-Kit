# 更新日志

所有此项目的显著更改都将记录在此文件中。

## [1.4.0] - 2026-09-20

### Provider 与协议

- 建立统一 Provider 工厂与抽象请求模型，补齐 LLM、Embedding、Rerank 的能力声明、配置校验、缓存与同步/异步生命周期管理。
- 新增 `complete()`/`acomplete()` 高级调用入口，统一支持多轮消息、工具调用、结构化输出、多模态内容、usage 与结束原因；`invoke()`/`ainvoke()` 保留文本流兼容行为。
- 修复多轮工具调用的历史归一化：共享的 `normalize_messages` 不再强制要求工具调用带 `id`，因为 Gemini 的 `FunctionCall.id` 是可选字段且 SDK 默认 `None`，否则把上一轮工具调用回填给 Google 会在请求发出前被本地拒绝；需要 `id` 的 Chat Completions 与 Responses 改为显式校验。同一处还修复了 `input` 参数别名被静默清空，以及 `function_call`/`custom_tool_call` 类型未映射为 `function` 的问题。
- 支持 Chat Completions、Responses、Google Generate Content 和 Anthropic Messages 四类线协议；OpenAI 兼容渠道与 Ark 可通过 `protocol = "responses"` 选择 Responses，Ark 的 `ark` 取值是 Chat Completions 的别名。
- Responses 对非官方端点要求 `options.server_verified_protocols = ["responses"]` 显式登记；未登记时工厂、Provider 和资源 Facade 均在请求前拒绝，避免把本地 SDK 资源误发到未实现 `/responses` 的服务端。
- Ark Responses 在构造阶段拒绝 `instructions` 与 `caching={"type": "enabled"}` 同时出现：官方规定配置 `instructions` 后本轮请求无法写入或使用缓存，`caching` 为 `enabled` 时服务端直接报错，而 SDK 不做本地校验。错误信息会指出 `instructions` 来自 `system_prompt` 的兼容默认值，并给出改走 `messages` 的修复方式。
- 新增原生资源 Facade，显式暴露文件、批处理、缓存、向量库、token 计数、调优等 SDK 能力，不再把资源生命周期混入普通聊天请求。
- 深度适配 Google GenAI、Anthropic、Volcengine Ark、Jina 与 SiliconFlow Rerank；OpenAI 兼容渠道复用 `openai` SDK，Jina 与 SiliconFlow Rerank 使用 HTTP JSON。
- 收紧动态资源树的能力门禁：此前按路径分段判断时只认复数 `responses`，而 `input_items`（实际请求 `/responses/{id}/input_items`）不含该分段，在未登记 `server_verified_protocols` 的渠道上也能把请求发到远端；同时动态路径只做能力检查、不执行 Facade 的参数校验，可带着互斥参数直接发出。现在两条路径共用同一套校验，`extra_body` 里的 `instructions`/`caching` 同样参与互斥检查。
- 顶层 Responses item 形态（`{"type": "input_audio", ...}`）补齐变体校验：此前只在缺少 `content` 键时检查，加一个无关的 `content` 键即可改走消息分支绕过全部校验；现在按 `type` 分流并复用同一套内容块实现。
- 修复 Responses 流式工具调用的 `id` 语义：`id` 此前在 delta 事件取输出项 ID、在 done 事件取调用 ID、合并时又互相覆盖，同一轮工具调用在 delta 与 completed 事件里得到不同的值，调用方按 `tool_call["id"]` 回填 `role=tool` 时会配不上。现在与非流式路径统一取调用标识符。
- 放宽 Ark Responses 的失败判定：内置工具调用项（`web_search_call`、`mcp_call`、`mcp_list_tools`、`image_process`、`agent_tool_call` 等）都是 SDK 输出项联合的正式成员，且都是需要后续轮次的中间态，此前被误判为「响应结构与预期不符」而无法处理。空摘要的 `reasoning` 不在豁免之列——它不是工具调用，豁免会让「completed 但什么都没有」静默返回空成功。
- 动态资源路径的能力门禁补齐：兼容渠道的路径映射此前只覆盖 responses/files，其余资源树（batches、vector_stores、images 等）拿到空能力直接放行，而显式 Facade 名会被拦；Ark 的动态路径按动词白名单决定是否做参数校验，漏掉 `async_create` 这类写法。两者都改为按资源分段与参数内容判断。判断依据用「参数里是否出现受校验字段」而不是动词，`extra_body` 不在此列——它是 retrieve/delete/list 的合法参数，放进触发集会把只读操作误判为创建请求、要求提供 `model` 与 `input`，使 Facade 可用而动态路径不可用。
- 资源代理不再经私有名转发到底层 SDK：`proxy.__dict__["_client"]` 此前能拿到未包装的原始客户端，绕开全部凭证扫描与能力门禁；`dir()` 也把这些通路提示给使用者。出口只保留文档化的 `.native`。
- 修正动态资源路径的能力映射与官方声明集不一致：SDK 客户端的属性是复数 `moderations`，段映射只写单数会让 `resources.moderations.create` 拿到空能力直接放行，而 `create_moderation` 会被拦；同时段映射列出了 `skills`/`realtime`/`webhooks`/`admin`/`content_provenance_checks` 五项官方声明集之外的能力名，把官方端点从放行变成显式报错——而这些资源在 `capabilities` 里声明为可用。段映射现在限定在声明集内。
- Google 的实验性资源门禁此前只认显式 Facade 名：动态路径是点分形式（`google.interactions.create`），资源名是独立分段，按前缀匹配全部落空、直接放行——用户拿一个不含该资源的客户端走 `resources.interactions` 就能绕开。改为同时按路径分段匹配。
- Embedding 区分文档与查询任务类型：Google 使用 `RETRIEVAL_DOCUMENT` 与 `RETRIEVAL_QUERY`。

### 安全

- 新增配置、请求与日志边界的凭证校验：模型级 `options` 与请求级扩展不得携带密钥、请求头、query 或 Base URL 覆盖，资源 Facade 的关键字与位置参数同样受检。
- 凭证键识别补齐火山引擎的 `ak`/`sk` 与 Azure 存储的 `account_key`；此前这些键可绕过请求覆盖校验，而 `api_key` 会被拒绝。
- 日志与错误信息统一脱敏常见凭证表示（Bearer、API key、URL userinfo 与 query）；补齐 `ak=`/`sk=`/`account_key=` 形式，以及服务端错误里「首尾可见、中间掩码」的凭证形态（如 `sk-abc***...***xyz`，此前会原样落进日志）。
- 修复脱敏日志 formatter 的缓存泄漏：`logging.Formatter` 会把格式化后的 traceback 缓存在 `record.exc_text` 上供后续 handler 复用，先格式化后脱敏会让同一 logger 上的非脱敏 handler 输出原始凭证；现在脱敏结果会写回缓存。
- 文本脱敏的键名识别与配置边界对齐：此前词表更窄，且键名前要求非字母数字字符，`dbPassword`、`myApiKey` 这类驼峰键在配置边界被拒绝、在日志里却明文输出。同时给值加上形态约束，避免 `auth: none`、`cookie: enabled` 这类普通文本被误判为凭证而让合法 options 在边界被拒。
- 词边界改用「非 ASCII 字母数字」而非 `\b`：中文属 `\w`，`\b` 在 `密钥sk-...` 两侧都不成立，国产网关（SiliconFlow、Ark、DashScope）的中文错误消息此前会让密钥整串落进终端与日志。
- 键名识别改用「非字母数字或 camelCase 边界」：既让 `dbPassword`、`myApiKey` 这类驼峰键脱敏，又不把 `topsecret`、`sessiontoken` 从词中间切开（配置边界按 camelCase 切词判它们非敏感，切开会导致合法配置被误拒）。排除误判源改用非凭证字面量枚举，而不是值的长度/字符构成——后者会把 `password: FakePwOnly` 这类短纯字母凭证一并放过。
- 修复掩码规则的二次回溯：尾部前瞻逐字符重复扫描剩余串，在尾部无冒号时退化成 O(n²)（4000 字符的掩码密钥前缀加长 token 耗时 1.4 秒）。该函数挂在每条日志的 formatter 上，一个回显错误体的网关响应即可拖住进程。
- 掩码规则不再吞掉键值对的键名后留下孤儿值：`sk-abc***xxxtoken=<secret>` 里的键名不是敏感键（按 camelCase 切词判非敏感），键值规则不会接手，而掩码又已把键名吃掉，值失去锚点后明文落进日志。现在掩码匹配把键值尾部一并纳入，只保留键名作为可读上下文。
- 修复替换函数里的二次回溯：尾部游程查找用无锚点的 `re.search(r"[A-Za-z0-9_-]+$")`，在每个起点重试 `+$`；尾部以 `.` 这类掩码占位符收尾时游程够不到 `$`，退化成 O(n²)（32010 字符 9.6 秒，4010 字符 86ms）。仓库原有线性测试的用例恰好无尾随字符，所以只跑 1.2ms 就通过、没能拦住。改为单遍扫描，并给该测试补上以掩码占位符收尾的形态。
- 非凭证字面量枚举改为大小写不敏感：`auth: None`、`token: Null`、`cookie: Enabled` 此前不被识别为状态字面量而判成凭证，`find_sensitive_option_paths` 据此把合法配置在边界误拒。
- 文本脱敏的界断言补齐第三条驼峰分支：`is_sensitive_option_key` 的切词含「大写串接小写词」，文本侧只实现了「非字母数字」与「小写接大写」两条，`HTTPBearer`/`HTTPSSecret` 这类首字母缩写键在配置边界判敏感、在日志里却明文输出。
- 配置校验失败不再把凭证交给解释器默认 handler：`Settings()` 在 import 期就被调用（`log_manager.get_module_logger`），早于任何入口的 `try`，pydantic 默认渲染会把 `input_value` 明文打印。现在在 `get_settings` 边界重抛已脱敏的消息，并给 `run_cli` 补上覆盖整个启动阶段的异常边界。
- 兼容渠道不再被当作官方 OpenAI 端点放行 SDK 专属资源；仅精确匹配 `https://api.openai.com/v1` 时按 SDK 契约放行。
- 收紧失败语义：凭证、SDK 或远端调用失败时显式报错，不再以空 embedding、零分 rerank 或占位回答掩盖失败。

### 移除与调整

- 移除 Dify 上游子模块；`src/etl/`、`src/retrieval/` 等移植代码继续遵守 `DIFY_LICENSE` 并保留上游版权声明。
- 清理死代码：移除 `has_system_message`、`_configured_extra_body_keys`、`SnapshotRepository.write_stats`/`load_stats`/`load_active_manifest`，以及 `recursive_text_splitter` 中未被引用的 `DEFAULT_CHILD_CHUNK_*` 常量。这些逻辑均已在调用点内联或由框架通过装饰器调用，保留会造成同一行为的两处实现。
- 移除 iFlow 渠道及其配置示例。
- 移除 qwen rerank 配置示例；该渠道不提供 rerank 能力，调用时会显式报错。
- 发布脚本新增目标环境校验，`--target` 与主机系统/架构不匹配时在清理构建产物前显式失败，不再生成错误架构的发布包。
- Qwen Base URL 改为 OpenAI 兼容接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`；Volcengine Base URL 改为 Ark API `https://ark.cn-beijing.volces.com/api/v3`。配置仍为旧值时会在加载时给出显式升级警告：DashScope 原生 `api/v1` 不提供本仓库请求的 `{base_url}/chat/completions` 路径，火山旧域名 `maas-api.ml-platform-cn-beijing.volces.com` 则已停止服务。百炼的 `/responses` 只挂在业务空间专属域名下，默认共享域名不可用，文档已补充说明。
- Jina Rerank 与 SiliconFlow Rerank 一致，`return_documents` 固定为 `True` 且不允许被 `options` 覆盖。

### 测试与文档

- 新增 Provider 协议适配、SDK 能力、凭证边界、Rerank 契约与失败语义回归测试；测试总数增至 1037。新增断言经红绿验证（回退对应源码后测试变红）；其中一部分是**反向守卫**——回退源码不会变红，需要定向变异（例如「让守卫连 Ark 一起拒」）才能验证，这类断言守的是「不能过度收紧」，同样有区分度。
- 修复三处测试有效性缺陷：掩码尾部可见片段的断言用 `startswith("[REDACTED]")`，而掩码前缀在任何实现下都会被替换成 `[REDACTED]`，该断言恒真——尾部片段原样泄漏时也能通过，改为全文相等；短纯字母凭证的 7 个用例里有 4 个取值 ≥12 字符，旧规则本就能命中、对被测属性零区分度，改为真正短于 12 字符的值并断言值本身被替换；掩码线性度用例用单次采样配 100ms 阈值，实测本机 p99.9 即到 100ms、满载时更高，会间歇误报，改为放大规模到线性与二次相差三个数量级、取多轮最小值。
- 为活动快照的 embedding 兼容性检测补充回归测试：快照记录的 embedding provider 或模型名与当前运行配置不一致时必须显式失败，避免用错向量空间后静默产出错误检索结果。
- 引入 Ruff、Bandit 与 MyPy 到开发依赖，并补齐对应配置；新增 `.github/workflows/quality.yml`，在 PR 与 `main` 推送时执行格式化检查、lint、类型检查、安全扫描与完整测试，此前这些工具只在本地手动运行。
- 全仓库应用 `ruff format`（行宽 100、双引号、4 空格），此前未配置 formatter，`main.py`、`src/`、`scripts/`、`tests/` 中存在混用单引号、行尾空白和手工对齐等不一致；格式化只改表示不改语义，已用 AST 比对确认语法树等价，`AGENTS.md` 的质量检查与风格段落同步更新。
- 更新 `README.md`、`AGENTS.md` 与 `docs/`，同步协议选择、`options` 边界、原生资源入口和 Vertex ADC 配置口径。
- 核对各渠道官方现状并刷新配置示例：Google Embedding 改用 `gemini-embedding-2`（`text-embedding-004` 已于 2026-01-14 退役），OpenAI 改用 `gpt-5.6-*`（`gpt-3.5-turbo` 于 2026-10-23 退役），Anthropic 改用 `claude-sonnet-4-6`（`claude-3-5-sonnet-20240620` 已于 2025-10-28 退役），DeepSeek 改用 `deepseek-v4-pro`/`deepseek-v4-flash`（`deepseek-chat` 已于 2026-07-24 退役），火山改用 `doubao-seed-2-0-lite-260428` 与 `doubao-embedding-text-240715`（`doubao-pro-32k`、`bge-large-zh` 已不在方舟模型列表），Qwen 改用 `qwen3.8-max`/`qwen3.7-plus`，SiliconFlow 改用 `Qwen/Qwen3.5-27B`/`deepseek-ai/DeepSeek-V3.2`（复核发现 `Qwen/Qwen3-8B` 已不在该平台在售列表，其 Qwen 对话模型现从 Qwen3.5-27B 起步），Grok 改用 `grok-4.6`（`llama3-70b-8192` 实为 Groq 的 ID，不属于 xAI），Ollama 改用 `llama3.1`/`gemma3`。示例配置的键名同步改为与模型一致，模型内置默认值一并更新；内置 embedding 与 rerank 兜底值此前含无效 ID（Google 的裸名 `embedding-001`、SiliconFlow 不存在的 `alibaba/` 命名空间与已下线的 `bge-reranker-large`、Ollama 的 `llama3`），现分别改为 `gemini-embedding-2`、`BAAI/bge-large-zh-v1.5`、`BAAI/bge-reranker-v2-m3` 与 `llama3.1`。
- 需要重建知识库快照：`gemini-embedding-2` 与 `text-embedding-004` 的向量空间不兼容，旧 FAISS 索引不能复用。火山的 `doubao-embedding-text-240715` 已于 2025-12-26 停止新购（EOM），且已不在方舟「向量化能力」模型列表中；官方下线公告给出的迁移目标是 `doubao-embedding-vision-251215`，该模型同时接受纯文本输入，但它按 `/api/v3/embeddings/multimodal` 提供，与本项目使用的文本 `embeddings.create` 路径不同，迁移前需实测。embedding 模型只有 EOM 阶段、不涉及 EOS，存量接入点不受影响，因此示例配置暂未改动该值。
- 修复 Anthropic 采样字段弃用识别：家族名不再限定 `opus`/`sonnet`/`haiku`，覆盖 5 代新增的 `claude-fable-5`、`claude-mythos-5` 等命名；此前这些模型会被透传 `temperature`/`top_p`/`top_k`，而 Python SDK v1.0+ 已移除这些参数，请求会直接失败。

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
