# Repository Guidelines

本指南适用于仓库根目录的 PyRAG-Kit。项目要求 Python 3.11 或更高版本，使用 `uv` 管理依赖。

## Project Structure & Module Organization

- `main.py` 是交互式入口；`src/` 按职责划分为 `chat/`、`etl/`、`models/`、`providers/`、`retrieval/`、`retrieval_test/`、`runtime/`、`services/`、`ui/` 和 `utils/`。
- `scripts/` 放置知识库快照构建、发布打包和发布说明提取脚本；`tests/` 按源码模块组织 pytest 测试。
- `knowledge_base/` 保存本地 Markdown 原文；`data/kb/` 和 `data/logs/` 保存运行产物，均不应提交。仓库不再包含 Dify 上游子模块，Dify 相关逻辑以 `src/` 中的移植代码维护。
- 核心流程为 ETL -> embedding -> FAISS 快照 -> 混合检索 -> provider-backed chat。

## Build, Test, and Development Commands

```bash
uv sync
uv sync --group dev
uv run main.py --smoke-test
uv run main.py
uv run python -m scripts.embed_knowledge_base --mode standard
uv run pytest
uv run pytest tests/providers/test_factory.py
uv run python scripts/build_binary_release.py --target macos-arm64 --validate
```

`standard` 也可替换为 `hierarchical`。发布目标支持 `windows-x64`、`macos-x64`、`macos-arm64`、`linux-x64` 和 `linux-arm64`；标签 `v*` 会触发 GitHub Release 工作流。
发布脚本不做跨平台交叉编译，`--target` 必须匹配当前主机或 CI runner 的系统与 CPU 架构；目标不匹配时会在清理构建产物前显式失败。

## Quality Checks

开发依赖包含 Ruff、Bandit 和 MyPy。提交前可运行：

```bash
uv run ruff check main.py src scripts tests
uv run bandit -r main.py src scripts -ll
uv run mypy --cache-dir /tmp/pyrag-kit-mypy main.py src
uv run python -m compileall -q main.py src scripts tests
uv lock --check
git diff --check
```

MyPy 使用任务专用缓存目录，避免多个进程共享 `.mypy_cache` 造成锁等待；优先修复本次改动引入的问题，并记录未覆盖的历史基线告警。

## Coding Style & Naming Conventions

使用 4 个空格缩进、明确的公开函数类型标注和简短客观的 docstring。函数、变量和模块使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_CASE`。项目使用 Ruff 做 lint 检查，未配置独立 formatter；修改时遵循相邻代码风格并运行 `git diff --check`。保持服务、provider、检索和快照边界，不用静默回退或占位结果掩盖失败。

## Testing Guidelines

测试框架为 pytest，配置见 `pytest.ini`；该配置禁用未使用的 `langsmith` 插件。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。新增行为应在对应 `tests/` 子目录补充回归测试；优先运行相关文件，再运行完整 `uv run pytest`。仓库未设置强制覆盖率阈值。

## Commit & Pull Request Guidelines

提交历史使用 `feat:`、`fix:`、`docs:`、`build:`、`ci:`、`chore:` 等约定式前缀，并以简短祈使句说明变更。PR 应说明目的、影响范围和验证命令，注明配置或生成数据影响；只有 UI 输出改变时才附截图。发布相关变更同步更新 `CHANGELOG.md`。

## Security & Configuration Tips

用 `cp config.toml.example config.toml` 和 `cp .env.example .env` 创建本地配置。密钥只放环境变量或本地 `.env`，不得写入源码、示例、日志或提交。根项目框架采用 MIT；`src/etl/`、`src/retrieval/` 等 Dify 衍生部分仍须遵守 `DIFY_LICENSE`，并保留上游版权声明。移除上游子模块不改变这些许可义务。
知识快照保留 Dify 兼容的 pickle 文件格式；只从本项目生成的本地快照或明确受信的 legacy 文件加载，不导入不可信来源的 `.pkl` 文件。

## API Channel Integration

Provider 必须在 `src/providers/factory.py` 声明能力并通过对应抽象接口校验。OpenAI 兼容渠道复用 `openai` SDK；Google、Anthropic、Volcengine 使用各自 SDK；Jina 和 SiliconFlow Rerank 使用 HTTP JSON。新增或调整渠道时同步更新 `config.toml.example`、用户文档和 provider 契约测试。Google 的 Auth Token、Anthropic Beta 资源和 Ark Classification 资源都需通过显式 Facade/入口调用，不能把缺少资源挂载静默当成成功。

`local-hash` 是无需远端凭证的离线 Embedding provider，默认配置使用它构建本地知识快照。

LLM 配置可通过 `protocol = "responses"` 选择 OpenAI/Ark Responses；省略时保持 `chat_completions` 兼容行为。Responses 仅在目标 Base URL 确实提供 `/responses` 时可用，`protocol_status()` 的 `adapter_supported` 不等于真实服务端验收；兼容渠道和 Ark 都必须在 `options.server_verified_protocols` 中显式登记已验证协议。Ark Responses 可使用 SDK 支持的 `input_audio`、`input_video` 和 `image_pixel_limit`，OpenAI 兼容适配器不得把这些 Ark 专属块静默发送。Embedding 和 Rerank 配置不得填写 `protocol`。
