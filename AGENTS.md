# 仓库指南 (Repository Guidelines)

## 项目结构与模块组织 (Project Structure & Module Organization)

本仓库是一个 Python 3.11+ 的 RAG 工具集。核心代码位于 `src/` 目录下，按职责拆分为以下主要模块：
`src/chat/`、`src/providers/`、`src/retrieval/`、`src/etl/`、`src/ui/` 和 `src/utils/`。入口程序和脚本位于
`main.py` 和 `scripts/`。测试代码位于 `tests/`，其目录结构与源码对应。运行产物（如 `data/` 目录和生成的日志）不应提交至版本控制。

## 构建、测试与开发命令 (Build, Test, and Development Commands)

- `uv sync`：根据 `pyproject.toml` 安装依赖并创建本地环境。
- `uv run main.py`：启动应用程序。
- `uv run python -m scripts.embed_knowledge_base`：重建知识库向量化缓存。
- `uv run pytest`：运行完整测试套件。
- `uv run pytest tests/test_config.py`：在迭代特定更改时运行单个测试文件。

## 编码风格与命名规范 (Coding Style & Naming Conventions)

遵循现有的 Python 风格：使用 4 空格缩进，公开函数必须包含明确的类型标注，命名应具有描述性。
函数、变量和模块使用 `snake_case`；类使用 `PascalCase`；常量使用 `UPPER_CASE`。保持文档字符串（docstrings）和注释简短且客观。修改代码时应贴合周围代码风格，避免引入新的格式化样式。

## 测试指南 (Testing Guidelines)

所有自动化测试统一使用 `pytest`。测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。建议将针对特定行为的测试放在该行为相关的代码附近，特别是对于配置解析、提供商工厂、ETL 和检索逻辑。项目没有强制的覆盖率阈值；针对每次行为变更都应补充测试，并优先运行最小相关测试集。

## 提交与拉取请求指引 (Commit & Pull Request Guidelines)

Git 提交历史应使用约定式前缀，如 `feat:`、`refactor:`、`test:` 和 `chore:`，后接简短摘要。提交信息应具体且使用祈使句。拉取请求（PR）需说明改动内容，列出验证命令，并注明是否对配置或数据文件产生影响。仅在 UI 输出发生变化时附带截图。

## 安全与配置提示 (Security & Configuration Tips)

严禁提交密钥或本地覆盖配置。请复制 `config.toml.example` 为本地 `config.toml`，并使用 `.env` 文件处理私密数值。`config.toml` 为本地文件，默认不纳入版本控制。将 `data/` 目录和生成的日志视为本地产物。若在 `dify/` 子目录下工作，请先阅读该目录下的 `AGENTS.md`，因为子目录指令优先于此根目录指南。
