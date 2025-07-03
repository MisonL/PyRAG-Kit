# 计划：实现召回测试功能

此计划旨在以最小的侵入性、最大化地复用现有代码，将一个功能完备的召回测试模块集成到项目中。

## 第一部分：修改主程序入口 (`main.py`)

1.  **更新主菜单 `display_menu()`**:
    *   在菜单中增加一个新的选项“召回测试”，并将其置于第二位。
    *   自动将后续选项的序号顺延。更新后的菜单将如下所示：
        ```
        1. 知识库文档向量化处理
        2. 召回测试
        3. 启动聊天机器人会话
        4. 退出程序
        ```

2.  **扩展主循环 `main()`**:
    *   在处理用户输入的 `while` 循环中，为 `choice == '2'` 添加一个新的 `elif` 分支。
    *   此分支将负责调用新的召回测试功能。为保持主文件整洁并实现延迟加载，它将从一个新模块中导入并执行 `run_retrieval_test()` 函数。
    *   相应地，将原有的选项 `2` 和 `3` 的逻辑调整为 `3` 和 `4`。
    *   更新输入提示，将选项范围从 `(1-3)` 修改为 `(1-4)`。

## 第二部分：创建召回测试核心模块 (`src/retrieval_test/core.py`)

1.  **创建新文件**:
    *   在 `src/` 目录下创建一个新的子目录 `retrieval_test`，并在其中创建一个新文件 `core.py`。这种结构有助于保持代码的模块化和可维护性。

2.  **实现 `run_retrieval_test()` 函数**:
    *   此函数将作为召回测试的用户交互界面和逻辑控制器。
    *   **步骤 1：获取用户输入** - 使用 `prompt_toolkit` 提示用户输入要测试的查询语句。
    *   **步骤 2：加载配置与资源** - 通过 `get_settings()` 加载项目配置，并使用 `VectorStoreFactory.get_default_vector_store()` 获取已初始化的向量存储实例。
    *   **步骤 3：执行检索** - 调用 `src/retrieval/retriever.py` 中已经存在的 `retrieve_documents()` 函数。这个函数是项目检索能力的核心，我们直接复用它，将用户的查询、向量存储实例以及从配置中读取的检索参数（如 `top_k`, `retrieval_method` 等）传递给它。
    *   **步骤 4：格式化并展示结果** - 将 `retrieve_documents()` 返回的文档列表进行美化处理。使用 `rich` 库的 `Table` 或 `Panel`，清晰地展示每个检索结果的**排名**、**内容**、**来源**及**相关性分数**。

## 计划流程图

```mermaid
graph TD
    subgraph "阶段一: 主程序修改"
        A[开始] --> B[修改 main.py];
        B --> C[更新 display_menu 函数，增加“召回测试”];
        B --> D[更新 main 函数，为选项 '2' 添加处理逻辑];
    end

    subgraph "阶段二: 核心功能实现"
        E[创建新文件 src/retrieval_test/core.py];
        E --> F[定义 run_retrieval_test 函数];
        F --> G[获取用户查询];
        F --> H[加载配置和向量库];
        F --> I[调用现有的 retrieve_documents 函数执行检索];
        F --> J[使用 Rich 库格式化并展示检索结果];
    end

    subgraph "阶段三: 集成与完成"
        D --> K[从 main.py 中调用 run_retrieval_test];
        J --> L[召回测试功能完成];
        K --> L;
    end