# Difytopython 重构方案：对齐Dify核心逻辑

## 1. 总体设计目标

将当前紧耦合的脚本化实现，重构为 **“配置 + 抽象基类 + 工厂模式”** 的模块化架构。此方案旨在将Dify官方源码中先进的、松耦合的架构思想，应用到我们轻量级的本地项目中，使核心运转逻辑与Dify对齐，从而将本项目转变为 **“Dify核心逻辑的本地验证器”**。

## 2. 核心重构蓝图

我们将围绕以下三个核心支柱进行重构：

1.  **模型管理系统 (Provider)**: 实现可插拔的模型提供商。
2.  **向量存储系统 (VectorStore)**: 实现可替换的向量数据库。
3.  **文档处理流水线 (Pipeline)**: 实现模块化的数据处理流程。

---

### 支柱一：模型管理系统重构

**目标**：废除当前在 `config.py` 中硬编码实例的方式，引入Dify的工厂模式，实现模型提供商的动态加载和实例化。

**新文件结构:**

```
src/
├── providers/
│   ├── __base__/
│   │   └── model_provider.py  # (新增) 定义模型抽象基类
│   ├── factory.py             # (新增) 模型提供商工厂
│   ├── google.py              # (修改)
│   ├── openai.py              # (修改)
│   └── ... (其他provider)     # (修改)
└── utils/
    └── config.py              # (修改)
```

**实施步骤:**

1.  **定义抽象基类 (`src/providers/__base__/model_provider.py`)**:
    *   创建 `LargeLanguageModel`、`TextEmbeddingModel`、`RerankModel` 等抽象基类。
    *   每个基类都定义标准接口，例如 `invoke()` 方法。

2.  **改造具体实现 (例如 `src/providers/google.py`)**:
    *   创建 `GoogleProvider` 类，继承自对应的基类（如 `LargeLanguageModel`）。
    *   将原有的函数逻辑（如 `invoke_gemini`）封装到 `invoke` 方法中。

3.  **创建工厂 (`src/providers/factory.py`)**:
    *   创建 `ModelProviderFactory` 类。
    *   实现 `get_provider(provider_key: str)` 方法。此方法会：
        *   从 `config.py` 读取 `provider_key` 对应的配置（模型名称、API密钥等）。
        *   动态导入并实例化对应的Provider类（如 `GoogleProvider`）。
        *   返回一个符合标准接口的实例。

4.  **改造核心逻辑 (`src/chat/core.py`, `src/knowledge_base/knowledge_base.py`)**:
    *   不再从 `config.py` 直接导入模型实例。
    *   改为在使用时，通过 `ModelProviderFactory.get_provider()` 按需获取模型实例。

---

### 支柱二：向量存储系统重构

**目标**：解除与FAISS的强耦合关系，建立一个可插拔的向量数据库层，为未来支持Chroma等其他VDB做准备。

**新文件结构:**

```
src/
└── retrieval/
    ├── vdb/
    │   ├── base.py          # (新增) VDB抽象基类
    │   ├── faiss_store.py   # (新增) FAISS的具体实现
    │   └── factory.py       # (新增) VDB工厂
    └── hybrid_search.py     # (修改)
```

**实施步骤:**

1.  **定义抽象基类 (`src/retrieval/vdb/base.py`)**:
    *   创建 `VectorStoreBase` 抽象类。
    *   定义标准接口，如 `add_documents()`, `search()`, `save_local()`, `load_local()`。

2.  **封装FAISS实现 (`src/retrieval/vdb/faiss_store.py`)**:
    *   创建 `FaissStore` 类，继承自 `VectorStoreBase`。
    *   将项目中所有与FAISS直接交互的逻辑（创建索引、搜索、保存/加载PKL文件）全部移入此类。

3.  **创建工厂 (`src/retrieval/vdb/factory.py`)**:
    *   创建 `VectorStoreFactory` 类。
    *   实现 `get_vector_store()` 方法，该方法会读取 `.env` 中的配置（如 `VECTOR_STORE="faiss"`），并返回对应的VDB实例（当前是 `FaissStore`）。

4.  **改造核心逻辑 (`knowledge_base.py`, `hybrid_search.py`)**:
    *   所有需要与向量数据库交互的地方，都通过 `VectorStoreFactory` 获取 `VectorStoreBase` 实例，并调用其标准接口。

---

### 支柱三：文档处理流水线重构

**目标**：将目前混合在一起的文档加载与处理，重构为与Dify类似的、清晰的“抽取 -> 清洗 -> 分割”流水线。

**新文件结构:**

```
src/
└── knowledge_base/
    ├── extractors/          # (新增) 抽取器
    │   ├── base.py
    │   └── pdf_extractor.py
    ├── cleaners/            # (新增) 清洗器
    │   ├── base.py
    │   └── whitespace_cleaner.py
    ├── splitters/           # (新增) 分割器
    │   ├── base.py
    │   └── recursive_splitter.py
    └── knowledge_base.py    # (修改)
```

**实施步骤:**

1.  **创建独立的处理器模块**:
    *   在 `extractors`, `cleaners`, `splitters` 目录中，分别创建处理具体任务的类，并为每种类型定义一个抽象基类。

2.  **重构 `knowledge_base.py`**:
    *   其核心方法 `_load_and_process_documents` 将被重写为一个流水线管理器。
    *   它将根据文件类型和配置，依次调用不同的 **抽取器**、一系列 **清洗器** 和一个 **分割器**，最终将处理好的文档块交给向量存储系统。

---

### 3. 新架构示意图

```mermaid
graph TD
    subgraph "用户入口 (main.py)"
        A[Chat Logic] --> B{ModelProviderFactory};
        C[KB Builder] --> B;
        C --> D{VectorStoreFactory};
        C --> E{Pipeline};
    end

    subgraph "支柱一: 模型管理 (src/providers)"
        B -- get_provider("google") --> B1[GoogleProvider];
        B -- get_provider("openai") --> B2[OpenAIProvider];
        B1 -- inherits --> B_Base(ModelBase);
        B2 -- inherits --> B_Base;
    end

    subgraph "支柱二: 向量存储 (src/retrieval/vdb)"
        D -- get_store("faiss") --> D1[FaissStore];
        D1 -- inherits --> D_Base(VectorStoreBase);
    end

    subgraph "支柱三: 文档处理 (src/knowledge_base)"
        E[Pipeline Manager] --> E1[Extractor];
        E1 --> E2[Cleaner];
        E2 --> E3[Splitter];
        E3 --> D;
    end

    F[.env Config] --> B;
    F --> D;
    F --> E;
```

### 4. 实施建议

这是一个系统性的重构工程。我强烈建议我们按照上述三个支柱的顺序，分阶段进行：

1.  **第一阶段**：完成 **模型管理系统** 的重构。这是核心中的核心，能最快地让项目架构向Dify看齐。
2.  **第二阶段**：完成 **向量存储系统** 的重构。
3.  **第三阶段**：完成 **文档处理流水线** 的重构。

请审阅此调整方案。如果满意，我们将切换到“代码”模式，并从第一阶段开始实施。