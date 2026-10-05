import sys  # 导入 sys 模块
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.providers.__base__.model_provider import (
    LargeLanguageModel,
    RerankModel,
    TextEmbeddingModel,
)
from src.providers.factory import (
    ModelProviderFactory,  # 在这里导入 ModelProviderFactory
)
from src.utils.config import ModelDetail, Settings


# Mock 类定义
class MockLLMProvider(LargeLanguageModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """模拟 LLM 聊天响应"""
        if stream:
            yield f"Mock LLM stream response for {self.model_name}"
        else:
            yield f"Mock LLM response for {self.model_name}"

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str | None = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """模拟异步 LLM 聊天响应"""
        if stream:
            yield f"Mock Async LLM stream response for {self.model_name}"
        else:
            yield f"Mock Async LLM response for {self.model_name}"


class MockEmbeddingProvider(TextEmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """模拟文档嵌入"""
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """模拟查询嵌入"""
        return [0.2] * 10

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """模拟异步文档嵌入"""
        return [[0.1] * 10 for _ in texts]


class MockRerankProvider(RerankModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str], top_n: int) -> tuple[list[int], list[float]]:
        """模拟重排，返回原始顺序和递减的分数"""
        indices = list(range(len(documents)))
        scores = [1.0 - i * 0.1 for i in range(len(documents))]
        return indices[:top_n], scores[:top_n]

    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> tuple[list[int], list[float]]:
        """模拟异步重排"""
        indices = list(range(len(documents)))
        scores = [1.0 - i * 0.1 for i in range(len(documents))]
        return indices[:top_n], scores[:top_n]


@pytest.fixture(scope="module")
def mock_settings():
    """模拟全局设置对象"""
    mock_llm_config = ModelDetail(provider="mock_llm", model_name="mock-llm-model")
    mock_embedding_config = ModelDetail(
        provider="mock_embedding", model_name="mock-embedding-model"
    )
    mock_rerank_config = ModelDetail(provider="mock_rerank", model_name="mock-rerank-model")
    mock_siliconflow_rerank_config = ModelDetail(
        provider="siliconflow", model_name="mock-siliconflow-rerank-model"
    )
    mock_qwen_rerank_config = ModelDetail(provider="qwen", model_name="qwen-rerank-model")

    mock_settings_instance = MagicMock(spec=Settings)
    mock_settings_instance.llm_configurations = {"default_llm": mock_llm_config}
    mock_settings_instance.embedding_configurations = {"default_embedding": mock_embedding_config}
    mock_settings_instance.rerank_configurations = {
        "default_rerank": mock_rerank_config,
        "siliconflow_rerank_key": mock_siliconflow_rerank_config,
        "qwen_rerank_key": mock_qwen_rerank_config,
    }
    mock_settings_instance.siliconflow_api_key = "mock_siliconflow_api_key"
    mock_settings_instance.siliconflow_base_url = "http://mock-siliconflow-base-url.com"
    mock_settings_instance.google_api_key = "mock_google_api_key"

    return mock_settings_instance


@pytest.fixture(scope="function", autouse=True)
def patch_settings(mock_settings, monkeypatch):
    """在测试期间替换全局 get_settings 函数，使其返回模拟的 settings 对象，并修改 _provider_map"""
    with patch("src.utils.config.get_settings", return_value=mock_settings):
        # 清除 ModelProviderFactory 及其依赖模块的缓存
        # 这确保了 ModelProviderFactory 在每次测试时都使用最新的模拟配置
        modules_to_clear = [
            "src.providers.factory",
            "src.providers.google",
            "src.providers.openai",
            "src.providers.anthropic",
            "src.providers.qwen",
            "src.providers.volcengine",
            "src.providers.siliconflow",
            "src.providers.ollama",
            "src.providers.lm_studio",
            "src.providers.deepseek",
            "src.providers.grok",
            "src.providers.jina",
            "src.providers.siliconflow_rerank",
        ]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # 重新导入 ModelProviderFactory，确保它加载的是最新的版本
        from src.providers.factory import (
            ModelProviderFactory as ReloadedModelProviderFactory,
        )

        # 直接模拟 _get_provider_class 方法
        def mock_get_provider_class(provider_name: str):
            if provider_name == "mock_llm":
                return MockLLMProvider
            elif provider_name == "mock_embedding":
                return MockEmbeddingProvider
            elif provider_name == "mock_rerank" or provider_name == "siliconflow_rerank":
                return MockRerankProvider
            elif provider_name == "google" or provider_name == "qwen":
                return MockLLMProvider
            else:
                raise ValueError(f"不支持的模型提供商: {provider_name}")

        monkeypatch.setattr(
            ReloadedModelProviderFactory, "_get_provider_class", mock_get_provider_class
        )

        # 将重新导入的 ModelProviderFactory 赋值给全局 ModelProviderFactory，以便测试函数使用
        global ModelProviderFactory
        ModelProviderFactory = ReloadedModelProviderFactory
        yield


# 测试用例
def test_get_llm_provider_success():
    """测试成功获取 LLM 提供商"""
    llm_provider = ModelProviderFactory.get_llm_provider("default_llm")
    assert isinstance(llm_provider, MockLLMProvider)
    assert llm_provider.model_name == "mock-llm-model"


def test_get_llm_provider_not_found():
    """测试获取不存在的 LLM 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在LLM配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_llm_provider("non_existent_key")


def test_factory_uses_explicit_configuration_mapping():
    configurations = {"custom": ModelDetail(provider="mock_llm", model_name="custom-model")}

    provider = ModelProviderFactory.get_llm_provider("custom", configurations)

    assert provider.model_name == "custom-model"


def test_factory_does_not_pass_fixed_protocol_to_provider_constructor():
    configurations = {
        "google-model": ModelDetail(
            provider="google", model_name="gemini-model", protocol="generate_content"
        )
    }

    provider = ModelProviderFactory.get_llm_provider("google-model", configurations)

    assert isinstance(provider, MockLLMProvider)
    assert provider.model_name == "gemini-model"


def test_factory_normalizes_direct_string_protocol_values():
    provider = ModelProviderFactory._create_provider(
        "mock_llm",
        "mock-model",
        "llm",
        LargeLanguageModel,
        protocol="chat",
    )

    assert isinstance(provider, MockLLMProvider)


def test_factory_rejects_options_for_provider_without_options_parameter():
    with pytest.raises(ValueError, match="不支持 options"):
        ModelProviderFactory._create_provider(
            "mock_embedding",
            "mock-model",
            "embedding",
            TextEmbeddingModel,
            options={"dimensions": 2},
        )


def test_get_embedding_provider_success():
    """测试成功获取 Embedding 提供商"""
    embedding_provider = ModelProviderFactory.get_embedding_provider("default_embedding")
    assert isinstance(embedding_provider, MockEmbeddingProvider)
    assert embedding_provider.model_name == "mock-embedding-model"


def test_get_embedding_provider_not_found():
    """测试获取不存在的 Embedding 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在Embedding配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_embedding_provider("non_existent_key")


def test_get_embedding_provider_rejects_llm_only_provider():
    configurations = {"chat-only": ModelDetail(provider="deepseek", model_name="deepseek-chat")}

    with pytest.raises(TypeError, match="提供商 deepseek 不支持 embedding 能力"):
        ModelProviderFactory.get_embedding_provider("chat-only", configurations)


def test_get_embedding_provider_rejects_protocol_configuration():
    configurations = {
        "embedding": ModelDetail(
            provider="mock_embedding", model_name="embedding-model", protocol="responses"
        )
    }

    with pytest.raises(ValueError, match="Embedding 配置不支持 protocol"):
        ModelProviderFactory.get_embedding_provider("embedding", configurations)


def test_get_rerank_provider_rejects_protocol_configuration():
    configurations = {
        "rerank": ModelDetail(
            provider="mock_rerank", model_name="rerank-model", protocol="responses"
        )
    }

    with pytest.raises(ValueError, match="Rerank 配置不支持 protocol"):
        ModelProviderFactory.get_rerank_provider("rerank", configurations)


def test_get_rerank_provider_success():
    """测试成功获取 Rerank 提供商"""
    rerank_provider = ModelProviderFactory.get_rerank_provider("default_rerank")
    assert isinstance(rerank_provider, MockRerankProvider)
    assert rerank_provider.model_name == "mock-rerank-model"


def test_get_rerank_provider_siliconflow_mapping():
    """测试 Siliconflow Rerank 提供商的特殊映射"""
    rerank_provider = ModelProviderFactory.get_rerank_provider("siliconflow_rerank_key")
    assert isinstance(rerank_provider, MockRerankProvider)
    assert rerank_provider.model_name == "mock-siliconflow-rerank-model"


def test_get_rerank_provider_not_found():
    """测试获取不存在的 Rerank 提供商时抛出 ValueError"""
    with pytest.raises(ValueError, match="在Rerank配置中未找到key: non_existent_key"):
        ModelProviderFactory.get_rerank_provider("non_existent_key")


def test_get_rerank_provider_rejects_provider_without_rerank_capability():
    with pytest.raises(TypeError, match="提供商 qwen 不支持 rerank 能力"):
        ModelProviderFactory.get_rerank_provider("qwen_rerank_key")


def test_unsupported_provider_type():
    """测试 _get_provider_class 方法处理不支持的提供商名称"""
    with pytest.raises(ValueError, match="不支持的模型提供商: unsupported_provider"):
        ModelProviderFactory._get_provider_class("unsupported_provider")


# ── 回归：provider key 的形态校验 ──
#
# ``_get_model_detail`` 直接写 ``if provider_key not in configurations``，随后
# ``configurations[provider_key]`` 与错误消息里的 f-string 插值。三条失败路径都
# 与「配置里没有这个 key」无关：
#   ``key not in dict`` 对不可哈希的取值抛 ``TypeError: unhashable type: 'list'``
#     （它不是 ``ValueError``，穿透 ``Chatbot.reload_llm`` 的
#     ``except Exception`` 分类与上游的 ``except ValueError``）；
#   ``f"...{provider_key}"`` 对位数超限的整数抛
#     ``ValueError: Exceeds the limit (4300 digits)``——类型对，内容与字段无关。
#
# 同一个判据在 ``src/runtime/contracts.py`` 的 ``_require_known_field`` 里已经
# 认领过（同样的 ``key not in ...`` + f-string 组合），provider 工厂漏了。
# key 来源是 ``SessionConfig.active_llm_configuration`` 等可直写属性。


@pytest.mark.parametrize("bad_key", [[1], {1: 2}, {1, 2}], ids=["list", "dict", "set"])
def test_get_llm_provider_rejects_unhashable_key_readably(bad_key):
    """不可哈希的 key 必须给 ``ValueError``，不是裸 ``TypeError: unhashable type``。"""
    with pytest.raises(ValueError, match="llm_configurations"):
        ModelProviderFactory.get_llm_provider(bad_key, {})


@pytest.mark.parametrize("bad_key", [10**4301, -(10**4301)], ids=["huge", "negative-huge"])
def test_get_llm_provider_rejects_oversized_key_readably(bad_key):
    """位数超限的 key 必须给出字段名，而不是整数转换超限。"""
    with pytest.raises(ValueError) as excinfo:
        ModelProviderFactory.get_llm_provider(bad_key, {})

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "llm_configurations" in message


@pytest.mark.parametrize("bad_key", [[1], 10**4301], ids=["list", "huge-int"])
def test_get_embedding_provider_rejects_malformed_key_readably(bad_key):
    """Embedding 与 Rerank 走同一 helper，判据必须一致。"""
    with pytest.raises(ValueError, match="embedding_configurations"):
        ModelProviderFactory.get_embedding_provider(bad_key, {})


def test_get_llm_provider_still_reports_missing_key():
    """收紧不能误伤：合法的字符串 key 照常给出原有消息。"""
    with pytest.raises(ValueError, match="在LLM配置中未找到key: nope"):
        ModelProviderFactory.get_llm_provider("nope", {})


# ── 回归：provider 配置**取值**的形态校验 ──
#
# `_get_model_detail` 只做 `provider_key not in configurations` 的**成员**判定，
# 然后把 `configurations[provider_key]` 原样返回。键在、值不可用（`None`／dict／
# int）时不会走「未找到key」分支，`None` 一路带到三个入口的 `config.provider`，
# 在那里抛裸 `AttributeError`——它不是 `ValueError`，穿透
# `Chatbot.reload_llm` 的 `except ValueError` 分类。
# 值来源是 `SessionConfig.llm_configurations` 等可直写映射。


@pytest.mark.parametrize(
    "bad_detail", [None, 5, "x", {"provider": "deepseek"}], ids=["None", "int", "str", "dict"]
)
def test_get_llm_provider_rejects_non_detail_value_readably(bad_detail):
    """配置取值不是 `ModelDetail` 时必须给 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="llm_configurations"):
        ModelProviderFactory.get_llm_provider("k", {"k": bad_detail})


def test_get_embedding_provider_rejects_non_detail_value_readably():
    """Embedding 与 LLM 走同一 helper，判据必须一致。"""
    with pytest.raises(ValueError, match="embedding_configurations"):
        ModelProviderFactory.get_embedding_provider("k", {"k": None})


def test_get_rerank_provider_rejects_non_detail_value_readably():
    """Rerank 同上。"""
    with pytest.raises(ValueError, match="rerank_configurations"):
        ModelProviderFactory.get_rerank_provider("k", {"k": None})


def test_get_llm_provider_still_accepts_valid_explicit_configuration():
    """收紧不能误伤：合法的 ``ModelDetail`` 照常返回实例。"""
    configurations = {"k": ModelDetail(provider="mock_llm", model_name="m")}

    provider = ModelProviderFactory.get_llm_provider("k", configurations)

    assert provider.model_name == "m"


# ── 回归：provider 配置**容器**形态的校验 ──
#
# `_get_model_detail` 直接写 `provider_key not in configurations`。取值不可迭代
# 时抛裸 `TypeError: argument of type 'int' is not iterable`；是字符串时走**子串**
# 判定（`"k" in "k"` 为真），随后 `"k"["k"]` 抛
# `TypeError: string indices must be integers`。两者都不是 `ValueError`。
# 取值来自可直写的 `SessionConfig.llm_configurations`，经 `Chatbot.reload_llm`
# 原样传到这里。


@pytest.mark.parametrize(
    "bad_configurations", [5, "x", 10**4301, object()], ids=["int", "str", "huge-int", "object"]
)
def test_get_llm_provider_rejects_non_mapping_configurations_readably(bad_configurations):
    """配置容器不是映射时必须给 ``ValueError``，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="llm_configurations"):
        ModelProviderFactory.get_llm_provider("k", bad_configurations)


def test_get_embedding_and_rerank_providers_share_the_mapping_guard():
    """Embedding／Rerank 走同一 helper，判据必须一致。"""
    with pytest.raises(ValueError, match="embedding_configurations"):
        ModelProviderFactory.get_embedding_provider("k", 5)
    with pytest.raises(ValueError, match="rerank_configurations"):
        ModelProviderFactory.get_rerank_provider("k", 5)


def test_get_llm_provider_still_accepts_a_plain_dict():
    """收紧不能误伤：普通 ``dict`` 仍是合法映射。"""
    configurations = {"k": ModelDetail(provider="mock_llm", model_name="m")}

    provider = ModelProviderFactory.get_llm_provider("k", configurations)

    assert provider.model_name == "m"


# ── 回归：provider 配置**取值内部字段**的形态校验 ──
#
# `_get_model_detail` 只做键/容器/取值形态的校验，取值的**内部字段**
# （`provider`／`model_name`）原样交给 `_create_provider`。而 `_create_provider`
# 会把 `provider_name` 拿去 `_role_aliases.get(...)`／`_provider_map.get(...)`
# 做字典查找、`_get_provider_class` 做成员判定、并在「不支持的模型提供商」消息里
# f-string 插值；`model_name` 则原样透传给 provider 构造器（`local-hash` 直接
# `.rsplit()`）。
#
# 判据只做了一半：`832abf0` 给**诊断**入口 `_provider_info_for_role` 加了
# `provider_name` 形态守卫，展示层 `display_chat_config` 与
# `RetrievalService._get_rerank_provider` 也各自用 `require_provider_name` 收敛，
# 唯独**真正建 provider** 的 `_create_provider` 与三个 getter 漏了。同一个
# `provider=["a"]`：走诊断入口得到带指引的 `ValueError`，走 getter 得到裸
# `TypeError: unhashable type: 'list'`。
#
# 可达路径：`SessionConfig` 是公开类、普通属性没有 `__setattr__` 守卫，
# `llm_configurations`／`embedding_configurations`／`rerank_configurations`
# 的 `_VALIDATORS` 明确不登记（「按原样写入，由使用方负责其内部结构」），嵌入方
# 可以往映射里放任意取值的 detail；`Chatbot.reload_llm` 把它原样传到这里。
# `ModelDetail` 本身会拒绝非字符串 `provider`，但属性赋值没有 `__setattr__` 守卫
# （实测 `detail.provider = ["a"]` 直接成功），且 `SimpleNamespace` 替身同样能提供
# 这些属性（`require_detail_attributes` 只要求属性**存在**）。


def _detail_with(**overrides: Any) -> ModelDetail:
    """构造一个 `ModelDetail` 再直写字段，绕过 pydantic 的构造期校验。"""
    detail = ModelDetail(provider="mock_llm", model_name="m")
    for field, value in overrides.items():
        setattr(detail, field, value)
    return detail


@pytest.mark.parametrize(
    "bad_provider",
    [["a"], {"k": 1}, 10**4301],
    ids=["list", "dict", "huge-int"],
)
def test_get_llm_provider_rejects_malformed_provider_field_readably(bad_provider):
    """`detail.provider` 形态非法时必须给 ``ValueError``，不是裸 ``TypeError``／整数转换错。"""
    configurations = {"k": _detail_with(provider=bad_provider)}

    with pytest.raises(ValueError) as excinfo:
        ModelProviderFactory.get_llm_provider("k", configurations)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "provider" in message


def test_embedding_and_rerank_getters_share_the_provider_field_guard():
    """Embedding／Rerank 走同一 helper，判据必须一致。"""
    for getter, field in (
        (ModelProviderFactory.get_embedding_provider, "embedding_configurations"),
        (ModelProviderFactory.get_rerank_provider, "rerank_configurations"),
    ):
        with pytest.raises(ValueError, match="provider"):
            getter("k", {"k": _detail_with(provider=["a"])})


@pytest.mark.parametrize(
    "bad_model_name",
    [["m"], {"m": 1}, None, 5, 10**4301],
    ids=["list", "dict", "none", "int", "huge-int"],
)
def test_get_embedding_provider_rejects_malformed_model_name_readably(bad_model_name):
    """`detail.model_name` 形态非法时必须给 ``ValueError``，不是裸 ``AttributeError``。

    `model_name` 全程没有形态校验：`local-hash` 的 `_parse_dimension` 直接
    `.rsplit()`（非字符串抛裸 `AttributeError`），其它 provider 把它交给 SDK。
    位数超限的整数还会在日志 f-string 与错误消息里抛 `ValueError: Exceeds the limit`。
    """
    configurations = {"k": _detail_with(provider="local-hash", model_name=bad_model_name)}

    with pytest.raises(ValueError) as excinfo:
        ModelProviderFactory.get_embedding_provider("k", configurations)

    message = str(excinfo.value)
    assert "Exceeds the limit" not in message, message
    assert "model_name" in message


def test_getters_still_accept_valid_details():
    """收紧不能误伤：合法的 provider／model_name 照常建出实例。"""
    provider = ModelProviderFactory.get_llm_provider(
        "k", {"k": ModelDetail(provider="mock_llm", model_name="m")}
    )

    assert provider.model_name == "m"


# ── 回归：`_get_provider_class` 的白名单守卫 ──
#
# 本文件 `patch_settings` 是 `autouse=True`，它把 `_get_provider_class`
# **整体换成** `mock_get_provider_class`（含一份自己的「不支持的模型提供商」
# 分支）。于是 `test_unsupported_provider_type` 断言的消息来自替身，真实守卫
# （`factory.py` 里 `if provider_name not in _provider_map: raise ValueError`）
# 在整套测试里从未被执行——把它的 `raise` 换成 `pass`，3160 条测试全绿。
#
# 这个判据的价值在于它是**唯一**的白名单执行者：`require_provider_name` 的注释
# 写明「白名单是工厂的职责（`_get_provider_class` 会给『不支持的模型提供商』）」，
# 而 `_create_provider` 在 `_provider_map.get(...)` 返回 `None` 时会落到这里
# （`factory.py:327`）。删掉守卫后，`get_llm_provider`（经 `_create_provider`）与
# 诊断入口直调 `_get_provider_class` 都改抛裸 `KeyError: 'unknown'`，不再带指引。
#
# fixture 替换的是**重新导入**出的那个类；模块顶部导入的 `_ORIGINAL_FACTORY` 仍是
# 未被打补丁的原始类，用它才能钉住源码里那份实现。
_ORIGINAL_FACTORY = ModelProviderFactory


def test_get_provider_class_enforces_the_provider_map_whitelist():
    """未知渠道名必须给带指引的 ``ValueError``，不能是裸 ``KeyError``。"""
    real = _ORIGINAL_FACTORY._get_provider_class.__wrapped__

    with pytest.raises(ValueError) as excinfo:
        real("definitely_not_a_provider")

    assert "不支持的模型提供商: definitely_not_a_provider" in str(excinfo.value)


def test_create_provider_rejects_unknown_provider_with_value_error():
    """真正建 provider 的路径（`_create_provider`）同样收敛成 ``ValueError``。"""
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY._create_provider(
            "definitely_not_a_provider", "m", "llm", LargeLanguageModel
        )

    assert "不支持的模型提供商: definitely_not_a_provider" in str(excinfo.value)


# ── 回归：`_create_provider` 协议／能力两条守卫 ──
#
# 同一函数内四条守卫只有一条被钉住：`test_factory_rejects_options_for_provider_without_options_parameter`
# 覆盖了「构造函数不支持 options」（342 行），而 315／329 两条在整套测试里从未执行
# （`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3186 条全绿，
# 且没有任何测试断言这两条消息）。删除后行为确实改变（实测）：
#   * 315 行 `protocol` 归一化为空（`protocol=""`）→ 从 `ValueError: provider
#     protocol 不能为空。` 变成裸 `AttributeError: 'NoneType' object has no
#     attribute 'value'`（把带字段名的 `ValueError` 换成了别的异常）；
#   * 329 行 `issubclass` 失败 → **静默返回**一个不是 `expected_type` 子类的
#     实例（`local-hash` 传 `expected_type=LargeLanguageModel` 时拿到
#     `LocalHashEmbeddingProvider`），类型契约被无声破坏。
# 两条都经公开 getter 可达（`config.protocol` 与 `config.provider` 由
# `ModelDetail` 承载，`_detail_with` 直写字段即可构造）。用 `_ORIGINAL_FACTORY`
# 是因为 `patch_settings` 把真实 `_get_provider_class` 换成了替身，只有原始类能
# 走到 `_provider_map` 里的真实 provider 信息。
#
# 同一函数的 326 行（协议不在该 provider 的 `protocols` 白名单内）经甄别是**等价
# 变异**，故不补测试：删除后仍是 `ValueError`、结论不变——OpenAI 兼容渠道落到适配器
# 自己的 `_normalize_protocol`（消息换成「OpenAI 兼容渠道不支持 X 协议」），
# google/anthropic 落到 360 行（「构造函数不支持 protocol=X」）。三种收场都拒绝且
# 都带指引，没有静默接受，不满足「行为确实改变」的判据。
def test_create_provider_rejects_empty_protocol_readably():
    """`protocol=""` 归一化为空时必须给带字段名的 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY._create_provider("openai", "m", "llm", LargeLanguageModel, protocol="")

    assert "protocol" in str(excinfo.value) and "不能为空" in str(excinfo.value)


def test_create_provider_rejects_class_that_is_not_the_expected_type():
    """provider 类不是 `expected_type` 子类时必须 ``TypeError``，不能静默返回。

    `local-hash` 是纯 embedding 渠道（`LocalHashEmbeddingProvider`），把它按
    `expected_type=LargeLanguageModel` 取用时，能力表那一关（role 在
    `capabilities` 内）放行，随后 `issubclass` 判据必须拦住——否则调用方拿到
    一个不是 `LargeLanguageModel` 的实例，类型契约被无声破坏。
    """
    with pytest.raises(TypeError) as excinfo:
        _ORIGINAL_FACTORY._create_provider("local-hash", "m", "embedding", LargeLanguageModel)

    assert "需要实现 LargeLanguageModel" in str(excinfo.value)


# ── 回归：`_provider_info_for_role` 与诊断入口的六条守卫 ──
#
# 这些判据在整套测试里从未执行（行覆盖证实；把它们的 `raise` 换成 `pass` 后
# 全量 3471 条仍全绿）。删除后行为确实改变（实测，经公开诊断入口
# `capabilities`／`capability_report`／`protocol_status` 与公开 getter）：
#   * 236 模块导入失败 → 删掉 ``except ImportError`` 的 ``raise`` 后
#     ``ProviderClass`` 未赋值，函数**静默返回 None**，调用方拿到 ``None`` 并在
#     后面某处才崩；
#   * 244 模块里找不到 provider 类 → 同上，静默返回 ``None``；
#   * 277 渠道名不在 ``_provider_map`` → 带指引的 ``ValueError: 不支持的模型
#     提供商: nope`` 变成裸 ``AttributeError: 'NoneType' object has no attribute
#     'get'``（``info.get(...)``），诊断界面拿到的是崩溃而不是「渠道名写错了」；
#   * 280 角色不在该渠道的 ``capabilities`` 内 → **静默返回**该渠道的 provider
#     信息（实测返回 ``local-hash`` 的 embedding 实现），角色契约被无声破坏；
#   * 412 ``server_verified_protocols`` 非字符串非序列 → 裸 ``TypeError: 'int'
#     object is not iterable``（``for value in configured``）；
#   * 431 登记了未知协议名 → ``normalized_protocol`` 未赋值，漏出裸
#     ``UnboundLocalError: cannot access local variable 'normalized_protocol'``；
#   * 554 ``role=None`` 且渠道名无法推断出任何角色 → 消息从「无法为提供商 nope
#     推断模型角色，请显式传入 role。」退化成「模型角色必须是字符串，但得到
#     None。」——把「渠道名不存在」说成「角色名不是字符串」。
#
# 同批的 360 行经甄别是**死代码**，不补测试：它的条件
# （``protocol_value not in provider_info["protocols"]``）是 326 行那条的严格
# 子集，326 先命中，因此这条分支在现有配置下不可达。
#
# 用 `_ORIGINAL_FACTORY` 的原因同上：``patch_settings`` 替换的是重新导入的
# 那个类，只有原始类能走到真实的 ``_provider_map`` 与 provider 信息。


def test_provider_info_for_role_rejects_unknown_provider():
    """渠道名不在 ``_provider_map`` 时必须给带指引的 ``ValueError``。"""
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY._provider_info_for_role("nope", "llm")

    assert "不支持的模型提供商: nope" in str(excinfo.value)


def test_provider_info_for_role_rejects_unsupported_capability():
    """角色不在该渠道 ``capabilities`` 内必须报错。

    ``local-hash`` 是纯 embedding 渠道；按 ``llm`` 角色查询时判据必须拦住，
    否则调用方拿到 embedding 实现，角色契约被无声破坏。
    """
    with pytest.raises(TypeError) as excinfo:
        _ORIGINAL_FACTORY._provider_info_for_role("local-hash", "llm")

    assert "不支持 llm 能力" in str(excinfo.value)


@pytest.mark.parametrize(
    "options",
    [
        {"server_verified_protocols": 5},
        {"server_verified_protocols": {"responses": 1}},
    ],
    ids=["int", "dict"],
)
def test_runtime_verified_protocols_rejects_non_sequence_container(options):
    """``server_verified_protocols`` 必须是字符串或协议序列。

    映射是 ``Sequence`` 的反面：``for value in {"responses": 1}`` 会迭代出键
    而不是报错，因此这里要显式排除。
    """
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY._runtime_verified_protocols(options, frozenset({"responses"}), "qwen")

    assert "server_verified_protocols 必须是字符串或协议序列" in str(excinfo.value)


def test_runtime_verified_protocols_rejects_unknown_protocol():
    """登记了未知协议名必须给带可选值列表的消息。

    删除守卫后 ``normalized_protocol`` 从未赋值，漏出裸 ``UnboundLocalError``。
    """
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY._runtime_verified_protocols(
            {"server_verified_protocols": ["nope"]}, frozenset({"responses"}), "qwen"
        )

    message = str(excinfo.value)
    assert "不支持的 server_verified_protocols 协议: nope" in message
    assert "可选值" in message


def test_protocol_status_requires_explicit_role_for_unknown_provider():
    """``role=None`` 且渠道名不存在时必须说「无法推断角色」，而不是「角色不是字符串」。"""
    with pytest.raises(ValueError) as excinfo:
        _ORIGINAL_FACTORY.protocol_status("nope", "chat_completions")

    assert "推断模型角色" in str(excinfo.value)


@pytest.mark.parametrize(
    ("module", "class_name", "expected"),
    [
        ("src.providers.nonexistent_module", "X", ImportError),
        ("src.providers.jina", "NoSuchProviderClass", AttributeError),
    ],
    ids=["missing-module", "missing-class"],
)
def test_get_provider_class_reports_import_failures_readably(module, class_name, expected):
    """动态导入失败必须报成带模块名的异常，不能静默返回 ``None``。

    删除守卫后函数在 ``except`` 里走空，落到函数末尾隐式返回 ``None``——调用方
    拿到 ``None`` 后会在更远的地方崩，错误现场与真正的病因分开了。
    """
    real = _ORIGINAL_FACTORY._get_provider_class.__wrapped__
    saved = _ORIGINAL_FACTORY._provider_map["local-hash"]
    _ORIGINAL_FACTORY._provider_map["local-hash"] = {
        "module": module,
        "class": class_name,
        "capabilities": {"embedding"},
    }
    try:
        with pytest.raises(expected) as excinfo:
            real("local-hash")
    finally:
        _ORIGINAL_FACTORY._provider_map["local-hash"] = saved

    message = str(excinfo.value)
    assert module in message or class_name in message
