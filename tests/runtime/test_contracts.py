"""``SessionConfig`` 是运行期可变的独立配置入口，必须自带字段校验。

``Settings`` 只在启动时校验一次；UI（``src/ui/config_menu.py``）与库调用方拿到的是
``SessionConfig``，它们经 ``__setitem__`` 写入的值此前完全不校验。修复前实测
``chat_config["top_k"] = 10**9`` 能一路走到 ``faiss_index.search()``：FAISS 既不报错
也不截断，按 10 亿条分配（实测 10**6 时就已分配满槽位并耗时 1.56s）。
"""

import pickle

import pytest

from src.runtime.contracts import (
    KnowledgeSnapshotManifest,
    SessionConfig,
    build_session_config,
)
from src.utils.config import (
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    RetrievalMethod,
    Settings,
)


@pytest.fixture
def session_config() -> SessionConfig:
    return build_session_config(Settings(_env_file=None))


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("top_k", 10**9),
        ("top_k", 0),
        ("top_k", -1),
        ("top_k", 99999999999999999999),
        ("top_k", "5"),
        ("top_k", True),
        ("top_k", 5.5),
        ("retrieval_candidate_multiplier", 10**6),
        ("retrieval_candidate_multiplier", 0),
        ("vector_weight", -5),
        ("vector_weight", 1.5),
        ("vector_weight", float("nan")),
        ("keyword_weight", float("inf")),
        ("score_threshold", -999),
        ("score_threshold", 2.0),
        ("retrieval_method", "totally-invalid"),
        ("hybrid_fusion_strategy", "bogus"),
        ("rerank_enabled", "yes"),
        ("chat_temperature", -1.0),
        ("chat_temperature", float("nan")),
    ],
)
def test_session_config_rejects_illegal_values(session_config, field_name, bad_value):
    """各字段的非法值必须在写入时拒绝，不能拖到检索期。"""
    with pytest.raises(ValueError, match=field_name):
        session_config[field_name] = bad_value


# ── 回归：不可哈希取值让 ``ValueError`` 契约漏成 ``TypeError`` ──
#
# ``_validate_fusion_strategy`` 原写 ``value not in {"rrf", "weighted"}``。集合
# 成员判断对不可哈希的值抛 ``TypeError: unhashable type``，而 ``SessionConfig``
# 的契约是 ``ValueError``——``__setitem__`` 与构造器都据此分类，调用方的
# ``except ValueError`` 接不住 ``TypeError``。输入可达：配置来自 TOML，
# ``hybrid_fusion_strategy = ["rrf"]`` 是合法的 TOML 数组。
#
# 同文件里的 ``retrieval_method``／``rerank_enabled`` 早已显式判类型，只有这一
# 条漏了——所以本用例按字段参数化，逐个断言，而不是只测出问题的那一个。
@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("hybrid_fusion_strategy", ["rrf"]),
        ("hybrid_fusion_strategy", {"strategy": "rrf"}),
        ("hybrid_fusion_strategy", {"rrf"}),
        ("retrieval_method", ["向量检索"]),
        ("retrieval_method", {"a": 1}),
        ("rerank_enabled", ["yes"]),
    ],
    ids=[
        "fusion-list",
        "fusion-dict",
        "fusion-set",
        "method-list",
        "method-dict",
        "rerank-list",
    ],
)
def test_session_config_unhashable_values_raise_value_error(session_config, field_name, bad_value):
    """不可哈希的取值必须收敛成 ``ValueError``，不能漏出 ``TypeError``。

    断言异常类型而不是消息：这里守的是「调用方的 ``except ValueError`` 能不能
    接住」，消息措辞无关紧要。
    """
    with pytest.raises(ValueError):
        session_config[field_name] = bad_value

    with pytest.raises(ValueError):
        SessionConfig(**{**session_config.to_dict(), field_name: bad_value})


def test_session_config_rejects_both_weights_zero(session_config):
    """两个融合权重不得同时为零。

    同时为零时 ``weighted`` 融合的每个文档得分都是 0.0，再被
    ``chat_score_threshold`` 全部过滤——检索静默返回空结果且无任何报错，
    把配置错误伪装成检索质量差。``Settings`` 启动时校验过，但
    ``SessionConfig`` 是运行期独立入口，必须自己守住。
    """
    session_config["vector_weight"] = 0.0
    with pytest.raises(ValueError, match="不能同时为 0"):
        session_config["keyword_weight"] = 0.0
    # 顺序反过来也要拒绝。
    session_config["vector_weight"] = 0.5
    session_config["keyword_weight"] = 0.0
    with pytest.raises(ValueError, match="不能同时为 0"):
        session_config["vector_weight"] = 0.0


def test_session_config_constructor_rejects_both_weights_zero(session_config):
    """构造器同样要拒绝，否则 ``SessionConfig(...)`` 是绕过守卫的口子。"""
    payload = {**session_config.to_dict(), "vector_weight": 0.0, "keyword_weight": 0.0}
    with pytest.raises(ValueError, match="不能同时为 0"):
        SessionConfig(**payload)


def test_session_config_rejects_unknown_field(session_config):
    """拼错字段名应报错，而不是静默新增一个属性。"""
    with pytest.raises(KeyError, match="没有字段"):
        session_config["不存在的字段"] = 1


# ── 回归：字段名校验与兄弟字段取值不能漏出非 ValueError ──
#
# ``key not in self.to_dict()`` 对不可哈希的 key 抛 ``TypeError: unhashable
# type``，而 ``f"{key!r}"`` 对位数超限的整数抛 ``ValueError: Exceeds the limit``
# ——两条都与「字段名写错了」无关。同文件的 ``_validate_fusion_strategy`` 早已
# 认领同一类线索（见其 docstring），只有字段名这一处漏了。
#
# 兄弟字段的取值可能**没经过 validator**：``SessionConfig`` 是公开类，普通属性
# 没有 ``__setattr__`` 守卫，嵌入方可以直接 ``cfg.vector_weight = 10**400``。
# 于是组合校验里的 ``float(...)``／``int(...)`` 分别抛 ``OverflowError``（它是
# ``ArithmeticError`` 子类，**不是** ``ValueError``）与 ``ValueError``——
# 调用方按 ``ValueError`` 写的分类接不住前者。


@pytest.mark.parametrize(
    ("label", "bad_key"),
    [
        ("不可哈希的列表", [1, 2]),
        ("不可哈希的字典", {"a": 1}),
        ("位数超限的整数", 10**4301),
        ("None", None),
    ],
    ids=["list", "dict", "huge-int", "none"],
)
def test_session_config_setitem_rejects_malformed_field_names(session_config, label, bad_key):
    """字段名形态非法时必须报 ``ValueError``，不能漏出 ``TypeError`` 或位数上限错误。"""
    with pytest.raises(ValueError) as excinfo:
        session_config[bad_key] = 1

    message = str(excinfo.value)
    assert "字段名必须是字符串" in message, f"{label}: {message}"
    assert "Exceeds the limit" not in message, f"{label}: {message}"


@pytest.mark.parametrize(
    ("label", "bad_key"),
    [
        ("位数超限的整数", 10**4301),
        ("None", None),
    ],
    ids=["huge-int", "none"],
)
def test_session_config_set_many_rejects_malformed_field_names(session_config, label, bad_key):
    """``set_many`` 走同一个字段名校验，不能只修 ``__setitem__``。

    这里不测不可哈希的 key（列表／字典）：``set_many`` 收的是 ``Mapping``，
    构造 ``{[1, 2]: 1}`` 这一步在调用方就抛 ``TypeError``，够不到被测代码。
    """
    with pytest.raises(ValueError) as excinfo:
        session_config.set_many({bad_key: 1})

    assert "字段名必须是字符串" in str(excinfo.value), label


def test_session_config_setitem_still_rejects_unknown_string_fields(session_config):
    """收紧不能误伤：未知的**字符串**字段名仍然报 ``KeyError``。"""
    with pytest.raises(KeyError, match="没有字段"):
        session_config["不存在的字段"] = 1


# ── 回归：``__getitem__`` 必须与 ``__setitem__`` 同判据 ──
#
# 本类声明 ``MutableMapping[str, Any]``，而 ``__setitem__`` 会经
# ``_require_known_field`` 拒绝未知字段（``KeyError``）与非法形态
# （``ValueError``）。读取侧此前直接写 ``getattr(self, key)``，对同一 key 抛的
# 是 ``AttributeError``——判据只做了一半。
#
# 后果不是「消息不好看」：``in``／``.get(k, default)``／``.pop(k, default)``／
# ``.setdefault`` 这些 ``MutableMapping`` 的混入实现**只接 ``KeyError``**。实测
# ``"nope" in cfg`` 与 ``cfg.get("nope", "d")`` 都抛 ``AttributeError``，而它们
# 本该分别返回 ``False`` 与 ``"d"``。另外 ``getattr`` 会把 ``__class__``／
# ``__dict__``／``to_dict`` 这类非字段属性一并返回（实测 ``cfg["to_dict"]`` 拿到
# 绑定方法），而写入侧对同名 key 抛 ``KeyError``。


@pytest.mark.parametrize(
    ("label", "bad_key"),
    [
        ("未知字符串字段", "不存在的字段"),
        ("非字段属性名", "to_dict"),
        ("dunder", "__class__"),
    ],
    ids=["unknown", "method-name", "dunder"],
)
def test_session_config_getitem_rejects_unknown_fields(session_config, label, bad_key):
    """未知字段名（含非字段属性）在读取侧同样报 ``KeyError``。"""
    with pytest.raises(KeyError, match="没有字段"):
        session_config[bad_key]


@pytest.mark.parametrize(
    ("label", "bad_key"),
    [
        ("不可哈希的列表", [1, 2]),
        ("位数超限的整数", 10**4301),
        ("None", None),
    ],
    ids=["list", "huge-int", "none"],
)
def test_session_config_getitem_rejects_malformed_field_names(session_config, label, bad_key):
    """字段名形态非法时必须报 ``ValueError``，不能漏出 ``TypeError`` 或位数上限错误。"""
    with pytest.raises(ValueError) as excinfo:
        session_config[bad_key]

    message = str(excinfo.value)
    assert "字段名必须是字符串" in message, f"{label}: {message}"
    assert "Exceeds the limit" not in message, f"{label}: {message}"


def test_session_config_mapping_protocol_works_for_unknown_keys(session_config):
    """``MutableMapping`` 的混入实现（只接 ``KeyError``）必须照常工作。"""
    assert "不存在的字段" not in session_config
    assert session_config.get("不存在的字段", "dflt") == "dflt"
    assert session_config.pop("不存在的字段", "dflt") == "dflt"
    assert "top_k" in session_config
    assert session_config.get("top_k") == session_config["top_k"]


def test_session_config_getitem_still_returns_known_fields(session_config):
    """收紧不能误伤：已知字段照常可取，且与属性访问一致。"""
    assert session_config["top_k"] == session_config.top_k
    assert session_config["chat_temperature"] == session_config.chat_temperature


@pytest.mark.parametrize(
    ("label", "attribute", "value"),
    [
        ("权重超出双精度", "vector_weight", 10**400),
        ("权重是 inf", "vector_weight", float("inf")),
        ("权重是 nan", "vector_weight", float("nan")),
        ("top_k 是 inf", "top_k", float("inf")),
        ("top_k 是 nan", "top_k", float("nan")),
    ],
    ids=["weight-huge", "weight-inf", "weight-nan", "topk-inf", "topk-nan"],
)
def test_session_config_combination_check_survives_direct_attribute_writes(
    session_config, label, attribute, value
):
    """属性直写绕过 validator 后，组合校验必须仍报 ``ValueError``。

    ``SessionConfig`` 是公开类，普通属性没有 ``__setattr__`` 守卫——嵌入方
    （或携带超大整数的 pickle 快照）可以直接写 ``cfg.vector_weight = 10**400``。
    随后任一次 ``set_many``／``__setitem__`` 都会读到该兄弟字段并做运算：
    ``float(10**400)`` 抛 ``OverflowError``（``ArithmeticError`` 子类，
    **不是** ``ValueError``）、``int(inf)`` 同样抛 ``OverflowError``、
    ``int(nan)`` 抛 ``ValueError``。前两条会穿透只接 ``ValueError`` 的调用方。
    """
    object.__setattr__(session_config, attribute, value)

    with pytest.raises(ValueError) as excinfo:
        session_config.set_many({"retrieval_candidate_multiplier": 5})

    message = str(excinfo.value)
    assert attribute in message, f"{label}: {message}"
    assert "Exceeds the limit" not in message, f"{label}: {message}"


def test_candidate_pool_message_survives_huge_integer_siblings(session_config):
    """``top_k`` 被直写成位数超限整数时，组合校验的消息自己不能抛。

    消息里插值 ``{candidates}``（= ``top_k * multiplier``）撞上 CPython 的整数
    位数上限，于是在**正要报「候选规模过大」**的那一步抛 ``ValueError:
    Exceeds the limit``——类型对但内容与字段无关。
    """
    object.__setattr__(session_config, "top_k", 10**4301)

    with pytest.raises(ValueError) as excinfo:
        session_config.set_many({"retrieval_candidate_multiplier": 5})

    message = str(excinfo.value)
    assert "不能超过" in message
    assert "Exceeds the limit" not in message


@pytest.mark.parametrize(
    "field_name,good_value,expected",
    [
        ("top_k", 20, 20),
        # 单字段上界：fixture 的倍率是 3，而乘积上界是 2000，直接用
        # _MAX_RETRIEVAL_TOP_K 会超界——那正是组合校验要拦的情形（另有用例
        # 覆盖），这里取「上界内且乘积不越界」的值。
        ("top_k", _MAX_RETRIEVAL_TOP_K // 4, _MAX_RETRIEVAL_TOP_K // 4),
        ("retrieval_candidate_multiplier", _MAX_RETRIEVAL_MULTIPLIER, _MAX_RETRIEVAL_MULTIPLIER),
        ("vector_weight", 0.3, 0.3),
        ("score_threshold", 0.0, 0.0),
        ("score_threshold", 1.0, 1.0),
    ],
)
def test_session_config_accepts_legal_boundary_values(
    session_config, field_name, good_value, expected
):
    """上界与下界本身必须可用——收紧不能把合法边界一起拒掉。"""
    session_config[field_name] = good_value
    assert session_config[field_name] == expected


def test_session_config_normalizes_retrieval_method_string(session_config):
    """UI 传的是枚举值的中文 value，应被归一化成枚举成员。"""
    session_config["retrieval_method"] = "全文检索"
    assert session_config["retrieval_method"] is RetrievalMethod.FULL_TEXT_SEARCH

    session_config["retrieval_method"] = RetrievalMethod.HYBRID_SEARCH
    assert session_config["retrieval_method"] is RetrievalMethod.HYBRID_SEARCH


def test_session_config_validator_registry_covers_scalar_fields():
    """注册表必须覆盖所有标量字段。

    漏登记会让该字段回落到「原样写入」，也就是本次修复前的无校验状态——
    这类静默回归不会让任何行为测试变红，只能靠结构断言盯住。
    """
    scalar_fields = {
        "retrieval_method",
        "vector_weight",
        "keyword_weight",
        "hybrid_fusion_strategy",
        "retrieval_candidate_multiplier",
        "rerank_enabled",
        "top_k",
        "score_threshold",
        "chat_temperature",
    }
    registered = set(SessionConfig._VALIDATORS)
    assert scalar_fields <= registered, f"以下标量字段没有登记校验器: {scalar_fields - registered}"
    # 反方向：注册表里的字段必须真实存在，否则是拼错名字的死登记
    known_fields = set(build_session_config(Settings(_env_file=None)).to_dict())
    assert registered <= known_fields, f"注册表里有不存在的字段: {registered - known_fields}"


def test_session_config_to_dict_still_reflects_writes(session_config):
    """写入后 ``to_dict`` 必须能看到新值（否则 UI 显示会与实际不符）。"""
    session_config["top_k"] = 42
    assert session_config.to_dict()["top_k"] == 42


# ── 回归：构造器必须走与 __setitem__ 同一组校验器 ──


@pytest.mark.parametrize(
    "vector_weight,keyword_weight",
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (float("-inf"), 1.0),
        (-1.0, 1.0),
        (2.0, 1.0),
        (1.0, float("nan")),
    ],
)
def test_session_config_constructor_rejects_invalid_weights(
    session_config, vector_weight, keyword_weight
):
    """``SessionConfig(...)`` 不能绕过 ``__setitem__`` 的逐字段校验。

    原构造器只判「两个权重是否同时为零」，放过了 ``nan``／``inf`` 与越界值。
    实测 ``SessionConfig(vector_weight=float("nan"), ...)`` 被接受，
    ``HybridReranker`` 随后算出 ``nan`` 得分：排序结果不确定，且 ``nan`` 与
    分数阈值的比较恒为 False，整批结果被静默丢弃。
    """
    payload = {
        **session_config.to_dict(),
        "vector_weight": vector_weight,
        "keyword_weight": keyword_weight,
    }

    with pytest.raises(ValueError, match="vector_weight|keyword_weight"):
        SessionConfig(**payload)


def test_session_config_constructor_accepts_valid_weights(session_config):
    """收紧不能误伤：合法权重照常接受。"""
    payload = {**session_config.to_dict(), "vector_weight": 0.7, "keyword_weight": 0.3}

    config = SessionConfig(**payload)

    assert config["vector_weight"] == 0.7
    assert config["keyword_weight"] == 0.3


# 构造器绕过校验是**整类**问题，不只是权重：实测修复前它对 top_k、score_threshold、
# chat_temperature 等每一个受校验字段都照单全收。这里逐字段比对两条入口的结论，
# 避免以后新增字段时只补了 __setitem__ 那一侧。


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("top_k", 10**9),
        ("top_k", 0),
        ("top_k", True),
        ("top_k", 5.5),
        ("retrieval_candidate_multiplier", 10**6),
        ("score_threshold", float("nan")),
        ("score_threshold", -999),
        ("chat_temperature", float("nan")),
        ("chat_temperature", -1.0),
        ("hybrid_fusion_strategy", "bogus"),
        ("rerank_enabled", "yes"),
    ],
)
def test_session_config_constructor_matches_setitem_rejection(
    session_config, field_name, bad_value
):
    """构造器与 ``__setitem__`` 对同一非法值必须都拒绝。

    只测其中一条入口会漏掉另一条的绕过口子——``top_k=10**9`` 就是先在
    ``__setitem__`` 上修好、而构造器仍然放行，一路走到 ``faiss_index.search()``，
    FAISS 既不报错也不截断，按 10 亿条分配。
    """
    with pytest.raises(ValueError):
        SessionConfig(**{**session_config.to_dict(), field_name: bad_value})

    writable = SessionConfig(**session_config.to_dict())
    with pytest.raises(ValueError):
        writable[field_name] = bad_value


def test_session_config_constructor_and_setitem_agree_on_legal_values(session_config):
    """收紧不能误伤：合法取值在两条入口上都必须通过。"""
    payload = session_config.to_dict()

    from_constructor = SessionConfig(**payload)
    writable = SessionConfig(**payload)
    for field_name, value in payload.items():
        writable[field_name] = value

    assert from_constructor.to_dict() == writable.to_dict()


# ── 回归：候选规模的上界必须落在 top_k × multiplier 的乘积上 ──
#
# 两个字段各自有上界（top_k ≤ 2000、multiplier ≤ 100），但真正送进
# faiss_index.search() 的是乘积，而 FAISS 对超过索引条数的 k 既不报错也不
# 截断，按 k 全量分配。实测 dim=1536、k=10**6 时峰值 RSS 涨约 35 GB。


def test_session_config_rejects_oversized_candidate_pool(session_config):
    """两个字段各自合法、乘积越界时必须拒绝。"""
    payload = session_config.to_dict()

    with pytest.raises(ValueError, match="top_k × retrieval_candidate_multiplier"):
        SessionConfig(**{**payload, "top_k": 2000, "retrieval_candidate_multiplier": 100})


def test_session_config_setitem_rejects_oversized_candidate_pool(session_config):
    """``__setitem__`` 改任一个字段后都要按新组合判一次。"""
    config = SessionConfig(**{**session_config.to_dict(), "top_k": 20})
    config["retrieval_candidate_multiplier"] = 100  # 20 × 100 = 2000，边界内

    with pytest.raises(ValueError, match="top_k × retrieval_candidate_multiplier"):
        config["top_k"] = 2000  # 2000 × 100 越界


def test_session_config_accepts_candidate_pool_at_the_bound(session_config):
    """收紧不能误伤：恰好等于上界的组合必须接受。"""
    config = SessionConfig(
        **{**session_config.to_dict(), "top_k": 2000, "retrieval_candidate_multiplier": 1}
    )

    assert config["top_k"] == 2000


# ── 回归：manifest 的 TOML 序列化必须能原样读回 ──
#
# ``to_toml`` 用 ``f'key = "{value}"'`` 直接拼字符串，没有做 TOML 字符串转义，
# 而这些值都来自 ``config.toml``／``.env``：``embedding_model`` 可以是自建网关的
# 模型路径（``models\qwen-7b``）、``chunk_mode`` 与 ``snapshot_id`` 也由调用方决定。
# 实测模型名含反斜杠时 ``write_manifest`` 正常落盘，``load_manifest`` 抛
# ``TOMLDecodeError: Unescaped '\' in a string``——快照写得出、读不回，而报错
# 现场离真正的原因（写入时未转义）很远；含引号或换行的值则连落盘的 TOML 都
# 不可解析。同一份 manifest 在内存里是好的，落盘一圈就坏了。


@pytest.mark.parametrize(
    "value",
    [
        r"models\qwen-7b",  # 反斜杠：自建网关的模型路径写法
        'model"name',  # 双引号
        "model'name",  # 单引号
        "model\nname",  # 换行
        "model\tname",  # 制表符
        "model\x00name",  # 控制字符
        "model[name]",  # 方括号
        "model#name",  # 井号（TOML 注释起始）
        "model = name",  # 等号
        "普通模型名",  # 中文
    ],
)
def test_manifest_to_toml_round_trips_arbitrary_values(value):
    """任意取值都必须能写进 TOML 并原样读回。"""
    import tomllib

    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id="kb-roundtrip",
        store_type="faiss",
        embedding_provider="openai",
        embedding_model=value,
        chunk_mode="standard",
        source_digest="digest",
        document_count=1,
        chunk_count=1,
    )

    parsed = tomllib.loads(manifest.to_toml())

    assert parsed["embedding_model"] == value


@pytest.mark.parametrize(
    "field_name",
    ["snapshot_id", "store_type", "embedding_provider", "chunk_mode", "source_digest"],
)
def test_manifest_to_toml_round_trips_arbitrary_values_in_every_string_field(field_name):
    """每个字符串字段都要转义，不能只修被报告的那一个。"""
    import tomllib

    hostile = 'v"al\\ue\n[x]'
    payload = {
        "snapshot_id": "kb-roundtrip",
        "store_type": "faiss",
        "embedding_provider": "openai",
        "embedding_model": "m",
        "chunk_mode": "standard",
        "source_digest": "digest",
        "document_count": 1,
        "chunk_count": 1,
        field_name: hostile,
    }

    parsed = tomllib.loads(KnowledgeSnapshotManifest.create(**payload).to_toml())

    assert parsed[field_name] == hostile


def test_manifest_to_toml_keeps_counts_as_integers():
    """收紧不能误伤：计数字段必须仍是 TOML 整数而不是被引号包成字符串。"""
    import tomllib

    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id="kb-counts",
        store_type="faiss",
        embedding_provider="openai",
        embedding_model="m",
        chunk_mode="standard",
        source_digest="d",
        document_count=7,
        chunk_count=11,
    )

    parsed = tomllib.loads(manifest.to_toml())

    assert parsed["document_count"] == 7
    assert parsed["chunk_count"] == 11
    assert isinstance(parsed["document_count"], int)


# ---------------------------------------------------------------------------
# 超出双精度范围的整数
# ---------------------------------------------------------------------------
#
# ``float(10**400)`` 抛 ``OverflowError``——它是 ``ArithmeticError`` 的子类，
# **不是** ``ValueError``。``SessionConfig`` 的每个受校验字段都承诺抛
# ``ValueError``，库调用方与 UI 都只接 ``ValueError``；漏出 ``OverflowError``
# 会让整条 ``except ValueError`` 链失效。而 ``config.toml`` 的整数是无上界的
# TOML 整数，``SessionConfig`` 又是运行期可变入口，这条路径完全可达。
#
# 参数 id 必须显式给出：``10**4301`` 超过 ``sys.get_int_max_str_digits()``，
# pytest 为它自动生成 id 时调 ``str(val)`` 会自己抛 ``ValueError``，测试在
# **收集阶段**就崩掉。
_BEYOND_DOUBLE_INTS = [
    pytest.param(10**400, id="400digits"),
    pytest.param(-(10**400), id="negative-400digits"),
    pytest.param(10**4301, id="4301digits-over-int-max-str-digits"),
]


# ── 回归：清单的**写入**侧也要收敛非字符串取值 ──
#
# ``from_mapping``（读取侧）早已按同一判据修过，但 ``to_toml``（写入侧）原先
# 写的是 ``str(value)`` 与 ``int(value)``。同一条缺陷只修了一个方向：写入侧
# 抛出的消息与「快照写入失败」毫无关系，而 ``int(inf)`` 更抛 ``OverflowError``
# ——``ArithmeticError`` 子类，**不是** ``ValueError``。
#
# 值的来源与读取侧相同：``config.toml``、以及磁盘上的畸形快照。


@pytest.mark.parametrize("bad_value", _BEYOND_DOUBLE_INTS)
@pytest.mark.parametrize("field_name", ["snapshot_id", "store_type", "chunk_mode"])
def test_manifest_to_toml_survives_huge_integer_text_fields(field_name, bad_value):
    """文本字段是位数超限的整数时，``to_toml`` 必须给出可读结果或可读错误。"""
    payload = {
        "snapshot_id": "kb-write",
        "store_type": "faiss",
        "embedding_provider": "openai",
        "embedding_model": "m",
        "chunk_mode": "standard",
        "source_digest": "digest",
        "document_count": 1,
        "chunk_count": 1,
        field_name: bad_value,
    }
    manifest = KnowledgeSnapshotManifest.create(**payload)

    try:
        text = manifest.to_toml()
    except ValueError as exc:
        # 允许拒绝，但消息里不能是「整数转字符串超限」这种与字段无关的内容。
        assert "Exceeds the limit" not in str(exc), str(exc)
        return
    # 或者降级成量级描述并成功落盘。
    assert "Exceeds the limit" not in text


@pytest.mark.parametrize(
    ("label", "bad_value"),
    [
        ("inf", float("inf")),
        ("nan", float("nan")),
        ("负数", -1),
        ("小数", 1.5),
    ],
    ids=["inf", "nan", "negative", "float"],
)
@pytest.mark.parametrize("field_name", ["document_count", "chunk_count"])
def test_manifest_to_toml_rejects_malformed_counts(field_name, label, bad_value):
    """计数字段非法时必须报 ``ValueError``（带字段名），不能漏出 ``OverflowError``。

    ``int(inf)`` 抛 ``OverflowError``——``ArithmeticError`` 子类而**不是**
    ``ValueError``，会穿透 ``write_manifest`` 调用方的错误分类。
    """
    payload = {
        "snapshot_id": "kb-counts",
        "store_type": "faiss",
        "embedding_provider": "openai",
        "embedding_model": "m",
        "chunk_mode": "standard",
        "source_digest": "digest",
        "document_count": 1,
        "chunk_count": 1,
        field_name: bad_value,
    }
    manifest = KnowledgeSnapshotManifest.create(**payload)

    with pytest.raises(ValueError, match=field_name) as excinfo:
        manifest.to_toml()

    assert "Exceeds the limit" not in str(excinfo.value), label


@pytest.mark.parametrize("bad_value", _BEYOND_DOUBLE_INTS)
@pytest.mark.parametrize("field_name", ["document_count", "chunk_count"])
def test_manifest_to_toml_rejects_out_of_range_counts(field_name, bad_value):
    """位数超限的计数必须在**写入前**被拒，且消息里带字段名。

    原先它一路走到 f-string 插值才抛 ``ValueError: Exceeds the limit``——类型对，
    但用户看到的是整数转换问题，而真实原因是计数不合法。
    """
    payload = {
        "snapshot_id": "kb-range",
        "store_type": "faiss",
        "embedding_provider": "openai",
        "embedding_model": "m",
        "chunk_mode": "standard",
        "source_digest": "digest",
        "document_count": 1,
        "chunk_count": 1,
        field_name: bad_value,
    }
    manifest = KnowledgeSnapshotManifest.create(**payload)

    with pytest.raises(ValueError, match=field_name) as excinfo:
        manifest.to_toml()

    assert "Exceeds the limit" not in str(excinfo.value)


def test_manifest_to_toml_accepts_realistic_counts():
    """收紧不能误伤：真实量级的计数照常落盘且是 TOML 整数。"""
    import tomllib

    manifest = KnowledgeSnapshotManifest.create(
        snapshot_id="kb-ok",
        store_type="faiss",
        embedding_provider="local-hash",
        embedding_model="local-hash-256",
        chunk_mode="standard",
        source_digest="d",
        document_count=39,
        chunk_count=115,
    )

    parsed = tomllib.loads(manifest.to_toml())

    assert parsed["document_count"] == 39
    assert parsed["chunk_count"] == 115
    assert isinstance(parsed["document_count"], int)


@pytest.mark.parametrize("bad_value", _BEYOND_DOUBLE_INTS)
@pytest.mark.parametrize("field_name", ["vector_weight", "keyword_weight", "score_threshold"])
def test_session_config_rejects_weights_beyond_double_range(session_config, field_name, bad_value):
    """权重与阈值超出双精度范围时必须是 ``ValueError``，不是 ``OverflowError``。"""
    with pytest.raises(ValueError, match=field_name):
        session_config[field_name] = bad_value


@pytest.mark.parametrize("bad_value", _BEYOND_DOUBLE_INTS)
def test_session_config_constructor_rejects_temperature_beyond_double_range(bad_value):
    """构造器与 ``__setitem__`` 走同一组校验器，这条路径也必须收敛成 ``ValueError``。"""
    config = build_session_config(Settings(_env_file=None))
    with pytest.raises(ValueError, match="chat_temperature"):
        config["chat_temperature"] = bad_value


# ── 回归：快照清单字段是位数超限的整数时不得崩 ──
#
# ``from_mapping`` 原写 ``str(data[key])``。清单文件来自磁盘，而本项目加载
# pickle 格式的快照——pickle 是二进制格式、不受 CPython 十进制位数限制
# （实测 ``pickle.loads(pickle.dumps(10**4301))`` 正常往返），因此畸形快照能让
# ``str()`` 在正要解析清单时抛 ``ValueError: Exceeds the limit``，异常类型与
# 内容都与「清单字段不合法」无关。


def _manifest_mapping() -> dict[str, object]:
    return {
        "schema_version": "2",
        "snapshot_id": "snap-1",
        "created_at": "2026-01-01T00:00:00Z",
        "store_type": "faiss",
        "embedding_provider": "local-hash",
        "embedding_model": "demo",
        "chunk_mode": "standard",
        "source_digest": "deadbeef",
        "document_count": 1,
        "chunk_count": 1,
    }


@pytest.mark.parametrize(
    "field_name",
    ["schema_version", "snapshot_id", "created_at", "store_type", "source_digest"],
)
def test_snapshot_manifest_survives_huge_integer_field(field_name):
    data = _manifest_mapping()
    data[field_name] = pickle.loads(pickle.dumps(10**4301))

    manifest = KnowledgeSnapshotManifest.from_mapping(data)

    assert isinstance(getattr(manifest, field_name), str)
    assert getattr(manifest, field_name)  # 非空，仍可用于日志与报错
