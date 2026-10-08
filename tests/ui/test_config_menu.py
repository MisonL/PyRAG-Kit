"""配置编辑器的上界必须与 `SessionConfig`/`Settings` 的守卫一致。

UI 的校验器如果只按单字段上界放行，用户设一个「字段合法、组合越界」的值后，
`__setitem__` 会在应用配置时抛 `ValueError`——而该异常落在 chat 主循环之外
（主循环只捕获 `KeyboardInterrupt`/`EOFError`，外层 `try` 只有 `finally`），
会以裸 traceback 终止整个会话。
"""

from __future__ import annotations

import pytest
import questionary

from src.runtime.contracts import RetrievalMethod, SessionConfig, build_session_config
from src.ui.config_menu import _bounded_int_validator, _multiplier_ceiling, _top_k_ceiling
from src.utils.config import (
    _MAX_RETRIEVAL_CANDIDATES,
    _MAX_RETRIEVAL_MULTIPLIER,
    _MAX_RETRIEVAL_TOP_K,
    Settings,
)


@pytest.fixture
def session_config() -> SessionConfig:
    return build_session_config(Settings(_env_file=None))


@pytest.mark.parametrize("multiplier", [1, 3, 20, 100])
def test_top_k_ceiling_is_accepted_by_session_config(session_config, multiplier):
    """UI 给出的 top_k 上界，配上当前倍率必须能被 ``SessionConfig`` 接受。"""
    ceiling = _top_k_ceiling(multiplier)

    config = SessionConfig(
        **{
            **session_config.to_dict(),
            "top_k": ceiling,
            "retrieval_candidate_multiplier": multiplier,
        }
    )

    assert config["top_k"] == ceiling


@pytest.mark.parametrize("top_k", [1, 5, 100, 2000])
def test_multiplier_ceiling_is_accepted_by_session_config(session_config, top_k):
    """UI 给出的倍率上界，配上当前 top_k 必须能被 ``SessionConfig`` 接受。"""
    ceiling = _multiplier_ceiling(top_k)

    config = SessionConfig(
        **{
            **session_config.to_dict(),
            "top_k": top_k,
            "retrieval_candidate_multiplier": ceiling,
        }
    )

    assert config["retrieval_candidate_multiplier"] == ceiling


def test_ceilings_never_exceed_the_individual_bounds():
    """上界不得高于字段自身的上界。"""
    assert _top_k_ceiling(1) <= _MAX_RETRIEVAL_TOP_K
    assert _multiplier_ceiling(1) <= _MAX_RETRIEVAL_MULTIPLIER


def test_ceilings_shrink_as_the_other_field_grows():
    """另一字段越大，本字段的上界越小——否则乘积会越界。"""
    assert _top_k_ceiling(100) < _top_k_ceiling(1)
    assert _multiplier_ceiling(2000) < _multiplier_ceiling(1)


def test_ceiling_stays_usable_when_the_other_field_is_at_its_bound():
    """另一字段取到上界时，本字段仍须至少能取到 1。

    取 0 会让 UI 校验器无条件拒绝，用户被卡在一个改不动的字段上。
    """
    assert _top_k_ceiling(_MAX_RETRIEVAL_CANDIDATES) >= 1
    assert _multiplier_ceiling(_MAX_RETRIEVAL_CANDIDATES) >= 1


# ── 回归：输入框校验器不得把异常抛给 prompt_toolkit 的事件循环 ──
#
# questionary 的 ``build_validator`` 只把 ``ValidationError`` 转成「输入无效」
# 提示；校验器抛出的其它异常会一路冒到 prompt_toolkit 的事件循环之外。用户按下
# 回车后看到的是 ``Unhandled exception in event loop`` 加一段 traceback，输入框
# 不再响应——而触发它的只是一个上标数字。
#
# 根因是 ``text.isdigit()``：它对 128 个 Unicode 码位为真（``²`` ``①`` ``፩``
# ``₅`` …），而 ``int()`` 对它们一律抛 ``ValueError``。实测（真实 PromptSession
# + 管道输入）输入 ``²`` 再回车，异常从
# ``prompt_toolkit/shortcuts/prompt.py::_accept_input`` →
# ``buffer.validate_and_handle`` → ``questionary/prompts/common.py:571`` 抛出。
#
# 全码位穷举：``isdigit()`` 为真的 788 个码位里 128 个 ``int()`` 抛错；
# ``isdecimal()`` 为真的 660 个码位 ``int()`` 全部可解析。


@pytest.mark.parametrize(
    "text",
    ["²", "¹", "①", "፹", "₅", "⑴", "²²", "5²"],
)
def test_bounded_int_validator_does_not_raise_for_unparseable_digits(text):
    """校验器对任何输入都必须返回布尔值，绝不能抛异常。"""
    validate = _bounded_int_validator(_top_k_ceiling(3))

    assert validate(text) is False


@pytest.mark.parametrize("text", ["٣", "１２", "۷"])
def test_bounded_int_validator_accepts_parseable_unicode_digits(text):
    """收紧不能误伤：``int()`` 真能解析的 Unicode 十进制数字照常放行。"""
    validate = _bounded_int_validator(20)

    assert validate(text) is True


def test_bounded_int_validator_keeps_the_original_bounds():
    """正常输入的接受／拒绝边界不得改变。"""
    validate = _bounded_int_validator(5)

    assert validate("1") is True
    assert validate("5") is True
    assert validate("6") is False
    assert validate("0") is False
    assert validate("abc") is False
    assert validate("") is False


def test_bounded_int_validator_returns_bool_not_truthy():
    """questionary 只把 ``False`` 当作「无效」；返回非布尔真值会被当成通过。

    见 ``questionary/prompts/common.py`` 的 ``_InputValidator``：``if verdict is
    not True:``——返回 ``1`` 之类会被当成「不是 True」而报无效，返回 ``0`` 则
    恰好等价于 False。这里要求严格返回 ``bool``，避免依赖这种巧合。
    """
    validate = _bounded_int_validator(5)

    for text in ["3", "9", "²", "abc", ""]:
        assert type(validate(text)) is bool, f"{text!r} 返回了非布尔值"


def test_bounded_int_validator_survives_digit_strings_beyond_the_conversion_limit():
    """超长数字串必须被拒绝，而不是抛 ``ValueError``。

    ``isdecimal()`` 为真不代表 ``int()`` 能解析：``"9" * 4301`` 是合法十进制整数，
    但位数超过 ``sys.get_int_max_str_digits()``（默认 4300），``int()`` 照样抛
    ``ValueError``。这类异常同样会冒到 prompt_toolkit 事件循环之外——把上标数字
    换成「粘贴一长串数字」，收场完全一样。
    """
    import sys

    validate = _bounded_int_validator(_top_k_ceiling(3))

    assert validate("9" * (sys.get_int_max_str_digits() + 1)) is False
    assert validate("0" * 5000 + "1") is False
    # 恰好在上限内且超过 ceiling 的值同样只是「被拒」，不是崩溃
    assert validate("9" * 100) is False


# ── 回归：菜单与提示里的取值必须可渲染 ──
#
# ``edit_retrieval_params`` 直接把 ``chat_config`` 的字段插进 questionary 的
# 菜单项与提示串。``SessionConfig`` 是公开类、普通属性没有 ``__setattr__`` 守卫，
# 嵌入方（或携带超大整数的 pickle 快照）可以直写 ``cfg.top_k = 10**4301``，
# 随后 ``chat_config["top_k"]`` 就是这个值。裸插值会抛
# ``ValueError: Exceeds the limit``——用户看到的是整数转换错误，而且异常落在
# questionary 渲染阶段，菜单根本显示不出来。
#
# 可达性已核实：这些取值全部经 ``chat_config[...]`` 读取，与
# ``SessionConfig`` 的「属性直写可绕过 validator」是同一条路径
# （见 ``_require_finite_number`` 的 docstring）。
# 权重字段与 top_k／倍率同源：``SessionConfig`` 的属性没有 ``__setattr__``
# 守卫，直写后菜单串、提示串、``default=`` 与 logger 四处都会裸插值。
# 前三个在 questionary 渲染阶段抛 ``ValueError: Exceeds the limit``，菜单
# 根本显示不出来；logger 那处在记录日志时抛。
@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("vector_weight", "向量"),
        ("keyword_weight", "关键词"),
    ],
)
def test_weight_renderings_survive_a_directly_written_value(field, label):
    from src.utils.config import safe_repr

    huge = 10**4301

    # 与 config_menu 里的拼装方式一致；裸插值在这里就会抛。
    menu = f"向量: {safe_repr(huge)} / 关键词: {safe_repr(huge)}"
    prompt = f"输入新的向量权重 (0.0-1.0, 当前: {safe_repr(huge)}):"
    default = safe_repr(huge)

    for text in (menu, prompt, default):
        assert "Exceeds the limit" not in text
        assert "位十进制" in text
    assert label in menu


def test_weight_renderings_keep_small_values_verbatim():
    """收紧不能误伤：正常权重照旧原样显示，不加引号、不截断。"""
    from src.utils.config import safe_repr

    assert safe_repr(0.7) == "0.7"
    assert safe_repr(0.3) == "0.3"
    assert f"向量: {safe_repr(0.7)} / 关键词: {safe_repr(0.3)}" == "向量: 0.7 / 关键词: 0.3"


def test_menu_labels_render_values_beyond_the_conversion_limit():
    """菜单项与提示串里的取值都要经 ``safe_repr``，超限整数照常显示。"""
    from src.utils.config import safe_repr

    huge = 10**4301

    # 与 config_menu 里的拼装方式一致；裸插值在这里就会抛。
    labels = [
        f"2. 检索数量 Top K (当前: {safe_repr(huge)})",
        f"6. 候选过量招募倍率 (当前: {safe_repr(huge)})",
        f"输入新的Top K值 (1-{safe_repr(_top_k_ceiling(3))}, 当前: {safe_repr(huge)}):",
        f"输入候选过量招募倍率 (1-{safe_repr(_multiplier_ceiling(5))}, 当前: {safe_repr(huge)}):",
    ]

    for label in labels:
        assert "Exceeds the limit" not in label
        assert "位十进制" in label


def test_menu_label_renderer_is_not_the_naive_interpolation():
    """反向守卫：裸插值确实会抛——否则上面那条断言证明不了什么。"""
    huge = 10**4301

    with pytest.raises(ValueError, match="Exceeds the limit"):
        f"当前: {huge}"


def test_ceilings_stay_usable_across_every_legal_config_value():
    """在 ``SessionConfig`` 允许的取值范围内，两个上界都不得退化成 0。

    上界为 0 会让编辑器拒绝一切输入（区间 ``[1, 0]`` 为空），用户无从下手。
    实测：合法范围内 ``top_k`` 上界最小 20、倍率上界最小 1，因此不会退化。

    被直写的越界倍率（如 ``10**4301``）确实会让上界算成 0，但**有意不修**：
    把上界钳到 1 反而允许用户提交一个乘法积越界的组合，使 ``__setitem__``
    抛 ``ValueError`` 冲穿 chat 主循环——比「编辑器拒绝输入」更糟。此处只锁
    「合法范围不退化为 0」这一真实不变量。
    """
    from src.utils.config import (
        _MAX_RETRIEVAL_MULTIPLIER,
        _MAX_RETRIEVAL_TOP_K,
    )

    assert min(_top_k_ceiling(m) for m in range(1, _MAX_RETRIEVAL_MULTIPLIER + 1)) >= 1
    assert min(_multiplier_ceiling(k) for k in range(1, _MAX_RETRIEVAL_TOP_K + 1)) >= 1


# ── 回归：模型菜单必须能对不可用的配置**取值**保持可用 ──
#
# `edit_model_params` 原先直接写 `list(chat_config["llm_configurations"].keys())`。
# `SessionConfig._VALIDATORS` 明确不登记 `*_configurations`（「按原样写入，由
# 使用方负责其内部结构」），嵌入方可以把它直写成 `5`，于是抛裸
# `AttributeError: 'int' object has no attribute 'keys'`——不是 `ValueError`，
# 会以裸 traceback 终止会话。
#
# 关键取舍：菜单**只判容器、不判取值**。`display_chat_config` 在上游读值时会
# 直接拒绝形态坏掉的配置项，菜单是用户唯一还能把激活项切走的入口；如果这里
# 也按取值判，配置一旦写坏就彻底没有修复路径了。


class _Choice:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


def _fake_select(answers):
    """返回一个按顺序吐答案的 ``questionary.select`` 替身。

    真实调用点把**菜单项**作为位置参数传入，替身必须收下（否则
    ``questionary.Choice`` 的求值路径根本没被覆盖到）。
    """
    queue = list(answers)

    def select(_prompt, *_args, **_kwargs):
        return _Choice(queue.pop(0) if queue else None)

    return select


def test_edit_model_params_reports_non_mapping_llm_configurations(monkeypatch):
    from src.ui import config_menu

    monkeypatch.setattr(config_menu.questionary, "select", _fake_select(["llm"]))
    chat_config = {
        "llm_configurations": 5,
        "active_llm_configuration": "broken",
        "active_rerank_configuration": "",
    }

    with pytest.raises(ValueError, match="LLM配置"):
        config_menu.edit_model_params(chat_config)


def test_edit_model_params_reports_non_mapping_rerank_configurations(monkeypatch):
    from src.ui import config_menu

    monkeypatch.setattr(config_menu.questionary, "select", _fake_select(["rerank"]))
    chat_config = {
        "rerank_configurations": "x",
        "active_llm_configuration": "",
        "active_rerank_configuration": "broken",
    }

    with pytest.raises(ValueError, match="Rerank配置"):
        config_menu.edit_model_params(chat_config)


def test_edit_model_params_still_lists_configuration_keys(monkeypatch):
    """收紧不能误伤：正常映射照常列出键并完成切换。"""
    from src.ui import config_menu

    prompts: list[str] = []
    # 第一轮进 "llm" 分支、选到新的键；第二轮退出。主菜单必须真的退出，
    # 否则 `while True` 会一直转（替身返回同一个答案就会死循环）。
    answers = iter(["llm", "b", "back"])

    def select(prompt, choices=None, **_kwargs):
        prompts.append(prompt)
        if prompt == "选择新的LLM配置:":
            # 调用点把配置键列表原样当候选传入
            assert [c.value if hasattr(c, "value") else c for c in choices] == ["a", "b"]
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "llm_configurations": {"a": object(), "b": object()},
        "active_llm_configuration": "a",
        "active_rerank_configuration": "",
    }

    assert config_menu.edit_model_params(chat_config) is True
    assert chat_config["active_llm_configuration"] == "b"
    assert "选择新的LLM配置:" in prompts


def test_edit_model_params_lists_unusable_details_without_reading_them(monkeypatch):
    """取舍锁死：取值不可用时菜单仍要列出键，不能连修复入口一起断掉。"""
    from src.ui import config_menu

    def select(prompt, choices=None, **_kwargs):
        if prompt == "选择新的LLM配置:":
            # 坏条目必须仍然出现在候选里——这是它能被切走的唯一入口
            assert [c.value if hasattr(c, "value") else c for c in choices] == ["broken"]
            return _Choice("broken")
        return _Choice("back")

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "llm_configurations": {"broken": None},
        "active_llm_configuration": "broken",
        "active_rerank_configuration": "",
    }

    assert config_menu.edit_model_params(chat_config) is False
    assert chat_config["active_llm_configuration"] == "broken"


# ── 回归：模型菜单的候选项必须能交给 questionary ──
#
# `edit_model_params` 把 `_configuration_keys` 返回的**原始键**直接当候选传给
# `questionary.select`。`SessionConfig` 的 `_VALIDATORS` 明确不登记
# `*_configurations`（「按原样写入，由使用方负责其内部结构」），嵌入方或携带
# 异常取值的 pickle 快照可以放进任意可哈希键。questionary 对每个候选调
# `choice.get(...)`（`Choice` 对象才有该方法），非字符串键——`int`、`tuple`、
# 乃至 `None`——一律抛裸 `AttributeError: 'int' object has no attribute 'get'`，
# 菜单根本显示不出来。位数超限的整数则更早在渲染候选标题时抛
# `ValueError: Exceeds the limit`。
#
# 同一处还有第二条：`default=current_llm` 传的是**原始**激活键，而候选若被
# 渲染成字符串标题，questionary 会拿它去候选集合里找，找不到就抛
# `ValueError: Invalid 'default' value passed`——激活项被删掉、或它本身是位数
# 超限的整数时都会命中。
#
# 修法：候选一律包成 `questionary.Choice(title=_render_value(key), value=key)`，
# `default` 传原始键（`Choice.value` 就是原始键，能对上），并把不在候选集合里的
# `default` 收敛成 `None`（questionary 的「未作答」哨兵，合法）。


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(1, id="int"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(10**4301, id="huge-int"),
    ],
)
def test_edit_model_params_selects_over_non_string_configuration_keys(monkeypatch, key):
    """非字符串的配置键必须能进候选，而不是让 questionary 抛裸 ``AttributeError``。"""
    from src.ui import config_menu

    seen: list[list] = []
    answers = iter(["llm", None, "back"])

    def select(prompt, choices=None, **_kwargs):
        if prompt == "选择新的LLM配置:":
            # 真实 questionary 走 ``Choice.build``：``Choice`` 原样返回、``str``
            # 包成 ``Choice(c, c)``，**其余类型**才落到 ``c.get(...)`` 上抛裸
            # ``AttributeError``。替身按同一契约校验。
            seen.append(list(choices))
            for choice in choices:
                assert isinstance(choice, (str, questionary.Choice)), (
                    f"候选不是 questionary 能接受的形态: {choice!r}"
                )
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "llm_configurations": {key: None},
        "active_llm_configuration": key,
        "active_rerank_configuration": "",
    }

    assert config_menu.edit_model_params(chat_config) is False
    assert seen, "菜单必须真的构造过候选"


def test_edit_model_params_survives_stale_active_configuration(monkeypatch):
    """激活项不在候选集合里时，``default`` 不得让 questionary 抛 ``ValueError``。"""
    from src.ui import config_menu

    answers = iter(["llm", None, "back"])

    def select(prompt, choices=None, default=None, **_kwargs):
        if prompt == "选择新的LLM配置:":
            # 收敛后 default 要么是候选里的原始键，要么是 ``None``（未作答哨兵）
            values = [c.value for c in choices]
            assert default is None or default in values, (default, values)
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "llm_configurations": {"a": None, "b": None},
        # 激活项已被删掉：不在候选里
        "active_llm_configuration": "removed",
        "active_rerank_configuration": "",
    }

    assert config_menu.edit_model_params(chat_config) is False


# ── 回归：融合策略的 ``default=`` 必须落在候选集合里 ──
#
# `edit_retrieval_params` 的融合策略分支把 `chat_config["hybrid_fusion_strategy"]`
# **原始**取值当 `questionary.select(default=...)`。候选只有 `"rrf"`／`"weighted"`
# 两个字面量，而 `SessionConfig` 的普通属性没有 `__setattr__` 守卫，嵌入方或携带
# 异常取值的 pickle 快照可以直写该字段。实测三种收场都不在契约内：
#
#   `"RRF"`／`" rrf "`（大小写／空白未归一） -> questionary 拿它去候选集合里
#     找不到，抛 `ValueError: Invalid 'default' value passed`；
#   位数超限的整数 -> 抛 `ValueError: Exceeds the limit`；
#   非字符串（`1`／`None`／`["RRF"]`） -> 同样落到 `Invalid 'default'`。
#
# 菜单是用户唯一还能改配置的入口（`display_chat_config` 在上游读值时会直接拒绝），
# 而异常落在 `edit_retrieval_params` 内、由 `launch_config_editor` 经
# `asyncio.to_thread` 调用——调用点只接 `(KeyboardInterrupt, EOFError)`，会终止
# 整个会话。
#
# 修法：`default=` 走与 `_choice_default` 同一判据——取值经 `_coerce_fusion_strategy`
# 归一后若在候选集合里就用它，否则退回 `None`（questionary 的「未作答」哨兵）。


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("RRF", id="uppercase"),
        pytest.param(" rrf ", id="padded"),
        pytest.param(1, id="int"),
        pytest.param(10**4301, id="huge-int"),
        pytest.param(None, id="none"),
    ],
)
def test_fusion_default_survives_unnormalized_or_non_string_values(monkeypatch, value):
    """融合策略的 ``default=`` 必须是候选集合里的取值或 ``None``。"""
    from src.ui import config_menu

    seen: list = []
    answers = iter(["fusion", None, "back"])

    def select(prompt, choices=None, default=None, **_kwargs):
        if prompt == "选择新的混合融合策略:":
            values = [c.value for c in choices]
            seen.append(default)
            assert default is None or default in values, (default, values)
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "hybrid_fusion_strategy": value,
        "retrieval_candidate_multiplier": 3,
        "top_k": 5,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "rerank_enabled": False,
    }

    config_menu.edit_retrieval_params(chat_config)

    assert seen, "菜单必须真的构造过融合策略候选"


def test_fusion_default_keeps_normalized_value(monkeypatch):
    """收紧不能误伤：规范的 ``"rrf"`` 照旧作为默认项选中。"""
    from src.ui import config_menu

    seen: list = []
    answers = iter(["fusion", None, "back"])

    def select(prompt, choices=None, default=None, **_kwargs):
        if prompt == "选择新的混合融合策略:":
            seen.append(default)
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", select)
    chat_config = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 3,
        "top_k": 5,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "rerank_enabled": False,
    }

    config_menu.edit_retrieval_params(chat_config)

    assert seen == ["rrf"]


def test_weight_validator_rejects_non_numeric_and_out_of_range(monkeypatch):
    """权重校验器：非数值与越界返回 False，合法值返回 True。

    删除 ``except ValueError: return False`` 后，``float("abc")`` 的
    ``ValueError`` 冒泡到 questionary 的 ``buffer.validate_and_handle``
    ——输入一个非数字会让整个菜单崩溃，而校验器的契约是「返回 False
    表示输入无效」。
    """
    from src.ui import config_menu

    captured = {}

    def text(prompt, validate=None, default=None, **_kwargs):
        if "向量权重" in prompt:
            captured["validate"] = validate
        return _Choice(None)

    monkeypatch.setattr(config_menu.questionary, "text", text)
    monkeypatch.setattr(config_menu.questionary, "select", _fake_select(["weights", "back"]))
    chat_config = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 3,
        "top_k": 5,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "rerank_enabled": False,
    }
    config_menu.edit_retrieval_params(chat_config)

    validate = captured["validate"]
    assert validate("0.5") is True
    assert validate("0") is True
    assert validate("1") is True
    assert validate("1.5") is False
    assert validate("-0.1") is False
    assert validate("abc") is False
    assert validate("") is False


def test_edit_retrieval_params_writes_rerank_toggle(monkeypatch):
    """确认框返回布尔时写回 ``rerank_enabled``（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后判据反转：用户确认「启用」时不再写回配置，只有
    「未作答」（``None``）才写入——开关永远不生效。
    """
    from src.ui import config_menu

    answers = iter(["rerank", "back"])

    def _select(*_args, **_kwargs):
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", _select)
    monkeypatch.setattr(config_menu.questionary, "confirm", lambda *_a, **_k: _Choice(True))
    chat_config = {
        "retrieval_method": RetrievalMethod.HYBRID_SEARCH,
        "hybrid_fusion_strategy": "rrf",
        "retrieval_candidate_multiplier": 3,
        "top_k": 5,
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "rerank_enabled": False,
    }

    config_menu.edit_retrieval_params(chat_config)

    assert chat_config["rerank_enabled"] is True


def test_edit_model_params_writes_selected_rerank_configuration(monkeypatch):
    """选中新的 Rerank 配置时写回激活项（``is not None`` 不是 ``is None``）。

    ``is None`` 变异后判据反转：用户选中的配置不再写回，激活项保持原值——
    菜单显示切换成功而配置没变。
    """
    from src.ui import config_menu

    answers = iter(["rerank", "b", "back"])

    def _select(*_args, **_kwargs):
        return _Choice(next(answers))

    monkeypatch.setattr(config_menu.questionary, "select", _select)
    chat_config = {
        "llm_configurations": {"a": object()},
        "rerank_configurations": {"a": object(), "b": object()},
        "active_llm_configuration": "a",
        "active_rerank_configuration": "a",
    }

    config_menu.edit_model_params(chat_config)

    assert chat_config["active_rerank_configuration"] == "b"
