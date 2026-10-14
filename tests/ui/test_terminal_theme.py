"""终端主题检测与适配样式的单元测试。"""

import pytest

from src.ui.terminal_theme import (
    detect_light_background,
    prompt_style_for_theme,
    questionary_style_for_theme,
)


class TestDetectLightBackground:
    def test_explicit_light_flag_wins(self):
        assert detect_light_background({"PYRAG_LIGHT": "1", "COLORFGBG": "15;0"}) is True

    def test_explicit_dark_flag_wins(self):
        assert detect_light_background({"PYRAG_DARK": "1", "COLORFGBG": "0;15"}) is False

    def test_dark_when_background_component_low(self):
        assert detect_light_background({"COLORFGBG": "15;0"}) is False

    def test_light_when_background_component_high(self):
        assert detect_light_background({"COLORFGBG": "0;15"}) is True

    def test_konsole_three_segment_form(self):
        assert detect_light_background({"COLORFGBG": "0;15;0"}) is False

    def test_missing_signal_defaults_to_dark(self):
        assert detect_light_background({}) is False

    def test_malformed_signal_defaults_to_dark(self):
        assert detect_light_background({"COLORFGBG": "not-a-number"}) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
    def test_flag_value_variants(self, value):
        assert detect_light_background({"PYRAG_LIGHT": value}) is True


class TestThemeStyles:
    """两套样式的颜色必须按主题切换（红/绿语义：亮色给深变体）。"""

    def test_prompt_style_differs_between_themes(self):
        dark = str(prompt_style_for_theme(light=False).style_rules)
        light = str(prompt_style_for_theme(light=True).style_rules)
        assert dark != light
        assert "#87d7af" in dark  # 暗色沿用原浅色
        assert "#0e7490" in light  # 亮色深变体

    def test_questionary_pointer_differs_between_themes(self):
        dark = str(questionary_style_for_theme(light=False).style_rules)
        light = str(questionary_style_for_theme(light=True).style_rules)
        assert dark != light
        assert "ansiyellow" in dark
        assert "#a16207" in light

    def test_styles_parse_with_prompt_toolkit(self):
        # from_dict 已在构造时验证语法；这里再断言产物可用（不抛）。
        assert prompt_style_for_theme(light=False) is not None
        assert questionary_style_for_theme(light=True) is not None
