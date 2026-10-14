"""终端亮/暗主题检测与适配样式。

本项目 TUI 的三处提示色（主菜单 ``prompt``、聊天提示符、questionary 的
pointer）原先写死浅色系（xterm 116 ``skyblue1``、``fg:yellow``）——它们在
暗底上对比度约 12:1，在白底上只有 1.7-2.3:1（WCAG AA 正文要求 4.5:1，实测
低于该阈值），亮色终端下几乎不可读。rich 的语义色不受影响：``Console``
按终端类型自动选择亮/暗变体，问题集中在 prompt_toolkit / questionary 的
固定色。

检测策略（按优先级）：

1. 显式环境变量 ``PYRAG_LIGHT=1`` / ``PYRAG_DARK=1``——用户强制指定，
   优先于一切自动检测（嵌入式运行或自动检测误判时的逃生通道）。
2. ``COLORFGBG`` 环境变量（``fg;bg`` 或 ``fg;bg;mode`` 格式）：背景分量
   数值 ≥ 8 视为亮色背景。xterm/rxvt/Konsole 等主流模拟器会设置；
   iTerm2 / VS Code 终端默认不设。
3. 兜底暗色：无任何信号时按暗色处理——终端用户以暗色主题为主流，而
   亮色主题下错误选择的代价（浅色文字不可读）比暗色主题下更严重。

背景探测只读环境变量，不做 OSC 11 轮询：OSC 查询需要应答式读 tty，
在 ``/config`` 编辑器线程与主循环并发的场景下有竞争风险，不值得为一层
检测精度引入。
"""

from __future__ import annotations

import os

from prompt_toolkit.styles import Style

__all__ = ["detect_light_background", "prompt_style_for_theme", "questionary_style_for_theme"]

# 亮色背景下的替代色（均按白底 ≥4.5:1、黑底 ≥3.8:1 双向校验过）：
# 暗色沿用原浅色（黑底 ~12:1），亮色换深变体（白底 ~5:1）。
_PROMPT_COLOR_LIGHT = "#0e7490"  # 深青，白底 5.36:1
_PROMPT_COLOR_DARK = "#87d7af"  # skyblue1 / xterm 116，黑底 12.37:1
_POINTER_COLOR_LIGHT = "#a16207"  # 深黄，白底 4.92:1
_POINTER_COLOR_DARK = "ansiyellow"  # 原值，黑底可用


def detect_light_background(environ: dict[str, str] | None = None) -> bool:
    """返回当前终端是否为亮色背景。

    ``environ`` 供测试注入；默认读 ``os.environ``。
    """
    env = os.environ if environ is None else environ
    forced_light = env.get("PYRAG_LIGHT", "").strip().lower() in ("1", "true", "yes")
    forced_dark = env.get("PYRAG_DARK", "").strip().lower() in ("1", "true", "yes")
    if forced_light and not forced_dark:
        return True
    if forced_dark:
        return False

    colorfgbg = env.get("COLORFGBG", "")
    # 常见形态："15;0"（fg=15 白 fg / bg=0 黑 bg，暗色终端）、
    # "0;15"（暗 fg / 亮 bg，亮色终端）、"0;15;x"（Konsole 三段式）。
    # 取**最后一个**数值段作背景（两段式里它是 bg；三段式里 mode 不可能
    # 出现在 bg 之前）。
    parts = [segment for segment in colorfgbg.split(";") if segment.strip().isdigit()]
    if len(parts) >= 2:
        return int(parts[-1]) >= 8
    return False


def _prompt_color(light: bool) -> str:
    return _PROMPT_COLOR_LIGHT if light else _PROMPT_COLOR_DARK


def prompt_style_for_theme(light: bool | None = None) -> Style:
    """主菜单 / 聊天提示符的 prompt_toolkit 样式。

    ``light=None`` 时自动检测。原实现把 ``<skyblue>`` 当 HTML class 交给
    prompt_toolkit 的 xterm 色名表解析（实际渲染 xterm 116 浅青绿，白底
    1.7:1 不可读）；现在显式按主题给定 ``prompt`` 类的颜色。
    """
    if light is None:
        light = detect_light_background()
    return Style.from_dict({"prompt": f"bold {_prompt_color(light)}"})


def questionary_style_for_theme(light: bool | None = None) -> Style:
    """questionary 选择菜单的 pointer 样式（两处子菜单共用）。"""
    if light is None:
        light = detect_light_background()
    pointer = _POINTER_COLOR_LIGHT if light else _POINTER_COLOR_DARK
    return Style.from_dict({"pointer": f"bold fg:{pointer}"})
