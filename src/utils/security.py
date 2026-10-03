"""配置、请求和日志边界的安全校验工具。"""

import copy
import os
import re
from collections.abc import Collection, Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

FORBIDDEN_OPTION_CONTAINERS = frozenset(
    {
        "headers",
        "defaultheaders",
        "setdefaultheaders",
        "extraheaders",
        "header",
        "requestheaders",
        "query",
        "queryparams",
        "params",
        "searchparams",
        "urlparams",
        "requestquery",
        "defaultquery",
        "setdefaultquery",
        "extraquery",
        "httpoptions",
    }
)

SENSITIVE_OPTION_KEYS = frozenset(
    {
        "apikey",
        "xapikey",
        "accesskey",
        "secretkey",
        # 火山引擎 SDK 的客户端凭证键名（``ak``/``sk``）；本项目自己就用它们
        # 承载 VOLC_ACCESS_KEY / VOLC_SECRET_KEY，因此必须和 ``api_key`` 一样
        # 被识别为凭证。按规范化后的完整键名精确匹配，``task``/``risk``/``ask``
        # 这类普通字段不会命中。
        "ak",
        "sk",
        # Azure 存储的 ``account_key`` 也是凭证键名。
        "accountkey",
        "token",
        "password",
        "passwd",
        "authorization",
        "auth",
        "bearer",
        "accesstoken",
        "xaccesstoken",
        "authtoken",
        "xauthtoken",
        "authkey",
        "credential",
        "credentials",
        "cookie",
        "setcookie",
        "proxyauthorization",
        "secret",
        "privatekey",
        "sshkey",
        "sshprivatekey",
        "signingkey",
        "encryptionkey",
        "googkey",
        "googlekey",
        "clientsecret",
        "secretvalue",
        "secretcredential",
        "adminapikey",
        "webhooksecret",
        "workloadidentity",
        "httpclient",
        "httpxclient",
        "httpxasyncclient",
        "aiohttpclient",
        "clientargs",
        "asyncclientargs",
        "baseurl",
        "websocketbaseurl",
    }
)

_RESOURCE_EXTENSION_KEYS = frozenset({"extraheaders", "extraquery", "extrabody"})
# These names are ordinary business parameters for some official SDK resource
# methods (for example ``vector_stores.search(query=...)``). They are safe only
# at the resource-argument container itself; nested keys and values remain
# subject to credential scanning.
_SAFE_RESOURCE_PARAMETER_CONTAINERS = frozenset({"query", "params"})

# 用「非 ASCII 字母数字」而不是 ``\b`` 作边界：中文属 ``\w``，``\b`` 在
# ``密钥sk-...`` 两侧都不成立，而国产网关（SiliconFlow、Ark、DashScope）的
# 错误消息正是中文，密钥会整串落进日志。
_BEARER_TEXT_RE = re.compile(r"(?i)(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/=-]+")
# A punctuation separator is unambiguous.  Whitespace-only forms are useful
# for redacting messages such as ``token sk-...``, but treating every following
# word as a credential causes ordinary text (``token usage``) to be rejected by
# ``find_sensitive_option_paths``.  Require a credential-like value for that
# form: at least twelve token characters and one digit or token punctuation.
# 文本脱敏与 is_sensitive_option_key 必须使用同一套凭证键名，否则会出现
# 「配置边界判为敏感、日志里却明文输出」的漏洞（client_secret、private_key、
# credentials 都曾如此）。下面显式列出可在文本中安全识别的键名，所有文本正则
# 均由它派生；``[_-]?`` 同时覆盖 ``client_secret`` 与 ``clientsecret``。
#
# 不含 ak/sk：这两个键太短，需要独立的词边界正则（见 _SHORT_CREDENTIAL_KEY_RE），
# 否则 ``task = value`` 里的 sk 会被误脱敏。
# 也不含 base_url / client_args / http_client 等连接类键名：它们在配置边界被禁止
# 是因为会覆盖请求边界，本身不是凭证；纳入文本识别会让普通配置值被误判为凭证。
_TEXT_CREDENTIAL_KEYS = (
    "secret[_-]?access[_-]?key",
    "ssh[_-]?private[_-]?key",
    "proxy[_-]?authorization",
    "secret[_-]?credential",
    "workload[_-]?identity",
    "x[_-]?access[_-]?token",
    "x[_-]?auth[_-]?token",
    "x[_-]?api[_-]?key",
    "admin[_-]?api[_-]?key",
    "encryption[_-]?key",
    "signing[_-]?key",
    "private[_-]?key",
    "client[_-]?secret",
    "secret[_-]?value",
    "webhook[_-]?secret",
    "account[_-]?key",
    "access[_-]?token",
    "auth[_-]?token",
    "access[_-]?key",
    "secret[_-]?key",
    "auth[_-]?key",
    "set[_-]?cookie",
    "google[_-]?key",
    "goog[_-]?key",
    "ssh[_-]?key",
    "api[_-]?key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "cookie",
    "secret",
    "token",
    "auth",
    "bearer",
)
_TEXT_CREDENTIAL_KEY_PATTERN = "|".join(_TEXT_CREDENTIAL_KEYS)

# 键名前的界断言。这里必须比 ``\b`` 宽、比「任意位置」窄：
#
# - 不要 ``(?<![A-Za-z0-9])``：``is_sensitive_option_key`` 按 camelCase 边界
#   切词，``dbPassword``/``myApiKey``/``userToken`` 都判为敏感；要求键名前是
#   非字母数字，这些键名会在日志里明文输出。
# - 也不能完全不要断言：那样 ``oauth``、``topsecret``、``sessiontoken`` 会从
#   词中间匹配到 ``auth``/``secret``/``token``，而配置边界判它们是**非敏感**
#   （切词后是单个词，不命中）。因 find_sensitive_option_paths 用
#   「值是否被改写」判断值里有没有凭证，这会误拒合法配置。
#
# 因此界断言取「非字母数字**或** camelCase 边界」：
# ``dbPassword`` 的 ``Password`` 前是小写字母接大写，属驼峰边界，命中；
# ``topsecret`` 里的 ``secret`` 前是小写字母接小写，不是边界，不命中——
# 与 ``is_sensitive_option_key`` 的切词口径一致。
#
# 注意断言必须保持大小写敏感，因此 ``(?i)`` 不能写在最前面——否则
# ``(?<=[a-z0-9])(?=[A-Z])`` 的 ``[A-Z]`` 会匹配任意大小写字母，``oauth``
# 里的 ``auth``（前一个字符是 ``o``）会被当成驼峰边界。忽略大小写只作用于
# 键名本身（``(?i:...)`` 局部开启）。
_CREDENTIAL_KEY_HEAD = r"(?:(?<![A-Za-z0-9])|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))"
#
# 非凭证字面量：``auth: none``、``cookie: enabled``、``secret: false``、
# ``token: 0`` 是文档与提示词里的常见写法，判为凭证会让合法配置在边界被误拒
# （find_sensitive_option_paths 用「值是否被改写」判断值里有没有凭证）。
#
# 这里用**枚举字面量**而不是「值的长度/字符构成」来排除：形态启发式无法区分
# ``password: FakePwOnly`` 与 ``auth: none``——两者都是 12 字符以下的纯字母值，
# 加长度下限会把前者这类真凭证一并放过（净漏检），加字符类型要求同理。
# 误判源是有限的几个状态字面量，枚举它们更精确。
_NON_CREDENTIAL_VALUE = (
    r"(?:none|true|false|null|nil|enabled|disabled|on|off|auto|yes|no"
    # ``bearer`` 的值是 token 本身而不是状态，但 ``The bearer: standard``
    # 这类描述里 ``standard`` 是普通词，与 ``auth: none`` 同属误判源。
    r"|standard|required|optional|default|basic|empty|unset"
    r"|\d+)"
)

_ANY_VALUE_SHAPE = r"""(?:"[^"]*"|'[^']*'|[^\s,;}']+)"""


# 键名候选：以凭证词开头、后接任意标识符字符。用「宽泛捕获 + 回调判定」
# 而不是把每个完整键名写进正则：``passwordHash``/``tokenCount`` 这类
# 「凭证词 + 另一个词」的键，在配置边界按切词判为敏感，正则里却列不全。
# 候选只负责圈出可能的键名，是否敏感由 ``_is_credential_key`` 决定——与配置
# 边界同一个函数，一致性由构造保证。
#
# 前缀 ``_CREDENTIAL_KEY_HEAD`` 保持大小写敏感：``topsecret``/``oauth`` 里的
# ``secret``/``auth`` 前面是小写字母接小写，不是驼峰边界，不会从词中间切开。
_CREDENTIAL_KEY_CANDIDATE = r"(?i:(?:" + "|".join(_TEXT_CREDENTIAL_KEYS) + r")[A-Za-z0-9_\-]*)"
_CREDENTIAL_KEY_VALUE_RE = re.compile(
    _CREDENTIAL_KEY_HEAD
    # 尾随引号单独成组，替换时原样保留：``{"password": "x"}`` 里若把 ``"``
    # 一起吃掉，会输出 ``{"password: [REDACTED]}`` 破坏 JSON 结构。
    + r"("
    + _CREDENTIAL_KEY_CANDIDATE
    + r")([\"']?)"
    + r"""(\s*[:=]\s*|\s+(?=["']|"""
    + r"""(?=[A-Za-z0-9._~+/=-]{12,}(?:[\s,;}']|$))[A-Za-z0-9._~+/=-]*[0-9._~+/=-]))"""
    + r"("
    + _ANY_VALUE_SHAPE
    + r")"
)

# 裸关键词：值可能只是状态字面量（``auth: none``/``password: none``），
# 需要形态约束。从 ``_TEXT_CREDENTIAL_KEYS`` 派生而不是手写，否则词表扩充
# 时两处会失步——此前手写漏掉 ``password``/``credential`` 等五个，使
# ``password: none`` 从「不脱敏」变成「脱敏」。
_BARE_KEYWORD_KEYS = frozenset(key for key in _TEXT_CREDENTIAL_KEYS if re.fullmatch(r"[a-z]+", key))


def _redact_credential_pair(match: re.Match[str]) -> str:
    """键名判定为凭证时替换其值，否则原样返回。

    键名交给 ``_is_credential_key`` 判定，与配置边界用的是同一个函数。
    此前文本层用正则的字符断言去逼近切词逻辑，``passwordHash``/``tokenCount``
    这类键在配置边界判敏感、在文本里却明文输出（而它们的合法配置值又会被
    边界误拒），两个方向都不一致。
    """
    key, quote = match.group(1), match.group(2)
    if not _is_credential_key(key):
        return match.group(0)
    value = match.group(4).strip("\"'")
    # 裸关键词后接状态字面量时是普通文本（``auth: none``），不是凭证赋值。
    # 显式凭证键名（``password``/``api_key``…）的值一律脱敏。
    if normalize_option_key(key) in _BARE_KEYWORD_KEYS and re.fullmatch(
        _NON_CREDENTIAL_VALUE, value, re.IGNORECASE
    ):
        return match.group(0)
    return f"{key}{quote}{match.group(3)}[REDACTED]"


_URL_USERINFO_RE = re.compile(r"(?i)(https?://)([^\s/@:]+):([^\s/@]+)@")
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:" + _TEXT_CREDENTIAL_KEY_PATTERN + r"|ak|sk)=)[^&#\s]+"
)
# ``ak``/``sk`` 是火山引擎凭证键名，``account_key`` 是 Azure 存储凭证键名。
# 这些键很短，必须用词边界约束，否则 ``task = value`` 里的 ``sk`` 会被误脱敏。
#
# 值同样要排除状态字面量：``sk``/``ak`` 在配置里常作开关（``sk: off``），
# 与裸关键词（``auth: none``）是同一类误判源。此前只给裸关键词加了约束，
# 这两个短键没有，口径不一致。
_SHORT_CREDENTIAL_KEY_RE = re.compile(
    r"""(?i)(?<![A-Za-z0-9])"""
    r"""((?:ak|sk|account[_-]?key|secret[_-]?access[_-]?key))["']?"""
    r"""(\s*[:=]\s*)"""
    r"""(?!""" + _NON_CREDENTIAL_VALUE + r"""(?![A-Za-z0-9._~+/=-]))"""
    r"""(?:"[^"]*"|'[^']*'|[^\s,;}']+)"""
)
_OPENAI_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:sk|rk|sess)-[A-Za-z0-9_-]{8,}(?![A-Za-z0-9])", re.IGNORECASE
)
# 服务端返回的错误信息常带「首尾可见、中间掩码」的凭证形态，例如
# ``sk-abc***...***xyz``。``_OPENAI_KEY_RE`` 要求 ``sk-`` 后连续 8 个以上
# 字母数字，星号会中断匹配，于是整串原样落进日志。这里单独覆盖掩码形态。
_MASKED_CREDENTIAL_RE = re.compile(
    # 掩码段可含 ``*``、``.``、``…`` 等占位字符，尾部可能还有可见片段，
    # 因此尾部字符类要覆盖字母数字，否则 ``sk-abc***...***xyz`` 只吃掉前半段。
    #
    # 但尾部不能吞掉紧邻的键名：``sk-abc***token=<secret>`` 里的 ``token``
    # 若被吞进掩码匹配，后面的 ``token=<secret>`` 就失去锚点，值会从脱敏变成
    # 明文（净漏检）。
    #
    # 尾部用单字符类贪婪匹配（无前瞻、无回溯），裁剪交给替换函数
    # ``_redact_masked_credential``：正则里加断言会让引擎在游程的每个起点
    # 重复扫描剩余串，尾部无冒号时退化成 O(n²)——``"sk-abc***" + "deadbeef"*500``
    # 从 0.01ms 涨到 11ms，4000 字符时 1.4 秒。而 redact_sensitive_text 挂在
    # 每条日志的 formatter 上，一个回显掩码密钥前缀加长 token 的错误体就能
    # 拖住进程。
    r"(?i)(?<![A-Za-z0-9])(?:sk|rk|sess)-[A-Za-z0-9_.-]{2,}[*\u2026.]{2,}"
    r"([A-Za-z0-9_.*-]*)"
    # 尾部游程后紧跟 ``=`` 时它是键值对的键名：``sk-abc***xxxtoken=<secret>``。
    # 键名本身可能不是敏感键（``xxxtoken`` 按 camelCase 切词判非敏感），
    # 键值规则不会接手，值就会明文落进日志。这里把键值尾部一并纳入匹配，
    # 由替换函数决定如何脱敏——掩码前缀已表明它是凭证的可见尾部。
    # 尾随可选组不引入回溯：``=`` 不在前一个字符类里，引擎无需回退。
    r"(?:(=)" + _ANY_VALUE_SHAPE + r")?"
)


def _redact_masked_credential(match: re.Match[str]) -> str:
    """替换掩码凭证，并处理混进尾部的键名与值。

    尾部游程后跟 ``=`` 时（``sk-abc***xxxtoken=<secret>``），游程是键值对的
    键名、后面是它的值；键名可能不是敏感键，键值规则不会接手，因此这里把
    值一并脱敏，只保留键名作为可读上下文。

    没有分隔符时，游程可能是凭证的可见片段（``sk-abc***xyz``），不能整段
    留下；但敏感键名（``token``/``myApiKey``）要保留，让
    ``_KEY_VALUE_TEXT_RE`` 接续处理 ``sk-abc***token: <secret>`` 这类写法。
    """
    # 键名游程由正则的 ``([A-Za-z0-9_.*-]*)`` 捕获（``*``/``.``/``…`` 等掩码
    # 占位符不在其中），整段判断是否为敏感键：``token``/``myApiKey`` 命中并保留，
    # ``defghijkl``/``xyz`` 这类可见片段不命中，随掩码一起吃掉。
    # 不用「从某处截断取后缀」：``sk`` 本身就是火山凭证键名，从 ``s`` 起算会
    # 把 ``sk-abc***`` 切碎。
    run = match.group(1)
    # 键值尾部：游程是键名，后面的值同样属于凭证的可见尾部，一并脱敏。
    # 这里不能只保留键名交给 ``_KEY_VALUE_TEXT_RE``——键名可能不是敏感键
    # （``xxxtoken``/``oauth``），那条规则不会接手，值就明文落进日志。
    if match.group(2) is not None:
        return "[REDACTED]" + run + "=[REDACTED]"
    # 没有分隔符时游程可能是凭证的可见片段（``sk-abc***xyz``），不能整段留下；
    # 但敏感键名（``token``/``myApiKey``）要保留，让键值规则接续处理
    # ``sk-abc***token: <secret>`` 这类冒号分隔的写法。
    if run and is_sensitive_option_key(run):
        return "[REDACTED]" + run
    return "[REDACTED]"


_GOOGLE_API_KEY_RE = re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}(?![A-Za-z0-9])")
# 有固定前缀、可自证的凭证。与 ``_GOOGLE_API_KEY_RE`` 同一类：**不带键名**也能
# 认出是凭证，因此不依赖 ``key=value`` 上下文。
#
# 为什么需要：SDK 的异常消息常回显请求头或 URL（``safe_exception_text`` 的
# docstring 写明了这一点），此时凭证是**裸串**，没有键名可锚定。本项目的
# ``_TEXT_CREDENTIAL_KEYS`` 覆盖的是键名，``volc_access_key=...`` 这类会被
# 脱敏，但 ``invalid access key 'AKLT...'`` 这种回显形态不会——实测
# ``RedactingFormatter`` 对它原样输出。火山引擎（Ark）与阿里云都使用这种
# 固定前缀的长串凭证。
#
# 前缀足够长（4 字符）且有长度下限，因此误伤风险很低：对真实知识库 4489 行、
# 全部源码与文档扫描，零命中。
_AK_SK_PREFIXED_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:AKLT|LTAI)[0-9A-Za-z_-]{12,}(?![A-Za-z0-9])"
)


# PEM 私钥块。这是本项目**自己点名要防**的一类凭证——``SENSITIVE_OPTION_KEYS``
# 里同时列了 ``privatekey``／``sshkey``／``sshprivatekey``，Google Vertex AI 的
# ADC 服务账号 JSON 里就带 ``private_key``。但键名规则只挡得住「键名 + 值」的
# 形态：私钥**正文**作为一段独立文本出现时（异常消息回显文件内容、URL 参数、
# 单行 base64 正文）没有任何锚点，会原样落进终端与日志。实测
# ``validate_secret_free_options`` 把私钥当键名拒绝时，回显的消息里就带着完整
# 正文；裸 PEM 文本走 ``redact_sensitive_text`` 同样原样返回。
#
# 覆盖 BEGIN/END 成对与裸 ``BEGIN`` 两种：前者吃掉整块（含正文），后者兜住被
# 截断的片段（``safe_exception_text`` 会把消息截到 240 字符，END 可能已被切掉）。
_PEM_PRIVATE_KEY_RE = re.compile(
    r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
    r".*?"
    r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)


def _display_repr(value: Any) -> str:
    """把任意取值渲染成字符串，位数超限时退回量级描述。

    本模块**不能**导入 ``config.safe_repr``：``config`` 反过来从本模块导入
    ``redact_sensitive_text`` 与 ``validate_secret_free_options``，跨模块引用会
    形成循环导入。判据与 ``safe_repr`` 一致（位数超限时用 ``bit_length()`` 给
    出量级），改一处时两处都要看。
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except (ValueError, RuntimeError, RecursionError):
        if isinstance(value, int):
            digits = value.bit_length() * 30103 // 100000 + 1  # log10(2) ≈ 0.30103
            return f"<整数，约 {digits} 位十进制>"
        return f"<{type(value).__name__}>"


def _require_string_keys(mapping: Mapping[Any, Any], label: str) -> None:
    """校验映射的**顶层**键都是字符串。

    非字符串键本身不是凭证，但它会让下游的 ``str(key)`` 与 ``sorted(...)`` 崩在
    与字段无关的地方：``str(10**4301)`` 抛
    ``ValueError: Exceeds the limit``（类型对、内容错），``", ".join(sorted(...))``
    对普通整数键抛裸 ``TypeError: sequence item 0: expected str instance``。
    ``ModelDetail.options`` 声明为 ``dict[str, Any]``，pydantic 会强制这一形态，
    因此**配置来源**到不了这里；但 provider 是**公开类**——嵌入方可以直接
    ``JinaProvider("m", {1: "x"})``（本仓测试就这么构造）。各校验器的其余判据都
    按 ``ValueError`` 收敛，这里必须对齐。

    嵌套层级的同类判据在 ``find_sensitive_option_paths`` 的递归里，两处共用
    ``_display_repr`` 渲染越界整数。
    """
    for key in mapping:
        if not isinstance(key, str):
            raise ValueError(
                f"{label} 的键必须是字符串，但得到 {_display_repr(key)}。"
                " 请检查配置里的 options 表（TOML 表的键本就是字符串，"
                "非字符串键通常来自直接构造 Provider 的库调用方）。"
            )


def redact_sensitive_text(value: Any) -> str:
    """脱敏异常、日志和请求错误中的常见凭证表示。"""
    if not isinstance(value, str):
        # ``str(10**4301)`` 撞上 CPython 的整数位数上限抛 ``ValueError``。这个
        # 分支明确声明支持非字符串（调用方有传异常、URL、映射的），而 pickle 是
        # 二进制格式、不受十进制位数限制，畸形快照能带进这种值。
        #
        # 这里内联判据而**不**导入 ``config.safe_repr``：``config`` 反过来从本模块
        # 导入 ``redact_sensitive_text`` 与 ``validate_secret_free_options``，跨模块
        # 引用会形成循环导入。判据与 ``safe_repr`` 一致（位数超限时用
        # ``bit_length()`` 给出量级），改一处时两处都要看。
        try:
            value = str(value)
        except (ValueError, RuntimeError, RecursionError):
            if isinstance(value, int):
                digits = value.bit_length() * 30103 // 100000 + 1  # log10(2) ≈ 0.30103
                value = f"<整数，约 {digits} 位十进制>"
            else:
                value = f"<{type(value).__name__}>"
    # 掩码形态必须最先处理：``_BEARER_TEXT_RE`` 会先吃掉 ``Bearer sk-abc``
    # 的可见前缀，使后续的 ``sk-`` 锚点失效，尾部掩码片段就会残留。
    redacted = _MASKED_CREDENTIAL_RE.sub(_redact_masked_credential, value)
    redacted = _BEARER_TEXT_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _CREDENTIAL_KEY_VALUE_RE.sub(_redact_credential_pair, redacted)
    redacted = _SHORT_CREDENTIAL_KEY_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED]:[REDACTED]@", redacted)
    redacted = _URL_QUERY_SECRET_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED]", redacted)
    redacted = _GOOGLE_API_KEY_RE.sub("[REDACTED]", redacted)
    redacted = _AK_SK_PREFIXED_KEY_RE.sub("[REDACTED]", redacted)
    # PEM 块放最后：它的正文是多行 base64，前面几条规则可能已经改写了其中的
    # 片段（例如正文里恰好出现 ``sk-`` 形态），但 BEGIN/END 标记不受影响，
    # 整块仍能被吃掉。放前面则会把已被部分脱敏的块再匹配一次，结果相同但多
    # 一次全串扫描。
    return _PEM_PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", redacted)


def safe_exception_text(exc: BaseException) -> str:
    """把异常转成可安全展示的文本：脱敏凭证，并限制长度。

    终端与日志是凭证最容易泄漏的出口——SDK 的异常消息常回显请求头或
    URL。所有面向用户的异常输出都应经过这里，而不是直接 ``str(exc)``
    或把异常对象交给 ``console.print`` / ``logger.error``。

    ``str(exc)`` 自己会抛，必须兜住：``str(10**4301)`` 撞上 CPython 的整数
    位数上限（``sys.get_int_max_str_digits()``，默认 4300）抛 ``ValueError``。
    本函数是**最后一道展示边界**，它在 22 处 ``except`` 里被调用——它再抛，
    用户看到的就是解释器默认 handler 打出的裸 traceback，连脱敏都没有，
    比「消息不好看」严重得多。

    这类值确实可达：``pickle`` 是二进制格式，不受十进制位数限制，实测
    ``pickle.loads(pickle.dumps(10**4301))`` 正常往返，而本项目加载 pickle
    知识快照。JSON 与 TOML 两条路径送不进来（解析器自己就抛），所以此前没被
    发现；快照里的畸形值可以。

    兜底不用 ``repr``（同样会抛），用 ``safe_repr``：它对位数超限的整数退回
    ``bit_length()`` 给出量级，既保住了「是哪个字段」的信息，也不再抛。
    """
    try:
        message = str(exc).strip()
    except Exception:  # noqa: BLE001 - 展示边界不得再抛
        # 只有异常自身的 args 含位数超限的整数会走到这里。
        #
        # 逐个渲染 args 而不是把整个元组交给 ``safe_repr``：``repr(tuple)`` 会
        # 递归到元素上，对含超大整数的元组抛错，``safe_repr`` 兜住后只能退回
        # ``<tuple>``，把「是哪个值出的问题」也丢掉了。逐元素渲染既保住信息，
        # 又因为每个元素单独兜底而不会抛。
        #
        # 局部导入 ``safe_repr``：``config`` 在模块层导入本模块
        # （``from src.utils.security import redact_sensitive_text, ...``），
        # 模块层反向导入会成环。这里只在异常兜底分支执行，代价可忽略。
        from src.utils.config import safe_repr

        rendered = ", ".join(safe_repr(item) for item in exc.args) if exc.args else ""
        return f"{type(exc).__name__}: {rendered}" if rendered else type(exc).__name__
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {redact_sensitive_text(message)[:240]}"


def _copy_nested(value: Any) -> Any:
    """递归复制配置容器，尽量隔离可变的第三方 SDK 对象。"""
    if isinstance(value, Mapping):
        return {key: _copy_nested(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_nested(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_copy_nested(nested) for nested in value)
    if isinstance(value, set):
        return {_copy_nested(nested) for nested in value}
    if isinstance(value, frozenset):
        return frozenset(_copy_nested(nested) for nested in value)
    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        try:
            copied = model_copy(deep=True)
        except Exception:  # noqa: BLE001 - SDK model copy is best effort
            copied = value
        if copied is not value:
            return copied
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - preserve opaque runtime handles
        return value


def _model_dump_mapping(value: Any) -> Mapping[str, Any] | None:
    """读取第三方 SDK 模型的公开字典表示（若可用）。

    Pydantic 型 SDK 配置对象（例如 ``google.genai.types.HttpOptions``）不是
    ``Mapping``，只遍历普通字典会让其中的 headers/base URL 绕过配置边界。
    仅使用 SDK 明确提供的 ``model_dump``，无法读取时保持原有兼容行为。

    返回 ``None`` 的语义是「**这个值检视不了**」，不是「这个值是安全的」——
    调用方 ``find_sensitive_option_paths`` 会就此跳过它。这是**有意的
    fail-open**，与边界对其它不可检视值的处理一致：没有 ``model_dump`` 的对象、
    任意 ``object()``、``bytes`` 同样被跳过（实测四者都返回 ``[]``）。收紧成
    fail-closed 会让所有非 pydantic 的第三方配置对象（``httpx.Client``、
    ``openai.OpenAI`` 都属此类）连同其中的普通配置一起被拒，那是行为破坏而非
    加固。

    此处原先的注释写「third-party model_dump must fail closed」，与代码相反；
    那句话描述的是**意图**（不要因为读不到就当成已检查），而实现选择的是
    「读不到就不检查、也不拒绝」。注释已按实际行为改写，避免下一个人照着
    「fail closed」去推断边界强度。
    """
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        dumped = model_dump(exclude_none=True)
    except TypeError:
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 - 读不到就不检查，见 docstring 的 fail-open 说明
            return None
    except Exception:  # noqa: BLE001 - 读不到就不检查，见 docstring 的 fail-open 说明
        return None
    if dumped is value or not isinstance(dumped, Mapping):
        return None
    return dumped


def normalize_option_key(key: Any) -> str:
    """将配置键规范化为大小写无关、分隔符无关的形式。

    非字符串键走 ``_display_repr``：``str(10**4301)`` 撞上 CPython 的整数位数
    上限抛 ``ValueError: Exceeds the limit``。本函数在配置边界的**每一层**键上
    调用（``find_sensitive_option_paths`` 的递归、``_is_credential_key``），
    也在文本脱敏的正则回调里调用，因此对不可 ``str()`` 的键必须降级而不是崩。
    归一化结果对非字符串键没有语义（调用方另有 ``_require_string_keys`` 判据），
    只要不抛即可。
    """
    return "".join(char for char in _display_repr(key).lower() if char.isalnum())


def _is_credential_key(key: Any) -> bool:
    """判定键名是否为凭证键（不含连接类容器键）。

    连接类容器键（``query``/``headers``/``params``）在配置边界同样被禁止，
    但原因是会覆盖请求边界、本身不是凭证。文本脱敏只关心凭证，把它们一并
    纳入会让普通配置值（``query: foo``）被误判为凭证。
    """
    normalized = normalize_option_key(key)
    if normalized in SENSITIVE_OPTION_KEYS:
        return True

    # 只对明确的词边界进行组合键识别；例如 ``secretary`` 不会命中，
    # 而 ``client_secret``、``api-token`` 等显式凭证命名会命中。
    words = [
        part.lower()
        for part in re.split(
            r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            _display_repr(key),
        )
        if part
    ]
    if any(
        word
        in {
            "apikey",
            "accesstoken",
            "authtoken",
            "apitoken",
            "bearer",
            "password",
            "cookie",
            "token",
        }
        for word in words
    ):
        return True
    if any(word == "secret" for word in words):
        return True
    if any(
        left
        in {
            "api",
            "access",
            "auth",
            "client",
            "secret",
            "private",
            "ssh",
            "signing",
            "encryption",
            "goog",
            "google",
        }
        and right in {"key", "token", "secret", "credential"}
        for left, right in pairwise(words)
    ):
        return True
    compact = "".join(words)
    return compact in SENSITIVE_OPTION_KEYS


def is_sensitive_option_key(key: Any) -> bool:
    """识别凭证键与连接覆盖键，避免把普通单词的 ``secret`` 子串误判为凭证。"""
    if normalize_option_key(key) in FORBIDDEN_OPTION_CONTAINERS:
        return True
    return _is_credential_key(key)


def _url_credential_paths(value: Any, path: str) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = urlsplit(value)
    except ValueError:
        # urlsplit 对畸形 URL（例如 ``https://[::1``）抛 ValueError。这不是
        # 「值不是 URL」而是「无法解析」，不能就此放弃检查：该分支是叶子值的
        # 唯一入口，直接 return 会让任意凭证随一个畸形前缀整体绕过边界校验。
        # 退回手工解析，保持与正常 URL 路径一致的严格度：既要看值里有没有
        # 凭证，也要看 query 的键名（``?myApiKey=`` 这种空值键在正常路径下会被
        # ``is_sensitive_option_key`` 拦下，只做值脱敏会漏掉它）。
        fallback = [path or "value"] if redact_sensitive_text(value) != value else []
        marker = value.find("?")
        if marker != -1:
            for pair in value[marker + 1 :].split("&"):
                key = pair.partition("=")[0]
                if key and is_sensitive_option_key(key):
                    fallback.append(f"{path}?{key}")
        return fallback
    found: list[str] = []
    if redact_sensitive_text(value) != value:
        found.append(path or "value")
    if parsed.username is not None or parsed.password is not None:
        found.append(f"{path} userinfo")
    if parsed.query and (parsed.scheme or parsed.netloc or "://" in value):
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            if is_sensitive_option_key(key):
                found.append(f"{path}?{key}")
    return found


def find_sensitive_option_paths(
    options: Mapping[str, Any],
    *,
    allowed_containers: Collection[str] = (),
    label: str = "Provider options",
) -> list[str]:
    """递归查找凭证、连接覆盖和 URL 中的凭证路径。

    ``allowed_containers`` 只用于供应商明确支持的安全容器（例如 Google
    ``http_options.headers``）；容器内的键和值仍会继续递归检查。

    每一层的键都必须是字符串：本函数是**唯一的**递归点，四个兄弟校验器都经由
    它扫描，因此判据放在这里才能覆盖嵌套层级。此前判据只在
    ``validate_secret_free_options`` 的顶层循环里，嵌套的 ``{10**4301: "x"}``
    会在这里的 ``str(key)`` 上抛 ``ValueError: Exceeds the limit``（类型对、内容
    与字段无关），而普通的 ``{1: "x"}`` 则静默通过、随后在调用方的
    ``", ".join(sorted(...))`` 处抛裸 ``TypeError``。
    """
    found: list[str] = []
    allowed = {normalize_option_key(value) for value in allowed_containers}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            _require_string_keys(value, label if not path else f"{label} 的 {path}")
            for key, nested in value.items():
                current = f"{path}.{key}" if path else str(key)
                if is_sensitive_option_key(key) and normalize_option_key(key) not in allowed:
                    found.append(current)
                    continue
                visit(nested, current)
            return
        dumped = _model_dump_mapping(value)
        if dumped is not None:
            visit(dumped, path)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
            return
        found.extend(_url_credential_paths(value, path))

    visit(options)
    return sorted(set(found))


def validate_secret_free_options(
    options: Mapping[str, Any] | None,
    provider: str,
    *,
    allowed_containers: Collection[str] = (),
) -> dict[str, Any]:
    """校验模型级 options 不会携带凭证或请求边界覆盖。

    允许的容器只放宽容器本身，绝不会跳过其子项的敏感键检查。
    """
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError(f"{provider} options 必须是对象。")
    # 键形态判据（顶层与嵌套）统一由 ``find_sensitive_option_paths`` 承担：
    # 它是四个校验器共用的递归点，判据放在那里才覆盖每一层。
    found = find_sensitive_option_paths(
        options,
        allowed_containers=allowed_containers,
        label=f"{provider} options",
    )
    if found:
        raise ValueError(
            f"{provider} options 不允许包含凭证或请求头/query 配置: " + ", ".join(found)
        )
    # 不能只复制最外层：模型配置通常包含 ``extra_body``、嵌套路由或
    # SDK 配置对象。递归复制可以防止调用方在 Provider 初始化后修改
    # 原始字典，进而改变后续请求。
    return _copy_nested(dict(options))


def validate_secret_free_request_overrides(
    extra_headers: Mapping[str, Any] | None = None,
    extra_query: Mapping[str, Any] | None = None,
    provider: str = "Provider",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """校验请求级 HTTP 覆盖项，不允许携带凭证。

    请求级扩展仍可用于追踪头、租户标识等非敏感字段；认证由 Provider
    自己管理，避免调用方通过 ``extra_headers``/``extra_query`` 绕过密钥边界。
    """

    normalized: dict[str, Mapping[str, Any] | None] = {
        "extra_headers": extra_headers,
        "extra_query": extra_query,
    }
    copies: dict[str, dict[str, Any] | None] = {}
    for field_name, value in normalized.items():
        if value is None:
            copies[field_name] = None
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{provider} {field_name} 必须是对象。")
        found = find_sensitive_option_paths(value, label=f"{provider} {field_name}")
        if found:
            raise ValueError(
                f"{provider} 请求头/query 不允许包含凭证: {field_name}." + ", ".join(found)
            )
        copies[field_name] = _copy_nested(dict(value))
    return copies["extra_headers"], copies["extra_query"]


def validate_secret_free_payload(
    value: Mapping[str, Any] | None,
    provider: str,
    field_name: str,
) -> dict[str, Any]:
    """校验可扩展请求体中的敏感字段并返回递归隔离的副本。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{provider} {field_name} 必须是对象。")
    found = find_sensitive_option_paths(value, label=f"{provider} {field_name}")
    if found:
        raise ValueError(f"{provider} {field_name} 不允许包含凭证或连接字段: " + ", ".join(found))
    return _copy_nested(dict(value))


def validate_secret_free_resource_kwargs(
    kwargs: Mapping[str, Any] | None,
    provider: str,
) -> dict[str, Any]:
    """校验并复制资源 Facade 的 HTTP/body 扩展参数。

    SDK 资源方法普遍接受 ``extra_headers``、``extra_query`` 和
    ``extra_body``。统一在 Facade 边界校验，避免某个资源方法遗漏安全检查。
    """

    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"{provider} 资源参数必须是对象。")
    # 顶层键单独判：下面的扫描器只收到 ``generic_kwargs``（已按扩展键名过滤），
    # 非字符串键若恰好归一化成扩展键名就会绕过那一层的判据。
    _require_string_keys(kwargs, f"{provider} 资源参数")
    sanitized = _copy_nested(dict(kwargs))

    # SDK client/resource methods also expose client-level overrides such as
    # ``api_key``, ``default_headers`` and ``base_url``.  They are not part of
    # the three supported request extensions below and must not pass through a
    # Facade just because a particular SDK accepts them.
    generic_kwargs = {
        key: value
        for key, value in sanitized.items()
        if normalize_option_key(key) not in _RESOURCE_EXTENSION_KEYS
    }
    found = find_sensitive_option_paths(
        generic_kwargs,
        allowed_containers=_SAFE_RESOURCE_PARAMETER_CONTAINERS,
        label=f"{provider} 资源参数",
    )
    if found:
        raise ValueError(
            f"{provider} 资源参数不允许包含凭证或请求头/query 配置: " + ", ".join(found)
        )

    normalized_keys: dict[str, Any] = {}
    duplicate_extension_keys: list[str] = []
    for key in sanitized:
        normalized = normalize_option_key(key)
        if normalized not in _RESOURCE_EXTENSION_KEYS:
            continue
        if normalized in normalized_keys:
            duplicate_extension_keys.append(str(key))
            continue
        normalized_keys[normalized] = key
    if duplicate_extension_keys:
        raise ValueError(
            f"{provider} 资源参数包含重复的扩展字段: " + ", ".join(duplicate_extension_keys)
        )
    headers_key = normalized_keys.get("extraheaders")
    query_key = normalized_keys.get("extraquery")
    body_key = normalized_keys.get("extrabody")
    headers, query = validate_secret_free_request_overrides(
        sanitized.get(headers_key) if headers_key is not None else None,
        sanitized.get(query_key) if query_key is not None else None,
        provider,
    )
    if headers_key is not None:
        sanitized[headers_key] = headers
    if query_key is not None:
        sanitized[query_key] = query
    if body_key is not None and sanitized[body_key] is not None:
        sanitized[body_key] = validate_secret_free_payload(
            sanitized[body_key], provider, "extra_body"
        )
    return sanitized


def validate_secret_free_resource_args(
    args: tuple[Any, ...] | None,
    provider: str,
) -> tuple[Any, ...]:
    """校验并复制资源 Facade 的位置参数。

    大多数 SDK 资源方法把业务参数声明为关键字参数，但少数入口（例如
    OpenAI Responses WebSocket 的 ``connect``）允许用位置参数传入
    ``extra_query``、``extra_headers`` 或连接选项。统一扫描位置参数可以
    防止这些入口绕过关键字参数的凭证边界，同时保留普通字符串、ID 和
    文件对象的转发兼容性。
    """

    if args is None:
        return ()
    if not isinstance(args, tuple):
        raise ValueError(f"{provider} 资源位置参数必须是元组。")
    sanitized = _copy_nested(args)
    found: list[str] = []
    for index, value in enumerate(sanitized):
        # Wrapping the value in a mapping lets the existing recursive scanner
        # retain useful paths (for example ``args[0].Authorization``) and the
        # same safe business containers used by keyword resource parameters.
        found.extend(
            find_sensitive_option_paths(
                {f"args[{index}]": value},
                allowed_containers=_SAFE_RESOURCE_PARAMETER_CONTAINERS,
                label=f"{provider} 资源位置参数",
            )
        )
    if found:
        raise ValueError(
            f"{provider} 资源位置参数不允许包含凭证或请求头/query 配置: "
            + ", ".join(sorted(set(found)))
        )
    return sanitized


def resolve_within(
    candidate: str | os.PathLike[str],
    root: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    """把一个路径解析到 ``root`` 之内，越界即拒绝。

    用于 pickle 一类「解析即执行」的加载入口：``AGENTS.md`` 要求知识快照只从
    本项目生成的本地快照或明确受信的 legacy 文件加载，但路径本身并不能证明
    来源，所以调用方必须给出信任根，由这里判断归属。

    两个路径都先 ``resolve()`` 再比较，因此符号链接指向根外时会被拦下
    （``root/link -> /etc`` 这类逃逸）。``resolve()`` 用默认的
    ``strict=False``：目标可以尚不存在，此处只判断归属，存在性由调用方另行校验。
    """
    # ``Path.resolve()`` 对符号链接环抛 ``RuntimeError: Symlink loop from ...``，
    # 它**不是** ``ValueError``，会穿透本函数的契约与调用方的 ``except ValueError``
    # ——``ensure_trusted_source`` 只接 ``ValueError``，于是一条「拒绝加载」的消息
    # 变成 pathlib 的内部错误。``strict=False`` 不覆盖这种情况：环是**解析期**的
    # 失败，与目标是否存在无关。这里统一收敛成 ``ValueError``。
    #
    # ``Path(...)`` 对非路径取值抛的 ``TypeError``（``None``／``int``／``list``：
    # ``expected str, bytes or os.PathLike object, not ...``）同样要收敛：它不是
    # ``ValueError``，且消息里没有字段名，而本函数的两个调用方
    # （``ensure_trusted_source`` 与 ``FaissStore.load_snapshot``）都按
    # ``ValueError`` 分类。形态判据放在构造之前，让消息带上是哪一个参数错了。
    for role, value in (("路径", candidate), ("受信根", root)):
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(
                f"{label} 的{role}必须是字符串或路径，但得到 {type(value).__name__}。"
                " 请检查该路径来源（通常来自 RunConfig 的路径字段）。"
            )
    try:
        resolved_root = Path(root).resolve()
        resolved_candidate = Path(candidate).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"{label} 无法解析（{type(exc).__name__}: {exc}）。"
            " 路径可能包含符号链接环或过长的间接层，请检查配置。"
        ) from exc
    if resolved_candidate != resolved_root and not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(
            f"{label} 必须位于受信根目录内。"
            f" 路径={resolved_candidate}，受信根={resolved_root}。"
            " 请只加载本项目生成的快照；如需导入外部文件，请显式把它放入受信目录。"
        )
    return resolved_candidate


def ensure_trusted_source(
    candidate: str | os.PathLike[str],
    trusted_roots: Iterable[str | os.PathLike[str]] | None,
    *,
    label: str,
) -> Path:
    """要求 ``candidate`` 命中给定的受信根之一，否则拒绝加载。

    ``trusted_roots`` 为 ``None`` 或空集合时一律拒绝——调用方没有声明任何
    信任来源时，不能退化成「任意路径都可加载」，否则这个边界等于不存在。

    ``trusted_roots`` 是**可迭代**，而 ``str``／``bytes``／``bytearray`` 也是
    合法的可迭代：直接 ``for root in trusted_roots`` 会把一个字符串信任根拆成
    单字符根。实测 ``trusted_roots="/etc"`` 拆出 ``"/"``（解析为文件系统根），
    于是**任意绝对路径都被接受**——``/anywhere/evil.pkl`` 也返回成功，正好绕开
    「不导入不可信来源的 .pkl」这条边界。这与同模块对 ``Sequence`` 参数的既有
    判据一致（``normalize_messages``／``_require_documents`` 都显式排除
    ``str``／``bytes``／``bytearray``），必须同样拦下。

    每个根还要能构成 ``Path``：``Path(None)``／``Path(5)``／``Path([...])`` 抛
    裸 ``TypeError: expected str, bytes or os.PathLike object, not ...``——它
    不是 ``ValueError``，会穿透 ``load`` 调用方按 ``ValueError`` 建立的错误
    分类，消息里也没有字段名。收敛成带 label 的 ``ValueError``，与
    ``resolve_within`` 对候选路径的同类形态同判据。
    """
    if isinstance(trusted_roots, (str, bytes, bytearray)):
        raise ValueError(
            f"{label} 的受信来源必须是路径序列，但得到 {type(trusted_roots).__name__}。"
            " 字符串会被逐字符拆成信任根（其中 ``/`` 会放行任意绝对路径），"
            "请传入路径列表，例如 [snapshot_root]。"
        )
    # 非可迭代的形态也要在这里拦下：下面直接 ``for root in trusted_roots``，对
    # ``5``／``True``／``object()`` 抛裸 ``TypeError: 'int' object is not
    # iterable``——不是本函数承诺的带 label 的 ``ValueError``，而调用方
    # ``FaissStore.load`` 按 ``ValueError`` 分类。``str``／``bytes`` 已在上一条
    # 显式拦下（它们**是**可迭代，但会被逐字符拆开，是另一种失败），这里补的是
    # 「根本不是序列」这一类。可达路径：``load`` 是 ``VectorStoreBase`` 的契约
    # 方法，嵌入方直接调用时把 ``trusted_paths`` 写错形态即命中。
    if trusted_roots is not None and not isinstance(trusted_roots, Iterable):
        raise ValueError(
            f"{label} 的受信来源必须是路径序列，但得到 {type(trusted_roots).__name__}。"
            " 请传入路径列表，例如 [snapshot_root]。"
        )
    roots: list[Path] = []
    if trusted_roots is not None:
        for root in trusted_roots:
            try:
                roots.append(Path(root))
            except TypeError as exc:
                raise ValueError(
                    f"{label} 的受信来源必须是路径，但得到 {type(root).__name__}。"
                    " 请传入路径列表（通常来自 RunConfig 的 snapshot_root 或"
                    " legacy_pkl_path）。"
                ) from exc
    if not roots:
        raise ValueError(
            f"缺少 {label} 的受信来源声明，已拒绝加载。"
            " 请在调用处显式传入 trusted_paths/trusted_roots（通常来自 RunConfig 的"
            " snapshot_root 或 legacy_pkl_path）。"
        )
    for root in roots:
        try:
            return resolve_within(candidate, root, label=label)
        except ValueError:
            # 逐个根试探，全部落空后由下面的消息统一报告：单条 ``resolve_within``
            # 的消息只提一个根，而这里的结论是「一个都没命中」，把根列表完整
            # 列出来更有用。收集各条子消息再拼进去会让消息变长却不增加信息量。
            continue
    # ``candidate`` 的形态先在这里判一次，不能只依赖 ``resolve_within``：后者对
    # 非路径取值抛的 ``ValueError`` 会被上面的 ``except ValueError: continue``
    # 吞掉，于是全部根都「没命中」，最后落到下面那条消息里的 ``Path(candidate)``
    # ——那里对 ``None``／``int``／``list`` 抛的是裸 ``TypeError``。本函数的契约
    # 是带 label 的 ``ValueError``（调用方 ``FaissStore.load`` 按它分类）。
    if not isinstance(candidate, (str, os.PathLike)):
        raise ValueError(
            f"{label} 的路径必须是字符串或路径，但得到 {type(candidate).__name__}。"
            " 请检查该路径来源（通常来自 RunConfig 的 legacy_pkl_path）。"
        )
    raise ValueError(
        f"{label} 不在任何受信路径内。路径={Path(candidate)}。"
        f" 受信路径={[str(root) for root in roots]}。"
    )
