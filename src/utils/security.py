"""配置、请求和日志边界的安全校验工具。"""

import copy
import re
from collections.abc import Collection, Mapping
from itertools import pairwise
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
_CREDENTIAL_KEY_HEAD = r"(?:(?<![A-Za-z0-9])|(?<=[a-z0-9])(?=[A-Z]))"
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

_BARE_CREDENTIAL_KEYWORD_PATTERN = "|".join(
    key for key in _TEXT_CREDENTIAL_KEYS if re.fullmatch(r"[a-z]+", key)
)
_QUALIFIED_CREDENTIAL_KEY_PATTERN = "|".join(
    key for key in _TEXT_CREDENTIAL_KEYS if not re.fullmatch(r"[a-z]+", key)
)


def _key_value_pattern(keys: str, value_prefix: str = "") -> str:
    """拼出 ``键名 = 值`` 的文本脱敏模式。

    ``keys`` 里每个分支都是完整键名（可含 ``[_-]?``），配合
    ``_CREDENTIAL_KEY_HEAD`` 的「非字母数字或 camelCase 边界」断言，既让
    ``dbPassword``/``myApiKey`` 命中，又不把 ``topsecret``/``sessiontoken``
    从词中间切开——与 ``is_sensitive_option_key`` 的切词口径一致。

    ``value_prefix`` 插在取值之前，用于排除非凭证字面量。
    """
    return (
        _CREDENTIAL_KEY_HEAD + r"((?i:" + keys + r"))[\"']?"
        r"""(\s*[:=]\s*|\s+(?=["']|"""
        r"""(?=[A-Za-z0-9._~+/=-]{12,}(?:[\s,;}']|$))[A-Za-z0-9._~+/=-]*[0-9._~+/=-]))"""
        + value_prefix
        + _ANY_VALUE_SHAPE
    )


# 显式凭证键名（``password``、``api_key``、``client_secret``…）的值就是凭证，
# 任何取值都脱敏；裸关键词（``auth``/``cookie``/``secret``/``token``/``bearer``）
# 额外排除非凭证字面量。
_KEY_VALUE_TEXT_RE = re.compile(_key_value_pattern(_QUALIFIED_CREDENTIAL_KEY_PATTERN))
_BARE_KEYWORD_TEXT_RE = re.compile(
    _key_value_pattern(
        _BARE_CREDENTIAL_KEYWORD_PATTERN,
        r"(?!(?:" + _NON_CREDENTIAL_VALUE + r")(?![A-Za-z0-9._~+/=-]))",
    )
)
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)([^\s/@:]+):([^\s/@]+)@")
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:" + _TEXT_CREDENTIAL_KEY_PATTERN + r"|ak|sk)=)[^&#\s]+"
)
# ``ak``/``sk`` 是火山引擎凭证键名，``account_key`` 是 Azure 存储凭证键名。
# 这些键很短，必须用词边界约束，否则 ``task = value`` 里的 ``sk`` 会被误脱敏。
_SHORT_CREDENTIAL_KEY_RE = re.compile(
    r"""(?i)(?<![A-Za-z0-9])"""
    r"""(ak|sk|account[_-]?key|secret[_-]?access[_-]?key)["']?"""
    r"""(\s*[:=]\s*)"""
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
    r"(?i)(?<![A-Za-z0-9])(?:sk|rk|sess)-[A-Za-z0-9_.-]{2,}[*\u2026.]{2,}[A-Za-z0-9_.*-]*"
)


def _redact_masked_credential(match: re.Match[str]) -> str:
    """替换掩码凭证，但把混进尾部的敏感键名留给键值规则处理。

    ``sk-abc***token=<secret>`` 里的 ``token`` 若被掩码整体吃掉，后面的
    ``=<secret>`` 就失去锚点，值会从脱敏变成明文（净漏检）。这里从尾部
    游程中找出最长的敏感键名后缀并保留，让 ``_KEY_VALUE_TEXT_RE`` 接续
    处理。

    只保留**完整键名**（能通过 ``is_sensitive_option_key``）的后缀：掩码
    尾部本身可能就是可见片段（``sk-abc***xyz``），不能整段留下。
    """
    text = match.group(0)
    # 键名是尾部那段连续的键名字符（``*``/``.``/``…`` 等掩码占位符不在其中），
    # 整段判断是否为敏感键：``token``/``myApiKey`` 命中并保留，
    # ``defghijkl``/``xyz`` 这类可见片段不命中，随掩码一起吃掉。
    # 不用「从某处截断取后缀」：``sk`` 本身就是火山凭证键名，从 ``s`` 起算会
    # 把 ``sk-abc***`` 切碎。
    trailing = re.search(r"[A-Za-z0-9_-]+$", text)
    if trailing is not None and is_sensitive_option_key(trailing.group(0)):
        return "[REDACTED]" + trailing.group(0)
    return "[REDACTED]"


_GOOGLE_API_KEY_RE = re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}(?![A-Za-z0-9])")


def redact_sensitive_text(value: Any) -> str:
    """脱敏异常、日志和请求错误中的常见凭证表示。"""
    if not isinstance(value, str):
        value = str(value)
    # 掩码形态必须最先处理：``_BEARER_TEXT_RE`` 会先吃掉 ``Bearer sk-abc``
    # 的可见前缀，使后续的 ``sk-`` 锚点失效，尾部掩码片段就会残留。
    redacted = _MASKED_CREDENTIAL_RE.sub(_redact_masked_credential, value)
    redacted = _BEARER_TEXT_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _KEY_VALUE_TEXT_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _BARE_KEYWORD_TEXT_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _SHORT_CREDENTIAL_KEY_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED]:[REDACTED]@", redacted)
    redacted = _URL_QUERY_SECRET_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED]", redacted)
    return _GOOGLE_API_KEY_RE.sub("[REDACTED]", redacted)


def safe_exception_text(exc: BaseException) -> str:
    """把异常转成可安全展示的文本：脱敏凭证，并限制长度。

    终端与日志是凭证最容易泄漏的出口——SDK 的异常消息常回显请求头或
    URL。所有面向用户的异常输出都应经过这里，而不是直接 ``str(exc)``
    或把异常对象交给 ``console.print`` / ``logger.error``。
    """
    message = str(exc).strip()
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
    """
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        dumped = model_dump(exclude_none=True)
    except TypeError:
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 - third-party model_dump must fail closed
            return None
    except Exception:  # noqa: BLE001 - third-party model_dump must fail closed
        return None
    if dumped is value or not isinstance(dumped, Mapping):
        return None
    return dumped


def normalize_option_key(key: Any) -> str:
    """将配置键规范化为大小写无关、分隔符无关的形式。"""
    return "".join(char for char in str(key).lower() if char.isalnum())


def is_sensitive_option_key(key: Any) -> bool:
    """识别凭证键，避免把普通单词的 ``secret`` 子串误判为凭证。"""
    normalized = normalize_option_key(key)
    if normalized in FORBIDDEN_OPTION_CONTAINERS or normalized in SENSITIVE_OPTION_KEYS:
        return True

    # 只对明确的词边界进行组合键识别；例如 ``secretary`` 不会命中，
    # 而 ``client_secret``、``api-token`` 等显式凭证命名会命中。
    words = [
        part.lower()
        for part in re.split(
            r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            str(key),
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
    return compact in SENSITIVE_OPTION_KEYS or compact in FORBIDDEN_OPTION_CONTAINERS


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
) -> list[str]:
    """递归查找凭证、连接覆盖和 URL 中的凭证路径。

    ``allowed_containers`` 只用于供应商明确支持的安全容器（例如 Google
    ``http_options.headers``）；容器内的键和值仍会继续递归检查。
    """
    found: list[str] = []
    allowed = {normalize_option_key(value) for value in allowed_containers}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
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
    found = find_sensitive_option_paths(options, allowed_containers=allowed_containers)
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
        found = find_sensitive_option_paths(value)
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
    found = find_sensitive_option_paths(value)
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
            )
        )
    if found:
        raise ValueError(
            f"{provider} 资源位置参数不允许包含凭证或请求头/query 配置: "
            + ", ".join(sorted(set(found)))
        )
    return sanitized
