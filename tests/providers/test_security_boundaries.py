"""凭证边界与脱敏的回归测试。"""

import pytest

from src.utils.security import (
    is_sensitive_option_key,
    redact_sensitive_text,
    validate_secret_free_options,
)


@pytest.mark.parametrize("key", ["ak", "sk", "AK", "SK", "account_key"])
def test_security_recognizes_short_credential_keys(key):
    """火山 ``ak``/``sk`` 与 Azure ``account_key`` 必须按凭证键处理。"""
    assert is_sensitive_option_key(key) is True


@pytest.mark.parametrize("key", ["task", "risk", "ask", "mask", "disk", "sketch", "ak_region"])
def test_security_does_not_flag_ordinary_keys_containing_ak_or_sk(key):
    """短键必须精确匹配，普通业务字段不能被误判。"""
    assert is_sensitive_option_key(key) is False


def test_security_rejects_volcengine_ak_sk_request_overrides():
    """把火山凭证塞进 extra_headers 必须被拒绝。"""
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"ak": "AKLTxxx", "sk": "SKxxx"}, "Volcengine")


@pytest.mark.parametrize(
    "value",
    [
        "ak=AKLTabcdefghijklmnop",
        "sk=SKabcdefghijklmnop",
        "account_key=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6I",
        '{"ak": "AKLT1234567890", "sk": "SK1234567890"}',
    ],
)
def test_security_redacts_short_credential_key_assignments(value):
    redacted = redact_sensitive_text(value)
    assert "[REDACTED]" in redacted
    assert "AKLTabcdefghijklmnop" not in redacted
    assert "SKabcdefghijklmnop" not in redacted


@pytest.mark.parametrize(
    "value",
    ["task = value", "risk=high", "ask me later", "disk=full", "mask=none"],
)
def test_security_does_not_redact_ordinary_text_with_ak_or_sk(value):
    """普通文本里的 ``sk``/``ak`` 子串不能被当成凭证脱敏。"""
    assert redact_sensitive_text(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "Incorrect API key provided: sk-abc***...***xyz.",
        "Invalid token: rk-abc*******************def",
        "key sess-abcd****wxyz rejected",
    ],
)
def test_security_redacts_masked_credential_shapes(value):
    """服务端错误里「首尾可见、中间掩码」的凭证形态也要脱敏。

    ``_OPENAI_KEY_RE`` 要求 ``sk-`` 后连续 8 个以上字母数字，星号会中断匹配，
    这类字符串此前会原样落进日志。
    """
    redacted = redact_sensitive_text(value)
    assert "[REDACTED]" in redacted
    assert "sk-abc" not in redacted
    assert "rk-abc" not in redacted
    assert "sess-abcd" not in redacted


# ── 回归：脱敏与配置边界必须使用同一套凭证键名 ──

# 这些键在 is_sensitive_option_key 中返回 True，却曾因文本正则词表更小而在
# 日志里明文输出（client_secret 是 Google OAuth 的常见形态）。
_TEXT_REDACTION_KEYS = [
    "client_secret",
    "clientsecret",
    "private_key",
    "privatekey",
    "credentials",
    "ssh_private_key",
    "secret_access_key",
    "encryption_key",
    "webhook_secret",
]


@pytest.mark.parametrize("key", _TEXT_REDACTION_KEYS)
def test_security_redacts_every_sensitive_key_name_in_text(key):
    """配置边界判为敏感的键名，在文本里也必须脱敏。"""
    secret = "SUPERSECRETVALUE12345"
    assert is_sensitive_option_key(key) is True
    assert secret not in redact_sensitive_text(f"{key}={secret}")


@pytest.mark.parametrize("key", _TEXT_REDACTION_KEYS)
def test_security_redacts_sensitive_key_names_in_url_query(key):
    """URL query 形态同样要覆盖，不能只处理 key=value。"""
    secret = "SUPERSECRETVALUE12345"
    assert secret not in redact_sensitive_text(f"https://example.invalid/x?{key}={secret}")


@pytest.mark.parametrize(
    "text",
    ["token usage is high", "task = value", "the secret sauce", "auth flow"],
)
def test_security_keeps_ordinary_text_intact(text):
    """扩大键名词表后，普通文本不能被误脱敏。"""
    assert redact_sensitive_text(text) == text


# ── 回归：掩码形态在 Bearer 前缀下也要完整脱敏 ──


@pytest.mark.parametrize(
    "value",
    [
        "Bearer sk-abc***...***xyz rejected",
        "Authorization: Bearer rk-abcd****wxyz",
        "Bearer sess-abcdef****tail",
        "sk-abc***...***xyz",
        "rk-abcd****wxyz",
    ],
)
def test_security_redacts_masked_credentials_after_bearer_prefix(value):
    """``_BEARER_TEXT_RE`` 会先吃掉 ``Bearer sk-abc`` 前缀，使掩码正则失配，
    导致尾部掩码片段残留。掩码规则必须优先于 Bearer 规则执行。"""
    redacted = redact_sensitive_text(value)
    assert "***" not in redacted
    assert "****" not in redacted


# ── 回归：urlsplit 无法解析的值不能让凭证整体绕过边界 ──


def test_security_rejects_credential_hidden_behind_unparseable_url():
    """``urlsplit`` 对畸形 URL 抛 ValueError。若该分支直接 return，任何凭证
    都能随一个畸形前缀整体绕过边界校验（这是叶子值的唯一检查入口）。"""
    unparseable = "https://[::1 token sk-FAKE0000FAKE0000FAKE0000FAKE0000"
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"note": unparseable}, "Provider")


def test_security_parseable_and_unparseable_urls_are_equally_strict():
    """同样藏凭证的两个值，只因 URL 可解析与否而一个被拒一个放行，即为漏洞。"""
    secret = "token sk-FAKE0000FAKE0000FAKE0000FAKE0000"
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"note": f"https://h/x {secret}"}, "Provider")
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"note": f"https://[::1 {secret}"}, "Provider")


# ── 回归：面向用户的异常输出必须脱敏 ──


def test_safe_exception_text_redacts_credentials():
    """SDK 异常常回显请求头或 URL，终端与日志是凭证最容易泄漏的出口。"""
    from src.utils.security import safe_exception_text

    secret = "sk-FAKE0000FAKE0000FAKE0000FAKE0000"
    rendered = safe_exception_text(RuntimeError(f"auth failed for {secret}"))

    assert secret not in rendered
    assert "RuntimeError" in rendered


def test_safe_exception_text_handles_empty_and_long_messages():
    from src.utils.security import safe_exception_text

    assert safe_exception_text(ValueError()) == "ValueError"
    assert len(safe_exception_text(RuntimeError("x" * 500))) <= 260


def test_user_facing_error_paths_do_not_print_raw_exceptions():
    """``retrieval_test`` 与顶层入口曾把裸异常交给 console/logger，凭证会
    原样落到终端与日志；``chat`` 路径已做脱敏，三者必须一致。"""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    for relative in ("src/retrieval_test/core.py", "main.py", "src/chat/core.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "safe_exception_text" in source, relative
        assert 'console.print(f"[red]召回测试出错: {exc}' not in source, relative
        assert '{e}")' not in source, relative


# ── 回归：文本键名口径必须与 is_sensitive_option_key 一致 ──


# 键名识别按 camelCase 边界切词，因此这些驼峰键在配置边界判为敏感。
# 文本正则若要求键名前是非字母数字字符，它们会在日志里明文输出——即
# 「配置边界拦住、文本边界放过」的不一致，正是本文件要消灭的形态。
_CAMEL_CASE_REDACTION_KEYS = [
    "dbPassword",
    "myApiKey",
    "userToken",
    "myAccessKey",
    "clientSecret",
]


@pytest.mark.parametrize("key", _CAMEL_CASE_REDACTION_KEYS)
def test_security_redacts_camel_case_sensitive_key_names(key):
    """驼峰键名在文本与配置两个边界上必须同口径。"""
    secret = "SUPERSECRETVALUE12345"
    assert is_sensitive_option_key(key) is True
    assert secret not in redact_sensitive_text(f"{key}={secret}")


def test_security_rejects_camel_case_credential_hidden_in_ordinary_option():
    """驼峰键藏进普通选项值时也必须被拦下，不能因前缀是字母而漏检。"""
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"note": "myApiKey=SUPERSECRETVALUE12345"}, "Provider")


# ── 回归：裸关键词加标点不能把普通文本判成凭证 ──


@pytest.mark.parametrize(
    "text",
    [
        "Set auth: none to disable",
        "cookie: enabled",
        "secret: false",
        "token: 0",
        "The bearer: standard",
    ],
)
def test_security_keeps_benign_keyword_assignments_intact(text):
    """``auth``/``cookie``/``secret`` 等裸关键词后接普通词，是文档与提示词里的
    常见写法。判为凭证会让合法配置在边界被误拒。"""
    assert redact_sensitive_text(text) == text


def test_security_accepts_benign_keyword_text_in_options():
    """误判不止影响日志：``find_sensitive_option_paths`` 用值是否被改写来判断
    值里有没有凭证，误脱敏会让合法 options 直接被拒。"""
    assert validate_secret_free_options(
        {"instructions": "Set auth: none to disable"}, "Provider"
    ) == {"instructions": "Set auth: none to disable"}


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer sk-FAKE0000SHORT0000FAKE0000",
        '{"api_key": "secret-value-xyz"}',
        "token: sk-FAKE0000FAKE0000FAKE0000FAKE0000",
    ],
)
def test_security_still_redacts_real_credentials_after_value_shape_check(value):
    """加了值的形态约束后，真凭证不能跟着一起放过。"""
    assert "[REDACTED]" in redact_sensitive_text(value)


# ── 回归：掩码尾部不能吞掉紧邻的键名 ──


def test_security_masked_credential_does_not_swallow_adjacent_key_name():
    """``sk-abc***token=<secret>`` 里的 ``token`` 若被掩码规则吞进匹配，
    后面的键值对就失去锚点，值会从脱敏变成明文——净漏检。"""
    secret = "SECRETVALUE1234567890"
    redacted = redact_sensitive_text(f"sk-abc***token={secret}")
    assert secret not in redacted
    # 键名保留是预期行为（脱敏的是值），关键是掩码规则没有把 ``token`` 吞掉
    # 而让后续的 ``=<secret>`` 失去锚点。
    assert redacted.count("[REDACTED]") == 2


# ── 回归：urlsplit 回退分支也要检查 query 键名 ──


def test_security_unparseable_url_still_checks_query_key_names():
    """回退分支若只做值脱敏，``?myApiKey=`` 这类空值敏感键会在畸形 URL 下漏检，
    而同样的值在可解析 URL 下会被拦下。"""
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"endpoint": "https://[::1?myApiKey="}, "Provider")


def test_security_parseable_and_unparseable_urls_check_query_keys_equally():
    with pytest.raises(ValueError, match="凭证"):
        validate_secret_free_options({"endpoint": "https://host/v1?myApiKey="}, "Provider")


def test_security_accepts_benign_unparseable_url():
    """严格度提升不能把无害的畸形 URL 也一并拒掉。"""
    assert validate_secret_free_options({"endpoint": "https://[::1/v1/models"}, "Provider") == {
        "endpoint": "https://[::1/v1/models"
    }
