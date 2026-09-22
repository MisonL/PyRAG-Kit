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
