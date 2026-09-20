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
