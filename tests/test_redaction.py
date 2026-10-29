import hashlib
import time

import pytest

from src.redaction import (
    REDACTED,
    context_summary,
    is_credential_key,
    redact,
    redact_text,
)

JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun\n"
    "VTLw7onLRnrq0/IzW7yWR7QkrmBL7jTKEn5u+qKhbwKfBstIs+bMY2Zkp18gnTxK\n"
    "-----END RSA PRIVATE KEY-----"
)
SECRETS = {
    "bearer": ("curl -H 'Authorization: Bearer abc.DEF-123_xyz'", "abc.DEF-123_xyz"),
    "jwt": (f"my session is {JWT} please decode", JWT),
    "openai": (
        "key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 fails",
        "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    ),
    "anthropic": (
        "why does sk-ant-api03-AbCdEfGhIjKlMnOpQrSt_uv-XYZ return 401",
        "sk-ant-api03-AbCdEfGhIjKlMnOpQrSt_uv-XYZ",
    ),
    "slack_bot": (
        "token xoxb-1234567890-0987654321-AbCdEfGhIjKl is invalid",
        "xoxb-1234567890-0987654321-AbCdEfGhIjKl",
    ),
    "slack_user": (
        "我的 xoxp-1111111111-2222222222-AbCdEfGh 不能用",
        "xoxp-1111111111-2222222222-AbCdEfGh",
    ),
    "slack_app": (
        "app token xapp-1-A0123456789-1234567890-abcdef",
        "xapp-1-A0123456789-1234567890-abcdef",
    ),
    "github": (
        "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 leaked",
        "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    ),
    "private_key": (f"debug this key\n{PRIVATE_KEY}\nthanks", "MIIEowIBAAKCAQEAu1SU1"),
    "private_key_assignment": (f"PRIVATE_KEY={PRIVATE_KEY}", "VTLw7onLRnrq0"),
    "password_assignment": ("db password = hunter2!\nnext line", "hunter2!"),
    "json_api_key": ('{"api_key": "abc123secret"}', "abc123secret"),
    "env_line": ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG", "wJalrXUtnFEMI"),
}


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_credentials_are_masked(name):
    text, secret = SECRETS[name]

    redacted = redact_text(text)

    assert secret not in redacted
    assert REDACTED in redacted


def test_text_around_a_credential_survives():
    redacted = redact_text(SECRETS["slack_bot"][0] + "\nsecond line stays")

    assert redacted.startswith("token [REDACTED] is invalid")
    assert redacted.endswith("second line stays")
    assert redact_text(SECRETS["slack_user"][0]) == "我的 [REDACTED] 不能用"


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "Analyze this data",
        "Review this Python traceback and refactor the function",
        "請幫我總結這份設計文件，重點放在延遲與成本。",
        "max_tokens=100 and token count: 5 tokens",
        "sk-learn is a library; the task-force met at the desk-1234",
        "The password policy requires twelve characters.",
    ],
)
def test_ordinary_text_is_unchanged(text):
    assert redact_text(text) == text


def test_redaction_is_idempotent():
    for text, _secret in SECRETS.values():
        once = redact_text(text)
        assert redact_text(once) == once


def test_redact_masks_credential_keys_and_nested_strings():
    payload = {
        "api_key": "plain",
        "nested": {"slack_bot_token": "xoxb-x", "note": f"Bearer {JWT}"},
        "items": ["ok", "password: hunter2"],
        "max_tokens": 5,
        "total_tokens": 9,
        "token_count": 3,
    }

    assert redact(payload) == {
        "api_key": REDACTED,
        "nested": {"slack_bot_token": REDACTED, "note": "Bearer [REDACTED]"},
        "items": ["ok", "password: [REDACTED]"],
        "max_tokens": 5,
        "total_tokens": 9,
        "token_count": 3,
    }


@pytest.mark.parametrize(
    "key",
    ["password", "API_KEY", "x-api-key", "authorization", "slack_signing_secret"],
)
def test_credential_keys(key):
    assert is_credential_key(key)


@pytest.mark.parametrize("key", ["max_tokens", "token_count", "query_text", "keys"])
def test_non_credential_keys(key):
    assert not is_credential_key(key)


@pytest.mark.parametrize(
    "text",
    [
        "a-" * 100_000,
        "eyJ" * 70_000,
        "password" + " " * 200_000,
        "x" * 200_000,
        "-----BEGIN " + "A" * 200_000,
    ],
)
def test_adversarial_input_is_linear(text):
    started = time.perf_counter()

    redact_text(text)

    assert time.perf_counter() - started < 1.0


def test_context_summary_reports_size_and_hash_only():
    context = "其他成員的訊息 secret thread"

    assert context_summary(context) == {
        "context_chars": len(context),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }
    assert context_summary(None) == {"context_chars": 0, "context_sha256": None}
    assert context_summary("") == {"context_chars": 0, "context_sha256": None}
