import time

import pytest

from flink.logic import build_request_enriched_event
from src.llm_router_part3_pipeline import build_request_raw_event
from src.utils.schema import QueryRequest, UserTier
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


def test_context_summary_reports_size_only():
    context = "其他成員的訊息 secret thread"

    assert context_summary(context) == {"context_chars": len(context)}
    assert context_summary(None) == {"context_chars": 0}
    assert context_summary("") == {"context_chars": 0}


PGP_KEY = (
    "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
    "\n"
    "lQOYBF0Y8mMBCADGu7yWkzxV0ZyA9dZf3zU1Ym7Y0Gf0r1qkQ8cW6pJ1u0Kz3Xx0\n"
    "=Ab12\n"
    "-----END PGP PRIVATE KEY BLOCK-----"
)
MORE_SECRETS = {
    "url_password": (
        "connect to postgres://admin:hunter2@db.internal:5432/app",
        "hunter2",
    ),
    "database_url": (
        "DATABASE_URL=postgresql://svc:S3cr3t-Pa55@db:5432/app?sslmode=require",
        "S3cr3t-Pa55",
    ),
    "redis_url_no_user": ("redis://:r3dis-pass@cache:6379/0 is down", "r3dis-pass"),
    "curl_user": ("curl -u admin:hunter2 https://api.example.com", "hunter2"),
    "curl_long_user": ("curl --user 'deploy:Tr0ub4dor&3' https://x", "Tr0ub4dor&3"),
    "pgp_private_key": (f"my key:\n{PGP_KEY}\nwhy?", "lQOYBF0Y8mMBCADGu7yW"),
    "stripe_secret": (
        "stripe key sk_live_51H8xYzAbCdEfGhIjKlMnOp is rejected",
        "sk_live_51H8xYzAbCdEfGhIjKlMnOp",
    ),
    "stripe_restricted": (
        "rk_live_51H8xYzAbCdEfGhIjKlMnOp",
        "rk_live_51H8xYzAbCdEfGhIjKlMnOp",
    ),
    "slack_webhook": (
        "post to https://hooks.slack.com/services/T0123ABCD/B0456EFGH/"
        "a1B2c3D4e5F6g7H8i9J0k1L2 please",
        "a1B2c3D4e5F6g7H8i9J0k1L2",
    ),
    "huggingface": (
        "HF says hf_AbCdEfGhIjKlMnOpQrStUvWxYz01234567 is invalid",
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz01234567",
    ),
    "google_oauth": (
        "access token ya29.a0AfH6SMBx3yZ-AbCdEfGhIjKlMnOpQrStUv expired",
        "ya29.a0AfH6SMBx3yZ-AbCdEfGhIjKlMnOpQrStUv",
    ),
    "cookie_header": (
        "Cookie: session=9f8e7d6c5b4a; csrftoken=abc123def456",
        "9f8e7d6c5b4a",
    ),
    "db_pass": ("DB_PASS=Zx9!kLm2qR in the env", "Zx9!kLm2qR"),
    "passwd": ("passwd: s3cr3tValue", "s3cr3tValue"),
    "pwd": ("pwd=Qw3rty!99", "Qw3rty!99"),
    "fullwidth_colon": ("我的 password：hunter2 登不進去", "hunter2"),
    "aws_pair": (
        "AKIAIOSFODNN7EXAMPLE wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    "aws_secret_name": (
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    "authorization_basic": ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
}


@pytest.mark.parametrize("name", sorted(MORE_SECRETS))
def test_more_credential_formats_are_masked(name):
    text, secret = MORE_SECRETS[name]

    redacted = redact_text(text)

    assert secret not in redacted
    assert REDACTED in redacted
    assert redact_text(redacted) == redacted


def test_url_redaction_keeps_the_rest_of_the_url():
    assert (
        redact_text("postgres://admin:hunter2@db.internal:5432/app")
        == "postgres://admin:[REDACTED]@db.internal:5432/app"
    )


def test_assignment_masks_only_the_value_not_the_rest_of_the_line():
    assert (
        redact_text("password=hunter2 and then explain the error")
        == "password=[REDACTED] and then explain the error"
    )


def test_a_bare_forty_character_string_is_not_treated_as_an_aws_secret():
    text = "commit 0123456789abcdef0123456789abcdef01234567 broke the build"

    assert redact_text(text) == text


# Ordinary, code and CJK queries: none holds a credential, so redaction must
# leave each one unchanged and Flink must classify it the same way.
ORDINARY_QUERIES = [
    "Who is the bearer of this note?",
    "Token: what is a JWT?",
    "def login(password=None): pass",
    "Bearer\n\nnext paragraph",
    "Secret: debug this refactor of the architecture, root cause please",
    "Password reset emails are not arriving; how do I debug SMTP?",
    "Explain OAuth bearer tokens versus API keys",
    "token = get_token()\nheaders = {'Authorization': f'Bearer {token}'}",
    "config.password = os.environ['DB_PASSWORD']",
    "api_key = settings.OPENAI_API_KEY",
    "password = input('Password: ')",
    "Write a Python function that validates a password: at least 8 characters",
    "SELECT user, password_hash FROM users WHERE token_expires_at < now()",
    "Set secret: true in the YAML and token: null when unused",
    "export API_KEY=$OPENAI_API_KEY",
    "The API key rotation runbook: step 1, revoke; step 2, reissue.",
    "Cookie: what is a cookie?",
    "Use `curl -u` to pass credentials",
    "Connection strings look like postgres://user@host/db",
    "Bearer authentication-and-authorization flow diagram",
    "請幫我總結這份設計文件，重點放在延遲與成本。",
    "密碼重設信沒有寄出，要怎麼排查？",
    "token：請解釋 refresh token 的用途",
    "Compare the tradeoffs of Redis vs Kafka for the session store",
    "Critical production outage, respond ASAP",
]


@pytest.mark.parametrize("query", ORDINARY_QUERIES)
def test_ordinary_queries_pass_redaction_unchanged(query):
    assert redact_text(query) == query


FLINK_FEATURE_KEYS = (
    "query_type",
    "query_complexity",
    "query_token_estimate",
    "long_context",
    "code_heavy",
    "requires_high_reasoning",
    "requires_low_latency",
    "route_to_fast_lane",
    "priority",
)


@pytest.mark.parametrize("query", ORDINARY_QUERIES)
def test_flink_features_match_for_old_and_new_events(query):
    context = "Earlier in the thread: we discussed the rollout plan."
    metadata = {"latency_sla": "interactive"}
    request = QueryRequest(
        query=query,
        user_id="u1",
        user_tier=UserTier.PREMIUM,
        context=context,
        metadata=metadata,
    )
    old_event = {
        "request_id": request.request_id,
        "user_id": "u1",
        "user_tier": "premium",
        "query_text": request.query,
        "context": request.context,
        "metadata": metadata,
        "max_tokens": request.max_tokens,
        "priority": request.priority,
    }

    new_features = build_request_enriched_event(build_request_raw_event(request))
    old_features = build_request_enriched_event(old_event)

    assert {key: new_features[key] for key in FLINK_FEATURE_KEYS} == {
        key: old_features[key] for key in FLINK_FEATURE_KEYS
    }
