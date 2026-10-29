"""Redaction by context: structured values are masked whole, prose is left alone.

The cases come from a review probe of `src/redaction.py`.
"""

import random
import string
import time

import pytest

from flink.logic import build_request_enriched_event
from src.llm_router_part3_pipeline import build_request_raw_event
from src.redaction import REDACTED, is_credential_key, redact_text
from src.utils.schema import QueryRequest, UserTier

AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# (label, text, secret that must not survive)
STRUCTURED_SECRETS = [
    ("env quoted dq", 'OPENAI_API_KEY="abc123def456ghi789"', "abc123def456ghi789"),
    ("env quoted sq", "DB_PASSWORD='p@ss w0rd!'", "w0rd!"),
    ("env export", "export SLACK_BOT_TOKEN=xoxb-123-456-abcdefghij", "xoxb-123-456"),
    ("json api_key", '{"api_key": "abc123secret"}', "abc123secret"),
    ("json apiKey camel", '{"apiKey": "abc123secretZZ"}', "abc123secretZZ"),
    ("json clientSecret camel", '{"clientSecret": "Zq8x7Wv6Ut5"}', "Zq8x7Wv6Ut5"),
    ("json client_secret", '{"client_secret": "Zq8x7Wv6Ut5"}', "Zq8x7Wv6Ut5"),
    ("json password with comma", '{"password": "a,b9c!d"}', "b9c!d"),
    ("json password with space", '{"password": "my secret 9pass"}', "9pass"),
    ("json password letters only", '{"password": "hunterhunter"}', "hunterhunter"),
    ("json escaped quote", r'{"password": "ab\"cd\/ef9"}', "cd"),
    (
        "docker auth",
        '{"auths": {"r.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}}',
        "dXNlcjpw",
    ),
    (
        "yaml password letters",
        "password: CorrectHorseBatteryStaple",
        "CorrectHorseBatteryStaple",
    ),
    ("yaml quoted", 'password: "hunter2!"', "hunter2!"),
    ("yaml token", "token: 9f8e7d6c5b4a3210", "9f8e7d6c5b4a3210"),
    ("yaml nested secret", "  client_secret: Zq8x7Wv6Ut5", "Zq8x7Wv6Ut5"),
    ("yaml secret with braces", "password: abc{123}def", "123}def"),
    ("yaml secret with ampersand", "password: abc&123", "&123"),
    ("yaml secret with backslash", "password: abc\\123x", "123x"),
    ("yaml secret with semicolon", "password: abc;123x", "123x"),
    ("yaml secret with comma", "password: abc,123x", "123x"),
    ("yaml secret with paren", "password: abc(123x)", "123x"),
    ("yaml secret leading paren", "password: (abc123x", "abc123x"),
    ("yaml secret leading bracket", "password: [abc123x]", "abc123x"),
    ("yaml secret with lt", "password: abc<123x>", "123x"),
    ("yaml secret with apostrophe", "password: it's9secret", "9secret"),
    ("yaml secret unicode", "password: pässwörd123", "rd123"),
    ("yaml dollar start", "password: $ecretP4ss", "ecretP4ss"),
    ("yaml percent start", "password: %ecretP4ss", "ecretP4ss"),
    ("assign dotted secret", "password=abc.def", "abc.def"),
    ("assign dotted secret 2", "secret: my.Secret.Value", "my.Secret.Value"),
    ("assign short", "pwd=a1b2c", "a1b2c"),
    ("assign multi-space", "password    =    hunter2!", "hunter2!"),
    ("assign tab", "password\t=\thunter2!", "hunter2!"),
    ("assign colon then newline", "password:\n  hunter2!", "hunter2!"),
    ("yaml block", "credentials:\n  password: hunter2!", "hunter2!"),
    ("passphrase", "passphrase: purple-monkey-dishwasher", "purple-monkey-dishwasher"),
    ("assign value is word", "password: Summer", "Summer"),
    ("keyword then is", "my password is hunter2", "hunter2"),
    ("密碼", "我的密碼是 hunter2!", "hunter2!"),
    ("密碼 colon", "密碼：hunter2!", "hunter2!"),
    ("密碼 colon ascii", "密碼: hunter2!", "hunter2!"),
    ("密码 simplified", "数据库密码：Zx9kLm2q", "Zx9kLm2q"),
    ("令牌", "令牌: ab12cd34ef56", "ab12cd34ef56"),
    ("密鑰", "API 密鑰 = Qw3rty99xyz", "Qw3rty99xyz"),
    ("密钥", "密钥是 Qw3rty99xyz", "Qw3rty99xyz"),
    ("口令", "口令：s3cr3t!", "s3cr3t!"),
    ("token fullwidth equals", "token＝abc123def456", "abc123def456"),
    ("conn string special chars", "postgres://user:p@ss:w/rd@host:5432/db", "ss:w"),
    ("conn string with slash", "postgres://user:pa/ss@host/db", "pa/ss"),
    ("conn string at in pw", "postgres://user:p@ssword@host/db", "ssword"),
    (
        "conn string with space-encoded",
        "mongodb+srv://u:p%40ss%20w@cluster0.mongodb.net/",
        "p%40ss",
    ),
    (
        "conn string jdbc",
        "jdbc:mysql://host:3306/db?user=root&password=s3cr3t!",
        "s3cr3t!",
    ),
    (
        "conn string query password",
        "https://x/?token=abc123def456ghi",
        "abc123def456ghi",
    ),
    (
        "conn string query api_key",
        "https://x/?api_key=abc123def456ghi&x=1",
        "abc123def456ghi",
    ),
    (
        "ado.net",
        "Server=tcp:x.database.windows.net;Password=Pa$$w0rd;User ID=sa",
        "Pa$$w0rd",
    ),
    ("ado.net Pwd", "Server=x;Uid=sa;Pwd=Pa55w0rd!;", "Pa55w0rd!"),
    (
        "azure storage",
        "DefaultEndpointsProtocol=https;AccountName=x;"
        "AccountKey=abcDEF123+/xyz==;EndpointSuffix=core.windows.net",
        "abcDEF123",
    ),
    (
        "azure sas",
        "https://x.blob.core.windows.net/c?sv=2020&sig=abcDEF123%2Bxyz%3D",
        "abcDEF123",
    ),
    (
        "kubeconfig",
        "users:\n- name: admin\n  user:\n    client-key-data: LS0tLS1CRUdJTiBSU0Eg",
        "LS0tLS1CRUdJTiBSU0Eg",
    ),
    (
        "auth header token scheme",
        "Authorization: token ghx_notaprefixbutsecret123",
        "ghx_notaprefix",
    ),
    (
        "auth header ApiKey scheme",
        "Authorization: ApiKey abc123def456ghi789",
        "abc123def456ghi789",
    ),
    (
        "auth header AWS4",
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/"
        "us-east-1/s3/aws4_request, SignedHeaders=host, Signature=fe5f80f77d5fa3be"
        "ca038a248ff027d0445342fe2855ddc963176630326f1024",
        "fe5f80f77d5fa3beca038a248ff027d0",
    ),
    ("auth header SSWS", "Authorization: SSWS 00abcDEFghiJKLmno123", "00abcDEFghi"),
    ("auth header Splunk", "Authorization: Splunk 0a1b2c3d-4e5f-6789", "0a1b2c3d-4e5f"),
    ("auth header Key", "Authorization: Key a1b2c3d4e5f6", "a1b2c3d4e5f6"),
    (
        "auth header DeepL",
        "Authorization: DeepL-Auth-Key 1a2b3c4d-5e6f:fx",
        "1a2b3c4d-5e6f",
    ),
    ("auth header Bearer short", "Authorization: Bearer abc123", "abc123"),
    ("bearer short alone", "Bearer abc123def", "abc123def"),
    ("bearer 15 chars", "-H 'Bearer abc123def456ghi'", "abc123def456ghi"),
    ("x-api-key header", "x-api-key: abc123def456ghi789", "abc123def456ghi789"),
    ("X-Api-Key header", "X-API-Key: abc123def456", "abc123def456"),
    ("api-key header", "api-key: abc123def456", "abc123def456"),
    (
        "Ocp-Apim",
        "Ocp-Apim-Subscription-Key: 0123456789abcdef0123456789abcdef",
        "0123456789abcdef",
    ),
    ("x-auth-token", "X-Auth-Token: abc123def456ghi", "abc123def456ghi"),
    ("PRIVATE-TOKEN header", "PRIVATE-TOKEN: abc123def456", "abc123def456"),
    ("API key spaced", "API key: Zx9kLm2qR7", "Zx9kLm2qR7"),
    ("api key spaced equals", "api key = Zx9kLm2qR7", "Zx9kLm2qR7"),
    ("secret access key spaced", f"secret access key {AWS_SECRET}", "wJalrXUtnFEMI"),
    (
        "aws configure transcript",
        "AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE\n"
        f"AWS Secret Access Key [None]: {AWS_SECRET}",
        "wJalrXUtnFEMI",
    ),
    (
        "aws csv",
        f"Access key ID,Secret access key\nAKIAIOSFODNN7EXAMPLE,{AWS_SECRET}",
        "wJalrXUtnFEMI",
    ),
    (
        "aws ini",
        "[default]\naws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"
        f"aws_secret_access_key = {AWS_SECRET}",
        "wJalrXUtnFEMI",
    ),
    (
        "aws session token",
        "aws_session_token = IQoJb3JpZ2luX2VjEJr//////////wEaCXVzLWVhc3QtMSJHMEUCIQ",
        "IQoJb3JpZ2lu",
    ),
    (
        "aws env pair",
        f"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY={AWS_SECRET}",
        "wJalrXUtnFEMI",
    ),
    (
        "aws json",
        '{"AccessKeyId": "ASIAIOSFODNN7EXAMPLE", "SecretAccessKey": '
        f'"{AWS_SECRET}", "SessionToken": "IQoJb3JpZ2luX2VjEJr"}}',
        "wJalrXUtnFEMI",
    ),
    (
        "gcp sa json",
        '{"type": "service_account", "private_key_id": "abc123def456", '
        '"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC'
        '\\n-----END PRIVATE KEY-----\\n"}',
        "MIIEvQIBADAN",
    ),
    (
        "gcp sa private_key_id",
        '{"private_key_id": "0123456789abcdef0123456789abcdef01234567"}',
        "0123456789abcdef",
    ),
    ("gcp api key", "AIzaSyA1234567890abcdefghijklmnopqrstu", "AIzaSyA1234567890"),
    ("ghp", "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "ghp_AbCd"),
    (
        "github_pat",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ",
        "github_pat_11",
    ),
    ("gho", "gho_16C7e42F292c6912E7710c838347Ae178B4a", "gho_16C7"),
    ("sk-proj", "sk-proj-abcDEF123_ghiJKL456-mnoPQR789stuVWX", "sk-proj-abc"),
    ("sk-ant", "sk-ant-api03-abcDEF123_ghiJKL456-mnoPQR789stuVWX-AAAA", "sk-ant-api03"),
    ("sk-ant admin", "sk-ant-admin01-abcDEF123_ghiJKL456-mnoPQR789", "sk-ant-admin01"),
    (
        "xoxb",
        "xoxb-" "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
        "xoxb-1234",
    ),
    ("xoxe", "xoxe.xoxp-" "1-Mi0yLTEyMzQ1Njc4OTAx", "Mi0yLTEyMzQ1"),
    ("xoxe-1", "xoxe-" "1-My0xLTEyMzQ1Njc4OTAx", "My0xLTEyMzQ1"),
    (
        "slack signing secret",
        "SLACK_SIGNING_SECRET=8f742231b10e8888abcd99yyyzzz85a5",
        "8f742231b10e",
    ),
    (
        "discord webhook",
        "https://discord.com/api/webhooks/123456789012345678/abcDEF-ghiJKL_mnopqrstu",
        "abcDEF-ghiJKL",
    ),
    (
        "npm token",
        "//registry.npmjs.org/:_authToken=npm_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "npm_AbCd",
    ),
    ("npm token bare", "npm_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "npm_AbCd"),
    (
        "sendgrid",
        "SG." "abcdefghijklmnopqrstuv.abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",
        "SG.abcdef",
    ),
    ("twilio", "SK" "0123456789abcdef0123456789abcdef", "SK0123456789"),
    (
        "pypi",
        "pypi-AgEIcHlwaS5vcmcCJDAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMAACKlsz",
        "pypi-AgEI",
    ),
    (
        "doppler",
        "DOPPLER: dp.st.dev.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "dp.st.dev.AbCd",
    ),
    ("openrouter", "sk-or-v1-0123456789abcdef0123456789abcdef", "sk-or-v1"),
    (
        "gemini key in url",
        "https://generativelanguage.googleapis.com/v1/models?key="
        "AIzaSyA1234567890abcdefghijklmnopqrstu",
        "AIzaSyA1234567890",
    ),
    ("Set-Cookie", "Set-Cookie: sid=abc123; Path=/; HttpOnly", "abc123"),
    ("cookie lowercase header in curl", "curl -H 'cookie: sid=abc123def'", "abc123def"),
    ("curl -b", "curl -b 'sid=abc123def456' https://x", "abc123def456"),
    ("curl --cookie", 'curl --cookie "sid=abc123def456" https://x', "abc123def456"),
    ("curl -u no space", "curl -uadmin:hunter2 https://x", "hunter2"),
    ("curl -u double quote", 'curl -u "admin:hunter 2" https://x', "hunter"),
    ("curl -u with colon in pw", "curl -u admin:pa:ss https://x", "ss"),
    ("wget password", "wget --password=hunter2! https://x", "hunter2!"),
    ("mysql -p", "mysql -u root -phunter2 db", "hunter2"),
    ("psql PGPASSWORD", "PGPASSWORD=hunter2 psql -h x", "hunter2"),
    ("docker login", "docker login -u me -p hunter2! registry", "hunter2!"),
    ("basic auth url with port", "http://admin:hunter2@127.0.0.1:8080", "hunter2"),
    ("url pw in markdown", "<https://admin:hunter2@x.com>", "hunter2"),
    ("url pw in parens", "(https://admin:hunter2@x.com)", "hunter2"),
    ("url scheme 1 char", "s://admin:hunter2@x", "hunter2"),
    (
        "private key escaped",
        '"-----BEGIN OPENSSH PRIVATE KEY-----\\nb3BlbnNzaC1rZXktdjEAAAAA'
        '\\n-----END OPENSSH PRIVATE KEY-----"',
        "b3BlbnNzaC1rZXkt",
    ),
    (
        "private key encrypted",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHDBOBgkqhkiG9w0BBQ0w\n"
        "-----END ENCRYPTED PRIVATE KEY-----",
        "MIIFHDBOBgkq",
    ),
    (
        "private key lowercase",
        "-----begin private key-----\nMIIFHDBOBgkqhkiG9w0BBQ0w\n-----end private key-----",
        "MIIFHDBOBgkq",
    ),
    (
        "private key without end",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAu1SU1LfVLPHCozMx\n"
        "H2Mo4lgOEePzNm0tRgeLezV6ffAt0gun",
        "H2Mo4lgOEePzNm0t",
    ),
    (
        "putty key",
        "PuTTY-User-Key-File-3: ssh-ed25519\nPrivate-Lines: 1\n"
        "AAAAIBqT1CzMaS7+5GOD6VTRgpm3pV9rM",
        "AAAAIBqT1Cz",
    ),
    (
        "jwt",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "dozjgNryP4J3",
    ),
    (
        "token in querystring",
        "GET /api?access_token=abc123def456ghi HTTP/1.1",
        "abc123def456ghi",
    ),
    ("password in toml", 'password = "hunter2!"', "hunter2!"),
    ("password in xml", "<password>hunter2!</password>", "hunter2!"),
    ("password in xml attr", '<add key="Password" value="hunter2!"/>', "hunter2!"),
    ("password in properties", "spring.datasource.password=hunter2!", "hunter2!"),
    ("password k8s secret b64", "data:\n  password: aHVudGVyMiE=", "aHVudGVyMiE="),
    ("password in python kwarg", "connect(host='x', password='hunter2!')", "hunter2!"),
    ("password in python kwarg dq", 'connect(password="hunter2!")', "hunter2!"),
    ("password with leading space quoted", 'password: " hunter2!"', "hunter2!"),
    ("password = quoted + space", "password = 'hunter2! more'", "hunter2!"),
    ("password: arrow", "password => 'hunter2!'", "hunter2!"),
    ("password hash ruby", ":password => 'hunter2!'", "hunter2!"),
    ("password php array", "'password' => 'hunter2!'", "hunter2!"),
    ("password go struct", 'Password: "hunter2!",', "hunter2!"),
    ("password long 600", "password=" + "a1" * 300, "a1a1a1"),
    ("api_key in cli flag", "--api-key abc123def456ghi", "abc123def456ghi"),
    ("api_key in cli flag eq", "--api-key=abc123def456ghi", "abc123def456ghi"),
    ("token in cli flag", "--token abc123def456ghi", "abc123def456ghi"),
    ("password cli flag", "--password hunter2!", "hunter2!"),
    ("set password word", "PASSWORD=letmein", "letmein"),
    ("SECRET_KEY django", "SECRET_KEY = 'django-insecure-abc'", "django-insecure-abc"),
    ("hex secret word-like", "secret: deadbeefdeadbeefdeadbeef", "deadbeefdeadbeef"),
    ("password in angle brackets", 'DB_PASSWORD="<_b:-1w9&1!(JW*Cshw>"', "1w9&1!"),
    ("code-like password", "password: V$poI4$Y[GcYXxWbVoPQ", "GcYXxW"),
]
# Deliberately not masked:
# - "PASSWORD=correct horse battery staple": an unquoted value ends at the first
#   space, so only "correct" is masked.
# - A bare 40-character string ("secret is wJalr..."): too many false positives
#   unless it sits next to an access key id or after an aws secret name.


@pytest.mark.parametrize(
    "text, secret", [case[1:] for case in STRUCTURED_SECRETS], ids=lambda v: str(v)[:40]
)
def test_structured_secrets_are_masked(text, secret):
    redacted = redact_text(text)

    assert secret not in redacted
    assert REDACTED in redacted
    assert redact_text(redacted) == redacted


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Authorization: Bearer abc123def456",
            "Authorization: Bearer [REDACTED]",
        ),
        (
            "Authorization: AWS4-HMAC-SHA256 Credential=AKIA1/x, Signature=ab12cd",
            "Authorization: AWS4-HMAC-SHA256 [REDACTED] Signature=[REDACTED]",
        ),
        (
            "Cookie: sid=abc123; theme=dark\nnext line",
            "Cookie: sid=[REDACTED]; theme=[REDACTED]\nnext line",
        ),
        (
            "Set-Cookie: sid=abc123; Path=/; HttpOnly",
            "Set-Cookie: sid=[REDACTED]; Path=/; HttpOnly",
        ),
        (
            '{"password": "a,b9c!d", "user": "bob"}',
            '{"password": "[REDACTED]", "user": "bob"}',
        ),
        ("DB_PASSWORD=Xk7&mQ2;pL9{zR} is set", "DB_PASSWORD=[REDACTED] is set"),
        (
            "postgres://user:p@ss:w/rd@host:5432/db",
            "postgres://user:[REDACTED]@host:5432/db",
        ),
        ("mysql -u root -phunter2 db", "mysql -u root -p[REDACTED] db"),
        ("curl -u admin:hunter2 https://x", "curl -u admin:[REDACTED] https://x"),
        ("--password=hunter2! --verbose", "--password=[REDACTED] --verbose"),
        ("password=" + "a1" * 300 + " tail", "password=[REDACTED] tail"),
    ],
)
def test_structured_masking_keeps_names_schemes_and_the_rest_of_the_line(
    text, expected
):
    assert redact_text(text) == expected


ORDINARY_TEXT = [
    "--- PASS: TestLoginFlow2 (0.00s)",
    "--- PASS: TestHandler_Redirects/case_1 (0.01s)\n--- FAIL: TestOther (0.02s)",
    "=== RUN   TestRedact\n--- PASS: TestRedact (0.00s)\nPASS\nok  \tgithub.com/x/y\t0.012s",
    "PWD=/home/runner/work/repo\nHOME=/home/runner",
    "pwd: /Users/alice/code/project",
    "set max_token=128000 for the long-context model",
    "Token: 2024-05-01 release notes are attached",
    "Authorization: Bearer YOUR_TOKEN_HERE",
    "Authorization: Basic <base64>",
    "Authorization: Bearer <token> is the format; explain why",
    "Use Authorization: Token placeholder123 and explain",
    "the authorization: requires_admin_role_v2 check fails",
    "Authorization: user-must-be-admin rule",
    "authorization=RBAC_v2 policy",
    "secret: v1.2.3",
    "secret_version: 3",
    "TOKEN_LIMIT=4096",
    "token: 4096 tokens max",
    "token=100000",
    "PASSWORD_MIN_LENGTH=12",
    "password: 8+ characters with 1 number",
    "api_key: string (required) - your API key",
    "api_key: str | None = None",
    "api_key: Optional[str] = None",
    "token: str = Depends(oauth2_scheme)",
    "secret_key: bytes = os.urandom(32)",
    "access_token: string;  // JWT from /login",
    "def f(password: str, token: int = 0): ...",
    "password = 'changeme123' is the default; how do I change it?",
    "api_key=YOUR_API_KEY",
    "API_KEY=<your-key-here>",
    "API_KEY=sk-...",
    "token: ${{ secrets.GITHUB_TOKEN }}",
    "password: ${DB_PASSWORD}",
    "password: {{ .Values.password }}",
    "password: !vault |",
    "SECRET=$(openssl rand -hex 32)",
    "accessToken: this.authService.getToken()",
    "token: response.data.access_token",
    "token: response.data[0].token",
    "secret: process.env.SECRET",
    "api_key: config['openai']['api_key']",
    "password: get_password(user_id=42)",
    "TOKEN = os.getenv('TOKEN', 'dev')",
    "access_token = oauth2_token",
    "api_key = sha256_hex",
    "pass: 3/3 tests",
    "Bypass: cache_v2 enabled",
    "compass: N45E",
    "PASS: 12/12",
    "Bearer tokens (RFC-6750) are sent in the Authorization header",
    "Bearer token-based-authentication-2fa explained",
    "Bearer 2024-roadmap-planning-doc",
    "the Cookie: header contains name=value pairs separated by semicolons; explain each",
    "curl -u $USER:$PASS https://x",
    "curl -u user:password https://x",
    "ssh -u admin:admin",
    "git clone https://github.com/org/repo.git",
    "https://user@github.com/org/repo",
    "http://localhost:8080/path@v2",
    "mailto:someone@example.com",
    "see https://example.com/a:b@c for details",
    "npm install @scope/pkg@1.2.3",
    "ftp://anonymous:guest@ftp.example.com",
    "postgres://user:password@localhost/db is the example DSN",
    "redis://:@localhost:6379",
    "時間：2024-05-01，token：abc 是什麼？",
    "請問 password：與 passwd 有什麼差別",
    "密碼：至少 8 個字元",
    "我的 token: 在哪裡設定？",
    "密钥 secret：用來簽章",
    "價格：$100，pass：通過",
    "AKIAIOSFODNN7EXAMPLE is the docs example id",
    "s3 key prefix: logs/2024/05/01/",
    "the private key: id_rsa (not .pub)",
    "the access key: is required for S3",
    "sk-learn and sk-image libraries",
    "x-api-key: header is required",
    "hf_hub_download(repo_id='x')",
    "ya29 is a prefix",
    "whsec_ prefix explained",
    "My file starts with -----BEGIN RSA PRIVATE KEY----- but ssh says invalid format",
    "-----BEGIN PRIVATE KEY-----\n\nWhy does openssl reject this header?",
    "API key rotation runbook: step 1, revoke; step 2, reissue.",
    "Where do I put my API key? The docs say api key = <your key>.",
    "What does secret access key mean in AWS?",
    "Use the --token flag to pass your token",
    "the --password option is deprecated in mysql 8",
    "Should I pass --api-key as an argument or read it from the environment?",
]


@pytest.mark.parametrize("text", ORDINARY_TEXT, ids=lambda v: v[:40])
def test_ordinary_text_passes_through_unchanged(text):
    assert redact_text(text) == text


def test_a_pem_header_in_prose_never_masks_to_the_end_of_the_text():
    text = (
        "My key starts with -----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMx\n"
        "\n"
        "Why does ssh-keygen reject it? Please debug."
    )

    assert redact_text(text) == (
        "My key starts with [REDACTED]\n\nWhy does ssh-keygen reject it? Please debug."
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Cookie: theme=dark is not persisted; why does the login function reset it?",
            "Cookie: theme=[REDACTED] is not persisted; why does the login function "
            "reset it?",
        ),
        (
            "My request sends Cookie: sid=abc but the server returns 401",
            "My request sends Cookie: sid=[REDACTED] but the server returns 401",
        ),
        ("password: at-least-8-chars", "password: [REDACTED]"),
        (
            "password: P@ssw0rd is an example of a weak password",
            "password: [REDACTED] is an example of a weak password",
        ),
    ],
)
def test_structured_values_in_prose_are_masked_word_for_word(text, expected):
    assert redact_text(text) == expected


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
FLINK_QUERIES = ORDINARY_TEXT + [
    "My file starts with -----BEGIN RSA PRIVATE KEY----- but ssh says invalid format; "
    "write python code to convert it and debug the function",
    "PASSWORD=Xk7&mQ2;pL9{zR} is in my .env and the python function fails, debug it",
    "My request sends Cookie: sid=abc but the server returns 401, what is the bug in "
    "my Python code? Please debug and refactor the function",
    "Cookie: theme=dark is not persisted; why does the login function reset it? "
    "Explain and debug the code",
    "--- PASS: TestHandler_Redirects/case_1 (0.01s)\n--- FAIL: TestOther (0.02s)\n"
    "why does TestOther fail?",
    "password: P@ssw0rd is an example of a weak password; write a python function to "
    "validate passwords",
]


@pytest.mark.parametrize("query", FLINK_QUERIES, ids=lambda v: v[:40])
def test_flink_features_do_not_change_when_the_query_is_redacted(query):
    metadata = {"latency_sla": "interactive"}
    request = QueryRequest(
        query=query, user_id="u1", user_tier=UserTier.PREMIUM, metadata=metadata
    )
    old_event = {
        "request_id": request.request_id,
        "user_id": "u1",
        "user_tier": "premium",
        "query_text": request.query,
        "metadata": metadata,
        "max_tokens": request.max_tokens,
        "priority": request.priority,
    }

    new_features = build_request_enriched_event(build_request_raw_event(request))
    old_features = build_request_enriched_event(old_event)

    assert {key: new_features[key] for key in FLINK_FEATURE_KEYS} == {
        key: old_features[key] for key in FLINK_FEATURE_KEYS
    }


N = 200_000
ADVERSARIAL = {
    "x space": "x " * (N // 2),
    "password: repeated": "password: " * (N // 10),
    "password:x repeated": "password:x " * (N // 11),
    "pwd_ repeated": "a_pwd " * (N // 6),
    "_ run": "_" * N,
    "- run with pass": ("-" * 63 + "pass ") * (N // 68),
    "long name then pass": (" " + "a" * 64 + "_" * 1 + "x") * (N // 67),
    "ab:// repeated": "ab://" * (N // 5),
    "ab://x: + y run": ("ab://x:" + "y" * 250 + " ") * (N // 258),
    "ab://x:y no @ long": "ab://" + "x" * 255 + ":" + "y" * N,
    "url user run": ("ab://" + "u" * 255 + ":" + "p" * 255 + "/") * (N // 516),
    "Cookie:": "Cookie: " * (N // 8),
    "Cookie: name no =": ("Cookie: " + "a" * 256 + " ") * (N // 265),
    "Cookie pairs": "Cookie: " + "a=b; " * (N // 5),
    "curl -u": "curl " + ("-u " + "a" * 255 + ":" + "b" * 255 + "'") * (N // 515),
    "-u x:": "curl " + "-u x:" * (N // 5),
    "Bearer ": "Bearer " * (N // 7),
    "Bearer long": "Bearer " + "a" * N,
    "authorization:": "authorization: " * (N // 15),
    "authorization: Bearer": "authorization: Bearer " * (N // 22),
    "AKIA sep": ("AKIA" + "A" * 16 + " " * 16) * (N // 36),
    "AKIA sep 40": ("AKIA" + "A" * 16 + " " + "a" * 41) * (N // 62),
    "hooks.slack": "https://hooks.slack.com/services/" * (N // 33),
    "BEGIN": "-----BEGIN A B C PRIVATE KEY-----" * (N // 33),
    "BEGIN then END": "-----BEGIN PRIVATE KEY-----" * (N // 27)
    + "!"
    + "-----END PRIVATE KEY-----",
    "BEGIN newlines": "-----BEGIN PRIVATE KEY-----" + "\n" * N,
    "eyJ.": ("eyJ" + "a" * 8 + ".") * (N // 12),
    "dotted value": "password=" + "a." * 256,
    "assign value 512 x many": ("password=" + "a1" * 300 + "\n") * (N // 610),
    "fullwidth": "password：" * (N // 9),
    "nested quotes": "password\"'" * (N // 10),
    "quoted unterminated": 'password: "' + "a" * N,
    "quoted alternating": ("password: \"x' " + "password: 'y\" ") * (N // 30),
    "x_secret run": "x_secret" * (N // 8),
    "secret= nested": "secret=" * (N // 7),
    "sk- run": "sk-" + "a" * N,
    "CJK": "密碼" * N,
    "CJK is": "密碼是" * (N // 3),
    "mysql -p": "mysql " + "-p " * (N // 3),
    "flag run": "--password " * (N // 11),
    "spaced names": "api key " * (N // 8),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_long_adversarial_input_is_linear(name):
    started = time.perf_counter()

    redact_text(ADVERSARIAL[name])

    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "template",
    ["DB_PASSWORD={}", 'DB_PASSWORD="{}"', '{{"password": "{}"}}', "password: {}"],
)
def test_random_passwords_are_masked_whole(template):
    rng = random.Random(7)
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?/~"
    leaks = []
    for _ in range(500):
        password = "".join(rng.choice(alphabet) for _ in range(20))
        if '"' in template and '"' in password:
            continue
        output = redact_text(template.format(password))
        if any(password[i : i + 6] in output for i in range(len(password) - 5)):
            leaks.append((password, output))

    assert leaks == []


@pytest.mark.parametrize(
    "key",
    [
        "clientSecret",
        "accessToken",
        "sessionToken",
        "AccountKey",
        "client-key-data",
        "db_pass",
        "passwd",
        "pwd",
        "Cookie",
    ],
)
def test_more_credential_keys(key):
    assert is_credential_key(key)


@pytest.mark.parametrize("key", ["bypass", "compass", "pass_count", "tokens_used"])
def test_more_non_credential_keys(key):
    assert not is_credential_key(key)
