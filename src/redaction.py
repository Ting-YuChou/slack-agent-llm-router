"""Mask credentials in text and payloads before they leave the process.

Based on the rules in `agent-runtime/src/telemetry.ts` (`redact`), plus
provider key formats, credentials in URLs, headers and command lines. Query
text is also classified by Flink after redaction, so the rules mask only a
credential-shaped value, never the rest of a line, and ordinary prose and code
pass through unchanged. Every pattern is bounded so a long input cannot make
the regex engine backtrack quadratically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

REDACTED = "[REDACTED]"

_LONG_SUFFIX = (
    r"(?:AUTHORIZATION|SECRET[_-]?ACCESS[_-]?KEY|SECRET[_-]?KEY|PRIVATE[_-]?KEY"
    r"|ACCESS[_-]?KEY|API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN"
    r"|GATEWAY[_-]?TOKEN|SIGNING[_-]?SECRET|PASSWORD|PASSPHRASE|SECRET|TOKEN)"
)
# Short words that also end ordinary words ("bypass"), so they count only as a
# whole name or after a separator ("DB_PASS").
_SHORT_SUFFIX = r"(?:PASSWD|PASS|PWD)"
_CREDENTIAL_KEY = re.compile(
    rf"(?:^|[_-])(?:{_LONG_SUFFIX}|{_SHORT_SUFFIX}|COOKIE)$", re.IGNORECASE
)
_CREDENTIAL_NAME = (
    rf"(?:[A-Za-z0-9_-]{{0,64}}{_LONG_SUFFIX}"
    rf"|(?:[A-Za-z0-9_-]{{0,64}}[_-])?{_SHORT_SUFFIX})"
)
# One printable ASCII token: no spaces, quotes, brackets, commas or CJK text.
_VALUE_CHARS = r"[A-Za-z0-9._~+/=!@#$%^*?:|-]"
_TOKEN_CHARS = r"[A-Za-z0-9._~+/=-]"

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9]{1,32} ){0,3}PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
    r"(?:-----END (?:[A-Z0-9]{1,32} ){0,3}PRIVATE KEY(?: BLOCK)?-----|\Z)"
)
# A bare 40-character string is too common to mask; next to an access key id
# it is the secret key.
_AWS_KEY_PAIR = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?P<sep>[\s,:;'\"|]{1,16})"
    r"[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
)
_URL_PASSWORD = re.compile(
    r"((?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{1,31}://[^\s:/@]{0,256}:)"
    r"[^\s/@]{1,256}(?=@)"
)
_SLACK_WEBHOOK = re.compile(
    r"(https://hooks\.slack\.com/(?:services|workflows|triggers)/)[A-Za-z0-9/_-]{8,}"
)
_AUTHORIZATION = re.compile(
    r"(?P<head>(?<![A-Za-z0-9_-])(?:proxy-)?authorization[\"']?[\t ]{0,8}[:=]"
    r"[\t ]{0,8}[\"']?"
    r"(?P<scheme>(?:Bearer|Basic|Token|Digest|Bot|Negotiate)[\t ]+)?)"
    rf"(?P<value>{_TOKEN_CHARS}{{8,}})",
    re.IGNORECASE,
)
_BEARER = re.compile(
    rf"(?P<head>(?<![A-Za-z0-9_-])Bearer[\t ]+)(?P<value>{_TOKEN_CHARS}{{16,}})",
    re.IGNORECASE,
)
_COOKIE = re.compile(
    r"((?<![A-Za-z0-9_-])(?:Set-)?Cookie[\t ]{0,8}:[\t ]{0,8})"
    r"[^\s=;:]{1,256}=[^\r\n]{0,8192}",
    re.IGNORECASE,
)
_CURL_USER = re.compile(
    r"((?<!\S)(?:-u|--user)(?:[\t ]+|=)[\"']?[^\s:\"']{1,256}:)[^\s\"']{1,256}"
)
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,65536}"
    r"\.[A-Za-z0-9_-]{0,4096}"
)
_PROVIDER_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-[A-Za-z0-9_-]{16,}"  # OpenAI, Anthropic (sk-ant-), OpenRouter (sk-or-)
    r"|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe secret / restricted
    r"|whsec_[A-Za-z0-9]{16,}"
    r"|xox[a-z]-[A-Za-z0-9-]{10,}"  # Slack bot/user/app-level tokens
    r"|xapp-[A-Za-z0-9-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|tvly-[A-Za-z0-9_-]{16,}"
    r"|ya29\.[A-Za-z0-9._-]{20,}"
    r"|AIza[A-Za-z0-9_-]{35}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r")"
)
# `name = value`, `name: value`, `"name": "value"`, also with a fullwidth colon.
_ASSIGNMENT = re.compile(
    rf"(?P<head>(?<![A-Za-z0-9_-]){_CREDENTIAL_NAME}"
    r"[\"']?[\t ]{0,8}[:=\uff1a][\t ]{0,8}[\"']?)"
    rf"(?P<value>{_VALUE_CHARS}{{1,512}})",
    re.IGNORECASE,
)

_NON_SECRET_VALUES = frozenset(
    {"none", "null", "nil", "true", "false", "undefined", "empty", "redacted"}
)
_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_WORD_OR_IDENTIFIER = re.compile(r"[A-Za-z_-]+")


def _credential_shaped(value: str, min_length: int) -> bool:
    """True for a value that could be a secret rather than a word or code."""
    if len(value) < min_length:
        return False
    if value.lower() in _NON_SECRET_VALUES or value[0] in "$%":
        return False  # literals and references such as $OPENAI_API_KEY
    if "://" in value or _DOTTED_IDENTIFIER.fullmatch(value):
        return False  # URLs and code such as os.environ or settings.API_KEY
    # A secret has a digit or a symbol; letters, '_' and '-' alone read as a
    # word or an identifier.
    return not _WORD_OR_IDENTIFIER.fullmatch(value)


def _mask_authorization(match: re.Match) -> str:
    if match.group("scheme") or _credential_shaped(match.group("value"), 8):
        return match.group("head") + REDACTED
    return match.group(0)


def _mask_bearer(match: re.Match) -> str:
    if _credential_shaped(match.group("value"), 16):
        return match.group("head") + REDACTED
    return match.group(0)


def _mask_assignment(match: re.Match) -> str:
    following = match.string[match.end() : match.end() + 1]
    if following in ("(", "[") or not _credential_shaped(match.group("value"), 6):
        return match.group(0)  # a call, an index or an ordinary word
    return match.group("head") + REDACTED


def is_credential_key(key: Any) -> bool:
    return isinstance(key, str) and bool(_CREDENTIAL_KEY.search(key))


def redact_text(text: str) -> str:
    """Return `text` with credentials replaced by `[REDACTED]`."""
    if not text:
        return text
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _AWS_KEY_PAIR.sub(lambda m: REDACTED + m.group("sep") + REDACTED, text)
    text = _URL_PASSWORD.sub(rf"\1{REDACTED}", text)
    text = _SLACK_WEBHOOK.sub(rf"\1{REDACTED}", text)
    text = _AUTHORIZATION.sub(_mask_authorization, text)
    text = _BEARER.sub(_mask_bearer, text)
    text = _COOKIE.sub(rf"\1{REDACTED}", text)
    text = _CURL_USER.sub(rf"\1{REDACTED}", text)
    text = _JWT.sub(REDACTED, text)
    text = _PROVIDER_KEY.sub(REDACTED, text)
    return _ASSIGNMENT.sub(_mask_assignment, text)


def redact(value: Any) -> Any:
    """Redact strings anywhere in a JSON-like value; mask credential-named keys."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            key: REDACTED if is_credential_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def context_summary(context: Optional[str]) -> Dict[str, Any]:
    """Describe a prompt context by its size instead of its text.

    No hash: an unsalted hash of a short context can be confirmed by guessing.
    """
    return {"context_chars": len(context or "")}
