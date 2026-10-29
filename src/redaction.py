"""Mask credentials in text and payloads before they leave the process.

A port of the rules in `agent-runtime/src/telemetry.ts` (`redact`), plus
well-known provider key prefixes. Only credentials are masked; ordinary text
passes through unchanged. Every pattern is bounded so a long input cannot make
the regex engine backtrack quadratically.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional

REDACTED = "[REDACTED]"

_CREDENTIAL_SUFFIX = (
    r"(?:AUTHORIZATION|SECRET[_-]?ACCESS[_-]?KEY|SECRET[_-]?KEY|PRIVATE[_-]?KEY"
    r"|ACCESS[_-]?KEY|API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN"
    r"|GATEWAY[_-]?TOKEN|SIGNING[_-]?SECRET|PASSWORD|PASSPHRASE|SECRET|TOKEN)"
)
_CREDENTIAL_KEY = re.compile(rf"(?:^|[_-]){_CREDENTIAL_SUFFIX}$", re.IGNORECASE)

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9]{1,32} )?PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END (?:[A-Z0-9]{1,32} )?PRIVATE KEY-----|\Z)"
)
_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]+", re.IGNORECASE)
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,65536}"
    r"\.[A-Za-z0-9_-]{0,4096}"
)
_PROVIDER_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-[A-Za-z0-9_-]{16,}"  # OpenAI, Anthropic (sk-ant-), OpenRouter (sk-or-)
    r"|xox[a-z]-[A-Za-z0-9-]{10,}"  # Slack bot/user/app-level tokens
    r"|xapp-[A-Za-z0-9-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|tvly-[A-Za-z0-9_-]{16,}"
    r"|AIza[A-Za-z0-9_-]{35}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r")"
)
# `name = value` / `name: value` where the name ends in a credential word.
# The rest of the line is masked rather than guessing the value's grammar.
_ASSIGNMENT = re.compile(
    rf"((?<![A-Za-z0-9_-])[A-Za-z0-9_-]{{0,128}}{_CREDENTIAL_SUFFIX}"
    r"[\"']?[\t ]{0,32}[:=][\t ]{0,32})"
    r"(?![\t ]*\[REDACTED\](?=\r|\n|\Z))[^\r\n]+(?=\r|\n|\Z)",
    re.IGNORECASE,
)


def is_credential_key(key: Any) -> bool:
    return isinstance(key, str) and bool(_CREDENTIAL_KEY.search(key))


def redact_text(text: str) -> str:
    """Return `text` with credentials replaced by `[REDACTED]`."""
    if not text:
        return text
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _BEARER.sub(f"Bearer {REDACTED}", text)
    text = _JWT.sub(REDACTED, text)
    text = _PROVIDER_KEY.sub(REDACTED, text)
    return _ASSIGNMENT.sub(rf"\1{REDACTED}", text)


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
    """Describe a prompt context by size and hash instead of its text."""
    text = context or ""
    return {
        "context_chars": len(text),
        "context_sha256": (
            hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
            if text
            else None
        ),
    }
