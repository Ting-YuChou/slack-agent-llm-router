"""Mask credentials in text and payloads before they leave the process.

Based on the rules in `agent-runtime/src/telemetry.ts` (`redact`), extended
with provider key formats and the places credentials are written. Rules come
in two tiers:

- Structured contexts mask the whole value, whatever it looks like: quoted
  values after a credential name, env/config assignments at the start of a
  line, query and connection-string parameters, command-line flags, the
  Authorization and Cookie headers, URL passwords and private key blocks.
- A credential name in the middle of a sentence (`my password is hunter2`)
  masks the value only when it looks like a secret, so ordinary questions,
  which Flink also classifies, pass through unchanged.

References ($VAR, ${...}, {{ ... }}), placeholders (<your-key>, YOUR_TOKEN_HERE),
paths, numbers, versions, dates and code (os.environ[...], get_token()) are
not secrets and stay. Every pattern runs in linear time.
"""

from __future__ import annotations

import bisect
import re
from typing import Any, Callable, Dict, List, Match, Optional, Pattern, Tuple

REDACTED = "[REDACTED]"

# ---------------------------------------------------------------------------
# Credential names
# ---------------------------------------------------------------------------

_LONG_SUFFIX = (
    r"(?:AUTHORIZATION"
    r"|(?:SECRET|PRIVATE|ACCESS|API|ACCOUNT|SUBSCRIPTION)[_-]?KEY(?:[_-]?ID)?"
    r"|KEY[_-]DATA|PASSWORD|PASSPHRASE|SECRET|TOKEN)"
)
# Whole names, or after a separator: `DB_PASS`, but never `PASS` or `bypass`.
_NAME = (
    r"(?<![A-Za-z0-9_.-])(?:"
    rf"[A-Za-z0-9_.-]{{0,64}}{_LONG_SUFFIX}"
    r"|(?:[A-Za-z0-9_.-]{0,64}[_.-])?(?:PASSWD|PWD)"
    r"|[A-Za-z0-9_.-]{0,64}[_.-]PASS"
    r")"
)
_SEPARATOR = r"[\t ]*(?:=>|[:=\uff1a\uff1d])[\t ]*"
_CREDENTIAL_KEY = re.compile(
    rf"(?:^|[_.-])(?:{_LONG_SUFFIX}|PASSWD|PWD|COOKIE)$|[_.-](?:PASS|AUTH)$|^AUTH$",
    re.IGNORECASE,
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PASSWORD_NAME = re.compile(r"pass|pwd|密碼|密码|口令", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Values that are not secrets
# ---------------------------------------------------------------------------

_REFERENCE = re.compile(
    r"\$\{\{?[\w.\s:-]*\}{0,2}|\$\([\w.\s-]*\)?|\{\{[\w.\s-]*(?:\}\})?"
    r"|%\(\w*\)?[sd]?|\$[A-Z_][A-Z0-9_]*|%[A-Za-z_]\w*%|![A-Za-z]+"
)
_PLACEHOLDER = re.compile(
    r"<[A-Za-z][\w .-]{0,63}>|your[_-][\w-]*|[\w-]*[_-]here|x{3,}|\*{3,}"
    r"|[\w.-]*\.\.\.|placeholder\w*|changeme\w*",
    re.IGNORECASE,
)
_PLACEHOLDER_WORDS = frozenset(
    {
        "password",
        "passwd",
        "pass",
        "secret",
        "token",
        "guest",
        "none",
        "null",
        "nil",
        "true",
        "false",
        "undefined",
        "empty",
    }
)
_NUMBER = re.compile(r"\d+(?:[.,_]\d+)*")
_QUANTITY = re.compile(r"[<>~=]?\d+(?:[.,]\d+)?[+%]")
_VERSION = re.compile(r"v?\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.]+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?")
_PATH = re.compile(r"(?:/|~/|\.{1,2}/)[\w.@~+-]*(?:/[\w.@~+-]*)*")
_TYPE_NAMES = frozenset(
    {
        "str",
        "int",
        "bytes",
        "bool",
        "float",
        "string",
        "number",
        "boolean",
        "any",
        "object",
        "dict",
        "list",
        "optional",
        "unknown",
        "text",
    }
)
_CODE_CHARS = re.compile(r"[\w$.()\[\]'\"=,:/ -]+")
_CODE_CALL = re.compile(
    r"(?P<callee>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)[\[(](?P<arg>.?)"
)
_CODE_ROOT_CHAIN = re.compile(
    r"(?:this|self|cls|os|sys|process|response|resp|res|req|request|config|cfg"
    r"|conf|settings|env|environ|window|document|app|ctx|context|data|args"
    r"|kwargs|opts|options|params|props|state|store|secrets|vars|module|exports)"
    r"(?:\.[A-Za-z_$][\w$]*)+"
)
_MAX_EXPRESSION = 512
_TRAILING_ATTRIBUTES = re.compile(r"(?:\.[A-Za-z_$][\w$]*)+$")
_SNAKE_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
_LOWER_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")
_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_WORD_OR_IDENTIFIER = re.compile(r"[A-Za-z_-]+")
_ALPHA_WORD = re.compile(r"[A-Za-z]+")
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def _bare(value: str) -> str:
    return value.strip().rstrip(",;").strip("\"'")


def _is_reference_or_placeholder(value: str) -> bool:
    """Empty, already masked, a variable reference or an obvious placeholder."""
    bare = _bare(value)
    if not bare or REDACTED in value:
        return True
    return bool(
        _REFERENCE.fullmatch(bare)
        or _PLACEHOLDER.fullmatch(bare)
        or bare.lower() in _PLACEHOLDER_WORDS
    )


def _is_literal_data(value: str, name: str) -> bool:
    """A number, version, date or path: data, not a credential."""
    bare = _bare(value)
    if _QUANTITY.fullmatch(bare) or _VERSION.fullmatch(bare) or _DATE.fullmatch(bare):
        return True
    if _NUMBER.fullmatch(bare) and not _PASSWORD_NAME.search(name):
        return True  # max_token=128000; a numeric *password* is still masked
    return bool(_PATH.fullmatch(bare))


def _is_code(value: str) -> bool:
    """A type annotation or an expression such as os.environ['X'] or get()."""
    bare = value.strip().rstrip(",;")
    if bare.lower() in _TYPE_NAMES or _CODE_ROOT_CHAIN.fullmatch(bare):
        return True
    # A call or subscript, optionally followed by attributes: x.get(...).token
    if not _CODE_CHARS.fullmatch(bare):
        return False
    if not _TRAILING_ATTRIBUTES.sub("", bare).endswith((")", "]")):
        return False
    if bare.count("(") != bare.count(")") or bare.count("[") != bare.count("]"):
        return False
    call = _CODE_CALL.match(bare)
    if not call:
        return False
    callee, arg = call.group("callee"), call.group("arg")
    return (
        "." in callee or "_" in callee or arg in (")", "]", "'", '"') or (arg.isalpha())
    )


def _credential_shaped(value: str, min_length: int) -> bool:
    """True for a value that could be a secret rather than a word or code."""
    if len(value) < min_length:
        return False
    if _DOTTED_IDENTIFIER.fullmatch(value) or _SNAKE_IDENTIFIER.fullmatch(value):
        return False
    # A secret has a digit or a symbol; letters, '_' and '-' alone read as a
    # word or an identifier.
    return not _WORD_OR_IDENTIFIER.fullmatch(value)


# ---------------------------------------------------------------------------
# Private keys
# ---------------------------------------------------------------------------

_PEM_BEGIN = re.compile(
    r"-----BEGIN (?:[A-Z0-9]{1,32} ){0,3}PRIVATE KEY(?: BLOCK)?-----", re.IGNORECASE
)
_PEM_END = re.compile(
    r"-----END (?:[A-Z0-9]{1,32} ){0,3}PRIVATE KEY(?: BLOCK)?-----", re.IGNORECASE
)
_PEM_NOT_BODY = re.compile(r"[^A-Za-z0-9+/=\s\\:,.()-]")
_PEM_MAX_BODY = 131072
_NEWLINE = r"(?:\r?\n|\\r?\\n)"
# Without an END marker, the body is the run of whole base64 (or armor header)
# lines that follows; the first line of prose ends it.
_PEM_BODY = re.compile(
    rf"(?:(?:[\t ]*{_NEWLINE})+[\t ]*"
    r"(?:(?:Proc-Type|DEK-Info|Comment|Version|Hash|Charset):[^\r\n\\]{0,256}"
    rf"|[A-Za-z0-9+/=]{{4,1024}}(?=[\t ]*(?:{_NEWLINE}|$|-----))))+"
    r"|(?:[\t ]+[A-Za-z0-9+/=]{16,1024})+"
)
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{16}")
_PEM_TRAILING_END = re.compile(
    rf"(?:[\t ]*{_NEWLINE})*[\t ]*-----END (?:[A-Z0-9]{{1,32}} ){{0,3}}"
    r"PRIVATE KEY(?: BLOCK)?-----",
    re.IGNORECASE,
)
_PUTTY = re.compile(
    r"(?P<head>Private-Lines:[\t ]*\d{1,4}[\t ]*\r?\n)"
    r"[A-Za-z0-9+/=]{1,128}(?:\r?\n[A-Za-z0-9+/=]{1,128})*"
    r"|(?P<mac>Private-MAC:[\t ]*)[0-9A-Fa-f]{16,}"
)


def _mask_private_keys(text: str) -> str:
    """Mask BEGIN..END, or BEGIN plus its base64 body; never to end of text."""
    begins = list(_PEM_BEGIN.finditer(text))
    if not begins:
        return text
    ends = list(_PEM_END.finditer(text))
    end_starts = [end.start() for end in ends]
    parts: List[str] = []
    position = 0
    known_bad = -1  # first non-body character found by an earlier search
    for begin in begins:
        if begin.start() < position:
            continue
        stop: Optional[int] = None
        index = bisect.bisect_left(end_starts, begin.end())
        if index < len(ends) and end_starts[index] - begin.end() <= _PEM_MAX_BODY:
            limit = end_starts[index]
            if not begin.end() <= known_bad < limit:
                bad = _PEM_NOT_BODY.search(text, begin.end(), limit)
                if bad is None:
                    stop = ends[index].end()
                else:
                    known_bad = bad.start()
        if stop is None:
            body = _PEM_BODY.match(text, begin.end())
            if body and _BASE64_RUN.search(body.group()):
                stop = body.end()
                trailing_end = _PEM_TRAILING_END.match(text, stop)
                if trailing_end:
                    stop = trailing_end.end()
        if stop is None:
            continue
        parts.append(text[position : begin.start()])
        parts.append(REDACTED)
        position = stop
    parts.append(text[position:])
    return "".join(parts)


def _mask_putty(match: Match[str]) -> str:
    return (match.group("head") or match.group("mac")) + REDACTED


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_AWS_KEY_PAIR = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?P<sep>[\s,:;'\"|]{1,16})"
    r"[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
)

# Up to the last '@' before the host, so a raw '@' in the password is masked.
# A numeric "password" followed by '/' is a port.
_URL_PASSWORD = re.compile(
    r"(?P<head>(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,31}://[^\s:/@]{0,128}:)"
    r"(?!\d{1,5}(?:[/?#\s]|$))(?P<value>[^\s/@][^\s]{0,255})@(?=[A-Za-z0-9\[])"
)
_WEBHOOK = re.compile(
    r"(?P<head>https://(?:hooks\.slack\.com/(?:services|workflows|triggers)/"
    r"|(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\d{5,20}/))"
    r"[A-Za-z0-9/_-]{8,}"
)
_AUTHORIZATION = re.compile(
    r"(?P<head>(?<![\w-])(?:proxy-)?authorization[\"']?[\t ]*[:=][\t ]*[\"']?"
    r"(?:(?P<scheme>[A-Za-z][\w-]{0,31})[\t ]+)?)(?P<value>[^\s\"']+)",
    re.IGNORECASE,
)
_SIGNATURE = re.compile(
    r"(?P<head>(?<![\w-])(?:x-amz-)?(?:signature|sig)=)(?P<value>[^\s&;,\"']+)",
    re.IGNORECASE,
)
_COOKIE_START = re.compile(
    r"(?<![\w-])(?:set-)?cookie[\"']?[\t ]*:[\t ]*[\"']?"
    r"|(?<!\S)--cookie(?:[\t ]+|=)[\"']?",
    re.IGNORECASE,
)
_CURL_COOKIE_START = re.compile(r"(?<!\S)-b(?:[\t ]+|=)?[\"']?")
_COOKIE_PAIR = re.compile(
    r"(?P<name>[^\s=;,\"']{1,256})=(?P<value>[^\s;,\"']*)(?P<sep>;[\t ]*)?"
)
_COOKIE_ATTRIBUTES = frozenset(
    {
        "path",
        "domain",
        "expires",
        "max-age",
        "samesite",
        "secure",
        "httponly",
        "priority",
        "partitioned",
    }
)
_COMMAND = re.compile(
    r"(?<![\w-])(?P<command>curl|mysql|mysqldump|mysqladmin|mariadb"
    r"|docker[\t ]+login)\b[^\r\n]*",
    re.IGNORECASE,
)
_QUOTED_BODY = r"(?P<q>[\"'])(?P<value>(?:\\.|(?!(?P=q))[^\\\r\n])*)(?P=q)"
_USER_FLAG = (
    r"(?P<head>(?<!\S){flag}(?:[\t ]+|=)?(?:(?P<q>[\"'])[^\s:\"']{{0,256}}:"
    r"|[^\s:\"']{{1,256}}:))"
)
_CURL_USER_QUOTED = re.compile(
    _USER_FLAG.format(flag="(?:-u|--user)")
    + r"(?P<value>(?:\\.|(?!(?P=q))[^\\\r\n])*)(?=(?P=q))"
)
_CURL_USER = re.compile(
    _USER_FLAG.format(flag="(?:-u|--user)").replace("(?P<q>", "(?P<noq>")
    + r"(?P<value>[^\s\"']+)"
)
_LONG_USER_QUOTED = re.compile(
    _USER_FLAG.format(flag="--user")
    + r"(?P<value>(?:\\.|(?!(?P=q))[^\\\r\n])*)(?=(?P=q))"
)
_LONG_USER = re.compile(
    _USER_FLAG.format(flag="--user").replace("(?P<q>", "(?P<noq>")
    + r"(?P<value>[^\s\"']+)"
)
_MYSQL_PASSWORD = re.compile(r"(?P<head>(?<!\S)-p)(?P<value>[^\s\"']+)")
_DOCKER_PASSWORD = re.compile(r"(?P<head>(?<!\S)-p[\t ]+)(?P<value>[^\s\"']+)")
_FLAG = re.compile(
    r"(?P<head>(?<!\S)--[A-Za-z0-9-]{0,48}(?:password|passwd|passphrase|token"
    r"|secret|api-?key|access-key|secret-key|private-key)(?:=|[\t ]+))"
    r"(?:" + _QUOTED_BODY.replace("value", "qvalue") + r"|(?P<value>[^\s\"']+))",
    re.IGNORECASE,
)
# Words that follow a flag *named* in a sentence rather than given a value.
_PROSE_AFTER_FLAG = frozenset(
    {
        "a",
        "an",
        "and",
        "arg",
        "argument",
        "as",
        "flag",
        "for",
        "from",
        "in",
        "is",
        "isn't",
        "of",
        "on",
        "option",
        "or",
        "parameter",
        "switch",
        "the",
        "to",
        "was",
        "when",
        "with",
    }
)
_AWS_CONFIGURE = re.compile(
    r"^(?P<head>[\t ]*AWS[\t ]+(?:Secret[\t ]+Access[\t ]+Key|Session[\t ]+Token)"
    r"[\t ]*(?:\[[^\]\r\n]{0,64}\])?[\t ]*:[\t ]*)(?P<value>\S+)",
    re.IGNORECASE | re.MULTILINE,
)
_QUOTED_ASSIGNMENT = re.compile(
    rf"(?P<head>(?:[:@]?[\"']?{_NAME}|[\"']auth)[\"']?{_SEPARATOR})" + _QUOTED_BODY,
    re.IGNORECASE,
)
_XML_ELEMENT = re.compile(
    rf"(?P<head><(?P<tag>{_NAME})>)(?P<value>[^<\r\n]+)(?=</(?P=tag)>)",
    re.IGNORECASE,
)
_XML_KEY_VALUE = re.compile(
    rf"(?P<head>\bkey=(?P<kq>[\"']){_NAME}(?P=kq)[\t ]+value=)" + _QUOTED_BODY,
    re.IGNORECASE,
)
_PARAMETER = re.compile(
    rf"(?P<head>(?<=[?&;]){_NAME}=)(?P<value>[^\s&;#\"']+)", re.IGNORECASE
)
_LINE_ASSIGNMENT = re.compile(
    r"^(?P<head>[\t ]*(?:(?:export|set|setx)[\t ]+|-[\t ]+)?"
    rf"(?P<name>{_NAME})(?P<sep>[\"']?[\t ]*(?:=>|[:=\uff1a\uff1d])"
    r"(?:[\t ]*\r?\n[\t ]+|[\t ]*)))(?P<value>\S+)",
    re.IGNORECASE | re.MULTILINE,
)
_SPACED_NAME = re.compile(
    r"(?P<head>(?<![\w-])(?:secret[\t ]+access[\t ]+key|access[\t ]+key"
    r"|secret[\t ]+key|api[\t ]+(?:key|token)|access[\t ]+token|refresh[\t ]+token"
    r"|auth[\t ]+token|session[\t ]+token|client[\t ]+secret|private[\t ]+key)"
    r"(?:[\t ]*[:=\uff1a\uff1d][\t ]*|[\t ]+(?:is[\t ]+)?))"
    r"(?P<value>[A-Za-z0-9._~+/=!@#$%^*?:|-]{1,1024})",
    re.IGNORECASE,
)
_CJK_NAME = re.compile(
    r"(?P<head>(?:密碼|密码|令牌|密鑰|密钥|口令)[\t ]*"
    r"(?:[:=\uff1a\uff1d]|是|為|为)[\t ]*)(?P<value>[!-~]+)"
)
_BEARER = re.compile(
    r"(?P<head>(?<![\w-])Bearer[\t ]+)(?P<value>[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
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
    r"|xoxe\.xox[a-z]-[A-Za-z0-9-]{10,}"  # Slack refreshable tokens
    r"|xox[a-z]-[A-Za-z0-9-]{10,}"  # Slack bot/user/app-level tokens
    r"|xapp-[A-Za-z0-9-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|pypi-[A-Za-z0-9_-]{40,}"
    r"|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"  # SendGrid
    r"|dp\.(?:st|ct|sa|scim|audit|pt)\.[A-Za-z0-9_.-]{20,}"  # Doppler
    r"|tvly-[A-Za-z0-9_-]{16,}"
    r"|ya29\.[A-Za-z0-9._-]{20,}"  # Google OAuth access token
    r"|AIza[A-Za-z0-9_-]{30,}"
    r"|SK[0-9a-fA-F]{32}(?![A-Za-z0-9])"  # Twilio API key
    r"|(?:AKIA|ASIA)(?![A-Z0-9]{9}EXAMPLE)[A-Z0-9]{16}"
    r")"
)
_PROSE_ASSIGNMENT = re.compile(
    rf"(?P<head>(?P<name>{_NAME})(?:[\"']?{_SEPARATOR}|[\t ]+is[\t ]+))"
    r"(?P<value>[A-Za-z0-9._~+/=!@#$%^*?:|-]{1,1024})",
    re.IGNORECASE,
)


def _masker(
    keep: Callable[[Match[str]], bool], tail: str = ""
) -> Callable[[Match[str]], str]:
    def mask(match: Match[str]) -> str:
        if keep(match):
            return match.group(0)
        return match.group("head") + REDACTED + tail

    return mask


def _quoted_masker(match: Match[str]) -> str:
    value = match.group("value")
    if _is_reference_or_placeholder(value):
        return match.group(0)
    quote = match.group("q")
    return f"{match.group('head')}{quote}{REDACTED}{quote}"


def _keep_authorization(match: Match[str]) -> bool:
    value = match.group("value")
    if _is_reference_or_placeholder(value):
        return True
    if match.group("scheme"):
        return bool(re.fullmatch(r"[a-z]+", value))  # "authorization: x check"
    return not _credential_shaped(value, 8)


def _keep_flag_value(match: Match[str]) -> bool:
    value = match.group("value")
    return value.startswith("-") or _is_reference_or_placeholder(value)


def _keep_user_password(match: Match[str]) -> bool:
    value = match.group("value")
    return _is_reference_or_placeholder(value) or bool(_NUMBER.fullmatch(value))


def _mask_quoted_flag(match: Match[str]) -> str:
    if match.group("qvalue") is not None:
        if _is_reference_or_placeholder(match.group("qvalue")):
            return match.group(0)
        quote = match.group("q")
        return f"{match.group('head')}{quote}{REDACTED}{quote}"
    if _keep_flag_value(match):
        return match.group(0)
    if not match.group("head").endswith("=") and (
        match.group("value").lower() in _PROSE_AFTER_FLAG
    ):
        return match.group(0)  # "the --password option is deprecated"
    return match.group("head") + REDACTED


def _mask_line_assignment(match: Match[str]) -> str:
    name, sep, value = match.group("name"), match.group("sep"), match.group("value")
    if name.lower().endswith("authorization"):
        return match.group(0)  # the Authorization rule owns it
    if (
        _is_reference_or_placeholder(value)
        or _is_literal_data(value, name)
        or not re.search(r"[A-Za-z0-9]", value)
    ):
        return match.group(0)
    spaced_equals = re.fullmatch(r"[\t ]+=[\t ]+", sep) is not None
    text, end = match.string, match.end()
    line_end = text.find("\n", end, end + _MAX_EXPRESSION)
    if line_end < 0:
        line_end = end + _MAX_EXPRESSION
    expression = value + text[end:line_end].rstrip("\r")
    if (":" in sep or "\uff1a" in sep or spaced_equals) and (
        _is_code(value) or _is_code(expression)
    ):
        return match.group(0)  # api_key: Optional[str], TOKEN = os.getenv(...)
    if spaced_equals and name == name.lower() and _LOWER_IDENTIFIER.fullmatch(value):
        return match.group(0)  # Python: access_token = oauth2_token
    following = match.string[match.end() : match.end() + 64]
    if (
        (":" in sep or "\uff1a" in sep)
        and _ALPHA_WORD.fullmatch(value)
        and re.match(r"[\t ]+\S", following)
    ):
        return match.group(0)  # prose: "Token: what is a JWT?"
    return match.group("head") + REDACTED


def _keep_prose(match: Match[str], min_length: int = 6) -> bool:
    value = match.group("value")
    name = match.groupdict().get("name") or match.group("head")
    if name.lower().endswith("authorization"):
        return True
    following = match.string[match.end() : match.end() + 1]
    return (
        following in ("(", "[")
        or not _credential_shaped(value, min_length)
        or _is_reference_or_placeholder(value)
        or _is_literal_data(value, name)
    )


def _keep_bearer(match: Match[str]) -> bool:
    value = match.group("value")
    return (
        not _credential_shaped(value, 8)
        or bool(_SLUG.fullmatch(value))
        or _is_reference_or_placeholder(value)
    )


def _mask_cookie_pairs(text: str, start: int) -> Tuple[str, int]:
    parts: List[str] = []
    position = start
    while True:
        pair = _COOKIE_PAIR.match(text, position)
        if not pair:
            break
        value = pair.group("value")
        if pair.group("name").lower() in _COOKIE_ATTRIBUTES or (
            _is_reference_or_placeholder(value)
        ):
            parts.append(pair.group(0))
        else:
            parts.append(f"{pair.group('name')}={REDACTED}{pair.group('sep') or ''}")
        position = pair.end()
        if not pair.group("sep"):
            break
    return "".join(parts), position


def _mask_after(text: str, starts: Pattern[str]) -> str:
    parts: List[str] = []
    position = 0
    for start in starts.finditer(text):
        if start.start() < position:
            continue
        masked, end = _mask_cookie_pairs(text, start.end())
        parts.append(text[position : start.end()])
        parts.append(masked)
        position = max(end, start.end())
    parts.append(text[position:])
    return "".join(parts)


def _mask_command(match: Match[str]) -> str:
    command = match.group("command").lower()
    line = match.group(0)
    if command == "curl":
        line = _CURL_USER_QUOTED.sub(_masker(_keep_user_password), line)
        line = _CURL_USER.sub(_masker(_keep_user_password), line)
        return _mask_after(line, _CURL_COOKIE_START)
    if command.startswith("docker"):
        return _DOCKER_PASSWORD.sub(_masker(_keep_flag_value), line)
    return _MYSQL_PASSWORD.sub(_masker(_keep_flag_value), line)


def _keep_parameter(match: Match[str]) -> bool:
    value = match.group("value")
    name = match.group("head")
    return _is_reference_or_placeholder(value) or (
        _NUMBER.fullmatch(value) is not None and not _PASSWORD_NAME.search(name)
    )


def _keep_structured(match: Match[str]) -> bool:
    return _is_reference_or_placeholder(match.group("value"))


_STEPS: Tuple[Callable[[str], str], ...] = (
    _mask_private_keys,
    lambda text: _PUTTY.sub(_mask_putty, text),
    lambda text: _AWS_KEY_PAIR.sub(
        lambda m: REDACTED + m.group("sep") + REDACTED, text
    ),
    lambda text: _URL_PASSWORD.sub(_masker(_keep_structured, "@"), text),
    lambda text: _WEBHOOK.sub(lambda m: m.group("head") + REDACTED, text),
    lambda text: _AUTHORIZATION.sub(_masker(_keep_authorization), text),
    lambda text: _SIGNATURE.sub(_masker(_keep_structured), text),
    lambda text: _mask_after(text, _COOKIE_START),
    lambda text: _COMMAND.sub(_mask_command, text),
    lambda text: _LONG_USER_QUOTED.sub(_masker(_keep_user_password), text),
    lambda text: _LONG_USER.sub(_masker(_keep_user_password), text),
    lambda text: _FLAG.sub(_mask_quoted_flag, text),
    lambda text: _AWS_CONFIGURE.sub(_masker(_keep_structured), text),
    lambda text: _XML_KEY_VALUE.sub(_quoted_masker, text),
    lambda text: _QUOTED_ASSIGNMENT.sub(_quoted_masker, text),
    lambda text: _XML_ELEMENT.sub(_masker(_keep_structured), text),
    lambda text: _PARAMETER.sub(_masker(_keep_parameter), text),
    lambda text: _LINE_ASSIGNMENT.sub(_mask_line_assignment, text),
    lambda text: _CJK_NAME.sub(_masker(_keep_structured), text),
    lambda text: _SPACED_NAME.sub(
        _masker(lambda m: _keep_prose(m, min_length=8)), text
    ),
    lambda text: _BEARER.sub(_masker(_keep_bearer), text),
    lambda text: _JWT.sub(REDACTED, text),
    lambda text: _PROVIDER_KEY.sub(REDACTED, text),
    lambda text: _PROSE_ASSIGNMENT.sub(_masker(_keep_prose), text),
)


def is_credential_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    return bool(_CREDENTIAL_KEY.search(_CAMEL_BOUNDARY.sub("_", key)))


def redact_text(text: str) -> str:
    """Return `text` with credentials replaced by `[REDACTED]`."""
    if not text:
        return text
    for step in _STEPS:
        text = step(text)
    return text


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
