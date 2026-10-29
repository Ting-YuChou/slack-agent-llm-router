"""Download Slack-hosted files with the bot token, and nothing else.

Shared by the chat bot's attachments and the agent bootstrap. The bot token
only ever goes to an https Slack file host: redirects are followed by hand and
every hop is checked again, and the body is streamed and dropped as soon as it
passes the byte cap, whatever size Slack declared.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

import httpx

SLACK_FILE_HOSTS = frozenset({"files.slack.com"})
# Modes whose bytes Slack does not host or no longer serves.
WITHHELD_FILE_MODES = frozenset({"external", "tombstone", "hidden_by_limit"})
MAX_DOWNLOAD_HOPS = 4
# Slack-native documents (canvases), as the agent bootstrap recognises them.
SLACK_DOCUMENT_MIME_TYPES = frozenset({"application/vnd.slack-docs"})
SLACK_DOCUMENT_FILETYPES = frozenset({"canvas", "quip"})


class SlackFileDownloadError(ValueError):
    """A download refused for a reason named by its message."""


def is_allowed_slack_file_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in SLACK_FILE_HOSTS


def is_slack_hosted_file(file_payload: Mapping[str, Any]) -> bool:
    """False for remote files and files whose content Slack withholds."""
    if file_payload.get("is_external"):
        return False
    return str(file_payload.get("mode") or "") not in WITHHELD_FILE_MODES


def declares_html(file_payload: Mapping[str, Any]) -> bool:
    """True when an HTML body is the file's own content, not a sign-in page.

    That covers HTML files and Slack's native documents (canvases), which
    Slack serves as HTML.
    """
    mime_type = str(file_payload.get("mimetype") or "").lower()
    filetype = str(file_payload.get("filetype") or "").lower()
    name = str(file_payload.get("name") or "").lower()
    return (
        "html" in mime_type
        or mime_type in SLACK_DOCUMENT_MIME_TYPES
        or filetype in {"html", "htm"} | SLACK_DOCUMENT_FILETYPES
        or name.endswith((".html", ".htm"))
    )


async def download_slack_file(
    client: httpx.AsyncClient,
    url: str,
    *,
    bot_token: str,
    max_bytes: int,
    allow_html: bool = True,
) -> bytes:
    """Return the file body or raise `SlackFileDownloadError`/`httpx.HTTPError`.

    With `allow_html=False` an HTML body is refused: Slack answers a request it
    will not serve with its sign-in page.
    """
    current_url = url
    for hop in range(MAX_DOWNLOAD_HOPS):
        if not is_allowed_slack_file_url(current_url):
            raise SlackFileDownloadError(
                "unsafe_redirect" if hop else "unsafe_download_url"
            )
        async with client.stream(
            "GET",
            current_url,
            headers={"Authorization": f"Bearer {bot_token}"},
            follow_redirects=False,
        ) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise SlackFileDownloadError("redirect_without_location")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if not allow_html and content_type.startswith("text/html"):
                raise SlackFileDownloadError("unexpected_html")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > max_bytes:
                    raise SlackFileDownloadError("resource_too_large")
                content.extend(chunk)
            return bytes(content)
    raise SlackFileDownloadError("too_many_redirects")


def error_name(exc: BaseException) -> str:
    """A short, URL-free name for `exc` that is safe to log or show."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    response = getattr(exc, "response", None)
    data = getattr(response, "data", None)
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    message = str(exc).strip()
    if re.fullmatch(r"[a-z][a-z0-9_:-]{0,79}", message):
        return message
    return exc.__class__.__name__.lower()
