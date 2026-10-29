"""Read the newest part of a Slack thread within a message and page budget.

`conversations.replies` pages from the parent and the oldest replies
forward, so a single page of a long thread is its oldest part. These helpers
page through it, keep only the newest eligible messages and fill a character
budget from the newest message back. Chat context and the agent bootstrap
share them.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

DEFAULT_MAX_THREAD_PAGES = 10
# conversations.replies accepts at most this many messages per page.
MAX_THREAD_PAGE_LIMIT = 100

PageRequest = Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]]


@dataclass
class ThreadWindow:
    """The newest eligible messages of a thread, oldest first."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    eligible_count: int = 0
    pages: int = 0
    # Pagination stopped before the thread's end (cap, missing cursor, timeout).
    incomplete: bool = False
    page_cap_reached: bool = False
    # The timeout that ended pagination after the first page, if any.
    stopped_by: Optional[BaseException] = None

    @property
    def truncated(self) -> bool:
        return self.incomplete or self.eligible_count > len(self.messages)


def raise_for_slack_error(response: Any) -> None:
    if isinstance(response, dict) and response.get("ok") is False:
        raise RuntimeError(str(response.get("error") or "slack_api_error"))


def timestamp_key(value: Any) -> Tuple[int, int]:
    try:
        seconds, fraction = str(value).split(".", 1)
    except ValueError:
        seconds, fraction = str(value or "0"), "0"
    try:
        return int(seconds), int(fraction)
    except ValueError:
        return 0, 0


async def _direct(request: Callable[[], Awaitable[Any]]) -> Any:
    return await request()


async def fetch_newest_thread_messages(
    method: Callable[..., Awaitable[Any]],
    kwargs: Dict[str, Any],
    *,
    max_messages: int,
    eligible: Callable[[Dict[str, Any]], bool],
    max_pages: int = DEFAULT_MAX_THREAD_PAGES,
    send: PageRequest = _direct,
    next_page_limit: Optional[int] = None,
    keep_parent: bool = False,
) -> ThreadWindow:
    """Page `conversations.replies` and keep its newest `max_messages`.

    The first call uses `kwargs` as given; later calls add `cursor` and, when
    `next_page_limit` is set, ask for that many messages per page instead.
    `send` wraps each call, e.g. in a timeout. A timeout on the first page
    propagates; a later one ends pagination and is kept in `stopped_by`.
    Messages repeated across pages (Slack can resend the parent) count once.
    With `keep_parent`, the thread parent (`ts == kwargs["ts"]`) stays in the
    result even when newer messages fill the window.
    """
    max_messages = max(max_messages, 1)
    max_pages = max(max_pages, 1)
    window: deque[Dict[str, Any]] = deque(maxlen=max_messages)
    result = ThreadWindow()
    cursor: Optional[str] = None
    seen_ts: Set[str] = set()
    parent_ts = str(kwargs.get("ts")) if keep_parent and kwargs.get("ts") else None
    parent: Optional[Dict[str, Any]] = None
    while True:
        page_kwargs = dict(kwargs)
        if cursor:
            page_kwargs["cursor"] = cursor
            if next_page_limit:
                page_kwargs["limit"] = next_page_limit
        try:
            response = await send(lambda: method(**page_kwargs))
        except asyncio.TimeoutError as exc:
            if not result.pages:
                raise
            result.incomplete = True
            result.stopped_by = exc
            break
        result.pages += 1
        raise_for_slack_error(response)
        page_messages = list((response or {}).get("messages", []) or [])
        eligible_messages = []
        for message in page_messages:
            if not isinstance(message, dict):
                continue
            ts = message.get("ts")
            if ts is not None:
                if str(ts) in seen_ts:
                    continue
                seen_ts.add(str(ts))
            if not eligible(message):
                continue
            if parent_ts is not None and str(ts) == parent_ts:
                parent = message
            eligible_messages.append(message)
        result.eligible_count += len(eligible_messages)
        window.extend(eligible_messages)
        metadata = (response or {}).get("response_metadata", {}) or {}
        next_cursor = str(metadata.get("next_cursor") or "")
        has_more = bool((response or {}).get("has_more") or next_cursor)
        if not has_more:
            break
        if not next_cursor:
            result.incomplete = True
            break
        if result.pages >= max_pages:
            result.incomplete = True
            result.page_cap_reached = True
            break
        cursor = next_cursor

    kept = list(window)
    if parent is not None and not any(message is parent for message in kept):
        kept.append(parent)
    result.messages = sorted(kept, key=lambda message: timestamp_key(message.get("ts")))
    return result


def bounded_join_newest(parts: List[str], limit: int) -> Tuple[str, int, bool]:
    """Fill the budget from the newest part back, then restore order.

    Returns the joined text, how many parts it includes (a partially kept
    part counts) and whether anything was dropped or cut.
    """
    kept: List[str] = []
    used = 0
    truncated = False
    for raw_part in reversed(parts):
        part = str(raw_part).strip()
        if not part:
            continue
        separator = 1 if kept else 0
        if used + separator + len(part) <= limit:
            kept.append(part)
            used += separator + len(part)
            continue
        remaining = limit - used - separator
        if remaining > 3:
            kept.append(part[: remaining - 3].rstrip() + "...")
        truncated = True
        break
    kept.reverse()
    return "\n".join(kept), len(kept), truncated
