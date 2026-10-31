"""Slack conversations: per-thread history, query traces, active threads
and the SlackBot mixin that fetches Slack history as context."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)
from dataclasses import dataclass
from slack_sdk.web.async_client import AsyncWebClient

from slack.agent_bootstrap import AgentBootstrapBuilder
from slack.thread_history import (
    DEFAULT_MAX_THREAD_PAGES,
    MAX_THREAD_PAGE_LIMIT,
    bounded_join_newest,
    fetch_newest_thread_messages,
    raise_for_slack_error,
)
from src.prompt_assembly import protect_untrusted_boundary
from src.utils.schema import (
    UserTier,
    is_valid_request_id,
)
from slack.formatting import (
    _is_slack_notice_text,
    slack_chat_request_id,
)
from slack.state_store import (
    SLACK_SESSION_TIMEOUT,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


# Message subtypes a person's follow-up can carry: a plain message, one with
# files, and a reply also sent to the channel. Edits, deletions, joins and the
# like are never a new request. The same set the Agent thread context reads.
CHAT_MESSAGE_SUBTYPES = frozenset(AgentBootstrapBuilder._INCLUDED_SUBTYPES)
# How the bot's own messages appear in the Slack history a request sees.
SLACK_OWN_MESSAGE_SENDER = "assistant (you)"
# Fence name for fetched Slack history: `<untrusted_slack_history>`.
SLACK_HISTORY_FENCE = "slack_history"
# Most characters of per-user conversation summary a request carries.
CONVERSATION_SUMMARY_MAX_CHARS = 8000


@dataclass(frozen=True)
class SlackQueryTrace:
    """What ties a Slack chat query to its logs, events and feedback."""

    request_id: Optional[str] = None
    # mention | thread | dm | slash
    surface: Optional[str] = None
    event_id: Optional[str] = None
    # How long the event waited in the bot's work queue before a worker
    # took it (ms); inference adds its own scheduler wait to it.
    queue_wait_ms: Optional[int] = None

    @classmethod
    def for_message(
        cls, event: Dict[str, Any], *, team_id: Optional[str]
    ) -> "SlackQueryTrace":
        thread_ts = event.get("thread_ts")
        if event.get("channel_type") == "im":
            surface = "dm"
        elif thread_ts and thread_ts != event.get("ts"):
            # A reply in a thread, mentioning the bot or not.
            surface = "thread"
        else:
            surface = "mention"
        event_id = event.get("_event_id")
        return cls(
            request_id=slack_chat_request_id(
                team_id, event.get("channel"), event.get("ts")
            ),
            surface=surface,
            event_id=event_id if isinstance(event_id, str) else None,
            queue_wait_ms=_queue_wait_ms(event.get(SLACK_QUEUE_WAIT_KEY)),
        )

    @classmethod
    def for_slash(
        cls, request_id: Optional[str], *, queue_wait_ms: Any = None
    ) -> "SlackQueryTrace":
        return cls(
            request_id=request_id if is_valid_request_id(request_id) else None,
            surface="slash",
            queue_wait_ms=_queue_wait_ms(queue_wait_ms),
        )

    def add_metadata(
        self, metadata: Dict[str, Any], *, message_ts: Optional[str]
    ) -> None:
        if self.surface:
            metadata["surface"] = self.surface
        if self.queue_wait_ms is not None:
            metadata["ingress_queue_wait_ms"] = self.queue_wait_ms
        slack = metadata.get("slack")
        if isinstance(slack, dict):
            slack["message_ts"] = message_ts
            slack["event_id"] = self.event_id


# The event key a worker stamps with the item's queue wait at pickup.
SLACK_QUEUE_WAIT_KEY = "_ingress_queue_wait_ms"


def _queue_wait_ms(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


@dataclass
class ConversationContext:
    """Context for maintaining conversation continuity"""

    user_id: str
    channel_id: str
    thread_ts: Optional[str]
    conversation_history: List[Dict[str, str]]
    user_tier: UserTier
    preferences: Dict[str, Any]
    last_activity: datetime
    session_id: str

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

        # Keep only last 20 messages for context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        self.last_activity = datetime.now()


@dataclass
class SlackFetchedContext:
    """Request-scoped Slack channel/thread context fetched from Slack history."""

    context: str = ""
    source: str = "none"
    message_count: int = 0
    truncated: bool = False
    error: Optional[str] = None


class ConversationManager:
    """Manages conversation contexts and history"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversations = {}
        self.session_timeout = SLACK_SESSION_TIMEOUT

    def get_or_create_context(
        self, user_id: str, channel_id: str, thread_ts: str = None
    ) -> ConversationContext:
        """Get existing conversation context or create new one"""
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"

        if context_key in self.conversations:
            context = self.conversations[context_key]
            # Check if session is still valid
            if datetime.now() - context.last_activity < self.session_timeout:
                return context
            else:
                # Session expired, create new one
                del self.conversations[context_key]

        # Create new context
        context = ConversationContext(
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            conversation_history=[],
            user_tier=UserTier.FREE,  # Will be updated by UserManager
            preferences={},
            last_activity=datetime.now(),
            session_id=str(uuid.uuid4()),
        )

        self.conversations[context_key] = context
        return context

    def cleanup_expired_sessions(self, return_keys: bool = False):
        """Clean up expired conversation sessions"""
        current_time = datetime.now()
        expired_keys = []

        for key, context in self.conversations.items():
            if current_time - context.last_activity > self.session_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self.conversations[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired conversation sessions")

        if return_keys:
            return expired_keys
        return len(expired_keys)

    def clear_contexts(self, user_id: str, channel_id: str, return_keys: bool = False):
        """Clear all contexts for a user in a channel."""
        prefix = f"{user_id}:{channel_id}:"
        matching_keys = [key for key in self.conversations if key.startswith(prefix)]

        for key in matching_keys:
            del self.conversations[key]

        if return_keys:
            return matching_keys
        return len(matching_keys)

    def get_conversation_summary(
        self, user_id: str, channel_id: str, thread_ts: str = None
    ) -> str:
        """Get this user's recent turns with the bot, one line per turn.

        Only for requests without a Slack transcript. Whitespace collapses
        to single spaces, so no turn can forge a line of its own; user turns
        keep their first 200 characters and answers stay whole, within an
        overall budget filled from the newest turn back.
        """
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"
        context = self.conversations.get(context_key)

        if not context or not context.conversation_history:
            return ""

        summary_parts = []
        for msg in context.conversation_history[-10:]:
            role = msg["role"]
            content = " ".join(str(msg["content"] or "").split())
            if role != "assistant":
                content = content[:200]
            summary_parts.append(f"{role}: {content}")

        summary, _, _ = bounded_join_newest(
            summary_parts, CONVERSATION_SUMMARY_MAX_CHARS
        )
        return summary

    def get_last_answer(
        self, user_id: str, channel_id: str, thread_ts: str = None
    ) -> str:
        """This user's last answer from the bot, whitespace-normalized."""
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"
        context = self.conversations.get(context_key)
        for msg in reversed(getattr(context, "conversation_history", None) or []):
            if msg.get("role") == "assistant":
                return " ".join(str(msg.get("content") or "").split())[
                    :CONVERSATION_SUMMARY_MAX_CHARS
                ]
        return ""


class SlackBotConversationMixin:
    """SlackBot's Slack history context and active threads."""

    def _resolve_context_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve Slack history context settings with backwards-compatible defaults."""
        settings = {
            "enabled": True,
            "strategy": "thread_first",
            "max_thread_messages": 20,
            "max_channel_messages": 10,
            "max_context_chars": 4000,
            "include_bot_messages": False,
            "timeout_seconds": 3.0,
            "fail_open": True,
        }
        if isinstance(config, dict):
            settings.update(config)

        settings["strategy"] = str(settings.get("strategy") or "thread_first").lower()
        settings["max_thread_messages"] = max(
            1, int(settings.get("max_thread_messages") or 20)
        )
        settings["max_channel_messages"] = max(
            1, int(settings.get("max_channel_messages") or 10)
        )
        settings["max_context_chars"] = max(
            1, int(settings.get("max_context_chars") or 4000)
        )
        settings["timeout_seconds"] = max(
            0.1, float(settings.get("timeout_seconds") or 3.0)
        )
        settings["enabled"] = bool(settings.get("enabled", True))
        settings["include_bot_messages"] = bool(
            settings.get("include_bot_messages", False)
        )
        settings["fail_open"] = bool(settings.get("fail_open", True))
        return settings

    async def _build_slack_context(
        self,
        client: AsyncWebClient,
        channel_id: Optional[str],
        thread_ts: Optional[str],
        message_ts: Optional[str],
    ) -> SlackFetchedContext:
        """Fetch request-scoped Slack thread/channel context for the current query."""
        if not self.context_settings.get("enabled", True):
            return SlackFetchedContext()
        if not client or not channel_id:
            return SlackFetchedContext()

        # The deadline applies per Slack call, so a slow later page ends the
        # paging and keeps the pages already fetched instead of all of them.
        deadline = asyncio.get_running_loop().time() + float(
            self.context_settings.get("timeout_seconds", 3.0)
        )
        try:
            return await self._fetch_slack_context(
                client, channel_id, thread_ts, message_ts, deadline=deadline
            )
        except Exception as exc:
            if not self.context_settings.get("fail_open", True):
                raise
            error_name = self._slack_context_error_name(exc)
            logger.warning("Slack context fetch failed: %s", error_name)
            return SlackFetchedContext(error=error_name)

    async def _fetch_slack_context(
        self,
        client: AsyncWebClient,
        channel_id: str,
        thread_ts: Optional[str],
        message_ts: Optional[str],
        deadline: Optional[float] = None,
    ) -> SlackFetchedContext:
        """Fetch Slack history using thread context first, then channel context."""

        async def send(request: Callable[[], Awaitable[Any]]) -> Any:
            if deadline is None:
                return await request()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            return await asyncio.wait_for(request(), timeout=remaining)

        strategy = self.context_settings.get("strategy", "thread_first")
        if strategy != "thread_first":
            raise ValueError(f"Unsupported Slack context strategy: {strategy}")

        use_thread = bool(thread_ts and (not message_ts or thread_ts != message_ts))
        if use_thread:
            method_name = "conversations_replies"
            method = getattr(client, method_name, None)
            if not callable(method):
                raise RuntimeError(f"{method_name}_unavailable")

            max_messages = int(self.context_settings.get("max_thread_messages", 20))
            kwargs: Dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": min(max(max_messages, 1), MAX_THREAD_PAGE_LIMIT),
            }
            if message_ts:
                kwargs["latest"] = message_ts
                kwargs["inclusive"] = False
            window = await fetch_newest_thread_messages(
                method,
                kwargs,
                max_messages=max_messages,
                eligible=lambda message: self._should_include_slack_context_message(
                    message, message_ts
                ),
                max_pages=int(
                    self.context_settings.get(
                        "max_thread_pages", DEFAULT_MAX_THREAD_PAGES
                    )
                ),
                send=send,
                next_page_limit=MAX_THREAD_PAGE_LIMIT,
                keep_parent=True,
            )
            fetched = self._format_slack_context(
                window.messages, "thread", parent_ts=thread_ts
            )
            fetched.truncated = fetched.truncated or window.truncated
            return fetched

        if not message_ts:
            return SlackFetchedContext()
        method_name = "conversations_history"
        method = getattr(client, method_name, None)
        if not callable(method):
            raise RuntimeError(f"{method_name}_unavailable")

        response = await send(
            lambda: method(
                channel=channel_id,
                latest=message_ts,
                inclusive=False,
                limit=int(self.context_settings.get("max_channel_messages", 10)),
            )
        )
        raise_for_slack_error(response)

        messages = list((response or {}).get("messages", []) or [])
        messages.sort(key=lambda message: self._slack_message_sort_key(message))
        filtered_messages = [
            message
            for message in messages
            if self._should_include_slack_context_message(message, message_ts)
        ]
        return self._format_slack_context(filtered_messages, "channel")

    def _is_own_slack_message(self, message: Dict[str, Any]) -> bool:
        """Whether this bot posted the message, e.g. one of its own answers."""
        return bool(self.bot_user_id) and message.get("user") == self.bot_user_id

    def _should_include_slack_context_message(
        self, message: Dict[str, Any], current_ts: Optional[str]
    ) -> bool:
        """Filter Slack system/other-bot/current-query messages out of context.

        The bot's own answers stay: a follow-up from anyone in the thread
        needs them. Other bots follow `include_bot_messages`.
        """
        if self._same_slack_timestamp(message.get("ts"), current_ts):
            return False
        if message.get("subtype") not in CHAT_MESSAGE_SUBTYPES:
            return False
        if self._is_own_slack_message(message):
            if _is_slack_notice_text(message.get("text")):
                return False
        elif not self.context_settings.get("include_bot_messages", False):
            if message.get("bot_id"):
                return False
        if not self._slack_context_message_text(message):
            return False
        return bool(self._slack_message_sender(message))

    def _slack_context_message_text(self, message: Dict[str, Any]) -> str:
        """A message's text, or the names of the files it shared without any."""
        text = self._clean_slack_message_text(message.get("text", ""))
        if text:
            return text
        names = [
            self._clean_slack_message_text(item.get("name") or item.get("title"))
            for item in (message.get("files") or [])
            if isinstance(item, dict)
        ]
        names = [name for name in names if name]
        return f"[shared file: {', '.join(names)}]" if names else ""

    def _format_slack_context(
        self,
        messages: List[Dict[str, Any]],
        source: str,
        parent_ts: Optional[str] = None,
    ) -> SlackFetchedContext:
        """Render Slack messages into a compact deterministic context block.

        The budget fills from the newest message back, so a long history
        loses its oldest messages, and the block keeps chronological order.
        The thread parent (`parent_ts`), usually the opening question, is
        always kept, cut to a third of the budget at most.
        """
        if not messages:
            return SlackFetchedContext(source=source)

        header = (
            "Slack thread context:" if source == "thread" else "Recent channel context:"
        )
        # Other people's messages are data for the model, never instructions:
        # fenced, with any look-alike of the fence's tags escaped.
        candidate_lines = [
            protect_untrusted_boundary(
                f"{self._slack_context_sender(message)}: "
                f"{self._slack_context_message_text(message)}"
            )
            for message in messages
        ]
        opening = f"<untrusted_{SLACK_HISTORY_FENCE}>"
        closing = f"</untrusted_{SLACK_HISTORY_FENCE}>"

        max_chars = int(self.context_settings.get("max_context_chars", 4000))
        budget = max_chars - len(header) - len(opening) - len(closing) - 3
        parent_lines: List[str] = []
        parent_truncated = False
        if parent_ts is not None and self._same_slack_timestamp(
            messages[0].get("ts"), parent_ts
        ):
            parent_line = candidate_lines.pop(0)
            parent_limit = max(budget // 3, 0)
            # Only a thread that does not fit cuts its parent.
            whole_length = len(parent_line) + sum(
                len(line) + 1 for line in candidate_lines
            )
            if whole_length > budget and len(parent_line) > parent_limit:
                parent_line = parent_line[: max(parent_limit - 3, 0)].rstrip() + "..."
                parent_truncated = True
            parent_lines.append(parent_line)
            budget -= len(parent_line) + 1
        body, included_count, truncated = bounded_join_newest(candidate_lines, budget)
        truncated = truncated or parent_truncated
        if parent_lines:
            body = "\n".join([*parent_lines, body]) if body else parent_lines[0]
            included_count += 1
            candidate_lines.insert(0, parent_lines[0])
        if included_count < len(candidate_lines):
            truncated = True
        if included_count == 0:
            return SlackFetchedContext(source=source, truncated=True)

        return SlackFetchedContext(
            context=f"{header}\n{opening}\n{body}\n{closing}",
            source=source,
            message_count=included_count,
            truncated=truncated,
        )

    def _slack_context_sender(self, message: Dict[str, Any]) -> str:
        if self._is_own_slack_message(message):
            return SLACK_OWN_MESSAGE_SENDER
        return self._slack_message_sender(message)

    def _clean_slack_message_text(self, text: Any) -> str:
        """Normalize Slack text without resolving user IDs or making extra API calls."""
        return " ".join(str(text or "").split())

    def _slack_message_sender(self, message: Dict[str, Any]) -> str:
        user_id = message.get("user")
        if user_id:
            return f"<@{user_id}>"
        username = message.get("username")
        if username:
            return str(username)
        return ""

    def _same_slack_timestamp(self, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return False
        return str(left) == str(right)

    def _slack_message_sort_key(self, message: Dict[str, Any]) -> float:
        try:
            return float(message.get("ts", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _slack_context_error_name(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if isinstance(response, dict) and response.get("error"):
            return str(response["error"])
        return str(exc) or type(exc).__name__

    def _mark_thread_active(self, channel_id: Optional[str], thread_ts: Optional[str]):
        """Mark a bot thread as active for follow-up replies."""
        if not channel_id or not thread_ts:
            return
        self.active_threads[self._thread_key(channel_id, thread_ts)] = datetime.now()

    def _is_active_thread(
        self,
        channel_id: Optional[str],
        thread_ts: Optional[str],
        *,
        refresh: bool = True,
    ) -> bool:
        """Return whether a Slack thread is still active.

        By default this also extends a live thread and forgets an expired one;
        refresh=False only looks.
        """
        if not channel_id or not thread_ts:
            return False

        key = self._thread_key(channel_id, thread_ts)
        last_activity = self.active_threads.get(key)
        if not last_activity:
            return False

        if datetime.now() - last_activity > self.active_thread_timeout:
            if refresh:
                self.active_threads.pop(key, None)
            return False

        if refresh:
            self.active_threads[key] = datetime.now()
        return True

    def _cleanup_expired_threads(self, return_keys: bool = False):
        """Drop inactive thread markers after the configured timeout."""
        now = datetime.now()
        expired_keys = [
            key
            for key, last_activity in self.active_threads.items()
            if now - last_activity > self.active_thread_timeout
        ]

        for key in expired_keys:
            del self.active_threads[key]

        if return_keys:
            return expired_keys
        return len(expired_keys)

    def clear_active_threads(
        self, channel_id: Optional[str] = None, return_keys: bool = False
    ):
        """Clear tracked active threads, optionally limited to one channel."""
        if channel_id is None:
            cleared_keys = list(self.active_threads.keys())
            self.active_threads.clear()
            if return_keys:
                return cleared_keys
            return len(cleared_keys)

        prefix = f"{channel_id}:"
        matching_keys = [key for key in self.active_threads if key.startswith(prefix)]
        for key in matching_keys:
            del self.active_threads[key]
        if return_keys:
            return matching_keys
        return len(matching_keys)
