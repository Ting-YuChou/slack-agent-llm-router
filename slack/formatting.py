"""Slack reply text and shaping: notices, error texts, answers and the
SlackBot mixin that posts replies (placeholder, split, fallback)."""

import logging
import math
import re
import uuid
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)

from src.utils.schema import (
    UserTier,
    is_valid_request_id,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


MEMORY_SLASH_ONLY_MESSAGE = (
    "Memory commands are only available through Slack slash commands. "
    "Use `/llm remember`, `/llm memories`, or `/llm forget`."
)
SLACK_BUSY_MESSAGE = "The LLM Router is busy right now. Please retry in a moment."
SLACK_SLASH_WORKING_TEXT = "Working on it…"
SLACK_EMPTY_ANSWER_MESSAGE = (
    "🤔 The model returned an empty answer. Please try again or rephrase "
    "your question."
)
_RATE_LIMITED_TEXT = (
    "⏳ The model provider is rate-limiting requests right now. "
    "Please try again in a minute."
)
_TIMEOUT_TEXT = (
    "⌛ The model took too long to answer. Please try again, or ask for "
    "something shorter."
)
_TOO_LONG_TEXT = (
    "❌ That request is too long for the model. Try shortening it or splitting "
    "it into smaller requests."
)
# What a failed answer tells the user. A precise InferenceResponse.error_code
# wins; else its error_type decides; anything else gets the generic message,
# which reveals nothing internal. A rate_limited error_code is the provider's
# 429 only when error_type says so too; otherwise our own capacity turned the
# request away (an admission rejection), and the router is busy.
SLACK_ERROR_CODE_MESSAGES = {
    "context_length_exceeded": _TOO_LONG_TEXT,
    "prompt_too_large": _TOO_LONG_TEXT,
    "model_not_entitled": (
        "❌ The model this request needs is not available for your tier."
    ),
    "retrieval_unavailable": (
        "⚠️ Document search is unavailable right now. Please try again in a "
        "few minutes."
    ),
    "provider_unavailable": (
        "⚠️ The model provider is unavailable right now. Please try again in a "
        "few minutes."
    ),
    "timeout": _TIMEOUT_TEXT,
}
SLACK_INFERENCE_ERROR_MESSAGES = {
    "rate_limited": _RATE_LIMITED_TEXT,
    "timeout": _TIMEOUT_TEXT,
    # Every ValueError lands here, internal ones too: blame nothing.
    "user_or_policy": (
        "❌ I couldn't complete that request. Rephrasing it may help; "
        "otherwise please try again later."
    ),
}
# A note appended to an answer that came back degraded, by response warning
# (its part before any ":detail"). Other warnings add nothing.
SLACK_RESPONSE_WARNING_NOTES = {
    "rag_unavailable": (
        "_Note: document search was unavailable, so this answer isn't based on "
        "your documents._"
    ),
    "rag_degraded": (
        "_Note: document search only partly worked, so this answer may miss "
        "parts of your documents._"
    ),
    "web_search_unavailable": (
        "_Note: live web search was unavailable; recent information couldn't "
        "be checked._"
    ),
    "response_truncated": "_Note: the answer was cut off._",
    "response_incomplete": "_Note: the answer may be incomplete._",
    "response_filtered": (
        "_Note: part of the answer was withheld by the model's content filter._"
    ),
    "empty_response": (
        "_Note: the model returned almost no text, so the answer may be incomplete._"
    ),
}
# The same, as a plain-text label that leads the answer in history: the
# conversation summary keeps only each message's first 200 characters.
SLACK_RESPONSE_WARNING_LABELS = {
    "rag_unavailable": "document search was unavailable",
    "rag_degraded": "document search only partly worked",
    "web_search_unavailable": "live web search was unavailable",
    "response_truncated": "answer was cut off",
    "response_incomplete": "answer may be incomplete",
    "response_filtered": "part of the answer was withheld by the content filter",
    "empty_response": "model returned almost no text",
}
# The progress placeholder the bot posts before an answer replaces it. Never
# part of a transcript: it says nothing, and one left behind when the update
# failed would read as an answer.
SLACK_THINKING_TEXT = "_Thinking…_"
# A handoff notice the bot no longer posts; still kept out of transcripts.
SLACK_AGENT_HANDOFF_TEXT = "_Passed to the Agent run in this thread._"
SLACK_GENERIC_ERROR_MESSAGE = (
    "❌ Sorry, I couldn't process that request right now. Please try again in a "
    "moment."
)


def _escape_mrkdwn(text: str) -> str:
    """Escape what Slack reads as links, mentions and entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_fenced_text(text: str, limit: int) -> List[str]:
    """Split Markdown into chunks of at most `limit` characters.

    Each chunk is as long as it can be, so no tiny fragments are left
    between parts. A chunk breaks on a newline when one is near the limit,
    else on a space, else mid-text; a code fence left open at a break is
    closed and reopened so each chunk renders alone.
    """
    chunks: List[str] = []
    remaining = text
    open_fence = False
    while remaining:
        prefix = "```\n" if open_fence else ""
        if len(prefix) + len(remaining) <= limit:
            piece, remaining = remaining, ""
        else:
            budget = max(limit - len(prefix) - len("\n```"), 1)
            cut = remaining.rfind("\n", 0, budget)
            space = remaining.rfind(" ", 0, budget)
            if cut >= budget // 2:
                piece, remaining = remaining[:cut], remaining[cut:]
                # Inside code only the break's own newline goes (the fence
                # markers stand in for it); blank code lines stay.
                remaining = (
                    remaining[1:]
                    if _fence_is_open(prefix + piece)
                    else remaining.lstrip("\n")
                )
            elif space >= budget // 2 and not _fence_is_open(
                prefix + remaining[:space]
            ):
                # Prose breaks on a space, which the break replaces.
                piece, remaining = remaining[:space], remaining[space + 1 :]
            else:
                # Code keeps every character: a hard cut, but never inside a
                # run of backticks, which would flip every later fence.
                cut = budget
                while 0 < cut < len(remaining) and remaining[cut - 1 : cut + 1] == "``":
                    cut -= 1
                if cut == 0:
                    cut = budget
                piece, remaining = remaining[:cut], remaining[cut:]
                if not _fence_is_open(prefix + piece):
                    remaining = remaining.lstrip("\n")
                elif remaining.startswith("\n"):
                    remaining = remaining[1:]
        chunk = prefix + piece
        open_fence = _fence_is_open(chunk)
        if open_fence and remaining:
            chunk += "\n```"
            closing_line, _, after = remaining.partition("\n")
            if closing_line.startswith("```") and not closing_line[3:].strip():
                # The break fell just before the fence's own closing line:
                # that closes this chunk, so the next one is not reopened
                # (which would render an empty code block).
                open_fence = False
                remaining = after.lstrip("\n")
        chunks.append(chunk)
    return chunks


def _fence_is_open(text: str) -> bool:
    return text.count("```") % 2 == 1


# Slack API errors that reject a message's blocks, not the message.
SLACK_BLOCK_REJECTION_ERRORS = frozenset(
    {"invalid_blocks", "invalid_blocks_format", "msg_blocks_too_long"}
)


def _is_block_rejection(exc: Exception) -> bool:
    """Whether Slack refused a message for its blocks (a SlackApiError's code)."""
    response = getattr(exc, "response", None)
    getter = getattr(response, "get", None)
    try:
        error = getter("error") if callable(getter) else None
    except Exception:
        return False
    return error in SLACK_BLOCK_REJECTION_ERRORS


def _without_blocks(message: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in message.items() if key != "blocks"}


# Slack's limit on the text of one `markdown` block.
SLACK_MARKDOWN_BLOCK_CHARS = 12000


def slack_chat_request_id(
    team_id: Optional[str], channel_id: Optional[str], message_ts: Optional[str]
) -> Optional[str]:
    """The request id of a chat query asked by the Slack message at message_ts.

    Deterministic, so a redelivered event gets the id it had the first time.
    """
    if not channel_id or not message_ts:
        return None
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"slack:{team_id or ''}:{channel_id}:{message_ts}"
        )
    )


# Thumbs up / down under a chat answer; the value is the answered request id.
CHAT_FEEDBACK_ACTIONS = {"chat_feedback_up": "up", "chat_feedback_down": "down"}
CHAT_FEEDBACK_THANKS_TEXT = "Thanks for the feedback."
CHAT_FEEDBACK_NOT_RECORDED_TEXT = (
    "Sorry, your feedback could not be recorded right now. Please try again."
)
# The send was not confirmed in time but may still arrive: no retry prompt,
# since a retry is a new click and would count as a second vote.
CHAT_FEEDBACK_UNCONFIRMED_TEXT = (
    "Your feedback was sent, but it may not have been recorded."
)
CHAT_FEEDBACK_CLICKS_LIMIT = 1000


def _chat_feedback_block(request_id: str) -> Dict[str, Any]:
    return {
        "type": "actions",
        "block_id": "chat_feedback",
        "elements": [
            {
                "type": "button",
                "action_id": action_id,
                "text": {"type": "plain_text", "text": emoji, "emoji": True},
                "value": request_id,
                "accessibility_label": label,
            }
            for action_id, emoji, label in (
                ("chat_feedback_up", "👍", "Good answer"),
                ("chat_feedback_down", "👎", "Bad answer"),
            )
        ],
    }


class SlackAnswer(str):
    """A model's answer text, carrying what the Slack footer shows about it."""

    model_name: Optional[str]
    latency_ms: Optional[float]
    cached: bool
    trailer_lines: FrozenSet[str]
    # The request it answers: what a feedback click rates.
    request_id: Optional[str]

    def __new__(
        cls,
        text: str,
        *,
        model_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
        cached: bool = False,
        trailer_lines: Iterable[str] = (),
        request_id: Optional[str] = None,
    ) -> "SlackAnswer":
        answer = super().__new__(cls, text)
        answer.model_name = model_name
        answer.latency_ms = latency_ms
        answer.cached = cached
        answer.request_id = request_id if is_valid_request_id(request_id) else None
        # The bot's own lines after the model's text: notes and sources.
        answer.trailer_lines = frozenset(line for line in trailer_lines if line)
        return answer

    def markdown(self, part: str) -> str:
        """`part` for a Markdown block, which joins single newlines.

        Adjacent note and source lines get a blank line between them so
        each stays on its own line; the model's own text is left alone.
        """
        lines = part.split("\n")
        out = lines[:1]
        for previous, line in zip(lines, lines[1:]):
            if previous in self.trailer_lines and line in self.trailer_lines:
                out.append("")
            out.append(line)
        return "\n".join(out)

    def footer(self) -> Optional[str]:
        """`model · 1.2 s · cached`; never the cost."""
        fields = []
        if self.model_name and self.model_name != "unknown":
            fields.append(self.model_name)
        if self.latency_ms is not None and self.latency_ms >= 0:
            fields.append(f"{self.latency_ms / 1000:.1f} s")
        if self.cached:
            fields.append("cached")
        return " · ".join(fields) or None


# Slack's limit on the text of one `section` block.
SLACK_SECTION_TEXT_CHARS = 3000


class SlackNotice(str):
    """An error reply's text, carrying a short reference to its request.

    The text is what the user always saw; the reference (the request id's
    first 8 characters) shows under it so a report can be matched to logs.
    """

    reference: Optional[str]

    def __new__(cls, text: str, *, request_id: Optional[str] = None) -> "SlackNotice":
        notice = super().__new__(cls, text)
        notice.reference = (
            request_id[:8] if is_valid_request_id(request_id) and request_id else None
        )
        return notice

    def blocks(self, part: str) -> Optional[List[Dict[str, Any]]]:
        """`part` as a section with the reference under it; None without one."""
        if not self.reference or len(part) > SLACK_SECTION_TEXT_CHARS:
            return None
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": part}},
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Reference: `{self.reference}`"}
                ],
            },
        ]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _format_wait(seconds: float) -> str:
    """A wait rounded up: seconds under a minute, whole minutes beyond."""
    whole_seconds = max(1, math.ceil(seconds))
    if whole_seconds < 60:
        return _plural(whole_seconds, "second")
    return _plural(math.ceil(whole_seconds / 60), "minute")


def _retry_wait(retry_after_seconds: Any) -> Optional[str]:
    """`_format_wait` of a positive retry-after; None for anything else."""
    if (
        isinstance(retry_after_seconds, (int, float))
        and not isinstance(retry_after_seconds, bool)
        and retry_after_seconds > 0
    ):
        return _format_wait(retry_after_seconds)
    return None


def _known_response_warnings(warnings: Any, *, skip: Tuple[str, ...] = ()) -> List[str]:
    """A response's warnings that have a note, each once, in a fixed order."""
    if not isinstance(warnings, (list, tuple)):
        return []
    present = {str(warning).split(":", 1)[0] for warning in warnings}
    return [
        warning
        for warning in SLACK_RESPONSE_WARNING_NOTES
        if warning in present and warning not in skip
    ]


def _response_warning_notes(warnings: Any, *, skip: Tuple[str, ...] = ()) -> str:
    """The Slack notes for a response's known warnings."""
    return "\n".join(
        SLACK_RESPONSE_WARNING_NOTES[warning]
        for warning in _known_response_warnings(warnings, skip=skip)
    )


def _response_warning_label(warnings: Any) -> str:
    """The plain-text history label for a response's known warnings."""
    labels = [
        SLACK_RESPONSE_WARNING_LABELS[warning]
        for warning in _known_response_warnings(warnings)
    ]
    return f"[{'; '.join(labels)}]" if labels else ""


# Admission reasons that are a quota rather than capacity (src/admission.py),
# by the kind of quota. The user's own:
ADMISSION_USER_QUOTA_REASONS = {
    "identity_request_rate_exceeded": "request",
    "user_request_rate_exceeded": "request",
    "user_token_budget_exceeded": "token",
}
# One bucket shared by every user of a tier, so not this user's doing:
ADMISSION_TIER_QUOTA_REASONS = {
    "tier_request_rate_exceeded": "request",
    "tier_token_budget_exceeded": "token",
}
# Every other rejection is shared capacity: the router is busy.


def _busy_text(wait: Optional[str]) -> str:
    if wait:
        return f"⏳ The LLM Router is busy right now. Please try again in {wait}."
    return SLACK_BUSY_MESSAGE


def _admission_rejection_text(
    decision: Any, user_tier: Optional[UserTier] = None
) -> str:
    """The reply when admission turns a query away: a quota, or capacity."""
    reason = str(getattr(decision, "reason", "") or "")
    if reason == "scheduler_queue_timeout":
        # The queue wait ended on whatever turned the request away last.
        metadata = getattr(decision, "metadata", None) or {}
        reason = str(metadata.get("last_reason") or "")
    wait = _retry_wait(getattr(decision, "retry_after_seconds", None))
    when = f"Please try again in {wait}." if wait else "Please try again later."
    if reason in ADMISSION_USER_QUOTA_REASONS:
        quota = ADMISSION_USER_QUOTA_REASONS[reason]
        return f"⏳ You've reached your {quota} quota for now. {when}"
    if reason in ADMISSION_TIER_QUOTA_REASONS:
        quota = ADMISSION_TIER_QUOTA_REASONS[reason]
        tier = f"The {user_tier.value.title()} tier's" if user_tier else "A tier's"
        return f"⏳ {tier} shared {quota} quota is used up right now. {when}"
    return _busy_text(wait)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


# What the bot posts instead of an answer: progress, errors, busy and empty
# answer notices. A transcript skips them, since none of them is an answer.
_SLACK_NOTICE_TEXTS = frozenset(
    _normalized_text(text)
    for text in (
        SLACK_THINKING_TEXT,
        SLACK_AGENT_HANDOFF_TEXT,
        SLACK_BUSY_MESSAGE,
        SLACK_GENERIC_ERROR_MESSAGE,
        MEMORY_SLASH_ONLY_MESSAGE,
        *SLACK_ERROR_CODE_MESSAGES.values(),
        *SLACK_INFERENCE_ERROR_MESSAGES.values(),
    )
)
# Notices with a variable part (a wait, a quota, notes): the busy, provider
# rate-limit and quota templates above, and the empty answer with its notes.
_SLACK_NOTICE_PATTERN = re.compile(
    r"(?:⏳ The LLM Router is busy right now\. Please try again in "
    r"|⏳ The model provider is rate-limiting requests right now\. "
    r"Please try again in "
    r"|⏳ You've reached your \w+ quota for now\. Please try again "
    r"|⏳ (?:The \w+ tier's|A tier's) shared \w+ quota is used up right now\. "
    r"Please try again "
    rf"|{re.escape(_normalized_text(SLACK_EMPTY_ANSWER_MESSAGE))})"
)


def _is_slack_notice_text(text: Any) -> bool:
    normalized = _normalized_text(text)
    return normalized in _SLACK_NOTICE_TEXTS or bool(
        _SLACK_NOTICE_PATTERN.match(normalized)
    )


_SLACK_LINK_PATTERN = re.compile(r"<((?:https?|mailto):[^<>|]*)(?:\|([^<>]*))?>")


def _slack_text_as_posted(text: Any) -> str:
    """Slack message text back as the bot wrote it, for comparison only.

    Unwraps `<url>` and `<url|label>` links, unescapes `&lt;`, `&gt;` and
    `&amp;`, and collapses whitespace.
    """
    unwrapped = _SLACK_LINK_PATTERN.sub(
        lambda match: match.group(2) or match.group(1), str(text or "")
    )
    unescaped = (
        unwrapped.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    )
    return _normalized_text(unescaped)


class SlackBotReplyMixin:
    """SlackBot's reply delivery."""

    def _answer_message(self, part: str, answer: str, *, last: bool) -> Dict[str, Any]:
        """The message fields for one part of a reply.

        A model's answer renders as a Markdown block, with the plain text
        kept as the notification fallback, and its last part carries the
        footer. An error's last part carries its reference; command replies
        stay plain text.
        """
        message: Dict[str, Any] = {"text": part}
        if isinstance(answer, SlackNotice):
            notice_blocks = answer.blocks(part) if last else None
            if notice_blocks:
                message["blocks"] = notice_blocks
            return message
        if (
            not isinstance(answer, SlackAnswer)
            or len(part) > SLACK_MARKDOWN_BLOCK_CHARS
        ):
            return message
        markdown = answer.markdown(part)
        if len(markdown) > SLACK_MARKDOWN_BLOCK_CHARS:
            return message
        blocks: List[Dict[str, Any]] = [{"type": "markdown", "text": markdown}]
        response_settings = self.config.get("response_settings", {})
        footer = answer.footer() if last else None
        if footer and response_settings.get("answer_footer", True):
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "plain_text", "text": footer}],
                }
            )
        if last and answer.request_id and response_settings.get("feedback_buttons"):
            blocks.append(_chat_feedback_block(answer.request_id))
        message["blocks"] = blocks
        return message

    def _typing_indicator_enabled(self) -> bool:
        return bool(
            self.config.get("response_settings", {}).get("typing_indicator", True)
        )

    def _socket_ack_payload(self, req: Any) -> Optional[Dict[str, Any]]:
        """Tell the user a model-backed slash command is underway."""
        if getattr(req, "type", None) != "slash_commands":
            return None
        if not self._typing_indicator_enabled():
            return None
        payload = getattr(req, "payload", None) or {}
        waits_on_model = getattr(self.message_handler, "command_waits_on_model", None)
        if callable(waits_on_model) and not waits_on_model(
            payload.get("text", "") or ""
        ):
            return None
        return {"response_type": "ephemeral", "text": SLACK_SLASH_WORKING_TEXT}

    def _waits_on_model(self, event: Dict[str, Any]) -> bool:
        """Whether a message waits on a model answer, so progress is shown."""
        if self._is_query_event(event):
            return True
        extract_command_text = getattr(
            self.message_handler, "extract_prefixed_command_text", None
        )
        waits_on_model = getattr(self.message_handler, "command_waits_on_model", None)
        if not (callable(extract_command_text) and callable(waits_on_model)):
            return False
        command_text = extract_command_text((event.get("text") or "").strip())
        return command_text is not None and bool(waits_on_model(command_text))

    async def _post_placeholder(
        self, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> Optional[str]:
        """Show progress where the answer will appear; its ts, or None.

        Only a client that can update the placeholder gets one.
        """
        if not self._typing_indicator_enabled():
            return None
        if not callable(getattr(self.web_client, "chat_update", None)):
            return None
        try:
            response = await self.web_client.chat_postMessage(
                channel=channel_id, text=SLACK_THINKING_TEXT, thread_ts=thread_ts
            )
        except Exception as exc:
            logger.warning("Could not post the Slack progress placeholder: %s", exc)
            return None
        try:
            ts = response.get("ts") if response is not None else None
        except Exception:
            ts = None
        return ts if isinstance(ts, str) and ts else None

    async def _replace_placeholder(
        self, channel_id: Optional[str], placeholder_ts: str, **message: Any
    ) -> bool:
        """Turn the placeholder into `message`; False when Slack refused.

        An update refused for its blocks is tried once more as plain text.
        Any other failure (a timeout, a 5xx) may have been applied, so it is
        not resent here: the caller posts the message instead.
        """
        try:
            await self.web_client.chat_update(
                channel=channel_id, ts=placeholder_ts, **message
            )
            return True
        except Exception as exc:
            logger.warning("Could not update the Slack progress placeholder: %s", exc)
            if "blocks" not in message or not _is_block_rejection(exc):
                return False
        try:
            await self.web_client.chat_update(
                channel=channel_id, ts=placeholder_ts, **_without_blocks(message)
            )
        except Exception as exc:
            logger.warning("Could not update the Slack progress placeholder: %s", exc)
            return False
        return True

    async def _post_reply(
        self, channel_id: Optional[str], thread_ts: Optional[str], **message: Any
    ) -> None:
        """Post one reply message; as plain text if Slack refuses its blocks.

        Only a block rejection is resent: any other failure may have posted.
        """
        try:
            await self.web_client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts, **message
            )
        except Exception as exc:
            if "blocks" not in message or not _is_block_rejection(exc):
                raise
            logger.warning("Slack refused the answer's blocks; posting text: %s", exc)
            await self.web_client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts, **_without_blocks(message)
            )

    def _reply_thread_ts(
        self, event: Dict[str, Any], user_prefs: Dict[str, Any]
    ) -> Optional[str]:
        """The thread a reply to this message goes in; None for the channel."""
        if event.get("thread_ts"):
            return event.get("thread_ts")
        if (
            self.config.get("response_settings", {}).get("thread_replies", True)
            and user_prefs.get("threading", True)
            and event.get("ts")
        ):
            return event.get("ts")
        return None

    def _split_response(self, text: str, max_length: int) -> List[str]:
        """Split a long answer into Slack messages of at most `max_length`.

        Code fences stay balanced in every part (see `_split_fenced_text`).
        """
        if len(text) <= max_length:
            return [text]
        parts = [
            part.strip("\n")
            for part in _split_fenced_text(text, max_length)
            if part.strip()
        ]
        return parts or [text[:max_length]]
