"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)
from dataclasses import dataclass

from slack_sdk.http_retry.builtin_async_handlers import (
    AsyncRateLimitErrorRetryHandler,
)
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.aiohttp import (
    SocketModeClient as AsyncSocketModeClient,
)
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
import httpx

from src.agent_runtime import (
    AgentRuntimeBusy,
    AgentRuntimeConflict,
    AgentRuntimeError,
    AgentRuntimeTimeout,
    AgentRuntimeUnauthorized,
    AgentRuntimeUnavailable,
    configured_model_refs,
    is_transient_error,
)
from slack.agent_bootstrap import (
    AgentBootstrapBuilder,
    AgentBootstrapContext,
    AgentBootstrapFetchError,
)
from slack.agent_feedback import (
    AGENT_FEEDBACK_CALLBACK_ID,
    AGENT_TERMINAL_STATUSES,
    AgentFeedbackNonce,
    build_agent_feedback_modal,
    parse_agent_feedback_submission,
)
from slack.agent_shortcut import (
    AgentShortcutNonce,
    AgentShortcutSubmission,
    build_agent_shortcut_modal,
    parse_agent_shortcut_submission,
)
from slack.files import (
    declares_html,
    download_slack_file,
    error_name,
    is_allowed_slack_file_url,
    is_slack_hosted_file,
)
from src.admission import AdmissionRejectedError
from src.memory import MemoryManager, build_memory_scope
from src.utils.schema import (
    Attachment,
    AttachmentType,
    QueryRequest,
    ToolPolicy,
    UserTier,
)
from src.utils.metrics import SLACK_METRICS

logger = logging.getLogger(__name__)

MAX_QUERY_TOKENS = 8192
STATE_SCHEMA_VERSION = 1
# How long a conversation, and the bot thread it lives in, stays live.
SLACK_SESSION_TIMEOUT = timedelta(hours=2)
# Redis expiry for tracked Agent runs; user settings never expire.
SLACK_AGENT_RUN_TTL_SECONDS = 30 * 24 * 3600
# Requests older than this no longer count against a user's rate limit.
RATE_LIMIT_WINDOW_SECONDS = 3600
POSTED_CI_JOBS_LIMIT = 1000
HANDLED_MESSAGES_LIMIT = 1000
UNVERIFIED_LABEL = "⚠️ 未驗證（沒有 verification profile）"
MEMORY_SLASH_ONLY_MESSAGE = (
    "Memory commands are only available through Slack slash commands. "
    "Use `/llm remember`, `/llm memories`, or `/llm forget`."
)
SLACK_BUSY_MESSAGE = "The LLM Router is busy right now. Please retry in a moment."
SLACK_THINKING_TEXT = "_Thinking…_"
SLACK_SLASH_WORKING_TEXT = "Working on it…"
SLACK_AGENT_HANDOFF_TEXT = "_Passed to the Agent run in this thread._"
# Commands that answer through the model, though they spend no query quota.
MODEL_BACKED_COMMANDS = frozenset({"web", "fast"})
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
# The windows UserManager.check_rate_limit counts requests in.
RATE_LIMIT_WINDOW_SECONDS = 3600
RATE_LIMIT_BURST_WINDOW_SECONDS = 300
# Message subtypes a person's follow-up can carry: a plain message, one with
# files, and a reply also sent to the channel. Edits, deletions, joins and the
# like are never a new request. The same set the Agent thread context reads.
CHAT_MESSAGE_SUBTYPES = frozenset(AgentBootstrapBuilder._INCLUDED_SUBTYPES)
SLACK_PRIORITY_WORK_KINDS = frozenset({"interactive"})
AGENT_TASK_ECHO_LIMIT = 2500
AGENT_SHORTCUT_NONCE_TTL_SECONDS = 300
# How long an expired form is remembered, so its submission gets a notice.
AGENT_SHORTCUT_NONCE_GRACE_SECONDS = 3600
AGENT_STARTING_MESSAGE = (
    "🛠️ Starting the Pi coding agent; it will open a thread in this channel "
    "when its worktree is ready."
)
SLACK_SECTION_TEXT_LIMIT = 3000
AGENT_DELIVERY_CARD_LIMIT = 3000
AGENT_DELIVERY_MAX_FILES = 15
AGENT_DELIVERY_FILES_TEXT_LIMIT = AGENT_DELIVERY_CARD_LIMIT // 2
# Jev acts on a route label only at this confidence; the discuss hint uses the same.
AGENT_DISCUSS_HINT_PROBABILITY = 0.9
GITHUB_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
AGENT_TERMINAL_STATUSES = {
    "completed",
    "rejected",
    "cancelled",
    "failed",
    "timed_out",
    "interrupted",
}


def _agent_idempotency_key(
    team_id: Optional[str], channel_id: str, request_id: str, user_id: str, op: str
) -> str:
    """Stable per Slack request, so a redelivered event cannot start a second run."""
    raw = "\n".join([team_id or "", channel_id, request_id, user_id, op])
    return "slack-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _agent_idempotency_kwargs(
    team_id: Optional[str],
    channel_id: str,
    request_id: Optional[str],
    user_id: str,
    op: str,
) -> Dict[str, str]:
    """The runtime call's `idempotency_key`, when the Slack request has an id."""
    if not request_id:
        return {}
    return {
        "idempotency_key": _agent_idempotency_key(
            team_id, channel_id, request_id, user_id, op
        )
    }


# Next step shown after a failed run's reason, keyed by error code or cause_code.
AGENT_ERROR_HINTS = {
    "scope_violation": (
        "Ask for the change again within the profile's allowed paths, "
        "or widen allowedPaths if those files belong in scope."
    ),
    "diff_limit_exceeded": (
        "Split the request into smaller changes, each within the diff limits."
    ),
    "session_integrity_failed": (
        "The saved agent session changed after it was stored, so it was not "
        "continued; start a new thread to begin a fresh session."
    ),
    "disk_quota_exceeded": (
        "The worktree outgrew its disk quota; avoid generating large files or "
        "dependencies in the repository, then retry."
    ),
    "agent_tool_limit": (
        "Narrow the request or split it so the agent needs fewer tool calls."
    ),
    "agent_turn_limit": (
        "Narrow the request or split it so the agent can finish in fewer turns."
    ),
    "grader_timeout": (
        "The grader timed out; retry the run, and report it if it keeps happening."
    ),
    "insufficient_evidence": (
        "The grader could not confirm the change; add tests or point to how "
        "the behaviour can be checked, then retry."
    ),
    "docker_unavailable": (
        "Docker is unavailable on the runtime host; ask an operator to start it, "
        "then retry."
    ),
    "snapshot_too_large": (
        "The change was too large to snapshot for verification; split it into "
        "smaller changes or drop large generated files, then retry."
    ),
    "candidate_changed": (
        "The worktree changed while it was being verified; retry the run, "
        "and report it if it keeps happening."
    ),
    "candidate_artifact_missing": (
        "The verified change could not be found; retry the run, and ask an "
        "operator to check the runtime's artifact storage if it repeats."
    ),
    "verification_stop_unconfirmed": (
        "The verification container could not be confirmed stopped; ask an "
        "operator to check the runtime host before retrying."
    ),
    "grader_did_not_submit": (
        "The grader finished without a verdict; retry the run, and report it "
        "if it keeps happening."
    ),
    "grader_invalid_submission": (
        "The grader's verdict was malformed and rejected; retry the run, and "
        "report it if it keeps happening."
    ),
    "grader_execution_failed": (
        "The grader could not run (often a model or budget limit); retry later, "
        "or ask an operator to check the model gateway."
    ),
    "investigation_timeout": (
        "The investigation ran out of time; narrow the question or point to the "
        "relevant files, then retry."
    ),
    "investigation_missing_structured_result": (
        "The investigation ended without a structured result; retry the run, "
        "and report it if it keeps happening."
    ),
    "investigation_invalid_submission": (
        "The investigation's result was malformed and rejected; retry the run, "
        "and report it if it keeps happening."
    ),
    "investigation_execution_failed": (
        "The investigation could not run (often a model or budget limit); retry "
        "later, or ask an operator to check the model gateway."
    ),
    "investigation_reported_unknown": (
        "The investigation could not find the cause; add the error, logs or "
        "steps to reproduce, then retry."
    ),
    "investigation_stop_unconfirmed": (
        "The investigation container could not be confirmed stopped; ask an "
        "operator to check the runtime host before retrying."
    ),
    "verification_timeout": (
        "Verification ran out of time; narrow the change or ask an operator to "
        "raise the slow check's timeout, then retry."
    ),
    "verification_incomplete": (
        "A verification check never reported a result; retry the run, and ask "
        "an operator to check the verification image if it repeats."
    ),
    "delivery_expired": (
        "The delivery approval expired; rerun the request and approve it in time."
    ),
    "delivery_rejected": (
        "Delivery was rejected; rerun with updated instructions if you still "
        "want the change."
    ),
    "archive_failed": (
        "The change could not be archived, so it stays in the isolated worktree; "
        "ask an operator to recover it before retrying."
    ),
}


def _format_agent_status_line(result: Dict[str, Any]) -> str:
    """Lead every terminal agent result with an explicit outcome."""
    status = str(result.get("status") or "")
    if status not in AGENT_TERMINAL_STATUSES:
        return ""
    run_id = result.get("run_id")
    label = f"Agent run `{run_id}`" if run_id else "Agent run"
    if status == "completed":
        return f"✅ {label} *completed*"
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    code = str(error.get("code") or "").strip()
    message = str(error.get("message") or "").strip()
    reason = ": ".join(part for part in (code, message) if part)
    if not reason:
        reason = {
            "rejected": "the requested action was rejected or the approval expired",
            "cancelled": "the run was cancelled",
            "timed_out": "the run exceeded its deadline",
            "interrupted": "the runtime stopped before the run finished",
        }.get(status, "no error detail was reported")
    line = f"⚠️ {label} *{status}* — {reason}"
    cause = str(error.get("cause_code") or "").strip()
    hint = AGENT_ERROR_HINTS.get(cause) or AGENT_ERROR_HINTS.get(code)
    if hint:
        line = f"{line}\nNext step: {hint}"
    if result.get("patch_retained") is True:
        line = f"{line}\nThe undelivered patch was kept on the runtime host."
    return line


def _agent_run_started_at(run: Dict[str, Any]) -> Optional[float]:
    """The run's start in epoch seconds, from the runtime's `created_at`."""
    value = run.get("created_at")
    if not isinstance(value, str) or not value:
        return None
    try:
        started = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.timestamp()


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


async def _show_agent_status(
    client: Any,
    tracked: Dict[str, Any],
    text: str,
    status_ts: Optional[str] = None,
) -> bool:
    """Make a Slack message the run's status message, which its monitor keeps current.

    Rewrites the message at `status_ts` when given, otherwise posts a reply in
    the run's thread. Returns False when Slack did not take the message.
    """
    try:
        if status_ts:
            if not hasattr(client, "chat_update"):
                return False
            await client.chat_update(
                channel=tracked["channel_id"], ts=status_ts, text=text
            )
        else:
            if not hasattr(client, "chat_postMessage"):
                return False
            posted = await client.chat_postMessage(
                channel=tracked["channel_id"], thread_ts=tracked["thread_ts"], text=text
            )
            status_ts = posted.get("ts") if hasattr(posted, "get") else None
    except Exception:
        logger.warning("Agent status message could not be posted", exc_info=True)
        return False
    if not isinstance(status_ts, str) or not status_ts:
        return False
    tracked["status_ts"] = status_ts
    tracked["status_text"] = text
    return True


def _escape_mrkdwn(text: str) -> str:
    """Escape what Slack reads as links, mentions and entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _int_field(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ci_repair_line(target: Dict[str, Any]) -> str:
    """Name the CI workflow, run, PR and failed jobs a repair approval is for."""
    workflow = _escape_mrkdwn(
        str(target.get("workflow") or "CI")[:100].replace("*", "")
    )
    repository = str(target.get("repository") or "")
    linkable = bool(GITHUB_REPOSITORY_PATTERN.match(repository))
    line = f"🔧 Repairing CI workflow *{workflow}*"
    run = _int_field(target.get("run"))
    if run is not None:
        line += (
            f" <https://github.com/{repository}/actions/runs/{run}|run {run}>"
            if linkable
            else f" run {run}"
        )
    attempt = _int_field(target.get("attempt"))
    if attempt is not None:
        line += f" (attempt {attempt})"
    pr = _int_field(target.get("pr"))
    if pr is not None:
        line += (
            f" on <https://github.com/{repository}/pull/{pr}|PR #{pr}>"
            if linkable
            else f" on PR #{pr}"
        )
    jobs = [
        f"`{_escape_mrkdwn(str(job)[:100].replace('`', chr(39)))}`"
        for job in target.get("failed_jobs") or []
    ]
    if not jobs:
        return f"{line}\nFailed jobs: _not reported_"
    # Keep room for the "and N more" marker so the section stays within its limit.
    budget = SLACK_SECTION_TEXT_LIMIT - len(line) - 40
    shown: List[str] = []
    for job in jobs[:10]:
        if sum(len(item) + 2 for item in shown) + len(job) > budget:
            break
        shown.append(job)
    listed = ", ".join(shown)
    if len(jobs) > len(shown):
        listed += f" _…and {len(jobs) - len(shown)} more_"
    return f"{line}\nFailed jobs: {listed}"


def _format_grader_finding(finding: Dict[str, Any]) -> str:
    location = str(finding.get("path") or "unknown")
    if finding.get("line") is not None:
        location = f"{location}:{finding.get('line')}"
    return f"• `{location}` {finding.get('reason') or ''}".rstrip()


AGENT_FOLLOWUP_FLAG_PATTERN = re.compile(
    r"\s*(--discuss|--implement|--workflow)(?:\s+|$)"
)


AGENT_RUNTIME_CREDENTIALS_TEXT = (
    "Agent mode is misconfigured: the runtime rejected the bot's credentials."
)
AGENT_STATUS_DELAYED_TEXT = (
    "⏳ Agent status is delayed: the Agent runtime is not answering. "
    "I'll keep checking until the run's deadline."
)
AGENT_START_ERROR_TEXT = {
    "session_busy": "This Agent session already has a run in progress; wait for it or use `agent stop`.",
    "verification_profile_required": "Implement runs need a trusted verification profile here; use `--discuss` for a read-only run.",
    "invalid_model": "That Agent model is not allowlisted.",
    "provider_not_configured": "That Agent model's provider is not configured.",
}


AGENT_PATCH_OWNER_TEXT = "Only the user who started this Agent run can use its patch."
AGENT_PATCH_RESUME_ERROR_TEXT = {
    "session_busy": AGENT_START_ERROR_TEXT["session_busy"],
    "patch_conflict": (
        "The kept patch no longer applies cleanly to this session's worktree; "
        "download it or start a new task."
    ),
    "worktree_not_clean": (
        "This session's worktree has uncommitted changes, so the kept patch "
        "was not applied."
    ),
    "patch_truncated": (
        "The kept patch was truncated when it was archived, so it cannot be "
        "resumed; download it instead."
    ),
}


AGENT_RESULT_BLOCK_CHARS = 3000
AGENT_RESULT_BLOCKS_PER_MESSAGE = 10
AGENT_RESULT_MAX_CHARS = 60000


def _split_agent_result(text: str, limit: int = AGENT_RESULT_BLOCK_CHARS) -> List[str]:
    """Split a result into Slack section-sized chunks without dropping content.

    Chunks break on a newline when one is near the limit, and a code fence
    left open at a break is closed and reopened so each block renders alone.
    """
    if len(text) > AGENT_RESULT_MAX_CHARS:
        omitted = len(text) - AGENT_RESULT_MAX_CHARS
        text = (
            text[:AGENT_RESULT_MAX_CHARS].rstrip()
            + f"\n\n_…answer truncated: {omitted:,} more characters were not posted._"
        )
    return _split_fenced_text(text, limit)


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


SLACK_RATE_LIMIT_RETRIES = 2
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


class SlackAnswer(str):
    """A model's answer text, carrying what the Slack footer shows about it."""

    model_name: Optional[str]
    latency_ms: Optional[float]
    cached: bool
    trailer_lines: FrozenSet[str]

    def __new__(
        cls,
        text: str,
        *,
        model_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
        cached: bool = False,
        trailer_lines: Iterable[str] = (),
    ) -> "SlackAnswer":
        answer = super().__new__(cls, text)
        answer.model_name = model_name
        answer.latency_ms = latency_ms
        answer.cached = cached
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


def _add_rate_limit_retry(client: Any) -> None:
    """Retry Slack 429s (honouring Retry-After) instead of failing the post."""
    retry_handlers = getattr(client, "retry_handlers", None)
    if isinstance(retry_handlers, list):
        retry_handlers.append(
            AsyncRateLimitErrorRetryHandler(max_retry_count=SLACK_RATE_LIMIT_RETRIES)
        )


def _agent_task_echo(task: str) -> str:
    """Quote a submitted task back so a refused form does not lose it."""
    task = task.replace("```", "`\u200b``")
    if len(task) > AGENT_TASK_ECHO_LIMIT:
        task = task[:AGENT_TASK_ECHO_LIMIT] + "…"
    return f"\nYour task was:\n```{task}```"


def _is_complete_agent_run_record(record: Any) -> bool:
    """Whether a persisted run record names its run, session, thread and owner."""
    return isinstance(record, dict) and all(
        record.get(field)
        for field in (
            "run_id",
            "session_id",
            "channel_id",
            "thread_ts",
            "owner_user_id",
        )
    )


def _is_agent_command_text(command_text: str) -> bool:
    """Whether an `/llm` or `!llm` payload is an `agent` command."""
    parts = command_text.split(maxsplit=1)
    return bool(parts) and parts[0].lower() == "agent"


def _agent_auth_error_text(exc: AgentRuntimeError, owner_text: str) -> str:
    """Tell a bad bot token (`unauthorized`) apart from a non-owner (`*_forbidden`)."""
    if exc.code == "unauthorized":
        return AGENT_RUNTIME_CREDENTIALS_TEXT
    return owner_text


AGENT_FEEDBACK_RETRY_TEXT = "Result feedback could not be recorded. Please try again."
AGENT_FEEDBACK_UNAVAILABLE_TEXT = (
    "Result feedback was not recorded: this Agent runtime stores feedback "
    "through telemetry, which is turned off (PI_AGENT_TELEMETRY_ENABLED)."
)


def _agent_feedback_error_text(exc: AgentRuntimeError) -> str:
    """Retrying cannot help when the runtime has no feedback capture at all."""
    if exc.code == "feedback_unavailable":
        return AGENT_FEEDBACK_UNAVAILABLE_TEXT
    return AGENT_FEEDBACK_RETRY_TEXT


@dataclass(frozen=True)
class AgentFollowupFlags:
    text: str
    task_mode: Optional[str] = None
    workflow_mode: Optional[str] = None
    consumed: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class SlackWorkItem:
    kind: str
    payload: Dict[str, Any]
    name: str


class SlackStateStore:
    """Base class for Slack state persistence backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def initialize(self):
        """Initialize the backend."""

    async def load_state(self) -> Dict[str, Any]:
        """Load a previously persisted state snapshot."""
        return {}

    async def save_state(self, snapshot: Dict[str, Any]):
        """Persist a state snapshot."""

    async def shutdown(self):
        """Release backend resources."""


class MemorySlackStateStore(SlackStateStore):
    """No-op in-memory state backend."""


class FileSlackStateStore(SlackStateStore):
    """Persist Slack state snapshots to a local JSON file."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = Path(config.get("file_path", "queries/slack_state.json"))

    async def load_state(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {}

        try:
            return json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to load Slack state file {self.file_path}: {exc}")
            return {}

    async def save_state(self, snapshot: Dict[str, Any]):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
            payload = json.dumps(snapshot, ensure_ascii=True, sort_keys=True, indent=2)
            # The snapshot holds Slack user and thread data: owner-only, and
            # durable before the rename makes it the live state file.
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.file_path)
            dir_fd = os.open(self.file_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception as exc:
            logger.warning(f"Failed to write Slack state file {self.file_path}: {exc}")


class RedisSlackStateStore(SlackStateStore):
    """Persist Slack state snapshots in Redis."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        redis_config = config.get("redis", {})
        self.redis_config = redis_config
        self.key_prefix = redis_config.get("key_prefix", "llm_router_slack_state")
        self.users_index_key = f"{self.key_prefix}:users"
        self.rate_limits_index_key = f"{self.key_prefix}:rate_limits"
        self.conversations_index_key = f"{self.key_prefix}:conversations"
        self.active_threads_index_key = f"{self.key_prefix}:active_threads"
        self.agent_runs_index_key = f"{self.key_prefix}:agent_runs"
        self.posted_ci_jobs_key = f"{self.key_prefix}:posted_ci_jobs"
        self.session_ttl_seconds = int(SLACK_SESSION_TIMEOUT.total_seconds())
        self.rate_limit_ttl_seconds = RATE_LIMIT_WINDOW_SECONDS
        self.agent_run_ttl_seconds = SLACK_AGENT_RUN_TTL_SECONDS
        self.client = None

    async def initialize(self):
        try:
            import redis.asyncio as redis_asyncio

            password = None
            password_env = self.redis_config.get("password_env")
            if password_env:
                password = os.getenv(str(password_env))

            redis_url = self.redis_config.get("url")
            if redis_url and hasattr(redis_asyncio.Redis, "from_url"):
                self.client = redis_asyncio.Redis.from_url(redis_url, password=password)
            else:
                self.client = redis_asyncio.Redis(
                    host=self.redis_config.get("host", "localhost"),
                    port=self.redis_config.get("port", 6379),
                    db=self.redis_config.get("db", 0),
                    password=password,
                )

            if hasattr(self.client, "ping"):
                await self.client.ping()
        except Exception as exc:
            logger.error(f"Failed to initialize Redis Slack state store: {exc}")
            raise

    async def load_state(self) -> Dict[str, Any]:
        if not self.client:
            return {}

        try:
            users = await self._load_json_map(
                self.users_index_key, lambda item_id: self._user_key(item_id)
            )
            rate_limits = await self._load_json_map(
                self.rate_limits_index_key,
                lambda item_id: self._rate_limit_key(item_id),
            )
            conversations = await self._load_json_map(
                self.conversations_index_key,
                lambda item_id: self._conversation_key(item_id),
            )
            active_threads = await self._load_json_map(
                self.active_threads_index_key,
                lambda item_id: self._active_thread_key(item_id),
            )
            agent_runs = await self._load_json_map(
                self.agent_runs_index_key,
                lambda item_id: self._agent_run_key(item_id),
            )
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "users": users,
                "rate_limits": rate_limits,
                "conversations": conversations,
                "active_threads": active_threads,
                "agent_runs": agent_runs,
                "posted_ci_jobs": sorted(await self._smembers(self.posted_ci_jobs_key)),
            }
        except Exception as exc:
            logger.warning("Failed to load Slack state from Redis: %s", exc)
            return {}

    async def save_state(self, snapshot: Dict[str, Any]):
        if not self.client:
            return

        users = snapshot.get("users", {})
        for user_id, user_payload in users.items():
            rate_limit_payload = snapshot.get("rate_limits", {}).get(user_id, [])
            await self.persist_user_state(user_id, user_payload, rate_limit_payload)

        conversations = snapshot.get("conversations", {})
        for context_key, context_payload in conversations.items():
            await self.persist_conversation_state(context_key, context_payload)

        active_threads = snapshot.get("active_threads", {})
        for thread_key, thread_payload in active_threads.items():
            await self.persist_active_thread_state(thread_key, thread_payload)

        for run_id, run_payload in snapshot.get("agent_runs", {}).items():
            await self.persist_agent_run_state(run_id, run_payload)

        for run_id in snapshot.get("posted_ci_jobs", []):
            await self.add_posted_ci_job(run_id)

    async def shutdown(self):
        if self.client and hasattr(self.client, "close"):
            await self.client.close()

    async def persist_user_state(
        self,
        user_id: str,
        user_payload: Dict[str, Any],
        rate_limit_payload: List[Any],
    ):
        """Persist one user's preferences/tier and rate-limit counters."""
        if not self.client:
            return

        try:
            if user_payload:
                resolved_user_payload = await self._merge_user_payload(
                    user_id, user_payload
                )
                await self._set_json(self._user_key(user_id), resolved_user_payload)
                await self._sadd(self.users_index_key, user_id)

            merged_rate_limits = await self._merge_rate_limits(
                user_id, rate_limit_payload
            )
            if merged_rate_limits:
                await self._set_json(
                    self._rate_limit_key(user_id),
                    merged_rate_limits,
                    ttl_seconds=self.rate_limit_ttl_seconds,
                )
                await self._sadd(self.rate_limits_index_key, user_id)
            else:
                await self._delete(self._rate_limit_key(user_id))
                await self._srem(self.rate_limits_index_key, user_id)
        except Exception as exc:
            logger.warning("Failed to persist Slack user state to Redis: %s", exc)

    async def delete_user_state(self, user_id: str):
        """Delete one user's persisted state."""
        if not self.client:
            return

        try:
            await self._delete(self._user_key(user_id))
            await self._delete(self._rate_limit_key(user_id))
            await self._srem(self.users_index_key, user_id)
            await self._srem(self.rate_limits_index_key, user_id)
        except Exception as exc:
            logger.warning("Failed to delete Slack user state from Redis: %s", exc)

    async def persist_conversation_state(
        self, context_key: str, context_payload: Dict[str, Any]
    ):
        """Persist one conversation context keyed by user/channel/thread."""
        if not self.client:
            return

        try:
            current_payload = await self._get_json(self._conversation_key(context_key))
            if self._is_payload_newer(
                current_payload, context_payload, "last_activity"
            ):
                return

            await self._set_json(
                self._conversation_key(context_key),
                context_payload,
                ttl_seconds=self.session_ttl_seconds,
            )
            await self._sadd(self.conversations_index_key, context_key)
        except Exception as exc:
            logger.warning("Failed to persist Slack conversation to Redis: %s", exc)

    async def delete_conversation_states(self, context_keys: List[str]):
        """Delete one or more persisted conversation contexts."""
        if not self.client:
            return

        try:
            for context_key in context_keys:
                await self._delete(self._conversation_key(context_key))
                await self._srem(self.conversations_index_key, context_key)
        except Exception as exc:
            logger.warning("Failed to delete Slack conversations from Redis: %s", exc)

    async def persist_active_thread_state(self, thread_key: str, last_activity: str):
        """Persist active bot-thread tracking using one key per thread."""
        if not self.client:
            return

        try:
            current_payload = await self._get_json(self._active_thread_key(thread_key))
            incoming_payload = {"last_activity": last_activity}
            if self._is_payload_newer(
                current_payload, incoming_payload, "last_activity"
            ):
                return

            await self._set_json(
                self._active_thread_key(thread_key),
                incoming_payload,
                ttl_seconds=self.session_ttl_seconds,
            )
            await self._sadd(self.active_threads_index_key, thread_key)
        except Exception as exc:
            logger.warning("Failed to persist Slack active thread to Redis: %s", exc)

    async def delete_active_thread_states(self, thread_keys: List[str]):
        """Delete one or more active-thread markers."""
        if not self.client:
            return

        try:
            for thread_key in thread_keys:
                await self._delete(self._active_thread_key(thread_key))
                await self._srem(self.active_threads_index_key, thread_key)
        except Exception as exc:
            logger.warning("Failed to delete Slack active threads from Redis: %s", exc)

    async def persist_agent_run_state(self, run_id: str, payload: Dict[str, Any]):
        """Persist one tracked Pi agent run so a restart can resume monitoring."""
        if not self.client:
            return

        try:
            await self._set_json(
                self._agent_run_key(run_id),
                payload,
                ttl_seconds=self.agent_run_ttl_seconds,
            )
            await self._sadd(self.agent_runs_index_key, run_id)
        except Exception as exc:
            logger.warning("Failed to persist Slack agent run to Redis: %s", exc)

    async def load_agent_run_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """One persisted Pi agent run, which another bot process may track."""
        if not self.client:
            return None

        try:
            payload = await self._get_json(self._agent_run_key(run_id))
        except Exception as exc:
            logger.warning("Failed to load Slack agent run from Redis: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None

    async def delete_agent_run_state(self, run_id: str):
        """Stop tracking one Pi agent run across restarts."""
        if not self.client:
            return

        try:
            await self._delete(self._agent_run_key(run_id))
            await self._srem(self.agent_runs_index_key, run_id)
        except Exception as exc:
            logger.warning("Failed to delete Slack agent run from Redis: %s", exc)

    async def add_posted_ci_job(self, run_id: str):
        """Remember that a CI repair job already has a Slack thread."""
        if not self.client:
            return

        try:
            await self._sadd(self.posted_ci_jobs_key, run_id)
        except Exception as exc:
            logger.warning("Failed to persist posted CI job to Redis: %s", exc)

    async def remove_posted_ci_job(self, run_id: str):
        """Forget a CI repair job claim so it can be posted again."""
        if not self.client:
            return

        try:
            await self._srem(self.posted_ci_jobs_key, run_id)
        except Exception as exc:
            logger.warning("Failed to delete posted CI job from Redis: %s", exc)

    def _user_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:user:{user_id}"

    def _rate_limit_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:rate_limit:{user_id}"

    def _conversation_key(self, context_key: str) -> str:
        return f"{self.key_prefix}:conversation:{context_key}"

    def _active_thread_key(self, thread_key: str) -> str:
        return f"{self.key_prefix}:active_thread:{thread_key}"

    def _agent_run_key(self, run_id: str) -> str:
        return f"{self.key_prefix}:agent_run:{run_id}"

    async def _load_json_map(self, index_key: str, item_key_builder) -> Dict[str, Any]:
        """Load a keyed JSON map backed by a Redis set index."""
        results = {}
        item_ids = await self._smembers(index_key)
        for item_id in item_ids:
            payload = await self._get_json(item_key_builder(item_id))
            if payload is None:
                await self._srem(index_key, item_id)
                continue
            if (
                isinstance(payload, dict)
                and "last_activity" in payload
                and len(payload) == 1
            ):
                results[item_id] = payload["last_activity"]
            else:
                results[item_id] = payload
        return results

    async def _get_json(self, key: str) -> Optional[Any]:
        payload = await self.client.get(key)
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload)

    async def _set_json(
        self, key: str, payload: Any, *, ttl_seconds: Optional[int] = None
    ):
        """Store JSON that expires after ttl_seconds, or never when that is None."""
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        if hasattr(self.client, "set"):
            await self.client.set(key, serialized, ex=ttl_seconds)
        elif hasattr(self.client, "setex"):
            # setex needs an expiry; this fallback has always used 30 days.
            await self.client.setex(
                key, ttl_seconds or SLACK_AGENT_RUN_TTL_SECONDS, serialized
            )

    async def _delete(self, key: str):
        if hasattr(self.client, "delete"):
            await self.client.delete(key)

    async def _sadd(self, key: str, member: str):
        if hasattr(self.client, "sadd"):
            await self.client.sadd(key, member)

    async def _srem(self, key: str, member: str):
        if hasattr(self.client, "srem"):
            await self.client.srem(key, member)

    async def _smembers(self, key: str) -> List[str]:
        if not hasattr(self.client, "smembers"):
            return []
        members = await self.client.smembers(key)
        normalized = []
        for member in members or []:
            if isinstance(member, bytes):
                normalized.append(member.decode("utf-8"))
            else:
                normalized.append(str(member))
        return normalized

    async def _merge_user_payload(
        self, user_id: str, incoming_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        current_payload = await self._get_json(self._user_key(user_id))
        if not isinstance(current_payload, dict):
            return incoming_payload

        if self._is_payload_newer(current_payload, incoming_payload, "updated_at"):
            return current_payload

        return incoming_payload

    async def _merge_rate_limits(
        self, user_id: str, incoming_payload: List[Any]
    ) -> List[Any]:
        current_payload = await self._get_json(self._rate_limit_key(user_id))
        merged_values: List[Any] = []

        if isinstance(current_payload, list):
            merged_values.extend(current_payload)
        if incoming_payload:
            merged_values.extend(incoming_payload)

        if not merged_values:
            return []

        cutoff = time.time() - RATE_LIMIT_WINDOW_SECONDS
        deduped = sorted(set(_rate_limit_entries(merged_values, cutoff)))
        return [_rate_limit_entry(*entry) for entry in deduped]

    def _is_payload_newer(
        self,
        current_payload: Optional[Dict[str, Any]],
        incoming_payload: Dict[str, Any],
        field: str,
    ) -> bool:
        """Return whether the currently stored payload is newer than the incoming one."""
        if not isinstance(current_payload, dict):
            return False

        current_value = current_payload.get(field)
        incoming_value = incoming_payload.get(field)
        if not current_value or not incoming_value:
            return False

        current_ts = self._parse_timestamp(current_value)
        incoming_ts = self._parse_timestamp(incoming_value)
        if current_ts is None or incoming_ts is None:
            return False
        return current_ts > incoming_ts

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None


def build_slack_state_store(config: Dict[str, Any]) -> SlackStateStore:
    """Construct the configured Slack state backend."""
    backend = (config or {}).get("backend", "memory").lower()
    if backend == "file":
        return FileSlackStateStore(config)
    if backend == "redis":
        return RedisSlackStateStore(config)
    return MemorySlackStateStore(config)


def resolve_slack_state_store_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Support both nested and legacy flat Slack state configuration."""
    resolved_config = dict((config or {}).get("state", {}) or {})

    if "backend" not in resolved_config:
        resolved_config["backend"] = (config or {}).get("state_backend", "memory")

    if "file_path" not in resolved_config and (config or {}).get("state_file"):
        resolved_config["file_path"] = config["state_file"]

    legacy_redis_config = dict((config or {}).get("redis", {}) or {})
    nested_redis_config = dict(resolved_config.get("redis", {}) or {})
    merged_redis_config = dict(legacy_redis_config)
    merged_redis_config.update(nested_redis_config)

    state_key_prefix = merged_redis_config.get("key_prefix") or (config or {}).get(
        "state_key_prefix"
    )
    if state_key_prefix:
        merged_redis_config["key_prefix"] = state_key_prefix

    if merged_redis_config:
        resolved_config["redis"] = merged_redis_config

    return resolved_config


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


RATE_LIMIT_CHAT = "chat"
RATE_LIMIT_AGENT = "agent"


def _rate_limit_entry(at: float, bucket: str) -> Any:
    """One counted request; a chat request stays a bare timestamp.

    That is the shape saved before Agent requests had a bucket of their own.
    """
    return at if bucket == RATE_LIMIT_CHAT else {"at": at, "bucket": bucket}


def _read_rate_limit_entry(entry: Any) -> Optional[Tuple[float, str]]:
    """The (timestamp, bucket) of a counted request; None when malformed."""
    try:
        if isinstance(entry, dict):
            return float(entry["at"]), str(entry.get("bucket") or RATE_LIMIT_CHAT)
        return float(entry), RATE_LIMIT_CHAT
    except (KeyError, TypeError, ValueError):
        return None


def _rate_limit_entries(values: Any, since: float) -> List[Tuple[float, str]]:
    """The well-formed counted requests after `since`."""
    entries = (_read_rate_limit_entry(value) for value in values or [])
    return [entry for entry in entries if entry is not None and entry[0] > since]


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


class UserManager:
    """Manages user data and preferences"""

    def __init__(
        self,
        default_tier: str = UserTier.FREE.value,
        tier_overrides: Optional[Dict[str, str]] = None,
    ):
        self.default_tier = self._normalize_tier_name(default_tier)
        self.tier_overrides = {
            user_id: self._normalize_tier_name(tier_name)
            for user_id, tier_name in (tier_overrides or {}).items()
        }
        self.users = {}  # In production, use a database
        self.rate_limits = {}

    def get_user_tier(self, user_id: str) -> UserTier:
        """The user's tier from config: their override, else the default.

        The tier stored on a user record is only a record of it, so removing
        an override or lowering the default takes effect on the next request.
        """
        return UserTier(self._configured_tier_name(user_id))

    def _configured_tier_name(self, user_id: str) -> str:
        return self.tier_overrides.get(user_id) or self.default_tier

    def check_rate_limit(
        self, user_id: str, config: Dict[str, Any], bucket: str = RATE_LIMIT_CHAT
    ) -> bool:
        """Check if user is within rate limits.

        Chat queries and Agent requests fill separate buckets with the same
        limits, so a burst of one never blocks the other. Each request still
        counts as one Slack query in the user's list.
        """
        current_time = time.time()
        hour_window = current_time - 3600  # 1 hour window

        # Clean old requests
        entries = _rate_limit_entries(self.rate_limits.get(user_id), hour_window)
        self.rate_limits[user_id] = [_rate_limit_entry(*entry) for entry in entries]
        bucket_requests = [at for at, kind in entries if kind == bucket]

        # Check limits
        requests_per_hour = config.get("requests_per_hour", 100)
        burst_requests = config.get("burst_requests", 5)

        # Check hourly limit
        if len(bucket_requests) >= requests_per_hour:
            return False

        # Check burst limit (last 5 minutes)
        burst_window = current_time - 300  # 5 minutes
        recent_requests = [
            req_time for req_time in bucket_requests if req_time > burst_window
        ]

        if len(recent_requests) >= burst_requests:
            return False

        # Add current request
        self.rate_limits[user_id].append(_rate_limit_entry(current_time, bucket))
        return True

    def requests_since(
        self, user_id: str, since: float, bucket: str = RATE_LIMIT_CHAT
    ) -> List[float]:
        """Times of the user's counted requests in one bucket after `since`."""
        return [
            at
            for at, kind in _rate_limit_entries(self.rate_limits.get(user_id), since)
            if kind == bucket
        ]

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        return self.users.get(user_id, {}).get(
            "preferences",
            {
                "response_length": "medium",
                "technical_level": "intermediate",
                "preferred_models": [],
                "threading": True,
                "web_search_auto": True,
            },
        )

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        if user_id not in self.users:
            self.users[user_id] = {}

        if "preferences" not in self.users[user_id]:
            self.users[user_id]["preferences"] = {}

        self.users[user_id]["preferences"].update(preferences)
        self.users[user_id]["updated_at"] = datetime.now().isoformat()

    def ensure_user_record(self, user_id: str) -> bool:
        """Record the configured tier on the user's record; True if it changed.

        A user on the implicit free default gets no record for it alone.
        """
        if not user_id:
            return False

        previous_tier = self.users.get(user_id, {}).get("tier")
        resolved_tier = self._configured_tier_name(user_id)
        if previous_tier == resolved_tier:
            return False
        if (
            previous_tier is None
            and user_id not in self.tier_overrides
            and resolved_tier == UserTier.FREE.value
        ):
            return False

        if user_id not in self.users:
            self.users[user_id] = {}

        self.users[user_id]["tier"] = resolved_tier
        self.users[user_id]["updated_at"] = datetime.now().isoformat()
        return True

    def _normalize_tier_name(self, tier_name: Optional[str]) -> str:
        """Normalize tier names used in config and persisted user state."""
        normalized = (tier_name or UserTier.FREE.value).strip().lower()
        try:
            return UserTier(normalized).value
        except ValueError:
            return UserTier.FREE.value


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
        """Get conversation summary for context"""
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"
        context = self.conversations.get(context_key)

        if not context or not context.conversation_history:
            return ""

        # Create summary of recent conversation
        recent_messages = context.conversation_history[-10:]  # Last 10 messages
        summary_parts = []

        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"][:200]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)


class SlackMessageHandler:
    """Handles different types of Slack messages and commands"""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.commands = {
            "help": self._handle_help_command,
            "settings": self._handle_settings_command,
            "status": self._handle_status_command,
            "models": self._handle_models_command,
            "analytics": self._handle_analytics_command,
            "clear": self._handle_clear_command,
            "web": self._handle_web_command,
            "fast": self._handle_fast_command,
            "agent": self._handle_agent_command,
            "remember": self._handle_remember_command,
            "memories": self._handle_memories_command,
            "forget": self._handle_forget_command,
        }

    def extract_prefixed_command_text(self, text: str) -> Optional[str]:
        """Strip inline `/llm` / `!llm` prefixes while preserving the raw payload."""
        normalized = (text or "").strip()
        for prefix in ("/llm", "!llm"):
            if normalized == prefix:
                return ""
            if normalized.startswith(f"{prefix} "):
                return normalized[len(prefix) :].strip()
        return None

    def is_supported_command(self, command_text: str) -> bool:
        """Return whether the first token maps to a supported Slack command."""
        normalized = (command_text or "").strip()
        if not normalized:
            return True
        command_name = normalized.split(maxsplit=1)[0].lower()
        return command_name in self.commands

    def command_consumes_query_quota(self, command_text: str) -> bool:
        """Return whether a command performs a model-backed query."""
        normalized = (command_text or "").strip()
        if not normalized:
            return False
        command_name = normalized.split(maxsplit=1)[0].lower()
        if command_name == "agent":
            parts = normalized.split()
            return len(parts) < 2 or parts[1].lower() not in {"status", "stop", "close"}
        return command_name not in self.commands

    def command_waits_on_model(self, command_text: str) -> bool:
        """Whether a command waits on a model answer or an Agent start.

        That is every quota-spending command plus `web` and `fast` with a
        query, which are worth a progress indicator.
        """
        if self.command_consumes_query_quota(command_text):
            return True
        parts = (command_text or "").strip().split(maxsplit=1)
        return len(parts) == 2 and parts[0].lower() in MODEL_BACKED_COMMANDS

    async def handle_message(
        self, event: Dict[str, Any], client: AsyncWebClient
    ) -> Optional[str]:
        """Handle incoming message"""
        text = event.get("text", "").strip()
        routing_text = text
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        message_ts = event.get("ts")
        team_id = self.bot._extract_team_id(event)
        attachments = list(event.get("_query_attachments", []) or [])

        # The bot decides chat or Agent once per message, before it shows
        # progress or downloads files; a thread that became an Agent thread
        # since then does not change this message's lane.
        decided_agent = event.get("_agent_request")
        if not isinstance(decided_agent, bool):
            is_agent_thread = getattr(
                self.bot, "_is_agent_thread", lambda *_args: False
            )
            decided_agent = is_agent_thread(channel_id, event.get("thread_ts"))
        in_agent_thread = bool(event.get("thread_ts")) and decided_agent
        # The chat path does not download an Agent thread's files.
        agent_files = [
            item for item in event.get("files") or [] if isinstance(item, dict)
        ]

        # Check if it's a command
        if not text and not attachments and not (in_agent_thread and agent_files):
            return await self._handle_help_command([], user_id, channel_id, client)

        command_text = self.extract_prefixed_command_text(text)
        if command_text is not None:
            return await self._handle_command_or_query(
                command_text,
                user_id,
                channel_id,
                thread_ts,
                client,
                attachments=attachments,
                team_id=team_id,
                message_ts=message_ts,
            )

        if not text and attachments:
            text = self._build_attachment_only_query(attachments)
        elif not text and in_agent_thread:
            text = self._files_only_query(
                [str(item.get("name") or "") for item in agent_files]
            )

        if in_agent_thread:
            flags = self._parse_followup_agent_flags(text)
            if flags.error:
                return flags.error
            if flags.consumed:
                text = flags.text
                routing_text = flags.text
            files_note = ""
            files = [
                item for item in event.get("files") or [] if isinstance(item, dict)
            ]
            builder = getattr(self.bot, "agent_bootstrap_builder", None)
            if files and builder is not None:
                # Like the files of an initial request's thread, these reach
                # the agent as untrusted reference context for the task.
                loaded = await builder.build_message_files(
                    client, files=files, owner_user_id=user_id, task=text
                )
                text = loaded.prompt
                files_note = self._attached_files_note(loaded)
            return await self._start_agent_prompt(
                text,
                user_id,
                channel_id,
                client,
                team_id=team_id,
                thread_ts=event.get("thread_ts"),
                routing_text=routing_text,
                request_id=event.get("ts"),
                **({"status_note": files_note} if files_note else {}),
                **(
                    {"workflow_mode": flags.workflow_mode}
                    if flags.workflow_mode
                    else {}
                ),
                **({"task_mode": flags.task_mode} if flags.task_mode else {}),
            )

        # Regular query - process through inference engine
        return await self._handle_query(
            text,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
            message_ts=message_ts,
        )

    @staticmethod
    def _parse_followup_agent_flags(text: str) -> "AgentFollowupFlags":
        """Consume leading `--discuss` / `--implement` / `--workflow` flags."""
        task_mode: Optional[str] = None
        workflow_mode: Optional[str] = None
        remaining = text
        consumed = False
        while True:
            match = AGENT_FOLLOWUP_FLAG_PATTERN.match(remaining)
            if not match:
                break
            option = match.group(1)
            if option == "--workflow":
                workflow_mode = "structured"
            else:
                requested = option[2:]
                if task_mode and task_mode != requested:
                    return AgentFollowupFlags(
                        text=text,
                        error="Conflicting task modes: choose --discuss or --implement.",
                    )
                task_mode = requested
            remaining = remaining[match.end() :]
            consumed = True
        return AgentFollowupFlags(
            text=remaining.strip() if consumed else text,
            task_mode=task_mode,
            workflow_mode=workflow_mode,
            consumed=consumed,
        )

    @staticmethod
    def _attached_files_note(loaded: AgentBootstrapContext) -> str:
        details = [f"Loaded {loaded.resource_count} attached resources."]
        if loaded.truncated:
            details.append("Attachments truncated to configured limits.")
        if loaded.partial:
            details.append("Attachments are partial: the time budget ran out.")
        if loaded.warnings:
            details.append(
                "Skipped or partial resources: `"
                + "`, `".join(loaded.warnings[:10])
                + "`."
            )
        return "\n".join(details)

    @staticmethod
    def _agent_mode_summary(task_mode: str, workflow_mode: Optional[str]) -> str:
        summary = f"Mode: *{task_mode}*"
        if task_mode == "discuss":
            summary += " (read-only; reply with `--implement` to make changes)"
        if workflow_mode == "structured":
            summary += " · structured workflow"
        return summary

    async def _handle_command_or_query(
        self,
        command_text: str,
        user_id: str,
        channel_id: str,
        thread_ts: Optional[str],
        client: AsyncWebClient,
        attachments: Optional[List[Attachment]] = None,
        team_id: Optional[str] = None,
        command_surface: str = "message",
        message_ts: Optional[str] = None,
    ) -> str:
        """Handle inline `/llm` / `!llm` payloads as commands or free-form queries."""
        return await self._handle_command(
            command_text,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments or [],
            team_id=team_id,
            command_surface=command_surface,
            message_ts=message_ts,
        )

    async def _handle_command(
        self,
        command_text: str,
        user_id: str,
        channel_id: str,
        thread_ts: Optional[str],
        client: AsyncWebClient,
        attachments: Optional[List[Attachment]] = None,
        team_id: Optional[str] = None,
        command_surface: str = "message",
        message_ts: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Handle bot commands, falling back to a free-form query when appropriate."""
        parts = command_text.split()
        if not parts:
            return await self._handle_help_command([], user_id, channel_id, client)

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in self.commands:
            if command == "web":
                return await self._handle_web_command(
                    args,
                    user_id,
                    channel_id,
                    client,
                    thread_ts=thread_ts,
                    attachments=attachments,
                    team_id=team_id,
                    message_ts=message_ts,
                )
            if command == "fast":
                return await self._handle_fast_command(
                    args,
                    user_id,
                    channel_id,
                    client,
                    thread_ts=thread_ts,
                    attachments=attachments,
                    team_id=team_id,
                    message_ts=message_ts,
                )
            if command in {"remember", "memories", "forget"}:
                return await self.commands[command](
                    args,
                    user_id,
                    channel_id,
                    client,
                    team_id=team_id,
                    thread_ts=thread_ts,
                    command_surface=command_surface,
                )
            if command == "agent":
                return await self._handle_agent_command(
                    args,
                    user_id,
                    channel_id,
                    client,
                    team_id=team_id,
                    thread_ts=thread_ts,
                    message_ts=message_ts,
                    request_id=request_id,
                    background_start=command_surface == "slash",
                )
            return await self.commands[command](args, user_id, channel_id, client)
        return await self._handle_query(
            command_text,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
            message_ts=message_ts,
        )

    async def _handle_web_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
        attachments: Optional[List[Attachment]] = None,
        team_id: Optional[str] = None,
        message_ts: Optional[str] = None,
    ) -> str:
        """Handle `/llm web <query>` by requiring the web_search tool."""
        query = " ".join(args).strip()
        if not query:
            return "Usage: `/llm web <query>`"
        return await self._handle_query(
            query,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
            tool_policy=ToolPolicy.REQUIRED,
            allowed_tools=["web_search"],
            message_ts=message_ts,
        )

    async def _handle_fast_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
        attachments: Optional[List[Attachment]] = None,
        team_id: Optional[str] = None,
        message_ts: Optional[str] = None,
    ) -> str:
        """Handle `/llm fast <query>` by marking this request low-latency."""
        query = " ".join(args).strip()
        if not query:
            return "Usage: `/llm fast <query>`"
        return await self._handle_query(
            query,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
            routing_metadata={
                "requires_low_latency": True,
                "latency_sla": "interactive",
                "routing_intent_source": "slack_fast_command",
            },
            message_ts=message_ts,
        )

    async def _handle_agent_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        team_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
        message_ts: Optional[str] = None,
        request_id: Optional[str] = None,
        background_start: bool = False,
    ) -> str:
        """Start or control a stateful Pi coding-agent thread.

        With `background_start`, a new task is acknowledged at once and its
        bootstrap and `create_session` run in a background task, so a slow
        start never holds a Slack worker.

        `request_id` (the triggering message's ts by default) keys each runtime
        call, so a redelivered message or slash command reuses its first key.
        """
        if not args:
            return "Usage: `/llm agent <task>`"

        runtime = getattr(self.bot, "agent_runtime_client", None)
        if runtime is None:
            return "Agent mode is currently disabled."
        request_id = request_id or message_ts

        def keyed(op: str) -> Dict[str, str]:
            return _agent_idempotency_kwargs(
                team_id, channel_id, request_id, user_id, op
            )

        action = args[0].lower()
        if action in {"status", "stop", "close"} and len(args) == 1:
            tracked = self.bot._agent_run_for(user_id, channel_id, thread_ts)
            if not tracked:
                return "No active Agent session was found for you."
            if action != "close" and not tracked.get("run_id"):
                # A session restored from the runtime may have no run to act on.
                return (
                    f"Agent session `{tracked['session_id']}` has no active run "
                    "right now. Reply in its thread to start one, or use "
                    "`agent close` to close the session."
                )
            try:
                if action == "status":
                    run = await runtime.get_run(tracked["run_id"])
                    return self._format_agent_status(run)
                if action == "stop":
                    await runtime.cancel(tracked["run_id"], user_id, **keyed("cancel"))
                    return f"Agent run `{tracked['run_id']}` is stopping."
                await runtime.close_session(tracked["session_id"], user_id)
                await self.bot._forget_agent_run(tracked)
                return f"Agent session `{tracked['session_id']}` is closed; its branch and commits were kept."
            except AgentRuntimeConflict:
                return "That Agent action has already completed or is no longer valid."
            except AgentRuntimeUnauthorized as exc:
                return _agent_auth_error_text(
                    exc, "Only the Agent session owner can perform that action."
                )
            except AgentRuntimeError:
                return "Agent mode could not complete that control action."

        if action == "resume" and len(args) == 2:
            try:
                result = await runtime.resume(args[1], user_id, **keyed("resume"))
            except AgentRuntimeError:
                return "Workflow cannot resume. Check ownership, baseline, artifacts and whether writing already started."
            resumed = f"🛠️ Workflow `{result['run_id']}` resumed."
            # The run goes on in its own thread wherever this came from; a
            # slash command carries no thread at all.
            home = await self.bot._agent_run_home(result, user_id) or {}
            run_channel = home.get("channel_id") or channel_id
            run_thread = home.get("thread_ts") or thread_ts
            status_ts = None
            if not run_thread and hasattr(client, "chat_postMessage"):
                # Nothing names its thread, so like a CI repair it opens one here.
                started = await client.chat_postMessage(
                    channel=channel_id,
                    text=f"🛠️ Resuming workflow `{result['run_id']}`…",
                )
                run_thread = status_ts = started.get("ts")
            if not run_thread:
                return f"Workflow `{result['run_id']}` resumed. Use `agent status` to inspect it."
            tracked = self.bot._track_agent_run(
                result,
                team_id=home.get("team_id") or team_id or "unknown-team",
                channel_id=run_channel,
                thread_ts=run_thread,
                owner_user_id=user_id,
                mode=home.get("mode"),
            )
            shown = await _show_agent_status(client, tracked, resumed, status_ts)
            self.bot._start_agent_monitor(tracked)
            if (run_channel, run_thread) != (channel_id, thread_ts) and not status_ts:
                return (
                    f"{resumed} Its progress, approvals and result appear in its "
                    f"thread in <#{run_channel}>."
                )
            return "" if shown else resumed

        if action in {"repair-ci", "publish-repair"}:
            ci_args = list(args)
            ci_workflow = (
                action == "repair-ci"
                and len(ci_args) > 1
                and ci_args[1] == "--workflow"
            )
            if ci_workflow:
                ci_args.pop(1)
            if len(ci_args) != 2:
                return "Usage: `/llm agent repair-ci <PR_URL|RUN_ID>` or `/llm agent publish-repair <RUN_ID>`"
            try:
                repair_op = (
                    "repair_ci" if action == "repair-ci" else "retry_publication"
                )
                result = await getattr(runtime, repair_op)(
                    ci_args[1],
                    user_id,
                    team_id=team_id or "unknown-team",
                    channel_id=channel_id,
                    **({"workflow_mode": "structured"} if ci_workflow else {}),
                    **keyed(repair_op),
                )
            except AgentRuntimeError:
                return "CI repair could not start. Check repository profile, owner mapping, failed workflow and GitHub App configuration."
            claimed = await self.bot._claim_ci_job(result["run_id"])
            if not claimed and action == "repair-ci":
                # The CI inbox already opened a review thread for this job.
                return f"CI repair `{result['run_id']}` is awaiting approval in the thread."
            root_ts = thread_ts
            if not root_ts:
                started = await client.chat_postMessage(
                    channel=channel_id,
                    text="Investigated CI failure; owner approval is required.",
                )
                root_ts = started.get("ts")
            tracked = self.bot._track_agent_run(
                result,
                team_id=team_id or "unknown-team",
                channel_id=channel_id,
                thread_ts=root_ts,
                owner_user_id=user_id,
                mode="ci_repair",
            )
            self.bot._start_agent_monitor(tracked)
            return f"CI repair `{result['run_id']}` is awaiting approval in the thread."

        model = None
        include_thread_context = True
        selected_workflow = None
        selected_task_mode = None
        task_args = list(args)
        while task_args and task_args[0].startswith("--"):
            option = task_args.pop(0)
            if option in {"--discuss", "--implement"}:
                requested_task_mode = option[2:]
                if selected_task_mode and selected_task_mode != requested_task_mode:
                    return "Conflicting task modes: choose --discuss or --implement."
                selected_task_mode = requested_task_mode
                continue
            if option == "--workflow":
                selected_workflow = "structured"
                continue
            if option == "--no-thread-context":
                include_thread_context = False
                continue
            if option == "--model":
                if not task_args:
                    return "Usage: `/llm agent --model <provider/model> <task>`"
                model = task_args.pop(0).strip()
                continue
            if option.startswith("--model="):
                model = option.split("=", 1)[1].strip()
                if model:
                    continue
            return "Usage: `/llm agent [--discuss|--implement] [--workflow] [--model provider/model] [--no-thread-context] <task>`"
        task = " ".join(task_args).strip()
        if not task:
            return "Usage: `/llm agent [--model provider/model] <task>`"

        start = self._bootstrap_and_start_agent(
            task,
            user_id,
            channel_id,
            client,
            team_id=team_id,
            thread_ts=thread_ts,
            message_ts=message_ts,
            request_id=request_id,
            model=model,
            include_thread_context=include_thread_context,
            workflow_mode=selected_workflow,
            task_mode=selected_task_mode,
        )
        if not background_start:
            return await start
        cap_message = self.bot._agent_cap_message(user_id, channel_id, thread_ts)
        if cap_message:
            start.close()
            return cap_message
        self.bot._spawn_background_task(
            self._finish_background_agent_start(start, user_id, channel_id),
            "pi_agent_start",
        )
        return AGENT_STARTING_MESSAGE

    async def _finish_background_agent_start(
        self, start: Awaitable[str], user_id: str, channel_id: str
    ) -> None:
        """Run a deferred agent start and tell the owner how it ended."""
        try:
            response = await start
        except Exception:
            logger.exception("Background Agent start failed")
            response = "Agent mode could not start this task. Chat mode was not used as fallback."
        if response:
            await self.bot._post_command_response(channel_id, user_id, response)

    async def _bootstrap_and_start_agent(
        self,
        task: str,
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        *,
        team_id: Optional[str],
        thread_ts: Optional[str],
        message_ts: Optional[str],
        request_id: Optional[str],
        model: Optional[str],
        include_thread_context: bool,
        workflow_mode: Optional[str],
        task_mode: Optional[str],
    ) -> str:
        bootstrap = AgentBootstrapContext(prompt=task)
        builder = getattr(self.bot, "agent_bootstrap_builder", None)
        if include_thread_context and thread_ts and message_ts and builder is not None:
            is_public_channel = getattr(self.bot, "_is_public_agent_channel", None)
            if not callable(is_public_channel) or not await is_public_channel(
                channel_id
            ):
                # Same egress rule as the shortcut: thread content from private,
                # DM or Slack Connect channels never reaches a model provider.
                bootstrap = AgentBootstrapContext(
                    prompt=task, error="channel_not_public"
                )
            else:
                try:
                    bootstrap = await builder.build(
                        client,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        owner_user_id=user_id,
                        task=task,
                        latest_ts=message_ts,
                    )
                except AgentBootstrapFetchError as exc:
                    return f"Agent mode did not start because Slack thread context could not be loaded: `{exc}`."

        note = (
            f"Agent started without prior Slack thread context: `{bootstrap.error}`."
            if bootstrap.error
            else ""
        )
        response = await self._start_agent_prompt(
            bootstrap.prompt,
            user_id,
            channel_id,
            client,
            team_id=team_id,
            thread_ts=thread_ts,
            model=model,
            routing_text=task,
            request_id=request_id,
            **({"workflow_mode": workflow_mode} if workflow_mode else {}),
            **({"task_mode": task_mode} if task_mode else {}),
            **({"status_note": note} if note else {}),
        )
        if note and response:
            response += f"\n{note}"
        return response

    async def _start_agent_prompt(
        self,
        task: str,
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        *,
        team_id: Optional[str],
        thread_ts: Optional[str],
        model: Optional[str] = None,
        routing_text: Optional[str] = None,
        workflow_mode: Optional[str] = None,
        task_mode: Optional[str] = None,
        request_id: Optional[str] = None,
        status_ts: Optional[str] = None,
        status_note: str = "",
    ) -> str:
        """Start a run unless the owner already has their maximum of active runs.

        The run's status message is the message at `status_ts` when given, the
        new root message of a run started outside a thread, or else a new reply
        in the thread; `status_note` is added to it. Returns "" once the status
        message shows the start, otherwise the reply for the caller to post.
        """
        cap_message = getattr(self.bot, "_agent_cap_message", None)
        reserve = getattr(self.bot, "_reserve_agent_start", None)
        if not callable(cap_message) or not callable(reserve):
            reservation = None
        else:
            # Check and reserve with no await in between, so concurrent starts
            # by one owner cannot both pass the cap.
            refusal = cap_message(user_id, channel_id, thread_ts)
            if refusal:
                return refusal
            reservation = reserve(user_id, channel_id, thread_ts)
        try:
            return await self._start_agent_prompt_unchecked(
                task,
                user_id,
                channel_id,
                client,
                team_id=team_id,
                thread_ts=thread_ts,
                model=model,
                routing_text=routing_text,
                workflow_mode=workflow_mode,
                task_mode=task_mode,
                request_id=request_id,
                status_ts=status_ts,
                status_note=status_note,
            )
        finally:
            if reservation is not None:
                self.bot._release_agent_start(user_id, reservation)

    async def _start_agent_prompt_unchecked(
        self,
        task: str,
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        *,
        team_id: Optional[str],
        thread_ts: Optional[str],
        model: Optional[str] = None,
        routing_text: Optional[str] = None,
        workflow_mode: Optional[str] = None,
        task_mode: Optional[str] = None,
        request_id: Optional[str] = None,
        status_ts: Optional[str] = None,
        status_note: str = "",
    ) -> str:
        runtime = getattr(self.bot, "agent_runtime_client", None)
        if runtime is None:
            return "Agent mode is currently disabled."

        root_ts = thread_ts
        root_message_ts = None
        if not root_ts and client is not None:
            started = await client.chat_postMessage(
                channel=channel_id,
                text="🛠️ Pi coding agent is preparing an isolated worktree…",
            )
            root_ts = started.get("ts")
            root_message_ts = root_ts
        if not root_ts:
            return "Agent mode needs a Slack thread to start this task."

        if task_mode is None:
            # A follow-up never silently upgrades a read-only discussion: the
            # runtime defaults to implement, so carry discuss forward unless
            # the owner explicitly passes `--implement`.
            # Only this thread's own run counts, never the user's latest run
            # elsewhere.
            agent_threads = getattr(self.bot, "agent_threads", None) or {}
            thread_key = getattr(self.bot, "_thread_key", None)
            previous = (
                agent_threads.get(thread_key(channel_id, thread_ts))
                if thread_ts and callable(thread_key)
                else None
            )
            if (
                previous
                and previous.get("owner_user_id") == user_id
                and previous.get("mode") == "discuss"
            ):
                task_mode = "discuss"
        effective_task_mode = task_mode or "implement"
        mode_summary = self._agent_mode_summary(effective_task_mode, workflow_mode)

        try:
            create_args = (
                team_id or "unknown-team",
                channel_id,
                root_ts,
                user_id,
                task,
            )
            create_kwargs = {}
            if task_mode is not None:
                create_kwargs["task_mode"] = task_mode
            if workflow_mode is not None:
                create_kwargs["workflow_mode"] = workflow_mode
            if model:
                create_kwargs["model"] = model
            if routing_text:
                create_kwargs["routing_text"] = routing_text
            if request_id:
                create_kwargs["idempotency_key"] = _agent_idempotency_key(
                    team_id, channel_id, request_id, user_id, "create"
                )
            result = await runtime.create_session(*create_args, **create_kwargs)
        except AgentRuntimeBusy:
            message = (
                "Agent mode is busy right now. Chat mode was not used as fallback."
            )
        except AgentRuntimeTimeout:
            message = "Agent mode timed out while starting. Chat mode was not used as fallback."
        except AgentRuntimeUnavailable:
            message = "Agent mode is temporarily unavailable. Chat mode was not used as fallback."
        except AgentRuntimeUnauthorized as exc:
            message = (
                _agent_auth_error_text(
                    exc, "Only the Agent session owner can continue this thread."
                )
                + " Chat mode was not used as fallback."
            )
        except AgentRuntimeError as exc:
            if exc.code == "idempotency_key_reused":
                message = await self._follow_redelivered_start(
                    team_id or "unknown-team", channel_id, root_ts
                )
            else:
                known = AGENT_START_ERROR_TEXT.get(exc.code or "")
                if known is None and isinstance(exc, AgentRuntimeConflict):
                    known = "This Agent thread cannot accept that request (for example, a different model)."
                if known is None:
                    logger.warning(
                        "Agent runtime could not complete the Slack task (code=%s)",
                        exc.code,
                    )
                    known = "Agent mode could not start this task."
                message = f"{known} Chat mode was not used as fallback."
        else:
            tracked = self.bot._track_agent_run(
                result,
                team_id=team_id or "unknown-team",
                channel_id=channel_id,
                thread_ts=root_ts,
                owner_user_id=user_id,
                mode=effective_task_mode,
            )
            started_text = (
                f"🛠️ Pi coding agent started run `{result['run_id']}`"
                f"{f' with `{model}`' if model else ''}.\n"
                f"{mode_summary}\n"
                + (
                    "Read-only discussion: no edits, verification or commit."
                    if effective_task_mode == "discuss"
                    else (
                        "Edits in this worktree run automatically; risky shell "
                        "commands need approval here, and nothing is committed "
                        "until the owner approves the final diff."
                    )
                )
                + (f"\n{status_note}" if status_note else "")
            )
            target_ts = root_message_ts or status_ts
            if tracked.get("status_ts") and not target_ts:
                # The runtime replayed a start it already accepted: same run,
                # same status message.
                shown = True
            else:
                shown = await _show_agent_status(
                    client, tracked, started_text, target_ts
                )
            self.bot._start_agent_monitor(tracked)
            if shown or root_message_ts:
                return ""
            return f"🛠️ Agent run `{result['run_id']}` started.\n{mode_summary}"

        if root_message_ts and client is not None and hasattr(client, "chat_update"):
            await client.chat_update(
                channel=channel_id, ts=root_message_ts, text=message
            )
            return ""
        return message

    async def _follow_redelivered_start(
        self, team_id: str, channel_id: str, thread_ts: str
    ) -> str:
        """Follow the run a redelivered Slack request already started.

        The runtime refuses the request's key because its body changed (for
        example a newer thread context), so the thread's session names the run
        the first delivery started.
        """
        tracked = await self.bot._adopt_agent_session(team_id, channel_id, thread_ts)
        if tracked and tracked.get("run_id") and not tracked.get("result_posted"):
            return (
                f"🛠️ Agent run `{tracked['run_id']}` is already running for this "
                "request; its progress and approvals will appear in this thread."
            )
        return "Agent mode already received this request, so no second run was started."

    @staticmethod
    def _format_agent_response(
        result: Dict[str, Any], max_length: Optional[int] = None
    ) -> str:
        answer = str(result.get("answer", "")).strip()
        detail_sections = []
        changed_files = [str(item) for item in result.get("changed_files", [])]
        if changed_files:
            detail_sections.append(
                "*Changed files:*\n"
                + "\n".join(f"• `{item}`" for item in changed_files[:50])
            )
        if result.get("diff_stat"):
            detail_sections.append(f"*Diff:* `{result['diff_stat']}`")
        if all(result.get(field) for field in ("mcp_server", "mcp_tool")):
            mcp_status = (
                "succeeded"
                if result.get("mcp_success") is True
                else "failed"
                if result.get("mcp_success") is False
                else "started"
            )
            parts = [f"`{result['mcp_server']}`", f"`{result['mcp_tool']}`"]
            if result.get("mcp_repository"):
                parts.append(f"`{result['mcp_repository']}`")
            detail_sections.append("*MCP:* " + " / ".join(parts) + f" / *{mcp_status}*")
        if result.get("task_mode") == "discuss":
            detail_sections.append(
                "*Mode:* Read-only discussion; no verification or commit."
            )
        if result.get("verification_status") or result.get("grader_status"):
            verification = result.get("verification_status", "pending")
            if verification == "unverified":
                verification = UNVERIFIED_LABEL
            detail_sections.append(
                f"*Verification:* {verification} / "
                f"*Grader:* {result.get('grader_status', 'pending')} / "
                f"*Phase:* {result.get('phase', 'unknown')}"
            )
        if result.get("workflow_mode") == "structured":
            summary = result.get("workflow_summary") or {}
            detail_sections.append(
                f"*Investigation:* {summary.get('completed', 0)}/{summary.get('total', 0)} nodes complete; revision {summary.get('revision', 0)}"
            )
        if result.get("estimated_cost_usd") is not None:
            detail_sections.append(
                f"*Known model cost:* ${result['estimated_cost_usd']:.4f}"
            )
        timing = result.get("timing") or {}
        if timing:
            detail_sections.append(
                f"*Time:* wall {timing.get('wall_ms', 0) / 1000:.1f}s / "
                f"active {timing.get('active_ms', 0) / 1000:.1f}s / "
                f"human wait {timing.get('human_wait_ms', 0) / 1000:.1f}s / "
                f"suspended {result.get('suspended_ms', 0) / 1000:.1f}s"
            )
        findings = [
            finding
            for finding in result.get("grader_findings") or []
            if isinstance(finding, dict)
        ]
        if findings:
            detail_sections.append(
                "*Grader findings:*\n"
                + "\n".join(_format_grader_finding(finding) for finding in findings[:5])
            )
        delivery = []
        if result.get("branch") and result.get("task_mode") != "discuss":
            delivery.append(f"Branch: `{result['branch']}`")
        if result.get("commit"):
            delivery.append(f"Commit: `{result['commit']}`")
        if result.get("cherry_pick"):
            delivery.append(f"Apply: `{result['cherry_pick']}`")
        if delivery:
            detail_sections.append("*Delivery:*\n" + "\n".join(delivery))

        details = "\n\n".join(detail_sections)
        status_line = _format_agent_status_line(result)
        if not answer and not details and not status_line:
            answer = "_The agent finished without producing an answer._"
        if status_line:
            if max_length is not None:
                status_line = status_line[: max(1, max_length)]
                max_length = max(0, max_length - len(status_line) - 2)
            body = SlackMessageHandler._fit_agent_response(answer, details, max_length)
            return "\n\n".join(section for section in (status_line, body) if section)
        return SlackMessageHandler._fit_agent_response(answer, details, max_length)

    @staticmethod
    def _fit_agent_response(
        answer: str, details: str, max_length: Optional[int]
    ) -> str:
        response = "\n\n".join(section for section in (answer, details) if section)
        if max_length is None or len(response) <= max_length:
            return response

        if details and len(details) < max_length:
            available_answer_chars = max_length - len(details) - 2
            if available_answer_chars > 1:
                truncated_answer = answer[: available_answer_chars - 1].rstrip() + "…"
                return f"{truncated_answer}\n\n{details}"
            return details[:max_length]
        return response[: max(0, max_length - 1)].rstrip() + "…"

    @staticmethod
    def _format_agent_status(run: Dict[str, Any]) -> str:
        model = "/".join(
            item for item in (run.get("provider"), run.get("model")) if item
        )
        suffix = f" using `{model}`" if model else ""
        return f"Agent run `{run.get('run_id', 'unknown')}` is *{run.get('status', 'unknown')}*{suffix}."

    async def _handle_help_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle help command"""
        lines = [
            "🤖 *LLM Router Bot Help*",
            "",
            "*Basic Usage:*",
            "Mention me in a channel, use `/llm ...`, or reply inside an active bot thread.",
            "",
            "*Commands:*",
            "• `/llm help` - Show this help message",
            "• `/llm settings` - View/update your preferences",
            "• `/llm status` - Show system status and your usage",
            "• `/llm models` - List available models and their capabilities",
            "• `/llm analytics` - Show usage analytics (premium users)",
            "• `/llm clear` - Clear conversation history",
            "• `/llm web <query>` - Search the web before answering",
            "• `/llm fast <query>` - Prefer an explicit low-latency route for this query",
        ]
        if self.bot.agent_runtime_client is not None:
            lines.extend(
                [
                    "• `/llm agent <task>` - Start a stateful Pi coding agent with the default model",
                    "• `/llm agent --discuss <question>` - Discuss the project with read-only tools; no verification or commit",
                    "• `/llm agent --implement <task>` - Implement changes with the configured verification gates",
                    "• `/llm agent --workflow <task>` - Run the structured investigation before the writer",
                    "• `/llm agent --model <provider/model> <task>` - Select an Agent model (`--model=<provider/model>` also works)",
                    "• Use the message shortcut *Run Pi Agent* to start from an existing Slack thread",
                    "• `/llm agent --no-thread-context <task>` - Start without loading prior thread messages",
                    f"• {await self._agent_models_text()}",
                    "• `/llm agent /skill:test-gap <task>` - Run the approved test-gap workflow",
                    "• `/llm agent status|stop|close` - Inspect or control your latest Agent session",
                    "• `/llm agent resume <RUN_ID>` - Resume an interrupted structured workflow",
                    "• `/llm agent repair-ci [--workflow] <PR_URL|RUN_ID>` - Investigate a failed CI run and propose a repair for approval",
                    "• `/llm agent publish-repair <RUN_ID>` - Ask again to publish a CI repair whose publication failed",
                    "• Reply in an Agent thread to continue it; start the reply with `--discuss`, `--implement` or `--workflow` to choose its mode",
                ]
            )
        lines.extend(
            [
                "• `/llm remember [--global] <text>` - Save an explicit long-term memory (`--global` for every channel)",
                "• `/llm memories [--all] [--global] [query]` - List or search your memories",
                "• `/llm forget <memory_id|all>` - Delete saved memories",
                "",
                "*Examples:*",
                '• "Write a Python function to calculate fibonacci numbers"',
                '• "Explain quantum computing in simple terms"',
                '• "Analyze this CSV data: [attach file]"',
                '• "Help me plan a project roadmap"',
                "",
                "*Features:*",
                "✅ Intelligent model routing",
                "✅ Conversation continuity",
                "✅ Context compression for long conversations",
                "✅ Response caching for faster replies",
                "✅ Usage analytics and cost tracking",
                "",
                "*User Tiers:*",
                getattr(self.bot, "_format_tier_limit_summary", lambda: "")(),
            ]
        )
        return "\n".join(lines)

    async def _agent_models_text(self) -> str:
        """Name the Agent models a fresh runtime health check reports.

        A cached allowlist may be stale, so without an answer say so instead.
        """
        runtime = getattr(self.bot, "agent_runtime_client", None)
        health_details = getattr(runtime, "health_details", None)
        details = await health_details() if callable(health_details) else None
        if not isinstance(details, dict):
            return "Agent models are unavailable: the Agent runtime did not answer its health check."
        models = configured_model_refs(details)
        if not models:
            return "Agent models: none are configured on the Agent runtime."
        return "Agent models: " + ", ".join(f"`{model}`" for model in models)

    async def _handle_settings_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle settings command"""
        user_prefs = self.bot.user_manager.get_user_preferences(user_id)
        user_tier = self.bot.user_manager.get_user_tier(user_id)

        if not args:
            # Show current settings
            settings_text = f"""
🔧 *Your Settings*

*User Tier:* {user_tier.value.title()}
*Response Length:* {user_prefs.get('response_length', 'medium')}
*Technical Level:* {user_prefs.get('technical_level', 'intermediate')}
*Threading:* {'Enabled' if user_prefs.get('threading', True) else 'Disabled'}
*Auto Web Search:* {'Enabled' if user_prefs.get('web_search_auto', True) else 'Disabled'}
*Preferred Models:* {', '.join(user_prefs.get('preferred_models', [])) or 'Auto-select'}

*Update Settings:*
• `/llm settings response_length short|medium|long`
• `/llm settings technical_level beginner|intermediate|expert`
• `/llm settings threading on|off`
• `/llm settings web_search_auto on|off`
• `/llm settings preferred_models model1,model2`
            """
            return settings_text.strip()

        elif len(args) >= 2:
            # Update setting
            setting_name = args[0]
            setting_value = " ".join(args[1:])

            if setting_name == "response_length" and setting_value in [
                "short",
                "medium",
                "long",
            ]:
                user_prefs["response_length"] = setting_value
            elif setting_name == "technical_level" and setting_value in [
                "beginner",
                "intermediate",
                "expert",
            ]:
                user_prefs["technical_level"] = setting_value
            elif setting_name == "threading":
                user_prefs["threading"] = setting_value.lower() in ["on", "true", "yes"]
            elif setting_name == "web_search_auto":
                user_prefs["web_search_auto"] = setting_value.lower() in [
                    "on",
                    "true",
                    "yes",
                ]
            elif setting_name == "preferred_models":
                normalized_value = setting_value.strip().lower()
                if normalized_value in {"auto", "none", "clear"}:
                    models = []
                else:
                    models = [m.strip() for m in setting_value.split(",") if m.strip()]
                    if not models:
                        return "Invalid preferred_models value. Use a comma-separated model list or `auto` to clear."

                    available_models = set(
                        getattr(getattr(self.bot, "router", None), "models", {})
                    )
                    if available_models:
                        unknown_models = [
                            model_name
                            for model_name in models
                            if model_name not in available_models
                        ]
                        if unknown_models:
                            return "Unknown model(s): " + ", ".join(
                                f"`{model_name}`" for model_name in unknown_models
                            )
                user_prefs["preferred_models"] = models
            else:
                return f"Invalid setting: `{setting_name}` or value: `{setting_value}`"

            self.bot.user_manager.update_user_preferences(user_id, user_prefs)
            persist_user_state = getattr(self.bot, "_persist_user_state", None)
            if persist_user_state is not None:
                await persist_user_state(user_id)
            return f"✅ Updated {setting_name} to: {setting_value}"

        else:
            return "Usage: `/llm settings [setting_name] [value]` or `/llm settings` to view current settings"

    async def _handle_status_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle status command"""
        try:
            # Get system status
            system_health = await self.bot.get_system_status()

            # Get user stats
            user_stats = await self.bot.get_user_stats(user_id)
            runtime = getattr(self.bot, "agent_runtime_client", None)
            agent_models = ""
            if runtime is None:
                agent_health = "⚪ Disabled"
            else:
                health_details = getattr(runtime, "health_details", None)
                if health_details is not None:
                    details = await health_details()
                    agent_health = (
                        "🟢 Healthy"
                        if details and details.get("status") == "healthy"
                        else "🔴 Unavailable"
                    )
                    configured = configured_model_refs(details)
                    if configured:
                        agent_models = f"\n*Agent Models:* {', '.join(f'`{model}`' for model in configured)}"
                else:
                    agent_health = (
                        "🟢 Healthy" if await runtime.health() else "🔴 Unavailable"
                    )

            status_text = f"""
📊 *System Status*

*Health:* {'🟢 Healthy' if system_health.get('healthy', False) else '🔴 Issues Detected'}
*Active Models:* {len(system_health.get('available_models', []))}
*Response Time:* {system_health.get('avg_response_time', 0):.0f}ms
*Uptime:* {system_health.get('uptime', 'Unknown')}
*Agent Runtime:* {agent_health}
{agent_models}

*Your Usage (Last 24h):*
📝 Queries: {user_stats.get('queries_24h', 0)}
💰 Cost: ${user_stats.get('cost_24h', 0):.4f}
⚡ Avg Latency: {user_stats.get('avg_latency', 0):.0f}ms
🎯 Success Rate: {user_stats.get('success_rate', 0):.1f}%

*Rate Limits:*
Remaining this hour: {user_stats.get('remaining_requests', 0)}
            """
            return status_text.strip()

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return "❌ Error retrieving status information"

    async def _handle_models_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle models command"""
        try:
            available_models = await self.bot.get_available_models(user_id)

            models_text = "🤖 *Available Models*\n\n"

            for model in available_models:
                status_emoji = "🟢" if model.get("available", False) else "🔴"
                cost_info = (
                    f"${model.get('cost_per_1k_tokens', 0):.4f}/1K tokens"
                    if model.get("cost_per_1k_tokens")
                    else "Free"
                )

                models_text += (
                    f"""
{status_emoji} *{model['name']}*
• Provider: {model.get('provider', 'Unknown')}
• Capabilities: {', '.join(model.get('capabilities', []))}
• Max Tokens: {model.get('max_tokens', 'Unknown'):,}
• Cost: {cost_info}
• Best For: {model.get('description', 'General use')}
                """.strip()
                    + "\n\n"
                )

            if getattr(self.bot, "agent_runtime_client", None) is not None:
                models_text += f"*{await self._agent_models_text()}*\n\n"

            models_text += """
*Auto-Routing:* The bot automatically selects the best model for your query based on:
• Query type and complexity
• Your user tier and preferences  
• Model availability and performance
• Cost optimization
            """

            return models_text.strip()

        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return "❌ Error retrieving model information"

    async def _handle_analytics_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle analytics command"""
        user_tier = self.bot.user_manager.get_user_tier(user_id)

        if user_tier == UserTier.FREE:
            return "📊 Analytics are available for Premium and Enterprise users. Upgrade to access detailed usage analytics!"

        try:
            analytics = await self.bot.get_user_analytics(user_id)

            analytics_text = f"""
📈 *Your Analytics (Last 7 Days)*

*Usage Overview:*
• Total Queries: {analytics.get('total_queries', 0)}
• Total Tokens: {analytics.get('total_tokens', 0):,}
• Total Cost: ${analytics.get('total_cost', 0):.4f}
• Daily Average: {analytics.get('daily_avg_queries', 0):.1f} queries

*Performance:*
• Avg Response Time: {analytics.get('avg_latency', 0):.0f}ms
• Success Rate: {analytics.get('success_rate', 0):.1f}%
• Cache Hit Rate: {analytics.get('cache_hit_rate', 0):.1f}%

*Model Usage:*
            """

            for model, usage in analytics.get("model_breakdown", {}).items():
                analytics_text += f"• {model}: {usage.get('queries', 0)} queries (${usage.get('cost', 0):.4f})\n"

            analytics_text += f"""

*Query Types:*
            """

            for query_type, count in analytics.get("query_type_breakdown", {}).items():
                analytics_text += f"• {query_type.replace('_', ' ').title()}: {count}\n"

            return analytics_text.strip()

        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return "❌ Error retrieving analytics data"

    async def _handle_clear_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle clear conversation command"""
        cleared_keys = self.bot.conversation_manager.clear_contexts(
            user_id, channel_id, return_keys=True
        )

        if not cleared_keys:
            return "🧹 No saved conversation history was found for this channel."

        delete_conversations = getattr(self.bot, "_delete_conversation_states", None)
        if delete_conversations is not None:
            await delete_conversations(cleared_keys)
        return "🧹 Conversation history cleared for this channel."

    async def _handle_remember_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        team_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
        command_surface: str = "message",
    ) -> str:
        """Store an explicit long-term memory for this Slack user."""
        if command_surface != "slash":
            return MEMORY_SLASH_ONLY_MESSAGE

        memory_manager = getattr(self.bot, "memory_manager", None)
        if not memory_manager or not getattr(memory_manager, "enabled", False):
            return "Memory is not enabled for this Slack bot."

        visibility = "channel"
        text_args = list(args)
        if text_args and text_args[0] == "--global":
            visibility = "global"
            text_args = text_args[1:]

        text = " ".join(text_args).strip()
        if not text:
            return "Usage: `/llm remember [--global] <something useful to remember>`"

        scope = self.bot._memory_scope(team_id, user_id)
        metadata = self.bot._build_memory_metadata(
            channel_id=channel_id,
            thread_ts=thread_ts,
            visibility=visibility,
        )
        try:
            item = await memory_manager.remember(scope, text, metadata=metadata)
        except ValueError as exc:
            return f"Could not save memory: {exc}"
        except Exception:
            logger.exception("Failed to save Slack memory")
            return "Could not save memory right now."

        if item is None:
            return "Could not save an empty memory."
        return f"Saved memory `{item.memory_id}`."

    async def _handle_memories_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        team_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
        command_surface: str = "message",
    ) -> str:
        """List or search this Slack user's long-term memories."""
        if command_surface != "slash":
            return MEMORY_SLASH_ONLY_MESSAGE

        memory_manager = getattr(self.bot, "memory_manager", None)
        if not memory_manager or not getattr(memory_manager, "enabled", False):
            return "Memory is not enabled for this Slack bot."

        scope = self.bot._memory_scope(team_id, user_id)
        include_all = False
        global_only = False
        query_args = []
        for arg in args:
            if arg == "--all":
                include_all = True
            elif arg == "--global":
                global_only = True
            else:
                query_args.append(arg)
        query = " ".join(query_args).strip()
        metadata = self._memory_visibility_metadata(
            channel_id=channel_id,
            include_all=include_all,
            global_only=global_only,
        )
        try:
            if query:
                results = await memory_manager.search(scope, query, metadata=metadata)
                items = [result.item for result in results]
            else:
                items = await memory_manager.list_memories(
                    scope, limit=50, metadata=metadata
                )
        except Exception:
            logger.exception("Failed to read Slack memories")
            return "Could not read memories right now."

        if not items:
            return "No memories found."

        lines = ["Your memories:"]
        for item in items[:10]:
            preview = " ".join(item.text.split())
            if len(preview) > 120:
                preview = preview[:120].rstrip() + "..."
            visibility = item.metadata.get("visibility", "channel")
            lines.append(f"- `{item.memory_id}` [{visibility}] {preview}")
        return "\n".join(lines)

    async def _handle_forget_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        team_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
        command_surface: str = "message",
    ) -> str:
        """Delete one or all long-term memories for this Slack user."""
        if command_surface != "slash":
            return MEMORY_SLASH_ONLY_MESSAGE

        memory_manager = getattr(self.bot, "memory_manager", None)
        if not memory_manager or not getattr(memory_manager, "enabled", False):
            return "Memory is not enabled for this Slack bot."

        if not args:
            return "Usage: `/llm forget <memory_id>` or `/llm forget all`"

        scope = self.bot._memory_scope(team_id, user_id)
        target = " ".join(args).strip()
        try:
            if target.lower() == "all":
                deleted_count = await memory_manager.forget_all(scope)
                return f"Forgot {deleted_count} memories."

            deleted = await memory_manager.forget(scope, target)
        except Exception:
            logger.exception("Failed to delete Slack memory")
            return "Could not delete memory right now."

        if not deleted:
            return f"No memory found for `{target}`."
        return f"Forgot memory `{target}`."

    def _memory_visibility_metadata(
        self,
        channel_id: Optional[str],
        include_all: bool,
        global_only: bool,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"source": "slack"}
        if include_all:
            return metadata
        if global_only:
            metadata["visibility"] = "global"
            return metadata
        metadata["visibility_scope"] = "channel_or_global"
        if channel_id:
            metadata["channel_id"] = channel_id
        return metadata

    async def _handle_query(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        thread_ts: str,
        client: AsyncWebClient,
        attachments: Optional[List[Attachment]] = None,
        team_id: Optional[str] = None,
        tool_policy: ToolPolicy = ToolPolicy.AUTO,
        allowed_tools: Optional[List[str]] = None,
        routing_metadata: Optional[Dict[str, Any]] = None,
        message_ts: Optional[str] = None,
    ) -> str:
        """Handle regular query through inference engine"""
        try:
            # Get or create conversation context
            context = self.bot.conversation_manager.get_or_create_context(
                user_id, channel_id, thread_ts
            )

            # Get user tier and preferences
            user_tier = self.bot.user_manager.get_user_tier(user_id)
            user_prefs = self.bot.user_manager.get_user_preferences(user_id)
            context.user_tier = user_tier
            context.preferences = user_prefs

            # Build request-scoped Slack and bot conversation context.
            conversation_context = (
                self.bot.conversation_manager.get_conversation_summary(
                    user_id, channel_id, thread_ts
                )
            )
            slack_context = SlackFetchedContext()
            slack_context_builder = getattr(self.bot, "_build_slack_context", None)
            if callable(slack_context_builder):
                slack_context = await slack_context_builder(
                    client=client,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    message_ts=message_ts,
                )

            request_context = self._join_context_sections(
                [slack_context.context, conversation_context]
            )
            memory_manager = getattr(self.bot, "memory_manager", None)
            if memory_manager and getattr(memory_manager, "enabled", False):
                scope = self.bot._memory_scope(team_id, user_id)
                try:
                    memory_results = await memory_manager.search(
                        scope,
                        text,
                        metadata=self._memory_visibility_metadata(
                            channel_id=channel_id,
                            include_all=False,
                            global_only=False,
                        ),
                    )
                    request_context = memory_manager.build_context(
                        memory_results,
                        request_context,
                    )
                except Exception:
                    logger.exception("Failed to search Slack memory")
                    request_context = self._join_context_sections(
                        [slack_context.context, conversation_context]
                    )

            # Create query request
            conversation_id_builder = getattr(
                self.bot, "_conversation_context_key", None
            )
            if callable(conversation_id_builder):
                conversation_id = conversation_id_builder(
                    user_id=user_id,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                )
            else:
                conversation_id = f"{user_id}:{channel_id}:{thread_ts or 'main'}"

            query_metadata = (
                self.bot._build_query_metadata(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    attachments=attachments or [],
                    user_preferences=user_prefs,
                    team_id=team_id,
                )
                if hasattr(self.bot, "_build_query_metadata")
                else {}
            )
            query_metadata.update(dict(routing_metadata or {}))
            self._add_slack_context_metadata(query_metadata, slack_context)

            query_request = QueryRequest(
                query=text,
                user_id=user_id,
                user_tier=user_tier,
                context=request_context,
                max_tokens=self._get_max_tokens_for_user(user_tier, user_prefs),
                temperature=0.7,
                attachments=attachments or [],
                tool_policy=tool_policy,
                allowed_tools=allowed_tools or [],
                session_id=context.session_id,
                conversation_id=conversation_id,
                metadata=query_metadata,
            )

            # Process through inference engine
            response = await self.bot.inference_engine.process_query(query_request)

            if getattr(response, "error", None):
                logger.error(
                    "Slack query failed for user %s in channel %s: %s",
                    user_id,
                    channel_id,
                    response.error,
                )
                SLACK_METRICS.errors.labels(error_type="inference_error_response").inc()
                return self._inference_error_message(
                    getattr(response, "error_type", None),
                    getattr(response, "error_code", None),
                    retry_after_seconds=getattr(response, "retry_after_seconds", None),
                )

            warnings = getattr(response, "warnings", None)
            if not (getattr(response, "response_text", "") or "").strip():
                # Nothing to post or remember: an empty turn would only
                # mislead the next request's context.
                logger.warning(
                    "Slack query for user %s in channel %s got an empty answer "
                    "from %s",
                    user_id,
                    channel_id,
                    getattr(response, "model_name", "unknown"),
                )
                SLACK_METRICS.errors.labels(error_type="empty_response").inc()
                notes = _response_warning_notes(warnings, skip=("empty_response",))
                return "\n\n".join(filter(None, [SLACK_EMPTY_ANSWER_MESSAGE, notes]))

            # A degraded answer says so, in Slack after it and in history
            # before it, so the conversation never treats a cut-off answer
            # as complete: the summary keeps each message's first 200 chars.
            answer = response.response_text
            notes = _response_warning_notes(warnings)
            label = _response_warning_label(warnings)
            if notes:
                answer = f"{answer.rstrip()}\n\n{notes}"

            # Add to conversation history
            context.add_message("user", text)
            context.add_message(
                "assistant",
                f"{label} {response.response_text}" if label else answer,
            )

            # Update metrics
            SLACK_METRICS.messages_processed.labels(user_tier=user_tier.value).inc()
            SLACK_METRICS.response_time.observe(response.latency_ms / 1000)

            return self._format_response_with_sources(response, answer)

        except AdmissionRejectedError as e:
            logger.warning(
                "Slack query for user %s was not admitted: %s",
                user_id,
                e.decision.reason,
            )
            SLACK_METRICS.errors.labels(error_type=type(e).__name__).inc()
            return _admission_rejection_text(
                e.decision, self.bot.user_manager.get_user_tier(user_id)
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            SLACK_METRICS.errors.labels(error_type=type(e).__name__).inc()
            return self._inference_error_message(None)

    def _inference_error_message(
        self,
        error_type: Optional[str],
        error_code: Optional[str] = None,
        *,
        retry_after_seconds: Any = None,
    ) -> str:
        """What a failed answer tells the user: by error code, else error type."""
        wait = _retry_wait(retry_after_seconds)
        if error_code == "rate_limited" and error_type != "rate_limited":
            # Our own capacity, e.g. an admission-rejected primary whose
            # fallbacks failed too: not the provider's 429.
            return _busy_text(wait)
        for messages, key in (
            (SLACK_ERROR_CODE_MESSAGES, error_code),
            (SLACK_INFERENCE_ERROR_MESSAGES, error_type),
        ):
            message = messages.get(key) if isinstance(key, str) else None
            if message == _RATE_LIMITED_TEXT and wait:
                return (
                    "⏳ The model provider is rate-limiting requests right now. "
                    f"Please try again in {wait}."
                )
            if message:
                return message
        error_builder = getattr(self.bot, "_build_user_safe_error_message", None)
        if callable(error_builder):
            return error_builder()
        return "❌ Sorry, I couldn't process that request right now. Please try again in a moment."

    def _join_context_sections(self, sections: List[str]) -> str:
        """Join non-empty context sections without adding persistence semantics."""
        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _add_slack_context_metadata(
        self, metadata: Dict[str, Any], slack_context: SlackFetchedContext
    ) -> None:
        """Attach Slack history fetch metadata only when a fetch happened or failed."""
        if slack_context.source != "none" or slack_context.error:
            metadata["slack_context_source"] = slack_context.source
            metadata["slack_context_message_count"] = slack_context.message_count
            metadata["slack_context_truncated"] = slack_context.truncated
        if slack_context.error:
            metadata["slack_context_error"] = slack_context.error

    def _format_response_with_sources(
        self, response: Any, response_text: Optional[str] = None
    ) -> str:
        """Append compact source links for web-search enriched answers.

        `response_text` replaces the response's own text, e.g. with notes added.
        """
        if response_text is None:
            response_text = getattr(response, "response_text", "") or ""
        sources = list(getattr(response, "sources", []) or [])
        if sources:
            lines = [response_text.rstrip(), "", "Sources:"]
            for source in sources[:5]:
                title = getattr(source, "title", "") or getattr(source, "url", "")
                url = getattr(source, "url", "")
                rank = getattr(source, "rank", len(lines))
                if not url:
                    continue
                lines.append(f"[{rank}] {title} - {url}")
            response_text = "\n".join(lines).strip()
        model_name = getattr(response, "model_name", None)
        latency_ms = getattr(response, "latency_ms", None)
        model_text = (getattr(response, "response_text", "") or "").rstrip()
        trailer = (
            response_text[len(model_text) :]
            if model_text and response_text.startswith(model_text)
            else ""
        )
        return SlackAnswer(
            response_text,
            trailer_lines=trailer.split("\n"),
            model_name=model_name if isinstance(model_name, str) else None,
            latency_ms=(
                float(latency_ms)
                if isinstance(latency_ms, (int, float))
                and not isinstance(latency_ms, bool)
                else None
            ),
            cached=getattr(response, "cached", False) is True,
        )

    def _get_max_tokens_for_user(
        self, user_tier: UserTier, preferences: Dict[str, Any]
    ) -> int:
        """Get max tokens based on user tier and preferences"""
        base_tokens = {
            UserTier.FREE: 1000,
            UserTier.PREMIUM: 4000,
            UserTier.ENTERPRISE: 8000,
        }

        length_multiplier = {"short": 0.5, "medium": 1.0, "long": 2.0}

        base = base_tokens.get(user_tier, 1000)
        multiplier = length_multiplier.get(
            preferences.get("response_length", "medium"), 1.0
        )

        return min(int(base * multiplier), MAX_QUERY_TOKENS)

    def _build_attachment_only_query(self, attachments: List[Attachment]) -> str:
        """Create a fallback query when a Slack message only contains files."""
        return self._files_only_query([attachment.name for attachment in attachments])

    @staticmethod
    def _files_only_query(names: List[str]) -> str:
        attachment_names = ", ".join(name for name in names[:3] if name)
        if attachment_names:
            return f"Analyze the attached Slack file(s): {attachment_names}"
        return "Analyze the attached Slack file(s)."


class SlackBot:
    """Main Slack bot class"""

    def __init__(
        self,
        config: Dict[str, Any],
        inference_engine,
        router=None,
        monitoring_service=None,
        analytics_service=None,
        services: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        # Agent settings live in the top-level `agent` section, not under `slack`.
        self.agent_config = dict(
            agent_config if agent_config is not None else config.get("agent") or {}
        )
        self.inference_engine = inference_engine
        resolved_services = services or {}
        self.services = resolved_services
        self.router = (
            router
            or resolved_services.get("router")
            or getattr(inference_engine, "router", None)
        )
        self.monitoring_service = monitoring_service or resolved_services.get(
            "monitoring"
        )
        self.analytics_service = analytics_service or resolved_services.get("pipeline")
        self.agent_runtime_client = resolved_services.get("agent_runtime")

        tier_settings = config.get("user_tiers", {})
        default_user_tier = tier_settings.get("default", UserTier.FREE.value)
        tier_overrides = tier_settings.get("overrides", {})

        # Initialize components
        self.user_manager = UserManager(
            default_tier=default_user_tier,
            tier_overrides=tier_overrides,
        )
        self.conversation_manager = ConversationManager(config)
        self.memory_manager = MemoryManager(config.get("memory", {}))
        self.message_handler = SlackMessageHandler(self)

        # Slack clients
        self.web_client = None
        self.socket_client = None
        self.bot_token = None

        # Bot configuration
        self.bot_user_id = None
        self.allowed_channels = config.get("channels", [])
        self.allowed_channel_ids: Set[str] = set()
        self.allowed_channel_names: Set[str] = set()
        self.rate_limiting = config.get("rate_limiting", {})
        self.context_settings = self._resolve_context_settings(
            config.get("context", {})
        )
        self.agent_context_settings = dict(config.get("agent_context", {}) or {})
        self.attachment_settings = {
            "enabled": True,
            "max_files": 10,
            "max_file_size_bytes": 10_000_000,
            "download_timeout_seconds": 30,
        }
        self.attachment_settings.update(config.get("attachments", {}))
        # Created on the first attachment download and closed on shutdown.
        self._attachment_http_client: Optional[httpx.AsyncClient] = None
        self.agent_bootstrap_builder = AgentBootstrapBuilder(
            self.agent_context_settings,
            bot_token=self._resolve_secret("bot_token", "bot_token_env"),
        )
        self.state_store = build_slack_state_store(
            resolve_slack_state_store_config(config)
        )
        self.state_lock = asyncio.Lock()
        self.active_threads: Dict[str, datetime] = {}
        self.active_thread_timeout = self.conversation_manager.session_timeout
        self.agent_threads: Dict[str, Dict[str, Any]] = {}
        self.agent_runs: Dict[str, Dict[str, Any]] = {}
        # Each owner's tracked sessions, one per thread, the latest last.
        self.agent_user_runs: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_monitoring: Set[str] = set()
        # Ordered so the oldest claims are evicted first.
        self.posted_ci_jobs: Dict[str, None] = {}
        # "channel:ts" of messages already handled, oldest first. Slack sends
        # both a message and an app_mention event for one mention.
        self.handled_messages: Dict[str, None] = {}
        # "channel:ts" of messages already given a busy reply, so a mention's
        # message and app_mention events get one between them.
        self.busy_replied_messages: Dict[str, None] = {}
        self.agent_monitor_tasks: Dict[str, asyncio.Task] = {}
        self.agent_monitor_poll_seconds = float(
            self.agent_config.get("monitor_poll_seconds", 2)
        )
        self.agent_progress_interval_seconds = float(
            self.agent_config.get("progress_interval_seconds", 10)
        )
        # Wall-clock seconds, so elapsed time compares with the run's created_at.
        self.agent_progress_clock: Callable[[], float] = time.time
        # How long a monitor keeps reattaching to a run it cannot reach: the
        # runtime's own deadline (PI_AGENT_DEADLINE_MS) bounds the run anyway.
        self.agent_run_deadline_seconds = float(
            self.agent_config.get("run_deadline_seconds", 900)
        )
        self.agent_monitor_reattach_seconds = float(
            self.agent_config.get("monitor_reattach_seconds", 5)
        )
        self.agent_monitor_reattach_max_seconds = float(
            self.agent_config.get("monitor_reattach_max_seconds", 60)
        )
        self.agent_monitor_clock: Callable[[], float] = time.monotonic
        self.agent_monitor_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self.agent_monitor_random: Callable[[], float] = random.random
        self.agent_max_active_runs_per_user = max(
            1, int(self.agent_config.get("max_active_runs_per_user", 1))
        )
        # Starts that passed the cap but have no run yet: owner -> token -> thread key.
        self.agent_pending_starts: Dict[str, Dict[str, Optional[str]]] = {}
        self.agent_shortcut_nonces: Dict[str, AgentShortcutNonce] = {}
        self.agent_feedback_nonces: Dict[str, AgentFeedbackNonce] = {}

        # Running state
        self.initialized = False
        self.running = False
        self.started_at: Optional[datetime] = None
        self.background_tasks: Set[asyncio.Task] = set()
        work_queue_config = dict(config.get("work_queue", {}) or {})
        self._work_queue = asyncio.Queue(
            maxsize=max(1, int(work_queue_config.get("capacity", 256)))
        )
        self._work_concurrency = max(1, int(work_queue_config.get("concurrency", 16)))
        # Button clicks get their own queue and workers, so approvals still go
        # through while every regular worker waits on a slow agent start.
        self._priority_queue: asyncio.Queue = asyncio.Queue(
            maxsize=max(1, int(work_queue_config.get("capacity", 256)))
        )
        self._priority_concurrency = max(
            1, int(work_queue_config.get("priority_concurrency", 2))
        )
        self._overload_reply_timeout_seconds = float(
            work_queue_config.get("overload_reply_timeout_seconds", 2)
        )
        self._work_workers: List[asyncio.Task] = []
        self._work_running = 0
        self._accepting_work = True
        self._socket_disconnect_lock = asyncio.Lock()
        self._socket_disconnected = False

    async def initialize(self):
        """Initialize Slack bot"""
        try:
            # Initialize Slack clients
            bot_token = self._resolve_secret("bot_token", "bot_token_env")
            app_token = self._resolve_secret("app_token", "app_token_env")

            if not bot_token or not app_token:
                raise ValueError(
                    "Slack bot_token/app_token are required directly or via bot_token_env/app_token_env"
                )

            self.web_client = AsyncWebClient(token=bot_token)
            _add_rate_limit_retry(self.web_client)
            self.bot_token = bot_token
            self.socket_client = AsyncSocketModeClient(
                app_token=app_token, web_client=self.web_client
            )

            # Get bot user ID
            auth_response = await self.web_client.auth_test()
            self.bot_user_id = auth_response["user_id"]
            self.agent_bootstrap_builder = AgentBootstrapBuilder(
                self.agent_context_settings,
                bot_user_id=self.bot_user_id,
                bot_token=bot_token,
            )

            # Register event handlers
            self.socket_client.socket_mode_request_listeners.append(
                self._handle_socket_mode_request
            )
            await self.state_store.initialize()
            await self._restore_state()
            try:
                await self.memory_manager.initialize()
            except Exception:
                logger.exception("Slack memory initialization failed; disabling memory")
                self.memory_manager.enabled = False
            else:
                self._warn_if_memory_uses_shared_redis()
            await self._resolve_allowed_channels()
            await self._refresh_agent_model_cache()
            self.initialized = True

            logger.info(
                f"Slack bot initialized successfully. Bot ID: {self.bot_user_id}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Slack bot: {e}")
            raise

    async def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack bot...")
        self.running = True
        self.started_at = datetime.now()

        try:
            # Start socket mode client
            await self.socket_client.connect()
            self._start_work_workers()
            self._spawn_background_task(
                self._refresh_agent_model_cache_periodically(),
                "slack_agent_model_cache_refresh",
            )

            self._spawn_background_task(
                self._monitor_ci_inbox(), "slack_ci_repair_inbox"
            )
            self._reconcile_agent_runs()

            # Start cleanup task
            self._spawn_background_task(
                self._cleanup_sessions_periodically(), "slack_state_cleanup"
            )

            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Slack bot error: {e}")
            raise
        finally:
            await self._disconnect_socket_once()

    async def _monitor_ci_inbox(self):
        while self.running:
            await self._poll_ci_inbox_once()
            await asyncio.sleep(5)

    async def _poll_ci_inbox_once(self):
        runtime = self.agent_runtime_client
        if runtime is None:
            return
        try:
            jobs = await runtime.ci_repairs()
        except AgentRuntimeError:
            logger.warning("CI repair inbox unavailable")
            return
        for job in jobs:
            try:
                await self._post_ci_inbox_job(job)
            except Exception:
                logger.warning(
                    "CI repair Slack notification failed for %s", job.get("run_id")
                )

    async def _post_ci_inbox_job(self, job: Dict[str, Any]):
        run_id = job["run_id"]
        if run_id in self.agent_runs:
            await self._claim_ci_job(run_id)
            return
        if not await self._claim_ci_job(run_id):
            return
        try:
            message = await self.web_client.chat_postMessage(
                channel=job["channel_id"],
                client_msg_id=run_id,
                text=f"CI repair `{run_id}` requires the repository owner's review.",
            )
        except Exception:
            await self._release_ci_job(run_id)
            raise
        tracked = self._track_agent_run(
            job,
            team_id=job["team_id"],
            channel_id=job["channel_id"],
            thread_ts=message["ts"],
            owner_user_id=job["owner_user_id"],
            mode="ci_repair",
        )
        self._start_agent_monitor(tracked)

    async def _claim_ci_job(self, run_id: str) -> bool:
        """Claim a CI repair job for posting; False when it was already claimed."""
        if run_id in self.posted_ci_jobs:
            return False
        # Claim synchronously so the inbox and /llm agent repair-ci cannot
        # both pass this check before either one awaits.
        self.posted_ci_jobs[run_id] = None
        evicted = []
        while len(self.posted_ci_jobs) > POSTED_CI_JOBS_LIMIT:
            oldest = next(iter(self.posted_ci_jobs))
            self.posted_ci_jobs.pop(oldest)
            evicted.append(oldest)
        if self._uses_granular_redis_state():
            await self.state_store.add_posted_ci_job(run_id)
            for oldest in evicted:
                await self.state_store.remove_posted_ci_job(oldest)
        else:
            await self._persist_state()
        return True

    async def _release_ci_job(self, run_id: str):
        self.posted_ci_jobs.pop(run_id, None)
        if self._uses_granular_redis_state():
            await self.state_store.remove_posted_ci_job(run_id)
        else:
            await self._persist_state()

    async def _handle_socket_mode_request(
        self, client: AsyncSocketModeClient, req: SocketModeRequest
    ):
        """Handle incoming socket mode requests"""
        try:
            ack_payload = self._socket_ack_payload(req)
            response = (
                SocketModeResponse(envelope_id=req.envelope_id, payload=ack_payload)
                if ack_payload
                else SocketModeResponse(envelope_id=req.envelope_id)
            )
            await client.send_socket_mode_response(response)

            if req.type == "events_api":
                event = req.payload.get("event", {})
                team_id = self._extract_team_id(req.payload)
                if team_id and "_team_id" not in event:
                    event["_team_id"] = team_id
                await self._enqueue_work(
                    SlackWorkItem(
                        kind="event",
                        payload=event,
                        name=f"slack_event:{event.get('type', 'unknown')}",
                    )
                )

            elif req.type == "slash_commands":
                command = req.payload
                await self._enqueue_work(
                    SlackWorkItem(
                        kind="slash",
                        payload=command,
                        name="slack_slash_command",
                    )
                )

            elif (
                req.type == "interactive" and req.payload.get("type") == "block_actions"
            ):
                # trigger_id expires after ~3s, so the modal cannot wait in the queue.
                if await self._open_agent_feedback_modal(req.payload):
                    return
                await self._enqueue_work(
                    SlackWorkItem(
                        kind="interactive",
                        payload=req.payload,
                        name="slack_agent_approval",
                    )
                )
            elif (
                req.type == "interactive"
                and req.payload.get("type") == "message_action"
                and req.payload.get("callback_id") == "run_pi_agent_from_thread"
            ):
                await self._handle_agent_shortcut(req.payload)
            elif (
                req.type == "interactive"
                and req.payload.get("type") == "view_submission"
                and req.payload.get("view", {}).get("callback_id")
                in {"run_pi_agent_from_thread_submit", AGENT_FEEDBACK_CALLBACK_ID}
            ):
                await self._enqueue_work(
                    SlackWorkItem(
                        kind="agent_view",
                        payload=req.payload,
                        name="slack_agent_shortcut_submission",
                    )
                )

        except Exception as e:
            logger.error(f"Error handling socket mode request: {e}")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle Slack events"""
        event_type = event.get("type")

        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "app_mention":
            await self._handle_mention_event(event)

    async def _handle_message_event(self, event: Dict[str, Any]):
        """Handle message events"""
        # Skip bot messages, edits and other subtypes nobody asks with
        subtype = event.get("subtype")
        if (
            subtype not in CHAT_MESSAGE_SUBTYPES
            or event.get("bot_id")
            or event.get("user") == self.bot_user_id
        ):
            return

        channel_id = event.get("channel")
        if not await self._is_channel_allowed(channel_id, event.get("channel_name")):
            return

        thread_ts = event.get("thread_ts")
        if subtype is not None:
            # A file share or broadcast continues a chat thread only: an Agent
            # thread still takes plain follow-ups, as it did before S13.
            if not self._is_active_thread(channel_id, thread_ts, refresh=False):
                return
            if not self._is_agent_thread(channel_id, thread_ts):
                await self._restore_agent_thread(event)
            if self._is_agent_thread(channel_id, thread_ts):
                return
        elif await self._reject_non_owner_agent_message(event):
            return
        if not thread_ts or not self._is_active_thread(channel_id, thread_ts):
            return
        if not self._claim_message(event):
            return

        self._strip_bot_mention(event)
        await self._process_message(event)

    def _claim_message(
        self, event: Dict[str, Any], claimed: Optional[Dict[str, None]] = None
    ) -> bool:
        """True only the first time a Slack message is claimed.

        Claims go in handled_messages unless another bounded set is given.
        """
        if claimed is None:
            claimed = self.handled_messages
        channel_id = event.get("channel")
        ts = event.get("ts")
        if not channel_id or not ts:
            return True
        key = self._thread_key(channel_id, ts)
        if key in claimed:
            return False
        claimed[key] = None
        while len(claimed) > HANDLED_MESSAGES_LIMIT:
            claimed.pop(next(iter(claimed)))
        return True

    def _strip_bot_mention(self, event: Dict[str, Any]) -> None:
        if self.bot_user_id:
            text = event.get("text", "")
            event["text"] = text.replace(f"<@{self.bot_user_id}>", "").strip()

    async def _reject_non_owner_agent_message(self, event: Dict[str, Any]) -> bool:
        """Turn away anyone but the owner in an Agent thread; True if turned away."""
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        if not thread_ts:
            return False
        if not self._is_agent_thread(channel_id, thread_ts):
            await self._restore_agent_thread(event)
        tracked = self.agent_threads.get(self._thread_key(channel_id, thread_ts))
        if not tracked or tracked.get("owner_user_id") == event.get("user"):
            return False
        if self._claim_message(event) and hasattr(
            self.web_client, "chat_postEphemeral"
        ):
            await self.web_client.chat_postEphemeral(
                channel=channel_id,
                user=event.get("user"),
                text=(
                    "Only the user who started this Agent session can send "
                    "coding prompts in this thread."
                ),
            )
        return True

    async def _restored_agent_mode(self, session: Dict[str, Any]) -> str:
        """The task mode of a thread known only to the runtime.

        A follow-up inherits discuss from its thread, so a mode nobody can
        name restores as discuss: the owner upgrades with `--implement`
        instead of a read-only thread silently gaining write access.
        """
        mode = session.get("task_mode")
        if mode in {"discuss", "implement"}:
            return str(mode)
        runtime = self.agent_runtime_client
        run_id = session.get("run_id")
        if run_id and runtime is not None and hasattr(runtime, "get_run"):
            try:
                run = await runtime.get_run(run_id)
            except AgentRuntimeError:
                run = None
            if isinstance(run, dict):
                # Runs created before task_mode existed were implement runs.
                mode = run.get("task_mode") or "implement"
                if mode in {"discuss", "implement"}:
                    return str(mode)
        return "discuss"

    async def _restore_agent_thread(self, event: Dict[str, Any]):
        await self._adopt_agent_session(
            self._extract_team_id(event) or "unknown-team",
            event.get("channel"),
            event.get("thread_ts"),
        )

    async def _adopt_agent_session(
        self, team_id: str, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Track the runtime's open session for a thread and watch its active run.

        Returns the tracked record, or None when the runtime has no open session
        for the thread or cannot be asked. A run this bot already follows keeps
        its Slack-side state.
        """
        runtime = self.agent_runtime_client
        if runtime is None or not hasattr(runtime, "lookup_session"):
            return None
        try:
            session = await runtime.lookup_session(team_id, channel_id, thread_ts)
        except AgentRuntimeError:
            return None
        if not session:
            return None
        known = self.agent_runs.get(session.get("run_id") or "")
        if known is not None:
            if session.get("active"):
                self._start_agent_monitor(known)
            return known
        mode = await self._restored_agent_mode(session)
        if session.get("run_id"):
            tracked = self._track_agent_run(
                session,
                team_id=team_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                owner_user_id=session["owner_user_id"],
                mode=mode,
            )
            if session.get("active"):
                self._start_agent_monitor(tracked)
            else:
                # Without Slack-side state we cannot tell whether the result
                # was posted; never repost it on a later restart.
                tracked["result_posted"] = True
            return tracked
        tracked = {
            "session_id": session["session_id"],
            "run_id": None,
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "owner_user_id": session["owner_user_id"],
            "mode": mode,
        }
        self.agent_threads[self._thread_key(channel_id, thread_ts)] = tracked
        self._remember_agent_session(tracked)
        self._mark_thread_active(channel_id, thread_ts)
        return tracked

    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Handle app mentions"""
        channel_id = event.get("channel")
        if not await self._is_channel_allowed(channel_id, event.get("channel_name")):
            return
        if await self._reject_non_owner_agent_message(event):
            return
        if not self._claim_message(event):
            return

        self._strip_bot_mention(event)
        await self._process_message(event)

    async def _handle_slash_command(self, command: Dict[str, Any]):
        """Handle slash commands"""
        command_text = command.get("text", "")
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")
        team_id = self._extract_team_id(command)

        if not await self._is_channel_allowed(channel_id, command.get("channel_name")):
            await self._post_command_response(
                channel_id,
                user_id,
                "This bot is not enabled in this channel.",
            )
            return

        user_state_changed = self._synchronize_user_tier(user_id)
        is_supported_command = True
        message_handler_is_supported_command = getattr(
            self.message_handler, "is_supported_command", None
        )
        if callable(message_handler_is_supported_command):
            is_supported_command = message_handler_is_supported_command(command_text)
        consumes_query_quota = not is_supported_command
        message_handler_consumes_query_quota = getattr(
            self.message_handler, "command_consumes_query_quota", None
        )
        if callable(message_handler_consumes_query_quota):
            consumes_query_quota = message_handler_consumes_query_quota(command_text)
        if consumes_query_quota:
            agent_command = _is_agent_command_text(command_text)
            if not self._within_rate_limit(user_id, agent=agent_command):
                await self._post_command_response(
                    channel_id,
                    user_id,
                    self._build_rate_limit_message(user_id, agent=agent_command),
                )
                if user_state_changed:
                    await self._persist_user_state(user_id)
                return

        trigger_id = command.get("trigger_id")
        response_text = await self.message_handler._handle_command(
            command_text,
            user_id,
            channel_id,
            None,
            self.web_client,
            team_id=team_id,
            command_surface="slash",
            # Slack resends a slash command with the same trigger_id.
            request_id=f"slash:{trigger_id}" if trigger_id else None,
        )

        if response_text:
            await self._post_command_response(channel_id, user_id, response_text)
        if consumes_query_quota:
            await self._persist_message_state(
                user_id=user_id,
                conversation_key=self._conversation_context_key(
                    user_id=user_id,
                    channel_id=channel_id,
                    thread_ts=None,
                ),
            )
        elif user_state_changed:
            await self._persist_user_state(user_id)

    async def _process_message(self, event: Dict[str, Any]):
        """Process incoming message"""
        placeholder_ts: Optional[str] = None
        try:
            channel_id = event.get("channel")
            user_id = event.get("user")
            user_state_changed = self._synchronize_user_tier(user_id)
            user_prefs = self.user_manager.get_user_preferences(user_id)
            agent_request = self._is_agent_request(event)
            event["_agent_request"] = agent_request

            if self._is_query_event(event):
                if not self._within_rate_limit(user_id, agent=agent_request):
                    await self._send_rate_limit_message(
                        channel_id,
                        user_id,
                        thread_ts=event.get("thread_ts"),
                        agent=agent_request,
                    )
                    if user_state_changed:
                        await self._persist_user_state(user_id)
                    return

            reply_thread_ts = self._reply_thread_ts(event, user_prefs)
            active_thread_key = None
            if reply_thread_ts and not agent_request:
                # Accepted: a follow-up sent while this answer is generating
                # belongs to the thread too. An Agent thread becomes active
                # once the runtime accepts the run.
                self._mark_thread_active(channel_id, reply_thread_ts)
                active_thread_key = self._thread_key(channel_id, reply_thread_ts)

            if not agent_request and self._waits_on_model(event):
                placeholder_ts = await self._post_placeholder(
                    channel_id, reply_thread_ts
                )

            if not agent_request:
                # An Agent request loads its own files, as untrusted context.
                attachments = await self._extract_query_attachments(event)
                if attachments:
                    event["_query_attachments"] = attachments

            # Process message through handler
            response_text = await self.message_handler.handle_message(
                event, self.web_client
            )

            state_changed = active_thread_key is not None
            agent_thread_ts = event.get("thread_ts") or event.get("ts")
            if response_text:
                thread_ts = reply_thread_ts

                # Split long responses
                max_length = self.config.get("response_settings", {}).get(
                    "max_response_length", 2000
                )
                responses = (
                    self._split_response(response_text, max_length)
                    if len(response_text) > max_length
                    else [response_text]
                )
                for index, response_part in enumerate(responses):
                    message = self._answer_message(
                        response_part,
                        response_text,
                        last=index == len(responses) - 1,
                    )
                    if index == 0 and placeholder_ts:
                        replaced = await self._replace_placeholder(
                            channel_id, placeholder_ts, **message
                        )
                        # The answer owns the placeholder now: a later
                        # failure must not overwrite it with the error.
                        placeholder_ts = None
                        if replaced:
                            continue
                    await self._post_reply(channel_id, thread_ts, **message)

                if thread_ts:
                    self._mark_thread_active(channel_id, thread_ts)
                    state_changed = True
                    active_thread_key = self._thread_key(channel_id, thread_ts)
            else:
                # No reply text: the handler posted the reply itself (an Agent
                # run's status message), or there is no answer.
                handed_to_agent = self._is_agent_thread(channel_id, agent_thread_ts)
                if placeholder_ts:
                    notice = (
                        SLACK_AGENT_HANDOFF_TEXT
                        if handed_to_agent
                        else SLACK_EMPTY_ANSWER_MESSAGE
                    )
                    replaced = await self._replace_placeholder(
                        channel_id, placeholder_ts, text=notice
                    )
                    if not replaced and not handed_to_agent:
                        await self._post_reply(channel_id, reply_thread_ts, text=notice)
                if handed_to_agent:
                    state_changed = True
                    active_thread_key = self._thread_key(channel_id, agent_thread_ts)

            query_event = self._is_query_event(event)
            if query_event or user_state_changed:
                state_changed = True

            if state_changed:
                conversation_key = None
                if query_event:
                    conversation_key = self._conversation_context_key(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_ts=event.get("thread_ts") or event.get("ts"),
                    )
                await self._persist_message_state(
                    user_id=user_id if (query_event or user_state_changed) else None,
                    conversation_key=conversation_key,
                    active_thread_key=active_thread_key,
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_text = self._build_user_safe_error_message()
            if placeholder_ts and await self._replace_placeholder(
                event.get("channel"), placeholder_ts, text=error_text
            ):
                return
            await self.web_client.chat_postMessage(
                channel=event.get("channel"),
                text=error_text,
                thread_ts=event.get("thread_ts") or event.get("ts"),
            )

    def _answer_message(self, part: str, answer: str, *, last: bool) -> Dict[str, Any]:
        """The message fields for one part of a reply.

        A model's answer renders as a Markdown block, with the plain text
        kept as the notification fallback, and its last part carries the
        footer. Command replies and errors stay plain text.
        """
        message: Dict[str, Any] = {"text": part}
        if (
            not isinstance(answer, SlackAnswer)
            or len(part) > SLACK_MARKDOWN_BLOCK_CHARS
        ):
            return message
        markdown = answer.markdown(part)
        if len(markdown) > SLACK_MARKDOWN_BLOCK_CHARS:
            return message
        blocks: List[Dict[str, Any]] = [{"type": "markdown", "text": markdown}]
        footer = answer.footer() if last else None
        if footer and self.config.get("response_settings", {}).get(
            "answer_footer", True
        ):
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "plain_text", "text": footer}],
                }
            )
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

        A refused update is tried once more as plain text, without blocks.
        """
        attempts = [message]
        if "blocks" in message:
            attempts.append(_without_blocks(message))
        for attempt in attempts:
            try:
                await self.web_client.chat_update(
                    channel=channel_id, ts=placeholder_ts, **attempt
                )
            except Exception as exc:
                logger.warning(
                    "Could not update the Slack progress placeholder: %s", exc
                )
                continue
            return True
        return False

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

        try:
            return await asyncio.wait_for(
                self._fetch_slack_context(client, channel_id, thread_ts, message_ts),
                timeout=float(self.context_settings.get("timeout_seconds", 3.0)),
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
    ) -> SlackFetchedContext:
        """Fetch Slack history using thread context first, then channel context."""
        strategy = self.context_settings.get("strategy", "thread_first")
        if strategy != "thread_first":
            raise ValueError(f"Unsupported Slack context strategy: {strategy}")

        use_thread = bool(thread_ts and (not message_ts or thread_ts != message_ts))
        if use_thread:
            method_name = "conversations_replies"
            method = getattr(client, method_name, None)
            if not callable(method):
                raise RuntimeError(f"{method_name}_unavailable")

            kwargs: Dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": int(self.context_settings.get("max_thread_messages", 20)),
            }
            if message_ts:
                kwargs["latest"] = message_ts
                kwargs["inclusive"] = False
            response = await method(**kwargs)
            source = "thread"
        else:
            if not message_ts:
                return SlackFetchedContext()
            method_name = "conversations_history"
            method = getattr(client, method_name, None)
            if not callable(method):
                raise RuntimeError(f"{method_name}_unavailable")

            response = await method(
                channel=channel_id,
                latest=message_ts,
                inclusive=False,
                limit=int(self.context_settings.get("max_channel_messages", 10)),
            )
            source = "channel"

        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(str(response.get("error") or "slack_api_error"))

        messages = list((response or {}).get("messages", []) or [])
        messages.sort(key=lambda message: self._slack_message_sort_key(message))
        filtered_messages = [
            message
            for message in messages
            if self._should_include_slack_context_message(message, message_ts)
        ]
        return self._format_slack_context(filtered_messages, source)

    def _should_include_slack_context_message(
        self, message: Dict[str, Any], current_ts: Optional[str]
    ) -> bool:
        """Filter Slack system/bot/current-query messages out of fetched context."""
        if self._same_slack_timestamp(message.get("ts"), current_ts):
            return False
        if message.get("subtype"):
            return False
        if not self.context_settings.get("include_bot_messages", False):
            if message.get("bot_id") or message.get("user") == self.bot_user_id:
                return False
        if not self._clean_slack_message_text(message.get("text", "")):
            return False
        return bool(self._slack_message_sender(message))

    def _format_slack_context(
        self, messages: List[Dict[str, Any]], source: str
    ) -> SlackFetchedContext:
        """Render Slack messages into a compact deterministic context block."""
        if not messages:
            return SlackFetchedContext(source=source)

        header = (
            "Slack thread context:" if source == "thread" else "Recent channel context:"
        )
        candidate_lines = [
            f"{self._slack_message_sender(message)}: "
            f"{self._clean_slack_message_text(message.get('text', ''))}"
            for message in messages
        ]

        max_chars = int(self.context_settings.get("max_context_chars", 4000))
        output_lines = [header]
        included_count = 0
        truncated = False

        for line in candidate_lines:
            current_text = "\n".join(output_lines)
            separator_len = 1 if current_text else 0
            remaining = max_chars - len(current_text) - separator_len
            if remaining <= 0:
                truncated = True
                break
            if len(line) <= remaining:
                output_lines.append(line)
                included_count += 1
                continue
            if remaining > 4:
                output_lines.append(line[: remaining - 3].rstrip() + "...")
                included_count += 1
            truncated = True
            break

        if included_count < len(candidate_lines):
            truncated = True
        if included_count == 0:
            return SlackFetchedContext(source=source, truncated=True)

        return SlackFetchedContext(
            context="\n".join(output_lines),
            source=source,
            message_count=included_count,
            truncated=truncated,
        )

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

    async def _send_rate_limit_message(
        self,
        channel_id: str,
        user_id: str,
        *,
        thread_ts: Optional[str] = None,
        agent: bool = False,
    ):
        """Tell only the user that they hit a limit, in their thread if any.

        Pass thread_ts only for a reply in an existing thread: Slack never
        shows an ephemeral sent into a thread that has no replies yet, so a
        notice for a new top-level request stays in the channel.
        """
        await self.web_client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=self._build_rate_limit_message(user_id, agent=agent),
            thread_ts=thread_ts,
        )

    def _build_rate_limit_message(self, user_id: str, *, agent: bool = False) -> str:
        """Say which of the user's limits was hit and when to try again.

        The limits are a rolling 60-minute window and a 5-minute burst, each
        counted per bucket: chat queries and Agent requests apart.
        """
        user_tier = self.user_manager.get_user_tier(user_id)
        limits = self._get_rate_limit_config_for_tier(user_tier)
        hourly_limit = limits["requests_per_hour"]
        burst_limit = limits["burst_requests"]
        kind = "Agent request" if agent else "chat request"
        now = time.time()
        counted = sorted(
            self.user_manager.requests_since(
                user_id,
                now - RATE_LIMIT_WINDOW_SECONDS,
                bucket=RATE_LIMIT_AGENT if agent else RATE_LIMIT_CHAT,
            )
        )
        recent = [at for at in counted if at > now - RATE_LIMIT_BURST_WINDOW_SECONDS]

        # The first request that must leave a full window frees a slot in it;
        # with both full, the later of the two is when a retry goes through.
        reason = None
        retry_at = 0.0
        if 0 < hourly_limit <= len(counted):
            reason = (
                f"You've sent {_plural(len(counted), kind)} in the last "
                "60 minutes, the hourly limit for your tier."
            )
            retry_at = counted[len(counted) - hourly_limit] + RATE_LIMIT_WINDOW_SECONDS
        if 0 < burst_limit <= len(recent):
            burst_retry_at = (
                recent[len(recent) - burst_limit] + RATE_LIMIT_BURST_WINDOW_SECONDS
            )
            if reason is None or burst_retry_at > retry_at:
                reason = (
                    f"You've sent {_plural(len(recent), kind)} in the last "
                    "5 minutes, the burst limit for your tier."
                )
                retry_at = burst_retry_at
        if hourly_limit <= 0 or burst_limit <= 0:
            # check_rate_limit refuses every request against a zero limit.
            summary = (
                f"The {user_tier.value.title()} tier doesn't allow {kind}s. "
                "Ask an admin if you need access."
            )
        elif reason is None:
            article = "an" if agent else "a"
            summary = (
                f"You've reached {article} {kind} limit. Try again in a few minutes."
            )
        else:
            summary = f"{reason} Try again in {_format_wait(retry_at - now)}."

        return "\n".join(
            [
                "🚫 *Rate Limit Exceeded*",
                "",
                summary,
                "",
                f"*Your tier:* {user_tier.value.title()}",
                f"*Limits:* {hourly_limit} requests per rolling 60 minutes and "
                f"{burst_limit} per 5 minutes, for chat and Agent requests each",
            ]
        )

    async def _cleanup_sessions_periodically(self):
        """Periodically clean up expired sessions"""
        while self.running:
            try:
                expired_sessions = self.conversation_manager.cleanup_expired_sessions(
                    return_keys=True
                )
                expired_threads = self._cleanup_expired_threads(return_keys=True)
                if expired_sessions or expired_threads:
                    await self._delete_conversation_states(expired_sessions)
                    await self._delete_active_thread_states(expired_threads)
                await asyncio.sleep(3600)  # Clean up every hour
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(3600)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for status command"""
        health = {}
        if hasattr(self.inference_engine, "get_health_status"):
            health = self.inference_engine.get_health_status()

        provider_health = health.get("providers", {})
        available_models = list(getattr(self.router, "models", {}).keys())
        model_stats = getattr(self.router, "model_stats", {})
        avg_latency = 0.0
        if model_stats:
            avg_latency = sum(
                stats.get("avg_latency", 0.0) for stats in model_stats.values()
            ) / max(len(model_stats), 1)

        healthy = all(
            provider.get("status", "healthy") == "healthy"
            for provider in provider_health.values()
        )
        if not provider_health:
            healthy = self.initialized

        return {
            "healthy": healthy,
            "available_models": available_models,
            "avg_response_time": avg_latency,
            "uptime": self._get_uptime_text(),
        }

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        # The quota shown is the chat bucket's; Agent requests have their own.
        recent_requests = self.user_manager.requests_since(user_id, time.time() - 3600)
        rate_limit_config = self._get_rate_limit_config_for_user(user_id)
        remaining_requests = max(
            0,
            rate_limit_config.get("requests_per_hour", 100) - len(recent_requests),
        )

        analytics = {}
        if self.analytics_service and hasattr(
            self.analytics_service, "get_query_analytics"
        ):
            analytics = await self.analytics_service.get_query_analytics(
                user_id, hours=24
            )
        elif self.analytics_service and hasattr(
            self.analytics_service, "get_analytics"
        ):
            analytics = await self.analytics_service.get_analytics(user_id, hours=24)

        return {
            "queries_24h": analytics.get("total_queries", 0),
            "cost_24h": analytics.get("total_cost", 0.0),
            "avg_latency": analytics.get("avg_latency", 0.0),
            "success_rate": analytics.get("success_rate", 0.0),
            "remaining_requests": remaining_requests,
        }

    async def get_available_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Get available models for user"""
        if not self.router:
            return []

        user_tier = self.user_manager.get_user_tier(user_id)
        check_user_access = getattr(self.router, "_check_user_access", None)
        models = []
        for model_name in getattr(self.router, "models", {}):
            model_info = self.router.get_model_info(model_name)
            if not model_info:
                continue

            model_config = getattr(self.router, "models", {}).get(model_name)
            if callable(check_user_access) and model_config is not None:
                if not check_user_access(model_config, user_tier.value):
                    continue

            config = model_info["config"]
            provider_name = config.get("provider", "unknown")
            provider = getattr(self.inference_engine, "providers", {}).get(
                provider_name
            )
            provider_health = (
                provider.get_health_status()
                if provider and hasattr(provider, "get_health_status")
                else {}
            )
            available = (
                provider is not None
                and provider_health.get("status", "healthy") == "healthy"
            )

            models.append(
                {
                    "name": model_name,
                    "provider": provider_name,
                    "capabilities": config.get("capabilities", []),
                    "max_tokens": config.get("max_tokens", 0),
                    "cost_per_1k_tokens": config.get("cost_per_token", 0.0) * 1000,
                    "description": ", ".join(config.get("capabilities", []))
                    or "General use",
                    "available": available,
                }
            )

        return models

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        analytics = {}
        if self.analytics_service and hasattr(
            self.analytics_service, "get_query_analytics"
        ):
            analytics = await self.analytics_service.get_query_analytics(
                user_id, hours=168
            )
        elif self.analytics_service and hasattr(
            self.analytics_service, "get_analytics"
        ):
            analytics = await self.analytics_service.get_analytics(user_id, hours=168)

        total_queries = analytics.get("total_queries", 0)
        analytics.setdefault(
            "daily_avg_queries", total_queries / 7 if total_queries else 0.0
        )
        analytics.setdefault("cache_hit_rate", 0.0)
        analytics.setdefault("model_breakdown", {})
        analytics.setdefault("query_type_breakdown", {})
        analytics.setdefault("total_tokens", 0)
        analytics.setdefault("total_cost", 0.0)
        analytics.setdefault("avg_latency", 0.0)
        analytics.setdefault("success_rate", 0.0)
        return analytics

    def is_healthy(self) -> bool:
        """Check whether the Slack bot initialized correctly."""
        return (
            self.initialized
            and self.web_client is not None
            and self.socket_client is not None
        )

    async def shutdown(self):
        """Shutdown the Slack bot"""
        logger.info("Shutting down Slack bot...")
        self.running = False
        self._accepting_work = False
        await self._stop_work_workers()

        for task in list(self.background_tasks):
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        if self.socket_client:
            await self._disconnect_socket_once()

        if not self._uses_granular_redis_state():
            await self._persist_state()
        await self.state_store.shutdown()
        try:
            await self.memory_manager.shutdown()
        except Exception:
            logger.warning("Slack memory shutdown failed", exc_info=True)
        await self._close_attachment_http_client()

        logger.info("Slack bot shutdown complete")

    async def _disconnect_socket_once(self) -> None:
        async with self._socket_disconnect_lock:
            if self._socket_disconnected or not self.socket_client:
                return
            await self.socket_client.disconnect()
            self._socket_disconnected = True

    def _resolve_secret(self, value_key: str, env_key: str) -> Optional[str]:
        """Resolve a secret from config value or named environment variable."""
        direct_value = self.config.get(value_key)
        if direct_value:
            return direct_value

        env_name = self.config.get(env_key)
        if env_name:
            return os.getenv(env_name)

        return None

    async def _resolve_allowed_channels(self):
        """Resolve configured channel names to Slack channel IDs when possible."""
        self.allowed_channel_ids = set()
        self.allowed_channel_names = set()

        unresolved_names = set()
        for channel in self.allowed_channels:
            raw_value = (channel or "").strip().lstrip("#")
            if not raw_value:
                continue
            if self._looks_like_channel_id(raw_value):
                self.allowed_channel_ids.add(raw_value)
            else:
                unresolved_names.add(raw_value.lower())

        if not unresolved_names or not self.web_client:
            self.allowed_channel_names = unresolved_names
            return

        cursor = None
        try:
            while True:
                response = await self.web_client.conversations_list(
                    limit=1000,
                    types="public_channel",
                    cursor=cursor,
                )
                for channel in response.get("channels", []):
                    channel_name = self._normalize_channel_name(channel.get("name"))
                    channel_id = channel.get("id")
                    if channel_name in unresolved_names and channel_id:
                        self.allowed_channel_ids.add(channel_id)
                        unresolved_names.discard(channel_name)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor or not unresolved_names:
                    break
        except Exception as exc:
            logger.warning(f"Failed to resolve Slack channel allowlist: {exc}")

        self.allowed_channel_names = unresolved_names
        if unresolved_names:
            logger.warning(
                "Unresolved Slack channels remain in allowlist: %s",
                ", ".join(sorted(unresolved_names)),
            )

    async def _is_channel_allowed(
        self, channel_id: Optional[str], channel_name: Optional[str] = None
    ) -> bool:
        """Check whether the bot is enabled in the target channel."""
        if not self.allowed_channels:
            return True

        if channel_id and channel_id in self.allowed_channel_ids:
            return True

        normalized_name = self._normalize_channel_name(channel_name)
        if normalized_name and normalized_name in self.allowed_channel_names:
            return True

        if (
            channel_id
            and self.allowed_channel_names
            and self.web_client is not None
            and hasattr(self.web_client, "conversations_info")
        ):
            try:
                response = await self.web_client.conversations_info(channel=channel_id)
                resolved_name = self._normalize_channel_name(
                    response.get("channel", {}).get("name")
                )
                if resolved_name and resolved_name in self.allowed_channel_names:
                    self.allowed_channel_ids.add(channel_id)
                    return True
            except Exception as exc:
                logger.warning(
                    f"Failed to resolve Slack channel info for allowlist check: {exc}"
                )

        return False

    def _normalize_channel_name(self, channel_name: Optional[str]) -> str:
        """Normalize channel names for consistent allowlist matching."""
        if not channel_name:
            return ""
        return channel_name.strip().lstrip("#").lower()

    def _looks_like_channel_id(self, value: str) -> bool:
        """Return whether a configured channel resembles a Slack channel ID."""
        return (
            len(value) >= 9
            and value[0] in {"C", "G", "D"}
            and value.upper() == value
            and value.isalnum()
        )

    def _thread_key(self, channel_id: str, thread_ts: str) -> str:
        """Build a stable key for active thread tracking."""
        return f"{channel_id}:{thread_ts}"

    def _is_agent_thread(
        self, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> bool:
        if not channel_id or not thread_ts:
            return False
        return self._thread_key(channel_id, thread_ts) in self.agent_threads

    def _track_agent_run(
        self,
        accepted: Dict[str, Any],
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        owner_user_id: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        previous = self.agent_runs.get(accepted["run_id"]) or {}
        tracked = {
            "session_id": accepted["session_id"],
            "run_id": accepted["run_id"],
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "owner_user_id": owner_user_id,
            "mode": mode,
            "posted_approval_ids": list(previous.get("posted_approval_ids", [])),
            "result_posted": False,
            # Times the progress line when the runtime reports no created_at.
            "tracked_at": self.agent_progress_clock(),
        }
        if previous.get("approval_cards"):
            tracked["approval_cards"] = dict(previous["approval_cards"])
        for field in ("status_ts", "status_text"):
            if previous.get(field):
                tracked[field] = previous[field]
        self._register_agent_run(tracked)
        return tracked

    def _register_agent_run(self, tracked: Dict[str, Any]) -> None:
        channel_id = tracked["channel_id"]
        thread_ts = tracked["thread_ts"]
        self.agent_threads[self._thread_key(channel_id, thread_ts)] = tracked
        self.agent_runs[tracked["run_id"]] = tracked
        self._remember_agent_session(tracked)
        self._mark_thread_active(channel_id, thread_ts)

    def _agent_sessions_of(self, user_id: str) -> List[Dict[str, Any]]:
        """The owner's tracked sessions, one per thread, the latest last."""
        sessions = self.agent_user_runs.get(user_id)
        if isinstance(sessions, dict):
            # One record per owner, the shape this map held before it kept them all.
            sessions = self.agent_user_runs[user_id] = [sessions]
        return sessions or []

    def _remember_agent_session(self, tracked: Dict[str, Any]) -> None:
        """Make a run its owner's latest session, replacing its thread's older one."""
        thread = (tracked.get("channel_id"), tracked.get("thread_ts"))
        self.agent_user_runs[tracked["owner_user_id"]] = [
            other
            for other in self._agent_sessions_of(tracked["owner_user_id"])
            if (other.get("channel_id"), other.get("thread_ts")) != thread
        ] + [tracked]

    @staticmethod
    def _agent_run_record(tracked: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "run_id": tracked["run_id"],
            "session_id": tracked.get("session_id"),
            "team_id": tracked.get("team_id"),
            "channel_id": tracked.get("channel_id"),
            "thread_ts": tracked.get("thread_ts"),
            "owner_user_id": tracked.get("owner_user_id"),
            "mode": tracked.get("mode"),
            "posted_approval_ids": list(tracked.get("posted_approval_ids", [])),
            "result_posted": bool(tracked.get("result_posted", False)),
        }
        # The root message the monitor rewrites with progress, when there is one.
        for field in ("status_ts", "status_text"):
            if isinstance(tracked.get(field), str) and tracked[field]:
                record[field] = tracked[field]
        # Message ts of each approval card still showing buttons, by approval_id.
        cards = tracked.get("approval_cards")
        if isinstance(cards, dict) and cards:
            record["approval_cards"] = {
                str(key): str(value) for key, value in cards.items()
            }
        return record

    async def _persist_agent_run(self, tracked: Dict[str, Any]):
        """Persist one tracked run and drop finished runs it superseded."""
        superseded = [
            run_id
            for run_id, other in self.agent_runs.items()
            if run_id != tracked["run_id"]
            and other.get("result_posted")
            and other.get("channel_id") == tracked["channel_id"]
            and other.get("thread_ts") == tracked["thread_ts"]
        ]
        for run_id in superseded:
            self.agent_runs.pop(run_id, None)

        if self._uses_granular_redis_state():
            await self.state_store.persist_agent_run_state(
                tracked["run_id"], self._agent_run_record(tracked)
            )
            for run_id in superseded:
                await self.state_store.delete_agent_run_state(run_id)
            return

        await self._persist_state()

    async def _forget_agent_run(self, tracked: Dict[str, Any]):
        """Stop tracking a run in memory and in the configured state backend."""
        self._forget_agent_session(tracked)
        if self._uses_granular_redis_state():
            if tracked.get("run_id"):
                await self.state_store.delete_agent_run_state(tracked["run_id"])
            return

        await self._persist_state()

    async def _agent_run_home(
        self, accepted: Dict[str, Any], owner_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """A Slack record placing a run: its own, or one of its session's.

        A session lives in one thread. The runtime keeps no Slack thread for a
        run, so tracked state answers, then the shared Redis state another bot
        process may have written; a record found there is tracked from now on,
        so its posted approvals are not posted again.
        """
        known = self.agent_runs.get(accepted["run_id"])
        if known is None:
            known = next(
                (
                    tracked
                    for tracked in reversed(self._agent_sessions_of(owner_user_id))
                    if tracked.get("session_id") == accepted.get("session_id")
                ),
                None,
            )
        if known is None and self._uses_granular_redis_state():
            record = await self.state_store.load_agent_run_state(accepted["run_id"])
            if _is_complete_agent_run_record(record):
                known = self._agent_run_record(record)
                self._register_agent_run(known)
        return known

    def _agent_run_for(
        self,
        user_id: str,
        channel_id: Optional[str],
        thread_ts: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if channel_id and thread_ts:
            tracked = self.agent_threads.get(self._thread_key(channel_id, thread_ts))
            if tracked and tracked.get("owner_user_id") == user_id:
                return tracked
        sessions = self._agent_sessions_of(user_id)
        return sessions[-1] if sessions else None

    def _agent_cap_message(
        self,
        user_id: str,
        channel_id: Optional[str],
        thread_ts: Optional[str],
    ) -> Optional[str]:
        """Why the owner cannot start another agent now, or None if they can.

        A run counts while its monitor is watching it: a monitor that gave up
        must not lock its owner out. A follow-up in a thread reuses that
        thread's session, so runs and starts in the same thread never count.
        CI repair runs are opened by the CI inbox, not by the owner, and do
        not count either.
        """
        this_thread = (
            self._thread_key(channel_id, thread_ts)
            if channel_id and thread_ts
            else None
        )
        active = [
            tracked
            for tracked in self.agent_runs.values()
            if tracked.get("owner_user_id") == user_id
            and not tracked.get("result_posted")
            and self._is_agent_run_monitored(tracked.get("run_id"))
            and tracked.get("mode") != "ci_repair"
            and (
                this_thread is None
                or self._thread_key(
                    tracked.get("channel_id", ""), tracked.get("thread_ts", "")
                )
                != this_thread
            )
        ]
        pending = [
            key
            for key in self.agent_pending_starts.get(user_id, {}).values()
            if key is None or key != this_thread
        ]
        if len(active) + len(pending) < self.agent_max_active_runs_per_user:
            return None
        if active:
            run = active[0]
            return (
                f"You already have an active Agent run `{run['run_id']}` in "
                f"<#{run['channel_id']}>. Wait for it to finish, or stop it with "
                "`/llm agent stop`, then start the new task."
            )
        return (
            "You already have an Agent run starting. Wait for it to open its "
            "thread, then start the new task."
        )

    def _is_agent_run_monitored(self, run_id: Optional[str]) -> bool:
        task = self.agent_monitor_tasks.get(run_id) if run_id else None
        return task is not None and not task.done()

    def _reserve_agent_start(
        self, user_id: str, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> str:
        token = uuid.uuid4().hex
        self.agent_pending_starts.setdefault(user_id, {})[token] = (
            self._thread_key(channel_id, thread_ts)
            if channel_id and thread_ts
            else None
        )
        return token

    def _release_agent_start(self, user_id: str, token: str) -> None:
        pending = self.agent_pending_starts.get(user_id)
        if pending is None:
            return
        pending.pop(token, None)
        if not pending:
            self.agent_pending_starts.pop(user_id, None)

    def _forget_agent_session(self, tracked: Dict[str, Any]):
        self.agent_runs.pop(tracked.get("run_id"), None)
        self.agent_threads.pop(
            self._thread_key(
                tracked.get("channel_id", ""), tracked.get("thread_ts", "")
            ),
            None,
        )
        owner_user_id = tracked.get("owner_user_id") or ""
        sessions = [
            other
            for other in self._agent_sessions_of(owner_user_id)
            if other is not tracked
        ]
        if sessions:
            self.agent_user_runs[owner_user_id] = sessions
        else:
            self.agent_user_runs.pop(owner_user_id, None)

    @staticmethod
    def _agent_approval_blocks(
        approval: Dict[str, Any], ci: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build approval card blocks that each stay within Slack's section limit.

        `ci` is a CI repair's target; the card then names the job it repairs.
        """
        # Titles name agent-chosen paths and commands, which must not ping or link.
        title = _escape_mrkdwn(str(approval.get("title") or "Approval required")[:200])
        # Keep embedded fences from closing the code block early.
        detail = str(approval.get("detail") or "").replace("```", "`\u200b``")
        fence = "```"
        text = f"{fence}{detail}{fence}" if detail else "_No detail provided._"
        if len(text) > SLACK_SECTION_TEXT_LIMIT:
            marker_reserve = len(
                f"\n_Truncated: showing {len(detail)}/{len(detail)} chars._"
            )
            shown = SLACK_SECTION_TEXT_LIMIT - 2 * len(fence) - marker_reserve
            text = (
                f"{fence}{detail[:shown]}{fence}\n"
                f"_Truncated: showing {shown}/{len(detail)} chars._"
            )

        expiry = "This request expires soon."
        expires_at = approval.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expiry = (
                    f"This request expires at <!date^{int(parsed.timestamp())}"
                    f"^{{time}}|{expires_at}>."
                )
            except ValueError:
                pass

        repairing = (
            [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _ci_repair_line(ci)},
                }
            ]
            if isinstance(ci, dict)
            else []
        )
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
            *repairing,
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Only the session owner can decide. {expiry}",
                    }
                ],
            },
        ]

    @staticmethod
    def _agent_delivery_blocks(approval: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the Deliver/Discard card; all of its text stays within 3000 chars."""
        delivery = approval.get("delivery")
        delivery = delivery if isinstance(delivery, dict) else {}
        files = [str(item) for item in delivery.get("files") or []]
        # The list takes at most half the card so the stat and diff still fit.
        file_lines: List[str] = []
        used = 0
        for index, name in enumerate(files[:AGENT_DELIVERY_MAX_FILES]):
            line = f"• `{name[:200]}`"
            rest = len(files) - index - 1
            marker = len(f"\n_…and {len(files)} more._") if rest else 0
            if used + len(line) + 1 + marker > AGENT_DELIVERY_FILES_TEXT_LIMIT:
                break
            file_lines.append(line)
            used += len(line) + 1
        if len(files) > len(file_lines):
            file_lines.append(f"_…and {len(files) - len(file_lines)} more._")
        title = f"*Deliver {len(files)} changed file{'s' if len(files) != 1 else ''}?*"
        files_text = "\n".join(file_lines) or "_No file list provided._"
        checks = (
            f"Verification: *{delivery.get('verification_status') or 'unknown'}*"
            f" · Grader: *{delivery.get('grader_status') or 'unknown'}*"
        )

        expiry = "This request expires soon."
        expires_at = approval.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expiry = (
                    f"Expires at <!date^{int(parsed.timestamp())}"
                    f"^{{date_short_pretty}} {{time}}|{expires_at}>; "
                    "unanswered changes are discarded."
                )
            except ValueError:
                pass
        footer = f"Only the session owner can deliver. {expiry}"

        fence = "```"
        # Spend whatever the fixed parts leave on the stat and the diff excerpt.
        budget = AGENT_DELIVERY_CARD_LIMIT - sum(
            len(part) for part in (title, files_text, checks, footer)
        )
        extras = []
        for raw in (delivery.get("stat"), delivery.get("excerpt")):
            text = str(raw or "").replace(fence, "`\u200b``")
            if not text or budget <= 2 * len(fence) + 20:
                continue
            room = budget - 2 * len(fence)
            if len(text) > room:
                text = text[: room - 2] + "\n…"
            extras.append(f"{fence}{text}{fence}")
            budget -= len(extras[-1])

        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": title}},
            {"type": "section", "text": {"type": "mrkdwn", "text": files_text}},
            *(
                {"type": "section", "text": {"type": "mrkdwn", "text": text}}
                for text in extras
            ),
            {"type": "section", "text": {"type": "mrkdwn", "text": checks}},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": footer}],
            },
        ]

    def _reconcile_agent_runs(self) -> None:
        """Resume monitoring every restored run whose result was never posted."""
        if self.agent_runtime_client is None:
            return
        for tracked in list(self.agent_runs.values()):
            if tracked.get("run_id") and not tracked.get("result_posted"):
                self._start_agent_monitor(tracked)

    def _start_agent_monitor(self, tracked: Dict[str, Any]) -> asyncio.Task:
        """Spawn a monitor for a run unless one is already running."""
        run_id = tracked["run_id"]
        task = self.agent_monitor_tasks.get(run_id)
        if task is not None and not task.done():
            return task
        task = self._spawn_background_task(
            self._monitor_agent_run(tracked), f"pi_agent_monitor:{run_id}"
        )
        self.agent_monitor_tasks[run_id] = task

        def release(done: asyncio.Task) -> None:
            if self.agent_monitor_tasks.get(run_id) is done:
                self.agent_monitor_tasks.pop(run_id, None)

        task.add_done_callback(release)
        return task

    async def _monitor_agent_run(self, tracked: Dict[str, Any]):
        run_id = tracked["run_id"]
        if run_id in self.agent_monitoring:
            return
        self.agent_monitoring.add(run_id)
        try:
            await self._watch_agent_run(tracked)
        finally:
            self.agent_monitoring.discard(run_id)

    async def _watch_agent_run(self, tracked: Dict[str, Any]):
        """Post each new approval card and the final result exactly once."""
        runtime = self.agent_runtime_client
        if runtime is None or self.web_client is None:
            return
        await self._persist_agent_run(tracked)
        # The runtime pauses a run's deadline while it waits for approval, so
        # each approval restarts the window the monitor keeps reattaching in.
        deadline = self.agent_monitor_clock() + self.agent_run_deadline_seconds
        failures = 0
        delay_posted = False
        while True:
            try:
                run = await runtime.wait_for_update(
                    tracked["run_id"],
                    on_progress=lambda update: self._update_agent_progress(
                        tracked, update
                    ),
                    # The status message is rewritten at most this often anyway.
                    refresh_seconds=self.agent_progress_interval_seconds,
                )
            except AgentRuntimeError as exc:
                if exc.status == 404:
                    logger.info(
                        "Agent run %s is gone; forgetting it", tracked["run_id"]
                    )
                    await self._forget_agent_run(tracked)
                    return
                remaining = deadline - self.agent_monitor_clock()
                if is_transient_error(exc) and remaining > 0:
                    logger.warning(
                        "Agent runtime unreachable for run %s; reattaching: %s",
                        tracked["run_id"],
                        exc,
                    )
                    if not delay_posted:
                        delay_posted = True
                        await self.web_client.chat_postMessage(
                            channel=tracked["channel_id"],
                            thread_ts=tracked["thread_ts"],
                            text=AGENT_STATUS_DELAYED_TEXT,
                        )
                    ceiling = min(
                        self.agent_monitor_reattach_max_seconds,
                        self.agent_monitor_reattach_seconds * 2**failures,
                    )
                    failures += 1
                    await self.agent_monitor_sleep(
                        min(remaining, self.agent_monitor_random() * ceiling)
                    )
                    continue
                logger.warning(
                    "Agent run monitor failed for %s: %s", tracked["run_id"], exc
                )
                await self.web_client.chat_postMessage(
                    channel=tracked["channel_id"],
                    thread_ts=tracked["thread_ts"],
                    text="Agent status could not be retrieved; Chat mode was not used as fallback.",
                )
                return
            failures = 0
            await self._close_agent_approval_cards(tracked, run)
            if run.get("status") == "awaiting_approval":
                deadline = self.agent_monitor_clock() + self.agent_run_deadline_seconds
                await self._update_agent_progress(tracked, run)
                await self._post_agent_approval(tracked, run)
                await asyncio.sleep(self.agent_monitor_poll_seconds)
                continue
            if not tracked.get("result_posted"):
                await self._update_agent_progress(tracked, run, force=True)
                await self._post_agent_result(tracked, run)
                tracked["result_posted"] = True
                await self._persist_agent_run(tracked)
            return

    @staticmethod
    def _agent_progress_line(run: Dict[str, Any], elapsed: float) -> str:
        try:
            cost = float(run.get("estimated_cost_usd") or 0)
        except (TypeError, ValueError):
            cost = 0.0
        line = (
            f"Phase: *{run.get('phase') or run.get('status') or 'unknown'}*"
            f" · Tools: {int(run.get('tool_count') or 0)}"
            f" · Turns: {int(run.get('turn_count') or 0)}"
            f" · Cost: ${cost:.3f}"
            f" · Elapsed: {_format_elapsed(elapsed)}"
        )
        if run.get("last_tool"):
            line += f" · Last tool: `{str(run['last_tool'])[:80]}`"
        return line

    @staticmethod
    def _ci_repair_target(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The CI run a repair job answers to; only CI repair runs carry one."""
        target = run.get("target")
        return target if isinstance(target, dict) else None

    @staticmethod
    def _agent_discuss_hint(run: Dict[str, Any]) -> Optional[str]:
        """Suggest --discuss when Jev is confident an implement run is read-only.

        Only a hint: the run keeps its task mode, permissions and approvals.
        """
        routing = run.get("routing")
        if run.get("task_mode") != "implement" or not isinstance(routing, dict):
            return None
        probability = routing.get("probability")
        if (
            # Shadow routing is observed, never acted on.
            routing.get("source") != "jev"
            or routing.get("label") != "read_only"
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or probability < AGENT_DISCUSS_HINT_PROBABILITY
        ):
            return None
        return (
            f"💡 This request looks read-only (route `read_only`, p={probability:.2f})."
            " `--discuss` may fit better: it uses read-only tools, with no"
            " verification or commit. This run continues as an implement run."
        )

    async def _update_agent_progress(
        self, tracked: Dict[str, Any], run: Dict[str, Any], *, force: bool = False
    ) -> None:
        """Rewrite the run's status message, at most once per progress interval."""
        status_ts = tracked.get("status_ts")
        if not status_ts or not hasattr(self.web_client, "chat_update"):
            return
        now = self.agent_progress_clock()
        last = tracked.get("progress_updated_at")
        if (
            not force
            and last is not None
            and now - last < self.agent_progress_interval_seconds
        ):
            return
        header = (
            tracked.get("status_text")
            or f"🛠️ Pi coding agent run `{run.get('run_id')}`."
        )
        started = _agent_run_started_at(run)
        if started is None:
            # A run restored after a bot restart is timed from its first update.
            started = tracked.setdefault("tracked_at", now)
        text = f"{header}\n{self._agent_progress_line(run, now - started)}"
        hint = self._agent_discuss_hint(run)
        if hint:
            text = f"{text}\n{hint}"
        if text == tracked.get("progress_text"):
            return
        try:
            await self.web_client.chat_update(
                channel=tracked["channel_id"], ts=status_ts, text=text
            )
        except Exception:
            logger.warning("Agent status message could not be updated", exc_info=True)
            return
        tracked["progress_updated_at"] = now
        tracked["progress_text"] = text

    async def _post_agent_approval(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        approval = next(
            (
                event
                for event in reversed(run.get("events", []))
                if event.get("type") == "approval"
            ),
            None,
        )
        posted = tracked.setdefault("posted_approval_ids", [])
        if not approval or approval.get("approval_id") in posted:
            return
        value = json.dumps(
            {
                "run_id": run["run_id"],
                "session_id": run["session_id"],
                "approval_id": approval["approval_id"],
            },
            separators=(",", ":"),
        )
        kind = approval.get("kind")
        labels = {
            "publish": ("Publish repair Draft PR", "Keep local result"),
            "delivery": ("Deliver", "Discard"),
        }.get(str(kind), ("Approve", "Reject"))
        card = (
            self._agent_delivery_blocks(approval)
            if kind == "delivery"
            else self._agent_approval_blocks(approval, self._ci_repair_target(run))
        )
        text = (
            "Ready to deliver: review the changes"
            if kind == "delivery"
            else "Approval required: "
            + _escape_mrkdwn(str(approval.get("title") or "Agent action"))
        )
        owner = tracked.get("owner_user_id")
        if owner:
            # Notify the owner: a slash-command thread does not include them.
            first = card[0]
            card[0] = {
                **first,
                "text": {
                    **first["text"],
                    "text": f"<@{owner}> {first['text']['text']}",
                },
            }
            text = f"{text} (<@{owner}>)"
        response = await self.web_client.chat_postMessage(
            channel=tracked["channel_id"],
            thread_ts=tracked["thread_ts"],
            text=text,
            blocks=[
                *card,
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "pi_agent_approve",
                            "text": {"type": "plain_text", "text": labels[0]},
                            "style": "primary",
                            "value": value,
                        },
                        {
                            "type": "button",
                            "action_id": "pi_agent_reject",
                            "text": {"type": "plain_text", "text": labels[1]},
                            "style": "danger",
                            "value": value,
                        },
                    ],
                },
            ],
        )
        posted.append(approval["approval_id"])
        card_ts = response.get("ts") if hasattr(response, "get") else None
        if isinstance(card_ts, str) and card_ts:
            tracked.setdefault("approval_cards", {})[approval["approval_id"]] = card_ts
        await self._persist_agent_run(tracked)

    @staticmethod
    def _closed_agent_card_blocks(
        blocks: List[Dict[str, Any]], outcome: str
    ) -> List[Dict[str, Any]]:
        """Keep a card's sections; its buttons and expiry footer give way to the outcome."""
        kept = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "section"
        ]
        return [
            *kept,
            {"type": "context", "elements": [{"type": "mrkdwn", "text": outcome}]},
        ]

    @staticmethod
    def _agent_card_outcome(verdict: str, user_id: Optional[str] = None) -> str:
        now = int(time.time())
        when = f"<!date^{now}^{{date_short_pretty}} {{time}}|{now}>"
        by = f" by <@{user_id}>" if user_id else ""
        return f"{verdict}{by} at {when}"

    async def _update_agent_card(
        self,
        channel_id: str,
        ts: str,
        blocks: List[Dict[str, Any]],
        outcome: str,
    ) -> None:
        if not hasattr(self.web_client, "chat_update"):
            return
        try:
            await self.web_client.chat_update(
                channel=channel_id,
                ts=ts,
                text=outcome,
                blocks=self._closed_agent_card_blocks(blocks, outcome),
            )
        except Exception:
            logger.warning("Agent approval card could not be updated", exc_info=True)

    async def _close_agent_approval_cards(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        """Replace the buttons of every card the run has moved past."""
        cards = tracked.get("approval_cards")
        if not isinstance(cards, dict) or not cards:
            return
        events = [event for event in run.get("events") or [] if isinstance(event, dict)]
        pending = None
        if run.get("status") == "awaiting_approval":
            pending = next(
                (
                    event.get("approval_id")
                    for event in reversed(events)
                    if event.get("type") == "approval"
                ),
                None,
            )
        closed = False
        for approval_id, ts in list(cards.items()):
            if approval_id == pending:
                continue
            approval = next(
                (
                    event
                    for event in events
                    if event.get("type") == "approval"
                    and event.get("approval_id") == approval_id
                ),
                {},
            )
            decision = next(
                (
                    event.get("decision")
                    for event in events
                    if event.get("type") == "approval_decision"
                    and event.get("approval_id") == approval_id
                ),
                None,
            )
            verdict = {"approved": "Approved", "rejected": "Rejected"}.get(
                str(decision)
            )
            outcome = self._agent_card_outcome(verdict) if verdict else "Expired"
            blocks = (
                self._agent_delivery_blocks(approval)
                if approval.get("kind") == "delivery"
                else self._agent_approval_blocks(approval, self._ci_repair_target(run))
                if approval
                else []
            )
            cards.pop(approval_id, None)
            closed = True
            await self._update_agent_card(tracked["channel_id"], ts, blocks, outcome)
        if closed:
            await self._persist_agent_run(tracked)

    async def _post_agent_result(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        # Final answers are posted whole: split into section-sized blocks
        # across as many thread messages as needed instead of cut at a limit.
        formatted = self.message_handler._format_agent_response(run)
        if tracked.get("owner_user_id"):
            formatted = f"<@{tracked['owner_user_id']}> {formatted}"
        chunks = _split_agent_result(formatted) or [formatted]
        messages = [
            chunks[offset : offset + AGENT_RESULT_BLOCKS_PER_MESSAGE]
            for offset in range(0, len(chunks), AGENT_RESULT_BLOCKS_PER_MESSAGE)
        ]
        actions: List[Dict[str, Any]] = []
        # Hide the buttons only when the runtime says it cannot store feedback.
        cached_capture = getattr(
            getattr(self, "agent_runtime_client", None), "cached_feedback_capture", None
        )
        feedback_capture = cached_capture() if callable(cached_capture) else None
        if (
            run.get("status") in AGENT_TERMINAL_STATUSES
            and self.agent_config.get("feedback_enabled", False)
            and feedback_capture is not False
        ):
            value = json.dumps(
                {
                    "run_id": run["run_id"],
                    "team_id": tracked.get("team_id"),
                    "channel_id": tracked["channel_id"],
                    "thread_ts": tracked["thread_ts"],
                }
            )
            actions = [
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": action_id,
                            "text": {"type": "plain_text", "text": label},
                            "value": value,
                        }
                        for action_id, label in [
                            ("pi_agent_accept_result", "Accept result"),
                            ("pi_agent_needs_changes", "Needs changes"),
                        ]
                    ],
                },
            ]
        if (
            run.get("status") in AGENT_TERMINAL_STATUSES
            and run.get("status") != "completed"
            and run.get("patch_retained") is True
        ):
            patch_value = json.dumps(
                {
                    "run_id": run["run_id"],
                    "session_id": tracked.get("session_id") or run.get("session_id"),
                    "team_id": tracked.get("team_id"),
                    "channel_id": tracked["channel_id"],
                    "thread_ts": tracked["thread_ts"],
                }
            )
            actions.insert(
                0,
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": action_id,
                            "text": {"type": "plain_text", "text": label},
                            "value": patch_value,
                        }
                        for action_id, label in [
                            ("pi_agent_download_patch", "Download patch"),
                            ("pi_agent_resume_from_patch", "Resume from patch"),
                        ]
                    ],
                },
            )
        for index, message_chunks in enumerate(messages):
            blocks: List[Dict[str, Any]] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
                for chunk in message_chunks
            ]
            if index == len(messages) - 1:
                blocks.extend(actions)
            await self.web_client.chat_postMessage(
                channel=tracked["channel_id"],
                thread_ts=tracked["thread_ts"],
                text="\n".join(message_chunks),
                blocks=blocks,
            )

    async def _open_agent_feedback_modal(self, payload: Dict[str, Any]) -> bool:
        """Open the Needs-changes modal; False lets the click record a bare verdict."""
        actions = payload.get("actions") or []
        if not actions or actions[0].get("action_id") != "pi_agent_needs_changes":
            return False
        action = actions[0]
        trigger_id = payload.get("trigger_id")
        user_id = payload.get("user", {}).get("id")
        click_id = str(action.get("action_ts") or "")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
        except (KeyError, TypeError, ValueError):
            return False
        team = payload.get("team")
        team_id = value.get("team_id") or (
            team.get("id") if isinstance(team, dict) else None
        )
        channel_id = value.get("channel_id") or payload.get("channel", {}).get("id")
        thread_ts = value.get("thread_ts") or (payload.get("container") or {}).get(
            "thread_ts"
        )
        if not all((trigger_id, user_id, click_id, channel_id, thread_ts)):
            return False

        now = time.time()
        self.agent_feedback_nonces = {
            key: nonce
            for key, nonce in self.agent_feedback_nonces.items()
            if nonce.expires_at > now
        }
        nonce = uuid.uuid4().hex
        self.agent_feedback_nonces[nonce] = AgentFeedbackNonce(
            run_id=run_id,
            team_id=team_id or "unknown-team",
            channel_id=channel_id,
            thread_ts=thread_ts,
            owner_user_id=user_id,
            feedback_id=click_id,
            expires_at=now + 1800,
        )
        try:
            await self.web_client.views_open(
                trigger_id=trigger_id, view=build_agent_feedback_modal(nonce)
            )
        except Exception:
            logger.warning("Agent feedback modal could not be opened", exc_info=True)
            self.agent_feedback_nonces.pop(nonce, None)
            return False
        return True

    async def _handle_agent_feedback_submission(self, payload: Dict[str, Any]) -> None:
        """Consume a one-shot feedback nonce, record it, and optionally follow up."""
        view = payload.get("view", {}) or {}
        nonce_value = str(view.get("private_metadata") or "")
        nonce = self.agent_feedback_nonces.get(nonce_value)
        user_id = payload.get("user", {}).get("id")
        if not nonce:
            return
        if nonce.expires_at <= time.time():
            self.agent_feedback_nonces.pop(nonce_value, None)
            return
        if nonce.owner_user_id != user_id:
            return
        self.agent_feedback_nonces.pop(nonce_value, None)

        submission = parse_agent_feedback_submission(view)
        try:
            await self.agent_runtime_client.feedback(
                nonce.run_id,
                user_id,
                "needs_changes",
                nonce.feedback_id,
                reason_category=submission.reason_category,
                comment=submission.comment or None,
            )
            message = "Result feedback recorded."
        except AgentRuntimeUnauthorized as exc:
            await self.web_client.chat_postEphemeral(
                channel=nonce.channel_id,
                user=user_id,
                text=_agent_auth_error_text(
                    exc, "Only the Agent run owner can submit result feedback."
                ),
            )
            return
        except AgentRuntimeError as exc:
            # The owner's requested follow-up does not depend on the label
            # being stored, so a lost label must not also lose the next run.
            message = _agent_feedback_error_text(exc)
        await self.web_client.chat_postEphemeral(
            channel=nonce.channel_id, user=user_id, text=message
        )
        if not (submission.continue_session and submission.comment):
            return

        if not self._within_rate_limit(user_id, agent=True):
            await self._post_command_response(
                nonce.channel_id,
                user_id,
                self._build_rate_limit_message(user_id, agent=True),
            )
            return
        response = await self.message_handler._start_agent_prompt(
            submission.comment,
            user_id,
            nonce.channel_id,
            self.web_client,
            team_id=nonce.team_id,
            thread_ts=nonce.thread_ts,
            routing_text=submission.comment,
            request_id=f"feedback:{nonce.feedback_id}",
        )
        if response:
            await self.web_client.chat_postMessage(
                channel=nonce.channel_id, thread_ts=nonce.thread_ts, text=response
            )
        await self._persist_message_state(user_id=user_id)

    async def _handle_agent_shortcut(self, payload: Dict[str, Any]) -> None:
        """Open the Agent bootstrap modal while the Slack trigger is still valid."""
        trigger_id = payload.get("trigger_id")
        team_id = payload.get("team", {}).get("id") or self._extract_team_id(payload)
        user_id = payload.get("user", {}).get("id")
        channel_id = payload.get("channel", {}).get("id")
        message = payload.get("message", {}) or {}
        selected_ts = message.get("ts")
        thread_ts = message.get("thread_ts") or selected_ts
        if not all((trigger_id, team_id, user_id, channel_id, selected_ts, thread_ts)):
            return
        if not await self._is_channel_allowed(channel_id):
            return
        if not await self._is_public_agent_channel(channel_id):
            return

        now = time.time()
        self.agent_shortcut_nonces = {
            key: value
            for key, value in self.agent_shortcut_nonces.items()
            if value.expires_at + AGENT_SHORTCUT_NONCE_GRACE_SECONDS > now
        }
        nonce = uuid.uuid4().hex
        self.agent_shortcut_nonces[nonce] = AgentShortcutNonce(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            selected_message_ts=selected_ts,
            owner_user_id=user_id,
            expires_at=now + AGENT_SHORTCUT_NONCE_TTL_SECONDS,
        )

        runtime = self.agent_runtime_client
        cached_models = getattr(runtime, "cached_configured_models", None)
        model_refs = list(cached_models()) if callable(cached_models) else []
        # Never preselect an upgrade of a read-only discussion thread.
        thread = self.agent_threads.get(self._thread_key(channel_id, thread_ts)) or {}
        initial_task_mode = (
            "discuss"
            if thread.get("owner_user_id") == user_id
            and thread.get("mode") == "discuss"
            else "implement"
        )
        await self.web_client.views_open(
            trigger_id=trigger_id,
            view=build_agent_shortcut_modal(nonce, model_refs, initial_task_mode),
        )

    async def _is_public_agent_channel(self, channel_id: str) -> bool:
        """Fail closed unless Slack confirms a non-shared public channel."""
        method = getattr(self.web_client, "conversations_info", None)
        if not callable(method):
            logger.warning("Cannot verify public channel for Agent shortcut")
            return False
        try:
            response = await method(channel=channel_id)
            if not response.get("ok", True):
                return False
            channel = response.get("channel", {}) or {}
            return bool(channel.get("is_channel")) and not any(
                channel.get(flag)
                for flag in (
                    "is_private",
                    "is_im",
                    "is_mpim",
                    "is_shared",
                    "is_ext_shared",
                )
            )
        except Exception:
            logger.warning(
                "Failed to verify public channel for Agent shortcut", exc_info=True
            )
            return False

    async def _refresh_agent_model_cache(self) -> None:
        runtime = self.agent_runtime_client
        health_details = getattr(runtime, "health_details", None)
        if not callable(health_details):
            return
        try:
            await health_details()
        except Exception:
            logger.warning("Agent model health cache refresh failed", exc_info=True)

    async def _refresh_agent_model_cache_periodically(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            if self.running:
                await self._refresh_agent_model_cache()

    async def _handle_agent_view_submission(self, payload: Dict[str, Any]) -> None:
        """Consume a one-shot modal nonce and start one quota-counted Agent prompt."""
        view = payload.get("view", {}) or {}
        if view.get("callback_id") == AGENT_FEEDBACK_CALLBACK_ID:
            await self._handle_agent_feedback_submission(payload)
            return
        nonce_value = str(view.get("private_metadata") or "")
        nonce = self.agent_shortcut_nonces.get(nonce_value)
        user_id = payload.get("user", {}).get("id")
        if not nonce:
            return
        if nonce.expires_at <= time.time():
            self.agent_shortcut_nonces.pop(nonce_value, None)
            if nonce.owner_user_id == user_id:
                task = parse_agent_shortcut_submission(view).task
                await self.web_client.chat_postEphemeral(
                    channel=nonce.channel_id,
                    user=user_id,
                    text=(
                        "This Agent form expired before it was submitted. "
                        "Run the shortcut again."
                        + (_agent_task_echo(task) if task else "")
                    ),
                )
            return
        if nonce.owner_user_id != user_id:
            return
        self.agent_shortcut_nonces.pop(nonce_value, None)

        submission = parse_agent_shortcut_submission(view)
        if not submission.task:
            return
        if not await self._is_public_agent_channel(nonce.channel_id):
            return
        cap_message = self._agent_cap_message(
            user_id, nonce.channel_id, nonce.thread_ts
        )
        if cap_message:
            await self.web_client.chat_postEphemeral(
                channel=nonce.channel_id,
                user=user_id,
                text=cap_message + _agent_task_echo(submission.task),
            )
            return

        if not self._within_rate_limit(user_id, agent=True):
            await self._post_command_response(
                nonce.channel_id,
                user_id,
                self._build_rate_limit_message(user_id, agent=True),
            )
            return

        status = await self.web_client.chat_postMessage(
            channel=nonce.channel_id,
            thread_ts=nonce.thread_ts,
            text="🛠️ Pi coding agent is loading the Slack thread snapshot…",
        )
        # Bootstrap and create_session can take minutes; never hold a worker.
        self._spawn_background_task(
            self._start_agent_from_submission(
                nonce, nonce_value, submission, user_id, status.get("ts")
            ),
            "pi_agent_shortcut_start",
        )

    async def _start_agent_from_submission(
        self,
        nonce: AgentShortcutNonce,
        nonce_value: str,
        submission: AgentShortcutSubmission,
        user_id: str,
        status_ts: Optional[str],
    ) -> None:
        try:
            await self._bootstrap_agent_submission(
                nonce, nonce_value, submission, user_id, status_ts
            )
        except Exception:
            logger.exception("Agent shortcut start failed")
            if status_ts:
                await self.web_client.chat_update(
                    channel=nonce.channel_id,
                    ts=status_ts,
                    text="Agent mode could not start this task. Chat mode was not used as fallback.",
                )
        finally:
            await self._persist_message_state(user_id=user_id)

    async def _bootstrap_agent_submission(
        self,
        nonce: AgentShortcutNonce,
        nonce_value: str,
        submission: AgentShortcutSubmission,
        user_id: str,
        status_ts: Optional[str],
    ) -> None:
        bootstrap = AgentBootstrapContext(prompt=submission.task)
        if submission.include_context:
            try:
                bootstrap = await self.agent_bootstrap_builder.build(
                    self.web_client,
                    channel_id=nonce.channel_id,
                    thread_ts=nonce.thread_ts,
                    owner_user_id=user_id,
                    task=submission.task,
                    selected_ts=nonce.selected_message_ts,
                )
            except AgentBootstrapFetchError as exc:
                if status_ts:
                    await self.web_client.chat_update(
                        channel=nonce.channel_id,
                        ts=status_ts,
                        text=(
                            "Agent mode did not start because Slack thread context "
                            f"could not be loaded: `{exc}`."
                        ),
                    )
                return

        details = []
        if submission.include_context:
            details.append(
                f"Loaded {bootstrap.message_count} prior Slack messages and "
                f"{bootstrap.resource_count} resources."
            )
            if bootstrap.truncated:
                details.append("Context truncated to configured limits.")
            if bootstrap.partial:
                details.append(
                    "Context is partial: the Slack fetch time budget ran out."
                )
            if bootstrap.error:
                details.append(
                    f"Started without prior Slack context: `{bootstrap.error}`."
                )
            if bootstrap.warnings:
                details.append(
                    "Skipped or partial resources: `"
                    + "`, `".join(bootstrap.warnings[:10])
                    + "`."
                )
            details.append("Loaded context was sent to the selected model provider.")
        # The loading message becomes the run's status message once it starts.
        response = await self.message_handler._start_agent_prompt(
            bootstrap.prompt,
            user_id,
            nonce.channel_id,
            self.web_client,
            team_id=nonce.team_id,
            thread_ts=nonce.thread_ts,
            model=submission.model,
            routing_text=submission.task,
            request_id=f"shortcut:{nonce_value}",
            task_mode=submission.task_mode,
            workflow_mode=submission.workflow_mode,
            status_ts=status_ts,
            status_note="\n".join(details),
        )
        if response and status_ts:
            await self.web_client.chat_update(
                channel=nonce.channel_id,
                ts=status_ts,
                text="\n".join(part for part in (response, *details) if part),
            )

    async def _handle_agent_interactive(self, payload: Dict[str, Any]):
        actions = payload.get("actions") or []
        if not actions:
            return
        action = actions[0]
        if action.get("action_id") in {
            "pi_agent_download_patch",
            "pi_agent_resume_from_patch",
        }:
            await self._handle_agent_patch_action(payload, action)
            return
        if action.get("action_id") in {
            "pi_agent_accept_result",
            "pi_agent_needs_changes",
        }:
            user_id = payload.get("user", {}).get("id")
            channel_id = payload.get("channel", {}).get("id")
            try:
                value = json.loads(action.get("value", "{}"))
                verdict = (
                    "accepted"
                    if action["action_id"] == "pi_agent_accept_result"
                    else "needs_changes"
                )
                click_id = str(
                    action.get("action_ts") or payload.get("trigger_id") or ""
                )
                if not user_id or not click_id:
                    return
                await self.agent_runtime_client.feedback(
                    value["run_id"], user_id, verdict, click_id
                )
                message = "Result feedback recorded."
            except AgentRuntimeUnauthorized as exc:
                message = _agent_auth_error_text(
                    exc, "Only the Agent run owner can submit result feedback."
                )
            except AgentRuntimeError as exc:
                message = _agent_feedback_error_text(exc)
            except (KeyError, TypeError, ValueError):
                message = AGENT_FEEDBACK_RETRY_TEXT
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=message
            )
            return
        if action.get("action_id") not in {"pi_agent_approve", "pi_agent_reject"}:
            return
        user_id = payload.get("user", {}).get("id")
        channel_id = payload.get("channel", {}).get("id")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
            approval_id = value["approval_id"]
        except (KeyError, TypeError, ValueError):
            return
        tracked = self.agent_runs.get(run_id)
        if tracked and tracked.get("owner_user_id") != user_id:
            if hasattr(self.web_client, "chat_postEphemeral"):
                await self.web_client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="Only the user who started this Agent session can approve it.",
                )
            return
        decision = "approve" if action["action_id"] == "pi_agent_approve" else "reject"
        message = payload.get("message") or {}
        card_ts = message.get("ts") or (payload.get("container") or {}).get(
            "message_ts"
        )
        card_blocks = message.get("blocks") or []
        # Claim the card first so the monitor does not close it without the decider.
        cards = tracked.get("approval_cards") if tracked else None
        claimed = cards.pop(approval_id, None) if isinstance(cards, dict) else None
        try:
            await self.agent_runtime_client.decide(
                run_id, approval_id, user_id, decision
            )
        except AgentRuntimeError as exc:
            if isinstance(exc, AgentRuntimeConflict):
                if channel_id and card_ts:
                    await self._update_agent_card(
                        channel_id, card_ts, card_blocks, "Expired"
                    )
            elif claimed is not None and isinstance(cards, dict):
                cards[approval_id] = claimed
            if isinstance(exc, AgentRuntimeUnauthorized):
                text = _agent_auth_error_text(
                    exc, "Only the Agent session owner can approve this action."
                )
            elif isinstance(exc, AgentRuntimeConflict):
                text = "This approval has expired or was already used."
            else:
                logger.warning("Agent decision failed for %s", run_id)
                text = "Agent decision could not be recorded. Please try again."
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=text
            )
            return
        if channel_id and card_ts:
            await self._update_agent_card(
                channel_id,
                card_ts,
                card_blocks,
                self._agent_card_outcome(
                    "Approved" if decision == "approve" else "Rejected", user_id
                ),
            )
        if not tracked:
            # The bot lost this run (e.g. restarted without state); the runtime
            # accepted the owner's decision, so resume tracking from the card.
            container = payload.get("container") or {}
            message = payload.get("message") or {}
            thread_ts = (
                container.get("thread_ts")
                or message.get("thread_ts")
                or container.get("message_ts")
                or message.get("ts")
            )
            if not value.get("session_id") or not channel_id or not thread_ts:
                return
            team = payload.get("team")
            team_id = team.get("id") if isinstance(team, dict) else None
            tracked = self._track_agent_run(
                value,
                team_id=team_id or self._extract_team_id(payload) or "unknown-team",
                channel_id=channel_id,
                thread_ts=thread_ts,
                owner_user_id=user_id,
            )
        if approval_id not in tracked.setdefault("posted_approval_ids", []):
            tracked["posted_approval_ids"].append(approval_id)
        self._start_agent_monitor(tracked)

    async def _handle_agent_patch_action(
        self, payload: Dict[str, Any], action: Dict[str, Any]
    ) -> None:
        """Download a failed run's kept patch, or start a new run from it (owner only)."""
        user_id = payload.get("user", {}).get("id")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
        except (KeyError, TypeError, ValueError):
            return
        tracked = self.agent_runs.get(run_id) or {}
        channel_id = (
            value.get("channel_id")
            or tracked.get("channel_id")
            or payload.get("channel", {}).get("id")
        )
        thread_ts = value.get("thread_ts") or tracked.get("thread_ts")
        team = payload.get("team")
        team_id = (
            value.get("team_id")
            or tracked.get("team_id")
            or (team.get("id") if isinstance(team, dict) else None)
        )
        if not user_id or not channel_id or not thread_ts:
            return

        async def tell(text: str) -> None:
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=text
            )

        if tracked and tracked.get("owner_user_id") != user_id:
            await tell(AGENT_PATCH_OWNER_TEXT)
            return
        runtime = self.agent_runtime_client
        if action["action_id"] == "pi_agent_download_patch":
            try:
                patch = await runtime.get_run_patch(run_id, user_id)
            except AgentRuntimeError as exc:
                await tell(self._agent_patch_error_text(exc))
                return
            comment = f"Kept patch of Agent run `{run_id}` (secrets redacted)."
            if patch.get("truncated"):
                comment += " It was truncated to the download size limit."
            try:
                await self.web_client.files_upload_v2(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    content=patch["patch"],
                    filename=f"pi-agent-{run_id}.patch",
                    title=f"Agent run {run_id} patch",
                    initial_comment=comment,
                )
            except Exception:
                logger.warning("Agent patch upload failed for %s", run_id)
                await tell(
                    "The patch could not be uploaded to Slack. Please try again."
                )
            return
        try:
            accepted = await runtime.resume_from_patch(
                run_id,
                user_id,
                # One resume per run and owner, however often the button is clicked.
                _agent_idempotency_key(
                    team_id, channel_id, run_id, user_id, "resume_from_patch"
                ),
            )
        except AgentRuntimeError as exc:
            await tell(self._agent_patch_error_text(exc))
            return
        resumed = self._track_agent_run(
            accepted,
            team_id=team_id or "unknown-team",
            channel_id=channel_id,
            thread_ts=thread_ts,
            owner_user_id=user_id,
            mode=tracked.get("mode"),
        )
        text = (
            f"🛠️ Pi coding agent resumed from the kept patch of run `{run_id}` "
            f"as run `{accepted['run_id']}`."
        )
        await _show_agent_status(self.web_client, resumed, text)
        self._start_agent_monitor(resumed)

    @staticmethod
    def _agent_patch_error_text(exc: AgentRuntimeError) -> str:
        if isinstance(exc, AgentRuntimeUnauthorized) or exc.status == 403:
            return _agent_auth_error_text(exc, AGENT_PATCH_OWNER_TEXT)
        if exc.status == 404:
            return "No kept patch is available for this Agent run any more."
        known = AGENT_PATCH_RESUME_ERROR_TEXT.get(exc.code or "")
        if known:
            return known
        if isinstance(exc, AgentRuntimeConflict):
            return "This Agent run cannot be resumed from its patch right now."
        logger.warning("Agent patch action failed (code=%s)", exc.code)
        return "The Agent patch action failed. Please try again."

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

    def _is_supported_inline_command(self, text: str) -> bool:
        """Return whether inline `/llm` / `!llm` text maps to a supported command."""
        extract_command_text = getattr(
            self.message_handler, "extract_prefixed_command_text", None
        )
        if not callable(extract_command_text):
            return False
        command_text = extract_command_text(text)
        if command_text is None:
            return False
        is_supported_command = getattr(
            self.message_handler, "is_supported_command", None
        )
        if not callable(is_supported_command):
            return False
        return is_supported_command(command_text)

    def _is_query_text(self, text: str) -> bool:
        """Return whether a Slack message should count against rate limits."""
        normalized = (text or "").strip()
        if not normalized:
            return False
        extract_command_text = getattr(
            self.message_handler, "extract_prefixed_command_text", None
        )
        consumes_query_quota = getattr(
            self.message_handler, "command_consumes_query_quota", None
        )
        if callable(extract_command_text) and callable(consumes_query_quota):
            command_text = extract_command_text(normalized)
            if command_text is not None:
                return consumes_query_quota(command_text)
        return not self._is_supported_inline_command(normalized)

    def _is_query_event(self, event: Dict[str, Any]) -> bool:
        """Return whether a Slack event should consume query quota."""
        return self._is_query_text(event.get("text", "")) or bool(event.get("files"))

    def _is_agent_request(self, event: Dict[str, Any]) -> bool:
        """Whether a user's message goes to Agent mode rather than chat.

        That is an `agent` command, or any other message in an Agent thread,
        which continues its session. A command shared with a file or sent to
        the channel too is one as well, so it takes the Agent lane and bucket
        whichever of Slack's events for it arrives first.
        """
        if event.get("subtype") not in CHAT_MESSAGE_SUBTYPES or event.get("bot_id"):
            return False
        text = event.get("text") or ""
        if self.bot_user_id:
            text = text.replace(f"<@{self.bot_user_id}>", "")
        text = text.strip()
        extract_command_text = getattr(
            self.message_handler, "extract_prefixed_command_text", None
        )
        if callable(extract_command_text):
            command_text = extract_command_text(text)
            if command_text is not None:
                return _is_agent_command_text(command_text)
        return bool(text or event.get("files")) and self._is_agent_thread(
            event.get("channel"), event.get("thread_ts")
        )

    def _synchronize_user_tier(self, user_id: Optional[str]) -> bool:
        """Apply configured user-tier defaults/overrides to the in-memory state."""
        return self.user_manager.ensure_user_record(user_id or "")

    def _get_rate_limit_config_for_tier(self, user_tier: UserTier) -> Dict[str, int]:
        """Resolve rate-limit settings for a concrete tier."""
        base_limits = {
            "requests_per_hour": int(
                self.rate_limiting.get("requests_per_hour", 100) or 100
            ),
            "burst_requests": int(self.rate_limiting.get("burst_requests", 5) or 5),
        }
        tier_overrides = self.rate_limiting.get("by_tier", {}).get(user_tier.value, {})
        if isinstance(tier_overrides, dict):
            for key in ("requests_per_hour", "burst_requests"):
                if key in tier_overrides:
                    base_limits[key] = int(tier_overrides[key])
        return base_limits

    def _get_rate_limit_config_for_user(self, user_id: str) -> Dict[str, int]:
        """Resolve rate-limit settings for the user's current tier."""
        return self._get_rate_limit_config_for_tier(
            self.user_manager.get_user_tier(user_id)
        )

    def _within_rate_limit(self, user_id: str, *, agent: bool) -> bool:
        """Count one request in the user's chat or Agent bucket, unless it is full."""
        return self.user_manager.check_rate_limit(
            user_id,
            self._get_rate_limit_config_for_user(user_id),
            **({"bucket": RATE_LIMIT_AGENT} if agent else {}),
        )

    def _format_tier_limit_summary(self) -> str:
        """Render the configured tier limits for Slack help text."""
        summaries = []
        for tier in (UserTier.FREE, UserTier.PREMIUM, UserTier.ENTERPRISE):
            limits = self._get_rate_limit_config_for_tier(tier)
            summaries.append(
                f"{self._tier_emoji(tier)} {tier.value.title()}: "
                f"{limits['requests_per_hour']} requests/hour, "
                f"{limits['burst_requests']} burst requests / 5 min"
            )
        return "\n".join(summaries)

    def _tier_emoji(self, tier: UserTier) -> str:
        if tier == UserTier.PREMIUM:
            return "💎"
        if tier == UserTier.ENTERPRISE:
            return "🏢"
        return "🆓"

    async def _post_command_response(self, channel_id: str, user_id: str, text: str):
        """Respond to a slash command, preferring ephemeral responses."""
        max_length = self.config.get("response_settings", {}).get(
            "max_response_length", 2000
        )
        responses = self._split_response(text, max_length)

        if hasattr(self.web_client, "chat_postEphemeral"):
            for response_text in responses:
                await self.web_client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=response_text,
                )
            return

        for response_text in responses:
            await self.web_client.chat_postMessage(
                channel=channel_id, text=response_text
            )

    def _spawn_background_task(self, coroutine, name: str):
        """Track background tasks so request acks stay fast without losing errors."""
        task = asyncio.create_task(coroutine, name=name)
        self.background_tasks.add(task)
        task.add_done_callback(self._finalize_background_task)
        return task

    def _start_work_workers(self) -> None:
        if self._work_workers:
            return
        self._accepting_work = True
        self._work_workers = [
            asyncio.create_task(
                self._work_worker(index, self._work_queue),
                name=f"slack_work_worker:{index}",
            )
            for index in range(self._work_concurrency)
        ] + [
            asyncio.create_task(
                self._work_worker(index, self._priority_queue),
                name=f"slack_priority_worker:{index}",
            )
            for index in range(self._priority_concurrency)
        ]

    async def _stop_work_workers(self) -> None:
        if not self._work_workers:
            return
        self._accepting_work = False
        queues = (self._work_queue, self._priority_queue)
        try:
            for queue in queues:
                await queue.join()
        finally:
            workers, self._work_workers = self._work_workers, []
            for queue in queues:
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    else:
                        queue.task_done()
                        SLACK_METRICS.work_rejected.labels(reason="shutdown").inc()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            SLACK_METRICS.work_queue_depth.set(0)
            SLACK_METRICS.work_running.set(0)

    def _is_ignored_message_event(self, event: Dict[str, Any]) -> bool:
        """A message _handle_message_event always drops.

        That is a bot's post, an edit or another subtype nobody asks with, or a
        file share or broadcast in a known Agent thread. Dropping it before the
        queue keeps the bot from answering its own busy reply with another one.
        """
        if event.get("type") != "message":
            return False
        subtype = event.get("subtype")
        return bool(
            subtype not in CHAT_MESSAGE_SUBTYPES
            or event.get("bot_id")
            or (self.bot_user_id and event.get("user") == self.bot_user_id)
            or (
                subtype is not None
                and self._is_agent_thread(event.get("channel"), event.get("thread_ts"))
            )
        )

    def _wants_busy_reply(self, item: SlackWorkItem) -> bool:
        """Whether someone is waiting on the bot: a slash command, mention or thread.

        Read-only: a busy reply neither extends nor expires a thread.
        """
        if item.kind == "slash":
            return True
        if item.kind != "event":
            return False
        event_type = item.payload.get("type")
        if event_type == "app_mention":
            return True
        if event_type != "message":
            return False
        channel_id = item.payload.get("channel")
        thread_ts = item.payload.get("thread_ts")
        return self._is_agent_thread(channel_id, thread_ts) or self._is_active_thread(
            channel_id, thread_ts, refresh=False
        )

    async def _enqueue_work(self, item: SlackWorkItem) -> bool:
        if item.kind == "event" and self._is_ignored_message_event(item.payload):
            return False
        if not self._accepting_work:
            await self._reply_busy(item)
            return False
        queue = (
            self._priority_queue
            if item.kind in SLACK_PRIORITY_WORK_KINDS
            else self._work_queue
        )
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            SLACK_METRICS.work_rejected.labels(reason="capacity").inc()
            await self._reply_busy(item)
            return False
        SLACK_METRICS.work_queue_depth.set(self._queued_work())
        return True

    def _queued_work(self) -> int:
        return self._work_queue.qsize() + self._priority_queue.qsize()

    async def _work_worker(self, index: int, queue: asyncio.Queue) -> None:
        while True:
            item = await queue.get()
            self._work_running += 1
            SLACK_METRICS.work_queue_depth.set(self._queued_work())
            SLACK_METRICS.work_running.set(self._work_running)
            try:
                if item.kind == "slash":
                    await self._handle_slash_command(item.payload)
                elif item.kind == "interactive":
                    await self._handle_agent_interactive(item.payload)
                elif item.kind == "agent_view":
                    await self._handle_agent_view_submission(item.payload)
                elif self._is_agent_request(item.payload):
                    # Bootstrap and create_session can take minutes, like a
                    # slash start's; the thread gets the status or the error.
                    self._spawn_background_task(
                        self._handle_event(item.payload), "pi_agent_start"
                    )
                else:
                    await self._handle_event(item.payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Slack work item failed: %s", item.name)
            finally:
                self._work_running -= 1
                SLACK_METRICS.work_running.set(self._work_running)
                queue.task_done()

    async def _reply_busy(self, item: SlackWorkItem) -> None:
        if not self._wants_busy_reply(item):
            return
        if item.kind == "event" and not self._claim_message(
            item.payload, self.busy_replied_messages
        ):
            return

        async def send_reply() -> bool:
            channel = item.payload.get("channel") or item.payload.get("channel_id")
            user = item.payload.get("user") or item.payload.get("user_id")
            if item.kind == "slash" and hasattr(self.web_client, "chat_postEphemeral"):
                # Like _handle_slash_command, which answers in any channel.
                await self.web_client.chat_postEphemeral(
                    channel=channel, user=user, text=SLACK_BUSY_MESSAGE
                )
                return True
            # Like _handle_*_event: stay silent where the bot is not enabled.
            if not await self._is_channel_allowed(
                channel, item.payload.get("channel_name")
            ):
                return False
            kwargs = {"channel": channel, "text": SLACK_BUSY_MESSAGE}
            thread_ts = item.payload.get("thread_ts") or item.payload.get("ts")
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            await self.web_client.chat_postMessage(**kwargs)
            return True

        try:
            sent = await asyncio.wait_for(
                send_reply(), timeout=self._overload_reply_timeout_seconds
            )
        except Exception:
            SLACK_METRICS.busy_replies.labels(outcome="failed").inc()
            logger.warning("Failed to send Slack overload reply", exc_info=True)
        else:
            if sent:
                SLACK_METRICS.busy_replies.labels(outcome="sent").inc()

    def _finalize_background_task(self, task: asyncio.Task):
        """Log unexpected background task failures and drop completed tasks."""
        self.background_tasks.discard(task)

        if task.cancelled():
            return

        try:
            exception = task.exception()
        except Exception as exc:
            logger.error(f"Failed to inspect Slack background task: {exc}")
            return

        if exception:
            logger.error(f"Slack background task failed: {exception}")

    def _get_uptime_text(self) -> str:
        """Render a human-readable uptime string."""
        if not self.started_at:
            return "Not started"

        uptime = datetime.now() - self.started_at
        total_seconds = int(uptime.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _seconds = divmod(remainder, 60)

        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _build_query_metadata(
        self,
        channel_id: str,
        thread_ts: Optional[str],
        attachments: List[Attachment],
        user_preferences: Optional[Dict[str, Any]] = None,
        team_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach Slack-specific metadata to routed queries."""
        user_preferences = user_preferences or {}
        preferred_models = [
            model_name
            for model_name in user_preferences.get("preferred_models", [])
            if model_name
        ]
        response_style_instructions = self._build_response_style_instructions(
            user_preferences
        )
        return {
            "source": "slack",
            "workspace_id": team_id or "default",
            "preferred_models": preferred_models,
            "response_style_instructions": response_style_instructions,
            "web_search_auto": user_preferences.get("web_search_auto", True),
            "slack": {
                "team_id": team_id,
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "attachment_count": len(attachments),
                "attachment_names": [attachment.name for attachment in attachments],
                "technical_level": user_preferences.get(
                    "technical_level", "intermediate"
                ),
                "threading": user_preferences.get("threading", True),
                "response_length": user_preferences.get("response_length", "medium"),
            },
        }

    def _build_response_style_instructions(
        self, user_preferences: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Translate Slack user preferences into provider-friendly style instructions."""
        preferences = user_preferences or {}
        technical_level = preferences.get("technical_level", "intermediate")
        response_length = preferences.get("response_length", "medium")

        technical_guidance = {
            "beginner": "Explain in simple terms, define jargon, and include short practical examples.",
            "intermediate": "Assume working familiarity with the topic and balance clarity with technical depth.",
            "expert": "Use precise technical terminology, skip foundational explanations, and focus on advanced detail.",
        }
        length_guidance = {
            "short": "Keep the response concise and focused on the key answer.",
            "medium": "Use a moderate amount of detail.",
            "long": "Provide a thorough, detailed answer when the topic warrants it.",
        }

        instructions = []
        if technical_level in technical_guidance:
            instructions.append(technical_guidance[technical_level])
        if response_length in length_guidance:
            instructions.append(length_guidance[response_length])

        if not instructions:
            return None
        return " ".join(instructions)

    def _build_user_safe_error_message(self) -> str:
        """Return a generic Slack-safe error message without leaking internals."""
        return "❌ Sorry, I couldn't process that request right now. Please try again in a moment."

    def _extract_team_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract Slack team/workspace id from event or command payloads."""
        if not payload:
            return None
        for key in ("team_id", "team", "_team_id"):
            value = payload.get(key)
            if value:
                return str(value)

        authorizations = payload.get("authorizations") or []
        if authorizations and isinstance(authorizations[0], dict):
            value = authorizations[0].get("team_id")
            if value:
                return str(value)

        event_context = payload.get("event_context")
        if isinstance(event_context, str) and event_context:
            parts = event_context.split("-")
            if len(parts) > 1 and parts[1].startswith("T"):
                return parts[1]
        return None

    def _memory_scope(self, team_id: Optional[str], user_id: str) -> str:
        """Build the per-workspace/per-user long-term memory scope."""
        return build_memory_scope(team_id, user_id)

    def _build_memory_metadata(
        self,
        channel_id: Optional[str],
        thread_ts: Optional[str],
        visibility: str = "channel",
    ) -> Dict[str, Any]:
        """Build stable metadata stored with explicit Slack memories."""
        normalized_visibility = "global" if visibility == "global" else "channel"
        metadata = {"source": "slack", "visibility": normalized_visibility}
        if normalized_visibility == "channel" and channel_id:
            metadata["channel_id"] = channel_id
        if normalized_visibility == "channel" and thread_ts:
            metadata["thread_ts"] = thread_ts
        return metadata

    def _warn_if_memory_uses_shared_redis(self):
        """Warn when Redis Stack memory is likely sharing the response cache DB."""
        memory_config = self.config.get("memory", {}) or {}
        if not memory_config.get("enabled"):
            return
        if str(memory_config.get("backend", "")).lower() != "redis_stack":
            return

        memory_redis = dict(memory_config.get("redis", {}) or {})
        if memory_redis.get("url"):
            return
        db = int(memory_redis.get("db", 3))
        host = memory_redis.get("host", "localhost")
        if db == 0:
            logger.warning(
                "Slack memory Redis is configured for DB 0 on %s. "
                "Use a dedicated Redis Stack service or at least a separate DB/prefix "
                "to avoid evicting response cache entries.",
                host,
            )

    def _conversation_context_key(
        self, user_id: str, channel_id: str, thread_ts: Optional[str]
    ) -> str:
        """Build the canonical conversation key used by the conversation manager."""
        return f"{user_id}:{channel_id}:{thread_ts or 'main'}"

    async def _extract_query_attachments(
        self, event: Dict[str, Any]
    ) -> List[Attachment]:
        """Convert Slack file payloads into QueryRequest attachments."""
        if not self.attachment_settings.get("enabled", True):
            return []

        files = list(event.get("files", []) or [])
        if not files:
            return []

        max_files = min(
            int(self.attachment_settings.get("max_files", 10) or 10),
            10,
        )
        attachments = []

        for file_payload in files[:max_files]:
            attachment = await self._build_attachment_from_slack_file(file_payload)
            if attachment is not None:
                attachments.append(attachment)

        return attachments

    async def _build_attachment_from_slack_file(
        self, file_payload: Dict[str, Any]
    ) -> Optional[Attachment]:
        """Build a normalized attachment object from Slack file metadata."""
        if not is_slack_hosted_file(file_payload):
            return None

        download_url = file_payload.get("url_private_download") or file_payload.get(
            "url_private"
        )
        mime_type = file_payload.get("mimetype") or "application/octet-stream"
        attachment_type = self._infer_attachment_type(mime_type)
        name = (
            file_payload.get("name")
            or file_payload.get("title")
            or file_payload.get("id")
        )
        if not name:
            return None

        size_bytes = int(file_payload.get("size") or 0)
        content = None
        max_file_size_bytes = int(
            self.attachment_settings.get("max_file_size_bytes", 10_000_000)
            or 10_000_000
        )

        if download_url and size_bytes <= max_file_size_bytes:
            content = await self._download_attachment_content(
                file_payload, str(download_url), max_file_size_bytes
            )
            if content is not None and size_bytes <= 0:
                size_bytes = len(content)

        if size_bytes <= 0:
            if content:
                size_bytes = len(content)
            else:
                logger.warning(
                    "Skipping Slack file without usable size metadata: %s", name
                )
                return None

        return Attachment(
            id=file_payload.get("id") or str(uuid.uuid4()),
            name=name,
            type=attachment_type,
            size_bytes=size_bytes,
            mime_type=mime_type,
            url=download_url,
            content=content,
        )

    async def _download_attachment_content(
        self, file_payload: Dict[str, Any], download_url: str, max_bytes: int
    ) -> Optional[bytes]:
        """Download a private Slack file using the bot token.

        The token only goes to Slack file hosts. Logs name the file id and the
        failure, never the URL.
        """
        if not download_url or not self.bot_token:
            return None

        file_id = str(file_payload.get("id") or "unknown")
        if not is_allowed_slack_file_url(download_url):
            logger.warning("Skipping Slack attachment %s: unsafe_download_url", file_id)
            return None

        try:
            return await asyncio.wait_for(
                download_slack_file(
                    self._get_attachment_http_client(),
                    download_url,
                    bot_token=self.bot_token,
                    max_bytes=max_bytes,
                    allow_html=declares_html(file_payload),
                ),
                timeout=self._attachment_download_timeout(),
            )
        except Exception as exc:
            logger.warning(
                "Failed to download Slack attachment %s: %s",
                file_id,
                error_name(exc),
            )
            return None

    def _attachment_download_timeout(self) -> float:
        return float(self.attachment_settings.get("download_timeout_seconds", 30) or 30)

    def _get_attachment_http_client(self) -> httpx.AsyncClient:
        if self._attachment_http_client is None:
            self._attachment_http_client = httpx.AsyncClient(
                timeout=self._attachment_download_timeout(),
                follow_redirects=False,
            )
        return self._attachment_http_client

    async def _close_attachment_http_client(self) -> None:
        client = self._attachment_http_client
        self._attachment_http_client = None
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as exc:
            logger.warning(
                "Slack attachment HTTP client close failed: %s", error_name(exc)
            )

    def _infer_attachment_type(self, mime_type: str) -> AttachmentType:
        """Map a MIME type to the internal attachment type enum."""
        normalized = (mime_type or "").lower()
        if normalized.startswith("image/"):
            return AttachmentType.IMAGE
        if normalized.startswith("audio/"):
            return AttachmentType.AUDIO
        if (
            normalized.startswith("text/")
            or "pdf" in normalized
            or "document" in normalized
            or "spreadsheet" in normalized
            or "presentation" in normalized
            or "json" in normalized
            or "csv" in normalized
        ):
            return AttachmentType.DOCUMENT
        return AttachmentType.FILE

    def _uses_granular_redis_state(self) -> bool:
        """Return whether the configured state backend supports granular Redis writes."""
        return isinstance(self.state_store, RedisSlackStateStore)

    async def _persist_user_state(self, user_id: str):
        """Persist one user's settings and rate-limit counters."""
        if not user_id:
            return

        if self._uses_granular_redis_state():
            await self.state_store.persist_user_state(
                user_id,
                self.user_manager.users.get(user_id, {}),
                self.user_manager.rate_limits.get(user_id, []),
            )
            return

        await self._persist_state()

    async def _persist_conversation_state(self, context_key: Optional[str]):
        """Persist one conversation context."""
        if not context_key:
            return

        if self._uses_granular_redis_state():
            context = self.conversation_manager.conversations.get(context_key)
            if context is None:
                await self.state_store.delete_conversation_states([context_key])
                return

            await self.state_store.persist_conversation_state(
                context_key,
                self._serialize_conversation_context(context),
            )
            return

        await self._persist_state()

    async def _delete_conversation_states(self, context_keys: List[str]):
        """Delete one or more persisted conversation contexts."""
        if not context_keys:
            return

        if self._uses_granular_redis_state():
            await self.state_store.delete_conversation_states(context_keys)
            return

        await self._persist_state()

    async def _persist_active_thread_state(self, thread_key: Optional[str]):
        """Persist one active bot-thread marker."""
        if not thread_key:
            return

        if self._uses_granular_redis_state():
            last_activity = self.active_threads.get(thread_key)
            if last_activity is None:
                await self.state_store.delete_active_thread_states([thread_key])
                return

            await self.state_store.persist_active_thread_state(
                thread_key,
                last_activity.isoformat(),
            )
            return

        await self._persist_state()

    async def _delete_active_thread_states(self, thread_keys: List[str]):
        """Delete one or more active bot-thread markers."""
        if not thread_keys:
            return

        if self._uses_granular_redis_state():
            await self.state_store.delete_active_thread_states(thread_keys)
            return

        await self._persist_state()

    async def _persist_message_state(
        self,
        user_id: Optional[str] = None,
        conversation_key: Optional[str] = None,
        active_thread_key: Optional[str] = None,
    ):
        """Persist only the pieces of state changed by a handled Slack message."""
        if user_id:
            await self._persist_user_state(user_id)
        if conversation_key:
            await self._persist_conversation_state(conversation_key)
        if active_thread_key:
            await self._persist_active_thread_state(active_thread_key)

    async def _restore_state(self):
        """Hydrate in-memory Slack state from the configured backend."""
        snapshot = await self.state_store.load_state()
        if not snapshot:
            return

        self.user_manager.users = dict(snapshot.get("users", {}))
        self.user_manager.rate_limits = {
            user_id: [
                _rate_limit_entry(*entry)
                for entry in _rate_limit_entries(values, float("-inf"))
            ]
            for user_id, values in snapshot.get("rate_limits", {}).items()
        }
        self.conversation_manager.conversations = {
            key: self._deserialize_conversation_context(value)
            for key, value in snapshot.get("conversations", {}).items()
        }
        self.active_threads = {
            key: self._deserialize_datetime(value)
            for key, value in snapshot.get("active_threads", {}).items()
        }
        self.posted_ci_jobs = dict.fromkeys(
            str(run_id) for run_id in snapshot.get("posted_ci_jobs", [])
        )
        for record in snapshot.get("agent_runs", {}).values():
            if not _is_complete_agent_run_record(record):
                continue
            self._register_agent_run(self._agent_run_record(record))

        expired_sessions = self.conversation_manager.cleanup_expired_sessions(
            return_keys=True
        )
        expired_threads = self._cleanup_expired_threads(return_keys=True)
        if expired_sessions or expired_threads:
            await self._delete_conversation_states(expired_sessions)
            await self._delete_active_thread_states(expired_threads)

    async def _persist_state(self):
        """Persist the current Slack state through the configured backend."""
        async with self.state_lock:
            snapshot = {
                "schema_version": STATE_SCHEMA_VERSION,
                "saved_at": datetime.now().isoformat(),
                "users": self.user_manager.users,
                "rate_limits": self.user_manager.rate_limits,
                "conversations": {
                    key: self._serialize_conversation_context(context)
                    for key, context in self.conversation_manager.conversations.items()
                },
                "active_threads": {
                    key: value.isoformat() for key, value in self.active_threads.items()
                },
                "agent_runs": {
                    run_id: self._agent_run_record(tracked)
                    for run_id, tracked in self.agent_runs.items()
                    if tracked.get("run_id") and tracked.get("session_id")
                },
                "posted_ci_jobs": list(self.posted_ci_jobs),
            }
            await self.state_store.save_state(snapshot)

    def _serialize_conversation_context(
        self, context: ConversationContext
    ) -> Dict[str, Any]:
        """Serialize a conversation context to JSON-friendly data."""
        return {
            "user_id": context.user_id,
            "channel_id": context.channel_id,
            "thread_ts": context.thread_ts,
            "conversation_history": context.conversation_history,
            "user_tier": context.user_tier.value,
            "preferences": context.preferences,
            "last_activity": context.last_activity.isoformat(),
            "session_id": context.session_id,
        }

    def _deserialize_conversation_context(
        self, payload: Dict[str, Any]
    ) -> ConversationContext:
        """Restore a serialized conversation context."""
        user_tier = payload.get("user_tier", UserTier.FREE.value)
        try:
            tier = UserTier(user_tier)
        except ValueError:
            tier = UserTier.FREE

        return ConversationContext(
            user_id=payload.get("user_id", ""),
            channel_id=payload.get("channel_id", ""),
            thread_ts=payload.get("thread_ts"),
            conversation_history=list(payload.get("conversation_history", [])),
            user_tier=tier,
            preferences=dict(payload.get("preferences", {})),
            last_activity=self._deserialize_datetime(payload.get("last_activity")),
            session_id=payload.get("session_id", str(uuid.uuid4())),
        )

    def _deserialize_datetime(self, value: Optional[str]) -> datetime:
        """Restore datetimes stored in Slack state snapshots."""
        if not value:
            return datetime.now()

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()
