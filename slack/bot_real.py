"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import logging
import os
import random
import re
import time
import uuid
from datetime import (
    datetime,
)
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
)
from collections import deque
from dataclasses import dataclass, field, replace

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
)
from slack.agent_bootstrap import (
    AgentBootstrapBuilder,
    AgentBootstrapContext,
    AgentBootstrapFetchError,
)
from slack.agent_feedback import (
    AGENT_FEEDBACK_CALLBACK_ID,
    AgentFeedbackNonce,
)
from slack.agent_shortcut import (
    AgentShortcutNonce,
)
from src.admission import AdmissionRejectedError
from src.memory import MemoryManager, build_memory_scope
from src.prompt_assembly import protect_untrusted_boundary
from src.utils.schema import (
    Attachment,
    QueryRequest,
    ToolPolicy,
    UserTier,
    is_valid_request_id,
)
from src.utils.bounded_state import BoundedTTLMap
from src.utils.metrics import SLACK_METRICS
from slack.formatting import (
    CHAT_FEEDBACK_ACTIONS,
    CHAT_FEEDBACK_CLICKS_LIMIT,
    CHAT_FEEDBACK_NOT_RECORDED_TEXT,
    CHAT_FEEDBACK_THANKS_TEXT,
    MEMORY_SLASH_ONLY_MESSAGE,
    _RATE_LIMITED_TEXT,
    SLACK_BUSY_MESSAGE,
    SLACK_EMPTY_ANSWER_MESSAGE,
    SLACK_ERROR_CODE_MESSAGES,
    SLACK_GENERIC_ERROR_MESSAGE,
    SLACK_INFERENCE_ERROR_MESSAGES,
    SLACK_THINKING_TEXT,
    SlackAnswer,
    SlackBotReplyMixin,
    SlackNotice,
    _admission_rejection_text,
    _busy_text,
    _normalized_text,
    _response_warning_label,
    _response_warning_notes,
    _retry_wait,
    slack_chat_request_id,
    _slack_text_as_posted,
)
from slack.users import (
    SlackBotUsersMixin,
    UserManager,
    _rate_limit_entries,
    _rate_limit_entry,
)
from slack.state_store import (
    STATE_SCHEMA_VERSION,
    FileSlackStateStore,
    RedisSlackStateStore,
    build_slack_state_store,
    resolve_slack_state_store_config,
)
from slack.conversation import (
    CHAT_MESSAGE_SUBTYPES,
    SLACK_QUEUE_WAIT_KEY,
    ConversationContext,
    ConversationManager,
    SlackBotConversationMixin,
    SlackFetchedContext,
    SlackQueryTrace,
    _queue_wait_ms,
)
from slack.attachments import (
    SlackBotAttachmentsMixin,
)
from slack.bot_agent import (
    AGENT_ERROR_HINTS,
    AGENT_START_ERROR_TEXT,
    AGENT_TERMINAL_STATUSES,
    UNVERIFIED_LABEL,
    AgentFollowupFlags,
    SlackBotAgentMixin,
    _agent_auth_error_text,
    _agent_idempotency_key,
    _format_agent_status_line,
    _format_grader_finding,
    _is_agent_command_text,
    _is_complete_agent_run_record,
    _show_agent_status,
)

logger = logging.getLogger(__name__)

MAX_QUERY_TOKENS = 8192
HANDLED_MESSAGES_LIMIT = 1000
# Threads whose Agent runtime lookup a message event skips (S7): chat threads
# known not to be Agent threads, and threads whose lookup just failed.
AGENT_LOOKUP_SKIPS_LIMIT = 10_000
# Channels outside a name allowlist, so conversations.info runs once per TTL.
CHANNEL_ALLOWLIST_MISSES_LIMIT = 1000
CHANNEL_ALLOWLIST_MISS_TTL_SECONDS = 300.0
# A failed conversations.info call (a 429, a network error) is retried sooner.
CHANNEL_ALLOWLIST_ERROR_TTL_SECONDS = 10.0
# Commands that answer through the model, though they spend no query quota.
MODEL_BACKED_COMMANDS = frozenset({"web", "fast"})
# How much of the last answer must appear in the transcript to count as there.
LAST_ANSWER_PROBE_CHARS = 200
SLACK_PRIORITY_WORK_KINDS = frozenset({"interactive"})
AGENT_STARTING_MESSAGE = (
    "🛠️ Starting the Pi coding agent; it will open a thread in this channel "
    "when its worktree is ready."
)


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


AGENT_FOLLOWUP_FLAG_PATTERN = re.compile(
    r"\s*(--discuss|--implement|--workflow)(?:\s+|$)"
)


SLACK_RATE_LIMIT_RETRIES = 2


def _add_rate_limit_retry(client: Any) -> None:
    """Retry Slack 429s (honouring Retry-After) instead of failing the post."""
    retry_handlers = getattr(client, "retry_handlers", None)
    if isinstance(retry_handlers, list):
        retry_handlers.append(
            AsyncRateLimitErrorRetryHandler(max_retry_count=SLACK_RATE_LIMIT_RETRIES)
        )


@dataclass(frozen=True)
class SlackWorkItem:
    kind: str
    payload: Dict[str, Any]
    name: str
    # time.monotonic() when the item was queued; None if it never was.
    enqueued_at: Optional[float] = None


def _stamp_queue_wait(item: SlackWorkItem) -> None:
    """Put the item's wait so far (queued and parked) on its payload, in ms.

    Only the bot's own stamp counts: a value already there is dropped.
    """
    payload = item.payload
    if not isinstance(payload, dict):
        return
    payload.pop(SLACK_QUEUE_WAIT_KEY, None)
    if item.enqueued_at is None:
        return
    waited = time.monotonic() - item.enqueued_at
    payload[SLACK_QUEUE_WAIT_KEY] = max(0, int(waited * 1000))


@dataclass
class SlackThreadWork:
    """Chat events parked behind the one a worker is answering in a thread."""

    pending: Deque[SlackWorkItem] = field(default_factory=deque)


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
        trace = SlackQueryTrace.for_message(event, team_id=team_id)

        # The bot decides chat or Agent once per message, before it shows
        # progress or downloads files; a thread that became an Agent thread
        # since then does not change this message's lane.
        decided_agent = event.get("_agent_request")
        if not isinstance(decided_agent, bool):
            decided_agent = self.bot._is_agent_thread(
                channel_id, event.get("thread_ts")
            )
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
                trace=trace,
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
            builder = self.bot.agent_bootstrap_builder
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
            trace=trace,
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
        trace: Optional["SlackQueryTrace"] = None,
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
            trace=trace,
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
        trace: Optional["SlackQueryTrace"] = None,
        queue_wait_ms: Optional[int] = None,
    ) -> str:
        """Handle bot commands, falling back to a free-form query when appropriate."""
        parts = command_text.split()
        if not parts:
            return await self._handle_help_command([], user_id, channel_id, client)

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        if command_surface == "slash":
            # Slack resends a slash command with the same trigger_id.
            trace = SlackQueryTrace.for_slash(request_id, queue_wait_ms=queue_wait_ms)

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
                    trace=trace,
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
                    trace=trace,
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
            trace=trace,
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
        trace: Optional["SlackQueryTrace"] = None,
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
            trace=trace,
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
        trace: Optional["SlackQueryTrace"] = None,
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
            trace=trace,
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

        runtime = self.bot.agent_runtime_client
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
        builder = self.bot.agent_bootstrap_builder
        if include_thread_context and thread_ts and message_ts and builder is not None:
            if not await self.bot._is_public_agent_channel(channel_id):
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
        # Check and reserve with no await in between, so concurrent starts
        # by one owner cannot both pass the cap.
        refusal = self.bot._agent_cap_message(user_id, channel_id, thread_ts)
        if refusal:
            return refusal
        reservation = self.bot._reserve_agent_start(user_id, channel_id, thread_ts)
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
        runtime = self.bot.agent_runtime_client
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
            previous = (
                self.bot.agent_threads.get(self.bot._thread_key(channel_id, thread_ts))
                if thread_ts
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
                self.bot._format_tier_limit_summary(),
            ]
        )
        return "\n".join(lines)

    async def _agent_models_text(self) -> str:
        """Name the Agent models a fresh runtime health check reports.

        A cached allowlist may be stale, so without an answer say so instead.
        """
        runtime = self.bot.agent_runtime_client
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
                        getattr(self.bot.router, "models", None) or {}
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
            await self.bot._persist_user_state(user_id)
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
            runtime = self.bot.agent_runtime_client
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

            if self.bot.agent_runtime_client is not None:
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

        await self.bot._delete_conversation_states(cleared_keys)
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

        memory_manager = self.bot.memory_manager
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

        memory_manager = self.bot.memory_manager
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

        memory_manager = self.bot.memory_manager
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
        trace: Optional["SlackQueryTrace"] = None,
    ) -> str:
        """Handle regular query through inference engine"""
        trace = trace or SlackQueryTrace()
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

            # Build request-scoped Slack and bot conversation context. The
            # Slack history and the memory search are independent, so they
            # run concurrently and each fails open on its own.
            slack_context, memory_results = await asyncio.gather(
                self._fetch_query_slack_context(
                    client=client,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    message_ts=message_ts,
                ),
                self._search_query_memory(
                    text, user_id=user_id, channel_id=channel_id, team_id=team_id
                ),
            )
            conversation_context = self._conversation_fallback(
                slack_context,
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )

            request_context = self._join_context_sections(
                [slack_context.context, conversation_context]
            )
            # Memory hits change with every query, so they ride in metadata
            # and inference places them after the history, not before it.
            memory_context = ""
            if memory_results:
                try:
                    memory_context = self.bot.memory_manager.build_context(
                        memory_results, ""
                    )
                except Exception:
                    logger.exception("Failed to build Slack memory context")

            # Create query request
            conversation_id = self.bot._conversation_context_key(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )

            query_metadata = self.bot._build_query_metadata(
                channel_id=channel_id,
                thread_ts=thread_ts,
                attachments=attachments or [],
                user_preferences=user_prefs,
                team_id=team_id,
            )
            query_metadata.update(dict(routing_metadata or {}))
            self._add_slack_context_metadata(query_metadata, slack_context)
            if memory_context:
                query_metadata["memory_context"] = memory_context
            # Server-written trace keys: an external /route caller cannot set
            # them (they are not client metadata keys).
            trace.add_metadata(query_metadata, message_ts=message_ts)

            query_request = QueryRequest(
                **({"request_id": trace.request_id} if trace.request_id else {}),
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

            # Process through the ingress /route shares (requests.raw, API
            # metrics, error mapping); the per-user limiter already admitted it.
            ingress = (self.bot.services or {}).get("chat_ingress")
            if ingress is not None and callable(getattr(ingress, "handle", None)):
                response = await ingress.handle(query_request, source="slack")
            else:
                response = await self.bot.inference_engine.process_query(query_request)

            if getattr(response, "error", None):
                logger.error(
                    "Slack query %s failed for user %s in channel %s: %s",
                    query_request.request_id,
                    user_id,
                    channel_id,
                    response.error,
                )
                SLACK_METRICS.errors.labels(error_type="inference_error_response").inc()
                return self._inference_error_message(
                    getattr(response, "error_type", None),
                    getattr(response, "error_code", None),
                    retry_after_seconds=getattr(response, "retry_after_seconds", None),
                    request_id=query_request.request_id,
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

            return self._format_response_with_sources(
                response, answer, request_id=query_request.request_id
            )

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
            return self._inference_error_message(None, request_id=trace.request_id)

    def _inference_error_message(
        self,
        error_type: Optional[str],
        error_code: Optional[str] = None,
        *,
        retry_after_seconds: Any = None,
        request_id: Optional[str] = None,
    ) -> str:
        """What a failed answer tells the user, with the request's reference."""
        return SlackNotice(
            self._inference_error_text(
                error_type, error_code, retry_after_seconds=retry_after_seconds
            ),
            request_id=request_id,
        )

    def _inference_error_text(
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
        return self.bot._build_user_safe_error_message()

    def _conversation_fallback(
        self,
        slack_context: SlackFetchedContext,
        *,
        user_id: str,
        channel_id: str,
        thread_ts: Optional[str],
    ) -> str:
        """What the per-user history adds to the Slack transcript.

        Without a thread transcript, or when the bot cannot recognise its own
        messages (no bot user id), the whole summary. Otherwise only the last
        answer, and only when the transcript misses its start, e.g. an answer
        posted after the follow-up (queued work) is past `latest`. Only the
        start is compared, so a split answer whose first part is in the
        transcript counts as present.
        """
        conversations = self.bot.conversation_manager
        if (
            slack_context.source != "thread"
            or slack_context.error
            or not self.bot.bot_user_id
        ):
            return conversations.get_conversation_summary(
                user_id, channel_id, thread_ts
            )
        last_answer = conversations.get_last_answer(user_id, channel_id, thread_ts)
        if not last_answer:
            return ""
        # History leads a degraded answer with a "[...]" label Slack lacks,
        # and Slack returns the text escaped with its links wrapped.
        probe = _normalized_text(re.sub(r"^\[[^\]]*\]\s*", "", last_answer))
        probe = probe[:LAST_ANSWER_PROBE_CHARS]
        if probe and probe in _slack_text_as_posted(slack_context.context):
            return ""
        return protect_untrusted_boundary(f"assistant: {last_answer}")

    async def _fetch_query_slack_context(
        self,
        *,
        client: AsyncWebClient,
        channel_id: str,
        thread_ts: Optional[str],
        message_ts: Optional[str],
    ) -> SlackFetchedContext:
        """Slack history for a query; the builder fails open as configured."""
        return await self.bot._build_slack_context(
            client=client,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message_ts=message_ts,
        )

    async def _search_query_memory(
        self,
        text: str,
        *,
        user_id: str,
        channel_id: str,
        team_id: Optional[str],
    ) -> List[Any]:
        """Long-term memory hits for a query; none when off or on failure."""
        memory_manager = self.bot.memory_manager
        if not memory_manager or not getattr(memory_manager, "enabled", False):
            return []
        try:
            scope = self.bot._memory_scope(team_id, user_id)
            return list(
                await memory_manager.search(
                    scope,
                    text,
                    metadata=self._memory_visibility_metadata(
                        channel_id=channel_id,
                        include_all=False,
                        global_only=False,
                    ),
                )
                or []
            )
        except Exception:
            logger.exception("Failed to search Slack memory")
            return []

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
        self,
        response: Any,
        response_text: Optional[str] = None,
        *,
        request_id: Optional[str] = None,
    ) -> str:
        """Append compact source links for web-search enriched answers.

        `response_text` replaces the response's own text, e.g. with notes added.
        `request_id` is the request answered, for feedback on the answer.
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
            request_id=request_id or getattr(response, "request_id", None),
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


class SlackBot(
    SlackBotAgentMixin,
    SlackBotAttachmentsMixin,
    SlackBotConversationMixin,
    SlackBotUsersMixin,
    SlackBotReplyMixin,
):
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
        # Channel IDs conversations.info showed are outside the allowlist.
        self._channel_allowlist_misses: BoundedTTLMap[str, bool] = BoundedTTLMap(
            max_entries=CHANNEL_ALLOWLIST_MISSES_LIMIT,
            ttl_seconds=CHANNEL_ALLOWLIST_MISS_TTL_SECONDS,
        )
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
        # "channel:thread_ts" -> "chat" or "error": a message event in such a
        # thread does not ask the runtime whether it is an Agent thread. Local
        # only, so after a restart each thread is looked up once again.
        self._agent_lookup_skips: BoundedTTLMap[str, str] = BoundedTTLMap(
            max_entries=AGENT_LOOKUP_SKIPS_LIMIT,
            ttl_seconds=self.active_thread_timeout.total_seconds(),
        )
        self.agent_runs: Dict[str, Dict[str, Any]] = {}
        # Each owner's tracked sessions, one per thread, the latest last.
        self.agent_user_runs: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_monitoring: Set[str] = set()
        # Ordered so the oldest claims are evicted first.
        self.posted_ci_jobs: Dict[str, None] = {}
        # "channel:ts" of messages already handled, oldest first. Slack sends
        # both a message and an app_mention event for one mention.
        self.handled_messages: Dict[str, None] = {}
        # Feedback clicks already recorded, by action_ts (bounded).
        self._chat_feedback_clicks: Dict[str, None] = {}
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
        # Threads a worker is answering, with the chat events parked behind
        # the current one (S12). An entry lives only while its worker runs,
        # so the map holds at most one entry per worker.
        self._busy_threads: Dict[str, SlackThreadWork] = {}
        # Parked events across all threads; with the work queue's items they
        # stay within its capacity.
        self._thread_pending = 0
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
                # The envelope's event id, kept for tracing: a retry of the
                # same event carries the same id.
                event_id = req.payload.get("event_id")
                if isinstance(event_id, str) and event_id and "_event_id" not in event:
                    event["_event_id"] = event_id
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
                actions = req.payload.get("actions") or [{}]
                chat_feedback = actions[0].get("action_id") in CHAT_FEEDBACK_ACTIONS
                await self._enqueue_work(
                    SlackWorkItem(
                        kind="interactive",
                        payload=req.payload,
                        name=(
                            "slack_chat_feedback"
                            if chat_feedback
                            else "slack_agent_approval"
                        ),
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
            await self._look_up_agent_thread(event)
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

    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Handle app mentions"""
        channel_id = event.get("channel")
        if not await self._is_channel_allowed(channel_id, event.get("channel_name")):
            return
        if await self._reject_non_owner_agent_message(event):
            return
        if not self._claim_message(event):
            return

        if not event.get("thread_ts") and not self._is_agent_request(event):
            # The answer starts a new thread at this message; an Agent start
            # here registers its thread locally, which is checked first.
            self._remember_chat_thread(channel_id, event.get("ts"))
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
        queue_wait_ms = _queue_wait_ms(command.get(SLACK_QUEUE_WAIT_KEY))
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
            **({"queue_wait_ms": queue_wait_ms} if queue_wait_ms is not None else {}),
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
                # No reply text. A placeholder is only posted on the chat lane
                # (the lane handle_message is held to), so there it means no
                # answer, never an Agent handoff.
                if placeholder_ts and not await self._replace_placeholder(
                    channel_id, placeholder_ts, text=SLACK_EMPTY_ANSWER_MESSAGE
                ):
                    await self._post_reply(
                        channel_id, reply_thread_ts, text=SLACK_EMPTY_ANSWER_MESSAGE
                    )
                if self._is_agent_thread(channel_id, agent_thread_ts):
                    # The handler posted the reply itself: an Agent run's
                    # status message.
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
            error_text = SlackNotice(
                self._build_user_safe_error_message(),
                request_id=self._event_request_id(event),
            )
            error_message = self._answer_message(error_text, error_text, last=True)
            if placeholder_ts and await self._replace_placeholder(
                event.get("channel"), placeholder_ts, **error_message
            ):
                return
            await self._post_reply(
                event.get("channel"),
                event.get("thread_ts") or event.get("ts"),
                **error_message,
            )

    def _event_request_id(self, event: Dict[str, Any]) -> Optional[str]:
        """The chat request id of a message, as its trace computed it."""
        try:
            return slack_chat_request_id(
                self._extract_team_id(event), event.get("channel"), event.get("ts")
            )
        except Exception:
            return None

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
        self._channel_allowlist_misses.clear()

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
            and channel_id not in self._channel_allowlist_misses
            and self.web_client is not None
            and hasattr(self.web_client, "conversations_info")
        ):
            try:
                response = await self.web_client.conversations_info(channel=channel_id)
                resolved_name = self._normalize_channel_name(
                    response.get("channel", {}).get("name")
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to resolve Slack channel info for allowlist check: {exc}"
                )
                self._channel_allowlist_misses.set(
                    channel_id, True, ttl_seconds=CHANNEL_ALLOWLIST_ERROR_TTL_SECONDS
                )
                return False
            if resolved_name and resolved_name in self.allowed_channel_names:
                self.allowed_channel_ids.add(channel_id)
                return True
            # Outside the allowlist: ask Slack again only after the TTL.
            self._channel_allowlist_misses[channel_id] = True

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

    async def _handle_chat_feedback(
        self, payload: Dict[str, Any], action: Dict[str, Any]
    ) -> None:
        """Record a thumbs up / down on a chat answer, once per click.

        Anyone who sees the answer may rate it; the event names who clicked.
        """
        rating = CHAT_FEEDBACK_ACTIONS[action["action_id"]]
        request_id = action.get("value")
        user_id = (payload.get("user") or {}).get("id")
        channel_id = (payload.get("channel") or {}).get("id")
        click_id = str(action.get("action_ts") or payload.get("trigger_id") or "")
        if not is_valid_request_id(request_id) or not user_id or not click_id:
            return
        # Slack may deliver one click twice; a replay is not a second vote.
        # The click is claimed while it is being recorded and kept only once
        # it was, so a click that failed can be recorded when retried.
        clicks = self._chat_feedback_clicks
        if click_id in clicks:
            return
        clicks[click_id] = None

        recorded = False
        producer = (getattr(self, "services", None) or {}).get("event_producer")
        produce = getattr(producer, "produce_chat_feedback", None)
        if not callable(produce):
            logger.warning(
                "No event producer: chat feedback for request %s not recorded",
                request_id,
            )
        else:
            # A producer that can confirm the send is asked to: the user is
            # thanked, and the click kept, only once Kafka has the event.
            confirm = getattr(producer, "confirms_chat_feedback_delivery", False)
            try:
                recorded = bool(
                    await produce(
                        request_id=request_id,
                        user_id=str(user_id),
                        rating=rating,
                        click_id=click_id,
                        **({"wait_for_delivery": True} if confirm is True else {}),
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to record chat feedback for request %s", request_id
                )
            else:
                if not recorded:
                    logger.warning(
                        "Chat feedback for request %s was not delivered",
                        request_id,
                    )
        if not recorded:
            clicks.pop(click_id, None)
        while len(clicks) > CHAT_FEEDBACK_CLICKS_LIMIT:
            clicks.pop(next(iter(clicks)))
        if channel_id and hasattr(self.web_client, "chat_postEphemeral"):
            container = payload.get("container") or {}
            message = payload.get("message") or {}
            thread_ts = container.get("thread_ts") or message.get("thread_ts")
            await self.web_client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=(
                    CHAT_FEEDBACK_THANKS_TEXT
                    if recorded
                    else CHAT_FEEDBACK_NOT_RECORDED_TEXT
                ),
                **({"thread_ts": thread_ts} if thread_ts else {}),
            )

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
        item = replace(item, enqueued_at=time.monotonic())
        try:
            if queue is self._work_queue and self._work_queue_full():
                raise asyncio.QueueFull
            queue.put_nowait(item)
        except asyncio.QueueFull:
            SLACK_METRICS.work_rejected.labels(reason="capacity").inc()
            await self._reply_busy(item)
            return False
        SLACK_METRICS.work_queue_depth.set(self._queued_work())
        return True

    def _work_queue_full(self) -> bool:
        """Queued and parked chat work share the work queue's capacity."""
        return (
            self._work_queue.qsize() + self._thread_pending >= self._work_queue.maxsize
        )

    def _queued_work(self) -> int:
        return (
            self._work_queue.qsize()
            + self._priority_queue.qsize()
            + self._thread_pending
        )

    async def _work_worker(self, index: int, queue: asyncio.Queue) -> None:
        while True:
            item = await queue.get()
            self._work_running += 1
            SLACK_METRICS.work_queue_depth.set(self._queued_work())
            SLACK_METRICS.work_running.set(self._work_running)
            try:
                if item.kind == "slash":
                    _stamp_queue_wait(item)
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
                    await self._handle_chat_work(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Slack work item failed: %s", item.name)
            finally:
                self._work_running -= 1
                SLACK_METRICS.work_running.set(self._work_running)
                queue.task_done()

    def _chat_thread_key(self, event: Dict[str, Any]) -> Optional[str]:
        """The thread a chat event belongs to: its root's channel:ts."""
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        if not channel_id or not thread_ts:
            return None
        return self._thread_key(channel_id, thread_ts)

    async def _drops_before_thread(self, event: Dict[str, Any]) -> bool:
        """Cheap checks _handle_*_event would drop the event on anyway.

        That is an ignored message, a message already claimed (Slack's
        message and app_mention for one mention) or a channel the bot is not
        enabled in. Such an event need not wait behind a busy thread.
        """
        if event.get("type") == "message" and self._is_ignored_message_event(event):
            return True
        channel_id = event.get("channel")
        ts = event.get("ts")
        if (
            channel_id
            and ts
            and self._thread_key(channel_id, ts) in (self.handled_messages)
        ):
            return True
        return not await self._is_channel_allowed(channel_id, event.get("channel_name"))

    async def _handle_chat_work(self, item: SlackWorkItem) -> None:
        """Answer one Slack thread one chat event at a time (S12).

        A follow-up then sees the answer before it in the thread's history and
        is posted after it. An event for a thread already being answered is
        parked on that thread and the worker goes back to the queue; the
        worker answering the thread takes the parked events in order before
        it lets the thread go, so one thread never holds more than one worker.
        """
        key = self._chat_thread_key(item.payload)
        if key is None:
            _stamp_queue_wait(item)
            await self._handle_event(item.payload)
            return
        if key in self._busy_threads:
            if await self._drops_before_thread(item.payload):
                return
        # Read again after the await: the owner may have let the thread go
        # meanwhile, and an event parked on its entry would never be answered.
        busy = self._busy_threads.get(key)
        if busy is not None:
            if self._work_queue_full():
                SLACK_METRICS.work_rejected.labels(reason="capacity").inc()
                await self._reply_busy(item)
                return
            busy.pending.append(item)
            self._thread_pending += 1
            SLACK_METRICS.work_queue_depth.set(self._queued_work())
            return
        busy = self._busy_threads[key] = SlackThreadWork()
        try:
            current = item
            while True:
                # Time parked behind the thread's earlier events counts.
                _stamp_queue_wait(current)
                try:
                    await self._handle_event(current.payload)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Slack work item failed: %s", current.name)
                if not busy.pending:
                    break
                current = busy.pending.popleft()
                self._thread_pending -= 1
                SLACK_METRICS.work_queue_depth.set(self._queued_work())
        finally:
            del self._busy_threads[key]
            if busy.pending:
                # Cancelled at shutdown with follow-ups still parked.
                self._thread_pending -= len(busy.pending)
                SLACK_METRICS.work_rejected.labels(reason="shutdown").inc(
                    len(busy.pending)
                )
                busy.pending.clear()

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
        return SLACK_GENERIC_ERROR_MESSAGE

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
