"""SlackMessageHandler: commands, chat queries and Agent starts."""

import asyncio
import logging
import re
from typing import (
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
)
from slack_sdk.web.async_client import AsyncWebClient

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
    AgentBootstrapContext,
    AgentBootstrapFetchError,
)
from src.admission import AdmissionRejectedError
from src.prompt_assembly import protect_untrusted_boundary
from src.utils.schema import (
    Attachment,
    QueryRequest,
    ToolPolicy,
    UserTier,
)
from src.utils.metrics import SLACK_METRICS
from slack.formatting import (
    MEMORY_SLASH_ONLY_MESSAGE,
    _RATE_LIMITED_TEXT,
    SLACK_EMPTY_ANSWER_MESSAGE,
    SLACK_ERROR_CODE_MESSAGES,
    SLACK_INFERENCE_ERROR_MESSAGES,
    SlackAnswer,
    SlackNotice,
    _admission_rejection_text,
    _busy_text,
    _normalized_text,
    _response_warning_label,
    _response_warning_notes,
    _retry_wait,
    _slack_text_as_posted,
)
from slack.conversation import (
    SlackFetchedContext,
    SlackQueryTrace,
)
from slack.bot_agent import (
    AGENT_START_ERROR_TEXT,
    UNVERIFIED_LABEL,
    AgentFollowupFlags,
    _agent_auth_error_text,
    _agent_idempotency_key,
    _format_agent_status_line,
    _format_grader_finding,
    _show_agent_status,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


MAX_QUERY_TOKENS = 8192
# Commands that answer through the model, though they spend no query quota.
MODEL_BACKED_COMMANDS = frozenset({"web", "fast"})
# How much of the last answer must appear in the transcript to count as there.
LAST_ANSWER_PROBE_CHARS = 200
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
            # The reference the user sees is in the log line too.
            logger.exception("Error processing Slack query %s", trace.request_id)
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
