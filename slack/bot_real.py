"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import logging
import os
import random
import time
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

from slack.agent_bootstrap import (
    AgentBootstrapBuilder,
)
from slack.agent_feedback import (
    AGENT_FEEDBACK_CALLBACK_ID,
    AgentFeedbackNonce,
)
from slack.agent_shortcut import (
    AgentShortcutNonce,
)
from src.memory import MemoryManager, build_memory_scope
from src.utils.schema import (
    Attachment,
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
    SLACK_BUSY_MESSAGE,
    SLACK_EMPTY_ANSWER_MESSAGE,
    SLACK_ERROR_CODE_MESSAGES,
    SLACK_GENERIC_ERROR_MESSAGE,
    SLACK_THINKING_TEXT,
    SlackBotReplyMixin,
    SlackNotice,
    slack_chat_request_id,
)
from slack.users import (
    SlackBotUsersMixin,
    UserManager,
)
from slack.state_store import (
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
    AGENT_TERMINAL_STATUSES,
    SlackBotAgentMixin,
    _agent_idempotency_key,
    _format_agent_status_line,
    _is_agent_command_text,
)
from slack.chat_handler import (
    AGENT_FOLLOWUP_FLAG_PATTERN,
    SlackMessageHandler,
)
from slack.bot_persistence import (
    SlackBotPersistenceMixin,
)

logger = logging.getLogger(__name__)
HANDLED_MESSAGES_LIMIT = 1000
# Threads whose Agent runtime lookup a message event skips (S7): chat threads
# known not to be Agent threads, and threads whose lookup just failed.
AGENT_LOOKUP_SKIPS_LIMIT = 10_000
# Channels outside a name allowlist, so conversations.info runs once per TTL.
CHANNEL_ALLOWLIST_MISSES_LIMIT = 1000
CHANNEL_ALLOWLIST_MISS_TTL_SECONDS = 300.0
# A failed conversations.info call (a 429, a network error) is retried sooner.
CHANNEL_ALLOWLIST_ERROR_TTL_SECONDS = 10.0
SLACK_PRIORITY_WORK_KINDS = frozenset({"interactive"})


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


class SlackBot(
    SlackBotPersistenceMixin,
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
