"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.async_client import AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
import httpx

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
MEMORY_SLASH_ONLY_MESSAGE = (
    "Memory commands are only available through Slack slash commands. "
    "Use `/llm remember`, `/llm memories`, or `/llm forget`."
)


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
            tmp_path.write_text(
                json.dumps(snapshot, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self.file_path)
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
        self.client = None

    async def initialize(self):
        try:
            import redis.asyncio as redis_asyncio

            redis_url = self.redis_config.get("url")
            if redis_url and hasattr(redis_asyncio.Redis, "from_url"):
                self.client = redis_asyncio.Redis.from_url(redis_url)
            else:
                self.client = redis_asyncio.Redis(
                    host=self.redis_config.get("host", "localhost"),
                    port=self.redis_config.get("port", 6379),
                    db=self.redis_config.get("db", 0),
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
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "users": users,
                "rate_limits": rate_limits,
                "conversations": conversations,
                "active_threads": active_threads,
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

    async def shutdown(self):
        if self.client and hasattr(self.client, "close"):
            await self.client.close()

    async def persist_user_state(
        self,
        user_id: str,
        user_payload: Dict[str, Any],
        rate_limit_payload: List[float],
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
                await self._set_json(self._rate_limit_key(user_id), merged_rate_limits)
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

            await self._set_json(self._conversation_key(context_key), context_payload)
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

            await self._set_json(self._active_thread_key(thread_key), incoming_payload)
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

    def _user_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:user:{user_id}"

    def _rate_limit_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:rate_limit:{user_id}"

    def _conversation_key(self, context_key: str) -> str:
        return f"{self.key_prefix}:conversation:{context_key}"

    def _active_thread_key(self, thread_key: str) -> str:
        return f"{self.key_prefix}:active_thread:{thread_key}"

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

    async def _set_json(self, key: str, payload: Any):
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        if hasattr(self.client, "set"):
            await self.client.set(key, serialized)
        elif hasattr(self.client, "setex"):
            await self.client.setex(key, 30 * 24 * 3600, serialized)

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
        self, user_id: str, incoming_payload: List[float]
    ) -> List[float]:
        current_payload = await self._get_json(self._rate_limit_key(user_id))
        merged_values = []

        if isinstance(current_payload, list):
            merged_values.extend(float(value) for value in current_payload)
        if incoming_payload:
            merged_values.extend(float(value) for value in incoming_payload)

        if not merged_values:
            return []

        cutoff = time.time() - 3600
        deduped = sorted({value for value in merged_values if value > cutoff})
        return deduped

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
        """Get user tier from user data"""
        user_data = self.users.get(user_id, {})
        tier_name = (
            user_data.get("tier")
            or self.tier_overrides.get(user_id)
            or self.default_tier
        )

        try:
            return UserTier(tier_name)
        except ValueError:
            return UserTier.FREE

    def check_rate_limit(self, user_id: str, config: Dict[str, Any]) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        hour_window = current_time - 3600  # 1 hour window

        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Clean old requests
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id] if req_time > hour_window
        ]

        # Check limits
        requests_per_hour = config.get("requests_per_hour", 100)
        burst_requests = config.get("burst_requests", 5)

        # Check hourly limit
        if len(self.rate_limits[user_id]) >= requests_per_hour:
            return False

        # Check burst limit (last 5 minutes)
        burst_window = current_time - 300  # 5 minutes
        recent_requests = [
            req_time
            for req_time in self.rate_limits[user_id]
            if req_time > burst_window
        ]

        if len(recent_requests) >= burst_requests:
            return False

        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True

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
        """Apply configured tier defaults/overrides to the in-memory user record."""
        if not user_id:
            return False

        previous_tier = self.users.get(user_id, {}).get("tier")
        resolved_tier = self.tier_overrides.get(user_id)
        if (
            resolved_tier is None
            and previous_tier is None
            and self.default_tier != UserTier.FREE.value
        ):
            resolved_tier = self.default_tier

        if resolved_tier is None:
            return False

        if user_id not in self.users:
            self.users[user_id] = {}

        if previous_tier == resolved_tier:
            return False

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
        self.session_timeout = timedelta(hours=2)  # Sessions expire after 2 hours

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

    async def handle_message(
        self, event: Dict[str, Any], client: AsyncWebClient
    ) -> Optional[str]:
        """Handle incoming message"""
        text = event.get("text", "").strip()
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        team_id = self.bot._extract_team_id(event)
        attachments = list(event.get("_query_attachments", []) or [])

        # Check if it's a command
        if not text and not attachments:
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
            )

        if not text and attachments:
            text = self._build_attachment_only_query(attachments)

        # Regular query - process through inference engine
        return await self._handle_query(
            text,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
        )

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
            return await self.commands[command](args, user_id, channel_id, client)
        return await self._handle_query(
            command_text,
            user_id,
            channel_id,
            thread_ts,
            client,
            attachments=attachments,
            team_id=team_id,
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
        )

    async def _handle_help_command(
        self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient
    ) -> str:
        """Handle help command"""
        help_text = """
🤖 *LLM Router Bot Help*

*Basic Usage:*
Mention me in a channel, use `/llm ...`, or reply inside an active bot thread.

*Commands:*
• `/llm help` - Show this help message
• `/llm settings` - View/update your preferences  
• `/llm status` - Show system status and your usage
• `/llm models` - List available models and their capabilities
• `/llm analytics` - Show usage analytics (premium users)
• `/llm clear` - Clear conversation history
• `/llm web <query>` - Search the web before answering
• `/llm remember <text>` - Save an explicit long-term memory
• `/llm memories [query]` - List or search your memories
• `/llm forget <memory_id|all>` - Delete saved memories

*Examples:*
• "Write a Python function to calculate fibonacci numbers"
• "Explain quantum computing in simple terms"
• "Analyze this CSV data: [attach file]"
• "Help me plan a project roadmap"

*Features:*
✅ Intelligent model routing
✅ Conversation continuity  
✅ Context compression for long conversations
✅ Response caching for faster replies
✅ Usage analytics and cost tracking

*User Tiers:*
{self.bot._format_tier_limit_summary()}
        """
        return help_text.strip()

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

            status_text = f"""
📊 *System Status*

*Health:* {'🟢 Healthy' if system_health.get('healthy', False) else '🔴 Issues Detected'}
*Active Models:* {len(system_health.get('available_models', []))}
*Response Time:* {system_health.get('avg_response_time', 0):.0f}ms
*Uptime:* {system_health.get('uptime', 'Unknown')}

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
                results = await memory_manager.search(
                    scope, query, metadata=metadata
                )
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

            # Build conversation context
            conversation_context = (
                self.bot.conversation_manager.get_conversation_summary(
                    user_id, channel_id, thread_ts
                )
            )
            request_context = conversation_context
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
                        conversation_context,
                    )
                except Exception:
                    logger.exception("Failed to search Slack memory")
                    request_context = conversation_context

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

            query_request = QueryRequest(
                query=text,
                user_id=user_id,
                user_tier=user_tier,
                context=request_context,
                max_tokens=self._get_max_tokens_for_user(user_tier, user_prefs),
                temperature=0.7,
                priority=1 if user_tier != UserTier.FREE else 3,
                attachments=attachments or [],
                tool_policy=tool_policy,
                allowed_tools=allowed_tools or [],
                session_id=context.session_id,
                conversation_id=conversation_id,
                metadata=(
                    self.bot._build_query_metadata(
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        attachments=attachments or [],
                        user_preferences=user_prefs,
                        team_id=team_id,
                    )
                    if hasattr(self.bot, "_build_query_metadata")
                    else {}
                ),
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
                error_builder = getattr(
                    self.bot, "_build_user_safe_error_message", None
                )
                if callable(error_builder):
                    return error_builder()
                return "❌ Sorry, I couldn't process that request right now. Please try again in a moment."

            # Add to conversation history
            context.add_message("user", text)
            context.add_message("assistant", response.response_text)

            # Update metrics
            SLACK_METRICS.messages_processed.labels(user_tier=user_tier.value).inc()
            SLACK_METRICS.response_time.observe(response.latency_ms / 1000)

            return self._format_response_with_sources(response)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            SLACK_METRICS.errors.labels(error_type=type(e).__name__).inc()
            error_builder = getattr(self.bot, "_build_user_safe_error_message", None)
            if callable(error_builder):
                return error_builder()
            return "❌ Sorry, I couldn't process that request right now. Please try again in a moment."

    def _format_response_with_sources(self, response: Any) -> str:
        """Append compact source links for web-search enriched answers."""
        response_text = getattr(response, "response_text", "") or ""
        sources = list(getattr(response, "sources", []) or [])
        if not sources:
            return response_text

        lines = [response_text.rstrip(), "", "Sources:"]
        for source in sources[:5]:
            title = getattr(source, "title", "") or getattr(source, "url", "")
            url = getattr(source, "url", "")
            rank = getattr(source, "rank", len(lines))
            if not url:
                continue
            lines.append(f"[{rank}] {title} - {url}")
        return "\n".join(lines).strip()

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
        attachment_names = ", ".join(attachment.name for attachment in attachments[:3])
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
    ):
        self.config = config
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
        self.attachment_settings = {
            "enabled": True,
            "max_files": 10,
            "max_file_size_bytes": 10_000_000,
            "download_timeout_seconds": 30,
        }
        self.attachment_settings.update(config.get("attachments", {}))
        self.state_store = build_slack_state_store(
            resolve_slack_state_store_config(config)
        )
        self.state_lock = asyncio.Lock()
        self.active_threads: Dict[str, datetime] = {}
        self.active_thread_timeout = self.conversation_manager.session_timeout

        # Running state
        self.initialized = False
        self.running = False
        self.started_at: Optional[datetime] = None
        self.background_tasks: Set[asyncio.Task] = set()

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
            self.bot_token = bot_token
            self.socket_client = AsyncSocketModeClient(
                app_token=app_token, web_client=self.web_client
            )

            # Get bot user ID
            auth_response = await self.web_client.auth_test()
            self.bot_user_id = auth_response["user_id"]

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

            # Start cleanup task
            self._spawn_background_task(
                self._cleanup_sessions_periodically(), "slack_state_cleanup"
            )

            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Slack bot error: {e}")
        finally:
            await self.socket_client.disconnect()

    async def _handle_socket_mode_request(
        self, client: AsyncSocketModeClient, req: SocketModeRequest
    ):
        """Handle incoming socket mode requests"""
        try:
            response = SocketModeResponse(envelope_id=req.envelope_id)
            await client.send_socket_mode_response(response)

            if req.type == "events_api":
                event = req.payload.get("event", {})
                team_id = self._extract_team_id(req.payload)
                if team_id and "_team_id" not in event:
                    event["_team_id"] = team_id
                self._spawn_background_task(
                    self._handle_event(event),
                    f"slack_event:{event.get('type', 'unknown')}",
                )

            elif req.type == "slash_commands":
                command = req.payload
                self._spawn_background_task(
                    self._handle_slash_command(command), "slack_slash_command"
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
        # Skip bot messages
        if (
            event.get("subtype")
            or event.get("bot_id")
            or event.get("user") == self.bot_user_id
        ):
            return

        channel_id = event.get("channel")
        if not await self._is_channel_allowed(channel_id, event.get("channel_name")):
            return

        thread_ts = event.get("thread_ts")
        if not thread_ts or not self._is_active_thread(channel_id, thread_ts):
            return

        await self._process_message(event)

    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Handle app mentions"""
        channel_id = event.get("channel")
        if not await self._is_channel_allowed(channel_id, event.get("channel_name")):
            return

        # Remove mention from text
        text = event.get("text", "")
        mention_pattern = f"<@{self.bot_user_id}>"
        text = text.replace(mention_pattern, "").strip()

        # Update event with cleaned text
        event["text"] = text

        # Process as regular message
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
        if not is_supported_command:
            rate_limit_config = self._get_rate_limit_config_for_user(user_id)
            if not self.user_manager.check_rate_limit(user_id, rate_limit_config):
                await self._post_command_response(
                    channel_id,
                    user_id,
                    self._build_rate_limit_message(user_id),
                )
                if user_state_changed:
                    await self._persist_user_state(user_id)
                return

        response_text = await self.message_handler._handle_command(
            command_text,
            user_id,
            channel_id,
            None,
            self.web_client,
            team_id=team_id,
            command_surface="slash",
        )

        await self._post_command_response(channel_id, user_id, response_text)
        if not is_supported_command:
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
        try:
            channel_id = event.get("channel")
            user_id = event.get("user")
            user_state_changed = self._synchronize_user_tier(user_id)
            user_prefs = self.user_manager.get_user_preferences(user_id)

            if self._is_query_event(event):
                rate_limit_config = self._get_rate_limit_config_for_user(user_id)
                if not self.user_manager.check_rate_limit(user_id, rate_limit_config):
                    await self._send_rate_limit_message(channel_id, user_id)
                    if user_state_changed:
                        await self._persist_user_state(user_id)
                    return

            attachments = await self._extract_query_attachments(event)
            if attachments:
                event["_query_attachments"] = attachments

            # Process message through handler
            response_text = await self.message_handler.handle_message(
                event, self.web_client
            )

            if response_text:
                # Determine if we should reply in thread
                thread_ts = None
                threading_enabled = user_prefs.get("threading", True)
                if event.get("thread_ts"):
                    thread_ts = event.get("thread_ts")
                elif (
                    self.config.get("response_settings", {}).get("thread_replies", True)
                    and threading_enabled
                    and event.get("ts")
                ):
                    thread_ts = event.get("ts")

                # Split long responses
                max_length = self.config.get("response_settings", {}).get(
                    "max_response_length", 2000
                )
                if len(response_text) > max_length:
                    responses = self._split_response(response_text, max_length)

                    for i, response_part in enumerate(responses):
                        await self.web_client.chat_postMessage(
                            channel=channel_id,
                            text=response_part,
                            thread_ts=thread_ts,
                            reply_broadcast=(i == 0),  # Only broadcast first message
                        )
                        await asyncio.sleep(0.5)  # Small delay between parts
                else:
                    await self.web_client.chat_postMessage(
                        channel=channel_id, text=response_text, thread_ts=thread_ts
                    )

                state_changed = False
                active_thread_key = None
                if thread_ts:
                    self._mark_thread_active(channel_id, thread_ts)
                    state_changed = True
                    active_thread_key = self._thread_key(channel_id, thread_ts)

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
                        user_id=user_id
                        if (query_event or user_state_changed)
                        else None,
                        conversation_key=conversation_key,
                        active_thread_key=active_thread_key,
                    )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.web_client.chat_postMessage(
                channel=event.get("channel"),
                text=self._build_user_safe_error_message(),
            )

    def _split_response(self, text: str, max_length: int) -> List[str]:
        """Split long response into multiple parts"""
        if len(text) <= max_length:
            return [text]

        def append_chunk(chunk: str):
            chunk = chunk.strip()
            if not chunk:
                return

            if len(chunk) <= max_length:
                parts.append(chunk)
                return

            for start in range(0, len(chunk), max_length):
                parts.append(chunk[start : start + max_length])

        parts = []
        current_part = ""

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(current_part) + len(paragraph) + 2 <= max_length:
                if current_part:
                    current_part += "\n\n" + paragraph
                else:
                    current_part = paragraph
            else:
                if current_part:
                    append_chunk(current_part)
                    current_part = paragraph
                else:
                    # Paragraph is too long, split by sentences
                    sentences = paragraph.split(". ")
                    for sentence in sentences:
                        if len(current_part) + len(sentence) + 2 <= max_length:
                            if current_part:
                                current_part += ". " + sentence
                            else:
                                current_part = sentence
                        else:
                            if current_part:
                                append_chunk(current_part)
                            current_part = sentence

        if current_part:
            append_chunk(current_part)

        return parts

    async def _send_rate_limit_message(self, channel_id: str, user_id: str):
        """Send rate limit exceeded message"""
        await self.web_client.chat_postMessage(
            channel=channel_id,
            text=self._build_rate_limit_message(user_id),
        )

    def _build_rate_limit_message(self, user_id: str) -> str:
        """Build a tier-aware rate-limit message for Slack responses."""
        user_tier = self.user_manager.get_user_tier(user_id)
        limits = self._get_rate_limit_config_for_tier(user_tier)
        return f"""
🚫 *Rate Limit Exceeded*

<@{user_id}>, you've reached your hourly request limit.

*Your tier:* {user_tier.value.title()}
*Hourly limit:* {limits.get('requests_per_hour', 0)}
*Burst limit:* {limits.get('burst_requests', 0)} requests / 5 min
*Limit resets:* At the top of each hour
        """.strip()

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
        rate_limit_window = time.time() - 3600
        recent_requests = [
            request_time
            for request_time in self.user_manager.rate_limits.get(user_id, [])
            if request_time > rate_limit_window
        ]
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

        for task in list(self.background_tasks):
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        if self.socket_client:
            await self.socket_client.disconnect()

        if not self._uses_granular_redis_state():
            await self._persist_state()
        await self.state_store.shutdown()
        try:
            await self.memory_manager.shutdown()
        except Exception:
            logger.warning("Slack memory shutdown failed", exc_info=True)

        logger.info("Slack bot shutdown complete")

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
                    types="public_channel,private_channel",
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

    def _mark_thread_active(self, channel_id: Optional[str], thread_ts: Optional[str]):
        """Mark a bot thread as active for follow-up replies."""
        if not channel_id or not thread_ts:
            return
        self.active_threads[self._thread_key(channel_id, thread_ts)] = datetime.now()

    def _is_active_thread(
        self, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> bool:
        """Return whether a Slack thread is still active."""
        if not channel_id or not thread_ts:
            return False

        key = self._thread_key(channel_id, thread_ts)
        last_activity = self.active_threads.get(key)
        if not last_activity:
            return False

        if datetime.now() - last_activity > self.active_thread_timeout:
            self.active_threads.pop(key, None)
            return False

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
        return not self._is_supported_inline_command(normalized)

    def _is_query_event(self, event: Dict[str, Any]) -> bool:
        """Return whether a Slack event should consume query quota."""
        return self._is_query_text(event.get("text", "")) or bool(event.get("files"))

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
        for key in ("team_id", "team"):
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
        if file_payload.get("mode") == "tombstone":
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
            content = await self._download_attachment_content(download_url)
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

    async def _download_attachment_content(self, download_url: str) -> Optional[bytes]:
        """Download a private Slack file using the bot token."""
        if not download_url or not self.bot_token:
            return None

        timeout_seconds = self.attachment_settings.get("download_timeout_seconds", 30)
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(
                    download_url,
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                return getattr(response, "content", None)
        except Exception as exc:
            logger.warning(
                f"Failed to download Slack attachment from {download_url}: {exc}"
            )
            return None

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
            user_id: [float(value) for value in values]
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
