"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.async_client import AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
import httpx

from src.utils.logger import setup_logging
from src.utils.schema import QueryRequest, UserTier
from src.utils.metrics import (
    INFERENCE_METRICS,
    SLACK_METRICS,
    USER_METRICS,
    histogram_average,
)

logger = logging.getLogger(__name__)
DEFAULT_SLACK_STATE_FILE = "data/slack_state.json"
DEFAULT_USER_PREFERENCES = {
    "response_length": "medium",
    "technical_level": "intermediate",
    "preferred_models": [],
    "threading": True,
}


def default_user_preferences() -> Dict[str, Any]:
    """Return a fresh copy of default Slack user preferences."""
    return {
        "response_length": DEFAULT_USER_PREFERENCES["response_length"],
        "technical_level": DEFAULT_USER_PREFERENCES["technical_level"],
        "preferred_models": list(DEFAULT_USER_PREFERENCES["preferred_models"]),
        "threading": DEFAULT_USER_PREFERENCES["threading"],
    }


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
    _on_update: Optional[Callable[["ConversationContext"], None]] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def attach_update_hook(
        self, callback: Optional[Callable[["ConversationContext"], None]]
    ):
        """Attach a persistence callback for context mutations."""
        self._on_update = callback

    def _notify_updated(self):
        """Persist the context after a mutation when a callback is attached."""
        if self._on_update is not None:
            self._on_update(self)

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

        # Keep only last 20 messages for context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        self.last_activity = datetime.now()
        self._notify_updated()

    def to_record(self) -> Dict[str, Any]:
        """Serialize the context for persistent storage."""
        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "conversation_history": list(self.conversation_history),
            "user_tier": self.user_tier.value,
            "preferences": dict(self.preferences),
            "last_activity": self.last_activity.isoformat(),
            "session_id": self.session_id,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "ConversationContext":
        """Deserialize a stored conversation context."""
        tier_name = record.get("user_tier", UserTier.FREE.value)
        try:
            user_tier = UserTier(tier_name)
        except ValueError:
            user_tier = UserTier.FREE

        last_activity = record.get("last_activity")
        return cls(
            user_id=record.get("user_id", ""),
            channel_id=record.get("channel_id", ""),
            thread_ts=record.get("thread_ts"),
            conversation_history=list(record.get("conversation_history", [])),
            user_tier=user_tier,
            preferences=dict(record.get("preferences", {})),
            last_activity=(
                datetime.fromisoformat(last_activity)
                if isinstance(last_activity, str)
                else datetime.now()
            ),
            session_id=record.get("session_id", str(uuid.uuid4())),
        )


class SlackStateStore:
    """Storage backend for Slack user and conversation state."""

    def get_user_record(self, user_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def update_user_record(self, user_id: str, updates: Dict[str, Any]):
        raise NotImplementedError

    def list_user_records(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def get_rate_limit_requests(self, user_id: str) -> List[float]:
        raise NotImplementedError

    def set_rate_limit_requests(self, user_id: str, requests: List[float]):
        raise NotImplementedError

    def get_conversation(self, context_key: str) -> Optional[ConversationContext]:
        raise NotImplementedError

    def set_conversation(self, context_key: str, context: ConversationContext):
        raise NotImplementedError

    def delete_conversation(self, context_key: str):
        raise NotImplementedError

    def list_conversations(self) -> Dict[str, ConversationContext]:
        raise NotImplementedError


class MemorySlackStateStore(SlackStateStore):
    """In-memory Slack state store."""

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.conversations: Dict[str, ConversationContext] = {}

    def get_user_record(self, user_id: str) -> Dict[str, Any]:
        return dict(self.users.get(user_id, {}))

    def update_user_record(self, user_id: str, updates: Dict[str, Any]):
        existing = dict(self.users.get(user_id, {}))
        existing.update(updates)
        self.users[user_id] = existing

    def list_user_records(self) -> Dict[str, Dict[str, Any]]:
        return {user_id: dict(record) for user_id, record in self.users.items()}

    def get_rate_limit_requests(self, user_id: str) -> List[float]:
        return list(self.rate_limits.get(user_id, []))

    def set_rate_limit_requests(self, user_id: str, requests: List[float]):
        self.rate_limits[user_id] = list(requests)

    def get_conversation(self, context_key: str) -> Optional[ConversationContext]:
        return self.conversations.get(context_key)

    def set_conversation(self, context_key: str, context: ConversationContext):
        self.conversations[context_key] = context

    def delete_conversation(self, context_key: str):
        self.conversations.pop(context_key, None)

    def list_conversations(self) -> Dict[str, ConversationContext]:
        return self.conversations


class FileSlackStateStore(SlackStateStore):
    """JSON-file backed Slack state store."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load_state()

    def _default_state(self) -> Dict[str, Any]:
        return {"users": {}, "rate_limits": {}, "conversations": {}}

    def _load_state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default_state()

        try:
            raw_state = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Slack state file: {self.path}") from exc

        state = self._default_state()
        if isinstance(raw_state, dict):
            state["users"] = dict(raw_state.get("users", {}))
            state["rate_limits"] = {
                user_id: list(requests)
                for user_id, requests in dict(raw_state.get("rate_limits", {})).items()
            }
            state["conversations"] = dict(raw_state.get("conversations", {}))
        return state

    def _persist_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def get_user_record(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state["users"].get(user_id, {}))

    def update_user_record(self, user_id: str, updates: Dict[str, Any]):
        with self._lock:
            existing = dict(self._state["users"].get(user_id, {}))
            existing.update(updates)
            self._state["users"][user_id] = existing
            self._persist_locked()

    def list_user_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                user_id: dict(record)
                for user_id, record in self._state["users"].items()
            }

    def get_rate_limit_requests(self, user_id: str) -> List[float]:
        with self._lock:
            return list(self._state["rate_limits"].get(user_id, []))

    def set_rate_limit_requests(self, user_id: str, requests: List[float]):
        with self._lock:
            self._state["rate_limits"][user_id] = list(requests)
            self._persist_locked()

    def get_conversation(self, context_key: str) -> Optional[ConversationContext]:
        with self._lock:
            record = self._state["conversations"].get(context_key)
        if record is None:
            return None
        return ConversationContext.from_record(record)

    def set_conversation(self, context_key: str, context: ConversationContext):
        with self._lock:
            self._state["conversations"][context_key] = context.to_record()
            self._persist_locked()

    def delete_conversation(self, context_key: str):
        with self._lock:
            self._state["conversations"].pop(context_key, None)
            self._persist_locked()

    def list_conversations(self) -> Dict[str, ConversationContext]:
        with self._lock:
            raw_conversations = dict(self._state["conversations"])
        return {
            context_key: ConversationContext.from_record(record)
            for context_key, record in raw_conversations.items()
        }


class RedisSlackStateStore(SlackStateStore):
    """Redis-backed Slack state store."""

    def __init__(self, config: Dict[str, Any], key_prefix: str = "slack_state"):
        try:
            import redis as redis_sync
        except ImportError as exc:
            raise RuntimeError(
                "redis package is required for Slack redis state backend"
            ) from exc

        password = config.get("password")
        password_env = config.get("password_env")
        if not password and password_env:
            password = os.getenv(password_env)

        self.client = redis_sync.Redis(
            host=config.get("host", "localhost"),
            port=int(config.get("port", 6379) or 6379),
            db=int(config.get("db", 0) or 0),
            username=config.get("username"),
            password=password,
            ssl=bool(config.get("ssl", False)),
            decode_responses=True,
        )
        self.client.ping()

        normalized_prefix = str(key_prefix or "slack_state").strip() or "slack_state"
        self._users_key = f"{normalized_prefix}:users"
        self._rate_limits_key = f"{normalized_prefix}:rate_limits"
        self._conversations_key = f"{normalized_prefix}:conversations"

    def _load_json(self, raw_value: Any, default: Any):
        if raw_value is None:
            return default

        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")

        try:
            return json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return default

    def get_user_record(self, user_id: str) -> Dict[str, Any]:
        return dict(self._load_json(self.client.hget(self._users_key, user_id), {}))

    def update_user_record(self, user_id: str, updates: Dict[str, Any]):
        existing = self.get_user_record(user_id)
        existing.update(updates)
        self.client.hset(self._users_key, user_id, json.dumps(existing))

    def list_user_records(self) -> Dict[str, Dict[str, Any]]:
        return {
            user_id: dict(self._load_json(record, {}))
            for user_id, record in self.client.hgetall(self._users_key).items()
        }

    def get_rate_limit_requests(self, user_id: str) -> List[float]:
        return list(
            self._load_json(self.client.hget(self._rate_limits_key, user_id), [])
        )

    def set_rate_limit_requests(self, user_id: str, requests: List[float]):
        self.client.hset(self._rate_limits_key, user_id, json.dumps(list(requests)))

    def get_conversation(self, context_key: str) -> Optional[ConversationContext]:
        record = self._load_json(
            self.client.hget(self._conversations_key, context_key), None
        )
        if record is None:
            return None
        return ConversationContext.from_record(record)

    def set_conversation(self, context_key: str, context: ConversationContext):
        self.client.hset(
            self._conversations_key,
            context_key,
            json.dumps(context.to_record()),
        )

    def delete_conversation(self, context_key: str):
        self.client.hdel(self._conversations_key, context_key)

    def list_conversations(self) -> Dict[str, ConversationContext]:
        return {
            context_key: ConversationContext.from_record(
                self._load_json(record, {})
            )
            for context_key, record in self.client.hgetall(
                self._conversations_key
            ).items()
        }


class UserManager:
    """Manages user data and preferences"""

    MAX_QUERY_HISTORY = 500
    QUERY_HISTORY_RETENTION_DAYS = 7

    def __init__(self, store: Optional[SlackStateStore] = None):
        self.store = store or MemorySlackStateStore()
        self._refresh_user_tier_metrics()

    def get_user_tier(self, user_id: str) -> UserTier:
        """Get user tier from user data"""
        user_data = self.store.get_user_record(user_id)
        tier_name = user_data.get("tier", "free")

        try:
            return UserTier(tier_name)
        except ValueError:
            return UserTier.FREE

    def set_user_tier(self, user_id: str, tier: UserTier):
        """Persist a user tier."""
        self.store.update_user_record(user_id, {"tier": tier.value})
        self._refresh_user_tier_metrics()

    def check_rate_limit(self, user_id: str, config: Dict[str, Any]) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        hour_window = current_time - 3600  # 1 hour window
        rate_limit_requests = self.store.get_rate_limit_requests(user_id)

        # Clean old requests
        rate_limit_requests = [
            req_time for req_time in rate_limit_requests if req_time > hour_window
        ]

        # Check limits
        requests_per_hour = config.get("requests_per_hour", 100)
        burst_requests = config.get("burst_requests", 5)

        # Check hourly limit
        if len(rate_limit_requests) >= requests_per_hour:
            return False

        # Check burst limit (last 5 minutes)
        burst_window = current_time - 300  # 5 minutes
        recent_requests = [
            req_time
            for req_time in rate_limit_requests
            if req_time > burst_window
        ]

        if len(recent_requests) >= burst_requests:
            return False

        # Add current request
        rate_limit_requests.append(current_time)
        self.store.set_rate_limit_requests(user_id, rate_limit_requests)
        return True

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        user_data = self.store.get_user_record(user_id)
        preferences = default_user_preferences()
        preferences.update(dict(user_data.get("preferences", {})))
        return preferences

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        user_data = self.store.get_user_record(user_id)
        stored_preferences = default_user_preferences()
        stored_preferences.update(dict(user_data.get("preferences", {})))
        stored_preferences.update(preferences)
        self.store.update_user_record(user_id, {"preferences": stored_preferences})
        self._refresh_user_tier_metrics()

    def record_query_event(
        self,
        user_id: str,
        user_tier: UserTier,
        *,
        query_type: str,
        response=None,
        success: bool,
        latency_ms: int,
    ):
        """Persist a bounded per-user query history for Slack analytics fallback."""
        user_data = self.store.get_user_record(user_id)
        history = list(user_data.get("query_history", []))
        event = {
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "success": bool(success),
            "latency_ms": max(0, int(latency_ms or 0)),
            "model_name": getattr(response, "model_name", None),
            "provider": getattr(response, "provider", None),
            "total_tokens": int(getattr(response, "total_tokens", 0) or 0),
            "cost_usd": float(getattr(response, "cost_usd", 0.0) or 0.0),
            "cached": bool(getattr(response, "cached", False)),
        }
        history.append(event)
        history = self._prune_query_history(history)
        self.store.update_user_record(
            user_id,
            {
                "tier": user_tier.value,
                "query_history": history,
            },
        )
        self._refresh_user_tier_metrics()

    def get_query_history(
        self, user_id: str, *, hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return bounded query history for a user, optionally filtered by recency."""
        user_data = self.store.get_user_record(user_id)
        history = self._prune_query_history(list(user_data.get("query_history", [])))
        if not history:
            return []

        if hours is None:
            return history

        cutoff = datetime.now() - timedelta(hours=hours)
        filtered_history = []
        for event in history:
            event_time = self._parse_event_timestamp(event.get("timestamp"))
            if event_time and event_time >= cutoff:
                filtered_history.append(event)
        return filtered_history

    def get_remaining_requests(self, user_id: str, config: Dict[str, Any]) -> int:
        """Return remaining requests in the current rolling hour window."""
        current_time = time.time()
        hour_window = current_time - 3600
        recent_requests = [
            request_time
            for request_time in self.store.get_rate_limit_requests(user_id)
            if request_time > hour_window
        ]
        requests_per_hour = int(config.get("requests_per_hour", 100) or 100)
        return max(0, requests_per_hour - len(recent_requests))

    def _parse_event_timestamp(self, raw_timestamp: Any) -> Optional[datetime]:
        if not isinstance(raw_timestamp, str):
            return None
        try:
            return datetime.fromisoformat(raw_timestamp)
        except ValueError:
            return None

    def _prune_query_history(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(days=self.QUERY_HISTORY_RETENTION_DAYS)
        retained_history = []
        for event in history:
            event_time = self._parse_event_timestamp(event.get("timestamp"))
            if event_time and event_time >= cutoff:
                retained_history.append(event)
        return retained_history[-self.MAX_QUERY_HISTORY :]

    def _refresh_user_tier_metrics(self):
        """Synchronize per-tier user gauges from persisted Slack user records."""
        records = self.store.list_user_records()
        counts = {tier.value: 0 for tier in UserTier}
        for record in records.values():
            tier_name = record.get("tier", UserTier.FREE.value)
            if tier_name not in counts:
                tier_name = UserTier.FREE.value
            counts[tier_name] += 1

        for tier_name, count in counts.items():
            USER_METRICS.users_by_tier.labels(user_tier=tier_name).set(count)


class ConversationManager:
    """Manages conversation contexts and history"""

    def __init__(
        self,
        config: Dict[str, Any],
        store: Optional[SlackStateStore] = None,
    ):
        self.config = config
        self.store = store or MemorySlackStateStore()
        self.session_timeout = timedelta(hours=2)  # Sessions expire after 2 hours

    def _context_key(
        self, user_id: str, channel_id: str, thread_ts: Optional[str] = None
    ) -> str:
        return f"{user_id}:{channel_id}:{thread_ts or 'main'}"

    @property
    def conversations(self) -> Dict[str, ConversationContext]:
        """Expose current conversations for diagnostics and tests."""
        return self.store.list_conversations()

    def get_or_create_context(
        self, user_id: str, channel_id: str, thread_ts: str = None
    ) -> ConversationContext:
        """Get existing conversation context or create new one"""
        context_key = self._context_key(user_id, channel_id, thread_ts)
        context = self.store.get_conversation(context_key)
        if context is not None:
            context.attach_update_hook(self.save_context)
            # Check if session is still valid
            if datetime.now() - context.last_activity < self.session_timeout:
                return context

            self.store.delete_conversation(context_key)

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
        context.attach_update_hook(self.save_context)
        self.store.set_conversation(context_key, context)
        return context

    def save_context(self, context: ConversationContext):
        """Persist the current context state."""
        context.attach_update_hook(self.save_context)
        self.store.set_conversation(
            self._context_key(context.user_id, context.channel_id, context.thread_ts),
            context,
        )

    def clear_context(
        self, user_id: str, channel_id: str, thread_ts: Optional[str] = None
    ):
        """Delete a stored conversation context."""
        self.store.delete_conversation(self._context_key(user_id, channel_id, thread_ts))

    def cleanup_expired_sessions(self):
        """Clean up expired conversation sessions"""
        current_time = datetime.now()
        expired_keys = []

        for key, context in self.store.list_conversations().items():
            if current_time - context.last_activity > self.session_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            self.store.delete_conversation(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired conversation sessions")

    def get_conversation_summary(
        self, user_id: str, channel_id: str, thread_ts: str = None
    ) -> str:
        """Get conversation summary for context"""
        context_key = self._context_key(user_id, channel_id, thread_ts)
        context = self.store.get_conversation(context_key)

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

    def has_active_thread_context(self, channel_id: str, thread_ts: Optional[str]) -> bool:
        """Return whether the bot has an active conversation in the given thread."""
        if not thread_ts:
            return False

        current_time = datetime.now()
        for context in self.store.list_conversations().values():
            if context.channel_id != channel_id:
                continue
            if context.thread_ts != thread_ts:
                continue
            if current_time - context.last_activity < self.session_timeout:
                return True

        return False


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
        }

    async def handle_message(
        self, event: Dict[str, Any], client: AsyncWebClient
    ) -> Optional[str]:
        """Handle incoming message"""
        text = event.get("text", "").strip()
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")
        allow_bare_command = event.get("type") == "app_mention"

        command_text = self._extract_command_text(text, allow_bare_command)
        if command_text is not None:
            return await self._handle_command(
                command_text,
                user_id,
                channel_id,
                client,
                thread_ts=thread_ts,
            )

        # Regular query - process through inference engine
        return await self._handle_query(text, user_id, channel_id, thread_ts, client)

    def _extract_command_text(
        self, text: str, allow_bare_command: bool = False
    ) -> Optional[str]:
        """Return normalized command text when the message should be handled as a bot command."""
        normalized_text = text.strip()
        if not normalized_text:
            return None

        for prefix in ("/llm ", "!llm "):
            if normalized_text.startswith(prefix):
                return normalized_text[len(prefix) :].strip()

        if allow_bare_command:
            command_name = normalized_text.split(maxsplit=1)[0].lower()
            if command_name in self.commands:
                return normalized_text

        return None

    async def _handle_command(
        self,
        command_text: str,
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
    ) -> str:
        """Handle bot commands"""
        parts = command_text.split()
        if not parts:
            return await self._handle_help_command(
                [], user_id, channel_id, client, thread_ts=thread_ts
            )

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in self.commands:
            return await self.commands[command](
                args, user_id, channel_id, client, thread_ts=thread_ts
            )
        else:
            return f"Unknown command: `{command}`. Type `/llm help` for available commands."

    async def _handle_help_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
    ) -> str:
        """Handle help command"""
        help_text = """
🤖 *LLM Router Bot Help*

*Basic Usage:*
• Mention me in a channel to start a reply thread
• Keep follow-up questions inside that thread
• Use `/llm ...` for explicit commands

*Commands:*
• `/llm help` - Show this help message
• `/llm settings` - View/update your preferences  
• `/llm status` - Show system status and your usage
• `/llm models` - List available models and their capabilities
• `/llm analytics` - Show usage analytics (premium users)
• `/llm clear` - Clear conversation history

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
🆓 Free: Basic models, 100 requests/hour
💎 Premium: All models, 500 requests/hour, priority support
🏢 Enterprise: Custom limits, dedicated resources
        """
        return help_text.strip()

    async def _handle_settings_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
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
*Preferred Models:* {', '.join(user_prefs.get('preferred_models', [])) or 'Auto-select'}

*Update Settings:*
• `/llm settings response_length short|medium|long`
• `/llm settings technical_level beginner|intermediate|expert`
• `/llm settings threading on|off`
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
            elif setting_name == "preferred_models":
                models = [m.strip() for m in setting_value.split(",")]
                user_prefs["preferred_models"] = models
            else:
                return f"Invalid setting: `{setting_name}` or value: `{setting_value}`"

            self.bot.user_manager.update_user_preferences(user_id, user_prefs)
            return f"✅ Updated {setting_name} to: {setting_value}"

        else:
            return "Usage: `/llm settings [setting_name] [value]` or `/llm settings` to view current settings"

    async def _handle_status_command(
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
    ) -> str:
        """Handle status command"""
        try:
            # Get system status
            system_health = await self.bot.get_system_status(user_id=user_id)

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
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
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
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
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
        self,
        args: List[str],
        user_id: str,
        channel_id: str,
        client: AsyncWebClient,
        thread_ts: Optional[str] = None,
    ) -> str:
        """Handle clear conversation command"""
        self.bot.conversation_manager.clear_context(user_id, channel_id, thread_ts)
        if thread_ts:
            return "🧹 Cleared this thread's conversation history."
        return "🧹 Cleared your channel-level conversation history."

    async def _handle_query(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        thread_ts: str,
        client: AsyncWebClient,
    ) -> str:
        """Handle regular query through inference engine"""
        request_started_at = time.time()
        query_type = self._classify_query_type(text)
        user_tier = self.bot.user_manager.get_user_tier(user_id)
        try:
            # Get or create conversation context
            context = self.bot.conversation_manager.get_or_create_context(
                user_id, channel_id, thread_ts
            )

            # Get user tier and preferences
            user_prefs = self.bot.user_manager.get_user_preferences(user_id)
            context.user_tier = user_tier
            context.preferences = user_prefs
            self.bot.conversation_manager.save_context(context)

            # Build conversation context
            conversation_context = (
                self.bot.conversation_manager.get_conversation_summary(
                    user_id, channel_id, thread_ts
                )
            )

            # Create query request
            query_request = QueryRequest(
                query=text,
                user_id=user_id,
                user_tier=user_tier,
                context=conversation_context,
                max_tokens=self._get_max_tokens_for_user(user_tier, user_prefs),
                temperature=0.7,
                priority=1 if user_tier != UserTier.FREE else 3,
            )

            # Process through inference engine
            response = await self.bot.inference_engine.process_query(query_request)

            # Add to conversation history
            context.add_message("user", text)
            context.add_message("assistant", response.response_text)

            # Update metrics
            SLACK_METRICS.messages_processed.labels(user_tier=user_tier.value).inc()
            SLACK_METRICS.response_time.observe(response.latency_ms / 1000)
            USER_METRICS.queries_per_user.labels(user_tier=user_tier.value).inc()
            USER_METRICS.tokens_per_user.labels(
                user_tier=user_tier.value, token_type="input"
            ).inc(response.token_count_input)
            USER_METRICS.tokens_per_user.labels(
                user_tier=user_tier.value, token_type="output"
            ).inc(response.token_count_output)
            USER_METRICS.cost_per_user.labels(user_tier=user_tier.value).inc(
                response.cost_usd
            )
            self.bot.user_manager.record_query_event(
                user_id,
                user_tier,
                query_type=query_type,
                response=response,
                success=not bool(response.error),
                latency_ms=response.latency_ms,
            )

            return response.response_text

        except Exception:
            logger.exception("Error processing query")
            SLACK_METRICS.errors.labels(error_type="query_processing").inc()
            self.bot.user_manager.record_query_event(
                user_id,
                user_tier,
                query_type=query_type,
                response=None,
                success=False,
                latency_ms=int((time.time() - request_started_at) * 1000),
            )
            return "❌ Sorry, I couldn't process that request right now. Please try again."

    def _classify_query_type(self, text: str) -> str:
        """Return the router classifier's best query type, falling back to general."""
        router = getattr(self.bot.inference_engine, "router", None)
        classifier = getattr(router, "classifier", None)
        classify_query = getattr(classifier, "classify_query", None)
        if callable(classify_query):
            try:
                query_type, _confidence = classify_query(text)
                return query_type.value if hasattr(query_type, "value") else str(query_type)
            except Exception:
                logger.exception("Failed to classify Slack query for analytics")
        return "general"

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

        return int(base * multiplier)


class SlackBot:
    """Main Slack bot class"""

    def __init__(
        self,
        config: Dict[str, Any],
        inference_engine,
        services: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.inference_engine = inference_engine
        self.services = services if services is not None else {}
        self.started_at = datetime.now()

        # Initialize components
        self.state_backend = config.get("state_backend", "memory")
        self.state_store = self._build_state_store()
        self.user_manager = UserManager(store=self.state_store)
        self.conversation_manager = ConversationManager(
            config,
            store=self.state_store,
        )
        self.message_handler = SlackMessageHandler(self)

        # Slack clients
        self.web_client = None
        self.socket_client = None

        # Bot configuration
        self.bot_user_id = None
        self.allowed_channels = config.get("channels", [])
        self._allowed_channel_names = {
            channel.lstrip("#")
            for channel in self.allowed_channels
            if isinstance(channel, str)
        }
        self._channel_name_cache: Dict[str, Optional[str]] = {}
        self.rate_limiting = config.get("rate_limiting", {})

        # Running state
        self.running = False

        if self.state_backend == "memory":
            logger.warning(
                "Slack bot is using in-memory state for users and conversations; configure a persistent backend before horizontal scaling."
            )

    def _build_state_store(self) -> SlackStateStore:
        """Create the configured Slack state backend."""
        if self.state_backend == "file":
            state_file = self.config.get("state_file", DEFAULT_SLACK_STATE_FILE)
            return FileSlackStateStore(state_file)
        if self.state_backend == "redis":
            return RedisSlackStateStore(
                dict(self.config.get("redis", {})),
                key_prefix=self.config.get("state_key_prefix", "slack_state"),
            )
        return MemorySlackStateStore()

    def _resolve_secret(self, value_key: str, env_key: str) -> Optional[str]:
        """Resolve secrets from direct config values or referenced env vars."""
        direct_value = self.config.get(value_key)
        if direct_value:
            return direct_value

        env_var_name = self.config.get(env_key)
        if env_var_name:
            return os.getenv(env_var_name)

        return None

    async def initialize(self):
        """Initialize Slack bot"""
        try:
            # Initialize Slack clients
            bot_token = self._resolve_secret("bot_token", "bot_token_env")
            app_token = self._resolve_secret("app_token", "app_token_env")

            if not bot_token or not app_token:
                raise ValueError("Slack bot_token and app_token are required")

            self.web_client = AsyncWebClient(token=bot_token)
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

        try:
            # Start socket mode client
            await self.socket_client.connect()

            # Start cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_sessions_periodically())

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
            if req.type == "events_api":
                # Handle Events API
                event = req.payload.get("event", {})
                await self._handle_event(event)

            elif req.type == "slash_commands":
                # Handle slash commands
                command = req.payload
                await self._handle_slash_command(command)

            # Acknowledge the request
            response = SocketModeResponse(envelope_id=req.envelope_id)
            await client.send_socket_mode_response(response)

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
        if event.get("bot_id") or event.get("user") == self.bot_user_id:
            return

        if event.get("subtype"):
            return

        channel_id = event.get("channel")
        if not await self._is_allowed_channel(event):
            return

        user_id = event.get("user")
        if not self._should_process_message_event(event):
            return

        # Check rate limiting
        if not self.user_manager.check_rate_limit(user_id, self.rate_limiting):
            await self._send_rate_limit_message(
                channel_id,
                user_id,
                thread_ts=self._get_reply_thread_ts(event),
            )
            return

        # Process message
        await self._process_message(event)

    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Handle app mentions"""
        channel_id = event.get("channel")
        user_id = event.get("user")

        if not await self._is_allowed_channel(event):
            return

        if not self.user_manager.check_rate_limit(user_id, self.rate_limiting):
            await self._send_rate_limit_message(
                channel_id,
                user_id,
                thread_ts=self._get_reply_thread_ts(event),
            )
            return

        # Remove mention from text
        text = event.get("text", "")
        mention_pattern = f"<@{self.bot_user_id}>"
        text = text.replace(mention_pattern, "").strip()
        if not text:
            text = "help"

        # Update event with cleaned text
        event["text"] = text

        # Process as regular message
        await self._process_message(event)

    def _should_process_message_event(self, event: Dict[str, Any]) -> bool:
        """Only accept threaded replies for active bot conversations."""
        text = (event.get("text") or "").strip()
        if not text:
            return False

        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        if not thread_ts:
            return False

        if event.get("parent_user_id") == self.bot_user_id:
            return True

        return self.conversation_manager.has_active_thread_context(channel_id, thread_ts)

    async def _is_allowed_channel(self, event: Dict[str, Any]) -> bool:
        """Accept configured Slack channel IDs or channel names."""
        if not self.allowed_channels:
            return True

        channel_id = event.get("channel")
        if channel_id in self.allowed_channels:
            return True

        channel_name = event.get("channel_name")
        if not channel_name and channel_id in self._channel_name_cache:
            channel_name = self._channel_name_cache[channel_id]

        if not channel_name and self.web_client and channel_id:
            try:
                response = await self.web_client.conversations_info(channel=channel_id)
                channel_name = response.get("channel", {}).get("name")
            except Exception as exc:
                logger.warning(
                    "Failed to resolve Slack channel name for allowlist check: %s", exc
                )
                channel_name = None
            self._channel_name_cache[channel_id] = channel_name

        if not channel_name:
            return False

        return channel_name in self._allowed_channel_names

    async def _handle_slash_command(self, command: Dict[str, Any]):
        """Handle slash commands"""
        command_text = command.get("text", "")
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")
        thread_ts = command.get("thread_ts")

        # Process command
        response_text = await self.message_handler._handle_command(
            command_text,
            user_id,
            channel_id,
            self.web_client,
            thread_ts=thread_ts,
        )

        # Send response as ephemeral command feedback.
        await self.web_client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=response_text,
        )

    async def _process_message(self, event: Dict[str, Any]):
        """Process incoming message"""
        try:
            channel_id = event.get("channel")
            thread_ts = self._get_reply_thread_ts(event)

            # Process message through handler
            response_text = await self.message_handler.handle_message(
                event, self.web_client
            )

            if response_text:
                # Split long responses
                max_length = self.config.get("response_settings", {}).get(
                    "max_response_length", 2000
                )
                if len(response_text) > max_length:
                    responses = self._split_response(response_text, max_length)

                    for response_part in responses:
                        await self.web_client.chat_postMessage(
                            channel=channel_id,
                            text=response_part,
                            thread_ts=thread_ts,
                        )
                        await asyncio.sleep(0.5)  # Small delay between parts
                else:
                    await self.web_client.chat_postMessage(
                        channel=channel_id, text=response_text, thread_ts=thread_ts
                    )

        except Exception:
            logger.exception("Error processing message")
            await self.web_client.chat_postMessage(
                channel=event.get("channel"),
                text="❌ Sorry, I couldn't handle that message right now. Please try again.",
                thread_ts=self._get_reply_thread_ts(event),
            )

    def _get_reply_thread_ts(self, event: Dict[str, Any]) -> Optional[str]:
        """Return the Slack thread ts to reply into for this event."""
        if self.config.get("response_settings", {}).get("thread_replies", True):
            return event.get("thread_ts") or event.get("ts")
        return None

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

    async def _send_rate_limit_message(
        self, channel_id: str, user_id: str, thread_ts: Optional[str] = None
    ):
        """Send rate limit exceeded message"""
        user_tier = self.user_manager.get_user_tier(user_id)

        message = f"""
🚫 *Rate Limit Exceeded*

<@{user_id}>, you've reached your hourly request limit.

*Your tier:* {user_tier.value.title()}
*Limit resets:* At the top of each hour

*Upgrade for higher limits:*
💎 Premium: 500 requests/hour
🏢 Enterprise: Custom limits
        """

        await self.web_client.chat_postMessage(
            channel=channel_id,
            text=message.strip(),
            thread_ts=thread_ts,
        )

    async def _cleanup_sessions_periodically(self):
        """Periodically clean up expired sessions"""
        while self.running:
            try:
                self.conversation_manager.cleanup_expired_sessions()
                await asyncio.sleep(3600)  # Clean up every hour
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(3600)

    async def get_system_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get system status for status command"""
        router = getattr(self.inference_engine, "router", None)
        monitoring = self.services.get("monitoring")
        pipeline = self.services.get("pipeline")
        available_models = await self.get_available_models(user_id or "")
        service_health = {
            "router": bool(router and getattr(router, "is_healthy", lambda: False)()),
            "inference": bool(
                getattr(self.inference_engine, "is_healthy", lambda: False)()
            ),
        }
        if pipeline is not None:
            service_health["pipeline"] = bool(
                getattr(pipeline, "is_healthy", lambda: False)()
            )
        if monitoring is not None:
            service_health["monitoring"] = bool(
                getattr(monitoring, "is_healthy", lambda: False)()
            )

        avg_response_time = self._get_process_local_avg_latency_ms()
        if pipeline is not None:
            analytics = await pipeline.get_query_analytics(hours=24)
            if analytics:
                avg_response_time = float(analytics.get("avg_latency", 0.0) or 0.0)

        return {
            "healthy": all(service_health.values()),
            "available_models": [
                model["name"] for model in available_models if model.get("available", False)
            ],
            "avg_response_time": avg_response_time,
            "uptime": self._format_uptime(),
            "services": service_health,
        }

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        recent_history = self.user_manager.get_query_history(user_id, hours=24)
        aggregated = self._aggregate_query_history(recent_history)
        pipeline = self.services.get("pipeline")

        if pipeline is not None:
            pipeline_analytics = await pipeline.get_query_analytics(user_id=user_id, hours=24)
            if pipeline_analytics:
                aggregated.update(
                    {
                        "queries_24h": int(
                            pipeline_analytics.get("total_queries", 0) or 0
                        ),
                        "cost_24h": float(
                            pipeline_analytics.get("total_cost", 0.0) or 0.0
                        ),
                        "avg_latency": float(
                            pipeline_analytics.get("avg_latency", 0.0) or 0.0
                        ),
                        "success_rate": float(
                            pipeline_analytics.get("success_rate", 0.0) or 0.0
                        ),
                    }
                )

        return {
            "queries_24h": aggregated["queries"],
            "cost_24h": aggregated["cost"],
            "avg_latency": aggregated["avg_latency_ms"],
            "success_rate": aggregated["success_rate_pct"],
            "remaining_requests": self.user_manager.get_remaining_requests(
                user_id, self.rate_limiting
            ),
        }

    async def get_available_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Get available models for user"""
        user_tier = self.user_manager.get_user_tier(user_id)
        router = getattr(self.inference_engine, "router", None)
        if router is None:
            return []

        provider_health = self._provider_health_by_name()
        models = []
        for model_name, model_config in getattr(router, "models", {}).items():
            check_access = getattr(router, "_check_user_access", None)
            if callable(check_access) and not check_access(model_config, user_tier):
                continue

            model_info = (
                router.get_model_info(model_name)
                if hasattr(router, "get_model_info")
                else None
            ) or {}
            model_stats = dict(model_info.get("stats", {}))
            provider_name = str(getattr(model_config, "provider", "")).lower()
            provider_is_healthy = provider_health.get(provider_name, True)
            models.append(
                {
                    "name": model_name,
                    "provider": getattr(model_config, "provider", "unknown"),
                    "capabilities": list(getattr(model_config, "capabilities", [])),
                    "max_tokens": int(getattr(model_config, "max_tokens", 0) or 0),
                    "cost_per_1k_tokens": float(
                        getattr(model_config, "cost_per_token", 0.0) or 0.0
                    )
                    * 1000.0,
                    "description": self._describe_model(model_config),
                    "available": bool(provider_is_healthy),
                    "success_rate": model_stats.get("success_rate"),
                    "avg_latency_ms": model_stats.get("avg_latency"),
                }
            )

        return sorted(
            models,
            key=lambda model: (
                not model.get("available", False),
                float(model.get("avg_latency_ms") or float("inf")),
                model["name"],
            ),
        )

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        recent_history = self.user_manager.get_query_history(user_id, hours=24 * 7)
        aggregated = self._aggregate_query_history(recent_history)
        analytics = {
            "total_queries": aggregated["queries"],
            "total_tokens": aggregated["total_tokens"],
            "total_cost": aggregated["cost"],
            "daily_avg_queries": aggregated["queries"] / 7.0,
            "avg_latency": aggregated["avg_latency_ms"],
            "success_rate": aggregated["success_rate_pct"],
            "cache_hit_rate": aggregated["cache_hit_rate_pct"],
            "model_breakdown": aggregated["model_breakdown"],
            "query_type_breakdown": aggregated["query_type_breakdown"],
        }

        pipeline = self.services.get("pipeline")
        if pipeline is not None:
            pipeline_analytics = await pipeline.get_query_analytics(
                user_id=user_id, hours=24 * 7
            )
            if pipeline_analytics:
                analytics.update(
                    {
                        "total_queries": int(
                            pipeline_analytics.get("total_queries", 0) or 0
                        ),
                        "total_tokens": int(
                            pipeline_analytics.get("total_tokens", 0) or 0
                        ),
                        "total_cost": float(
                            pipeline_analytics.get("total_cost", 0.0) or 0.0
                        ),
                        "daily_avg_queries": float(
                            int(pipeline_analytics.get("total_queries", 0) or 0) / 7.0
                        ),
                        "avg_latency": float(
                            pipeline_analytics.get("avg_latency", 0.0) or 0.0
                        ),
                        "success_rate": float(
                            pipeline_analytics.get("success_rate", 0.0) or 0.0
                        ),
                        "model_breakdown": dict(
                            pipeline_analytics.get("model_breakdown", {})
                        ),
                        "query_type_breakdown": dict(
                            pipeline_analytics.get("query_type_breakdown", {})
                        ),
                    }
                )

        return analytics

    def _provider_health_by_name(self) -> Dict[str, bool]:
        """Return provider health keyed by provider name."""
        if not hasattr(self.inference_engine, "get_health_status"):
            return {}

        health_status = self.inference_engine.get_health_status()
        providers = dict(health_status.get("providers", {}))
        return {
            provider_name.lower(): provider_payload.get("status") == "healthy"
            for provider_name, provider_payload in providers.items()
        }

    def _describe_model(self, model_config: Any) -> str:
        """Build a concise description from the actual router model config."""
        capabilities = list(getattr(model_config, "capabilities", []))
        capability_text = ", ".join(capabilities[:3]) if capabilities else "general"
        return (
            f"{getattr(model_config, 'provider', 'unknown')} model "
            f"for {capability_text}"
        )

    def _format_uptime(self) -> str:
        """Return a compact human-readable uptime string."""
        delta = max(datetime.now() - self.started_at, timedelta())
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _get_process_local_avg_latency_ms(self) -> float:
        """Return a best-effort local average latency when pipeline analytics are absent."""
        inference_stats = getattr(self.inference_engine, "inference_stats", {})
        total_requests = sum(
            int(stats.get("total_requests", 0) or 0) for stats in inference_stats.values()
        )
        total_time_seconds = sum(
            float(stats.get("total_time", 0.0) or 0.0)
            for stats in inference_stats.values()
        )
        if total_requests > 0 and total_time_seconds > 0:
            return (total_time_seconds / total_requests) * 1000.0
        return histogram_average(INFERENCE_METRICS.request_duration) * 1000.0

    def _aggregate_query_history(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate persisted Slack user query history."""
        total_queries = len(history)
        success_count = sum(1 for event in history if event.get("success"))
        total_latency = sum(int(event.get("latency_ms", 0) or 0) for event in history)
        total_tokens = sum(int(event.get("total_tokens", 0) or 0) for event in history)
        total_cost = sum(float(event.get("cost_usd", 0.0) or 0.0) for event in history)
        cached_count = sum(1 for event in history if event.get("cached"))

        model_breakdown: Dict[str, Dict[str, Any]] = {}
        query_type_breakdown: Dict[str, int] = {}
        for event in history:
            model_name = event.get("model_name")
            if model_name:
                model_entry = model_breakdown.setdefault(
                    model_name, {"queries": 0, "cost": 0.0}
                )
                model_entry["queries"] += 1
                model_entry["cost"] += float(event.get("cost_usd", 0.0) or 0.0)

            query_type = str(event.get("query_type") or "general")
            query_type_breakdown[query_type] = (
                query_type_breakdown.get(query_type, 0) + 1
            )

        return {
            "queries": total_queries,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "avg_latency_ms": (
                total_latency / total_queries if total_queries > 0 else 0.0
            ),
            "success_rate_pct": (
                (success_count / total_queries) * 100.0 if total_queries > 0 else 0.0
            ),
            "cache_hit_rate_pct": (
                (cached_count / total_queries) * 100.0 if total_queries > 0 else 0.0
            ),
            "model_breakdown": model_breakdown,
            "query_type_breakdown": query_type_breakdown,
        }

    async def shutdown(self):
        """Shutdown the Slack bot"""
        logger.info("Shutting down Slack bot...")
        self.running = False

        if self.socket_client:
            await self.socket_client.disconnect()

        logger.info("Slack bot shutdown complete")
