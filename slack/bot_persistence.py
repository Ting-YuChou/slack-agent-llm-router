"""SlackBot mixin that saves and restores the bot's state."""

import logging
import uuid
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.utils.schema import UserTier
from slack.users import (
    _rate_limit_entries,
    _rate_limit_entry,
)
from slack.state_store import (
    STATE_SCHEMA_VERSION,
    RedisSlackStateStore,
)
from slack.conversation import ConversationContext
from slack.bot_agent import _is_complete_agent_run_record

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


class SlackBotPersistenceMixin:
    """SlackBot's state persistence."""

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
