"""Slack bot state persistence backends (memory, file, Redis)."""

import json
import logging
import os
import time
from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from slack.users import (
    RATE_LIMIT_WINDOW_SECONDS,
    _rate_limit_entries,
    _rate_limit_entry,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


STATE_SCHEMA_VERSION = 1
# How long a conversation, and the bot thread it lives in, stays live.
SLACK_SESSION_TIMEOUT = timedelta(hours=2)
# Redis expiry for tracked Agent runs; user settings never expire.
SLACK_AGENT_RUN_TTL_SECONDS = 30 * 24 * 3600


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
