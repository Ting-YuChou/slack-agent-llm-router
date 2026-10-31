"""
Policy cache materialization for Flink routing hints.

This layer keeps Kafka and Flink off the synchronous API path:
- background workers consume requests.enriched / fast_lane_hints
- background workers consume routing.guardrails
- hints are materialized into Redis with a tiny in-process L1 cache
- API workers read the shared policy cache during routing
"""

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition
from prometheus_client import Counter, Gauge
import redis.asyncio as redis

from src.event_version import UnsupportedEventVersionError, is_supported_event_version
from src.kafka_topics import dead_letter_topic_name, dlq_enabled
from src.redaction import redact_text
from src.utils.metrics import PIPELINE_METRICS
from src.utils.bounded_state import BoundedTTLMap

logger = logging.getLogger(__name__)

GUARDRAIL_SCOPE_TYPES: Tuple[str, ...] = ("model", "provider")
REDIS_STATES: Tuple[str, ...] = ("available", "unavailable", "recovering")

POLICY_CACHE_UNAVAILABLE = Counter(
    "llm_router_policy_cache_unavailable_total",
    "Routing policy reads answered without the shared policy cache",
    ["reason"],
)
POLICY_CACHE_REDIS_STATE = Gauge(
    "llm_router_policy_cache_redis_state",
    "Routing policy cache Redis state (1 for the current state)",
    ["state"],
)
POLICY_CACHE_RECOVERY = Counter(
    "llm_router_policy_cache_recovery_total",
    "Routing policy cache Redis recovery attempts",
    ["outcome"],
)
POLICY_CACHE_STALE_WRITES = Counter(
    "llm_router_policy_cache_stale_writes_total",
    "Policy events not written: past their TTL, or older than the stored record",
    ["namespace", "reason"],
)

# Compare-and-set on event time (A6). KEYS[1] is the record, KEYS[2] an
# optional index set; ARGV is the payload, its TTL in seconds, its event time
# in ms and the index member. A stored record with a later event_time_ms wins;
# records without one (written before A6) are overwritten.
CAS_SET_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current then
  local ok, decoded = pcall(cjson.decode, current)
  if ok and type(decoded) == 'table' then
    local stored = tonumber(decoded['event_time_ms'])
    if stored and stored > tonumber(ARGV[3]) then
      return 0
    end
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
if #KEYS > 1 then
  redis.call('SADD', KEYS[2], ARGV[4])
end
return 1
"""

# Lifts a guardrail (A2) unless the stored one is newer, leaving a tombstone
# so an older avoid that arrives later (a replay, a rollback, another
# partition) is rejected by CAS_SET_SCRIPT. KEYS[1] is the guardrail record,
# KEYS[2] its index set; ARGV is the clear's event time in ms for the
# comparison, the index member, the tombstone payload and its TTL in seconds.
CAS_CLEAR_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current then
  local ok, decoded = pcall(cjson.decode, current)
  if ok and type(decoded) == 'table' then
    local stored = tonumber(decoded['event_time_ms'])
    if stored and stored > tonumber(ARGV[1]) then
      return 0
    end
  end
end
redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
redis.call('SREM', KEYS[2], ARGV[2])
return 1
"""
# A clear without an event time lifts any stored guardrail.
_LATEST_EVENT_TIME_MS = 2**53 - 1

# Event times before 2000-01-01 are counters or defaults, not wall clock.
_MIN_EVENT_TIME_MS = 946_684_800_000


def _parse_event_time_ms(value: Any) -> Optional[int]:
    """Epoch milliseconds from an epoch-ms number or an ISO-8601 string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        millis = int(value)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        millis = int(parsed.timestamp() * 1000)
    else:
        return None
    return millis if millis >= _MIN_EVENT_TIME_MS else None


def policy_event_time_ms(event: Dict[str, Any], *fields: str) -> Optional[int]:
    """The first usable event time among fields, in epoch milliseconds."""
    for field in fields:
        millis = _parse_event_time_ms(event.get(field))
        if millis is not None:
            return millis
    return None


POLICY_MATERIALIZER_CONSUMER_RESTARTS = Counter(
    "llm_router_policy_materializer_consumer_restarts_total",
    "Policy materializer consumer task restarts",
    ["topic"],
)
POLICY_MATERIALIZER_REDIS_RETRIES = Counter(
    "llm_router_policy_materializer_redis_retries_total",
    "Policy events held and retried in place while the cache's Redis is down",
    ["topic"],
)
POLICY_MATERIALIZER_INVALID_MESSAGES = Counter(
    "llm_router_policy_materializer_invalid_messages_total",
    "Malformed policy events moved past: dead-lettered, or skipped without a DLQ",
    ["topic", "destination"],
)


class PolicyEventInvalid(ValueError):
    """A policy event whose payload does not have the expected types (A7)."""


def _require_event_object(event: Any) -> None:
    if event is not None and not isinstance(event, dict):
        raise PolicyEventInvalid(
            f"policy event must be a JSON object, got {type(event).__name__}"
        )


def _event_text(event: Dict[str, Any], field: str, default: str = "") -> str:
    """A string field; numbers are accepted, containers are not."""
    value = event.get(field)
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise PolicyEventInvalid(f"{field} must be a string, got {type(value).__name__}")


def _event_number(event: Dict[str, Any], field: str) -> float:
    value = event.get(field)
    if value is None or value == "" or value is False:
        return 0.0
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except ValueError as exc:
            # The message names the field, never the value.
            raise PolicyEventInvalid(f"{field} must be a number") from exc
        if not math.isfinite(number):
            raise PolicyEventInvalid(f"{field} must be finite")
        return number
    raise PolicyEventInvalid(f"{field} must be a number, got {type(value).__name__}")


def _event_int(event: Dict[str, Any], field: str) -> int:
    return int(_event_number(event, field))


def _event_float(event: Dict[str, Any], field: str) -> float:
    return _event_number(event, field)


def _event_names(event: Dict[str, Any], field: str) -> List[str]:
    """A list of names; anything else, a bare string included, is invalid."""
    value = event.get(field)
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        raise PolicyEventInvalid(f"{field} must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise PolicyEventInvalid(
                f"{field} items must be strings, got {type(item).__name__}"
            )
    return list(value)


def _event_mapping(event: Dict[str, Any], field: str) -> Dict[str, Any]:
    value = event.get(field)
    if value is None or value == "":
        return {}
    if not isinstance(value, dict):
        raise PolicyEventInvalid(
            f"{field} must be an object, got {type(value).__name__}"
        )
    return dict(value)


def describe_policy_event_error(error: BaseException) -> str:
    """A value-free description of why an event was rejected.

    PolicyEventInvalid messages name the field and the expected type only;
    other errors (a JSON or UTF-8 decode error can quote the payload) are
    reduced to their type and position.
    """
    if isinstance(error, PolicyEventInvalid):
        return str(error)
    if isinstance(error, json.JSONDecodeError):
        return f"JSONDecodeError at char {error.pos}"
    if isinstance(error, UnicodeDecodeError):
        return f"UnicodeDecodeError at byte {error.start}"
    return type(error).__name__


class UnsupportedPolicyEventVersion(PolicyEventInvalid, UnsupportedEventVersionError):
    """A policy event with an unsupported ``event_version`` major (A12b)."""


class PolicyCacheWriteRejected(PolicyEventInvalid):
    """Redis rejected one write deterministically (WRONGTYPE, a script or
    argument error): retrying cannot help, so the event is dead-lettered or
    skipped like a malformed one, and Redis is not marked failed."""


# Error replies that mean "try again later", not "this write is wrong".
_TRANSIENT_REPLY_CODES = frozenset(
    {
        "BUSY",
        "LOADING",
        "TRYAGAIN",
        "MASTERDOWN",
        "CLUSTERDOWN",
        "READONLY",
        "NOREPLICAS",
        "MOVED",
        "ASK",
    }
)


def _is_rejected_by_redis(exc: BaseException) -> bool:
    """A deterministic error reply, as opposed to an outage or a busy server."""
    try:
        from redis import exceptions as redis_exceptions
    except ImportError:  # pragma: no cover - redis always ships
        return False
    if not isinstance(exc, redis_exceptions.ResponseError):
        return False
    transient_classes = tuple(
        cls
        for cls in (
            getattr(redis_exceptions, name, None)
            for name in (
                "BusyLoadingError",
                "ReadOnlyError",
                "TryAgainError",
                "ClusterDownError",
                "MasterDownError",
                "AskError",
                "MovedError",
            )
        )
        if isinstance(cls, type)
    )
    if transient_classes and isinstance(exc, transient_classes):
        return False
    words = str(exc).split()
    return not (words and words[0].upper() in _TRANSIENT_REPLY_CODES)


def _commit_lost_errors() -> Tuple[type, ...]:
    """aiokafka's errors for a commit lost to a rebalance or a left group."""
    try:
        from aiokafka import errors as kafka_errors
    except ImportError:  # pragma: no cover - aiokafka or its test stub
        return ()
    return tuple(
        cls
        for cls in (
            getattr(kafka_errors, "CommitFailedError", None),
            getattr(kafka_errors, "IllegalStateError", None),
        )
        if isinstance(cls, type) and issubclass(cls, BaseException)
    )


class PolicyCacheUnavailableError(RuntimeError):
    """The shared policy cache's Redis is unavailable or in its cooldown."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or f"routing policy cache unavailable: {reason}")
        self.reason = reason


class RoutingPolicyCache:
    """Shared routing policy cache with Redis durability and a small L1 cache."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        redis_config = self.config.get("redis", {})
        self.redis_host = redis_config.get("host", "localhost")
        self.redis_port = int(redis_config.get("port", 6379))
        self.redis_db = int(redis_config.get("db", 1))
        self.redis_password_env = redis_config.get("password_env")
        self.key_prefix = redis_config.get("key_prefix", "routing_policy")
        self.redis_connect_timeout_ms = int(redis_config.get("connect_timeout_ms", 100))
        self.redis_socket_timeout_ms = int(redis_config.get("socket_timeout_ms", 100))
        self.redis_max_connections = int(redis_config.get("max_connections", 50))
        self.redis_health_check_interval_seconds = int(
            redis_config.get("health_check_interval_seconds", 30)
        )
        self.recovery_cooldown_ms = max(
            0,
            int(
                self.config.get(
                    "recovery_cooldown_ms",
                    redis_config.get("recovery_cooldown_ms", 1000),
                )
            ),
        )
        # The router waits this long for get_effective_policy, then routes
        # without the shared policy.
        self.read_timeout_ms = max(1, int(self.config.get("read_timeout_ms", 50)))
        # This many router deadline misses in a row, with no successful read
        # between, start the cooldown: a black-holed Redis never answers
        # within the 50 ms deadline, so its socket timeout never fires. One
        # stall (GC, a busy loop) does not.
        self.read_timeout_failure_threshold = max(
            1, int(self.config.get("read_timeout_failure_threshold", 5))
        )
        self._consecutive_read_timeouts = 0
        self.request_ttl_seconds = int(self.config.get("request_ttl_seconds", 300))
        self.session_ttl_seconds = int(self.config.get("session_ttl_seconds", 600))
        self.user_ttl_seconds = int(self.config.get("user_ttl_seconds", 900))
        self.guardrail_ttl_seconds = int(self.config.get("guardrail_ttl_seconds", 180))
        self.local_cache_ttl_seconds = int(
            self.config.get("local_cache_ttl_seconds", 5)
        )
        self.local_cache_max_entries = int(
            self.config.get("local_cache_max_entries", 4096)
        )
        self.redis_client: Any = None
        self._initialized = False
        self._redis_available = True
        self._redis_failure_at: Optional[float] = None
        self._last_redis_error: Optional[str] = None
        self._recovery_lock = asyncio.Lock()
        self._time_func = time.monotonic
        self._wall_time_func = time.time
        self._local_guardrail_index: Dict[str, set[str]] = {
            "model": set(),
            "provider": set(),
        }
        self._local_cache: BoundedTTLMap[str, Dict[str, Any]] = BoundedTTLMap(
            max_entries=self.local_cache_max_entries,
            ttl_seconds=max(1, self.local_cache_ttl_seconds),
            on_evict=self._on_local_cache_evict,
            metric_name="policy_l1",
        )

    def _on_local_cache_evict(
        self, cache_key: str, value: Dict[str, Any], reason: str
    ) -> None:
        guardrail_prefix = f"{self.key_prefix}:guardrail:"
        if not cache_key.startswith(guardrail_prefix):
            return
        remainder = cache_key[len(guardrail_prefix) :]
        scope_type, separator, scope_key = remainder.partition(":")
        if separator and scope_type in self._local_guardrail_index:
            self._local_guardrail_index[scope_type].discard(scope_key)

    @property
    def read_timeout_seconds(self) -> float:
        return self.read_timeout_ms / 1000.0

    def _build_redis_client(self) -> Any:
        password = None
        if self.redis_password_env:
            password = os.getenv(str(self.redis_password_env))
        return redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=self.redis_connect_timeout_ms / 1000.0,
            socket_timeout=self.redis_socket_timeout_ms / 1000.0,
            max_connections=self.redis_max_connections,
            health_check_interval=self.redis_health_check_interval_seconds,
        )

    async def initialize(self):
        """Initialize Redis if the policy cache is enabled.

        A failure leaves the cache enabled but unavailable: reads fail open
        with policy_source "unavailable" and a ping after the cooldown brings
        Redis back.
        """
        if self._initialized or not self.enabled:
            self._initialized = True
            return

        self._initialized = True
        try:
            self.redis_client = self._build_redis_client()
            await self._run_redis_operation(self.redis_client.ping())
        except Exception as e:
            self._mark_redis_failure("initialize", e)
            logger.warning(f"Failed to initialize routing policy cache: {e}")
            return
        self._mark_redis_available()
        logger.info("Routing policy cache initialized")

    def is_healthy(self) -> bool:
        """Return whether the policy cache can serve routing.

        Routing fails open without Redis, so an unavailable Redis degrades the
        cache (see is_degraded and get_health_status) rather than making it
        unhealthy.
        """
        return True

    def is_degraded(self) -> bool:
        """Return whether the cache is enabled but its Redis is unavailable."""
        return bool(self._redis_enabled() and not self._redis_available)

    def get_health_status(self) -> Dict[str, Any]:
        """Policy cache status for readiness detail."""
        if not self.enabled:
            status = "disabled"
        elif self.is_degraded():
            status = "degraded"
        else:
            status = "healthy"
        return {
            "status": status,
            "redis_available": bool(self._redis_available),
            "last_error": self._last_redis_error,
            "unavailable_for_seconds": (
                round(self._time_func() - self._redis_failure_at, 3)
                if self._redis_failure_at is not None
                else None
            ),
        }

    def _redis_enabled(self) -> bool:
        """Whether reads and writes go to Redis rather than the local L1."""
        return bool(
            self.enabled
            and (self.redis_client is not None or not self._redis_available)
        )

    async def _run_redis_operation(self, awaitable: Any) -> Any:
        # redis-py applies the socket timeouts at the transport; injected
        # clients without a connection pool get the same bound here.
        if getattr(self.redis_client, "connection_pool", None) is not None:
            return await awaitable
        return await asyncio.wait_for(
            awaitable, timeout=max(0.001, self.redis_socket_timeout_ms / 1000.0)
        )

    def _note_redis_read(self) -> None:
        """A read Redis answered: deadline misses are no longer consecutive."""
        self._consecutive_read_timeouts = 0

    def _mark_redis_available(self) -> None:
        self._consecutive_read_timeouts = 0
        self._redis_available = True
        self._redis_failure_at = None
        self._set_redis_state("available")

    @staticmethod
    def _is_pool_exhausted(exc: BaseException) -> bool:
        """redis-py's "Too many connections": this replica is busy, Redis is up."""
        try:
            from redis.exceptions import ConnectionError as RedisConnectionError
            from redis.exceptions import MaxConnectionsError
        except ImportError:  # pragma: no cover - redis always ships these
            return False
        if isinstance(exc, MaxConnectionsError):
            return True
        return isinstance(exc, RedisConnectionError) and (
            "too many connections" in str(exc).lower()
        )

    def _note_redis_error(self, operation: str, exc: BaseException) -> str:
        """Classify a Redis error; only a real failure starts the cooldown.

        Returns the unavailability reason: "pool_exhausted" when the local
        connection pool is full, "redis_rejected" for a deterministic error
        reply (WRONGTYPE, script or argument errors), "redis_error" otherwise.
        """
        if self._is_pool_exhausted(exc):
            return "pool_exhausted"
        if _is_rejected_by_redis(exc):
            return "redis_rejected"
        self._mark_redis_failure(operation, exc)
        return "redis_error"

    def _mark_redis_failure(self, operation: str, exc: BaseException) -> None:
        self._redis_available = False
        self._redis_failure_at = self._time_func()
        self._last_redis_error = f"{operation}: {exc or type(exc).__name__}"
        self._set_redis_state("unavailable")

    @staticmethod
    def _set_redis_state(state: str) -> None:
        for candidate in REDIS_STATES:
            POLICY_CACHE_REDIS_STATE.labels(state=candidate).set(
                1 if candidate == state else 0
            )

    async def _recover_if_due(self) -> bool:
        """Return whether Redis is available, pinging once after the cooldown."""
        if self._redis_available and self.redis_client is not None:
            return True
        cooldown_seconds = self.recovery_cooldown_ms / 1000.0
        if (
            self._redis_failure_at is not None
            and self._time_func() - self._redis_failure_at < cooldown_seconds
        ):
            return False

        async with self._recovery_lock:
            if self._redis_available and self.redis_client is not None:
                return True
            if (
                self._redis_failure_at is not None
                and self._time_func() - self._redis_failure_at < cooldown_seconds
            ):
                return False
            POLICY_CACHE_RECOVERY.labels(outcome="attempt").inc()
            self._set_redis_state("recovering")
            try:
                if self.redis_client is None:
                    self.redis_client = self._build_redis_client()
                await self._run_redis_operation(self.redis_client.ping())
            except asyncio.CancelledError as e:
                # A caller's deadline cut the ping short: count it as a failure
                # so the next read waits out the cooldown instead.
                POLICY_CACHE_RECOVERY.labels(outcome="failure").inc()
                self._mark_redis_failure("recovery_ping", e)
                raise
            except Exception as e:
                POLICY_CACHE_RECOVERY.labels(outcome="failure").inc()
                self._mark_redis_failure("recovery_ping", e)
                return False
            POLICY_CACHE_RECOVERY.labels(outcome="success").inc()
            self._mark_redis_available()
            logger.info("Routing policy cache Redis recovered")
            return True

    async def _require_redis(self) -> None:
        if not await self._recover_if_due():
            raise PolicyCacheUnavailableError("cooldown")

    def record_read_timeout(self) -> None:
        """Count a policy read the router gave up on.

        The router's deadline is shorter than the socket timeout and also
        fires on event-loop stalls, so it says nothing certain about Redis:
        it does not mark Redis failed or start a cooldown.
        """
        POLICY_CACHE_UNAVAILABLE.labels(reason="timeout").inc()
        self._consecutive_read_timeouts += 1
        if (
            self._consecutive_read_timeouts >= self.read_timeout_failure_threshold
            and self._redis_enabled()
            and self._redis_available
        ):
            self._mark_redis_failure(
                "read",
                TimeoutError(
                    f"{self._consecutive_read_timeouts} policy reads in a row "
                    "missed the router deadline"
                ),
            )

    @staticmethod
    def _write_error(reason: str, exc: BaseException) -> Exception:
        if reason == "redis_rejected":
            words = str(exc).split()
            code = words[0] if words and words[0].isupper() else type(exc).__name__
            return PolicyCacheWriteRejected(f"redis rejected the write: {code}")
        return PolicyCacheUnavailableError(reason, str(exc))

    @staticmethod
    def _record_unavailable(reason: str) -> None:
        POLICY_CACHE_UNAVAILABLE.labels(reason=reason).inc()

    def _unavailable_policy(self) -> Dict[str, Any]:
        return {
            "policy_source": "unavailable",
            **self._empty_guardrail_policy(),
        }

    @staticmethod
    def _empty_guardrail_policy() -> Dict[str, Any]:
        return {
            "blocked_models": [],
            "warn_models": [],
            "blocked_providers": [],
            "warn_providers": [],
            "guardrail_reasons": {},
        }

    def _cache_key(self, namespace: str, identifier: str) -> str:
        return f"{self.key_prefix}:{namespace}:{identifier}"

    def _guardrail_index_key(self, scope_type: str) -> str:
        return self._cache_key("guardrails", scope_type)

    def _guardrail_cache_key(self, scope_type: str, identifier: str) -> str:
        return self._cache_key(f"guardrail:{scope_type}", identifier)

    def _request_hint_cache_key(self, request_id: str) -> str:
        """fast_lane_hints' own request record; reads merge it over requests.enriched."""
        return self._cache_key("request_hint", request_id)

    def _merge_request_records(
        self,
        enriched: Optional[Dict[str, Any]],
        hint: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """One request policy from both sources, the hint over the enrichment.

        Each consumer writes its own key, so neither drops the other's
        fields however their events are ordered (A6 review).
        """
        if enriched and hint:
            return self._merge_policy_records(enriched, hint)
        return enriched or hint or None

    def _user_hint_cache_key(self, user_id: str) -> str:
        return self._cache_key("user_hint", user_id)

    def _user_state_cache_key(self, user_id: str) -> str:
        return self._cache_key("user_state", user_id)

    def _local_lookup(self, cache_key: str) -> Tuple[bool, Any]:
        """(hit, value) from L1; a hit may hold None, a cached absence."""
        record = self._local_cache.get(cache_key)
        if not record:
            return False, None

        if record["expires_at"] <= time.time():
            self._local_cache.pop(cache_key, None)
            return False, None

        value = record["value"]
        return True, dict(value) if isinstance(value, dict) else None

    def _local_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        _hit, value = self._local_lookup(cache_key)
        return value if isinstance(value, dict) else None

    def _local_set(self, cache_key: str, value: Dict[str, Any], ttl_seconds: int):
        self._local_store(cache_key, dict(value), ttl_seconds)

    def _local_store(self, cache_key: str, value: Any, ttl_seconds: int):
        """Store an L1 value: a record, or None for a cached absence."""
        ttl_seconds = max(1, ttl_seconds)
        self._local_cache.set(
            cache_key,
            {"value": value, "expires_at": time.time() + ttl_seconds},
            ttl_seconds=ttl_seconds,
        )

    def _decode_record(self, cache_key: str, payload: Any) -> Optional[Dict[str, Any]]:
        """Decode a Redis payload and cache it in L1, absences included."""
        if not payload:
            self._local_store(cache_key, None, self.local_cache_ttl_seconds)
            return None
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON payload in routing policy cache: {cache_key}")
            return None
        if not isinstance(value, dict):
            logger.warning(f"Ignoring non-object routing policy payload: {cache_key}")
            return None
        self._local_set(cache_key, value, self.local_cache_ttl_seconds)
        return value

    async def _get_json(self, cache_key: str) -> Optional[Dict[str, Any]]:
        hit, cached = self._local_lookup(cache_key)
        if hit:
            return cached if isinstance(cached, dict) else None

        if not self._redis_enabled():
            return None

        if not await self._recover_if_due():
            self._record_unavailable("cooldown")
            return None
        try:
            payload = await self._run_redis_operation(self.redis_client.get(cache_key))
            self._note_redis_read()
        except Exception as e:
            self._record_unavailable(self._note_redis_error("read", e))
            logger.warning(f"Failed to read routing policy cache key {cache_key}: {e}")
            return None

        return self._decode_record(cache_key, payload)

    async def _read_policy_state(
        self, record_keys: List[str], scope_types: Tuple[str, ...]
    ) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, List[str]]]:
        """Read policy records and guardrail index members in one round trip.

        Records in L1, absences included, are not read again. The guardrail
        indexes are read from Redis every time, so a guardrail another replica
        sets or lifts applies at once. Returns the records by key and the
        index members by scope type. Without Redis the records come from L1
        and the members from the local guardrail index. Raises
        PolicyCacheUnavailableError when Redis is enabled but unavailable.
        """
        records: Dict[str, Optional[Dict[str, Any]]] = {}
        members: Dict[str, List[str]] = {}
        if not self._redis_enabled():
            for cache_key in record_keys:
                records[cache_key] = self._local_get(cache_key)
            for scope_type in scope_types:
                members[scope_type] = sorted(
                    self._local_guardrail_index.get(scope_type, set())
                )
            return records, members

        pending_keys = []
        for cache_key in record_keys:
            hit, value = self._local_lookup(cache_key)
            if hit:
                records[cache_key] = value if isinstance(value, dict) else None
            else:
                pending_keys.append(cache_key)
        if not pending_keys and not scope_types:
            return records, members

        await self._require_redis()
        try:
            pipeline = self.redis_client.pipeline(transaction=False)
            for cache_key in pending_keys:
                pipeline.get(cache_key)
            for scope_type in scope_types:
                pipeline.smembers(self._guardrail_index_key(scope_type))
            results = list(await self._run_redis_operation(pipeline.execute()))
            self._note_redis_read()
        except Exception as e:
            reason = self._note_redis_error("read", e)
            logger.warning(f"Failed to read routing policy cache: {e}")
            raise PolicyCacheUnavailableError(reason, str(e)) from e

        for cache_key, payload in zip(pending_keys, results):
            records[cache_key] = self._decode_record(cache_key, payload)
        for scope_type, scope_members in zip(scope_types, results[len(pending_keys) :]):
            members[scope_type] = sorted(
                str(member) for member in (scope_members or [])
            )
        return records, members

    def _remaining_ttl_seconds(
        self, ttl_seconds: int, event_time_ms: Optional[int]
    ) -> int:
        """The configured TTL less the event's age; <= 0 means already expired."""
        if event_time_ms is None:
            return ttl_seconds
        age_seconds = max(0.0, self._wall_time_func() - event_time_ms / 1000.0)
        return min(ttl_seconds, int(math.ceil(ttl_seconds - age_seconds)))

    @staticmethod
    def _namespace_of(cache_key: str, key_prefix: str) -> str:
        remainder = cache_key[len(key_prefix) + 1 :]
        return remainder.split(":", 1)[0] or "unknown"

    async def _compare_and_set(
        self,
        cache_key: str,
        value: Dict[str, Any],
        ttl_seconds: int,
        event_time_ms: int,
        index_key: Optional[str] = None,
        index_member: str = "",
    ) -> bool:
        """Write value unless Redis holds a record with a later event time."""
        keys = [cache_key] + ([index_key] if index_key else [])
        written = await self._run_redis_operation(
            self.redis_client.eval(
                CAS_SET_SCRIPT,
                len(keys),
                *keys,
                json.dumps(value),
                str(ttl_seconds),
                str(event_time_ms),
                index_member,
            )
        )
        if int(written or 0) != 1:
            POLICY_CACHE_STALE_WRITES.labels(
                namespace=self._namespace_of(cache_key, self.key_prefix),
                reason="older",
            ).inc()
            logger.debug(
                "Skipped policy write for %s: a newer record is stored", cache_key
            )
            return False
        return True

    def _record_expired(self, cache_key: str) -> None:
        POLICY_CACHE_STALE_WRITES.labels(
            namespace=self._namespace_of(cache_key, self.key_prefix),
            reason="expired",
        ).inc()
        logger.debug(
            "Skipped policy write for %s: the event outlived its TTL", cache_key
        )

    async def _set_json(
        self,
        cache_key: str,
        value: Dict[str, Any],
        ttl_seconds: int,
        event_time_ms: Optional[int] = None,
    ):
        """Write a policy record.

        On Redis, an event time shortens the TTL by the event's age, drops an
        expired event and never overwrites a record with a later event time
        (A6). The L1-only path keeps the configured TTL.
        """
        if not self._redis_enabled():
            self._local_set(cache_key, value, ttl_seconds)
            return

        if event_time_ms is not None:
            ttl_seconds = self._remaining_ttl_seconds(ttl_seconds, event_time_ms)
            if ttl_seconds <= 0:
                self._record_expired(cache_key)
                return
            value = {**value, "event_time_ms": event_time_ms}

        await self._require_redis()
        try:
            if event_time_ms is None:
                await self._run_redis_operation(
                    self.redis_client.setex(cache_key, ttl_seconds, json.dumps(value))
                )
            elif not await self._compare_and_set(
                cache_key, value, ttl_seconds, event_time_ms
            ):
                return
        except Exception as e:
            reason = self._note_redis_error("write", e)
            logger.warning(f"Failed to write routing policy cache key {cache_key}: {e}")
            raise self._write_error(reason, e) from e
        self._local_set(cache_key, value, ttl_seconds)

    async def get_request_policy(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request-scoped policy hints if present."""
        if not request_id:
            return None
        return self._merge_request_records(
            await self._get_json(self._cache_key("request", request_id)),
            await self._get_json(self._request_hint_cache_key(request_id)),
        )

    async def get_user_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the user-scoped policy view: the rolling routing.policy_state.

        Single fast-lane hints stay request-scoped (see
        materialize_fast_lane_hint), so no user_hint key is ever written or
        read on the routing path.
        """
        if not user_id:
            return None
        return await self.get_user_state_policy(user_id)

    async def get_user_hint_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user-scoped fast-lane hint policy if present (never written)."""
        if not user_id:
            return None
        return await self._get_json(self._user_hint_cache_key(user_id))

    async def get_user_state_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get rolling user-scoped policy state if present."""
        if not user_id:
            return None
        return await self._get_json(self._user_state_cache_key(user_id))

    async def get_session_policy(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session-scoped rolling routing policy if present."""
        if not session_id:
            return None
        return await self._get_json(self._cache_key("session", session_id))

    async def get_effective_policy(
        self,
        request_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge request, session, user, and guardrail policy views for the router.

        One Redis round trip reads the three records and the guardrail
        indexes; guardrail payloads not in L1 take one MGET more.
        """
        request_key = self._cache_key("request", request_id) if request_id else None
        hint_key = self._request_hint_cache_key(request_id) if request_id else None
        session_key = self._cache_key("session", session_id) if session_id else None
        user_key = self._user_state_cache_key(user_id) if user_id else None
        try:
            records, members = await self._read_policy_state(
                [key for key in (request_key, hint_key, session_key, user_key) if key],
                GUARDRAIL_SCOPE_TYPES,
            )
            active_guardrails = await self._guardrail_policy(members)
        except PolicyCacheUnavailableError as e:
            self._record_unavailable(e.reason)
            return self._unavailable_policy()
        request_policy = self._merge_request_records(
            records.get(request_key) if request_key else None,
            records.get(hint_key) if hint_key else None,
        )
        session_policy = records.get(session_key) if session_key else None
        user_policy = records.get(user_key) if user_key else None
        merged: Dict[str, Any] = {}
        policy_sources: List[str] = []

        if user_policy:
            merged = dict(user_policy)
            policy_sources.append("user")
        if session_policy:
            merged = self._merge_policy_records(merged, session_policy)
            policy_sources.append("session")
        if request_policy:
            merged = self._merge_policy_records(merged, request_policy)
            policy_sources.append("request")

        if policy_sources:
            merged["policy_source"] = "+".join(reversed(policy_sources))
            merged.update(active_guardrails)
            return merged

        return dict(active_guardrails)

    def _merge_policy_records(
        self, base_policy: Dict[str, Any], override_policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge request and user policies while preserving list semantics."""
        merged = dict(base_policy)
        list_fields = {
            "preferred_models",
            "avoid_models",
            "avoid_providers",
        }

        for key, value in override_policy.items():
            if value in (None, "", [], {}):
                continue

            if key in list_fields:
                combined = []
                for item in list(value) + list(merged.get(key, []) or []):
                    if item not in combined:
                        combined.append(item)
                merged[key] = combined
                continue

            merged[key] = value

        return merged

    async def materialize_request_enriched(self, event: Dict[str, Any]):
        """Materialize requests.enriched events into request/user caches."""
        _require_event_object(event)
        if not event:
            return

        request_id = _event_text(event, "request_id")
        user_id = _event_text(event, "user_id")
        route_to_fast_lane = bool(event.get("route_to_fast_lane", False))
        priority = _event_text(event, "priority")
        updated_at = event.get("emitted_at")

        request_policy = {
            "request_id": request_id,
            "user_id": user_id,
            "priority": priority,
            "route_to_fast_lane": route_to_fast_lane,
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": bool(event.get("requires_low_latency", False)),
            "requires_high_reasoning": bool(
                event.get("requires_high_reasoning", False)
            ),
            "long_context": bool(event.get("long_context", False)),
            "attachment_heavy": bool(event.get("attachment_heavy", False)),
            "code_heavy": bool(event.get("code_heavy", False)),
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "latency_sla": event.get("latency_sla"),
            "routing_intent_source": event.get("routing_intent_source"),
            "preferred_models": _event_names(event, "preferred_models"),
            "avoid_models": _event_names(event, "avoid_models"),
            "avoid_providers": _event_names(event, "avoid_providers"),
            "hint_reason": event.get("enrichment_stage", "priority_classification"),
            "source_event_type": event.get("event_type", "requests.enriched"),
            "updated_at": updated_at,
        }

        if request_id:
            await self._set_json(
                self._cache_key("request", request_id),
                request_policy,
                self.request_ttl_seconds,
                event_time_ms=policy_event_time_ms(event, "emitted_at"),
            )

        # Request enrichment is late relative to the synchronous API path. Keep it
        # request-scoped so a single task cannot promote future user fast-lane state.

    async def materialize_fast_lane_hint(self, event: Dict[str, Any]):
        """Materialize fast_lane_hints into request/user caches."""
        _require_event_object(event)
        if not event:
            return

        request_id = _event_text(event, "request_id")
        user_id = _event_text(event, "user_id")
        selected_model = _event_text(event, "selected_model") or None
        preferred_models = [selected_model] if selected_model else []
        updated_at = event.get("emitted_at")
        route_to_fast_lane = bool(event.get("route_to_fast_lane", False))
        requires_low_latency = event.get("requires_low_latency")
        requires_high_reasoning = event.get("requires_high_reasoning")
        long_context = event.get("long_context")
        attachment_heavy = event.get("attachment_heavy")
        code_heavy = event.get("code_heavy")

        request_policy = {
            "request_id": request_id,
            "user_id": user_id,
            "priority": event.get("priority"),
            "route_to_fast_lane": route_to_fast_lane,
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": (
                bool(requires_low_latency) if requires_low_latency is not None else None
            ),
            "requires_high_reasoning": (
                bool(requires_high_reasoning)
                if requires_high_reasoning is not None
                else None
            ),
            "long_context": bool(long_context) if long_context is not None else None,
            "attachment_heavy": (
                bool(attachment_heavy) if attachment_heavy is not None else None
            ),
            "code_heavy": bool(code_heavy) if code_heavy is not None else None,
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "latency_sla": event.get("latency_sla"),
            "routing_intent_source": event.get("routing_intent_source"),
            "preferred_models": preferred_models,
            "avoid_models": _event_names(event, "avoid_models"),
            "avoid_providers": _event_names(event, "avoid_providers"),
            "hint_reason": event.get("hint_reason", "fast_lane_candidate"),
            "source_event_type": event.get("event_type", "fast_lane_hints"),
            "updated_at": updated_at,
        }
        # The hint has its own key: no read-modify-write of the enrichment.
        if request_id:
            await self._set_json(
                self._request_hint_cache_key(request_id),
                request_policy,
                self.request_ttl_seconds,
                event_time_ms=policy_event_time_ms(event, "emitted_at"),
            )

        # Fast-lane hints are observability/request evidence. User or session
        # policy must come from rolling routing.policy_state, not a single hint.

    async def materialize_routing_policy_state(self, event: Dict[str, Any]):
        """Materialize rolling user/session policy state emitted by Flink analytics."""
        _require_event_object(event)
        if not event:
            return

        scope_type = _event_text(event, "scope_type").lower()
        scope_key = _event_text(event, "scope_key")
        if scope_type not in {"user", "session"} or not scope_key:
            return

        payload = {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "user_id": event.get("user_id"),
            "session_id": event.get("session_id"),
            "user_tier": event.get("user_tier"),
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": bool(event.get("requires_low_latency", False)),
            "requires_high_reasoning": bool(
                event.get("requires_high_reasoning", False)
            ),
            "route_to_fast_lane": bool(event.get("route_to_fast_lane", False)),
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "burst_protection_active": bool(
                event.get("burst_protection_active", False)
            ),
            "enterprise_priority_active": bool(
                event.get("enterprise_priority_active", False)
            ),
            "recent_request_count": _event_int(event, "recent_request_count"),
            "recent_error_rate": _event_float(event, "recent_error_rate"),
            "avg_total_tokens": _event_float(event, "avg_total_tokens"),
            "avg_cost_usd": _event_float(event, "avg_cost_usd"),
            "avg_latency_ms": _event_float(event, "avg_latency_ms"),
            "fast_lane_hit_rate": _event_float(event, "fast_lane_hit_rate"),
            "query_type_breakdown": _event_mapping(event, "query_type_breakdown"),
            "preferred_models": _event_names(event, "preferred_models"),
            "avoid_models": _event_names(event, "avoid_models"),
            "avoid_providers": _event_names(event, "avoid_providers"),
            "hint_reason": event.get("hint_reason", "rolling_policy_state"),
            "source_event_type": event.get("event_type", "routing.policy_state"),
            "updated_at": event.get("emitted_at"),
        }

        cache_key = self._cache_key("session", scope_key)
        ttl_seconds = self.session_ttl_seconds
        if scope_type == "user":
            cache_key = self._user_state_cache_key(scope_key)
            ttl_seconds = self.user_ttl_seconds
        # Flink stamps emitted_at when it emits, so a replayed window looks
        # new; latest_event_at is the newest completion the state covers.
        await self._set_json(
            cache_key,
            payload,
            ttl_seconds,
            event_time_ms=policy_event_time_ms(event, "latest_event_at", "emitted_at"),
        )

    async def materialize_routing_guardrail(self, event: Dict[str, Any]):
        """Materialize routing.guardrails events into model/provider guardrail views."""
        _require_event_object(event)
        if not event:
            return

        scope_type = _event_text(event, "scope_type").lower()
        scope_key = _event_text(event, "scope_key")
        if scope_type not in {"model", "provider"} or not scope_key:
            return

        payload = {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "guardrail_action": str(event.get("guardrail_action", "avoid") or "avoid"),
            "severity": str(event.get("severity", "warning") or "warning"),
            "trigger_type": str(event.get("trigger_type", "unknown") or "unknown"),
            "description": _event_text(event, "description"),
            "provider": event.get("provider"),
            "model_name": event.get("model_name"),
            "updated_at": event.get("emitted_at"),
            "source_event_type": event.get("event_type", "routing.guardrails"),
        }

        if payload["guardrail_action"] == "clear":
            await self._clear_routing_guardrail(event, scope_type, scope_key)
            return

        if not self._redis_enabled():
            self._local_guardrail_index.setdefault(scope_type, set()).add(scope_key)
            self._local_set(
                self._guardrail_cache_key(scope_type, scope_key),
                payload,
                self.guardrail_ttl_seconds,
            )
            return

        cache_key = self._guardrail_cache_key(scope_type, scope_key)
        index_key = self._guardrail_index_key(scope_type)
        # The window's end is when the guardrail's evidence is from.
        event_time_ms = policy_event_time_ms(event, "window_end_ms", "emitted_at")
        ttl_seconds = self._remaining_ttl_seconds(
            self.guardrail_ttl_seconds, event_time_ms
        )
        if ttl_seconds <= 0:
            self._record_expired(cache_key)
            return
        if event_time_ms is not None:
            payload["event_time_ms"] = event_time_ms
        await self._require_redis()
        try:
            if event_time_ms is None:
                transaction = self.redis_client.pipeline(transaction=True)
                transaction.setex(cache_key, ttl_seconds, json.dumps(payload))
                transaction.sadd(index_key, scope_key)
                await self._run_redis_operation(transaction.execute())
            elif not await self._compare_and_set(
                cache_key,
                payload,
                ttl_seconds,
                event_time_ms,
                index_key=index_key,
                index_member=scope_key,
            ):
                return
        except Exception as e:
            reason = self._note_redis_error("write", e)
            logger.warning(
                f"Failed to update routing guardrail index {scope_type}:{scope_key}: {e}"
            )
            raise self._write_error(reason, e) from e

        self._local_set(cache_key, payload, ttl_seconds)

    async def _clear_routing_guardrail(
        self, event: Dict[str, Any], scope_type: str, scope_key: str
    ) -> None:
        """Lift a guardrail on a Flink clear event: delete it and its index member.

        On Redis a stored guardrail with a later event time stays.
        """
        cache_key = self._guardrail_cache_key(scope_type, scope_key)
        if self._redis_enabled():
            event_time_ms = policy_event_time_ms(event, "window_end_ms", "emitted_at")
            # Readers skip a tombstone: it is not in the index and its action
            # is "clear".
            tombstone = {
                "scope_type": scope_type,
                "scope_key": scope_key,
                "guardrail_action": "clear",
                "tombstone": True,
                "trigger_type": _event_text(event, "trigger_type", "unknown"),
                "event_time_ms": (
                    event_time_ms
                    if event_time_ms is not None
                    else int(self._wall_time_func() * 1000)
                ),
                "updated_at": event.get("emitted_at"),
            }
            await self._require_redis()
            try:
                cleared = await self._run_redis_operation(
                    self.redis_client.eval(
                        CAS_CLEAR_SCRIPT,
                        2,
                        cache_key,
                        self._guardrail_index_key(scope_type),
                        str(
                            _LATEST_EVENT_TIME_MS
                            if event_time_ms is None
                            else event_time_ms
                        ),
                        scope_key,
                        json.dumps(tombstone),
                        str(self.guardrail_ttl_seconds),
                    )
                )
            except Exception as e:
                reason = self._note_redis_error("write", e)
                logger.warning(
                    f"Failed to clear routing guardrail {scope_type}:{scope_key}: {e}"
                )
                raise self._write_error(reason, e) from e
            if int(cleared or 0) != 1:
                POLICY_CACHE_STALE_WRITES.labels(
                    namespace="guardrail", reason="older"
                ).inc()
                return
        self._local_guardrail_index.setdefault(scope_type, set()).discard(scope_key)
        self._local_cache.pop(cache_key, None)

    async def get_active_guardrails(self) -> Dict[str, Any]:
        """Return active model/provider guardrails for router candidate filtering."""
        try:
            _records, members = await self._read_policy_state([], GUARDRAIL_SCOPE_TYPES)
            return await self._guardrail_policy(members)
        except PolicyCacheUnavailableError as e:
            self._record_unavailable(e.reason)
            return self._empty_guardrail_policy()

    async def _guardrail_policy(self, members: Dict[str, List[str]]) -> Dict[str, Any]:
        """The guardrail lists for the router, from the index members read."""
        active_guardrails = await self._read_guardrail_payloads(members)

        policy: Dict[str, Any] = self._empty_guardrail_policy()

        for scope_type in GUARDRAIL_SCOPE_TYPES:
            for scope_key, payload in active_guardrails.get(scope_type, {}).items():
                guardrail_action = str(
                    payload.get("guardrail_action", "avoid") or "avoid"
                )
                if guardrail_action == "clear":
                    continue
                reason = payload.get("description") or payload.get("trigger_type")
                if scope_type == "model":
                    target_key = (
                        "blocked_models"
                        if guardrail_action == "avoid"
                        else "warn_models"
                    )
                else:
                    target_key = (
                        "blocked_providers"
                        if guardrail_action == "avoid"
                        else "warn_providers"
                    )
                if scope_key not in policy[target_key]:
                    policy[target_key].append(scope_key)
                policy["guardrail_reasons"][f"{scope_type}:{scope_key}"] = reason

        return policy

    async def _read_guardrail_payloads(
        self, members: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Active guardrail payloads by scope; expired index members are pruned.

        Payloads in L1 are not read again; the rest take one MGET.
        """
        active: Dict[str, Dict[str, Dict[str, Any]]] = {
            scope_type: {} for scope_type in members
        }

        if not self._redis_enabled():
            for scope_type, scope_keys in members.items():
                for scope_key in scope_keys:
                    payload = self._local_get(
                        self._guardrail_cache_key(scope_type, scope_key)
                    )
                    if payload is None:
                        self._local_guardrail_index.setdefault(
                            scope_type, set()
                        ).discard(scope_key)
                        continue
                    active[scope_type][scope_key] = payload
            return active

        pending: List[Tuple[str, str, str]] = []
        for scope_type, scope_keys in members.items():
            for scope_key in scope_keys:
                cache_key = self._guardrail_cache_key(scope_type, scope_key)
                payload = self._local_get(cache_key)
                if payload is not None:
                    active[scope_type][scope_key] = payload
                else:
                    pending.append((scope_type, scope_key, cache_key))
        if not pending:
            return active

        try:
            payloads = list(
                await self._run_redis_operation(
                    self.redis_client.mget([cache_key for *_, cache_key in pending])
                )
            )
        except Exception as e:
            reason = self._note_redis_error("read", e)
            logger.warning(f"Failed to read routing guardrails: {e}")
            raise PolicyCacheUnavailableError(reason, str(e)) from e

        expired: List[Tuple[str, str]] = []
        for (scope_type, scope_key, cache_key), raw_payload in zip(pending, payloads):
            if not raw_payload:
                expired.append((scope_type, scope_key))
                continue
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON payload in routing policy cache: {cache_key}"
                )
                continue
            if not isinstance(payload, dict):
                continue
            self._local_set(cache_key, payload, self.local_cache_ttl_seconds)
            active[scope_type][scope_key] = payload

        if expired:
            await self._prune_expired_guardrails(expired)
        return active

    async def _prune_expired_guardrails(self, expired: List[Tuple[str, str]]) -> None:
        """Drop index members whose payload expired from the Redis index."""
        try:
            pipeline = self.redis_client.pipeline(transaction=False)
            for scope_type, scope_key in expired:
                pipeline.srem(self._guardrail_index_key(scope_type), scope_key)
            await self._run_redis_operation(pipeline.execute())
        except Exception as e:
            logger.warning(f"Failed to prune expired routing guardrails: {e}")

    async def shutdown(self):
        """Shutdown the shared policy cache."""
        if self.redis_client and hasattr(self.redis_client, "close"):
            await self.redis_client.close()
        self.redis_client = None


class PolicyMaterializer:
    """Background worker that materializes Flink hints into the shared policy cache."""

    def __init__(self, kafka_config: Dict[str, Any], policy_cache: RoutingPolicyCache):
        self.kafka_config = kafka_config or {}
        self.policy_cache = policy_cache
        self.topics = self.kafka_config.get("topics", {})
        consumer_config = self.kafka_config.get("consumer", {})
        materializer_config = policy_cache.config.get("consumer", {})
        self.consumer_config = {
            "bootstrap_servers": self.kafka_config.get(
                "bootstrap_servers", ["localhost:9092"]
            ),
            # Its own group: sharing the sink's would rebalance both (D4).
            "group_id": materializer_config.get(
                "group_id", "llm-router-policy-materializer"
            ),
            "auto_offset_reset": materializer_config.get(
                "auto_offset_reset",
                consumer_config.get("auto_offset_reset", "latest"),
            ),
            "enable_auto_commit": materializer_config.get(
                "enable_auto_commit",
                consumer_config.get("enable_auto_commit", True),
            ),
            "max_poll_records": materializer_config.get(
                "max_poll_records", consumer_config.get("max_poll_records", 500)
            ),
        }
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.topic_names_by_key: Dict[str, str] = {}
        self.consumer_task_status: Dict[str, Dict[str, Any]] = {}
        self.consumer_restart_counts: Dict[str, int] = {}
        self.supervisor_initial_backoff_seconds = float(
            materializer_config.get(
                "supervisor_initial_backoff_seconds",
                consumer_config.get("supervisor_initial_backoff_seconds", 1.0),
            )
        )
        self.supervisor_max_backoff_seconds = float(
            materializer_config.get(
                "supervisor_max_backoff_seconds",
                consumer_config.get("supervisor_max_backoff_seconds", 30.0),
            )
        )
        # Malformed events go to a DLQ when one is enabled; without one they
        # are skipped. Either way the offset moves on. The DLQ is on when the
        # shared kafka setting says so (the one provisioning reads, so the
        # topic exists); policy_cache.consumer.dlq_enabled can only turn it
        # off (D7).
        dlq_config = dict(self.kafka_config.get("dlq", {}) or {})
        self.dlq_enabled = dlq_enabled(self.kafka_config) and bool(
            materializer_config.get("dlq_enabled", True)
        )
        self.dlq_send_timeout_seconds = float(
            dlq_config.get("send_timeout_seconds", 10.0)
        )
        self.dlq_producer: Optional[Any] = None
        # While the cache's Redis is down a message is held and retried in
        # place: no commit, no consumer restart and no rebalance.
        self.redis_retry_initial_backoff_seconds = float(
            materializer_config.get(
                "redis_retry_initial_backoff_seconds",
                self.supervisor_initial_backoff_seconds,
            )
        )
        self.redis_retry_max_backoff_seconds = float(
            materializer_config.get(
                "redis_retry_max_backoff_seconds", self.supervisor_max_backoff_seconds
            )
        )
        self._redis_backoff_topics: set[str] = set()
        self.running = False

    async def initialize(self):
        """Initialize Kafka consumers for the policy materializer."""
        if not self.policy_cache.enabled:
            logger.info("Policy materializer disabled")
            return

        # A cache whose Redis failed to initialize stays enabled and recovers
        # on its own; consumers start anyway, and a write that fails leaves
        # its offset uncommitted for the supervisor to retry.
        await self.policy_cache.initialize()
        if not self.policy_cache.enabled:
            return

        await self._start_dlq_producer()

        topic_keys = (
            "requests_enriched",
            "fast_lane_hints",
            "routing_guardrails",
            "routing_policy_state",
        )
        for topic_key in topic_keys:
            topic_name = self.topics.get(topic_key)
            if not topic_name:
                continue
            consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
            await consumer.start()
            self.consumers[topic_key] = consumer
            self.topic_names_by_key[topic_key] = topic_name
            self.consumer_task_status[topic_key] = {
                "running": False,
                "last_error": None,
                "last_started_at": None,
                "last_stopped_at": None,
            }
            logger.info(f"Policy materializer consumer initialized: {topic_name}")

    async def _start_dlq_producer(self) -> None:
        if not self.dlq_enabled or self.dlq_producer is not None:
            return
        producer = AIOKafkaProducer(
            bootstrap_servers=self.consumer_config["bootstrap_servers"],
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8") if key else None,
            acks="all",
            enable_idempotence=True,
        )
        try:
            await producer.start()
        except Exception as e:
            logger.warning(
                "Policy materializer DLQ producer failed to start; malformed "
                "events will be skipped: %s",
                e,
            )
            return
        self.dlq_producer = producer

    async def start(self):
        """Start consuming policy events."""
        if not self.consumers:
            return

        self.running = True
        async with asyncio.TaskGroup() as task_group:
            for topic_key, consumer in self.consumers.items():
                task_group.create_task(
                    self._supervise_consumer(topic_key, consumer),
                    name=f"policy_materializer_{topic_key}",
                )

    async def _supervise_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        """Restart policy materializer consumers after task-level failures."""
        backoff_seconds = self.supervisor_initial_backoff_seconds
        while self.running:
            self.consumer_task_status.setdefault(topic_key, {})
            self.consumer_task_status[topic_key].update(
                {
                    "running": True,
                    "last_error": None,
                    "last_started_at": time.time(),
                }
            )
            try:
                await self._run_consumer(topic_key, consumer)
                if not self.running:
                    break
                raise RuntimeError("policy materializer consumer exited unexpectedly")
            except asyncio.CancelledError:
                self.consumer_task_status[topic_key].update(
                    {"running": False, "last_stopped_at": time.time()}
                )
                raise
            except Exception as e:
                self.consumer_task_status[topic_key].update(
                    {
                        "running": False,
                        "last_error": str(e),
                        "last_stopped_at": time.time(),
                    }
                )
                self.consumer_restart_counts[topic_key] = (
                    self.consumer_restart_counts.get(topic_key, 0) + 1
                )
                POLICY_MATERIALIZER_CONSUMER_RESTARTS.labels(topic=topic_key).inc()
                logger.error(
                    "Policy materializer consumer for %s stopped; restarting in %.1fs: %s",
                    topic_key,
                    backoff_seconds,
                    e,
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    max(backoff_seconds * 2, self.supervisor_initial_backoff_seconds),
                    self.supervisor_max_backoff_seconds,
                )
                consumer = await self._restart_consumer(topic_key)

    async def _restart_consumer(self, topic_key: str) -> AIOKafkaConsumer:
        old_consumer = self.consumers.get(topic_key)
        if old_consumer is not None:
            try:
                await old_consumer.stop()
            except Exception as e:
                logger.warning(
                    f"Failed to stop unhealthy policy consumer {topic_key}: {e}"
                )
        topic_name = self.topic_names_by_key.get(topic_key) or self.topics[topic_key]
        consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
        await consumer.start()
        self.consumers[topic_key] = consumer
        return consumer

    async def _run_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        handlers = {
            "requests_enriched": self._consume_requests_enriched,
            "fast_lane_hints": self._consume_fast_lane_hints,
            "routing_guardrails": self._consume_routing_guardrails,
            "routing_policy_state": self._consume_routing_policy_state,
        }
        handler = handlers.get(topic_key)
        if handler is None:
            raise RuntimeError(f"No policy materializer handler for {topic_key}")
        await handler(consumer)

    @staticmethod
    def _pause_assignment(consumer: Any) -> set:
        assignment = getattr(consumer, "assignment", None)
        pause = getattr(consumer, "pause", None)
        if not callable(assignment) or not callable(pause):
            return set()
        try:
            partitions = set(assignment() or ())
            if partitions:
                pause(*partitions)
            return partitions
        except Exception as e:
            logger.warning("Failed to pause policy consumer partitions: %s", e)
            return set()

    @staticmethod
    def _resume_assignment(consumer: Any, partitions: set) -> None:
        resume = getattr(consumer, "resume", None)
        if not partitions or not callable(resume):
            return
        try:
            resume(*partitions)
        except Exception as e:
            logger.warning("Failed to resume policy consumer partitions: %s", e)

    async def _wait_in_group(self, consumer: Any, seconds: float, paused: set) -> None:
        """Wait while polling the paused consumer so it stays in its group.

        aiokafka leaves the group when it is not polled for
        max_poll_interval_ms; paused partitions return no records. Should a
        poll return any anyway, the consumer is sought back to them.
        """
        getmany = getattr(consumer, "getmany", None)
        if not paused or not callable(getmany):
            await asyncio.sleep(seconds)
            return
        records = await getmany(timeout_ms=max(1, int(seconds * 1000)))
        for partition, batch in (records or {}).items():
            offsets = [
                getattr(record, "offset", None)
                for record in batch
                if getattr(record, "offset", None) is not None
            ]
            if offsets:
                consumer.seek(partition, min(offsets))

    async def _materialize_with_retry(
        self,
        topic_key: str,
        materialize: Any,
        payload: Dict[str, Any],
        consumer: Any = None,
    ) -> bool:
        """Materialize one event, waiting out an unavailable cache.

        While waiting the consumer's partitions are paused and it keeps
        polling, so it stays in its group past max_poll_interval_ms. Returns
        False when the materializer stops while waiting; the caller then
        leaves the offset uncommitted.
        """
        backoff_seconds = max(0.0, self.redis_retry_initial_backoff_seconds)
        paused: set = set()
        try:
            while True:
                try:
                    await materialize(payload)
                    return True
                except PolicyCacheUnavailableError as e:
                    if not self.running:
                        return False
                    self._redis_backoff_topics.add(topic_key)
                    POLICY_MATERIALIZER_REDIS_RETRIES.labels(topic=topic_key).inc()
                    logger.warning(
                        "Policy cache unavailable for %s (%s); retrying in %.2fs",
                        topic_key,
                        e,
                        backoff_seconds,
                    )
                    if not paused:
                        paused = self._pause_assignment(consumer)
                    await self._wait_in_group(consumer, backoff_seconds, paused)
                    if not self.running:
                        return False
                    backoff_seconds = min(
                        max(
                            backoff_seconds * 2,
                            self.redis_retry_initial_backoff_seconds,
                        ),
                        self.redis_retry_max_backoff_seconds,
                    )
        finally:
            self._resume_assignment(consumer, paused)
            self._redis_backoff_topics.discard(topic_key)

    async def _commit_processed_message(self, consumer: AIOKafkaConsumer, message: Any):
        """Commit a processed Kafka message when manual commits are enabled."""
        if self.consumer_config.get("enable_auto_commit", True):
            return

        topic = getattr(message, "topic", None)
        partition = getattr(message, "partition", None)
        offset = getattr(message, "offset", None)
        if topic is None or partition is None or offset is None:
            return

        try:
            await consumer.commit({TopicPartition(topic, partition): int(offset) + 1})
        except _commit_lost_errors() as e:
            # The group moved on (a rebalance, or a long Redis wait): the
            # message is redelivered and its rewrite is safe (A6), so carry
            # on rather than restart the consumer.
            logger.warning(
                "Policy offset commit for %s[%s]@%s was lost; it will be "
                "redelivered: %s",
                topic,
                partition,
                offset,
                type(e).__name__,
            )

    @staticmethod
    def _decode_message_value(message: Any) -> Dict[str, Any]:
        value = getattr(message, "value", None)
        if isinstance(value, bytes):
            value = json.loads(value.decode("utf-8"))
        elif isinstance(value, str):
            value = json.loads(value)
        elif not isinstance(value, dict):
            raise ValueError(f"Unsupported Kafka message payload type: {type(value)}")
        if not isinstance(value, dict):
            raise PolicyEventInvalid(
                f"policy event must be a JSON object, got {type(value).__name__}"
            )
        if not is_supported_event_version(value.get("event_version")):
            raise UnsupportedPolicyEventVersion(value.get("event_version"))
        return value

    async def _skip_invalid_message(
        self,
        topic_key: str,
        consumer: AIOKafkaConsumer,
        message: Any,
        error: Exception,
    ) -> None:
        """Dead-letter or skip a malformed event, then commit past it (A7)."""
        destination = "skipped"
        if self.dlq_producer is not None:
            try:
                await self._publish_dead_letter(topic_key, message, error)
                destination = "dlq"
            except Exception as e:
                destination = "dlq_failed"
                logger.error(
                    "Failed to dead-letter a malformed %s message; skipping it: %s",
                    topic_key,
                    e,
                )
        POLICY_MATERIALIZER_INVALID_MESSAGES.labels(
            topic=topic_key, destination=destination
        ).inc()
        logger.warning(
            "Skipping invalid %s message at %s[%s]@%s (%s): %s",
            topic_key,
            getattr(message, "topic", None),
            getattr(message, "partition", None),
            getattr(message, "offset", None),
            destination,
            describe_policy_event_error(error),
        )
        await self._commit_processed_message(consumer, message)

    @staticmethod
    def _payload_keys(message: Any) -> List[str]:
        """Top-level field names of the payload; values never reach the DLQ."""
        value = getattr(message, "value", None)
        try:
            if isinstance(value, bytes):
                value = json.loads(value.decode("utf-8"))
            elif isinstance(value, str):
                value = json.loads(value)
        except (ValueError, UnicodeDecodeError):
            return []
        return sorted(str(key) for key in value) if isinstance(value, dict) else []

    def _dead_letter_topic(self, topic_key: str, source_topic: str) -> str:
        """The DLQ topic, resolved by the helper the sink and provisioning use."""
        return dead_letter_topic_name(self.kafka_config, topic_key, source_topic)

    async def _publish_dead_letter(
        self, topic_key: str, message: Any, error: Exception
    ) -> None:
        source_topic = (
            getattr(message, "topic", None)
            or self.topic_names_by_key.get(topic_key)
            or self.topics.get(topic_key)
            or topic_key
        )
        dlq_topic = self._dead_letter_topic(topic_key, source_topic)
        offset = getattr(message, "offset", None)
        partition = getattr(message, "partition", None)
        payload = {
            "event_type": "dead_letter",
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "source_topic_key": topic_key,
            "source_topic": source_topic,
            "source_partition": partition,
            "source_offset": offset,
            "failure_stage": "validate",
            "error_type": type(error).__name__,
            # Parse errors can quote payload values.
            "error_message": redact_text(describe_policy_event_error(error))[:500],
            "payload_keys": self._payload_keys(message),
        }
        failure_reason = getattr(error, "reason", None)
        if isinstance(failure_reason, str) and failure_reason:
            payload["failure_reason"] = failure_reason
        key = f"{source_topic}:{partition}:{offset}"
        await asyncio.wait_for(
            self.dlq_producer.send_and_wait(dlq_topic, key=key, value=payload),
            timeout=self.dlq_send_timeout_seconds,
        )
        PIPELINE_METRICS.dead_letter_messages.labels(topic=topic_key).inc()

    async def _consume_requests_enriched(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    if not await self._materialize_with_retry(
                        "requests_enriched",
                        self.policy_cache.materialize_request_enriched,
                        payload,
                        consumer,
                    ):
                        break
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    await self._skip_invalid_message(
                        "requests_enriched", consumer, message, e
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize requests.enriched message: {e}"
                    )
                    raise
        except Exception as e:
            logger.error(f"requests.enriched consumer error: {e}")
            raise

    async def _consume_fast_lane_hints(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    if not await self._materialize_with_retry(
                        "fast_lane_hints",
                        self.policy_cache.materialize_fast_lane_hint,
                        payload,
                        consumer,
                    ):
                        break
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    await self._skip_invalid_message(
                        "fast_lane_hints", consumer, message, e
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize fast_lane_hints message: {e}"
                    )
                    raise
        except Exception as e:
            logger.error(f"fast_lane_hints consumer error: {e}")
            raise

    async def _consume_routing_guardrails(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    if not await self._materialize_with_retry(
                        "routing_guardrails",
                        self.policy_cache.materialize_routing_guardrail,
                        payload,
                        consumer,
                    ):
                        break
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    await self._skip_invalid_message(
                        "routing_guardrails", consumer, message, e
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize routing_guardrails message: {e}"
                    )
                    raise
        except Exception as e:
            logger.error(f"routing_guardrails consumer error: {e}")
            raise

    async def _consume_routing_policy_state(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    if not await self._materialize_with_retry(
                        "routing_policy_state",
                        self.policy_cache.materialize_routing_policy_state,
                        payload,
                        consumer,
                    ):
                        break
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    await self._skip_invalid_message(
                        "routing_policy_state", consumer, message, e
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize routing_policy_state message: {e}"
                    )
                    raise
        except Exception as e:
            logger.error(f"routing_policy_state consumer error: {e}")
            raise

    @staticmethod
    def _deserialize_message(message: bytes) -> Dict[str, Any]:
        return json.loads(message.decode("utf-8"))

    def is_healthy(self) -> bool:
        """Return whether the materializer is configured and its consumers run.

        A down Redis does not make it unhealthy: routing fails open, messages
        wait in place, and is_degraded() reports it.
        """
        if not self.policy_cache.enabled:
            return True
        if not self.consumers:
            return False
        return not any(
            status.get("last_error") for status in self.consumer_task_status.values()
        )

    def is_degraded(self) -> bool:
        """True while the cache is degraded or a consumer waits on Redis."""
        if self._redis_backoff_topics:
            return True
        cache_degraded = getattr(self.policy_cache, "is_degraded", None)
        return bool(cache_degraded()) if callable(cache_degraded) else False

    def get_health_status(self) -> Dict[str, Any]:
        """Return materializer consumer supervisor status."""
        cache_status_getter = getattr(self.policy_cache, "get_health_status", None)
        cache_status = cache_status_getter() if callable(cache_status_getter) else {}
        return {
            "healthy": self.is_healthy(),
            "degraded": self.is_degraded(),
            "waiting_on_redis": sorted(self._redis_backoff_topics),
            "policy_cache": cache_status,
            "topics": dict(self.consumer_task_status),
            "restart_counts": dict(self.consumer_restart_counts),
        }

    async def shutdown(self):
        """Shutdown Kafka consumers. The shared cache is owned by the platform."""
        self.running = False
        stops = [consumer.stop() for consumer in self.consumers.values()]
        if self.dlq_producer is not None:
            stops.append(self.dlq_producer.stop())
        results = await asyncio.gather(*stops, return_exceptions=True)
        self.consumers.clear()
        self.dlq_producer = None
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]
