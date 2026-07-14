"""Redis-backed API gateway admission control."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import redis.asyncio as redis
from redis.exceptions import MaxConnectionsError

from src.utils.metrics import ADMISSION_METRICS


logger = logging.getLogger(__name__)

QUEUE_PRIORITY_STRIDE = 1_000_000_000_000


ADMIT_SCRIPT = """
-- admission:atomic_admit
local queue_enabled = tonumber(ARGV[1])
local queue_member = ARGV[2]
local queue_score = tonumber(ARGV[3])
local queue_expiry_ms = tonumber(ARGV[4])
local queue_enqueue = tonumber(ARGV[5])
local queue_max_depth = tonumber(ARGV[6])
local queue_depth = 0
local pruned_count = 0

if queue_enabled == 1 then
    local queue_order_key = KEYS[#KEYS - 3]
    local queue_expiry_key = KEYS[#KEYS - 2]
    local queue_depth_key = KEYS[#KEYS - 1]
    local queue_sequence_key = KEYS[#KEYS]
    queue_depth = tonumber(redis.call("GET", queue_depth_key) or "0")
    if queue_depth > 0 then
        local expired = redis.call("ZRANGEBYSCORE", queue_expiry_key, "-inf", tonumber(ARGV[9]))
        for _, member in ipairs(expired) do
            local removed = redis.call("ZREM", queue_order_key, member)
            redis.call("ZREM", queue_expiry_key, member)
            if removed == 1 then
                queue_depth = math.max(0, queue_depth - 1)
                pruned_count = pruned_count + 1
            end
        end
        if queue_depth == 0 then
            redis.call("DEL", queue_depth_key)
            redis.call("DEL", queue_sequence_key)
        else
            redis.call("SET", queue_depth_key, queue_depth)
        end
    end
    if queue_enqueue == 0 and queue_depth > 0 then
        return {0, "queue_required", 0, queue_order_key, queue_depth, pruned_count, queue_score}
    end
    if queue_enqueue == 1 then
        local existing = redis.call("ZSCORE", queue_order_key, queue_member)
        if queue_max_depth and queue_max_depth > 0 and existing == false and queue_depth >= queue_max_depth then
            return {0, "queue_depth_exceeded", 1000, queue_order_key, queue_depth, pruned_count, queue_score}
        end
        if existing == false then
            local sequence = redis.call("INCR", queue_sequence_key)
            if queue_depth > 0 and sequence >= 1000000000000 then
                return {0, "queue_sequence_exhausted", 0, queue_order_key, queue_depth, pruned_count, queue_score}
            end
            queue_score = queue_score * 1000000000000 + sequence
            redis.call("ZADD", queue_order_key, queue_score, queue_member)
            queue_depth = queue_depth + 1
            redis.call("SET", queue_depth_key, queue_depth)
        else
            queue_score = tonumber(existing)
        end
        redis.call("ZADD", queue_expiry_key, queue_expiry_ms, queue_member)
        local head = redis.call("ZRANGE", queue_order_key, 0, 0)[1]
        if head ~= queue_member then
            return {0, "queue_wait", 0, queue_order_key, queue_depth, pruned_count, queue_score}
        end
    end
end

local reservation_key = KEYS[1]
local reservation_ttl_ms = tonumber(ARGV[7])
local bucket_ttl_ms = tonumber(ARGV[8])
local now_ms = tonumber(ARGV[9])
local active_count = tonumber(ARGV[10])
local arg_index = 11

for index = 1, active_count do
    local limit = tonumber(ARGV[arg_index])
    arg_index = arg_index + 1
    if limit and limit > 0 then
        local current = tonumber(redis.call("GET", KEYS[1 + index]) or "0")
        if current >= limit then
            return {0, "active_limit", 100, KEYS[1 + index], queue_depth, pruned_count, queue_score}
        end
    end
end

local bucket_count = tonumber(ARGV[arg_index])
arg_index = arg_index + 1
local bucket_key_start = 2 + active_count
local next_tokens = {}
local bucket_capacities = {}

for index = 1, bucket_count do
    local capacity = tonumber(ARGV[arg_index])
    local refill_per_ms = tonumber(ARGV[arg_index + 1])
    local cost = tonumber(ARGV[arg_index + 2])
    local reason = ARGV[arg_index + 3]
    arg_index = arg_index + 4

    local bucket_key = KEYS[bucket_key_start + index - 1]
    local values = redis.call("HMGET", bucket_key, "tokens", "updated_ms")
    local tokens = tonumber(values[1])
    local updated_ms = tonumber(values[2])
    if tokens == nil then
        tokens = capacity
    end
    if updated_ms == nil then
        updated_ms = now_ms
    end
    if now_ms > updated_ms and refill_per_ms and refill_per_ms > 0 then
        tokens = math.min(capacity, tokens + ((now_ms - updated_ms) * refill_per_ms))
    end
    if cost > 0 and tokens + 0.000001 < cost then
        local retry_ms = 1000
        if refill_per_ms and refill_per_ms > 0 then
            retry_ms = math.ceil((cost - tokens) / refill_per_ms)
        end
        return {0, reason, retry_ms, bucket_key, queue_depth, pruned_count, queue_score}
    end
    next_tokens[index] = tokens - cost
    bucket_capacities[index] = capacity
end

local reservation_created = redis.call(
    "SET",
    reservation_key,
    "1",
    "PX",
    reservation_ttl_ms,
    "NX"
)
if not reservation_created then
    return {0, "reservation_conflict", 1000, reservation_key, queue_depth, pruned_count, queue_score}
end

for index = 1, active_count do
    local active_key = KEYS[1 + index]
    redis.call("INCR", active_key)
    redis.call("PEXPIRE", active_key, reservation_ttl_ms)
end

for index = 1, bucket_count do
    local bucket_key = KEYS[bucket_key_start + index - 1]
    redis.call(
        "HSET",
        bucket_key,
        "tokens",
        tostring(next_tokens[index]),
        "updated_ms",
        tostring(now_ms)
    )
    redis.call("PEXPIRE", bucket_key, bucket_ttl_ms)
end

if queue_enabled == 1 and queue_enqueue == 1 then
    local removed = redis.call("ZREM", KEYS[#KEYS - 3], queue_member)
    redis.call("ZREM", KEYS[#KEYS - 2], queue_member)
    if removed == 1 then
        queue_depth = math.max(0, queue_depth - 1)
        if queue_depth == 0 then
            redis.call("DEL", KEYS[#KEYS - 1])
            redis.call("DEL", KEYS[#KEYS])
        else
            redis.call("SET", KEYS[#KEYS - 1], queue_depth)
        end
    end
end
return {1, "allowed", 0, "", queue_depth, pruned_count, queue_score}
"""

QUEUE_CLEANUP_SCRIPT = """
-- admission:queue_cleanup
local removed = redis.call("ZREM", KEYS[1], ARGV[1])
redis.call("ZREM", KEYS[2], ARGV[1])
local depth = tonumber(redis.call("GET", KEYS[3]) or "0")
if removed == 1 then
    depth = math.max(0, depth - 1)
end
if depth == 0 then
    redis.call("DEL", KEYS[3])
    redis.call("DEL", KEYS[4])
else
    redis.call("SET", KEYS[3], depth)
end
return {1, depth}
"""


RELEASE_SCRIPT = """
-- admission:release
local reservation_key = KEYS[1]
local deleted = redis.call("DEL", reservation_key)
if deleted == 0 then
    return {0}
end

local now_ms = tonumber(ARGV[1])
local bucket_ttl_ms = tonumber(ARGV[2])
local active_count = tonumber(ARGV[3])
local token_count = tonumber(ARGV[4])
local arg_index = 5

for index = 1, active_count do
    local active_key = KEYS[1 + index]
    local current = tonumber(redis.call("GET", active_key) or "0")
    if current <= 1 then
        redis.call("DEL", active_key)
    else
        redis.call("DECR", active_key)
    end
end

local token_key_start = 2 + active_count
for index = 1, token_count do
    local capacity = tonumber(ARGV[arg_index])
    local delta = tonumber(ARGV[arg_index + 1])
    arg_index = arg_index + 2
    local token_key = KEYS[token_key_start + index - 1]
    local current = tonumber(redis.call("HGET", token_key, "tokens"))
    if current == nil then
        current = capacity
    end
    local next_value = current + delta
    if delta > 0 then
        next_value = math.min(capacity, next_value)
    end
    redis.call(
        "HSET",
        token_key,
        "tokens",
        tostring(next_value),
        "updated_ms",
        tostring(now_ms)
    )
    redis.call("PEXPIRE", token_key, bucket_ttl_ms)
end

return {1}
"""


@dataclass
class BucketSpec:
    key: str
    capacity: float
    refill_per_ms: float
    cost: float
    reason: str
    scope: str


@dataclass
class ActiveSpec:
    key: str
    limit: int
    reason: str
    scope: str


@dataclass(frozen=True)
class QueueAdmissionRequest:
    scope: str
    member_id: str
    priority_score: float
    expiry_timestamp_ms: int
    enqueue: bool
    order_key: str = ""
    expiry_key: str = ""
    depth_key: str = ""
    sequence_key: str = ""
    max_depth: int = 0
    priority_band: int = 0


@dataclass
class AdmissionReservation:
    reservation_id: str
    stage: str
    active_specs: List[ActiveSpec] = field(default_factory=list)
    token_specs: List[BucketSpec] = field(default_factory=list)
    reserved_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdmissionDecision:
    allowed: bool
    status_code: int = 200
    error: Optional[str] = None
    reason: str = "allowed"
    message: str = ""
    retry_after_seconds: Optional[int] = None
    reservation: Optional[AdmissionReservation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        reservation: Optional[AdmissionReservation],
        *,
        reason: str = "allowed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AdmissionDecision":
        return cls(
            allowed=True,
            reason=reason,
            reservation=reservation,
            metadata=metadata or {},
        )

    @classmethod
    def reject(
        cls,
        *,
        status_code: int,
        error: str,
        reason: str,
        message: str,
        retry_after_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AdmissionDecision":
        return cls(
            allowed=False,
            status_code=status_code,
            error=error,
            reason=reason,
            message=message,
            retry_after_seconds=retry_after_seconds,
            metadata=metadata or {},
        )


class AdmissionRejectedError(Exception):
    """Raised when provider-level admission rejects a request."""

    def __init__(self, decision: AdmissionDecision):
        super().__init__(decision.reason)
        self.decision = decision


class RedisAdmissionController:
    """Redis-backed global API gateway admission controller."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        redis_client: Optional[Any] = None,
        time_func: Optional[Any] = None,
        sleep_func: Optional[Any] = None,
    ):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.failure_mode = str(self.config.get("failure_mode", "closed")).lower()
        if self.failure_mode not in {"open", "closed"}:
            self.failure_mode = "closed"
        self.redis_config = dict(self.config.get("redis", {}) or {})
        self.queue_config = dict(self.config.get("queue", {}) or {})
        self.key_prefix = str(self.redis_config.get("key_prefix", "api_gateway"))
        self.reservation_ttl_ms = int(
            self.config.get("reservation_ttl_ms", 15 * 60 * 1000)
        )
        self.bucket_ttl_ms = int(self.config.get("bucket_ttl_ms", 60 * 60 * 1000))
        self.redis_client = redis_client
        self._owns_redis_client = redis_client is None
        self._redis_available = redis_client is not None
        self._redis_failure_at: Optional[float] = None
        self._recovery_lock = asyncio.Lock()
        self._time_func = time_func or time.time
        self._sleep_func = sleep_func or asyncio.sleep
        self._set_redis_state("available" if self._redis_available else "unavailable")

    async def initialize(self):
        """Initialize the Redis connection used for atomic admission decisions."""
        if not self.enabled:
            return
        if self.redis_client is not None:
            try:
                await self._ping()
            except Exception as exc:
                ADMISSION_METRICS.redis_errors.labels(operation="initialize").inc()
                self._mark_redis_failure("initialize", exc)
                logger.warning("Failed to initialize API admission controller: %s", exc)
            return

        try:
            password = None
            password_env = self.redis_config.get("password_env")
            if password_env:
                password = os.getenv(str(password_env))

            redis_url = self.redis_config.get("url")
            client_options = {
                "socket_connect_timeout": int(
                    self.redis_config.get("connect_timeout_ms", 100)
                )
                / 1000.0,
                "socket_timeout": int(self.redis_config.get("socket_timeout_ms", 100))
                / 1000.0,
                "max_connections": int(self.redis_config.get("max_connections", 100)),
                "health_check_interval": 30,
            }
            if redis_url and hasattr(redis.Redis, "from_url"):
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    password=password,
                    decode_responses=True,
                    **client_options,
                )
            else:
                self.redis_client = redis.Redis(
                    host=self.redis_config.get("host", "localhost"),
                    port=int(self.redis_config.get("port", 6379)),
                    db=int(self.redis_config.get("db", 4)),
                    password=password,
                    decode_responses=True,
                    **client_options,
                )
            await self._ping()
            logger.info("API admission controller initialized")
        except Exception as exc:
            ADMISSION_METRICS.redis_errors.labels(operation="initialize").inc()
            self._mark_redis_failure("initialize", exc)
            logger.warning("Failed to initialize API admission controller: %s", exc)

    async def _ping(self):
        if self.redis_client is None:
            self._redis_available = False
            self._set_redis_state("unavailable")
            return
        await self._run_redis_operation(self.redis_client.ping())
        self._mark_redis_available()

    def is_healthy(self) -> bool:
        if not self.enabled:
            return True
        return self._redis_available or self.failure_mode == "open"

    async def shutdown(self):
        if self.redis_client and self._owns_redis_client:
            close_method = getattr(self.redis_client, "close", None)
            if close_method:
                result = close_method()
                if hasattr(result, "__await__"):
                    await result
        self.redis_client = None
        self._redis_available = False
        self._redis_failure_at = None
        self._set_redis_state("unavailable")

    async def acquire_http(
        self,
        *,
        endpoint: str,
        method: str,
        identifier: Optional[str] = None,
    ) -> AdmissionDecision:
        """Acquire a lightweight gateway reservation for protected API endpoints."""
        active_specs = self._global_active_specs()
        bucket_specs = self._global_request_bucket_specs()
        metadata = {"endpoint": endpoint, "method": method}
        if identifier:
            bucket_specs.extend(self._per_identity_request_bucket_specs(identifier))
        return await self._acquire(
            stage="http",
            active_specs=active_specs,
            bucket_specs=bucket_specs,
            token_specs=[],
            reserved_tokens=0,
            metadata=metadata,
        )

    async def acquire_route(self, request: Any) -> AdmissionDecision:
        """Acquire request-level admission after /route body validation."""
        active_specs = self._global_active_specs()
        bucket_specs = self._global_request_bucket_specs()
        bucket_specs.extend(
            self._user_and_tier_request_bucket_specs(
                user_id=str(request.user_id),
                user_tier=self._tier_value(request.user_tier),
            )
        )
        return await self._acquire(
            stage="route",
            active_specs=active_specs,
            bucket_specs=bucket_specs,
            token_specs=[],
            reserved_tokens=0,
            metadata={
                "user_tier": self._tier_value(request.user_tier),
                "request_id": str(getattr(request, "request_id", "")),
            },
        )

    async def acquire_provider(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
        queue_request: Optional[QueueAdmissionRequest] = None,
    ) -> AdmissionDecision:
        """Acquire model/provider capacity and token budget before provider IO."""
        provider_name = provider.lower()
        normalized_model = self._scope_value(model_name)
        active_specs = self._model_provider_active_specs(
            model_name=normalized_model,
            provider=provider_name,
        )
        bucket_specs = self._model_provider_request_bucket_specs(
            model_name=normalized_model,
            provider=provider_name,
        )
        reserved_tokens = max(0, int(estimated_input_tokens)) + max(
            0, int(getattr(request, "max_tokens", 0) or 0)
        )
        token_specs = self._token_bucket_specs(
            request=request,
            model_name=normalized_model,
            provider=provider_name,
            reserved_tokens=reserved_tokens,
        )
        return await self._acquire(
            stage="provider",
            active_specs=active_specs,
            bucket_specs=bucket_specs,
            token_specs=token_specs,
            reserved_tokens=reserved_tokens if token_specs else 0,
            metadata={
                "model": model_name,
                "provider": provider_name,
                "user_tier": self._tier_value(request.user_tier),
            },
            queue_request=queue_request,
        )

    async def release(
        self,
        reservation: Optional[AdmissionReservation],
        *,
        actual_tokens: Optional[int] = None,
    ) -> bool:
        """Release an admission reservation, idempotently."""
        if not reservation or not self.enabled or self.redis_client is None:
            return False

        delta = 0
        if actual_tokens is not None and reservation.reserved_tokens > 0:
            delta = reservation.reserved_tokens - max(0, int(actual_tokens))

        token_args: List[str] = []
        token_keys: List[str] = []
        for spec in reservation.token_specs:
            token_keys.append(spec.key)
            token_args.extend([str(spec.capacity), str(delta)])

        keys = (
            [self._reservation_key(reservation.reservation_id)]
            + [spec.key for spec in reservation.active_specs]
            + token_keys
        )
        args = [
            str(self._now_ms()),
            str(self.bucket_ttl_ms),
            str(len(reservation.active_specs)),
            str(len(token_keys)),
            *token_args,
        ]

        try:
            result = await self._eval(RELEASE_SCRIPT, keys, args)
            released = bool(int(self._result_value(result, 0, 0)))
            if released:
                for active_spec in reservation.active_specs:
                    ADMISSION_METRICS.active_reservations.labels(
                        stage=reservation.stage,
                        scope=active_spec.scope,
                    ).dec()
                if reservation.reserved_tokens > 0:
                    ADMISSION_METRICS.tokens.labels(
                        stage=reservation.stage,
                        token_type="actual",
                    ).inc(max(0, int(actual_tokens or reservation.reserved_tokens)))
                    if delta > 0:
                        ADMISSION_METRICS.tokens.labels(
                            stage=reservation.stage,
                            token_type="refunded",
                        ).inc(delta)
                    elif delta < 0:
                        ADMISSION_METRICS.tokens.labels(
                            stage=reservation.stage,
                            token_type="overage",
                        ).inc(abs(delta))
            return released
        except Exception as exc:
            ADMISSION_METRICS.redis_errors.labels(operation="release").inc()
            self._mark_redis_failure("release", exc)
            logger.warning("Admission release failed: %s", exc)
            return False

    async def _acquire(
        self,
        *,
        stage: str,
        active_specs: List[ActiveSpec],
        bucket_specs: List[BucketSpec],
        token_specs: List[BucketSpec],
        reserved_tokens: int,
        metadata: Dict[str, Any],
        queue_request: Optional[QueueAdmissionRequest] = None,
    ) -> AdmissionDecision:
        if not self.enabled:
            self._record_decision(stage, "allowed", "disabled")
            return AdmissionDecision.allow(None, reason="disabled", metadata=metadata)

        if not self._redis_available:
            await self._recover_if_due()
            if not self._redis_available:
                unavailable = self._unavailable_decision(stage)
                if unavailable:
                    return unavailable
                return AdmissionDecision.allow(
                    None, reason="fail_open", metadata=metadata
                )

        external_queue = queue_request is not None
        queue_enabled = bool(self.queue_config.get("enabled", True))
        if queue_request is None and queue_enabled:
            member_id = uuid.uuid4().hex
            version = str(self.queue_config.get("control_plane_version", "v2"))
            prefix = self._key(version, "queue", "global")
            queue_request = QueueAdmissionRequest(
                scope="global",
                member_id=member_id,
                priority_score=0,
                priority_band=0,
                expiry_timestamp_ms=self._queue_expiry_timestamp_ms(self._now_ms()),
                enqueue=False,
                order_key=f"{prefix}:order",
                expiry_key=f"{prefix}:expiry",
                depth_key=f"{prefix}:depth",
                sequence_key=f"{prefix}:sequence",
                max_depth=int(self.queue_config.get("max_depth", 0) or 0),
            )

        queued = False
        queue_started = self._time_func()
        first_attempt = True
        try:
            timeout_ms = int(self.queue_config.get("timeout_ms", 0) or 0)
            poll_interval_ms = max(
                1, int(self.queue_config.get("poll_interval_ms", 25) or 25)
            )
            max_poll_interval_ms = max(
                poll_interval_ms,
                int(self.queue_config.get("max_poll_interval_ms", 250) or 250),
            )
            jitter_ratio = max(
                0.0, float(self.queue_config.get("poll_jitter_ratio", 0.2) or 0.0)
            )
            deadline = self._now_ms() + timeout_ms
            last_denial: Optional[AdmissionDecision] = None

            while True:
                decision = await self._try_admit(
                    stage=stage,
                    active_specs=active_specs,
                    bucket_specs=[*bucket_specs, *token_specs],
                    token_specs=token_specs,
                    reserved_tokens=reserved_tokens,
                    metadata=metadata,
                    queue_request=queue_request,
                )
                if first_attempt:
                    ADMISSION_METRICS.fast_path.labels(
                        stage=stage,
                        outcome="allowed" if decision.allowed else decision.reason,
                    ).inc()
                    first_attempt = False
                if decision.allowed:
                    wait_seconds = max(0.0, self._time_func() - queue_started)
                    ADMISSION_METRICS.queue_wait.labels(stage=stage).observe(
                        wait_seconds
                    )
                    return decision

                last_denial = decision
                if external_queue:
                    return decision
                if decision.reason == "queue_required" or (
                    not queued and self._is_waitable_denial(decision.reason)
                ):
                    if queue_request is None or timeout_ms <= 0:
                        return decision
                    queued = True
                    queue_request = QueueAdmissionRequest(
                        **{
                            **queue_request.__dict__,
                            "enqueue": True,
                        }
                    )
                    continue
                if decision.reason == "queue_wait":
                    queued = True
                elif not self._is_waitable_denial(decision.reason):
                    return decision
                now_ms = self._now_ms()
                if timeout_ms <= 0 or now_ms >= deadline:
                    return decision
                remaining_ms = max(0, deadline - now_ms)
                jitter = 1.0
                if jitter_ratio:
                    digest = int(
                        hashlib.sha256(str(now_ms).encode()).hexdigest()[:8], 16
                    )
                    jitter += (((digest / 0xFFFFFFFF) * 2.0) - 1.0) * jitter_ratio
                sleep_ms = min(remaining_ms, max(1, int(poll_interval_ms * jitter)))
                ADMISSION_METRICS.queue_polls.labels(stage=stage).inc()
                await self._sleep_func(sleep_ms / 1000.0)
                poll_interval_ms = min(max_poll_interval_ms, poll_interval_ms * 2)

            return last_denial or self._deny_rate_limited(stage, "admission_rejected")
        finally:
            if queued and queue_request is not None:
                await self._cleanup_queue(queue_request)

    async def _try_admit(
        self,
        *,
        stage: str,
        active_specs: List[ActiveSpec],
        bucket_specs: List[BucketSpec],
        token_specs: List[BucketSpec],
        reserved_tokens: int,
        metadata: Dict[str, Any],
        queue_request: Optional[QueueAdmissionRequest] = None,
    ) -> AdmissionDecision:
        reservation_id = uuid.uuid4().hex
        keys = [self._reservation_key(reservation_id)]
        keys.extend(spec.key for spec in active_specs)
        keys.extend(spec.key for spec in bucket_specs)

        if queue_request is not None:
            keys.extend(
                [
                    queue_request.order_key,
                    queue_request.expiry_key,
                    queue_request.depth_key,
                    queue_request.sequence_key,
                ]
            )

        args: List[str] = [
            "1" if queue_request is not None else "0",
            queue_request.member_id if queue_request else "",
            str(queue_request.priority_band if queue_request else 0),
            str(queue_request.expiry_timestamp_ms if queue_request else 0),
            "1" if queue_request and queue_request.enqueue else "0",
            str(queue_request.max_depth if queue_request else 0),
            str(self.reservation_ttl_ms),
            str(self.bucket_ttl_ms),
            str(self._now_ms()),
            str(len(active_specs)),
        ]
        args.extend(str(spec.limit) for spec in active_specs)
        args.append(str(len(bucket_specs)))
        for spec in bucket_specs:
            args.extend(
                [
                    str(spec.capacity),
                    str(spec.refill_per_ms),
                    str(spec.cost),
                    spec.reason,
                ]
            )

        try:
            result = await self._eval(ADMIT_SCRIPT, keys, args)
        except Exception as exc:
            ADMISSION_METRICS.redis_errors.labels(operation="admit").inc()
            logger.warning("Admission admit failed: %s", exc)
            self._mark_redis_failure("admit", exc)
            unavailable = self._unavailable_decision(stage)
            if unavailable:
                return unavailable
            return AdmissionDecision.allow(None, reason="fail_open", metadata=metadata)

        allowed = bool(int(self._result_value(result, 0, 0)))
        reason = str(self._decode(self._result_value(result, 1, "unknown")))
        retry_ms = int(float(self._result_value(result, 2, 0) or 0))
        denied_key = str(self._decode(self._result_value(result, 3, "")))
        queue_depth = int(float(self._result_value(result, 4, 0) or 0))
        pruned_count = int(float(self._result_value(result, 5, 0) or 0))
        queue_score = float(self._result_value(result, 6, 0) or 0)
        if queue_request is not None:
            ADMISSION_METRICS.queue_depth.labels(scope=queue_request.scope).set(
                queue_depth
            )
            ADMISSION_METRICS.queue_pruned.labels(scope=queue_request.scope).inc(
                pruned_count
            )

        if not allowed:
            retry_after_seconds = max(1, math.ceil(retry_ms / 1000.0))
            if reason in {"queue_required", "queue_wait"}:
                return AdmissionDecision.reject(
                    status_code=429,
                    error="rate_limited",
                    reason=reason,
                    message="Admission queue wait required",
                    retry_after_seconds=retry_after_seconds,
                    metadata={
                        **metadata,
                        "queue_depth": queue_depth,
                        "queue_pruned": pruned_count,
                        "queue_score": queue_score,
                    },
                )
            denial_reason = self._normalize_denial_reason(reason, denied_key)
            decision = self._deny_rate_limited(
                stage,
                denial_reason,
                retry_after_seconds=retry_after_seconds,
                metadata={
                    **metadata,
                    "redis_key": denied_key,
                    "queue_depth": queue_depth,
                    "queue_pruned": pruned_count,
                    "queue_score": queue_score,
                },
            )
            return decision

        reservation = AdmissionReservation(
            reservation_id=reservation_id,
            stage=stage,
            active_specs=active_specs,
            token_specs=token_specs,
            reserved_tokens=reserved_tokens,
            metadata=metadata,
        )
        self._record_decision(stage, "allowed", "allowed")
        for active_spec in active_specs:
            ADMISSION_METRICS.active_reservations.labels(
                stage=stage,
                scope=active_spec.scope,
            ).inc()
        if reserved_tokens > 0:
            ADMISSION_METRICS.tokens.labels(
                stage=stage,
                token_type="reserved",
            ).inc(reserved_tokens)
        return AdmissionDecision.allow(
            reservation,
            metadata={
                **metadata,
                "queue_depth": queue_depth,
                "queue_pruned": pruned_count,
                "queue_score": queue_score,
            },
        )

    async def _cleanup_queue(self, request: QueueAdmissionRequest) -> int:
        try:
            result = await self._eval(
                QUEUE_CLEANUP_SCRIPT,
                [
                    request.order_key,
                    request.expiry_key,
                    request.depth_key,
                    request.sequence_key,
                ],
                [request.member_id],
            )
            depth = int(float(self._result_value(result, 1, 0) or 0))
            ADMISSION_METRICS.queue_depth.labels(scope=request.scope).set(depth)
            return depth
        except Exception as exc:
            ADMISSION_METRICS.redis_errors.labels(operation="queue_cleanup").inc()
            self._mark_redis_failure("queue_cleanup", exc)
            return int(0)

    def _queue_expiry_timestamp_ms(self, started_ms: int) -> int:
        timeout_ms = int(self.queue_config.get("timeout_ms", 0) or 0)
        grace_value = self.queue_config.get("queue_lease_grace_ms", 1000)
        grace_ms = max(0, int(1000 if grace_value is None else grace_value))
        return started_ms + timeout_ms + grace_ms

    @staticmethod
    def _is_waitable_denial(reason: str) -> bool:
        return reason.endswith("_active_requests_exceeded") or reason in {
            "active_limit",
            "queue_wait",
            "queue_required",
        }

    async def _recover_if_due(self) -> bool:
        if self.redis_client is None or self._redis_available:
            return self._redis_available
        cooldown_seconds = (
            max(0, int(self.redis_config.get("recovery_cooldown_ms", 1000))) / 1000.0
        )
        if (
            self._redis_failure_at is not None
            and self._time_func() - self._redis_failure_at < cooldown_seconds
        ):
            return False

        async with self._recovery_lock:
            if self._redis_available:
                return True
            if (
                self._redis_failure_at is not None
                and self._time_func() - self._redis_failure_at < cooldown_seconds
            ):
                return False
            ADMISSION_METRICS.redis_recovery.labels(outcome="attempt").inc()
            self._set_redis_state("recovering")
            try:
                await self._run_redis_operation(self.redis_client.ping())
            except Exception as exc:
                ADMISSION_METRICS.redis_recovery.labels(outcome="failure").inc()
                self._mark_redis_failure("recovery_ping", exc)
                return False
            ADMISSION_METRICS.redis_recovery.labels(outcome="success").inc()
            self._mark_redis_available()
            return True

    def _mark_redis_available(self):
        self._redis_available = True
        self._redis_failure_at = None
        self._set_redis_state("available")

    def _mark_redis_failure(self, operation: str, exc: Exception):
        self._redis_available = False
        self._redis_failure_at = self._time_func()
        self._set_redis_state("unavailable")
        if isinstance(exc, MaxConnectionsError):
            ADMISSION_METRICS.redis_pool_exhaustion.labels(
                operation=operation,
                failure_mode=self.failure_mode,
            ).inc()

    @staticmethod
    def _set_redis_state(state: str):
        for candidate in ("available", "unavailable", "recovering"):
            ADMISSION_METRICS.redis_state.labels(state=candidate).set(
                1 if candidate == state else 0
            )

    def _unavailable_decision(self, stage: str) -> Optional[AdmissionDecision]:
        if self.redis_client is not None and self._redis_available:
            return None
        if self.failure_mode == "open":
            self._record_decision(stage, "allowed", "redis_unavailable_fail_open")
            return None
        self._record_decision(stage, "denied", "redis_unavailable")
        return AdmissionDecision.reject(
            status_code=503,
            error="admission_unavailable",
            reason="redis_unavailable",
            message="API admission control is enabled but Redis is unavailable",
            retry_after_seconds=1,
        )

    def _deny_rate_limited(
        self,
        stage: str,
        reason: str,
        *,
        retry_after_seconds: Optional[int] = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdmissionDecision:
        self._record_decision(stage, "denied", reason)
        return AdmissionDecision.reject(
            status_code=429,
            error="rate_limited",
            reason=reason,
            message="Request rejected by API admission control",
            retry_after_seconds=retry_after_seconds,
            metadata=metadata or {},
        )

    def _record_decision(self, stage: str, outcome: str, reason: str):
        ADMISSION_METRICS.decisions.labels(
            stage=stage,
            outcome=outcome,
            reason=reason,
        ).inc()

    async def _eval(
        self, script: str, keys: Sequence[str], args: Sequence[str]
    ) -> Sequence[Any]:
        if self.redis_client is None:
            raise RuntimeError("Redis client is not initialized")
        result = await self._run_redis_operation(
            self.redis_client.eval(script, len(keys), *keys, *args)
        )
        self._mark_redis_available()
        return result

    async def _run_redis_operation(self, awaitable: Any) -> Any:
        # redis-py applies socket_connect_timeout and socket_timeout at the
        # transport boundary. A second timeout around the high-level command
        # would also count connection-pool/client/server scheduling time and
        # falsely fail healthy Redis under high concurrency. Keep the wrapper
        # only for injected clients that do not expose a configured pool.
        if getattr(self.redis_client, "connection_pool", None) is not None:
            return await awaitable
        timeout_seconds = max(
            0.001,
            int(self.redis_config.get("socket_timeout_ms", 100)) / 1000.0,
        )
        async with asyncio.timeout(timeout_seconds):
            return await awaitable

    def _global_active_specs(self) -> List[ActiveSpec]:
        global_limits = dict(self.config.get("global_limits", {}) or {})
        limit = int(global_limits.get("active_requests") or 0)
        if limit <= 0:
            return []
        return [
            ActiveSpec(
                key=self._key("active", "global"),
                limit=limit,
                reason="global_active_requests_exceeded",
                scope="global",
            )
        ]

    def _model_provider_active_specs(
        self, *, model_name: str, provider: str
    ) -> List[ActiveSpec]:
        specs: List[ActiveSpec] = []
        provider_limit = self._scope_config("providers", provider).get(
            "active_requests"
        )
        model_limit = self._scope_config("models", model_name).get("active_requests")
        if provider_limit:
            specs.append(
                ActiveSpec(
                    key=self._key("active", "provider", provider),
                    limit=int(provider_limit),
                    reason="provider_active_requests_exceeded",
                    scope=f"provider:{provider}",
                )
            )
        if model_limit:
            specs.append(
                ActiveSpec(
                    key=self._key("active", "model", model_name),
                    limit=int(model_limit),
                    reason="model_active_requests_exceeded",
                    scope=f"model:{model_name}",
                )
            )
        return specs

    def _global_request_bucket_specs(self) -> List[BucketSpec]:
        global_limits = dict(self.config.get("global_limits", {}) or {})
        requests_per_minute = int(
            global_limits.get("requests_per_minute")
            or self.config.get("requests_per_minute")
            or 0
        )
        burst_size = int(
            global_limits.get("burst_size") or self.config.get("burst_size") or 0
        )
        spec = self._request_bucket_spec(
            key=self._key("bucket", "requests", "global"),
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            reason="global_request_rate_exceeded",
            scope="global",
        )
        return [spec] if spec else []

    def _per_identity_request_bucket_specs(self, identifier: str) -> List[BucketSpec]:
        per_user = dict(self.config.get("per_user", {}) or {})
        return self._request_scope_specs(
            config=per_user,
            key=self._key("bucket", "requests", "identity", self._hash_id(identifier)),
            reason="identity_request_rate_exceeded",
            scope="identity",
        )

    def _user_and_tier_request_bucket_specs(
        self, *, user_id: str, user_tier: str
    ) -> List[BucketSpec]:
        specs: List[BucketSpec] = []
        per_user = dict(self.config.get("per_user", {}) or {})
        specs.extend(
            self._request_scope_specs(
                config=per_user,
                key=self._key("bucket", "requests", "user", self._hash_id(user_id)),
                reason="user_request_rate_exceeded",
                scope="user",
            )
        )
        tier_config = self._tier_config(user_tier)
        specs.extend(
            self._request_scope_specs(
                config=tier_config,
                key=self._key(
                    "bucket", "requests", "tier", self._scope_value(user_tier)
                ),
                reason="tier_request_rate_exceeded",
                scope=f"tier:{self._scope_value(user_tier)}",
            )
        )
        return specs

    def _model_provider_request_bucket_specs(
        self, *, model_name: str, provider: str
    ) -> List[BucketSpec]:
        specs: List[BucketSpec] = []
        specs.extend(
            self._request_scope_specs(
                config=self._scope_config("providers", provider),
                key=self._key("bucket", "requests", "provider", provider),
                reason="provider_request_rate_exceeded",
                scope=f"provider:{provider}",
            )
        )
        specs.extend(
            self._request_scope_specs(
                config=self._scope_config("models", model_name),
                key=self._key("bucket", "requests", "model", model_name),
                reason="model_request_rate_exceeded",
                scope=f"model:{model_name}",
            )
        )
        return specs

    def _request_scope_specs(
        self, *, config: Dict[str, Any], key: str, reason: str, scope: str
    ) -> List[BucketSpec]:
        requests_per_minute = config.get("requests_per_minute")
        burst_size = config.get("burst_size")
        spec = self._request_bucket_spec(
            key=key,
            requests_per_minute=int(requests_per_minute or 0),
            burst_size=int(burst_size or 0),
            reason=reason,
            scope=scope,
        )
        return [spec] if spec else []

    def _request_bucket_spec(
        self,
        *,
        key: str,
        requests_per_minute: int,
        burst_size: int,
        reason: str,
        scope: str,
    ) -> Optional[BucketSpec]:
        if requests_per_minute <= 0 or burst_size <= 0:
            return None
        return BucketSpec(
            key=key,
            capacity=float(burst_size),
            refill_per_ms=float(requests_per_minute) / 60_000.0,
            cost=1.0,
            reason=reason,
            scope=scope,
        )

    def _token_bucket_specs(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        reserved_tokens: int,
    ) -> List[BucketSpec]:
        token_budget = dict(self.config.get("token_budget", {}) or {})
        if not bool(token_budget.get("enabled", False)) or reserved_tokens <= 0:
            return []

        specs: List[BucketSpec] = []
        global_token_config = dict(self.config.get("global_limits", {}) or {})
        for key in ("tokens_per_minute", "burst_tokens"):
            if token_budget.get(key):
                global_token_config[key] = token_budget[key]
        global_spec = self._token_scope_spec(
            config=global_token_config,
            key=self._key("bucket", "tokens", "global"),
            reason="global_token_budget_exceeded",
            scope="global",
            reserved_tokens=reserved_tokens,
        )
        if global_spec:
            specs.append(global_spec)

        user_tier = self._tier_value(request.user_tier)
        tier_spec = self._token_scope_spec(
            config=self._tier_config(user_tier),
            key=self._key("bucket", "tokens", "tier", self._scope_value(user_tier)),
            reason="tier_token_budget_exceeded",
            scope=f"tier:{self._scope_value(user_tier)}",
            reserved_tokens=reserved_tokens,
        )
        if tier_spec:
            specs.append(tier_spec)

        user_spec = self._token_scope_spec(
            config=dict(self.config.get("per_user", {}) or {}),
            key=self._key(
                "bucket", "tokens", "user", self._hash_id(str(request.user_id))
            ),
            reason="user_token_budget_exceeded",
            scope="user",
            reserved_tokens=reserved_tokens,
        )
        if user_spec:
            specs.append(user_spec)

        provider_spec = self._token_scope_spec(
            config=self._scope_config("providers", provider),
            key=self._key("bucket", "tokens", "provider", provider),
            reason="provider_token_budget_exceeded",
            scope=f"provider:{provider}",
            reserved_tokens=reserved_tokens,
        )
        if provider_spec:
            specs.append(provider_spec)

        model_spec = self._token_scope_spec(
            config=self._scope_config("models", model_name),
            key=self._key("bucket", "tokens", "model", model_name),
            reason="model_token_budget_exceeded",
            scope=f"model:{model_name}",
            reserved_tokens=reserved_tokens,
        )
        if model_spec:
            specs.append(model_spec)
        return specs

    def _token_scope_spec(
        self,
        *,
        config: Dict[str, Any],
        key: str,
        reason: str,
        scope: str,
        reserved_tokens: int,
    ) -> Optional[BucketSpec]:
        tokens_per_minute = int(config.get("tokens_per_minute") or 0)
        burst_tokens = int(config.get("burst_tokens") or 0)
        if tokens_per_minute <= 0 or burst_tokens <= 0:
            return None
        return BucketSpec(
            key=key,
            capacity=float(burst_tokens),
            refill_per_ms=float(tokens_per_minute) / 60_000.0,
            cost=float(reserved_tokens),
            reason=reason,
            scope=scope,
        )

    def _scope_config(self, section: str, name: str) -> Dict[str, Any]:
        scoped = dict(self.config.get(section, {}) or {})
        return dict(
            scoped.get(name, {}) or scoped.get(self._scope_value(name), {}) or {}
        )

    def _tier_config(self, tier: str) -> Dict[str, Any]:
        by_tier = dict(self.config.get("by_tier", {}) or {})
        normalized = self._scope_value(tier)
        return dict(by_tier.get(normalized, {}) or by_tier.get(tier, {}) or {})

    def _normalize_denial_reason(self, reason: str, redis_key: str) -> str:
        if reason != "active_limit":
            return reason
        if ":provider:" in redis_key:
            return "provider_active_requests_exceeded"
        if ":model:" in redis_key:
            return "model_active_requests_exceeded"
        return "global_active_requests_exceeded"

    def _reservation_key(self, reservation_id: str) -> str:
        return self._key("reservation", reservation_id)

    def _key(self, *parts: str) -> str:
        normalized_parts = [self._scope_value(str(part)) for part in parts]
        return ":".join([self.key_prefix, *normalized_parts])

    def _hash_id(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    def _scope_value(self, value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"[^a-z0-9_.-]+", "_", normalized)
        return normalized.strip("_") or "unknown"

    def _tier_value(self, tier: Any) -> str:
        return str(getattr(tier, "value", tier) or "free")

    def _now_ms(self) -> int:
        return int(self._time_func() * 1000)

    def _result_value(self, result: Sequence[Any], index: int, default: Any) -> Any:
        try:
            return result[index]
        except (IndexError, TypeError):
            return default

    def _decode(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value
