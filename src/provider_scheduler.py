"""Provider/model capacity scheduling for inference-time provider calls."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from src.admission import AdmissionDecision, AdmissionRejectedError
from src.utils.metrics import INFERENCE_METRICS

logger = logging.getLogger(__name__)


SCHEDULER_FALLBACK_REASONS = {
    "provider_active_requests_exceeded",
    "model_active_requests_exceeded",
    "provider_token_budget_exceeded",
    "model_token_budget_exceeded",
    "circuit_open",
}

SCHEDULER_WAITABLE_REASONS = {
    "global_active_requests_exceeded",
    "provider_active_requests_exceeded",
    "model_active_requests_exceeded",
    "global_request_rate_exceeded",
    "provider_request_rate_exceeded",
    "model_request_rate_exceeded",
    "global_token_budget_exceeded",
    "provider_token_budget_exceeded",
    "model_token_budget_exceeded",
}


@dataclass
class SchedulerLease:
    """Capacity lease held while one provider call is in flight."""

    provider: str
    model_name: str
    reservation: Optional[Any] = None
    circuit_probe_key: Optional[str] = None
    queued: bool = False
    queue_key: Optional[str] = None
    queue_member: Optional[str] = None
    acquired_at: float = field(default_factory=time.time)
    wait_seconds: float = 0.0


class ProviderCapacityScheduler:
    """Coordinate provider capacity, wait ordering, retries, and circuit state."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        admission_controller: Optional[Any] = None,
        *,
        time_func: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], Awaitable[Any]]] = None,
        redis_client: Optional[Any] = None,
    ):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.queue_enabled = bool(self.config.get("queue_enabled", False))
        self.wait_timeout_ms = int(self.config.get("wait_timeout_ms", 250) or 0)
        self.poll_interval_ms = max(
            1, int(self.config.get("poll_interval_ms", 25) or 25)
        )
        self.failure_mode = str(self.config.get("failure_mode", "closed")).lower()
        if self.failure_mode not in {"open", "closed"}:
            self.failure_mode = "closed"
        self.allow_fallback_on_provider_rejection = bool(
            self.config.get("allow_fallback_on_provider_rejection", True)
        )
        self.key_prefix = str(self.config.get("key_prefix", "provider_scheduler"))
        self.admission_controller = admission_controller
        self.redis_client = redis_client
        self._time_func = time_func or time.time
        self._sleep_func = sleep_func or asyncio.sleep
        self.retry_config = dict(self.config.get("retry", {}) or {})
        self.circuit_config = dict(self.config.get("circuit_breaker", {}) or {})
        self._local_circuit_state: Dict[str, Dict[str, Any]] = {}
        self._local_circuit_probes: Dict[str, tuple[int, float]] = {}
        self._local_retry_budget: Dict[str, tuple[int, float]] = {}
        self._last_queue_depths: Dict[tuple[str, str], int] = {}
        self._last_circuit_states: Dict[tuple[str, str], str] = {}
        self._last_rejection: Optional[Dict[str, Any]] = None

    async def run_provider_call(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run a non-streaming provider call under scheduler control."""
        if not self.enabled:
            self._record_decision(provider, model_name, "allowed", "disabled")
            return await operation()

        attempts = max(
            1, int(self.retry_config.get("max_attempts_per_request", 1) or 1)
        )
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < attempts:
            attempt += 1
            lease = await self.acquire(
                request=request,
                model_name=model_name,
                provider=provider,
                estimated_input_tokens=estimated_input_tokens,
            )
            response = None
            try:
                response = await operation()
                await self.release(
                    lease,
                    actual_tokens=getattr(response, "total_tokens", None),
                )
                await self.record_success(provider=provider, model_name=model_name)
                if attempt > 1:
                    self._record_retry_metric(provider, model_name, "success")
                return response
            except Exception as exc:
                last_exc = exc
                await self.release(lease, actual_tokens=None)
                await self.record_failure(
                    provider=provider, model_name=model_name, error=exc
                )
                if attempt >= attempts or not self._is_retryable_exception(exc):
                    raise
                if not await self._consume_retry_budget(provider, model_name):
                    self._record_retry_metric(provider, model_name, "budget_exhausted")
                    raise
                self._record_retry_metric(provider, model_name, "retry")
                await self._sleep_func(self._retry_backoff_seconds(attempt))

        if last_exc:
            raise last_exc
        raise RuntimeError(
            "provider scheduler exited without running the provider call"
        )

    async def acquire(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
    ) -> SchedulerLease:
        """Acquire scheduler/provider capacity before provider I/O starts."""
        if not self.enabled or self.admission_controller is None:
            self._record_decision(provider, model_name, "allowed", "disabled")
            return SchedulerLease(provider=provider, model_name=model_name)

        circuit_probe_key = await self._acquire_circuit_probe_or_reject(
            provider, model_name
        )

        try:
            if self.queue_enabled:
                lease = await self._acquire_with_queue(
                    request=request,
                    model_name=model_name,
                    provider=provider,
                    estimated_input_tokens=estimated_input_tokens,
                )
            else:
                lease = await self._try_acquire(
                    request=request,
                    model_name=model_name,
                    provider=provider,
                    estimated_input_tokens=estimated_input_tokens,
                    queued=False,
                    queue_key=None,
                    queue_member=None,
                    wait_seconds=0.0,
                )
        except Exception:
            await self._release_circuit_probe(circuit_probe_key)
            raise

        lease.circuit_probe_key = circuit_probe_key
        return lease

    async def release(
        self, lease: Optional[SchedulerLease], *, actual_tokens: Optional[int] = None
    ) -> bool:
        if not lease:
            return False

        released = False
        if lease.reservation and self.admission_controller is not None:
            release_method = getattr(self.admission_controller, "release", None)
            if release_method:
                try:
                    released = bool(
                        await release_method(
                            lease.reservation, actual_tokens=actual_tokens
                        )
                    )
                except Exception:
                    logger.warning(
                        "Provider scheduler failed to release admission reservation",
                        exc_info=True,
                    )

        await self._release_circuit_probe(lease.circuit_probe_key)
        return released

    def allow_fallback_for_rejection(
        self, decision: Optional[AdmissionDecision]
    ) -> bool:
        if not self.allow_fallback_on_provider_rejection or decision is None:
            return False
        if str(decision.reason) == "scheduler_queue_timeout":
            return (
                str((decision.metadata or {}).get("last_reason"))
                in SCHEDULER_FALLBACK_REASONS
            )
        return str(decision.reason) in SCHEDULER_FALLBACK_REASONS

    async def record_success(self, *, provider: str, model_name: str) -> None:
        if not bool(self.circuit_config.get("enabled", False)):
            return
        await self._write_circuit_state(
            provider,
            model_name,
            {
                "state": "closed",
                "failure_count": 0,
                "opened_until": 0.0,
                "updated_at": self._time_func(),
            },
        )
        self._set_circuit_metric(provider, model_name, "closed")

    async def record_failure(
        self, *, provider: str, model_name: str, error: Exception
    ) -> None:
        if not bool(self.circuit_config.get("enabled", False)):
            return
        if not self._is_retryable_exception(error):
            return

        state = await self._read_circuit_state(provider, model_name)
        failure_count = int(state.get("failure_count", 0) or 0) + 1
        threshold = max(1, int(self.circuit_config.get("failure_threshold", 5) or 5))
        next_state = "closed"
        opened_until = 0.0
        if state.get("state") == "half_open" or failure_count >= threshold:
            next_state = "open"
            opened_until = self._time_func() + (
                max(0, int(self.circuit_config.get("recovery_timeout_ms", 30_000) or 0))
                / 1000.0
            )
        await self._write_circuit_state(
            provider,
            model_name,
            {
                "state": next_state,
                "failure_count": failure_count,
                "opened_until": opened_until,
                "last_error": str(error),
                "updated_at": self._time_func(),
            },
        )
        self._set_circuit_metric(provider, model_name, next_state)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "queue_enabled": self.queue_enabled,
            "wait_timeout_ms": self.wait_timeout_ms,
            "poll_interval_ms": self.poll_interval_ms,
            "failure_mode": self.failure_mode,
            "last_rejection": dict(self._last_rejection)
            if self._last_rejection
            else None,
            "circuit_breaker_enabled": bool(self.circuit_config.get("enabled", False)),
            "retry_budget_enabled": bool(
                self.retry_config.get("budget_enabled", False)
            ),
            "queue_depths": [
                {"provider": provider, "model": model_name, "depth": depth}
                for (provider, model_name), depth in sorted(
                    self._last_queue_depths.items()
                )
            ],
            "circuit_states": [
                {"provider": provider, "model": model_name, "state": state}
                for (provider, model_name), state in sorted(
                    self._last_circuit_states.items()
                )
            ],
        }

    async def _acquire_with_queue(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
    ) -> SchedulerLease:
        client = self._redis_client()
        if client is None:
            if self.failure_mode == "open":
                self._record_decision(
                    provider, model_name, "allowed", "queue_unavailable"
                )
                return await self._try_acquire(
                    request=request,
                    model_name=model_name,
                    provider=provider,
                    estimated_input_tokens=estimated_input_tokens,
                    queued=False,
                    queue_key=None,
                    queue_member=None,
                    wait_seconds=0.0,
                )
            decision = self._reject(
                provider=provider,
                model_name=model_name,
                reason="scheduler_queue_unavailable",
                status_code=503,
                retry_after_seconds=1,
            )
            raise AdmissionRejectedError(decision)

        queue_key = self._queue_key(provider, model_name)
        queue_member = self._queue_member(request)
        started_at = self._time_func()
        await self._queue_add(
            client, queue_key, queue_member, self._queue_score(request)
        )
        await self._set_queue_depth(provider, model_name, queue_key)

        last_decision: Optional[AdmissionDecision] = None
        try:
            deadline = started_at + (max(0, self.wait_timeout_ms) / 1000.0)
            while True:
                now = self._time_func()
                if last_decision is not None and (
                    self.wait_timeout_ms <= 0 or now >= deadline
                ):
                    decision = self._timeout_decision(
                        provider=provider,
                        model_name=model_name,
                        last_decision=last_decision,
                    )
                    self._record_rejection(provider, model_name, decision)
                    raise AdmissionRejectedError(decision)
                if await self._queue_is_head(client, queue_key, queue_member):
                    try:
                        return await self._try_acquire(
                            request=request,
                            model_name=model_name,
                            provider=provider,
                            estimated_input_tokens=estimated_input_tokens,
                            queued=True,
                            queue_key=queue_key,
                            queue_member=queue_member,
                            wait_seconds=max(0.0, self._time_func() - started_at),
                        )
                    except AdmissionRejectedError as exc:
                        last_decision = exc.decision
                        if not self._is_waitable_reason(exc.decision.reason):
                            raise

                now = self._time_func()
                if self.wait_timeout_ms <= 0 or now >= deadline:
                    decision = self._timeout_decision(
                        provider=provider,
                        model_name=model_name,
                        last_decision=last_decision,
                    )
                    self._record_rejection(provider, model_name, decision)
                    raise AdmissionRejectedError(decision)
                sleep_for = min(
                    self.poll_interval_ms / 1000.0, max(deadline - now, 0.0)
                )
                await self._sleep_func(sleep_for)
        finally:
            await self._queue_remove(client, queue_key, queue_member)
            await self._set_queue_depth(provider, model_name, queue_key)

    async def _try_acquire(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
        queued: bool,
        queue_key: Optional[str],
        queue_member: Optional[str],
        wait_seconds: float,
    ) -> SchedulerLease:
        acquire_provider = getattr(self.admission_controller, "acquire_provider", None)
        if acquire_provider is None:
            self._record_decision(provider, model_name, "allowed", "admission_missing")
            return SchedulerLease(
                provider=provider,
                model_name=model_name,
                queued=queued,
                queue_key=queue_key,
                queue_member=queue_member,
                wait_seconds=wait_seconds,
            )
        decision = await acquire_provider(
            request=request,
            model_name=model_name,
            provider=provider,
            estimated_input_tokens=estimated_input_tokens,
        )
        if not decision.allowed:
            self._record_rejection(provider, model_name, decision)
            raise AdmissionRejectedError(decision)
        reason = "queued" if queued else "allowed"
        self._record_decision(provider, model_name, "allowed", reason)
        if queued:
            INFERENCE_METRICS.scheduler_queue_wait.labels(
                provider=provider,
                model=model_name,
            ).observe(wait_seconds)
        return SchedulerLease(
            provider=provider,
            model_name=model_name,
            reservation=decision.reservation,
            queued=queued,
            queue_key=queue_key,
            queue_member=queue_member,
            wait_seconds=wait_seconds,
        )

    def _queue_score(self, request: Any) -> float:
        metadata = getattr(request, "metadata", None) or {}
        latency_sla = str(metadata.get("latency_sla", "") or "").lower()
        explicit_fast = bool(
            metadata.get("requires_low_latency")
            or metadata.get("route_to_fast_lane")
            or metadata.get("fast_lane")
            or latency_sla in {"low", "interactive"}
        )
        latency_rank = 0 if explicit_fast else 1
        priority_rank = max(0, 5 - int(getattr(request, "priority", 1) or 1))
        tier_value = getattr(getattr(request, "user_tier", "free"), "value", None)
        tier = str(tier_value or getattr(request, "user_tier", "free")).lower()
        tier_rank = {"enterprise": 0, "premium": 1, "free": 2}.get(tier, 2)
        now_ms = int(self._time_func() * 1000)
        return (
            latency_rank * 1_000_000_000_000
            + priority_rank * 10_000_000_000
            + tier_rank * 100_000_000
            + now_ms
        )

    def _queue_key(self, provider: str, model_name: str) -> str:
        return (
            f"{self.key_prefix}:queue:{self._scope(provider)}:{self._scope(model_name)}"
        )

    def _queue_member(self, request: Any) -> str:
        request_id = str(getattr(request, "request_id", "") or uuid.uuid4().hex)
        digest = hashlib.sha256(uuid.uuid4().hex.encode("utf-8")).hexdigest()[:12]
        return f"{request_id}:{digest}"

    async def _queue_add(
        self, client: Any, queue_key: str, queue_member: str, score: float
    ) -> None:
        await client.zadd(queue_key, {queue_member: score})

    async def _queue_remove(
        self, client: Any, queue_key: str, queue_member: str
    ) -> None:
        try:
            await client.zrem(queue_key, queue_member)
        except Exception as exc:
            logger.warning("Failed to remove provider scheduler queue member: %s", exc)

    async def _queue_is_head(
        self, client: Any, queue_key: str, queue_member: str
    ) -> bool:
        raw = await client.zrange(queue_key, 0, 0)
        if not raw:
            return False
        head = raw[0]
        if isinstance(head, bytes):
            head = head.decode("utf-8")
        return str(head) == queue_member

    async def _set_queue_depth(
        self, provider: str, model_name: str, queue_key: str
    ) -> None:
        client = self._redis_client()
        if client is None:
            return
        try:
            depth = int(await client.zcard(queue_key))
        except Exception:
            return
        self._last_queue_depths[(provider, model_name)] = depth
        INFERENCE_METRICS.scheduler_queue_depth.labels(
            provider=provider,
            model=model_name,
        ).set(depth)

    def _timeout_decision(
        self,
        *,
        provider: str,
        model_name: str,
        last_decision: Optional[AdmissionDecision],
    ) -> AdmissionDecision:
        retry_after = last_decision.retry_after_seconds if last_decision else 1
        return self._reject(
            provider=provider,
            model_name=model_name,
            reason="scheduler_queue_timeout",
            status_code=429,
            retry_after_seconds=retry_after or 1,
            metadata={
                "last_reason": last_decision.reason if last_decision else None,
                "provider": provider,
                "model": model_name,
            },
        )

    def _reject(
        self,
        *,
        provider: str,
        model_name: str,
        reason: str,
        status_code: int,
        retry_after_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdmissionDecision:
        return AdmissionDecision.reject(
            status_code=status_code,
            error="rate_limited" if status_code == 429 else "scheduler_unavailable",
            reason=reason,
            message="Request rejected by provider capacity scheduler",
            retry_after_seconds=retry_after_seconds,
            metadata={"provider": provider, "model": model_name, **(metadata or {})},
        )

    async def _acquire_circuit_probe_or_reject(
        self, provider: str, model_name: str
    ) -> Optional[str]:
        if not bool(self.circuit_config.get("enabled", False)):
            return None
        state = await self._read_circuit_state(provider, model_name)
        circuit_state = state.get("state")
        if circuit_state not in {"open", "half_open"}:
            return None
        opened_until = float(state.get("opened_until", 0.0) or 0.0)
        if circuit_state == "open" and opened_until <= self._time_func():
            await self._write_circuit_state(
                provider,
                model_name,
                {
                    **state,
                    "state": "half_open",
                    "updated_at": self._time_func(),
                },
            )
            self._set_circuit_metric(provider, model_name, "half_open")
            return await self._acquire_half_open_probe(provider, model_name)
        if circuit_state == "half_open":
            return await self._acquire_half_open_probe(provider, model_name)
        retry_after = max(1, math.ceil(opened_until - self._time_func()))
        decision = self._reject(
            provider=provider,
            model_name=model_name,
            reason="circuit_open",
            status_code=429,
            retry_after_seconds=retry_after,
            metadata={"opened_until": opened_until},
        )
        self._record_rejection(provider, model_name, decision)
        raise AdmissionRejectedError(decision)

    async def _acquire_half_open_probe(self, provider: str, model_name: str) -> str:
        max_probes = max(
            1, int(self.circuit_config.get("half_open_max_requests", 1) or 1)
        )
        key = self._circuit_probe_key(provider, model_name)
        ttl_seconds = max(
            1,
            math.ceil(
                max(1, int(self.circuit_config.get("recovery_timeout_ms", 30_000) or 1))
                / 1000.0
            ),
        )
        client = self._redis_client()
        if client is not None:
            try:
                current = int(await client.incr(key))
                if current == 1:
                    expire = getattr(client, "expire", None)
                    if expire:
                        result = expire(key, ttl_seconds)
                        if hasattr(result, "__await__"):
                            await result
                if current <= max_probes:
                    return key
                decr = getattr(client, "decr", None)
                if decr:
                    result = decr(key)
                    if hasattr(result, "__await__"):
                        await result
                decision = self._reject(
                    provider=provider,
                    model_name=model_name,
                    reason="circuit_open",
                    status_code=429,
                    retry_after_seconds=ttl_seconds,
                    metadata={"state": "half_open", "max_probe_requests": max_probes},
                )
                self._record_rejection(provider, model_name, decision)
                raise AdmissionRejectedError(decision)
            except AdmissionRejectedError:
                raise
            except Exception:
                logger.debug("Falling back to local circuit probes", exc_info=True)

        now = self._time_func()
        current, expires_at = self._local_circuit_probes.get(
            key, (0, now + ttl_seconds)
        )
        if expires_at <= now:
            current = 0
            expires_at = now + ttl_seconds
        if current >= max_probes:
            decision = self._reject(
                provider=provider,
                model_name=model_name,
                reason="circuit_open",
                status_code=429,
                retry_after_seconds=ttl_seconds,
                metadata={"state": "half_open", "max_probe_requests": max_probes},
            )
            self._record_rejection(provider, model_name, decision)
            raise AdmissionRejectedError(decision)
        self._local_circuit_probes[key] = (current + 1, expires_at)
        return key

    async def _release_circuit_probe(self, probe_key: Optional[str]) -> None:
        if not probe_key:
            return
        client = self._redis_client()
        if client is not None:
            try:
                decr = getattr(client, "decr", None)
                if decr:
                    result = decr(probe_key)
                    if hasattr(result, "__await__"):
                        current = await result
                    else:
                        current = result
                    if int(current or 0) <= 0:
                        delete = getattr(client, "delete", None)
                        if delete:
                            result = delete(probe_key)
                            if hasattr(result, "__await__"):
                                await result
                    return
            except Exception:
                logger.debug(
                    "Falling back to local circuit probe release", exc_info=True
                )

        current, expires_at = self._local_circuit_probes.get(probe_key, (0, 0.0))
        if current <= 1:
            self._local_circuit_probes.pop(probe_key, None)
        else:
            self._local_circuit_probes[probe_key] = (current - 1, expires_at)

    async def _read_circuit_state(
        self, provider: str, model_name: str
    ) -> Dict[str, Any]:
        key = self._circuit_key(provider, model_name)
        client = self._redis_client()
        if client is not None:
            try:
                raw = await client.hgetall(key)
                if raw:
                    return {
                        self._decode(k): self._coerce_state_value(v)
                        for k, v in raw.items()
                    }
            except Exception:
                logger.debug("Falling back to local circuit state", exc_info=True)
        return dict(self._local_circuit_state.get(key, {}))

    async def _write_circuit_state(
        self, provider: str, model_name: str, state: Dict[str, Any]
    ) -> None:
        key = self._circuit_key(provider, model_name)
        self._local_circuit_state[key] = dict(state)
        client = self._redis_client()
        if client is None:
            return
        try:
            await client.hset(key, mapping={k: str(v) for k, v in state.items()})
            ttl = int(self.circuit_config.get("state_ttl_seconds", 3600) or 3600)
            expire = getattr(client, "expire", None)
            if expire:
                result = expire(key, ttl)
                if hasattr(result, "__await__"):
                    await result
        except Exception:
            logger.debug("Failed to persist scheduler circuit state", exc_info=True)

    async def _consume_retry_budget(self, provider: str, model_name: str) -> bool:
        if not bool(self.retry_config.get("budget_enabled", False)):
            return True
        limit = int(self.retry_config.get("budget_tokens", 0) or 0)
        if limit <= 0:
            return False
        window_seconds = max(
            1, int(self.retry_config.get("budget_window_seconds", 60) or 60)
        )
        now = self._time_func()
        key = self._retry_budget_key(provider, model_name, now, window_seconds)
        client = self._redis_client()
        if client is not None:
            try:
                current = int(await client.incr(key))
                if current == 1:
                    expire = getattr(client, "expire", None)
                    if expire:
                        result = expire(key, window_seconds)
                        if hasattr(result, "__await__"):
                            await result
                return current <= limit
            except Exception:
                logger.debug("Falling back to local retry budget", exc_info=True)

        current, expires_at = self._local_retry_budget.get(
            key, (0, now + window_seconds)
        )
        if expires_at <= now:
            current = 0
            expires_at = now + window_seconds
        current += 1
        self._local_retry_budget[key] = (current, expires_at)
        return current <= limit

    def _retry_budget_key(
        self, provider: str, model_name: str, now: float, window_seconds: int
    ) -> str:
        window = int(now // window_seconds)
        return (
            f"{self.key_prefix}:retry_budget:{self._scope(provider)}:"
            f"{self._scope(model_name)}:{window}"
        )

    def _circuit_key(self, provider: str, model_name: str) -> str:
        return f"{self.key_prefix}:circuit:{self._scope(provider)}:{self._scope(model_name)}"

    def _circuit_probe_key(self, provider: str, model_name: str) -> str:
        return (
            f"{self.key_prefix}:circuit_probe:"
            f"{self._scope(provider)}:{self._scope(model_name)}"
        )

    def _redis_client(self) -> Optional[Any]:
        return self.redis_client or getattr(
            self.admission_controller, "redis_client", None
        )

    def _is_waitable_reason(self, reason: str) -> bool:
        return str(reason) in SCHEDULER_WAITABLE_REASONS

    def _is_retryable_exception(self, exc: Exception) -> bool:
        if isinstance(exc, AdmissionRejectedError):
            return False
        if isinstance(exc, (ValueError, KeyError, TypeError)):
            return False
        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response else 0
            return status in {408, 409, 425, 429, 500, 502, 503, 504}
        name = type(exc).__name__.lower()
        transient_names = (
            "ratelimit",
            "timeout",
            "connection",
            "serviceunavailable",
            "internalserver",
            "overload",
            "apierror",
        )
        if any(marker in name for marker in transient_names):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "timeout",
                "temporar",
                "rate limit",
                "429",
                "500",
                "502",
                "503",
                "504",
                "overload",
                "no healthy",
            )
        )

    def _retry_backoff_seconds(self, attempt: int) -> float:
        initial_ms = max(0, int(self.retry_config.get("initial_backoff_ms", 100) or 0))
        max_ms = max(
            initial_ms, int(self.retry_config.get("max_backoff_ms", 1000) or 1000)
        )
        return min(initial_ms * (2 ** max(attempt - 1, 0)), max_ms) / 1000.0

    def _record_rejection(
        self, provider: str, model_name: str, decision: AdmissionDecision
    ) -> None:
        self._last_rejection = {
            "provider": provider,
            "model": model_name,
            "reason": decision.reason,
            "status_code": decision.status_code,
            "timestamp": self._time_func(),
        }
        self._record_decision(provider, model_name, "denied", decision.reason)

    def _record_decision(
        self, provider: str, model_name: str, outcome: str, reason: str
    ) -> None:
        INFERENCE_METRICS.scheduler_decisions.labels(
            provider=provider,
            model=model_name,
            outcome=outcome,
            reason=reason,
        ).inc()

    def _record_retry_metric(
        self, provider: str, model_name: str, outcome: str
    ) -> None:
        INFERENCE_METRICS.scheduler_retries.labels(
            provider=provider,
            model=model_name,
            outcome=outcome,
        ).inc()

    def _set_circuit_metric(self, provider: str, model_name: str, state: str) -> None:
        self._last_circuit_states[(provider, model_name)] = state
        for candidate in ("closed", "open", "half_open"):
            INFERENCE_METRICS.scheduler_circuit_state.labels(
                provider=provider,
                model=model_name,
                state=candidate,
            ).set(1 if candidate == state else 0)

    def _scope(self, value: str) -> str:
        return str(value or "unknown").lower().replace(":", "_").replace("/", "_")

    def _decode(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _coerce_state_value(self, value: Any) -> Any:
        decoded = self._decode(value)
        if decoded in {"closed", "open", "half_open"}:
            return decoded
        try:
            if "." in decoded:
                return float(decoded)
            return int(decoded)
        except ValueError:
            return decoded
