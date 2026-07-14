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

from src.admission import (
    AdmissionDecision,
    AdmissionRejectedError,
    QueueAdmissionRequest,
)
from src.utils.metrics import INFERENCE_METRICS

logger = logging.getLogger(__name__)


_CIRCUIT_LUA = r"""
local key = KEYS[1]
local action = ARGV[1]
local now = tonumber(ARGV[2])
local threshold = tonumber(ARGV[3])
local recovery = tonumber(ARGV[4])
local max_probes = tonumber(ARGV[5])
local request_started = tonumber(ARGV[6])
local permit_epoch = tonumber(ARGV[7])
local permit_probe = tonumber(ARGV[8])
local ttl = tonumber(ARGV[9])

local state = redis.call('HGET', key, 'state') or 'closed'
local failures = tonumber(redis.call('HGET', key, 'failure_count') or '0')
local opened_until = tonumber(redis.call('HGET', key, 'opened_until_ms') or '0')
local epoch = tonumber(redis.call('HGET', key, 'epoch') or '0')
local probes = tonumber(redis.call('HGET', key, 'probe_count') or '0')
local last_failure = tonumber(redis.call('HGET', key, 'last_failure_at_ms') or '0')

if action == 'acquire' then
  if state == 'open' and opened_until <= now then
    state = 'half_open'
    probes = 0
  end
  if state == 'open' then
    return {0, epoch, 0, state, opened_until, failures, probes, last_failure}
  end
  if state == 'half_open' then
    if probes >= max_probes then
      return {0, epoch, 0, state, opened_until, failures, probes, last_failure}
    end
    probes = probes + 1
    redis.call('HSET', key, 'state', state, 'failure_count', failures,
      'opened_until_ms', opened_until, 'epoch', epoch, 'probe_count', probes,
      'last_failure_at_ms', last_failure, 'updated_at_ms', now)
    redis.call('EXPIRE', key, ttl)
    return {1, epoch, 1, state, opened_until, failures, probes, last_failure}
  end
  return {1, epoch, 0, state, opened_until, failures, probes, last_failure}
end

if action == 'success' then
  if permit_probe == 1 and permit_epoch ~= epoch then
    return {1, epoch, permit_probe, state, opened_until, failures, probes, last_failure}
  end
  if permit_probe == 1 then
    if state == 'half_open' and permit_epoch == epoch then
      state = 'closed'; failures = 0; opened_until = 0; probes = 0
    end
  elseif state == 'closed' and request_started >= last_failure then
    failures = 0
  end
elseif action == 'failure' then
  if permit_probe == 1 and permit_epoch ~= epoch then
    return {1, epoch, permit_probe, state, opened_until, failures, probes, last_failure}
  end
  if state ~= 'open' then
    failures = failures + 1
    last_failure = now
    if state == 'half_open' or failures >= threshold then
      state = 'open'; opened_until = now + recovery; probes = 0; epoch = epoch + 1
    end
  end
elseif action == 'release' then
  if permit_probe == 1 and state == 'half_open' and permit_epoch == epoch and probes > 0 then
    probes = probes - 1
  end
end

redis.call('HSET', key, 'state', state, 'failure_count', failures,
  'opened_until_ms', opened_until, 'epoch', epoch, 'probe_count', probes,
  'last_failure_at_ms', last_failure, 'updated_at_ms', now)
redis.call('EXPIRE', key, ttl)
return {1, epoch, permit_probe, state, opened_until, failures, probes, last_failure}
"""


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
class CircuitPermit:
    """Epoch-bound permission for a circuit-protected provider attempt."""

    epoch: int = 0
    probe: bool = False
    provider: str = ""
    model_name: str = ""


@dataclass
class SchedulerLease:
    """Capacity lease held while one provider call is in flight."""

    provider: str
    model_name: str
    reservation: Optional[Any] = None
    circuit_permit: Optional[CircuitPermit] = None
    queued: bool = False
    queue_key: Optional[str] = None
    queue_member: Optional[str] = None
    acquired_at: float = field(default_factory=time.time)
    wait_seconds: float = 0.0
    released: bool = False
    release_task: Optional[asyncio.Task] = None


@dataclass
class _QueuePollScope:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    candidates: Dict[str, tuple[float, int]] = field(default_factory=dict)
    events: Dict[str, asyncio.Event] = field(default_factory=dict)
    authoritative_members: set[str] = field(default_factory=set)
    sequence: int = 0
    active_member: Optional[str] = None
    next_poll_at: float = 0.0
    interval_ms: int = 0


class _QueuePollCoordinator:
    """Allow one Redis queue poll at a time for each local provider scope."""

    def __init__(
        self,
        *,
        base_interval_ms: int,
        max_interval_ms: int,
        jitter_ratio: float,
        time_func: Callable[[], float],
        sleep_func: Callable[[float], Awaitable[Any]],
    ):
        self._base_interval_ms = base_interval_ms
        self._max_interval_ms = max_interval_ms
        self._jitter_ratio = jitter_ratio
        self._time_func = time_func
        self._sleep_func = sleep_func
        self._scopes: Dict[str, _QueuePollScope] = {}

    @staticmethod
    def _wake_best(state: _QueuePollScope) -> None:
        if state.active_member is not None or not state.candidates:
            return
        best = min(state.candidates, key=lambda item: state.candidates[item])
        state.events[best].set()

    async def wait_turn(
        self,
        scope: str,
        member: str,
        priority: int,
        *,
        deadline: float,
        order_score: Optional[float] = None,
    ) -> None:
        state = self._scopes.setdefault(
            scope,
            _QueuePollScope(
                interval_ms=min(self._max_interval_ms, self._base_interval_ms * 2)
            ),
        )
        async with state.condition:
            if member not in state.candidates:
                state.sequence += 1
                score = (
                    float(order_score)
                    if order_score is not None
                    else float(priority * 1_000_000_000_000 + state.sequence)
                )
                state.candidates[member] = (score, state.sequence)
                state.events[member] = asyncio.Event()
                if order_score is not None:
                    state.authoritative_members.add(member)
                if len(state.candidates) == 1:
                    state.next_poll_at = self._time_func() + (
                        self._base_interval_ms / 1000.0
                    )
            event = state.events[member]
            self._wake_best(state)

        while True:
            await event.wait()
            event.clear()
            async with state.condition:
                if member not in state.candidates:
                    raise asyncio.CancelledError
                best = min(state.candidates, key=lambda item: state.candidates[item])
                if state.active_member is None and best == member:
                    state.active_member = member
                    delay = min(
                        max(0.0, state.next_poll_at - self._time_func()),
                        max(0.0, deadline - self._time_func()),
                    )
                else:
                    self._wake_best(state)
                    continue

            if delay:
                await self._sleep_func(delay)

            async with state.condition:
                best = min(state.candidates, key=lambda item: state.candidates[item])
                if state.active_member == member and best == member:
                    return
                if state.active_member == member:
                    state.active_member = None
                    self._wake_best(state)

    async def poll_denied(
        self, scope: str, member: str, *, blocked_by_queue_head: bool
    ) -> None:
        state = self._scopes[scope]
        async with state.condition:
            if state.active_member == member:
                state.active_member = None
            if (
                blocked_by_queue_head
                and member in state.candidates
                and member not in state.authoritative_members
            ):
                score, _ = state.candidates[member]
                state.sequence += 1
                priority_base = (int(score) // 1_000_000_000_000) * 1_000_000_000_000
                state.candidates[member] = (
                    float(priority_base + state.sequence),
                    state.sequence,
                )
                state.next_poll_at = self._time_func()
                state.interval_ms = self._base_interval_ms
                self._wake_best(state)
                return
            interval_ms = state.interval_ms
            jitter = 1.0
            if self._jitter_ratio:
                digest = int(
                    hashlib.sha256(str(self._time_func()).encode()).hexdigest()[:8],
                    16,
                )
                jitter += (((digest / 0xFFFFFFFF) * 2.0) - 1.0) * self._jitter_ratio
            state.next_poll_at = self._time_func() + (
                (max(1.0, interval_ms * jitter) / 1000.0) + 1e-12
            )
            state.interval_ms = min(self._max_interval_ms, interval_ms * 2)
            self._wake_best(state)

    async def remove(self, scope: str, member: str, *, wake: bool) -> None:
        state = self._scopes.get(scope)
        if state is None:
            return
        async with state.condition:
            state.candidates.pop(member, None)
            state.events.pop(member, None)
            state.authoritative_members.discard(member)
            if state.active_member == member:
                state.active_member = None
            if wake:
                state.next_poll_at = self._time_func()
                state.interval_ms = self._base_interval_ms
            self._wake_best(state)
            if not state.candidates and state.active_member is None:
                self._scopes.pop(scope, None)

    async def wake(self, scope: str) -> None:
        state = self._scopes.get(scope)
        if state is None:
            return
        async with state.condition:
            state.next_poll_at = self._time_func()
            state.interval_ms = self._base_interval_ms
            self._wake_best(state)


@dataclass
class RequestExecutionBudget:
    """One monotonic deadline and transport-attempt budget for a request."""

    max_attempts: int
    deadline_seconds: float = 60.0
    started_at: float = field(default_factory=time.monotonic)
    attempts_used: int = 0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - (time.monotonic() - self.started_at))

    def consume_attempt(self) -> bool:
        if self.remaining_seconds <= 0 or self.attempts_used >= self.max_attempts:
            return False
        self.attempts_used += 1
        return True

    def ensure_available(self) -> None:
        if self.remaining_seconds <= 0:
            raise asyncio.TimeoutError("request execution deadline exceeded")
        if self.attempts_used >= self.max_attempts:
            raise RuntimeError("request transport attempt budget exhausted")


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
        self.max_poll_interval_ms = max(
            self.poll_interval_ms,
            int(self.config.get("max_poll_interval_ms", 250) or 250),
        )
        self.poll_jitter_ratio = max(
            0.0, min(1.0, float(self.config.get("poll_jitter_ratio", 0.2) or 0.0))
        )
        self.queue_lease_grace_ms = max(
            0, int(self.config.get("queue_lease_grace_ms", 1000) or 0)
        )
        self.control_plane_version = str(
            self.config.get("control_plane_version", "v2") or "v2"
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
        self._circuit_lock = asyncio.Lock()
        self._local_retry_budget: Dict[str, tuple[int, float]] = {}
        self._last_queue_depths: Dict[tuple[str, str], int] = {}
        self._last_circuit_states: Dict[tuple[str, str], str] = {}
        self._last_rejection: Optional[Dict[str, Any]] = None
        self._queue_poll_coordinator = _QueuePollCoordinator(
            base_interval_ms=self.poll_interval_ms,
            max_interval_ms=self.max_poll_interval_ms,
            jitter_ratio=self.poll_jitter_ratio,
            time_func=self._time_func,
            sleep_func=self._sleep_func,
        )

    def create_execution_budget(self) -> RequestExecutionBudget:
        return RequestExecutionBudget(
            max_attempts=max(
                1, int(self.retry_config.get("max_attempts_per_request", 1) or 1)
            ),
            deadline_seconds=float(
                self.config.get("request_deadline_seconds", 60) or 60
            ),
        )

    async def run_provider_call(
        self,
        *,
        request: Any,
        model_name: str,
        provider: str,
        estimated_input_tokens: int,
        operation: Callable[[], Awaitable[Any]],
        execution_budget: Optional[RequestExecutionBudget] = None,
    ) -> Any:
        """Run a non-streaming provider call under scheduler control."""
        if not self.enabled:
            self._record_decision(provider, model_name, "allowed", "disabled")
            if execution_budget is None:
                return await operation()
            execution_budget.ensure_available()
            if not execution_budget.consume_attempt():
                execution_budget.ensure_available()
            async with asyncio.timeout(execution_budget.remaining_seconds):
                return await operation()

        attempts = max(
            1, int(self.retry_config.get("max_attempts_per_request", 1) or 1)
        )
        budget = execution_budget or RequestExecutionBudget(
            max_attempts=attempts,
            deadline_seconds=float(
                self.config.get("request_deadline_seconds", 60) or 60
            ),
        )
        last_exc: Optional[Exception] = None
        local_attempt = 0
        while local_attempt < attempts:
            budget.ensure_available()
            lease: Optional[SchedulerLease] = None
            response = None
            try:
                request_started_at_ms = int(self._time_func() * 1000)
                async with asyncio.timeout(budget.remaining_seconds):
                    lease = await self.acquire(
                        request=request,
                        model_name=model_name,
                        provider=provider,
                        estimated_input_tokens=estimated_input_tokens,
                    )
                    if not budget.consume_attempt():
                        budget.ensure_available()
                    local_attempt += 1
                    response = await operation()
                await self.record_success(
                    provider=provider,
                    model_name=model_name,
                    request_started_at_ms=request_started_at_ms,
                    circuit_permit=lease.circuit_permit if lease else None,
                )
                if local_attempt > 1:
                    self._record_retry_metric(provider, model_name, "success")
                return response
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if local_attempt > 0:
                    await self.record_failure(
                        provider=provider,
                        model_name=model_name,
                        error=exc,
                        request_started_at_ms=request_started_at_ms,
                        circuit_permit=lease.circuit_permit if lease else None,
                    )
                if (
                    local_attempt >= attempts
                    or budget.attempts_used >= budget.max_attempts
                    or budget.remaining_seconds <= 0
                    or not self._is_retryable_exception(exc)
                ):
                    raise
                if not await self._consume_retry_budget(provider, model_name):
                    self._record_retry_metric(provider, model_name, "budget_exhausted")
                    raise
                self._record_retry_metric(provider, model_name, "retry")
                backoff = self._retry_backoff_seconds(local_attempt)
                if backoff >= budget.remaining_seconds:
                    raise asyncio.TimeoutError("request execution deadline exceeded")
                await self._sleep_func(backoff)
            finally:
                await self.release(
                    lease,
                    actual_tokens=getattr(response, "total_tokens", None),
                )

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

        circuit_permit = await self._acquire_circuit_probe_or_reject(
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
                    queue_request=None,
                )
        except Exception:
            await self._release_circuit_probe(circuit_permit)
            raise

        lease.circuit_permit = circuit_permit
        return lease

    async def release(
        self, lease: Optional[SchedulerLease], *, actual_tokens: Optional[int] = None
    ) -> bool:
        if not lease:
            return False
        if lease.released:
            return False

        if lease.release_task is None:
            lease.release_task = asyncio.create_task(
                self._release_once(lease, actual_tokens=actual_tokens)
            )
        try:
            return bool(await asyncio.shield(lease.release_task))
        except asyncio.CancelledError:
            await lease.release_task
            raise

    async def _release_once(
        self, lease: SchedulerLease, *, actual_tokens: Optional[int]
    ) -> bool:
        """Release reservation and probe once, even if the caller is cancelled."""

        released = False
        try:
            if lease.reservation and self.admission_controller is not None:
                release_method = getattr(self.admission_controller, "release", None)
                if release_method:
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
        finally:
            try:
                await self._release_circuit_probe(lease.circuit_permit)
            finally:
                lease.released = True
                await self._queue_poll_coordinator.wake(
                    f"provider:{lease.provider}:{lease.model_name}"
                )
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

    async def record_success(
        self,
        *,
        provider: str,
        model_name: str,
        request_started_at_ms: Optional[int] = None,
        circuit_permit: Optional[CircuitPermit] = None,
    ) -> None:
        if not bool(self.circuit_config.get("enabled", False)):
            return
        try:
            result = await self._circuit_transition_with_fallback(
                provider,
                model_name,
                "success",
                request_started_at_ms=request_started_at_ms,
                permit=circuit_permit,
            )
        except Exception:
            logger.error(
                "Circuit success transition failed after provider response",
                exc_info=True,
            )
            return
        self._set_circuit_metric(provider, model_name, str(result["state"]))

    async def record_failure(
        self,
        *,
        provider: str,
        model_name: str,
        error: Exception,
        request_started_at_ms: Optional[int] = None,
        circuit_permit: Optional[CircuitPermit] = None,
    ) -> None:
        if not bool(self.circuit_config.get("enabled", False)):
            return
        if not self._is_retryable_exception(error):
            return

        result = await self._circuit_transition_with_fallback(
            provider,
            model_name,
            "failure",
            request_started_at_ms=request_started_at_ms,
            permit=circuit_permit,
        )
        self._set_circuit_metric(provider, model_name, str(result["state"]))

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
                    queue_request=None,
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
        expiry_key = self._queue_expiry_key(provider, model_name)
        depth_key = self._queue_depth_key(provider, model_name)
        sequence_key = self._queue_sequence_key(provider, model_name)
        queue_member = self._queue_member(request)
        started_at = self._time_func()
        expiry_ms = int(
            (started_at * 1000) + self.wait_timeout_ms + self.queue_lease_grace_ms
        )
        queue_request = QueueAdmissionRequest(
            scope=f"provider:{provider}:{model_name}",
            member_id=queue_member,
            priority_score=0,
            priority_band=self._queue_priority_band(request),
            expiry_timestamp_ms=expiry_ms,
            enqueue=False,
            order_key=queue_key,
            expiry_key=expiry_key,
            depth_key=depth_key,
            sequence_key=sequence_key,
        )

        queue_scope = queue_request.scope
        last_decision: Optional[AdmissionDecision] = None
        coordinator_active = False
        acquired = False
        try:
            deadline = started_at + (max(0, self.wait_timeout_ms) / 1000.0)
            while True:
                now = self._time_func()
                if last_decision is not None and (
                    self.wait_timeout_ms <= 0 or now >= deadline - 1e-9
                ):
                    decision = self._timeout_decision(
                        provider=provider,
                        model_name=model_name,
                        last_decision=last_decision,
                    )
                    self._record_rejection(provider, model_name, decision)
                    raise AdmissionRejectedError(decision)
                try:
                    lease = await self._try_acquire(
                        request=request,
                        model_name=model_name,
                        provider=provider,
                        estimated_input_tokens=estimated_input_tokens,
                        queued=queue_request.enqueue,
                        queue_key=queue_key,
                        queue_member=queue_member,
                        wait_seconds=max(0.0, self._time_func() - started_at),
                        queue_request=queue_request,
                    )
                    acquired = True
                    return lease
                except AdmissionRejectedError as exc:
                    last_decision = exc.decision
                    if not self._is_waitable_reason(
                        exc.decision.reason
                    ) and exc.decision.reason not in {
                        "queue_required",
                        "queue_wait",
                    }:
                        raise
                    if not queue_request.enqueue:
                        queue_request = QueueAdmissionRequest(
                            **{**queue_request.__dict__, "enqueue": True}
                        )
                        continue

                    if coordinator_active:
                        await self._queue_poll_coordinator.poll_denied(
                            queue_scope,
                            queue_member,
                            blocked_by_queue_head=exc.decision.reason
                            in {"queue_required", "queue_wait"},
                        )
                        coordinator_active = False

                    await self._queue_poll_coordinator.wait_turn(
                        queue_scope,
                        queue_member,
                        queue_request.priority_band,
                        deadline=deadline,
                        order_score=(exc.decision.metadata or {}).get("queue_score"),
                    )
                    coordinator_active = True
                    INFERENCE_METRICS.scheduler_queue_polls.labels(
                        provider=provider, model=model_name
                    ).inc()
        finally:
            await self._queue_poll_coordinator.remove(
                queue_scope,
                queue_member,
                wake=not acquired,
            )
            cleanup = getattr(self.admission_controller, "_cleanup_queue", None)
            if cleanup is not None and queue_request.enqueue:
                depth = int(await cleanup(queue_request))
            else:
                depth = self._last_queue_depths.get((provider, model_name), 0)
            self._last_queue_depths[(provider, model_name)] = depth
            INFERENCE_METRICS.scheduler_queue_depth.labels(
                provider=provider, model=model_name
            ).set(depth)

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
        queue_request: Optional[QueueAdmissionRequest],
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
            queue_request=queue_request,
        )
        if queue_request is not None:
            depth = int(decision.metadata.get("queue_depth", 0) or 0)
            pruned = int(decision.metadata.get("queue_pruned", 0) or 0)
            self._last_queue_depths[(provider, model_name)] = depth
            INFERENCE_METRICS.scheduler_queue_depth.labels(
                provider=provider, model=model_name
            ).set(depth)
            if pruned:
                INFERENCE_METRICS.scheduler_queue_pruned.labels(
                    provider=provider, model=model_name
                ).inc(pruned)
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
        return float(self._queue_priority_band(request))

    def _queue_priority_band(self, request: Any) -> int:
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
        return latency_rank * 100 + priority_rank * 10 + tier_rank

    def _queue_key(self, provider: str, model_name: str) -> str:
        return (
            f"{self.key_prefix}:{self.control_plane_version}:queue:"
            f"{self._scope(provider)}:{self._scope(model_name)}:order"
        )

    def _queue_expiry_key(self, provider: str, model_name: str) -> str:
        return self._queue_key(provider, model_name).removesuffix(":order") + ":expiry"

    def _queue_depth_key(self, provider: str, model_name: str) -> str:
        return self._queue_key(provider, model_name).removesuffix(":order") + ":depth"

    def _queue_sequence_key(self, provider: str, model_name: str) -> str:
        return (
            self._queue_key(provider, model_name).removesuffix(":order") + ":sequence"
        )

    def _queue_member(self, request: Any) -> str:
        request_id = str(getattr(request, "request_id", "") or uuid.uuid4().hex)
        digest = hashlib.sha256(uuid.uuid4().hex.encode("utf-8")).hexdigest()[:12]
        return f"{request_id}:{digest}"

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
    ) -> Optional[CircuitPermit]:
        if not bool(self.circuit_config.get("enabled", False)):
            return None
        try:
            state = await self._circuit_transition(provider, model_name, "acquire")
        except Exception:
            logger.warning("Circuit control plane unavailable", exc_info=True)
            if self.failure_mode == "open":
                state = await self._local_circuit_transition(
                    provider, model_name, "acquire"
                )
            else:
                decision = self._reject(
                    provider=provider,
                    model_name=model_name,
                    reason="circuit_unavailable",
                    status_code=503,
                    retry_after_seconds=1,
                )
                self._record_rejection(provider, model_name, decision)
                raise AdmissionRejectedError(decision)
        self._set_circuit_metric(provider, model_name, str(state["state"]))
        if state["allowed"]:
            return CircuitPermit(
                epoch=int(state["epoch"]),
                probe=bool(state["probe"]),
                provider=provider,
                model_name=model_name,
            )
        retry_after = max(
            1,
            math.ceil(
                (int(state["opened_until_ms"]) - int(self._time_func() * 1000)) / 1000.0
            ),
        )
        decision = self._reject(
            provider=provider,
            model_name=model_name,
            reason="circuit_open",
            status_code=429,
            retry_after_seconds=retry_after,
            metadata={
                "state": state["state"],
                "opened_until_ms": state["opened_until_ms"],
                "max_probe_requests": self._max_circuit_probes(),
            },
        )
        self._record_rejection(provider, model_name, decision)
        raise AdmissionRejectedError(decision)

    async def _release_circuit_probe(self, permit: Optional[CircuitPermit]) -> None:
        if not permit or not permit.probe:
            return
        # Release is best-effort and epoch-checked. Success/failure may already have
        # consumed the probe by changing state, making this an intentional no-op.
        try:
            await self._circuit_transition(
                permit.provider, permit.model_name, "release", permit=permit
            )
        except Exception:
            if self.failure_mode == "open":
                logger.warning(
                    "Using process-local circuit probe release after Redis failure",
                    exc_info=True,
                )
                await self._local_circuit_transition(
                    permit.provider,
                    permit.model_name,
                    "release",
                    permit=permit,
                )
            else:
                logger.error(
                    "Circuit probe release failed closed; future acquires remain "
                    "dependent on the Redis control plane",
                    exc_info=True,
                )

    async def _read_circuit_state(
        self, provider: str, model_name: str
    ) -> Dict[str, Any]:
        key = self._circuit_key(provider, model_name)
        client = self._redis_client()
        if client is not None and hasattr(client, "eval"):
            try:
                raw = await client.hgetall(key)
                if raw:
                    state = {
                        self._decode(k): self._coerce_state_value(v)
                        for k, v in raw.items()
                    }
                    state["opened_until"] = (
                        int(state.get("opened_until_ms", 0) or 0) / 1000.0
                    )
                    return state
            except Exception:
                logger.debug("Falling back to local circuit state", exc_info=True)
        state = dict(self._local_circuit_state.get(key, {}))
        if state:
            state["opened_until"] = int(state.get("opened_until_ms", 0) or 0) / 1000.0
        return state

    async def _write_circuit_state(
        self, provider: str, model_name: str, state: Dict[str, Any]
    ) -> None:
        key = self._circuit_key(provider, model_name)
        normalized = self._normalize_circuit_state(state)
        self._local_circuit_state[key] = dict(normalized)
        client = self._redis_client()
        if client is None:
            return
        try:
            await client.hset(key, mapping={k: str(v) for k, v in normalized.items()})
            ttl = int(self.circuit_config.get("state_ttl_seconds", 3600) or 3600)
            expire = getattr(client, "expire", None)
            if expire:
                result = expire(key, ttl)
                if hasattr(result, "__await__"):
                    await result
        except Exception:
            logger.debug("Failed to persist scheduler circuit state", exc_info=True)

    async def _circuit_transition(
        self,
        provider: str,
        model_name: str,
        action: str,
        *,
        request_started_at_ms: Optional[int] = None,
        permit: Optional[CircuitPermit] = None,
    ) -> Dict[str, Any]:
        client = self._redis_client()
        if client is None or not hasattr(client, "eval"):
            raise ConnectionError(
                "Redis circuit control plane is not configured or does not support eval"
            )
        now_ms = int(self._time_func() * 1000)
        result = await client.eval(
            _CIRCUIT_LUA,
            1,
            self._circuit_key(provider, model_name),
            action,
            now_ms,
            self._circuit_threshold(),
            self._circuit_recovery_ms(),
            self._max_circuit_probes(),
            request_started_at_ms if request_started_at_ms is not None else now_ms,
            permit.epoch if permit else -1,
            1 if permit and permit.probe else 0,
            int(self.circuit_config.get("state_ttl_seconds", 3600) or 3600),
        )
        parsed = self._parse_circuit_result(result)
        self._local_circuit_state[self._circuit_key(provider, model_name)] = {
            key: parsed[key]
            for key in (
                "state",
                "failure_count",
                "opened_until_ms",
                "epoch",
                "probe_count",
                "last_failure_at_ms",
                "updated_at_ms",
            )
        }
        return parsed

    async def _circuit_transition_with_fallback(
        self,
        provider: str,
        model_name: str,
        action: str,
        *,
        request_started_at_ms: Optional[int] = None,
        permit: Optional[CircuitPermit] = None,
    ) -> Dict[str, Any]:
        try:
            return await self._circuit_transition(
                provider,
                model_name,
                action,
                request_started_at_ms=request_started_at_ms,
                permit=permit,
            )
        except Exception:
            if self.failure_mode != "open":
                raise
            logger.warning(
                "Using process-local circuit after Redis transition failure",
                exc_info=True,
            )
            return await self._local_circuit_transition(
                provider,
                model_name,
                action,
                request_started_at_ms=request_started_at_ms,
                permit=permit,
            )

    async def _local_circuit_transition(
        self,
        provider: str,
        model_name: str,
        action: str,
        *,
        request_started_at_ms: Optional[int] = None,
        permit: Optional[CircuitPermit] = None,
    ) -> Dict[str, Any]:
        async with self._circuit_lock:
            key = self._circuit_key(provider, model_name)
            now_ms = int(self._time_func() * 1000)
            state = self._normalize_circuit_state(
                self._local_circuit_state.get(key, {})
            )
            current = str(state["state"])
            if action == "acquire":
                if current == "open" and int(state["opened_until_ms"]) <= now_ms:
                    current = "half_open"
                    state["state"] = current
                    state["probe_count"] = 0
                if current == "open":
                    return self._local_result(state, allowed=False, probe=False)
                if current == "half_open":
                    if int(state["probe_count"]) >= self._max_circuit_probes():
                        return self._local_result(state, allowed=False, probe=False)
                    state["probe_count"] = int(state["probe_count"]) + 1
                    state["updated_at_ms"] = now_ms
                    self._local_circuit_state[key] = state
                    return self._local_result(state, allowed=True, probe=True)
                return self._local_result(state, allowed=True, probe=False)

            permit_matches = bool(
                permit and permit.probe and permit.epoch == int(state["epoch"])
            )
            if action == "success":
                if permit_matches and current == "half_open":
                    state.update(
                        state="closed",
                        failure_count=0,
                        opened_until_ms=0,
                        probe_count=0,
                    )
                elif not (permit and permit.probe) and current == "closed":
                    started = request_started_at_ms or now_ms
                    if started >= int(state["last_failure_at_ms"]):
                        state["failure_count"] = 0
            elif action == "failure":
                if permit and permit.probe and not permit_matches:
                    return self._local_result(state, allowed=True, probe=True)
                if current != "open":
                    state["failure_count"] = int(state["failure_count"]) + 1
                    state["last_failure_at_ms"] = now_ms
                    if (
                        current == "half_open"
                        or int(state["failure_count"]) >= self._circuit_threshold()
                    ):
                        state["state"] = "open"
                        state["opened_until_ms"] = now_ms + self._circuit_recovery_ms()
                        state["probe_count"] = 0
                        state["epoch"] = int(state["epoch"]) + 1
            elif action == "release":
                if (
                    permit_matches
                    and current == "half_open"
                    and int(state["probe_count"]) > 0
                ):
                    state["probe_count"] = int(state["probe_count"]) - 1
            state["updated_at_ms"] = now_ms
            self._local_circuit_state[key] = state
            return self._local_result(
                state, allowed=True, probe=bool(permit and permit.probe)
            )

    def _normalize_circuit_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        opened_ms = state.get("opened_until_ms")
        if opened_ms is None:
            opened_ms = int(float(state.get("opened_until", 0) or 0) * 1000)
        updated_ms = state.get("updated_at_ms")
        if updated_ms is None:
            updated_ms = int(float(state.get("updated_at", self._time_func())) * 1000)
        return {
            "state": str(state.get("state", "closed") or "closed"),
            "failure_count": int(state.get("failure_count", 0) or 0),
            "opened_until_ms": int(opened_ms or 0),
            "epoch": int(state.get("epoch", 0) or 0),
            "probe_count": int(state.get("probe_count", 0) or 0),
            "last_failure_at_ms": int(state.get("last_failure_at_ms", 0) or 0),
            "updated_at_ms": int(updated_ms or 0),
        }

    def _parse_circuit_result(self, result: Any) -> Dict[str, Any]:
        values = list(result)
        return {
            "allowed": bool(int(values[0])),
            "epoch": int(values[1]),
            "probe": bool(int(values[2])),
            "state": self._decode(values[3]),
            "opened_until_ms": int(values[4]),
            "failure_count": int(values[5]),
            "probe_count": int(values[6]),
            "last_failure_at_ms": int(values[7]),
            "updated_at_ms": int(self._time_func() * 1000),
        }

    def _local_result(
        self, state: Dict[str, Any], *, allowed: bool, probe: bool
    ) -> Dict[str, Any]:
        return {**state, "allowed": allowed, "probe": probe}

    def _circuit_threshold(self) -> int:
        return max(1, int(self.circuit_config.get("failure_threshold", 5) or 5))

    def _circuit_recovery_ms(self) -> int:
        return max(0, int(self.circuit_config.get("recovery_timeout_ms", 30_000) or 0))

    def _max_circuit_probes(self) -> int:
        # A single probe is part of the v2 circuit contract. Keep raw-dict callers
        # safe even when they bypass schema validation.
        return 1

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
        return (
            f"{self.key_prefix}:{self.control_plane_version}:circuit:"
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
