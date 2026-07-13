import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.admission import AdmissionDecision, AdmissionRejectedError
from src.provider_scheduler import (
    CircuitPermit,
    ProviderCapacityScheduler,
    RequestExecutionBudget,
    SchedulerLease,
)
from src.utils.schema import QueryRequest, UserTier


class FakeSchedulerRedis:
    def __init__(self):
        self.zsets = {}
        self.hashes = {}
        self.values = {}

    async def zadd(self, key, mapping):
        zset = self.zsets.setdefault(key, {})
        zset.update(mapping)
        return len(mapping)

    async def zrem(self, key, member):
        zset = self.zsets.setdefault(key, {})
        existed = member in zset
        zset.pop(member, None)
        return 1 if existed else 0

    async def zrange(self, key, start, stop):
        zset = self.zsets.get(key, {})
        members = [
            member for member, _score in sorted(zset.items(), key=lambda item: item[1])
        ]
        if stop == -1:
            return members[start:]
        return members[start : stop + 1]

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping=None, **_kwargs):
        self.hashes[key] = dict(mapping or {})
        return len(self.hashes[key])

    async def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    async def decr(self, key):
        self.values[key] = int(self.values.get(key, 0)) - 1
        return self.values[key]

    async def delete(self, key):
        existed = key in self.values
        self.values.pop(key, None)
        return 1 if existed else 0

    async def expire(self, _key, _seconds):
        return True


class BrokenCircuitRedis:
    async def eval(self, *_args):
        raise ConnectionError("redis unavailable")

    async def hgetall(self, *_args):
        raise ConnectionError("redis unavailable")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class RecordingAdmission:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls = []
        self.releases = []
        self.redis_client = FakeSchedulerRedis()

    async def acquire_provider(self, **kwargs):
        self.calls.append(kwargs)
        queue_request = kwargs.get("queue_request")
        if queue_request is not None and queue_request.enqueue:
            self.redis_client.zsets.setdefault(queue_request.order_key, {})[
                queue_request.member_id
            ] = queue_request.priority_score
            self.redis_client.zsets.setdefault(queue_request.expiry_key, {})[
                queue_request.member_id
            ] = queue_request.expiry_timestamp_ms
            self.redis_client.values[queue_request.depth_key] = len(
                self.redis_client.zsets[queue_request.order_key]
            )
        if self.decisions:
            decision = self.decisions.pop(0)
            if decision is not None:
                return decision
        if queue_request is not None and queue_request.enqueue:
            await self._cleanup_queue(queue_request)
        return AdmissionDecision.allow(
            SimpleNamespace(reservation_id=f"r{len(self.calls)}", stage="provider")
        )

    async def release(self, reservation, *, actual_tokens=None):
        self.releases.append((reservation, actual_tokens))
        return True

    async def _cleanup_queue(self, queue_request):
        self.redis_client.zsets.setdefault(queue_request.order_key, {}).pop(
            queue_request.member_id, None
        )
        self.redis_client.zsets.setdefault(queue_request.expiry_key, {}).pop(
            queue_request.member_id, None
        )
        depth = len(self.redis_client.zsets[queue_request.order_key])
        self.redis_client.values[queue_request.depth_key] = depth
        return depth


@pytest.mark.asyncio
async def test_scheduler_fast_path_passes_try_only_provider_queue_request():
    admission = RecordingAdmission()
    scheduler = ProviderCapacityScheduler({"queue_enabled": True}, admission)

    await scheduler.acquire(
        request=QueryRequest(query="hello", user_id="u1"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
    )

    queue_request = admission.calls[0]["queue_request"]
    assert queue_request.enqueue is False
    assert queue_request.scope == "provider:openai:gpt-5"
    assert ":v2:queue:" in queue_request.order_key
    assert queue_request.depth_key.endswith(":depth")
    assert queue_request.sequence_key.endswith(":sequence")


def _deny(reason, status_code=429):
    return AdmissionDecision.reject(
        status_code=status_code,
        error="rate_limited",
        reason=reason,
        message="denied",
        retry_after_seconds=1,
    )


@pytest.mark.asyncio
async def test_scheduler_acquires_and_releases_provider_capacity():
    admission = RecordingAdmission()
    scheduler = ProviderCapacityScheduler({}, admission)
    request = QueryRequest(query="hello", user_id="u1", max_tokens=3)

    response = SimpleNamespace(total_tokens=7)
    result = await scheduler.run_provider_call(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=4,
        operation=AsyncMock(return_value=response),
    )

    assert result is response
    assert admission.calls[0]["model_name"] == "gpt-5"
    assert admission.releases[0][1] == 7


@pytest.mark.asyncio
async def test_disabled_scheduler_bypasses_admission_and_scheduler_retries():
    admission = RecordingAdmission()
    scheduler = ProviderCapacityScheduler(
        {"enabled": False, "retry": {"max_attempts_per_request": 3}},
        admission,
    )
    request = QueryRequest(query="hello", user_id="u1")
    operation = AsyncMock(side_effect=RuntimeError("503 temporarily overloaded"))

    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
        )

    assert operation.await_count == 1
    assert admission.calls == []
    assert admission.releases == []


def test_scheduler_queue_score_orders_fast_priority_tier_then_age():
    clock = Clock()
    scheduler = ProviderCapacityScheduler({}, None, time_func=clock)
    free = QueryRequest(query="a", user_id="u1", priority=1, user_tier=UserTier.FREE)
    premium = QueryRequest(
        query="a", user_id="u2", priority=1, user_tier=UserTier.PREMIUM
    )
    fast = QueryRequest(
        query="a",
        user_id="u3",
        priority=1,
        user_tier=UserTier.FREE,
        metadata={"requires_low_latency": True},
    )
    high_priority = QueryRequest(
        query="a", user_id="u4", priority=5, user_tier=UserTier.FREE
    )

    assert scheduler._queue_score(fast) < scheduler._queue_score(high_priority)
    assert scheduler._queue_score(high_priority) < scheduler._queue_score(premium)
    assert scheduler._queue_score(premium) < scheduler._queue_score(free)


@pytest.mark.asyncio
async def test_scheduler_queue_timeout_cleans_up_wait_record():
    clock = Clock()
    admission = RecordingAdmission(
        decisions=[
            _deny("provider_active_requests_exceeded"),
            _deny("provider_active_requests_exceeded"),
            _deny("provider_active_requests_exceeded"),
        ]
    )
    scheduler = ProviderCapacityScheduler(
        {"queue_enabled": True, "wait_timeout_ms": 10, "poll_interval_ms": 5},
        admission,
        time_func=clock,
        sleep_func=clock.sleep,
    )
    request = QueryRequest(query="hello", user_id="u1")

    with pytest.raises(AdmissionRejectedError) as exc:
        await scheduler.acquire(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    assert exc.value.decision.reason == "scheduler_queue_timeout"
    queue_key = scheduler._queue_key("openai", "gpt-5")
    assert admission.redis_client.zsets[queue_key] == {}
    assert scheduler.get_health_status()["queue_depths"] == [
        {"provider": "openai", "model": "gpt-5", "depth": 0}
    ]


@pytest.mark.asyncio
async def test_scheduler_cleanup_preserves_other_member_depth():
    clock = Clock()
    admission = RecordingAdmission(
        decisions=[
            _deny("provider_active_requests_exceeded"),
            _deny("provider_active_requests_exceeded"),
        ]
    )
    scheduler = ProviderCapacityScheduler(
        {
            "queue_enabled": True,
            "wait_timeout_ms": 1,
            "poll_interval_ms": 1,
            "poll_jitter_ratio": 0,
        },
        admission,
        time_func=clock,
        sleep_func=clock.sleep,
    )
    queue_key = scheduler._queue_key("openai", "gpt-5")
    expiry_key = scheduler._queue_expiry_key("openai", "gpt-5")
    depth_key = scheduler._queue_depth_key("openai", "gpt-5")
    admission.redis_client.zsets[queue_key] = {"other": 1}
    admission.redis_client.zsets[expiry_key] = {"other": 9999999}
    admission.redis_client.values[depth_key] = 1

    with pytest.raises(AdmissionRejectedError):
        await scheduler.acquire(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    assert scheduler.get_health_status()["queue_depths"] == [
        {"provider": "openai", "model": "gpt-5", "depth": 1}
    ]


@pytest.mark.asyncio
async def test_scheduler_adaptive_poll_sequence_caps_and_respects_deadline():
    sleeps = []
    now = 1000.0

    def time_func():
        return now

    async def sleep_func(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    admission = RecordingAdmission(
        decisions=[_deny("provider_active_requests_exceeded")] * 20
    )
    scheduler = ProviderCapacityScheduler(
        {
            "queue_enabled": True,
            "wait_timeout_ms": 700,
            "poll_interval_ms": 25,
            "max_poll_interval_ms": 250,
            "poll_jitter_ratio": 0,
        },
        admission,
        time_func=time_func,
        sleep_func=sleep_func,
    )

    with pytest.raises(AdmissionRejectedError):
        await scheduler.acquire(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    assert sleeps == pytest.approx([0.025, 0.05, 0.1, 0.2, 0.25, 0.075])


@pytest.mark.asyncio
async def test_scheduler_adaptive_poll_jitter_stays_bounded_by_deadline():
    sleeps = []
    now = 1000.0

    def time_func():
        return now

    async def sleep_func(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    scheduler = ProviderCapacityScheduler(
        {
            "queue_enabled": True,
            "wait_timeout_ms": 400,
            "poll_interval_ms": 25,
            "max_poll_interval_ms": 250,
            "poll_jitter_ratio": 0.2,
        },
        RecordingAdmission(decisions=[_deny("provider_active_requests_exceeded")] * 20),
        time_func=time_func,
        sleep_func=sleep_func,
    )

    with pytest.raises(AdmissionRejectedError):
        await scheduler.acquire(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    nominal = [0.025, 0.05, 0.1, 0.2]
    for actual, expected in zip(sleeps, nominal):
        assert expected * 0.8 <= actual <= expected * 1.2
    assert sum(sleeps) == pytest.approx(0.4)


def test_scheduler_allows_fallback_for_queue_timeout_with_provider_last_reason():
    scheduler = ProviderCapacityScheduler({}, None)
    decision = AdmissionDecision.reject(
        status_code=429,
        error="rate_limited",
        reason="scheduler_queue_timeout",
        message="timed out",
        metadata={"last_reason": "provider_active_requests_exceeded"},
    )

    assert scheduler.allow_fallback_for_rejection(decision) is True


def test_scheduler_blocks_fallback_for_queue_timeout_with_global_last_reason():
    scheduler = ProviderCapacityScheduler({}, None)
    decision = AdmissionDecision.reject(
        status_code=429,
        error="rate_limited",
        reason="scheduler_queue_timeout",
        message="timed out",
        metadata={"last_reason": "global_token_budget_exceeded"},
    )

    assert scheduler.allow_fallback_for_rejection(decision) is False


@pytest.mark.asyncio
async def test_scheduler_queue_cancellation_cleans_up_wait_record():
    release_sleep = asyncio.Event()
    admission = RecordingAdmission(
        decisions=[_deny("provider_active_requests_exceeded")] * 10
    )

    async def sleep_forever(_seconds):
        await release_sleep.wait()

    scheduler = ProviderCapacityScheduler(
        {"queue_enabled": True, "wait_timeout_ms": 1000, "poll_interval_ms": 50},
        admission,
        sleep_func=sleep_forever,
    )
    request = QueryRequest(query="hello", user_id="u1")
    task = asyncio.create_task(
        scheduler.acquire(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )
    )
    while not admission.redis_client.zsets:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    queue_key = scheduler._queue_key("openai", "gpt-5")
    assert admission.redis_client.zsets[queue_key] == {}
    release_sleep.set()


@pytest.mark.asyncio
async def test_scheduler_circuit_opens_and_half_open_probe_closes():
    clock = Clock()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 1,
                "recovery_timeout_ms": 1000,
            },
        },
        RecordingAdmission(),
        time_func=clock,
        sleep_func=clock.sleep,
    )
    request = QueryRequest(query="hello", user_id="u1")

    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=AsyncMock(side_effect=RuntimeError("503 temporarily overloaded")),
        )

    with pytest.raises(AdmissionRejectedError) as exc:
        await scheduler.acquire(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )
    assert exc.value.decision.reason == "circuit_open"

    clock.now += 1.1
    response = SimpleNamespace(total_tokens=2)
    result = await scheduler.run_provider_call(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        operation=AsyncMock(return_value=response),
    )
    assert result is response
    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "closed"
    assert scheduler.get_health_status()["circuit_states"] == [
        {"provider": "openai", "model": "gpt-5", "state": "closed"}
    ]


@pytest.mark.asyncio
async def test_scheduler_half_open_allows_only_limited_probe_traffic():
    clock = Clock()
    admission = RecordingAdmission()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 1,
                "recovery_timeout_ms": 1000,
                "half_open_max_requests": 1,
            },
        },
        admission,
        time_func=clock,
    )
    await scheduler._write_circuit_state(
        "openai",
        "gpt-5",
        {
            "state": "open",
            "failure_count": 1,
            "opened_until": clock.now - 1,
            "updated_at": clock.now - 2,
        },
    )
    request = QueryRequest(query="hello", user_id="u1")

    first = await scheduler.acquire(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
    )
    with pytest.raises(AdmissionRejectedError) as exc:
        await scheduler.acquire(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    assert exc.value.decision.reason == "circuit_open"
    assert exc.value.decision.metadata["state"] == "half_open"
    assert len(admission.calls) == 1

    await scheduler.release(first)
    second = await scheduler.acquire(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
    )
    await scheduler.release(second)
    assert len(admission.calls) == 2


@pytest.mark.asyncio
async def test_scheduler_half_open_failure_reopens_circuit():
    clock = Clock()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout_ms": 1000,
            },
        },
        RecordingAdmission(),
        time_func=clock,
        sleep_func=clock.sleep,
    )
    await scheduler._write_circuit_state(
        "openai",
        "gpt-5",
        {
            "state": "half_open",
            "failure_count": 1,
            "opened_until": 0,
            "updated_at": clock.now,
        },
    )
    request = QueryRequest(query="hello", user_id="u1")

    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=AsyncMock(side_effect=RuntimeError("503 temporarily overloaded")),
        )

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "open"
    assert state["opened_until"] > clock.now


@pytest.mark.asyncio
async def test_late_success_does_not_clear_newer_failure_count():
    clock = Clock()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {"enabled": True, "failure_threshold": 5},
        },
        RecordingAdmission(),
        time_func=clock,
    )
    await scheduler.record_failure(
        provider="openai",
        model_name="gpt-5",
        error=RuntimeError("503 temporarily overloaded"),
        request_started_at_ms=999_000,
    )
    await scheduler.record_success(
        provider="openai", model_name="gpt-5", request_started_at_ms=999_000
    )

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["failure_count"] == 1


@pytest.mark.asyncio
async def test_stale_probe_epoch_cannot_close_circuit():
    clock = Clock()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {"enabled": True, "failure_threshold": 1},
        },
        RecordingAdmission(),
        time_func=clock,
    )
    await scheduler._write_circuit_state(
        "openai",
        "gpt-5",
        {"state": "half_open", "failure_count": 1, "epoch": 3, "probe_count": 1},
    )
    await scheduler.record_success(
        provider="openai",
        model_name="gpt-5",
        request_started_at_ms=1_000_000,
        circuit_permit=CircuitPermit(epoch=2, probe=True),
    )

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "half_open"


@pytest.mark.asyncio
async def test_open_state_late_failure_does_not_extend_recovery_window():
    clock = Clock()
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 1,
                "recovery_timeout_ms": 1000,
            },
        },
        RecordingAdmission(),
        time_func=clock,
    )
    await scheduler.record_failure(
        provider="openai",
        model_name="gpt-5",
        error=RuntimeError("503 temporarily overloaded"),
        request_started_at_ms=1_000_000,
    )
    opened = (await scheduler._read_circuit_state("openai", "gpt-5"))["opened_until_ms"]
    clock.now += 0.5
    await scheduler.record_failure(
        provider="openai",
        model_name="gpt-5",
        error=RuntimeError("503 temporarily overloaded"),
        request_started_at_ms=1_000_100,
    )
    assert (await scheduler._read_circuit_state("openai", "gpt-5"))[
        "opened_until_ms"
    ] == opened


@pytest.mark.asyncio
async def test_circuit_redis_error_obeys_failure_mode_on_acquire():
    request = QueryRequest(query="hello", user_id="u1")
    closed = ProviderCapacityScheduler(
        {"failure_mode": "closed", "circuit_breaker": {"enabled": True}},
        RecordingAdmission(),
        redis_client=BrokenCircuitRedis(),
    )
    with pytest.raises(AdmissionRejectedError) as exc:
        await closed.acquire(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )
    assert exc.value.decision.reason == "circuit_unavailable"

    opened = ProviderCapacityScheduler(
        {"failure_mode": "open", "circuit_breaker": {"enabled": True}},
        RecordingAdmission(),
        redis_client=BrokenCircuitRedis(),
    )
    lease = await opened.acquire(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
    )
    assert lease is not None


@pytest.mark.asyncio
async def test_circuit_missing_redis_client_rejects_when_failure_mode_closed():
    admission = RecordingAdmission()
    admission.redis_client = None
    scheduler = ProviderCapacityScheduler(
        {"failure_mode": "closed", "circuit_breaker": {"enabled": True}},
        admission,
    )

    with pytest.raises(AdmissionRejectedError) as exc:
        await scheduler.acquire(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
        )

    assert exc.value.decision.reason == "circuit_unavailable"


@pytest.mark.asyncio
async def test_probe_release_falls_back_to_epoch_checked_local_state_when_open():
    scheduler = ProviderCapacityScheduler(
        {"failure_mode": "open", "circuit_breaker": {"enabled": True}},
        RecordingAdmission(),
        redis_client=BrokenCircuitRedis(),
    )
    key = scheduler._circuit_key("openai", "gpt-5")
    scheduler._local_circuit_state[key] = scheduler._normalize_circuit_state(
        {"state": "half_open", "epoch": 4, "probe_count": 1}
    )

    await scheduler._release_circuit_probe(
        CircuitPermit(
            epoch=4,
            probe=True,
            provider="openai",
            model_name="gpt-5",
        )
    )

    assert scheduler._local_circuit_state[key]["probe_count"] == 0


@pytest.mark.asyncio
async def test_scheduler_retry_budget_exhaustion_stops_retry():
    scheduler = ProviderCapacityScheduler(
        {
            "retry": {
                "max_attempts_per_request": 3,
                "budget_enabled": True,
                "budget_tokens": 0,
            }
        },
        RecordingAdmission(),
    )
    request = QueryRequest(query="hello", user_id="u1")
    operation = AsyncMock(side_effect=RuntimeError("503 temporarily overloaded"))

    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
        )

    assert operation.await_count == 1


@pytest.mark.asyncio
async def test_request_execution_budget_is_shared_across_provider_calls():
    scheduler = ProviderCapacityScheduler(
        {"retry": {"max_attempts_per_request": 3, "initial_backoff_ms": 0}},
        RecordingAdmission(),
    )
    budget = RequestExecutionBudget(max_attempts=3, deadline_seconds=60)
    request = QueryRequest(query="hello", user_id="u1")
    first = AsyncMock(side_effect=RuntimeError("503 temporarily overloaded"))
    second = AsyncMock(side_effect=RuntimeError("503 temporarily overloaded"))

    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=first,
            execution_budget=budget,
        )
    with pytest.raises(RuntimeError):
        await scheduler.run_provider_call(
            request=request,
            model_name="local",
            provider="vllm",
            estimated_input_tokens=1,
            operation=second,
            execution_budget=budget,
        )

    assert first.await_count == 3
    assert second.await_count == 0


@pytest.mark.asyncio
async def test_request_execution_deadline_stops_provider_call_and_releases_lease():
    admission = RecordingAdmission()
    scheduler = ProviderCapacityScheduler(
        {"request_deadline_seconds": 0.01, "retry": {"max_attempts_per_request": 3}},
        admission,
    )
    budget = RequestExecutionBudget(max_attempts=3, deadline_seconds=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=lambda: asyncio.sleep(1),
            execution_budget=budget,
        )

    assert len(admission.releases) == 1


@pytest.mark.asyncio
async def test_acquire_timeout_does_not_record_provider_transport_failure():
    scheduler = ProviderCapacityScheduler({"request_deadline_seconds": 0.01})
    scheduler.acquire = AsyncMock(side_effect=asyncio.TimeoutError())
    scheduler.record_failure = AsyncMock()

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=QueryRequest(query="hello", user_id="u1"),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=AsyncMock(),
        )

    scheduler.record_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_cleanup_completes_when_caller_is_cancelled():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()
    admission = RecordingAdmission()

    async def release(_reservation, *, actual_tokens=None):
        started.set()
        await finish.wait()
        completed.set()
        return True

    admission.release = release
    scheduler = ProviderCapacityScheduler({}, admission)
    lease = SchedulerLease(provider="openai", model_name="gpt-5", reservation=object())
    task = asyncio.create_task(scheduler.release(lease))
    await started.wait()
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed.is_set()
    assert lease.released is True
