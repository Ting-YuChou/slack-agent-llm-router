"""Real Redis coverage for the v2 atomic admission queue."""

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio
import redis.asyncio as redis

from src.admission import QueueAdmissionRequest, RedisAdmissionController
from src.provider_scheduler import CircuitPermit, ProviderCapacityScheduler
from src.utils.schema import QueryRequest


@pytest_asyncio.fixture
async def redis_client():
    url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"real Redis unavailable at {url}: {exc}")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


def _controller(redis_client, *, max_depth=0):
    prefix = f"test:admission:{uuid.uuid4().hex}"
    return RedisAdmissionController(
        {
            "enabled": True,
            "failure_mode": "closed",
            "redis": {"key_prefix": prefix},
            "queue": {
                "enabled": True,
                "timeout_ms": 0,
                "queue_lease_grace_ms": 0,
                "max_depth": max_depth,
                "control_plane_version": "v2",
            },
            "global_limits": {"active_requests": 1},
            "providers": {"openai": {"active_requests": 1}},
        },
        redis_client=redis_client,
    )


def _queue(prefix, member, *, expiry_ms, enqueue, band=0, max_depth=0):
    return QueueAdmissionRequest(
        scope=prefix,
        member_id=member,
        priority_score=0,
        priority_band=band,
        expiry_timestamp_ms=expiry_ms,
        enqueue=enqueue,
        order_key=f"{prefix}:order",
        expiry_key=f"{prefix}:expiry",
        depth_key=f"{prefix}:depth",
        sequence_key=f"{prefix}:sequence",
        max_depth=max_depth,
    )


async def _saturate(controller):
    decision = await controller.acquire_provider(
        request=QueryRequest(query="hold", user_id="holder"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=None,
    )
    assert decision.allowed
    return decision.reservation


def _command_count(info, command):
    return int(info.get(f"cmdstat_{command}", {}).get("calls", 0))


def _circuit_scheduler(redis_client, *, threshold=5, recovery_ms=1000):
    return ProviderCapacityScheduler(
        {
            "key_prefix": f"test:scheduler:{uuid.uuid4().hex}",
            "control_plane_version": "v2",
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": threshold,
                "recovery_timeout_ms": recovery_ms,
                "half_open_max_requests": 1,
            },
        },
        redis_client=redis_client,
    )


@pytest.mark.asyncio
async def test_circuit_concurrent_failures_open_atomically(redis_client):
    scheduler = _circuit_scheduler(redis_client, threshold=5)

    await asyncio.gather(
        *(
            scheduler.record_failure(
                provider="openai",
                model_name="gpt-5",
                error=RuntimeError("503 temporarily overloaded"),
                request_started_at_ms=int(time.time() * 1000),
            )
            for _ in range(10)
        )
    )

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "open"
    assert state["failure_count"] == 5
    assert ":v2:circuit:" in scheduler._circuit_key("openai", "gpt-5")


@pytest.mark.asyncio
async def test_circuit_probe_epoch_and_cap_are_atomic(redis_client):
    scheduler = _circuit_scheduler(redis_client, threshold=1, recovery_ms=0)
    await scheduler.record_failure(
        provider="openai",
        model_name="gpt-5",
        error=RuntimeError("503 temporarily overloaded"),
        request_started_at_ms=int(time.time() * 1000),
    )

    results = await asyncio.gather(
        *(
            scheduler._circuit_transition("openai", "gpt-5", "acquire")
            for _ in range(10)
        )
    )
    admitted = [result for result in results if result["allowed"]]
    assert len(admitted) == 1
    permit = CircuitPermit(
        epoch=admitted[0]["epoch"],
        probe=True,
        provider="openai",
        model_name="gpt-5",
    )

    await scheduler.record_success(
        provider="openai",
        model_name="gpt-5",
        request_started_at_ms=int(time.time() * 1000),
        circuit_permit=CircuitPermit(epoch=permit.epoch - 1, probe=True),
    )
    assert (await scheduler._read_circuit_state("openai", "gpt-5"))[
        "state"
    ] == "half_open"
    await scheduler.record_success(
        provider="openai",
        model_name="gpt-5",
        request_started_at_ms=int(time.time() * 1000),
        circuit_permit=permit,
    )
    assert (await scheduler._read_circuit_state("openai", "gpt-5"))["state"] == "closed"


@pytest.mark.asyncio
async def test_stale_probe_success_does_not_mutate_hash_or_refresh_ttl(redis_client):
    scheduler = _circuit_scheduler(redis_client, threshold=1, recovery_ms=0)
    await scheduler.record_failure(
        provider="openai",
        model_name="gpt-5",
        error=RuntimeError("503 temporarily overloaded"),
        request_started_at_ms=int(time.time() * 1000),
    )
    acquired = await scheduler._circuit_transition("openai", "gpt-5", "acquire")
    key = scheduler._circuit_key("openai", "gpt-5")
    await redis_client.pexpire(key, 5000)
    before_updated_at = await redis_client.hget(key, "updated_at_ms")
    before_ttl = await redis_client.pttl(key)

    await scheduler.record_success(
        provider="openai",
        model_name="gpt-5",
        request_started_at_ms=int(time.time() * 1000),
        circuit_permit=CircuitPermit(
            epoch=int(acquired["epoch"]) - 1,
            probe=True,
            provider="openai",
            model_name="gpt-5",
        ),
    )

    after_updated_at = await redis_client.hget(key, "updated_at_ms")
    after_ttl = await redis_client.pttl(key)
    assert after_updated_at == before_updated_at
    assert 0 < after_ttl <= before_ttl <= 5000


@pytest.mark.asyncio
async def test_uncontended_fast_path_executes_no_sorted_set_commands(redis_client):
    controller = _controller(redis_client)
    before = await redis_client.info("commandstats")

    decision = await controller.acquire_route(QueryRequest(query="fast", user_id="u1"))

    assert decision.allowed
    after = await redis_client.info("commandstats")
    for command in ("zadd", "zrange", "zrem", "zcard", "zrangebyscore"):
        assert _command_count(after, command) - _command_count(before, command) == 0


@pytest.mark.asyncio
async def test_existing_waiter_blocks_fast_caller(redis_client):
    controller = _controller(redis_client)
    reservation = await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    queued = _queue(prefix, "first", expiry_ms=expiry, enqueue=True)
    fast = _queue(prefix, "fast", expiry_ms=expiry, enqueue=False)
    request = QueryRequest(query="wait", user_id="u1")

    await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=queued,
    )
    await controller.release(reservation)
    decision = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=fast,
    )

    assert not decision.allowed
    assert decision.reason == "queue_required"
    assert decision.metadata["queue_depth"] == 1


@pytest.mark.asyncio
async def test_concurrent_enqueue_enforces_max_depth(redis_client):
    controller = _controller(redis_client, max_depth=5)
    await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    request = QueryRequest(query="wait", user_id="u1")

    async def enqueue(index):
        return await controller.acquire_provider(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            queue_request=_queue(
                prefix,
                f"member-{index}",
                expiry_ms=expiry,
                enqueue=True,
                max_depth=5,
            ),
        )

    decisions = await asyncio.gather(*(enqueue(index) for index in range(20)))

    assert int(await redis_client.get(f"{prefix}:depth")) == 5
    assert await redis_client.zcard(f"{prefix}:order") == 5
    assert sum(d.reason == "queue_depth_exceeded" for d in decisions) == 15


@pytest.mark.asyncio
async def test_stale_members_are_pruned_before_fast_admission(redis_client):
    controller = _controller(redis_client)
    prefix = f"queue:{uuid.uuid4().hex}"
    await redis_client.zadd(f"{prefix}:order", {"stale": 1})
    await redis_client.zadd(f"{prefix}:expiry", {"stale": 1})
    await redis_client.set(f"{prefix}:depth", 1)
    await redis_client.set(f"{prefix}:sequence", 99)
    request = _queue(
        prefix,
        "fast",
        expiry_ms=int(time.time() * 1000) + 60_000,
        enqueue=False,
    )

    decision = await controller.acquire_provider(
        request=QueryRequest(query="fast", user_id="u1"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=request,
    )

    assert decision.allowed
    assert decision.metadata["queue_pruned"] == 1
    assert decision.metadata["queue_depth"] == 0
    assert await redis_client.get(f"{prefix}:sequence") is None


@pytest.mark.asyncio
async def test_atomic_sequence_preserves_fifo_for_same_time_arrivals(redis_client):
    controller = _controller(redis_client)
    await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    request = QueryRequest(query="wait", user_id="u1")

    for member in ("first", "second", "third"):
        await controller.acquire_provider(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            queue_request=_queue(
                prefix, member, expiry_ms=expiry, enqueue=True, band=3
            ),
        )

    assert await redis_client.zrange(f"{prefix}:order", 0, -1) == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_provider_priority_band_precedes_sequence(redis_client):
    controller = _controller(redis_client)
    await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    request = QueryRequest(query="wait", user_id="u1")

    for member, band in (("low", 100), ("fast", 0), ("medium", 50)):
        await controller.acquire_provider(
            request=request,
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            queue_request=_queue(
                prefix, member, expiry_ms=expiry, enqueue=True, band=band
            ),
        )

    assert await redis_client.zrange(f"{prefix}:order", 0, -1) == [
        "fast",
        "medium",
        "low",
    ]


@pytest.mark.asyncio
async def test_priority_band_precedes_preadvanced_sequence(redis_client):
    controller = _controller(redis_client)
    await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    request = QueryRequest(query="wait", user_id="u1")

    await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=_queue(prefix, "low", expiry_ms=expiry, enqueue=True, band=100),
    )
    await redis_client.set(f"{prefix}:sequence", 200)
    await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=_queue(prefix, "fast", expiry_ms=expiry, enqueue=True, band=0),
    )

    assert await redis_client.zrange(f"{prefix}:order", 0, -1) == ["fast", "low"]


@pytest.mark.asyncio
async def test_nonempty_queue_rejects_exhausted_sequence(redis_client):
    controller = _controller(redis_client)
    await _saturate(controller)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    request = QueryRequest(query="wait", user_id="u1")
    first = _queue(prefix, "first", expiry_ms=expiry, enqueue=True)

    await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=first,
    )
    await redis_client.set(first.sequence_key, 999_999_999_999)
    decision = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=_queue(prefix, "overflow", expiry_ms=expiry, enqueue=True),
    )

    assert not decision.allowed
    assert decision.reason == "queue_sequence_exhausted"
    assert await redis_client.zscore(first.order_key, "overflow") is None


@pytest.mark.asyncio
async def test_cleanup_of_last_member_resets_sequence(redis_client):
    controller = _controller(redis_client)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    member = _queue(prefix, "only", expiry_ms=expiry, enqueue=True)
    await redis_client.zadd(member.order_key, {member.member_id: 1})
    await redis_client.zadd(member.expiry_key, {member.member_id: expiry})
    await redis_client.set(member.depth_key, 1)
    await redis_client.set(member.sequence_key, 57)

    assert await controller._cleanup_queue(member) == 0
    assert await redis_client.get(member.sequence_key) is None


@pytest.mark.asyncio
async def test_successful_admission_of_last_waiter_resets_sequence(redis_client):
    controller = _controller(redis_client)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    member = _queue(prefix, "only", expiry_ms=expiry, enqueue=True)

    decision = await controller.acquire_provider(
        request=QueryRequest(query="ready", user_id="u1"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=member,
    )

    assert decision.allowed
    assert await redis_client.get(member.depth_key) is None
    assert await redis_client.get(member.sequence_key) is None


@pytest.mark.asyncio
async def test_cleanup_returns_depth_with_other_members(redis_client):
    controller = _controller(redis_client)
    prefix = f"queue:{uuid.uuid4().hex}"
    expiry = int(time.time() * 1000) + 60_000
    first = _queue(prefix, "first", expiry_ms=expiry, enqueue=True)
    await redis_client.zadd(first.order_key, {"first": 1, "other": 2})
    await redis_client.zadd(first.expiry_key, {"first": expiry, "other": expiry})
    await redis_client.set(first.depth_key, 2)

    depth = await controller._cleanup_queue(first)

    assert depth == 1
    assert await redis_client.get(first.depth_key) == "1"
    assert await redis_client.zrange(first.order_key, 0, -1) == ["other"]


@pytest.mark.asyncio
async def test_provider_queue_bypasses_global_queue(redis_client):
    controller = _controller(redis_client)
    global_prefix = controller._key("v2", "queue", "global")
    await redis_client.zadd(f"{global_prefix}:order", {"global-waiter": 1})
    await redis_client.zadd(f"{global_prefix}:expiry", {"global-waiter": 9e15})
    await redis_client.set(f"{global_prefix}:depth", 1)
    provider_prefix = f"queue:{uuid.uuid4().hex}"

    decision = await controller.acquire_provider(
        request=QueryRequest(query="provider", user_id="u1"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=_queue(
            provider_prefix,
            "provider-fast",
            expiry_ms=int(time.time() * 1000) + 60_000,
            enqueue=False,
        ),
    )

    assert decision.allowed
    assert await redis_client.get(f"{global_prefix}:depth") == "1"


def test_scheduler_uses_deterministic_priority_band_without_local_time():
    scheduler = ProviderCapacityScheduler({}, None, time_func=lambda: 1)
    request = QueryRequest(query="one", user_id="u1")
    first = scheduler._queue_priority_band(request)
    scheduler._time_func = lambda: 999999

    assert scheduler._queue_priority_band(request) == first
