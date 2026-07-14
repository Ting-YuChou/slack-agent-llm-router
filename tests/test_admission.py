import asyncio
import time

import pytest
from redis.exceptions import MaxConnectionsError

from src.admission import (
    QUEUE_PRIORITY_STRIDE,
    QueueAdmissionRequest,
    RedisAdmissionController,
)
from src.utils.metrics import ADMISSION_METRICS, metric_value_for_labels
from src.utils.schema import QueryRequest, UserTier


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.zsets = {}
        self.scripts = []

    async def ping(self):
        return True

    async def close(self):
        return None

    async def eval(self, script, numkeys, *keys_and_args):
        self.scripts.append(script)
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        if "admission:queue_enter" in script:
            return self._queue_enter(keys, args)
        if "admission:queue_leave" in script:
            return self._queue_leave(keys)
        if "admission:queue_cleanup" in script:
            member = args[0]
            removed = member in self.zsets.setdefault(keys[0], {})
            self.zsets[keys[0]].pop(member, None)
            self.zsets.setdefault(keys[1], {}).pop(member, None)
            depth = int(self.values.get(keys[2], 0))
            if removed:
                depth = max(0, depth - 1)
            if depth:
                self.values[keys[2]] = depth
            else:
                self.values.pop(keys[2], None)
                self.values.pop(keys[3], None)
            return [1, depth]
        if "admission:release" in script:
            return self._release(keys, args)
        if "admission:atomic_admit" in script:
            return self._admit(keys, args)
        raise AssertionError("unexpected script")

    def _queue_enter(self, keys, args):
        max_depth = int(args[0])
        current = int(self.values.get(keys[0], 0))
        if max_depth > 0 and current >= max_depth:
            return [0, current]
        current += 1
        self.values[keys[0]] = current
        return [1, current]

    def _queue_leave(self, keys):
        current = int(self.values.get(keys[0], 0))
        if current <= 1:
            self.values.pop(keys[0], None)
            return [1, 0]
        current -= 1
        self.values[keys[0]] = current
        return [1, current]

    def _admit(self, keys, args):
        queue_enabled = int(args[0])
        queue_member = args[1]
        queue_score = float(args[2])
        queue_expiry = float(args[3])
        queue_enqueue = int(args[4])
        queue_depth = 0
        pruned = 0
        now_ms = float(args[8])
        if queue_enabled:
            order_key, expiry_key, depth_key, sequence_key = keys[-4:]
            queue_depth = int(self.values.get(depth_key, 0))
            if queue_depth:
                expired = [
                    member
                    for member, expiry in self.zsets.get(expiry_key, {}).items()
                    if expiry <= now_ms
                ]
                for member in expired:
                    if member in self.zsets.setdefault(order_key, {}):
                        self.zsets[order_key].pop(member, None)
                        queue_depth -= 1
                        pruned += 1
                    self.zsets.setdefault(expiry_key, {}).pop(member, None)
                if queue_depth:
                    self.values[depth_key] = queue_depth
                else:
                    self.values.pop(depth_key, None)
                    self.values.pop(sequence_key, None)
            if not queue_enqueue and queue_depth:
                return [0, "queue_required", 0, order_key, queue_depth, pruned]
            if queue_enqueue:
                if queue_member not in self.zsets.setdefault(order_key, {}):
                    sequence = int(self.values.get(sequence_key, 0)) + 1
                    self.values[sequence_key] = sequence
                    if queue_depth > 0 and sequence >= QUEUE_PRIORITY_STRIDE:
                        return [
                            0,
                            "queue_sequence_exhausted",
                            0,
                            order_key,
                            queue_depth,
                            pruned,
                        ]
                    self.zsets[order_key][queue_member] = (
                        queue_score * QUEUE_PRIORITY_STRIDE + sequence
                    )
                    queue_depth += 1
                    self.values[depth_key] = queue_depth
                self.zsets.setdefault(expiry_key, {})[queue_member] = queue_expiry
                head = min(self.zsets[order_key], key=self.zsets[order_key].get)
                if head != queue_member:
                    return [0, "queue_wait", 0, order_key, queue_depth, pruned]

        active_count = int(args[9])
        arg_index = 10
        for index in range(active_count):
            limit = int(args[arg_index])
            arg_index += 1
            current = int(self.values.get(keys[1 + index], 0))
            if limit > 0 and current >= limit:
                return [0, "active_limit", 100, keys[1 + index], queue_depth, pruned]

        bucket_count = int(args[arg_index])
        arg_index += 1
        bucket_key_start = 1 + active_count
        next_tokens = []
        for index in range(bucket_count):
            capacity = float(args[arg_index])
            refill_per_ms = float(args[arg_index + 1])
            cost = float(args[arg_index + 2])
            reason = args[arg_index + 3]
            arg_index += 4
            bucket_key = keys[bucket_key_start + index]
            bucket = self.hashes.get(bucket_key, {})
            tokens = float(bucket.get("tokens", capacity))
            updated_ms = float(bucket.get("updated_ms", now_ms))
            if now_ms > updated_ms and refill_per_ms > 0:
                tokens = min(capacity, tokens + ((now_ms - updated_ms) * refill_per_ms))
            if cost > 0 and tokens + 0.000001 < cost:
                retry_ms = 1000
                if refill_per_ms > 0:
                    retry_ms = int((cost - tokens + refill_per_ms - 1) / refill_per_ms)
                return [0, reason, retry_ms, bucket_key, queue_depth, pruned]
            next_tokens.append((bucket_key, tokens - cost))

        reservation_key = keys[0]
        if reservation_key in self.values:
            return [
                0,
                "reservation_conflict",
                1000,
                reservation_key,
                queue_depth,
                pruned,
            ]
        self.values[reservation_key] = 1
        for index in range(active_count):
            active_key = keys[1 + index]
            self.values[active_key] = int(self.values.get(active_key, 0)) + 1
        for bucket_key, tokens in next_tokens:
            self.hashes[bucket_key] = {
                "tokens": tokens,
                "updated_ms": now_ms,
            }
        if queue_enabled and queue_enqueue:
            self.zsets[order_key].pop(queue_member, None)
            self.zsets[expiry_key].pop(queue_member, None)
            queue_depth -= 1
            if queue_depth:
                self.values[depth_key] = queue_depth
            else:
                self.values.pop(depth_key, None)
                self.values.pop(sequence_key, None)
        return [1, "allowed", 0, "", queue_depth, pruned]

    def _release(self, keys, args):
        reservation_key = keys[0]
        if reservation_key not in self.values:
            return [0]
        self.values.pop(reservation_key, None)
        now_ms = float(args[0])
        active_count = int(args[2])
        token_count = int(args[3])
        arg_index = 4
        for index in range(active_count):
            active_key = keys[1 + index]
            current = int(self.values.get(active_key, 0))
            if current <= 1:
                self.values.pop(active_key, None)
            else:
                self.values[active_key] = current - 1

        token_key_start = 1 + active_count
        for index in range(token_count):
            capacity = float(args[arg_index])
            delta = float(args[arg_index + 1])
            arg_index += 2
            token_key = keys[token_key_start + index]
            current = float(self.hashes.get(token_key, {}).get("tokens", capacity))
            next_value = current + delta
            if delta > 0:
                next_value = min(capacity, next_value)
            self.hashes[token_key] = {"tokens": next_value, "updated_ms": now_ms}
        return [1]


def test_queue_admission_request_carries_atomic_queue_contract():
    request = QueueAdmissionRequest(
        scope="provider:openai:gpt-5",
        member_id="r1",
        priority_score=42.0,
        expiry_timestamp_ms=1234,
        enqueue=False,
    )

    assert request.scope == "provider:openai:gpt-5"
    assert request.enqueue is False
    assert request.depth_key == ""
    assert request.sequence_key == ""
    assert request.priority_band == 0


@pytest.mark.asyncio
async def test_uncontended_queue_uses_one_atomic_eval_without_legacy_queue_scripts():
    redis_client = FakeRedis()
    controller = _controller(
        {"queue": {"enabled": True, "timeout_ms": 10}},
        redis_client=redis_client,
    )

    decision = await controller.acquire_route(QueryRequest(query="one", user_id="u1"))

    assert decision.allowed is True
    assert len(redis_client.scripts) == 1
    assert "admission:atomic_admit" in redis_client.scripts[0]
    assert "admission:queue_enter" not in redis_client.scripts[0]
    assert 'redis.call("GET", queue_depth_key)' in redis_client.scripts[0]
    assert "if queue_depth > 0 then" in redis_client.scripts[0]


@pytest.mark.asyncio
async def test_cleanup_queue_returns_remaining_depth():
    redis_client = FakeRedis()
    controller = _controller({}, redis_client=redis_client)
    request = QueueAdmissionRequest(
        scope="provider:openai:gpt-5",
        member_id="one",
        priority_score=0,
        priority_band=0,
        expiry_timestamp_ms=2000,
        enqueue=True,
        order_key="queue:order",
        expiry_key="queue:expiry",
        depth_key="queue:depth",
        sequence_key="queue:sequence",
    )
    redis_client.zsets[request.order_key] = {"one": 1, "two": 2}
    redis_client.zsets[request.expiry_key] = {"one": 2000, "two": 2000}
    redis_client.values[request.depth_key] = 2

    depth = await controller._cleanup_queue(request)

    assert depth == 1
    assert redis_client.values[request.depth_key] == 1
    assert redis_client.zsets[request.order_key] == {"two": 2}


@pytest.mark.asyncio
async def test_fake_redis_rejects_sequence_exhaustion_without_enqueuing():
    redis_client = FakeRedis()
    controller = _controller({}, redis_client=redis_client)
    request = QueueAdmissionRequest(
        scope="provider:openai:gpt-5",
        member_id="overflow",
        priority_score=0,
        priority_band=0,
        expiry_timestamp_ms=9_000_000_000_000_000,
        enqueue=True,
        order_key="queue:order",
        expiry_key="queue:expiry",
        depth_key="queue:depth",
        sequence_key="queue:sequence",
    )
    redis_client.zsets[request.order_key] = {"first": 1}
    redis_client.zsets[request.expiry_key] = {"first": 9_000_000_000_000_000}
    redis_client.values[request.depth_key] = 1
    redis_client.values[request.sequence_key] = 999_999_999_999

    decision = await controller.acquire_provider(
        request=QueryRequest(query="wait", user_id="u1"),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        queue_request=request,
    )

    assert not decision.allowed
    assert decision.reason == "queue_sequence_exhausted"
    assert request.member_id not in redis_client.zsets[request.order_key]


def test_api_queue_honors_zero_lease_grace():
    controller = _controller(
        {"queue": {"enabled": True, "timeout_ms": 10, "queue_lease_grace_ms": 0}}
    )
    assert controller._queue_expiry_timestamp_ms(1000) == 1010


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class RecoveringRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.available = False
        self.ping_calls = 0

    async def ping(self):
        self.ping_calls += 1
        if not self.available:
            raise ConnectionError("redis unavailable")
        return True

    async def eval(self, script, numkeys, *keys_and_args):
        if not self.available:
            raise ConnectionError("redis unavailable")
        return await super().eval(script, numkeys, *keys_and_args)


class PoolExhaustedRedis(FakeRedis):
    async def eval(self, script, numkeys, *keys_and_args):
        raise MaxConnectionsError("pool exhausted")


class RecoveringPingRedis(FakeRedis):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.ping_calls = 0

    async def ping(self):
        self.ping_calls += 1
        if self.error is not None:
            raise self.error
        return True


class BlockingRedis(FakeRedis):
    def __init__(self, delay_seconds=1.0):
        super().__init__()
        self.delay_seconds = delay_seconds

    async def ping(self):
        await asyncio.sleep(self.delay_seconds)
        return True

    async def eval(self, script, numkeys, *keys_and_args):
        await asyncio.sleep(self.delay_seconds)
        return await super().eval(script, numkeys, *keys_and_args)


def _controller(config, *, clock=None, redis_client=None):
    clock = clock or Clock()
    controller = RedisAdmissionController(
        {
            "enabled": True,
            "failure_mode": "closed",
            "queue": {"enabled": False, "timeout_ms": 0, "poll_interval_ms": 1},
            "global_limits": {"active_requests": 100},
            **config,
        },
        redis_client=redis_client or FakeRedis(),
        time_func=clock,
        sleep_func=clock.sleep,
    )
    return controller


@pytest.mark.asyncio
async def test_token_budget_refills_after_window():
    clock = Clock()
    controller = _controller(
        {
            "global_limits": {"active_requests": 100},
            "token_budget": {
                "enabled": True,
                "tokens_per_minute": 60,
                "burst_tokens": 60,
            },
        },
        clock=clock,
    )
    request = QueryRequest(query="hello", user_id="u1", max_tokens=1)

    first = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=59,
    )
    assert first.allowed is True

    second = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=59,
    )
    assert second.allowed is False
    assert second.reason == "global_token_budget_exceeded"

    clock.now += 60
    third = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=59,
    )
    assert third.allowed is True


@pytest.mark.asyncio
async def test_token_budget_can_use_global_limits_for_capacity():
    controller = _controller(
        {
            "global_limits": {
                "active_requests": 100,
                "tokens_per_minute": 10,
                "burst_tokens": 10,
            },
            "token_budget": {"enabled": True},
        }
    )
    request = QueryRequest(query="hello", user_id="u1", max_tokens=1)

    first = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=9,
    )
    second = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=9,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "global_token_budget_exceeded"


@pytest.mark.asyncio
async def test_route_admission_denies_per_user_before_tier():
    controller = _controller(
        {
            "requests_per_minute": 1000,
            "burst_size": 100,
            "per_user": {"requests_per_minute": 1, "burst_size": 1},
            "by_tier": {"free": {"requests_per_minute": 2, "burst_size": 2}},
        }
    )
    request = QueryRequest(query="hello", user_id="u1", user_tier=UserTier.FREE)

    first = await controller.acquire_route(request)
    second = await controller.acquire_route(request)

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "user_request_rate_exceeded"


@pytest.mark.asyncio
async def test_provider_admission_denies_provider_request_rate():
    controller = _controller(
        {
            "providers": {
                "openai": {
                    "requests_per_minute": 1,
                    "burst_size": 1,
                    "active_requests": 10,
                }
            }
        }
    )
    request = QueryRequest(query="hello", user_id="u1")

    first = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=10,
    )
    second = await controller.acquire_provider(
        request=request,
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=10,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "provider_request_rate_exceeded"


@pytest.mark.asyncio
async def test_queue_timeout_returns_active_limit_denial():
    clock = Clock()
    controller = _controller(
        {
            "queue": {
                "enabled": True,
                "max_depth": 10,
                "timeout_ms": 10,
                "poll_interval_ms": 5,
            },
            "global_limits": {"active_requests": 1},
        },
        clock=clock,
    )
    first = await controller.acquire_route(QueryRequest(query="one", user_id="u1"))

    second = await controller.acquire_route(QueryRequest(query="two", user_id="u2"))

    assert first.allowed is True
    assert second.allowed is False
    assert second.reason == "global_active_requests_exceeded"


@pytest.mark.asyncio
async def test_release_is_idempotent_and_frees_active_capacity():
    controller = _controller({"global_limits": {"active_requests": 1}})
    first = await controller.acquire_route(QueryRequest(query="one", user_id="u1"))
    blocked = await controller.acquire_route(QueryRequest(query="two", user_id="u2"))

    released = await controller.release(first.reservation)
    released_again = await controller.release(first.reservation)
    after_release = await controller.acquire_route(
        QueryRequest(query="two", user_id="u2")
    )

    assert first.allowed is True
    assert blocked.allowed is False
    assert released is True
    assert released_again is False
    assert after_release.allowed is True


@pytest.mark.asyncio
async def test_redis_failure_mode_closed_and_open():
    closed = RedisAdmissionController(
        {"enabled": True, "failure_mode": "closed"},
        redis_client=None,
    )
    open_mode = RedisAdmissionController(
        {"enabled": True, "failure_mode": "open"},
        redis_client=None,
    )
    request = QueryRequest(query="hello", user_id="u1")

    closed_decision = await closed.acquire_route(request)
    open_decision = await open_mode.acquire_route(request)

    assert closed_decision.allowed is False
    assert closed_decision.status_code == 503
    assert closed_decision.error == "admission_unavailable"
    assert open_decision.allowed is True


@pytest.mark.asyncio
async def test_transient_redis_failure_recovers_after_cooldown_without_restart():
    clock = Clock()
    client = RecoveringRedis()
    controller = _controller(
        {"redis": {"recovery_cooldown_ms": 1000}},
        clock=clock,
        redis_client=client,
    )
    request = QueryRequest(query="hello", user_id="u1")

    failed = await controller.acquire_route(request)
    assert failed.status_code == 503
    assert controller.redis_client is client

    client.available = True
    clock.now += 1.1
    recovered = await controller.acquire_route(request)

    assert recovered.allowed is True
    assert client.ping_calls == 1


@pytest.mark.asyncio
async def test_concurrent_recovery_uses_a_single_half_open_ping():
    clock = Clock()
    client = RecoveringRedis()
    controller = _controller(
        {"redis": {"recovery_cooldown_ms": 1000}},
        clock=clock,
        redis_client=client,
    )
    request = QueryRequest(query="hello", user_id="u1")
    await controller.acquire_route(request)
    client.available = True
    clock.now += 1.1

    decisions = await asyncio.gather(
        *(controller.acquire_route(request) for _ in range(100))
    )

    assert all(decision.allowed for decision in decisions)
    assert client.ping_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_mode", "allowed"), [("closed", False), ("open", True)]
)
async def test_pool_exhaustion_obeys_failure_mode(failure_mode, allowed):
    controller = _controller(
        {"failure_mode": failure_mode}, redis_client=PoolExhaustedRedis()
    )

    decision = await controller.acquire_route(QueryRequest(query="hello", user_id="u1"))

    assert decision.allowed is allowed
    if not allowed:
        assert decision.status_code == 503


@pytest.mark.asyncio
async def test_owned_redis_client_uses_bounded_timeouts_and_health_checks(monkeypatch):
    captured = {}

    def build_client(**kwargs):
        captured.update(kwargs)
        return FakeRedis()

    monkeypatch.setattr("src.admission.redis.Redis", build_client)
    controller = RedisAdmissionController({"enabled": True})

    await controller.initialize()

    assert captured["socket_connect_timeout"] == 0.1
    assert captured["socket_timeout"] == 0.1
    assert captured["max_connections"] == 100
    assert captured["health_check_interval"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_error",
    [ConnectionError("redis unavailable"), MaxConnectionsError("pool exhausted")],
)
async def test_injected_client_initialize_failure_recovers_after_cooldown(
    initial_error,
):
    clock = Clock()
    client = RecoveringPingRedis(initial_error)
    controller = _controller(
        {"redis": {"recovery_cooldown_ms": 1000}},
        clock=clock,
        redis_client=client,
    )
    initialize_errors_before = metric_value_for_labels(
        ADMISSION_METRICS.redis_errors, {"operation": "initialize"}
    )
    pool_errors_before = metric_value_for_labels(
        ADMISSION_METRICS.redis_pool_exhaustion,
        {"operation": "initialize", "failure_mode": "closed"},
    )

    await controller.initialize()

    assert controller._redis_available is False
    assert controller._redis_failure_at == clock.now
    assert metric_value_for_labels(
        ADMISSION_METRICS.redis_errors, {"operation": "initialize"}
    ) == (initialize_errors_before + 1)
    expected_pool_increment = 1 if isinstance(initial_error, MaxConnectionsError) else 0
    assert metric_value_for_labels(
        ADMISSION_METRICS.redis_pool_exhaustion,
        {"operation": "initialize", "failure_mode": "closed"},
    ) == (pool_errors_before + expected_pool_increment)

    client.error = None
    clock.now += 1.1
    recovered = await controller.acquire_route(
        QueryRequest(query="hello", user_id="u1")
    )

    assert recovered.allowed is True
    assert controller._redis_available is True
    assert client.ping_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ping", "eval"])
async def test_controller_bounds_blocking_redis_operations(operation):
    client = BlockingRedis()
    controller = _controller({"redis": {"socket_timeout_ms": 100}}, redis_client=client)
    started = time.monotonic()

    if operation == "ping":
        await controller.initialize()
        assert controller._redis_available is False
    else:
        decision = await controller.acquire_route(
            QueryRequest(query="hello", user_id="u1")
        )
        assert decision.allowed is False
        assert decision.status_code == 503

    elapsed = time.monotonic() - started
    assert 0.08 <= elapsed < 0.25


@pytest.mark.asyncio
async def test_blocking_redis_eval_times_out_and_fails_open_when_configured():
    controller = _controller(
        {
            "failure_mode": "open",
            "redis": {"socket_timeout_ms": 100},
        },
        redis_client=BlockingRedis(),
    )
    started = time.monotonic()

    decision = await controller.acquire_route(QueryRequest(query="hello", user_id="u1"))

    assert decision.allowed is True
    assert decision.reason == "fail_open"
    assert time.monotonic() - started < 0.25


@pytest.mark.asyncio
async def test_real_redis_operation_timeout_does_not_include_client_queue_delay():
    class ConnectionBackedRedis:
        connection_pool = object()

    controller = _controller(
        {"redis": {"socket_timeout_ms": 100}},
        redis_client=ConnectionBackedRedis(),
    )
    started = time.monotonic()

    result = await controller._run_redis_operation(
        asyncio.sleep(0.15, result="healthy redis response")
    )

    assert result == "healthy redis response"
    assert time.monotonic() - started >= 0.14
