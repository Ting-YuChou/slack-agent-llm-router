"""The policy cache's Redis is bounded, recovers and reports degraded (R10)."""

import asyncio
import time

import pytest

from src import llm_router_part3_policy as policy
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache
from src.utils.metrics import metric_value_for_labels
from src.utils.schema import PolicyCacheConfig, QueryRequest, UserTier


def _unavailable_count(reason):
    return metric_value_for_labels(policy.POLICY_CACHE_UNAVAILABLE, {"reason": reason})


class _HangingPipeline:
    def get(self, _key):
        return self

    def smembers(self, _key):
        return self

    async def execute(self):
        await asyncio.Event().wait()


class _HangingRedis:
    """A Redis black hole: commands are sent and never answered."""

    def __init__(self):
        self.pings = 0

    def pipeline(self, transaction=False):
        return _HangingPipeline()

    async def get(self, _key):
        await asyncio.Event().wait()

    async def ping(self):
        self.pings += 1
        await asyncio.Event().wait()


class _FlakyRedis:
    """Fails ping until told Redis is back, then answers empty reads."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.up = False
        self.pings = 0
        _FlakyRedis.instances.append(self)

    async def ping(self):
        self.pings += 1
        if not self.up:
            raise ConnectionError("redis down")
        return True

    def pipeline(self, transaction=False):
        redis = self

        class _Pipeline:
            def __init__(self):
                self.count = 0

            def get(self, _key):
                self.count += 1
                return self

            def smembers(self, _key):
                self.count += 1
                return self

            async def execute(self):
                if not redis.up:
                    raise ConnectionError("redis down")
                return [None] * self.count

        return _Pipeline()

    async def mget(self, keys):
        return [None for _ in keys]


def _router(policy_cache):
    return ModelRouter(
        {
            "default_model": "m1",
            "routing_strategy": "intelligent",
            "classifier": {"semantic_enabled": False},
            "models": {
                "m1": {
                    "provider": "openai",
                    "max_tokens": 8000,
                    "cost_per_token": 0.0,
                    "priority": 3,
                    "allowed_tiers": ["free", "premium", "enterprise"],
                    "capabilities": ["general"],
                }
            },
            "routing_rules": [],
        },
        policy_cache=policy_cache,
    )


def _request():
    return QueryRequest(query="hello", user_id="u1", user_tier=UserTier.PREMIUM)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.asyncio
async def test_routing_does_not_wait_on_a_redis_black_hole():
    cache = RoutingPolicyCache({"enabled": True})
    cache.redis_client = _HangingRedis()
    cache._initialized = True
    router = _router(cache)
    before = _unavailable_count("timeout")

    started = time.perf_counter()
    context = await router._get_policy_context(_request())
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert context["policy_source"] == "unavailable"
    assert context["blocked_models"] == []
    assert _unavailable_count("timeout") == before + 1


@pytest.mark.asyncio
async def test_two_argument_policy_stub_is_bounded_too():
    class _SlowTwoArgCache:
        async def get_effective_policy(self, request_id, user_id):
            await asyncio.Event().wait()

    router = _router(_SlowTwoArgCache())

    started = time.perf_counter()
    context = await router._get_policy_context(_request())

    assert time.perf_counter() - started < 0.1
    assert context["policy_source"] == "unavailable"


@pytest.mark.asyncio
async def test_failed_initialization_recovers_after_the_cooldown(monkeypatch):
    _FlakyRedis.instances.clear()
    monkeypatch.setattr(policy.redis, "Redis", _FlakyRedis)
    clock = _Clock()
    cache = RoutingPolicyCache({"enabled": True, "recovery_cooldown_ms": 1000})
    cache._time_func = clock

    await cache.initialize()

    client = _FlakyRedis.instances[0]
    assert cache.enabled is True
    assert cache.get_health_status()["status"] == "degraded"
    assert cache.is_degraded() is True
    # Inside the cooldown no ping is sent; the policy is reported unavailable.
    before = _unavailable_count("cooldown")
    result = await cache.get_effective_policy("r1", "u1")
    assert result["policy_source"] == "unavailable"
    assert client.pings == 1
    assert _unavailable_count("cooldown") == before + 1

    client.up = True
    clock.now += 1.5
    recovered = await cache.get_effective_policy("r2", "u1")

    assert client.pings == 2
    assert recovered.get("policy_source") != "unavailable"
    assert cache.is_degraded() is False
    assert cache.get_health_status()["status"] == "healthy"


@pytest.mark.asyncio
async def test_read_failure_marks_redis_unavailable_and_reports_it():
    clock = _Clock()
    cache = RoutingPolicyCache({"enabled": True})
    cache._time_func = clock
    flaky = _FlakyRedis()
    flaky.up = True
    cache.redis_client = flaky
    cache._initialized = True
    assert (await cache.get_effective_policy("r1", "u1")).get(
        "policy_source"
    ) != "unavailable"

    flaky.up = False
    before = _unavailable_count("redis_error")
    result = await cache.get_effective_policy("r2", "u2")

    assert result["policy_source"] == "unavailable"
    assert _unavailable_count("redis_error") == before + 1
    status = cache.get_health_status()
    assert status["status"] == "degraded"
    assert "redis down" in status["last_error"]


@pytest.mark.asyncio
async def test_healthy_redis_sends_no_recovery_ping():
    cache = RoutingPolicyCache({"enabled": True})
    flaky = _FlakyRedis()
    flaky.up = True
    cache.redis_client = flaky
    cache._initialized = True

    for index in range(5):
        await cache.get_effective_policy(f"r{index}", "u1", "s1")

    assert flaky.pings == 0


@pytest.mark.asyncio
async def test_policy_redis_client_is_bounded(monkeypatch):
    _FlakyRedis.instances.clear()
    monkeypatch.setattr(policy.redis, "Redis", _FlakyRedis)
    cache = RoutingPolicyCache({"enabled": True, "redis": {"host": "redis"}})

    await cache.initialize()

    kwargs = _FlakyRedis.instances[0].kwargs
    assert kwargs["socket_connect_timeout"] == pytest.approx(0.1)
    assert kwargs["socket_timeout"] == pytest.approx(0.1)
    assert kwargs["max_connections"] > 0
    assert kwargs["health_check_interval"] > 0


@pytest.mark.asyncio
async def test_materializer_keeps_consuming_after_a_cache_initialization_failure(
    monkeypatch,
):
    _FlakyRedis.instances.clear()
    monkeypatch.setattr(policy.redis, "Redis", _FlakyRedis)

    class _Consumer:
        def __init__(self, topic, **kwargs):
            self.topic = topic

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(policy, "AIOKafkaConsumer", _Consumer)
    cache = RoutingPolicyCache({"enabled": True})
    materializer = PolicyMaterializer(
        {"topics": {"routing_guardrails": "routing.guardrails"}}, cache
    )

    await materializer.initialize()

    assert set(materializer.consumers) == {"routing_guardrails"}
    status = materializer.get_health_status()
    assert status["degraded"] is True
    assert status["policy_cache"]["status"] == "degraded"


def test_policy_cache_schema_has_bounded_defaults():
    config = PolicyCacheConfig()

    assert config.read_timeout_ms == 50
    assert config.recovery_cooldown_ms == 1000
