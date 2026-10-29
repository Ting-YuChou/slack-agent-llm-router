"""Round-1 review: bounded policy Redis and materializer retries (R10)."""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiokafka.structs import TopicPartition
from redis.exceptions import ConnectionError as RedisConnectionError

from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import (
    PolicyCacheUnavailableError,
    PolicyMaterializer,
    RoutingPolicyCache,
)
from src.utils.metrics import PIPELINE_METRICS, metric_value_for_labels

NOW = 1_800_000_000.0


def _iso(seconds_ago):
    return datetime.fromtimestamp(NOW - seconds_ago, tz=timezone.utc).isoformat()


def _count(metric, **labels):
    return metric_value_for_labels(metric, labels)


class _Redis:
    """In-memory Redis running the cache's Lua scripts by identity."""

    def __init__(self):
        self.store = {}
        self.ttls = {}
        self.sets = {}
        self.gets = 0
        self.fail_with = None

    def _check(self):
        if self.fail_with is not None:
            raise self.fail_with

    async def get(self, key):
        self._check()
        self.gets += 1
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self._check()
        self.store[key] = value
        self.ttls[key] = ttl

    async def mget(self, keys):
        self._check()
        return [self.store.get(key) for key in keys]

    async def ping(self):
        self._check()
        return True

    def _stored_time(self, key):
        current = self.store.get(key)
        if current is None:
            return None
        return json.loads(current).get("event_time_ms")

    async def eval(self, script, numkeys, *keys_and_args):
        self._check()
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        stored = self._stored_time(keys[0])
        if script == policy.CAS_SET_SCRIPT:
            if stored is not None and int(stored) > int(args[2]):
                return 0
            self.store[keys[0]] = args[0]
            self.ttls[keys[0]] = int(args[1])
            if len(keys) > 1:
                self.sets.setdefault(keys[1], set()).add(args[3])
            return 1
        if script == policy.CAS_CLEAR_SCRIPT:
            if stored is not None and int(stored) > int(args[0]):
                return 0
            self.store[keys[0]] = args[2]
            self.ttls[keys[0]] = int(args[3])
            self.sets.get(keys[1], set()).discard(args[1])
            return 1
        raise AssertionError("unknown script")

    def pipeline(self, transaction=False):
        redis = self

        class _Pipeline:
            def __init__(self):
                self.commands = []

            def get(self, key):
                self.commands.append(("get", key))
                return self

            def smembers(self, key):
                self.commands.append(("smembers", key))
                return self

            def srem(self, key, *members):
                self.commands.append(("srem", key, members))
                return self

            async def execute(self):
                redis._check()
                results = []
                for command in self.commands:
                    if command[0] == "get":
                        redis.gets += 1
                        results.append(redis.store.get(command[1]))
                    elif command[0] == "smembers":
                        results.append(set(redis.sets.get(command[1], set())))
                    else:
                        for member in command[2]:
                            redis.sets.get(command[1], set()).discard(member)
                        results.append(len(command[2]))
                return results

        return _Pipeline()


def _cache(**config):
    cache = RoutingPolicyCache({"enabled": True, **config})
    cache.redis_client = _Redis()
    cache._initialized = True
    cache._wall_time_func = lambda: NOW
    return cache


# 1. A router deadline is not a Redis failure; writes retry in place.


@pytest.mark.asyncio
async def test_a_router_timeout_does_not_start_a_cooldown():
    cache = _cache()
    before = _count(policy.POLICY_CACHE_UNAVAILABLE, reason="timeout")

    cache.record_read_timeout()

    assert _count(policy.POLICY_CACHE_UNAVAILABLE, reason="timeout") == before + 1
    assert cache.is_degraded() is False
    result = await cache.get_effective_policy("r1", "u1")
    assert result.get("policy_source") != "unavailable"


@pytest.mark.asyncio
async def test_a_failed_write_raises_policy_cache_unavailable():
    cache = _cache()
    cache.redis_client.fail_with = RedisConnectionError("connection reset")

    with pytest.raises(PolicyCacheUnavailableError, match="connection reset"):
        await cache.materialize_request_enriched({"request_id": "r1"})


class _Consumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


def _guardrail_message(offset=3):
    return SimpleNamespace(
        value={"scope_type": "model", "scope_key": "m"},
        topic="routing.guardrails",
        partition=0,
        offset=offset,
    )


@pytest.mark.asyncio
async def test_an_unavailable_cache_is_retried_without_restarting_the_consumer():
    seen_degraded = []
    calls = []

    async def materialize(_event):
        calls.append(1)
        if len(calls) == 1:
            raise PolicyCacheUnavailableError("cooldown")
        seen_degraded.append(materializer.is_degraded())

    cache = SimpleNamespace(
        config={
            "consumer": {
                "enable_auto_commit": False,
                "redis_retry_initial_backoff_seconds": 0.001,
            }
        },
        enabled=True,
        materialize_routing_guardrail=materialize,
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    consumer = _Consumer([_guardrail_message()])

    await materializer._consume_routing_guardrails(consumer)

    assert len(calls) == 2
    assert seen_degraded == [True]
    assert materializer.is_degraded() is False
    consumer.commit.assert_awaited_once_with(
        {TopicPartition("routing.guardrails", 0): 4}
    )


@pytest.mark.asyncio
async def test_shutdown_during_a_redis_retry_leaves_the_offset_uncommitted():
    async def materialize(_event):
        materializer.running = False
        raise PolicyCacheUnavailableError("cooldown")

    cache = SimpleNamespace(
        config={
            "consumer": {
                "enable_auto_commit": False,
                "redis_retry_initial_backoff_seconds": 0.001,
            }
        },
        enabled=True,
        materialize_routing_guardrail=materialize,
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    consumer = _Consumer([_guardrail_message()])

    await materializer._consume_routing_guardrails(consumer)

    consumer.commit.assert_not_awaited()


# 2. Pool exhaustion is load, not an outage.


@pytest.mark.asyncio
async def test_pool_exhaustion_does_not_start_a_cooldown():
    cache = _cache()
    cache.redis_client.fail_with = RedisConnectionError("Too many connections")
    before = _count(policy.POLICY_CACHE_UNAVAILABLE, reason="pool_exhausted")

    result = await cache.get_effective_policy("r1", "u1")

    assert result["policy_source"] == "unavailable"
    assert _count(policy.POLICY_CACHE_UNAVAILABLE, reason="pool_exhausted") == (
        before + 1
    )
    assert cache.is_degraded() is False
    cache.redis_client.fail_with = None
    again = await cache.get_effective_policy("r2", "u1")
    assert again.get("policy_source") != "unavailable"


# 9. Readiness: degraded, not unhealthy, while Redis is down.


def test_materializer_is_degraded_but_healthy_while_the_cache_is_degraded():
    cache = RoutingPolicyCache({"enabled": True})
    cache._initialized = True
    cache._mark_redis_failure("initialize", ConnectionError("down"))
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.consumers = {"routing_guardrails": object()}
    materializer.consumer_task_status = {"routing_guardrails": {"last_error": None}}

    assert materializer.is_degraded() is True
    assert materializer.is_healthy() is True
    assert materializer.get_health_status()["degraded"] is True


def test_materializer_without_a_degraded_cache_is_not_degraded():
    cache = SimpleNamespace(config={}, enabled=True)
    materializer = PolicyMaterializer({"topics": {}}, cache)

    assert materializer.is_degraded() is False


# 10. Materializer restarts have their own metric.


@pytest.mark.asyncio
async def test_materializer_restarts_do_not_touch_the_sink_restart_metric():
    cache = SimpleNamespace(
        config={"consumer": {"supervisor_initial_backoff_seconds": 0}},
        enabled=True,
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    runs = []

    async def run_consumer(topic_key, consumer):
        runs.append(topic_key)
        if len(runs) == 1:
            raise RuntimeError("boom")
        materializer.running = False

    async def restart_consumer(topic_key):
        return object()

    materializer._run_consumer = run_consumer
    materializer._restart_consumer = restart_consumer
    sink_before = _count(PIPELINE_METRICS.consumer_restarts, topic="routing_guardrails")
    own_before = _count(
        policy.POLICY_MATERIALIZER_CONSUMER_RESTARTS, topic="routing_guardrails"
    )

    await asyncio.wait_for(
        materializer._supervise_consumer("routing_guardrails", object()), 5
    )

    assert (
        _count(PIPELINE_METRICS.consumer_restarts, topic="routing_guardrails")
        == sink_before
    )
    assert (
        _count(policy.POLICY_MATERIALIZER_CONSUMER_RESTARTS, topic="routing_guardrails")
        == own_before + 1
    )
