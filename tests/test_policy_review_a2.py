"""Round-1 review: guardrail clear tombstones (A2)."""

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


# 7. A clear leaves a tombstone that older avoids cannot overwrite.


def _guardrail(action, seconds_ago):
    end_ms = int((NOW - seconds_ago) * 1000)
    return {
        "scope_type": "model",
        "scope_key": "gpt-5",
        "trigger_type": "high_error_rate",
        "guardrail_action": action,
        "window_end_ms": end_ms,
    }


@pytest.mark.asyncio
async def test_an_older_avoid_after_a_clear_does_not_rearm():
    cache = _cache(guardrail_ttl_seconds=180)
    await cache.materialize_routing_guardrail(_guardrail("avoid", 60))
    await cache.materialize_routing_guardrail(_guardrail("clear", 10))

    await cache.materialize_routing_guardrail(_guardrail("avoid", 30))
    cache._local_cache.clear()

    assert (await cache.get_active_guardrails())["blocked_models"] == []
    tombstone = json.loads(
        cache.redis_client.store[cache._guardrail_cache_key("model", "gpt-5")]
    )
    assert tombstone["guardrail_action"] == "clear"
    assert cache.redis_client.ttls[cache._guardrail_cache_key("model", "gpt-5")] == 180

    await cache.materialize_routing_guardrail(_guardrail("avoid", 5))
    cache._local_cache.clear()
    assert (await cache.get_active_guardrails())["blocked_models"] == ["gpt-5"]


@pytest.mark.asyncio
async def test_a_tombstone_reads_as_absent():
    cache = _cache(guardrail_ttl_seconds=180)
    await cache.materialize_routing_guardrail(_guardrail("clear", 10))
    redis = cache.redis_client
    # Even if the index still lists it, a tombstone is not a guardrail.
    redis.sets[cache._guardrail_index_key("model")] = {"gpt-5"}

    guardrails = await cache.get_active_guardrails()

    assert guardrails["blocked_models"] == []
    assert guardrails["warn_models"] == []


@pytest_asyncio.fixture
async def real_redis():
    redis_asyncio = pytest.importorskip("redis.asyncio")
    url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    client = redis_asyncio.Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"real Redis unavailable at {url}: {exc}")
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_tombstone_on_real_redis(real_redis):
    prefix = f"test:policy_tombstone:{uuid.uuid4().hex}"
    cache = RoutingPolicyCache(
        {
            "enabled": True,
            "guardrail_ttl_seconds": 180,
            "redis": {"key_prefix": prefix},
        }
    )
    cache.redis_client = real_redis
    cache._initialized = True
    cache._wall_time_func = lambda: NOW
    key = cache._guardrail_cache_key("model", "gpt-5")
    index_key = cache._guardrail_index_key("model")
    try:
        await cache.materialize_routing_guardrail(_guardrail("avoid", 60))
        await cache.materialize_routing_guardrail(_guardrail("clear", 10))
        await cache.materialize_routing_guardrail(_guardrail("avoid", 30))

        stored = json.loads(await real_redis.get(key))
        assert stored["guardrail_action"] == "clear"
        assert 175 <= await real_redis.ttl(key) <= 180
        assert await real_redis.smembers(index_key) == set()

        await cache.materialize_routing_guardrail(_guardrail("avoid", 5))
        assert json.loads(await real_redis.get(key))["guardrail_action"] == "avoid"
        assert await real_redis.smembers(index_key) == {"gpt-5"}
    finally:
        await real_redis.delete(key, index_key)
