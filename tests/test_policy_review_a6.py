"""Round-1 review: request policy keeps both sources (A6)."""

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


# 3. requests.enriched and fast_lane_hints both survive, in either order.


def _enriched(seconds_ago):
    return {
        "request_id": "r1",
        "user_id": "u1",
        "priority": "critical",
        "avoid_models": ["gpt-x"],
        "avoid_providers": ["anthropic"],
        "emitted_at": _iso(seconds_ago),
    }


def _hint(seconds_ago):
    return {
        "request_id": "r1",
        "user_id": "u1",
        "selected_model": "mistral-7b",
        "route_to_fast_lane": True,
        "emitted_at": _iso(seconds_ago),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order", ["enriched-then-older-hint", "hint-then-older-enriched"]
)
async def test_request_policy_keeps_both_sources(order):
    cache = _cache()
    if order == "enriched-then-older-hint":
        await cache.materialize_request_enriched(_enriched(1))
        await cache.materialize_fast_lane_hint(_hint(5))
    else:
        await cache.materialize_fast_lane_hint(_hint(1))
        await cache.materialize_request_enriched(_enriched(5))
    cache._local_cache.clear()

    request_policy = await cache.get_request_policy("r1")
    effective = await cache.get_effective_policy("r1", "u1")

    for merged in (request_policy, effective):
        assert merged["preferred_models"] == ["mistral-7b"]
        assert merged["avoid_models"] == ["gpt-x"]
        assert merged["avoid_providers"] == ["anthropic"]
        assert merged["priority"] == "critical"
        assert merged["route_to_fast_lane"] is True
    assert effective["policy_source"] == "request"


@pytest.mark.asyncio
async def test_a_fast_lane_hint_write_does_not_read_first():
    cache = _cache()
    await cache.materialize_request_enriched(_enriched(1))
    cache._local_cache.clear()
    cache.redis_client.gets = 0

    await cache.materialize_fast_lane_hint(_hint(0))

    assert cache.redis_client.gets == 0
