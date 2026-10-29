"""Round-1 review: materializer DLQ (A7)."""

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


# 8. The DLQ redacts errors and honours topic overrides.


class _Consumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


class _DlqProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, key=None, value=None):
        self.sent.append((topic, value))


def _poison(topic="routing.policy_state"):
    return SimpleNamespace(
        value={
            "scope_type": "user",
            "scope_key": "u1",
            "recent_error_rate": "api_key=sk-abcdefghijklmnopqrstuvwxyz0123456789",
        },
        topic=topic,
        partition=0,
        offset=1,
    )


def _dlq_materializer(kafka_config):
    cache = RoutingPolicyCache(
        {"enabled": False, "consumer": {"enable_auto_commit": False}}
    )
    materializer = PolicyMaterializer(kafka_config, cache)
    materializer.running = True
    materializer.dlq_producer = _DlqProducer()
    return materializer


@pytest.mark.asyncio
async def test_dlq_error_message_is_redacted():
    materializer = _dlq_materializer({"topics": {}})

    await materializer._consume_routing_policy_state(_Consumer([_poison()]))

    ((_topic, payload),) = materializer.dlq_producer.sent
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in payload["error_message"]
    assert "recent_error_rate" in payload["error_message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kafka_config", "expected"),
    [
        (
            {"topics": {}, "dead_letter_topics": {"routing_policy_state": "ops.dlq"}},
            "ops.dlq",
        ),
        (
            {"topics": {"routing_policy_state_dlq": "policy-state-dead"}},
            "policy-state-dead",
        ),
        ({"topics": {}, "dlq": {"topic_suffix": "-dlq"}}, "routing.policy_state-dlq"),
    ],
)
async def test_dlq_topic_honours_overrides(kafka_config, expected):
    materializer = _dlq_materializer(kafka_config)

    await materializer._consume_routing_policy_state(_Consumer([_poison()]))

    ((topic, _payload),) = materializer.dlq_producer.sent
    assert topic == expected
