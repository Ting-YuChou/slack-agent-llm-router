"""Replayed policy events neither re-arm expired state nor overwrite newer (A6)."""

import json
import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from flink import logic
from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import RoutingPolicyCache
from src.utils.metrics import metric_value_for_labels

NOW = 1_800_000_000.0  # wall-clock seconds the tests run at


def _iso(seconds_ago):
    return datetime.fromtimestamp(NOW - seconds_ago, tz=timezone.utc).isoformat()


def _ms(seconds_ago):
    return int((NOW - seconds_ago) * 1000)


def _stale_count(reason):
    return metric_value_for_labels(policy.POLICY_CACHE_STALE_WRITES, {"reason": reason})


class _CasRedis:
    """In-memory Redis that runs the cache's compare-and-set scripts."""

    def __init__(self):
        self.store = {}
        self.ttls = {}
        self.sets = {}
        self.setex_calls = []
        self.eval_calls = []

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl))
        self.store[key] = value
        self.ttls[key] = ttl

    async def eval(self, script, numkeys, *keys_and_args):
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        self.eval_calls.append((keys, args))
        assert script == policy.CAS_SET_SCRIPT
        payload, ttl, event_ms = args[0], int(args[1]), int(args[2])
        current = self.store.get(keys[0])
        if current is not None:
            stored = json.loads(current).get("event_time_ms")
            if stored is not None and int(stored) > event_ms:
                return 0
        self.store[keys[0]] = payload
        self.ttls[keys[0]] = ttl
        if len(keys) > 1:
            self.sets.setdefault(keys[1], set()).add(args[3])
        return 1

    def pipeline(self, transaction=False):
        redis = self

        class _Pipeline:
            def __init__(self):
                self.commands = []

            def setex(self, key, ttl, value):
                self.commands.append(("setex", key, ttl, value))
                return self

            def sadd(self, key, member):
                self.commands.append(("sadd", key, member))
                return self

            async def execute(self):
                for command in self.commands:
                    if command[0] == "setex":
                        redis.setex_calls.append((command[1], command[2]))
                        redis.store[command[1]] = command[3]
                    else:
                        redis.sets.setdefault(command[1], set()).add(command[2])
                return [True] * len(self.commands)

        return _Pipeline()


def _redis_cache(**config):
    cache = RoutingPolicyCache({"enabled": True, **config})
    cache.redis_client = _CasRedis()
    cache._initialized = True
    cache._wall_time_func = lambda: NOW
    return cache


def _guardrail(seconds_ago, action="avoid", **extra):
    return {
        "event_type": "routing.guardrails",
        "scope_type": "provider",
        "scope_key": "openai",
        "guardrail_action": action,
        "window_start_ms": _ms(seconds_ago + 60),
        "window_end_ms": _ms(seconds_ago),
        "emitted_at": _iso(0),
        **extra,
    }


def _policy_state(seconds_ago, avoid_models):
    return {
        "event_type": "routing.policy_state",
        "scope_type": "user",
        "scope_key": "u1",
        "avoid_models": avoid_models,
        "latest_event_at": _iso(seconds_ago),
        # Flink stamps emitted_at at processing time, so a replay looks new.
        "emitted_at": _iso(0),
    }


@pytest.mark.asyncio
async def test_expired_guardrail_replay_is_not_written():
    cache = _redis_cache(guardrail_ttl_seconds=180)
    before = _stale_count("expired")

    await cache.materialize_routing_guardrail(_guardrail(seconds_ago=3600))

    assert cache.redis_client.store == {}
    assert cache.redis_client.sets == {}
    assert _stale_count("expired") == before + 1
    assert (await cache.get_active_guardrails())["blocked_providers"] == []


@pytest.mark.asyncio
async def test_guardrail_ttl_is_reduced_by_its_age():
    cache = _redis_cache(guardrail_ttl_seconds=180)

    await cache.materialize_routing_guardrail(_guardrail(seconds_ago=60))

    key = cache._guardrail_cache_key("provider", "openai")
    assert cache.redis_client.ttls[key] == 120
    assert cache.redis_client.sets[cache._guardrail_index_key("provider")] == {"openai"}


@pytest.mark.asyncio
async def test_older_guardrail_does_not_overwrite_a_newer_one():
    cache = _redis_cache(guardrail_ttl_seconds=180)
    await cache.materialize_routing_guardrail(_guardrail(seconds_ago=10))
    before = _stale_count("older")

    await cache.materialize_routing_guardrail(_guardrail(seconds_ago=70, action="warn"))

    key = cache._guardrail_cache_key("provider", "openai")
    assert json.loads(cache.redis_client.store[key])["guardrail_action"] == "avoid"
    assert _stale_count("older") == before + 1


@pytest.mark.asyncio
async def test_older_policy_state_does_not_overwrite_newer_state():
    cache = _redis_cache(user_ttl_seconds=900)
    await cache.materialize_routing_policy_state(_policy_state(5, ["gpt-5"]))

    # Flink restarts and replays an older window with a fresh emitted_at.
    await cache.materialize_routing_policy_state(_policy_state(300, ["mistral"]))

    stored = json.loads(cache.redis_client.store[cache._user_state_cache_key("u1")])
    assert stored["avoid_models"] == ["gpt-5"]
    assert cache.redis_client.ttls[cache._user_state_cache_key("u1")] == 895


@pytest.mark.asyncio
async def test_expired_policy_state_is_not_rearmed():
    cache = _redis_cache(user_ttl_seconds=900)

    await cache.materialize_routing_policy_state(_policy_state(3600, ["gpt-5"]))

    assert cache.redis_client.store == {}


@pytest.mark.asyncio
async def test_request_policy_replay_past_its_ttl_is_dropped():
    cache = _redis_cache(request_ttl_seconds=300)

    await cache.materialize_request_enriched(
        {"request_id": "r1", "user_id": "u1", "emitted_at": _iso(900)}
    )

    assert cache.redis_client.store == {}


@pytest.mark.asyncio
async def test_events_without_a_time_are_still_written():
    cache = _redis_cache(user_ttl_seconds=900, guardrail_ttl_seconds=180)

    await cache.materialize_routing_policy_state(
        {"scope_type": "user", "scope_key": "u1", "avoid_models": ["gpt-5"]}
    )
    await cache.materialize_routing_guardrail(
        {"scope_type": "model", "scope_key": "gpt-5", "guardrail_action": "avoid"}
    )

    redis = cache.redis_client
    assert (cache._user_state_cache_key("u1"), 900) in redis.setex_calls
    assert (cache._guardrail_cache_key("model", "gpt-5"), 180) in redis.setex_calls
    assert redis.eval_calls == []


@pytest.mark.asyncio
async def test_disabled_cache_keeps_old_events_in_l1():
    cache = RoutingPolicyCache({"enabled": False, "local_cache_ttl_seconds": 30})
    await cache.initialize()

    await cache.materialize_routing_guardrail(
        {
            "scope_type": "model",
            "scope_key": "gpt-5",
            "guardrail_action": "avoid",
            "window_end_ms": 1_000,
            "emitted_at": "2020-01-01T00:00:00Z",
        }
    )

    assert (await cache.get_active_guardrails())["blocked_models"] == ["gpt-5"]


def test_policy_state_event_carries_its_latest_event_time():
    event = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="u1",
        events=[
            {
                "user_id": "u1",
                "user_tier": "free",
                "query_type": "general",
                "selected_model": "gpt-5",
                "provider": "openai",
                "status": "success",
                "latency_ms": 100,
                "total_tokens": 10,
                "event_timestamp_ms": 1_776_000_000_000,
            }
        ],
    )

    assert (
        event["latest_event_at"]
        == datetime.fromtimestamp(1_776_000_000, tz=timezone.utc).isoformat()
    )


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
async def test_compare_and_set_script_on_real_redis(real_redis):
    prefix = f"test:policy_replay:{uuid.uuid4().hex}"
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
    key = cache._guardrail_cache_key("provider", "openai")
    index_key = cache._guardrail_index_key("provider")
    try:
        await cache.materialize_routing_guardrail(_guardrail(seconds_ago=10))
        await cache.materialize_routing_guardrail(
            _guardrail(seconds_ago=70, action="warn")
        )

        assert json.loads(await real_redis.get(key))["guardrail_action"] == "avoid"
        assert 165 <= await real_redis.ttl(key) <= 170
        assert await real_redis.smembers(index_key) == {"openai"}
    finally:
        await real_redis.delete(key, index_key)
