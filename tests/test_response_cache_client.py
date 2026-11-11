"""CH6: the response cache's Redis client is bounded and times out."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part2_inference import ResponseCache
from src.utils.metrics import INFERENCE_METRICS

REDIS = "src.llm_router_part2_inference.redis.Redis"


def _client():
    client = MagicMock(name="redis_client")
    client.ping = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_the_cache_client_is_bounded_and_times_out(monkeypatch):
    factory = MagicMock(return_value=_client())
    monkeypatch.setattr(REDIS, factory)
    monkeypatch.setenv("CACHE_REDIS_PASSWORD", "secret")
    cache = ResponseCache(
        {
            "redis": {
                "host": "redis",
                "port": 6380,
                "db": 4,
                "password_env": "CACHE_REDIS_PASSWORD",
            }
        }
    )

    await cache.initialize()

    kwargs = factory.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 0.25
    assert kwargs["socket_timeout"] == 0.25
    assert kwargs["max_connections"] == 100
    assert kwargs["health_check_interval"] == 30
    # The connection target is unchanged.
    assert kwargs["host"] == "redis"
    assert kwargs["port"] == 6380
    assert kwargs["db"] == 4
    assert kwargs["password"] == "secret"
    assert kwargs["decode_responses"] is True
    assert cache.enabled is True


@pytest.mark.asyncio
async def test_timeouts_and_pool_size_are_configurable(monkeypatch):
    factory = MagicMock(return_value=_client())
    monkeypatch.setattr(REDIS, factory)
    cache = ResponseCache(
        {
            "redis": {
                "connect_timeout_ms": 400,
                "socket_timeout_ms": 50,
                "max_connections": 8,
            }
        }
    )

    await cache.initialize()

    kwargs = factory.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 0.4
    assert kwargs["socket_timeout"] == 0.05
    assert kwargs["max_connections"] == 8


@pytest.mark.asyncio
async def test_a_timed_out_get_is_a_miss():
    cache = ResponseCache({})
    cache.redis_client = MagicMock()
    cache.redis_client.get = AsyncMock(side_effect=TimeoutError("read timed out"))
    misses = INFERENCE_METRICS.cache_misses._value.get()
    hits = INFERENCE_METRICS.cache_hits._value.get()

    assert await cache.get_cached_response("key") is None

    assert INFERENCE_METRICS.cache_misses._value.get() == misses + 1
    assert INFERENCE_METRICS.cache_hits._value.get() == hits


@pytest.mark.asyncio
async def test_a_timed_out_setex_is_swallowed():
    cache = ResponseCache({})
    cache.redis_client = MagicMock()
    cache.redis_client.setex = AsyncMock(side_effect=TimeoutError("write timed out"))

    await cache.cache_response("key", {"response_text": "hi"})

    cache.redis_client.setex.assert_awaited_once()


# --- review: outages, slow starts and bad settings ------------------------------


def _failing(method, error):
    cache = ResponseCache({})
    cache.redis_client = MagicMock()
    cache.redis_client.get = AsyncMock(return_value=None)
    cache.redis_client.setex = AsyncMock()
    getattr(cache.redis_client, method).side_effect = error
    clock = {"now": 100.0}
    cache._clock = lambda: clock["now"]
    return cache, clock


@pytest.mark.asyncio
async def test_after_an_error_the_cache_is_left_alone_for_a_cooldown():
    cache, clock = _failing("get", TimeoutError("connect timed out"))

    assert await cache.get_cached_response("key") is None
    # An unreachable Redis must not cost every request two more timeouts.
    assert await cache.get_cached_response("key") is None
    await cache.cache_response("key", {"response_text": "hi"})
    assert cache.redis_client.get.await_count == 1
    cache.redis_client.setex.assert_not_awaited()

    clock["now"] += 5.0
    cache.redis_client.get.side_effect = None
    assert await cache.get_cached_response("key") is None
    await cache.cache_response("key", {"response_text": "hi"})
    assert cache.redis_client.get.await_count == 2
    cache.redis_client.setex.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_write_also_starts_the_cooldown():
    cache, _clock = _failing("setex", ConnectionError("Connection refused"))

    await cache.cache_response("key", {"response_text": "hi"})
    assert await cache.get_cached_response("key") is None

    cache.redis_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_skipped_lookup_still_counts_as_a_miss():
    cache, _clock = _failing("get", TimeoutError("read timed out"))
    await cache.get_cached_response("key")
    misses = INFERENCE_METRICS.cache_misses._value.get()

    await cache.get_cached_response("key")

    assert INFERENCE_METRICS.cache_misses._value.get() == misses + 1


@pytest.mark.asyncio
async def test_a_full_pool_does_not_switch_the_cache_off():
    cache, _clock = _failing("get", ConnectionError("Too many connections"))

    await cache.get_cached_response("key")
    await cache.get_cached_response("key")

    assert cache.redis_client.get.await_count == 2


@pytest.mark.asyncio
async def test_the_cooldown_is_configurable():
    cache, clock = _failing("get", TimeoutError("read timed out"))
    cache.redis_config["error_cooldown_ms"] = 1000

    await cache.get_cached_response("key")
    clock["now"] += 1.0
    await cache.get_cached_response("key")

    assert cache.redis_client.get.await_count == 2


@pytest.mark.asyncio
async def test_a_slow_first_ping_is_retried_before_the_cache_is_given_up(monkeypatch):
    client = _client()
    client.ping = AsyncMock(side_effect=[TimeoutError("slow start"), True])
    monkeypatch.setattr(REDIS, MagicMock(return_value=client))
    monkeypatch.setattr("src.llm_router_part2_inference.asyncio.sleep", AsyncMock())
    cache = ResponseCache({})

    await cache.initialize()

    assert cache.enabled is True
    assert client.ping.await_count == 2


@pytest.mark.asyncio
async def test_a_redis_that_never_answers_at_startup_disables_the_cache(monkeypatch):
    client = _client()
    client.ping = AsyncMock(side_effect=TimeoutError("down"))
    monkeypatch.setattr(REDIS, MagicMock(return_value=client))
    monkeypatch.setattr("src.llm_router_part2_inference.asyncio.sleep", AsyncMock())
    cache = ResponseCache({})

    await cache.initialize()

    assert cache.enabled is False
    assert client.ping.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -5, None, "fast", True])
async def test_an_unusable_setting_falls_back_to_its_default(monkeypatch, bad, caplog):
    factory = MagicMock(return_value=_client())
    monkeypatch.setattr(REDIS, factory)
    cache = ResponseCache(
        {
            "redis": {
                "connect_timeout_ms": bad,
                "socket_timeout_ms": bad,
                "max_connections": bad,
            }
        }
    )

    with caplog.at_level("WARNING"):
        await cache.initialize()

    kwargs = factory.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 0.25
    assert kwargs["socket_timeout"] == 0.25
    assert kwargs["max_connections"] == 100
    assert cache.enabled is True
    assert "socket_timeout_ms" in caplog.text
