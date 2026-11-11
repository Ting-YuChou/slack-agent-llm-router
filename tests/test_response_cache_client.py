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
    assert kwargs["socket_connect_timeout"] == 1.0
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
