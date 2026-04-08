from src.llm_router_part3_policy import RoutingPolicyCache


import pytest


@pytest.mark.asyncio
async def test_policy_cache_materializes_request_and_user_fast_lane_hints():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "request_ttl_seconds": 300,
            "user_ttl_seconds": 900,
            "local_cache_ttl_seconds": 30,
        }
    )
    await cache.initialize()

    await cache.materialize_request_enriched(
        {
            "event_type": "requests.enriched",
            "request_id": "req-1",
            "user_id": "user-1",
            "priority": "critical",
            "route_to_fast_lane": True,
            "emitted_at": "2026-04-08T12:00:00Z",
        }
    )

    await cache.materialize_fast_lane_hint(
        {
            "event_type": "fast_lane_hints",
            "request_id": "req-1",
            "user_id": "user-1",
            "priority": "critical",
            "route_to_fast_lane": True,
            "selected_model": "mistral-7b",
            "hint_reason": "priority=critical",
            "emitted_at": "2026-04-08T12:00:01Z",
        }
    )

    request_policy = await cache.get_request_policy("req-1")
    user_policy = await cache.get_user_policy("user-1")
    effective_policy = await cache.get_effective_policy("req-1", "user-1")

    assert request_policy["preferred_models"] == ["mistral-7b"]
    assert user_policy["route_to_fast_lane"] is True
    assert user_policy["preferred_models"] == ["mistral-7b"]
    assert effective_policy["policy_source"] == "request+user"


@pytest.mark.asyncio
async def test_policy_cache_does_not_promote_user_on_non_fast_lane_event():
    cache = RoutingPolicyCache({"enabled": False, "local_cache_ttl_seconds": 30})
    await cache.initialize()

    await cache.materialize_request_enriched(
        {
            "event_type": "requests.enriched",
            "request_id": "req-2",
            "user_id": "user-2",
            "priority": "low",
            "route_to_fast_lane": False,
            "emitted_at": "2026-04-08T12:05:00Z",
        }
    )

    request_policy = await cache.get_request_policy("req-2")
    user_policy = await cache.get_user_policy("user-2")

    assert request_policy["route_to_fast_lane"] is False
    assert user_policy is None
