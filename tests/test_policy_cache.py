from src.llm_router_part3_policy import RoutingPolicyCache


import pytest
from unittest.mock import AsyncMock, MagicMock


def test_policy_l1_is_bounded_and_guardrail_index_tracks_eviction():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "local_cache_ttl_seconds": 30,
            "local_cache_max_entries": 2,
        }
    )
    cache._local_guardrail_index["model"].update({"m1", "m2"})
    cache._local_set(cache._guardrail_cache_key("model", "m1"), {"id": 1}, 30)
    cache._local_set(cache._guardrail_cache_key("model", "m2"), {"id": 2}, 30)
    cache._local_set(cache._cache_key("request", "r1"), {"id": 3}, 30)

    assert len(cache._local_cache) == 2
    assert "m1" not in cache._local_guardrail_index["model"]
    assert cache._local_get(cache._guardrail_cache_key("model", "m1")) is None


@pytest.mark.asyncio
async def test_redis_guardrails_do_not_accumulate_redundant_local_index():
    cache = RoutingPolicyCache(
        {"enabled": True, "local_cache_ttl_seconds": 30, "guardrail_ttl_seconds": 30}
    )
    cache.redis_client = MagicMock()
    cache.redis_client.pipeline.return_value.execute = AsyncMock(return_value=[True, 1])

    await cache.materialize_routing_guardrail(
        {"scope_type": "model", "scope_key": "gpt-5"}
    )

    assert cache._local_guardrail_index["model"] == set()


@pytest.mark.asyncio
async def test_policy_cache_keeps_fast_lane_hints_request_scoped():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "request_ttl_seconds": 300,
            "user_ttl_seconds": 900,
            "guardrail_ttl_seconds": 180,
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
            "query_type": "analysis",
            "query_complexity": "complex",
            "requires_high_reasoning": True,
            "cost_sensitivity": "medium",
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
    assert request_policy["query_type"] == "analysis"
    assert request_policy["requires_high_reasoning"] is True
    assert user_policy is None
    assert effective_policy["policy_source"] == "request"


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


@pytest.mark.asyncio
async def test_policy_cache_materializes_routing_guardrails():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "guardrail_ttl_seconds": 180,
            "local_cache_ttl_seconds": 30,
        }
    )
    await cache.initialize()

    await cache.materialize_routing_guardrail(
        {
            "event_type": "routing.guardrails",
            "scope_type": "model",
            "scope_key": "gpt-5",
            "guardrail_action": "avoid",
            "description": "Model error rate exceeded threshold",
            "emitted_at": "2026-04-08T12:00:00Z",
        }
    )
    await cache.materialize_routing_guardrail(
        {
            "event_type": "routing.guardrails",
            "scope_type": "provider",
            "scope_key": "anthropic",
            "guardrail_action": "warn",
            "description": "Provider latency regressed",
            "emitted_at": "2026-04-08T12:00:01Z",
        }
    )

    active_guardrails = await cache.get_active_guardrails()

    assert active_guardrails["blocked_models"] == ["gpt-5"]
    assert active_guardrails["warn_providers"] == ["anthropic"]
    assert (
        active_guardrails["guardrail_reasons"]["model:gpt-5"]
        == "Model error rate exceeded threshold"
    )


@pytest.mark.asyncio
async def test_policy_cache_merges_request_session_user_and_guardrails():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "request_ttl_seconds": 300,
            "session_ttl_seconds": 600,
            "user_ttl_seconds": 900,
            "guardrail_ttl_seconds": 180,
            "local_cache_ttl_seconds": 30,
        }
    )
    await cache.initialize()

    await cache.materialize_routing_policy_state(
        {
            "event_type": "routing.policy_state",
            "scope_type": "user",
            "scope_key": "user-9",
            "user_id": "user-9",
            "query_type": "analysis",
            "query_complexity": "moderate",
            "cost_sensitivity": "high",
            "preferred_models": [],
            "avoid_models": ["gpt-5"],
            "emitted_at": "2026-04-17T12:00:00Z",
        }
    )
    await cache.materialize_routing_policy_state(
        {
            "event_type": "routing.policy_state",
            "scope_type": "session",
            "scope_key": "session-9",
            "user_id": "user-9",
            "session_id": "session-9",
            "query_type": "code_analysis",
            "query_complexity": "complex",
            "requires_high_reasoning": True,
            "preferred_models": ["gpt-5"],
            "emitted_at": "2026-04-17T12:00:01Z",
        }
    )
    await cache.materialize_request_enriched(
        {
            "event_type": "requests.enriched",
            "request_id": "req-9",
            "user_id": "user-9",
            "query_type": "code_analysis",
            "query_complexity": "simple",
            "requires_low_latency": True,
            "route_to_fast_lane": False,
            "preferred_models": ["mistral-7b"],
            "emitted_at": "2026-04-17T12:00:02Z",
        }
    )
    await cache.materialize_routing_guardrail(
        {
            "event_type": "routing.guardrails",
            "scope_type": "provider",
            "scope_key": "anthropic",
            "guardrail_action": "avoid",
            "description": "Provider elevated error rate",
            "emitted_at": "2026-04-17T12:00:03Z",
        }
    )

    session_policy = await cache.get_session_policy("session-9")
    effective_policy = await cache.get_effective_policy(
        "req-9",
        "user-9",
        "session-9",
    )

    assert session_policy["preferred_models"] == ["gpt-5"]
    assert effective_policy["policy_source"] == "request+session+user"
    assert effective_policy["preferred_models"] == ["mistral-7b", "gpt-5"]
    assert effective_policy["avoid_models"] == ["gpt-5"]
    assert effective_policy["blocked_providers"] == ["anthropic"]


@pytest.mark.asyncio
async def test_policy_cache_uses_routing_policy_state_for_user_policy():
    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "request_ttl_seconds": 300,
            "session_ttl_seconds": 600,
            "user_ttl_seconds": 900,
            "local_cache_ttl_seconds": 30,
        }
    )
    await cache.initialize()

    await cache.materialize_request_enriched(
        {
            "event_type": "requests.enriched",
            "request_id": "req-user-hint-1",
            "user_id": "user-merged-1",
            "route_to_fast_lane": True,
            "preferred_models": ["mistral-7b"],
            "requires_low_latency": True,
            "emitted_at": "2026-04-17T12:10:00Z",
        }
    )
    await cache.materialize_routing_policy_state(
        {
            "event_type": "routing.policy_state",
            "scope_type": "user",
            "scope_key": "user-merged-1",
            "user_id": "user-merged-1",
            "query_type": "analysis",
            "query_complexity": "complex",
            "preferred_models": ["gpt-5"],
            "requires_high_reasoning": True,
            "emitted_at": "2026-04-17T12:10:05Z",
        }
    )

    user_hint_policy = await cache.get_user_hint_policy("user-merged-1")
    user_state_policy = await cache.get_user_state_policy("user-merged-1")
    user_policy = await cache.get_user_policy("user-merged-1")

    assert user_hint_policy is None
    assert user_state_policy["preferred_models"] == ["gpt-5"]
    assert user_policy["preferred_models"] == ["gpt-5"]
    assert user_policy["route_to_fast_lane"] is False


@pytest.mark.asyncio
async def test_enabled_policy_cache_does_not_update_l1_when_redis_write_fails():
    cache = RoutingPolicyCache({"enabled": True})
    cache.redis_client = MagicMock()
    cache.redis_client.setex = AsyncMock(side_effect=RuntimeError("redis down"))
    cache_key = cache._cache_key("request", "req-failed")

    with pytest.raises(RuntimeError, match="redis down"):
        await cache._set_json(cache_key, {"request_id": "req-failed"}, 30)

    assert cache._local_get(cache_key) is None


@pytest.mark.asyncio
async def test_guardrail_payload_and_index_are_transactional_before_l1_update():
    cache = RoutingPolicyCache({"enabled": True, "guardrail_ttl_seconds": 180})
    pipeline = MagicMock()
    pipeline.setex = MagicMock(return_value=pipeline)
    pipeline.sadd = MagicMock(return_value=pipeline)
    pipeline.execute = AsyncMock(side_effect=RuntimeError("transaction failed"))
    cache.redis_client = MagicMock()
    cache.redis_client.pipeline.return_value = pipeline

    with pytest.raises(RuntimeError, match="transaction failed"):
        await cache.materialize_routing_guardrail(
            {
                "event_type": "routing.guardrails",
                "scope_type": "model",
                "scope_key": "gpt-5",
                "guardrail_action": "avoid",
            }
        )

    cache.redis_client.pipeline.assert_called_once_with(transaction=True)
    pipeline.setex.assert_called_once()
    pipeline.sadd.assert_called_once()
    assert cache._local_get(cache._guardrail_cache_key("model", "gpt-5")) is None
    assert "gpt-5" not in cache._local_guardrail_index["model"]
