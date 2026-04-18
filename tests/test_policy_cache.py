from src.llm_router_part3_policy import RoutingPolicyCache


import pytest


@pytest.mark.asyncio
async def test_policy_cache_materializes_request_and_user_fast_lane_hints():
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
async def test_policy_cache_keeps_user_hint_and_user_state_separate():
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

    assert user_hint_policy["preferred_models"] == ["mistral-7b"]
    assert user_state_policy["preferred_models"] == ["gpt-5"]
    assert user_policy["preferred_models"] == ["mistral-7b", "gpt-5"]
    assert user_policy["route_to_fast_lane"] is True
