"""R12 round 2: the router's pick survives reroutes; the table stays eligible."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_pipeline import ClickHouseManager
from src.utils.schema import QueryRequest, QueryType, RoutingDecision, UserTier


def _engine():
    from src.llm_router_part2_inference import InferenceEngine

    return InferenceEngine({}, router=MagicMock())


def _decision(**updates):
    return RoutingDecision(
        selected_model="gpt-5",
        query_type=QueryType.GENERAL,
        routing_reason="Capability-based selection",
        token_count=1,
        estimated_cost=0.0,
        routing_time_ms=1,
        confidence=0.8,
        **updates,
    )


def test_vision_reroute_keeps_the_router_choice():
    rerouted = _engine()._build_capability_routing_decision(
        _decision(), "gpt-5", "claude-sonnet-4-6"
    )

    assert rerouted.selected_model == "claude-sonnet-4-6"
    assert rerouted.routed_model == "gpt-5"


def test_vision_reroute_keeps_an_earlier_router_choice():
    engine = _engine()
    first = engine._build_capability_routing_decision(
        _decision(routed_model="qwen"), "gpt-5", "claude-sonnet-4-6"
    )
    minimal = engine._build_capability_routing_decision(
        SimpleNamespace(selected_model="gpt-5", routing_reason="r"),
        "gpt-5",
        "vision",
    )

    assert first.routed_model == "qwen"
    assert minimal.selected_model == "vision"
    assert minimal.routed_model == "gpt-5"


@pytest.mark.asyncio
async def test_routing_decisions_gets_the_fallback_from_model_column():
    manager = ClickHouseManager(
        {"host": "localhost", "port": 8123, "database": "test_db"}
    )
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [call.args[1] for call in manager._run_blocking.await_args_list]
    assert (
        "ALTER TABLE test_db.routing_decisions ADD COLUMN IF NOT EXISTS "
        "fallback_from_model String DEFAULT ''"
    ) in statements


class _PolicyStub:
    def __init__(self, policy):
        self.policy = policy

    async def get_effective_policy(self, request_id, user_id):
        return dict(self.policy)


@pytest.mark.asyncio
async def test_round_robin_fail_open_without_capable_models_has_an_eligible_row():
    model = {
        "provider": "vllm",
        "max_tokens": 32768,
        "cost_per_token": 0.0,
        "priority": 3,
        "capabilities": ["vision"],
        "allowed_tiers": ["free", "premium", "enterprise"],
    }
    router = ModelRouter(
        {
            "default_model": "a",
            "routing_strategy": "round_robin",
            "models": {"a": dict(model), "b": dict(model)},
        },
        policy_cache=_PolicyStub({"blocked_models": ["a", "b"]}),
    )
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": 128

    decision = await router.route_query(
        QueryRequest(query="Hello", user_id="u1", user_tier=UserTier.PREMIUM)
    )

    assert "guardrails bypassed" in decision.routing_reason
    selected = [row for row in decision.candidates if row.selected]
    assert len(selected) == 1
    assert selected[0].eligible is True
    assert selected[0].reject_reason is None
    assert selected[0].score is not None
    assert selected[0].bypassed_guardrail == "blocked_model"
