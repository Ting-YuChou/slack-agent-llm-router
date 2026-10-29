"""R12 review: the decision table stays consistent with routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_pipeline import build_routing_decision_event
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
    InferenceResponse,
    QueryRequest,
    QueryType,
    UserTier,
)

# What RoutingPolicyCache.get_effective_policy returns with no state and no
# active guardrails.
EMPTY_GUARDRAILS = {
    "blocked_models": [],
    "warn_models": [],
    "blocked_providers": [],
    "warn_providers": [],
    "guardrail_reasons": {},
}
# What the policy branch's R10 returns on a timeout or outage.
UNAVAILABLE = {"policy_source": "unavailable"}


class _PolicyStub:
    def __init__(self, policy, mode=None):
        self.policy = policy
        if mode is not None:
            self.mode = mode

    async def get_effective_policy(self, request_id, user_id):
        return dict(self.policy)


def _router(config, policy_cache=None, query_type=QueryType.GENERAL):
    router = ModelRouter(config, policy_cache=policy_cache)
    router.classifier.classify_query = lambda _query: (query_type, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": 128
    return router


def _request(**kwargs):
    return QueryRequest(
        query="Hello there", user_id="u1", user_tier=UserTier.PREMIUM, **kwargs
    )


def _shadow(outcome):
    return ROUTER_METRICS.shadow_policy_decisions.labels(outcome=outcome)._value.get()


# 2. routing.decisions keeps the router's choice after a fallback.


def _response(model_name, fallback_from=None):
    return InferenceResponse(
        response_text="ok",
        model_name=model_name,
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
        provider="vllm",
        fallback_from_model=fallback_from,
    )


@pytest.mark.asyncio
async def test_fallback_keeps_the_router_choice_in_routing_decisions(router_config):
    from src.llm_router_part2_inference import InferenceEngine

    router = _router(router_config)
    request = _request()
    decision = await router.route_query(request)
    routed = decision.selected_model
    engine = InferenceEngine({}, router=MagicMock())

    first = engine._build_fallback_routing_decision(decision, routed, "mistral-7b")
    second = engine._build_fallback_routing_decision(first, "mistral-7b", "gpt-5")
    event = build_routing_decision_event(
        request, second, _response("gpt-5", fallback_from=routed)
    )

    assert routed != "gpt-5"
    assert event["selected_model"] == routed
    assert event["served_model"] == "gpt-5"
    assert event["fallback_from_model"] == routed
    selected_rows = [row for row in event["candidates"] if row["selected"]]
    assert [row["model"] for row in selected_rows] == [routed]


def test_fallback_on_a_minimal_decision_keeps_the_original_model():
    from src.llm_router_part2_inference import InferenceEngine

    engine = InferenceEngine({}, router=MagicMock())
    decision = SimpleNamespace(selected_model="gpt-5", routing_reason="r")

    fallback = engine._build_fallback_routing_decision(decision, "gpt-5", "mistral")
    event = build_routing_decision_event(_request(), fallback)

    assert fallback.selected_model == "mistral"
    assert event["selected_model"] == "gpt-5"


# 3. The selected row is never ineligible.


@pytest.mark.asyncio
async def test_default_model_without_the_capability_is_an_eligible_row():
    router = _router(
        {
            "default_model": "vision-only",
            "routing_strategy": "fallback",
            "models": {
                "vision-only": {
                    "provider": "vllm",
                    "max_tokens": 32768,
                    "cost_per_token": 0.0,
                    "priority": 3,
                    "capabilities": ["vision"],
                    "allowed_tiers": ["free", "premium", "enterprise"],
                },
                "general": {
                    "provider": "vllm",
                    "max_tokens": 32768,
                    "cost_per_token": 0.0,
                    "priority": 3,
                    "capabilities": ["general"],
                    "allowed_tiers": ["free", "premium", "enterprise"],
                },
            },
        }
    )

    decision = await router.route_query(_request())

    assert decision.selected_model == "vision-only"
    assert decision.reason_code == "default_model"
    selected = [row for row in decision.candidates if row.selected]
    assert len(selected) == 1
    assert selected[0].eligible is True
    assert selected[0].reject_reason is None
    assert selected[0].score is not None


# 5. A fail-open table shows the bypass instead of rejecting the selection.


@pytest.mark.asyncio
async def test_fail_open_table_marks_bypassed_guardrails(router_config):
    blocked = {"blocked_models": ["gpt-5", "qwen3.6-27b-fast", "mistral-7b"]}
    router = _router(router_config, _PolicyStub(blocked))

    decision = await router.route_query(_request())

    assert "guardrails bypassed" in decision.routing_reason
    rows = {row.model: row for row in decision.candidates}
    selected = rows[decision.selected_model]
    assert selected.selected is True
    assert selected.eligible is True
    assert selected.score is not None
    assert selected.bypassed_guardrail == "blocked_model"
    assert all(row.bypassed_guardrail == "blocked_model" for row in rows.values())
    eligible = [row.score for row in rows.values() if row.eligible]
    assert selected.score == max(eligible)
