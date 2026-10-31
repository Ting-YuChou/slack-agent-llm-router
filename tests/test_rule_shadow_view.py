"""RT2: shadow mode replays a policy that changes only what a rule reads."""

import pytest

from src.llm_router_part1_router import ModelRouter
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import QueryRequest, QueryType, UserTier


class _PolicyStub:
    mode = "shadow"

    def __init__(self, policy):
        self.policy = policy

    async def get_effective_policy(self, request_id, user_id, session_id):
        return dict(self.policy)


def _router(router_config, policy, condition):
    config = dict(router_config)
    config["routing_rules"] = [
        {"condition": condition, "models": ["mistral-7b"], "fallback": "mistral-7b"}
    ]
    router = ModelRouter(config, policy_cache=_PolicyStub(policy))
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": 128
    router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 100}
    return router


def _shadow_counter(outcome):
    return ROUTER_METRICS.shadow_policy_decisions.labels(outcome=outcome)._value.get()


def _request():
    return QueryRequest(query="Hello there", user_id="u1", user_tier=UserTier.PREMIUM)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, policy",
    [
        ("burst_protection_active == True", {"burst_protection_active": True}),
        ("policy_priority == 'high'", {"priority": "high"}),
        ("cost_sensitivity == 'low'", {"cost_sensitivity": "low"}),
        ("hint_reason == 'pin'", {"hint_reason": "pin"}),
    ],
)
async def test_a_policy_field_a_rule_reads_is_replayed(
    router_config, condition, policy
):
    differ_before = _shadow_counter("differ")
    router = _router(router_config, {"policy_source": "request", **policy}, condition)

    decision = await router.route_query(_request())

    assert decision.selected_model != "mistral-7b"
    assert decision.shadow_selected_model == "mistral-7b"
    assert decision.shadow_policy_source == "request"
    assert _shadow_counter("differ") == differ_before + 1


@pytest.mark.asyncio
async def test_a_policy_field_no_rule_reads_is_still_skipped(router_config):
    skipped_before = _shadow_counter("skipped_empty")
    router = _router(
        router_config,
        {"policy_source": "request", "burst_protection_active": True},
        "hint_reason == 'pin'",
    )

    decision = await router.route_query(_request())

    assert decision.shadow_selected_model is None
    assert _shadow_counter("skipped_empty") == skipped_before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, policy",
    [
        ("burst_protection_active == True", {"burst_protection_active": False}),
        ("burst_protection_active == True", {}),
        ("policy_priority == 'high'", {"priority": None}),
        ("hint_reason == 'pin'", {"hint_reason": "other"}),
    ],
)
async def test_a_policy_that_leaves_every_rule_outcome_alone_is_skipped(
    router_config, condition, policy
):
    skipped_before = _shadow_counter("skipped_empty")
    router = _router(router_config, {"policy_source": "request", **policy}, condition)

    decision = await router.route_query(_request())

    assert decision.shadow_selected_model is None
    assert _shadow_counter("skipped_empty") == skipped_before + 1
