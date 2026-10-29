"""A1 review: shadow mode replays only an actionable policy."""

import pytest

from src.llm_router_part1_router import ModelRouter
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
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


# 1. Shadow mode replays only an actionable policy.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy, outcome",
    [(EMPTY_GUARDRAILS, "skipped_empty"), (UNAVAILABLE, "unavailable")],
    ids=["empty-guardrails", "unavailable"],
)
async def test_shadow_skips_policies_with_nothing_to_replay(
    router_config, policy, outcome
):
    router = _router(router_config, _PolicyStub(policy, mode="shadow"))
    replays = []
    original = router._run_shadow_strategy

    async def counting(context):
        replays.append(context)
        return await original(context)

    router._run_shadow_strategy = counting
    before = {name: _shadow(name) for name in ("agree", "differ", outcome)}

    decision = await router.route_query(_request())

    assert replays == []
    assert decision.shadow_selected_model is None
    assert decision.shadow_policy_source is None
    assert _shadow(outcome) == before[outcome] + 1
    assert _shadow("agree") == before["agree"]
    assert _shadow("differ") == before["differ"]


@pytest.mark.asyncio
async def test_shadow_replays_an_actionable_guardrail(router_config):
    policy = {**EMPTY_GUARDRAILS, "blocked_models": ["qwen3.6-27b-fast"]}
    router = _router(router_config, _PolicyStub(policy, mode="shadow"))

    decision = await router.route_query(_request())

    assert decision.shadow_selected_model is not None
    assert decision.shadow_selected_model != "qwen3.6-27b-fast"
