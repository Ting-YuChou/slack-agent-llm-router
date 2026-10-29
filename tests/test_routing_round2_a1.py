"""A1 round 2: shadow replays whenever enforce would route differently."""

import pytest

from src.llm_router_part1_router import ModelRouter
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import QueryRequest, QueryType, UserTier

EMPTY_GUARDRAILS = {
    "blocked_models": [],
    "warn_models": [],
    "blocked_providers": [],
    "warn_providers": [],
    "guardrail_reasons": {},
}


class _PolicyStub:
    mode = "shadow"

    def __init__(self, policy=None, error=None):
        self.policy = policy
        self.error = error

    async def get_effective_policy(self, request_id, user_id):
        if self.error is not None:
            raise self.error
        return dict(self.policy)


def _router(router_config, cache):
    router = ModelRouter(router_config, policy_cache=cache)
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": 128
    replays = []
    original = router._run_shadow_strategy

    async def counting(context):
        replays.append(context)
        return await original(context)

    router._run_shadow_strategy = counting
    return router, replays


def _request(tier):
    return QueryRequest(query="Hello there", user_id="u1", user_tier=tier)


def _counter(outcome):
    return ROUTER_METRICS.shadow_policy_decisions.labels(outcome=outcome)._value.get()


@pytest.mark.asyncio
async def test_policy_that_lowers_cost_sensitivity_is_replayed(router_config):
    router, replays = _router(
        router_config, _PolicyStub({"cost_sensitivity": "medium"})
    )
    skipped = _counter("skipped_empty")

    decision = await router.route_query(_request(UserTier.FREE))

    assert len(replays) == 1
    assert replays[0]["cost_sensitivity"] == "medium"
    assert decision.shadow_selected_model is not None
    assert _counter("skipped_empty") == skipped


@pytest.mark.asyncio
async def test_policy_query_type_is_replayed(router_config):
    router, replays = _router(
        router_config, _PolicyStub({"query_type": "code_generation"})
    )

    await router.route_query(_request(UserTier.PREMIUM))

    assert len(replays) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [{"route_to_fast_lane": True}, {"requires_low_latency": True}, EMPTY_GUARDRAILS],
    ids=["fast-lane-only", "low-latency-only", "empty-guardrails"],
)
async def test_policy_that_routing_overwrites_is_skipped(router_config, policy):
    router, replays = _router(router_config, _PolicyStub(policy))
    skipped = _counter("skipped_empty")

    decision = await router.route_query(_request(UserTier.PREMIUM))

    assert replays == []
    assert decision.shadow_selected_model is None
    assert _counter("skipped_empty") == skipped + 1


@pytest.mark.asyncio
async def test_policy_fetch_failure_counts_as_unavailable(router_config):
    router, replays = _router(
        router_config, _PolicyStub(error=RuntimeError("redis down"))
    )
    unavailable = _counter("unavailable")
    skipped = _counter("skipped_empty")

    decision = await router.route_query(_request(UserTier.PREMIUM))

    assert replays == []
    assert decision.shadow_selected_model is None
    assert _counter("unavailable") == unavailable + 1
    assert _counter("skipped_empty") == skipped
