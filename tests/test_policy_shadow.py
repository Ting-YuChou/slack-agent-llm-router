"""A1: the policy cache in shadow mode records, but never changes, routing."""

from pathlib import Path

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_pipeline import (
    build_inference_completed_event,
    build_routing_decision_event,
    KafkaConsumerManager,
)
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
    InferenceResponse,
    PolicyCacheConfig,
    QueryRequest,
    QueryType,
    UserTier,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Blocks qwen, the model routing picks without a policy.
POLICY = {
    "policy_source": "request",
    "blocked_models": ["qwen3.6-27b-fast"],
    "hint_reason": "qwen error rate",
}


class _PolicyStub:
    def __init__(self, policy, mode=None, config_mode=None):
        self.policy = policy
        if mode is not None:
            self.mode = mode
        if config_mode is not None:
            self.config = {"mode": config_mode}
        self.calls = 0

    async def get_effective_policy(self, request_id, user_id):
        self.calls += 1
        return dict(self.policy)


def _router(router_config, policy_cache=None):
    router = ModelRouter(router_config, policy_cache=policy_cache)
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": 128
    router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 100}
    return router


def _request(**kwargs):
    return QueryRequest(
        query="Hello there", user_id="u1", user_tier=UserTier.PREMIUM, **kwargs
    )


def _shadow_counter(outcome):
    return ROUTER_METRICS.shadow_policy_decisions.labels(outcome=outcome)._value.get()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_kwargs", [{"mode": "shadow"}, {"config_mode": "shadow"}], ids=str
)
async def test_shadow_mode_routes_as_if_the_policy_were_off(
    router_config, cache_kwargs
):
    off = await _router(router_config).route_query(_request())
    enforced = await _router(router_config, _PolicyStub(POLICY)).route_query(_request())
    cache = _PolicyStub(POLICY, **cache_kwargs)

    shadow = await _router(router_config, cache).route_query(_request())

    assert off.selected_model == "qwen3.6-27b-fast"
    assert enforced.selected_model != "qwen3.6-27b-fast"
    assert shadow.selected_model == off.selected_model
    assert shadow.routing_reason == off.routing_reason
    assert shadow.policy_source == off.policy_source
    assert shadow.shadow_selected_model == enforced.selected_model
    assert shadow.shadow_policy_source == "request"
    assert cache.calls >= 1


@pytest.mark.asyncio
async def test_enforce_is_the_default_mode(router_config):
    decision = await _router(router_config, _PolicyStub(POLICY)).route_query(_request())

    assert decision.selected_model != "qwen3.6-27b-fast"
    assert decision.policy_source == "request"
    assert decision.shadow_selected_model is None
    assert PolicyCacheConfig().mode == "enforce"
    assert PolicyCacheConfig(mode="shadow").mode == "shadow"
    with pytest.raises(ValueError):
        PolicyCacheConfig(mode="sometimes")


@pytest.mark.asyncio
async def test_shadow_counts_agreement_and_difference(router_config):
    differ_before = _shadow_counter("differ")
    agree_before = _shadow_counter("agree")

    await _router(router_config, _PolicyStub(POLICY, mode="shadow")).route_query(
        _request()
    )
    agreeing = {"policy_source": "request", "warn_models": ["mistral-7b"]}
    decision = await _router(
        router_config, _PolicyStub(agreeing, mode="shadow")
    ).route_query(_request())

    assert decision.shadow_selected_model == decision.selected_model
    assert _shadow_counter("differ") == differ_before + 1
    assert _shadow_counter("agree") == agree_before + 1


@pytest.mark.asyncio
async def test_shadow_errors_never_change_routing(router_config):
    off = await _router(router_config).route_query(_request())
    router = _router(router_config, _PolicyStub(POLICY, mode="shadow"))
    errors_before = _shadow_counter("error")

    async def broken(_context):
        raise RuntimeError("shadow exploded")

    router._run_shadow_strategy = broken

    decision = await router.route_query(_request())

    assert decision.selected_model == off.selected_model
    assert decision.routing_reason == off.routing_reason
    assert decision.shadow_selected_model is None
    assert _shadow_counter("error") == errors_before + 1


@pytest.mark.asyncio
async def test_shadow_without_a_policy_is_a_no_op(router_config):
    off = await _router(router_config).route_query(_request())

    decision = await _router(router_config, _PolicyStub({}, mode="shadow")).route_query(
        _request()
    )

    assert decision.selected_model == off.selected_model
    assert decision.policy_source == off.policy_source
    assert decision.shadow_selected_model is None


@pytest.mark.asyncio
async def test_shadow_keeps_request_metadata_hints(router_config):
    request = _request(metadata={"avoid_models": ["qwen3.6-27b-fast"]})
    off = await _router(router_config).route_query(request)

    decision = await _router(
        router_config, _PolicyStub(POLICY, mode="shadow")
    ).route_query(_request(metadata={"avoid_models": ["qwen3.6-27b-fast"]}))

    assert off.selected_model != "qwen3.6-27b-fast"
    assert decision.selected_model == off.selected_model


@pytest.mark.asyncio
async def test_fallback_policy_context_in_shadow_mode_has_no_guardrails(
    router_config,
):
    router = _router(router_config, _PolicyStub(POLICY, mode="shadow"))

    context = await router._get_policy_context(_request())

    assert context["blocked_models"] == []
    assert context["policy_source"] == "none"


def _decision_with_shadow():
    from src.utils.schema import RoutingDecision

    return RoutingDecision(
        selected_model="gpt-5",
        query_type=QueryType.GENERAL,
        routing_reason="Capability-based selection",
        token_count=10,
        estimated_cost=0.0,
        routing_time_ms=1,
        confidence=0.8,
        shadow_selected_model="mistral-7b",
        shadow_policy_source="request",
    )


def test_events_carry_the_shadow_selection():
    request = _request(request_id="r-shadow")
    response = InferenceResponse(
        response_text="hi",
        model_name="gpt-5",
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
        provider="openai",
    )
    decision = _decision_with_shadow()

    routing_event = build_routing_decision_event(request, decision, response)
    completed_event = build_inference_completed_event(request, response, decision)

    assert routing_event["shadow_selected_model"] == "mistral-7b"
    assert routing_event["shadow_policy_source"] == "request"
    assert completed_event["shadow_selected_model"] == "mistral-7b"


def test_routing_decision_row_carries_the_shadow_selection():
    consumer = KafkaConsumerManager(
        {"bootstrap_servers": ["localhost:9092"], "topics": {}, "clickhouse": {}},
        None,
    )
    event = build_routing_decision_event(
        _request(request_id="r-row"), _decision_with_shadow()
    )

    row = consumer._build_routing_decision_entry(event).to_dict()

    assert row["shadow_selected_model"] == "mistral-7b"
    assert row["shadow_policy_source"] == "request"


def _yaml(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def test_checked_in_policy_cache_modes():
    compose = _yaml("config.compose.yaml")["policy_cache"]
    assert compose["enabled"] is True
    assert compose["mode"] == "shadow"
    for name in ("config.yaml", "config.demo.yaml"):
        section = _yaml(name)["policy_cache"]
        assert section["enabled"] is False
        assert section["mode"] == "shadow"
        assert PolicyCacheConfig(**section).mode == "shadow"
