"""Tier entitlement on every routing path (R3) and the decision it records (R11a)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import src.llm_router_part1_router as router_module
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine, classify_inference_error
from src.utils.schema import (
    ModelConfig,
    PlatformConfig,
    QueryRequest,
    QueryType,
    RoutingDecision,
    UserTier,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_CONFIGS = ("config.yaml", "config.compose.yaml", "config.demo.yaml")
ALL_TIERS = ["free", "premium", "enterprise"]

# The access each checked-in config grants. config.yaml keeps what the old
# priority rule gave; compose and demo keep what users actually got, which was
# gpt-5 for every tier through the default-model fallback.
EXPECTED_ALLOWED_TIERS = {
    "config.yaml": {
        "gpt-5": ["premium", "enterprise"],
        "claude-sonnet-4-6": ["enterprise"],
        "qwen3.6-27b-fast": ALL_TIERS,
        "mistral-7b": ALL_TIERS,
        "llama-3.1-70b": ALL_TIERS,
    },
    "config.compose.yaml": {
        "gpt-5": ALL_TIERS,
        "claude-sonnet-4-6": ["enterprise"],
    },
    "config.demo.yaml": {
        "gpt-5": ALL_TIERS,
    },
}


def _raw_router_section(filename):
    payload = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))
    return payload["router"]


def _declared_allowed_tiers(filename):
    return {
        model_name: list(model.get("allowed_tiers") or [])
        for model_name, model in _raw_router_section(filename)["models"].items()
    }


def _stub_router(router, query_type=QueryType.GENERAL, token_count=128):
    router.classifier.classify_query = lambda _query: (query_type, 0.9)
    router.token_counter.count_tokens = lambda _prompt, _model="default": token_count
    return router


def _router_from_checked_in_config(filename, **stub):
    # The same path main.py takes: validate the file, then hand the router the
    # dumped router section.
    validated = PlatformConfig.model_validate(
        {"router": _raw_router_section(filename)}
    ).model_dump(mode="json")
    router_config = {
        **validated["router"],
        "tokenizer_enabled": False,
        "classifier": {"semantic_enabled": False},
    }
    return _stub_router(ModelRouter(router_config), **stub)


def _request(user_tier, query="Help me with this", **overrides):
    return QueryRequest(
        query=query, user_id="u1", user_tier=UserTier(user_tier), **overrides
    )


def _single_model_router(model_overrides, **stub):
    model = {
        "provider": "openai",
        "max_tokens": 128000,
        "cost_per_token": 0.00003,
        "priority": 2,
        "capabilities": ["general", "reasoning"],
    }
    model.update(model_overrides)
    router = ModelRouter(
        {
            "default_model": "gpt-5",
            "routing_strategy": "intelligent",
            "tokenizer_enabled": False,
            "classifier": {"semantic_enabled": False},
            "models": {"gpt-5": model},
            "routing_rules": [],
        }
    )
    return _stub_router(router, **stub)


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_checked_in_configs_declare_allowed_tiers_explicitly(filename):
    assert _declared_allowed_tiers(filename) == EXPECTED_ALLOWED_TIERS[filename]


@pytest.mark.asyncio
@pytest.mark.parametrize("token_count", [128, 50_000])
@pytest.mark.parametrize("query_type", list(QueryType))
@pytest.mark.parametrize("user_tier", ALL_TIERS)
@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
async def test_checked_in_configs_route_every_tier_to_an_allowed_model(
    filename, user_tier, query_type, token_count
):
    router = _router_from_checked_in_config(
        filename, query_type=query_type, token_count=token_count
    )
    declared = _declared_allowed_tiers(filename)

    decision = await router.route_query(_request(user_tier))

    assert (
        user_tier in declared[decision.selected_model]
    ), f"{filename}: {user_tier} {query_type.value} got {decision.selected_model}"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_tier", ALL_TIERS)
@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
async def test_routing_error_still_returns_an_allowed_model_with_estimated_tokens(
    filename, user_tier
):
    router = _router_from_checked_in_config(filename, token_count=321)

    def broken_classifier(_query):
        raise RuntimeError("classifier exploded")

    router.classifier.classify_query = broken_classifier

    decision = await router.route_query(_request(user_tier))

    assert user_tier in _declared_allowed_tiers(filename)[decision.selected_model]
    assert decision.token_count == 321
    assert "routing error" in decision.routing_reason.lower()


@pytest.mark.asyncio
async def test_routing_error_rejects_prompt_too_large_for_the_tier():
    router = _router_from_checked_in_config("config.compose.yaml", token_count=200_000)

    def broken_classifier(_query):
        raise RuntimeError("classifier exploded")

    router.classifier.classify_query = broken_classifier

    with pytest.raises(router_module.PromptTooLargeError):
        await router.route_query(_request("free"))


@pytest.mark.asyncio
async def test_capability_mismatch_uses_best_allowed_model_that_fits():
    # config.yaml: only llama-3.1-70b is open to free and fits 50k tokens, and
    # it lacks the "general" capability. Do not upgrade to gpt-5 or Sonnet.
    router = _router_from_checked_in_config(
        "config.yaml", query_type=QueryType.GENERAL, token_count=50_000
    )

    decision = await router.route_query(_request("free"))

    assert decision.selected_model == "llama-3.1-70b"
    assert "capability" in decision.routing_reason.lower()


@pytest.mark.asyncio
async def test_prompt_too_large_for_every_allowed_model_is_rejected():
    # Sonnet (1M tokens) would fit, but compose opens it to enterprise only.
    router = _router_from_checked_in_config("config.compose.yaml", token_count=200_000)

    with pytest.raises(router_module.PromptTooLargeError) as exc_info:
        await router.route_query(_request("free"))

    error = exc_info.value
    assert isinstance(error, ValueError)
    assert error.error_code == "prompt_too_large"
    assert error.http_status == 413
    assert error.error_type == "user_or_policy"
    assert classify_inference_error(error) == "user_or_policy"


@pytest.mark.asyncio
async def test_enterprise_prompt_that_fits_sonnet_is_not_rejected():
    router = _router_from_checked_in_config("config.compose.yaml", token_count=200_000)

    decision = await router.route_query(_request("enterprise"))

    assert decision.selected_model == "claude-sonnet-4-6"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_overrides",
    [
        pytest.param({"priority": 2}, id="priority-rule"),
        pytest.param(
            {"priority": 3, "allowed_tiers": ["premium", "enterprise"]},
            id="allowed-tiers",
        ),
    ],
)
async def test_tier_without_any_model_is_not_entitled(model_overrides):
    router = _single_model_router(model_overrides)

    with pytest.raises(router_module.ModelNotEntitledError) as exc_info:
        await router.route_query(_request("free"))

    error = exc_info.value
    assert isinstance(error, ValueError)
    assert error.error_code == "model_not_entitled"
    assert error.http_status == 403
    assert error.error_type == "user_or_policy"
    assert classify_inference_error(error) == "user_or_policy"


@pytest.mark.asyncio
async def test_guardrails_excluding_every_allowed_model_is_not_entitled():
    class PolicyCacheStub:
        async def get_effective_policy(self, request_id, user_id):
            return {"policy_source": "user", "blocked_providers": ["openai"]}

    router = _router_from_checked_in_config("config.compose.yaml")
    router.policy_cache = PolicyCacheStub()

    with pytest.raises(router_module.ModelNotEntitledError, match="guardrail"):
        await router.route_query(_request("free"))


def test_allowed_tiers_override_the_priority_rule(router_config):
    router = ModelRouter(router_config)
    model = ModelConfig(
        **{**router.models["gpt-5"].model_dump(), "allowed_tiers": ["free"]}
    )

    assert router._check_user_access(model, "free") is True
    assert router._check_user_access(model, UserTier.FREE) is True
    assert router._check_user_access(model, "premium") is False
    assert router._check_user_access(model, "enterprise") is False


def test_unknown_tier_is_treated_as_free(router_config):
    router = ModelRouter(router_config)

    assert router._check_user_access(router.models["gpt-5"], "gold") is False
    assert router._check_user_access(router.models["mistral-7b"], "gold") is True


def test_router_warns_when_a_loaded_tier_has_no_model(caplog):
    with caplog.at_level("WARNING", logger=router_module.__name__):
        _single_model_router({"priority": 2})

    assert "free" in caplog.text
    assert "no loaded model" in caplog.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["round_robin", "weighted", "static"])
async def test_non_intelligent_strategies_only_return_allowed_models(
    router_config, strategy
):
    router = _stub_router(ModelRouter({**router_config, "routing_strategy": strategy}))

    selected = set()
    for _ in range(200):
        decision = await router.route_query(_request("free"))
        selected.add(decision.selected_model)

    # gpt-5 (priority 2) is not open to free under the priority rule.
    assert selected <= {"qwen3.6-27b-fast", "mistral-7b"}


@pytest.mark.asyncio
async def test_round_robin_cycles_through_allowed_models(router_config):
    router = _stub_router(
        ModelRouter({**router_config, "routing_strategy": "round_robin"})
    )

    selected = [
        (await router.route_query(_request("free"))).selected_model for _ in range(4)
    ]

    assert set(selected) == {"qwen3.6-27b-fast", "mistral-7b"}


@pytest.mark.asyncio
async def test_non_intelligent_strategy_rejects_tier_without_model():
    router = _single_model_router({"priority": 2})
    router.routing_strategy = "round_robin"

    with pytest.raises(router_module.ModelNotEntitledError):
        await router.route_query(_request("free"))


def test_select_best_model_never_returns_a_model_the_tier_cannot_use(router_config):
    router = ModelRouter(router_config)
    router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 100}
    router.model_stats["mistral-7b"] = {"success_rate": 0.5, "avg_latency": 2000}
    context = {
        "query_type": "code_generation",
        "token_count": 128,
        "user_tier": "free",
    }

    selected = router._select_best_model(["gpt-5", "mistral-7b"], context)

    assert selected == "mistral-7b"
    with pytest.raises(router_module.ModelNotEntitledError):
        router._select_best_model(["gpt-5"], context)


# R11a: the decision carries the tier, strategy and request features, and the
# inference fallback reads them from the decision.

LOCAL_VLLM_ROUTER_CONFIG = {
    "default_model": "qwen3.6-27b-fast",
    "routing_strategy": "intelligent",
    "tokenizer_enabled": False,
    "classifier": {"semantic_enabled": False},
    "models": {
        "qwen3.6-27b-fast": {
            "provider": "vllm",
            "model_path": "Qwen/Qwen3.6-27B-FP8",
            "max_tokens": 32768,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["general", "coding", "reasoning"],
        },
        "mistral-7b": {
            "provider": "vllm",
            "model_path": "/models/mistral",
            "max_tokens": 8192,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["general", "coding"],
        },
    },
    "routing_rules": [],
}

VLLM_MODEL_FALLBACK_CONFIG = {
    "compression": {"max_context_tokens": 4096, "compression_ratio": 0.5},
    "cache": {"enabled": True, "ttl": 3600, "max_size": "1GB"},
    "batching": {"enabled": False},
    "vllm": {
        "model_fallback": {
            "enabled": True,
            "fallbacks": {"qwen3.6-27b-fast": ["mistral-7b"]},
            "allowed_query_types": ["general", "code_generation"],
            "max_input_tokens": 2048,
            "max_output_tokens": 1024,
            "disallow_attachments": True,
            "disallow_complex_reasoning": True,
        }
    },
}


@pytest.mark.asyncio
async def test_decision_records_tier_strategy_and_features(router_config):
    router = _stub_router(ModelRouter(router_config), query_type=QueryType.ANALYSIS)

    decision = await router.route_query(
        _request("enterprise", query="Compare the tradeoffs of both designs")
    )

    assert decision.user_tier == UserTier.ENTERPRISE
    assert decision.routing_strategy == "intelligent"
    assert decision.features.query_complexity == "moderate"
    assert decision.features.requires_high_reasoning is True
    assert decision.features.long_context is False
    assert decision.features.classification_confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_decision_records_long_context_and_configured_strategy(router_config):
    router = _stub_router(
        ModelRouter({**router_config, "routing_strategy": "round_robin"})
    )

    decision = await router.route_query(_request("premium", max_tokens=4096))

    assert decision.user_tier == UserTier.PREMIUM
    assert decision.routing_strategy == "round_robin"
    assert decision.features.long_context is True
    assert decision.features.requires_high_reasoning is False


@pytest.mark.asyncio
async def test_routing_error_decision_records_tier_strategy_and_features():
    router = _stub_router(ModelRouter(LOCAL_VLLM_ROUTER_CONFIG), token_count=2000)

    def broken_classifier(_query):
        raise RuntimeError("classifier exploded")

    router.classifier.classify_query = broken_classifier

    decision = await router.route_query(
        _request("premium", query="Investigate the root cause of this incident")
    )

    assert "routing error" in decision.routing_reason.lower()
    assert decision.user_tier == UserTier.PREMIUM
    assert decision.routing_strategy == "intelligent"
    # The request is still read as complex, so the vLLM fallback stays closed.
    assert decision.features.query_complexity == "complex"
    assert decision.features.requires_high_reasoning is True
    assert decision.features.classification_confidence == 0.0


def _vllm_fallback_engine(router):
    engine = InferenceEngine(VLLM_MODEL_FALLBACK_CONFIG, router)
    engine.providers = {"vllm": AsyncMock()}
    engine.providers["vllm"].generate_response = AsyncMock(
        side_effect=RuntimeError("qwen temporarily unavailable")
    )
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    return engine


@pytest.mark.asyncio
async def test_complex_request_routed_to_qwen_does_not_fall_back_to_mistral():
    # A general query the router itself reads as complex: 2,000 tokens plus a
    # "root cause" marker. The client sends no complexity metadata.
    router = _stub_router(
        ModelRouter(LOCAL_VLLM_ROUTER_CONFIG),
        query_type=QueryType.GENERAL,
        token_count=2000,
    )
    request = _request("premium", query="Investigate the root cause of this incident")
    assert not request.metadata
    # The routing process_query will repeat, checked before stats change.
    decision = await router.route_query(request)
    assert decision.selected_model == "qwen3.6-27b-fast"
    assert decision.query_type == QueryType.GENERAL
    engine = _vllm_fallback_engine(router)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    assert [
        call.args[1]
        for call in engine.providers["vllm"].generate_response.await_args_list
    ] == ["qwen3.6-27b-fast"]
    assert decision.features.query_complexity == "complex"


def test_routing_decision_features_default_and_validate():
    decision_fields = {
        "selected_model": "gpt-5",
        "query_type": QueryType.GENERAL,
        "routing_reason": "Capability-based selection",
        "token_count": 1,
        "estimated_cost": 0.0,
        "routing_time_ms": 0,
        "confidence": 0.5,
    }

    features = RoutingDecision(**decision_fields).features

    assert features.query_complexity is None
    assert features.requires_high_reasoning is False
    assert features.long_context is False
    assert features.classification_confidence == 0.0
    with pytest.raises(ValueError):
        RoutingDecision(**decision_fields, features={"classification_confidence": 1.5})


@pytest.mark.parametrize(
    ("features", "allowed"),
    [
        pytest.param({}, True, id="simple"),
        pytest.param({"query_complexity": "complex"}, False, id="complex"),
        pytest.param({"requires_high_reasoning": True}, False, id="high-reasoning"),
    ],
)
def test_vllm_model_fallback_reads_decision_features(features, allowed):
    router = ModelRouter(LOCAL_VLLM_ROUTER_CONFIG)
    engine = InferenceEngine(VLLM_MODEL_FALLBACK_CONFIG, router)
    request = _request("premium", query="Explain this", max_tokens=256)
    decision = RoutingDecision(
        selected_model="qwen3.6-27b-fast",
        query_type=QueryType.GENERAL,
        routing_reason="Capability-based selection",
        token_count=64,
        estimated_cost=0.0,
        routing_time_ms=1,
        confidence=0.8,
        features=features,
    )
    context = engine._build_local_fallback_context(request, decision)

    assert (
        engine._vllm_model_fallback_allowed(
            engine._vllm_model_fallback_config(), request, decision, context
        )
        is allowed
    )


def test_vllm_model_fallback_accepts_decisions_without_features():
    router = ModelRouter(LOCAL_VLLM_ROUTER_CONFIG)
    engine = InferenceEngine(VLLM_MODEL_FALLBACK_CONFIG, router)
    request = _request("premium", query="Explain this", max_tokens=256)
    decision = SimpleNamespace(
        selected_model="qwen3.6-27b-fast",
        query_type=QueryType.GENERAL,
        token_count=64,
    )
    context = engine._build_local_fallback_context(request, decision)

    assert engine._vllm_model_fallback_allowed(
        engine._vllm_model_fallback_config(), request, decision, context
    )
