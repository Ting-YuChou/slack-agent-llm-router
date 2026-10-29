"""Reasoning effort follows the routed query, with per-model overrides (I7)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.llm_router_part2_inference import (
    InferenceEngine,
    OpenAIProvider,
    remember_routing_features,
)
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.utils.schema import (
    ModelConfig,
    QueryRequest,
    QueryType,
    RoutingDecision,
    RoutingDecisionFeatures,
    UserTier,
)


def _request() -> QueryRequest:
    return QueryRequest(
        query="Prove that the sum of two even numbers is even",
        user_id="user-123",
        user_tier=UserTier.PREMIUM,
        max_tokens=256,
        temperature=0.3,
    )


def _decision(query_type=QueryType.GENERAL, **features) -> RoutingDecision:
    return RoutingDecision(
        selected_model="gpt-5",
        query_type=query_type,
        routing_reason="test",
        token_count=10,
        estimated_cost=0.0,
        routing_time_ms=1,
        confidence=0.9,
        features=RoutingDecisionFeatures(**features),
    )


def _model(name: str, provider: str = "openai", **overrides) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        max_tokens=128000,
        cost_per_token=0.0,
        priority=2,
        capabilities=["general", "reasoning"],
        **overrides,
    )


def _kwargs(provider, request, model_name="gpt-5", decision=None):
    if decision is not None:
        remember_routing_features(request, decision)
    return provider._build_response_request_kwargs(request, model_name, "Prompt")


@pytest.mark.parametrize(
    "decision",
    [
        _decision(requires_high_reasoning=True),
        _decision(query_complexity="complex"),
        _decision(query_type=QueryType.REASONING),
        _decision(query_type=QueryType.MATH),
    ],
    ids=["requires_high_reasoning", "complex", "reasoning", "math"],
)
def test_complex_reasoning_query_uses_high_effort_on_gpt5(decision):
    provider = OpenAIProvider({"api_key": "test-key"})

    kwargs = _kwargs(provider, _request(), decision=decision)

    assert kwargs["reasoning"] == {"effort": "high"}
    assert "temperature" not in kwargs


@pytest.mark.parametrize(
    "decision",
    [
        None,
        _decision(),
        _decision(query_type=QueryType.CODE_GENERATION, query_complexity="simple"),
    ],
    ids=["no-routing", "general", "code-simple"],
)
def test_general_query_keeps_minimal_effort_on_gpt5(decision):
    provider = OpenAIProvider({"api_key": "test-key"})

    kwargs = _kwargs(provider, _request(), decision=decision)

    assert kwargs["reasoning"] == {"effort": "minimal"}


def test_model_without_reasoning_support_gets_no_reasoning_kwarg():
    provider = OpenAIProvider({"api_key": "test-key"})
    request = _request()

    kwargs = _kwargs(
        provider,
        request,
        model_name="gpt-4.1",
        decision=_decision(query_type=QueryType.REASONING),
    )

    assert "reasoning" not in kwargs
    assert kwargs["temperature"] == request.temperature


def test_model_config_overrides_both_efforts():
    provider = OpenAIProvider(
        {"api_key": "test-key"},
        model_configs={
            "gpt-5": _model(
                "gpt-5", reasoning_effort="low", reasoning_effort_high="medium"
            )
        },
    )

    general = _kwargs(provider, _request(), decision=_decision())
    complex_ = _kwargs(
        provider, _request(), decision=_decision(query_type=QueryType.MATH)
    )

    assert general["reasoning"] == {"effort": "low"}
    assert complex_["reasoning"] == {"effort": "medium"}


def test_model_config_enables_reasoning_and_temperature_by_name():
    provider = OpenAIProvider(
        {"api_key": "test-key"},
        model_configs={
            "o-custom": _model(
                "o-custom", reasoning_effort="low", supports_temperature=False
            ),
            "gpt-5-chat": _model("gpt-5-chat", supports_temperature=True),
        },
    )
    request = _request()

    custom = _kwargs(provider, request, model_name="o-custom", decision=_decision())
    chat = _kwargs(provider, request, model_name="gpt-5-chat", decision=_decision())

    assert custom["reasoning"] == {"effort": "low"}
    assert "temperature" not in custom
    assert chat["temperature"] == request.temperature
    # Unset effort fields keep the gpt-5 prefix rule.
    assert chat["reasoning"] == {"effort": "minimal"}


def test_high_effort_falls_back_to_configured_default_effort():
    provider = OpenAIProvider(
        {"api_key": "test-key"},
        model_configs={"o-custom": _model("o-custom", reasoning_effort="medium")},
    )

    kwargs = _kwargs(
        provider,
        _request(),
        model_name="o-custom",
        decision=_decision(requires_high_reasoning=True),
    )

    assert kwargs["reasoning"] == {"effort": "medium"}


def test_model_config_rejects_unknown_effort():
    with pytest.raises(ValidationError):
        _model("gpt-5", reasoning_effort="extreme")


def test_config_files_set_gpt5_efforts():
    import yaml

    for path in (
        "config/config.yaml",
        "config/config.compose.yaml",
        "config/config.demo.yaml",
    ):
        with open(path, encoding="utf-8") as handle:
            models = yaml.safe_load(handle)["router"]["models"]
        gpt5 = ModelConfig(name="gpt-5", **models["gpt-5"])
        assert gpt5.reasoning_effort == "minimal", path
        assert gpt5.reasoning_effort_high == "high", path


@pytest.mark.asyncio
async def test_initialize_gives_openai_provider_the_router_models():
    router = MagicMock()
    router.models = {"gpt-5": _model("gpt-5", reasoning_effort="low")}
    engine = InferenceEngine(
        {"compression": {"enabled": False}, "openai": {"enabled": True}},
        router,
    )
    engine.cache.initialize = AsyncMock()

    with patch("src.llm_router_part2_inference.openai.AsyncOpenAI"):
        await engine.initialize()

    kwargs = _kwargs(engine.providers["openai"], _request(), decision=_decision())
    assert kwargs["reasoning"] == {"effort": "low"}


def _openai_usage():
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=64),
    )


async def _run_process_query(decision):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=decision)
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    event_producer = AsyncMock()

    engine = InferenceEngine(
        {"compression": {"enabled": False}, "cache": {"enabled": False}},
        router,
        event_producer=event_producer,
    )
    with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        client.responses.create.return_value = SimpleNamespace(
            output_text="Because 2a + 2b = 2(a + b).",
            usage=_openai_usage(),
            status="completed",
        )
        mock_openai.return_value = client
        provider = OpenAIProvider({"api_key": "test-key"})
        await provider.initialize()
    engine.providers = {"openai": provider}
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()

    response = await engine.process_query(_request())
    return response, client.responses.create.await_args.kwargs, event_producer


@pytest.mark.asyncio
async def test_process_query_passes_routing_features_to_the_provider():
    response, kwargs, event_producer = await _run_process_query(
        _decision(requires_high_reasoning=True)
    )

    assert kwargs["reasoning"] == {"effort": "high"}
    assert response.reasoning_effort == "high"
    event = build_inference_completed_event(
        *event_producer.produce_inference_completed.await_args.args
    )
    assert event["reasoning_effort"] == "high"
    assert event["reasoning_tokens"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_type", "expected"),
    [("general", "minimal"), ("math", "high")],
)
async def test_process_query_accepts_decisions_without_features(query_type, expected):
    decision = SimpleNamespace(
        selected_model="gpt-5",
        query_type=SimpleNamespace(value=query_type),
        routing_reason="Rule-based selection",
    )

    response, kwargs, _ = await _run_process_query(decision)

    assert kwargs["reasoning"] == {"effort": expected}
    assert response.reasoning_effort == expected


def test_inference_completed_event_records_reasoning_effort(
    inference_response_factory,
):
    request = _request()
    with_effort = inference_response_factory(reasoning_effort="high")
    without = inference_response_factory()

    assert (
        build_inference_completed_event(request, with_effort)["reasoning_effort"]
        == "high"
    )
    assert build_inference_completed_event(request, without)["reasoning_effort"] is None
