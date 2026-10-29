"""X8: one price table in config, read by the router and the providers."""

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter
from src.pricing import resolve_model_pricing
from src.utils.metrics import INFERENCE_METRICS, ROUTER_METRICS
from src.utils.schema import ModelPricing, QueryType

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _metric(metric, **labels):
    return metric.labels(**labels)._value.get()


def _router_from(name):
    config = yaml.safe_load((CONFIG_DIR / name).read_text())
    return ModelRouter(config["router"])


@pytest.mark.parametrize(
    "name", ["config.yaml", "config.compose.yaml", "config.demo.yaml"]
)
def test_every_checked_in_model_is_priced_or_self_hosted(name):
    router = _router_from(name)

    for model_name, model in router.models.items():
        if model.provider.lower() == "vllm":
            continue
        pricing = resolve_model_pricing(router.models, model_name)
        assert pricing is not None, model_name
        assert pricing.input > 0 and pricing.output > 0
    assert router.unpriced_models() == []


def test_checked_in_prices_match_the_official_tables():
    router = _router_from("config.yaml")

    gpt5 = router.models["gpt-5"].pricing
    sonnet = router.models["claude-sonnet-4-6"].pricing
    assert (gpt5.input, gpt5.cached_input, gpt5.output) == (1.25, 0.125, 10.0)
    assert (
        sonnet.input,
        sonnet.cached_input,
        sonnet.output,
        sonnet.cache_write_5m,
        sonnet.cache_write_1h,
    ) == (3.0, 0.30, 15.0, 3.75, 6.0)


def test_unpriced_paid_model_warns_and_counts(caplog):
    config = {
        "default_model": "mystery",
        "models": {
            "mystery": {
                "provider": "openai",
                "max_tokens": 8192,
                "cost_per_token": 0.00001,
                "priority": 3,
                "capabilities": ["general"],
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        router = ModelRouter(config)

    assert router.unpriced_models() == ["mystery"]
    assert "mystery" in caplog.text and "pric" in caplog.text
    assert _metric(ROUTER_METRICS.unpriced_models, model="mystery") == 1


def test_gpt5_point_releases_inherit_gpt5_pricing_by_prefix():
    models = {
        "gpt-5": SimpleNamespace(
            pricing=ModelPricing(input=1.25, cached_input=0.125, output=10.0)
        ),
        "gpt-5.4": SimpleNamespace(pricing=None),
    }

    assert resolve_model_pricing(models, "gpt-5.4").input == 1.25
    assert resolve_model_pricing(models, "gpt-5-2025-08-07").output == 10.0
    # A different model family is not a point release.
    assert resolve_model_pricing(models, "gpt-5-mini") is None
    assert resolve_model_pricing(models, "gpt-50") is None
    assert resolve_model_pricing({}, "gpt-5") is None


def _usage(**values):
    return SimpleNamespace(**values)


def test_anthropic_one_hour_cache_writes_cost_twice_the_input_rate():
    from src.llm_router_part2_inference import AnthropicProvider

    models = {
        "claude-x": SimpleNamespace(
            provider="anthropic",
            cost_per_token=0.0,
            pricing=ModelPricing(input=3.0, output=15.0),
        )
    }
    provider = AnthropicProvider({"prompt_cache": {"ttl": "1h"}}, model_configs=models)

    cost = provider._calculate_cost(
        _usage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=0,
        ),
        "claude-x",
    )

    assert cost == pytest.approx(6.0)


def test_anthropic_cost_splits_cache_writes_by_ttl_and_uses_listed_prices():
    from src.llm_router_part2_inference import AnthropicProvider

    models = {
        "claude-sonnet-4-6": SimpleNamespace(
            provider="anthropic",
            cost_per_token=0.0,
            pricing=ModelPricing(
                input=3.0,
                cached_input=0.3,
                output=15.0,
                cache_write_5m=3.75,
                cache_write_1h=6.0,
            ),
        )
    }
    provider = AnthropicProvider({}, model_configs=models)

    cost = provider._calculate_cost(
        _usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=3_000_000,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=1_000_000,
                ephemeral_1h_input_tokens=2_000_000,
            ),
            cache_read_input_tokens=1_000_000,
        ),
        "claude-sonnet-4-6",
    )

    assert cost == pytest.approx(3.0 + 15.0 + 3.75 + 2 * 6.0 + 0.3)


def test_openai_uses_listed_cached_input_price():
    from src.llm_router_part2_inference import OpenAIProvider

    models = {
        "gpt-5": SimpleNamespace(
            provider="openai",
            cost_per_token=0.0,
            pricing=ModelPricing(input=1.25, cached_input=0.125, output=10.0),
        )
    }
    provider = OpenAIProvider({}, model_configs=models)

    cost = provider._calculate_cost(
        _usage(
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            input_tokens_details=SimpleNamespace(cached_tokens=1_000_000),
        ),
        "gpt-5.4",
    )

    assert cost == pytest.approx(1.25 + 0.125 + 10.0)


def test_unpriced_provider_cost_warns_and_counts(caplog):
    from src.llm_router_part2_inference import OpenAIProvider

    provider = OpenAIProvider({}, model_configs={})
    before = _metric(
        INFERENCE_METRICS.unpriced_cost_calculations,
        model="unknown-model",
        provider="openai",
    )

    with caplog.at_level(logging.WARNING):
        cost = provider._calculate_cost(
            _usage(input_tokens=10, output_tokens=10), "unknown-model"
        )

    assert cost == 0.0
    assert "unknown-model" in caplog.text
    assert (
        _metric(
            INFERENCE_METRICS.unpriced_cost_calculations,
            model="unknown-model",
            provider="openai",
        )
        == before + 1
    )


def test_vllm_cost_stays_zero_unless_priced():
    from src.llm_router_part2_inference import vLLMProvider

    unpriced = vLLMProvider({"base_url": "http://127.0.0.1:8001"})
    priced = vLLMProvider(
        {"base_url": "http://127.0.0.1:8001"},
        model_configs={
            "local": SimpleNamespace(
                provider="vllm",
                cost_per_token=0.0,
                pricing=ModelPricing(input=0.1, output=0.2),
            )
        },
    )
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    assert unpriced._calculate_cost(usage, "local") == 0.0
    assert priced._calculate_cost(usage, "local") == pytest.approx(0.3)


def _twin_router(weight=None, price_ratio=10.0):
    base = {
        "provider": "openai",
        "max_tokens": 128000,
        "cost_per_token": 0.00001,
        "priority": 2,
        "capabilities": ["general"],
    }
    scoring = {} if weight is None else {"cost_log_ratio_weight": weight}
    router = ModelRouter(
        {
            "default_model": "cheap",
            "scoring": {**scoring, "default_model_bonus": 0},
            "models": {
                "cheap": {**base, "pricing": {"input": 1.0, "output": 4.0}},
                "dear": {
                    **base,
                    "pricing": {
                        "input": 1.0 * price_ratio,
                        "output": 4.0 * price_ratio,
                    },
                },
            },
        }
    )
    return router


def _context(router, query_type="general"):
    return {
        "query_type": query_type,
        "token_count": 500,
        "user_tier": "enterprise",
        "max_tokens": 1000,
    }


@pytest.mark.parametrize("weight", [3.0, 7.5])
def test_ten_times_the_price_costs_a_configurable_score(weight):
    router = _twin_router(weight=weight)
    context = _context(router)

    table = {
        candidate.model: candidate
        for candidate in router._candidate_table(context, "cheap", "capability")
    }

    assert table["cheap"].score - table["dear"].score == pytest.approx(weight)
    assert router._select_best_model(["cheap", "dear"], context) == "cheap"


def test_cost_score_is_relative_to_the_cheapest_eligible_candidate():
    router = _twin_router(weight=4.0, price_ratio=100.0)
    context = _context(router)

    alone = router._score_components("dear", {**context, "cost_reference": None})
    reference = router._request_cost(router.models["cheap"], context)
    against_cheap = router._score_components(
        "dear", {**context, "cost_reference": reference}
    )

    cap = float(router.scoring["cost_weight_cap"])
    assert alone["cost"] == pytest.approx(cap - 4.0 * 2)
    assert against_cheap["cost"] == pytest.approx(cap - 4.0 * 2)
    only_dear = router._score_components(
        "dear",
        {
            **context,
            "cost_reference": router._request_cost(router.models["dear"], context),
        },
    )
    assert only_dear["cost"] == pytest.approx(cap)


def test_output_tokens_are_estimated_by_query_type():
    router = _twin_router()
    base = {"token_count": 100, "max_tokens": None}

    general = router._expected_output_tokens({**base, "query_type": "general"})
    code = router._expected_output_tokens({**base, "query_type": "code_generation"})
    capped = router._expected_output_tokens(
        {**base, "query_type": "code_generation", "max_tokens": 50}
    )

    assert code > general > 0
    assert capped == 50


def test_estimated_cost_uses_split_input_and_output_rates():
    router = _twin_router()
    context = {**_context(router), "query_type": "general"}
    output_tokens = router._expected_output_tokens(context)

    from src.utils.schema import ModelSelection

    cost = router._estimate_cost(
        ModelSelection(model_name="cheap", confidence=0.5, reason="r"), context
    )

    assert cost == pytest.approx((500 * 1.0 + output_tokens * 4.0) / 1_000_000)


def test_model_bonuses_come_from_scoring_config():
    router = ModelRouter(
        {
            "default_model": "a",
            "scoring": {
                "default_bonus_models": ["b"],
                "gpt5_default_bonus": 5,
                "enterprise_bonus_models": ["a"],
                "sonnet_enterprise_bonus": 9,
            },
            "models": {
                name: {
                    "provider": "vllm",
                    "max_tokens": 8192,
                    "cost_per_token": 0.0,
                    "priority": 3,
                    "capabilities": ["general"],
                }
                for name in ("a", "b", "gpt-5", "claude-sonnet-4-6")
            },
        }
    )
    premium = {
        "query_type": "code_generation",
        "token_count": 10,
        "user_tier": "premium",
    }
    enterprise = {**premium, "user_tier": "enterprise"}

    assert router._contextual_model_bonus("b", premium) == 5
    assert router._contextual_model_bonus("gpt-5", premium) == 0
    assert router._contextual_model_bonus("a", enterprise) == 9 + float(
        router.scoring["default_model_bonus"]
    )
    assert router._contextual_model_bonus("claude-sonnet-4-6", enterprise) == 0


def test_default_bonus_models_keep_the_old_names():
    router = ModelRouter(
        {
            "default_model": "gpt-5",
            "models": {
                "gpt-5": {
                    "provider": "openai",
                    "max_tokens": 8192,
                    "cost_per_token": 0.0,
                    "priority": 3,
                    "capabilities": ["general"],
                }
            },
        }
    )

    assert router.scoring["default_bonus_models"] == ["gpt-5"]
    assert router.scoring["enterprise_bonus_models"] == ["claude-sonnet-4-6"]
    assert router.scoring["difficult_task_bonus_models"] == ["claude-sonnet-4-6"]
    assert math.isclose(float(router.scoring["gpt5_default_bonus"]), 3.0)
    assert math.isclose(float(router.scoring["sonnet_enterprise_bonus"]), 12.0)


def test_unused_query_type_has_a_default_output_estimate():
    router = _twin_router()

    assert router._expected_output_tokens(
        {"query_type": QueryType.GENERAL.value, "token_count": 1}
    ) == router._expected_output_tokens({"token_count": 1})
