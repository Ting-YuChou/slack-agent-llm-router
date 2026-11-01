"""RT1: the router estimates with the per-token rate inference bills."""

from pathlib import Path

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import (
    AnthropicProvider,
    OpenAIProvider,
    vLLMProvider,
)
from src.utils.schema import ModelConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CHECKED_IN_CONFIGS = ("config.yaml", "config.compose.yaml", "config.demo.yaml")
MILLION = 1_000_000


def _billed_rates(models, model_name):
    """(input, output) USD per token that inference charges for the model."""
    provider_name = models[model_name].provider.lower()
    if provider_name == "vllm":
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001"}, models)
        input_cost = provider._calculate_cost(
            {"prompt_tokens": MILLION, "completion_tokens": 0}, model_name
        )
        output_cost = provider._calculate_cost(
            {"prompt_tokens": 0, "completion_tokens": MILLION}, model_name
        )
        return input_cost / MILLION, output_cost / MILLION
    provider_class = {"openai": OpenAIProvider, "anthropic": AnthropicProvider}[
        provider_name
    ]
    rates = provider_class({}, model_configs=models)._per_token_rates(
        model_name, provider_name
    )
    assert rates is not None, model_name
    return rates[0], rates[1]


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_router_rates_match_billed_rates_for_every_checked_in_model(filename):
    config = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    router = ModelRouter(config["router"])

    for model_name, model in router.models.items():
        assert router._unit_rates(model) == pytest.approx(
            _billed_rates(router.models, model_name)
        ), f"{filename}: {model_name}"


def test_unpriced_vllm_model_costs_nothing_to_the_router(router_config):
    config = dict(router_config)
    config["models"] = {
        **router_config["models"],
        "local": {
            "provider": "vllm",
            "max_tokens": 8192,
            "cost_per_token": 0.000004,
            "priority": 3,
            "capabilities": ["general"],
        },
    }
    router = ModelRouter(config)
    model = router.models["local"]

    assert router._unit_rates(model) == (0.0, 0.0)
    assert _billed_rates(router.models, "local") == (0.0, 0.0)


def test_priced_vllm_model_uses_its_pricing_on_both_sides(router_config):
    config = dict(router_config)
    config["models"] = {
        **router_config["models"],
        "local": {
            "provider": "vllm",
            "max_tokens": 8192,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["general"],
            "pricing": {"input": 0.5, "output": 2.0},
        },
    }
    router = ModelRouter(config)

    assert router._unit_rates(router.models["local"]) == pytest.approx(
        (0.5 / MILLION, 2.0 / MILLION)
    )
    assert _billed_rates(router.models, "local") == pytest.approx(
        (0.5 / MILLION, 2.0 / MILLION)
    )


@pytest.mark.parametrize("model_name", ["mistral-7b", "llama-3.1-70b"])
def test_checked_in_self_hosted_prices_keep_their_cost_per_token(model_name):
    config = yaml.safe_load((CONFIG_DIR / "config.yaml").read_text(encoding="utf-8"))
    model = ModelConfig(name=model_name, **config["router"]["models"][model_name])

    assert model.pricing is not None
    assert model.pricing.input == pytest.approx(model.cost_per_token * MILLION)
    assert model.pricing.output == pytest.approx(model.cost_per_token * MILLION)
    assert model.pricing.cached_input == pytest.approx(model.pricing.input)
