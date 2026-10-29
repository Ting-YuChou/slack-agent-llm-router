"""X8 review: cheaper tiers do not inherit a base price."""

import logging
from types import SimpleNamespace

from src.llm_router_part1_router import ModelRouter
from src.pricing import resolve_model_pricing
from src.utils.schema import ModelPricing


def test_cheaper_tiers_do_not_inherit_the_base_price():
    models = {
        "gpt-5": SimpleNamespace(pricing=ModelPricing(input=1.25, output=10.0)),
        "claude-sonnet-4-6": SimpleNamespace(
            pricing=ModelPricing(input=3.0, output=15.0)
        ),
    }

    for name in ("gpt-5.4", "gpt-5-2025-08-07", "gpt-5.4-2026-01-15"):
        assert resolve_model_pricing(models, name).input == 1.25, name
    assert resolve_model_pricing(models, "claude-sonnet-4-6-20250929").input == 3.0
    for name in (
        "gpt-5.4-mini",
        "gpt-5.1-codex-mini",
        "gpt-5.4-pro",
        "gpt-5-nano",
        "gpt-5.4.mini",
    ):
        assert resolve_model_pricing(models, name) is None, name


def test_versioned_cheaper_tier_is_reported_unpriced(caplog):
    with caplog.at_level(logging.WARNING):
        router = ModelRouter(
            {
                "default_model": "gpt-5",
                "models": {
                    "gpt-5": {
                        "provider": "openai",
                        "max_tokens": 8192,
                        "cost_per_token": 0.00001,
                        "pricing": {"input": 1.25, "output": 10.0},
                        "priority": 3,
                        "capabilities": ["general"],
                    },
                    "gpt-5.4-mini": {
                        "provider": "openai",
                        "max_tokens": 8192,
                        "cost_per_token": 0.00001,
                        "priority": 3,
                        "capabilities": ["general"],
                    },
                },
            }
        )

    assert router.unpriced_models() == ["gpt-5.4-mini"]
    assert "gpt-5.4-mini has no pricing" in caplog.text
