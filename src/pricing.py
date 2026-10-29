"""The router's single price table: router.models[*].pricing (X8).

Prices are USD per 1M tokens. The router scores and estimates with them and
the inference providers cost responses with them, so dashboards, routing and
the Slack footer read the same numbers.
"""

import logging
import re
from typing import Any, Mapping, Optional, Set, Tuple

from src.utils.metrics import INFERENCE_METRICS
from src.utils.schema import ModelPricing

logger = logging.getLogger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000

# What may follow a priced model's name for it to inherit the price: a point
# release, a dated snapshot, or both ("gpt-5" covers "gpt-5.4",
# "gpt-5-2025-08-07" and "claude-sonnet-4-6" covers its "-20250929"). Any
# other suffix is a different tier ("gpt-5.4-mini", "gpt-5.1-codex-mini") and
# stays unpriced.
_POINT_RELEASE = re.compile(r"(?:\.\d+)*(?:-\d{4}-\d{2}-\d{2}|-\d{8})?")

_warned_unpriced: Set[Tuple[str, str]] = set()


def _pricing_of(model_config: Any) -> Optional[ModelPricing]:
    if model_config is None:
        return None
    if isinstance(model_config, Mapping):
        pricing = model_config.get("pricing")
    else:
        pricing = getattr(model_config, "pricing", None)
    if pricing is None or isinstance(pricing, ModelPricing):
        return pricing
    try:
        return ModelPricing.model_validate(pricing)
    except Exception:
        logger.warning("Ignoring malformed pricing %r", pricing)
        return None


def resolve_model_pricing(
    model_configs: Optional[Mapping[str, Any]], model_name: str
) -> Optional[ModelPricing]:
    """The model's own pricing, else the longest priced name it extends."""
    configs = model_configs if isinstance(model_configs, Mapping) else {}
    own = _pricing_of(configs.get(model_name))
    if own is not None:
        return own
    best: Optional[Tuple[int, ModelPricing]] = None
    for name, config in configs.items():
        if name == model_name or not model_name.startswith(name):
            continue
        if not _POINT_RELEASE.fullmatch(model_name[len(name) :]):
            continue
        pricing = _pricing_of(config)
        if pricing is not None and (best is None or len(name) > best[0]):
            best = (len(name), pricing)
    return best[1] if best else None


def per_token(price_per_million: float) -> float:
    return float(price_per_million) / TOKENS_PER_PRICE_UNIT


def record_unpriced_cost(model_name: str, provider: str) -> None:
    """Count a response costed without a price; warn once per model."""
    INFERENCE_METRICS.unpriced_cost_calculations.labels(
        model=model_name, provider=provider
    ).inc()
    key = (provider, model_name)
    if key not in _warned_unpriced:
        _warned_unpriced.add(key)
        logger.warning(
            "No pricing configured for %s model %s; its cost falls back to "
            "cost_per_token (0 if the model is not configured). Set "
            "router.models.%s.pricing",
            provider,
            model_name,
            model_name,
        )
