"""Time-decayed model stats and a continuous latency term (R13)."""

import pytest

from src.llm_router_part1_router import ModelRouter


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _router(router_config, **overrides):
    clock = FakeClock()
    router = ModelRouter(
        {**router_config, "tokenizer_enabled": False, **overrides},
        stats_clock=clock,
    )
    return router, clock


def _context(**overrides):
    return {
        "token_count": 100,
        "query_type": "general",
        "user_tier": "premium",
        **overrides,
    }


def test_half_life_defaults_to_five_minutes_and_is_configurable(router_config):
    assert _router(router_config)[0].stats_half_life_seconds == 300.0
    router, _clock = _router(router_config, stats_half_life_seconds=60)

    assert router.stats_half_life_seconds == 60.0


def test_recent_stats_follow_an_outage_while_lifetime_stats_barely_move(
    router_config,
):
    router, clock = _router(router_config)
    for _ in range(1000):
        router.update_model_stats("gpt-5", success=True, latency_ms=200)
    clock.now += 3600
    for _ in range(20):
        clock.now += 5
        router.update_model_stats("gpt-5", success=False, latency_ms=5000)

    stats = router.model_stats["gpt-5"]

    assert stats["success_rate"] == pytest.approx(1000 / 1020)
    assert stats["avg_latency"] == pytest.approx((1000 * 200 + 20 * 5000) / 1020)
    assert stats["recent_success_rate"] < 0.05
    assert stats["recent_latency_ms"] > 4800


def test_one_half_life_halves_the_weight_of_older_requests(router_config):
    router, clock = _router(router_config)
    router.update_model_stats("gpt-5", success=True, latency_ms=100)
    clock.now += 300
    router.update_model_stats("gpt-5", success=False, latency_ms=400)

    stats = router.model_stats["gpt-5"]

    # Weights 0.5 (older) and 1 (newer).
    assert stats["recent_success_rate"] == pytest.approx(0.5 / 1.5)
    assert stats["recent_latency_ms"] == pytest.approx((0.5 * 100 + 400) / 1.5)


def test_score_prefers_recent_stats_over_lifetime_stats(router_config):
    router, _clock = _router(router_config)
    router.model_stats["gpt-5"] = {
        "success_rate": 0.99,
        "avg_latency": 300,
        "recent_success_rate": 0.2,
        "recent_latency_ms": 4000,
    }
    components = router._score_components("gpt-5", _context())

    assert components["reliability"] == pytest.approx(0.2 * 40.0)
    assert components["latency"] == pytest.approx(20.0 * 1000 / (1000 + 4000))


def test_score_uses_lifetime_stats_when_recent_ones_are_missing(router_config):
    router, _clock = _router(router_config)
    router.model_stats["gpt-5"] = {"success_rate": 0.5, "avg_latency": 3000}

    components = router._score_components("gpt-5", _context())

    assert components["reliability"] == pytest.approx(0.5 * 40.0)
    assert components["latency"] == pytest.approx(20.0 * 1000 / (1000 + 3000))


def test_latency_term_is_continuous_and_matches_the_old_value_at_one_second(
    router_config,
):
    router, _clock = _router(router_config)

    def latency(avg_latency):
        router.model_stats["gpt-5"] = {"success_rate": 0.95, "avg_latency": avg_latency}
        return router._score_components("gpt-5", _context())["latency"]

    # The old term was max(0, 20 - avg/100): 10 at 1 s and 0 from 2 s on.
    assert latency(1000) == pytest.approx(10.0)
    assert latency(2000) > latency(3000) > latency(10_000) > 0.0
    assert latency(0) == pytest.approx(20.0)
