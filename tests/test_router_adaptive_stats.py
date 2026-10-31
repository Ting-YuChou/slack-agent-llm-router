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
    # The prior (3 requests at the lifetime rate) keeps it off 0, but the
    # outage dominates.
    assert stats["recent_success_rate"] < 0.2
    # Failures slower than the success average (timeouts) count towards
    # latency, so a model that starts timing out loses its latency score.
    assert stats["recent_latency_ms"] > 4000


def test_one_half_life_halves_the_weight_of_older_requests(router_config):
    router, clock = _router(router_config)
    router.update_model_stats("gpt-5", success=True, latency_ms=100)
    clock.now += 300
    router.update_model_stats("gpt-5", success=True, latency_ms=400)
    clock.now += 300
    router.update_model_stats("gpt-5", success=False, latency_ms=50)

    stats = router.model_stats["gpt-5"]

    # Request weights 0.25, 0.5 and 1, plus a prior of 3 requests at the
    # lifetime rate before this update (2 of 2 succeeded).
    assert stats["recent_success_rate"] == pytest.approx((0.75 + 3.0) / (1.75 + 3.0))
    # Latency from the successes only, weights 0.25 and 0.5.
    assert stats["recent_latency_ms"] == pytest.approx((0.25 * 100 + 0.5 * 400) / 0.75)


def test_one_failure_after_an_idle_period_does_not_sink_the_model(router_config):
    router, clock = _router(router_config)
    for _ in range(100):
        router.update_model_stats("gpt-5", success=True, latency_ms=300)
    clock.now += 3600
    router.update_model_stats("gpt-5", success=False, latency_ms=300)

    stats = router.model_stats["gpt-5"]

    assert stats["recent_success_rate"] > 0.7
    assert stats["recent_latency_ms"] == pytest.approx(300)


def test_the_first_request_is_weighed_against_the_default_prior(router_config):
    router, _clock = _router(router_config)
    router.update_model_stats("gpt-5", success=False, latency_ms=300)

    stats = router.model_stats["gpt-5"]

    assert stats["recent_success_rate"] == pytest.approx((0 + 3 * 0.95) / (1 + 3))
    # No success yet and no slow failure: the score uses the default latency.
    assert "recent_latency_ms" not in stats


def test_prior_weight_is_configurable(router_config):
    router, _clock = _router(router_config, stats_prior_weight=0)
    router.update_model_stats("gpt-5", success=False, latency_ms=300)

    assert router.stats_prior_weight == 0.0
    assert router.model_stats["gpt-5"]["recent_success_rate"] == 0.0


def test_fast_failures_do_not_make_a_model_look_fast(router_config):
    router, clock = _router(router_config)
    router.update_model_stats("gpt-5", success=True, latency_ms=2000)
    for _ in range(10):
        clock.now += 1
        router.update_model_stats("gpt-5", success=False, latency_ms=5)

    assert router.model_stats["gpt-5"]["recent_latency_ms"] == pytest.approx(2000)


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


@pytest.mark.parametrize(
    ("low_latency", "expected_at_one_second"),
    [(False, 10.0), (True, 20.0)],
)
def test_latency_term_keeps_the_old_scale_with_the_low_latency_bonus(
    router_config, low_latency, expected_at_one_second
):
    router, _clock = _router(router_config)
    router.model_stats["gpt-5"] = {"success_rate": 0.95, "avg_latency": 1000}

    components = router._score_components(
        "gpt-5", _context(requires_low_latency=low_latency)
    )

    # The old term max(0, w - avg/100) with w = 20, or 30 with the bonus.
    assert components["latency"] == pytest.approx(expected_at_one_second)


def test_a_fast_model_beats_a_slow_one_when_low_latency_is_required(router_config):
    models = {
        name: {
            "provider": "vllm",
            "model_path": f"/models/{name}",
            "max_tokens": 32768,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["general"],
        }
        for name in ("fast-model", "slow-model")
    }
    router, _clock = _router(
        router_config,
        models=models,
        default_model="fast-model",
        fast_lane_models=[],
        routing_rules=[],
    )
    router.model_stats["fast-model"] = {"success_rate": 0.95, "avg_latency": 300}
    router.model_stats["slow-model"] = {"success_rate": 0.97, "avg_latency": 2500}
    context = _context(requires_low_latency=True)

    fast = router._score_components("fast-model", context)
    slow = router._score_components("slow-model", context)

    assert fast["latency"] - slow["latency"] > 10.0
    assert router._score_model("fast-model", context) > router._score_model(
        "slow-model", context
    )


def test_a_failing_new_model_does_not_look_faster_than_an_unseen_one(router_config):
    router, clock = _router(router_config)
    for _ in range(5):
        clock.now += 1
        router.update_model_stats("gpt-5", success=False, latency_ms=5)  # 429s

    failing = router._score_components("gpt-5", _context())
    unseen = router._score_components("mistral-7b", _context())

    # Not the lifetime 5 ms: the default 1000 ms, like an unseen model.
    assert failing["latency"] == pytest.approx(unseen["latency"])
    assert failing["reliability"] < unseen["reliability"]


def test_timeouts_count_towards_recent_latency(router_config):
    router, clock = _router(router_config)
    for _ in range(5):
        clock.now += 1
        router.update_model_stats("gpt-5", success=True, latency_ms=500)
    before = router._score_components("gpt-5", _context())["latency"]
    for _ in range(5):
        clock.now += 1
        router.update_model_stats("gpt-5", success=False, latency_ms=30_000)

    stats = router.model_stats["gpt-5"]

    assert stats["recent_latency_ms"] > 10_000
    assert router._score_components("gpt-5", _context())["latency"] < before / 2


def test_recent_success_rate_recovers_toward_the_prior_without_traffic(
    router_config,
):
    router, clock = _router(router_config)
    for _ in range(50):
        router.update_model_stats("gpt-5", success=True, latency_ms=300)
    clock.now += 1800
    for _ in range(10):
        clock.now += 1
        router.update_model_stats("gpt-5", success=False, latency_ms=300)
    just_after = router._score_components("gpt-5", _context())["reliability"]

    clock.now += 3600  # avoided: no requests for an hour
    later = router._score_components("gpt-5", _context())["reliability"]

    lifetime_before_outage_failures = 50 / 59
    assert just_after < 0.6 * 40.0
    assert later > just_after
    assert later == pytest.approx(lifetime_before_outage_failures * 40.0, rel=0.01)


@pytest.mark.parametrize("avg_latency", [0, 500, 1000, 3000])
def test_latency_term_never_drops_as_the_weight_grows(router_config, avg_latency):
    router, _clock = _router(router_config)

    values = [router._latency_score(weight, avg_latency) for weight in range(0, 41)]

    assert values == sorted(values)
    assert router._latency_score(20.0, 1000) == pytest.approx(10.0)
    assert router._latency_score(30.0, 1000) == pytest.approx(20.0)
