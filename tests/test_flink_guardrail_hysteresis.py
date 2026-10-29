"""Guardrails need enough samples and consecutive bad windows, and clear (A2)."""

import json
import os
import uuid
from datetime import datetime

import pytest
import pytest_asyncio

from flink import analytics_job, logic
from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import RoutingPolicyCache

TIMESTAMP = datetime(2026, 10, 29, 12, 0, 0)
WINDOW_MS = 60_000
BASE_MS = 1_800_000_000_000


class _ListState:
    def __init__(self):
        self.values = []

    def get(self):
        return list(self.values)

    def update(self, values):
        self.values = list(values)

    def clear(self):
        self.values = []


def _window(
    index=0,
    *,
    model="gpt-5",
    provider="openai",
    requests=10,
    errors=0,
    provider_failures=None,
    latency_ms=1000.0,
    tokens_per_second=50.0,
):
    start = BASE_MS + index * WINDOW_MS
    successes = requests - errors
    return logic.build_model_metrics_window_event(
        model_name=model,
        provider=provider,
        window_start_ms=start,
        window_end_ms=start + WINDOW_MS,
        window_size_seconds=60,
        request_count=requests,
        success_count=successes,
        error_count=errors,
        latency_sum_ms=latency_ms * requests,
        tokens_per_second_sum=tokens_per_second * requests,
        token_count_input=100 * requests,
        token_count_output=int(latency_ms / 1000 * tokens_per_second) * requests,
        total_tokens=200 * requests,
        total_cost_usd=0.001 * requests,
        cached_count=0,
        provider_failure_count=errors
        if provider_failures is None
        else provider_failures,
        provider_latency_sum_ms=latency_ms * successes,
        provider_latency_sample_count=successes,
        timestamp=TIMESTAMP,
    )


def _actions(guardrails):
    return {(g["trigger_type"], g["guardrail_action"]) for g in guardrails}


def _model_detector():
    detector = analytics_job.ModelRoutingGuardrailDetector()
    detector.latency_history_state = _ListState()
    detector.qps_history_state = _ListState()
    detector.error_rate_history_state = _ListState()
    detector.cache_hit_rate_history_state = _ListState()
    detector.cost_per_1k_history_state = _ListState()
    return detector


def _provider_detector():
    detector = analytics_job.ProviderRoutingGuardrailDetector()
    detector.latency_history_state = _ListState()
    detector.error_rate_history_state = _ListState()
    return detector


def _run(detector, window):
    return list(detector.process_element(window, None))


def _avoid(outputs):
    return [g for g in outputs if g["guardrail_action"] == "avoid"]


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(_window(requests=1, errors=1), id="one-failed-request"),
        pytest.param(
            _window(requests=1, latency_ms=7200.0, tokens_per_second=1.0),
            id="one-slow-request",
        ),
    ],
)
def test_a_single_request_window_never_avoids(window):
    model = logic.detect_model_routing_guardrails(
        window,
        latency_history=[],
        qps_history=[],
        error_rate_history=[],
        cache_hit_rate_history=[],
        cost_per_1k_history=[],
    )
    provider = logic.detect_provider_routing_guardrails(
        window, latency_history=[], error_rate_history=[]
    )

    assert _avoid(model) == []
    assert _avoid(provider) == []


def test_caller_failures_do_not_count_against_the_model():
    window = _window(requests=10, errors=6, provider_failures=0)

    guardrails = logic.detect_model_routing_guardrails(
        window,
        latency_history=[],
        qps_history=[],
        error_rate_history=[0.0, 0.0, 0.0],
        cache_hit_rate_history=[],
        cost_per_1k_history=[],
    )

    assert {g["trigger_type"] for g in guardrails} & {
        "high_error_rate",
        "error_rate_regression",
    } == set()


def test_a_long_answer_only_warns_but_a_slow_model_avoids():
    # 7 s at 70 tokens/s is a ~500-token answer at ~14 ms per token.
    long_answer = _window(latency_ms=7000.0, tokens_per_second=70.0)
    slow_model = _window(latency_ms=7000.0, tokens_per_second=4.0)

    def high_latency(window):
        model = logic.detect_model_routing_guardrails(
            window,
            latency_history=[],
            qps_history=[],
            error_rate_history=[],
            cache_hit_rate_history=[],
            cost_per_1k_history=[],
        )
        provider = logic.detect_provider_routing_guardrails(
            window, latency_history=[], error_rate_history=[]
        )
        return _actions(model) | _actions(provider)

    assert high_latency(long_answer) >= {
        ("high_latency", "warn"),
        ("provider_high_latency", "warn"),
    }
    assert ("high_latency", "avoid") not in high_latency(long_answer)
    assert high_latency(slow_model) >= {
        ("high_latency", "avoid"),
        ("provider_high_latency", "avoid"),
    }


def test_min_request_count_is_configurable():
    window = _window(requests=3, errors=3)

    guardrails = logic.detect_model_routing_guardrails(
        window,
        latency_history=[],
        qps_history=[],
        error_rate_history=[],
        cache_hit_rate_history=[],
        cost_per_1k_history=[],
        thresholds={"min_request_count": 3},
    )

    assert ("high_error_rate", "avoid") in _actions(guardrails)


def test_model_avoid_needs_two_bad_windows_and_clears_after_good_ones():
    detector = _model_detector()

    first = _run(detector, _window(0, errors=6))
    second = _run(detector, _window(1, errors=6))
    third = _run(detector, _window(2))
    fourth = _run(detector, _window(3))
    fifth = _run(detector, _window(4))

    assert _avoid(first) == []
    assert ("high_error_rate", "avoid") in _actions(second)
    assert third == []
    (clear,) = fourth
    assert clear["guardrail_action"] == "clear"
    assert clear["scope_type"] == "model"
    assert clear["scope_key"] == "gpt-5"
    assert clear["trigger_type"] == "high_error_rate"
    assert fifth == []


def test_two_bad_windows_out_of_three_enter_the_guardrail():
    detector = _model_detector()

    _run(detector, _window(0, errors=6))
    _run(detector, _window(1))
    third = _run(detector, _window(2, errors=6))

    assert ("high_error_rate", "avoid") in _actions(third)


def test_one_bad_window_in_three_never_avoids():
    detector = _model_detector()
    outputs = []
    for index in range(6):
        outputs += _run(detector, _window(index, errors=6 if index % 3 == 0 else 0))

    assert _avoid(outputs) == []


def test_the_provider_judges_all_its_models_in_a_window_together():
    detector = _provider_detector()
    outputs = []
    for index in range(2):
        # Each model alone is under min_request_count; together they are not.
        outputs += _run(detector, _window(index, model="gpt-5", requests=3, errors=3))
        outputs += _run(
            detector, _window(index, model="gpt-5-mini", requests=3, errors=3)
        )

    provider_avoids = [g for g in _avoid(outputs) if g["scope_key"] == "openai"]
    assert ("provider_high_error_rate", "avoid") in _actions(provider_avoids)
    # Two windows, two models each: history holds one rate per window.
    assert detector.error_rate_history_state.get() == [1.0, 1.0]


def test_a_quiet_provider_is_never_blocked():
    detector = _provider_detector()
    outputs = []
    for index in range(5):
        outputs += _run(detector, _window(index, requests=2, errors=2))

    assert _avoid(outputs) == []


def test_a_healthy_model_dilutes_a_failing_one_at_the_provider():
    detector = _provider_detector()
    outputs = []
    for index in range(2):
        outputs += _run(detector, _window(index, model="a", requests=40, errors=0))
        outputs += _run(detector, _window(index, model="b", requests=5, errors=5))

    # 5 failures in 45 requests is 11%, under the 20% threshold.
    assert _avoid(outputs) == []


def test_clear_is_emitted_once_per_episode_for_the_provider():
    detector = _provider_detector()
    outputs = []
    for index, errors in enumerate([10, 10, 0, 0, 0]):
        outputs += _run(detector, _window(index, errors=errors))

    clears = [g for g in outputs if g["guardrail_action"] == "clear"]
    assert len(clears) == 1
    assert clears[0]["scope_type"] == "provider"
    assert clears[0]["trigger_type"] == "provider_high_error_rate"


def _guardrail_event(action, window_end_ms, scope_key="gpt-5"):
    return {
        "event_type": "routing.guardrails",
        "scope_type": "model",
        "scope_key": scope_key,
        "trigger_type": "high_error_rate",
        "guardrail_action": action,
        "window_end_ms": window_end_ms,
    }


@pytest.mark.asyncio
async def test_clear_lifts_a_guardrail_from_the_local_cache():
    cache = RoutingPolicyCache({"enabled": False, "local_cache_ttl_seconds": 30})
    await cache.initialize()
    await cache.materialize_routing_guardrail(_guardrail_event("avoid", 1_000))

    await cache.materialize_routing_guardrail(_guardrail_event("clear", 2_000))

    assert (await cache.get_active_guardrails())["blocked_models"] == []
    assert "gpt-5" not in cache._local_guardrail_index["model"]


class _ClearRedis:
    """Runs the cache's set and clear scripts against dicts."""

    def __init__(self):
        self.store = {}
        self.sets = {}

    async def eval(self, script, numkeys, *keys_and_args):
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        current = self.store.get(keys[0])
        stored = json.loads(current).get("event_time_ms") if current else None
        if script == policy.CAS_SET_SCRIPT:
            if stored is not None and int(stored) > int(args[2]):
                return 0
            self.store[keys[0]] = args[0]
            self.sets.setdefault(keys[1], set()).add(args[3])
            return 1
        assert script == policy.CAS_CLEAR_SCRIPT
        if stored is not None and int(stored) > int(args[0]):
            return 0
        self.store.pop(keys[0], None)
        self.sets.get(keys[1], set()).discard(args[1])
        return 1

    def pipeline(self, transaction=False):
        redis = self

        class _Pipeline:
            """Only the guardrail index reads of get_active_guardrails."""

            def __init__(self):
                self.index_keys = []

            def smembers(self, key):
                self.index_keys.append(key)
                return self

            async def execute(self):
                return [set(redis.sets.get(key, set())) for key in self.index_keys]

        return _Pipeline()

    async def mget(self, keys):
        return [self.store.get(key) for key in keys]


def _redis_cache():
    cache = RoutingPolicyCache({"enabled": True, "guardrail_ttl_seconds": 180})
    cache.redis_client = _ClearRedis()
    cache._initialized = True
    cache._wall_time_func = lambda: BASE_MS / 1000 + 30
    return cache


@pytest.mark.asyncio
async def test_clear_deletes_the_guardrail_and_its_index_member_in_redis():
    cache = _redis_cache()
    redis = cache.redis_client
    await cache.materialize_routing_guardrail(_guardrail_event("avoid", BASE_MS))

    await cache.materialize_routing_guardrail(
        _guardrail_event("clear", BASE_MS + WINDOW_MS)
    )

    assert cache._guardrail_cache_key("model", "gpt-5") not in redis.store
    assert redis.sets[cache._guardrail_index_key("model")] == set()
    assert (await cache.get_active_guardrails())["blocked_models"] == []


@pytest.mark.asyncio
async def test_an_older_clear_leaves_a_newer_guardrail_alone():
    cache = _redis_cache()
    redis = cache.redis_client
    await cache.materialize_routing_guardrail(_guardrail_event("avoid", BASE_MS))

    await cache.materialize_routing_guardrail(
        _guardrail_event("clear", BASE_MS - WINDOW_MS)
    )

    assert cache._guardrail_cache_key("model", "gpt-5") in redis.store
    assert redis.sets[cache._guardrail_index_key("model")] == {"gpt-5"}


@pytest_asyncio.fixture
async def real_redis():
    redis_asyncio = pytest.importorskip("redis.asyncio")
    url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    client = redis_asyncio.Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"real Redis unavailable at {url}: {exc}")
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_clear_script_on_real_redis(real_redis):
    prefix = f"test:policy_clear:{uuid.uuid4().hex}"
    cache = RoutingPolicyCache(
        {
            "enabled": True,
            "guardrail_ttl_seconds": 180,
            "redis": {"key_prefix": prefix},
        }
    )
    cache.redis_client = real_redis
    cache._initialized = True
    cache._wall_time_func = lambda: BASE_MS / 1000 + 30
    key = cache._guardrail_cache_key("model", "gpt-5")
    index_key = cache._guardrail_index_key("model")
    try:
        await cache.materialize_routing_guardrail(_guardrail_event("avoid", BASE_MS))
        await cache.materialize_routing_guardrail(
            _guardrail_event("clear", BASE_MS - WINDOW_MS)
        )
        assert await real_redis.exists(key) == 1

        await cache.materialize_routing_guardrail(
            _guardrail_event("clear", BASE_MS + WINDOW_MS)
        )

        assert await real_redis.exists(key) == 0
        assert await real_redis.smembers(index_key) == set()
    finally:
        await real_redis.delete(key, index_key)
