"""Cache hits in Flink latency detectors and the rolling routing policy.

A cache hit reports its lookup time as latency, so it says nothing about a
model's latency. Windows of only cache hits must not move latency baselines, and
the rolling policy must average latency over uncached events.
"""

import json
from datetime import datetime

import pytest

from flink import analytics_job, logic

TIMESTAMP = datetime(2026, 10, 28, 12, 0, 0)
LATENCY_TRIGGERS = {
    "high_latency",
    "latency_regression",
    "provider_high_latency",
    "provider_latency_regression",
}


class _ListState:
    def __init__(self, values=()):
        self.values = list(values)

    def get(self):
        return list(self.values)

    def update(self, values):
        self.values = list(values)

    def clear(self):
        self.values = []


def _window(*, avg_latency_ms, cached):
    request_count = 5
    return logic.build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=request_count,
        success_count=request_count,
        error_count=0,
        latency_sum_ms=0.0 if cached else avg_latency_ms * request_count,
        tokens_per_second_sum=0.0 if cached else 50.0 * request_count,
        token_count_input=100,
        token_count_output=200,
        total_tokens=300,
        total_cost_usd=0.0 if cached else 0.01,
        cached_count=request_count if cached else 0,
        latency_sample_count=0 if cached else request_count,
        timestamp=TIMESTAMP,
    )


def _anomaly_detector(latency_history):
    detector = analytics_job.ModelMetricsAnomalyDetector()
    detector.latency_history_state = _ListState(latency_history)
    detector.qps_history_state = _ListState()
    detector.error_rate_history_state = _ListState()
    return detector


def _model_guardrail_detector(latency_history):
    detector = analytics_job.ModelRoutingGuardrailDetector()
    detector.latency_history_state = _ListState(latency_history)
    detector.qps_history_state = _ListState()
    detector.error_rate_history_state = _ListState()
    detector.cache_hit_rate_history_state = _ListState()
    detector.cost_per_1k_history_state = _ListState()
    return detector


def _provider_guardrail_detector(latency_history):
    detector = analytics_job.ProviderRoutingGuardrailDetector()
    detector.latency_history_state = _ListState(latency_history)
    detector.error_rate_history_state = _ListState()
    return detector


def _latency_triggers(outputs):
    """Latency guardrail trigger types and latency anomaly types in outputs."""
    return {
        output.get("trigger_type") or output.get("anomaly_type") for output in outputs
    } & LATENCY_TRIGGERS


@pytest.mark.parametrize(
    ("make_detector", "baseline", "normal_latency_ms"),
    [
        # Each normal window sits under the trigger for the real baseline but
        # over it once a 0 ms sample from the all-cached window is averaged in.
        pytest.param(_anomaly_detector, [1000.0] * 5, 1800.0, id="anomaly"),
        pytest.param(_model_guardrail_detector, [2000.0] * 3, 3800.0, id="model"),
        pytest.param(_provider_guardrail_detector, [3000.0] * 3, 5000.0, id="provider"),
    ],
)
def test_all_cached_window_leaves_latency_baseline_alone(
    make_detector, baseline, normal_latency_ms
):
    detector = make_detector(baseline)

    cached_outputs = list(
        detector.process_element(_window(avg_latency_ms=0.0, cached=True), None)
    )
    history_after_cached_window = detector.latency_history_state.get()
    normal_outputs = list(
        detector.process_element(
            _window(avg_latency_ms=normal_latency_ms, cached=False), None
        )
    )

    assert history_after_cached_window == baseline
    assert _latency_triggers(cached_outputs) == set()
    assert _latency_triggers(normal_outputs) == set()
    assert detector.latency_history_state.get() == [*baseline, normal_latency_ms]


def test_window_from_before_the_upgrade_still_feeds_latency_history():
    detector = _model_guardrail_detector([2000.0] * 3)
    legacy_window = _window(avg_latency_ms=1500.0, cached=False)
    del legacy_window["latency_sample_count"]

    list(detector.process_element(legacy_window, None))

    assert detector.latency_history_state.get() == [2000.0] * 3 + [1500.0]


def _policy_completion(**overrides):
    event = {
        "user_id": "user-1",
        "session_id": "session-1",
        "user_tier": "premium",
        "query_type": "general",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "success",
        "latency_ms": 1000,
        "total_tokens": 300,
        "cost_usd": 0.01,
    }
    event.update(overrides)
    return event


def test_rolling_policy_latency_ignores_cached_events():
    policy = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-1",
        events=[
            _policy_completion(latency_ms=1000),
            _policy_completion(latency_ms=3000),
            _policy_completion(latency_ms=2, cost_usd=0.0, cached_response=True),
        ],
        timestamp=TIMESTAMP,
    )

    assert policy["recent_request_count"] == 3
    assert policy["avg_latency_ms"] == pytest.approx(2000.0)


def test_rolling_policy_of_only_cache_hits_reports_no_latency():
    policy = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-1",
        events=[_policy_completion(latency_ms=2, cached_response=True)],
        timestamp=TIMESTAMP,
    )

    assert policy["avg_latency_ms"] == 0.0


def test_rolling_policy_aggregate_from_before_the_upgrade_averages_every_request():
    legacy_aggregate = logic._empty_routing_policy_aggregate()
    del legacy_aggregate["latency_sample_count"]
    legacy_aggregate.update(
        {"request_count": 2, "success_count": 2, "latency_sum_ms": 4000.0}
    )
    bucket = logic.add_routing_policy_event(
        logic._empty_routing_policy_aggregate(),
        _policy_completion(latency_ms=1000),
    )

    merged = logic.merge_routing_policy_aggregate(legacy_aggregate, bucket)
    policy = logic.build_routing_policy_state_event_from_aggregate(
        scope_type="user",
        scope_key="user-1",
        aggregate=merged,
        timestamp=TIMESTAMP,
    )

    assert policy["recent_request_count"] == 3
    assert policy["avg_latency_ms"] == round(5000.0 / 3, 2)


class _MapState:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)

    def keys(self):
        return list(self.values)

    def items(self):
        return list(self.values.items())

    def clear(self):
        self.values.clear()


class _ValueState:
    def __init__(self):
        self.current = None

    def value(self):
        return self.current

    def update(self, value):
        self.current = value

    def clear(self):
        self.current = None


class _TimerService:
    def __init__(self, now_ms):
        self.now_ms = now_ms

    def current_processing_time(self):
        return self.now_ms

    def current_watermark(self):
        return self.now_ms

    def register_event_time_timer(self, timestamp):
        pass

    def delete_event_time_timer(self, timestamp):
        pass


class _Context:
    def __init__(self, now_ms):
        self._timer_service = _TimerService(now_ms)

    def timer_service(self):
        return self._timer_service

    def get_current_key(self):
        return "user-1"


def _emitter_aggregate(events, now_ms=20_000):
    emitter = analytics_job.RollingScopePolicyEmitter(
        "user", window_size_seconds=300, bucket_size_seconds=5, emit_interval_seconds=5
    )
    emitter.recent_events_state = _ListState()
    emitter.bucket_state = _MapState()
    emitter.aggregate_state = _ValueState()
    emitter.cleanup_timer_state = _ValueState()
    emitter.cleanup_timer_kind_state = _ValueState()
    emitter.last_emit_state = _ValueState()
    emitter.dirty_state = _ValueState()
    emitter.migration_complete_state = _ValueState()
    emitter.arrival_sequence_state = _ValueState()
    ctx = _Context(now_ms)
    for offset, event in enumerate(events):
        completion = {
            "completion_timestamp": datetime.fromtimestamp(
                (now_ms - 1_000 + offset) / 1000
            ).isoformat(),
            **event,
        }
        list(emitter.process_element(completion, ctx))
    return json.loads(emitter.aggregate_state.value())


def test_rolling_policy_emitter_sees_cache_hits():
    aggregate = _emitter_aggregate(
        [
            _policy_completion(latency_ms=1000),
            _policy_completion(latency_ms=2, cost_usd=0.0, cached_response=True),
        ]
    )
    policy = logic.build_routing_policy_state_event_from_aggregate(
        scope_type="user",
        scope_key="user-1",
        aggregate=aggregate,
        timestamp=TIMESTAMP,
    )

    assert policy["recent_request_count"] == 2
    assert policy["avg_latency_ms"] == pytest.approx(1000.0)


@pytest.mark.parametrize(
    ("error_type", "failed_provider_counts"),
    [
        pytest.param("provider_error", {"openai": 1}, id="provider_error"),
        pytest.param("user_or_policy", {}, id="user_or_policy"),
    ],
)
def test_rolling_policy_emitter_sees_error_type(error_type, failed_provider_counts):
    aggregate = _emitter_aggregate(
        [_policy_completion(status="error", error_type=error_type)]
    )

    assert aggregate["failed_provider_counts"] == failed_provider_counts
