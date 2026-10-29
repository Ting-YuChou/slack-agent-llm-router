"""ModelMetricsWindowAggregator: what goes into each 1-minute window."""

import json
from datetime import datetime, timezone

import pytest

from flink import analytics_job, logic

WINDOW_START_MS = 1_710_000_000_000


class _ListState:
    def __init__(self):
        self.values = []

    def get(self):
        return list(self.values)

    def update(self, values):
        self.values = list(values)

    def clear(self):
        self.values = []


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
    def __init__(self, timer_service, key):
        self._timer_service = timer_service
        self._key = key

    def timer_service(self):
        return self._timer_service

    def get_current_key(self):
        return self._key


def _completion(offset_ms=1_000, **overrides):
    timestamp = datetime.fromtimestamp(
        (WINDOW_START_MS + offset_ms) / 1000, tz=timezone.utc
    )
    event = {
        "event_type": "inference.completed",
        "request_id": f"request-{offset_ms}",
        "completion_timestamp": timestamp.isoformat(),
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "success",
        "latency_ms": 1000,
        "tokens_per_second": 50.0,
        "token_count_input": 10,
        "token_count_output": 20,
        "total_tokens": 30,
        "cost_usd": 0.01,
        "cached_response": False,
    }
    event.update(overrides)
    return event


def _window_metric(events, restored_buckets=()):
    aggregator = analytics_job.ModelMetricsWindowAggregator(
        window_size_seconds=60, allowed_lateness_seconds=15
    )
    aggregator.window_buckets_state = _ListState()
    aggregator.window_buckets_state.update(
        [json.dumps(bucket) for bucket in restored_buckets]
    )
    aggregator.flush_timer_state = _ValueState()
    aggregator.flush_timer_kind_state = _ValueState()
    ctx = _Context(_TimerService(now_ms=WINDOW_START_MS), key="gpt-5\0openai")

    for event in events:
        assert list(aggregator.process_element(event, ctx)) == []
    (metric,) = list(aggregator.on_timer(WINDOW_START_MS + 75_000, ctx))
    return metric


def test_window_latency_averages_only_uncached_events():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000, latency_ms=1000, cost_usd=0.01),
            _completion(offset_ms=2_000, latency_ms=3000, cost_usd=0.03),
            _completion(
                offset_ms=3_000, latency_ms=2, cost_usd=0.0, cached_response=True
            ),
        ]
    )

    assert metric["request_count"] == 3
    assert metric["cached_count"] == 1
    assert metric["cache_hit_rate"] == pytest.approx(1 / 3)
    assert metric["total_cost_usd"] == pytest.approx(0.04)
    assert metric["avg_latency_ms"] == pytest.approx(2000.0)


def test_window_of_only_cache_hits_reports_no_latency():
    metric = _window_metric(
        [_completion(latency_ms=3, cost_usd=0.0, cached_response=True)]
    )

    assert metric["request_count"] == 1
    assert metric["cached_count"] == 1
    assert metric["latency_sample_count"] == 0
    assert metric["avg_latency_ms"] == 0.0
    assert metric["avg_tokens_per_second"] == 0.0


def test_window_throughput_averages_only_uncached_events():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000, tokens_per_second=40.0),
            _completion(offset_ms=2_000, tokens_per_second=60.0),
            # A cache hit carries the original call's (stale) throughput.
            _completion(
                offset_ms=3_000,
                tokens_per_second=500.0,
                latency_ms=2,
                cost_usd=0.0,
                cached_response=True,
            ),
        ]
    )

    assert metric["latency_sample_count"] == 2
    assert metric["avg_tokens_per_second"] == pytest.approx(50.0)


def test_window_restored_from_older_state_keeps_averaging_every_request():
    legacy_bucket = {
        "window_start_ms": WINDOW_START_MS,
        "request_count": 2,
        "success_count": 2,
        "error_count": 0,
        "latency_sum_ms": 4000.0,
        "tokens_per_second_sum": 100.0,
        "token_count_input": 20,
        "token_count_output": 40,
        "total_tokens": 60,
        "total_cost_usd": 0.02,
        "cached_count": 0,
    }

    metric = _window_metric(
        [_completion(latency_ms=1000)], restored_buckets=[legacy_bucket]
    )

    assert metric["request_count"] == 3
    assert metric["avg_latency_ms"] == pytest.approx(5000.0 / 3)


def test_window_counts_provider_attributable_failures_beside_all_errors():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000),
            _completion(offset_ms=2_000, status="error", error_type="provider_error"),
            _completion(offset_ms=3_000, status="error", error_type="timeout"),
            _completion(offset_ms=4_000, status="error", error_type="user_or_policy"),
            _completion(offset_ms=5_000, status="error", error_type="internal"),
            # Older producers sent no error_type; their failures still count.
            _completion(offset_ms=6_000, status="error"),
        ]
    )

    assert metric["error_count"] == 5
    assert metric["provider_failure_count"] == 3
    # The user_or_policy and internal failures are not provider requests.
    assert metric["provider_request_count"] == 4
    assert metric["provider_failure_rate"] == pytest.approx(3 / 4)


def test_window_restored_from_older_state_counts_its_errors_as_provider_failures():
    legacy_bucket = {
        "window_start_ms": WINDOW_START_MS,
        "request_count": 2,
        "success_count": 0,
        "error_count": 2,
        "latency_sum_ms": 200.0,
        "tokens_per_second_sum": 0.0,
        "token_count_input": 0,
        "token_count_output": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "cached_count": 0,
    }

    metric = _window_metric(
        [_completion(status="error", error_type="user_or_policy")],
        restored_buckets=[legacy_bucket],
    )

    assert metric["error_count"] == 3
    assert metric["provider_failure_count"] == 2


def test_window_cost_per_1k_prices_only_uncached_events():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000, total_tokens=1000, cost_usd=0.04),
            _completion(offset_ms=2_000, total_tokens=1000, cost_usd=0.04),
            # A cache hit keeps its tokens but now costs nothing.
            _completion(
                offset_ms=3_000,
                total_tokens=2000,
                cost_usd=0.0,
                latency_ms=2,
                cached_response=True,
            ),
        ]
    )
    detector = analytics_job.ModelRoutingGuardrailDetector()
    detector.latency_history_state = _ListState()
    detector.qps_history_state = _ListState()
    detector.error_rate_history_state = _ListState()
    detector.cache_hit_rate_history_state = _ListState()
    detector.cost_per_1k_history_state = _ListState()
    detector.cost_per_1k_history_state.update([0.02] * 3)

    guardrails = list(detector.process_element(metric, None))

    # Uncached: $0.08 over 2000 tokens is $0.04 per 1k, double the baseline.
    # Counting the cached tokens would give $0.02 and hide the spike.
    assert metric["total_cost_usd"] == pytest.approx(0.08)
    assert detector.cost_per_1k_history_state.get()[-1] == pytest.approx(0.04)
    assert "cost_spike" in {g["trigger_type"] for g in guardrails}


def test_caller_failure_latency_does_not_reach_the_provider_guardrail():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000, latency_ms=3_000),
            # Its latency spans RAG, web search and every fallback attempt.
            _completion(
                offset_ms=2_000,
                latency_ms=13_000,
                status="error",
                error_type="user_or_policy",
                total_tokens=0,
                cost_usd=0.0,
            ),
        ]
    )

    guardrails = logic.detect_provider_routing_guardrails(
        metric, latency_history=[], error_rate_history=[]
    )

    # Model metrics still see every uncached request.
    assert metric["avg_latency_ms"] == pytest.approx(8_000.0)
    assert metric["provider_avg_latency_ms"] == pytest.approx(3_000.0)
    assert metric["provider_failure_rate"] == 0.0
    assert {g["trigger_type"] for g in guardrails} == set()


def test_provider_failure_rate_is_not_diluted_by_cache_hits_or_caller_failures():
    cache_hits = [
        _completion(
            offset_ms=10_000 + i, latency_ms=2, cost_usd=0.0, cached_response=True
        )
        for i in range(3)
    ]
    caller_failures = [
        _completion(offset_ms=20_000 + i, status="error", error_type=error_type)
        for i, error_type in enumerate(["user_or_policy", "internal"])
    ]
    metric = _window_metric(
        [
            _completion(offset_ms=1_000),
            _completion(offset_ms=2_000),
            _completion(offset_ms=3_000, status="error", error_type="provider_error"),
            _completion(offset_ms=4_000, status="error", error_type="provider_error"),
            *cache_hits,
            *caller_failures,
        ]
    )

    assert metric["request_count"] == 9
    assert metric["provider_request_count"] == 4
    assert metric["provider_failure_rate"] == pytest.approx(0.5)


def test_provider_latency_averages_successes_only():
    metric = _window_metric(
        [
            _completion(offset_ms=1_000, latency_ms=1_000),
            _completion(offset_ms=2_000, latency_ms=3_000),
            # A provider failure counts in the failure rate, not the latency.
            _completion(
                offset_ms=3_000,
                latency_ms=30_000,
                status="error",
                error_type="provider_error",
            ),
        ]
    )

    assert metric["provider_latency_sample_count"] == 2
    assert metric["provider_avg_latency_ms"] == pytest.approx(2_000.0)
    assert metric["provider_failure_rate"] == pytest.approx(1 / 3)


def test_window_restored_from_older_state_keeps_todays_provider_figures():
    legacy_bucket = {
        "window_start_ms": WINDOW_START_MS,
        "request_count": 2,
        "success_count": 1,
        "error_count": 1,
        "latency_sum_ms": 4000.0,
        "tokens_per_second_sum": 0.0,
        "token_count_input": 0,
        "token_count_output": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "cached_count": 0,
    }

    metric = _window_metric(
        [_completion(latency_ms=1000)], restored_buckets=[legacy_bucket]
    )

    assert metric["provider_request_count"] == 3
    assert metric["provider_failure_rate"] == pytest.approx(1 / 3)
    assert metric["provider_avg_latency_ms"] == pytest.approx(5000.0 / 3)
