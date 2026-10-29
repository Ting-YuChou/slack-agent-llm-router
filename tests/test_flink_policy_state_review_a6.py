"""Round-1 review: latest_event_at is the newest event time (A6)."""

from datetime import datetime, timezone

from flink import analytics_job, logic

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


def _window(index, *, model="gpt-5", requests=10, errors=0):
    start = BASE_MS + index * WINDOW_MS
    return logic.build_model_metrics_window_event(
        model_name=model,
        provider="openai",
        window_start_ms=start,
        window_end_ms=start + WINDOW_MS,
        window_size_seconds=60,
        request_count=requests,
        success_count=requests - errors,
        error_count=errors,
        latency_sum_ms=1000.0 * requests,
        tokens_per_second_sum=50.0 * requests,
        token_count_input=100 * requests,
        token_count_output=50 * requests,
        total_tokens=150 * requests,
        total_cost_usd=0.001 * requests,
        cached_count=0,
        timestamp=TIMESTAMP,
    )


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


def _actions(outputs):
    return [g["guardrail_action"] for g in outputs]


# 6. latest_event_at is the newest event time, not the last arrival.


def _completion(event_timestamp_ms):
    return {
        "user_id": "u1",
        "user_tier": "free",
        "query_type": "general",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "success",
        "latency_ms": 100,
        "total_tokens": 10,
        "event_timestamp_ms": event_timestamp_ms,
    }


def test_latest_event_at_is_the_maximum_event_time():
    newest_ms = 1_776_000_002_000
    event = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="u1",
        # The newer completion arrives first; a late one follows.
        events=[_completion(newest_ms), _completion(1_776_000_001_000)],
    )

    assert (
        event["latest_event_at"]
        == datetime.fromtimestamp(newest_ms / 1000, tz=timezone.utc).isoformat()
    )


def test_merged_buckets_keep_the_maximum_event_time():
    newer = logic.add_routing_policy_event(
        logic._empty_routing_policy_aggregate(),
        _completion(1_776_000_002_000),
        event_timestamp_ms=1_776_000_002_000,
        arrival_sequence=1,
    )
    late = logic.add_routing_policy_event(
        logic._empty_routing_policy_aggregate(),
        _completion(1_776_000_001_000),
        event_timestamp_ms=1_776_000_001_000,
        arrival_sequence=2,
    )
    aggregate = logic._empty_routing_policy_aggregate()
    logic.merge_routing_policy_aggregate(aggregate, newer)
    logic.merge_routing_policy_aggregate(aggregate, late)

    event = logic.build_routing_policy_state_event_from_aggregate(
        scope_type="user", scope_key="u1", aggregate=aggregate
    )

    assert event["latest_event_at"].startswith(
        datetime.fromtimestamp(1_776_000_002, tz=timezone.utc).isoformat()[:19]
    )
