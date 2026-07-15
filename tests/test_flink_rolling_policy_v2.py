import json
from datetime import datetime

import pytest

from flink import logic
from flink import analytics_job
from flink.analytics_job import RollingScopePolicyEmitter


def _event(timestamp_ms=10_000, **overrides):
    event = {
        "request_id": f"request-{timestamp_ms}",
        "completion_timestamp": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
        "user_id": "user-1",
        "session_id": "session-1",
        "user_tier": "premium",
        "query_type": "analysis",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "success",
        "latency_ms": 100,
        "total_tokens": 200,
        "cost_usd": 0.01,
        "actual_fast_lane_hit": True,
        "route_to_fast_lane": True,
    }
    event.update(overrides)
    return event


def test_aggregate_policy_builder_matches_event_compatibility_builder():
    events = [
        _event(10_000),
        _event(
            12_000,
            status="error",
            query_type="summarization",
            selected_model="claude-sonnet",
            provider="anthropic",
            latency_ms=300,
            total_tokens=600,
            cost_usd=0.03,
            actual_fast_lane_hit=False,
        ),
    ]

    aggregate = logic.aggregate_routing_policy_events(events)
    from_events = logic.build_routing_policy_state_event(
        scope_type="user", scope_key="user-1", events=events
    )
    from_aggregate = logic.build_routing_policy_state_event_from_aggregate(
        scope_type="user", scope_key="user-1", aggregate=aggregate
    )

    ignored = {"emitted_at"}
    assert {k: v for k, v in from_aggregate.items() if k not in ignored} == {
        k: v for k, v in from_events.items() if k not in ignored
    }


class _ListState:
    def __init__(self, values=()):
        self.values = list(values)
        self.fail_clear = False
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        return list(self.values)

    def update(self, values):
        self.values = list(values)

    def clear(self):
        if self.fail_clear:
            raise RuntimeError("clear failed")
        self.values.clear()


class _MapState:
    def __init__(self):
        self.values = {}
        self.put_calls = 0
        self.fail_on_put_call = None

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.put_calls += 1
        if self.put_calls == self.fail_on_put_call:
            raise RuntimeError("map put failed")
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
    def __init__(self, value=None):
        self.current = value

    def value(self):
        return self.current

    def update(self, value):
        self.current = value

    def clear(self):
        self.current = None


class _TimerService:
    def __init__(self, now_ms=10_000, watermark_ms=10_000):
        self.now_ms = now_ms
        self.watermark_ms = watermark_ms
        self.registered = []
        self.deleted = []

    def current_processing_time(self):
        return self.now_ms

    def current_watermark(self):
        return self.watermark_ms

    def register_event_time_timer(self, timestamp):
        self.registered.append(timestamp)

    def delete_event_time_timer(self, timestamp):
        self.deleted.append(timestamp)


class _Context:
    def __init__(self, timer_service, key="user-1"):
        self._timer_service = timer_service
        self._key = key

    def timer_service(self):
        return self._timer_service

    def get_current_key(self):
        return self._key


def _emitter(*, legacy=(), window_seconds=10, bucket_seconds=5, emit_seconds=5):
    emitter = RollingScopePolicyEmitter(
        "user",
        window_size_seconds=window_seconds,
        bucket_size_seconds=bucket_seconds,
        emit_interval_seconds=emit_seconds,
    )
    emitter.recent_events_state = _ListState(legacy)
    emitter.bucket_state = _MapState()
    emitter.aggregate_state = _ValueState()
    emitter.cleanup_timer_state = _ValueState()
    emitter.cleanup_timer_kind_state = _ValueState()
    emitter.last_emit_state = _ValueState()
    emitter.dirty_state = _ValueState()
    emitter.migration_complete_state = _ValueState()
    emitter.arrival_sequence_state = _ValueState()
    return emitter


def test_out_of_order_event_identity_uses_arrival_order_then_restores_remaining_bucket():
    timer = _TimerService(now_ms=20_000, watermark_ms=20_000)
    ctx = _Context(timer)
    emitter = _emitter(window_seconds=10)

    first = list(emitter.process_element(_event(20_000, user_tier="premium"), ctx))
    list(emitter.process_element(_event(12_000, user_tier="enterprise"), ctx))
    aggregate = json.loads(emitter.aggregate_state.value())

    assert first[0]["user_tier"] == "premium"
    assert aggregate["latest_event"]["user_tier"] == "enterprise"

    list(emitter.on_timer(27_000, ctx))
    aggregate = json.loads(emitter.aggregate_state.value())
    assert aggregate["latest_event"]["user_tier"] == "premium"


def test_preserved_legacy_out_of_order_fixture_migrates_with_last_arrival_identity():
    legacy = [
        json.dumps(
            dict(_event(20_000, user_tier="premium"), event_timestamp_ms=20_000)
        ),
        json.dumps(
            dict(_event(12_000, user_tier="enterprise"), event_timestamp_ms=12_000)
        ),
    ]
    emitter = _emitter(legacy=legacy, window_seconds=20)
    timer = _TimerService(now_ms=21_000, watermark_ms=21_000)
    ctx = _Context(timer)

    list(emitter.process_element(_event(21_000, user_tier="free"), ctx))
    aggregate = json.loads(emitter.aggregate_state.value())

    assert aggregate["latest_event"]["user_tier"] == "free"
    assert aggregate["latest_arrival_sequence"] == 2
    migrated = [json.loads(value) for value in emitter.bucket_state.values.values()]
    enterprise_bucket = next(
        bucket
        for bucket in migrated
        if bucket.get("latest_event", {}).get("user_tier") == "enterprise"
    )
    assert enterprise_bucket["latest_arrival_sequence"] == 1


def test_first_event_emits_immediately_then_hot_key_waits_for_cadence_timer():
    timer = _TimerService(now_ms=10_000, watermark_ms=10_000)
    ctx = _Context(timer)
    emitter = _emitter()

    first = list(emitter.process_element(_event(10_000), ctx))
    second = list(emitter.process_element(_event(11_000), ctx))

    assert first[0]["recent_request_count"] == 1
    assert second == []
    assert 15_000 in timer.registered

    emitted = list(emitter.on_timer(15_000, ctx))
    assert emitted[0]["recent_request_count"] == 2


def test_one_thousand_hot_key_events_emit_at_most_twice_per_five_seconds():
    timer = _TimerService(now_ms=10_000, watermark_ms=10_000)
    ctx = _Context(timer)
    emitter = _emitter(window_seconds=300)

    outputs = []
    for sequence in range(1_000):
        timestamp_ms = 10_000 + min(sequence, 4_999)
        timer.now_ms = timestamp_ms
        timer.watermark_ms = timestamp_ms
        outputs.extend(emitter.process_element(_event(timestamp_ms), ctx))
    outputs.extend(emitter.on_timer(15_000, ctx))

    assert len(outputs) == 2
    assert outputs[-1]["recent_request_count"] == 1_000
    assert emitter.recent_events_state.get_calls == 1


def test_bucket_expiry_subtracts_aggregate_and_empty_state_emits_nothing():
    timer = _TimerService(now_ms=10_000, watermark_ms=10_000)
    ctx = _Context(timer)
    emitter = _emitter(window_seconds=10)
    list(emitter.process_element(_event(10_000), ctx))

    assert list(emitter.on_timer(25_000, ctx)) == []
    assert emitter.bucket_state.values == {}
    assert emitter.aggregate_state.value() is None


def test_late_event_inside_window_joins_its_event_time_bucket():
    timer = _TimerService(now_ms=20_000, watermark_ms=20_000)
    ctx = _Context(timer)
    emitter = _emitter(window_seconds=20)
    list(emitter.process_element(_event(20_000), ctx))
    list(emitter.process_element(_event(12_000), ctx))

    assert sorted(emitter.bucket_state.values) == [10_000, 20_000]
    aggregate = json.loads(emitter.aggregate_state.value())
    assert aggregate["request_count"] == 2


def test_legacy_list_state_migrates_lazily_and_only_clears_after_success():
    legacy_event = dict(_event(10_000), event_timestamp_ms=10_000)
    emitter = _emitter(legacy=[json.dumps(legacy_event)])
    timer = _TimerService(now_ms=11_000, watermark_ms=11_000)
    ctx = _Context(timer)

    list(emitter.process_element(_event(11_000), ctx))

    assert emitter.recent_events_state.values == []
    assert json.loads(emitter.aggregate_state.value())["request_count"] == 2


def test_first_event_migration_prunes_legacy_entries_outside_window():
    stale = dict(_event(1_000), event_timestamp_ms=1_000)
    current = _event(20_000)
    emitter = _emitter(legacy=[json.dumps(stale)], window_seconds=10)
    timer = _TimerService(now_ms=20_000, watermark_ms=20_000)
    ctx = _Context(timer)

    outputs = list(emitter.process_element(current, ctx))

    assert outputs[0]["recent_request_count"] == 1
    assert json.loads(emitter.aggregate_state.value())["request_count"] == 1
    assert emitter.recent_events_state.values == []


def test_partial_v2_migration_write_is_overwritten_without_duplicates_on_retry():
    legacy = [
        json.dumps(dict(_event(10_000), event_timestamp_ms=10_000)),
        json.dumps(dict(_event(16_000), event_timestamp_ms=16_000)),
    ]
    emitter = _emitter(legacy=legacy, window_seconds=20)
    emitter.bucket_state.fail_on_put_call = 2
    timer = _TimerService(now_ms=20_000, watermark_ms=20_000)
    ctx = _Context(timer)

    with pytest.raises(RuntimeError, match="map put failed"):
        list(emitter.process_element(_event(20_000), ctx))

    assert emitter.recent_events_state.values == legacy
    assert len(emitter.bucket_state.values) == 1

    emitter.bucket_state.fail_on_put_call = None
    outputs = list(emitter.process_element(_event(20_000), ctx))

    assert outputs == []
    assert json.loads(emitter.aggregate_state.value())["request_count"] == 3
    assert len(emitter.bucket_state.values) == 3


def test_failed_legacy_migration_preserves_old_list_state(monkeypatch):
    raw = json.dumps(dict(_event(10_000), event_timestamp_ms=10_000))
    emitter = _emitter(legacy=[raw])
    timer = _TimerService(now_ms=11_000, watermark_ms=11_000)
    ctx = _Context(timer)
    monkeypatch.setattr(
        emitter,
        "_write_migrated_state",
        lambda buckets: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        list(emitter.process_element(_event(11_000), ctx))

    assert emitter.recent_events_state.values == [raw]
    assert emitter.bucket_state.values == {}


def test_rolling_policy_operator_config_defaults_to_five_second_buckets_and_emit():
    assert analytics_job._rolling_policy_operator_config({}) == {
        "bucket_size_seconds": 5,
        "emit_interval_seconds": 5,
    }


def test_watermark_strategy_applies_sixty_second_idleness(monkeypatch):
    calls = []

    class _Duration:
        @staticmethod
        def of_seconds(seconds):
            return seconds

    class _Strategy:
        @classmethod
        def for_bounded_out_of_orderness(cls, duration):
            calls.append(("bounded", duration))
            return cls()

        def with_timestamp_assigner(self, assigner):
            calls.append(("assigner", type(assigner).__name__))
            return self

        def with_idleness(self, duration):
            calls.append(("idle", duration))
            return self

    monkeypatch.setattr(analytics_job, "Duration", _Duration)
    monkeypatch.setattr(analytics_job, "WatermarkStrategy", _Strategy)
    monkeypatch.setattr(
        analytics_job,
        "CompletionTimestampAssigner",
        type("CompletionTimestampAssigner", (), {}),
    )

    analytics_job._completion_watermark_strategy(15, 60)

    assert calls == [
        ("bounded", 15),
        ("assigner", "CompletionTimestampAssigner"),
        ("idle", 60),
    ]


def test_negative_uninitialized_watermark_does_not_drop_first_metrics_event():
    timer = _TimerService(now_ms=1_000_000_000, watermark_ms=-1)
    ctx = _Context(timer, key="gpt-5\0openai")
    aggregator = analytics_job.ModelMetricsWindowAggregator(
        window_size_seconds=60, allowed_lateness_seconds=15
    )
    aggregator.window_buckets_state = _ListState()
    aggregator.flush_timer_state = _ValueState()
    aggregator.flush_timer_kind_state = _ValueState()

    outputs = list(aggregator.process_element(_event(10_000), ctx))

    assert outputs == []
    buckets = [json.loads(item) for item in aggregator.window_buckets_state.values]
    assert buckets[0]["request_count"] == 1


def test_negative_uninitialized_watermark_does_not_drop_first_policy_event():
    timer = _TimerService(now_ms=1_000_000_000, watermark_ms=-1)
    ctx = _Context(timer)
    emitter = _emitter(window_seconds=10)

    outputs = list(emitter.process_element(_event(10_000), ctx))

    assert outputs[0]["recent_request_count"] == 1


def test_processing_time_fallback_remains_when_watermark_api_is_absent():
    class _ProcessingOnlyTimer:
        pass

    assert (
        analytics_job._current_time_progress_ms(
            _ProcessingOnlyTimer(), processing_time_ms=123_456
        )
        == 123_456
    )
