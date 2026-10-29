"""Round-1 review: guardrail windows, expiry and partial windows (A2)."""

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


# 4. Windows must be consecutive, and an active guardrail expires.


def test_bad_windows_separated_by_a_gap_do_not_enter():
    detector = _model_detector()

    _run(detector, _window(0, errors=6))
    after_gap = _run(detector, _window(2, errors=6))

    assert "avoid" not in _actions(after_gap)


def test_an_active_guardrail_expires_after_its_ttl_without_windows():
    detector = _model_detector()
    _run(detector, _window(0, errors=6))
    assert "avoid" in _actions(_run(detector, _window(1, errors=6)))

    # The model goes quiet for longer than the guardrail TTL (180 s).
    resumed = _run(detector, _window(10, errors=6))

    assert "avoid" not in _actions(resumed)
    # It must pass the enter gate again.
    assert "avoid" in _actions(_run(detector, _window(11, errors=6)))


# 5. A provider avoid decided on a partial window is retracted.


def test_a_partial_window_avoid_is_cleared_when_the_window_turns_healthy():
    detector = _provider_detector()
    _run(detector, _window(0, model="a", requests=6, errors=6))
    entered = _run(detector, _window(1, model="a", requests=6, errors=6))
    assert "avoid" in _actions(entered)

    # The rest of window 1 arrives: a busy healthy model.
    rest = _run(detector, _window(1, model="b", requests=100, errors=0))

    assert _actions(rest) == ["clear"]
    assert rest[0]["scope_type"] == "provider"
    assert _run(detector, _window(2, model="a", requests=10, errors=0)) == []
