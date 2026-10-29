"""Provider failure attribution in the Flink routing policy and guardrail input.

Only provider-attributable failures should count against a provider; failures
caused by the caller, policy or our own code should not trip provider guardrails.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from flink import analytics_job, logic

_LEGACY = object()


def _completion(error_type=_LEGACY, **overrides):
    event = {
        "event_type": "inference.completed",
        "request_id": "request-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "user_tier": "premium",
        "query_type": "general",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "error",
        "latency_ms": 100,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    if error_type is not _LEGACY:
        event["error_type"] = error_type
    event.update(overrides)
    return event


@pytest.mark.parametrize(
    ("error_type", "counted"),
    [
        ("provider_error", True),
        ("timeout", True),
        ("rate_limited", True),
        ("user_or_policy", False),
        ("internal", False),
        (None, True),
        (_LEGACY, True),
    ],
)
def test_routing_policy_counts_only_provider_attributable_failures(error_type, counted):
    aggregate = logic.add_routing_policy_event(
        logic._empty_routing_policy_aggregate(),
        _completion(error_type=error_type),
    )

    assert aggregate["failed_provider_counts"] == ({"openai": 1} if counted else {})


@pytest.mark.parametrize(
    ("error_type", "avoid_providers"),
    [
        ("provider_error", ["openai"]),
        ("user_or_policy", []),
        (_LEGACY, ["openai"]),
    ],
)
def test_routing_policy_avoids_a_provider_only_for_its_own_failures(
    error_type, avoid_providers
):
    policy = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-1",
        events=[
            _completion(error_type=error_type, request_id=f"request-{index}")
            for index in range(3)
        ],
        timestamp=datetime(2026, 10, 28, 12, 0, 0),
    )

    assert policy["avoid_providers"] == avoid_providers


class _Stream:
    """Records the operator graph create_flink_job builds."""

    def __init__(self, nodes, kind="source", fn=None, parent=None):
        self.kind = kind
        self.fn = fn
        self.parent = parent
        self.sink_name = None
        nodes.append(self)
        self._nodes = nodes

    def _then(self, kind, fn):
        return _Stream(self._nodes, kind, fn, self)

    def map(self, fn, output_type=None):
        return self._then("map", fn)

    def filter(self, fn):
        return self._then("filter", fn)

    def key_by(self, fn):
        return self._then("key_by", fn)

    def process(self, fn, output_type=None):
        return self._then("process", fn)

    def assign_timestamps_and_watermarks(self, strategy):
        return self._then("watermarks", strategy)

    def add_sink(self, sink):
        return self._then("sink", sink)

    def name(self, name):
        self.sink_name = name
        return self

    def uid(self, uid):
        return self


class _Env:
    def __init__(self):
        self.nodes = []

    def set_parallelism(self, parallelism):
        pass

    def enable_checkpointing(self, interval_ms):
        pass

    def add_source(self, consumer, name):
        return _Stream(self.nodes)


@pytest.fixture
def job_graph(monkeypatch):
    env = _Env()
    monkeypatch.setattr(
        analytics_job,
        "StreamExecutionEnvironment",
        SimpleNamespace(get_execution_environment=lambda: env),
    )
    monkeypatch.setattr(analytics_job, "FlinkKafkaConsumer", lambda **kwargs: kwargs)
    monkeypatch.setattr(analytics_job, "FlinkKafkaProducer", lambda **kwargs: kwargs)
    monkeypatch.setattr(analytics_job, "SimpleStringSchema", lambda: None)
    monkeypatch.setattr(analytics_job, "WatermarkStrategy", None)
    analytics_job.create_flink_job({})
    return env.nodes


def _aggregator_feeding(node):
    """Walk up from an operator to the metrics window aggregator it consumes."""
    upstream = node.parent
    while upstream.kind != "process":
        upstream = upstream.parent
    assert isinstance(upstream.fn, analytics_job.ModelMetricsWindowAggregator)
    return upstream


def _request_ids_reaching(aggregator, payloads):
    """Push raw Kafka payloads through the stateless operators before aggregator."""
    path = []
    upstream = aggregator.parent
    while upstream is not None:
        path.append(upstream)
        upstream = upstream.parent
    path.reverse()

    reached = set()
    for payload in payloads:
        value = json.dumps(payload)
        for node in path:
            if node.kind == "map":
                value = (
                    node.fn.map(value) if hasattr(node.fn, "map") else node.fn(value)
                )
            elif node.kind == "filter":
                keep = (
                    node.fn.filter(value)
                    if hasattr(node.fn, "filter")
                    else node.fn(value)
                )
                if not keep:
                    break
            else:
                assert node.kind in {"source", "key_by", "watermarks"}
        else:
            reached.add(value["request_id"])
    return reached


_PAYLOADS = [
    _completion(request_id="success", status="success"),
    _completion(error_type="provider_error", request_id="provider_error"),
    _completion(error_type="user_or_policy", request_id="user_or_policy"),
    _completion(request_id="legacy"),
]


def test_provider_guardrail_reads_the_shared_model_metrics_windows(job_graph):
    (detector,) = [
        node
        for node in job_graph
        if node.kind == "process"
        and isinstance(node.fn, analytics_job.ProviderRoutingGuardrailDetector)
    ]
    (sink,) = [
        node
        for node in job_graph
        if node.kind == "sink" and node.sink_name == "Model Metrics 1m Sink"
    ]
    aggregators = [
        node
        for node in job_graph
        if node.kind == "process"
        and isinstance(node.fn, analytics_job.ModelMetricsWindowAggregator)
    ]

    assert aggregators == [_aggregator_feeding(sink)]
    assert _aggregator_feeding(detector) is aggregators[0]


def test_model_metrics_still_count_every_failure(job_graph):
    (sink,) = [
        node
        for node in job_graph
        if node.kind == "sink" and node.sink_name == "Model Metrics 1m Sink"
    ]

    reached = _request_ids_reaching(_aggregator_feeding(sink), _PAYLOADS)

    assert reached == {"success", "provider_error", "user_or_policy", "legacy"}


def _metric_event(error_count, provider_failure_count):
    return logic.build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=10,
        success_count=10 - error_count,
        error_count=error_count,
        latency_sum_ms=10_000.0,
        tokens_per_second_sum=500.0,
        token_count_input=100,
        token_count_output=200,
        total_tokens=300,
        total_cost_usd=0.1,
        cached_count=0,
        provider_failure_count=provider_failure_count,
        timestamp=datetime(2026, 10, 28, 12, 0, 0),
    )


def test_metric_event_reports_provider_failures_beside_all_errors():
    metric_event = _metric_event(error_count=6, provider_failure_count=2)

    assert metric_event["error_rate"] == pytest.approx(0.6)
    assert metric_event["provider_failure_count"] == 2
    assert metric_event["provider_failure_rate"] == pytest.approx(0.2)


def test_metric_event_without_provider_failure_count_treats_every_error_as_one():
    metric_event = logic.build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=0,
        window_end_ms=60_000,
        window_size_seconds=60,
        request_count=4,
        success_count=3,
        error_count=1,
        latency_sum_ms=400.0,
        tokens_per_second_sum=0.0,
        token_count_input=0,
        token_count_output=0,
        total_tokens=0,
        total_cost_usd=0.0,
        cached_count=0,
    )

    assert metric_event["provider_failure_count"] == 1
    assert metric_event["provider_failure_rate"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("error_count", "provider_failure_count", "tripped"),
    [
        pytest.param(6, 0, False, id="caller-errors-only"),
        pytest.param(6, 6, True, id="provider-failures"),
    ],
)
def test_provider_guardrail_trips_on_provider_failures_only(
    error_count, provider_failure_count, tripped
):
    guardrails = logic.detect_provider_routing_guardrails(
        _metric_event(error_count, provider_failure_count),
        latency_history=[],
        error_rate_history=[],
        timestamp=datetime(2026, 10, 28, 12, 1, 0),
    )

    assert (
        any(g["trigger_type"] == "provider_high_error_rate" for g in guardrails)
        is tripped
    )


class _HistoryState:
    def __init__(self):
        self.values = []

    def get(self):
        return list(self.values)

    def update(self, values):
        self.values = list(values)


def test_provider_guardrail_history_tracks_provider_failure_rate():
    detector = analytics_job.ProviderRoutingGuardrailDetector()
    detector.latency_history_state = _HistoryState()
    detector.error_rate_history_state = _HistoryState()

    list(
        detector.process_element(
            _metric_event(error_count=6, provider_failure_count=2), ctx=None
        )
    )

    assert detector.error_rate_history_state.values == [pytest.approx(0.2)]
