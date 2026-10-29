"""Default timestamps are true UTC whatever the host timezone is (X7)."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from flink.logic import _isoformat as flink_isoformat
from src.llm_router_part3_pipeline import (
    _parse_datetime,
    build_inference_completed_event,
)
from src.llm_router_part4_monitor import MetricsCollector
from src.utils.schema import (
    ComponentHealth,
    ErrorResponse,
    HealthStatus,
    InferenceResponse,
    QueryRequest,
    QueryType,
    RoutingDecision,
    SystemMetric,
    UserSession,
    UserTier,
)

TOLERANCE = timedelta(seconds=2)
TAIPEI_OFFSET_SECONDS = 8 * 3600


@pytest.fixture
def taipei_tz(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    monkeypatch.setenv("TZ", "Asia/Taipei")
    time.tzset()
    assert time.localtime().tm_gmtoff == TAIPEI_OFFSET_SECONDS
    try:
        yield
    finally:
        monkeypatch.undo()
        time.tzset()


def _assert_true_utc(value: datetime) -> None:
    assert value.tzinfo is not None, f"{value!r} is naive"
    assert value.utcoffset() == timedelta(0), f"{value!r} is not UTC"
    assert abs(value - datetime.now(timezone.utc)) <= TOLERANCE


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _query_request() -> QueryRequest:
    return QueryRequest(query="hello", user_id="user-1")


def _inference_response() -> InferenceResponse:
    return InferenceResponse(
        response_text="world",
        model_name="gpt-5",
        provider="openai",
        token_count_input=3,
        token_count_output=5,
        total_tokens=8,
        latency_ms=12,
        tokens_per_second=120.0,
        cost_usd=0.01,
    )


def _health_status() -> HealthStatus:
    return HealthStatus(service="api", status="healthy")


DEFAULT_TIMESTAMP_FIELDS = [
    pytest.param(_query_request, "timestamp", id="QueryRequest.timestamp"),
    pytest.param(_inference_response, "timestamp", id="InferenceResponse.timestamp"),
    pytest.param(
        lambda: RoutingDecision(
            selected_model="gpt-5",
            query_type=QueryType.GENERAL,
            routing_reason="default",
            token_count=1,
            estimated_cost=0.0,
            routing_time_ms=1,
            confidence=0.5,
        ),
        "timestamp",
        id="RoutingDecision.timestamp",
    ),
    pytest.param(
        lambda: SystemMetric(name="cpu_usage", value=1.0),
        "timestamp",
        id="SystemMetric.timestamp",
    ),
    pytest.param(
        lambda: UserSession(user_id="user-1", user_tier=UserTier.FREE),
        "start_time",
        id="UserSession.start_time",
    ),
    pytest.param(
        lambda: UserSession(user_id="user-1", user_tier=UserTier.FREE),
        "last_activity",
        id="UserSession.last_activity",
    ),
    pytest.param(
        lambda: ErrorResponse(error_type="x", error_message="y"),
        "timestamp",
        id="ErrorResponse.timestamp",
    ),
    pytest.param(_health_status, "timestamp", id="HealthStatus.timestamp"),
    pytest.param(
        lambda: ComponentHealth(
            overall_status="healthy", components={"api": _health_status()}
        ),
        "timestamp",
        id="ComponentHealth.timestamp",
    ),
]


@pytest.mark.parametrize("factory, field_name", DEFAULT_TIMESTAMP_FIELDS)
def test_schema_default_timestamps_are_true_utc(taipei_tz, factory, field_name):
    _assert_true_utc(getattr(factory(), field_name))


def test_inference_completed_event_timestamps_are_true_utc(taipei_tz):
    event = build_inference_completed_event(_query_request(), _inference_response())

    for key in ("emitted_at", "request_timestamp", "completion_timestamp"):
        _assert_true_utc(_parse_iso(event[key]))


def test_flink_isoformat_default_is_true_utc(taipei_tz):
    serialized = flink_isoformat()

    assert serialized.endswith("+00:00")
    _assert_true_utc(datetime.fromisoformat(serialized))


def test_parse_datetime_reads_naive_strings_as_utc(taipei_tz):
    parsed = _parse_datetime("2026-10-28T07:50:00")

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parsed == datetime(2026, 10, 28, 7, 50, tzinfo=timezone.utc)


def test_parse_datetime_keeps_naive_as_utc_contract(taipei_tz):
    expected = datetime(2026, 10, 28, 7, 50, tzinfo=timezone.utc)

    assert _parse_datetime(datetime(2026, 10, 28, 7, 50)) == expected
    assert _parse_datetime(datetime(2026, 10, 28, 7, 50)).tzinfo is timezone.utc
    assert _parse_datetime("2026-10-28T07:50:00Z") == expected
    assert _parse_datetime("2026-10-28T15:50:00+08:00") == expected
    assert _parse_datetime("2026-10-28T15:50:00+08:00").utcoffset() == timedelta(0)


def _record_stream_events(collector: MetricsCollector, emitted_at: str) -> None:
    collector.record_stream_model_metrics(
        {
            "event_type": "analytics.model_metrics_1m",
            "emitted_at": emitted_at,
            "model_name": "gpt-5",
            "provider": "openai",
            "request_count": 4,
        }
    )
    collector.record_request_routing_features(
        {"event_type": "requests.enriched", "emitted_at": emitted_at}
    )
    collector.record_routing_guardrail(
        {"event_type": "routing.guardrails", "emitted_at": emitted_at}
    )
    collector.record_routing_policy_state(
        {"event_type": "routing.policy_state", "emitted_at": emitted_at}
    )


def _stream_counts(collector: MetricsCollector) -> dict:
    return {
        "fresh_requests": collector.get_stream_metrics_summary(hours=1)[
            "request_count"
        ],
        "recent_windows": collector.get_stream_metrics_summary(hours=1)["sample_count"],
        "routing_features": collector.get_routing_feature_summary(hours=1)[
            "request_count"
        ],
        "guardrails": collector.get_routing_guardrail_summary(hours=1)[
            "guardrail_count"
        ],
        "policy_states": collector.get_routing_policy_state_summary(hours=1)[
            "state_count"
        ],
    }


def test_monitor_keeps_fresh_utc_stream_events(taipei_tz):
    collector = MetricsCollector({"stream_metrics_staleness_seconds": 300})

    _record_stream_events(collector, flink_isoformat())

    assert _stream_counts(collector) == {
        "fresh_requests": 4,
        "recent_windows": 1,
        "routing_features": 1,
        "guardrails": 1,
        "policy_states": 1,
    }


def test_monitor_drops_stale_utc_stream_events(taipei_tz):
    collector = MetricsCollector({"stream_metrics_staleness_seconds": 300})
    two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)

    _record_stream_events(collector, two_hours_ago.isoformat().replace("+00:00", "Z"))

    assert _stream_counts(collector) == {
        "fresh_requests": 0,
        "recent_windows": 0,
        "routing_features": 0,
        "guardrails": 0,
        "policy_states": 0,
    }


def test_monitor_reads_naive_stream_timestamps_as_host_local(taipei_tz):
    collector = MetricsCollector({"stream_metrics_staleness_seconds": 300})

    _record_stream_events(collector, datetime.now().isoformat())

    assert _stream_counts(collector)["fresh_requests"] == 4
    assert _stream_counts(collector)["policy_states"] == 1
