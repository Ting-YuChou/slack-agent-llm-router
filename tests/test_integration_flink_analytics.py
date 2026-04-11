from datetime import datetime

import pytest

from flink.logic import (
    build_model_metrics_window_event,
    detect_metric_anomalies,
    validate_inference_completed_event,
)


@pytest.mark.integration
def test_flink_analytics_builds_windowed_model_metrics():
    event = validate_inference_completed_event(
        {
            "event_type": "inference.completed",
            "event_version": "1.0",
            "request_id": "integration-analytics-completed",
            "query_id": "integration-analytics-completed",
            "selected_model": "gpt-5",
            "provider": "openai",
            "status": "success",
            "latency_ms": 180,
            "token_count_input": 12,
            "token_count_output": 24,
            "total_tokens": 36,
            "tokens_per_second": 120.0,
            "cost_usd": 0.02,
        }
    )
    assert event["event_type"] == "inference.completed"

    metric_event = build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=3,
        success_count=2,
        error_count=1,
        latency_sum_ms=900.0,
        tokens_per_second_sum=330.0,
        token_count_input=36,
        token_count_output=72,
        total_tokens=108,
        total_cost_usd=0.06,
        cached_count=1,
        timestamp=datetime(2026, 4, 8, 12, 0, 0),
    )
    assert metric_event["event_type"] == "analytics.model_metrics_1m"
    assert metric_event["avg_latency_ms"] == 300.0
    assert metric_event["success_rate"] == pytest.approx(2 / 3)
    assert metric_event["error_rate"] == pytest.approx(1 / 3)
    assert metric_event["cache_hit_rate"] == pytest.approx(1 / 3)


@pytest.mark.integration
def test_flink_analytics_detects_high_latency_error_and_volume():
    metric_event = build_model_metrics_window_event(
        model_name="mistral-7b",
        provider="vllm",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=12,
        success_count=8,
        error_count=4,
        latency_sum_ms=7200.0,
        tokens_per_second_sum=1200.0,
        token_count_input=120,
        token_count_output=360,
        total_tokens=480,
        total_cost_usd=0.0,
        cached_count=0,
        timestamp=datetime(2026, 4, 8, 12, 1, 0),
    )

    alerts = detect_metric_anomalies(
        metric_event,
        latency_history=[200.0, 240.0, 210.0, 220.0, 230.0],
        qps_history=[0.05, 0.07, 0.06, 0.05, 0.04],
        error_rate_history=[0.0, 0.01, 0.0, 0.02, 0.01],
        timestamp=datetime(2026, 4, 8, 12, 2, 0),
    )

    anomaly_types = {alert["anomaly_type"] for alert in alerts}
    assert "high_latency" in anomaly_types
    assert "high_volume" in anomaly_types
    assert "high_error_rate" in anomaly_types
