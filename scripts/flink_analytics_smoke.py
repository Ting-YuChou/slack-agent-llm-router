#!/usr/bin/env python3
"""
Smoke test for Flink analytics logic without requiring a local PyFlink runtime.
"""

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flink.logic import (
    build_model_metrics_window_event,
    detect_metric_anomalies,
    validate_inference_completed_event,
)


def main():
    event = validate_inference_completed_event(
        {
            "event_type": "inference.completed",
            "event_version": "1.0",
            "request_id": "analytics-smoke-1",
            "query_id": "analytics-smoke-1",
            "user_id": "u1",
            "selected_model": "gpt-5",
            "provider": "openai",
            "status": "success",
            "latency_ms": 200,
            "token_count_input": 10,
            "token_count_output": 20,
            "total_tokens": 30,
            "tokens_per_second": 100.0,
            "cost_usd": 0.01,
            "cached_response": False,
        }
    )
    assert event["event_type"] == "inference.completed"

    metric_event = build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=6,
        success_count=4,
        error_count=2,
        latency_sum_ms=4200.0,
        tokens_per_second_sum=900.0,
        token_count_input=120,
        token_count_output=240,
        total_tokens=360,
        total_cost_usd=0.12,
        cached_count=1,
        timestamp=datetime(2026, 4, 8, 12, 0, 0),
    )
    assert metric_event["event_type"] == "analytics.model_metrics_1m"
    assert metric_event["request_count"] == 6
    assert metric_event["avg_latency_ms"] == 700.0
    assert metric_event["error_rate"] == 2 / 6

    alerts = detect_metric_anomalies(
        metric_event,
        latency_history=[100.0, 120.0, 110.0, 130.0, 115.0],
        qps_history=[0.02, 0.03, 0.04, 0.02, 0.03],
        error_rate_history=[0.0, 0.01, 0.0, 0.02, 0.01],
        timestamp=datetime(2026, 4, 8, 12, 1, 0),
    )
    anomaly_types = {alert["anomaly_type"] for alert in alerts}
    assert "high_latency" in anomaly_types
    assert "high_volume" in anomaly_types
    assert "high_error_rate" in anomaly_types

    print("Flink analytics smoke test passed: metrics aggregation and anomaly detection are consistent")


if __name__ == "__main__":
    main()
