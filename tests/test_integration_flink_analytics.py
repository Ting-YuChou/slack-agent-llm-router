from datetime import datetime

import pytest

from flink.analytics_job import (
    ModelMetricsWindowAggregator,
    _model_provider_stream_key,
    _split_model_provider_stream_key,
)
from flink.logic import (
    build_model_metrics_window_event,
    build_routing_policy_state_event,
    detect_metric_anomalies,
    detect_model_routing_guardrails,
    detect_provider_routing_guardrails,
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
def test_flink_analytics_keeps_same_model_name_separate_across_providers():
    openai_key = _model_provider_stream_key("gpt-5", "openai")
    azure_key = _model_provider_stream_key("gpt-5", "azure")

    assert openai_key != azure_key
    assert _split_model_provider_stream_key(openai_key) == ("gpt-5", "openai")
    assert _split_model_provider_stream_key(azure_key) == ("gpt-5", "azure")

    aggregator = ModelMetricsWindowAggregator()
    bucket = {
        "window_start_ms": 1_710_000_000_000,
        "request_count": 4,
        "success_count": 4,
        "error_count": 0,
        "latency_sum_ms": 800.0,
        "tokens_per_second_sum": 400.0,
        "token_count_input": 100,
        "token_count_output": 200,
        "total_tokens": 300,
        "total_cost_usd": 0.04,
        "cached_count": 1,
    }

    openai_metric = aggregator._build_metric_event(model_name=openai_key, bucket=bucket)
    azure_metric = aggregator._build_metric_event(model_name=azure_key, bucket=bucket)

    assert openai_metric["model_name"] == "gpt-5"
    assert openai_metric["provider"] == "openai"
    assert azure_metric["model_name"] == "gpt-5"
    assert azure_metric["provider"] == "azure"


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


@pytest.mark.integration
def test_flink_analytics_emits_model_and_provider_routing_guardrails():
    metric_event = build_model_metrics_window_event(
        model_name="gpt-5",
        provider="openai",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=10,
        success_count=6,
        error_count=4,
        latency_sum_ms=62000.0,
        tokens_per_second_sum=900.0,
        token_count_input=600,
        token_count_output=1400,
        total_tokens=2000,
        total_cost_usd=0.20,
        cached_count=0,
        timestamp=datetime(2026, 4, 8, 12, 3, 0),
    )

    model_guardrails = detect_model_routing_guardrails(
        metric_event,
        latency_history=[900.0, 1100.0, 950.0],
        qps_history=[0.9, 1.1, 1.0],
        error_rate_history=[0.01, 0.02, 0.01],
        cache_hit_rate_history=[0.50, 0.45, 0.48],
        cost_per_1k_history=[0.03, 0.04, 0.035],
        timestamp=datetime(2026, 4, 8, 12, 4, 0),
    )
    provider_guardrails = detect_provider_routing_guardrails(
        metric_event,
        latency_history=[1000.0, 1200.0, 950.0],
        error_rate_history=[0.01, 0.02, 0.01],
        timestamp=datetime(2026, 4, 8, 12, 4, 0),
    )

    assert any(
        guardrail["event_type"] == "routing.guardrails"
        and guardrail["scope_type"] == "model"
        and guardrail["scope_key"] == "gpt-5"
        for guardrail in model_guardrails
    )
    assert any(
        guardrail["event_type"] == "routing.guardrails"
        and guardrail["scope_type"] == "provider"
        and guardrail["scope_key"] == "openai"
        for guardrail in provider_guardrails
    )


@pytest.mark.integration
def test_flink_analytics_guardrail_thresholds_are_configurable():
    metric_event = build_model_metrics_window_event(
        model_name="mistral-7b",
        provider="vllm",
        window_start_ms=1_710_000_000_000,
        window_end_ms=1_710_000_060_000,
        window_size_seconds=60,
        request_count=8,
        success_count=6,
        error_count=2,
        latency_sum_ms=16000.0,
        tokens_per_second_sum=960.0,
        token_count_input=200,
        token_count_output=400,
        total_tokens=600,
        total_cost_usd=0.03,
        cached_count=0,
        timestamp=datetime(2026, 4, 8, 12, 5, 0),
    )

    model_guardrails = detect_model_routing_guardrails(
        metric_event,
        latency_history=[1000.0, 1100.0, 1200.0],
        qps_history=[0.8, 0.9, 0.85],
        error_rate_history=[0.05, 0.08, 0.07],
        cache_hit_rate_history=[0.2, 0.25, 0.22],
        cost_per_1k_history=[0.03, 0.031, 0.032],
        thresholds={"high_error_rate": 0.20, "high_latency_ms": 1500},
        timestamp=datetime(2026, 4, 8, 12, 6, 0),
    )
    provider_guardrails = detect_provider_routing_guardrails(
        metric_event,
        latency_history=[1000.0, 1200.0, 1100.0],
        error_rate_history=[0.05, 0.06, 0.07],
        thresholds={"high_error_rate": 0.20, "high_latency_ms": 1500},
        timestamp=datetime(2026, 4, 8, 12, 6, 0),
    )

    assert any(g["trigger_type"] == "high_error_rate" for g in model_guardrails)
    assert any(g["trigger_type"] == "high_latency" for g in model_guardrails)
    assert any(
        g["trigger_type"] == "provider_high_latency" for g in provider_guardrails
    )


@pytest.mark.integration
def test_flink_analytics_builds_user_rolling_policy_state():
    policy_event = build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-rolling-1",
        window_size_seconds=300,
        events=[
            {
                "user_id": "user-rolling-1",
                "user_tier": "free",
                "query_type": "summarization",
                "selected_model": "gpt-5",
                "provider": "openai",
                "status": "success",
                "latency_ms": 600,
                "total_tokens": 900,
                "cost_usd": 0.030,
                "route_to_fast_lane": True,
                "actual_fast_lane_hit": True,
            }
            for _ in range(10)
        ],
        timestamp=datetime(2026, 4, 17, 12, 5, 0),
    )

    assert policy_event["event_type"] == "routing.policy_state"
    assert policy_event["scope_type"] == "user"
    assert policy_event["burst_protection_active"] is True
    assert policy_event["cost_sensitivity"] == "high"
    assert policy_event["avoid_models"] == ["gpt-5"]
    assert policy_event["route_to_fast_lane"] is True


@pytest.mark.integration
def test_flink_analytics_builds_session_model_pin_policy_state():
    policy_event = build_routing_policy_state_event(
        scope_type="session",
        scope_key="session-rolling-1",
        window_size_seconds=300,
        events=[
            {
                "user_id": "enterprise-1",
                "session_id": "session-rolling-1",
                "user_tier": "enterprise",
                "query_type": "analysis",
                "selected_model": "gpt-5",
                "provider": "openai",
                "status": "success",
                "latency_ms": 1200,
                "total_tokens": 1800,
                "cost_usd": 0.025,
                "route_to_fast_lane": False,
                "actual_fast_lane_hit": False,
            }
            for _ in range(4)
        ],
        timestamp=datetime(2026, 4, 17, 12, 6, 0),
    )

    assert policy_event["scope_type"] == "session"
    assert policy_event["preferred_models"] == ["gpt-5"]
    assert policy_event["requires_high_reasoning"] is True
    assert policy_event["query_complexity"] in {"moderate", "complex"}


@pytest.mark.integration
def test_flink_analytics_fast_lane_hit_rate_uses_actual_hits():
    policy_event = build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-rolling-2",
        window_size_seconds=300,
        events=[
            {
                "user_id": "user-rolling-2",
                "user_tier": "premium",
                "query_type": "summarization",
                "selected_model": "gpt-5",
                "provider": "openai",
                "status": "success",
                "latency_ms": 300,
                "total_tokens": 300,
                "cost_usd": 0.005,
                "route_to_fast_lane": True,
                "actual_fast_lane_hit": False,
            }
            for _ in range(6)
        ],
        timestamp=datetime(2026, 4, 17, 12, 7, 0),
    )

    assert policy_event["fast_lane_hit_rate"] == 0.0
    assert policy_event["route_to_fast_lane"] is False


@pytest.mark.integration
def test_flink_analytics_rolling_policy_uses_configurable_thresholds():
    policy_event = build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-rolling-3",
        window_size_seconds=300,
        policy_config={
            "burst_request_count": 3,
            "burst_min_avg_total_tokens": 200,
            "free_high_cost_threshold_usd": 0.01,
        },
        events=[
            {
                "user_id": "user-rolling-3",
                "user_tier": "free",
                "query_type": "summarization",
                "selected_model": "gpt-5",
                "provider": "openai",
                "status": "success",
                "latency_ms": 450,
                "total_tokens": 300,
                "cost_usd": 0.02,
                "route_to_fast_lane": True,
                "actual_fast_lane_hit": True,
            }
            for _ in range(3)
        ],
        timestamp=datetime(2026, 4, 17, 12, 8, 0),
    )

    assert policy_event["burst_protection_active"] is True
    assert policy_event["cost_sensitivity"] == "high"
