from datetime import datetime

import pytest

from flink.logic import (
    build_fast_lane_hint_event,
    build_request_enriched_event,
    classify_request_event,
)


@pytest.mark.integration
def test_flink_classifier_routes_enterprise_and_urgent_queries():
    timestamp = datetime(2026, 3, 13, 12, 0, 0)

    enterprise_event = classify_request_event(
        {
            "request_id": "integration-flink-enterprise",
            "query_text": "Investigate the incident report",
            "user_tier": "enterprise",
            "user_id": "enterprise-user",
        },
        timestamp=timestamp,
    )
    assert enterprise_event["priority"] == "high"
    assert enterprise_event["route_to_fast_lane"] is True
    assert enterprise_event["query_type"] == "analysis"
    assert enterprise_event["requires_high_reasoning"] is True

    urgent_enriched = build_request_enriched_event(
        {
            "request_id": "integration-flink-urgent",
            "query_text": "Critical production outage, respond ASAP",
            "user_tier": "free",
            "user_id": "free-user",
        },
        timestamp=timestamp,
    )
    assert urgent_enriched["event_type"] == "requests.enriched"
    assert urgent_enriched["priority"] == "critical"
    assert urgent_enriched["route_to_fast_lane"] is True
    assert urgent_enriched["requires_low_latency"] is True

    urgent_hint = build_fast_lane_hint_event(urgent_enriched, timestamp=timestamp)
    assert urgent_hint["event_type"] == "fast_lane_hints"
    assert urgent_hint["route_to_fast_lane"] is True
    assert urgent_hint["hint_type"] == "fast_lane_candidate"


@pytest.mark.integration
def test_flink_classifier_keeps_standard_queries_off_fast_lane():
    event = classify_request_event(
        {
            "request_id": "integration-flink-standard",
            "query_text": "Please summarize this design document",
            "user_tier": "premium",
            "user_id": "premium-user",
        }
    )
    assert event["priority"] == "medium"
    assert event["route_to_fast_lane"] is False


@pytest.mark.integration
def test_flink_feature_generator_marks_code_heavy_long_context_requests():
    enriched_event = build_request_enriched_event(
        {
            "request_id": "integration-flink-code-heavy",
            "query_text": "Review this Python traceback and refactor the function for better reliability",
            "user_tier": "premium",
            "user_id": "premium-user",
            "context": "x" * 5000,
            "attachments_count": 3,
            "metadata": {
                "session_query_count": 12,
                "cost_sensitivity": "high",
                "avoid_providers": ["openai"],
            },
        }
    )

    assert enriched_event["query_type"] == "code_analysis"
    assert enriched_event["query_complexity"] in {"moderate", "complex"}
    assert enriched_event["long_context"] is True
    assert enriched_event["attachment_heavy"] is True
    assert enriched_event["code_heavy"] is True
    assert enriched_event["session_hotness"] == "hot"
    assert enriched_event["cost_sensitivity"] == "high"
    assert enriched_event["avoid_providers"] == ["openai"]
