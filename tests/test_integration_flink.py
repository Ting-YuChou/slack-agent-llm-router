from datetime import datetime

import pytest

from flink.logic import (
    build_fast_lane_hint_event,
    build_request_enriched_event,
    classify_request_event,
)


@pytest.mark.integration
def test_flink_classifier_uses_explicit_sla_for_fast_lane():
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
    assert enterprise_event["priority"] == "low"
    assert enterprise_event["route_to_fast_lane"] is False
    assert enterprise_event["query_type"] == "analysis"
    assert enterprise_event["requires_high_reasoning"] is True

    urgent_keyword_event = build_request_enriched_event(
        {
            "request_id": "integration-flink-urgent",
            "query_text": "Critical production outage, respond ASAP",
            "user_tier": "free",
            "user_id": "free-user",
        },
        timestamp=timestamp,
    )
    assert urgent_keyword_event["event_type"] == "requests.enriched"
    assert urgent_keyword_event["priority"] == "low"
    assert urgent_keyword_event["route_to_fast_lane"] is False
    assert urgent_keyword_event["requires_low_latency"] is False

    priority_enriched = build_request_enriched_event(
        {
            "request_id": "integration-flink-priority",
            "query_text": "Summarize this thread",
            "user_tier": "free",
            "user_id": "free-user",
            "priority": 4,
        },
        timestamp=timestamp,
    )
    assert priority_enriched["priority"] == "high"
    assert priority_enriched["route_to_fast_lane"] is True
    assert priority_enriched["requires_low_latency"] is True

    priority_hint = build_fast_lane_hint_event(priority_enriched, timestamp=timestamp)
    assert priority_hint["event_type"] == "fast_lane_hints"
    assert priority_hint["route_to_fast_lane"] is True
    assert priority_hint["hint_type"] == "fast_lane_candidate"


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
    assert event["priority"] == "low"
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
