from datetime import datetime

import pytest

from flink.logic import build_fast_lane_event, classify_query_event


@pytest.mark.integration
def test_flink_classifier_routes_enterprise_and_urgent_queries():
    timestamp = datetime(2026, 3, 13, 12, 0, 0)

    enterprise_event = classify_query_event(
        {
            "query_id": "integration-flink-enterprise",
            "query_text": "Investigate the incident report",
            "user_tier": "enterprise",
            "user_id": "enterprise-user",
        },
        timestamp=timestamp,
    )
    assert enterprise_event["priority"] == "high"
    assert enterprise_event["route_to_fast_lane"] is True

    urgent_event = build_fast_lane_event(
        {
            "query_id": "integration-flink-urgent",
            "query_text": "Critical production outage, respond ASAP",
            "user_tier": "free",
            "user_id": "free-user",
        },
        timestamp=timestamp,
    )
    assert urgent_event["priority"] == "critical"
    assert urgent_event["route_to_fast_lane"] is True
    assert urgent_event["fast_lane_processed"] is True


@pytest.mark.integration
def test_flink_classifier_keeps_standard_queries_off_fast_lane():
    event = classify_query_event(
        {
            "query_id": "integration-flink-standard",
            "query_text": "Please summarize this design document",
            "user_tier": "premium",
            "user_id": "premium-user",
        }
    )
    assert event["priority"] == "medium"
    assert event["route_to_fast_lane"] is False
