#!/usr/bin/env python3
"""
Smoke test for Flink job logic without requiring a local PyFlink runtime.
"""

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flink.logic import (
    build_fast_lane_hint_event,
    build_request_enriched_event,
    classify_request_event,
)


def main():
    timestamp = datetime(2026, 3, 13, 12, 0, 0)

    enterprise_event = classify_request_event(
        {
            "request_id": "flink-smoke-enterprise",
            "query_text": "Summarize the latest production incident",
            "user_tier": "enterprise",
            "user_id": "enterprise-user",
        },
        timestamp=timestamp,
    )
    assert enterprise_event["priority"] == "high"
    assert enterprise_event["route_to_fast_lane"] is True

    enriched_event = build_request_enriched_event(
        {
            "request_id": "flink-smoke-urgent",
            "query_text": "Critical outage in production, fix immediately",
            "user_tier": "free",
            "user_id": "free-user",
        },
        timestamp=timestamp,
    )
    assert enriched_event["event_type"] == "requests.enriched"
    assert enriched_event["priority"] == "critical"
    assert enriched_event["route_to_fast_lane"] is True

    fast_lane_hint = build_fast_lane_hint_event(enriched_event, timestamp=timestamp)
    assert fast_lane_hint["event_type"] == "fast_lane_hints"
    assert fast_lane_hint["route_to_fast_lane"] is True
    assert fast_lane_hint["hint_type"] == "fast_lane_candidate"

    premium_event = classify_request_event(
        {
            "request_id": "flink-smoke-premium",
            "query_text": "Analyze this dashboard for me",
            "user_tier": "premium",
            "user_id": "premium-user",
        },
        timestamp=timestamp,
    )
    assert premium_event["priority"] == "medium"
    assert premium_event["route_to_fast_lane"] is False

    print("Flink smoke test passed: enrichment and fast-lane hint outputs match expectations")


if __name__ == "__main__":
    main()
