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

from flink.logic import build_fast_lane_event, classify_query_event


def main():
    timestamp = datetime(2026, 3, 13, 12, 0, 0)

    enterprise_event = classify_query_event(
        {
            "query_id": "flink-smoke-enterprise",
            "query_text": "Summarize the latest production incident",
            "user_tier": "enterprise",
            "user_id": "enterprise-user",
            "selected_model": "gpt-5",
        },
        timestamp=timestamp,
    )
    assert enterprise_event["priority"] == "high"
    assert enterprise_event["route_to_fast_lane"] is True

    urgent_event = build_fast_lane_event(
        {
            "query_id": "flink-smoke-urgent",
            "query_text": "Critical outage in production, fix immediately",
            "user_tier": "free",
            "user_id": "free-user",
            "selected_model": "gpt-5",
        },
        timestamp=timestamp,
    )
    assert urgent_event["priority"] == "critical"
    assert urgent_event["route_to_fast_lane"] is True
    assert urgent_event["fast_lane_processed"] is True

    premium_event = classify_query_event(
        {
            "query_id": "flink-smoke-premium",
            "query_text": "Analyze this dashboard for me",
            "user_tier": "premium",
            "user_id": "premium-user",
            "selected_model": "gpt-5",
        },
        timestamp=timestamp,
    )
    assert premium_event["priority"] == "medium"
    assert premium_event["route_to_fast_lane"] is False

    print("Flink smoke test passed: classifier and fast-lane outputs match expectations")


if __name__ == "__main__":
    main()
