"""
Pure-Python logic shared by the Flink job and local integration tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


HIGH_PRIORITY_KEYWORDS = [
    "urgent",
    "critical",
    "emergency",
    "asap",
    "immediately",
    "production",
    "outage",
    "down",
    "error",
    "bug",
]


def calculate_priority(query_text: str, user_tier: str, user_id: str = "") -> str:
    """Calculate priority using the same rules as the Flink job."""
    normalized_query = (query_text or "").lower()
    normalized_tier = (user_tier or "free").lower()

    if normalized_tier == "enterprise":
        return "high"

    if any(keyword in normalized_query for keyword in HIGH_PRIORITY_KEYWORDS):
        return "critical"

    if normalized_tier == "premium":
        return "medium"

    return "low"


def classify_query_event(data: Dict[str, Any], timestamp: datetime | None = None) -> Dict[str, Any]:
    """Apply Flink query-priority enrichment to an event payload."""
    event = dict(data)
    priority = calculate_priority(
        query_text=event.get("query_text", ""),
        user_tier=event.get("user_tier", "free"),
        user_id=event.get("user_id", ""),
    )
    event["priority"] = priority
    event["processing_timestamp"] = (timestamp or datetime.now()).isoformat()
    event["route_to_fast_lane"] = priority in {"high", "critical"}
    return event


def build_fast_lane_event(data: Dict[str, Any], timestamp: datetime | None = None) -> Dict[str, Any]:
    """Build the fast-lane output payload produced by the Flink job."""
    event = classify_query_event(data, timestamp=timestamp)
    event["fast_lane_processed"] = event["route_to_fast_lane"]
    event["processing_timestamp"] = (timestamp or datetime.now()).isoformat()
    return event
