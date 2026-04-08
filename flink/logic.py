"""
Pure-Python logic shared by the Flink job and local integration tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

EVENT_SCHEMA_VERSION = "1.0"
REQUESTS_RAW_EVENT = "requests.raw"
REQUESTS_ENRICHED_EVENT = "requests.enriched"
INFERENCE_COMPLETED_EVENT = "inference.completed"
FAST_LANE_HINT_EVENT = "fast_lane_hints"
ALERT_EVENT = "alerts"
MODEL_METRICS_1M_EVENT = "analytics.model_metrics_1m"
REQUIRED_REQUEST_FIELDS = ("request_id", "query_text", "user_id", "user_tier")
REQUIRED_COMPLETION_FIELDS = (
    "request_id",
    "selected_model",
    "provider",
    "status",
    "latency_ms",
)

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


def _isoformat(timestamp: datetime | None = None) -> str:
    """Serialize timestamps consistently for event payloads."""
    return (timestamp or datetime.now()).isoformat()


def normalize_request_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize alias fields so Flink can process API request events consistently."""
    event = dict(data)

    if event.get("query") and not event.get("query_text"):
        event["query_text"] = event["query"]

    request_id = event.get("request_id") or event.get("query_id")
    if request_id:
        event["request_id"] = request_id
        event.setdefault("query_id", request_id)

    event.setdefault("event_type", REQUESTS_RAW_EVENT)
    event.setdefault("event_version", EVENT_SCHEMA_VERSION)
    return event


def validate_request_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the minimal contract required by the enrichment job."""
    event = normalize_request_event(data)
    missing_fields = [
        field for field in REQUIRED_REQUEST_FIELDS if not event.get(field)
    ]
    if missing_fields:
        raise ValueError(
            f"Missing required request fields: {', '.join(sorted(missing_fields))}"
        )
    return event


def normalize_inference_completed_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize post-inference events so analytics jobs can validate them."""
    event = dict(data)

    request_id = event.get("request_id") or event.get("query_id")
    if request_id:
        event["request_id"] = request_id
        event.setdefault("query_id", request_id)

    event.setdefault("event_type", INFERENCE_COMPLETED_EVENT)
    event.setdefault("event_version", EVENT_SCHEMA_VERSION)
    event.setdefault("status", "success")
    event.setdefault("cached_response", False)
    event.setdefault("token_count_input", 0)
    event.setdefault("token_count_output", 0)
    event.setdefault("total_tokens", 0)
    event.setdefault("tokens_per_second", 0.0)
    event.setdefault("cost_usd", 0.0)
    event.setdefault("completion_timestamp", event.get("emitted_at"))
    return event


def validate_inference_completed_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the minimal contract required by the analytics job."""
    event = normalize_inference_completed_event(data)
    missing_fields = [
        field for field in REQUIRED_COMPLETION_FIELDS if event.get(field) is None
    ]
    if missing_fields:
        raise ValueError(
            "Missing required inference.completed fields: "
            + ", ".join(sorted(missing_fields))
        )
    return event


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


def classify_query_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Apply Flink query-priority enrichment to an event payload."""
    event = validate_request_event(data)
    priority = calculate_priority(
        query_text=event.get("query_text", ""),
        user_tier=event.get("user_tier", "free"),
        user_id=event.get("user_id", ""),
    )
    event["priority"] = priority
    event["processing_timestamp"] = _isoformat(timestamp)
    event["route_to_fast_lane"] = priority in {"high", "critical"}
    return event


def build_request_enriched_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Build the enrichment output payload produced by Flink."""
    event = classify_query_event(data, timestamp=timestamp)
    event["event_type"] = REQUESTS_ENRICHED_EVENT
    event["emitted_at"] = _isoformat(timestamp)
    event["enrichment_stage"] = "priority_classification"
    return event


def build_fast_lane_hint_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Build a routing hint for requests that should enter the fast lane."""
    event = classify_query_event(data, timestamp=timestamp)
    return {
        "event_type": FAST_LANE_HINT_EVENT,
        "event_version": event.get("event_version", EVENT_SCHEMA_VERSION),
        "emitted_at": _isoformat(timestamp),
        "request_id": event["request_id"],
        "query_id": event["query_id"],
        "user_id": event.get("user_id"),
        "user_tier": event.get("user_tier"),
        "query_text": event.get("query_text"),
        "priority": event.get("priority"),
        "route_to_fast_lane": event.get("route_to_fast_lane", False),
        "hint_type": "fast_lane_candidate",
        "hint_reason": f"priority={event.get('priority')}",
        "selected_model": event.get("selected_model"),
        "query_type": event.get("query_type"),
    }


def build_alert_event(
    data: Dict[str, Any] | str,
    *,
    alert_type: str,
    description: str,
    severity: str = "warning",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Build an alert payload for malformed or rejected request events."""
    original_payload = data if isinstance(data, dict) else {"raw_message": str(data)}
    normalized_payload = normalize_request_event(original_payload)

    return {
        "event_type": ALERT_EVENT,
        "event_version": normalized_payload.get("event_version", EVENT_SCHEMA_VERSION),
        "emitted_at": _isoformat(timestamp),
        "request_id": normalized_payload.get("request_id"),
        "query_id": normalized_payload.get("query_id"),
        "user_id": normalized_payload.get("user_id"),
        "alert_type": alert_type,
        "severity": severity,
        "description": description,
        "original_event_type": normalized_payload.get("event_type"),
        "original_payload": original_payload,
    }


def build_model_metrics_window_event(
    *,
    model_name: str,
    provider: str,
    window_start_ms: int,
    window_end_ms: int,
    window_size_seconds: int,
    request_count: int,
    success_count: int,
    error_count: int,
    latency_sum_ms: float,
    tokens_per_second_sum: float,
    token_count_input: int,
    token_count_output: int,
    total_tokens: int,
    total_cost_usd: float,
    cached_count: int,
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Build a 1-minute model metrics aggregate event."""
    request_count = max(0, int(request_count))
    success_count = max(0, int(success_count))
    error_count = max(0, int(error_count))

    avg_latency_ms = latency_sum_ms / request_count if request_count > 0 else 0.0
    avg_tokens_per_second = (
        tokens_per_second_sum / request_count if request_count > 0 else 0.0
    )
    success_rate = success_count / request_count if request_count > 0 else 0.0
    error_rate = error_count / request_count if request_count > 0 else 0.0
    cache_hit_rate = cached_count / request_count if request_count > 0 else 0.0
    queries_per_second = request_count / max(1, window_size_seconds)

    return {
        "event_type": MODEL_METRICS_1M_EVENT,
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat(timestamp),
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "window_size_seconds": window_size_seconds,
        "model_name": model_name,
        "provider": provider,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "avg_latency_ms": avg_latency_ms,
        "avg_tokens_per_second": avg_tokens_per_second,
        "queries_per_second": queries_per_second,
        "token_count_input": int(token_count_input),
        "token_count_output": int(token_count_output),
        "total_tokens": int(total_tokens),
        "total_cost_usd": float(total_cost_usd),
        "cache_hit_rate": cache_hit_rate,
        "cached_count": int(cached_count),
    }


def build_metric_anomaly_alert_event(
    metric_event: Dict[str, Any],
    *,
    anomaly_type: str,
    description: str,
    severity: str = "warning",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Build an alert event tied to an aggregated model metrics window."""
    return {
        "event_type": ALERT_EVENT,
        "event_version": metric_event.get("event_version", EVENT_SCHEMA_VERSION),
        "emitted_at": _isoformat(timestamp),
        "request_id": None,
        "query_id": None,
        "user_id": None,
        "alert_type": "anomaly_detected",
        "anomaly_type": anomaly_type,
        "severity": severity,
        "description": description,
        "original_event_type": metric_event.get("event_type"),
        "model_name": metric_event.get("model_name"),
        "provider": metric_event.get("provider"),
        "window_start_ms": metric_event.get("window_start_ms"),
        "window_end_ms": metric_event.get("window_end_ms"),
        "metrics": metric_event,
    }


def detect_metric_anomalies(
    metric_event: Dict[str, Any],
    *,
    latency_history: list[float],
    qps_history: list[float],
    error_rate_history: list[float],
    threshold_multiplier: float = 2.0,
    minimum_history: int = 5,
    timestamp: datetime | None = None,
) -> list[Dict[str, Any]]:
    """Detect simple anomalies from recent model metrics history."""
    alerts: list[Dict[str, Any]] = []

    current_latency = float(metric_event.get("avg_latency_ms", 0.0) or 0.0)
    current_qps = float(metric_event.get("queries_per_second", 0.0) or 0.0)
    current_error_rate = float(metric_event.get("error_rate", 0.0) or 0.0)

    if len(latency_history) >= minimum_history:
        avg_latency = sum(latency_history) / len(latency_history)
        if avg_latency > 0 and current_latency > avg_latency * threshold_multiplier:
            alerts.append(
                build_metric_anomaly_alert_event(
                    metric_event,
                    anomaly_type="high_latency",
                    description=(
                        f"Average latency {current_latency:.2f}ms is "
                        f"{current_latency / avg_latency:.2f}x above baseline"
                    ),
                    timestamp=timestamp,
                )
            )

    if len(qps_history) >= minimum_history:
        avg_qps = sum(qps_history) / len(qps_history)
        if avg_qps > 0 and current_qps > avg_qps * threshold_multiplier:
            alerts.append(
                build_metric_anomaly_alert_event(
                    metric_event,
                    anomaly_type="high_volume",
                    description=(
                        f"Queries per second {current_qps:.2f} is "
                        f"{current_qps / avg_qps:.2f}x above baseline"
                    ),
                    timestamp=timestamp,
                )
            )

    if len(error_rate_history) >= minimum_history:
        avg_error_rate = sum(error_rate_history) / len(error_rate_history)
        comparison_floor = max(avg_error_rate * threshold_multiplier, 0.05)
        if current_error_rate > comparison_floor:
            alerts.append(
                build_metric_anomaly_alert_event(
                    metric_event,
                    anomaly_type="high_error_rate",
                    description=(
                        f"Error rate {current_error_rate:.2%} exceeded "
                        f"baseline threshold {comparison_floor:.2%}"
                    ),
                    severity="error",
                    timestamp=timestamp,
                )
            )

    return alerts


def classify_request_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Alias for the enrichment classifier used by the Flink request path."""
    return classify_query_event(data, timestamp=timestamp)


def build_fast_lane_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Backward-compatible alias for legacy smoke tests."""
    return build_fast_lane_hint_event(data, timestamp=timestamp)
