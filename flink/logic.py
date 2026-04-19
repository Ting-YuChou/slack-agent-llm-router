"""
Pure-Python logic shared by the Flink job and local integration tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict

EVENT_SCHEMA_VERSION = "1.0"
REQUESTS_RAW_EVENT = "requests.raw"
REQUESTS_ENRICHED_EVENT = "requests.enriched"
INFERENCE_COMPLETED_EVENT = "inference.completed"
FAST_LANE_HINT_EVENT = "fast_lane_hints"
ALERT_EVENT = "alerts"
MODEL_METRICS_1M_EVENT = "analytics.model_metrics_1m"
ROUTING_GUARDRAIL_EVENT = "routing.guardrails"
ROUTING_POLICY_STATE_EVENT = "routing.policy_state"
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

LOW_LATENCY_KEYWORDS = [
    "urgent",
    "critical",
    "asap",
    "immediately",
    "fast",
    "quickly",
    "now",
    "outage",
    "incident",
]

CODE_KEYWORDS = [
    "code",
    "python",
    "javascript",
    "typescript",
    "java",
    "golang",
    "rust",
    "function",
    "class",
    "stack trace",
    "traceback",
    "refactor",
    "bug",
]

QUERY_TYPE_RULES = [
    (
        "code_analysis",
        [
            "debug",
            "fix this code",
            "review this code",
            "refactor",
            "stack trace",
            "traceback",
            "optimize this function",
        ],
    ),
    (
        "code_generation",
        [
            "write a",
            "implement",
            "build a",
            "create a function",
            "generate code",
            "script",
            "class ",
            "def ",
            "```",
        ],
    ),
    (
        "translation",
        [
            "translate",
            "translation",
            "convert this to english",
            "convert this to chinese",
        ],
    ),
    (
        "summarization",
        [
            "summarize",
            "summary",
            "tldr",
            "tl;dr",
            "brief overview",
        ],
    ),
    (
        "brainstorming",
        [
            "brainstorm",
            "ideas",
            "suggestions",
            "alternatives",
        ],
    ),
    (
        "planning",
        [
            "plan",
            "roadmap",
            "milestones",
            "timeline",
            "strategy",
        ],
    ),
    (
        "math",
        [
            "calculate",
            "equation",
            "derivative",
            "integral",
            "probability",
        ],
    ),
    (
        "reasoning",
        [
            "why",
            "reason through",
            "step by step",
            "root cause",
            "tradeoff",
            "deduce",
        ],
    ),
    (
        "analysis",
        [
            "analyze",
            "investigate",
            "evaluate",
            "assess",
            "compare",
            "architecture",
            "incident report",
        ],
    ),
]

HIGH_REASONING_QUERY_TYPES = {
    "analysis",
    "reasoning",
    "planning",
    "code_analysis",
    "math",
}

DEFAULT_MODEL_GUARDRAIL_CONFIG = {
    "high_error_rate": 0.30,
    "high_latency_ms": 5000,
    "regression_error_rate_floor": 0.15,
    "regression_latency_floor_ms": 3500,
    "cache_hit_baseline_min": 0.15,
    "cache_hit_drop_ratio": 0.5,
    "cache_hit_min_request_count": 5,
    "throughput_baseline_min_qps": 0.20,
    "throughput_drop_ratio": 0.4,
    "throughput_min_request_count": 3,
    "cost_spike_ratio": 1.75,
}

DEFAULT_PROVIDER_GUARDRAIL_CONFIG = {
    "high_error_rate": 0.20,
    "high_latency_ms": 6000,
    "regression_error_rate_floor": 0.10,
    "regression_latency_floor_ms": 4500,
}

DEFAULT_ROLLING_POLICY_CONFIG = {
    "hot_request_count": 10,
    "warm_request_count": 4,
    "complex_total_tokens": 2400,
    "complex_reasoning_tokens": 1200,
    "moderate_total_tokens": 900,
    "moderate_latency_ms": 2500,
    "moderate_latency_min_tokens": 600,
    "enterprise_priority_request_count": 4,
    "burst_request_count": 10,
    "burst_min_avg_total_tokens": 400,
    "session_pin_min_success_count": 3,
    "user_pin_min_success_count": 4,
    "pin_min_share": 0.60,
    "expensive_model_min_count": 3,
    "expensive_model_min_share": 0.70,
    "free_high_cost_threshold_usd": 0.015,
    "paid_high_cost_threshold_usd": 0.030,
    "enterprise_medium_cost_threshold_usd": 0.050,
    "failing_model_min_count": 2,
    "failing_model_min_share": 0.60,
    "failing_provider_min_count": 2,
    "failing_provider_min_share": 0.70,
    "fast_lane_hit_rate_threshold": 0.60,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merged_config(
    defaults: Dict[str, Any], overrides: Dict[str, Any] | None
) -> Dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, "", ()):
        return []
    return [str(value)]


def _normalize_preference(value: Any, default: str) -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return default


def _isoformat(timestamp: datetime | None = None) -> str:
    """Serialize timestamps consistently for event payloads."""
    return (timestamp or datetime.now()).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None


def event_timestamp_ms(value: Any, *, default_ms: int | None = None) -> int:
    parsed = _parse_datetime(value)
    if parsed is None:
        return default_ms if default_ms is not None else 0
    return int(parsed.timestamp() * 1000)


def completion_event_timestamp_ms(
    event: Dict[str, Any], *, default_ms: int | None = None
) -> int:
    return event_timestamp_ms(
        event.get("completion_timestamp")
        or event.get("emitted_at")
        or event.get("request_timestamp"),
        default_ms=default_ms,
    )


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
    event.setdefault("route_to_fast_lane", False)
    event.setdefault("actual_fast_lane_hit", False)
    event.setdefault("session_id", None)
    event.setdefault("conversation_id", None)
    event.setdefault("policy_source", None)
    event.setdefault("hint_reason", None)
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


def estimate_query_tokens(query_text: str, context: str = "") -> int:
    """Estimate prompt size cheaply for Flink-side routing features."""
    combined = " ".join(part for part in [query_text, context] if part)
    word_count = len(combined.split())
    if word_count == 0:
        return 0
    return max(word_count, int(word_count * 1.3))


def infer_query_type(query_text: str) -> str:
    """Infer a coarse query type without depending on the API router runtime."""
    normalized_query = (query_text or "").strip().lower()
    if not normalized_query:
        return "general"

    for query_type, rules in QUERY_TYPE_RULES:
        if any(rule in normalized_query for rule in rules):
            return query_type

    if re.search(r"\b(def|class|function)\b", normalized_query):
        return "code_generation"

    return "general"


def calculate_query_complexity(
    *,
    query_type: str,
    query_text: str,
    token_estimate: int,
    context_length: int,
    attachments_count: int,
) -> str:
    """Estimate whether the request is simple, moderate, or complex."""
    score = 0.0
    normalized_query = (query_text or "").lower()

    if query_type in HIGH_REASONING_QUERY_TYPES:
        score += 1.0

    if token_estimate >= 1800 or context_length >= 12000:
        score += 2.0
    elif token_estimate >= 700 or context_length >= 3000:
        score += 1.0

    if attachments_count >= 3:
        score += 1.0
    elif attachments_count >= 1:
        score += 0.5

    if any(
        marker in normalized_query
        for marker in [
            "compare",
            "tradeoff",
            "root cause",
            "step by step",
            "design",
            "architecture",
            "incident",
            "investigate",
        ]
    ):
        score += 1.0

    if score >= 3.0:
        return "complex"
    if score >= 1.5:
        return "moderate"
    return "simple"


def derive_session_hotness(metadata: Dict[str, Any]) -> str:
    """Infer whether a session is cold, warm, or hot from recent request volume."""
    recent_query_count = max(
        _safe_int(metadata.get("recent_query_count")),
        _safe_int(metadata.get("session_query_count")),
        _safe_int(metadata.get("session_message_count")),
        _safe_int(metadata.get("recent_request_count")),
    )

    if recent_query_count >= 10:
        return "hot"
    if recent_query_count >= 4:
        return "warm"
    return "cold"


def build_routing_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate request-side routing features from the raw API payload."""
    metadata = data.get("metadata") or {}
    query_text = str(data.get("query_text", "") or "")
    context = str(data.get("context", "") or "")
    attachments = data.get("attachments") or []
    attachments_count = _safe_int(
        data.get(
            "attachments_count",
            len(attachments) if isinstance(attachments, list) else 0,
        )
    )
    attachment_bytes = 0
    if isinstance(attachments, list):
        attachment_bytes = sum(
            _safe_int(attachment.get("size_bytes"))
            for attachment in attachments
            if isinstance(attachment, dict)
        )

    query_type = infer_query_type(query_text)
    token_estimate = estimate_query_tokens(query_text, context)
    long_context = (
        len(context) >= 4000
        or token_estimate >= 2500
        or _safe_int(data.get("max_tokens")) >= 4096
    )
    attachment_heavy = attachments_count >= 3 or attachment_bytes >= 5_000_000
    code_heavy = query_type in {"code_generation", "code_analysis"} or any(
        keyword in query_text.lower() for keyword in CODE_KEYWORDS
    )
    query_complexity = calculate_query_complexity(
        query_type=query_type,
        query_text=query_text,
        token_estimate=token_estimate,
        context_length=len(context),
        attachments_count=attachments_count,
    )
    session_hotness = derive_session_hotness(metadata)

    user_tier = str(data.get("user_tier", "free") or "free").lower()
    default_cost_sensitivity = {
        "free": "high",
        "premium": "medium",
        "enterprise": "low",
    }.get(user_tier, "medium")
    default_error_sensitivity = {
        "free": "medium",
        "premium": "medium",
        "enterprise": "high",
    }.get(user_tier, "medium")

    requires_high_reasoning = bool(metadata.get("requires_high_reasoning")) or (
        query_type in HIGH_REASONING_QUERY_TYPES or query_complexity == "complex"
    )
    requires_low_latency = bool(metadata.get("requires_low_latency")) or (
        any(keyword in query_text.lower() for keyword in LOW_LATENCY_KEYWORDS)
        or (_safe_int(data.get("priority"), 1) >= 4 and query_complexity != "complex")
        or (session_hotness == "hot" and not requires_high_reasoning)
    )

    return {
        "query_type": query_type,
        "query_complexity": query_complexity,
        "query_token_estimate": token_estimate,
        "long_context": long_context,
        "attachment_heavy": attachment_heavy,
        "code_heavy": code_heavy,
        "session_hotness": session_hotness,
        "cost_sensitivity": _normalize_preference(
            metadata.get("cost_sensitivity"), default_cost_sensitivity
        ),
        "error_sensitivity": _normalize_preference(
            metadata.get("error_sensitivity"), default_error_sensitivity
        ),
        "requires_low_latency": requires_low_latency,
        "requires_high_reasoning": requires_high_reasoning,
        "preferred_models": _coerce_list(metadata.get("preferred_models")),
        "avoid_models": _coerce_list(metadata.get("avoid_models")),
        "avoid_providers": [
            provider.lower()
            for provider in _coerce_list(metadata.get("avoid_providers"))
        ],
    }


def classify_query_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Apply Flink query-priority enrichment to an event payload."""
    event = validate_request_event(data)
    routing_features = build_routing_features(event)
    priority = calculate_priority(
        query_text=event.get("query_text", ""),
        user_tier=event.get("user_tier", "free"),
        user_id=event.get("user_id", ""),
    )
    route_to_fast_lane = priority in {"high", "critical"} or (
        routing_features["requires_low_latency"]
        and not routing_features["requires_high_reasoning"]
        and not routing_features["long_context"]
    )

    event.update(routing_features)
    event["priority"] = priority
    event["processing_timestamp"] = _isoformat(timestamp)
    event["route_to_fast_lane"] = route_to_fast_lane
    return event


def build_request_enriched_event(
    data: Dict[str, Any], timestamp: datetime | None = None
) -> Dict[str, Any]:
    """Build the enrichment output payload produced by Flink."""
    event = classify_query_event(data, timestamp=timestamp)
    event["event_type"] = REQUESTS_ENRICHED_EVENT
    event["emitted_at"] = _isoformat(timestamp)
    event["enrichment_stage"] = "routing_feature_generation"
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
        "query_type": event.get("query_type"),
        "query_complexity": event.get("query_complexity"),
        "priority": event.get("priority"),
        "route_to_fast_lane": event.get("route_to_fast_lane", False),
        "requires_low_latency": event.get("requires_low_latency", False),
        "requires_high_reasoning": event.get("requires_high_reasoning", False),
        "long_context": event.get("long_context", False),
        "attachment_heavy": event.get("attachment_heavy", False),
        "code_heavy": event.get("code_heavy", False),
        "session_hotness": event.get("session_hotness"),
        "cost_sensitivity": event.get("cost_sensitivity"),
        "error_sensitivity": event.get("error_sensitivity"),
        "preferred_models": list(event.get("preferred_models", []) or []),
        "avoid_models": list(event.get("avoid_models", []) or []),
        "avoid_providers": list(event.get("avoid_providers", []) or []),
        "hint_type": "fast_lane_candidate",
        "hint_reason": f"priority={event.get('priority')}",
        "selected_model": event.get("selected_model"),
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


def _cost_per_1k_tokens(metric_event: Dict[str, Any]) -> float:
    total_tokens = max(0, int(metric_event.get("total_tokens", 0) or 0))
    total_cost = float(metric_event.get("total_cost_usd", 0.0) or 0.0)
    if total_tokens <= 0:
        return 0.0
    return total_cost * 1000.0 / total_tokens


def build_routing_guardrail_event(
    metric_event: Dict[str, Any],
    *,
    scope_type: str,
    scope_key: str,
    trigger_type: str,
    description: str,
    guardrail_action: str = "avoid",
    severity: str = "warning",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Build a routing guardrail event for the policy cache path."""
    return {
        "event_type": ROUTING_GUARDRAIL_EVENT,
        "event_version": metric_event.get("event_version", EVENT_SCHEMA_VERSION),
        "emitted_at": _isoformat(timestamp),
        "scope_type": scope_type,
        "scope_key": scope_key,
        "trigger_type": trigger_type,
        "guardrail_action": guardrail_action,
        "severity": severity,
        "description": description,
        "model_name": metric_event.get("model_name"),
        "provider": metric_event.get("provider"),
        "window_start_ms": metric_event.get("window_start_ms"),
        "window_end_ms": metric_event.get("window_end_ms"),
        "metric_snapshot": metric_event,
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


def detect_model_routing_guardrails(
    metric_event: Dict[str, Any],
    *,
    latency_history: list[float],
    qps_history: list[float],
    error_rate_history: list[float],
    cache_hit_rate_history: list[float],
    cost_per_1k_history: list[float],
    threshold_multiplier: float = 2.0,
    minimum_history: int = 3,
    thresholds: Dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> list[Dict[str, Any]]:
    """Detect model-level routing guardrails from recent aggregate history."""
    guardrails: list[Dict[str, Any]] = []
    config = _merged_config(DEFAULT_MODEL_GUARDRAIL_CONFIG, thresholds)

    model_name = str(metric_event.get("model_name", "unknown") or "unknown")
    current_latency = float(metric_event.get("avg_latency_ms", 0.0) or 0.0)
    current_qps = float(metric_event.get("queries_per_second", 0.0) or 0.0)
    current_error_rate = float(metric_event.get("error_rate", 0.0) or 0.0)
    current_cache_hit_rate = float(metric_event.get("cache_hit_rate", 0.0) or 0.0)
    request_count = max(0, int(metric_event.get("request_count", 0) or 0))
    current_cost_per_1k = _cost_per_1k_tokens(metric_event)
    high_error_rate_threshold = _safe_float(config.get("high_error_rate"), 0.30)
    high_latency_ms = _safe_float(config.get("high_latency_ms"), 5000)
    regression_error_floor = _safe_float(
        config.get("regression_error_rate_floor"), 0.15
    )
    regression_latency_floor_ms = _safe_float(
        config.get("regression_latency_floor_ms"), 3500
    )
    cache_hit_baseline_min = _safe_float(config.get("cache_hit_baseline_min"), 0.15)
    cache_hit_drop_ratio = _safe_float(config.get("cache_hit_drop_ratio"), 0.5)
    cache_hit_min_request_count = _safe_int(
        config.get("cache_hit_min_request_count"), 5
    )
    throughput_baseline_min_qps = _safe_float(
        config.get("throughput_baseline_min_qps"), 0.20
    )
    throughput_drop_ratio = _safe_float(config.get("throughput_drop_ratio"), 0.4)
    throughput_min_request_count = _safe_int(
        config.get("throughput_min_request_count"), 3
    )
    cost_spike_ratio = _safe_float(config.get("cost_spike_ratio"), 1.75)

    if current_error_rate >= high_error_rate_threshold:
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="model",
                scope_key=model_name,
                trigger_type="high_error_rate",
                description=(
                    f"Model error rate {current_error_rate:.2%} exceeded the routing guardrail threshold"
                ),
                severity="error",
                timestamp=timestamp,
            )
        )

    if current_latency >= high_latency_ms:
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="model",
                scope_key=model_name,
                trigger_type="high_latency",
                description=(
                    f"Model latency {current_latency:.0f}ms exceeded the routing guardrail threshold"
                ),
                severity="error",
                timestamp=timestamp,
            )
        )

    if len(error_rate_history) >= minimum_history:
        avg_error_rate = sum(error_rate_history) / len(error_rate_history)
        if avg_error_rate >= 0 and current_error_rate > max(
            regression_error_floor, avg_error_rate * threshold_multiplier
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="model",
                    scope_key=model_name,
                    trigger_type="error_rate_regression",
                    description=(
                        f"Model error rate {current_error_rate:.2%} regressed relative to the recent baseline"
                    ),
                    severity="error",
                    timestamp=timestamp,
                )
            )

    if len(latency_history) >= minimum_history:
        avg_latency = sum(latency_history) / len(latency_history)
        if avg_latency > 0 and current_latency > max(
            regression_latency_floor_ms, avg_latency * threshold_multiplier
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="model",
                    scope_key=model_name,
                    trigger_type="latency_regression",
                    description=(
                        f"Model latency {current_latency:.0f}ms regressed relative to the recent baseline"
                    ),
                    severity="error",
                    timestamp=timestamp,
                )
            )

    if len(cache_hit_rate_history) >= minimum_history:
        avg_cache_hit_rate = sum(cache_hit_rate_history) / len(cache_hit_rate_history)
        if (
            avg_cache_hit_rate > cache_hit_baseline_min
            and request_count >= cache_hit_min_request_count
            and current_cache_hit_rate < avg_cache_hit_rate * cache_hit_drop_ratio
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="model",
                    scope_key=model_name,
                    trigger_type="cache_hit_drop",
                    description=(
                        f"Model cache hit rate dropped to {current_cache_hit_rate:.2%} from a recent baseline near {avg_cache_hit_rate:.2%}"
                    ),
                    guardrail_action="warn",
                    severity="warning",
                    timestamp=timestamp,
                )
            )

    if len(qps_history) >= minimum_history:
        avg_qps = sum(qps_history) / len(qps_history)
        if (
            avg_qps > throughput_baseline_min_qps
            and request_count >= throughput_min_request_count
            and current_qps < avg_qps * throughput_drop_ratio
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="model",
                    scope_key=model_name,
                    trigger_type="throughput_drop",
                    description=(
                        f"Model throughput dropped to {current_qps:.2f} qps from a recent baseline near {avg_qps:.2f} qps"
                    ),
                    guardrail_action="warn",
                    severity="warning",
                    timestamp=timestamp,
                )
            )

    if len(cost_per_1k_history) >= minimum_history:
        avg_cost_per_1k = sum(cost_per_1k_history) / len(cost_per_1k_history)
        if (
            avg_cost_per_1k > 0
            and current_cost_per_1k > avg_cost_per_1k * cost_spike_ratio
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="model",
                    scope_key=model_name,
                    trigger_type="cost_spike",
                    description=(
                        f"Model cost rose to ${current_cost_per_1k:.4f} per 1k tokens from a recent baseline near ${avg_cost_per_1k:.4f}"
                    ),
                    guardrail_action="warn",
                    severity="warning",
                    timestamp=timestamp,
                )
            )

    return guardrails


def detect_provider_routing_guardrails(
    metric_event: Dict[str, Any],
    *,
    latency_history: list[float],
    error_rate_history: list[float],
    threshold_multiplier: float = 2.0,
    minimum_history: int = 3,
    thresholds: Dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> list[Dict[str, Any]]:
    """Detect provider-level routing guardrails from recent aggregate history."""
    guardrails: list[Dict[str, Any]] = []
    config = _merged_config(DEFAULT_PROVIDER_GUARDRAIL_CONFIG, thresholds)

    provider = str(metric_event.get("provider", "unknown") or "unknown")
    current_latency = float(metric_event.get("avg_latency_ms", 0.0) or 0.0)
    current_error_rate = float(metric_event.get("error_rate", 0.0) or 0.0)
    high_error_rate_threshold = _safe_float(config.get("high_error_rate"), 0.20)
    high_latency_ms = _safe_float(config.get("high_latency_ms"), 6000)
    regression_error_floor = _safe_float(
        config.get("regression_error_rate_floor"), 0.10
    )
    regression_latency_floor_ms = _safe_float(
        config.get("regression_latency_floor_ms"), 4500
    )

    if current_error_rate >= high_error_rate_threshold:
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="provider",
                scope_key=provider,
                trigger_type="provider_high_error_rate",
                description=(
                    f"Provider error rate {current_error_rate:.2%} exceeded the routing guardrail threshold"
                ),
                severity="error",
                timestamp=timestamp,
            )
        )

    if current_latency >= high_latency_ms:
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="provider",
                scope_key=provider,
                trigger_type="provider_high_latency",
                description=(
                    f"Provider latency {current_latency:.0f}ms exceeded the routing guardrail threshold"
                ),
                severity="error",
                timestamp=timestamp,
            )
        )

    if len(error_rate_history) >= minimum_history:
        avg_error_rate = sum(error_rate_history) / len(error_rate_history)
        if avg_error_rate > 0 and current_error_rate > max(
            regression_error_floor, avg_error_rate * threshold_multiplier
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="provider",
                    scope_key=provider,
                    trigger_type="provider_error_regression",
                    description=(
                        f"Provider error rate {current_error_rate:.2%} regressed relative to the recent baseline"
                    ),
                    severity="error",
                    timestamp=timestamp,
                )
            )

    if len(latency_history) >= minimum_history:
        avg_latency = sum(latency_history) / len(latency_history)
        if avg_latency > 0 and current_latency > max(
            regression_latency_floor_ms, avg_latency * threshold_multiplier
        ):
            guardrails.append(
                build_routing_guardrail_event(
                    metric_event,
                    scope_type="provider",
                    scope_key=provider,
                    trigger_type="provider_latency_regression",
                    description=(
                        f"Provider latency {current_latency:.0f}ms regressed relative to the recent baseline"
                    ),
                    severity="error",
                    timestamp=timestamp,
                )
            )

    return guardrails


def _derive_hotness_from_request_count(
    request_count: int, *, hot_request_count: int = 10, warm_request_count: int = 4
) -> str:
    if request_count >= hot_request_count:
        return "hot"
    if request_count >= warm_request_count:
        return "warm"
    return "cold"


def _derive_complexity_from_rolling_state(
    *,
    dominant_query_type: str,
    avg_total_tokens: float,
    avg_latency_ms: float,
    policy_config: Dict[str, Any] | None = None,
) -> str:
    config = _merged_config(DEFAULT_ROLLING_POLICY_CONFIG, policy_config)
    complex_total_tokens = _safe_float(config.get("complex_total_tokens"), 2400)
    complex_reasoning_tokens = _safe_float(config.get("complex_reasoning_tokens"), 1200)
    moderate_total_tokens = _safe_float(config.get("moderate_total_tokens"), 900)
    moderate_latency_ms = _safe_float(config.get("moderate_latency_ms"), 2500)
    moderate_latency_min_tokens = _safe_float(
        config.get("moderate_latency_min_tokens"), 600
    )

    if avg_total_tokens >= complex_total_tokens or (
        dominant_query_type in HIGH_REASONING_QUERY_TYPES
        and avg_total_tokens >= complex_reasoning_tokens
    ):
        return "complex"
    if (
        avg_total_tokens >= moderate_total_tokens
        or dominant_query_type in HIGH_REASONING_QUERY_TYPES
    ):
        return "moderate"
    if (
        avg_latency_ms >= moderate_latency_ms
        and avg_total_tokens >= moderate_latency_min_tokens
    ):
        return "moderate"
    return "simple"


def _dominant_key(
    counts: Dict[str, int],
    *,
    minimum_count: int = 1,
    minimum_share: float = 0.0,
) -> str | None:
    if not counts:
        return None

    total = sum(max(0, count) for count in counts.values())
    if total <= 0:
        return None

    key, count = max(counts.items(), key=lambda item: item[1])
    if count < minimum_count:
        return None
    if (count / total) < minimum_share:
        return None
    return key


def build_routing_policy_state_event(
    *,
    scope_type: str,
    scope_key: str,
    events: list[Dict[str, Any]],
    window_size_seconds: int = 300,
    policy_config: Dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Derive rolling user/session routing policy state from recent completions."""
    if scope_type not in {"user", "session"}:
        raise ValueError(f"Unsupported routing policy scope_type: {scope_type}")
    if not scope_key:
        raise ValueError("scope_key is required for routing policy state")
    if not events:
        raise ValueError("events is required to build routing policy state")

    request_count = len(events)
    success_count = sum(1 for event in events if event.get("status") == "success")
    error_count = request_count - success_count
    recent_error_rate = error_count / request_count if request_count > 0 else 0.0
    avg_total_tokens = (
        sum(_safe_int(event.get("total_tokens")) for event in events) / request_count
    )
    avg_cost_usd = (
        sum(_safe_float(event.get("cost_usd")) for event in events) / request_count
    )
    avg_latency_ms = (
        sum(_safe_float(event.get("latency_ms")) for event in events) / request_count
    )
    fast_lane_hits = sum(
        1 for event in events if bool(event.get("actual_fast_lane_hit", False))
    )
    fast_lane_hit_rate = fast_lane_hits / request_count if request_count > 0 else 0.0

    query_type_counts: Dict[str, int] = {}
    model_success_counts: Dict[str, int] = {}
    model_all_counts: Dict[str, int] = {}
    failed_model_counts: Dict[str, int] = {}
    failed_provider_counts: Dict[str, int] = {}

    latest_event = events[-1]
    user_tier = str(latest_event.get("user_tier", "free") or "free").lower()
    user_id = latest_event.get("user_id")
    session_id = latest_event.get("session_id")
    config = _merged_config(DEFAULT_ROLLING_POLICY_CONFIG, policy_config)

    for event in events:
        query_type = str(event.get("query_type", "general") or "general").lower()
        query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1

        selected_model = str(event.get("selected_model", "") or "")
        if selected_model:
            model_all_counts[selected_model] = (
                model_all_counts.get(selected_model, 0) + 1
            )
            if event.get("status") == "success":
                model_success_counts[selected_model] = (
                    model_success_counts.get(selected_model, 0) + 1
                )
            else:
                failed_model_counts[selected_model] = (
                    failed_model_counts.get(selected_model, 0) + 1
                )

        provider = str(event.get("provider", "") or "").lower()
        if provider and event.get("status") != "success":
            failed_provider_counts[provider] = (
                failed_provider_counts.get(provider, 0) + 1
            )

    dominant_query_type = _dominant_key(query_type_counts) or "general"
    session_hotness = _derive_hotness_from_request_count(
        request_count,
        hot_request_count=_safe_int(config.get("hot_request_count"), 10),
        warm_request_count=_safe_int(config.get("warm_request_count"), 4),
    )
    query_complexity = _derive_complexity_from_rolling_state(
        dominant_query_type=dominant_query_type,
        avg_total_tokens=avg_total_tokens,
        avg_latency_ms=avg_latency_ms,
        policy_config=config,
    )
    requires_high_reasoning = (
        dominant_query_type in HIGH_REASONING_QUERY_TYPES
        or query_complexity == "complex"
    )

    enterprise_priority_active = (
        user_tier == "enterprise"
        and request_count
        >= _safe_int(config.get("enterprise_priority_request_count"), 4)
    )
    burst_protection_active = (
        user_tier != "enterprise"
        and request_count >= _safe_int(config.get("burst_request_count"), 10)
        and avg_total_tokens
        >= _safe_float(config.get("burst_min_avg_total_tokens"), 400)
    )

    preferred_models: list[str] = []
    avoid_models: list[str] = []
    avoid_providers: list[str] = []

    pinned_model = _dominant_key(
        model_success_counts,
        minimum_count=(
            _safe_int(config.get("session_pin_min_success_count"), 3)
            if scope_type == "session"
            else _safe_int(config.get("user_pin_min_success_count"), 4)
        ),
        minimum_share=_safe_float(config.get("pin_min_share"), 0.60),
    )
    if pinned_model and recent_error_rate <= 0.15:
        if scope_type == "session":
            preferred_models.append(pinned_model)
        elif enterprise_priority_active:
            preferred_models.append(pinned_model)

    expensive_model = _dominant_key(
        model_all_counts,
        minimum_count=_safe_int(config.get("expensive_model_min_count"), 3),
        minimum_share=_safe_float(config.get("expensive_model_min_share"), 0.70),
    )
    high_cost_threshold = (
        _safe_float(config.get("free_high_cost_threshold_usd"), 0.015)
        if user_tier == "free"
        else _safe_float(config.get("paid_high_cost_threshold_usd"), 0.030)
    )
    if (
        scope_type == "user"
        and user_tier != "enterprise"
        and expensive_model
        and avg_cost_usd >= high_cost_threshold
        and burst_protection_active
        and expensive_model not in preferred_models
    ):
        avoid_models.append(expensive_model)

    failing_model = _dominant_key(
        failed_model_counts,
        minimum_count=_safe_int(config.get("failing_model_min_count"), 2),
        minimum_share=_safe_float(config.get("failing_model_min_share"), 0.60),
    )
    if (
        failing_model
        and failing_model not in preferred_models
        and failing_model not in avoid_models
    ):
        avoid_models.append(failing_model)

    failing_provider = _dominant_key(
        failed_provider_counts,
        minimum_count=_safe_int(config.get("failing_provider_min_count"), 2),
        minimum_share=_safe_float(config.get("failing_provider_min_share"), 0.70),
    )
    if failing_provider:
        avoid_providers.append(failing_provider)

    cost_sensitivity = {
        "free": "high",
        "premium": "medium",
        "enterprise": "low",
    }.get(user_tier, "medium")
    if user_tier != "enterprise" and (
        burst_protection_active or avg_cost_usd >= high_cost_threshold
    ):
        cost_sensitivity = "high"
    elif user_tier == "enterprise" and avg_cost_usd >= _safe_float(
        config.get("enterprise_medium_cost_threshold_usd"), 0.050
    ):
        cost_sensitivity = "medium"

    error_sensitivity = (
        "high" if (user_tier == "enterprise" or recent_error_rate >= 0.20) else "medium"
    )
    requires_low_latency = (
        burst_protection_active
        or fast_lane_hit_rate
        >= _safe_float(config.get("fast_lane_hit_rate_threshold"), 0.60)
        or (session_hotness == "hot" and not requires_high_reasoning)
    )
    route_to_fast_lane = (
        requires_low_latency
        and not requires_high_reasoning
        and query_complexity != "complex"
    )

    hint_reasons: list[str] = []
    if burst_protection_active:
        hint_reasons.append("burst_protection")
    if preferred_models:
        hint_reasons.append(
            "session_model_pinning"
            if scope_type == "session"
            else "stable_model_preference"
        )
    if avoid_models or avoid_providers:
        hint_reasons.append("recent_failure_avoidance")
    if enterprise_priority_active:
        hint_reasons.append("enterprise_priority")
    if route_to_fast_lane and fast_lane_hit_rate > 0:
        hint_reasons.append("fast_lane_affinity")

    query_type_breakdown = {
        query_type: round(count / request_count, 4)
        for query_type, count in sorted(
            query_type_counts.items(), key=lambda item: (-item[1], item[0])
        )
    }

    return {
        "event_type": ROUTING_POLICY_STATE_EVENT,
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat(timestamp),
        "scope_type": scope_type,
        "scope_key": scope_key,
        "user_id": user_id,
        "session_id": session_id,
        "user_tier": user_tier,
        "window_size_seconds": window_size_seconds,
        "recent_request_count": request_count,
        "recent_success_count": success_count,
        "recent_error_count": error_count,
        "recent_error_rate": round(recent_error_rate, 4),
        "avg_total_tokens": round(avg_total_tokens, 2),
        "avg_cost_usd": round(avg_cost_usd, 6),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "dominant_query_type": dominant_query_type,
        "query_type": dominant_query_type,
        "query_type_breakdown": query_type_breakdown,
        "fast_lane_hit_rate": round(fast_lane_hit_rate, 4),
        "session_hotness": session_hotness,
        "query_complexity": query_complexity,
        "requires_low_latency": requires_low_latency,
        "requires_high_reasoning": requires_high_reasoning,
        "route_to_fast_lane": route_to_fast_lane,
        "cost_sensitivity": cost_sensitivity,
        "error_sensitivity": error_sensitivity,
        "burst_protection_active": burst_protection_active,
        "enterprise_priority_active": enterprise_priority_active,
        "preferred_models": preferred_models,
        "avoid_models": avoid_models,
        "avoid_providers": avoid_providers,
        "hint_reason": ",".join(hint_reasons)
        if hint_reasons
        else "rolling_policy_state",
    }


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
