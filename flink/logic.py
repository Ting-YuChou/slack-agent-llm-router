"""
Pure-Python logic shared by the Flink job and local integration tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.event_version import (
    EVENT_SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
    is_supported_event_version,
)
from src.redaction import context_summary
from src.routing_features import (
    CODE_KEYWORDS as CODE_KEYWORDS,
    HIGH_REASONING_QUERY_TYPES,
    QUERY_TYPE_RULES as QUERY_TYPE_RULES,
    build_routing_features,
    calculate_priority,
    calculate_query_complexity as calculate_query_complexity,
    derive_session_hotness as derive_session_hotness,
    estimate_query_tokens as estimate_query_tokens,
    infer_query_type as infer_query_type,
)

REQUESTS_RAW_EVENT = "requests.raw"
REQUESTS_ENRICHED_EVENT = "requests.enriched"
INFERENCE_COMPLETED_EVENT = "inference.completed"
FAST_LANE_HINT_EVENT = "fast_lane_hints"
ALERT_EVENT = "alerts"
MODEL_METRICS_1M_EVENT = "analytics.model_metrics_1m"
ROUTING_GUARDRAIL_EVENT = "routing.guardrails"
ROUTING_POLICY_STATE_EVENT = "routing.policy_state"
# Mapper output for an input the job drops; no filter or sink selects it.
DROPPED_EVENT = "dropped"
UNSUPPORTED_EVENT_VERSION_COUNTER = "unsupported_event_version_dropped"
REQUIRED_REQUEST_FIELDS = ("request_id", "query_text", "user_id", "user_tier")
REQUIRED_COMPLETION_FIELDS = (
    "request_id",
    "selected_model",
    "provider",
    "status",
    "latency_ms",
)

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

# Shared by model and provider guardrails (A2). An avoid trigger needs
# min_request_count samples in its window; a high latency avoids only when
# the time per output token is also high (a long answer is not a slow
# model), and only warns otherwise. The detectors block after
# enter_bad_windows bad windows among the last enter_window_lookback, and
# emit guardrail_action "clear" after exit_good_windows good ones in a row.
DEFAULT_GUARDRAIL_GATING_CONFIG = {
    "min_request_count": 5,
    "high_latency_ms_per_output_token": 100,
    "enter_bad_windows": 2,
    "enter_window_lookback": 3,
    "exit_good_windows": 2,
    # Matches policy_cache.guardrail_ttl_seconds: an entered scope with no
    # bad window for this long has lapsed in the cache and must enter again.
    "active_ttl_seconds": 180,
}
DEFAULT_MODEL_GUARDRAIL_CONFIG.update(DEFAULT_GUARDRAIL_GATING_CONFIG)

DEFAULT_PROVIDER_GUARDRAIL_CONFIG = {
    "high_error_rate": 0.20,
    "high_latency_ms": 6000,
    "regression_error_rate_floor": 0.10,
    "regression_latency_floor_ms": 4500,
    **DEFAULT_GUARDRAIL_GATING_CONFIG,
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


# Event timestamps before 2000-01-01 are sequence numbers, not wall clock.
_MIN_WALL_CLOCK_MS = 946_684_800_000


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


def _isoformat(timestamp: datetime | None = None) -> str:
    """Serialize timestamps consistently for event payloads; naive values are UTC."""
    timestamp = timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.isoformat()


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


def unsupported_event_version_drop(data: Any) -> Optional[Dict[str, Any]]:
    """The dropped-event marker for an event with an unsupported version (A12b).

    Returns None for events the jobs can read: a missing version (legacy) or
    one with the supported major. The marker carries no input fields.
    """
    if isinstance(data, dict) and not is_supported_event_version(
        data.get("event_version")
    ):
        return {"event_type": DROPPED_EVENT}
    return None


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


def serialize_event(event: Dict[str, Any]) -> str:
    """JSON for the Kafka sinks, with non-ASCII text left as UTF-8."""
    text = json.dumps(event, ensure_ascii=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate has no UTF-8 form; escaped JSON still round-trips it.
        return json.dumps(event)
    return text


def _drop_context_text(event: Dict[str, Any]) -> None:
    """Keep a legacy event's `context` text out of everything Flink emits."""
    if "context" not in event:
        return
    context = event.pop("context")
    summary = context_summary(None if context is None else str(context))
    for key, value in summary.items():
        event.setdefault(key, value)


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
        priority=event.get("priority"),
    )
    route_to_fast_lane = bool(routing_features.get("route_to_fast_lane", False))

    event.update(routing_features)
    _drop_context_text(event)
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
        "latency_sla": event.get("latency_sla"),
        "routing_intent_source": event.get("routing_intent_source"),
        "preferred_models": list(event.get("preferred_models", []) or []),
        "avoid_models": list(event.get("avoid_models", []) or []),
        "avoid_providers": list(event.get("avoid_providers", []) or []),
        "hint_type": "fast_lane_candidate",
        "hint_reason": event.get("routing_intent_source")
        or f"priority={event.get('priority')}",
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
    """Build an alert for a malformed or rejected request event.

    The alert names the request and the error; it never embeds the payload.
    """
    original_payload = data if isinstance(data, dict) else {}
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
    latency_sample_count: int | None = None,
    provider_failure_count: int | None = None,
    uncached_total_tokens: int | None = None,
    uncached_cost_usd: float | None = None,
    provider_request_count: int | None = None,
    provider_latency_sum_ms: float | None = None,
    provider_latency_sample_count: int | None = None,
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Build a 1-minute model metrics aggregate event.

    latency_sample_count is the number of uncached requests summed into
    latency_sum_ms and tokens_per_second_sum (a cache hit's latency is the
    lookup and its throughput is the original call's); it defaults to
    request_count. uncached_total_tokens and uncached_cost_usd are those
    requests' tokens and cost, for pricing per 1k tokens; they default to the
    window totals.

    Provider health reads its own figures. provider_request_count is the
    uncached requests that say something about the provider (cache hits and
    user_or_policy or internal failures are left out; see
    counts_toward_provider_health), and provider_failure_count the failures among
    them. provider_latency_sum_ms and provider_latency_sample_count cover
    uncached successes only, since a failure's latency spans RAG, web search and
    every fallback. They default to request_count, error_count and the model
    latency figures.
    """
    request_count = max(0, int(request_count))
    success_count = max(0, int(success_count))
    error_count = max(0, int(error_count))
    provider_failures = (
        error_count
        if provider_failure_count is None
        else max(0, int(provider_failure_count))
    )
    latency_samples = (
        request_count
        if latency_sample_count is None
        else max(0, int(latency_sample_count))
    )
    provider_requests = (
        request_count
        if provider_request_count is None
        else max(0, int(provider_request_count))
    )
    provider_latency_samples = (
        latency_samples
        if provider_latency_sample_count is None
        else max(0, int(provider_latency_sample_count))
    )
    provider_latency_sum = (
        latency_sum_ms if provider_latency_sum_ms is None else provider_latency_sum_ms
    )

    avg_latency_ms = latency_sum_ms / latency_samples if latency_samples > 0 else 0.0
    avg_tokens_per_second = (
        tokens_per_second_sum / latency_samples if latency_samples > 0 else 0.0
    )
    success_rate = success_count / request_count if request_count > 0 else 0.0
    error_rate = error_count / request_count if request_count > 0 else 0.0
    provider_failure_rate = (
        provider_failures / provider_requests if provider_requests > 0 else 0.0
    )
    provider_avg_latency_ms = (
        provider_latency_sum / provider_latency_samples
        if provider_latency_samples > 0
        else 0.0
    )
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
        "provider_request_count": provider_requests,
        "provider_failure_count": provider_failures,
        "provider_failure_rate": provider_failure_rate,
        "provider_latency_sample_count": provider_latency_samples,
        "provider_avg_latency_ms": provider_avg_latency_ms,
        "latency_sample_count": latency_samples,
        "avg_latency_ms": avg_latency_ms,
        "avg_tokens_per_second": avg_tokens_per_second,
        "queries_per_second": queries_per_second,
        "token_count_input": int(token_count_input),
        "token_count_output": int(token_count_output),
        "total_tokens": int(total_tokens),
        "total_cost_usd": float(total_cost_usd),
        "uncached_total_tokens": int(
            total_tokens if uncached_total_tokens is None else uncached_total_tokens
        ),
        "uncached_cost_usd": float(
            total_cost_usd if uncached_cost_usd is None else uncached_cost_usd
        ),
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


def metric_cost_per_1k_tokens(metric_event: Dict[str, Any]) -> float:
    """Cost per 1k tokens of a metrics window's uncached requests.

    A cache hit keeps its tokens but costs nothing, so pricing it in would make
    the model look cheaper than it is. Windows from before the uncached totals
    existed price every request.
    """
    tokens = metric_event.get("uncached_total_tokens")
    cost = metric_event.get("uncached_cost_usd")
    if tokens is None or cost is None:
        tokens = metric_event.get("total_tokens")
        cost = metric_event.get("total_cost_usd")
    tokens = max(0, _safe_int(tokens or 0))
    if tokens <= 0:
        return 0.0
    return _safe_float(cost or 0.0) * 1000.0 / tokens


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


def metric_has_uncached_sample(metric_event: Dict[str, Any]) -> bool:
    """Return whether a metrics window has any uncached request to measure.

    A window of only cache hits has no model latency and no priced tokens, so
    latency and cost detectors skip it rather than read its zeros as a
    baseline. Windows from before latency_sample_count existed are treated as
    measured.
    """
    samples = metric_event.get("latency_sample_count")
    return samples is None or _safe_int(samples, 0) > 0


def metric_has_priced_tokens(metric_event: Dict[str, Any]) -> bool:
    """Return whether a metrics window has uncached tokens to price per 1k.

    Cache hits cost nothing and failures usually bill no tokens, so a window of
    only those has no cost per 1k; cost detectors skip it rather than read $0
    as a baseline. Windows from before uncached_total_tokens existed count.
    """
    if not metric_has_uncached_sample(metric_event):
        return False
    tokens = metric_event.get("uncached_total_tokens")
    return tokens is None or _safe_int(tokens, 0) > 0


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

    if metric_has_uncached_sample(metric_event) and (
        len(latency_history) >= minimum_history
    ):
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


def metric_latency_sample_count(metric_event: Dict[str, Any]) -> int:
    """Uncached requests behind a window's latency; older windows: all requests."""
    samples = metric_event.get("latency_sample_count")
    if samples is None:
        samples = metric_event.get("request_count")
    return max(0, _safe_int(samples, 0))


def metric_provider_request_count(metric_event: Dict[str, Any]) -> int:
    """Requests behind provider_failure_rate; older windows: all requests."""
    requests = metric_event.get("provider_request_count")
    if requests is None:
        requests = metric_event.get("request_count")
    return max(0, _safe_int(requests, 0))


def metric_provider_latency_sample_count(metric_event: Dict[str, Any]) -> int:
    """Uncached successes behind provider_avg_latency_ms.

    Older windows fall back to the model latency samples.
    """
    samples = metric_event.get("provider_latency_sample_count")
    if samples is None:
        return metric_latency_sample_count(metric_event)
    return max(0, _safe_int(samples, 0))


def metric_ms_per_output_token(metric_event: Dict[str, Any]) -> float | None:
    """Milliseconds per output token from the window's average throughput."""
    tokens_per_second = _safe_float(metric_event.get("avg_tokens_per_second"), 0.0)
    if tokens_per_second <= 0:
        return None
    return 1000.0 / tokens_per_second


def _high_latency_action(metric_event: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Avoid only when each output token is slow too; otherwise warn."""
    threshold = _safe_float(config.get("high_latency_ms_per_output_token"), 100)
    ms_per_token = metric_ms_per_output_token(metric_event)
    if ms_per_token is not None and ms_per_token >= threshold:
        return "avoid"
    return "warn"


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
    has_uncached = metric_has_uncached_sample(metric_event)
    current_latency = float(metric_event.get("avg_latency_ms", 0.0) or 0.0)
    current_qps = float(metric_event.get("queries_per_second", 0.0) or 0.0)
    # Caller, policy and internal failures are not the model's (A2).
    current_error_rate = metric_provider_failure_rate(metric_event)
    min_request_count = max(1, _safe_int(config.get("min_request_count"), 5))
    enough_requests = metric_provider_request_count(metric_event) >= min_request_count
    enough_latency_samples = (
        metric_latency_sample_count(metric_event) >= min_request_count
    )
    current_cache_hit_rate = float(metric_event.get("cache_hit_rate", 0.0) or 0.0)
    request_count = max(0, int(metric_event.get("request_count", 0) or 0))
    current_cost_per_1k = metric_cost_per_1k_tokens(metric_event)
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

    if enough_requests and current_error_rate >= high_error_rate_threshold:
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

    if has_uncached and enough_latency_samples and current_latency >= high_latency_ms:
        action = _high_latency_action(metric_event, config)
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="model",
                scope_key=model_name,
                trigger_type="high_latency",
                description=(
                    f"Model latency {current_latency:.0f}ms exceeded the routing guardrail threshold"
                ),
                guardrail_action=action,
                severity="error" if action == "avoid" else "warning",
                timestamp=timestamp,
            )
        )

    if enough_requests and len(error_rate_history) >= minimum_history:
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

    if (
        has_uncached
        and enough_latency_samples
        and len(latency_history) >= minimum_history
    ):
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

    if metric_has_priced_tokens(metric_event) and (
        len(cost_per_1k_history) >= minimum_history
    ):
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


def metric_provider_failure_rate(metric_event: Dict[str, Any]) -> float:
    """Failure rate that counts against the provider, from a metrics window.

    Windows emitted before provider_failure_rate existed fall back to error_rate.
    """
    rate = metric_event.get("provider_failure_rate")
    if rate is None:
        rate = metric_event.get("error_rate")
    return _safe_float(rate, 0.0)


def metric_provider_avg_latency_ms(metric_event: Dict[str, Any]) -> float:
    """Provider latency over a window's uncached successes.

    Windows emitted before provider_avg_latency_ms existed fall back to
    avg_latency_ms.
    """
    latency = metric_event.get("provider_avg_latency_ms")
    if latency is None:
        latency = metric_event.get("avg_latency_ms")
    return _safe_float(latency, 0.0)


def metric_has_provider_latency_sample(metric_event: Dict[str, Any]) -> bool:
    """Return whether a window has an uncached success to time the provider by.

    Windows emitted before provider_latency_sample_count existed fall back to
    metric_has_uncached_sample.
    """
    samples = metric_event.get("provider_latency_sample_count")
    if samples is None:
        return metric_has_uncached_sample(metric_event)
    return _safe_int(samples, 0) > 0


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
    min_request_count = max(1, _safe_int(config.get("min_request_count"), 5))
    has_provider_latency = metric_has_provider_latency_sample(metric_event) and (
        metric_provider_latency_sample_count(metric_event) >= min_request_count
    )
    enough_requests = metric_provider_request_count(metric_event) >= min_request_count
    current_latency = metric_provider_avg_latency_ms(metric_event)
    current_error_rate = metric_provider_failure_rate(metric_event)
    high_error_rate_threshold = _safe_float(config.get("high_error_rate"), 0.20)
    high_latency_ms = _safe_float(config.get("high_latency_ms"), 6000)
    regression_error_floor = _safe_float(
        config.get("regression_error_rate_floor"), 0.10
    )
    regression_latency_floor_ms = _safe_float(
        config.get("regression_latency_floor_ms"), 4500
    )

    if enough_requests and current_error_rate >= high_error_rate_threshold:
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

    if has_provider_latency and current_latency >= high_latency_ms:
        action = _high_latency_action(metric_event, config)
        guardrails.append(
            build_routing_guardrail_event(
                metric_event,
                scope_type="provider",
                scope_key=provider,
                trigger_type="provider_high_latency",
                description=(
                    f"Provider latency {current_latency:.0f}ms exceeded the routing guardrail threshold"
                ),
                guardrail_action=action,
                severity="error" if action == "avoid" else "warning",
                timestamp=timestamp,
            )
        )

    if enough_requests and len(error_rate_history) >= minimum_history:
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

    if has_provider_latency and len(latency_history) >= minimum_history:
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


def _gating_config(
    defaults: Dict[str, Any], thresholds: Dict[str, Any] | None
) -> Dict[str, int]:
    config = _merged_config(defaults, thresholds)
    lookback = max(1, _safe_int(config.get("enter_window_lookback"), 3))
    return {
        "enter_bad_windows": min(
            lookback, max(1, _safe_int(config.get("enter_bad_windows"), 2))
        ),
        "enter_window_lookback": lookback,
        "exit_good_windows": max(1, _safe_int(config.get("exit_good_windows"), 2)),
        "active_ttl_ms": max(1, _safe_int(config.get("active_ttl_seconds"), 180))
        * 1000,
    }


def _window_end_ms(metric_event: Dict[str, Any], window_start_ms: int) -> int:
    end_ms = _safe_int(metric_event.get("window_end_ms"), 0)
    if end_ms > window_start_ms:
        return end_ms
    size_seconds = _safe_int(metric_event.get("window_size_seconds"), 0)
    return window_start_ms + max(0, size_seconds) * 1000


def _hysteresis_clear_event(
    metric_event: Dict[str, Any],
    *,
    scope_type: str,
    scope_key: str,
    trigger_type: str,
    description: str,
    timestamp: datetime | None,
) -> Dict[str, Any]:
    return build_routing_guardrail_event(
        metric_event,
        scope_type=scope_type,
        scope_key=scope_key,
        trigger_type=trigger_type,
        description=description,
        guardrail_action="clear",
        severity="info",
        timestamp=timestamp,
    )


def apply_guardrail_hysteresis(
    state: Dict[str, Any],
    guardrails: list[Dict[str, Any]],
    *,
    metric_event: Dict[str, Any],
    scope_type: str,
    scope_key: str,
    thresholds: Dict[str, Any] | None = None,
    defaults: Dict[str, Any] | None = None,
    replace_window: bool = False,
    timestamp: datetime | None = None,
) -> list[Dict[str, Any]]:
    """Gate one scope's guardrails on consecutive windows (A2).

    state is the scope's JSON-safe hysteresis state, updated in place. A
    window is bad when any guardrail in it would avoid. The scope enters
    (emits avoid) once enter_bad_windows of the last enter_window_lookback
    windows are bad; until then an avoid is downgraded to warn. While it is
    in, avoids repeat once per bad window to keep the cache entry alive and
    warns are held back so they cannot overwrite the avoid. After
    exit_good_windows good windows in a row it emits one guardrail_action
    "clear" for the trigger that entered.

    Windows count only when consecutive in event time: a window that does
    not start where the previous one ended starts the count again. An
    entered scope that sees no bad window for active_ttl_seconds (the cache
    TTL of its last avoid) has expired: it emits a clear and must pass the
    enter gate again.

    The same window may be evaluated again as more of it arrives: with
    replace_window the newest verdict replaces the window's, and if that
    turns the window that entered the scope healthy, the avoid is retracted
    with a clear. Otherwise a window stays bad once it was. Earlier windows
    arriving late are ignored.
    """
    gating = _gating_config(defaults or DEFAULT_GUARDRAIL_GATING_CONFIG, thresholds)
    window_start_ms = _safe_int(metric_event.get("window_start_ms"), 0)
    window_end_ms = _window_end_ms(metric_event, window_start_ms)
    avoids = [g for g in guardrails if g.get("guardrail_action") == "avoid"]
    warns = [g for g in guardrails if g.get("guardrail_action") != "avoid"]
    bad = bool(avoids)
    outputs: list[Dict[str, Any]] = []

    windows: list[list[Any]] = []
    for item in state.get("windows", []):
        start = _safe_int(item[0], 0)
        end = _safe_int(item[2], 0) if len(item) > 2 else 0
        windows.append([start, bool(item[1]), end if end > start else start])
    if windows and window_start_ms < windows[-1][0]:
        return []

    if state.get("active") and window_start_ms >= _safe_int(
        state.get("active_until_ms"), 0
    ):
        # The cache entry has lapsed; the scope must enter again.
        outputs.append(
            _hysteresis_clear_event(
                metric_event,
                scope_type=scope_type,
                scope_key=scope_key,
                trigger_type=str(state.get("trigger_type") or "unknown"),
                description=(
                    f"{scope_type.capitalize()} {scope_key} guardrail expired "
                    "without recent windows"
                ),
                timestamp=timestamp,
            )
        )
        state["active"] = False
        state["trigger_type"] = None
        state["activated_window"] = None

    retracted = False
    if windows and windows[-1][0] == window_start_ms:
        was_bad = windows[-1][1]
        windows[-1][1] = bad if replace_window else (was_bad or bad)
        retracted = (
            replace_window
            and was_bad
            and not bad
            and bool(state.get("active"))
            and _safe_int(state.get("activated_window"), -1) == window_start_ms
        )
    else:
        if windows and window_start_ms > windows[-1][2]:
            # A gap in event time: earlier windows are not consecutive.
            windows = []
        windows.append([window_start_ms, bad, window_end_ms])
        state["emitted"] = []
    keep = max(gating["enter_window_lookback"], gating["exit_good_windows"])
    windows = windows[-keep:]
    state["windows"] = windows
    emitted = set(state.get("emitted", []))

    def emit(candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        fresh = []
        for guardrail in candidates:
            marker = (
                f"{guardrail.get('trigger_type')}:{guardrail.get('guardrail_action')}"
            )
            if marker not in emitted:
                emitted.add(marker)
                fresh.append(guardrail)
        return fresh

    def refresh_active() -> None:
        state["active_until_ms"] = window_end_ms + gating["active_ttl_ms"]

    lookback = windows[-gating["enter_window_lookback"] :]
    bad_count = sum(1 for window in lookback if window[1])
    if retracted:
        # The rest of the window that entered the scope arrived healthy.
        outputs.append(
            _hysteresis_clear_event(
                metric_event,
                scope_type=scope_type,
                scope_key=scope_key,
                trigger_type=str(state.get("trigger_type") or "unknown"),
                description=(
                    f"{scope_type.capitalize()} {scope_key} window was healthy "
                    "once complete"
                ),
                timestamp=timestamp,
            )
        )
        state["active"] = False
        state["trigger_type"] = None
        state["activated_window"] = None
        emitted = set()
    elif not state.get("active"):
        if bad and bad_count >= gating["enter_bad_windows"]:
            state["active"] = True
            state["trigger_type"] = avoids[0].get("trigger_type")
            state["activated_window"] = window_start_ms
            refresh_active()
            outputs.extend(emit(avoids))
        else:
            pending = [
                {
                    **guardrail,
                    "guardrail_action": "warn",
                    "severity": "warning",
                    "hysteresis": "pending",
                }
                for guardrail in avoids
            ]
            outputs.extend(emit(warns + pending))
    elif bad:
        refresh_active()
        outputs.extend(emit(avoids))
    else:
        good_streak = 0
        for window in reversed(windows):
            if window[1]:
                break
            good_streak += 1
        if good_streak >= gating["exit_good_windows"]:
            outputs.extend(
                emit(
                    [
                        _hysteresis_clear_event(
                            metric_event,
                            scope_type=scope_type,
                            scope_key=scope_key,
                            trigger_type=str(state.get("trigger_type") or "unknown"),
                            description=(
                                f"{scope_type.capitalize()} {scope_key} recovered "
                                f"after {good_streak} healthy windows"
                            ),
                            timestamp=timestamp,
                        )
                    ]
                )
            )
            state["active"] = False
            state["trigger_type"] = None
            state["activated_window"] = None
    state["emitted"] = sorted(emitted)
    return outputs


_PROVIDER_WINDOW_SUM_FIELDS = (
    "request_count",
    "success_count",
    "error_count",
    "provider_request_count",
    "provider_failure_count",
    "provider_latency_sample_count",
    "provider_latency_sum_ms",
    "latency_sample_count",
    "latency_sum_ms",
    "tokens_per_second_sum",
    "token_count_output",
)


def _provider_window_contribution(metric_event: Dict[str, Any]) -> Dict[str, float]:
    """One (model, provider) window's sums, for adding up per provider."""
    request_count = max(0, _safe_int(metric_event.get("request_count"), 0))
    provider_requests = metric_provider_request_count(metric_event)
    provider_failures = metric_event.get("provider_failure_count")
    if provider_failures is None:
        provider_failures = round(
            metric_provider_failure_rate(metric_event) * provider_requests
        )
    latency_samples = metric_latency_sample_count(metric_event)
    provider_latency_samples = metric_provider_latency_sample_count(metric_event)
    if not metric_has_provider_latency_sample(metric_event):
        provider_latency_samples = 0
    return {
        "request_count": request_count,
        "success_count": max(0, _safe_int(metric_event.get("success_count"), 0)),
        "error_count": max(0, _safe_int(metric_event.get("error_count"), 0)),
        "provider_request_count": provider_requests,
        "provider_failure_count": max(0, _safe_int(provider_failures, 0)),
        "provider_latency_sample_count": provider_latency_samples,
        "provider_latency_sum_ms": metric_provider_avg_latency_ms(metric_event)
        * provider_latency_samples,
        "latency_sample_count": latency_samples,
        "latency_sum_ms": _safe_float(metric_event.get("avg_latency_ms"), 0.0)
        * latency_samples,
        "tokens_per_second_sum": _safe_float(
            metric_event.get("avg_tokens_per_second"), 0.0
        )
        * latency_samples,
        "token_count_output": max(
            0, _safe_int(metric_event.get("token_count_output"), 0)
        ),
    }


def merge_provider_window(
    aggregate: Dict[str, Any] | None, metric_event: Dict[str, Any]
) -> Dict[str, Any]:
    """Add a (model, provider) metrics window to its provider's window."""
    merged = dict(aggregate or {})
    if not merged:
        merged = {field: 0 for field in _PROVIDER_WINDOW_SUM_FIELDS}
        merged["models"] = []
        for field in (
            "provider",
            "window_start_ms",
            "window_end_ms",
            "window_size_seconds",
            "event_version",
        ):
            merged[field] = metric_event.get(field)
    for field, value in _provider_window_contribution(metric_event).items():
        merged[field] = merged.get(field, 0) + value
    model_name = metric_event.get("model_name")
    if model_name and model_name not in merged["models"]:
        merged["models"] = sorted([*merged["models"], str(model_name)])
    return merged


def provider_window_metric_event(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """The provider-level metrics window the provider guardrails read."""
    requests = _safe_int(aggregate.get("request_count"), 0)
    provider_requests = _safe_int(aggregate.get("provider_request_count"), 0)
    provider_latency_samples = _safe_int(
        aggregate.get("provider_latency_sample_count"), 0
    )
    latency_samples = _safe_int(aggregate.get("latency_sample_count"), 0)
    return {
        "event_type": MODEL_METRICS_1M_EVENT,
        "event_version": aggregate.get("event_version", EVENT_SCHEMA_VERSION),
        "window_start_ms": aggregate.get("window_start_ms"),
        "window_end_ms": aggregate.get("window_end_ms"),
        "window_size_seconds": aggregate.get("window_size_seconds"),
        "model_name": None,
        "models": list(aggregate.get("models", [])),
        "provider": aggregate.get("provider"),
        "request_count": requests,
        "success_count": _safe_int(aggregate.get("success_count"), 0),
        "error_count": _safe_int(aggregate.get("error_count"), 0),
        "error_rate": (
            _safe_int(aggregate.get("error_count"), 0) / requests if requests else 0.0
        ),
        "provider_request_count": provider_requests,
        "provider_failure_count": _safe_int(aggregate.get("provider_failure_count"), 0),
        "provider_failure_rate": (
            _safe_int(aggregate.get("provider_failure_count"), 0) / provider_requests
            if provider_requests
            else 0.0
        ),
        "provider_latency_sample_count": provider_latency_samples,
        "provider_avg_latency_ms": (
            _safe_float(aggregate.get("provider_latency_sum_ms"), 0.0)
            / provider_latency_samples
            if provider_latency_samples
            else 0.0
        ),
        "latency_sample_count": latency_samples,
        "avg_latency_ms": (
            _safe_float(aggregate.get("latency_sum_ms"), 0.0) / latency_samples
            if latency_samples
            else 0.0
        ),
        "avg_tokens_per_second": (
            _safe_float(aggregate.get("tokens_per_second_sum"), 0.0) / latency_samples
            if latency_samples
            else 0.0
        ),
        "token_count_output": _safe_int(aggregate.get("token_count_output"), 0),
    }


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


def _empty_routing_policy_aggregate() -> Dict[str, Any]:
    return {
        "request_count": 0,
        "success_count": 0,
        "latency_sum_ms": 0.0,
        "latency_sample_count": 0,
        "total_tokens_sum": 0,
        "cost_sum_usd": 0.0,
        "fast_lane_hits": 0,
        "query_type_counts": {},
        "model_success_counts": {},
        "model_all_counts": {},
        "failed_model_counts": {},
        "failed_provider_counts": {},
        "latest_event": {},
        "latest_event_timestamp_ms": -1,
        "max_event_timestamp_ms": -1,
        "latest_arrival_sequence": -1,
    }


def _latency_sample_count(aggregate: Dict[str, Any]) -> int:
    """Uncached requests in latency_sum_ms; older state summed every request."""
    return _safe_int(
        aggregate.get("latency_sample_count", aggregate.get("request_count", 0))
    )


def _increment_count(counts: Dict[str, int], key: str, amount: int = 1) -> None:
    if not key:
        return
    updated = int(counts.get(key, 0)) + amount
    if updated > 0:
        counts[key] = updated
    else:
        counts.pop(key, None)


PROVIDER_ATTRIBUTABLE_ERROR_TYPES = frozenset(
    {"provider_error", "timeout", "rate_limited"}
)


def counts_toward_provider_health(event: Dict[str, Any]) -> bool:
    """Return whether a completion event belongs in a provider's health signal.

    Failures classified as user_or_policy or internal say nothing about the
    provider. Events without error_type (successes and older producers) count.
    """
    error_type = event.get("error_type")
    if not error_type:
        return True
    return str(error_type).lower() in PROVIDER_ATTRIBUTABLE_ERROR_TYPES


def add_routing_policy_event(
    aggregate: Dict[str, Any],
    event: Dict[str, Any],
    *,
    event_timestamp_ms: int = 0,
    arrival_sequence: int = 0,
) -> Dict[str, Any]:
    """Increment a routing-policy aggregate with one completion event."""
    latency_samples = _latency_sample_count(aggregate)
    aggregate["request_count"] = int(aggregate.get("request_count", 0)) + 1
    succeeded = event.get("status") == "success"
    if succeeded:
        aggregate["success_count"] = int(aggregate.get("success_count", 0)) + 1
    # A cache hit's latency is the lookup, not the model.
    if not bool(event.get("cached_response", False)):
        aggregate["latency_sum_ms"] = _safe_float(
            aggregate.get("latency_sum_ms")
        ) + _safe_float(event.get("latency_ms"))
        latency_samples += 1
    aggregate["latency_sample_count"] = latency_samples
    aggregate["total_tokens_sum"] = _safe_int(
        aggregate.get("total_tokens_sum")
    ) + _safe_int(event.get("total_tokens"))
    aggregate["cost_sum_usd"] = _safe_float(
        aggregate.get("cost_sum_usd")
    ) + _safe_float(event.get("cost_usd"))
    if bool(event.get("actual_fast_lane_hit", False)):
        aggregate["fast_lane_hits"] = int(aggregate.get("fast_lane_hits", 0)) + 1

    query_type = str(event.get("query_type", "general") or "general").lower()
    _increment_count(aggregate.setdefault("query_type_counts", {}), query_type)
    selected_model = str(event.get("selected_model", "") or "")
    if selected_model:
        _increment_count(aggregate.setdefault("model_all_counts", {}), selected_model)
        if succeeded:
            _increment_count(
                aggregate.setdefault("model_success_counts", {}), selected_model
            )
        elif counts_toward_provider_health(event):
            # A failure the caller, policy or our code caused is not the model's.
            _increment_count(
                aggregate.setdefault("failed_model_counts", {}), selected_model
            )
    provider = str(event.get("provider", "") or "").lower()
    if provider and not succeeded and counts_toward_provider_health(event):
        _increment_count(aggregate.setdefault("failed_provider_counts", {}), provider)

    aggregate["max_event_timestamp_ms"] = max(
        _safe_int(aggregate.get("max_event_timestamp_ms"), -1), event_timestamp_ms
    )
    if arrival_sequence >= int(aggregate.get("latest_arrival_sequence", -1)):
        aggregate["latest_arrival_sequence"] = arrival_sequence
        aggregate["latest_event_timestamp_ms"] = event_timestamp_ms
        aggregate["latest_event"] = {
            key: event.get(key) for key in ("user_id", "session_id", "user_tier")
        }
    return aggregate


def aggregate_routing_policy_events(
    events: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compatibility adapter from the historic event list to an aggregate."""
    aggregate = _empty_routing_policy_aggregate()
    for sequence, event in enumerate(events):
        explicit_timestamp = event.get("event_timestamp_ms")
        timestamp_ms = (
            _safe_int(explicit_timestamp)
            if explicit_timestamp is not None
            else sequence
        )
        add_routing_policy_event(
            aggregate,
            event,
            event_timestamp_ms=timestamp_ms,
            arrival_sequence=sequence,
        )
    return aggregate


def merge_routing_policy_aggregate(
    target: Dict[str, Any], source: Dict[str, Any], *, sign: int = 1
) -> Dict[str, Any]:
    """Add or subtract a bucket aggregate from a rolling aggregate."""
    target["latency_sample_count"] = _latency_sample_count(
        target
    ) + sign * _latency_sample_count(source)
    for key in (
        "request_count",
        "success_count",
        "total_tokens_sum",
        "fast_lane_hits",
    ):
        target[key] = int(target.get(key, 0)) + sign * int(source.get(key, 0))
    for key in ("latency_sum_ms", "cost_sum_usd"):
        target[key] = _safe_float(target.get(key)) + sign * _safe_float(source.get(key))
    for key in (
        "query_type_counts",
        "model_success_counts",
        "model_all_counts",
        "failed_model_counts",
        "failed_provider_counts",
    ):
        target_counts = target.setdefault(key, {})
        for item, count in source.get(key, {}).items():
            _increment_count(target_counts, item, sign * int(count))
    if sign > 0:
        target["max_event_timestamp_ms"] = max(
            _safe_int(target.get("max_event_timestamp_ms"), -1),
            _safe_int(
                source.get(
                    "max_event_timestamp_ms", source.get("latest_event_timestamp_ms")
                ),
                -1,
            ),
        )
    if sign > 0 and int(source.get("latest_arrival_sequence", -1)) >= int(
        target.get("latest_arrival_sequence", -1)
    ):
        target["latest_arrival_sequence"] = int(
            source.get("latest_arrival_sequence", -1)
        )
        target["latest_event_timestamp_ms"] = int(
            source.get("latest_event_timestamp_ms", -1)
        )
        target["latest_event"] = dict(source.get("latest_event", {}))
    return target


def build_routing_policy_state_event(
    *,
    scope_type: str,
    scope_key: str,
    events: list[Dict[str, Any]],
    window_size_seconds: int = 300,
    policy_config: Dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Derive policy state through the event-list compatibility adapter."""
    if not events:
        raise ValueError("events is required to build routing policy state")
    return build_routing_policy_state_event_from_aggregate(
        scope_type=scope_type,
        scope_key=scope_key,
        aggregate=aggregate_routing_policy_events(events),
        window_size_seconds=window_size_seconds,
        policy_config=policy_config,
        timestamp=timestamp,
    )


def build_routing_policy_state_event_from_aggregate(
    *,
    scope_type: str,
    scope_key: str,
    aggregate: Dict[str, Any],
    window_size_seconds: int = 300,
    policy_config: Dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Derive rolling user/session routing policy state from counters and sums."""
    if scope_type not in {"user", "session"}:
        raise ValueError(f"Unsupported routing policy scope_type: {scope_type}")
    if not scope_key:
        raise ValueError("scope_key is required for routing policy state")
    request_count = int(aggregate.get("request_count", 0))
    if request_count <= 0:
        raise ValueError("aggregate is required to build routing policy state")
    success_count = int(aggregate.get("success_count", 0))
    error_count = request_count - success_count
    recent_error_rate = error_count / request_count if request_count > 0 else 0.0
    avg_total_tokens = _safe_int(aggregate.get("total_tokens_sum")) / request_count
    avg_cost_usd = _safe_float(aggregate.get("cost_sum_usd")) / request_count
    latency_samples = _latency_sample_count(aggregate)
    avg_latency_ms = (
        _safe_float(aggregate.get("latency_sum_ms")) / latency_samples
        if latency_samples > 0
        else 0.0
    )
    fast_lane_hits = int(aggregate.get("fast_lane_hits", 0))
    fast_lane_hit_rate = fast_lane_hits / request_count if request_count > 0 else 0.0

    query_type_counts = dict(aggregate.get("query_type_counts", {}))
    model_success_counts = dict(aggregate.get("model_success_counts", {}))
    model_all_counts = dict(aggregate.get("model_all_counts", {}))
    failed_model_counts = dict(aggregate.get("failed_model_counts", {}))
    failed_provider_counts = dict(aggregate.get("failed_provider_counts", {}))

    latest_event = dict(aggregate.get("latest_event", {}))
    user_tier = str(latest_event.get("user_tier", "free") or "free").lower()
    user_id = latest_event.get("user_id")
    session_id = latest_event.get("session_id")
    config = _merged_config(DEFAULT_ROLLING_POLICY_CONFIG, policy_config)

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
    # emitted_at is when Flink emitted the state; latest_event_at is the
    # newest completion time it covers (the maximum, not the last arrival),
    # which a replay leaves unchanged.
    latest_event_ms = max(
        _safe_int(aggregate.get("max_event_timestamp_ms"), -1),
        _safe_int(aggregate.get("latest_event_timestamp_ms"), -1),
    )
    latest_event_at = (
        _isoformat(datetime.fromtimestamp(latest_event_ms / 1000.0, tz=timezone.utc))
        if latest_event_ms >= _MIN_WALL_CLOCK_MS
        else None
    )

    return {
        "event_type": ROUTING_POLICY_STATE_EVENT,
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat(timestamp),
        "latest_event_at": latest_event_at,
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
