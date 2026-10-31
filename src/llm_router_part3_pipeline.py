"""
LLM Router Part 3: Kafka Pipeline - Stream Processing and Data Storage
Handles real-time data ingestion, processing, and storage to ClickHouse
"""

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import re
import time
from datetime import datetime, timezone
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import TopicPartition

try:
    from aiokafka import ConsumerRebalanceListener
except ImportError:  # pragma: no cover - CI stubs aiokafka without it
    ConsumerRebalanceListener = object  # type: ignore[assignment,misc]
import clickhouse_connect
from clickhouse_connect.driver import Client

from src.kafka_topics import dead_letter_topic_name, dlq_enabled
from src.prompt_assembly import PROMPT_CONTENT_METADATA_KEYS
from src.redaction import context_summary, redact, redact_text
from src.routing_features import estimate_query_tokens
from src.utils.logger import setup_logging
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import QueryRequest, InferenceResponse, SystemMetric, UserTier

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = "1.0"
ROUTING_GUARDRAIL_EVENT_TYPE = "routing.guardrails"
# What an alert row keeps in `payload_json`: identifiers and the error, never
# the request that triggered it. `metrics` holds only window aggregates.
ALERT_PAYLOAD_FIELDS = (
    "event_type",
    "event_version",
    "emitted_at",
    "request_id",
    "query_id",
    "user_id",
    "alert_type",
    "anomaly_type",
    "trigger_type",
    "severity",
    "description",
    "original_event_type",
    "model_name",
    "provider",
    "window_start_ms",
    "window_end_ms",
    "metrics",
    "error",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class _DeliveryEnvelope:
    topic_key: str
    default_topic: str
    key: Optional[str]
    value: Dict[str, Any]


def _isoformat_utc(timestamp: Optional[datetime] = None) -> str:
    """Serialize a timestamp using an explicit UTC ISO-8601 string."""
    timestamp = timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _enum_value(value: Any) -> Any:
    """Return the primitive enum value when present."""
    return getattr(value, "value", value)


def _parse_datetime(
    value: Optional[Any],
    *,
    default: Optional[datetime] = None,
    strict: bool = False,
) -> datetime:
    """Parse platform timestamps into aware UTC datetimes; naive values are UTC.

    Unparseable or out-of-range values fall back to ``default`` (or the current
    time), or raise ``ValueError`` when ``strict`` is set.
    """
    parsed: Optional[datetime] = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug(f"Failed to parse datetime value: {value}")

    if parsed is not None:
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        try:
            return parsed.astimezone(timezone.utc)
        except (OverflowError, ValueError):
            logger.debug(f"Datetime value is out of range in UTC: {value}")

    if strict:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return default or datetime.now(timezone.utc)


_UINT32_MAX = 2**32 - 1


def _window_hours(hours: Any) -> int:
    """Coerce a dashboard look-back window to a non-negative UInt32 hour count."""
    return min(max(int(hours), 0), _UINT32_MAX)


def _decode_kafka_key(key: Any) -> str:
    """Decode a Kafka key for diagnostic payloads."""
    if key is None:
        return ""
    if isinstance(key, bytes):
        return key.decode("utf-8", errors="replace")
    return str(key)


def _json_safe_payload(value: Any) -> Any:
    """Return a JSON-safe representation of an arbitrary Kafka payload."""
    if isinstance(value, bytes):
        try:
            return json.loads(value.decode("utf-8"))
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_bytes(value: Any, default: Optional[Callable[[Any], Any]] = None) -> bytes:
    """Serialize as UTF-8 JSON; non-ASCII text is not expanded to escapes."""
    try:
        return json.dumps(value, default=default, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate has no UTF-8 form; escaped JSON still round-trips it.
        return json.dumps(value, default=default).encode("utf-8")


def _json_text(value: Any, default: Optional[Callable[[Any], Any]] = None) -> str:
    return _json_bytes(value, default=default).decode("utf-8")


# Free-text fields never copied into a dead-letter payload (X6).
_PRIVATE_ROW_FIELDS = frozenset({"query_text", "query", "context", "response_text"})


def _private_free_payload(value: Any) -> Any:
    """Redact credentials, drop free text and replace `context` with its size."""
    if isinstance(value, dict) and _PRIVATE_ROW_FIELDS & set(value):
        value = dict(value)
        if "context" in value:
            context = value.pop("context")
            summary = context_summary(None if context is None else str(context))
            for key, item in summary.items():
                value.setdefault(key, item)
        for key in _PRIVATE_ROW_FIELDS:
            value.pop(key, None)
    return redact(value)


def _dead_letter_payload(value: Any, error: BaseException) -> Any:
    """Return a private-free DLQ copy of a source message value.

    A value that is not JSON (truncated or corrupt) cannot be filtered field
    by field and may hold user text, so it is replaced by its length, a hash
    prefix and the failure reason.
    """
    payload = _json_safe_payload(value)
    if isinstance(payload, (dict, list)) or payload is None:
        return _private_free_payload(payload)
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raw = str(value).encode("utf-8", errors="replace")
    return {
        "unparseable": True,
        "length": len(raw),
        "sha256_prefix": hashlib.sha256(raw).hexdigest()[:16],
        "reason": type(error).__name__,
    }


def _alert_payload(data: Dict[str, Any]) -> Any:
    if data.get("event_type") == ROUTING_GUARDRAIL_EVENT_TYPE:
        # Built by Flink from metric windows; get_routing_guardrails reads
        # its scope fields back from payload_json.
        return redact(data)
    return redact({key: data[key] for key in ALERT_PAYLOAD_FIELDS if key in data})


REQUEST_SURFACES = frozenset({"mention", "thread", "dm", "slash", "api"})


def request_source_and_surface(query_request: Any) -> Tuple[str, Optional[str]]:
    """Where a request came from: ("slack", "mention" | "thread" | "dm" |
    "slash") or ("api", "api").

    Only the server writes these metadata keys (external /route metadata is
    allowlisted), so a request without them came through the API.
    """
    metadata = getattr(query_request, "metadata", None) or {}
    source = metadata.get("source")
    if not isinstance(source, str) or not source:
        return "api", "api"
    surface = metadata.get("surface")
    if surface not in REQUEST_SURFACES:
        surface = None
    return source, surface


def thread_key_hash(query_request: Any) -> Optional[str]:
    """A Slack thread's key, hashed: requests of one thread share it."""
    metadata = getattr(query_request, "metadata", None) or {}
    slack = metadata.get("slack")
    if not isinstance(slack, dict):
        return None
    channel_id = slack.get("channel_id")
    thread_ts = slack.get("thread_ts")
    if not channel_id or not thread_ts:
        return None
    key = f"slack:{slack.get('team_id') or ''}:{channel_id}:{thread_ts}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _first_event_time(data: Dict[str, Any], *keys: str) -> datetime:
    """The first of keys that holds a valid timestamp, never the consume time.

    An event with none of them is malformed: ValueError, a permanent failure.
    """
    for key in keys:
        try:
            return _parse_datetime(data.get(key), strict=True)
        except ValueError:
            continue
    raise ValueError(f"Event has no valid timestamp in {', '.join(keys)}")


def _uint_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def build_chat_request_row(
    data: Dict[str, Any], kafka_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One chat_requests row from an inference.completed event.

    An explicit field list: no query text, context or attachment names.
    """
    request_id = str(data.get("request_id") or data.get("query_id") or "")
    if not request_id:
        raise ValueError("inference.completed event has no request_id")
    # Only the event's own times: a replay must get the same ts and version.
    completed_at = _first_event_time(data, "completion_timestamp", "emitted_at")
    ts = _first_event_time(
        data, "request_timestamp", "completion_timestamp", "emitted_at"
    )
    emitted_at = _first_event_time(data, "emitted_at", "completion_timestamp")
    return {
        "request_id": request_id,
        "ts": ts,
        "completed_at": completed_at,
        # The latest completion of a request id wins on merge; a replay of
        # the same event has the same version.
        "version": int(emitted_at.timestamp() * 1000),
        "user_id": str(data.get("user_id") or ""),
        "tier": str(data.get("user_tier") or ""),
        "source": str(data.get("source") or "api"),
        "surface": str(data.get("surface") or ""),
        "thread_key_hash": str(data.get("thread_key_hash") or ""),
        "query_type": str(data.get("query_type") or ""),
        "model": str(data.get("selected_model") or ""),
        "provider": str(data.get("provider") or ""),
        "status": str(data.get("status") or ""),
        "error_type": str(data.get("error_type") or ""),
        "error_code": str(data.get("error_code") or ""),
        "finish_reason": str(data.get("finish_reason") or ""),
        "fallback_from_model": str(data.get("fallback_from_model") or ""),
        "attempts": _uint_or_none(data.get("attempts")) or 0,
        "cached": bool(data.get("cached_response", False)),
        "routing_strategy": str(data.get("routing_strategy") or ""),
        "policy_source": str(data.get("policy_source") or ""),
        "reasoning_effort": str(data.get("reasoning_effort") or ""),
        "latency_ms": _uint_or_none(data.get("latency_ms")) or 0,
        "routing_ms": _uint_or_none(data.get("routing_ms")),
        "queue_wait_ms": _uint_or_none(data.get("queue_wait_ms")),
        "ttft_ms": _uint_or_none(data.get("ttft_ms")),
        "token_count_input": _uint_or_none(data.get("token_count_input")) or 0,
        "token_count_output": _uint_or_none(data.get("token_count_output")) or 0,
        "reasoning_tokens": _uint_or_none(data.get("reasoning_tokens")) or 0,
        "provider_cached_input_tokens": (
            _uint_or_none(data.get("provider_cached_input_tokens")) or 0
        ),
        "cost_usd": float(data.get("cost_usd") or 0.0),
        "event_id": "",
        "kafka_topic": "",
        "kafka_partition": -1,
        "kafka_offset": -1,
        **(kafka_metadata or {}),
    }


CHAT_FEEDBACK_RATINGS = frozenset({"up", "down"})


def build_chat_feedback_event(
    *,
    request_id: str,
    user_id: str,
    rating: str,
    click_id: str,
    source: str = "slack",
) -> Dict[str, Any]:
    """A user's rating of one chat answer; an explicit field list, no text."""
    if rating not in CHAT_FEEDBACK_RATINGS:
        raise ValueError(f"Unknown chat feedback rating: {rating!r}")
    return {
        "event_type": "chat.feedback",
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat_utc(),
        "request_id": request_id,
        "user_id": user_id,
        "rating": rating,
        "click_id": click_id,
        "source": source,
    }


def build_chat_feedback_row(
    data: Dict[str, Any], kafka_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """One chat_feedback row from a chat.feedback event."""
    request_id = str(data.get("request_id") or "")
    rating = str(data.get("rating") or "")
    if not request_id or rating not in CHAT_FEEDBACK_RATINGS:
        raise ValueError("chat.feedback event needs a request_id and a rating")
    clicked_at = _first_event_time(data, "emitted_at")
    return {
        "request_id": request_id,
        "user_id": str(data.get("user_id") or ""),
        "rating": rating,
        "click_id": str(data.get("click_id") or ""),
        "source": str(data.get("source") or ""),
        "clicked_at": clicked_at,
        "version": int(clicked_at.timestamp() * 1000),
        "event_id": "",
        "kafka_topic": "",
        "kafka_partition": -1,
        "kafka_offset": -1,
        **(kafka_metadata or {}),
    }


def _event_query_text(query_request: QueryRequest) -> str:
    """The query as events carry it: redacted, and for Slack without the
    attached files' names (a file-only message asks about them by name)."""
    query = query_request.query
    if (query_request.metadata or {}).get("source") == "slack":
        for attachment in query_request.attachments:
            if attachment.name:
                query = query.replace(attachment.name, "[attachment]")
    return redact_text(query)


def _without_attachment_names(slack_metadata: Any) -> Any:
    if not isinstance(slack_metadata, dict):
        return slack_metadata
    return {
        key: value for key, value in slack_metadata.items() if key != "attachment_names"
    }


def build_request_raw_event(
    query_request: QueryRequest,
    classification: Optional[Tuple[str, float]] = None,
) -> Dict[str, Any]:
    """Build the pre-inference event emitted by the API request path.

    `classification` is the (query type, confidence) the chat ingress got from
    the router, if any (R13).
    """
    # A Slack file's name can say what is in it (X6): Slack requests send
    # only each file's type and size. /route keeps its existing shape.
    from_slack = (query_request.metadata or {}).get("source") == "slack"
    attachments = [
        {
            "id": attachment.id,
            **({} if from_slack else {"name": attachment.name}),
            "type": _enum_value(attachment.type),
            "size_bytes": attachment.size_bytes,
            "mime_type": attachment.mime_type,
        }
        for attachment in query_request.attachments
    ]

    return {
        "event_type": "requests.raw",
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat_utc(),
        "request_id": query_request.request_id,
        "query_id": query_request.request_id,
        "user_id": query_request.user_id,
        "user_tier": _enum_value(query_request.user_tier),
        "request_timestamp": _isoformat_utc(query_request.timestamp),
        "query_text": _event_query_text(query_request),
        # Other people's Slack messages ride in `context`; send only its shape.
        **context_summary(query_request.context),
        "query_token_estimate": estimate_query_tokens(
            query_request.query, query_request.context or ""
        ),
        "max_tokens": query_request.max_tokens,
        "temperature": query_request.temperature,
        "priority": query_request.priority,
        "tool_policy": _enum_value(query_request.tool_policy),
        "allowed_tools": list(query_request.allowed_tools or []),
        "web_search_options": query_request.web_search_options.model_dump(mode="json")
        if query_request.web_search_options
        else None,
        "session_id": query_request.session_id,
        "conversation_id": query_request.conversation_id,
        # Prompt content assembled in process (Slack memory hits) stays out.
        "metadata": redact(
            {
                key: (
                    _without_attachment_names(value)
                    if from_slack and key == "slack"
                    else value
                )
                for key, value in (query_request.metadata or {}).items()
                if key not in PROMPT_CONTENT_METADATA_KEYS
            }
        ),
        "attachments": attachments,
        "attachments_count": len(attachments),
        # The router's classification when the ingress got one (R13); Flink
        # keeps it instead of inferring its own type.
        "query_type": classification[0] if classification else None,
        "classifier_confidence": classification[1] if classification else None,
    }


def build_inference_completed_event(
    query_request: QueryRequest,
    inference_response: InferenceResponse,
    routing_decision: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the post-inference event emitted after request completion."""
    query_type = getattr(routing_decision, "query_type", None)
    routing_strategy = getattr(routing_decision, "routing_strategy", None)
    route_to_fast_lane = bool(getattr(routing_decision, "route_to_fast_lane", False))
    actual_fast_lane_hit = bool(
        getattr(routing_decision, "actual_fast_lane_hit", False)
    )
    policy_source = getattr(routing_decision, "policy_source", None)
    hint_reason = getattr(routing_decision, "hint_reason", None)
    tool_calls = list(getattr(inference_response, "tool_calls", []) or [])
    sources = list(getattr(inference_response, "sources", []) or [])
    web_search_call = next(
        (
            tool_call
            for tool_call in tool_calls
            if getattr(tool_call, "name", "") == "web_search"
        ),
        None,
    )
    source_domains = [
        domain
        for domain in (getattr(source, "source_domain", None) for source in sources)
        if domain
    ]
    source, surface = request_source_and_surface(query_request)
    routing_ms = getattr(routing_decision, "routing_time_ms", None)

    return {
        "event_type": "inference.completed",
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat_utc(),
        "request_id": query_request.request_id,
        "query_id": query_request.request_id,
        "user_id": query_request.user_id,
        "user_tier": _enum_value(query_request.user_tier),
        "session_id": query_request.session_id,
        "conversation_id": query_request.conversation_id,
        "source": source,
        "surface": surface,
        "thread_key_hash": thread_key_hash(query_request),
        "request_timestamp": _isoformat_utc(query_request.timestamp),
        "completion_timestamp": _isoformat_utc(inference_response.timestamp),
        "query_text": _event_query_text(query_request),
        "query_type": _enum_value(query_type),
        "selected_model": inference_response.model_name,
        "provider": inference_response.provider,
        "status": "error" if inference_response.error else "success",
        "error_message": inference_response.error or "",
        "error_type": inference_response.error_type,
        "error_code": inference_response.error_code,
        "finish_reason": inference_response.finish_reason,
        "provider_request_id": inference_response.provider_request_id,
        "latency_ms": inference_response.latency_ms,
        # Where the time went: routing; the ingress and provider queues;
        # time to first token (no streaming yet).
        "routing_ms": (
            int(routing_ms)
            if isinstance(routing_ms, (int, float)) and not isinstance(routing_ms, bool)
            else None
        ),
        "queue_wait_ms": _uint_or_none(
            getattr(inference_response, "queue_wait_ms", None)
        ),
        "ttft_ms": None,
        "token_count_input": inference_response.token_count_input,
        "token_count_output": inference_response.token_count_output,
        "total_tokens": inference_response.total_tokens,
        "provider_cached_input_tokens": inference_response.provider_cached_input_tokens,
        "provider_cache_creation_input_tokens": (
            inference_response.provider_cache_creation_input_tokens
        ),
        "provider_cache_read_input_tokens": (
            inference_response.provider_cache_read_input_tokens
        ),
        "reasoning_tokens": inference_response.reasoning_tokens,
        "reasoning_effort": getattr(inference_response, "reasoning_effort", None),
        "tokens_per_second": inference_response.tokens_per_second,
        "cost_usd": inference_response.cost_usd,
        "cached_response": inference_response.cached,
        "context_compressed": inference_response.compressed_context,
        "response_length_chars": len(inference_response.response_text or ""),
        "routing_reason": getattr(routing_decision, "routing_reason", None),
        "routing_strategy": _enum_value(routing_strategy),
        "route_to_fast_lane": route_to_fast_lane,
        "actual_fast_lane_hit": actual_fast_lane_hit,
        "policy_source": policy_source,
        "hint_reason": hint_reason,
        # A1 shadow mode: the model the policy cache would have chosen.
        "shadow_selected_model": getattr(
            routing_decision, "shadow_selected_model", None
        ),
        "fallback_models": list(getattr(routing_decision, "fallback_models", []) or []),
        "fallback_from_model": getattr(inference_response, "fallback_from_model", None),
        "attempts": getattr(inference_response, "attempts", 1),
        "tools_used": [getattr(tool_call, "name", "") for tool_call in tool_calls],
        "web_search_result_count": len(sources),
        "web_search_latency_ms": getattr(web_search_call, "latency_ms", 0)
        if web_search_call
        else 0,
        "web_search_provider": getattr(web_search_call, "provider", None)
        if web_search_call
        else None,
        "web_search_blocked_reason": (
            getattr(web_search_call, "error", None) if web_search_call else None
        ),
        "source_domains": source_domains,
    }


# Fields of a routing.decisions candidate entry; anything else is dropped.
ROUTING_CANDIDATE_FIELDS = (
    "model",
    "provider",
    "eligible",
    "reject_reason",
    "score",
    "score_breakdown",
    "selected",
    "bypassed_guardrail",
)


def _routing_candidate_payload(candidate: Any) -> Dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        candidate = candidate.model_dump()
    if not isinstance(candidate, dict):
        candidate = dict(vars(candidate))
    payload = {name: candidate.get(name) for name in ROUTING_CANDIDATE_FIELDS}
    payload["eligible"] = bool(payload["eligible"])
    payload["selected"] = bool(payload["selected"])
    payload["score_breakdown"] = {
        str(name): float(value)
        for name, value in dict(payload["score_breakdown"] or {}).items()
    }
    # Only rows routing kept despite a guardrail (fail-open) carry it.
    if not payload["bypassed_guardrail"]:
        payload.pop("bypassed_guardrail")
    return payload


def build_routing_decision_event(
    query_request: QueryRequest,
    routing_decision: Any,
    inference_response: Optional[InferenceResponse] = None,
) -> Dict[str, Any]:
    """Build the routing.decisions event: why the router chose the model.

    An explicit field list (X6): the routing context holds the raw query, so
    nothing from it is copied wholesale. Decisions built by tests or older
    callers may lack fields; those default.
    """
    features = getattr(routing_decision, "features", None)
    timestamp = getattr(routing_decision, "timestamp", None)
    return {
        "event_type": "routing.decision",
        "event_version": EVENT_SCHEMA_VERSION,
        "emitted_at": _isoformat_utc(),
        "decided_at": _isoformat_utc(
            timestamp if isinstance(timestamp, datetime) else None
        ),
        "request_id": query_request.request_id,
        "user_tier": _enum_value(
            getattr(routing_decision, "user_tier", None) or query_request.user_tier
        ),
        # The router's choice; after an inference fallback the decision's
        # selected_model is the fallback, which served_model reports.
        "selected_model": getattr(routing_decision, "routed_model", None)
        or getattr(routing_decision, "selected_model", None),
        "served_model": getattr(inference_response, "model_name", None),
        "fallback_from_model": getattr(inference_response, "fallback_from_model", None),
        "query_type": _enum_value(getattr(routing_decision, "query_type", None)),
        "routing_strategy": getattr(routing_decision, "routing_strategy", None),
        "reason_code": getattr(routing_decision, "reason_code", None),
        "routing_reason": getattr(routing_decision, "routing_reason", None),
        "policy_source": getattr(routing_decision, "policy_source", None),
        "token_count": getattr(routing_decision, "token_count", None),
        "estimated_cost": getattr(routing_decision, "estimated_cost", None),
        "confidence": getattr(routing_decision, "confidence", None),
        "classifier_confidence": getattr(features, "classification_confidence", None),
        "query_complexity": getattr(features, "query_complexity", None),
        "routing_time_ms": getattr(routing_decision, "routing_time_ms", None),
        "route_to_fast_lane": bool(
            getattr(routing_decision, "route_to_fast_lane", False)
        ),
        "actual_fast_lane_hit": bool(
            getattr(routing_decision, "actual_fast_lane_hit", False)
        ),
        "fallback_models": list(getattr(routing_decision, "fallback_models", []) or []),
        "shadow_selected_model": getattr(
            routing_decision, "shadow_selected_model", None
        ),
        "shadow_policy_source": getattr(routing_decision, "shadow_policy_source", None),
        "candidates": [
            _routing_candidate_payload(candidate)
            for candidate in list(getattr(routing_decision, "candidates", []) or [])
        ],
    }


@dataclass
class QueryLogEntry:
    """Structure for query log entries"""

    query_id: str
    timestamp: datetime
    user_id: str
    user_tier: str
    query_text: str
    query_type: str
    selected_model: str
    token_count_input: int
    token_count_output: int
    latency_ms: int
    cost_usd: float
    status: str
    error_message: str = ""
    context_compressed: bool = False
    compression_ratio: float = 0.0
    cached_response: bool = False
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            "event_id": self.event_id or self.query_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "user_tier": self.user_tier.value
            if isinstance(self.user_tier, UserTier)
            else str(self.user_tier),
            "query_text": redact_text(self.query_text)[:10000],
            "query_type": self.query_type.value
            if hasattr(self.query_type, "value")
            else str(self.query_type),
            "selected_model": self.selected_model,
            "token_count_input": self.token_count_input,
            "token_count_output": self.token_count_output,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error_message": self.error_message[:1000] if self.error_message else "",
            "context_compressed": self.context_compressed,
            "compression_ratio": self.compression_ratio,
            "cached_response": self.cached_response,
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


@dataclass
class MetricEntry:
    """Structure for system metrics"""

    timestamp: datetime
    service: str
    metric_name: str
    metric_value: float
    labels: Dict[str, str]
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        event_id = self.event_id or (
            f"{self.service}:{self.metric_name}:{self.timestamp.isoformat()}"
        )
        return {
            "event_id": event_id,
            "timestamp": self.timestamp,
            "service": self.service,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "labels": self.labels,  # ClickHouse Map type
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


@dataclass
class ModelPerformanceEntry:
    """Structure for Flink windowed model performance events."""

    timestamp: datetime
    model_name: str
    provider: str
    window_start_ms: int
    window_end_ms: int
    window_size_seconds: int
    requests_count: int
    success_count: int
    success_rate: float
    avg_latency_ms: float
    avg_tokens_per_second: float
    error_count: int
    total_tokens: int
    total_cost_usd: float
    queries_per_second: float
    cache_hit_rate: float
    cached_count: int
    gpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        event_id = self.event_id or (
            f"{self.model_name}:{self.provider}:{self.window_start_ms}:{self.window_end_ms}"
        )
        return {
            "event_id": event_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "provider": self.provider,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "window_size_seconds": self.window_size_seconds,
            "requests_count": self.requests_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "queries_per_second": self.queries_per_second,
            "cache_hit_rate": self.cache_hit_rate,
            "cached_count": self.cached_count,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage_gb": self.memory_usage_gb,
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


@dataclass
class AlertEventEntry:
    """Structure for persisted alert events."""

    timestamp: datetime
    alert_type: str
    severity: str
    description: str
    anomaly_type: str = ""
    source_event_type: str = ""
    request_id: str = ""
    query_id: str = ""
    user_id: str = ""
    model_name: str = ""
    provider: str = ""
    window_start_ms: int = 0
    window_end_ms: int = 0
    payload_json: str = "{}"
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        event_id = self.event_id or (
            f"{self.source_event_type}:{self.alert_type}:{self.request_id}:"
            f"{self.query_id}:{self.model_name}:{self.provider}:"
            f"{self.window_start_ms}:{self.window_end_ms}:{self.timestamp.isoformat()}"
        )
        return {
            "event_id": event_id,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "description": self.description,
            "anomaly_type": self.anomaly_type,
            "source_event_type": self.source_event_type,
            "request_id": self.request_id,
            "query_id": self.query_id,
            "user_id": self.user_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "payload_json": self.payload_json,
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


@dataclass
class RoutingPolicyStateEntry:
    """Structure for persisted routing.policy_state events."""

    timestamp: datetime
    scope_type: str
    scope_key: str
    user_id: str = ""
    session_id: str = ""
    user_tier: str = ""
    hint_reason: str = ""
    recent_request_count: int = 0
    recent_error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    fast_lane_hit_rate: float = 0.0
    dominant_query_type: str = ""
    query_complexity: str = ""
    requires_low_latency: bool = False
    requires_high_reasoning: bool = False
    route_to_fast_lane: bool = False
    burst_protection_active: bool = False
    enterprise_priority_active: bool = False
    preferred_models: List[str] = field(default_factory=list)
    avoid_models: List[str] = field(default_factory=list)
    avoid_providers: List[str] = field(default_factory=list)
    payload_json: str = "{}"
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        event_id = self.event_id or (
            f"{self.scope_type}:{self.scope_key}:{self.timestamp.isoformat()}"
        )
        return {
            "event_id": event_id,
            "timestamp": self.timestamp,
            "scope_type": self.scope_type,
            "scope_key": self.scope_key,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_tier": self.user_tier,
            "hint_reason": self.hint_reason,
            "recent_request_count": self.recent_request_count,
            "recent_error_rate": self.recent_error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "fast_lane_hit_rate": self.fast_lane_hit_rate,
            "dominant_query_type": self.dominant_query_type,
            "query_complexity": self.query_complexity,
            "requires_low_latency": self.requires_low_latency,
            "requires_high_reasoning": self.requires_high_reasoning,
            "route_to_fast_lane": self.route_to_fast_lane,
            "burst_protection_active": self.burst_protection_active,
            "enterprise_priority_active": self.enterprise_priority_active,
            "preferred_models": list(self.preferred_models),
            "avoid_models": list(self.avoid_models),
            "avoid_providers": list(self.avoid_providers),
            "payload_json": self.payload_json,
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


@dataclass
class RoutingDecisionEntry:
    """Structure for persisted routing.decisions events (no query, no context)."""

    timestamp: datetime
    request_id: str
    selected_model: str = ""
    served_model: str = ""
    fallback_from_model: str = ""
    user_tier: str = ""
    query_type: str = ""
    routing_strategy: str = ""
    reason_code: str = ""
    routing_reason: str = ""
    policy_source: str = ""
    token_count: int = 0
    estimated_cost: float = 0.0
    confidence: float = 0.0
    classifier_confidence: float = 0.0
    route_to_fast_lane: bool = False
    actual_fast_lane_hit: bool = False
    fallback_models: List[str] = field(default_factory=list)
    shadow_selected_model: str = ""
    shadow_policy_source: str = ""
    candidates_json: str = "[]"
    event_id: str = ""
    kafka_topic: str = ""
    kafka_partition: int = -1
    kafka_offset: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            # One decision per request: a replayed event replaces its row.
            "event_id": self.event_id or self.request_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "selected_model": self.selected_model,
            "served_model": self.served_model,
            "fallback_from_model": self.fallback_from_model,
            "user_tier": self.user_tier,
            "query_type": self.query_type,
            "routing_strategy": self.routing_strategy,
            "reason_code": self.reason_code,
            "routing_reason": self.routing_reason,
            "policy_source": self.policy_source,
            "token_count": self.token_count,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "classifier_confidence": self.classifier_confidence,
            "route_to_fast_lane": self.route_to_fast_lane,
            "actual_fast_lane_hit": self.actual_fast_lane_hit,
            "fallback_models": list(self.fallback_models),
            "shadow_selected_model": self.shadow_selected_model,
            "shadow_policy_source": self.shadow_policy_source,
            "candidates_json": self.candidates_json,
            "kafka_topic": self.kafka_topic,
            "kafka_partition": self.kafka_partition,
            "kafka_offset": self.kafka_offset,
        }


# ClickHouse insert failures fall in three kinds, matched by class name
# because clickhouse-connect is optional in CI:
#   "row":       only some rows can never be written; bisect and dead-letter them.
#   "schema":    the table does not match the code (unknown column or table).
#                Every row would fail, so keep the batch and back off loudly.
#   "transient": anything else; keep the batch and back off.
_ROW_ERROR_NAMES = frozenset({"DataError"})
_SCHEMA_ERROR_NAMES = frozenset({"ProgrammingError"})
_TRANSIENT_ERROR_NAMES = frozenset({"OperationalError"})
# Server-side "Code: N" values that depend only on the rows being inserted.
_ROW_CLICKHOUSE_ERROR_CODES = frozenset(
    {
        6,  # CANNOT_PARSE_TEXT
        26,  # CANNOT_PARSE_QUOTED_STRING
        27,  # CANNOT_PARSE_INPUT_ASSERTION_FAILED
        38,  # CANNOT_PARSE_DATE
        41,  # CANNOT_PARSE_DATETIME
        53,  # TYPE_MISMATCH
        69,  # ARGUMENT_OUT_OF_BOUND
        70,  # CANNOT_CONVERT_TYPE
        72,  # CANNOT_PARSE_NUMBER
        117,  # INCORRECT_DATA
        321,  # VALUE_IS_OUT_OF_RANGE_OF_DATA_TYPE
        691,  # UNKNOWN_ELEMENT_OF_ENUM
    }
)
# Server-side codes that mean the schema, not the rows, is wrong.
_SCHEMA_CLICKHOUSE_ERROR_CODES = frozenset(
    {
        16,  # NO_SUCH_COLUMN_IN_TABLE
        47,  # UNKNOWN_IDENTIFIER
        60,  # UNKNOWN_TABLE
        81,  # UNKNOWN_DATABASE
    }
)
_CLICKHOUSE_ERROR_CODE = re.compile(r"\bCode:\s*(\d+)")


def _classify_insert_error(error: Optional[BaseException]) -> str:
    """Return "row", "schema" or "transient" for a failed ClickHouse insert."""
    if error is None:
        return "transient"
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & _TRANSIENT_ERROR_NAMES:
        return "transient"
    if names & _SCHEMA_ERROR_NAMES:
        return "schema"
    if names & _ROW_ERROR_NAMES:
        return "row"
    if "DatabaseError" in names:
        match = _CLICKHOUSE_ERROR_CODE.search(str(error))
        code = int(match.group(1)) if match else None
        if code in _SCHEMA_CLICKHOUSE_ERROR_CODES:
            return "schema"
        if code in _ROW_CLICKHOUSE_ERROR_CODES:
            return "row"
        return "transient"
    # Row serialisation in to_dict() or in the driver's column packers.
    if isinstance(error, (TypeError, ValueError, OverflowError)):
        return "row"
    return "transient"


def _row_field(row: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dataclass row or a plain dict row."""
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _dead_letter_row_payload(row: Any) -> Dict[str, Any]:
    """Return a JSON-safe, private-free copy of a rejected ClickHouse row."""
    try:
        if isinstance(row, Mapping):
            data = dict(row)
        elif callable(getattr(row, "to_dict", None)):
            data = row.to_dict()
        else:
            data = {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    payload: Dict[str, Any] = {"row_type": type(row).__name__}
    for key, value in data.items():
        if key in _PRIVATE_ROW_FIELDS:
            continue
        if isinstance(value, datetime):
            value = _isoformat_utc(value)
        payload[key] = _json_safe_payload(value)
    return payload


class AckTimeoutError(asyncio.TimeoutError):
    """The record may already be on the broker: we only stopped waiting for its ack."""


def _is_ack_timeout(error: BaseException) -> bool:
    """Return whether a send failed after the record may have been written.

    A timeout before the record reached the producer's accumulator (metadata
    or buffer wait) is a plain TimeoutError and is safe to retry.
    """
    return isinstance(error, AckTimeoutError)


def _has_two_phase_send(producer: Any) -> bool:
    """Return whether ``producer.send`` enqueues and returns a delivery future.

    True for aiokafka's producer, whose class defines ``async def send``. Test
    doubles that only mock ``send_and_wait`` use the single-call path.
    """
    return inspect.iscoroutinefunction(getattr(type(producer), "send", None))


class KafkaProducerManager:
    """Manages Kafka message production"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.producer = None
        self.topics = config.get("topics", {})
        producer_config = config.get("producer", {})
        self.wait_for_ack = producer_config.get("wait_for_ack", True)
        self.retries = int(producer_config.get("retries", 3) or 0)
        self.retry_backoff_ms = int(producer_config.get("retry_backoff_ms", 100) or 0)
        self.send_timeout_seconds = float(
            producer_config.get(
                "send_timeout_seconds",
                producer_config.get("request_timeout_ms", 40000) / 1000,
            )
        )
        self.raise_on_failure = bool(producer_config.get("raise_on_failure", False))
        self.queue_capacity = int(producer_config.get("queue_capacity", 10000) or 10000)
        self.dispatcher_batch_size = int(
            producer_config.get("dispatcher_batch_size", 256) or 256
        )
        self.shutdown_drain_timeout_seconds = float(
            producer_config.get("shutdown_drain_timeout_seconds", 10) or 10
        )
        self.shutdown_cancel_timeout_seconds = float(
            producer_config.get("shutdown_cancel_timeout_seconds", 1) or 1
        )
        self.shutdown_producer_timeout_seconds = float(
            producer_config.get("shutdown_producer_timeout_seconds", 1) or 1
        )
        self._delivery_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_capacity)
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._accepting = False
        self.consecutive_failures = 0
        self.last_error: Optional[str] = None

        # Producer configuration
        enable_idempotence = bool(producer_config.get("enable_idempotence", True))
        acks = producer_config.get("acks", "all")
        if enable_idempotence and acks != "all":
            logger.warning("Kafka idempotent producer requires acks=all; overriding")
            acks = "all"

        self.producer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "value_serializer": self._serialize_message,
            "key_serializer": lambda x: x.encode("utf-8") if x else None,
            "acks": acks,
            "compression_type": producer_config.get("compression_type", "gzip"),
            "linger_ms": int(producer_config.get("linger_ms", 5) or 0),
            "max_batch_size": int(producer_config.get("batch_size", 16384) or 16384),
            "request_timeout_ms": int(
                producer_config.get("request_timeout_ms", 40000) or 40000
            ),
            "retry_backoff_ms": self.retry_backoff_ms,
            "enable_idempotence": enable_idempotence,
        }

    async def initialize(self):
        """Initialize Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(**self.producer_config)
            await self.producer.start()
            self._accepting = True
            self._dispatcher_task = asyncio.create_task(
                self._dispatch_loop(), name="kafka_producer_dispatcher"
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def is_healthy(self) -> bool:
        """Return whether the Kafka producer has been initialized."""
        failure_threshold = int(
            self.config.get("producer", {}).get("health_failure_threshold", 5) or 5
        )
        return (
            self.producer is not None and self.consecutive_failures < failure_threshold
        )

    def _topic_name(self, topic_key: str, default_topic: str) -> str:
        """Resolve a logical topic key to the configured Kafka topic name."""
        return self.topics.get(topic_key, default_topic)

    async def _send_message(
        self,
        topic_key: str,
        default_topic: str,
        *,
        key: Optional[str],
        value: Dict[str, Any],
    ) -> bool:
        """Send a JSON-serializable message to Kafka and update metrics."""
        topic_name = self._topic_name(topic_key, default_topic)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            started_at = time.perf_counter()
            try:
                await self._send_once(topic_name, key=key, value=value)
                PIPELINE_METRICS.messages_produced.labels(topic=topic_key).inc()
                PIPELINE_METRICS.producer_ack_latency.labels(topic=topic_key).observe(
                    time.perf_counter() - started_at
                )
                self.consecutive_failures = 0
                self.last_error = None
                return True
            except Exception as e:
                last_error = e
                if attempt >= self.retries or _is_ack_timeout(e):
                    # An ack timeout does not mean the broker dropped the record:
                    # it may already be written. Resending here would duplicate
                    # it, so leave delivery retries to the idempotent producer.
                    break
                backoff_seconds = (self.retry_backoff_ms / 1000.0) * (2**attempt)
                if backoff_seconds > 0:
                    await asyncio.sleep(min(backoff_seconds, 5.0))

        self.consecutive_failures += 1
        self.last_error = str(last_error) if last_error else "unknown producer error"
        logger.error(f"Failed to produce {topic_key} message: {self.last_error}")
        PIPELINE_METRICS.producer_errors.inc()
        if self.raise_on_failure and last_error is not None:
            raise last_error
        return False

    def _enqueue_message(
        self,
        topic_key: str,
        default_topic: str,
        *,
        key: Optional[str],
        value: Dict[str, Any],
    ) -> bool:
        """Accept a message for background delivery without yielding control."""
        if not self._accepting:
            PIPELINE_METRICS.producer_queue_dropped.labels(
                topic=topic_key, reason="not_accepting"
            ).inc()
            return False
        envelope = _DeliveryEnvelope(topic_key, default_topic, key, value)
        try:
            self._delivery_queue.put_nowait(envelope)
        except asyncio.QueueFull:
            PIPELINE_METRICS.producer_queue_dropped.labels(
                topic=topic_key, reason="queue_full"
            ).inc()
            return False
        PIPELINE_METRICS.producer_queue_accepted.labels(topic=topic_key).inc()
        PIPELINE_METRICS.producer_queue_depth.set(self._delivery_queue.qsize())
        return True

    async def _send_envelope(self, envelope) -> bool:
        topic_key = envelope.topic_key
        try:
            return await self._send_message(
                topic_key,
                envelope.default_topic,
                key=envelope.key,
                value=envelope.value,
            )
        except Exception:
            return False

    def _record_delivery(self, envelope, delivered: bool) -> None:
        topic_key = envelope.topic_key
        if delivered:
            PIPELINE_METRICS.producer_queue_delivered.labels(topic=topic_key).inc()
        else:
            PIPELINE_METRICS.producer_delivery_failed.labels(topic=topic_key).inc()

    async def _deliver_envelope(self, envelope) -> None:
        self._record_delivery(envelope, await self._send_envelope(envelope))

    @staticmethod
    def _consume_task_result(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _dispatch_loop(self) -> None:
        """Drain accepted messages in FIFO batches and deliver them concurrently."""
        while self._accepting or not self._delivery_queue.empty():
            envelope = await self._delivery_queue.get()
            batch = [envelope]
            while len(batch) < self.dispatcher_batch_size:
                try:
                    batch.append(self._delivery_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            PIPELINE_METRICS.producer_queue_depth.set(self._delivery_queue.qsize())
            deliveries = [
                asyncio.create_task(self._send_envelope(item)) for item in batch
            ]
            try:
                done, _ = await asyncio.wait(
                    deliveries, return_when=asyncio.ALL_COMPLETED
                )
                for item, task in zip(batch, deliveries):
                    if task in done:
                        self._record_delivery(item, task.result())
            except asyncio.CancelledError:
                unfinished = [task for task in deliveries if not task.done()]
                for task in unfinished:
                    task.cancel()
                if unfinished:
                    await asyncio.wait(
                        unfinished, timeout=self.shutdown_cancel_timeout_seconds
                    )
                for item, task in zip(batch, deliveries):
                    if task.done() and not task.cancelled():
                        self._record_delivery(item, task.result())
                    elif task.done():
                        PIPELINE_METRICS.producer_queue_dropped.labels(
                            topic=item.topic_key, reason="shutdown_timeout"
                        ).inc()
                    else:
                        PIPELINE_METRICS.producer_queue_dropped.labels(
                            topic=item.topic_key, reason="shutdown_abandoned"
                        ).inc()
                        task.add_done_callback(self._consume_task_result)
                raise
            finally:
                for _ in batch:
                    self._delivery_queue.task_done()

    def _drop_queued_messages(self, reason: str) -> None:
        while True:
            try:
                envelope = self._delivery_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            PIPELINE_METRICS.producer_queue_dropped.labels(
                topic=envelope.topic_key, reason=reason
            ).inc()
            self._delivery_queue.task_done()
        PIPELINE_METRICS.producer_queue_depth.set(0)

    async def _send_once(
        self,
        topic: str,
        *,
        key: Optional[str],
        value: Dict[str, Any],
    ):
        """Send one message and wait for the configured delivery guarantee."""
        if self.producer is None:
            raise RuntimeError("Kafka producer has not been initialized")

        if self.wait_for_ack and _has_two_phase_send(self.producer):
            # Phase 1: wait for metadata and buffer space. A timeout here means
            # the record was never enqueued, so the caller may retry it.
            delivery = await asyncio.wait_for(
                self.producer.send(topic=topic, key=key, value=value),
                timeout=self.send_timeout_seconds,
            )
            # Phase 2: the record is in the accumulator; the idempotent
            # producer retries delivery itself, so never resend after this.
            try:
                return await asyncio.wait_for(
                    delivery, timeout=self.send_timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                raise AckTimeoutError(str(exc)) from exc

        if self.wait_for_ack:
            send_and_wait = getattr(self.producer, "send_and_wait", None)
            if callable(send_and_wait):
                try:
                    return await asyncio.wait_for(
                        send_and_wait(topic=topic, key=key, value=value),
                        timeout=self.send_timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    # Cannot tell which phase timed out: do not resend.
                    raise AckTimeoutError(str(exc)) from exc

        send_result = await self.producer.send(topic=topic, key=key, value=value)
        if self.wait_for_ack and inspect.isawaitable(send_result):
            return await asyncio.wait_for(
                send_result,
                timeout=self.send_timeout_seconds,
            )
        return send_result

    async def produce_request_raw(
        self,
        query_request: QueryRequest,
        classification: Optional[Tuple[str, float]] = None,
    ):
        """Produce a pre-inference API request event."""
        return self._enqueue_message(
            "requests_raw",
            "requests.raw",
            key=query_request.user_id,
            value=build_request_raw_event(query_request, classification=classification),
        )

    async def produce_inference_completed(
        self,
        query_request: QueryRequest,
        inference_response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Produce a post-inference completion event."""
        return self._enqueue_message(
            "inference_completed",
            "inference.completed",
            key=query_request.request_id,
            value=build_inference_completed_event(
                query_request,
                inference_response,
                routing_decision=routing_decision,
            ),
        )

    async def produce_chat_feedback(
        self,
        *,
        request_id: str,
        user_id: str,
        rating: str,
        click_id: str,
        source: str = "slack",
    ):
        """Produce a chat.feedback event, keyed by the request it rates."""
        event = build_chat_feedback_event(
            request_id=request_id,
            user_id=user_id,
            rating=rating,
            click_id=click_id,
            source=source,
        )
        return self._enqueue_message(
            "chat_feedback",
            "chat.feedback",
            key=request_id,
            value=event,
        )

    async def produce_routing_decision(
        self,
        query_request: QueryRequest,
        routing_decision: Any,
        inference_response: Optional[InferenceResponse] = None,
    ):
        """Produce a routing.decisions event, keyed by request_id."""
        return self._enqueue_message(
            "routing_decisions",
            "routing.decisions",
            key=query_request.request_id,
            value=build_routing_decision_event(
                query_request, routing_decision, inference_response
            ),
        )

    async def produce_query_log(
        self,
        query_request: QueryRequest,
        inference_response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Produce query log message"""
        query_type = getattr(routing_decision, "query_type", "general")
        query_log = QueryLogEntry(
            query_id=query_request.request_id,
            timestamp=_parse_datetime(inference_response.timestamp),
            user_id=query_request.user_id,
            user_tier=query_request.user_tier,
            query_text=query_request.query,
            query_type=query_type.value
            if hasattr(query_type, "value")
            else str(query_type),
            selected_model=inference_response.model_name,
            token_count_input=inference_response.token_count_input,
            token_count_output=inference_response.token_count_output,
            latency_ms=inference_response.latency_ms,
            cost_usd=inference_response.cost_usd,
            status="error" if inference_response.error else "success",
            error_message=getattr(inference_response, "error", ""),
            context_compressed=getattr(inference_response, "compressed_context", False),
            compression_ratio=0.3
            if getattr(inference_response, "compressed_context", False)
            else 0.0,
            cached_response=inference_response.cached,
        )

        return self._enqueue_message(
            "queries",
            "llm-queries",
            key=query_log.user_id,
            value=query_log.to_dict(),
        )

    async def produce_response_log(self, response_data: Dict[str, Any]):
        """Produce response log message"""
        return self._enqueue_message(
            "responses",
            "llm-responses",
            key=response_data.get("query_id"),
            value=response_data,
        )

    async def produce_metric(
        self,
        service: str,
        metric_name: str,
        metric_value: float,
        labels: Dict[str, str] = None,
    ):
        """Produce system metric"""
        metric_entry = MetricEntry(
            timestamp=datetime.now(timezone.utc),
            service=service,
            metric_name=metric_name,
            metric_value=metric_value,
            labels=labels or {},
        )

        return self._enqueue_message(
            "metrics",
            "llm-metrics",
            key=f"{service}:{metric_name}",
            value=metric_entry.to_dict(),
        )

    async def produce_error(self, error_data: Dict[str, Any]):
        """Produce error message"""
        error_data["timestamp"] = datetime.now(timezone.utc)

        return self._enqueue_message(
            "errors",
            "llm-errors",
            key=error_data.get("error_id", str(uuid.uuid4())),
            value=error_data,
        )

    def _serialize_message(self, message: Any) -> bytes:
        """Serialize message to JSON bytes"""
        if isinstance(message, dict):
            # Handle datetime serialization
            def default_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            return _json_bytes(message, default=default_serializer)
        else:
            return _json_bytes(message)

    @staticmethod
    async def _wait_for_cancelled_dispatcher(dispatcher: asyncio.Task) -> None:
        """Wait until a cancelled dispatcher has accounted for its in-flight batch.

        The dispatcher's cancellation path is already bounded: it gives unfinished
        deliveries ``shutdown_cancel_timeout_seconds`` and then records each one as
        delivered, ``shutdown_timeout`` or ``shutdown_abandoned`` before re-raising.
        A second timer of the same length here would race that one (it starts a
        loop turn earlier), letting shutdown return before the batch is counted.
        """
        await asyncio.wait({dispatcher})

    async def shutdown(self):
        """Shutdown producer"""
        self._accepting = False
        try:
            if self._dispatcher_task:
                dispatcher = self._dispatcher_task
                try:
                    await asyncio.wait_for(
                        self._delivery_queue.join(),
                        timeout=self.shutdown_drain_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    dispatcher.cancel()
                    await self._wait_for_cancelled_dispatcher(dispatcher)
                    self._drop_queued_messages("shutdown_timeout")
                else:
                    if not dispatcher.done():
                        dispatcher.cancel()
                    await self._wait_for_cancelled_dispatcher(dispatcher)
                finally:
                    self._dispatcher_task = None
        finally:
            if self.producer:
                stop_task = asyncio.create_task(self.producer.stop())
                done, _ = await asyncio.wait(
                    {stop_task}, timeout=self.shutdown_producer_timeout_seconds
                )
                if stop_task in done:
                    await stop_task
                    logger.info("Kafka producer shutdown complete")
                else:
                    stop_task.cancel()
                    stop_task.add_done_callback(self._consume_task_result)
                    self.consecutive_failures += 1
                    self.last_error = "Kafka producer stop timed out"
                    PIPELINE_METRICS.producer_errors.inc()
                    logger.error(self.last_error)


class ClickHouseManager:
    """Manages ClickHouse database operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[Client] = None
        self.database = config.get("database", "llm_router")
        self.tables = config.get("tables", {})

        # Connection settings
        self.connection_params = {
            "host": config.get("host", "localhost"),
            "port": config.get("port", 8123),
            "username": config.get("username", "default"),
            "password": config.get("password", ""),
            "connect_timeout": config.get("connect_timeout_seconds", 2),
            "send_receive_timeout": config.get("send_receive_timeout_seconds", 10),
            # Dashboard queries run concurrently in worker threads. ClickHouse
            # rejects overlapping queries that reuse one generated session id.
            "autogenerate_session_id": False,
        }

        # Batch insertion settings
        self.batch_size = 1000
        self.batch_timeout = 30  # seconds
        self.pending_batches = {
            "query_logs": [],
            "metrics": [],
            "model_performance": [],
            "alert_events": [],
        }
        self.last_batch_time = time.time()

        # Cached liveness ping. ``None`` means no ping has run yet, which reads
        # as healthy so a freshly connected client is not reported down.
        self.health_ping_interval_seconds = max(
            0.001, float(config.get("health_ping_interval_seconds", 30) or 30)
        )
        self.health_ping_timeout_seconds = max(
            0.001, float(config.get("health_ping_timeout_seconds", 5) or 5)
        )
        self.last_ping_ok: Optional[bool] = None
        self.last_ping_at: Optional[float] = None
        self.last_ping_error: Optional[str] = None
        self._health_ping_task: Optional[asyncio.Task] = None
        self._health_pings_running = False

    async def ping(self) -> bool:
        """Ping ClickHouse once and cache the result for health checks."""
        client = self.client
        ok = False
        error: Optional[str] = None
        if client is None:
            error = "ClickHouse client is not initialized"
        else:
            ping = getattr(client, "ping", None)
            try:
                if callable(ping):
                    result = await asyncio.wait_for(
                        self._run_blocking(ping),
                        timeout=self.health_ping_timeout_seconds,
                    )
                    ok = bool(result)
                else:
                    await asyncio.wait_for(
                        self._run_blocking(client.query, "SELECT 1"),
                        timeout=self.health_ping_timeout_seconds,
                    )
                    ok = True
                if not ok:
                    error = "ClickHouse ping returned false"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = str(exc) or type(exc).__name__
        if not ok and self.last_ping_ok is not False:
            logger.warning(f"ClickHouse health ping failed: {error}")
        self.last_ping_ok = ok
        self.last_ping_at = time.time()
        self.last_ping_error = error
        return ok

    def is_connection_healthy(self) -> bool:
        """Return the cached connection health; unpinged clients read healthy."""
        return self.client is not None and self.last_ping_ok is not False

    async def _health_ping_loop(self) -> None:
        # The flag ends the loop even if Python 3.10's wait_for swallows the
        # cancellation that races a ping finishing.
        while self._health_pings_running:
            await asyncio.sleep(self.health_ping_interval_seconds)
            if not self._health_pings_running:
                return
            await self.ping()

    def start_health_pings(self) -> None:
        """Start the periodic background ping; idempotent."""
        if self._health_ping_task is None or self._health_ping_task.done():
            self._health_pings_running = True
            self._health_ping_task = asyncio.create_task(
                self._health_ping_loop(), name="clickhouse_health_ping"
            )

    async def stop_health_pings(self) -> None:
        self._health_pings_running = False
        task = self._health_ping_task
        self._health_ping_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run_blocking(self, func: Callable[..., Any], *args, **kwargs):
        """Run a blocking ClickHouse client call outside the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _insert_dict_rows(self, table_name: str, records: List[Dict[str, Any]]):
        """Insert dictionary records with explicit column ordering."""
        if not records:
            return

        columns = list(records[0].keys())
        rows = [[record.get(column) for column in columns] for record in records]
        await self._run_blocking(
            self.client.insert,
            table_name,
            rows,
            column_names=columns,
        )

    async def initialize(self):
        """Initialize ClickHouse connection and create tables"""
        try:
            admin_client = await self._run_blocking(
                clickhouse_connect.get_client,
                **self.connection_params,
                database="default",
            )

            # Test connection
            result = await self._run_blocking(admin_client.query, "SELECT 1")
            if result.result_rows:
                logger.info("ClickHouse connection established successfully")

            # Create database if not exists
            self.client = admin_client
            await self._create_database()

            if self.database != "default":
                await self._run_blocking(admin_client.close)
                self.client = await self._run_blocking(
                    clickhouse_connect.get_client,
                    **self.connection_params,
                    database=self.database,
                )
                await self._run_blocking(self.client.query, "SELECT 1")

            # Create tables
            await self._create_tables()

            logger.info("ClickHouse manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            raise

    async def _create_database(self):
        """Create database if not exists"""
        try:
            await self._run_blocking(
                self.client.command,
                f"CREATE DATABASE IF NOT EXISTS {self.database}",
            )
            logger.info(f"Database {self.database} created/verified")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise

    async def _create_tables(self):
        """Create all required tables"""
        # Query logs table
        query_logs_sql = """
        CREATE TABLE IF NOT EXISTS {database}.query_logs (
            event_id String,
            query_id String,
            timestamp DateTime64(3),
            user_id String,
            user_tier Enum8('free' = 1, 'premium' = 2, 'enterprise' = 3),
            query_text String,
            query_type String,
            selected_model String,
            token_count_input UInt32,
            token_count_output UInt32,
            latency_ms UInt32,
            cost_usd Float64,
            status Enum8('success' = 1, 'error' = 2, 'timeout' = 3),
            error_message String,
            context_compressed Boolean,
            compression_ratio Float32,
            cached_response Boolean,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, event_id)
        TTL toDateTime(timestamp) + INTERVAL 90 DAY
        """.format(
            database=self.database
        )

        # System metrics table
        metrics_sql = """
        CREATE TABLE IF NOT EXISTS {database}.system_metrics (
            event_id String,
            timestamp DateTime64(3),
            service String,
            metric_name String,
            metric_value Float64,
            labels Map(String, String),
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, service, metric_name, event_id)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        # Model performance table
        performance_sql = """
        CREATE TABLE IF NOT EXISTS {database}.model_performance (
            event_id String,
            timestamp DateTime64(3),
            model_name String,
            provider String DEFAULT '',
            window_start_ms UInt64 DEFAULT 0,
            window_end_ms UInt64 DEFAULT 0,
            window_size_seconds UInt32 DEFAULT 60,
            requests_count UInt32,
            success_count UInt32 DEFAULT 0,
            success_rate Float32,
            avg_latency_ms Float32,
            avg_tokens_per_second Float32,
            error_count UInt32,
            total_tokens UInt32 DEFAULT 0,
            total_cost_usd Float64,
            queries_per_second Float32 DEFAULT 0,
            cache_hit_rate Float32 DEFAULT 0,
            cached_count UInt32 DEFAULT 0,
            gpu_utilization Float32,
            memory_usage_gb Float32,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, model_name, provider, event_id)
        TTL toDateTime(timestamp) + INTERVAL 60 DAY
        """.format(
            database=self.database
        )

        # User analytics table
        user_analytics_sql = """
        CREATE TABLE IF NOT EXISTS {database}.user_analytics (
            user_id String,
            date Date,
            total_queries UInt32,
            total_tokens UInt32,
            total_cost_usd Float64,
            avg_query_length Float32,
            most_used_model String,
            query_types Array(String),
            satisfaction_score Float32
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (user_id, date)
        TTL date + INTERVAL 180 DAY
        """.format(
            database=self.database
        )

        alert_events_sql = """
        CREATE TABLE IF NOT EXISTS {database}.alert_events (
            event_id String,
            timestamp DateTime64(3),
            alert_type String,
            severity String,
            description String,
            anomaly_type String,
            source_event_type String,
            request_id String,
            query_id String,
            user_id String,
            model_name String,
            provider String,
            window_start_ms UInt64,
            window_end_ms UInt64,
            payload_json String,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (source_event_type, timestamp, event_id)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        routing_policy_state_sql = """
        CREATE TABLE IF NOT EXISTS {database}.routing_policy_state_events (
            event_id String,
            timestamp DateTime64(3),
            scope_type String,
            scope_key String,
            user_id String,
            session_id String,
            user_tier String,
            hint_reason String,
            recent_request_count UInt32,
            recent_error_rate Float32,
            avg_latency_ms Float32,
            fast_lane_hit_rate Float32,
            dominant_query_type String,
            query_complexity String,
            requires_low_latency Boolean,
            requires_high_reasoning Boolean,
            route_to_fast_lane Boolean,
            burst_protection_active Boolean,
            enterprise_priority_active Boolean,
            preferred_models Array(String),
            avoid_models Array(String),
            avoid_providers Array(String),
            payload_json String,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, scope_type, scope_key, event_id)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        # One row per chat request (inference.completed): no query text, no
        # context. A replay of one request (same id and ts) collapses to its
        # latest version; distinct calls that reuse an id keep their rows.
        chat_requests_sql = """
        CREATE TABLE IF NOT EXISTS {database}.chat_requests (
            request_id String,
            ts DateTime64(3),
            completed_at DateTime64(3),
            version UInt64,
            user_id String,
            tier LowCardinality(String),
            source LowCardinality(String),
            surface LowCardinality(String),
            thread_key_hash String,
            query_type LowCardinality(String),
            model LowCardinality(String),
            provider LowCardinality(String),
            status LowCardinality(String),
            error_type LowCardinality(String),
            error_code LowCardinality(String),
            finish_reason LowCardinality(String),
            fallback_from_model LowCardinality(String),
            attempts UInt8,
            cached Boolean,
            routing_strategy LowCardinality(String),
            policy_source LowCardinality(String),
            reasoning_effort LowCardinality(String),
            latency_ms UInt32,
            routing_ms Nullable(UInt32),
            queue_wait_ms Nullable(UInt32),
            ttft_ms Nullable(UInt32),
            token_count_input UInt32,
            token_count_output UInt32,
            reasoning_tokens UInt32,
            provider_cached_input_tokens UInt32,
            cost_usd Float64,
            event_id String,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree(version)
        PARTITION BY toYYYYMMDD(ts)
        ORDER BY (toStartOfHour(ts), request_id, ts)
        TTL toDateTime(ts) + INTERVAL 90 DAY
        """.format(
            database=self.database
        )

        routing_decisions_sql = """
        CREATE TABLE IF NOT EXISTS {database}.routing_decisions (
            event_id String,
            timestamp DateTime64(3),
            request_id String,
            selected_model String,
            served_model String,
            fallback_from_model String,
            user_tier String,
            query_type String,
            routing_strategy String,
            reason_code String,
            routing_reason String,
            policy_source String,
            token_count UInt32,
            estimated_cost Float64,
            confidence Float32,
            classifier_confidence Float32,
            route_to_fast_lane Boolean,
            actual_fast_lane_hit Boolean,
            fallback_models Array(String),
            shadow_selected_model String,
            shadow_policy_source String,
            candidates_json String,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, event_id)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        # A rating of a chat answer by the user who clicked it; each click
        # is one row, the latest per user and request is their rating.
        chat_feedback_sql = """
        CREATE TABLE IF NOT EXISTS {database}.chat_feedback (
            request_id String,
            user_id String,
            rating LowCardinality(String),
            click_id String,
            source LowCardinality(String),
            clicked_at DateTime64(3),
            version UInt64,
            event_id String,
            kafka_topic String DEFAULT '',
            kafka_partition Int32 DEFAULT -1,
            kafka_offset Int64 DEFAULT -1
        ) ENGINE = ReplacingMergeTree(version)
        PARTITION BY toYYYYMM(clicked_at)
        ORDER BY (request_id, user_id, click_id)
        TTL toDateTime(clicked_at) + INTERVAL 90 DAY
        """.format(
            database=self.database
        )

        # Execute table creation; new tables go at the end.
        tables = [
            ("query_logs", query_logs_sql),
            ("system_metrics", metrics_sql),
            ("model_performance", performance_sql),
            ("user_analytics", user_analytics_sql),
            ("alert_events", alert_events_sql),
            ("routing_policy_state_events", routing_policy_state_sql),
            ("chat_requests", chat_requests_sql),
            ("chat_feedback", chat_feedback_sql),
            ("routing_decisions", routing_decisions_sql),
        ]

        for table_name, sql in tables:
            try:
                await self._run_blocking(self.client.command, sql)
                logger.info(f"Table {table_name} created/verified")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise RuntimeError(
                    f"Failed to create required ClickHouse table {table_name}"
                ) from e

        alter_statements = [
            f"ALTER TABLE {self.database}.query_logs ADD COLUMN IF NOT EXISTS event_id String DEFAULT query_id",
            f"ALTER TABLE {self.database}.query_logs ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
            f"ALTER TABLE {self.database}.query_logs ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
            f"ALTER TABLE {self.database}.query_logs ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
            f"ALTER TABLE {self.database}.system_metrics ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(service, ':', metric_name, ':', toString(timestamp))",
            f"ALTER TABLE {self.database}.system_metrics ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
            f"ALTER TABLE {self.database}.system_metrics ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
            f"ALTER TABLE {self.database}.system_metrics ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(model_name, ':', toString(timestamp))",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS provider String DEFAULT ''",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_start_ms UInt64 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_end_ms UInt64 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_size_seconds UInt32 DEFAULT 60",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS success_count UInt32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS total_tokens UInt32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS queries_per_second Float32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS cache_hit_rate Float32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS cached_count UInt32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
            f"ALTER TABLE {self.database}.alert_events ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(source_event_type, ':', alert_type, ':', request_id, ':', query_id, ':', model_name, ':', provider, ':', toString(timestamp))",
            f"ALTER TABLE {self.database}.alert_events ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
            f"ALTER TABLE {self.database}.alert_events ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
            f"ALTER TABLE {self.database}.alert_events ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
            f"ALTER TABLE {self.database}.routing_policy_state_events ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(scope_type, ':', scope_key, ':', toString(timestamp))",
            f"ALTER TABLE {self.database}.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
            f"ALTER TABLE {self.database}.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
            f"ALTER TABLE {self.database}.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
            f"ALTER TABLE {self.database}.routing_decisions ADD COLUMN IF NOT EXISTS fallback_from_model String DEFAULT ''",
        ]

        for sql in alter_statements:
            try:
                await self._run_blocking(self.client.command, sql)
            except Exception as e:
                # ADD COLUMN IF NOT EXISTS is idempotent, so a failure here means
                # every insert using the column would fail. Fail startup instead.
                table_name = sql.split()[2]
                logger.error(f"Failed schema migration on {table_name}: {e}")
                raise RuntimeError(
                    f"Failed to apply ClickHouse schema migration for {table_name}"
                ) from e

    async def insert_query_log(self, query_log: QueryLogEntry):
        """Insert single query log entry"""
        try:
            data = [query_log.to_dict()]
            await self._insert_dict_rows("query_logs", data)
            logger.debug(f"Inserted query log: {query_log.query_id}")
        except Exception as e:
            logger.error(f"Failed to insert query log: {e}")
            raise

    async def batch_insert_query_logs(self, query_logs: List[QueryLogEntry]):
        """Batch insert query log entries"""
        if not query_logs:
            return

        try:
            data = [log.to_dict() for log in query_logs]
            await self._insert_dict_rows("query_logs", data)
            logger.info(f"Batch inserted {len(query_logs)} query logs")
            PIPELINE_METRICS.records_inserted.labels(table="query_logs").inc(
                len(query_logs)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert query logs: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def insert_metric(self, metric: MetricEntry):
        """Insert single metric entry"""
        try:
            data = [metric.to_dict()]
            await self._insert_dict_rows("system_metrics", data)
            logger.debug(f"Inserted metric: {metric.service}.{metric.metric_name}")
        except Exception as e:
            logger.error(f"Failed to insert metric: {e}")
            raise

    async def batch_insert_metrics(self, metrics: List[MetricEntry]):
        """Batch insert metric entries"""
        if not metrics:
            return

        try:
            data = [metric.to_dict() for metric in metrics]
            await self._insert_dict_rows("system_metrics", data)
            logger.info(f"Batch inserted {len(metrics)} metrics")
            PIPELINE_METRICS.records_inserted.labels(table="system_metrics").inc(
                len(metrics)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert metrics: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_model_performance(
        self, metrics: List[ModelPerformanceEntry]
    ):
        """Batch insert aggregated model performance windows."""
        if not metrics:
            return

        try:
            data = [metric.to_dict() for metric in metrics]
            await self._insert_dict_rows("model_performance", data)
            logger.info(f"Batch inserted {len(metrics)} model performance windows")
            PIPELINE_METRICS.records_inserted.labels(table="model_performance").inc(
                len(metrics)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert model performance: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_alert_events(self, alerts: List[AlertEventEntry]):
        """Batch insert alert events."""
        if not alerts:
            return

        try:
            data = [alert.to_dict() for alert in alerts]
            await self._insert_dict_rows("alert_events", data)
            logger.info(f"Batch inserted {len(alerts)} alert events")
            PIPELINE_METRICS.records_inserted.labels(table="alert_events").inc(
                len(alerts)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert alert events: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_chat_requests(self, rows: List[Dict[str, Any]]):
        """Batch insert chat_requests rows (see build_chat_request_row)."""
        if not rows:
            return
        try:
            await self._insert_dict_rows("chat_requests", rows)
            PIPELINE_METRICS.records_inserted.labels(table="chat_requests").inc(
                len(rows)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert chat requests: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_chat_feedback(self, rows: List[Dict[str, Any]]):
        """Batch insert chat_feedback rows (see build_chat_feedback_row)."""
        if not rows:
            return
        try:
            await self._insert_dict_rows("chat_feedback", rows)
            PIPELINE_METRICS.records_inserted.labels(table="chat_feedback").inc(
                len(rows)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert chat feedback: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_routing_policy_state(
        self, state_events: List[RoutingPolicyStateEntry]
    ):
        """Batch insert routing.policy_state events."""
        if not state_events:
            return

        try:
            data = [state_event.to_dict() for state_event in state_events]
            await self._insert_dict_rows("routing_policy_state_events", data)
            logger.info(
                f"Batch inserted {len(state_events)} routing policy state events"
            )
            PIPELINE_METRICS.records_inserted.labels(
                table="routing_policy_state_events"
            ).inc(len(state_events))
        except Exception as e:
            logger.error(f"Failed to batch insert routing policy states: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def batch_insert_routing_decisions(
        self, decisions: List[RoutingDecisionEntry]
    ):
        """Batch insert routing.decisions events."""
        if not decisions:
            return

        try:
            data = [decision.to_dict() for decision in decisions]
            await self._insert_dict_rows("routing_decisions", data)
            logger.info(f"Batch inserted {len(decisions)} routing decisions")
            PIPELINE_METRICS.records_inserted.labels(table="routing_decisions").inc(
                len(decisions)
            )
        except Exception as e:
            logger.error(f"Failed to batch insert routing decisions: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
            raise

    async def get_query_analytics(
        self,
        user_id: Optional[str] = None,
        hours: int = 24,
        *,
        raise_errors: bool = False,
    ) -> Dict[str, Any]:
        """Get query analytics for dashboard"""
        try:
            parameters: Dict[str, Any] = {"hours": _window_hours(hours)}
            where_clauses = ["timestamp >= now() - INTERVAL {hours:UInt32} HOUR"]
            if user_id:
                parameters["user_id"] = str(user_id)
                where_clauses.append("user_id = {user_id:String}")
            where_sql = " AND ".join(where_clauses)

            analytics_query = f"""
            SELECT
                grouping(selected_model) AS model_grouped,
                grouping(query_type) AS query_type_grouped,
                selected_model,
                query_type,
                count() as total_queries,
                ifNull(sum(token_count_input + token_count_output), 0) as total_tokens,
                ifNull(sum(cost_usd), 0.0) as total_cost,
                ifNull(avg(latency_ms), 0.0) as avg_latency,
                if(count() = 0, 0.0, countIf(status = 'success') * 100.0 / count()) as success_rate
            FROM {self.database}.query_logs FINAL
            WHERE {where_sql}
            GROUP BY GROUPING SETS ((), (selected_model), (query_type))
            """

            result = await self._run_blocking(
                self.client.query, analytics_query, parameters=parameters
            )
            analytics: Dict[str, Any] = {
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0,
                "model_breakdown": {},
                "query_type_breakdown": {},
            }

            for row in result.result_rows:
                model_grouped, query_type_grouped = int(row[0]), int(row[1])
                if model_grouped and query_type_grouped:
                    analytics.update(
                        {
                            "total_queries": int(row[4] or 0),
                            "total_tokens": int(row[5] or 0),
                            "total_cost": float(row[6] or 0.0),
                            "avg_latency": float(row[7] or 0.0),
                            "success_rate": float(row[8] or 0.0),
                        }
                    )
                elif not model_grouped and query_type_grouped:
                    analytics["model_breakdown"][row[2]] = {
                        "queries": int(row[4] or 0),
                        "cost": float(row[6] or 0.0),
                    }
                elif model_grouped and not query_type_grouped:
                    analytics["query_type_breakdown"][row[3]] = int(row[4] or 0)

            return analytics

        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            if raise_errors:
                raise
            return {}

    async def get_model_performance(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get model performance metrics"""
        try:
            time_window_hours = _window_hours(hours)
            query = f"""
            SELECT 
                model_name,
                provider,
                sum(requests_count) as requests,
                sum(success_count) * 100.0 / nullIf(sum(requests_count), 0) as success_rate,
                sum(avg_latency_ms * requests_count) / nullIf(sum(requests_count), 0) as avg_latency,
                sum(avg_tokens_per_second * requests_count) / nullIf(sum(requests_count), 0) as tokens_per_second,
                sum(error_count) as errors,
                sum(total_cost_usd) as total_cost
            FROM {self.database}.model_performance FINAL
            WHERE timestamp >= now() - INTERVAL {{hours:UInt32}} HOUR
            GROUP BY model_name, provider
            ORDER BY requests DESC, model_name ASC, provider ASC
            """

            result = await self._run_blocking(
                self.client.query,
                query,
                parameters={"hours": time_window_hours},
            )

            performance_data = []
            for row in result.result_rows:
                performance_data.append(
                    {
                        "model_name": row[0],
                        "provider": row[1],
                        "requests": row[2],
                        "success_rate": row[3],
                        "avg_latency_ms": row[4],
                        "tokens_per_second": row[5],
                        "error_count": row[6],
                        "total_cost": row[7],
                    }
                )

            if performance_data:
                return performance_data
            return []

        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            if raise_errors:
                raise
            return []

    async def get_routing_guardrails(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recent routing guardrail events persisted through the alert stream."""
        try:
            time_window_hours = _window_hours(hours)
            query = f"""
            SELECT
                timestamp,
                alert_type,
                severity,
                description,
                model_name,
                provider,
                payload_json
            FROM {self.database}.alert_events FINAL
            WHERE timestamp >= now() - INTERVAL {{hours:UInt32}} HOUR
              AND source_event_type = 'routing.guardrails'
            ORDER BY timestamp DESC
            LIMIT 50
            """

            result = await self._run_blocking(
                self.client.query,
                query,
                parameters={"hours": time_window_hours},
            )
            guardrails = []
            for row in result.result_rows:
                payload = {}
                raw_payload = row[6] or "{}"
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    payload = {}

                guardrails.append(
                    {
                        "timestamp": row[0].isoformat()
                        if hasattr(row[0], "isoformat")
                        else str(row[0]),
                        "trigger_type": row[1],
                        "severity": row[2],
                        "description": row[3],
                        "model_name": row[4],
                        "provider": row[5],
                        "scope_type": payload.get("scope_type"),
                        "scope_key": payload.get("scope_key"),
                        "guardrail_action": payload.get("guardrail_action"),
                        "source": "clickhouse",
                    }
                )

            return guardrails
        except Exception as e:
            logger.error(f"Failed to get routing guardrails: {e}")
            if raise_errors:
                raise
            return []

    async def get_routing_policy_state_events(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recent persisted routing.policy_state events for audit and tuning."""
        try:
            time_window_hours = _window_hours(hours)
            query = f"""
            SELECT
                timestamp,
                scope_type,
                scope_key,
                user_id,
                session_id,
                user_tier,
                hint_reason,
                recent_request_count,
                recent_error_rate,
                avg_latency_ms,
                fast_lane_hit_rate,
                dominant_query_type,
                query_complexity,
                requires_low_latency,
                requires_high_reasoning,
                route_to_fast_lane,
                burst_protection_active,
                enterprise_priority_active,
                preferred_models,
                avoid_models,
                avoid_providers,
                payload_json
            FROM {self.database}.routing_policy_state_events FINAL
            WHERE timestamp >= now() - INTERVAL {{hours:UInt32}} HOUR
            ORDER BY timestamp DESC
            LIMIT 50
            """

            result = await self._run_blocking(
                self.client.query,
                query,
                parameters={"hours": time_window_hours},
            )
            state_events = []
            for row in result.result_rows:
                payload = {}
                raw_payload = row[21] or "{}"
                try:
                    payload = json.loads(raw_payload)
                except Exception:
                    payload = {}

                state_events.append(
                    {
                        "timestamp": row[0].isoformat()
                        if hasattr(row[0], "isoformat")
                        else str(row[0]),
                        "scope_type": row[1],
                        "scope_key": row[2],
                        "user_id": row[3],
                        "session_id": row[4],
                        "user_tier": row[5],
                        "hint_reason": row[6],
                        "recent_request_count": row[7],
                        "recent_error_rate": row[8],
                        "avg_latency_ms": row[9],
                        "fast_lane_hit_rate": row[10],
                        "dominant_query_type": row[11],
                        "query_complexity": row[12],
                        "requires_low_latency": bool(row[13]),
                        "requires_high_reasoning": bool(row[14]),
                        "route_to_fast_lane": bool(row[15]),
                        "burst_protection_active": bool(row[16]),
                        "enterprise_priority_active": bool(row[17]),
                        "preferred_models": list(row[18] or []),
                        "avoid_models": list(row[19] or []),
                        "avoid_providers": list(row[20] or []),
                        "source": "clickhouse",
                        "payload": payload,
                    }
                )

            return state_events
        except Exception as e:
            logger.error(f"Failed to get routing policy state events: {e}")
            if raise_errors:
                raise
            return []

    def shutdown(self):
        """Shutdown ClickHouse connection"""
        if self.client:
            self.client.close()
            logger.info("ClickHouse connection closed")


@dataclass
class BatchBufferState:
    """Per-topic persistence backlog and backpressure state."""

    pending_rows: List[Any] = field(default_factory=list)
    pending_offsets: Dict[TopicPartition, int] = field(default_factory=dict)
    awaiting_commit_offsets: Dict[TopicPartition, int] = field(default_factory=dict)
    inflight_rows: int = 0
    oldest_enqueued_at: Optional[float] = None
    paused: bool = False
    paused_partitions: set[TopicPartition] = field(default_factory=set)
    flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    consecutive_failures: int = 0
    retry_delay_seconds: float = 0
    next_flush_at: float = 0.0


class _BatchBackpressureRebalanceListener(ConsumerRebalanceListener):
    """Reapply topic backpressure as soon as partitions are assigned."""

    def __init__(self, manager: "KafkaConsumerManager", topic_key: str, consumer):
        self.manager = manager
        self.topic_key = topic_key
        self.consumer = consumer

    async def on_partitions_revoked(self, revoked):
        state = self.manager.batch_buffer_states.get(self.topic_key)
        if state is None:
            return
        state.paused_partitions.difference_update(set(revoked or ()))
        PIPELINE_METRICS.consumer_paused_partitions.labels(topic=self.topic_key).set(
            len(state.paused_partitions)
        )

    async def on_partitions_assigned(self, assigned):
        await self.manager._pause_assigned_partitions_if_needed(
            self.topic_key, self.consumer, set(assigned or ())
        )


class KafkaConsumerManager:
    """Manages Kafka message consumption"""

    def __init__(self, config: Dict[str, Any], clickhouse_manager: ClickHouseManager):
        self.config = config
        self.clickhouse = clickhouse_manager
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.topics = config.get("topics", {})
        self.consumers = {}
        self.topic_names_by_key: Dict[str, str] = {}
        consumer_section = config.get("consumer", {})
        dlq_section = config.get("dlq", {})

        # Consumer configuration
        self.consumer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": consumer_section.get("group_id", "llm-router-consumer"),
            # The ClickHouse sink replays the retained backlog when its group is
            # new or reset; other consumers keep the shared auto_offset_reset.
            "auto_offset_reset": consumer_section.get(
                "sink_auto_offset_reset",
                consumer_section.get("auto_offset_reset", "latest"),
            ),
            "enable_auto_commit": consumer_section.get("enable_auto_commit", False),
            "max_poll_records": consumer_section.get("max_poll_records", 500),
        }
        self.supervisor_initial_backoff_seconds = float(
            consumer_section.get("supervisor_initial_backoff_seconds", 1.0)
        )
        self.supervisor_max_backoff_seconds = float(
            consumer_section.get("supervisor_max_backoff_seconds", 30.0)
        )
        self.dlq_enabled = dlq_enabled(config)
        self.dlq_topic_suffix = str(dlq_section.get("topic_suffix", ".dlq"))
        self.dead_letter_topics = dict(config.get("dead_letter_topics", {}) or {})
        self.dlq_send_timeout_seconds = float(
            dlq_section.get("send_timeout_seconds", 10.0)
        )
        self.dlq_producer: Optional[AIOKafkaProducer] = None
        self.consumer_task_status: Dict[str, Dict[str, Any]] = {}
        self.last_consumed_at: Dict[str, float] = {}
        self.consumer_restart_counts: Dict[str, int] = {}

        # Batch processing
        batched_topics = (
            "queries",
            "metrics",
            "analytics_model_metrics_1m",
            "routing_guardrails",
            "routing_policy_state",
            "alerts",
            "inference_completed",
            "chat_feedback",
            "routing_decisions",
        )
        self.batch_buffer_states: Dict[str, BatchBufferState] = {}
        # Compatibility views retained for callers and diagnostics that inspect
        # the old public dictionaries. BatchBufferState remains authoritative.
        self.batch_processors: Dict[str, List[Any]] = {}
        self.pending_commit_offsets: Dict[str, Dict[TopicPartition, int]] = {}
        self.awaiting_commit_offsets: Dict[str, Dict[TopicPartition, int]] = {}
        self.batch_flush_locks: Dict[str, asyncio.Lock] = {}
        for topic_key in batched_topics:
            self._add_batched_topic(topic_key)
        self.observers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {
            "requests_enriched": [],
            "analytics_model_metrics_1m": [],
            "routing_guardrails": [],
            "routing_policy_state": [],
            "alerts": [],
        }
        self.batch_size = int(consumer_section.get("batch_size", 100))
        self.batch_high_watermark_rows = int(
            consumer_section.get("batch_high_watermark_rows", 5000)
        )
        self.batch_low_watermark_rows = int(
            consumer_section.get("batch_low_watermark_rows", 2500)
        )
        if not 0 <= self.batch_low_watermark_rows < self.batch_high_watermark_rows:
            raise ValueError(
                "batch_low_watermark_rows must be non-negative and below "
                "batch_high_watermark_rows"
            )
        self.max_concurrent_flushes = max(
            1, int(consumer_section.get("max_concurrent_flushes", 4))
        )
        self.flush_retry_initial_backoff_seconds = max(
            0.0,
            float(consumer_section.get("flush_retry_initial_backoff_seconds", 5.0)),
        )
        self.flush_retry_max_backoff_seconds = max(
            self.flush_retry_initial_backoff_seconds,
            float(consumer_section.get("flush_retry_max_backoff_seconds", 120.0)),
        )
        self.last_batch_time = time.time()
        self.running = False
        self._rebalance_listeners: Dict[str, Any] = {}

    def _add_batched_topic(self, topic_key: str) -> BatchBufferState:
        """Create the buffer state and compatibility views for a sink topic."""
        state = BatchBufferState()
        self.batch_buffer_states[topic_key] = state
        self.batch_processors[topic_key] = state.pending_rows
        self.pending_commit_offsets[topic_key] = state.pending_offsets
        self.awaiting_commit_offsets[topic_key] = state.awaiting_commit_offsets
        self.batch_flush_locks[topic_key] = state.flush_lock
        return state

    def _install_rebalance_listener(
        self, topic_key: str, topic_name: str, consumer: AIOKafkaConsumer
    ) -> None:
        listener = _BatchBackpressureRebalanceListener(self, topic_key, consumer)
        consumer.subscribe(topics=[topic_name], listener=listener)
        self._rebalance_listeners[topic_key] = listener

    async def _create_consumer(
        self, topic_key: str, topic_name: str
    ) -> AIOKafkaConsumer:
        """Subscribe with the backpressure listener, then start.

        Re-subscribing a running consumer races its fetcher, so the listener
        is installed before start() on first start and on every restart.
        """
        consumer = AIOKafkaConsumer(**self.consumer_config)
        self._install_rebalance_listener(topic_key, topic_name, consumer)
        await consumer.start()
        return consumer

    async def initialize(self):
        """Initialize Kafka consumers"""
        try:
            if self.dlq_enabled:
                self.dlq_producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=self._serialize_message,
                    key_serializer=lambda x: x.encode("utf-8") if x else None,
                    acks="all",
                    enable_idempotence=True,
                )
                await self.dlq_producer.start()

            # Create consumer for each topic
            for topic_key, topic_name in self.topics.items():
                if topic_key in [
                    "queries",
                    "responses",
                    "metrics",
                    "errors",
                    "requests_enriched",
                    "analytics_model_metrics_1m",
                    "routing_guardrails",
                    "routing_policy_state",
                    "alerts",
                    "inference_completed",
                    "chat_feedback",
                    "routing_decisions",
                ]:
                    consumer = await self._create_consumer(topic_key, topic_name)
                    self.consumers[topic_key] = consumer
                    self.topic_names_by_key[topic_key] = topic_name
                    self.consumer_task_status[topic_key] = {
                        "running": False,
                        "last_error": None,
                        "last_started_at": None,
                        "last_stopped_at": None,
                    }
                    logger.info(f"Consumer for topic {topic_name} initialized")

            logger.info(f"Kafka consumers initialized for {len(self.consumers)} topics")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumers: {e}")
            raise

    async def start_consuming(self):
        """Start consuming messages from all topics"""
        self.running = True
        async with asyncio.TaskGroup() as task_group:
            for topic_key, consumer in self.consumers.items():
                task_group.create_task(
                    self._supervise_consumer(topic_key, consumer),
                    name=f"kafka_consumer_{topic_key}",
                )
            task_group.create_task(self._batch_processor(), name="kafka_batch_flush")

    async def _supervise_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        """Restart a topic consumer loop when it exits unexpectedly."""
        backoff_seconds = self.supervisor_initial_backoff_seconds
        while self.running:
            self.consumer_task_status.setdefault(topic_key, {})
            self.consumer_task_status[topic_key].update(
                {
                    "running": True,
                    "last_error": None,
                    "last_started_at": time.time(),
                }
            )
            try:
                await self._run_consumer(topic_key, consumer)
                if not self.running:
                    break
                raise RuntimeError("consumer loop exited unexpectedly")
            except asyncio.CancelledError:
                self.consumer_task_status[topic_key].update(
                    {"running": False, "last_stopped_at": time.time()}
                )
                raise
            except Exception as e:
                self.consumer_task_status[topic_key].update(
                    {
                        "running": False,
                        "last_error": str(e),
                        "last_stopped_at": time.time(),
                    }
                )
                self.consumer_restart_counts[topic_key] = (
                    self.consumer_restart_counts.get(topic_key, 0) + 1
                )
                PIPELINE_METRICS.consumer_restarts.labels(topic=topic_key).inc()
                logger.error(
                    "Kafka consumer for %s stopped; restarting in %.1fs: %s",
                    topic_key,
                    backoff_seconds,
                    e,
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    max(backoff_seconds * 2, self.supervisor_initial_backoff_seconds),
                    self.supervisor_max_backoff_seconds,
                )
                consumer = await self._restart_consumer(topic_key)

    async def _restart_consumer(self, topic_key: str) -> AIOKafkaConsumer:
        """Recreate a Kafka consumer after a task-level failure."""
        old_consumer = self.consumers.get(topic_key)
        if old_consumer is not None:
            try:
                await old_consumer.stop()
            except Exception as e:
                logger.warning(f"Failed to stop unhealthy {topic_key} consumer: {e}")

        topic_name = self.topic_names_by_key.get(topic_key) or self.topics[topic_key]
        consumer = await self._create_consumer(topic_key, topic_name)
        self.consumers[topic_key] = consumer
        await self._refresh_batch_backpressure(topic_key)
        return consumer

    async def _run_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        """Dispatch to the topic-specific consumer implementation."""
        handlers = {
            "queries": self._consume_queries,
            "responses": self._consume_responses,
            "metrics": self._consume_metrics,
            "errors": self._consume_errors,
            "requests_enriched": self._consume_requests_enriched,
            "analytics_model_metrics_1m": self._consume_analytics_model_metrics,
            "routing_guardrails": self._consume_routing_guardrails,
            "routing_policy_state": self._consume_routing_policy_state,
            "alerts": self._consume_alerts,
            "inference_completed": self._consume_inference_completed,
            "chat_feedback": self._consume_chat_feedback,
            "routing_decisions": self._consume_routing_decisions,
        }
        handler = handlers.get(topic_key)
        if handler is None:
            raise RuntimeError(f"No Kafka consumer handler for topic key {topic_key}")
        await handler(consumer)

    async def _consume_queries(self, consumer: AIOKafkaConsumer):
        """Consume query log messages"""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    query_log = QueryLogEntry(
                        query_id=data["query_id"],
                        timestamp=_parse_datetime(data["timestamp"], strict=True),
                        user_id=data["user_id"],
                        user_tier=data["user_tier"],
                        query_text=data["query_text"],
                        query_type=data["query_type"],
                        selected_model=data["selected_model"],
                        token_count_input=data["token_count_input"],
                        token_count_output=data["token_count_output"],
                        latency_ms=data["latency_ms"],
                        cost_usd=data["cost_usd"],
                        status=data["status"],
                        error_message=data.get("error_message", ""),
                        context_compressed=data.get("context_compressed", False),
                        compression_ratio=data.get("compression_ratio", 0.0),
                        cached_response=data.get("cached_response", False),
                        **self._payload_event_metadata(data, message),
                    )

                    # Add to batch
                    await self._append_batched_row("queries", query_log, message)
                    self._record_successful_consume("queries", consumer, message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="queries").inc()

                except Exception as e:
                    await self._handle_processing_failure(
                        "queries",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )

        except Exception as e:
            logger.error(f"Query consumer error: {e}")
            raise

    async def _consume_responses(self, consumer: AIOKafkaConsumer):
        """Consume response log messages"""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    # Process response data (for future analytics)
                    logger.debug(f"Processed response: {data.get('query_id')}")
                    await self._commit_processed_message("responses", message)
                    self._record_successful_consume("responses", consumer, message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="responses").inc()

                except Exception as e:
                    await self._handle_processing_failure(
                        "responses",
                        message,
                        e,
                        stage="consume",
                        batched=False,
                    )

        except Exception as e:
            logger.error(f"Response consumer error: {e}")
            raise

    async def _consume_metrics(self, consumer: AIOKafkaConsumer):
        """Consume system metrics messages"""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    metric_entry = MetricEntry(
                        timestamp=_parse_datetime(data["timestamp"], strict=True),
                        service=data["service"],
                        metric_name=data["metric_name"],
                        metric_value=data["metric_value"],
                        labels=data.get("labels", {}),
                        **self._kafka_event_metadata(message),
                    )

                    # Add to batch
                    await self._append_batched_row("metrics", metric_entry, message)
                    self._record_successful_consume("metrics", consumer, message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="metrics").inc()

                except Exception as e:
                    await self._handle_processing_failure(
                        "metrics",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )

        except Exception as e:
            logger.error(f"Metrics consumer error: {e}")
            raise

    async def _consume_errors(self, consumer: AIOKafkaConsumer):
        """Consume error messages"""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    logger.error(f"System error logged: {data}")
                    await self._commit_processed_message("errors", message)
                    self._record_successful_consume("errors", consumer, message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="errors").inc()

                except Exception as e:
                    await self._handle_processing_failure(
                        "errors",
                        message,
                        e,
                        stage="consume",
                        batched=False,
                    )

        except Exception as e:
            logger.error(f"Error consumer error: {e}")
            raise

    def register_observer(
        self, topic_key: str, callback: Callable[[Dict[str, Any]], Any]
    ):
        """Register an observer for consumed stream events."""
        self.observers.setdefault(topic_key, []).append(callback)

    async def _notify_observers(self, topic_key: str, payload: Dict[str, Any]):
        """Forward a consumed event to any registered in-process observers."""
        for callback in self.observers.get(topic_key, []):
            try:
                result = callback(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.warning(f"Observer for {topic_key} failed: {e}")

    def _decode_message_value(self, message: Any) -> Dict[str, Any]:
        """Decode a Kafka message payload without letting poison pills kill a task."""
        value = getattr(message, "value", None)
        if isinstance(value, dict):
            return value
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        raise ValueError(f"Unsupported Kafka message payload type: {type(value)}")

    def _kafka_event_metadata(self, message: Any) -> Dict[str, Any]:
        """Build stable idempotency metadata from Kafka coordinates."""
        topic = str(getattr(message, "topic", "") or "")
        partition = int(getattr(message, "partition", -1))
        offset = int(getattr(message, "offset", -1))
        event_id = f"{topic}:{partition}:{offset}" if topic and offset >= 0 else ""
        return {
            "event_id": event_id,
            "kafka_topic": topic,
            "kafka_partition": partition,
            "kafka_offset": offset,
        }

    def _payload_event_metadata(
        self, data: Dict[str, Any], message: Any
    ) -> Dict[str, Any]:
        """Prefer the producer's event identity so replays dedupe in ClickHouse.

        The Kafka coordinate is only a fallback: a resent or replayed message
        lands at a new offset, which would otherwise create a second row.
        """
        metadata = self._kafka_event_metadata(message)
        payload_event_id = str(data.get("event_id") or "").strip()
        if payload_event_id:
            metadata["event_id"] = payload_event_id
        return metadata

    def _dead_letter_topic(self, topic_key: str, message: Any) -> str:
        """Resolve the DLQ topic for a failed source message."""
        source_topic = (
            getattr(message, "topic", None)
            or self.topic_names_by_key.get(topic_key)
            or topic_key
        )
        return dead_letter_topic_name(
            {
                **self.config,
                "dead_letter_topics": self.dead_letter_topics,
                "dlq": {"topic_suffix": self.dlq_topic_suffix},
            },
            topic_key,
            source_topic,
        )

    async def _publish_dead_letter(
        self,
        topic_key: str,
        message: Any,
        error: Exception,
        *,
        stage: str,
    ) -> bool:
        """Publish a poison message to DLQ before committing past it."""
        if not self.dlq_enabled or self.dlq_producer is None:
            logger.error(
                "DLQ disabled or unavailable for failed %s message; offset retained",
                topic_key,
            )
            return False

        metadata = self._kafka_event_metadata(message)
        payload = {
            "event_type": "dead_letter",
            "event_version": EVENT_SCHEMA_VERSION,
            "emitted_at": _isoformat_utc(),
            "source_topic_key": topic_key,
            "source_topic": metadata["kafka_topic"],
            "source_partition": metadata["kafka_partition"],
            "source_offset": metadata["kafka_offset"],
            "source_key": _decode_kafka_key(getattr(message, "key", None)),
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": redact_text(str(error))[:2000],
            "payload": _dead_letter_payload(getattr(message, "value", None), error),
        }
        topic = self._dead_letter_topic(topic_key, message)
        key = metadata["event_id"] or str(uuid.uuid4())
        send_and_wait = getattr(self.dlq_producer, "send_and_wait", None)
        if callable(send_and_wait):
            await asyncio.wait_for(
                send_and_wait(topic=topic, key=key, value=payload),
                timeout=self.dlq_send_timeout_seconds,
            )
        else:
            await asyncio.wait_for(
                self.dlq_producer.send(topic=topic, key=key, value=payload),
                timeout=self.dlq_send_timeout_seconds,
            )
        PIPELINE_METRICS.dead_letter_messages.labels(topic=topic_key).inc()
        return True

    async def _handle_processing_failure(
        self,
        topic_key: str,
        message: Any,
        error: Exception,
        *,
        stage: str,
        batched: bool,
    ):
        """DLQ and commit a bad message after preserving earlier valid records."""
        logger.error(f"Failed to process {topic_key} message: {error}")
        PIPELINE_METRICS.consumer_errors.inc()

        if batched:
            # Only this topic's earlier rows must be durable before committing
            # past the poison message; other topics' failures are not ours.
            try:
                await self.flush_topic(topic_key)
            except Exception as flush_error:
                if _classify_insert_error(flush_error) != "schema":
                    raise
                # The topic is backing off on schema drift. Raising would
                # restart the consumer and re-read the buffered rows, so DLQ
                # the poison message now and commit its offset only together
                # with those rows, after they are written.
                if await self._publish_dead_letter(
                    topic_key, message, error, stage=stage
                ):
                    self._track_commit_offset(topic_key, message)
                return

        if await self._publish_dead_letter(topic_key, message, error, stage=stage):
            await self._commit_processed_message(topic_key, message)

    def _record_successful_consume(
        self, topic_key: str, consumer: AIOKafkaConsumer, message: Any
    ):
        """Update consumer observability after a message is handled."""
        now = time.time()
        self.last_consumed_at[topic_key] = now
        PIPELINE_METRICS.last_consumed_timestamp.labels(topic=topic_key).set(now)

        topic = getattr(message, "topic", None)
        partition = getattr(message, "partition", None)
        offset = getattr(message, "offset", None)
        if topic is None or partition is None or offset is None:
            return

        highwater = getattr(consumer, "highwater", None)
        if not callable(highwater):
            return

        try:
            latest_offset = highwater(TopicPartition(topic, partition))
            if latest_offset is None:
                return
            lag = max(0, int(latest_offset) - int(offset) - 1)
            PIPELINE_METRICS.consumer_lag.labels(
                topic=topic,
                partition=str(partition),
            ).set(lag)
        except Exception as e:
            logger.debug(f"Failed to update Kafka consumer lag: {e}")

    def is_healthy(self) -> bool:
        """Return whether all configured consumer supervisors are running."""
        if not self.consumers:
            return False
        for status in self.consumer_task_status.values():
            if status.get("last_error"):
                return False
        return True

    def get_health_status(self) -> Dict[str, Any]:
        """Return consumer task and DLQ health details."""
        return {
            "healthy": self.is_healthy(),
            "topics": dict(self.consumer_task_status),
            "last_consumed_at": dict(self.last_consumed_at),
            "restart_counts": dict(self.consumer_restart_counts),
            "dlq_enabled": self.dlq_enabled,
            "dlq_producer_healthy": (not self.dlq_enabled)
            or self.dlq_producer is not None,
        }

    def _build_model_performance_entry(
        self,
        data: Dict[str, Any],
        kafka_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelPerformanceEntry:
        """Convert a Flink analytics window event to a ClickHouse row."""
        window_start_ms = int(data.get("window_start_ms", 0) or 0)
        window_end_ms = int(data.get("window_end_ms", 0) or 0)
        model_name = str(data.get("model_name", "unknown"))
        provider = str(data.get("provider", ""))
        metadata = dict(kafka_metadata or {})
        if window_start_ms:
            # A replayed window is re-emitted with a new emitted_at and a new
            # offset; keying the row on the window keeps FINAL at one row.
            event_time = datetime.fromtimestamp(window_start_ms / 1000, tz=timezone.utc)
            metadata[
                "event_id"
            ] = f"{model_name}:{provider}:{window_start_ms}:{window_end_ms}"
        else:
            event_time = _parse_datetime(
                data.get("emitted_at"),
                default=datetime.fromtimestamp(window_end_ms / 1000, tz=timezone.utc)
                if window_end_ms
                else datetime.now(timezone.utc),
            )

        return ModelPerformanceEntry(
            timestamp=event_time,
            model_name=model_name,
            provider=provider,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            window_size_seconds=int(data.get("window_size_seconds", 60) or 60),
            requests_count=int(data.get("request_count", 0) or 0),
            success_count=int(data.get("success_count", 0) or 0),
            success_rate=float(data.get("success_rate", 0.0) or 0.0),
            avg_latency_ms=float(data.get("avg_latency_ms", 0.0) or 0.0),
            avg_tokens_per_second=float(data.get("avg_tokens_per_second", 0.0) or 0.0),
            error_count=int(data.get("error_count", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            queries_per_second=float(data.get("queries_per_second", 0.0) or 0.0),
            cache_hit_rate=float(data.get("cache_hit_rate", 0.0) or 0.0),
            cached_count=int(data.get("cached_count", 0) or 0),
            **metadata,
        )

    def _build_alert_event_entry(
        self,
        data: Dict[str, Any],
        kafka_metadata: Optional[Dict[str, Any]] = None,
    ) -> AlertEventEntry:
        """Convert an alert event payload to a ClickHouse row."""
        event_type = str(data.get("event_type", "") or "")
        alert_type = str(
            data.get("alert_type") or data.get("trigger_type") or "unknown"
        )
        anomaly_type = str(data.get("anomaly_type") or data.get("trigger_type", ""))
        source_event_type = str(data.get("original_event_type") or event_type)
        model_name = str(data.get("model_name", "") or "")
        provider = str(data.get("provider", "") or "")
        window_start_ms = int(data.get("window_start_ms", 0) or 0)
        window_end_ms = int(data.get("window_end_ms", 0) or 0)
        timestamp = _parse_datetime(data.get("emitted_at"))
        metadata = dict(kafka_metadata or {})
        if window_start_ms and window_end_ms:
            # Window alerts and guardrails are re-emitted with a new emitted_at
            # and offset when Flink restores; key them on what they describe.
            timestamp = datetime.fromtimestamp(window_end_ms / 1000, tz=timezone.utc)
            metadata["event_id"] = ":".join(
                str(part)
                for part in (
                    event_type,
                    data.get("scope_type", "") or "",
                    data.get("scope_key", "") or "",
                    alert_type,
                    anomaly_type,
                    data.get("guardrail_action", "") or "",
                    model_name,
                    provider,
                    window_start_ms,
                    window_end_ms,
                )
            )
        return AlertEventEntry(
            timestamp=timestamp,
            alert_type=alert_type,
            severity=str(data.get("severity", "warning")),
            description=str(data.get("description", "")),
            anomaly_type=anomaly_type,
            source_event_type=source_event_type,
            request_id=str(data.get("request_id", "") or ""),
            query_id=str(data.get("query_id", "") or ""),
            user_id=str(data.get("user_id", "") or ""),
            model_name=model_name,
            provider=provider,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            payload_json=_json_text(_alert_payload(data), default=str),
            **metadata,
        )

    def _build_routing_policy_state_entry(
        self,
        data: Dict[str, Any],
        kafka_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingPolicyStateEntry:
        """Convert a routing.policy_state payload to a ClickHouse row."""
        return RoutingPolicyStateEntry(
            timestamp=_parse_datetime(data.get("emitted_at")),
            scope_type=str(data.get("scope_type", "unknown") or "unknown"),
            scope_key=str(data.get("scope_key", "") or ""),
            user_id=str(data.get("user_id", "") or ""),
            session_id=str(data.get("session_id", "") or ""),
            user_tier=str(data.get("user_tier", "") or ""),
            hint_reason=str(data.get("hint_reason", "") or ""),
            recent_request_count=int(data.get("recent_request_count", 0) or 0),
            recent_error_rate=float(data.get("recent_error_rate", 0.0) or 0.0),
            avg_latency_ms=float(data.get("avg_latency_ms", 0.0) or 0.0),
            fast_lane_hit_rate=float(data.get("fast_lane_hit_rate", 0.0) or 0.0),
            dominant_query_type=str(data.get("dominant_query_type", "") or ""),
            query_complexity=str(data.get("query_complexity", "") or ""),
            requires_low_latency=bool(data.get("requires_low_latency", False)),
            requires_high_reasoning=bool(data.get("requires_high_reasoning", False)),
            route_to_fast_lane=bool(data.get("route_to_fast_lane", False)),
            burst_protection_active=bool(data.get("burst_protection_active", False)),
            enterprise_priority_active=bool(
                data.get("enterprise_priority_active", False)
            ),
            preferred_models=list(data.get("preferred_models", []) or []),
            avoid_models=list(data.get("avoid_models", []) or []),
            avoid_providers=list(data.get("avoid_providers", []) or []),
            payload_json=_json_text(data, default=str),
            **(kafka_metadata or {}),
        )

    async def _consume_requests_enriched(self, consumer: AIOKafkaConsumer):
        """Consume request-side Flink enrichment events for observers."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    await self._notify_observers("requests_enriched", data)
                    await self._commit_processed_message("requests_enriched", message)
                    self._record_successful_consume(
                        "requests_enriched", consumer, message
                    )
                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="requests_enriched"
                    ).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "requests_enriched",
                        message,
                        e,
                        stage="consume",
                        batched=False,
                    )
        except Exception as e:
            logger.error(f"requests.enriched consumer error: {e}")
            raise

    async def _consume_analytics_model_metrics(self, consumer: AIOKafkaConsumer):
        """Consume Flink model metrics window events."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    metric_entry = self._build_model_performance_entry(
                        data,
                        kafka_metadata=self._kafka_event_metadata(message),
                    )
                    await self._append_batched_row(
                        "analytics_model_metrics_1m", metric_entry, message
                    )
                    await self._notify_observers("analytics_model_metrics_1m", data)
                    self._record_successful_consume(
                        "analytics_model_metrics_1m", consumer, message
                    )

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="analytics_model_metrics_1m"
                    ).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "analytics_model_metrics_1m",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"Analytics model metrics consumer error: {e}")
            raise

    async def _consume_batched_rows(
        self,
        topic_key: str,
        consumer: AIOKafkaConsumer,
        build_row: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    ):
        """Batch each event of topic_key as the row build_row makes of it."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    row = build_row(data, self._kafka_event_metadata(message))
                    await self._append_batched_row(topic_key, row, message)
                    self._record_successful_consume(topic_key, consumer, message)
                    PIPELINE_METRICS.messages_consumed.labels(topic=topic_key).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        topic_key,
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"{topic_key} consumer error: {e}")
            raise

    async def _consume_inference_completed(self, consumer: AIOKafkaConsumer):
        """Persist inference.completed events as chat_requests rows."""
        await self._consume_batched_rows(
            "inference_completed", consumer, build_chat_request_row
        )

    async def _consume_chat_feedback(self, consumer: AIOKafkaConsumer):
        """Persist chat.feedback events as chat_feedback rows."""
        await self._consume_batched_rows(
            "chat_feedback", consumer, build_chat_feedback_row
        )

    async def _consume_alerts(self, consumer: AIOKafkaConsumer):
        """Consume Flink and platform alert events."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    alert_entry = self._build_alert_event_entry(
                        data,
                        kafka_metadata=self._kafka_event_metadata(message),
                    )
                    await self._append_batched_row("alerts", alert_entry, message)
                    await self._notify_observers("alerts", data)
                    self._record_successful_consume("alerts", consumer, message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="alerts").inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "alerts",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"Alert consumer error: {e}")
            raise

    async def _consume_routing_guardrails(self, consumer: AIOKafkaConsumer):
        """Consume routing guardrail events for persistence and observers."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    guardrail_entry = self._build_alert_event_entry(
                        data,
                        kafka_metadata=self._kafka_event_metadata(message),
                    )
                    await self._append_batched_row(
                        "routing_guardrails", guardrail_entry, message
                    )
                    await self._notify_observers("routing_guardrails", data)
                    self._record_successful_consume(
                        "routing_guardrails", consumer, message
                    )

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="routing_guardrails"
                    ).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "routing_guardrails",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"routing.guardrails consumer error: {e}")
            raise

    async def _consume_routing_policy_state(self, consumer: AIOKafkaConsumer):
        """Consume routing.policy_state events for persistence and observers."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    state_entry = self._build_routing_policy_state_entry(
                        data,
                        kafka_metadata=self._kafka_event_metadata(message),
                    )
                    await self._append_batched_row(
                        "routing_policy_state", state_entry, message
                    )
                    await self._notify_observers("routing_policy_state", data)
                    self._record_successful_consume(
                        "routing_policy_state", consumer, message
                    )

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="routing_policy_state"
                    ).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "routing_policy_state",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"routing.policy_state consumer error: {e}")
            raise

    def _build_routing_decision_entry(
        self,
        data: Dict[str, Any],
        kafka_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecisionEntry:
        """Convert a routing.decisions payload to a ClickHouse row."""
        request_id = str(data["request_id"])
        metadata = dict(kafka_metadata or {})
        # Keyed by request so a replay replaces the row instead of adding one.
        metadata["event_id"] = request_id
        return RoutingDecisionEntry(
            timestamp=_parse_datetime(data.get("decided_at") or data.get("emitted_at")),
            request_id=request_id,
            selected_model=str(data.get("selected_model") or ""),
            served_model=str(data.get("served_model") or ""),
            fallback_from_model=str(data.get("fallback_from_model") or ""),
            user_tier=str(data.get("user_tier") or ""),
            query_type=str(data.get("query_type") or ""),
            routing_strategy=str(data.get("routing_strategy") or ""),
            reason_code=str(data.get("reason_code") or ""),
            routing_reason=str(data.get("routing_reason") or ""),
            policy_source=str(data.get("policy_source") or ""),
            token_count=int(data.get("token_count") or 0),
            estimated_cost=float(data.get("estimated_cost") or 0.0),
            confidence=float(data.get("confidence") or 0.0),
            classifier_confidence=float(data.get("classifier_confidence") or 0.0),
            route_to_fast_lane=bool(data.get("route_to_fast_lane", False)),
            actual_fast_lane_hit=bool(data.get("actual_fast_lane_hit", False)),
            fallback_models=[
                str(model) for model in list(data.get("fallback_models") or [])
            ],
            shadow_selected_model=str(data.get("shadow_selected_model") or ""),
            shadow_policy_source=str(data.get("shadow_policy_source") or ""),
            candidates_json=_json_text(
                [
                    _routing_candidate_payload(candidate)
                    for candidate in list(data.get("candidates") or [])
                ]
            ),
            **metadata,
        )

    async def _consume_routing_decisions(self, consumer: AIOKafkaConsumer):
        """Consume routing.decisions events into ClickHouse."""
        try:
            async for message in consumer:
                try:
                    data = self._decode_message_value(message)
                    entry = self._build_routing_decision_entry(
                        data,
                        kafka_metadata=self._kafka_event_metadata(message),
                    )
                    await self._append_batched_row("routing_decisions", entry, message)
                    self._record_successful_consume(
                        "routing_decisions", consumer, message
                    )

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="routing_decisions"
                    ).inc()
                except Exception as e:
                    await self._handle_processing_failure(
                        "routing_decisions",
                        message,
                        e,
                        stage="consume",
                        batched=True,
                    )
        except Exception as e:
            logger.error(f"routing.decisions consumer error: {e}")
            raise

    async def _batch_processor(self):
        """Process batches periodically"""
        while self.running:
            try:
                current_time = time.time()
                reached_batch_size = any(
                    len(batch) >= self.batch_size
                    for batch in self.batch_processors.values()
                )
                batch_timeout_elapsed = current_time - self.last_batch_time > 30

                if reached_batch_size or batch_timeout_elapsed:
                    await self.flush_pending_batches(respect_backoff=True)

                # Update last batch time
                if batch_timeout_elapsed:
                    self.last_batch_time = current_time

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(10)

        await self.flush_pending_batches()

    def _replace_pending_rows(self, topic_key: str, rows: List[Any]):
        state = self.batch_buffer_states[topic_key]
        state.pending_rows = rows
        self.batch_processors[topic_key] = rows

    def _replace_pending_offsets(
        self, topic_key: str, offsets: Dict[TopicPartition, int]
    ):
        state = self.batch_buffer_states[topic_key]
        state.pending_offsets = offsets
        self.pending_commit_offsets[topic_key] = offsets

    def _replace_awaiting_offsets(
        self, topic_key: str, offsets: Dict[TopicPartition, int]
    ):
        state = self.batch_buffer_states[topic_key]
        state.awaiting_commit_offsets = offsets
        self.awaiting_commit_offsets[topic_key] = offsets

    def _buffered_row_count(self, topic_key: str) -> int:
        state = self.batch_buffer_states[topic_key]
        return len(state.pending_rows) + state.inflight_rows

    async def _pause_assigned_partitions_if_needed(
        self, topic_key: str, consumer: Any, partitions: set[TopicPartition]
    ) -> None:
        state = self.batch_buffer_states.get(topic_key)
        if state is None or not state.paused or not partitions:
            return
        pause_result = consumer.pause(*partitions)
        if inspect.isawaitable(pause_result):
            await pause_result
        state.paused_partitions.update(partitions)
        PIPELINE_METRICS.consumer_paused_partitions.labels(topic=topic_key).set(
            len(state.paused_partitions)
        )

    async def _refresh_batch_backpressure(self, topic_key: str):
        """Pause or resume the topic consumer based on its durable-write backlog."""
        state = self.batch_buffer_states[topic_key]
        buffered_rows = self._buffered_row_count(topic_key)
        PIPELINE_METRICS.consumer_buffer_rows.labels(topic=topic_key).set(buffered_rows)
        consumer = self.consumers.get(topic_key)
        if consumer is None:
            return

        assignment = getattr(consumer, "assignment", None)
        if assignment is None:
            return
        assigned = assignment()
        if inspect.isawaitable(assigned):
            assigned = await assigned
        partitions = set(assigned or ())
        if buffered_rows >= self.batch_high_watermark_rows or (
            state.paused and buffered_rows > self.batch_low_watermark_rows
        ):
            newly_assigned = partitions - state.paused_partitions
            if newly_assigned:
                pause_result = consumer.pause(*newly_assigned)
                if inspect.isawaitable(pause_result):
                    await pause_result
                state.paused_partitions.update(newly_assigned)
            was_paused = state.paused
            state.paused = True
            PIPELINE_METRICS.consumer_paused_partitions.labels(topic=topic_key).set(
                len(state.paused_partitions)
            )
            if not was_paused:
                PIPELINE_METRICS.consumer_backpressure_transitions.labels(
                    topic=topic_key, action="pause"
                ).inc()
        elif state.paused and buffered_rows <= self.batch_low_watermark_rows:
            if partitions:
                resume_result = consumer.resume(*partitions)
                if inspect.isawaitable(resume_result):
                    await resume_result
            state.paused = False
            state.paused_partitions.clear()
            PIPELINE_METRICS.consumer_paused_partitions.labels(topic=topic_key).set(0)
            PIPELINE_METRICS.consumer_backpressure_transitions.labels(
                topic=topic_key, action="resume"
            ).inc()

    async def _append_batched_row(self, topic_key: str, row: Any, message: Any):
        """Append one row and apply lossless consumer backpressure."""
        state = self.batch_buffer_states[topic_key]
        state.pending_rows.append(row)
        if state.oldest_enqueued_at is None:
            state.oldest_enqueued_at = time.monotonic()
        self._track_commit_offset(topic_key, message)
        await self._refresh_batch_backpressure(topic_key)

    def _track_commit_offset(self, topic_key: str, message: Any):
        """Track the latest offset eligible for commit after durable write."""
        topic = getattr(message, "topic", None)
        partition = getattr(message, "partition", None)
        offset = getattr(message, "offset", None)
        if topic is None or partition is None or offset is None:
            return

        topic_partition = TopicPartition(topic, partition)
        self.pending_commit_offsets.setdefault(topic_key, {})[topic_partition] = (
            int(offset) + 1
        )

    async def _commit_offsets(self, topic_key: str, offsets: Dict[TopicPartition, int]):
        """Commit explicit offsets for a topic when manual commits are enabled."""
        if self.consumer_config.get("enable_auto_commit", False) or not offsets:
            return

        consumer = self.consumers.get(topic_key)
        if consumer is None:
            return

        await consumer.commit(dict(offsets))

    async def _commit_processed_message(self, topic_key: str, message: Any):
        """Commit a non-batched message immediately after successful processing."""
        self._track_commit_offset(topic_key, message)
        offsets = dict(self.pending_commit_offsets.get(topic_key, {}))
        self.pending_commit_offsets[topic_key].clear()
        await self._commit_offsets(topic_key, offsets)

    async def _flush_batch_topic(
        self,
        topic_key: str,
        insert_method: Callable[[List[Any]], Any],
    ):
        """Flush one batched topic and commit offsets after durable persistence."""
        async with self.batch_flush_locks[topic_key]:
            if self.awaiting_commit_offsets.get(topic_key):
                await self._commit_offsets(
                    topic_key, self.awaiting_commit_offsets[topic_key]
                )
                self._replace_awaiting_offsets(topic_key, {})

            batch = self.batch_processors.get(topic_key, [])
            offsets = self.pending_commit_offsets.get(topic_key, {})
            if not batch:
                await self._refresh_batch_backpressure(topic_key)
                return

            # Swap both buffers before the first I/O await. Consumers can continue
            # appending to the new buffers while this detached batch is persisted.
            state = self.batch_buffer_states[topic_key]
            self._replace_pending_rows(topic_key, [])
            self._replace_pending_offsets(topic_key, {})
            state.inflight_rows = len(batch)
            state.oldest_enqueued_at = None
            await self._refresh_batch_backpressure(topic_key)

            try:
                await insert_method(batch)
            except Exception as error:
                kind = _classify_insert_error(error)
                unresolved: List[Any] = batch
                failure: Optional[Exception] = error
                if kind == "row":
                    unresolved, failure, kind = await self._resolve_rejected_batch(
                        topic_key, batch, insert_method, error
                    )
                if unresolved:
                    self._restore_detached_batch(topic_key, unresolved, offsets)
                    self._record_flush_failure(topic_key, kind)
                    await self._refresh_batch_backpressure(topic_key)
                    raise failure or error

            state.inflight_rows = 0
            self._record_flush_success(topic_key)
            self._replace_awaiting_offsets(topic_key, offsets)
            await self._refresh_batch_backpressure(topic_key)
            if self.awaiting_commit_offsets[topic_key]:
                await self._commit_offsets(
                    topic_key, self.awaiting_commit_offsets[topic_key]
                )
                self._replace_awaiting_offsets(topic_key, {})

    def _restore_detached_batch(
        self,
        topic_key: str,
        rows: List[Any],
        offsets: Dict[TopicPartition, int],
    ) -> None:
        """Put unwritten rows back ahead of new arrivals and keep their offsets."""
        state = self.batch_buffer_states[topic_key]
        self._replace_pending_rows(topic_key, rows + self.batch_processors[topic_key])
        current_offsets = self.pending_commit_offsets[topic_key]
        for partition, offset in offsets.items():
            current_offsets[partition] = max(
                offset, current_offsets.get(partition, offset)
            )
        state.inflight_rows = 0
        state.oldest_enqueued_at = time.monotonic()

    def _record_flush_failure(self, topic_key: str, kind: str) -> None:
        """Back off this topic's periodic flush exponentially, up to a cap."""
        state = self.batch_buffer_states[topic_key]
        state.consecutive_failures += 1
        delay = min(
            self.flush_retry_initial_backoff_seconds
            * (2 ** (state.consecutive_failures - 1)),
            self.flush_retry_max_backoff_seconds,
        )
        state.retry_delay_seconds = delay
        state.next_flush_at = time.monotonic() + delay
        PIPELINE_METRICS.clickhouse_flush_failures.labels(
            topic=topic_key, kind=kind
        ).inc()
        PIPELINE_METRICS.clickhouse_flush_retry_delay.labels(topic=topic_key).set(delay)
        log = logger.error if kind in ("schema", "dlq_failed") else logger.warning
        log(
            "ClickHouse flush for %s failed (%s, attempt %d); next retry in %.1fs",
            topic_key,
            kind,
            state.consecutive_failures,
            delay,
        )

    def _record_flush_success(self, topic_key: str) -> None:
        state = self.batch_buffer_states[topic_key]
        if state.consecutive_failures or state.next_flush_at:
            PIPELINE_METRICS.clickhouse_flush_retry_delay.labels(topic=topic_key).set(0)
        state.consecutive_failures = 0
        state.retry_delay_seconds = 0
        state.next_flush_at = 0.0

    async def _resolve_rejected_batch(
        self,
        topic_key: str,
        rows: List[Any],
        insert_method: Callable[[List[Any]], Any],
        error: Exception,
    ) -> Tuple[List[Any], Optional[Exception], str]:
        """Bisect a batch rejected for some rows: write good rows, DLQ bad ones.

        Returns the rows that are neither written nor dead-lettered (in their
        original order), the error that kept them and its failure kind. The
        caller keeps those rows and their offsets and backs off.
        """
        bad: List[Tuple[Any, Exception]] = []
        unwritten: List[Any] = []
        stop_error: Optional[Exception] = None

        async def isolate(group: List[Any], group_error: Exception) -> None:
            nonlocal stop_error
            if len(group) == 1:
                bad.append((group[0], group_error))
                return
            middle = len(group) // 2
            for half in (group[:middle], group[middle:]):
                if stop_error is not None:
                    unwritten.extend(half)
                    continue
                try:
                    await insert_method(half)
                except Exception as half_error:
                    if _classify_insert_error(half_error) == "row":
                        await isolate(half, half_error)
                    else:
                        unwritten.extend(half)
                        stop_error = half_error

        def in_order(kept: List[Any]) -> List[Any]:
            kept_ids = {id(row) for row in kept}
            return [row for row in rows if id(row) in kept_ids]

        await isolate(rows, error)
        if stop_error is not None:
            # Bisection was interrupted: nothing is dead-lettered this time.
            return (
                in_order(unwritten + [row for row, _ in bad]),
                stop_error,
                _classify_insert_error(stop_error),
            )
        # Only a schema-class error (classified before bisecting) keeps a
        # batch; rows rejected one by one are poison even if all of them are.
        for index, (row, row_error) in enumerate(bad):
            try:
                dead_lettered = await self._dead_letter_row(topic_key, row, row_error)
            except Exception as dlq_error:
                logger.error(f"Failed to dead-letter {topic_key} row: {dlq_error}")
                dead_lettered = False
            if not dead_lettered:
                # Stop at once: keep this and every later bad row for a retry.
                return [row for row, _ in bad[index:]], row_error, "dlq_failed"
        if bad:
            PIPELINE_METRICS.clickhouse_flush_failures.labels(
                topic=topic_key, kind="row"
            ).inc()
        return [], None, "row"

    async def _dead_letter_row(
        self, topic_key: str, row: Any, error: Exception
    ) -> bool:
        """Publish one row ClickHouse rejected permanently to the topic's DLQ."""
        message = SimpleNamespace(
            topic=str(_row_field(row, "kafka_topic", "") or ""),
            partition=int(_row_field(row, "kafka_partition", -1)),
            offset=int(_row_field(row, "kafka_offset", -1)),
            key=None,
            value=_dead_letter_row_payload(row),
        )
        if not await self._publish_dead_letter(
            topic_key, message, error, stage="persist"
        ):
            return False
        PIPELINE_METRICS.clickhouse_rows_dead_lettered.labels(topic=topic_key).inc()
        return True

    def _batch_flush_targets(
        self,
    ) -> Tuple[Tuple[str, Callable[[List[Any]], Any]], ...]:
        """Map each batched topic to its ClickHouse insert method."""
        return (
            ("queries", self.clickhouse.batch_insert_query_logs),
            ("metrics", self.clickhouse.batch_insert_metrics),
            (
                "analytics_model_metrics_1m",
                self.clickhouse.batch_insert_model_performance,
            ),
            ("routing_guardrails", self.clickhouse.batch_insert_alert_events),
            (
                "routing_policy_state",
                self.clickhouse.batch_insert_routing_policy_state,
            ),
            ("alerts", self.clickhouse.batch_insert_alert_events),
            ("inference_completed", self.clickhouse.batch_insert_chat_requests),
            ("chat_feedback", self.clickhouse.batch_insert_chat_feedback),
            ("routing_decisions", self.clickhouse.batch_insert_routing_decisions),
        )

    async def flush_topic(self, topic_key: str) -> None:
        """Flush one batched topic now, raising its failure (and only its own)."""
        for target_key, insert_method in self._batch_flush_targets():
            if target_key == topic_key:
                await self._flush_batch_topic(topic_key, insert_method)
                return

    async def flush_pending_batches(self, *, respect_backoff: bool = False):
        """Flush any pending Kafka batches to ClickHouse immediately.

        The periodic flusher passes ``respect_backoff`` so a topic that keeps
        failing is retried on its own exponential schedule; explicit flushes
        (shutdown, poison-message handling) always try every topic.
        """
        flushes = self._batch_flush_targets()
        if respect_backoff:
            now = time.monotonic()
            flushes = tuple(
                (topic_key, method)
                for topic_key, method in flushes
                if self.batch_buffer_states[topic_key].next_flush_at <= now
            )
        semaphore = asyncio.Semaphore(self.max_concurrent_flushes)

        async def limited_flush(topic_key, insert_method):
            async with semaphore:
                await self._flush_batch_topic(topic_key, insert_method)

        results = await asyncio.gather(
            *(limited_flush(topic_key, method) for topic_key, method in flushes),
            return_exceptions=True,
        )

        self.last_batch_time = time.time()
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]

    def _deserialize_message(self, message: bytes) -> Dict[str, Any]:
        """Deserialize message from JSON bytes"""
        return json.loads(message.decode("utf-8"))

    def _serialize_message(self, message: Any) -> bytes:
        """Serialize consumer-side DLQ payloads as JSON bytes."""
        return _json_bytes(message, default=str)

    async def shutdown(self):
        """Shutdown all consumers"""
        self.running = False
        failures: List[BaseException] = []
        try:
            await self.flush_pending_batches()
        except BaseException as exc:
            failures.append(exc)
        stop_results = await asyncio.gather(
            *(consumer.stop() for consumer in self.consumers.values()),
            return_exceptions=True,
        )
        failures.extend(
            result for result in stop_results if isinstance(result, BaseException)
        )
        if self.dlq_producer:
            try:
                await self.dlq_producer.stop()
            except BaseException as exc:
                failures.append(exc)
        logger.info("Kafka consumers shutdown complete")
        if failures:
            raise failures[0]


class KafkaIngestionPipeline:
    """Main Kafka ingestion pipeline coordinator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        clickhouse_config = config.get("clickhouse", {})
        self.clickhouse_manager = ClickHouseManager(clickhouse_config)
        self.producer_manager = KafkaProducerManager(config)
        self.consumer_manager = None  # Will be initialized after ClickHouse
        self._stream_handlers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {
            "requests_enriched": [],
            "analytics_model_metrics_1m": [],
            "routing_guardrails": [],
            "routing_policy_state": [],
            "alerts": [],
        }
        dashboard_config = clickhouse_config.get("dashboard", {})
        self._dashboard_cache_ttl_seconds = max(
            0.0, float(dashboard_config.get("cache_ttl_seconds", 5))
        )
        self._dashboard_cache_max_entries = max(
            1, int(dashboard_config.get("cache_max_entries", 32))
        )
        self._dashboard_query_semaphore = asyncio.Semaphore(
            max(1, int(dashboard_config.get("max_concurrent_queries", 4)))
        )
        self._dashboard_cache: OrderedDict[
            tuple[int, Optional[str]], tuple[float, Dict[str, Any]]
        ] = OrderedDict()
        self._dashboard_inflight: Dict[
            tuple[int, Optional[str]], asyncio.Task[Dict[str, Any]]
        ] = {}
        self.running = False

    async def initialize(self):
        """Initialize the complete pipeline"""
        logger.info("Initializing Kafka ingestion pipeline...")

        # Initialize ClickHouse first
        await self.clickhouse_manager.initialize()

        # Initialize producer
        await self.producer_manager.initialize()

        # Initialize consumer with ClickHouse manager
        self.consumer_manager = KafkaConsumerManager(
            self.config, self.clickhouse_manager
        )
        await self.consumer_manager.initialize()
        for topic_key, callbacks in self._stream_handlers.items():
            for callback in callbacks:
                self.consumer_manager.register_observer(topic_key, callback)

        logger.info("Kafka ingestion pipeline initialized successfully")

    async def start(self):
        """Start the pipeline"""
        logger.info("Starting Kafka ingestion pipeline...")
        self.running = True

        # Keep the cached ClickHouse health current while the pipeline runs.
        self.clickhouse_manager.start_health_pings()
        try:
            # Start consuming messages
            await self.consumer_manager.start_consuming()
        finally:
            await self.clickhouse_manager.stop_health_pings()

    async def log_query(
        self,
        query_request: QueryRequest,
        inference_response: InferenceResponse,
        routing_decision: Any,
    ):
        """Log query through the pipeline"""
        await self.producer_manager.produce_query_log(
            query_request, inference_response, routing_decision
        )

    async def log_metric(
        self,
        service: str,
        metric_name: str,
        metric_value: float,
        labels: Dict[str, str] = None,
    ):
        """Log metric through the pipeline"""
        await self.producer_manager.produce_metric(
            service, metric_name, metric_value, labels
        )

    async def log_error(self, error_data: Dict[str, Any]):
        """Log error through the pipeline"""
        await self.producer_manager.produce_error(error_data)

    async def get_analytics(
        self,
        user_id: Optional[str] = None,
        hours: int = 24,
        *,
        raise_errors: bool = False,
    ) -> Dict[str, Any]:
        """Get analytics from ClickHouse"""
        if not raise_errors:
            return await self.clickhouse_manager.get_query_analytics(user_id, hours)
        return await self.clickhouse_manager.get_query_analytics(
            user_id, hours, raise_errors=raise_errors
        )

    async def get_query_analytics(
        self,
        user_id: Optional[str] = None,
        hours: int = 24,
        *,
        raise_errors: bool = False,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for dashboard analytics lookups."""
        return await self.get_analytics(user_id, hours, raise_errors=raise_errors)

    async def get_model_performance(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get model performance from ClickHouse"""
        if not raise_errors:
            return await self.clickhouse_manager.get_model_performance(hours)
        return await self.clickhouse_manager.get_model_performance(
            hours, raise_errors=raise_errors
        )

    async def get_routing_guardrails(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get persisted routing guardrails from ClickHouse."""
        if not raise_errors:
            return await self.clickhouse_manager.get_routing_guardrails(hours)
        return await self.clickhouse_manager.get_routing_guardrails(
            hours, raise_errors=raise_errors
        )

    async def get_routing_policy_state_events(
        self, hours: int = 24, *, raise_errors: bool = False
    ) -> List[Dict[str, Any]]:
        """Get persisted routing.policy_state events from ClickHouse."""
        if not raise_errors:
            return await self.clickhouse_manager.get_routing_policy_state_events(hours)
        return await self.clickhouse_manager.get_routing_policy_state_events(
            hours, raise_errors=raise_errors
        )

    async def get_dashboard_bundle(
        self, user_id: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Read independent dashboard sections with cache and single-flight."""
        key = (_window_hours(hours), user_id)
        now = time.monotonic()
        cached = self._dashboard_cache.get(key)
        if cached is not None:
            expires_at, value = cached
            if expires_at > now:
                self._dashboard_cache.move_to_end(key)
                return copy.deepcopy(value)
            self._dashboard_cache.pop(key, None)

        task = self._dashboard_inflight.get(key)
        if task is None:
            task = asyncio.create_task(
                self._load_dashboard_bundle_and_cleanup(
                    key, user_id=user_id, hours=key[0]
                )
            )
            self._dashboard_inflight[key] = task

        return copy.deepcopy(await asyncio.shield(task))

    async def _load_dashboard_bundle_and_cleanup(
        self,
        key: tuple[int, Optional[str]],
        *,
        user_id: Optional[str],
        hours: int,
    ) -> Dict[str, Any]:
        try:
            return await self._load_dashboard_bundle(user_id=user_id, hours=hours)
        finally:
            current = asyncio.current_task()
            if self._dashboard_inflight.get(key) is current:
                self._dashboard_inflight.pop(key, None)

    async def _load_dashboard_bundle(
        self, *, user_id: Optional[str], hours: int
    ) -> Dict[str, Any]:
        sections: Dict[str, tuple[Callable[..., Any], Dict[str, Any], Any]] = {
            "analytics": (
                self.get_query_analytics,
                {"user_id": user_id, "hours": hours, "raise_errors": True},
                {},
            ),
            "model_performance": (
                self.get_model_performance,
                {"hours": hours, "raise_errors": True},
                [],
            ),
            "routing_guardrails": (
                self.get_routing_guardrails,
                {"hours": hours, "raise_errors": True},
                [],
            ),
            "routing_policy_state": (
                self.get_routing_policy_state_events,
                {"hours": hours, "raise_errors": True},
                [],
            ),
        }

        async def load(name, method, kwargs, default):
            async with self._dashboard_query_semaphore:
                try:
                    return name, await method(**kwargs), None
                except Exception as exc:
                    logger.error("Dashboard section %s failed: %s", name, exc)
                    return name, copy.deepcopy(default), str(exc)

        loaded = await asyncio.gather(
            *(
                load(name, method, kwargs, default)
                for name, (method, kwargs, default) in sections.items()
            )
        )
        bundle: Dict[str, Any] = {"errors": {}}
        for name, value, error in loaded:
            bundle[name] = value
            if error is not None:
                bundle["errors"][name] = error

        if not bundle["errors"] and self._dashboard_cache_ttl_seconds > 0:
            key = (hours, user_id)
            self._dashboard_cache[key] = (
                time.monotonic() + self._dashboard_cache_ttl_seconds,
                copy.deepcopy(bundle),
            )
            self._dashboard_cache.move_to_end(key)
            while len(self._dashboard_cache) > self._dashboard_cache_max_entries:
                self._dashboard_cache.popitem(last=False)

        return bundle

    def register_stream_handler(
        self, topic_key: str, callback: Callable[[Dict[str, Any]], Any]
    ):
        """Register an in-process handler for consumed stream topics."""
        self._stream_handlers.setdefault(topic_key, []).append(callback)
        if self.consumer_manager:
            self.consumer_manager.register_observer(topic_key, callback)

    def attach_monitoring_service(self, monitoring_service: Any):
        """Forward consumed analytics streams into the monitoring service."""
        request_feature_handler = getattr(
            monitoring_service, "ingest_stream_request_enriched", None
        )
        if callable(request_feature_handler):
            self.register_stream_handler("requests_enriched", request_feature_handler)

        model_metrics_handler = getattr(
            monitoring_service, "ingest_stream_model_metrics", None
        )
        if callable(model_metrics_handler):
            self.register_stream_handler(
                "analytics_model_metrics_1m", model_metrics_handler
            )

        routing_guardrail_handler = getattr(
            monitoring_service, "ingest_stream_routing_guardrail", None
        )
        if callable(routing_guardrail_handler):
            self.register_stream_handler(
                "routing_guardrails", routing_guardrail_handler
            )

        routing_policy_state_handler = getattr(
            monitoring_service, "ingest_stream_routing_policy_state", None
        )
        if callable(routing_policy_state_handler):
            self.register_stream_handler(
                "routing_policy_state", routing_policy_state_handler
            )

        alert_handler = getattr(monitoring_service, "ingest_stream_alert", None)
        if callable(alert_handler):
            self.register_stream_handler("alerts", alert_handler)

    async def flush(self):
        """Flush any pending consumer batches to ClickHouse"""
        if self.consumer_manager:
            await self.consumer_manager.flush_pending_batches()

    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status"""
        consumer_health = {"healthy": False}
        if self.consumer_manager:
            get_consumer_health = getattr(
                self.consumer_manager, "get_health_status", None
            )
            consumer_health = (
                get_consumer_health()
                if callable(get_consumer_health)
                else {"healthy": bool(getattr(self.consumer_manager, "consumers", {}))}
            )
        return {
            "pipeline_running": self.running,
            "producer_healthy": self.producer_manager.is_healthy(),
            "producer_last_error": self.producer_manager.last_error,
            "consumer_healthy": bool(consumer_health.get("healthy")),
            "consumer_status": consumer_health,
            "clickhouse_healthy": self.clickhouse_manager.is_connection_healthy(),
            "clickhouse_last_error": self.clickhouse_manager.last_ping_error,
            "total_consumers": len(self.consumer_manager.consumers)
            if self.consumer_manager
            else 0,
        }

    def is_healthy(self) -> bool:
        """Return whether the pipeline can produce, consume, and persist events."""
        health = self.get_health_status()
        # A failed ClickHouse ping only degrades analytics (see is_degraded);
        # it must not take a chat-serving instance out of readiness.
        return bool(
            health["producer_healthy"]
            and health["consumer_healthy"]
            and self.clickhouse_manager.client is not None
        )

    def is_degraded(self) -> bool:
        """Return whether the pipeline runs but its last ClickHouse ping failed."""
        return self.clickhouse_manager.last_ping_ok is False

    async def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down Kafka ingestion pipeline...")
        self.running = False

        failures: List[BaseException] = []
        if self.consumer_manager:
            try:
                await self.consumer_manager.shutdown()
            except BaseException as exc:
                failures.append(exc)

        try:
            await self.producer_manager.shutdown()
        except BaseException as exc:
            failures.append(exc)
        try:
            self.clickhouse_manager.shutdown()
        except BaseException as exc:
            failures.append(exc)

        logger.info("Kafka ingestion pipeline shutdown complete")
        if failures:
            raise failures[0]
