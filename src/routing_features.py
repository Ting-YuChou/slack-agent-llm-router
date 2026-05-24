"""
Shared request-shape features for synchronous routing and Flink enrichment.

Fast-lane eligibility is intentionally explicit-SLA only. Query text can still
inform task type and complexity, but it must not imply urgency by itself.
"""

from __future__ import annotations

import re
from typing import Any, Dict

EXPLICIT_LOW_LATENCY_SLA_VALUES = {"low", "interactive"}

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return False


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


def _normalize_latency_sla(value: Any) -> str:
    normalized = str(value or "normal").strip().lower()
    if normalized in {"normal", "low", "interactive"}:
        return normalized
    return "normal"


def calculate_priority(
    query_text: str = "",
    user_tier: str = "",
    user_id: str = "",
    priority: Any = None,
) -> str:
    """Convert explicit service/API priority to an analytics label."""
    del query_text, user_tier, user_id

    if isinstance(priority, str):
        normalized = priority.strip().lower()
        if normalized in {"low", "medium", "high", "critical"}:
            return normalized

    priority_value = _safe_int(priority, 1)
    if priority_value >= 5:
        return "critical"
    if priority_value >= 4:
        return "high"
    if priority_value >= 2:
        return "medium"
    return "low"


def estimate_query_tokens(query_text: str, context: str = "") -> int:
    """Estimate prompt size cheaply for request-side routing features."""
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


def _low_latency_intent_source(
    *, metadata: Dict[str, Any], latency_sla: str, priority: Any
) -> str | None:
    if _safe_bool(metadata.get("requires_low_latency")):
        return metadata.get("routing_intent_source") or "metadata.requires_low_latency"
    if latency_sla in EXPLICIT_LOW_LATENCY_SLA_VALUES:
        return metadata.get("routing_intent_source") or "metadata.latency_sla"
    if str(priority or "").strip().lower() in {"high", "critical"}:
        return metadata.get("routing_intent_source") or "api_priority"
    if _safe_int(priority, 1) >= 4:
        return metadata.get("routing_intent_source") or "api_priority"
    return None


def build_routing_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate request-side routing features from the raw API payload."""
    metadata = data.get("metadata") or {}
    query_text = str(data.get("query_text") or data.get("query") or "")
    context = str(data.get("context") or "")
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

    query_type = str(data.get("query_type") or infer_query_type(query_text)).lower()
    token_estimate = _safe_int(data.get("query_token_estimate"))
    if token_estimate <= 0:
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

    requires_high_reasoning = _safe_bool(metadata.get("requires_high_reasoning")) or (
        query_type in HIGH_REASONING_QUERY_TYPES or query_complexity == "complex"
    )
    latency_sla = _normalize_latency_sla(metadata.get("latency_sla"))
    routing_intent_source = _low_latency_intent_source(
        metadata=metadata,
        latency_sla=latency_sla,
        priority=data.get("priority"),
    )
    requires_low_latency = routing_intent_source is not None
    route_to_fast_lane = (
        requires_low_latency
        and not requires_high_reasoning
        and query_complexity != "complex"
        and not long_context
        and not attachment_heavy
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
        "route_to_fast_lane": route_to_fast_lane,
        "latency_sla": latency_sla,
        "routing_intent_source": routing_intent_source,
        "preferred_models": _coerce_list(metadata.get("preferred_models")),
        "avoid_models": _coerce_list(metadata.get("avoid_models")),
        "avoid_providers": [
            provider.lower()
            for provider in _coerce_list(metadata.get("avoid_providers"))
        ],
    }
