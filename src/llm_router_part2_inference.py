"""
LLM Router Part 2: Inference Engine - Multi-Provider Model Inference
Handles inference requests across OpenAI, Anthropic, and vLLM providers
"""

import asyncio
import base64
import email.utils
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Any,
    AsyncIterator,
    Tuple,
    Sequence,
)
import hashlib

import httpx
import openai
import anthropic

from src.admission import AdmissionRejectedError
from src.llm_router_part1_router import ModelRouter
from src.provider_scheduler import (
    REQUEST_SHAPED_STATUS_CODES,
    ProviderCapacityScheduler,
    RequestExecutionBudget,
    _exception_status_code,
)
from src.provider_scheduler import is_billing_exhausted as _is_billing_exhausted
from src.provider_scheduler import provider_error_text as _provider_error_text
from src.utils.metrics import INFERENCE_METRICS, RAG_METRICS
from src.utils.schema import (
    AttachmentType,
    QueryRequest,
    InferenceResponse,
    QueryType,
    RagPolicy,
    ToolCall,
    ToolPolicy,
)
from src.tools import ToolRegistry, WebSearchTool, WebSearchResult
from src.tools.web_search import WEB_SEARCH_UNAVAILABLE_WARNING

logger = logging.getLogger(__name__)


class _LazyRedisFactory:
    def __call__(self, *args, **kwargs):
        from redis.asyncio import Redis

        return Redis(*args, **kwargs)


class _LazyRedisNamespace:
    def __init__(self):
        self.Redis = _LazyRedisFactory()


redis = _LazyRedisNamespace()


@dataclass
class PromptSections:
    """Provider prompt sections split by cache stability."""

    static_cache_prefix: List[str]
    dynamic_context: List[str]
    query: str


@dataclass
class VLLMEndpointState:
    """Runtime state for one vLLM serving endpoint."""

    name: str
    base_url: str
    models: List[str] = field(default_factory=list)
    weight: float = 1.0
    max_outstanding: int = 0
    health_path: str = "/health"
    metrics_path: str = "/metrics"
    prefix_cache_enabled: bool = True
    enabled: bool = True
    client: Optional[httpx.AsyncClient] = None
    healthy: bool = True
    outstanding: int = 0
    queue_depth: int = 0
    running_requests: int = 0
    kv_cache_usage: Optional[float] = None
    last_health_check: float = 0.0
    last_metrics_refresh: float = 0.0
    cooldown_until: float = 0.0
    last_error: Optional[str] = None


RAG_AUTO_STRONG_TERMS = {
    "academic calendar",
    "admissions",
    "advisor",
    "attendance",
    "campus",
    "catalog",
    "class schedule",
    "curriculum",
    "degree",
    "department",
    "dorm",
    "enrollment",
    "exam",
    "faculty",
    "financial aid",
    "graduation",
    "handbook",
    "library",
    "major",
    "office hours",
    "prerequisite",
    "scholarship",
    "school",
    "semester",
    "syllabus",
    "transcript",
    "tuition",
    "university",
    "學生",
    "學校",
    "校園",
    "校曆",
    "課程",
    "選課",
    "註冊",
    "入學",
    "招生",
    "學費",
    "獎學金",
    "助學金",
    "退費",
    "宿舍",
    "圖書館",
    "系所",
    "學院",
    "教授",
    "老師",
    "辦公室時間",
    "成績單",
    "畢業",
    "學位",
    "主修",
    "先修",
    "考試",
    "期中",
    "期末",
    "手冊",
    "大綱",
    "文件",
    "PDF",
}

RAG_AUTO_CONTEXT_TERMS = {
    "course",
    "document",
    "file",
    "form",
    "handbook",
    "policy",
    "program",
    "registration",
    "schedule",
    "pdf",
    "規定",
    "政策",
    "文件",
    "表格",
    "手冊",
}

RAG_AUTO_SCHOOL_CONTEXT_TERMS = {
    "academic",
    "campus",
    "class",
    "college",
    "course",
    "department",
    "faculty",
    "school",
    "student",
    "university",
    "學校",
    "學生",
    "校園",
    "課程",
    "系所",
    "學院",
    "大學",
}

RAG_AUTO_EXCLUDE_TERMS = {
    "breaking",
    "current",
    "latest",
    "news",
    "now",
    "price",
    "release",
    "stock",
    "today",
    "web",
    "最新",
    "今天",
    "現在",
    "新聞",
    "即時",
    "搜尋",
    "查詢網頁",
}


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce SDK usage values without treating mocks or missing fields as tokens."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return default
    return default


def _usage_value(usage: Any, field_name: str, default: int = 0) -> int:
    """Read a usage field from either SDK objects or dictionaries."""
    if isinstance(usage, dict):
        return _safe_int(usage.get(field_name), default)
    return _safe_int(getattr(usage, field_name, None), default)


def _nested_usage_value(
    usage: Any, parent_name: str, field_name: str, default: int = 0
) -> int:
    """Read nested provider usage fields like prompt_tokens_details.cached_tokens."""
    parent = (
        usage.get(parent_name)
        if isinstance(usage, dict)
        else getattr(usage, parent_name, None)
    )
    if isinstance(parent, dict):
        return _safe_int(parent.get(field_name), default)
    return _safe_int(getattr(parent, field_name, None), default)


def _first_nested_usage_value(
    usage: Any, parent_names: Sequence[str], field_name: str, default: int = 0
) -> int:
    """Read a nested usage field from the first parent that reports it.

    The Responses API reports ``input_tokens_details``; Chat-Completions-compatible
    backends report ``prompt_tokens_details`` instead.
    """
    for parent_name in parent_names:
        parent = (
            usage.get(parent_name)
            if isinstance(usage, dict)
            else getattr(usage, parent_name, None)
        )
        if parent is None:
            continue
        value = (
            parent.get(field_name)
            if isinstance(parent, dict)
            else getattr(parent, field_name, None)
        )
        if value is not None:
            return _safe_int(value, default)
    return default


# HTTP status /route answers for each error_code; other failures stay 502.
ERROR_CODE_HTTP_STATUS: Dict[str, int] = {
    "context_length_exceeded": 413,
    "prompt_too_large": 413,
    "model_not_entitled": 403,
    "rate_limited": 429,
    "provider_unavailable": 503,
    "retrieval_unavailable": 503,
    "timeout": 504,
}
# Statuses whose responses may carry Retry-After (RFC 9110 section 10.2.3).
_RETRY_AFTER_HTTP_STATUSES = frozenset({429, 503})
# Phrases providers use for a prompt over the model's context window: OpenAI's
# error code and "context window"; Anthropic's "prompt is too long" and
# "input length and `max_tokens` exceed context limit"; vLLM's "maximum
# context length is N tokens" and "is longer than the maximum model length".
_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "prompt is too long",
    "exceed context limit",
    "maximum context length",
    "longer than the maximum model length",
    "context window",
)


def _string_attribute(exc: BaseException, name: str) -> Optional[str]:
    """A non-empty string attribute that a typed exception sets on purpose."""
    value = getattr(exc, name, None)
    return value if isinstance(value, str) and value else None


def _admission_rejection_reason(exc: AdmissionRejectedError) -> str:
    """The rejection reason, looking through a queue wait to what it waited on."""
    decision = getattr(exc, "decision", None)
    reason = str(getattr(decision, "reason", "") or "")
    if reason == "scheduler_queue_timeout":
        metadata = getattr(decision, "metadata", None) or {}
        reason = str(metadata.get("last_reason") or "")
    return reason


def _class_names(exc: BaseException) -> set:
    return {cls.__name__ for cls in type(exc).__mro__}


def _is_timeout_exception(exc: BaseException) -> bool:
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or any(
        "Timeout" in name for name in _class_names(exc)
    )


def _is_connection_exception(exc: BaseException) -> bool:
    return isinstance(exc, ConnectionError) or bool(
        _class_names(exc) & {"APIConnectionError", "TransportError"}
    )


def _is_context_length_error(exc: BaseException) -> bool:
    if getattr(exc, "code", None) == "context_length_exceeded":
        return True
    text = _provider_error_text(exc)
    return any(marker in text for marker in _CONTEXT_LENGTH_MARKERS)


def classify_inference_error(exc: BaseException) -> str:
    """Bucket a failed inference so analytics only blame providers for their faults.

    Matches SDK and httpx errors by attribute and class name, so it works without
    importing the provider SDKs. A typed exception that sets ``error_type``
    decides for itself.
    """
    explicit_error_type = _string_attribute(exc, "error_type")
    if explicit_error_type:
        return explicit_error_type
    if isinstance(exc, AdmissionRejectedError):
        # An open circuit (or a queue wait that ended on one) means the provider
        # has been failing; other rejections are our own capacity policy.
        if _admission_rejection_reason(exc) == "circuit_open":
            return "provider_error"
        return "user_or_policy"
    if _is_timeout_exception(exc):
        return "timeout"
    status_code = _exception_status_code(exc)
    if status_code is not None and status_code >= 400:
        if status_code == 408:
            return "timeout"
        if status_code == 429:
            return "rate_limited"
        if status_code in REQUEST_SHAPED_STATUS_CODES:
            # An exhausted account fails every request, whatever it says.
            if _is_billing_exhausted(exc):
                return "provider_error"
            return "user_or_policy"
        # 5xx, and 4xx such as 401/403 (revoked key) or 404 (retired model).
        return "provider_error"
    if _is_connection_exception(exc):
        return "provider_error"
    if isinstance(exc, json.JSONDecodeError):
        # A provider answered with a body that is not JSON (an HTML error page,
        # a truncated stream); JSONDecodeError is a ValueError, so check first.
        return "provider_error"
    if isinstance(exc, ValueError):
        return "user_or_policy"
    if isinstance(exc, RuntimeError):
        return "provider_error"
    return "internal"


def inference_error_code(exc: BaseException) -> Optional[str]:
    """Name a failure specifically enough for the caller to act on it.

    Typed exceptions (the router's entitlement and size rejections, retrieval
    failures) carry ``error_code`` themselves. Returns None when nothing more
    specific than "inference failed" is known.
    """
    explicit_error_code = _string_attribute(exc, "error_code")
    if explicit_error_code:
        return explicit_error_code
    if isinstance(exc, AdmissionRejectedError):
        decision_status = getattr(getattr(exc, "decision", None), "status_code", 0)
        if _admission_rejection_reason(exc) == "circuit_open" or (
            isinstance(decision_status, int) and decision_status >= 500
        ):
            return "provider_unavailable"
        return "rate_limited"
    if _is_timeout_exception(exc):
        return "timeout"
    status_code = _exception_status_code(exc)
    if status_code is not None and status_code >= 400:
        if status_code == 408:
            return "timeout"
        if status_code == 429:
            # An exhausted billing quota does not clear in seconds.
            if _is_billing_exhausted(exc):
                return "provider_unavailable"
            return "rate_limited"
        if status_code in REQUEST_SHAPED_STATUS_CODES:
            if _is_billing_exhausted(exc):
                return "provider_unavailable"
            if _is_context_length_error(exc):
                return "context_length_exceeded"
            return "prompt_too_large" if status_code == 413 else None
        # 401/403/404 mean our credentials or model name are wrong: a 502.
        return "provider_unavailable" if status_code >= 500 else None
    if _is_connection_exception(exc):
        return "provider_unavailable"
    if isinstance(exc, RuntimeError) and "no healthy" in str(exc).lower():
        return "provider_unavailable"
    return None


class RetrievalUnavailableError(RuntimeError):
    """rag_policy is REQUIRED, but document retrieval timed out or failed.

    The answer must come from the documents, so the request fails instead of
    answering without them. Neither the model nor its provider is at fault.
    """

    error_code = "retrieval_unavailable"
    error_type = "internal"
    http_status = 503

    def __init__(self, status: str, detail: Optional[str] = None):
        super().__init__(
            f"Document retrieval {status}" + (f" ({detail})" if detail else "")
        )
        self.status = status


def _header_value(headers: Any, name: str) -> Optional[str]:
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    for candidate in (name, name.title()):
        value = getter(candidate)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _retry_after_seconds(exc: BaseException) -> Optional[int]:
    """How long the provider (or our admission control) asked callers to wait."""
    if isinstance(exc, AdmissionRejectedError):
        seconds = getattr(getattr(exc, "decision", None), "retry_after_seconds", None)
        if isinstance(seconds, int) and not isinstance(seconds, bool):
            return max(0, seconds)
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None or _is_billing_exhausted(exc):
        # A quota error's Retry-After would only invite futile retries.
        return None
    try:
        milliseconds = _header_value(headers, "retry-after-ms")
        if milliseconds is not None:
            return max(0, math.ceil(float(milliseconds) / 1000))
        value = _header_value(headers, "retry-after")
        if value is None:
            return None
        try:
            return max(0, math.ceil(float(value)))
        except ValueError:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return max(0, math.ceil(delay))
    except (TypeError, ValueError, OverflowError):
        return None


def route_response_status(response: Any) -> Tuple[int, Dict[str, str]]:
    """The HTTP status (and headers) /route answers for an engine response."""
    if not getattr(response, "error", None):
        return 200, {}
    error_code = getattr(response, "error_code", None)
    status_code = ERROR_CODE_HTTP_STATUS.get(error_code or "", 502)
    headers: Dict[str, str] = {}
    retry_after = getattr(response, "retry_after_seconds", None)
    if status_code in _RETRY_AFTER_HTTP_STATUSES and isinstance(retry_after, int):
        headers["Retry-After"] = str(retry_after)
    return status_code, headers


_PRIMARY_ATTEMPT_EXCEPTION_ATTR = "_primary_attempt_exception"


def _primary_attempt_exception(exc: BaseException) -> BaseException:
    """Return the primary model's failure that exc (or its cause chain) carries.

    When every fallback also fails, the exception that surfaces belongs to the
    last fallback, or to the request deadline. Failure analytics describe the
    primary attempt, so the fallback chain attaches that attempt's exception.
    """
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        primary = getattr(current, _PRIMARY_ATTEMPT_EXCEPTION_ATTR, None)
        if isinstance(primary, BaseException):
            return primary
        current = current.__cause__ or current.__context__
    return exc


def _openai_cached_input_tokens(usage: Any) -> int:
    """Cached prompt tokens: the Responses API field first, then Chat Completions."""
    return _first_nested_usage_value(
        usage, ("input_tokens_details", "prompt_tokens_details"), "cached_tokens"
    )


# finish_reason values: "stop" for a complete answer, "length" when a token
# limit cut it off, "content_filter" when the provider refused or filtered it.
# Anything that is not a string (a mocked SDK object) counts as unknown: "stop".
_OPENAI_INCOMPLETE_REASONS = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}
_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
    "tool_use": "tool_calls",
}
# Warnings for answers that are returned but must not be cached; any other
# finish_reason but "stop" (vLLM abort, OpenAI failed or cancelled) gets
# the generic one.
_FINISH_REASON_WARNINGS = {
    "length": "response_truncated",
    "content_filter": "response_filtered",
}
_UNFINISHED_RESPONSE_WARNING = "response_incomplete"


def _field_value(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _openai_refused(response: Any) -> bool:
    """Whether a Responses API answer contains a refusal content part."""
    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        content = _field_value(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        if any(_field_value(part, "type") == "refusal" for part in content):
            return True
    return False


def _openai_finish_reason(response: Any) -> str:
    status = getattr(response, "status", None)
    if not isinstance(status, str):
        return "stop"
    if status == "incomplete":
        reason = _field_value(getattr(response, "incomplete_details", None), "reason")
        if isinstance(reason, str):
            return _OPENAI_INCOMPLETE_REASONS.get(reason, reason)
        return "incomplete"
    if status != "completed":
        return status
    return "content_filter" if _openai_refused(response) else "stop"


def _anthropic_finish_reason(response: Any) -> str:
    stop_reason = getattr(response, "stop_reason", None)
    if not isinstance(stop_reason, str):
        return "stop"
    return _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)


def _vllm_finish_reason(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    reason = _field_value(first_choice, "finish_reason")
    return reason if isinstance(reason, str) and reason else "stop"


def _sdk_request_id(response: Any) -> Optional[str]:
    """The SDK's ``_request_id``, taken from the provider's request-id header."""
    return _string_attribute(response, "_request_id")


def _exception_provider_request_id(exc: BaseException) -> Optional[str]:
    """The provider's request id on a failed call: SDK errors, then httpx."""
    request_id = _string_attribute(exc, "request_id")
    if request_id:
        return request_id
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return _header_value(headers, "x-request-id")


# Prompt line for a live web search that failed, or that a REQUIRED tool
# policy needed but that found nothing: no web context reached the model.
WEB_SEARCH_UNAVAILABLE_NOTICE = (
    "Live web search failed or returned no results for this request, so no "
    "current web information is included. Say that you cannot verify the "
    "latest information, and do not present older knowledge as current."
)
# Retrieval outcomes that mean nobody knows what the documents say.
_RAG_UNAVAILABLE_STATUSES = ("timeout", "error")
# Prompt line for rag_policy AUTO when document search timed out or failed.
RAG_UNAVAILABLE_NOTICE = (
    "Document search was unavailable for this request, so this answer is not "
    "grounded in the knowledge base. If the question depends on those "
    "documents, say so instead of guessing what they contain."
)
# inference.retrieval_budget_ratio when unset or invalid.
_DEFAULT_RETRIEVAL_BUDGET_RATIO = 0.5
# Least time a retrieval step leaves before the request deadline, so its own
# timeout always fires before the _within_request_deadline backstop.
_RETRIEVAL_BACKSTOP_MARGIN_SECONDS = 0.05


def _drop_metadata_key(request: QueryRequest, key: str) -> None:
    """Remove a key the engine owns from caller-supplied request metadata."""
    if request.metadata and key in request.metadata:
        request.metadata = {
            name: value for name, value in request.metadata.items() if name != key
        }


async def _await_within(
    awaitable: Awaitable[Any], time_limit_seconds: Optional[float]
) -> Any:
    """Await, raising asyncio.TimeoutError after time_limit_seconds (None: no limit)."""
    if time_limit_seconds is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=max(0.0, time_limit_seconds))


@dataclass
class InferenceContext:
    """Context information for inference requests"""

    request_id: str
    user_id: str
    model_name: str
    start_time: float
    token_count_input: int
    compressed_context: bool = False
    cached_response: bool = False


class ContextCompressor:
    """Advanced context compression for long sequences"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.compression_ratio = config.get("compression_ratio", 0.3)
        self.max_context_tokens = config.get("max_context_tokens", 100000)
        self.method = config.get("method", "semantic_graph")
        self.tokenizer = None

    async def initialize(self):
        """Initialize compression components"""
        if not self.enabled:
            logger.info("Context compression is disabled")
            return

        try:
            # Load tokenizer for token counting
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            logger.info("Context compressor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize context compressor: {e}")

    async def compress_context(self, context: str, target_length: int) -> str:
        """Compress context to target length while preserving key information"""
        if not self.enabled:
            return context
        if not context or len(context) < target_length:
            return context

        if self.method == "semantic_graph":
            return await self._semantic_graph_compression(context, target_length)
        elif self.method == "sketch_based":
            return await self._sketch_based_compression(context, target_length)
        elif self.method == "attention_based":
            return await self._attention_based_compression(context, target_length)
        else:
            return self._sliding_window_compression(context, target_length)

    async def _semantic_graph_compression(
        self, context: str, target_length: int
    ) -> str:
        """Semantic graph-based compression preserving key relationships"""
        # Split into sentences
        sentences = self._split_sentences(context)
        if len(sentences) <= 3:
            return context

        # Score sentences by importance
        sentence_scores = await self._score_sentences(sentences)

        # Select top sentences maintaining order
        selected_indices = sorted(
            sorted(
                range(len(sentences)), key=lambda i: sentence_scores[i], reverse=True
            )[: max(3, int(len(sentences) * self.compression_ratio))]
        )

        compressed = " ".join(sentences[i] for i in selected_indices)

        # Ensure target length
        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed

    async def _sketch_based_compression(self, context: str, target_length: int) -> str:
        """Sketch-based compression using key phrase extraction"""
        # Extract key phrases and entities
        key_phrases = self._extract_key_phrases(context)

        # Build compressed version focusing on key information
        sections = context.split("\n\n")
        important_sections = []

        for section in sections:
            if any(phrase.lower() in section.lower() for phrase in key_phrases):
                important_sections.append(section)

        compressed = "\n\n".join(important_sections)

        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed or context[:target_length]

    async def _attention_based_compression(
        self, context: str, target_length: int
    ) -> str:
        """Attention-based compression (simplified implementation)"""
        # This would ideally use a trained attention model
        # For now, implement a heuristic approach

        paragraphs = context.split("\n\n")
        if len(paragraphs) <= 2:
            return context[:target_length]

        # Score paragraphs by length and keyword density
        paragraph_scores = []
        important_keywords = self._extract_keywords(context)

        for para in paragraphs:
            score = len(para)  # Length score
            score += sum(para.lower().count(kw) for kw in important_keywords) * 50
            paragraph_scores.append(score)

        # Select top paragraphs
        num_select = max(2, int(len(paragraphs) * self.compression_ratio))
        top_indices = sorted(
            range(len(paragraphs)), key=lambda i: paragraph_scores[i], reverse=True
        )[:num_select]

        # Maintain original order
        selected_paragraphs = [paragraphs[i] for i in sorted(top_indices)]
        compressed = "\n\n".join(selected_paragraphs)

        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed

    def _sliding_window_compression(self, context: str, target_length: int) -> str:
        """Simple sliding window compression"""
        if len(context) <= target_length:
            return context

        # Take beginning and end, skip middle
        keep_start = target_length // 2
        keep_end = target_length - keep_start - 20  # Reserve space for separator

        start_part = context[:keep_start]
        end_part = context[-keep_end:]

        return f"{start_part}\n...[COMPRESSED]...\n{end_part}"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    async def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences by importance"""
        scores = []

        # Simple heuristic scoring
        for sentence in sentences:
            score = 0

            # Length factor (prefer medium-length sentences)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 2
            elif length > 30:
                score += 1

            # Keyword density
            keywords = ["important", "key", "main", "crucial", "significant", "primary"]
            score += sum(sentence.lower().count(kw) for kw in keywords)

            # Position factor (first and last sentences often important)
            if sentences.index(sentence) in [0, len(sentences) - 1]:
                score += 1

            scores.append(score)

        return scores

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - in production, use NLP libraries
        words = text.lower().split()

        # Find repeated phrases
        phrases = {}
        for i in range(len(words) - 1):
            phrase = " ".join(words[i : i + 2])
            phrases[phrase] = phrases.get(phrase, 0) + 1

        # Return top phrases
        return sorted(phrases.keys(), key=phrases.get, reverse=True)[:10]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        words = text.lower().split()

        # Filter out common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]

        # Count frequency
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1

        # Return top keywords
        return sorted(word_count.keys(), key=word_count.get, reverse=True)[:15]


class ResponseCache:
    """Intelligent response caching system"""

    ANALYTICS_METADATA_KEYS = {
        "memory_hit_count",
        "memory_hit_ids",
        "memory_match_sources",
        "memory_search_debug",
        "rag_context",
        "rag_hit_count",
        "rag_match_sources",
        "rag_search_debug",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.backend = str(config.get("backend", "redis")).lower()
        self.ttl = config.get("ttl", 3600)  # 1 hour
        self.max_size = config.get("max_size", "1GB")
        self.redis_config = dict(config.get("redis", {}))
        self.scope = str(config.get("scope", "user")).lower()
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        if not self.enabled:
            return

        if self.backend != "redis":
            logger.warning(
                "Unsupported response cache backend '%s'; disabling cache",
                self.backend,
            )
            self.enabled = False
            return

        try:
            password = None
            password_env = self.redis_config.get("password_env")
            if password_env:
                password = os.getenv(str(password_env))
            self.redis_client = redis.Redis(
                host=self.redis_config.get("host", "localhost"),
                port=int(self.redis_config.get("port", 6379)),
                db=int(self.redis_config.get("db", 0)),
                password=password,
                decode_responses=True,
            )
            await self.redis_client.ping()
            logger.info("Response cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}")
            self.enabled = False

    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        if not self.enabled or not self.redis_client:
            return None

        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                INFERENCE_METRICS.cache_hits.inc()
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        INFERENCE_METRICS.cache_misses.inc()
        return None

    async def cache_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache response data"""
        if not self.enabled or not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                cache_key, self.ttl, json.dumps(response_data)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def generate_cache_key(self, request: QueryRequest, model_name: str) -> str:
        """Generate cache key for request"""
        attachment_signatures = []
        for attachment in request.attachments:
            attachment_signatures.append(
                {
                    "name": attachment.name,
                    "type": attachment.type.value,
                    "size_bytes": attachment.size_bytes,
                    "mime_type": attachment.mime_type,
                    "url": attachment.url,
                    "content_sha1": hashlib.sha1(attachment.content).hexdigest()
                    if attachment.content
                    else None,
                }
            )
        rag_image_signatures = [
            {
                "figure_id": image.figure_id,
                "index_version": image.asset_ref.index_version,
                "checksum_sha256": (
                    image.content_checksum_sha256 or image.asset_ref.checksum_sha256
                ),
            }
            for image in getattr(request, "_rag_images", [])
        ]

        key_payload = {
            "cache_scope": self.scope,
            "model_name": model_name,
            "user_tier": request.user_tier.value,
            "query": request.query,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "priority": request.priority,
            "tool_policy": request.tool_policy.value,
            "allowed_tools": request.allowed_tools,
            "web_search_options": request.web_search_options.model_dump(mode="json")
            if request.web_search_options
            else None,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "metadata": self._cacheable_metadata(request.metadata or {}),
            "attachments": attachment_signatures,
            "rag_images": rag_image_signatures,
        }
        if self.scope != "shared":
            key_payload["user_id"] = request.user_id
        return hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _cacheable_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop analytics-only fields that should not fragment response cache keys."""
        return {
            key: value
            for key, value in metadata.items()
            if key not in self.ANALYTICS_METADATA_KEYS
        }


class RequestStageDeadlineError(Exception):
    """The request deadline passed in a step before any model was called.

    Retrieval or web search used up the time, so no model or provider is to
    blame: the attributes keep it out of provider health and map it to 504.
    """

    error_type = "internal"
    error_code = "timeout"
    http_status = 504


def _provider_http_timeout(read_seconds: float) -> httpx.Timeout:
    """Fail fast on connect and pool waits; allow the provider timeout per read."""
    return httpx.Timeout(connect=3.0, read=read_seconds, write=10.0, pool=2.0)


class BaseInferenceProvider(ABC):
    """Abstract base class for inference providers"""

    # Provider timeout when the config sets none (a dumped config holds None).
    DEFAULT_TIMEOUT_SECONDS = 60.0
    ATTACHMENT_PROMPT_CHAR_LIMIT = 12_000
    ATTACHMENT_PREVIEW_CHAR_LIMIT = 4_000
    TEXT_MIME_MARKERS = (
        "text/",
        "json",
        "csv",
        "xml",
        "yaml",
        "yml",
        "markdown",
        "javascript",
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

    def timeout_seconds(self) -> float:
        """Longest a single provider call may take, in seconds."""
        value = self.config.get("timeout") if isinstance(self.config, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return float(self.DEFAULT_TIMEOUT_SECONDS)
        return float(value)

    @abstractmethod
    async def initialize(self):
        """Initialize the provider"""
        pass

    @abstractmethod
    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response for the given request"""
        pass

    @abstractmethod
    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response for the given request"""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get provider health status"""
        pass

    def _build_prompt(self, request: QueryRequest) -> str:
        """Build a provider prompt that includes context plus extracted attachment content."""
        sections = self._build_prompt_sections(request)
        prompt_sections = [
            *sections.static_cache_prefix,
            *sections.dynamic_context,
            sections.query,
        ]
        return "\n\n".join(section for section in prompt_sections if section)

    def _build_prompt_sections(self, request: QueryRequest) -> PromptSections:
        """Split prompt into stable cache prefix, dynamic context, and user query."""
        static_cache_prefix = []
        dynamic_context = []

        response_style_instructions = (request.metadata or {}).get(
            "response_style_instructions"
        )
        if response_style_instructions:
            static_cache_prefix.append(
                f"Response style instructions:\n{response_style_instructions}"
            )

        attachment_prompt = self._build_attachment_prompt(request)
        if attachment_prompt:
            dynamic_context.append(attachment_prompt)
        if request.context:
            dynamic_context.append(f"Context: {request.context}")
        rag_context = (request.metadata or {}).get("rag_context")
        if rag_context:
            dynamic_context.append(str(rag_context))
        if (request.metadata or {}).get("rag_unavailable") in _RAG_UNAVAILABLE_STATUSES:
            dynamic_context.append(RAG_UNAVAILABLE_NOTICE)
        web_search_context = (request.metadata or {}).get("web_search_context")
        if web_search_context:
            dynamic_context.append(str(web_search_context))
        if (request.metadata or {}).get("web_search_unavailable") is True:
            dynamic_context.append(WEB_SEARCH_UNAVAILABLE_NOTICE)
        return PromptSections(
            static_cache_prefix=static_cache_prefix,
            dynamic_context=dynamic_context,
            query=f"Query: {request.query}",
        )

    def _prompt_cache_config(self) -> Dict[str, Any]:
        """Return provider prompt-cache config without assuming it exists."""
        return dict(self.config.get("prompt_cache", {}) or {})

    def _prompt_cache_enabled(self) -> bool:
        """Return whether provider-side prompt cache hints should be sent."""
        return bool(self._prompt_cache_config().get("enabled", False))

    def _prompt_cache_pricing(self) -> Dict[str, float]:
        """Return pricing multipliers for provider-side prompt cache accounting."""
        return dict(self._prompt_cache_config().get("pricing", {}) or {})

    def _build_attachment_prompt(self, request: QueryRequest) -> str:
        """Render attachment metadata and extracted text into the provider prompt."""
        if not request.attachments:
            return ""

        sections = [
            "Use the following attachment data when it is relevant to the user's request:"
        ]
        remaining_chars = self.ATTACHMENT_PROMPT_CHAR_LIMIT

        for index, attachment in enumerate(request.attachments, start=1):
            header = (
                f"[Attachment {index}] {attachment.name} "
                f"({attachment.mime_type}, {attachment.size_bytes} bytes)"
            )
            sections.append(header)

            extracted_text = self._extract_attachment_text(attachment)
            if not extracted_text:
                sections.append(
                    "Binary attachment metadata only. No text could be extracted automatically."
                )
                continue

            excerpt_limit = min(self.ATTACHMENT_PREVIEW_CHAR_LIMIT, remaining_chars)
            if excerpt_limit <= 0:
                sections.append(
                    "Additional extracted text omitted because the prompt limit was reached."
                )
                break

            excerpt = extracted_text[:excerpt_limit]
            if len(extracted_text) > excerpt_limit:
                excerpt = f"{excerpt}\n...[truncated]"
            sections.append(f"Content excerpt:\n{excerpt}")
            remaining_chars -= len(excerpt)

        return "\n\n".join(sections)

    def _extract_attachment_text(self, attachment) -> Optional[str]:
        """Decode text-like attachment content into a prompt-safe string."""
        if not attachment.content or not self._is_text_attachment(attachment):
            return None

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded = attachment.content.decode(encoding)
            except UnicodeDecodeError:
                continue

            normalized = decoded.strip()
            if not normalized:
                return None

            printable = sum(
                1
                for character in normalized
                if character.isprintable() or character in "\n\r\t"
            )
            if printable / max(len(normalized), 1) < 0.85:
                continue
            return normalized

        return None

    def _is_text_attachment(self, attachment) -> bool:
        """Return whether an attachment is safe to inline as text."""
        mime_type = (attachment.mime_type or "").lower()
        if attachment.type != AttachmentType.DOCUMENT:
            return False
        return mime_type.startswith("text/") or any(
            marker in mime_type for marker in self.TEXT_MIME_MARKERS
        )

    async def generate_batch_responses(
        self, requests: List[QueryRequest], model_name: str
    ) -> List[InferenceResponse]:
        """Generate responses for a batch of requests."""
        return await asyncio.gather(
            *(self.generate_response(request, model_name) for request in requests)
        )


class OpenAIProvider(BaseInferenceProvider):
    """OpenAI API provider implementation"""

    async def initialize(self):
        """Initialize OpenAI client"""
        api_key = self.config.get("api_key")
        api_key_env = self.config.get("api_key_env", "OPENAI_API_KEY")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
            timeout=_provider_http_timeout(self.timeout_seconds()),
            max_retries=0,
        )
        logger.info("OpenAI provider initialized")

    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)
            request_kwargs = self._build_response_request_kwargs(
                request,
                model_name,
                prompt,
            )
            response = await self.client.responses.create(**request_kwargs)

            usage = response.usage
            generated_text = response.output_text
            cached_tokens = _openai_cached_input_tokens(usage)
            reasoning_tokens = _nested_usage_value(
                usage, "output_tokens_details", "reasoning_tokens"
            )
            input_tokens = _usage_value(usage, "input_tokens")
            output_tokens = _usage_value(usage, "output_tokens")
            total_tokens = _usage_value(
                usage, "total_tokens", input_tokens + output_tokens
            )

            return InferenceResponse(
                response_text=generated_text,
                model_name=model_name,
                token_count_input=input_tokens,
                token_count_output=output_tokens,
                total_tokens=total_tokens,
                provider_cached_input_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=(output_tokens / max(time.time() - start_time, 1e-6)),
                cost_usd=self._calculate_cost(usage, model_name),
                provider="openai",
                cached=False,
                finish_reason=_openai_finish_reason(response),
                provider_request_id=_sdk_request_id(response),
            )

        except Exception as e:
            logger.error(f"OpenAI inference error: {e}")
            raise

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using OpenAI API"""
        try:
            prompt = self._build_prompt(request)
            request_kwargs = self._build_response_request_kwargs(
                request,
                model_name,
                prompt,
            )
            async with self.client.responses.stream(**request_kwargs) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    def _build_response_request_kwargs(
        self, request: QueryRequest, model_name: str, prompt: str
    ) -> Dict[str, Any]:
        """Build OpenAI Responses API kwargs, including optional prompt-cache hints."""
        provider_input: Any = prompt
        rag_images = getattr(request, "_rag_images", [])
        if rag_images:
            content: List[Dict[str, Any]] = []
            for image in rag_images:
                label = (
                    f"[S{image.source_rank}] Figure {image.figure_id}, "
                    f"page {image.page or 'unknown'}"
                )
                if image.caption:
                    label += f": {image.caption}"
                content.extend(
                    [
                        {"type": "input_text", "text": label},
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{image.asset_ref.media_type};base64,"
                                + base64.b64encode(image.content).decode("ascii")
                            ),
                        },
                    ]
                )
            content.append({"type": "input_text", "text": prompt})
            provider_input = [{"role": "user", "content": content}]
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": provider_input,
            "instructions": None,
            "max_output_tokens": request.max_tokens,
        }
        if self._supports_reasoning_effort(model_name):
            request_kwargs["reasoning"] = {"effort": "minimal"}
        if self._supports_temperature(model_name):
            request_kwargs["temperature"] = request.temperature

        prompt_cache_key = self._build_prompt_cache_key(request)
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = prompt_cache_key
            prompt_cache_retention = self._prompt_cache_retention()
            if prompt_cache_retention:
                request_kwargs["prompt_cache_retention"] = prompt_cache_retention
        else:
            request_kwargs["user"] = request.user_id

        return request_kwargs

    def _supports_temperature(self, model_name: str) -> bool:
        """Some reasoning models only support their default sampling settings."""
        return not model_name.startswith("gpt-5")

    def _supports_reasoning_effort(self, model_name: str) -> bool:
        """Keep GPT-5 short-answer requests from spending the full budget reasoning."""
        return model_name.startswith("gpt-5")

    def _build_prompt_cache_key(self, request: QueryRequest) -> Optional[str]:
        """Build a stable, non-PII key that helps OpenAI route similar prompts."""
        config = self._prompt_cache_config()
        if not bool(config.get("enabled", False)):
            return None

        strategy = str(config.get("key_strategy", "conversation")).lower()
        if strategy == "none":
            return None

        namespace = str(config.get("namespace", "llm-router"))
        parts = [namespace, strategy]
        if strategy == "shared":
            parts.append(str((request.metadata or {}).get("workspace_id", "shared")))
        elif strategy == "session":
            parts.extend(
                [
                    request.user_id,
                    request.session_id or request.conversation_id or "default",
                ]
            )
        elif strategy == "user":
            parts.append(request.user_id)
        else:
            parts.extend(
                [
                    request.user_id,
                    request.conversation_id or request.session_id or "default",
                ]
            )

        digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
        return f"{namespace}-{strategy}-{digest[:40]}"

    def _prompt_cache_retention(self) -> Optional[str]:
        """Return a valid OpenAI prompt cache retention hint, if configured."""
        config = self._prompt_cache_config()
        raw_retention = config.get("retention", "in_memory")
        if raw_retention is None:
            return None

        retention = str(raw_retention).strip().lower().replace("-", "_")
        if retention in {"in_memory", "24h"}:
            return retention

        logger.warning(
            "Ignoring unsupported OpenAI prompt_cache.retention value: %s",
            raw_retention,
        )
        return None

    def _calculate_cost(self, usage, model_name: str) -> float:
        """Calculate cost based on usage"""
        # OpenAI pricing (example rates - update with actual rates)
        pricing = {
            "gpt-5": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        }

        model_pricing = pricing.get(
            model_name, {"input": 0.001 / 1000, "output": 0.002 / 1000}
        )

        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        cached_tokens = min(_openai_cached_input_tokens(usage), input_tokens)
        uncached_tokens = max(input_tokens - cached_tokens, 0)
        cached_multiplier = float(
            self._prompt_cache_pricing().get("cached_input_multiplier", 0.5)
        )

        input_cost = (
            uncached_tokens * model_pricing["input"]
            + cached_tokens * model_pricing["input"] * cached_multiplier
        )
        output_cost = output_tokens * model_pricing["output"]

        return input_cost + output_cost

    def get_health_status(self) -> Dict[str, Any]:
        """Get OpenAI provider health status"""
        return {
            "provider": "openai",
            "status": "healthy" if self.client else "unhealthy",
            "models_available": ["gpt-5", "gpt-5.4", "gpt-3.5-turbo"],
        }


class AnthropicProvider(BaseInferenceProvider):
    """Anthropic API provider implementation"""

    async def initialize(self):
        """Initialize Anthropic client"""
        api_key = self.config.get("api_key")
        api_key_env = self.config.get("api_key_env", "ANTHROPIC_API_KEY")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self.config.get("base_url", "https://api.anthropic.com"),
            timeout=_provider_http_timeout(self.timeout_seconds()),
            max_retries=0,
        )
        logger.info("Anthropic provider initialized")

    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()

        try:
            messages = self._build_messages(request)

            # Make API call
            response = await self.client.messages.create(
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages,
            )

            # Extract all text blocks so future provider tool-use blocks do not break parsing.
            generated_text = "\n".join(
                str(getattr(block, "text", ""))
                for block in response.content
                if getattr(block, "text", None)
            )
            usage = response.usage
            uncached_input_tokens = _usage_value(usage, "input_tokens")
            cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
            cache_read_tokens = _usage_value(usage, "cache_read_input_tokens")
            total_input_tokens = (
                uncached_input_tokens + cache_creation_tokens + cache_read_tokens
            )
            output_tokens = _usage_value(usage, "output_tokens")

            return InferenceResponse(
                response_text=generated_text,
                model_name=model_name,
                token_count_input=total_input_tokens,
                token_count_output=output_tokens,
                total_tokens=total_input_tokens + output_tokens,
                provider_cached_input_tokens=cache_read_tokens,
                provider_cache_creation_input_tokens=cache_creation_tokens,
                provider_cache_read_input_tokens=cache_read_tokens,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=(output_tokens / max(time.time() - start_time, 1e-6)),
                cost_usd=self._calculate_cost(usage, model_name),
                provider="anthropic",
                cached=False,
                finish_reason=_anthropic_finish_reason(response),
                provider_request_id=_sdk_request_id(response),
            )

        except Exception as e:
            logger.error(f"Anthropic inference error: {e}")
            raise

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using Anthropic API"""
        try:
            messages = self._build_messages(request)

            async with self.client.messages.stream(
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    def _build_messages(self, request: QueryRequest) -> List[Dict[str, Any]]:
        """Build Anthropic messages with an optional cache breakpoint."""
        rag_images = getattr(request, "_rag_images", [])
        if rag_images:
            content: List[Dict[str, Any]] = []
            for image in rag_images:
                label = (
                    f"[S{image.source_rank}] Figure {image.figure_id}, "
                    f"page {image.page or 'unknown'}"
                )
                if image.caption:
                    label += f": {image.caption}"
                content.extend(
                    [
                        {"type": "text", "text": label},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.asset_ref.media_type,
                                "data": base64.b64encode(image.content).decode("ascii"),
                            },
                        },
                    ]
                )
            content.append({"type": "text", "text": self._build_prompt(request)})
            return [{"role": "user", "content": content}]
        sections = self._build_prompt_sections(request)
        if not self._prompt_cache_enabled() or not sections.static_cache_prefix:
            return [{"role": "user", "content": self._build_prompt(request)}]

        content = [
            {
                "type": "text",
                "text": "\n\n".join(sections.static_cache_prefix),
                "cache_control": self._anthropic_cache_control(),
            }
        ]
        if sections.dynamic_context:
            content.append(
                {
                    "type": "text",
                    "text": "\n\n".join(sections.dynamic_context),
                }
            )
        content.append({"type": "text", "text": sections.query})
        return [{"role": "user", "content": content}]

    def _anthropic_cache_control(self) -> Dict[str, str]:
        """Build Anthropic cache_control while only allowing supported TTLs."""
        config = self._prompt_cache_config()
        cache_control = dict(config.get("cache_control", {"type": "ephemeral"}) or {})
        if not cache_control.get("type"):
            cache_control["type"] = "ephemeral"

        ttl = config.get("ttl", cache_control.get("ttl"))
        if ttl is None:
            return cache_control

        normalized_ttl = str(ttl).strip().lower()
        if normalized_ttl in {"5m", "1h"}:
            if normalized_ttl == "1h":
                cache_control["ttl"] = normalized_ttl
            else:
                cache_control.pop("ttl", None)
            return cache_control

        logger.warning(
            "Ignoring unsupported Anthropic prompt_cache.ttl value: %s",
            ttl,
        )
        cache_control.pop("ttl", None)
        return cache_control

    def _calculate_cost(self, usage, model_name: str) -> float:
        """Calculate cost based on usage"""
        # Anthropic pricing, aligned to current Claude 4 family defaults.
        pricing = {
            "claude-sonnet-4-6": {"input": 0.003 / 1000, "output": 0.015 / 1000},
            "claude-opus-4-6": {"input": 0.005 / 1000, "output": 0.025 / 1000},
        }

        model_pricing = pricing.get(
            model_name, {"input": 0.003 / 1000, "output": 0.015 / 1000}
        )

        uncached_input_tokens = _usage_value(usage, "input_tokens")
        cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
        cache_read_tokens = _usage_value(usage, "cache_read_input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        pricing_config = self._prompt_cache_pricing()
        cache_creation_multiplier = float(
            pricing_config.get("cache_creation_input_multiplier", 1.25)
        )
        cache_read_multiplier = float(
            pricing_config.get("cache_read_input_multiplier", 0.1)
        )

        input_cost = (
            uncached_input_tokens * model_pricing["input"]
            + cache_creation_tokens * model_pricing["input"] * cache_creation_multiplier
            + cache_read_tokens * model_pricing["input"] * cache_read_multiplier
        )
        output_cost = output_tokens * model_pricing["output"]

        return input_cost + output_cost

    def get_health_status(self) -> Dict[str, Any]:
        """Get Anthropic provider health status"""
        return {
            "provider": "anthropic",
            "status": "healthy" if self.client else "unhealthy",
            "models_available": ["claude-sonnet-4-6", "claude-opus-4-6"],
        }


class vLLMProvider(BaseInferenceProvider):
    """vLLM self-hosted provider implementation"""

    DEFAULT_TIMEOUT_SECONDS = 300.0  # Local inference may take longer.

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoints = self._build_endpoint_states(config)
        self.base_url = self.endpoints[0].base_url
        self.api_mode = str(config.get("api_mode", "completions")).lower()
        self.routing_strategy = str(
            config.get("routing_strategy", "least_outstanding_prefix_aware")
        ).lower()
        self.health_check_interval_seconds = float(
            config.get("health_check_interval_seconds", 15)
        )
        self.failure_cooldown_seconds = float(
            config.get("failure_cooldown_seconds", 30)
        )
        self.metrics_refresh_seconds = float(config.get("metrics_refresh_seconds", 5))
        self.prefix_affinity_ttl_seconds = float(
            config.get("prefix_affinity_ttl_seconds", 300)
        )
        self.metrics_scrape_enabled = bool(config.get("metrics_scrape_enabled", False))
        self.validate_models_on_health = bool(
            config.get("validate_models_on_health", False)
        )
        self.http_client = None
        self._pool_lock = asyncio.Lock()
        self._prefix_affinity: Dict[str, Tuple[str, float]] = {}

    async def initialize(self):
        """Initialize vLLM clients and run endpoint health checks."""
        for endpoint in self.endpoints:
            if not endpoint.enabled:
                endpoint.healthy = False
                endpoint.last_error = "Endpoint disabled"
                continue

            # Connect fast, but let requests wait for a pooled connection and
            # send their body as long as they may wait for a reply: one
            # endpoint serves many concurrent generations.
            timeout_seconds = self.timeout_seconds()
            endpoint.client = httpx.AsyncClient(
                base_url=endpoint.base_url,
                timeout=httpx.Timeout(
                    connect=3.0,
                    read=timeout_seconds,
                    write=timeout_seconds,
                    pool=timeout_seconds,
                ),
            )
            if self.http_client is None:
                self.http_client = endpoint.client

            await self._refresh_endpoint_health(endpoint, force=True)
            await self._refresh_endpoint_metrics(endpoint, force=True)

        healthy_count = sum(
            1 for endpoint in self.endpoints if endpoint.enabled and endpoint.healthy
        )
        if healthy_count:
            logger.info(
                "vLLM provider initialized with %s healthy endpoint(s)",
                healthy_count,
            )
        else:
            logger.warning("vLLM provider initialized with no healthy endpoints")

    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using vLLM"""
        start_time = time.time()
        prompt = self._build_prompt(request)
        endpoint_path, payload = self._build_request_payload(
            prompt, model_name, request, stream=False
        )
        prefix_key = self._prompt_prefix_fingerprint(model_name, prompt)
        excluded_endpoints: set[str] = set()
        last_error: Optional[Exception] = None
        attempts = max(1, self._candidate_endpoint_count(model_name))

        for _ in range(attempts):
            endpoint = await self._reserve_endpoint(
                model_name, prefix_key, excluded_endpoints
            )

            try:
                client = self._endpoint_client(endpoint)
                if client is None:
                    raise RuntimeError(f"vLLM endpoint {endpoint.name} has no client")

                response = await client.post(endpoint_path, json=payload)
                response.raise_for_status()

                data = response.json()
                generated_text = self._extract_response_text(data)
                usage = data.get("usage", {})
                await self._mark_endpoint_success(endpoint, prefix_key)

                return InferenceResponse(
                    response_text=generated_text,
                    model_name=model_name,
                    token_count_input=usage.get("prompt_tokens", 0),
                    token_count_output=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0)
                    or (
                        usage.get("prompt_tokens", 0)
                        + usage.get("completion_tokens", 0)
                    ),
                    latency_ms=int((time.time() - start_time) * 1000),
                    tokens_per_second=(
                        usage.get("completion_tokens", 0)
                        / max(time.time() - start_time, 1e-6)
                    ),
                    cost_usd=0.0,  # Self-hosted models have no API cost
                    provider="vllm",
                    cached=False,
                    finish_reason=_vllm_finish_reason(data),
                    provider_request_id=_header_value(
                        getattr(response, "headers", None), "x-request-id"
                    ),
                )

            except Exception as e:
                if self._is_request_rejection(e):
                    logger.warning(
                        "vLLM endpoint %s rejected the request: %s", endpoint.name, e
                    )
                    raise
                last_error = e
                excluded_endpoints.add(endpoint.name)
                await self._mark_endpoint_failure(endpoint, e)
                logger.error(
                    "vLLM inference error on endpoint %s: %s", endpoint.name, e
                )

            finally:
                await self._release_endpoint(endpoint)

        if last_error:
            raise last_error
        raise RuntimeError(f"No healthy vLLM endpoint can serve model {model_name}")

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using vLLM"""
        endpoint: Optional[VLLMEndpointState] = None
        try:
            prompt = self._build_prompt(request)
            endpoint_path, payload = self._build_request_payload(
                prompt, model_name, request, stream=True
            )
            prefix_key = self._prompt_prefix_fingerprint(model_name, prompt)
            endpoint = await self._reserve_endpoint(model_name, prefix_key, set())
            client = self._endpoint_client(endpoint)
            if client is None:
                raise RuntimeError(f"vLLM endpoint {endpoint.name} has no client")

            async with client.stream("POST", endpoint_path, json=payload) as response:
                if response.status_code >= 400:
                    # Read the error body so the HTTPStatusError carries it.
                    await response.aread()
                    response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw_data = line[6:]
                    if raw_data == "[DONE]":
                        break
                    data = json.loads(raw_data)
                    delta = self._extract_stream_delta(data)
                    if delta:
                        yield delta
            await self._mark_endpoint_success(endpoint, prefix_key)

        except Exception as e:
            if self._is_request_rejection(e):
                logger.warning("vLLM endpoint rejected the stream request: %s", e)
                raise
            if endpoint is not None:
                await self._mark_endpoint_failure(endpoint, e)
            logger.error(f"vLLM streaming error: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                # str(e) names the internal endpoint URL; the chunk reaches
                # the user, so it carries only the status.
                yield f"Error: upstream returned HTTP {e.response.status_code}"
            else:
                yield f"Error: {str(e)}"

        finally:
            if endpoint is not None:
                await self._release_endpoint(endpoint)

    def _build_endpoint_states(self, config: Dict[str, Any]) -> List[VLLMEndpointState]:
        endpoint_configs = config.get("endpoints") or []
        if not endpoint_configs:
            endpoint_configs = [
                {
                    "name": "default",
                    "base_url": self._configured_base_url(config),
                    "models": config.get("models_available") or [],
                    "weight": config.get("weight", 1.0),
                    "max_outstanding": config.get("max_outstanding", 0),
                    "health_path": config.get("health_path", "/health"),
                    "metrics_path": config.get("metrics_path", "/metrics"),
                    "prefix_cache_enabled": config.get("prefix_cache_enabled", True),
                    "enabled": True,
                }
            ]

        endpoints: List[VLLMEndpointState] = []
        used_names: set[str] = set()
        for index, endpoint_config in enumerate(endpoint_configs):
            endpoint_dict = dict(endpoint_config or {})
            name = str(endpoint_dict.get("name") or f"endpoint-{index + 1}")
            if name in used_names:
                name = f"{name}-{index + 1}"
            used_names.add(name)
            endpoints.append(
                VLLMEndpointState(
                    name=name,
                    base_url=self._configured_base_url(endpoint_dict).rstrip("/"),
                    models=self._normalize_model_list(endpoint_dict.get("models", [])),
                    weight=max(float(endpoint_dict.get("weight", 1.0)), 0.001),
                    max_outstanding=max(
                        int(endpoint_dict.get("max_outstanding", 0)), 0
                    ),
                    health_path=self._normalize_path(
                        endpoint_dict.get("health_path", "/health")
                    ),
                    metrics_path=self._normalize_path(
                        endpoint_dict.get("metrics_path", "/metrics")
                    ),
                    prefix_cache_enabled=bool(
                        endpoint_dict.get("prefix_cache_enabled", True)
                    ),
                    enabled=bool(endpoint_dict.get("enabled", True)),
                )
            )

        return endpoints or [
            VLLMEndpointState(
                name="default", base_url=self._configured_base_url(config)
            )
        ]

    def _configured_base_url(self, config: Dict[str, Any]) -> str:
        configured_base_url = config.get("base_url")
        if configured_base_url:
            return str(configured_base_url).rstrip("/")
        return f"http://{config.get('host', 'localhost')}:{config.get('port', 8000)}"

    def _normalize_model_list(self, models: Any) -> List[str]:
        if models is None:
            return []
        if isinstance(models, str):
            return [models]
        if isinstance(models, Sequence):
            return [str(model) for model in models]
        return []

    def _normalize_path(self, path: Any) -> str:
        path_value = str(path or "/")
        return path_value if path_value.startswith("/") else f"/{path_value}"

    def _build_request_payload(
        self,
        prompt: str,
        model_name: str,
        request: QueryRequest,
        *,
        stream: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        if self.api_mode == "chat_completions":
            return (
                "/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": self._build_chat_messages(prompt),
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": stream,
                },
            )

        return (
            "/v1/completions",
            {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": stream,
                "logprobs": None,
            },
        )

    def _build_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build OpenAI-compatible chat messages for vLLM chat templates."""
        return [{"role": "user", "content": prompt}]

    def _prompt_prefix_fingerprint(self, model_name: str, prompt: str) -> str:
        prefix = prompt[:4096]
        payload = f"{model_name}\n{prefix}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _reserve_endpoint(
        self,
        model_name: str,
        prefix_key: str,
        excluded_names: set[str],
    ) -> VLLMEndpointState:
        await self._refresh_pool_state()
        async with self._pool_lock:
            endpoint = self._select_endpoint_locked(
                model_name, prefix_key, excluded_names
            )
            endpoint.outstanding += 1
            return endpoint

    async def _release_endpoint(self, endpoint: VLLMEndpointState):
        async with self._pool_lock:
            endpoint.outstanding = max(endpoint.outstanding - 1, 0)

    async def _mark_endpoint_success(
        self, endpoint: VLLMEndpointState, prefix_key: str
    ):
        async with self._pool_lock:
            endpoint.healthy = True
            endpoint.last_error = None
            endpoint.cooldown_until = 0.0
            if endpoint.prefix_cache_enabled and self._prefix_affinity_enabled():
                self._prefix_affinity[prefix_key] = (
                    endpoint.name,
                    time.time() + self.prefix_affinity_ttl_seconds,
                )

    @staticmethod
    def _is_request_rejection(error: Exception) -> bool:
        """400, 413 and 422 fault the request itself, not the endpoint.

        Every endpoint would reject it the same way (too many tokens, bad
        parameters), so it is neither retried elsewhere nor counted against
        the endpoint's health. Other 4xx can belong to one endpoint: a 404
        from one that does not load the model or has a wrong base_url path,
        401/403 from its credentials. Those fail over like a 5xx.
        """
        if not isinstance(error, httpx.HTTPStatusError):
            return False
        return error.response.status_code in REQUEST_SHAPED_STATUS_CODES

    async def _mark_endpoint_failure(
        self, endpoint: VLLMEndpointState, error: Exception
    ):
        async with self._pool_lock:
            endpoint.healthy = False
            endpoint.last_error = str(error)
            endpoint.cooldown_until = time.time() + self.failure_cooldown_seconds

    def _select_endpoint_locked(
        self,
        model_name: str,
        prefix_key: str,
        excluded_names: set[str],
    ) -> VLLMEndpointState:
        now = time.time()
        self._prune_prefix_affinity(now)

        if self._prefix_affinity_enabled():
            affinity = self._prefix_affinity.get(prefix_key)
            if affinity:
                endpoint_name, _ = affinity
                endpoint = self._endpoint_by_name(endpoint_name)
                if (
                    endpoint
                    and endpoint.name not in excluded_names
                    and endpoint.prefix_cache_enabled
                    and self._endpoint_available(endpoint, model_name, now)
                    and not self._endpoint_saturated(endpoint)
                ):
                    return endpoint

        candidates = [
            endpoint
            for endpoint in self.endpoints
            if endpoint.name not in excluded_names
            and self._endpoint_available(endpoint, model_name, now)
        ]
        unsaturated = [
            endpoint
            for endpoint in candidates
            if not self._endpoint_saturated(endpoint)
        ]
        if unsaturated:
            return min(unsaturated, key=self._endpoint_score)
        if candidates:
            return min(candidates, key=self._endpoint_score)
        raise RuntimeError(f"No healthy vLLM endpoint can serve model {model_name}")

    def _endpoint_available(
        self, endpoint: VLLMEndpointState, model_name: str, now: float
    ) -> bool:
        if not endpoint.enabled:
            return False
        if not self._endpoint_accepts_model(endpoint, model_name):
            return False
        if endpoint.healthy:
            return True
        return endpoint.cooldown_until <= now

    def _endpoint_accepts_model(
        self, endpoint: VLLMEndpointState, model_name: str
    ) -> bool:
        return not endpoint.models or model_name in endpoint.models

    def _endpoint_saturated(self, endpoint: VLLMEndpointState) -> bool:
        return bool(
            endpoint.max_outstanding
            and endpoint.outstanding >= endpoint.max_outstanding
        )

    def _endpoint_score(self, endpoint: VLLMEndpointState) -> float:
        load = endpoint.outstanding + endpoint.queue_depth + endpoint.running_requests
        return load / max(endpoint.weight, 0.001)

    def _endpoint_by_name(self, name: str) -> Optional[VLLMEndpointState]:
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        return None

    def _endpoint_client(self, endpoint: VLLMEndpointState) -> Optional[Any]:
        return endpoint.client or self.http_client

    def _candidate_endpoint_count(self, model_name: str) -> int:
        count = sum(
            1
            for endpoint in self.endpoints
            if endpoint.enabled and self._endpoint_accepts_model(endpoint, model_name)
        )
        return count or 1

    def _prefix_affinity_enabled(self) -> bool:
        return (
            "prefix" in self.routing_strategy and self.prefix_affinity_ttl_seconds > 0
        )

    def _prune_prefix_affinity(self, now: float):
        expired_keys = [
            prefix_key
            for prefix_key, (_, expires_at) in self._prefix_affinity.items()
            if expires_at <= now
        ]
        for prefix_key in expired_keys:
            self._prefix_affinity.pop(prefix_key, None)

    async def _refresh_pool_state(self):
        for endpoint in self.endpoints:
            await self._refresh_endpoint_health(endpoint)
            await self._refresh_endpoint_metrics(endpoint)

    async def _refresh_endpoint_health(
        self, endpoint: VLLMEndpointState, *, force: bool = False
    ):
        if not endpoint.enabled:
            return
        if endpoint.client is None:
            return

        now = time.time()
        if (
            not force
            and endpoint.last_health_check > 0
            and now - endpoint.last_health_check < self.health_check_interval_seconds
        ):
            return
        if not force and endpoint.last_health_check <= 0:
            return

        endpoint.last_health_check = now
        try:
            response = await endpoint.client.get(endpoint.health_path)
            if response.status_code != 200:
                raise RuntimeError(f"health check returned {response.status_code}")

            if self.validate_models_on_health:
                await self._validate_endpoint_models(endpoint)

            endpoint.healthy = True
            endpoint.last_error = None
            endpoint.cooldown_until = 0.0
        except Exception as e:
            endpoint.healthy = False
            endpoint.last_error = str(e)
            endpoint.cooldown_until = time.time() + self.failure_cooldown_seconds
            logger.warning("vLLM endpoint %s health check failed: %s", endpoint.name, e)

    async def _validate_endpoint_models(self, endpoint: VLLMEndpointState):
        if not endpoint.models:
            return
        response = await endpoint.client.get("/v1/models")
        if response.status_code != 200:
            raise RuntimeError(f"model list returned {response.status_code}")
        data = response.json()
        model_ids = {
            str(item.get("id"))
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
        missing = [model for model in endpoint.models if model not in model_ids]
        if missing:
            raise RuntimeError(
                f"endpoint missing configured model(s): {', '.join(missing)}"
            )

    async def _refresh_endpoint_metrics(
        self, endpoint: VLLMEndpointState, *, force: bool = False
    ):
        if not self.metrics_scrape_enabled or not endpoint.enabled:
            return
        if endpoint.client is None:
            return

        now = time.time()
        if (
            not force
            and endpoint.last_metrics_refresh > 0
            and now - endpoint.last_metrics_refresh < self.metrics_refresh_seconds
        ):
            return
        if not force and endpoint.last_metrics_refresh <= 0:
            return

        endpoint.last_metrics_refresh = now
        try:
            response = await endpoint.client.get(endpoint.metrics_path)
            if response.status_code != 200:
                raise RuntimeError(f"metrics returned {response.status_code}")
            metrics = self._parse_vllm_metrics(response.text)
            endpoint.running_requests = metrics["running_requests"]
            endpoint.queue_depth = metrics["queue_depth"]
            endpoint.kv_cache_usage = metrics["kv_cache_usage"]
        except Exception as e:
            endpoint.last_error = str(e)
            logger.warning(
                "vLLM endpoint %s metrics scrape failed: %s", endpoint.name, e
            )

    def _parse_vllm_metrics(self, metrics_text: str) -> Dict[str, Any]:
        running_requests = 0
        queue_depth = 0
        kv_cache_usage: Optional[float] = None

        for raw_line in metrics_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_name = parts[0].split("{", 1)[0]
            try:
                value = float(parts[-1])
            except ValueError:
                continue

            if metric_name == "vllm:num_requests_running":
                running_requests += int(value)
            elif metric_name == "vllm:num_requests_waiting":
                queue_depth += int(value)
            elif metric_name == "vllm:gpu_cache_usage_perc":
                kv_cache_usage = max(kv_cache_usage or 0.0, value)

        return {
            "running_requests": running_requests,
            "queue_depth": queue_depth,
            "kv_cache_usage": kv_cache_usage,
        }

    def _extract_response_text(self, data: Dict[str, Any]) -> str:
        choice = data["choices"][0]
        if self.api_mode == "chat_completions":
            return self._coerce_chat_content(choice.get("message", {}))
        return str(choice.get("text", ""))

    def _extract_stream_delta(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if self.api_mode == "chat_completions":
            return self._coerce_chat_content(choice.get("delta", {}))
        return str(choice.get("text", ""))

    def _coerce_chat_content(self, message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def get_health_status(self) -> Dict[str, Any]:
        """Get vLLM provider health status"""
        endpoint_statuses = [
            {
                "name": endpoint.name,
                "base_url": endpoint.base_url,
                "enabled": endpoint.enabled,
                "healthy": endpoint.healthy,
                "outstanding": endpoint.outstanding,
                "queue_depth": endpoint.queue_depth,
                "running_requests": endpoint.running_requests,
                "kv_cache_usage": endpoint.kv_cache_usage,
                "models": endpoint.models,
                "last_error": endpoint.last_error,
                "cooldown_until": endpoint.cooldown_until,
            }
            for endpoint in self.endpoints
        ]
        pool_has_client = bool(self.http_client) or any(
            endpoint.client for endpoint in self.endpoints
        )
        pool_healthy = any(
            endpoint.enabled and endpoint.healthy for endpoint in self.endpoints
        )
        return {
            "provider": "vllm",
            "status": "healthy" if pool_has_client and pool_healthy else "unhealthy",
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "models_available": self.config.get(
                "models_available", ["qwen3.6-27b-fast"]
            ),
            "routing_strategy": self.routing_strategy,
            "endpoints": endpoint_statuses,
        }

    async def aclose(self):
        """Close all endpoint HTTP clients."""
        closed_clients = set()
        for endpoint in self.endpoints:
            client = endpoint.client
            if client and id(client) not in closed_clients:
                await client.aclose()
                closed_clients.add(id(client))
        if self.http_client and id(self.http_client) not in closed_clients:
            await self.http_client.aclose()


class SingleFlightCoordinator:
    """Coalesce identical in-flight calls without delaying distinct work."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled", False)
        self.inflight_requests: Dict[Tuple[int, str, str], asyncio.Task] = {}
        self.batch_lock = asyncio.Lock()

    async def add_request(
        self, request: QueryRequest, provider: BaseInferenceProvider, model_name: str
    ) -> InferenceResponse:
        return await self.run(
            request,
            provider,
            model_name,
            lambda: provider.generate_response(request, model_name),
        )

    async def run(
        self,
        request: QueryRequest,
        provider: Any,
        model_name: str,
        operation: Callable[[], Awaitable[InferenceResponse]],
    ) -> InferenceResponse:
        """Run a leader immediately and shield it from follower cancellation."""
        if not self.enabled:
            return await operation()

        coalesce_key = self._build_coalesce_key(request, model_name)
        inflight_key = (id(provider), model_name, coalesce_key)

        async with self.batch_lock:
            shared_future = self.inflight_requests.get(inflight_key)
            if shared_future is None:
                shared_future = asyncio.create_task(operation())
                self.inflight_requests[inflight_key] = shared_future
                shared_future.add_done_callback(
                    lambda completed, key=inflight_key: self._on_completed(
                        key, completed
                    )
                )

        try:
            response = await asyncio.shield(shared_future)
            return response.model_copy(deep=True)
        finally:
            if shared_future.done():
                await self._remove_completed(inflight_key, shared_future)

    async def _remove_completed(self, key: Tuple[int, str, str], task: asyncio.Task):
        if task.done() and not task.cancelled():
            task.exception()
        async with self.batch_lock:
            if self.inflight_requests.get(key) is task:
                self.inflight_requests.pop(key, None)

    def _on_completed(self, key: Tuple[int, str, str], task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()
        if self.inflight_requests.get(key) is task:
            self.inflight_requests.pop(key, None)

    def _build_coalesce_key(self, request: QueryRequest, model_name: str) -> str:
        """Build a stable signature for in-flight request coalescing."""
        attachment_signatures = []
        for attachment in request.attachments:
            attachment_signatures.append(
                {
                    "name": attachment.name,
                    "type": attachment.type.value,
                    "size_bytes": attachment.size_bytes,
                    "mime_type": attachment.mime_type,
                    "url": attachment.url,
                    "content_sha1": hashlib.sha1(attachment.content).hexdigest()
                    if attachment.content
                    else None,
                }
            )
        rag_image_signatures = [
            {
                "figure_id": image.figure_id,
                "index_version": image.asset_ref.index_version,
                "checksum_sha256": (
                    image.content_checksum_sha256 or image.asset_ref.checksum_sha256
                ),
            }
            for image in getattr(request, "_rag_images", [])
        ]

        payload = {
            "model_name": model_name,
            "user_id": request.user_id,
            "user_tier": request.user_tier.value,
            "query": request.query,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "priority": request.priority,
            "tool_policy": request.tool_policy.value,
            "allowed_tools": request.allowed_tools,
            "web_search_options": request.web_search_options.model_dump(mode="json")
            if request.web_search_options
            else None,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "metadata": request.metadata,
            "attachments": attachment_signatures,
            "rag_images": rag_image_signatures,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    async def shutdown(self):
        """Wait for leaders that are still running."""
        async with self.batch_lock:
            tasks = list(self.inflight_requests.values())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


BatchProcessor = SingleFlightCoordinator


class InferenceEngine:
    """Main inference engine coordinating all providers"""

    def __init__(
        self,
        config: Dict[str, Any],
        router: ModelRouter,
        event_producer: Optional[Any] = None,
        rag_service: Optional[Any] = None,
        admission_controller: Optional[Any] = None,
    ):
        self.config = config
        self.router = router
        self.event_producer = event_producer
        self.rag_service = rag_service
        self.admission_controller = admission_controller
        self.providers = {}
        self.context_compressor = ContextCompressor(config.get("compression", {}))
        self.cache = ResponseCache(config.get("cache", {}))
        single_flight_config = config.get("single_flight")
        if single_flight_config is None:
            single_flight_config = config.get("batching", {})
            if single_flight_config:
                logger.warning(
                    "inference.batching is deprecated; use inference.single_flight"
                )
        elif config.get("batching", {}).get("enabled"):
            logger.warning(
                "inference.batching is deprecated; use inference.single_flight"
            )
        self.single_flight = SingleFlightCoordinator(single_flight_config or {})
        self.batch_processor = self.single_flight
        self.scheduler = ProviderCapacityScheduler(
            config.get("scheduler", {}),
            admission_controller=admission_controller,
        )
        self.tool_registry = ToolRegistry()
        self.web_search_tool: Optional[WebSearchTool] = None

        # Performance tracking
        self.inference_stats = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all providers and components"""
        logger.info("Initializing inference engine...")

        # Initialize context compressor
        await self.context_compressor.initialize()

        # Initialize cache
        await self.cache.initialize()

        tools_config = dict(self.config.get("tools", {}) or {})
        web_search_config = dict(tools_config.get("web_search", {}) or {})
        if web_search_config:
            self.web_search_tool = WebSearchTool(web_search_config)
            self.tool_registry.register(self.web_search_tool)

        # Initialize providers based on configuration
        openai_config = dict(self.config.get("openai", {}) or {})
        if (
            "openai" in self.config
            and openai_config.get("enabled", True)
            and self._router_uses_provider("openai")
        ):
            self.providers["openai"] = OpenAIProvider(openai_config)
            await self.providers["openai"].initialize()

        anthropic_config = dict(self.config.get("anthropic", {}) or {})
        if (
            "anthropic" in self.config
            and anthropic_config.get("enabled", True)
            and self._router_uses_provider("anthropic")
        ):
            self.providers["anthropic"] = AnthropicProvider(anthropic_config)
            await self.providers["anthropic"].initialize()

        vllm_config = dict(self.config.get("vllm", {}) or {})
        if self._vllm_configured(vllm_config):
            self.providers["vllm"] = vLLMProvider(vllm_config)
            await self.providers["vllm"].initialize()

        self._initialized = True
        logger.info(
            f"Inference engine initialized with {len(self.providers)} providers"
        )

    def _router_uses_provider(self, provider: str) -> bool:
        """Return whether the router has a concrete model for a provider."""
        models = getattr(self.router, "models", None)
        if not isinstance(models, dict) or not models:
            return True
        return any(
            str(getattr(model, "provider", "")).lower() == provider
            for model in models.values()
        )

    def _vllm_configured(self, config: Dict[str, Any]) -> bool:
        """Return whether a usable vLLM endpoint has been configured."""
        if not config or not config.get("enabled", True):
            return False

        if config.get("base_url"):
            return True

        if config.get("endpoints"):
            return True

        return bool(config.get("host")) and bool(config.get("port"))

    def is_healthy(self) -> bool:
        """Return whether the inference engine is ready to process requests."""
        return self._initialized and bool(self.providers)

    async def _compress_request_context_if_needed(self, request: QueryRequest) -> bool:
        """Canonicalize long context before cache keys are generated."""
        if not (
            request.context
            and len(request.context) > self.context_compressor.max_context_tokens
        ):
            return False

        original_length = len(request.context)
        target_length = int(original_length * self.context_compressor.compression_ratio)
        request.context = await self.context_compressor.compress_context(
            request.context,
            target_length,
        )
        logger.info(
            f"Compressed context from {original_length} to {len(request.context)} chars"
        )
        return True

    async def _get_cached_inference_response(
        self, request: QueryRequest, model_name: str
    ) -> Optional[InferenceResponse]:
        """Return an app-cache hit for a model, clearly separated from provider cache.

        The cached payload carries the original call's cost, latency and timestamp.
        A hit costs nothing, takes only the lookup time and happens now, so those
        fields are reset before every hit path publishes analytics.
        """
        if self._response_cache_bypassed(request):
            return None
        lookup_started_at = time.time()
        cache_key = self.cache.generate_cache_key(request, model_name)
        cached_response = await self.cache.get_cached_response(cache_key)
        if not cached_response:
            return None

        return InferenceResponse(
            **{
                **cached_response,
                "cached": True,
                "cost_usd": 0.0,
                "latency_ms": int((time.time() - lookup_started_at) * 1000),
                "timestamp": datetime.now(timezone.utc),
                # The fallback that produced the cached answer is history.
                "fallback_from_model": None,
                "attempts": 0,
            }
        )

    def _response_cache_bypassed(self, request: QueryRequest) -> bool:
        metadata = request.metadata or {}
        return bool(
            metadata.get("web_search_cache_bypass", False)
            or metadata.get("rag_cache_bypass", False)
        )

    async def process_query(self, request: QueryRequest) -> InferenceResponse:
        """Process query through the complete inference pipeline"""
        start_time = time.time()

        # Log with the caller's request id so logs join the analytics events.
        request_id = request.request_id
        # One deadline for the whole request: time spent on retrieval comes out
        # of the model calls' share instead of extending the request.
        execution_budget = (
            self.scheduler.create_execution_budget()
            if "scheduler" in self.config
            else None
        )

        try:
            # Step 1: Route query to appropriate model
            routing_decision = await self.router.route_query(request)
            model_name = routing_decision.selected_model

            # Step 2: Context compression before cache lookup keeps keys stable.
            compressed_context = await self._compress_request_context_if_needed(request)

            # Step 3: Optionally enrich school-document requests with RAG context.
            # Retrieval gets a share of the time left, so its own outcome
            # handling runs; the request deadline is only the backstop.
            rag_result = await self._within_request_deadline(
                self._run_rag_if_needed(
                    request,
                    routing_decision,
                    self._retrieval_time_limit(execution_budget),
                ),
                execution_budget,
                stage="rag",
            )
            if rag_result and rag_result.required_no_hits:
                response = self._build_required_rag_no_hit_response(
                    request,
                    model_name,
                    rag_result,
                    start_time,
                )
                await self._finalize_successful_response(
                    request,
                    response,
                    model_name,
                    routing_decision,
                    start_time,
                )
                return response
            routing_decision = await self._prepare_rag_images(
                request, routing_decision, rag_result
            )
            model_name = routing_decision.selected_model

            # Step 4: Optionally enrich current-info requests with web search.
            web_search_result = await self._within_request_deadline(
                self._run_web_search_if_needed(
                    request,
                    routing_decision,
                    self._retrieval_time_limit(execution_budget),
                ),
                execution_budget,
                stage="web_search",
            )

            # Step 5: Check full-response cache.
            response = await self._get_cached_inference_response(request, model_name)
            if response:
                logger.info(f"Cache hit for request {request_id}")
                await self._publish_analytics_events(
                    request,
                    response,
                    routing_decision=routing_decision,
                )
                return response

            # Step 6: Generate response, falling back to local models for cloud outages
            (
                response,
                model_name,
                routing_decision,
            ) = await self._generate_response_with_fallback(
                request,
                routing_decision,
                execution_budget=execution_budget,
            )
            response.compressed_context = compressed_context
            self._attach_rag_result(response, rag_result)
            self._attach_web_search_result(response, web_search_result)

            response_complete = self._flag_incomplete_response(request, response)

            # Step 7: Cache response; truncated, filtered or empty answers are
            # returned but not cached, so a retry can do better.
            if (
                response_complete
                and not response.cached
                and not self._response_cache_bypassed(request)
            ):
                cache_key = self.cache.generate_cache_key(request, model_name)
                # A cache hit makes no provider call: no provider request id.
                await self.cache.cache_response(
                    cache_key,
                    response.model_dump(mode="json", exclude={"provider_request_id"}),
                )

            await self._finalize_successful_response(
                request,
                response,
                model_name,
                routing_decision,
                start_time,
            )
            return response

        except AdmissionRejectedError:
            raise
        except Exception as exc:
            # Update error metrics
            if not getattr(exc, "_attempt_recorded", False):
                INFERENCE_METRICS.errors_total.labels(
                    model=routing_decision.selected_model
                    if "routing_decision" in locals()
                    else "unknown",
                    error_type=type(exc).__name__,
                ).inc()

            # Update router stats; a retrieval outage or a deadline spent
            # before any model call is not the model's fault.
            if (
                "routing_decision" in locals()
                and not getattr(exc, "_attempt_recorded", False)
                and not isinstance(
                    exc, (RetrievalUnavailableError, RequestStageDeadlineError)
                )
            ):
                self.router.update_model_stats(
                    model_name=routing_decision.selected_model,
                    success=False,
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            logger.exception(f"Inference failed for request {request_id}")

            # Return error response
            failed_model = (
                routing_decision.selected_model
                if "routing_decision" in locals()
                else "unknown"
            )
            primary_exc = _primary_attempt_exception(exc)
            error_response = InferenceResponse(
                response_text="Request processing failed",
                model_name=failed_model,
                token_count_input=0,
                token_count_output=0,
                total_tokens=0,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=0.0,
                cost_usd=0.0,
                provider="error",
                cached=False,
                error="inference_failed",
                error_type=classify_inference_error(primary_exc),
                error_code=inference_error_code(primary_exc),
                retry_after_seconds=_retry_after_seconds(primary_exc),
                provider_request_id=_exception_provider_request_id(primary_exc),
                finish_reason="error",
            )
            # The caller keeps the "error" provider sentinel; analytics get the
            # failed model's real provider so outages count against it.
            failed_provider = self._configured_provider_name(failed_model)
            await self._publish_analytics_events(
                request,
                (
                    error_response.model_copy(update={"provider": failed_provider})
                    if failed_provider
                    else error_response
                ),
                routing_decision=routing_decision
                if "routing_decision" in locals()
                else None,
            )
            return error_response

    def _flag_incomplete_response(
        self, request: QueryRequest, response: InferenceResponse
    ) -> bool:
        """Warn about a truncated, filtered or empty answer; True if complete."""
        warnings = []
        if response.finish_reason != "stop":
            warnings.append(
                _FINISH_REASON_WARNINGS.get(
                    response.finish_reason, _UNFINISHED_RESPONSE_WARNING
                )
            )
        if not (response.response_text or "").strip():
            warnings.append("empty_response")
        complete = response.finish_reason == "stop" and not warnings
        if not complete:
            logger.warning(
                "Model %s returned an incomplete answer for request %s "
                "(finish_reason=%s, empty=%s)",
                response.model_name,
                request.request_id,
                response.finish_reason,
                "empty_response" in warnings,
            )
        response.warnings = [
            *response.warnings,
            *(warning for warning in warnings if warning not in response.warnings),
        ]
        return complete

    async def _finalize_successful_response(
        self,
        request: QueryRequest,
        response: InferenceResponse,
        model_name: str,
        routing_decision: Any,
        started_at: float,
    ) -> None:
        inference_time = time.time() - started_at
        await self._update_stats(model_name, response, inference_time)
        self._record_provider_cache_metrics(model_name, response)
        if not response.cached:
            # A cache hit served by a fallback model never ran that model, so
            # its lookup time must not feed the router's latency and success.
            self.router.update_model_stats(
                model_name=model_name,
                success=True,
                latency_ms=response.latency_ms,
            )
        INFERENCE_METRICS.requests_total.labels(
            model=model_name,
            provider=response.provider,
        ).inc()
        INFERENCE_METRICS.request_duration.labels(
            model=model_name,
            provider=response.provider,
        ).observe(inference_time)
        await self._publish_analytics_events(
            request,
            response,
            routing_decision=routing_decision,
        )

    async def _run_web_search_if_needed(
        self,
        request: QueryRequest,
        routing_decision: Any,
        time_limit_seconds: Optional[float] = None,
    ) -> Optional[WebSearchResult]:
        """Run web search when the request needs it, within time_limit_seconds.

        Running out of time is a failed search (warning, prompt notice, no
        cache), not a request failure.
        """
        # Likewise only this method sets web_search_unavailable.
        _drop_metadata_key(request, "web_search_unavailable")
        if not self._should_run_web_search(request, routing_decision):
            return None

        request.metadata = dict(request.metadata or {})
        if self.web_search_tool is None:
            result = WebSearchResult(
                tool_call=ToolCall(
                    name="web_search",
                    provider="none",
                    arguments={"query": request.query},
                    error="web_search_not_configured",
                ),
                blocked_reason="web_search_not_configured",
            )
        else:
            started_at = time.time()
            try:
                result = await _await_within(
                    self.web_search_tool.search(
                        request.query,
                        user_id=request.user_id,
                        options=request.web_search_options,
                    ),
                    time_limit_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Web search for request %s ran out of its %ss share of the "
                    "request deadline",
                    request.request_id,
                    time_limit_seconds,
                )
                backend = getattr(self.web_search_tool, "backend", None)
                result = WebSearchResult(
                    tool_call=ToolCall(
                        name="web_search",
                        provider=str(getattr(backend, "provider_name", "unknown")),
                        arguments={"query": request.query},
                        result_count=0,
                        latency_ms=int((time.time() - started_at) * 1000),
                        error="web_search_timeout",
                    ),
                    blocked_reason="web_search_timeout",
                    warnings=[WEB_SEARCH_UNAVAILABLE_WARNING],
                )

        if result.context:
            request.metadata["web_search_context"] = result.context
            request.metadata["web_search_result_count"] = len(result.sources)
            if result.tool_call and result.tool_call.provider:
                request.metadata["web_search_provider"] = result.tool_call.provider
        elif result.blocked_reason:
            request.metadata["web_search_blocked_reason"] = result.blocked_reason
            request.metadata["web_search_cache_bypass"] = True
        search_failed = WEB_SEARCH_UNAVAILABLE_WARNING in (
            getattr(result, "warnings", None) or []
        )
        if search_failed or (
            request.tool_policy == ToolPolicy.REQUIRED and not result.context
        ):
            # The prompt tells the model not to pass older knowledge off as
            # current; such an answer is not worth caching either.
            request.metadata["web_search_unavailable"] = True
            request.metadata["web_search_cache_bypass"] = True
        return result

    async def _run_rag_if_needed(
        self,
        request: QueryRequest,
        routing_decision: Any,
        time_limit_seconds: Optional[float] = None,
    ) -> Optional[SimpleNamespace]:
        """Retrieve documents when the request needs them, within time_limit_seconds.

        Running out of time is a retrieval timeout: rag_policy AUTO answers
        without documents, REQUIRED raises RetrievalUnavailableError.
        """
        # Only this method decides whether the prompt says retrieval failed,
        # so a caller's own rag_unavailable never reaches the prompt.
        _drop_metadata_key(request, "rag_unavailable")
        if not self._should_run_rag(request, routing_decision):
            return None

        request.metadata = dict(request.metadata or {})
        started_at = time.time()
        options = request.rag_options
        max_results = options.max_results if options and options.max_results else None
        candidate_count = (
            options.candidate_count if options and options.candidate_count else None
        )
        min_score = (
            options.min_score if options and options.min_score is not None else None
        )
        outcome = await self._retrieve_rag_outcome(
            request,
            time_limit_seconds=time_limit_seconds,
            max_results=max_results,
            candidate_count=candidate_count,
            min_score=min_score,
        )
        results = list(outcome.results)
        rag_unavailable = outcome.status in _RAG_UNAVAILABLE_STATUSES
        latency_ms = int((time.time() - started_at) * 1000)
        tool_call = self.rag_service.tool_call_from_results(
            request.query,
            results,
            latency_ms,
        )
        if rag_unavailable and isinstance(tool_call, ToolCall):
            tool_call = tool_call.model_copy(update={"error": f"rag_{outcome.status}"})
        sources = self.rag_service.sources_from_results(results)
        warnings: List[str] = []

        if rag_unavailable:
            request.metadata["rag_hit_count"] = 0
            request.metadata["rag_unavailable"] = outcome.status
            request.metadata["rag_cache_bypass"] = True
            if request.rag_policy == RagPolicy.REQUIRED:
                raise RetrievalUnavailableError(
                    outcome.status, getattr(outcome, "detail", None)
                )
            logger.warning(
                "RAG retrieval %s for request %s; answering without documents",
                outcome.status,
                request.request_id,
            )
            warnings.append("rag_unavailable")
        elif results:
            if getattr(outcome, "degraded", False) is True:
                # Hits, but a failed branch may have held a better match:
                # answer with them, without caching the answer for the TTL.
                logger.warning(
                    "RAG retrieval for request %s degraded (%s)",
                    request.request_id,
                    getattr(outcome, "detail", None),
                )
                request.metadata["rag_cache_bypass"] = True
                warnings.append("rag_degraded")
            request.metadata["rag_context"] = self.rag_service.build_context(results)
            request.metadata["rag_hit_count"] = len(results)
            request.metadata["rag_chunk_ids"] = [
                result.chunk.chunk_id for result in results
            ]
            request.metadata["rag_index_versions"] = [
                result.index_version or "" for result in results
            ]
            request.metadata["rag_match_sources"] = [
                result.match_source for result in results
            ]
        else:
            request.metadata["rag_hit_count"] = 0
            request.metadata["rag_no_hits"] = True
            request.metadata["rag_cache_bypass"] = True

        return SimpleNamespace(
            results=results,
            sources=sources,
            tool_call=tool_call,
            latency_ms=latency_ms,
            required_no_hits=request.rag_policy == RagPolicy.REQUIRED and not results,
            warnings=warnings,
        )

    async def _retrieve_rag_outcome(
        self,
        request: QueryRequest,
        *,
        time_limit_seconds: Optional[float] = None,
        **retrieve_kwargs: Any,
    ) -> Any:
        """Retrieve with a typed outcome: hits, no_hits, timeout or error.

        time_limit_seconds (this step's share of the request deadline) can
        only shorten the RAG service's own deadline. The service works to it
        itself, so a hung branch still leaves the answered branch's hits; a
        timer around the call would cancel those too.

        RAG services without ``retrieve_with_outcome`` only answer a list, so
        a raised error is the only failure signal they can give.
        """
        retrieve_with_outcome = getattr(self.rag_service, "retrieve_with_outcome", None)
        if callable(retrieve_with_outcome):
            return await retrieve_with_outcome(
                request.query,
                knowledge_base_ids=request.knowledge_base_ids,
                deadline_seconds=time_limit_seconds,
                **retrieve_kwargs,
            )
        try:
            results = await _await_within(
                self.rag_service.retrieve(
                    request.query,
                    knowledge_base_ids=request.knowledge_base_ids,
                    **retrieve_kwargs,
                ),
                time_limit_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "RAG retrieval for request %s timed out (share of the request "
                "deadline: %ss)",
                request.request_id,
                time_limit_seconds,
            )
            return SimpleNamespace(status="timeout", results=[], detail=None)
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc, exc_info=True)
            return SimpleNamespace(
                status="error", results=[], detail=type(exc).__name__
            )
        return SimpleNamespace(
            status="hits" if results else "no_hits",
            results=list(results or []),
            detail=None,
        )

    def _retrieval_time_limit(
        self, execution_budget: Optional[RequestExecutionBudget]
    ) -> Optional[float]:
        """Seconds one retrieval step may use: a share of the request time left.

        At least the scheduler's minimum attempt window stays for the model;
        with less than that left, retrieval gets no time.
        """
        if execution_budget is None:
            return None
        try:
            ratio = float(
                self.config.get(
                    "retrieval_budget_ratio", _DEFAULT_RETRIEVAL_BUDGET_RATIO
                )
            )
        except (TypeError, ValueError):
            ratio = _DEFAULT_RETRIEVAL_BUDGET_RATIO
        if not 0.0 < ratio < 1.0:
            ratio = _DEFAULT_RETRIEVAL_BUDGET_RATIO
        # The step must end on its own clock before the request deadline's
        # backstop fires, or the backstop wins and the outcome is lost.
        model_window = max(
            _RETRIEVAL_BACKSTOP_MARGIN_SECONDS,
            float(getattr(self.scheduler, "min_attempt_window_seconds", 1.0)),
        )
        remaining = execution_budget.remaining_seconds
        return max(0.0, min(remaining * ratio, remaining - model_window))

    async def _prepare_rag_images(
        self,
        request: QueryRequest,
        routing_decision: Any,
        rag_result: Optional[SimpleNamespace],
    ) -> Any:
        if not rag_result or not rag_result.results:
            return routing_decision
        if not hasattr(self.rag_service, "materialize_image_inputs"):
            return routing_decision
        visual_config = getattr(self.rag_service, "visual_config", {}) or {}
        answer_config = dict(visual_config.get("answer_images", {}) or {})
        if answer_config.get("enabled", True) is False:
            return routing_decision
        images, warnings = await self.rag_service.materialize_image_inputs(
            rag_result.results,
            max_images=int(answer_config.get("max_images", 3)),
            max_bytes=int(answer_config.get("max_bytes", 5_000_000)),
            max_long_edge_pixels=int(answer_config.get("max_long_edge_pixels", 1568)),
        )
        request.metadata = dict(request.metadata or {})
        if warnings:
            request.metadata["rag_warnings"] = [
                *request.metadata.get("rag_warnings", []),
                *warnings,
            ]
            rag_result.warnings = list(request.metadata["rag_warnings"])
        if not images:
            if warnings:
                warning = (
                    "rag_vision_fallback: image materialization failed; used text only"
                )
                request.metadata["rag_warnings"] = [
                    *request.metadata.get("rag_warnings", []),
                    warning,
                ]
                rag_result.warnings = list(request.metadata["rag_warnings"])
                RAG_METRICS.vision_outcomes.labels(
                    "text_only_materialization_failure"
                ).inc()
            return routing_decision

        selected = routing_decision.selected_model
        selected_info = self.router.get_model_info(selected) or {}
        capabilities = {
            value.lower()
            for value in selected_info.get("config", {}).get("capabilities", [])
        }
        if "vision" not in capabilities:
            if hasattr(self.router, "eligible_models_with_capability"):
                eligible_vision_models = (
                    await self.router.eligible_models_with_capability(
                        request,
                        "vision",
                        exclude_models=[selected],
                    )
                )
                vision_model = (
                    eligible_vision_models[0] if eligible_vision_models else None
                )
                request.metadata["rag_vision_fallback_models"] = eligible_vision_models[
                    1:
                ]
            else:
                vision_model = await self.router.select_model_with_capability(
                    request,
                    "vision",
                    exclude_models=[selected],
                )
            if not vision_model:
                request.metadata["rag_warnings"] = [
                    *request.metadata.get("rag_warnings", []),
                    "rag_vision_fallback: no policy-eligible vision model",
                ]
                rag_result.warnings = list(request.metadata["rag_warnings"])
                RAG_METRICS.vision_outcomes.labels("text_only_no_model").inc()
                return routing_decision
            request.metadata["rag_original_model"] = selected
            routing_decision = self._build_capability_routing_decision(
                routing_decision, selected, vision_model
            )
            RAG_METRICS.vision_outcomes.labels("rerouted").inc()
        else:
            if hasattr(self.router, "eligible_models_with_capability"):
                request.metadata[
                    "rag_vision_fallback_models"
                ] = await self.router.eligible_models_with_capability(
                    request,
                    "vision",
                    exclude_models=[selected],
                )
                if hasattr(self.router, "eligible_models_without_capability"):
                    request.metadata[
                        "rag_text_fallback_models"
                    ] = await self.router.eligible_models_without_capability(
                        request,
                        "vision",
                        exclude_models=[
                            selected,
                            *request.metadata["rag_vision_fallback_models"],
                        ],
                    )
            RAG_METRICS.vision_outcomes.labels("selected_model").inc()
        request._rag_images = images
        return routing_decision

    def _build_capability_routing_decision(
        self, routing_decision: Any, original_model: str, selected_model: str
    ) -> Any:
        reason = (
            f"{getattr(routing_decision, 'routing_reason', 'RAG route')}; "
            f"vision reroute from {original_model} to {selected_model}"
        )
        if hasattr(routing_decision, "model_copy"):
            return routing_decision.model_copy(
                update={"selected_model": selected_model, "routing_reason": reason}
            )
        payload = dict(vars(routing_decision))
        payload.update(selected_model=selected_model, routing_reason=reason)
        return SimpleNamespace(**payload)

    def _should_run_rag(self, request: QueryRequest, routing_decision: Any) -> bool:
        if not self.rag_service or not getattr(self.rag_service, "enabled", False):
            return False
        if request.rag_policy == RagPolicy.DISABLED:
            return False
        if request.rag_policy == RagPolicy.REQUIRED:
            return True
        metadata = request.metadata or {}
        if metadata.get("enable_rag") is True:
            return True
        if metadata.get("rag_auto") is False:
            return False
        if request.knowledge_base_ids:
            return True
        if not bool(getattr(self.rag_service, "auto_retrieve", False)):
            return False
        return self._looks_like_school_document_query(request, routing_decision)

    def _looks_like_school_document_query(
        self, request: QueryRequest, routing_decision: Any
    ) -> bool:
        """Conservative intent gate for automatic school-document retrieval."""
        query = request.query or ""
        lowered = query.lower()

        query_type = getattr(routing_decision, "query_type", None)
        query_type_value = (
            query_type.value if hasattr(query_type, "value") else str(query_type or "")
        )
        if query_type_value in {
            QueryType.CODE_GENERATION.value,
            QueryType.CODE_ANALYSIS.value,
            QueryType.WEB_RESEARCH.value,
        }:
            return False

        strong_terms = self._rag_intent_terms("strong_terms", RAG_AUTO_STRONG_TERMS)
        context_terms = self._rag_intent_terms("context_terms", RAG_AUTO_CONTEXT_TERMS)
        school_terms = self._rag_intent_terms(
            "school_context_terms", RAG_AUTO_SCHOOL_CONTEXT_TERMS
        )
        exclude_terms = self._rag_intent_terms("exclude_terms", RAG_AUTO_EXCLUDE_TERMS)

        strong_match = any(str(term).lower() in lowered for term in strong_terms)
        context_match = any(
            str(term).lower() in lowered for term in context_terms
        ) and any(str(term).lower() in lowered for term in school_terms)

        if not (strong_match or context_match):
            return False

        if any(str(term).lower() in lowered for term in exclude_terms):
            return False

        return True

    def _rag_intent_terms(self, key: str, default: Sequence[str]) -> Sequence[str]:
        intent_config = dict(getattr(self.rag_service, "intent_gate_config", {}) or {})
        configured = intent_config.get(key)
        if isinstance(configured, (list, tuple, set)) and configured:
            return [str(term) for term in configured]
        return default

    def _build_required_rag_no_hit_response(
        self,
        request: QueryRequest,
        model_name: str,
        rag_result: SimpleNamespace,
        started_at: float,
    ) -> InferenceResponse:
        input_tokens = (
            self.router.token_counter.count_tokens(request.query)
            if hasattr(self.router, "token_counter")
            else len(request.query.split())
        )
        response_text = "文件庫沒有足夠資訊可以回答這個問題。"
        return InferenceResponse(
            response_text=response_text,
            model_name=model_name,
            provider="rag",
            token_count_input=input_tokens,
            token_count_output=len(response_text.split()),
            total_tokens=input_tokens + len(response_text.split()),
            latency_ms=int((time.time() - started_at) * 1000),
            tokens_per_second=0.0,
            cost_usd=0.0,
            cached=False,
            tool_calls=[rag_result.tool_call],
            tool_latency_ms=rag_result.latency_ms,
            finish_reason="rag_no_hits",
        )

    def _should_run_web_search(
        self,
        request: QueryRequest,
        routing_decision: Any,
    ) -> bool:
        metadata = request.metadata or {}
        if request.tool_policy == ToolPolicy.DISABLED:
            return False
        allowed_tools = set(request.allowed_tools or [])
        if allowed_tools and "web_search" not in allowed_tools:
            return False
        if request.tool_policy == ToolPolicy.REQUIRED:
            return True
        if metadata.get("enable_web_search") is True:
            return True
        if metadata.get("web_search_auto") is False:
            return False
        query_type = getattr(routing_decision, "query_type", None)
        query_type_value = (
            query_type.value if hasattr(query_type, "value") else query_type
        )
        return query_type_value == QueryType.WEB_RESEARCH.value

    def _attach_web_search_result(
        self,
        response: InferenceResponse,
        result: Optional[WebSearchResult],
    ) -> None:
        if result is None:
            return
        response.sources = [*response.sources, *list(result.sources)]
        response.warnings = [
            *response.warnings,
            *list(getattr(result, "warnings", []) or []),
        ]
        if result.tool_call:
            response.tool_calls = [*response.tool_calls, result.tool_call]
            response.tool_latency_ms += result.tool_call.latency_ms

    def _attach_rag_result(
        self,
        response: InferenceResponse,
        result: Optional[SimpleNamespace],
    ) -> None:
        if result is None:
            return
        response.sources = [*response.sources, *list(result.sources)]
        response.warnings = [
            *response.warnings,
            *list(getattr(result, "warnings", []) or []),
        ]
        if result.tool_call:
            response.tool_calls = [*response.tool_calls, result.tool_call]
            response.tool_latency_ms += result.latency_ms

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[str]:
        """Stream query response"""
        start_time = time.time()
        try:
            # Route query
            routing_decision = await self.router.route_query(request)
            model_name = routing_decision.selected_model
            await self._compress_request_context_if_needed(request)
            rag_result = await self._run_rag_if_needed(request, routing_decision)
            if rag_result and rag_result.required_no_hits:
                response = self._build_required_rag_no_hit_response(
                    request,
                    model_name,
                    rag_result,
                    start_time,
                )
                await self._finalize_successful_response(
                    request,
                    response,
                    model_name,
                    routing_decision,
                    start_time,
                )
                yield "文件庫沒有足夠資訊可以回答這個問題。"
                return
            routing_decision = await self._prepare_rag_images(
                request, routing_decision, rag_result
            )
            model_name = routing_decision.selected_model
            await self._run_web_search_if_needed(request, routing_decision)

            cached_response = await self._get_cached_inference_response(
                request,
                model_name,
            )
            if cached_response:
                await self._publish_analytics_events(
                    request,
                    cached_response,
                    routing_decision=routing_decision,
                )
                async for chunk in self._stream_cached_response(cached_response):
                    yield chunk
                return

            # Get provider
            provider = self._get_provider_for_model(model_name)
            if not provider:
                if getattr(request, "_rag_images", []):
                    response, _, _ = await self._execute_pre_token_vision_fallback(
                        request, routing_decision, model_name
                    )
                    yield response.response_text
                    return
                yield f"Error: No provider available for model: {model_name}"
                return

            model_info = self.router.get_model_info(model_name) or {}
            provider_name = (
                model_info.get("config", {}).get("provider")
                or getattr(provider, "provider_name", None)
                or "unknown"
            )
            request_started_at_ms = int(time.time() * 1000)
            lease = None
            pre_token_failure: Optional[Exception] = None
            emitted_token = False
            try:
                lease = await self.scheduler.acquire(
                    request=request,
                    model_name=model_name,
                    provider=str(provider_name),
                    estimated_input_tokens=self._estimate_provider_admission_input_tokens(
                        request
                    ),
                )
                try:
                    async for chunk in provider.stream_response(request, model_name):
                        emitted_token = True
                        yield chunk
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self.scheduler.record_failure(
                        provider=str(provider_name),
                        model_name=model_name,
                        error=exc,
                        request_started_at_ms=request_started_at_ms,
                        circuit_permit=lease.circuit_permit,
                    )
                    if emitted_token or not getattr(request, "_rag_images", []):
                        raise
                    pre_token_failure = exc
                else:
                    await self.scheduler.record_success(
                        provider=str(provider_name),
                        model_name=model_name,
                        request_started_at_ms=request_started_at_ms,
                        circuit_permit=lease.circuit_permit,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not getattr(request, "_rag_images", []):
                    raise
                pre_token_failure = exc
            finally:
                if lease is not None:
                    await self.scheduler.release(lease, actual_tokens=None)
            if pre_token_failure is not None:
                response, _, _ = await self._execute_pre_token_vision_fallback(
                    request, routing_decision, model_name
                )
                yield response.response_text
                return

        except AdmissionRejectedError as exc:
            yield f"Error: {exc.decision.reason}"
        except Exception:
            logger.exception("Streaming failed")
            yield "Error: Request processing failed"

    async def _stream_cached_response(
        self, response: InferenceResponse
    ) -> AsyncIterator[str]:
        """Yield cached full responses in deterministic chunks for streaming clients."""
        chunk_size = int(self.config.get("stream_cache_chunk_chars", 512) or 512)
        chunk_size = max(chunk_size, 1)
        for index in range(0, len(response.response_text), chunk_size):
            yield response.response_text[index : index + chunk_size]

    def _configured_provider_name(self, model_name: str) -> Optional[str]:
        """Return the provider configured for a model, or None when unknown."""
        try:
            model_info = self.router.get_model_info(model_name)
        except Exception:
            logger.debug("Provider lookup failed for %s", model_name, exc_info=True)
            return None
        config = model_info.get("config") if isinstance(model_info, dict) else None
        provider = config.get("provider") if isinstance(config, dict) else None
        return provider if isinstance(provider, str) and provider else None

    def _get_provider_for_model(
        self, model_name: str
    ) -> Optional[BaseInferenceProvider]:
        """Get the appropriate provider for a model"""
        # Get model configuration from router
        model_info = self.router.get_model_info(model_name)
        if not model_info:
            return None

        provider_name = model_info["config"]["provider"]
        return self.providers.get(provider_name)

    async def _within_request_deadline(
        self,
        awaitable: Awaitable[Any],
        execution_budget: Optional[RequestExecutionBudget],
        stage: str,
    ) -> Any:
        """Await a step before the model call, at most until the request deadline.

        Raises RequestStageDeadlineError when the deadline cuts the step off.
        """
        if execution_budget is None:
            return await awaitable
        try:
            return await asyncio.wait_for(
                awaitable, timeout=execution_budget.remaining_seconds
            )
        except asyncio.TimeoutError as exc:
            if execution_budget.remaining_seconds > 0.01:
                raise  # The step's own timeout, not the request deadline.
            raise RequestStageDeadlineError(
                f"request deadline reached during {stage}"
            ) from exc

    async def _generate_response_with_fallback(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget] = None,
    ) -> Tuple[InferenceResponse, str, Any]:
        if execution_budget is None:
            return await self._generate_response_with_fallback_unbounded(
                request, routing_decision, execution_budget=None
            )
        if execution_budget.remaining_seconds <= 0:
            # Routing, compression, image prep or the cache lookup used the
            # whole deadline: no model or provider is to blame.
            raise RequestStageDeadlineError(
                "request deadline reached before the model call"
            )
        execution_budget.ensure_available()
        return await asyncio.wait_for(
            self._generate_response_with_fallback_unbounded(
                request, routing_decision, execution_budget=execution_budget
            ),
            timeout=execution_budget.remaining_seconds,
        )

    async def _generate_response_with_fallback_unbounded(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget] = None,
    ) -> Tuple[InferenceResponse, str, Any]:
        """Generate a response and fall back to local models when cloud models fail.

        If the primary model fails and so does every recovery attempt, the last
        attempt's exception is raised unchanged, carrying the primary's exception.
        """
        primary_failures: List[BaseException] = []
        try:
            return await self._generate_response_with_fallback_attempts(
                request, routing_decision, execution_budget, primary_failures
            )
        except BaseException as exc:
            if primary_failures and exc is not primary_failures[0]:
                setattr(exc, _PRIMARY_ATTEMPT_EXCEPTION_ATTR, primary_failures[0])
            raise

    async def _generate_response_with_fallback_attempts(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget],
        primary_failures: List[BaseException],
    ) -> Tuple[InferenceResponse, str, Any]:
        primary_model = routing_decision.selected_model
        primary_started_at = time.time()
        # With a shared budget, a primary that has a fallback holds back part
        # of the time and one attempt for it. With a single attempt per
        # request the fallback could never run, so nothing is held back.
        fallback_reserve_seconds = 0.0
        reserve_attempts = 0
        if (
            execution_budget is not None
            and execution_budget.max_attempts >= 2
            and self._primary_has_fallback_route(request, routing_decision)
        ):
            fallback_reserve_seconds = execution_budget.fallback_reserve_seconds()
            reserve_attempts = 1
        started_calls: List[str] = []

        try:
            response = await self._execute_model_request(
                request,
                primary_model,
                execution_budget=execution_budget,
                fallback_reserve_seconds=fallback_reserve_seconds,
                reserve_attempts=reserve_attempts,
                started_calls=started_calls,
            )
            return response, primary_model, routing_decision
        except AdmissionRejectedError as exc:
            primary_failures.append(exc)
            if not self.scheduler.allow_fallback_for_rejection(exc.decision):
                raise
            # A circuit or capacity rejection: the model is unavailable.
            return await self._run_fallback_models(
                request,
                routing_decision,
                execution_budget,
                exc,
                cross_provider=True,
                started_calls=started_calls,
            )
        except Exception as exc:
            if primary_model in started_calls:
                primary_failures.append(exc)
                self._record_failed_attempt(primary_model, exc, primary_started_at)
            else:
                # The provider was never called (no time was left for an
                # attempt): neither the model nor its provider failed.
                setattr(exc, "_attempt_recorded", True)
                primary_failures.append(
                    RequestStageDeadlineError(
                        "request deadline reached before the model call"
                    )
                    if isinstance(exc, asyncio.TimeoutError)
                    else exc
                )
            return await self._run_fallback_models(
                request,
                routing_decision,
                execution_budget,
                exc,
                cross_provider=self.scheduler.is_availability_failure(exc),
                started_calls=started_calls,
            )

    async def _run_fallback_models(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget],
        primary_exc: Exception,
        *,
        cross_provider: bool,
        started_calls: List[str],
    ) -> Tuple[InferenceResponse, str, Any]:
        """Try the fallbacks of a failed primary model, in order.

        The configured chain (any provider) comes first, and only when the
        primary was unavailable; local candidates keep their own triggers.
        With no candidate the primary's exception is raised, else the last
        fallback's. ``started_calls`` lists the models whose provider was
        called for this request.
        """
        primary_model = routing_decision.selected_model
        policy_context = await self._fallback_policy_context(request)
        local_models = self._get_local_fallback_models(
            request,
            routing_decision,
            exclude_models=[primary_model],
            policy_context=policy_context,
        )
        chain_models = (
            self._get_fallback_chain_models(
                request, routing_decision, policy_context, exclude=[primary_model]
            )
            if cross_provider and not getattr(request, "_rag_images", [])
            else []
        )
        fallback_models = [
            *chain_models,
            *(model for model in local_models if model not in chain_models),
        ]

        if not fallback_models:
            if getattr(request, "_rag_images", []):
                return await self._text_only_fallback(
                    request,
                    routing_decision,
                    execution_budget,
                    started_calls=started_calls,
                )
            raise primary_exc

        logger.warning(
            "Primary model %s failed (%s); attempting fallback candidates %s",
            primary_model,
            getattr(getattr(primary_exc, "decision", None), "reason", None)
            or type(primary_exc).__name__,
            fallback_models,
        )

        last_exception: Exception = primary_exc
        for fallback_model in fallback_models:
            fallback_started_at = time.time()
            response = await self._get_cached_inference_response(
                request,
                fallback_model,
            )
            from_cache = bool(response)
            if not response:
                try:
                    response = await self._execute_model_request(
                        request,
                        fallback_model,
                        execution_budget=execution_budget,
                        started_calls=started_calls,
                    )
                except AdmissionRejectedError as fallback_rejection:
                    if not self.scheduler.allow_fallback_for_rejection(
                        fallback_rejection.decision
                    ):
                        raise
                    last_exception = fallback_rejection
                    continue
                except Exception as fallback_exc:
                    if fallback_model in started_calls:
                        self._record_failed_attempt(
                            fallback_model,
                            fallback_exc,
                            fallback_started_at,
                        )
                    else:
                        # No attempt or time was left to call it: the model
                        # did not fail, and the primary's failure is on record.
                        setattr(fallback_exc, "_attempt_recorded", True)
                    last_exception = fallback_exc
                    continue
            response.fallback_from_model = primary_model
            # Only models whose provider was called count: not those rejected
            # at admission or by an open circuit, nor a fallback cache hit. A
            # fallback that answered counts even when a coalesced call (the
            # single-flight leader's) reached its provider.
            response.attempts = (
                len(started_calls)
                if from_cache
                else len({*started_calls, fallback_model})
            )
            return (
                response,
                fallback_model,
                self._build_fallback_routing_decision(
                    routing_decision,
                    original_model=primary_model,
                    fallback_model=fallback_model,
                    from_cache=from_cache,
                    fallback_models=fallback_models,
                ),
            )

        if getattr(request, "_rag_images", []):
            return await self._text_only_fallback(
                request,
                routing_decision,
                execution_budget,
                started_calls=started_calls,
            )
        raise last_exception

    async def _text_only_fallback(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget],
        *,
        started_calls: List[str],
    ) -> Tuple[InferenceResponse, str, Any]:
        """The text-only retry of a failed vision request, marked as a fallback.

        A failure is recorded against the text-only model when it was called;
        the earlier failures are already on record, so the routed model is
        not blamed again.
        """
        primary_model = routing_decision.selected_model
        retry_started_at = time.time()
        retry_calls: List[str] = []
        try:
            result = await self._execute_text_only_after_vision_failure(
                request, routing_decision, execution_budget, started_calls=retry_calls
            )
        except AdmissionRejectedError:
            raise
        except Exception as exc:
            for model_name in retry_calls:
                self._record_failed_attempt(model_name, exc, retry_started_at)
            setattr(exc, "_attempt_recorded", True)
            raise
        finally:
            started_calls.extend(
                name for name in retry_calls if name not in started_calls
            )
        response, model_name, decision = result
        response.attempts = len({*started_calls, model_name})
        response.fallback_from_model = (
            primary_model if model_name != primary_model else None
        )
        return response, model_name, decision

    async def _fallback_policy_context(self, request: QueryRequest) -> Dict[str, Any]:
        """The router's current guardrail and avoid lists for this request.

        Read when a fallback is needed, so a model blocked since routing (or
        while the primary was failing) is not picked.
        """
        load_policy = getattr(self.router, "_get_policy_context", None)
        if not callable(load_policy):
            return {}
        try:
            policy = load_policy(request)
            if hasattr(policy, "__await__"):
                policy = await policy
        except Exception:
            logger.warning(
                "Routing policy unavailable for fallback selection", exc_info=True
            )
            return {}
        if not isinstance(policy, dict):
            return {}
        return {
            key: list(policy.get(key) or [])
            for key in (
                "blocked_models",
                "blocked_providers",
                "avoid_models",
                "avoid_providers",
            )
        }

    def _get_fallback_chain_models(
        self,
        request: QueryRequest,
        routing_decision: Any,
        policy_context: Dict[str, Any],
        exclude: Sequence[str],
    ) -> List[str]:
        """The configured fallback chain for the routed model, filtered like routing.

        Tier, context window and guardrails apply. Capability does not, the
        same as for routing rules that name their models.
        """
        chains = dict((self.config.get("fallback") or {}).get("chains") or {})
        candidates: List[str] = []
        for model_name in self._normalize_fallback_model_list(
            chains.get(routing_decision.selected_model)
        ):
            if model_name not in exclude and model_name not in candidates:
                candidates.append(model_name)
        if not candidates:
            return []
        context = self._build_local_fallback_context(
            request, routing_decision, policy_context
        )
        eligible = set(
            self.router._filter_models_for_context(
                candidates, context, check_capability=False
            )
        )
        return [
            model_name
            for model_name in candidates
            if model_name in eligible
            and self._get_provider_for_model(model_name) is not None
        ]

    def _tier_allows_model(self, model_name: str, request: QueryRequest) -> bool:
        """Whether the user's tier may use a model, by the router's access rule."""
        router = getattr(self, "router", None)
        models = getattr(router, "models", None)
        check_access = getattr(router, "_check_user_access", None)
        if not isinstance(models, dict) or not callable(check_access):
            return True  # A stub router has no access rules to apply.
        model_config = models.get(model_name)
        return model_config is not None and bool(
            check_access(model_config, request.user_tier)
        )

    async def _execute_text_only_after_vision_failure(
        self,
        request: QueryRequest,
        routing_decision: Any,
        execution_budget: Optional[RequestExecutionBudget],
        started_calls: Optional[List[str]] = None,
    ) -> Tuple[InferenceResponse, str, Any]:
        """Retry once without images only after all vision-capable routes fail."""
        request._rag_images = []
        request.metadata = dict(request.metadata or {})
        warning = "rag_vision_fallback: vision providers unavailable; used text only"
        RAG_METRICS.vision_outcomes.labels("text_only_provider_failure").inc()
        request.metadata["rag_warnings"] = [
            *request.metadata.get("rag_warnings", []),
            warning,
        ]
        text_fallbacks = list(request.metadata.get("rag_text_fallback_models") or [])
        candidates = [
            str(name)
            for name in (
                request.metadata.get("rag_original_model"),
                *text_fallbacks,
                routing_decision.selected_model,
            )
            if name
        ]
        # Request metadata can come from a caller: run only a model the user's
        # tier is entitled to.
        allowed = [
            name
            for name in dict.fromkeys(candidates)
            if self._tier_allows_model(name, request)
        ]
        if not allowed:
            raise ValueError("No text-only model is open to the user's tier")
        model_name = allowed[0]
        call_options: Dict[str, Any] = {"execution_budget": execution_budget}
        if started_calls is not None:
            call_options["started_calls"] = started_calls
        response = await self._execute_model_request(
            request, model_name, **call_options
        )
        response.warnings = [*response.warnings, warning]
        return (
            response,
            model_name,
            self._build_capability_routing_decision(
                routing_decision,
                routing_decision.selected_model,
                model_name,
            ),
        )

    async def _execute_pre_token_vision_fallback(
        self,
        request: QueryRequest,
        routing_decision: Any,
        failed_model: str,
    ) -> Tuple[InferenceResponse, str, Any]:
        """Retry another vision route, then downgrade before any token is emitted."""
        candidates = self._get_local_fallback_models(
            request,
            routing_decision,
            exclude_models=[failed_model],
            policy_context=await self._fallback_policy_context(request),
        )
        if candidates:
            fallback_decision = self._build_capability_routing_decision(
                routing_decision, failed_model, candidates[0]
            )
            return await self._generate_response_with_fallback(
                request, fallback_decision
            )
        return await self._execute_text_only_after_vision_failure(
            request, routing_decision, execution_budget=None
        )

    def _primary_has_fallback_route(
        self, request: QueryRequest, routing_decision: Any
    ) -> bool:
        """Whether a failed primary would have a fallback to try."""
        if getattr(request, "_rag_images", []):
            # A failed vision call is retried as text only.
            return True
        exclude = [routing_decision.selected_model]
        try:
            # Guardrails are only read once the primary fails, so a fallback
            # blocked by them still counts here.
            return bool(
                self._get_local_fallback_models(
                    request, routing_decision, exclude_models=exclude
                )
            ) or bool(
                self._get_fallback_chain_models(
                    request, routing_decision, {}, exclude=exclude
                )
            )
        except Exception:
            logger.debug("Fallback route lookup failed", exc_info=True)
            return False

    async def _execute_model_request(
        self,
        request: QueryRequest,
        model_name: str,
        execution_budget: Optional[RequestExecutionBudget] = None,
        fallback_reserve_seconds: float = 0.0,
        reserve_attempts: int = 0,
        started_calls: Optional[List[str]] = None,
    ) -> InferenceResponse:
        """Execute a request against a specific model.

        ``fallback_reserve_seconds`` of the request deadline and
        ``reserve_attempts`` of its shared attempts stay unused by this
        model's retries, so a fallback can still run. The model is added to
        ``started_calls`` once its provider is actually called.
        """
        provider = self._get_provider_for_model(model_name)
        if not provider:
            raise ValueError(f"No provider available for model: {model_name}")

        model_info = self.router.get_model_info(model_name) or {}
        provider_name = (
            model_info.get("config", {}).get("provider")
            or getattr(provider, "provider_name", None)
            or "unknown"
        )
        attempt_timeout_seconds = (
            provider.timeout_seconds()
            if isinstance(provider, BaseInferenceProvider)
            else None
        )

        async def call_provider() -> InferenceResponse:
            if started_calls is not None and model_name not in started_calls:
                started_calls.append(model_name)
            return await provider.generate_response(request, model_name)

        return await self.single_flight.run(
            request,
            provider,
            model_name,
            lambda: self.scheduler.run_provider_call(
                request=request,
                model_name=model_name,
                provider=str(provider_name),
                estimated_input_tokens=self._estimate_provider_admission_input_tokens(
                    request
                ),
                operation=call_provider,
                execution_budget=execution_budget,
                attempt_timeout_seconds=attempt_timeout_seconds,
                fallback_reserve_seconds=fallback_reserve_seconds,
                reserve_attempts=reserve_attempts,
            ),
        )

    def _estimate_provider_admission_input_tokens(self, request: QueryRequest) -> int:
        """Estimate provider input tokens for admission budget reservation."""
        estimator = getattr(self.router, "_estimate_request_token_count", None)
        if estimator is not None:
            try:
                return max(0, int(estimator(request)))
            except Exception:
                logger.debug("Provider admission token estimator failed", exc_info=True)

        token_counter = getattr(self.router, "token_counter", None)
        count_tokens = getattr(token_counter, "count_tokens", None)
        if count_tokens is not None:
            try:
                prompt = "\n\n".join(
                    part
                    for part in [
                        f"Context: {request.context}" if request.context else "",
                        f"Query: {request.query}",
                    ]
                    if part
                )
                return max(0, int(count_tokens(prompt)))
            except Exception:
                logger.debug("Fallback provider token estimator failed", exc_info=True)

        return max(1, len((request.query or "").split()))

    def _record_failed_attempt(
        self,
        model_name: str,
        exc: Exception,
        attempt_started_at: float,
    ):
        """Record metrics and router stats for a failed model attempt."""
        setattr(exc, "_attempt_recorded", True)
        INFERENCE_METRICS.errors_total.labels(
            model=model_name,
            error_type=type(exc).__name__,
        ).inc()
        self.router.update_model_stats(
            model_name=model_name,
            success=False,
            latency_ms=int((time.time() - attempt_started_at) * 1000),
        )
        if model_name not in self.inference_stats:
            self.inference_stats[model_name] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_time": 0.0,
                "error_count": 0,
            }
        self.inference_stats[model_name]["total_requests"] += 1
        self.inference_stats[model_name]["error_count"] += 1
        self.inference_stats[model_name]["total_time"] += max(
            time.time() - attempt_started_at,
            0.0,
        )

    def _get_local_fallback_models(
        self,
        request: QueryRequest,
        routing_decision: Any,
        exclude_models: Optional[List[str]] = None,
        policy_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Return capability-safe fallback candidates for the current request.

        ``policy_context`` carries the router's guardrail blocks and avoid
        lists; candidates they rule out are skipped.
        """
        selected_model = routing_decision.selected_model
        selected_model_info = self.router.get_model_info(selected_model)
        if not selected_model_info:
            return []

        selected_provider = selected_model_info["config"]["provider"].lower()
        exclude = set(exclude_models or [])
        context = self._build_local_fallback_context(
            request, routing_decision, policy_context
        )
        if getattr(request, "_rag_images", []):
            policy_candidates = request.metadata.get("rag_vision_fallback_models")
            if policy_candidates is not None:
                return [
                    model_name
                    for model_name in policy_candidates
                    if model_name not in exclude
                    and self._get_provider_for_model(model_name) is not None
                    and not self._ruled_out_by_policy(
                        model_name, self._configured_provider_name(model_name), context
                    )
                ]
            ranked = []
            for model_name, model_config in getattr(self.router, "models", {}).items():
                if model_name in exclude:
                    continue
                if self._ruled_out_by_policy(
                    model_name, model_config.provider, context
                ):
                    continue
                if "vision" not in {
                    capability.lower() for capability in model_config.capabilities
                }:
                    continue
                if model_config.provider not in self.providers:
                    continue
                if context["token_count"] > model_config.max_tokens:
                    continue
                if not self.router._check_user_access(
                    model_config, context["user_tier"]
                ):
                    continue
                ranked.append(
                    (self.router._score_model(model_name, context), model_name)
                )
            ranked.sort(reverse=True)
            return [model_name for _, model_name in ranked]

        if selected_provider == "vllm":
            return self._get_configured_vllm_model_fallbacks(
                selected_model,
                request,
                routing_decision,
                context,
                exclude,
            )

        return self._rank_local_vllm_fallback_candidates(context, exclude)

    def _build_local_fallback_context(
        self,
        request: QueryRequest,
        routing_decision: Any,
        policy_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_type = getattr(routing_decision, "query_type", QueryType.GENERAL.value)
        if hasattr(query_type, "value"):
            query_type = query_type.value
        else:
            query_type = str(query_type)

        token_count = getattr(routing_decision, "token_count", None)
        if token_count is None or (
            isinstance(token_count, (int, float)) and token_count <= 0
        ):
            # A routing error reports 0 tokens; estimate, so the context
            # window check still means something.
            token_count = self._estimate_provider_admission_input_tokens(request)

        context = {
            "query_type": query_type,
            "token_count": int(token_count),
            "user_tier": request.user_tier,
            "priority": request.priority,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "has_attachments": bool(request.attachments),
        }
        # Guardrail blocks and avoid lists, keyed as the router's context.
        for key, values in (policy_context or {}).items():
            context[key] = list(values or [])
        return context

    @staticmethod
    def _ruled_out_by_policy(
        model_name: str, provider_name: Optional[str], context: Dict[str, Any]
    ) -> bool:
        """Whether the guardrail blocks or avoid lists in context exclude a model."""
        blocked_models = {
            *(context.get("blocked_models") or []),
            *(context.get("avoid_models") or []),
        }
        if model_name in blocked_models:
            return True
        blocked_providers = {
            str(provider).lower()
            for provider in [
                *(context.get("blocked_providers") or []),
                *(context.get("avoid_providers") or []),
            ]
        }
        return bool(provider_name) and str(provider_name).lower() in blocked_providers

    def _get_configured_vllm_model_fallbacks(
        self,
        selected_model: str,
        request: QueryRequest,
        routing_decision: Any,
        context: Dict[str, Any],
        exclude: set[str],
    ) -> List[str]:
        policy = self._vllm_model_fallback_config()
        if not self._vllm_model_fallback_allowed(
            policy, request, routing_decision, context
        ):
            return []

        fallback_map = policy.get("fallbacks") or {}
        configured_models = self._normalize_fallback_model_list(
            fallback_map.get(selected_model, [])
        )

        fallback_models = []
        for fallback_model in configured_models:
            if fallback_model in exclude:
                continue
            if self._local_vllm_fallback_model_eligible(fallback_model, context):
                fallback_models.append(fallback_model)
        return fallback_models

    def _vllm_model_fallback_config(self) -> Dict[str, Any]:
        vllm_config = dict(self.config.get("vllm", {}) or {})
        return dict(vllm_config.get("model_fallback", {}) or {})

    def _normalize_fallback_model_list(self, models: Any) -> List[str]:
        if models is None:
            return []
        if isinstance(models, str):
            return [models]
        if isinstance(models, Sequence):
            return [str(model) for model in models]
        return []

    def _vllm_model_fallback_allowed(
        self,
        policy: Dict[str, Any],
        request: QueryRequest,
        routing_decision: Any,
        context: Dict[str, Any],
    ) -> bool:
        if not bool(policy.get("enabled", False)):
            return False

        allowed_query_types = set(
            self._normalize_fallback_model_list(
                policy.get(
                    "allowed_query_types",
                    [
                        QueryType.GENERAL.value,
                        QueryType.CODE_GENERATION.value,
                        QueryType.CODE_ANALYSIS.value,
                        QueryType.QUESTION_ANSWERING.value,
                    ],
                )
            )
        )
        if context["query_type"] not in allowed_query_types:
            return False

        max_input_tokens = int(policy.get("max_input_tokens", 2048) or 2048)
        if int(context["token_count"]) > max_input_tokens:
            return False

        max_output_tokens = int(policy.get("max_output_tokens", 1024) or 1024)
        if int(request.max_tokens) > max_output_tokens:
            return False

        if bool(policy.get("disallow_attachments", True)) and request.attachments:
            return False

        if bool(policy.get("disallow_required_tools", True)):
            tool_policy = getattr(
                request.tool_policy, "value", str(request.tool_policy)
            )
            if tool_policy == ToolPolicy.REQUIRED.value:
                return False

        if bool(policy.get("disallow_required_rag", True)):
            rag_policy = getattr(request.rag_policy, "value", str(request.rag_policy))
            if rag_policy == RagPolicy.REQUIRED.value:
                return False

        if bool(policy.get("disallow_complex_reasoning", True)):
            metadata = request.metadata or {}
            query_complexity = (
                getattr(routing_decision, "query_complexity", None)
                or metadata.get("query_complexity")
                or ""
            )
            requires_high_reasoning = bool(
                getattr(routing_decision, "requires_high_reasoning", False)
                or metadata.get("requires_high_reasoning", False)
            )
            if str(query_complexity).lower() == "complex":
                return False
            if requires_high_reasoning:
                return False
            if context["query_type"] in {
                QueryType.REASONING.value,
                QueryType.MATH.value,
                QueryType.ANALYSIS.value,
            }:
                return False

        return True

    def _rank_local_vllm_fallback_candidates(
        self, context: Dict[str, Any], exclude: set[str]
    ) -> List[str]:
        """Rank local vLLM candidates that can safely replace a failed cloud model."""

        ranked_candidates = []
        for model_name, model_config in getattr(self.router, "models", {}).items():
            if model_name in exclude:
                continue
            if not self._local_vllm_fallback_model_eligible(model_name, context):
                continue
            score = (
                self.router._score_model(model_name, context)
                if hasattr(self.router, "_score_model")
                else 0.0
            )
            ranked_candidates.append((score, model_name))

        ranked_candidates.sort(reverse=True)
        return [model_name for _score, model_name in ranked_candidates]

    def _local_vllm_fallback_model_eligible(
        self, model_name: str, context: Dict[str, Any]
    ) -> bool:
        model_config = getattr(self.router, "models", {}).get(model_name)
        if not model_config:
            return False
        if getattr(model_config, "provider", "").lower() != "vllm":
            return False
        if self._ruled_out_by_policy(model_name, model_config.provider, context):
            return False
        if context["token_count"] > model_config.max_tokens:
            return False
        if hasattr(self.router, "_has_capability") and not self.router._has_capability(
            model_config,
            context["query_type"],
        ):
            return False
        if hasattr(
            self.router, "_check_user_access"
        ) and not self.router._check_user_access(
            model_config,
            context["user_tier"],
        ):
            return False
        return True

    def _build_fallback_routing_decision(
        self,
        routing_decision: Any,
        original_model: str,
        fallback_model: str,
        from_cache: bool = False,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> Any:
        """Attach fallback metadata to the routing decision used for downstream events."""
        suffix = (
            f"{getattr(routing_decision, 'routing_reason', 'Selected model failed')}; "
            f"fallback from {original_model} to {fallback_model}"
        )
        if from_cache:
            suffix += " (cached)"
        update: Dict[str, Any] = {
            "selected_model": fallback_model,
            "routing_reason": suffix,
        }
        if fallback_models is not None:
            update["fallback_models"] = list(fallback_models)

        if hasattr(routing_decision, "model_copy"):
            return routing_decision.model_copy(update=update)

        payload = dict(vars(routing_decision))
        payload.update(update)
        return SimpleNamespace(**payload)

    async def _publish_inference_completed_event(
        self,
        request: QueryRequest,
        response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Best-effort publication of post-inference completion events."""
        if not self.event_producer:
            return

        publish_method = getattr(
            self.event_producer, "produce_inference_completed", None
        )
        if publish_method is None:
            return

        try:
            await publish_method(request, response, routing_decision)
        except Exception as exc:
            logger.warning(f"Failed to publish inference completion event: {exc}")

    async def _publish_query_log_event(
        self,
        request: QueryRequest,
        response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Best-effort publication of query log events for ClickHouse analytics."""
        if not self.event_producer:
            return

        publish_method = getattr(self.event_producer, "produce_query_log", None)
        if publish_method is None:
            return

        try:
            await publish_method(request, response, routing_decision)
        except Exception as exc:
            logger.warning(f"Failed to publish query log event: {exc}")

    async def _publish_analytics_events(
        self,
        request: QueryRequest,
        response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Best-effort publication of post-inference analytics events."""
        await self._publish_inference_completed_event(
            request,
            response,
            routing_decision=routing_decision,
        )
        await self._publish_query_log_event(
            request,
            response,
            routing_decision=routing_decision,
        )

    async def _update_stats(
        self, model_name: str, response: InferenceResponse, inference_time: float
    ):
        """Update inference statistics"""
        if model_name not in self.inference_stats:
            self.inference_stats[model_name] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_time": 0.0,
                "error_count": 0,
            }

        stats = self.inference_stats[model_name]
        stats["total_requests"] += 1
        stats["total_tokens"] += (
            response.token_count_input + response.token_count_output
        )
        stats["total_cost"] += response.cost_usd
        stats["total_time"] += inference_time

        if response.error:
            stats["error_count"] += 1

    def _record_provider_cache_metrics(
        self, model_name: str, response: InferenceResponse
    ):
        """Record provider-side prompt cache usage separately from Redis hits."""
        if response.cached or response.error:
            return

        cached_tokens = int(response.provider_cached_input_tokens or 0)
        read_tokens = int(response.provider_cache_read_input_tokens or 0)
        creation_tokens = int(response.provider_cache_creation_input_tokens or 0)

        if cached_tokens and cached_tokens != read_tokens:
            INFERENCE_METRICS.provider_cache_tokens.labels(
                model=model_name,
                provider=response.provider,
                cache_type="cached",
            ).inc(cached_tokens)
        if read_tokens:
            INFERENCE_METRICS.provider_cache_tokens.labels(
                model=model_name,
                provider=response.provider,
                cache_type="read",
            ).inc(read_tokens)
        if creation_tokens:
            INFERENCE_METRICS.provider_cache_tokens.labels(
                model=model_name,
                provider=response.provider,
                cache_type="creation",
            ).inc(creation_tokens)

        if read_tokens or cached_tokens:
            outcome = "hit"
        elif creation_tokens:
            outcome = "write"
        else:
            outcome = "miss"
        INFERENCE_METRICS.provider_cache_requests.labels(
            model=model_name,
            provider=response.provider,
            outcome=outcome,
        ).inc()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        provider_health = {}
        for name, provider in self.providers.items():
            provider_health[name] = provider.get_health_status()

        return {
            "inference_engine": "healthy",
            "providers": provider_health,
            "cache_enabled": self.cache.enabled,
            "compression_enabled": self.context_compressor.config.get("enabled", True),
            "web_search_enabled": bool(
                self.web_search_tool and self.web_search_tool.enabled
            ),
            "rag_enabled": bool(self.rag_service and self.rag_service.enabled),
            "scheduler": self.scheduler.get_health_status(),
            "stats": self.inference_stats,
        }

    async def shutdown(self):
        """Shutdown the inference engine"""
        logger.info("Shutting down inference engine...")

        await self.batch_processor.shutdown()
        await self.tool_registry.close()

        # Close HTTP clients
        for provider in self.providers.values():
            if hasattr(provider, "aclose"):
                await provider.aclose()
            elif hasattr(provider, "http_client") and provider.http_client:
                await provider.http_client.aclose()

        # Close Redis connection
        if self.cache.redis_client:
            await self.cache.redis_client.close()

        self._initialized = False
        logger.info("Inference engine shutdown complete")
