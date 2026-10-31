from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional
import re
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# Reasoning effort levels the OpenAI Responses API accepts.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class UserTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class QueryType(str, Enum):
    GENERAL = "general"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    CREATIVE_WRITING = "creative_writing"
    BRAINSTORMING = "brainstorming"
    PLANNING = "planning"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    MATH = "math"
    REASONING = "reasoning"
    WEB_RESEARCH = "web_research"


class ToolPolicy(str, Enum):
    DISABLED = "disabled"
    AUTO = "auto"
    REQUIRED = "required"


class RagPolicy(str, Enum):
    DISABLED = "disabled"
    AUTO = "auto"
    REQUIRED = "required"


class AttachmentType(str, Enum):
    FILE = "file"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"


class Attachment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=128)
    name: str = Field(..., max_length=1024)
    type: AttachmentType
    # Declared size. POST /route replaces it with len(content) when content is
    # sent; in-process callers build attachments directly, so the schema does
    # not compare them.
    size_bytes: int
    mime_type: str = Field(..., max_length=255)
    url: Optional[str] = Field(None, max_length=4096)
    content: Optional[bytes] = None

    @field_validator("size_bytes")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("File size must be positive")
        if v > 100_000_000:  # 100MB limit
            raise ValueError("File size too large")
        return v


class WebSearchOptions(BaseModel):
    max_results: Optional[int] = Field(None, ge=1, le=10)
    search_depth: Optional[str] = Field(None, pattern=r"^(basic|advanced)$")
    include_answer: Optional[bool] = None
    topic: Optional[str] = Field(None, pattern=r"^(general|news)$")
    days: Optional[int] = Field(None, ge=1)


class ResponseSource(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""
    source_domain: Optional[str] = None
    published_at: Optional[str] = None
    score: Optional[float] = Field(None, ge=0.0)
    rank: int = Field(ge=1)
    source_type: Optional[str] = None
    document_id: Optional[str] = None
    page: Optional[int] = Field(None, ge=1)
    bbox: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    index_version: Optional[str] = None
    figure_id: Optional[str] = None
    image_ref: Optional[str] = None


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = None
    result_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    error: Optional[str] = None


BoundedName = Annotated[str, Field(max_length=256)]

# What a request id may hold: it is echoed in X-Request-ID, logged, and used
# as a Kafka and Redis key.
REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]+$"
REQUEST_ID_MAX_LENGTH = 128
_REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)


def is_valid_request_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= REQUEST_ID_MAX_LENGTH
        and _REQUEST_ID_RE.fullmatch(value) is not None
    )


class QueryRequest(BaseModel):
    _rag_images: List[Any] = PrivateAttr(default_factory=list)
    # Set after routing: the decision's features and query type, so providers
    # can match per-query settings (reasoning effort) to how it was routed.
    _routing_features: Optional[Any] = PrivateAttr(default=None)
    _routing_query_type: Optional[str] = PrivateAttr(default=None)
    # Provider-scheduler lease waits of this request's attempts (T2).
    _scheduler_wait_seconds: Optional[float] = PrivateAttr(default=None)

    # Required fields
    query: str = Field(..., min_length=1, max_length=50000)
    user_id: str = Field(..., min_length=1, max_length=256)

    # Core fields with proper types
    user_tier: UserTier = UserTier.FREE
    # Only its length is checked, as before; X-Request-ID is held to
    # REQUEST_ID_PATTERN (main.py).
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), max_length=REQUEST_ID_MAX_LENGTH
    )
    timestamp: datetime = Field(default_factory=utc_now)

    # Optional context and settings
    context: Optional[str] = Field(None, max_length=100000)
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    priority: int = Field(
        1,
        ge=1,
        le=5,
        description=(
            "Request urgency: 1 = normal ... 5 = urgent; higher is served first. "
            "Priority >= 4 also selects the low-latency lane. User tier is "
            "ranked separately, so do not derive this from it."
        ),
    )

    # File handling
    attachments: List[Attachment] = Field(default_factory=list)

    # Tool handling
    tool_policy: ToolPolicy = ToolPolicy.AUTO
    allowed_tools: List[BoundedName] = Field(default_factory=list, max_length=32)
    web_search_options: Optional[WebSearchOptions] = None

    # RAG handling
    rag_policy: RagPolicy = RagPolicy.AUTO
    knowledge_base_ids: List[BoundedName] = Field(default_factory=list, max_length=32)
    rag_options: Optional["RagOptions"] = None

    # Additional metadata
    session_id: Optional[str] = Field(None, max_length=256)
    conversation_id: Optional[str] = Field(None, max_length=256)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # The router's per-request work (classification, token counts, policy),
    # keyed by the inputs each depends on, so routing it again is cheap.
    _routing_memo: Dict[Any, Any] = PrivateAttr(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("attachments")
    @classmethod
    def validate_attachments(cls, v: List[Attachment]) -> List[Attachment]:
        if len(v) > 10:  # Max 10 attachments
            raise ValueError("Too many attachments (max 10)")
        return v


class InferenceResponse(BaseModel):
    # Core response data
    response_text: str
    model_name: str
    provider: str

    # Token tracking
    token_count_input: int = Field(ge=0)
    token_count_output: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    provider_cached_input_tokens: int = Field(default=0, ge=0)
    provider_cache_creation_input_tokens: int = Field(default=0, ge=0)
    provider_cache_read_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    # Reasoning effort sent to the provider; None when it took no such setting.
    reasoning_effort: Optional[str] = None
    # Failure bucket for analytics: provider_error, timeout, rate_limited,
    # user_or_policy or internal. None on success.
    error_type: Optional[str] = None
    # What the caller can act on: context_length_exceeded, prompt_too_large,
    # model_not_entitled, rate_limited, provider_unavailable, timeout or
    # retrieval_unavailable. None on success and for unclassified failures.
    error_code: Optional[str] = None
    # Seconds the provider asked callers to wait before retrying, when it said.
    retry_after_seconds: Optional[int] = Field(default=None, ge=0)

    # Performance metrics
    latency_ms: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)
    # Time this request spent queued: the Slack bot's ingress queue
    # (metadata ingress_queue_wait_ms) plus provider-scheduler lease waits
    # summed over every attempt, fallbacks included. None when neither was
    # measured. Never cached: a cache hit reports its own request's wait.
    queue_wait_ms: Optional[int] = Field(default=None, ge=0)
    # The provider-scheduler wait behind this response, for single-flight
    # followers that share it (T2). Not serialized.
    _scheduler_wait_seconds: Optional[float] = PrivateAttr(default=None)

    # Cost tracking
    cost_usd: float = Field(ge=0.0)

    # System flags
    cached: bool = False
    compressed_context: bool = False
    # Set when a fallback answered: the routed model whose failure it replaced.
    fallback_from_model: Optional[str] = None
    # Models whose provider was called for this answer, the one that gave it
    # included; 0 when it came from the response cache.
    attempts: int = Field(default=1, ge=0)

    # Tool output
    sources: List[ResponseSource] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_latency_ms: int = Field(default=0, ge=0)

    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Error handling
    error: Optional[str] = None
    finish_reason: str = "stop"
    warnings: List[str] = Field(default_factory=list)
    # The provider's id for the call (OpenAI x-request-id, Anthropic
    # request-id, vLLM X-Request-Id) to quote in support tickets and logs.
    provider_request_id: Optional[str] = None
    # The request this answers (QueryRequest.request_id). Never cached: a
    # cache hit answers with the id of the request it served.
    request_id: Optional[str] = None

    # Timestamps
    timestamp: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_total_tokens(self):
        if self.total_tokens == 0:
            self.total_tokens = self.token_count_input + self.token_count_output
        return self


class RagOptions(BaseModel):
    max_results: Optional[int] = Field(None, ge=1, le=20)
    candidate_count: Optional[int] = Field(None, ge=1, le=100)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_debug: bool = False


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=50000)
    user_id: str = Field(..., min_length=1)
    knowledge_base_ids: List[str] = Field(default_factory=list)
    max_results: int = Field(5, ge=1, le=20)
    candidate_count: int = Field(30, ge=1, le=100)
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_rag_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class RagUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=1024)
    size_bytes: int = Field(..., ge=1)
    checksum_sha256: str = Field(..., min_length=1, max_length=256)
    document_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagBatchDocument(BaseModel):
    filename: str = Field(..., min_length=1)
    content_base64: Optional[str] = None
    storage_ref: Optional[str] = None
    document_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_source(self):
        if not self.content_base64 and not self.storage_ref:
            raise ValueError(
                "each batch document requires content_base64 or storage_ref"
            )
        return self


class RagBatchRequest(BaseModel):
    knowledge_base_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    documents: List[RagBatchDocument] = Field(..., min_length=1, max_length=1000)


class RoutingDecisionFeatures(BaseModel):
    """How the router read the request, for checks made after routing."""

    query_complexity: Optional[str] = None  # simple, moderate or complex
    requires_high_reasoning: bool = False
    long_context: bool = False
    classification_confidence: float = Field(0.0, ge=0.0, le=1.0)


class RoutingCandidate(BaseModel):
    """One model the router considered, and why it was or was not eligible."""

    model: str = Field(..., min_length=1)
    provider: Optional[str] = None
    eligible: bool = False
    # Why the model was filtered out: tier, capability, context_window,
    # avoid_model, blocked_model, avoid_provider or blocked_provider.
    reject_reason: Optional[str] = None
    score: Optional[float] = None
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    selected: bool = False
    # Routing failed open: the guardrail reason that would have excluded
    # this otherwise eligible model.
    bypassed_guardrail: Optional[str] = None


class RoutingDecision(BaseModel):
    # Core routing decision
    selected_model: str = Field(..., min_length=1)
    query_type: QueryType
    routing_reason: str = Field(..., min_length=1)

    # Metrics
    token_count: int = Field(ge=0)
    estimated_cost: float = Field(ge=0.0)
    routing_time_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Context
    timestamp: datetime = Field(default_factory=utc_now)
    fallback_models: List[str] = Field(default_factory=list)

    # Metadata
    routing_strategy: str = "intelligent"
    user_tier: UserTier = UserTier.FREE
    route_to_fast_lane: bool = False
    actual_fast_lane_hit: bool = False
    policy_source: str = "none"
    hint_reason: Optional[str] = None
    features: RoutingDecisionFeatures = Field(default_factory=RoutingDecisionFeatures)
    # Structured counterpart of routing_reason (which keeps its free text):
    # the routing path that chose the model, e.g. rule or capability.
    reason_code: Optional[str] = None
    # Every model the router considered. fallback_models stays the chain
    # inference actually tried.
    candidates: List[RoutingCandidate] = Field(default_factory=list)
    # policy_cache.mode shadow (A1): the model the policy would have chosen,
    # and the policy's source; None when not in shadow mode or no policy.
    shadow_selected_model: Optional[str] = None
    shadow_policy_source: Optional[str] = None
    # The router's choice, kept when inference falls back and selected_model
    # becomes the fallback; None while they are the same.
    routed_model: Optional[str] = None


class ModelSelection(BaseModel):
    model_name: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)
    reason_code: Optional[str] = None


class ModelPricing(BaseModel):
    """A model's list prices in USD per 1M tokens.

    Unset cache prices fall back to the provider's prompt_cache multipliers
    (cached input 0.5x or 0.1x, 5-minute writes 1.25x, 1-hour writes 2x).
    """

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cached_input: Optional[float] = Field(None, ge=0.0)
    cache_write_5m: Optional[float] = Field(None, ge=0.0)
    cache_write_1h: Optional[float] = Field(None, ge=0.0)


class ModelConfig(BaseModel):
    # Core model info
    name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model_path: Optional[str] = None

    # Capacity settings
    max_tokens: int = Field(ge=1)
    # Tokens of input plus output the model accepts (vLLM's max_model_len);
    # unset, it is max_tokens, which configs have always set to the window.
    context_window: Optional[int] = Field(None, ge=1)
    # The most output tokens the model generates; unset means no cap below
    # the window. request.max_tokens is clamped to it.
    max_output_tokens: Optional[int] = Field(None, ge=1)
    # The one rate used for input and output when pricing is unset.
    cost_per_token: float = Field(ge=0.0)
    # List prices; a model without them inherits a priced model's by prefix
    # (gpt-5.4 -> gpt-5), else falls back to cost_per_token with a warning.
    pricing: Optional[ModelPricing] = None
    priority: int = Field(ge=1)

    # Capabilities
    capabilities: List[str] = Field(..., min_length=1)

    # Resource requirements
    gpu_memory_gb: Optional[int] = Field(None, ge=1)

    # Authentication
    api_key_env: Optional[str] = None

    # Entitlement: the tiers that may use this model. None keeps the old rule,
    # which derived access from priority (see entitled_tiers).
    allowed_tiers: Optional[List[UserTier]] = None

    # Reasoning effort sent to the provider: reasoning_effort for most
    # queries, reasoning_effort_high for queries routed as complex reasoning.
    # None keeps the name rule (gpt-5* uses minimal and high, others send none).
    reasoning_effort: Optional[ReasoningEffort] = None
    reasoning_effort_high: Optional[ReasoningEffort] = None
    # Output tokens added on top of the caller's max_tokens when the effort is
    # above low, so reasoning does not eat the visible answer's budget. None
    # keeps the name rule (4096 for gpt-5*, 0 otherwise).
    reasoning_output_headroom_tokens: Optional[int] = Field(None, ge=0)
    # Whether the provider accepts a temperature. None keeps the name rule
    # (gpt-5* reasoning models only use their default sampling).
    supports_temperature: Optional[bool] = None

    def entitled_tiers(self) -> List[UserTier]:
        """Tiers that may use this model.

        Without allowed_tiers, free may use priority >= 3, premium >= 2 and
        enterprise any priority, as before allowed_tiers existed.
        """
        if self.allowed_tiers is not None:
            return list(dict.fromkeys(self.allowed_tiers))
        minimum_priority = {
            UserTier.FREE: 3,
            UserTier.PREMIUM: 2,
            UserTier.ENTERPRISE: 1,
        }
        return [
            tier
            for tier, minimum in minimum_priority.items()
            if self.priority >= minimum
        ]

    @model_validator(mode="after")
    def default_context_window(self) -> "ModelConfig":
        if self.context_window is None:
            self.context_window = self.max_tokens
        if (
            self.max_output_tokens is not None
            and self.max_output_tokens > self.context_window
        ):
            raise ValueError(
                f"max_output_tokens ({self.max_output_tokens}) exceeds "
                f"context_window ({self.context_window})"
            )
        return self

    def window_tokens(self) -> int:
        """The context window: input plus output tokens."""
        return self.context_window or self.max_tokens

    def output_tokens(self, requested: Any) -> int:
        """The output tokens a request for ``requested`` gets from this model."""
        tokens = max(0, int(requested or 0))
        if self.max_output_tokens is not None:
            return min(tokens, self.max_output_tokens)
        return tokens

    def fits_context(self, input_tokens: Any, requested_output: Any) -> bool:
        """Whether the prompt and its output fit the context window together."""
        return (
            int(input_tokens or 0) + self.output_tokens(requested_output)
            <= self.window_tokens()
        )

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: List[str]) -> List[str]:
        valid_capabilities = {
            "reasoning",
            "coding",
            "analysis",
            "writing",
            "creative",
            "general",
            "math",
            "translation",
            "tool_use",
            "web_search",
            "vision",
        }
        for cap in v:
            if cap not in valid_capabilities:
                raise ValueError(f"Invalid capability: {cap}")
        return v


class SystemMetric(BaseModel):
    name: str = Field(..., min_length=1)
    value: float
    timestamp: datetime = Field(default_factory=utc_now)
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("Metric name must be alphanumeric with underscores/dots")
        return v


# Additional models for specific use cases
class UserSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    user_tier: UserTier
    start_time: datetime = Field(default_factory=utc_now)
    last_activity: datetime = Field(default_factory=utc_now)
    query_count: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    total_cost: float = Field(0.0, ge=0.0)


class ErrorResponse(BaseModel):
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    request_id: Optional[str] = None
    retry_after: Optional[int] = None  # seconds


# Health check models
class HealthStatus(BaseModel):
    service: str
    status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=utc_now)
    details: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


class ComponentHealth(BaseModel):
    overall_status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    components: Dict[str, HealthStatus]
    timestamp: datetime = Field(default_factory=utc_now)


class ConfigModel(BaseModel):
    """Base model for platform configuration sections."""

    model_config = ConfigDict(extra="allow")


class ApiRateLimitRedisConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(6379, ge=1, le=65535)
    db: int = Field(4, ge=0)
    password_env: Optional[str] = "REDIS_PASSWORD"
    url: Optional[str] = None
    key_prefix: str = "api_gateway"
    connect_timeout_ms: int = Field(100, ge=1)
    socket_timeout_ms: int = Field(100, ge=1)
    max_connections: int = Field(100, ge=1)
    recovery_cooldown_ms: int = Field(1000, ge=0)


class ApiRateLimitQueueConfig(ConfigModel):
    enabled: bool = True
    max_depth: int = Field(1000, ge=0)
    timeout_ms: int = Field(250, ge=0)
    poll_interval_ms: int = Field(25, ge=1)
    max_poll_interval_ms: int = Field(250, ge=1)
    poll_jitter_ratio: float = Field(0.2, ge=0.0, le=1.0)
    queue_lease_grace_ms: int = Field(1000, ge=0)
    control_plane_version: str = "v2"


class ApiRateLimitBucketConfig(ConfigModel):
    enabled: bool = True
    requests_per_minute: int = Field(0, ge=0)
    burst_size: int = Field(0, ge=0)


class ApiRateLimitActiveConfig(ConfigModel):
    enabled: bool = True
    active_requests: int = Field(0, ge=0)


class ApiTokenBudgetConfig(ConfigModel):
    enabled: bool = False
    tokens_per_minute: int = Field(0, ge=0)
    burst_tokens: int = Field(0, ge=0)


class ApiRateLimitScopeConfig(ConfigModel):
    requests_per_minute: Optional[int] = Field(None, ge=0)
    burst_size: Optional[int] = Field(None, ge=0)
    active_requests: Optional[int] = Field(None, ge=0)
    tokens_per_minute: Optional[int] = Field(None, ge=0)
    burst_tokens: Optional[int] = Field(None, ge=0)


class ApiRateLimitGlobalConfig(ApiRateLimitScopeConfig):
    active_requests: Optional[int] = Field(100, ge=0)


class ApiRateLimitingConfig(ConfigModel):
    enabled: bool = False
    requests_per_minute: int = Field(1000, ge=1)
    burst_size: int = Field(100, ge=1)
    failure_mode: str = "closed"
    redis: ApiRateLimitRedisConfig = Field(default_factory=ApiRateLimitRedisConfig)
    queue: ApiRateLimitQueueConfig = Field(default_factory=ApiRateLimitQueueConfig)
    per_user: ApiRateLimitScopeConfig = Field(default_factory=ApiRateLimitScopeConfig)
    by_tier: Dict[str, ApiRateLimitScopeConfig] = Field(default_factory=dict)
    global_limits: ApiRateLimitGlobalConfig = Field(
        default_factory=ApiRateLimitGlobalConfig
    )
    providers: Dict[str, ApiRateLimitScopeConfig] = Field(default_factory=dict)
    models: Dict[str, ApiRateLimitScopeConfig] = Field(default_factory=dict)
    token_budget: ApiTokenBudgetConfig = Field(default_factory=ApiTokenBudgetConfig)
    # Active-request counters live in time buckets of this length. Keep it above
    # the longest request (deadline plus grace): a slot whose release is lost
    # stops counting after at most two windows.
    active_window_ms: int = Field(180_000, ge=1000)

    @field_validator("failure_mode")
    @classmethod
    def validate_failure_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"open", "closed"}:
            raise ValueError("failure_mode must be one of: open, closed")
        return normalized


class ApiConfig(ConfigModel):
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)
    log_level: str = "info"
    rate_limiting: ApiRateLimitingConfig = Field(default_factory=ApiRateLimitingConfig)


class LoggingConfig(ConfigModel):
    level: str = "INFO"
    file: str = "logs/llm_router.log"
    max_bytes: int = Field(10_485_760, ge=1)
    backup_count: int = Field(5, ge=0)
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class RouterClassifierConfig(ConfigModel):
    # Deprecated no-op (R16c): the classifier loads no semantic model. Kept so
    # configs that still set it load.
    semantic_enabled: bool = True


class RouterConfig(ConfigModel):
    default_model: str = "mistral-7b"
    routing_strategy: str = "intelligent"
    classifier: RouterClassifierConfig = Field(default_factory=RouterClassifierConfig)
    tokenizer_enabled: bool = True
    # Half-life of the time-decayed model stats routing scores with (R13).
    stats_half_life_seconds: float = Field(300.0, gt=0)
    # Requests' worth of prior toward the lifetime success rate (R13 review).
    stats_prior_weight: float = Field(3.0, ge=0)
    credential_gating: bool = False
    fast_lane_models: List[str] = Field(default_factory=list)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    routing_rules: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("models", mode="before")
    @classmethod
    def populate_model_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = {}
        for model_name, model_config in value.items():
            if isinstance(model_config, dict) and "name" not in model_config:
                normalized[model_name] = {"name": model_name, **model_config}
            else:
                normalized[model_name] = model_config
        return normalized

    @model_validator(mode="after")
    def require_a_model_for_every_tier(self) -> "RouterConfig":
        # The router rejects requests from a tier with no model instead of
        # upgrading them, so a config that leaves a tier out is a mistake.
        if not self.models:
            return self
        missing = [
            tier.value
            for tier in UserTier
            if not any(tier in model.entitled_tiers() for model in self.models.values())
        ]
        if missing:
            raise ValueError(
                f"router.models: no model is open to tier(s) {', '.join(missing)}; "
                "set allowed_tiers so every tier has at least one model"
            )
        return self


class ProviderPoolEndpointConfig(ConfigModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    models: List[str] = Field(default_factory=list)
    weight: float = Field(1.0, gt=0)
    max_outstanding: int = Field(0, ge=0)
    health_path: str = "/health"
    metrics_path: str = "/metrics"
    prefix_cache_enabled: bool = True
    enabled: bool = True


class ProviderModelFallbackConfig(ConfigModel):
    enabled: bool = False
    fallbacks: Dict[str, List[str]] = Field(default_factory=dict)
    allowed_query_types: List[str] = Field(
        default_factory=lambda: [
            QueryType.GENERAL.value,
            QueryType.CODE_GENERATION.value,
            QueryType.CODE_ANALYSIS.value,
            QueryType.TRANSLATION.value,
        ]
    )
    max_input_tokens: int = Field(2048, ge=1)
    max_output_tokens: int = Field(1024, ge=1)
    disallow_attachments: bool = True
    disallow_complex_reasoning: bool = True
    disallow_required_tools: bool = True
    disallow_required_rag: bool = True


class ProviderEndpointConfig(ConfigModel):
    enabled: bool = True
    base_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    timeout: Optional[int] = Field(None, ge=1)
    max_retries: Optional[int] = Field(None, ge=0)
    api_mode: Optional[str] = Field(None, pattern=r"^(completions|chat_completions)$")
    endpoints: List[ProviderPoolEndpointConfig] = Field(default_factory=list)
    routing_strategy: Optional[str] = Field(
        None, pattern=r"^(least_outstanding|least_outstanding_prefix_aware)$"
    )
    health_check_interval_seconds: Optional[float] = Field(None, gt=0)
    failure_cooldown_seconds: Optional[float] = Field(None, ge=0)
    metrics_refresh_seconds: Optional[float] = Field(None, gt=0)
    prefix_affinity_ttl_seconds: Optional[float] = Field(None, ge=0)
    metrics_scrape_enabled: bool = False
    validate_models_on_health: bool = False
    model_fallback: ProviderModelFallbackConfig = Field(
        default_factory=ProviderModelFallbackConfig
    )


class CacheConfig(ConfigModel):
    enabled: bool = False
    backend: str = "memory"
    ttl: int = Field(3600, ge=1)
    max_size: str = "1GB"
    redis: Dict[str, Any] = Field(default_factory=dict)


class BatchingConfig(ConfigModel):
    enabled: bool = False
    max_batch_size: int = Field(32, ge=1)
    max_wait_time_ms: int = Field(50, ge=1)


class SingleFlightConfig(ConfigModel):
    enabled: bool = False


class SchedulerRetryConfig(ConfigModel):
    max_attempts_per_request: int = Field(1, ge=1)
    initial_backoff_ms: int = Field(100, ge=0)
    max_backoff_ms: int = Field(1000, ge=0)
    budget_enabled: bool = False
    budget_tokens: int = Field(0, ge=0)
    budget_window_seconds: int = Field(60, ge=1)


class SchedulerCircuitBreakerConfig(ConfigModel):
    enabled: bool = False
    failure_threshold: int = Field(5, ge=1)
    recovery_timeout_ms: int = Field(30000, ge=0)
    half_open_max_requests: int = Field(1, ge=1, le=1)
    state_ttl_seconds: int = Field(3600, ge=1)


class ProviderSchedulerConfig(ConfigModel):
    enabled: bool = True
    queue_enabled: bool = False
    wait_timeout_ms: int = Field(250, ge=0)
    poll_interval_ms: int = Field(25, ge=1)
    max_poll_interval_ms: int = Field(250, ge=1)
    poll_jitter_ratio: float = Field(0.2, ge=0.0, le=1.0)
    queue_lease_grace_ms: int = Field(1000, ge=0)
    control_plane_version: str = "v2"
    failure_mode: str = "closed"
    allow_fallback_on_provider_rejection: bool = True
    key_prefix: str = "provider_scheduler"
    request_deadline_seconds: float = Field(60.0, gt=0)
    # Share of the time left when model calls start that a primary model
    # leaves for its fallback route, when it has one.
    fallback_reserve_ratio: float = Field(0.3, ge=0.0, lt=1.0)
    # A provider attempt (or retry) is not started with less time than this
    # left (capped at the provider timeout); it fails as a deadline timeout.
    min_attempt_window_seconds: float = Field(1.0, ge=0.0)
    # An attempt timeout counts against the provider's circuit only when the
    # attempt had at least this share of the longest window the request could
    # give it: the provider timeout, capped at the deadline less any reserve.
    circuit_timeout_min_window_ratio: float = Field(0.5, ge=0.0, le=1.0)
    retry: SchedulerRetryConfig = Field(default_factory=SchedulerRetryConfig)
    circuit_breaker: SchedulerCircuitBreakerConfig = Field(
        default_factory=SchedulerCircuitBreakerConfig
    )

    @field_validator("failure_mode")
    @classmethod
    def validate_failure_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"open", "closed"}:
            raise ValueError("failure_mode must be one of: open, closed")
        return normalized


class InferenceFallbackConfig(ConfigModel):
    # Models to try, in order, when a model is unavailable (retryable errors,
    # circuit or capacity rejections, attempt timeouts). Each must still pass
    # the user's tier, the context window and the routing guardrails.
    chains: Dict[str, List[str]] = Field(default_factory=dict)


class InferenceConfig(ConfigModel):
    vllm: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    openai: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    anthropic: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    single_flight: SingleFlightConfig = Field(default_factory=SingleFlightConfig)
    scheduler: ProviderSchedulerConfig = Field(default_factory=ProviderSchedulerConfig)
    fallback: InferenceFallbackConfig = Field(default_factory=InferenceFallbackConfig)
    # Share of the time left in the request deadline that one retrieval step
    # (RAG, then web search) may use, so the model call keeps the rest (and
    # never less than scheduler.min_attempt_window_seconds).
    retrieval_budget_ratio: float = Field(0.5, gt=0.0, lt=1.0)

    @model_validator(mode="before")
    @classmethod
    def map_legacy_batching_to_single_flight(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "single_flight" in value:
            return value
        batching = value.get("batching")
        if isinstance(batching, dict) and "enabled" in batching:
            mapped = dict(value)
            mapped["single_flight"] = {"enabled": bool(batching["enabled"])}
            return mapped
        return value


class WebSearchToolConfig(ConfigModel):
    enabled: bool = False
    provider: str = "tavily"
    api_key_env: str = "TAVILY_API_KEY"
    api_key: Optional[str] = None
    base_url: str = "https://api.tavily.com"
    timeout_seconds: float = Field(5.0, gt=0)
    max_results: int = Field(5, ge=1, le=10)
    cache_ttl_seconds: int = Field(300, ge=0)
    cache_max_entries: int = Field(512, ge=1)
    per_user_rate_limit: int = Field(20, ge=1)
    rate_limiter_max_users: int = Field(10000, ge=1)
    search_depth: str = Field("basic", pattern=r"^(basic|advanced)$")
    include_answer: bool = False
    max_results_per_domain: int = Field(2, ge=1)
    blocked_domains: List[str] = Field(default_factory=list)
    freshness_days: Optional[int] = Field(14, ge=1)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "tavily":
            raise ValueError("web_search provider must be tavily")
        return normalized


class ToolsConfig(ConfigModel):
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)


class KafkaProducerConfig(ConfigModel):
    acks: str = "all"
    retries: int = Field(3, ge=0)
    batch_size: int = Field(16_384, ge=1)
    linger_ms: int = Field(5, ge=0)
    compression_type: str = "gzip"
    request_timeout_ms: int = Field(40_000, ge=1)
    retry_backoff_ms: int = Field(100, ge=0)
    send_timeout_seconds: float = Field(40.0, gt=0)
    enable_idempotence: bool = True
    wait_for_ack: bool = True
    raise_on_failure: bool = False
    health_failure_threshold: int = Field(5, ge=1)
    queue_capacity: int = Field(10_000, ge=1)
    dispatcher_batch_size: int = Field(256, ge=1)
    shutdown_drain_timeout_seconds: float = Field(10.0, gt=0)
    shutdown_cancel_timeout_seconds: float = Field(1.0, gt=0)
    shutdown_producer_timeout_seconds: float = Field(1.0, gt=0)


class KafkaConsumerConfig(ConfigModel):
    group_id: str = "llm-router-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = Field(500, ge=1)
    batch_size: int = Field(100, ge=1)
    batch_high_watermark_rows: int = Field(5000, ge=1)
    batch_low_watermark_rows: int = Field(2500, ge=0)
    max_concurrent_flushes: int = Field(4, ge=1)
    dlq_enabled: bool = True
    supervisor_initial_backoff_seconds: float = Field(1.0, gt=0)
    supervisor_max_backoff_seconds: float = Field(30.0, gt=0)


class KafkaDlqConfig(ConfigModel):
    enabled: bool = True
    topic_suffix: str = ".dlq"
    send_timeout_seconds: float = Field(10.0, gt=0)


class KafkaConfig(ConfigModel):
    bootstrap_servers: List[str] = Field(default_factory=list)
    topics: Dict[str, str] = Field(default_factory=dict)
    dead_letter_topics: Dict[str, str] = Field(default_factory=dict)
    producer: KafkaProducerConfig = Field(default_factory=KafkaProducerConfig)
    consumer: KafkaConsumerConfig = Field(default_factory=KafkaConsumerConfig)
    dlq: KafkaDlqConfig = Field(default_factory=KafkaDlqConfig)
    enabled: bool = False


class ClickHouseConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(8123, ge=1, le=65535)
    database: str = "llm_router"
    username: str = "default"
    password: str = ""
    connect_timeout_seconds: float = Field(2.0, gt=0)
    send_receive_timeout_seconds: float = Field(10.0, gt=0)
    dashboard: Dict[str, Any] = Field(
        default_factory=lambda: {
            "cache_ttl_seconds": 5,
            "cache_max_entries": 32,
            "max_concurrent_queries": 4,
        }
    )
    enabled: bool = False


class MonitoringConfig(ConfigModel):
    prometheus_port: int = Field(8000, ge=1, le=65535)
    prometheus: Dict[str, Any] = Field(default_factory=dict)
    grafana: Dict[str, Any] = Field(default_factory=dict)
    alerts: Dict[str, Any] = Field(default_factory=dict)
    health_checks: Dict[str, Any] = Field(default_factory=dict)
    stream_metrics_staleness_seconds: int = Field(300, ge=0)
    enabled: bool = False


class SlackResponseSettingsConfig(ConfigModel):
    max_response_length: int = Field(2000, ge=1)
    typing_indicator: bool = True
    # A context line under each answer: model, latency and "cached".
    answer_footer: bool = True
    # Thumbs up / down under mention and thread answers, sent as chat.feedback.
    feedback_buttons: bool = False
    thread_replies: bool = True


class SlackRateLimitingConfig(ConfigModel):
    requests_per_hour: int = Field(100, ge=1)
    burst_requests: int = Field(5, ge=1)


class SlackContextConfig(ConfigModel):
    enabled: bool = True
    strategy: str = "thread_first"
    max_thread_messages: int = Field(20, ge=1)
    max_channel_messages: int = Field(10, ge=1)
    max_context_chars: int = Field(4000, ge=1)
    include_bot_messages: bool = False
    timeout_seconds: float = Field(3.0, gt=0)
    fail_open: bool = True

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"thread_first"}:
            raise ValueError("strategy must be one of: thread_first")
        return normalized


class SlackAgentContextConfig(ConfigModel):
    enabled: bool = True
    max_thread_messages: int = Field(20, ge=1, le=100)
    max_message_chars: int = Field(4000, ge=1)
    max_resources: int = Field(10, ge=0, le=20)
    max_resource_bytes: int = Field(262144, ge=1)
    max_resource_chars: int = Field(8000, ge=0)
    max_total_chars: int = Field(12000, ge=1)
    timeout_seconds: float = Field(10.0, gt=0)
    include_bot_messages: bool = False
    fail_open: bool = True


class SlackWorkQueueConfig(ConfigModel):
    capacity: int = Field(256, ge=1)
    concurrency: int = Field(16, ge=1)
    # Workers reserved for button clicks (approvals, feedback).
    priority_concurrency: int = Field(2, ge=1)
    overload_reply_timeout_seconds: float = Field(2.0, gt=0)


class SlackMemorySearchConfig(ConfigModel):
    max_results: int = Field(5, ge=1)
    max_context_chars: int = Field(2000, ge=1)
    max_item_chars: int = Field(500, ge=1)
    keyword_weight: float = Field(0.45, ge=0.0)
    vector_weight: float = Field(0.45, ge=0.0)
    recency_weight: float = Field(0.05, ge=0.0)
    importance_weight: float = Field(0.05, ge=0.0)
    # Cosine-similarity floor for vector matches; 0 keeps every neighbour.
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class SlackMemoryEmbeddingConfig(ConfigModel):
    provider: str = "none"
    model: str = "text-embedding-3-small"
    dimensions: int = Field(1536, ge=1)
    timeout: int = Field(10, ge=1)
    api_key: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    url: Optional[str] = None


class SlackMemoryRedisConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(6379, ge=1, le=65535)
    db: int = Field(3, ge=0)
    url: Optional[str] = None
    password_env: Optional[str] = None
    key_prefix: str = "slack_memory"
    dedicated_service_recommended: bool = True


class SlackMemoryConfig(ConfigModel):
    enabled: bool = False
    backend: str = "memory"
    key_prefix: str = "slack_memory"
    max_items_per_user: int = Field(500, ge=1)
    ttl_days: Optional[int] = Field(None, ge=1)
    retention_days: Optional[int] = Field(None, ge=1)
    search: SlackMemorySearchConfig = Field(default_factory=SlackMemorySearchConfig)
    embedding: SlackMemoryEmbeddingConfig = Field(
        default_factory=SlackMemoryEmbeddingConfig
    )
    redis: SlackMemoryRedisConfig = Field(default_factory=SlackMemoryRedisConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"memory", "redis_stack"}:
            raise ValueError("memory backend must be one of: memory, redis_stack")
        return normalized


class SlackConfig(ConfigModel):
    enabled: bool = False
    bot_token: Optional[str] = None
    bot_token_env: Optional[str] = None
    app_token: Optional[str] = None
    app_token_env: Optional[str] = None
    channels: List[str] = Field(default_factory=list)
    response_settings: SlackResponseSettingsConfig = Field(
        default_factory=SlackResponseSettingsConfig
    )
    rate_limiting: SlackRateLimitingConfig = Field(
        default_factory=SlackRateLimitingConfig
    )
    context: SlackContextConfig = Field(default_factory=SlackContextConfig)
    agent_context: SlackAgentContextConfig = Field(
        default_factory=SlackAgentContextConfig
    )
    work_queue: SlackWorkQueueConfig = Field(default_factory=SlackWorkQueueConfig)
    state_backend: str = "memory"
    state_file: str = "data/slack_state.json"
    state_key_prefix: str = "slack_state"
    redis: Dict[str, Any] = Field(default_factory=dict)
    memory: SlackMemoryConfig = Field(default_factory=SlackMemoryConfig)

    @field_validator("state_backend")
    @classmethod
    def validate_state_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"memory", "file", "redis"}:
            raise ValueError("state_backend must be one of: memory, file, redis")
        return normalized


class StreamlitConfig(ConfigModel):
    enabled: bool = True
    port: int = Field(8501, ge=1, le=65535)
    host: str = "0.0.0.0"
    theme: str = "dark"
    dashboard: Dict[str, Any] = Field(default_factory=dict)


class FlinkConfig(ConfigModel):
    enabled: bool = False
    job_manager_host: str = "localhost"
    job_manager_port: int = Field(8081, ge=1, le=65535)
    parallelism: int = Field(4, ge=1)
    checkpointing_interval_ms: int = Field(5000, ge=1)
    analytics: Dict[str, Any] = Field(default_factory=dict)


class PolicyCacheConfig(ConfigModel):
    enabled: bool = False
    # enforce: the policy changes routing. shadow (A1): routing ignores it and
    # records what it would have chosen (RoutingDecision.shadow_selected_model).
    mode: Literal["enforce", "shadow"] = "enforce"
    request_ttl_seconds: int = Field(300, ge=1)
    user_ttl_seconds: int = Field(900, ge=1)
    local_cache_ttl_seconds: int = Field(5, ge=0)
    local_cache_max_entries: int = Field(4096, ge=1)
    redis: Dict[str, Any] = Field(default_factory=dict)
    consumer: Dict[str, Any] = Field(default_factory=dict)
    # Redis bounds and recovery (R10): routing waits read_timeout_ms for the
    # policy, then routes without it; a failed Redis is pinged again after
    # recovery_cooldown_ms.
    read_timeout_ms: int = Field(50, ge=1)
    read_timeout_failure_threshold: int = Field(5, ge=1)
    recovery_cooldown_ms: int = Field(1000, ge=0)


class ApiKeyClientConfig(ConfigModel):
    """An API key mapped to a client and what that client may ask for."""

    client_id: str = Field(..., min_length=1, max_length=128)
    # Exactly one key reference: an environment variable holding the key, or
    # the hex SHA-256 of the key.
    key_env: Optional[str] = Field(None, min_length=1)
    sha256: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F]{64}$")
    # Highest user_tier the client may request; higher tiers are clamped.
    max_tier: UserTier = UserTier.FREE
    # Per-client request rate (0 disables it); burst defaults to rpm.
    rpm: int = Field(60, ge=0)
    burst: Optional[int] = Field(None, ge=1)
    # Per-client concurrent /route requests (0 disables it).
    active: int = Field(10, ge=0)
    # Whether the client's priority and latency metadata are honoured.
    allow_priority: bool = False

    @model_validator(mode="after")
    def validate_key_reference(self) -> "ApiKeyClientConfig":
        if bool(self.key_env) == bool(self.sha256):
            raise ValueError("set exactly one of key_env or sha256")
        return self


class ApiKeysSecurityConfig(ConfigModel):
    enabled: bool = False
    header_name: str = "X-API-Key"
    # Keys listed here without a client entry authenticate as an unmapped
    # principal: max_tier free, no priority, the unmapped_* limits below.
    env_var: str = "LLM_ROUTER_API_KEYS"
    clients: List[ApiKeyClientConfig] = Field(default_factory=list)
    unmapped_rpm: int = Field(60, ge=0)
    unmapped_active: int = Field(10, ge=0)

    @model_validator(mode="after")
    def validate_unique_clients(self) -> "ApiKeysSecurityConfig":
        client_ids = [client.client_id for client in self.clients]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("api_keys.clients client_id values must be unique")
        return self


class CorsSecurityConfig(ConfigModel):
    enabled: bool = False
    allow_origins: List[str] = Field(default_factory=list)
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = Field(default_factory=list)
    allow_credentials: bool = False


class SecurityConfig(ConfigModel):
    api_keys: ApiKeysSecurityConfig = Field(default_factory=ApiKeysSecurityConfig)
    cors: CorsSecurityConfig = Field(default_factory=CorsSecurityConfig)
    # Request bodies above this size get 413 before they are parsed.
    max_request_body_bytes: int = Field(2 * 1024 * 1024, ge=1)
    # Per-path caps by media type ("*" matches any), e.g.
    # {"/rag/batches": {"application/json": 20000000}}. A path listed here
    # replaces its default; /rag/documents and /rag/batches otherwise get caps
    # derived from rag.upload. Other media types use max_request_body_bytes.
    request_body_limits: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    # POST /route rejects an attachment whose inline content is above this size.
    max_attachment_bytes: int = Field(1024 * 1024, ge=1)

    @field_validator("request_body_limits")
    @classmethod
    def validate_request_body_limits(
        cls, value: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, int]]:
        for path, by_media_type in value.items():
            if not path.startswith("/"):
                raise ValueError(f"request_body_limits path must start with /: {path}")
            for media_type, limit in by_media_type.items():
                if limit < 1:
                    raise ValueError(
                        f"request_body_limits[{path}][{media_type}] must be >= 1"
                    )
        return {
            path: {
                media_type.strip().lower(): limit
                for media_type, limit in by_media_type.items()
            }
            for path, by_media_type in value.items()
        }


class DatabasePoolConfig(ConfigModel):
    max_connections: int = Field(20, ge=1)
    min_connections: int = Field(5, ge=0)

    @model_validator(mode="after")
    def validate_pool_bounds(self):
        if self.min_connections > self.max_connections:
            raise ValueError("min_connections cannot exceed max_connections")
        return self


class HttpPoolConfig(ConfigModel):
    max_connections: int = Field(100, ge=1)
    timeout: int = Field(30, ge=1)


class ConnectionPoolsConfig(ConfigModel):
    database: DatabasePoolConfig = Field(default_factory=DatabasePoolConfig)
    http: HttpPoolConfig = Field(default_factory=HttpPoolConfig)


class WorkerConfig(ConfigModel):
    api_workers: int = Field(1, ge=1)
    inference_workers: int = Field(1, ge=1)
    pipeline_workers: int = Field(1, ge=1)


class MemoryConfig(ConfigModel):
    max_heap_size: str = "8G"
    gc_strategy: str = "G1GC"


class PerformanceConfig(ConfigModel):
    connection_pools: ConnectionPoolsConfig = Field(
        default_factory=ConnectionPoolsConfig
    )
    workers: WorkerConfig = Field(default_factory=WorkerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class DevelopmentConfig(ConfigModel):
    debug: bool = False
    auto_reload: bool = False
    profiling: bool = False
    mock_external_apis: bool = False


class FeatureFlagsConfig(ConfigModel):
    semantic_caching: bool = True
    multi_modal: bool = False
    function_calling: bool = True


class PipelineConfig(ConfigModel):
    enabled: bool = False


class RagParserConfig(ConfigModel):
    provider: str = "docling"
    allow_cloud_fallback: bool = False
    max_file_size_bytes: int = Field(100_000_000, ge=1)
    timeout_seconds: int = Field(300, ge=1)
    ocr_enabled: bool = True
    ocr_languages: List[str] = Field(default_factory=list)
    table_structure_enabled: bool = True
    converter_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"docling", "text"}:
            raise ValueError("rag parser provider must be one of: docling, text")
        return normalized


class RagChunkingConfig(ConfigModel):
    strategy: str = "hybrid"
    chunk_size_tokens: int = Field(900, ge=100, le=4000)
    chunk_overlap_tokens: int = Field(120, ge=0, le=1000)
    repeat_table_headers: bool = True
    max_chunk_chars: int = Field(6000, ge=500)


class RagLayoutNormalizationConfig(ConfigModel):
    enabled: bool = True
    repeated_block_min_pages: int = Field(3, ge=2)
    repeated_block_min_ratio: float = Field(0.60, gt=0.0, le=1.0)
    top_margin_ratio: float = Field(0.10, ge=0.0, le=0.5)
    bottom_margin_ratio: float = Field(0.10, ge=0.0, le=0.5)
    page_seam_margin_ratio: float = Field(0.20, ge=0.0, le=0.5)
    unresolved_policy: str = "preserve_with_warning"


class RagEmbeddingConfig(ConfigModel):
    provider: str = "local_http"
    model: str = "BAAI/bge-m3"
    dimensions: int = Field(1024, ge=1)
    timeout: int = Field(30, ge=1)
    api_key: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    url: Optional[str] = "http://127.0.0.1:8002/v1"
    batch_size: int = Field(32, ge=1, le=2048)
    max_concurrent_batches: int = Field(2, ge=1, le=32)


class RagRedisConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(6380, ge=1, le=65535)
    db: int = Field(0, ge=0)
    url: Optional[str] = None
    password_env: Optional[str] = None
    key_prefix: str = "rag"
    pipeline_batch_size: int = Field(64, ge=1, le=1000)


class RagIndexingConfig(ConfigModel):
    control_plane_version: str = "v2"
    atomic_commit_max_chunks: int = Field(2000, ge=1, le=10000)
    staging_ttl_seconds: int = Field(86400, ge=60)


class RagRetrievalConfig(ConfigModel):
    top_k: int = Field(5, ge=1, le=20)
    candidate_count: int = Field(30, ge=1, le=100)
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    keyword_scorer: str = "BM25STD"
    keyword_score_normalization: str = "max"
    keyword_weight: float = Field(0.35, ge=0.0)
    vector_weight: float = Field(0.6, ge=0.0)
    recency_weight: float = Field(0.05, ge=0.0)
    max_context_chars: int = Field(6000, ge=500)
    max_concurrent_branches: int = Field(3, ge=1, le=3)
    deadline_seconds: float = Field(30.0, gt=0.0, le=300.0)


class RagRerankConfig(ConfigModel):
    enabled: bool = False
    provider: str = "sentence_transformers"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = Field(8, ge=1, le=50)
    timeout: int = Field(30, ge=1)
    url: Optional[str] = None


class RagIntentGateConfig(ConfigModel):
    strong_terms: List[str] = Field(default_factory=list)
    context_terms: List[str] = Field(default_factory=list)
    school_context_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)


class RagIngestionQueueConfig(ConfigModel):
    enabled: bool = False
    stream_key: str = "rag:ingestion:stream"
    group_name: str = "rag-ingestion-workers"
    dead_letter_stream_key: str = "rag:ingestion:dead_letter"
    retry_zset_key: str = "rag:ingestion:retry"
    consumer_count: int = Field(1, ge=1)
    concurrency: int = Field(1, ge=1)
    block_ms: int = Field(5000, ge=1)
    pending_idle_ms: int = Field(300000, ge=1)
    max_attempts: int = Field(3, ge=1)
    retry_backoff_seconds: float = Field(30.0, ge=0.0)
    stream_maxlen: int = Field(10000, ge=0)
    dead_letter_maxlen: int = Field(10000, ge=1)
    heartbeat_interval_seconds: float = Field(5.0, gt=0.0)
    heartbeat_ttl_seconds: int = Field(15, ge=1)


class RagStorageConfig(ConfigModel):
    staging_dir: str = "data/rag/uploads"
    cleanup_completed_files: bool = False
    completed_file_ttl_seconds: int = Field(86400, ge=0)
    failed_file_ttl_seconds: int = Field(604800, ge=1)
    orphan_file_ttl_seconds: int = Field(604800, ge=1)
    janitor_interval_seconds: float = Field(300.0, gt=0)
    max_staging_bytes: int = Field(10737418240, ge=1)
    high_watermark_ratio: float = Field(0.9, gt=0.0, le=1.0)


class RagUploadConfig(ConfigModel):
    json_max_decoded_bytes: int = Field(10_000_000, ge=1)
    multipart_max_bytes: int = Field(100_000_000, ge=1)
    stream_chunk_bytes: int = Field(1_048_576, ge=1024, le=16_777_216)


class RagVisualProviderConfig(ConfigModel):
    enabled: bool = False
    provider: str = ""
    model: str = ""
    url: Optional[str] = None
    timeout: int = Field(30, ge=1)


class RagVisualEmbeddingConfig(RagVisualProviderConfig):
    provider: str = "nomic_multimodal"
    model: str = "nomic-ai/nomic-embed-multimodal-3b"
    dimensions: int = Field(1024, ge=1)


class RagVisualRetrievalConfig(ConfigModel):
    top_k: int = Field(5, ge=1, le=20)
    weight: float = Field(0.4, ge=0.0)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class RagVisualStorageConfig(ConfigModel):
    assets_dir: str = "data/rag/assets"


class RagAnswerImagesConfig(ConfigModel):
    enabled: bool = True
    max_images: int = Field(3, ge=0, le=3)
    max_bytes: int = Field(5_000_000, ge=1)
    max_long_edge_pixels: int = Field(1568, ge=64)


class RagVisualConfig(ConfigModel):
    enabled: bool = False
    required: bool = False
    crop_dpi: int = Field(180, ge=72, le=600)
    min_crop_pixels: int = Field(64, ge=1)
    max_crops_per_document: int = Field(50, ge=0)
    ocr: RagVisualProviderConfig = Field(
        default_factory=lambda: RagVisualProviderConfig(
            provider="paddleocr_vl",
            model="PaddlePaddle/PaddleOCR-VL",
        )
    )
    caption: RagVisualProviderConfig = Field(
        default_factory=lambda: RagVisualProviderConfig(
            provider="qwen2_5_vl",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
        )
    )
    embedding: RagVisualEmbeddingConfig = Field(
        default_factory=RagVisualEmbeddingConfig
    )
    retrieval: RagVisualRetrievalConfig = Field(
        default_factory=RagVisualRetrievalConfig
    )
    storage: RagVisualStorageConfig = Field(default_factory=RagVisualStorageConfig)
    answer_images: RagAnswerImagesConfig = Field(default_factory=RagAnswerImagesConfig)

    @model_validator(mode="after")
    def reject_enabled_legacy_embedding(self):
        if self.embedding.enabled:
            raise ValueError("rag.visual.embedding.enabled=true is no longer supported")
        return self


class RagConfig(ConfigModel):
    enabled: bool = False
    backend: str = "redis_stack"
    auto_retrieve: bool = True
    default_knowledge_base_ids: List[str] = Field(default_factory=list)
    local_job_cache_max_entries: int = Field(1000, ge=1)
    local_batch_cache_max_entries: int = Field(100, ge=1)
    job_ttl_seconds: int = Field(86400, ge=60)
    parser: RagParserConfig = Field(default_factory=RagParserConfig)
    layout_normalization: RagLayoutNormalizationConfig = Field(
        default_factory=RagLayoutNormalizationConfig
    )
    chunking: RagChunkingConfig = Field(default_factory=RagChunkingConfig)
    embedding: RagEmbeddingConfig = Field(default_factory=RagEmbeddingConfig)
    redis: RagRedisConfig = Field(default_factory=RagRedisConfig)
    indexing: RagIndexingConfig = Field(default_factory=RagIndexingConfig)
    retrieval: RagRetrievalConfig = Field(default_factory=RagRetrievalConfig)
    rerank: RagRerankConfig = Field(default_factory=RagRerankConfig)
    visual: RagVisualConfig = Field(default_factory=RagVisualConfig)
    intent_gate: RagIntentGateConfig = Field(default_factory=RagIntentGateConfig)
    ingestion_queue: RagIngestionQueueConfig = Field(
        default_factory=RagIngestionQueueConfig
    )
    storage: RagStorageConfig = Field(default_factory=RagStorageConfig)
    upload: RagUploadConfig = Field(default_factory=RagUploadConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"memory", "redis_stack"}:
            raise ValueError("rag backend must be one of: memory, redis_stack")
        return normalized


class ShutdownConfig(ConfigModel):
    grace_period_seconds: float = Field(75.0, gt=0)
    api_drain_timeout_seconds: float = Field(65.0, gt=0)
    service_stop_timeout_seconds: float = Field(15.0, gt=0)


class AgentRuntimeConfig(ConfigModel):
    enabled: bool = False
    feedback_enabled: bool = False
    # Agent runs one Slack user may have running at once, across threads.
    max_active_runs_per_user: int = Field(1, ge=1)
    base_url: str = "http://127.0.0.1:3001"
    token_env: str = "AGENT_RUNTIME_TOKEN"
    connect_timeout_seconds: float = Field(2.0, gt=0)
    request_timeout_seconds: float = Field(910.0, gt=0)
    control_timeout_seconds: float = Field(10.0, gt=0)


class PlatformConfig(ConfigModel):
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    flink: FlinkConfig = Field(default_factory=FlinkConfig)
    policy_cache: PolicyCacheConfig = Field(default_factory=PolicyCacheConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agent: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
