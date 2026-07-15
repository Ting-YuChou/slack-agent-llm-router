from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: AttachmentType
    size_bytes: int
    mime_type: str
    url: Optional[str] = None
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


class QueryRequest(BaseModel):
    # Required fields
    query: str = Field(..., min_length=1, max_length=50000)
    user_id: str = Field(..., min_length=1)

    # Core fields with proper types
    user_tier: UserTier = UserTier.FREE
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    # Optional context and settings
    context: Optional[str] = Field(None, max_length=100000)
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    priority: int = Field(1, ge=1, le=5)

    # File handling
    attachments: List[Attachment] = Field(default_factory=list)

    # Tool handling
    tool_policy: ToolPolicy = ToolPolicy.AUTO
    allowed_tools: List[str] = Field(default_factory=list)
    web_search_options: Optional[WebSearchOptions] = None

    # RAG handling
    rag_policy: RagPolicy = RagPolicy.AUTO
    knowledge_base_ids: List[str] = Field(default_factory=list)
    rag_options: Optional["RagOptions"] = None

    # Additional metadata
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

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

    # Performance metrics
    latency_ms: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)

    # Cost tracking
    cost_usd: float = Field(ge=0.0)

    # System flags
    cached: bool = False
    compressed_context: bool = False

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

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.now)

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
    timestamp: datetime = Field(default_factory=datetime.now)
    fallback_models: List[str] = Field(default_factory=list)

    # Metadata
    routing_strategy: str = "intelligent"
    user_tier: UserTier = UserTier.FREE
    route_to_fast_lane: bool = False
    actual_fast_lane_hit: bool = False
    policy_source: str = "none"
    hint_reason: Optional[str] = None


class ModelSelection(BaseModel):
    model_name: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)


class ModelConfig(BaseModel):
    # Core model info
    name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model_path: Optional[str] = None

    # Capacity settings
    max_tokens: int = Field(ge=1)
    cost_per_token: float = Field(ge=0.0)
    priority: int = Field(ge=1)

    # Capabilities
    capabilities: List[str] = Field(..., min_length=1)

    # Resource requirements
    gpu_memory_gb: Optional[int] = Field(None, ge=1)

    # Authentication
    api_key_env: Optional[str] = None

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
        }
        for cap in v:
            if cap not in valid_capabilities:
                raise ValueError(f"Invalid capability: {cap}")
        return v


class SystemMetric(BaseModel):
    name: str = Field(..., min_length=1)
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
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
    start_time: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    query_count: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    total_cost: float = Field(0.0, ge=0.0)


class ErrorResponse(BaseModel):
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    retry_after: Optional[int] = None  # seconds


# Health check models
class HealthStatus(BaseModel):
    service: str
    status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


class ComponentHealth(BaseModel):
    overall_status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    components: Dict[str, HealthStatus]
    timestamp: datetime = Field(default_factory=datetime.now)


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


class ApiRateLimitQueueConfig(ConfigModel):
    enabled: bool = True
    max_depth: int = Field(1000, ge=0)
    timeout_ms: int = Field(250, ge=0)
    poll_interval_ms: int = Field(25, ge=1)


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
    cors_origins: List[str] = Field(default_factory=list)
    rate_limiting: ApiRateLimitingConfig = Field(default_factory=ApiRateLimitingConfig)


class LoggingConfig(ConfigModel):
    level: str = "INFO"
    file: str = "logs/llm_router.log"
    max_bytes: int = Field(10_485_760, ge=1)
    backup_count: int = Field(5, ge=0)
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class RouterConfig(ConfigModel):
    default_model: str = "mistral-7b"
    routing_strategy: str = "intelligent"
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
            QueryType.QUESTION_ANSWERING.value,
        ]
    )
    max_input_tokens: int = Field(2048, ge=1)
    max_output_tokens: int = Field(1024, ge=1)
    disallow_attachments: bool = True
    disallow_complex_reasoning: bool = True
    disallow_required_tools: bool = True
    disallow_required_rag: bool = True


class ProviderEndpointConfig(ConfigModel):
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


class CompressionConfig(ConfigModel):
    enabled: bool = False
    max_context_tokens: int = Field(100_000, ge=1)
    compression_ratio: float = Field(0.3, gt=0.0, le=1.0)
    method: str = "semantic_graph"


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
    half_open_max_requests: int = Field(1, ge=1)
    state_ttl_seconds: int = Field(3600, ge=1)


class ProviderSchedulerConfig(ConfigModel):
    enabled: bool = True
    queue_enabled: bool = False
    wait_timeout_ms: int = Field(250, ge=0)
    poll_interval_ms: int = Field(25, ge=1)
    failure_mode: str = "closed"
    allow_fallback_on_provider_rejection: bool = True
    key_prefix: str = "provider_scheduler"
    request_deadline_seconds: float = Field(60.0, gt=0)
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


class InferenceConfig(ConfigModel):
    vllm: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    openai: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    anthropic: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    single_flight: SingleFlightConfig = Field(default_factory=SingleFlightConfig)
    scheduler: ProviderSchedulerConfig = Field(default_factory=ProviderSchedulerConfig)

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
    per_user_rate_limit: int = Field(20, ge=1)
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
    tables: Dict[str, Any] = Field(default_factory=dict)
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


class SlackMemorySearchConfig(ConfigModel):
    max_results: int = Field(5, ge=1)
    max_context_chars: int = Field(2000, ge=1)
    max_item_chars: int = Field(500, ge=1)
    keyword_weight: float = Field(0.45, ge=0.0)
    vector_weight: float = Field(0.45, ge=0.0)
    recency_weight: float = Field(0.05, ge=0.0)
    importance_weight: float = Field(0.05, ge=0.0)


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
    signing_secret: Optional[str] = None
    signing_secret_env: Optional[str] = None
    channels: List[str] = Field(default_factory=list)
    response_settings: SlackResponseSettingsConfig = Field(
        default_factory=SlackResponseSettingsConfig
    )
    rate_limiting: SlackRateLimitingConfig = Field(
        default_factory=SlackRateLimitingConfig
    )
    context: SlackContextConfig = Field(default_factory=SlackContextConfig)
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
    request_ttl_seconds: int = Field(300, ge=1)
    user_ttl_seconds: int = Field(900, ge=1)
    local_cache_ttl_seconds: int = Field(5, ge=0)
    redis: Dict[str, Any] = Field(default_factory=dict)
    consumer: Dict[str, Any] = Field(default_factory=dict)


class ApiKeysSecurityConfig(ConfigModel):
    enabled: bool = False
    header_name: str = "X-API-Key"
    env_var: str = "LLM_ROUTER_API_KEYS"


class JwtSecurityConfig(ConfigModel):
    enabled: bool = False
    secret_key_env: str = "JWT_SECRET_KEY"
    algorithm: str = "HS256"
    expiration_hours: int = Field(24, ge=1)


class CorsSecurityConfig(ConfigModel):
    enabled: bool = False
    allow_origins: List[str] = Field(default_factory=list)
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = Field(default_factory=list)
    allow_credentials: bool = False


class SecurityConfig(ConfigModel):
    api_keys: ApiKeysSecurityConfig = Field(default_factory=ApiKeysSecurityConfig)
    jwt: JwtSecurityConfig = Field(default_factory=JwtSecurityConfig)
    cors: CorsSecurityConfig = Field(default_factory=CorsSecurityConfig)


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
    context_compression: bool = True
    semantic_caching: bool = True
    batch_processing: bool = True
    multi_modal: bool = False
    function_calling: bool = True
    streaming_responses: bool = True


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


class RagEmbeddingConfig(ConfigModel):
    provider: str = "local_http"
    model: str = "BAAI/bge-m3"
    dimensions: int = Field(1024, ge=1)
    timeout: int = Field(30, ge=1)
    api_key: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    url: Optional[str] = "http://127.0.0.1:8002/v1"


class RagRedisConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(6380, ge=1, le=65535)
    db: int = Field(0, ge=0)
    url: Optional[str] = None
    password_env: Optional[str] = None
    key_prefix: str = "rag"


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


class RagStorageConfig(ConfigModel):
    staging_dir: str = "data/rag/uploads"
    cleanup_completed_files: bool = False
    completed_file_ttl_seconds: int = Field(86400, ge=0)


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


class RagConfig(ConfigModel):
    enabled: bool = False
    backend: str = "redis_stack"
    auto_retrieve: bool = True
    default_knowledge_base_ids: List[str] = Field(default_factory=list)
    parser: RagParserConfig = Field(default_factory=RagParserConfig)
    chunking: RagChunkingConfig = Field(default_factory=RagChunkingConfig)
    embedding: RagEmbeddingConfig = Field(default_factory=RagEmbeddingConfig)
    redis: RagRedisConfig = Field(default_factory=RagRedisConfig)
    retrieval: RagRetrievalConfig = Field(default_factory=RagRetrievalConfig)
    rerank: RagRerankConfig = Field(default_factory=RagRerankConfig)
    visual: RagVisualConfig = Field(default_factory=RagVisualConfig)
    intent_gate: RagIntentGateConfig = Field(default_factory=RagIntentGateConfig)
    ingestion_queue: RagIngestionQueueConfig = Field(
        default_factory=RagIngestionQueueConfig
    )
    storage: RagStorageConfig = Field(default_factory=RagStorageConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"memory", "redis_stack"}:
            raise ValueError("rag backend must be one of: memory, redis_stack")
        return normalized


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
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
