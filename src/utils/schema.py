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

    # Performance metrics
    latency_ms: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)

    # Cost tracking
    cost_usd: float = Field(ge=0.0)

    # System flags
    cached: bool = False
    compressed_context: bool = False

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


class ApiRateLimitingConfig(ConfigModel):
    enabled: bool = False
    requests_per_minute: int = Field(1000, ge=1)
    burst_size: int = Field(100, ge=1)


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


class ProviderEndpointConfig(ConfigModel):
    base_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    timeout: Optional[int] = Field(None, ge=1)
    max_retries: Optional[int] = Field(None, ge=0)


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


class InferenceConfig(ConfigModel):
    vllm: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    openai: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    anthropic: ProviderEndpointConfig = Field(default_factory=ProviderEndpointConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)


class KafkaProducerConfig(ConfigModel):
    acks: str = "all"
    retries: int = Field(3, ge=0)
    batch_size: int = Field(16_384, ge=1)
    linger_ms: int = Field(5, ge=0)
    compression_type: str = "gzip"


class KafkaConsumerConfig(ConfigModel):
    group_id: str = "llm-router-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = Field(500, ge=1)


class KafkaConfig(ConfigModel):
    bootstrap_servers: List[str] = Field(default_factory=list)
    topics: Dict[str, str] = Field(default_factory=dict)
    producer: KafkaProducerConfig = Field(default_factory=KafkaProducerConfig)
    consumer: KafkaConsumerConfig = Field(default_factory=KafkaConsumerConfig)
    enabled: bool = False


class ClickHouseConfig(ConfigModel):
    host: str = "localhost"
    port: int = Field(8123, ge=1, le=65535)
    database: str = "llm_router"
    username: str = "default"
    password: str = ""
    tables: Dict[str, Any] = Field(default_factory=dict)
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
    state_backend: str = "memory"
    state_file: str = "data/slack_state.json"
    state_key_prefix: str = "slack_state"
    redis: Dict[str, Any] = Field(default_factory=dict)

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
    connection_pools: ConnectionPoolsConfig = Field(default_factory=ConnectionPoolsConfig)
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
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
