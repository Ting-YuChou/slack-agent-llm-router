from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid


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
