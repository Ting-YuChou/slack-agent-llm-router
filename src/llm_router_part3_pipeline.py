"""
LLM Router Part 3: Kafka Pipeline - Stream Processing and Data Storage
Handles real-time data ingestion, processing, and storage to ClickHouse
"""

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import TopicPartition
import clickhouse_connect
from clickhouse_connect.driver import Client

from src.utils.logger import setup_logging
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import QueryRequest, InferenceResponse, SystemMetric, UserTier

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = "1.0"


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
    value: Optional[Any], *, default: Optional[datetime] = None
) -> datetime:
    """Parse ISO timestamps emitted by the platform into timezone-aware datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug(f"Failed to parse datetime value: {value}")

    return default or datetime.now(timezone.utc)


def _escape_clickhouse_literal(value: Any) -> str:
    """Escape a value for safe interpolation inside a ClickHouse string literal."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


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


def build_request_raw_event(query_request: QueryRequest) -> Dict[str, Any]:
    """Build the pre-inference event emitted by the API request path."""
    attachments = [
        {
            "id": attachment.id,
            "name": attachment.name,
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
        "query_text": query_request.query,
        "context": query_request.context,
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
        "metadata": query_request.metadata or {},
        "attachments": attachments,
        "attachments_count": len(attachments),
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
        "request_timestamp": _isoformat_utc(query_request.timestamp),
        "completion_timestamp": _isoformat_utc(inference_response.timestamp),
        "query_text": query_request.query,
        "query_type": _enum_value(query_type),
        "selected_model": inference_response.model_name,
        "provider": inference_response.provider,
        "status": "error" if inference_response.error else "success",
        "error_message": inference_response.error or "",
        "latency_ms": inference_response.latency_ms,
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
        "fallback_models": list(getattr(routing_decision, "fallback_models", []) or []),
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
            "query_text": self.query_text[:10000],
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
                if attempt >= self.retries:
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

        if self.wait_for_ack:
            send_and_wait = getattr(self.producer, "send_and_wait", None)
            if callable(send_and_wait):
                return await asyncio.wait_for(
                    send_and_wait(topic=topic, key=key, value=value),
                    timeout=self.send_timeout_seconds,
                )

        send_result = await self.producer.send(topic=topic, key=key, value=value)
        if self.wait_for_ack and inspect.isawaitable(send_result):
            return await asyncio.wait_for(
                send_result,
                timeout=self.send_timeout_seconds,
            )
        return send_result

    async def produce_request_raw(self, query_request: QueryRequest):
        """Produce a pre-inference API request event."""
        await self._send_message(
            "requests_raw",
            "requests.raw",
            key=query_request.user_id,
            value=build_request_raw_event(query_request),
        )

    async def produce_inference_completed(
        self,
        query_request: QueryRequest,
        inference_response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Produce a post-inference completion event."""
        await self._send_message(
            "inference_completed",
            "inference.completed",
            key=query_request.request_id,
            value=build_inference_completed_event(
                query_request,
                inference_response,
                routing_decision=routing_decision,
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

        await self._send_message(
            "queries",
            "llm-queries",
            key=query_log.user_id,
            value=query_log.to_dict(),
        )

    async def produce_response_log(self, response_data: Dict[str, Any]):
        """Produce response log message"""
        await self._send_message(
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

        await self._send_message(
            "metrics",
            "llm-metrics",
            key=f"{service}:{metric_name}",
            value=metric_entry.to_dict(),
        )

    async def produce_error(self, error_data: Dict[str, Any]):
        """Produce error message"""
        error_data["timestamp"] = datetime.now(timezone.utc)

        await self._send_message(
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

            return json.dumps(message, default=default_serializer).encode("utf-8")
        else:
            return json.dumps(message).encode("utf-8")

    async def shutdown(self):
        """Shutdown producer"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer shutdown complete")


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
        ORDER BY event_id
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
        ORDER BY event_id
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
        ORDER BY event_id
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
        ORDER BY event_id
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
        ORDER BY event_id
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        # Execute table creation
        tables = [
            ("query_logs", query_logs_sql),
            ("system_metrics", metrics_sql),
            ("model_performance", performance_sql),
            ("user_analytics", user_analytics_sql),
            ("alert_events", alert_events_sql),
            ("routing_policy_state_events", routing_policy_state_sql),
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
        ]

        for sql in alter_statements:
            try:
                await self._run_blocking(self.client.command, sql)
            except Exception as e:
                logger.debug(f"Skipping optional schema migration: {e}")

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

    async def get_query_analytics(
        self, user_id: str = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get query analytics for dashboard"""
        try:
            time_window_hours = max(int(hours), 0)
            where_clauses = [f"timestamp >= now() - INTERVAL {time_window_hours} HOUR"]
            if user_id:
                escaped_user_id = _escape_clickhouse_literal(user_id)
                where_clauses.append(f"user_id = '{escaped_user_id}'")
            where_sql = " AND ".join(where_clauses)

            overall_query = f"""
            SELECT
                count() as total_queries,
                ifNull(sum(token_count_input + token_count_output), 0) as total_tokens,
                ifNull(sum(cost_usd), 0.0) as total_cost,
                ifNull(avg(latency_ms), 0.0) as avg_latency,
                if(count() = 0, 0.0, countIf(status = 'success') * 100.0 / count()) as success_rate
            FROM {self.database}.query_logs
            WHERE {where_sql}
            """
            model_breakdown_query = f"""
            SELECT
                selected_model,
                count() as total_queries,
                ifNull(sum(cost_usd), 0.0) as total_cost
            FROM {self.database}.query_logs
            WHERE {where_sql}
            GROUP BY selected_model
            """
            query_type_breakdown_query = f"""
            SELECT
                query_type,
                count() as total_queries
            FROM {self.database}.query_logs
            WHERE {where_sql}
            GROUP BY query_type
            """

            overall_result = await self._run_blocking(self.client.query, overall_query)
            model_result = await self._run_blocking(
                self.client.query, model_breakdown_query
            )
            query_type_result = await self._run_blocking(
                self.client.query, query_type_breakdown_query
            )

            overall_row = (
                overall_result.result_rows[0]
                if overall_result.result_rows
                else [0, 0, 0.0, 0.0, 0.0]
            )
            analytics = {
                "total_queries": int(overall_row[0] or 0),
                "total_tokens": int(overall_row[1] or 0),
                "total_cost": float(overall_row[2] or 0.0),
                "avg_latency": float(overall_row[3] or 0.0),
                "success_rate": float(overall_row[4] or 0.0),
                "model_breakdown": {},
                "query_type_breakdown": {},
            }

            for row in model_result.result_rows:
                model = row[0]
                if model not in analytics["model_breakdown"]:
                    analytics["model_breakdown"][model] = {"queries": 0, "cost": 0.0}
                analytics["model_breakdown"][model]["queries"] += int(row[1] or 0)
                analytics["model_breakdown"][model]["cost"] += float(row[2] or 0.0)

            for row in query_type_result.result_rows:
                query_type = row[0]
                if query_type not in analytics["query_type_breakdown"]:
                    analytics["query_type_breakdown"][query_type] = 0
                analytics["query_type_breakdown"][query_type] += int(row[1] or 0)

            return analytics

        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}

    async def get_model_performance(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get model performance metrics"""
        try:
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
            FROM {self.database}.model_performance
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            GROUP BY model_name, provider
            ORDER BY requests DESC, model_name ASC, provider ASC
            """

            result = await self._run_blocking(self.client.query, query)

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

            fallback_query = f"""
            SELECT 
                selected_model,
                count() as requests,
                countIf(status = 'success') * 100.0 / count() as success_rate,
                avg(latency_ms) as avg_latency,
                sum(token_count_input + token_count_output) / nullIf(sum(latency_ms), 0) * 1000 as tokens_per_second,
                countIf(status != 'success') as errors,
                sum(cost_usd) as total_cost
            FROM {self.database}.query_logs 
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            GROUP BY selected_model
            ORDER BY requests DESC
            """

            result = await self._run_blocking(self.client.query, fallback_query)
            performance_data = []
            for row in result.result_rows:
                performance_data.append(
                    {
                        "model_name": row[0],
                        "provider": "",
                        "requests": row[1],
                        "success_rate": row[2],
                        "avg_latency_ms": row[3],
                        "tokens_per_second": row[4],
                        "error_count": row[5],
                        "total_cost": row[6],
                    }
                )

            return performance_data

        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return []

    async def get_routing_guardrails(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent routing guardrail events persisted through the alert stream."""
        try:
            query = f"""
            SELECT
                timestamp,
                alert_type,
                severity,
                description,
                model_name,
                provider,
                payload_json
            FROM {self.database}.alert_events
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
              AND source_event_type = 'routing.guardrails'
            ORDER BY timestamp DESC
            LIMIT 50
            """

            result = await self._run_blocking(self.client.query, query)
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
            return []

    async def get_routing_policy_state_events(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent persisted routing.policy_state events for audit and tuning."""
        try:
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
            FROM {self.database}.routing_policy_state_events
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            ORDER BY timestamp DESC
            LIMIT 50
            """

            result = await self._run_blocking(self.client.query, query)
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
            return []

    def shutdown(self):
        """Shutdown ClickHouse connection"""
        if self.client:
            self.client.close()
            logger.info("ClickHouse connection closed")


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
            "auto_offset_reset": consumer_section.get("auto_offset_reset", "latest"),
            "enable_auto_commit": consumer_section.get("enable_auto_commit", False),
            "max_poll_records": consumer_section.get("max_poll_records", 500),
        }
        self.supervisor_initial_backoff_seconds = float(
            consumer_section.get("supervisor_initial_backoff_seconds", 1.0)
        )
        self.supervisor_max_backoff_seconds = float(
            consumer_section.get("supervisor_max_backoff_seconds", 30.0)
        )
        self.dlq_enabled = bool(
            dlq_section.get("enabled", consumer_section.get("dlq_enabled", True))
        )
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
        self.batch_processors = {
            "queries": [],
            "metrics": [],
            "analytics_model_metrics_1m": [],
            "routing_guardrails": [],
            "routing_policy_state": [],
            "alerts": [],
        }
        self.observers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {
            "requests_enriched": [],
            "analytics_model_metrics_1m": [],
            "routing_guardrails": [],
            "routing_policy_state": [],
            "alerts": [],
        }
        self.batch_size = 100
        self.last_batch_time = time.time()
        self.running = False
        self.pending_commit_offsets: Dict[str, Dict[TopicPartition, int]] = {
            topic_key: {} for topic_key in self.batch_processors
        }
        self.awaiting_commit_offsets: Dict[str, Dict[TopicPartition, int]] = {
            topic_key: {} for topic_key in self.batch_processors
        }

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
                ]:
                    consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
                    await consumer.start()
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
        tasks = []

        for topic_key, consumer in self.consumers.items():
            tasks.append(
                asyncio.create_task(
                    self._supervise_consumer(topic_key, consumer),
                    name=f"kafka_consumer_{topic_key}",
                )
            )

        # Start batch processor
        tasks.append(asyncio.create_task(self._batch_processor()))

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

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
        consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
        await consumer.start()
        self.consumers[topic_key] = consumer
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
                        timestamp=datetime.fromisoformat(
                            data["timestamp"].replace("Z", "+00:00")
                        ),
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
                        **self._kafka_event_metadata(message),
                    )

                    # Add to batch
                    self.batch_processors["queries"].append(query_log)
                    self._track_commit_offset("queries", message)
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
                        timestamp=datetime.fromisoformat(
                            data["timestamp"].replace("Z", "+00:00")
                        ),
                        service=data["service"],
                        metric_name=data["metric_name"],
                        metric_value=data["metric_value"],
                        labels=data.get("labels", {}),
                        **self._kafka_event_metadata(message),
                    )

                    # Add to batch
                    self.batch_processors["metrics"].append(metric_entry)
                    self._track_commit_offset("metrics", message)
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

    def _dead_letter_topic(self, topic_key: str, message: Any) -> str:
        """Resolve the DLQ topic for a failed source message."""
        if topic_key in self.dead_letter_topics:
            return self.dead_letter_topics[topic_key]
        configured_key = f"{topic_key}_dlq"
        if configured_key in self.topics:
            return self.topics[configured_key]
        source_topic = (
            getattr(message, "topic", None)
            or self.topic_names_by_key.get(topic_key)
            or topic_key
        )
        return f"{source_topic}{self.dlq_topic_suffix}"

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
            "error_message": str(error)[:2000],
            "payload": _json_safe_payload(getattr(message, "value", None)),
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
            await self.flush_pending_batches()

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
        window_end_ms = int(data.get("window_end_ms", 0) or 0)
        event_time = _parse_datetime(
            data.get("emitted_at"),
            default=datetime.fromtimestamp(window_end_ms / 1000, tz=timezone.utc)
            if window_end_ms
            else datetime.now(timezone.utc),
        )

        return ModelPerformanceEntry(
            timestamp=event_time,
            model_name=str(data.get("model_name", "unknown")),
            provider=str(data.get("provider", "")),
            window_start_ms=int(data.get("window_start_ms", 0) or 0),
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
            **(kafka_metadata or {}),
        )

    def _build_alert_event_entry(
        self,
        data: Dict[str, Any],
        kafka_metadata: Optional[Dict[str, Any]] = None,
    ) -> AlertEventEntry:
        """Convert an alert event payload to a ClickHouse row."""
        event_type = str(data.get("event_type", "") or "")
        return AlertEventEntry(
            timestamp=_parse_datetime(data.get("emitted_at")),
            alert_type=str(
                data.get("alert_type") or data.get("trigger_type") or "unknown"
            ),
            severity=str(data.get("severity", "warning")),
            description=str(data.get("description", "")),
            anomaly_type=str(data.get("anomaly_type") or data.get("trigger_type", "")),
            source_event_type=str(data.get("original_event_type") or event_type),
            request_id=str(data.get("request_id", "") or ""),
            query_id=str(data.get("query_id", "") or ""),
            user_id=str(data.get("user_id", "") or ""),
            model_name=str(data.get("model_name", "") or ""),
            provider=str(data.get("provider", "") or ""),
            window_start_ms=int(data.get("window_start_ms", 0) or 0),
            window_end_ms=int(data.get("window_end_ms", 0) or 0),
            payload_json=json.dumps(data, default=str),
            **(kafka_metadata or {}),
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
            payload_json=json.dumps(data, default=str),
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
                    self.batch_processors["analytics_model_metrics_1m"].append(
                        metric_entry
                    )
                    self._track_commit_offset("analytics_model_metrics_1m", message)
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
                    self.batch_processors["alerts"].append(alert_entry)
                    self._track_commit_offset("alerts", message)
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
                    self.batch_processors["routing_guardrails"].append(guardrail_entry)
                    self._track_commit_offset("routing_guardrails", message)
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
                    self.batch_processors["routing_policy_state"].append(state_entry)
                    self._track_commit_offset("routing_policy_state", message)
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
                    await self.flush_pending_batches()

                # Update last batch time
                if batch_timeout_elapsed:
                    self.last_batch_time = current_time

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(10)

        await self.flush_pending_batches()

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
        if self.awaiting_commit_offsets.get(topic_key):
            await self._commit_offsets(
                topic_key, self.awaiting_commit_offsets[topic_key]
            )
            self.awaiting_commit_offsets[topic_key].clear()

        batch = list(self.batch_processors.get(topic_key, []))
        if not batch:
            return

        await insert_method(batch)
        self.batch_processors[topic_key].clear()
        self.awaiting_commit_offsets[topic_key] = dict(
            self.pending_commit_offsets.get(topic_key, {})
        )
        self.pending_commit_offsets[topic_key].clear()

        if self.awaiting_commit_offsets[topic_key]:
            await self._commit_offsets(
                topic_key, self.awaiting_commit_offsets[topic_key]
            )
            self.awaiting_commit_offsets[topic_key].clear()

    async def flush_pending_batches(self):
        """Flush any pending Kafka batches to ClickHouse immediately"""
        await self._flush_batch_topic(
            "queries", self.clickhouse.batch_insert_query_logs
        )
        await self._flush_batch_topic("metrics", self.clickhouse.batch_insert_metrics)
        await self._flush_batch_topic(
            "analytics_model_metrics_1m",
            self.clickhouse.batch_insert_model_performance,
        )
        await self._flush_batch_topic(
            "routing_guardrails", self.clickhouse.batch_insert_alert_events
        )
        await self._flush_batch_topic(
            "routing_policy_state",
            self.clickhouse.batch_insert_routing_policy_state,
        )
        await self._flush_batch_topic(
            "alerts", self.clickhouse.batch_insert_alert_events
        )

        self.last_batch_time = time.time()

    def _deserialize_message(self, message: bytes) -> Dict[str, Any]:
        """Deserialize message from JSON bytes"""
        return json.loads(message.decode("utf-8"))

    def _serialize_message(self, message: Any) -> bytes:
        """Serialize consumer-side DLQ payloads as JSON bytes."""
        return json.dumps(message, default=str).encode("utf-8")

    async def shutdown(self):
        """Shutdown all consumers"""
        self.running = False
        await self.flush_pending_batches()
        for consumer in self.consumers.values():
            await consumer.stop()
        if self.dlq_producer:
            await self.dlq_producer.stop()
        logger.info("Kafka consumers shutdown complete")


class KafkaIngestionPipeline:
    """Main Kafka ingestion pipeline coordinator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clickhouse_manager = ClickHouseManager(config.get("clickhouse", {}))
        self.producer_manager = KafkaProducerManager(config)
        self.consumer_manager = None  # Will be initialized after ClickHouse
        self._stream_handlers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {
            "requests_enriched": [],
            "analytics_model_metrics_1m": [],
            "routing_guardrails": [],
            "routing_policy_state": [],
            "alerts": [],
        }
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

        # Start consuming messages
        await self.consumer_manager.start_consuming()

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
        self, user_id: str = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get analytics from ClickHouse"""
        return await self.clickhouse_manager.get_query_analytics(user_id, hours)

    async def get_query_analytics(
        self, user_id: str = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Backward-compatible alias for dashboard analytics lookups."""
        return await self.get_analytics(user_id, hours)

    async def get_model_performance(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get model performance from ClickHouse"""
        return await self.clickhouse_manager.get_model_performance(hours)

    async def get_routing_guardrails(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get persisted routing guardrails from ClickHouse."""
        return await self.clickhouse_manager.get_routing_guardrails(hours)

    async def get_routing_policy_state_events(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get persisted routing.policy_state events from ClickHouse."""
        return await self.clickhouse_manager.get_routing_policy_state_events(hours)

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
            "clickhouse_healthy": self.clickhouse_manager.client is not None,
            "total_consumers": len(self.consumer_manager.consumers)
            if self.consumer_manager
            else 0,
        }

    def is_healthy(self) -> bool:
        """Return whether the pipeline can produce, consume, and persist events."""
        health = self.get_health_status()
        return bool(
            health["producer_healthy"]
            and health["consumer_healthy"]
            and health["clickhouse_healthy"]
        )

    async def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down Kafka ingestion pipeline...")
        self.running = False

        if self.consumer_manager:
            await self.consumer_manager.shutdown()

        await self.producer_manager.shutdown()
        self.clickhouse_manager.shutdown()

        logger.info("Kafka ingestion pipeline shutdown complete")
