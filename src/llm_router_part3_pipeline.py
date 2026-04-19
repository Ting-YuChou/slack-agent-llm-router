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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
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
        }


@dataclass
class MetricEntry:
    """Structure for system metrics"""

    timestamp: datetime
    service: str
    metric_name: str
    metric_value: float
    labels: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            "timestamp": self.timestamp,
            "service": self.service,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "labels": self.labels,  # ClickHouse Map type
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
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
        }


class KafkaProducerManager:
    """Manages Kafka message production"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.producer = None
        self.topics = config.get("topics", {})

        # Producer configuration
        self.producer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "value_serializer": self._serialize_message,
            "key_serializer": lambda x: x.encode("utf-8") if x else None,
            "acks": config.get("producer", {}).get("acks", "all"),
            "compression_type": config.get("producer", {}).get(
                "compression_type", "gzip"
            ),
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
        return self.producer is not None

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
    ):
        """Send a JSON-serializable message to Kafka and update metrics."""
        try:
            await self.producer.send(
                topic=self._topic_name(topic_key, default_topic),
                key=key,
                value=value,
            )
            PIPELINE_METRICS.messages_produced.labels(topic=topic_key).inc()
        except Exception as e:
            logger.error(f"Failed to produce {topic_key} message: {e}")
            PIPELINE_METRICS.producer_errors.inc()

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
        routing_decision: Any,
    ):
        """Produce query log message"""
        query_log = QueryLogEntry(
            query_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            user_id=query_request.user_id,
            user_tier=query_request.user_tier,
            query_text=query_request.query,
            query_type=routing_decision.query_type.value
            if hasattr(routing_decision.query_type, "value")
            else str(routing_decision.query_type),
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
            cached_response Boolean
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, user_id, query_id)
        TTL toDateTime(timestamp) + INTERVAL 90 DAY
        """.format(
            database=self.database
        )

        # System metrics table
        metrics_sql = """
        CREATE TABLE IF NOT EXISTS {database}.system_metrics (
            timestamp DateTime64(3),
            service String,
            metric_name String,
            metric_value Float64,
            labels Map(String, String)
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (service, metric_name, timestamp)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        # Model performance table
        performance_sql = """
        CREATE TABLE IF NOT EXISTS {database}.model_performance (
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
            memory_usage_gb Float32
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (model_name, timestamp)
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
            payload_json String
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (severity, alert_type, timestamp)
        TTL toDateTime(timestamp) + INTERVAL 30 DAY
        """.format(
            database=self.database
        )

        routing_policy_state_sql = """
        CREATE TABLE IF NOT EXISTS {database}.routing_policy_state_events (
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
            payload_json String
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (scope_type, scope_key, timestamp)
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

        alter_statements = [
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS provider String DEFAULT ''",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_start_ms UInt64 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_end_ms UInt64 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS window_size_seconds UInt32 DEFAULT 60",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS success_count UInt32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS total_tokens UInt32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS queries_per_second Float32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS cache_hit_rate Float32 DEFAULT 0",
            f"ALTER TABLE {self.database}.model_performance ADD COLUMN IF NOT EXISTS cached_count UInt32 DEFAULT 0",
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
            base_query = f"""
            SELECT 
                count() as total_queries,
                sum(token_count_input + token_count_output) as total_tokens,
                sum(cost_usd) as total_cost,
                avg(latency_ms) as avg_latency,
                countIf(status = 'success') * 100.0 / count() as success_rate,
                selected_model,
                query_type
            FROM {self.database}.query_logs 
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            """

            if user_id:
                base_query += f" AND user_id = '{user_id}'"

            # Overall stats
            overall_query = base_query + " GROUP BY selected_model, query_type"
            result = await self._run_blocking(self.client.query, overall_query)

            # Process results
            analytics = {
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0,
                "model_breakdown": {},
                "query_type_breakdown": {},
            }

            for row in result.result_rows:
                analytics["total_queries"] += row[0]
                analytics["total_tokens"] += row[1]
                analytics["total_cost"] += row[2]

                model = row[5]
                query_type = row[6]

                if model not in analytics["model_breakdown"]:
                    analytics["model_breakdown"][model] = {"queries": 0, "cost": 0.0}
                analytics["model_breakdown"][model]["queries"] += row[0]
                analytics["model_breakdown"][model]["cost"] += row[2]

                if query_type not in analytics["query_type_breakdown"]:
                    analytics["query_type_breakdown"][query_type] = 0
                analytics["query_type_breakdown"][query_type] += row[0]

            # Calculate averages
            if analytics["total_queries"] > 0:
                analytics["avg_latency"] = sum(
                    row[3] for row in result.result_rows
                ) / len(result.result_rows)
                analytics["success_rate"] = sum(
                    row[4] for row in result.result_rows
                ) / len(result.result_rows)

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
                any(provider) as provider,
                sum(requests_count) as requests,
                sum(success_count) * 100.0 / nullIf(sum(requests_count), 0) as success_rate,
                sum(avg_latency_ms * requests_count) / nullIf(sum(requests_count), 0) as avg_latency,
                sum(avg_tokens_per_second * requests_count) / nullIf(sum(requests_count), 0) as tokens_per_second,
                sum(error_count) as errors,
                sum(total_cost_usd) as total_cost
            FROM {self.database}.model_performance
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            GROUP BY model_name
            ORDER BY requests DESC
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

        # Consumer configuration
        self.consumer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": config.get("consumer", {}).get(
                "group_id", "llm-router-consumer"
            ),
            "auto_offset_reset": config.get("consumer", {}).get(
                "auto_offset_reset", "latest"
            ),
            "enable_auto_commit": config.get("consumer", {}).get(
                "enable_auto_commit", False
            ),
            "max_poll_records": config.get("consumer", {}).get("max_poll_records", 500),
            "value_deserializer": self._deserialize_message,
        }

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
                    logger.info(f"Consumer for topic {topic_name} initialized")

            logger.info(f"Kafka consumers initialized for {len(self.consumers)} topics")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumers: {e}")
            raise

    async def start_consuming(self):
        """Start consuming messages from all topics"""
        self.running = True
        tasks = []

        # Start consumer tasks
        for topic_key, consumer in self.consumers.items():
            if topic_key == "queries":
                tasks.append(asyncio.create_task(self._consume_queries(consumer)))
            elif topic_key == "responses":
                tasks.append(asyncio.create_task(self._consume_responses(consumer)))
            elif topic_key == "metrics":
                tasks.append(asyncio.create_task(self._consume_metrics(consumer)))
            elif topic_key == "errors":
                tasks.append(asyncio.create_task(self._consume_errors(consumer)))
            elif topic_key == "requests_enriched":
                tasks.append(
                    asyncio.create_task(self._consume_requests_enriched(consumer))
                )
            elif topic_key == "analytics_model_metrics_1m":
                tasks.append(
                    asyncio.create_task(self._consume_analytics_model_metrics(consumer))
                )
            elif topic_key == "routing_guardrails":
                tasks.append(
                    asyncio.create_task(self._consume_routing_guardrails(consumer))
                )
            elif topic_key == "routing_policy_state":
                tasks.append(
                    asyncio.create_task(self._consume_routing_policy_state(consumer))
                )
            elif topic_key == "alerts":
                tasks.append(asyncio.create_task(self._consume_alerts(consumer)))

        # Start batch processor
        tasks.append(asyncio.create_task(self._batch_processor()))

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_queries(self, consumer: AIOKafkaConsumer):
        """Consume query log messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
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
                    )

                    # Add to batch
                    self.batch_processors["queries"].append(query_log)
                    self._track_commit_offset("queries", message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="queries").inc()

                except Exception as e:
                    logger.error(f"Failed to process query message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()

        except Exception as e:
            logger.error(f"Query consumer error: {e}")

    async def _consume_responses(self, consumer: AIOKafkaConsumer):
        """Consume response log messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    # Process response data (for future analytics)
                    logger.debug(f"Processed response: {data.get('query_id')}")
                    await self._commit_processed_message("responses", message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="responses").inc()

                except Exception as e:
                    logger.error(f"Failed to process response message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()

        except Exception as e:
            logger.error(f"Response consumer error: {e}")

    async def _consume_metrics(self, consumer: AIOKafkaConsumer):
        """Consume system metrics messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    metric_entry = MetricEntry(
                        timestamp=datetime.fromisoformat(
                            data["timestamp"].replace("Z", "+00:00")
                        ),
                        service=data["service"],
                        metric_name=data["metric_name"],
                        metric_value=data["metric_value"],
                        labels=data.get("labels", {}),
                    )

                    # Add to batch
                    self.batch_processors["metrics"].append(metric_entry)
                    self._track_commit_offset("metrics", message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="metrics").inc()

                except Exception as e:
                    logger.error(f"Failed to process metric message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()

        except Exception as e:
            logger.error(f"Metrics consumer error: {e}")

    async def _consume_errors(self, consumer: AIOKafkaConsumer):
        """Consume error messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    logger.error(f"System error logged: {data}")
                    await self._commit_processed_message("errors", message)

                    PIPELINE_METRICS.messages_consumed.labels(topic="errors").inc()

                except Exception as e:
                    logger.error(f"Failed to process error message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()

        except Exception as e:
            logger.error(f"Error consumer error: {e}")

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

    def _build_model_performance_entry(
        self, data: Dict[str, Any]
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
        )

    def _build_alert_event_entry(self, data: Dict[str, Any]) -> AlertEventEntry:
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
        )

    def _build_routing_policy_state_entry(
        self, data: Dict[str, Any]
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
        )

    async def _consume_requests_enriched(self, consumer: AIOKafkaConsumer):
        """Consume request-side Flink enrichment events for observers."""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    await self._notify_observers("requests_enriched", data)
                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="requests_enriched"
                    ).inc()
                except Exception as e:
                    logger.error(f"Failed to process requests.enriched event: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
        except Exception as e:
            logger.error(f"requests.enriched consumer error: {e}")

    async def _consume_analytics_model_metrics(self, consumer: AIOKafkaConsumer):
        """Consume Flink model metrics window events."""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    metric_entry = self._build_model_performance_entry(data)
                    self.batch_processors["analytics_model_metrics_1m"].append(
                        metric_entry
                    )
                    self._track_commit_offset("analytics_model_metrics_1m", message)
                    await self._notify_observers("analytics_model_metrics_1m", data)

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="analytics_model_metrics_1m"
                    ).inc()
                except Exception as e:
                    logger.error(f"Failed to process model metrics event: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
        except Exception as e:
            logger.error(f"Analytics model metrics consumer error: {e}")

    async def _consume_alerts(self, consumer: AIOKafkaConsumer):
        """Consume Flink and platform alert events."""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    alert_entry = self._build_alert_event_entry(data)
                    self.batch_processors["alerts"].append(alert_entry)
                    self._track_commit_offset("alerts", message)
                    await self._notify_observers("alerts", data)

                    PIPELINE_METRICS.messages_consumed.labels(topic="alerts").inc()
                except Exception as e:
                    logger.error(f"Failed to process alert event: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
        except Exception as e:
            logger.error(f"Alert consumer error: {e}")

    async def _consume_routing_guardrails(self, consumer: AIOKafkaConsumer):
        """Consume routing guardrail events for persistence and observers."""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    guardrail_entry = self._build_alert_event_entry(data)
                    self.batch_processors["routing_guardrails"].append(guardrail_entry)
                    self._track_commit_offset("routing_guardrails", message)
                    await self._notify_observers("routing_guardrails", data)

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="routing_guardrails"
                    ).inc()
                except Exception as e:
                    logger.error(f"Failed to process routing guardrail event: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
        except Exception as e:
            logger.error(f"routing.guardrails consumer error: {e}")

    async def _consume_routing_policy_state(self, consumer: AIOKafkaConsumer):
        """Consume routing.policy_state events for persistence and observers."""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    state_entry = self._build_routing_policy_state_entry(data)
                    self.batch_processors["routing_policy_state"].append(state_entry)
                    self._track_commit_offset("routing_policy_state", message)
                    await self._notify_observers("routing_policy_state", data)

                    PIPELINE_METRICS.messages_consumed.labels(
                        topic="routing_policy_state"
                    ).inc()
                except Exception as e:
                    logger.error(f"Failed to process routing policy state event: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
        except Exception as e:
            logger.error(f"routing.policy_state consumer error: {e}")

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

    async def shutdown(self):
        """Shutdown all consumers"""
        self.running = False
        await self.flush_pending_batches()
        for consumer in self.consumers.values():
            await consumer.stop()
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
        return {
            "pipeline_running": self.running,
            "producer_healthy": self.producer_manager.producer is not None,
            "consumer_healthy": bool(
                self.consumer_manager and self.consumer_manager.consumers
            ),
            "clickhouse_healthy": self.clickhouse_manager.client is not None,
            "total_consumers": len(self.consumer_manager.consumers)
            if self.consumer_manager
            else 0,
        }

    async def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down Kafka ingestion pipeline...")
        self.running = False

        if self.consumer_manager:
            await self.consumer_manager.shutdown()

        await self.producer_manager.shutdown()
        self.clickhouse_manager.shutdown()

        logger.info("Kafka ingestion pipeline shutdown complete")
