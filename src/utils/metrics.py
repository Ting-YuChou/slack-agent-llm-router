"""
Prometheus metrics definitions for LLM Router Platform
Comprehensive metrics collection for monitoring and observability
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info, Enum
from prometheus_client.metrics import MetricWrapperBase
import time
from typing import Dict, List, Optional, Any
from functools import wraps
import threading
from collections import defaultdict


# System-wide metrics
class SystemMetrics:
    """System-level metrics"""

    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "llm_router_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"],
        )

        self.request_duration = Histogram(
            "llm_router_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint", "method"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, float("inf")],
        )

        self.active_requests = Gauge(
            "llm_router_active_requests", "Number of active requests", ["endpoint"]
        )

        # Error metrics
        self.errors_total = Counter(
            "llm_router_errors_total",
            "Total number of errors",
            ["component", "error_type"],
        )

        # System resource metrics
        self.cpu_usage = Gauge("llm_router_cpu_usage_percent", "CPU usage percentage")

        self.memory_usage = Gauge(
            "llm_router_memory_usage_bytes", "Memory usage in bytes"
        )

        self.memory_usage_percent = Gauge(
            "llm_router_memory_usage_percent", "Memory usage percentage"
        )

        self.disk_usage = Gauge(
            "llm_router_disk_usage_percent", "Disk usage percentage", ["mount_point"]
        )

        # Connection metrics
        self.database_connections = Gauge(
            "llm_router_database_connections",
            "Number of database connections",
            ["database", "state"],
        )

        self.http_connections = Gauge(
            "llm_router_http_connections",
            "Number of HTTP connections",
            ["target", "state"],
        )

        # Application info
        self.info = Info("llm_router_info", "Application information")

        # Health status
        self.health_status = Enum(
            "llm_router_health_status",
            "Overall health status",
            states=["healthy", "degraded", "unhealthy"],
        )


class RouterMetrics:
    """Model router metrics"""

    def __init__(self):
        # Routing decisions
        self.routing_decisions = Counter(
            "llm_router_routing_decisions_total",
            "Total routing decisions",
            ["model", "query_type"],
        )

        self.routing_latency = Histogram(
            "llm_router_routing_latency_seconds",
            "Time taken for routing decisions",
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                float("inf"),
            ],
        )

        self.routing_confidence = Histogram(
            "llm_router_routing_confidence",
            "Confidence score for routing decisions",
            ["model", "query_type"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Model availability
        self.model_availability = Gauge(
            "llm_router_model_availability",
            "Model availability status (1=available, 0=unavailable)",
            ["model", "provider"],
        )

        # Query classification
        self.query_classifications = Counter(
            "llm_router_query_classifications_total",
            "Query classifications by type",
            ["query_type", "confidence_bucket"],
        )

        # Fallback usage
        self.fallback_usage = Counter(
            "llm_router_fallback_usage_total",
            "Fallback model usage",
            ["original_model", "fallback_model", "reason"],
        )


class InferenceMetrics:
    """Inference engine metrics"""

    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "llm_router_inference_requests_total",
            "Total inference requests",
            ["model", "provider"],
        )

        self.request_duration = Histogram(
            "llm_router_inference_duration_seconds",
            "Inference request duration",
            ["model", "provider"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")],
        )

        # Token metrics
        self.tokens_processed = Counter(
            "llm_router_tokens_processed_total",
            "Total tokens processed",
            ["model", "token_type"],  # input, output
        )

        self.tokens_per_second = Gauge(
            "llm_router_tokens_per_second",
            "Current tokens per second throughput",
            ["model", "provider"],
        )

        # Cost metrics
        self.cost_total = Counter(
            "llm_router_cost_total_usd", "Total cost in USD", ["model", "provider"]
        )

        self.cost_per_request = Histogram(
            "llm_router_cost_per_request_usd",
            "Cost per request in USD",
            ["model", "provider"],
            buckets=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, float("inf")],
        )

        # Quality metrics
        self.quality_scores = Histogram(
            "llm_router_response_quality_score",
            "Response quality scores",
            ["model"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.safety_scores = Histogram(
            "llm_router_response_safety_score",
            "Response safety scores",
            ["model"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Cache metrics
        self.cache_hits = Counter("llm_router_cache_hits_total", "Total cache hits")

        self.cache_misses = Counter(
            "llm_router_cache_misses_total", "Total cache misses"
        )

        self.cache_size = Gauge("llm_router_cache_size_bytes", "Cache size in bytes")

        # Context compression metrics
        self.context_compressions = Counter(
            "llm_router_context_compressions_total",
            "Total context compressions",
            ["method"],
        )

        self.compression_ratio = Histogram(
            "llm_router_compression_ratio",
            "Context compression ratios",
            ["method"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Error metrics
        self.errors_total = Counter(
            "llm_router_inference_errors_total",
            "Total inference errors",
            ["model", "error_type"],
        )

        # Model-specific metrics
        self.model_load_time = Gauge(
            "llm_router_model_load_time_seconds",
            "Model loading time",
            ["model", "provider"],
        )

        self.gpu_utilization = Gauge(
            "llm_router_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id", "model"],
        )

        self.gpu_memory_usage = Gauge(
            "llm_router_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["gpu_id", "model"],
        )


class PipelineMetrics:
    """Data pipeline metrics"""

    def __init__(self):
        # Kafka metrics
        self.messages_produced = Counter(
            "llm_router_kafka_messages_produced_total",
            "Total Kafka messages produced",
            ["topic"],
        )

        self.messages_consumed = Counter(
            "llm_router_kafka_messages_consumed_total",
            "Total Kafka messages consumed",
            ["topic"],
        )

        self.producer_errors = Counter(
            "llm_router_kafka_producer_errors_total", "Kafka producer errors"
        )

        self.consumer_errors = Counter(
            "llm_router_kafka_consumer_errors_total", "Kafka consumer errors"
        )

        self.consumer_lag = Gauge(
            "llm_router_kafka_consumer_lag",
            "Kafka consumer lag",
            ["topic", "partition"],
        )

        # ClickHouse metrics
        self.records_inserted = Counter(
            "llm_router_clickhouse_records_inserted_total",
            "Total records inserted to ClickHouse",
            ["table"],
        )

        self.insertion_errors = Counter(
            "llm_router_clickhouse_insertion_errors_total",
            "ClickHouse insertion errors",
        )

        self.query_duration = Histogram(
            "llm_router_clickhouse_query_duration_seconds",
            "ClickHouse query duration",
            ["query_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, float("inf")],
        )

        # Batch processing
        self.batch_size = Histogram(
            "llm_router_batch_size",
            "Batch processing sizes",
            ["processor"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")],
        )

        self.batch_processing_time = Histogram(
            "llm_router_batch_processing_time_seconds",
            "Batch processing time",
            ["processor"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, float("inf")],
        )


class SlackMetrics:
    """Slack bot metrics"""

    def __init__(self):
        # Message processing
        self.messages_received = Counter(
            "llm_router_slack_messages_received_total", "Total Slack messages received"
        )

        self.messages_processed = Counter(
            "llm_router_slack_messages_processed_total",
            "Total Slack messages processed",
            ["user_tier"],
        )

        self.commands_executed = Counter(
            "llm_router_slack_commands_executed_total",
            "Total Slack commands executed",
            ["command"],
        )

        # Response metrics
        self.response_time = Histogram(
            "llm_router_slack_response_time_seconds",
            "Slack response time",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")],
        )

        # User metrics
        self.active_users = Gauge(
            "llm_router_slack_active_users",
            "Number of active Slack users",
            ["time_window"],  # 1h, 24h, 7d
        )

        self.rate_limited_users = Counter(
            "llm_router_slack_rate_limited_users_total", "Total rate limited users"
        )

        # Error metrics
        self.errors = Counter(
            "llm_router_slack_errors_total", "Total Slack bot errors", ["error_type"]
        )

        # Conversation metrics
        self.conversation_length = Histogram(
            "llm_router_slack_conversation_length",
            "Conversation lengths in messages",
            buckets=[1, 5, 10, 25, 50, 100, float("inf")],
        )

        self.session_duration = Histogram(
            "llm_router_slack_session_duration_seconds",
            "Session durations",
            buckets=[60, 300, 900, 1800, 3600, 7200, float("inf")],
        )


class UserMetrics:
    """User-focused metrics"""

    def __init__(self):
        # Usage metrics
        self.queries_per_user = Counter(
            "llm_router_user_queries_total", "Total queries per user", ["user_tier"]
        )

        self.tokens_per_user = Counter(
            "llm_router_user_tokens_total",
            "Total tokens per user",
            ["user_tier", "token_type"],
        )

        self.cost_per_user = Counter(
            "llm_router_user_cost_total_usd", "Total cost per user", ["user_tier"]
        )

        # User distribution
        self.users_by_tier = Gauge(
            "llm_router_users_by_tier", "Number of users by tier", ["user_tier"]
        )

        # Satisfaction metrics
        self.user_satisfaction = Histogram(
            "llm_router_user_satisfaction_score",
            "User satisfaction scores",
            ["user_tier"],
            buckets=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


# Global metric instances
SYSTEM_METRICS = SystemMetrics()
ROUTER_METRICS = RouterMetrics()
INFERENCE_METRICS = InferenceMetrics()
PIPELINE_METRICS = PipelineMetrics()
SLACK_METRICS = SlackMetrics()
USER_METRICS = UserMetrics()


class MetricsCollector:
    """Collector for custom metrics and business logic"""

    def __init__(self):
        self._custom_metrics = {}
        self._lock = threading.Lock()

    def register_metric(self, name: str, metric: MetricWrapperBase):
        """Register a custom metric"""
        with self._lock:
            self._custom_metrics[name] = metric

    def get_metric(self, name: str) -> Optional[MetricWrapperBase]:
        """Get a registered metric"""
        return self._custom_metrics.get(name)

    def collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business-level metrics"""
        return {
            "total_requests_24h": self._get_counter_value_last_24h(
                SYSTEM_METRICS.requests_total
            ),
            "total_cost_24h": self._get_counter_value_last_24h(
                INFERENCE_METRICS.cost_total
            ),
            "average_response_time": self._get_histogram_average(
                SYSTEM_METRICS.request_duration
            ),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "error_rate": self._calculate_error_rate(),
            "model_distribution": self._get_model_usage_distribution(),
            "user_tier_distribution": self._get_user_tier_distribution(),
        }

    def _get_counter_value_last_24h(self, counter: Counter) -> float:
        """Get counter value for last 24 hours (simplified)"""
        # In a real implementation, this would query the time series data
        # For now, return current counter value
        return sum(counter._value.values())

    def _get_histogram_average(self, histogram: Histogram) -> float:
        """Get average value from histogram"""
        total_count = sum(histogram._count.values())
        total_sum = sum(histogram._sum.values())
        return total_sum / total_count if total_count > 0 else 0

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        hits = sum(INFERENCE_METRICS.cache_hits._value.values())
        misses = sum(INFERENCE_METRICS.cache_misses._value.values())
        total = hits + misses
        return hits / total if total > 0 else 0

    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        total_requests = sum(SYSTEM_METRICS.requests_total._value.values())
        total_errors = sum(SYSTEM_METRICS.errors_total._value.values())
        return total_errors / total_requests if total_requests > 0 else 0

    def _get_model_usage_distribution(self) -> Dict[str, float]:
        """Get model usage distribution"""
        model_counts = defaultdict(float)
        for labels, count in ROUTER_METRICS.routing_decisions._value.items():
            model = labels[0] if labels else "unknown"
            model_counts[model] += count

        total = sum(model_counts.values())
        if total == 0:
            return {}

        return {model: count / total for model, count in model_counts.items()}

    def _get_user_tier_distribution(self) -> Dict[str, float]:
        """Get user tier distribution"""
        tier_counts = {}
        for labels, count in USER_METRICS.users_by_tier._value.items():
            tier = labels[0] if labels else "unknown"
            tier_counts[tier] = count

        return tier_counts


# Performance tracking decorators
def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            SYSTEM_METRICS.active_requests.labels(endpoint=endpoint).inc()

            try:
                result = await func(*args, **kwargs)
                status = getattr(result, "status_code", 200)
                SYSTEM_METRICS.requests_total.labels(
                    endpoint=endpoint,
                    method="POST",  # Assume POST for async functions
                    status=str(status),
                ).inc()
                return result
            except Exception as e:
                SYSTEM_METRICS.requests_total.labels(
                    endpoint=endpoint, method="POST", status="500"
                ).inc()
                SYSTEM_METRICS.errors_total.labels(
                    component="api", error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                SYSTEM_METRICS.request_duration.labels(
                    endpoint=endpoint, method="POST"
                ).observe(duration)
                SYSTEM_METRICS.active_requests.labels(endpoint=endpoint).dec()

        return wrapper

    return decorator


def track_inference_metrics(model_name: str, provider: str):
    """Decorator to track inference metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Track successful inference
                INFERENCE_METRICS.requests_total.labels(
                    model=model_name, provider=provider
                ).inc()

                # Track tokens if available
                if hasattr(result, "token_count_input"):
                    INFERENCE_METRICS.tokens_processed.labels(
                        model=model_name, token_type="input"
                    ).inc(result.token_count_input)

                if hasattr(result, "token_count_output"):
                    INFERENCE_METRICS.tokens_processed.labels(
                        model=model_name, token_type="output"
                    ).inc(result.token_count_output)

                # Track cost
                if hasattr(result, "cost_usd"):
                    INFERENCE_METRICS.cost_total.labels(
                        model=model_name, provider=provider
                    ).inc(result.cost_usd)

                    INFERENCE_METRICS.cost_per_request.labels(
                        model=model_name, provider=provider
                    ).observe(result.cost_usd)

                # Track quality scores
                if (
                    hasattr(result, "quality_score")
                    and result.quality_score is not None
                ):
                    INFERENCE_METRICS.quality_scores.labels(model=model_name).observe(
                        result.quality_score
                    )

                if hasattr(result, "safety_score") and result.safety_score is not None:
                    INFERENCE_METRICS.safety_scores.labels(model=model_name).observe(
                        result.safety_score
                    )

                return result

            except Exception as e:
                INFERENCE_METRICS.errors_total.labels(
                    model=model_name, error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                INFERENCE_METRICS.request_duration.labels(
                    model=model_name, provider=provider
                ).observe(duration)

        return wrapper

    return decorator


class MetricsReporter:
    """Reporter for generating metrics reports"""

    def __init__(self):
        self.collector = MetricsCollector()

    def generate_system_report(self) -> Dict[str, Any]:
        """Generate system performance report"""
        return {
            "timestamp": time.time(),
            "system": {
                "cpu_usage": SYSTEM_METRICS.cpu_usage._value._value,
                "memory_usage_percent": SYSTEM_METRICS.memory_usage_percent._value._value,
                "active_requests": sum(SYSTEM_METRICS.active_requests._value.values()),
                "total_requests": sum(SYSTEM_METRICS.requests_total._value.values()),
                "total_errors": sum(SYSTEM_METRICS.errors_total._value.values()),
            },
            "inference": {
                "total_requests": sum(INFERENCE_METRICS.requests_total._value.values()),
                "total_tokens": sum(INFERENCE_METRICS.tokens_processed._value.values()),
                "total_cost": sum(INFERENCE_METRICS.cost_total._value.values()),
                "cache_hit_rate": self.collector._calculate_cache_hit_rate(),
                "error_rate": self.collector._calculate_error_rate(),
            },
            "business": self.collector.collect_business_metrics(),
        }

    def generate_model_report(self) -> Dict[str, Any]:
        """Generate model performance report"""
        models = {}

        # Collect per-model metrics
        for labels, value in ROUTER_METRICS.routing_decisions._value.items():
            model = labels[0] if labels else "unknown"
            if model not in models:
                models[model] = {"requests": 0, "tokens": 0, "cost": 0.0, "errors": 0}
            models[model]["requests"] += value

        # Add token and cost data
        for labels, value in INFERENCE_METRICS.tokens_processed._value.items():
            model = labels[0] if labels else "unknown"
            if model in models:
                models[model]["tokens"] += value

        for labels, value in INFERENCE_METRICS.cost_total._value.items():
            model = labels[0] if labels else "unknown"
            if model in models:
                models[model]["cost"] += value

        return {"timestamp": time.time(), "models": models}

    def generate_user_report(self) -> Dict[str, Any]:
        """Generate user analytics report"""
        return {
            "timestamp": time.time(),
            "users": {
                "by_tier": dict(USER_METRICS.users_by_tier._value),
                "total_queries": sum(USER_METRICS.queries_per_user._value.values()),
                "total_tokens": sum(USER_METRICS.tokens_per_user._value.values()),
                "total_cost": sum(USER_METRICS.cost_per_user._value.values()),
            },
            "slack": {
                "messages_processed": sum(
                    SLACK_METRICS.messages_processed._value.values()
                ),
                "active_users_1h": SLACK_METRICS.active_users.labels(
                    time_window="1h"
                )._value._value,
                "rate_limited_users": sum(
                    SLACK_METRICS.rate_limited_users._value.values()
                ),
            },
        }


class AlertThresholds:
    """Define alert thresholds for metrics"""

    # System thresholds
    CPU_USAGE_WARNING = 80.0
    CPU_USAGE_CRITICAL = 95.0
    MEMORY_USAGE_WARNING = 80.0
    MEMORY_USAGE_CRITICAL = 95.0
    DISK_USAGE_WARNING = 80.0
    DISK_USAGE_CRITICAL = 90.0

    # Performance thresholds
    RESPONSE_TIME_WARNING = 5.0  # seconds
    RESPONSE_TIME_CRITICAL = 10.0  # seconds
    ERROR_RATE_WARNING = 0.05  # 5%
    ERROR_RATE_CRITICAL = 0.10  # 10%

    # Business thresholds
    COST_SPIKE_THRESHOLD = 2.0  # 2x normal cost
    CACHE_HIT_RATE_WARNING = 0.5  # 50%

    @classmethod
    def check_system_thresholds(cls) -> List[Dict[str, Any]]:
        """Check system metrics against thresholds"""
        alerts = []

        # CPU usage
        cpu_usage = SYSTEM_METRICS.cpu_usage._value._value
        if cpu_usage > cls.CPU_USAGE_CRITICAL:
            alerts.append(
                {
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "threshold": cls.CPU_USAGE_CRITICAL,
                    "severity": "critical",
                }
            )
        elif cpu_usage > cls.CPU_USAGE_WARNING:
            alerts.append(
                {
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "threshold": cls.CPU_USAGE_WARNING,
                    "severity": "warning",
                }
            )

        # Memory usage
        memory_usage = SYSTEM_METRICS.memory_usage_percent._value._value
        if memory_usage > cls.MEMORY_USAGE_CRITICAL:
            alerts.append(
                {
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "threshold": cls.MEMORY_USAGE_CRITICAL,
                    "severity": "critical",
                }
            )
        elif memory_usage > cls.MEMORY_USAGE_WARNING:
            alerts.append(
                {
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "threshold": cls.MEMORY_USAGE_WARNING,
                    "severity": "warning",
                }
            )

        return alerts


# Utility functions
def reset_metrics():
    """Reset all metrics (useful for testing)"""
    for metric_class in [
        SYSTEM_METRICS,
        ROUTER_METRICS,
        INFERENCE_METRICS,
        PIPELINE_METRICS,
        SLACK_METRICS,
        USER_METRICS,
    ]:
        for attr_name in dir(metric_class):
            attr = getattr(metric_class, attr_name)
            if hasattr(attr, "clear"):
                attr.clear()


def get_metrics_summary() -> Dict[str, Any]:
    """Get summary of all metrics"""
    reporter = MetricsReporter()
    return {
        "system": reporter.generate_system_report(),
        "models": reporter.generate_model_report(),
        "users": reporter.generate_user_report(),
        "alerts": AlertThresholds.check_system_thresholds(),
    }


# Custom metric types for specific use cases
class SlidingWindowCounter:
    """Counter that maintains a sliding window of values"""

    def __init__(self, window_size: int = 3600):  # 1 hour default
        self.window_size = window_size
        self.values = []
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        """Increment counter"""
        current_time = time.time()
        with self._lock:
            self.values.append((current_time, amount))
            self._cleanup_old_values(current_time)

    def get_value(self) -> float:
        """Get current value in window"""
        current_time = time.time()
        with self._lock:
            self._cleanup_old_values(current_time)
            return sum(value for _, value in self.values)

    def _cleanup_old_values(self, current_time: float):
        """Remove values outside the window"""
        cutoff_time = current_time - self.window_size
        self.values = [
            (timestamp, value)
            for timestamp, value in self.values
            if timestamp > cutoff_time
        ]


class BusinessMetrics:
    """High-level business metrics"""

    def __init__(self):
        self.daily_active_users = SlidingWindowCounter(86400)  # 24 hours
        self.hourly_revenue = SlidingWindowCounter(3600)  # 1 hour
        self.customer_acquisition_cost = Gauge(
            "customer_acquisition_cost_usd", "Customer acquisition cost"
        )
        self.customer_lifetime_value = Gauge(
            "customer_lifetime_value_usd", "Customer lifetime value"
        )

    def track_user_activity(self, user_id: str):
        """Track user activity"""
        self.daily_active_users.inc()

    def track_revenue(self, amount: float):
        """Track revenue"""
        self.hourly_revenue.inc(amount)


# Global instances
BUSINESS_METRICS = BusinessMetrics()
METRICS_REPORTER = MetricsReporter()

# Export all metrics
__all__ = [
    "SYSTEM_METRICS",
    "ROUTER_METRICS",
    "INFERENCE_METRICS",
    "PIPELINE_METRICS",
    "SLACK_METRICS",
    "USER_METRICS",
    "BUSINESS_METRICS",
    "METRICS_REPORTER",
    "MetricsCollector",
    "MetricsReporter",
    "AlertThresholds",
    "SlidingWindowCounter",
    "BusinessMetrics",
    "track_request_metrics",
    "track_inference_metrics",
    "reset_metrics",
    "get_metrics_summary",
]
