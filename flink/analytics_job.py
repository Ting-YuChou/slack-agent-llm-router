"""
Apache Flink analytics job for post-inference completion events.

This job consumes `inference.completed` events, emits 1-minute model metrics
windows, and publishes anomaly alerts based on recent aggregate history.
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.functions import FilterFunction, KeyedProcessFunction, MapFunction
from pyflink.datastream.state import ListStateDescriptor, ValueStateDescriptor

from flink.logic import (
    ALERT_EVENT,
    INFERENCE_COMPLETED_EVENT,
    MODEL_METRICS_1M_EVENT,
    ROUTING_GUARDRAIL_EVENT,
    build_alert_event,
    build_model_metrics_window_event,
    build_routing_policy_state_event,
    completion_event_timestamp_ms,
    detect_metric_anomalies,
    detect_model_routing_guardrails,
    detect_provider_routing_guardrails,
    validate_inference_completed_event,
)

logger = logging.getLogger(__name__)


class InferenceCompletedMapper(MapFunction):
    """Map inference.completed messages into validated dict payloads or alerts."""

    def map(self, value: str) -> Dict[str, Any]:
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in inference.completed message: {value}")
            return build_alert_event(
                value,
                alert_type="invalid_json",
                description="Could not decode inference.completed payload as JSON",
                severity="error",
            )

        try:
            return validate_inference_completed_event(data)
        except Exception as exc:
            logger.error(f"Failed to validate inference.completed event: {exc}")
            return build_alert_event(
                data,
                alert_type="analytics_validation_error",
                description=str(exc),
                severity="error",
            )


class EventTypeFilter(FilterFunction):
    """Filter stream records by event_type."""

    def __init__(self, event_type: str):
        self.event_type = event_type

    def filter(self, value: Dict[str, Any]) -> bool:
        return value.get("event_type") == self.event_type


class NonEmptyFieldFilter(FilterFunction):
    """Filter stream records that contain a non-empty field."""

    def __init__(self, field_name: str):
        self.field_name = field_name

    def filter(self, value: Dict[str, Any]) -> bool:
        return bool(value.get(self.field_name))


class ModelMetricsWindowAggregator(KeyedProcessFunction):
    """Aggregate inference completion events into event-time windows."""

    def __init__(self, window_size_seconds: int = 60, allowed_lateness_seconds: int = 15):
        self.window_size_seconds = window_size_seconds
        self.window_size_ms = window_size_seconds * 1000
        self.allowed_lateness_ms = allowed_lateness_seconds * 1000

    def open(self, runtime_context):
        self.window_buckets_state = runtime_context.get_list_state(
            ListStateDescriptor("window_buckets", Types.STRING())
        )
        self.flush_timer_state = runtime_context.get_state(
            ValueStateDescriptor("flush_timer_ms", Types.LONG())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        current_processing_time = ctx.timer_service().current_processing_time()
        event_timestamp_ms = completion_event_timestamp_ms(
            value,
            default_ms=current_processing_time,
        )
        window_start_ms = (event_timestamp_ms // self.window_size_ms) * self.window_size_ms
        window_end_ms = window_start_ms + self.window_size_ms

        if window_end_ms <= current_processing_time - self.allowed_lateness_ms:
            logger.debug(
                "Dropping late inference.completed event for model metrics: %s",
                value.get("request_id"),
            )
            return

        buckets = self._read_buckets()
        bucket = buckets.get(window_start_ms) or {
            "window_start_ms": window_start_ms,
            "provider": str(value.get("provider", "unknown") or "unknown"),
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "latency_sum_ms": 0.0,
            "tokens_per_second_sum": 0.0,
            "token_count_input": 0,
            "token_count_output": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "cached_count": 0,
        }
        bucket["provider"] = str(value.get("provider", bucket.get("provider") or "unknown") or "unknown")
        bucket["request_count"] += 1
        bucket["success_count"] += 1 if value.get("status") == "success" else 0
        bucket["error_count"] += 1 if value.get("status") != "success" else 0
        bucket["latency_sum_ms"] += float(value.get("latency_ms", 0.0) or 0.0)
        bucket["tokens_per_second_sum"] += float(
            value.get("tokens_per_second", 0.0) or 0.0
        )
        bucket["token_count_input"] += int(value.get("token_count_input", 0) or 0)
        bucket["token_count_output"] += int(value.get("token_count_output", 0) or 0)
        bucket["total_tokens"] += int(value.get("total_tokens", 0) or 0)
        bucket["total_cost_usd"] += float(value.get("cost_usd", 0.0) or 0.0)
        bucket["cached_count"] += 1 if value.get("cached_response") else 0
        buckets[window_start_ms] = bucket

        ready_windows = [
            start_ms
            for start_ms in buckets
            if start_ms + self.window_size_ms + self.allowed_lateness_ms
            <= current_processing_time
        ]
        for start_ms in sorted(ready_windows):
            yield self._build_metric_event(
                model_name=str(ctx.get_current_key()),
                bucket=buckets.pop(start_ms),
            )

        self._write_buckets(buckets)
        self._schedule_next_flush(ctx, buckets)

    def on_timer(self, timestamp: int, ctx: "KeyedProcessFunction.OnTimerContext"):
        buckets = self._read_buckets()
        ready_windows = [
            start_ms
            for start_ms in buckets
            if start_ms + self.window_size_ms + self.allowed_lateness_ms <= timestamp
        ]
        for start_ms in sorted(ready_windows):
            yield self._build_metric_event(
                model_name=str(ctx.get_current_key()),
                bucket=buckets.pop(start_ms),
            )

        self._write_buckets(buckets)
        self._schedule_next_flush(ctx, buckets)

    def _read_buckets(self) -> Dict[int, Dict[str, Any]]:
        buckets: Dict[int, Dict[str, Any]] = {}
        for item in self.window_buckets_state.get():
            bucket = json.loads(item)
            buckets[int(bucket["window_start_ms"])] = bucket
        return buckets

    def _write_buckets(self, buckets: Dict[int, Dict[str, Any]]):
        if not buckets:
            self.window_buckets_state.clear()
            return
        self.window_buckets_state.update(
            [json.dumps(bucket) for _, bucket in sorted(buckets.items())]
        )

    def _schedule_next_flush(
        self, ctx: "KeyedProcessFunction.Context", buckets: Dict[int, Dict[str, Any]]
    ):
        existing_timer = self.flush_timer_state.value()
        if existing_timer:
            ctx.timer_service().delete_processing_time_timer(existing_timer)
            self.flush_timer_state.clear()

        if not buckets:
            return

        next_flush_ms = min(
            int(window_start_ms) + self.window_size_ms + self.allowed_lateness_ms
            for window_start_ms in buckets
        )
        ctx.timer_service().register_processing_time_timer(next_flush_ms)
        self.flush_timer_state.update(next_flush_ms)

    def _build_metric_event(
        self, *, model_name: str, bucket: Dict[str, Any]
    ) -> Dict[str, Any]:
        window_start_ms = int(bucket.get("window_start_ms", 0) or 0)
        return build_model_metrics_window_event(
            model_name=model_name,
            provider=str(bucket.get("provider", "unknown") or "unknown"),
            window_start_ms=window_start_ms,
            window_end_ms=window_start_ms + self.window_size_ms,
            window_size_seconds=self.window_size_seconds,
            request_count=int(bucket.get("request_count", 0) or 0),
            success_count=int(bucket.get("success_count", 0) or 0),
            error_count=int(bucket.get("error_count", 0) or 0),
            latency_sum_ms=float(bucket.get("latency_sum_ms", 0.0) or 0.0),
            tokens_per_second_sum=float(
                bucket.get("tokens_per_second_sum", 0.0) or 0.0
            ),
            token_count_input=int(bucket.get("token_count_input", 0) or 0),
            token_count_output=int(bucket.get("token_count_output", 0) or 0),
            total_tokens=int(bucket.get("total_tokens", 0) or 0),
            total_cost_usd=float(bucket.get("total_cost_usd", 0.0) or 0.0),
            cached_count=int(bucket.get("cached_count", 0) or 0),
        )


class ModelMetricsAnomalyDetector(KeyedProcessFunction):
    """Detect anomalies from recent aggregated model metrics history."""

    def __init__(self, threshold_multiplier: float = 2.0, history_size: int = 10):
        self.threshold_multiplier = threshold_multiplier
        self.history_size = history_size

    def open(self, runtime_context):
        self.latency_history_state = runtime_context.get_list_state(
            ListStateDescriptor("latency_history", Types.FLOAT())
        )
        self.qps_history_state = runtime_context.get_list_state(
            ListStateDescriptor("qps_history", Types.FLOAT())
        )
        self.error_rate_history_state = runtime_context.get_list_state(
            ListStateDescriptor("error_rate_history", Types.FLOAT())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        latency_history = list(self.latency_history_state.get())
        qps_history = list(self.qps_history_state.get())
        error_rate_history = list(self.error_rate_history_state.get())

        alerts = detect_metric_anomalies(
            value,
            latency_history=latency_history,
            qps_history=qps_history,
            error_rate_history=error_rate_history,
            threshold_multiplier=self.threshold_multiplier,
        )
        for alert in alerts:
            yield alert

        latency_history.append(float(value.get("avg_latency_ms", 0.0) or 0.0))
        qps_history.append(float(value.get("queries_per_second", 0.0) or 0.0))
        error_rate_history.append(float(value.get("error_rate", 0.0) or 0.0))

        self.latency_history_state.update(latency_history[-self.history_size :])
        self.qps_history_state.update(qps_history[-self.history_size :])
        self.error_rate_history_state.update(error_rate_history[-self.history_size :])


class ModelRoutingGuardrailDetector(KeyedProcessFunction):
    """Emit model-scoped routing guardrails from recent metrics windows."""

    def __init__(self, threshold_multiplier: float = 2.0, history_size: int = 10):
        self.threshold_multiplier = threshold_multiplier
        self.history_size = history_size

    def open(self, runtime_context):
        self.latency_history_state = runtime_context.get_list_state(
            ListStateDescriptor("guardrail_latency_history", Types.FLOAT())
        )
        self.qps_history_state = runtime_context.get_list_state(
            ListStateDescriptor("guardrail_qps_history", Types.FLOAT())
        )
        self.error_rate_history_state = runtime_context.get_list_state(
            ListStateDescriptor("guardrail_error_rate_history", Types.FLOAT())
        )
        self.cache_hit_rate_history_state = runtime_context.get_list_state(
            ListStateDescriptor("guardrail_cache_hit_history", Types.FLOAT())
        )
        self.cost_per_1k_history_state = runtime_context.get_list_state(
            ListStateDescriptor("guardrail_cost_per_1k_history", Types.FLOAT())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        latency_history = list(self.latency_history_state.get())
        qps_history = list(self.qps_history_state.get())
        error_rate_history = list(self.error_rate_history_state.get())
        cache_hit_rate_history = list(self.cache_hit_rate_history_state.get())
        cost_per_1k_history = list(self.cost_per_1k_history_state.get())

        guardrails = detect_model_routing_guardrails(
            value,
            latency_history=latency_history,
            qps_history=qps_history,
            error_rate_history=error_rate_history,
            cache_hit_rate_history=cache_hit_rate_history,
            cost_per_1k_history=cost_per_1k_history,
            threshold_multiplier=self.threshold_multiplier,
        )
        for guardrail in guardrails:
            yield guardrail

        total_tokens = max(0, int(value.get("total_tokens", 0) or 0))
        total_cost_usd = float(value.get("total_cost_usd", 0.0) or 0.0)
        cost_per_1k = (total_cost_usd * 1000.0 / total_tokens) if total_tokens > 0 else 0.0

        latency_history.append(float(value.get("avg_latency_ms", 0.0) or 0.0))
        qps_history.append(float(value.get("queries_per_second", 0.0) or 0.0))
        error_rate_history.append(float(value.get("error_rate", 0.0) or 0.0))
        cache_hit_rate_history.append(float(value.get("cache_hit_rate", 0.0) or 0.0))
        cost_per_1k_history.append(cost_per_1k)

        self.latency_history_state.update(latency_history[-self.history_size :])
        self.qps_history_state.update(qps_history[-self.history_size :])
        self.error_rate_history_state.update(error_rate_history[-self.history_size :])
        self.cache_hit_rate_history_state.update(
            cache_hit_rate_history[-self.history_size :]
        )
        self.cost_per_1k_history_state.update(cost_per_1k_history[-self.history_size :])


class ProviderRoutingGuardrailDetector(KeyedProcessFunction):
    """Emit provider-scoped routing guardrails from recent metrics windows."""

    def __init__(self, threshold_multiplier: float = 2.0, history_size: int = 10):
        self.threshold_multiplier = threshold_multiplier
        self.history_size = history_size

    def open(self, runtime_context):
        self.latency_history_state = runtime_context.get_list_state(
            ListStateDescriptor("provider_guardrail_latency_history", Types.FLOAT())
        )
        self.error_rate_history_state = runtime_context.get_list_state(
            ListStateDescriptor("provider_guardrail_error_rate_history", Types.FLOAT())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        latency_history = list(self.latency_history_state.get())
        error_rate_history = list(self.error_rate_history_state.get())

        guardrails = detect_provider_routing_guardrails(
            value,
            latency_history=latency_history,
            error_rate_history=error_rate_history,
            threshold_multiplier=self.threshold_multiplier,
        )
        for guardrail in guardrails:
            yield guardrail

        latency_history.append(float(value.get("avg_latency_ms", 0.0) or 0.0))
        error_rate_history.append(float(value.get("error_rate", 0.0) or 0.0))

        self.latency_history_state.update(latency_history[-self.history_size :])
        self.error_rate_history_state.update(error_rate_history[-self.history_size :])


class RollingScopePolicyEmitter(KeyedProcessFunction):
    """Emit rolling routing policy state for a user or session scope."""

    def __init__(self, scope_type: str, window_size_seconds: int = 300):
        self.scope_type = scope_type
        self.window_size_seconds = window_size_seconds
        self.window_size_ms = window_size_seconds * 1000

    def open(self, runtime_context):
        self.recent_events_state = runtime_context.get_list_state(
            ListStateDescriptor(
                f"{self.scope_type}_recent_completion_events",
                Types.STRING(),
            )
        )
        self.cleanup_timer_state = runtime_context.get_state(
            ValueStateDescriptor(f"{self.scope_type}_cleanup_timer_ms", Types.LONG())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        current_processing_time = ctx.timer_service().current_processing_time()
        cutoff_ms = current_processing_time - self.window_size_ms
        event_timestamp_ms = completion_event_timestamp_ms(
            value,
            default_ms=current_processing_time,
        )
        if event_timestamp_ms < cutoff_ms:
            logger.debug(
                "Dropping stale inference.completed event for %s rolling policy: %s",
                self.scope_type,
                value.get("request_id"),
            )
            return

        recent_events = self._prune_recent_events(
            [json.loads(item) for item in self.recent_events_state.get()],
            cutoff_ms,
        )
        recent_events.append(
            {
                "event_timestamp_ms": event_timestamp_ms,
                "user_id": value.get("user_id"),
                "session_id": value.get("session_id"),
                "user_tier": value.get("user_tier"),
                "query_type": value.get("query_type"),
                "selected_model": value.get("selected_model"),
                "provider": value.get("provider"),
                "status": value.get("status"),
                "latency_ms": value.get("latency_ms"),
                "total_tokens": value.get("total_tokens"),
                "cost_usd": value.get("cost_usd"),
                "actual_fast_lane_hit": bool(
                    value.get("actual_fast_lane_hit", False)
                ),
                "route_to_fast_lane": bool(value.get("route_to_fast_lane", False)),
            }
        )
        self.recent_events_state.update(
            [json.dumps(event) for event in recent_events]
        )
        self._schedule_next_cleanup(ctx, recent_events)

        yield build_routing_policy_state_event(
            scope_type=self.scope_type,
            scope_key=str(ctx.get_current_key()),
            events=recent_events,
            window_size_seconds=self.window_size_seconds,
        )

    def on_timer(self, timestamp: int, ctx: "KeyedProcessFunction.OnTimerContext"):
        cutoff_ms = timestamp - self.window_size_ms
        recent_events = self._prune_recent_events(
            [json.loads(item) for item in self.recent_events_state.get()],
            cutoff_ms,
        )
        self.cleanup_timer_state.clear()
        if recent_events:
            self.recent_events_state.update(
                [json.dumps(event) for event in recent_events]
            )
            self._schedule_next_cleanup(ctx, recent_events)
            return
        self.recent_events_state.clear()

    def _schedule_next_cleanup(
        self, ctx: "KeyedProcessFunction.Context", recent_events: list[Dict[str, Any]]
    ):
        existing_timer = self.cleanup_timer_state.value()
        if existing_timer:
            ctx.timer_service().delete_processing_time_timer(existing_timer)
            self.cleanup_timer_state.clear()

        if not recent_events:
            return

        oldest_event_ms = min(
            int(event.get("event_timestamp_ms", 0) or 0) for event in recent_events
        )
        next_cleanup_ms = oldest_event_ms + self.window_size_ms + 1000
        current_processing_time = ctx.timer_service().current_processing_time()
        next_cleanup_ms = max(current_processing_time + 1000, next_cleanup_ms)
        ctx.timer_service().register_processing_time_timer(next_cleanup_ms)
        self.cleanup_timer_state.update(next_cleanup_ms)

    @staticmethod
    def _prune_recent_events(
        recent_events: list[Dict[str, Any]], cutoff_ms: int
    ) -> list[Dict[str, Any]]:
        return [
            event
            for event in recent_events
            if int(event.get("event_timestamp_ms", 0) or 0) >= cutoff_ms
        ]


def create_flink_job(config: Dict[str, Any]):
    """Create and configure the Flink analytics job."""
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(config.get("parallelism", 4))
    env.enable_checkpointing(config.get("checkpoint_interval_ms", 60000))

    kafka_topics = config.get("kafka", {}).get("topics", {})
    kafka_props = {
        "bootstrap.servers": ",".join(
            config.get("kafka", {}).get("bootstrap_servers", ["localhost:9092"])
        ),
        "group.id": config.get("kafka", {}).get(
            "consumer_group", "flink-inference-analytics"
        ),
        "auto.offset.reset": config.get("kafka", {}).get(
            "auto_offset_reset", "latest"
        ),
    }

    completion_consumer = FlinkKafkaConsumer(
        topics=[kafka_topics.get("inference_completed", "inference.completed")],
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props,
    )
    metrics_producer = FlinkKafkaProducer(
        topic=kafka_topics.get(
            "analytics_model_metrics_1m", "analytics.model_metrics_1m"
        ),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )
    routing_guardrail_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("routing_guardrails", "routing.guardrails"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )
    routing_policy_state_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("routing_policy_state", "routing.policy_state"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )
    alerts_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("alerts", "alerts"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )

    completion_stream = env.add_source(
        completion_consumer,
        "Kafka Inference Completed Source",
    )
    mapped_stream = completion_stream.map(
        InferenceCompletedMapper(),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )

    validation_alerts_stream = mapped_stream.filter(EventTypeFilter(ALERT_EVENT))
    valid_completions_stream = mapped_stream.filter(
        EventTypeFilter(INFERENCE_COMPLETED_EVENT)
    )

    window_size_seconds = int(config.get("window_size_seconds", 60))
    allowed_lateness_seconds = int(config.get("allowed_lateness_seconds", 15))
    metrics_stream = valid_completions_stream.key_by(
        lambda event: event.get("selected_model", "unknown")
    ).process(
        ModelMetricsWindowAggregator(
            window_size_seconds=window_size_seconds,
            allowed_lateness_seconds=allowed_lateness_seconds,
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )
    anomaly_stream = metrics_stream.key_by(
        lambda event: event.get("model_name", "unknown")
    ).process(
        ModelMetricsAnomalyDetector(
            threshold_multiplier=float(config.get("anomaly_threshold_multiplier", 2.0))
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )
    model_guardrail_stream = metrics_stream.key_by(
        lambda event: event.get("model_name", "unknown")
    ).process(
        ModelRoutingGuardrailDetector(
            threshold_multiplier=float(config.get("guardrail_threshold_multiplier", 2.0))
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )
    provider_guardrail_stream = metrics_stream.key_by(
        lambda event: event.get("provider", "unknown")
    ).process(
        ProviderRoutingGuardrailDetector(
            threshold_multiplier=float(config.get("guardrail_threshold_multiplier", 2.0))
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )
    rolling_policy_window_seconds = int(
        config.get("rolling_policy_window_seconds", 300)
    )
    user_policy_state_stream = valid_completions_stream.filter(
        NonEmptyFieldFilter("user_id")
    ).key_by(
        lambda event: event.get("user_id", "unknown")
    ).process(
        RollingScopePolicyEmitter(
            scope_type="user",
            window_size_seconds=rolling_policy_window_seconds,
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )
    session_policy_state_stream = valid_completions_stream.filter(
        NonEmptyFieldFilter("session_id")
    ).key_by(
        lambda event: event.get("session_id", "unknown")
    ).process(
        RollingScopePolicyEmitter(
            scope_type="session",
            window_size_seconds=rolling_policy_window_seconds,
        ),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )

    metrics_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(metrics_producer).name("Model Metrics 1m Sink")
    validation_alerts_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(alerts_producer).name("Validation Alerts Sink")
    anomaly_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(alerts_producer).name("Anomaly Alerts Sink")
    model_guardrail_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(routing_guardrail_producer).name("Model Routing Guardrails Sink")
    provider_guardrail_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(routing_guardrail_producer).name("Provider Routing Guardrails Sink")
    user_policy_state_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(routing_policy_state_producer).name("User Routing Policy State Sink")
    session_policy_state_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(routing_policy_state_producer).name("Session Routing Policy State Sink")

    return env


def main():
    """Main entry point for the Flink analytics job."""
    config_json = os.environ.get("FLINK_CONFIG_JSON")
    config = {
        "parallelism": 4,
        "checkpoint_interval_ms": 60000,
        "window_size_seconds": 60,
        "allowed_lateness_seconds": 15,
        "rolling_policy_window_seconds": 300,
        "anomaly_threshold_multiplier": 2.0,
        "guardrail_threshold_multiplier": 2.0,
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "consumer_group": "flink-inference-analytics",
            "topics": {
                "inference_completed": "inference.completed",
                "analytics_model_metrics_1m": "analytics.model_metrics_1m",
                "routing_guardrails": "routing.guardrails",
                "routing_policy_state": "routing.policy_state",
                "alerts": "alerts",
            },
        },
    }

    if config_json:
        config = json.loads(config_json)

    env = create_flink_job(config)
    env.execute("LLM Router Inference Analytics Job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
