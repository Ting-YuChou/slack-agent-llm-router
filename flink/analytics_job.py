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
    build_alert_event,
    build_model_metrics_window_event,
    detect_metric_anomalies,
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


class ModelMetricsWindowAggregator(KeyedProcessFunction):
    """Aggregate inference completion events into fixed processing-time windows."""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size_seconds = window_size_seconds
        self.window_size_ms = window_size_seconds * 1000

    def open(self, runtime_context):
        self.request_count_state = runtime_context.get_state(
            ValueStateDescriptor("request_count", Types.LONG())
        )
        self.success_count_state = runtime_context.get_state(
            ValueStateDescriptor("success_count", Types.LONG())
        )
        self.error_count_state = runtime_context.get_state(
            ValueStateDescriptor("error_count", Types.LONG())
        )
        self.latency_sum_state = runtime_context.get_state(
            ValueStateDescriptor("latency_sum_ms", Types.FLOAT())
        )
        self.tokens_per_second_sum_state = runtime_context.get_state(
            ValueStateDescriptor("tokens_per_second_sum", Types.FLOAT())
        )
        self.token_input_sum_state = runtime_context.get_state(
            ValueStateDescriptor("token_input_sum", Types.LONG())
        )
        self.token_output_sum_state = runtime_context.get_state(
            ValueStateDescriptor("token_output_sum", Types.LONG())
        )
        self.total_tokens_sum_state = runtime_context.get_state(
            ValueStateDescriptor("total_tokens_sum", Types.LONG())
        )
        self.cost_sum_state = runtime_context.get_state(
            ValueStateDescriptor("cost_sum", Types.FLOAT())
        )
        self.cached_count_state = runtime_context.get_state(
            ValueStateDescriptor("cached_count", Types.LONG())
        )
        self.window_start_state = runtime_context.get_state(
            ValueStateDescriptor("window_start_ms", Types.LONG())
        )
        self.provider_state = runtime_context.get_state(
            ValueStateDescriptor("provider", Types.STRING())
        )

    def process_element(self, value: Dict[str, Any], ctx: "KeyedProcessFunction.Context"):
        current_processing_time = ctx.timer_service().current_processing_time()
        window_start_ms = self.window_start_state.value()

        if window_start_ms is None:
            window_start_ms = (
                current_processing_time // self.window_size_ms
            ) * self.window_size_ms
            self.window_start_state.update(window_start_ms)
            ctx.timer_service().register_processing_time_timer(
                window_start_ms + self.window_size_ms
            )

        self.provider_state.update(value.get("provider", self.provider_state.value() or "unknown"))
        self.request_count_state.update((self.request_count_state.value() or 0) + 1)
        self.success_count_state.update(
            (self.success_count_state.value() or 0)
            + (1 if value.get("status") == "success" else 0)
        )
        self.error_count_state.update(
            (self.error_count_state.value() or 0)
            + (1 if value.get("status") != "success" else 0)
        )
        self.latency_sum_state.update(
            float(self.latency_sum_state.value() or 0.0)
            + float(value.get("latency_ms", 0.0) or 0.0)
        )
        self.tokens_per_second_sum_state.update(
            float(self.tokens_per_second_sum_state.value() or 0.0)
            + float(value.get("tokens_per_second", 0.0) or 0.0)
        )
        self.token_input_sum_state.update(
            (self.token_input_sum_state.value() or 0)
            + int(value.get("token_count_input", 0) or 0)
        )
        self.token_output_sum_state.update(
            (self.token_output_sum_state.value() or 0)
            + int(value.get("token_count_output", 0) or 0)
        )
        self.total_tokens_sum_state.update(
            (self.total_tokens_sum_state.value() or 0)
            + int(value.get("total_tokens", 0) or 0)
        )
        self.cost_sum_state.update(
            float(self.cost_sum_state.value() or 0.0)
            + float(value.get("cost_usd", 0.0) or 0.0)
        )
        self.cached_count_state.update(
            (self.cached_count_state.value() or 0)
            + (1 if value.get("cached_response") else 0)
        )

    def on_timer(self, timestamp: int, ctx: "KeyedProcessFunction.OnTimerContext"):
        window_start_ms = self.window_start_state.value() or (
            timestamp - self.window_size_ms
        )
        request_count = self.request_count_state.value() or 0
        if request_count == 0:
            self.window_start_state.clear()
            self.provider_state.clear()
            return

        metric_event = build_model_metrics_window_event(
            model_name=ctx.get_current_key(),
            provider=self.provider_state.value() or "unknown",
            window_start_ms=window_start_ms,
            window_end_ms=timestamp,
            window_size_seconds=self.window_size_seconds,
            request_count=request_count,
            success_count=self.success_count_state.value() or 0,
            error_count=self.error_count_state.value() or 0,
            latency_sum_ms=float(self.latency_sum_state.value() or 0.0),
            tokens_per_second_sum=float(
                self.tokens_per_second_sum_state.value() or 0.0
            ),
            token_count_input=self.token_input_sum_state.value() or 0,
            token_count_output=self.token_output_sum_state.value() or 0,
            total_tokens=self.total_tokens_sum_state.value() or 0,
            total_cost_usd=float(self.cost_sum_state.value() or 0.0),
            cached_count=self.cached_count_state.value() or 0,
        )

        yield metric_event

        self.request_count_state.clear()
        self.success_count_state.clear()
        self.error_count_state.clear()
        self.latency_sum_state.clear()
        self.tokens_per_second_sum_state.clear()
        self.token_input_sum_state.clear()
        self.token_output_sum_state.clear()
        self.total_tokens_sum_state.clear()
        self.cost_sum_state.clear()
        self.cached_count_state.clear()
        self.provider_state.clear()
        self.window_start_state.clear()


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
    metrics_stream = valid_completions_stream.key_by(
        lambda event: event.get("selected_model", "unknown")
    ).process(
        ModelMetricsWindowAggregator(window_size_seconds=window_size_seconds),
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

    return env


def main():
    """Main entry point for the Flink analytics job."""
    config_json = os.environ.get("FLINK_CONFIG_JSON")
    config = {
        "parallelism": 4,
        "checkpoint_interval_ms": 60000,
        "window_size_seconds": 60,
        "anomaly_threshold_multiplier": 2.0,
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "consumer_group": "flink-inference-analytics",
            "topics": {
                "inference_completed": "inference.completed",
                "analytics_model_metrics_1m": "analytics.model_metrics_1m",
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
