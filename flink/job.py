"""
Apache Flink enrichment job for the LLM Router platform.

This job consumes pre-inference API request events, enriches them with
priority and fast-lane hints, and emits independent alert events for
malformed payloads.
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
from pyflink.datastream.functions import FilterFunction, MapFunction

from flink.logic import (
    ALERT_EVENT,
    REQUESTS_ENRICHED_EVENT,
    build_alert_event,
    build_fast_lane_hint_event,
    build_request_enriched_event,
)

logger = logging.getLogger(__name__)


class QueryEnrichmentMapper(MapFunction):
    """Map raw request events into enriched request events or alerts."""

    def map(self, value: str) -> Dict[str, Any]:
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in requests.raw message: {value}")
            return build_alert_event(
                value,
                alert_type="invalid_json",
                description="Could not decode requests.raw payload as JSON",
                severity="error",
            )

        try:
            return build_request_enriched_event(data)
        except Exception as exc:
            logger.error(f"Failed to enrich request event: {exc}")
            return build_alert_event(
                data,
                alert_type="enrichment_error",
                description=str(exc),
                severity="error",
            )


class EventTypeFilter(FilterFunction):
    """Filter stream records by event_type."""

    def __init__(self, event_type: str):
        self.event_type = event_type

    def filter(self, value: Dict[str, Any]) -> bool:
        return value.get("event_type") == self.event_type


class FastLaneHintFilter(FilterFunction):
    """Filter enriched events that should emit a fast-lane routing hint."""

    def filter(self, value: Dict[str, Any]) -> bool:
        return bool(value.get("route_to_fast_lane"))


def create_flink_job(config: Dict[str, Any]):
    """Create and configure the Flink enrichment job."""
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(config.get("parallelism", 4))
    env.enable_checkpointing(config.get("checkpoint_interval_ms", 60000))

    kafka_topics = config.get("kafka", {}).get("topics", {})
    kafka_props = {
        "bootstrap.servers": ",".join(
            config.get("kafka", {}).get("bootstrap_servers", ["localhost:9092"])
        ),
        "group.id": config.get("kafka", {}).get(
            "consumer_group", "flink-request-enrichment"
        ),
        "auto.offset.reset": config.get("kafka", {}).get(
            "auto_offset_reset", "latest"
        ),
    }

    request_consumer = FlinkKafkaConsumer(
        topics=[kafka_topics.get("requests_raw", "requests.raw")],
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props,
    )
    enriched_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("requests_enriched", "requests.enriched"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )
    fast_lane_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("fast_lane_hints", "fast_lane_hints"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )
    alert_producer = FlinkKafkaProducer(
        topic=kafka_topics.get("alerts", "alerts"),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props,
    )

    request_stream = env.add_source(request_consumer, "Kafka Requests Raw Source")
    mapped_stream = request_stream.map(
        QueryEnrichmentMapper(),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )

    alerts_stream = mapped_stream.filter(EventTypeFilter(ALERT_EVENT))
    enriched_stream = mapped_stream.filter(EventTypeFilter(REQUESTS_ENRICHED_EVENT))
    fast_lane_stream = enriched_stream.filter(FastLaneHintFilter()).map(
        lambda event: build_fast_lane_hint_event(event),
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )

    enriched_stream.map(lambda event: json.dumps(event), output_type=Types.STRING()).add_sink(
        enriched_producer
    ).name("Requests Enriched Sink")
    fast_lane_stream.map(
        lambda event: json.dumps(event),
        output_type=Types.STRING(),
    ).add_sink(fast_lane_producer).name("Fast Lane Hint Sink")
    alerts_stream.map(lambda event: json.dumps(event), output_type=Types.STRING()).add_sink(
        alert_producer
    ).name("Alerts Sink")

    return env


def main():
    """Main entry point for the Flink request enrichment job."""
    config_json = os.environ.get("FLINK_CONFIG_JSON")
    config = {
        "parallelism": 4,
        "checkpoint_interval_ms": 60000,
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "consumer_group": "flink-request-enrichment",
            "topics": {
                "requests_raw": "requests.raw",
                "requests_enriched": "requests.enriched",
                "fast_lane_hints": "fast_lane_hints",
                "alerts": "alerts",
            },
        },
    }

    if config_json:
        config = json.loads(config_json)

    env = create_flink_job(config)
    env.execute("LLM Router Request Enrichment Job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
