"""A4 review: retry only unsent records; window alerts dedupe across replays."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flink.logic import build_metric_anomaly_alert_event, build_routing_guardrail_event
from src.llm_router_part3_pipeline import KafkaConsumerManager, KafkaProducerManager

WINDOW = {
    "model_name": "gpt-5",
    "provider": "openai",
    "window_start_ms": 1_776_556_800_000,
    "window_end_ms": 1_776_556_860_000,
}


class TwoPhaseProducer:
    """Mimics aiokafka: ``send`` enqueues and returns a delivery future."""

    def __init__(self, enqueue_delays, deliver=True):
        self.enqueue_delays = list(enqueue_delays)
        self.deliver = deliver
        self.sends = 0

    async def send(self, topic, key=None, value=None):
        self.sends += 1
        delay = self.enqueue_delays.pop(0) if self.enqueue_delays else 0
        if delay:
            await asyncio.sleep(delay)
        future = asyncio.get_running_loop().create_future()
        if self.deliver:
            future.set_result(SimpleNamespace(offset=1))
        return future


def _manager(producer, retries=2):
    manager = KafkaProducerManager(
        {
            "topics": {"queries": "llm-queries"},
            "producer": {
                "retries": retries,
                "retry_backoff_ms": 0,
                "send_timeout_seconds": 0.05,
            },
        }
    )
    manager.producer = producer
    return manager


@pytest.mark.asyncio
async def test_timeout_before_the_record_is_enqueued_is_retried():
    producer = TwoPhaseProducer(enqueue_delays=[1.0, 0])
    manager = _manager(producer)

    sent = await manager._send_message("queries", "llm-queries", key="u", value={})

    assert sent is True
    assert producer.sends == 2


@pytest.mark.asyncio
async def test_timeout_waiting_for_the_ack_is_not_resent():
    producer = TwoPhaseProducer(enqueue_delays=[0], deliver=False)
    manager = _manager(producer)

    sent = await manager._send_message("queries", "llm-queries", key="u", value={})

    assert sent is False
    assert producer.sends == 1


def _consumer():
    return KafkaConsumerManager(
        {"topics": {"alerts": "alerts", "routing_guardrails": "routing.guardrails"}},
        MagicMock(),
    )


def _meta(consumer, topic, offset):
    return consumer._kafka_event_metadata(
        SimpleNamespace(topic=topic, partition=0, offset=offset)
    )


def _emitted(minute):
    return datetime(2026, 4, 19, 0, minute, tzinfo=timezone.utc)


def test_replayed_anomaly_alert_keeps_its_identity():
    consumer = _consumer()
    entries = [
        consumer._build_alert_event_entry(
            build_metric_anomaly_alert_event(
                WINDOW,
                anomaly_type="latency_spike",
                description="slow",
                timestamp=_emitted(minute),
            ),
            kafka_metadata=_meta(consumer, "alerts", offset),
        )
        for minute, offset in ((2, 5), (9, 77))
    ]

    first, second = entries
    assert first.to_dict()["event_id"] == second.to_dict()["event_id"]
    assert first.timestamp == second.timestamp
    assert first.timestamp == datetime.fromtimestamp(1_776_556_860, tz=timezone.utc)
    assert "alerts:0:5" not in first.event_id
    assert (first.kafka_offset, second.kafka_offset) == (5, 77)


def test_replayed_guardrail_keeps_its_identity_and_scopes_stay_apart():
    consumer = _consumer()

    def guardrail(scope_key, minute, offset, trigger="high_error_rate"):
        return consumer._build_alert_event_entry(
            build_routing_guardrail_event(
                WINDOW,
                scope_type="model",
                scope_key=scope_key,
                trigger_type=trigger,
                description="errors",
                timestamp=_emitted(minute),
            ),
            kafka_metadata=_meta(consumer, "routing.guardrails", offset),
        )

    first = guardrail("gpt-5", 2, 1)
    replay = guardrail("gpt-5", 8, 40)
    other_scope = guardrail("gpt-4o", 2, 2)
    other_trigger = guardrail("gpt-5", 2, 3, trigger="high_latency")

    assert first.event_id == replay.event_id
    assert first.timestamp == replay.timestamp
    assert len({first.event_id, other_scope.event_id, other_trigger.event_id}) == 3


def test_request_alert_without_a_window_keeps_existing_identity():
    consumer = _consumer()
    entry = consumer._build_alert_event_entry(
        {
            "event_type": "alert",
            "alert_type": "invalid_json",
            "request_id": "r1",
            "emitted_at": "2026-04-19T00:02:00+00:00",
        },
        kafka_metadata=_meta(consumer, "alerts", 9),
    )

    assert entry.event_id == "alerts:0:9"
    assert entry.timestamp == _emitted(2)


@pytest.mark.asyncio
async def test_poison_alert_dlq_key_is_still_the_coordinate():
    consumer = _consumer()
    consumer.dlq_producer = AsyncMock()

    class Messages:
        def __init__(self):
            self.items = [
                SimpleNamespace(
                    value=b"{bad", key=None, topic="alerts", partition=0, offset=4
                )
            ]
            self.commit = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    consumer.consumers["alerts"] = Messages()
    await consumer._consume_alerts(consumer.consumers["alerts"])

    kwargs = consumer.dlq_producer.send_and_wait.await_args.kwargs
    assert kwargs["key"] == "alerts:0:4"
    assert json.loads(json.dumps(kwargs["value"]))["source_offset"] == 4
