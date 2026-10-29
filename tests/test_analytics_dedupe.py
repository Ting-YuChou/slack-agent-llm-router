"""A4: analytics rows keep a replay-stable identity.

Kafka coordinates change when Flink replays a window or the producer resends a
message, so the ClickHouse ``event_id`` must come from the payload identity.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    KafkaConsumerManager,
    KafkaProducerManager,
)


class _MessageConsumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.fixture
def pipeline_config():
    return {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {
            "queries": "test-queries",
            "analytics_model_metrics_1m": "analytics.model_metrics_1m",
        },
        "consumer": {"group_id": "test-group"},
    }


QUERY_PAYLOAD = {
    "event_id": "query-1",
    "query_id": "query-1",
    "timestamp": "2026-04-19T00:00:00+00:00",
    "user_id": "user-1",
    "user_tier": "premium",
    "query_text": "hello",
    "query_type": "general",
    "selected_model": "gpt-5",
    "token_count_input": 1,
    "token_count_output": 2,
    "latency_ms": 10,
    "cost_usd": 0.01,
    "status": "success",
}

WINDOW_PAYLOAD = {
    "event_type": "analytics.model_metrics_1m",
    "model_name": "gpt-5",
    "provider": "openai",
    "window_start_ms": 1_776_556_800_000,
    "window_end_ms": 1_776_556_860_000,
    "window_size_seconds": 60,
    "request_count": 4,
    "success_count": 4,
    "success_rate": 1.0,
}


@pytest.mark.asyncio
async def test_query_replay_at_new_offset_keeps_payload_event_id(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    messages = [
        SimpleNamespace(
            value=json.dumps(QUERY_PAYLOAD).encode("utf-8"),
            topic="test-queries",
            partition=0,
            offset=offset,
        )
        for offset in (8, 41)
    ]

    await consumer._consume_queries(_MessageConsumer(messages))

    first, second = consumer.batch_processors["queries"]
    assert first.event_id == second.event_id == "query-1"
    assert first.to_dict()["event_id"] == second.to_dict()["event_id"]
    # Coordinates are still kept for lineage.
    assert (first.kafka_offset, second.kafka_offset) == (8, 41)


@pytest.mark.asyncio
async def test_query_dlq_key_is_still_the_kafka_coordinate(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    consumer.dlq_producer = AsyncMock()
    bad = {**QUERY_PAYLOAD, "timestamp": "garbage"}
    message = SimpleNamespace(
        value=json.dumps(bad).encode("utf-8"),
        key=b"user-1",
        topic="test-queries",
        partition=1,
        offset=5,
    )

    await consumer._consume_queries(_MessageConsumer([message]))

    kwargs = consumer.dlq_producer.send_and_wait.await_args.kwargs
    assert kwargs["key"] == "test-queries:1:5"


def test_window_replay_with_new_emitted_at_has_same_identity(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    first = consumer._build_model_performance_entry(
        {**WINDOW_PAYLOAD, "emitted_at": "2026-04-19T00:01:02+00:00"},
        kafka_metadata=consumer._kafka_event_metadata(
            SimpleNamespace(topic="analytics.model_metrics_1m", partition=0, offset=3)
        ),
    )
    second = consumer._build_model_performance_entry(
        {**WINDOW_PAYLOAD, "emitted_at": "2026-04-19T00:09:30+00:00"},
        kafka_metadata=consumer._kafka_event_metadata(
            SimpleNamespace(topic="analytics.model_metrics_1m", partition=0, offset=97)
        ),
    )

    assert first.event_id == second.event_id
    assert first.event_id == "gpt-5:openai:1776556800000:1776556860000"
    window_start = datetime.fromtimestamp(1_776_556_800, tz=timezone.utc)
    assert first.timestamp == second.timestamp == window_start
    assert first.to_dict()["event_id"] == second.to_dict()["event_id"]
    assert (first.kafka_offset, second.kafka_offset) == (3, 97)


def test_window_without_start_falls_back_to_emitted_at(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    payload = {
        key: value
        for key, value in WINDOW_PAYLOAD.items()
        if key not in {"window_start_ms", "window_end_ms"}
    }
    entry = consumer._build_model_performance_entry(
        {**payload, "emitted_at": "2026-04-19T00:01:02+00:00"}
    )

    assert entry.timestamp == datetime(2026, 4, 19, 0, 1, 2, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_producer_does_not_resend_after_ack_timeout(pipeline_config):
    config = {
        **pipeline_config,
        "producer": {"retries": 2, "retry_backoff_ms": 0, "send_timeout_seconds": 1},
    }
    manager = KafkaProducerManager(config)
    manager.producer = AsyncMock()
    manager.producer.send_and_wait = AsyncMock(side_effect=asyncio.TimeoutError())

    sent = await manager._send_message(
        "queries", "llm-queries", key="u1", value={"hello": "world"}
    )

    assert sent is False
    assert manager.producer.send_and_wait.await_count == 1
    assert manager.consecutive_failures == 1


@pytest.mark.asyncio
async def test_producer_still_retries_other_errors(pipeline_config):
    config = {
        **pipeline_config,
        "producer": {"retries": 2, "retry_backoff_ms": 0, "send_timeout_seconds": 1},
    }
    manager = KafkaProducerManager(config)
    manager.producer = AsyncMock()
    manager.producer.send_and_wait = AsyncMock(
        side_effect=[RuntimeError("a"), RuntimeError("b"), object()]
    )

    sent = await manager._send_message(
        "queries", "llm-queries", key="u1", value={"hello": "world"}
    )

    assert sent is True
    assert manager.producer.send_and_wait.await_count == 3


def test_idempotent_producer_is_the_default(pipeline_config):
    manager = KafkaProducerManager(pipeline_config)
    assert manager.producer_config["enable_idempotence"] is True


CLICKHOUSE_HOST = os.environ.get("CHAT_TEST_CLICKHOUSE_HOST")


@pytest.mark.skipif(
    not CLICKHOUSE_HOST,
    reason="set CHAT_TEST_CLICKHOUSE_HOST to run against a real ClickHouse",
)
@pytest.mark.asyncio
async def test_real_clickhouse_window_replay_collapses_under_final(pipeline_config):
    pytest.importorskip("clickhouse_connect")
    database = f"a4_dedupe_{uuid.uuid4().hex[:8]}"
    manager = ClickHouseManager(
        {
            "host": CLICKHOUSE_HOST,
            "port": int(os.environ.get("CHAT_TEST_CLICKHOUSE_PORT", "8123")),
            "username": os.environ.get("CHAT_TEST_CLICKHOUSE_USER", "default"),
            "password": os.environ.get("CHAT_TEST_CLICKHOUSE_PASSWORD", ""),
            "database": database,
        }
    )
    consumer = KafkaConsumerManager(pipeline_config, manager)
    await manager.initialize()
    # Use a recent window: the table TTL drops rows older than 60 days.
    window_start_ms = (int(time.time()) // 60 - 5) * 60_000
    window = {
        **WINDOW_PAYLOAD,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_start_ms + 60_000,
    }
    try:
        for offset, delay_s in ((3, 2), (97, 290)):
            emitted_at = datetime.fromtimestamp(
                window_start_ms / 1000 + 60 + delay_s, tz=timezone.utc
            )
            entry = consumer._build_model_performance_entry(
                {**window, "emitted_at": emitted_at.isoformat()},
                kafka_metadata=consumer._kafka_event_metadata(
                    SimpleNamespace(
                        topic="analytics.model_metrics_1m", partition=0, offset=offset
                    )
                ),
            )
            await manager.batch_insert_model_performance([entry])

        result = manager.client.query(
            f"SELECT count() FROM {database}.model_performance FINAL"
        )
        assert result.result_rows[0][0] == 1
    finally:
        manager.client.command(f"DROP DATABASE IF EXISTS {database}")
