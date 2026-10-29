"""A5 round 2: all-bad poison batches drain, schema backoff does not restart
consumers, row rejections are counted, and unparseable DLQ payloads stay private."""

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import KafkaConsumerManager, QueryLogEntry
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import UserTier


class DatabaseError(Exception):
    pass


class DataError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


TP = TopicPartition("test-queries", 0)


def _row(offset):
    return QueryLogEntry(
        query_id=f"q-{offset}",
        timestamp=datetime.now(timezone.utc),
        user_id="u",
        user_tier=UserTier.PREMIUM,
        query_text="secret question",
        query_type="general",
        selected_model="gpt-5",
        token_count_input=1,
        token_count_output=1,
        latency_ms=1,
        cost_usd=0.0,
        status="success",
        kafka_topic="test-queries",
        kafka_partition=0,
        kafka_offset=offset,
    )


def _consumer():
    consumer = KafkaConsumerManager(
        {"topics": {"queries": "test-queries", "requests_raw": "requests.raw"}},
        MagicMock(),
    )
    consumer.dlq_producer = AsyncMock()
    consumer.consumers["queries"] = AsyncMock()
    return consumer


def _buffer(consumer, rows):
    for row in rows:
        consumer.batch_processors["queries"].append(row)
        consumer.pending_commit_offsets["queries"][TP] = row.kafka_offset + 1


def _failures(kind):
    return PIPELINE_METRICS.clickhouse_flush_failures.labels(topic="queries", kind=kind)


@pytest.mark.parametrize(
    "error",
    [
        DataError("Failed to write column 'user_tier'"),
        ValueError("invalid literal for int()"),
        DatabaseError("Code: 691. DB::Exception: Unknown element (UNKNOWN_ENUM)"),
    ],
    ids=["DataError", "ValueError", "Code691"],
)
@pytest.mark.asyncio
async def test_batch_of_only_poison_rows_is_dead_lettered_not_kept(error):
    consumer = _consumer()
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(side_effect=error)
    _buffer(consumer, [_row(0), _row(1), _row(2)])
    before = _failures("row")._value.get()

    await consumer.flush_pending_batches()

    assert consumer.dlq_producer.send_and_wait.await_count == 3
    consumer.consumers["queries"].commit.assert_awaited_once_with({TP: 3})
    assert consumer.batch_processors["queries"] == []
    assert _failures("row")._value.get() == before + 1


@pytest.mark.asyncio
async def test_resolved_poison_batch_counts_one_row_failure():
    consumer = _consumer()
    inserted = []

    async def insert(batch):
        if any(row.kafka_offset == 1 for row in batch):
            raise DataError("bad row")
        inserted.extend(batch)

    consumer.clickhouse.batch_insert_query_logs = insert
    _buffer(consumer, [_row(0), _row(1), _row(2), _row(3)])
    before = _failures("row")._value.get()

    await consumer.flush_pending_batches()

    assert sorted(row.kafka_offset for row in inserted) == [0, 2, 3]
    assert _failures("row")._value.get() == before + 1


@pytest.mark.asyncio
async def test_poison_message_during_schema_backoff_does_not_raise():
    consumer = _consumer()
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(
        side_effect=ProgrammingError("Unrecognized column 'new_col'")
    )
    buffered = [_row(0)]
    _buffer(consumer, buffered)
    poison = SimpleNamespace(
        topic="test-queries", partition=0, offset=1, value=b"{", key=None
    )

    # Must not raise: raising restarts the consumer and re-reads the buffer.
    await consumer._handle_processing_failure(
        "queries", poison, ValueError("bad json"), stage="consume", batched=True
    )

    assert consumer.dlq_producer.send_and_wait.await_args.kwargs["key"] == (
        "test-queries:0:1"
    )
    # The poison offset is committed only with the buffered rows it follows.
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert consumer.batch_processors["queries"] == buffered
    assert consumer.pending_commit_offsets["queries"] == {TP: 2}

    consumer.clickhouse.batch_insert_query_logs = AsyncMock()
    await consumer.flush_pending_batches()
    consumer.consumers["queries"].commit.assert_awaited_once_with({TP: 2})


@pytest.mark.parametrize("topic_key", ["requests_raw", "queries"])
@pytest.mark.asyncio
async def test_unparseable_payload_is_replaced_by_a_summary(topic_key):
    consumer = _consumer()
    raw = '{"query_text": "secret question about my diagnosis'
    message = SimpleNamespace(
        value=raw.encode("utf-8"), key=None, topic="t", partition=0, offset=5
    )

    await consumer._publish_dead_letter(
        topic_key, message, ValueError("Unterminated string"), stage="consume"
    )

    payload = consumer.dlq_producer.send_and_wait.await_args.kwargs["value"]
    assert "secret" not in json.dumps(payload)
    summary = payload["payload"]
    assert summary["unparseable"] is True
    assert summary["length"] == len(raw.encode("utf-8"))
    assert summary["sha256_prefix"] == hashlib.sha256(raw.encode()).hexdigest()[:16]
    assert summary["reason"] == "ValueError"
