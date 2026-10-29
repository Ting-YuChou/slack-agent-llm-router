"""A5 review: schema drift is a stuck topic, DLQ failures stop the bisection,
and dead-lettered payloads stay private for dict rows too."""

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


def _row(offset, *, bad=False):
    return QueryLogEntry(
        query_id=f"q-{offset}",
        timestamp=datetime.now(timezone.utc),
        user_id="u",
        user_tier="bogus" if bad else UserTier.PREMIUM,
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


def _consumer(dlq=None):
    consumer = KafkaConsumerManager(
        {
            "topics": {"queries": "test-queries", "inference_completed": "ic"},
            "consumer": {"flush_retry_initial_backoff_seconds": 1},
        },
        MagicMock(),
    )
    consumer.dlq_producer = dlq if dlq is not None else AsyncMock()
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
        ProgrammingError("Unrecognized column 'new_col' in table query_logs"),
        DatabaseError(
            "Code: 16. DB::Exception: No such column new_col in table. "
            "(NO_SUCH_COLUMN_IN_TABLE)"
        ),
    ],
    ids=["ProgrammingError", "Code16"],
)
@pytest.mark.asyncio
async def test_schema_errors_keep_the_batch_instead_of_dead_lettering(error):
    consumer = _consumer()
    insert = AsyncMock(side_effect=error)
    consumer.clickhouse.batch_insert_query_logs = insert
    rows = [_row(0), _row(1), _row(2)]
    _buffer(consumer, rows)
    before = _failures("schema")._value.get()

    with pytest.raises(type(error)):
        await consumer.flush_pending_batches()

    assert insert.await_count == 1  # no bisection
    consumer.dlq_producer.send_and_wait.assert_not_awaited()
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert consumer.batch_processors["queries"] == rows
    assert consumer.pending_commit_offsets["queries"] == {TP: 3}
    assert _failures("schema")._value.get() == before + 1
    assert consumer.batch_buffer_states["queries"].next_flush_at > 0


@pytest.mark.asyncio
async def test_every_row_rejected_is_treated_as_schema_drift():
    consumer = _consumer()
    error = DatabaseError("Code: 53. DB::Exception: Type mismatch (TYPE_MISMATCH)")
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(side_effect=error)
    rows = [_row(0), _row(1), _row(2)]
    _buffer(consumer, rows)
    before = _failures("schema")._value.get()

    with pytest.raises(DatabaseError):
        await consumer.flush_pending_batches()

    consumer.dlq_producer.send_and_wait.assert_not_awaited()
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert consumer.batch_processors["queries"] == rows
    assert _failures("schema")._value.get() == before + 1


@pytest.mark.asyncio
async def test_dlq_failure_stops_at_once_and_keeps_the_bad_rows():
    dlq = AsyncMock()
    dlq.send_and_wait = AsyncMock(side_effect=RuntimeError("dlq down"))
    consumer = _consumer(dlq)
    inserted = []

    async def insert(batch):
        if any(row.user_tier == "bogus" for row in batch):
            raise DataError("bad tier")
        inserted.extend(batch)

    consumer.clickhouse.batch_insert_query_logs = insert
    bad_one, bad_two = _row(1, bad=True), _row(3, bad=True)
    _buffer(consumer, [_row(0), bad_one, _row(2), bad_two])
    before = _failures("dlq_failed")._value.get()

    with pytest.raises(DataError):
        await consumer.flush_pending_batches()

    assert dlq.send_and_wait.await_count == 1
    assert sorted(row.kafka_offset for row in inserted) == [0, 2]
    assert consumer.batch_processors["queries"] == [bad_one, bad_two]
    assert consumer.pending_commit_offsets["queries"] == {TP: 4}
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert _failures("dlq_failed")._value.get() == before + 1
    assert not consumer.batch_flush_locks["queries"].locked()


@pytest.mark.asyncio
async def test_transient_error_during_bisection_is_counted():
    consumer = _consumer()
    calls = []

    async def insert(batch):
        calls.append(len(batch))
        if len(calls) == 1:
            raise DataError("bad row somewhere")
        raise RuntimeError("clickhouse went away")

    consumer.clickhouse.batch_insert_query_logs = insert
    rows = [_row(0), _row(1), _row(2), _row(3)]
    _buffer(consumer, rows)
    before = _failures("transient")._value.get()

    with pytest.raises(RuntimeError):
        await consumer.flush_pending_batches()

    assert _failures("transient")._value.get() == before + 1
    assert consumer.batch_processors["queries"] == rows
    consumer.dlq_producer.send_and_wait.assert_not_awaited()


@pytest.mark.asyncio
async def test_consume_stage_dlq_drops_query_and_response_text():
    consumer = _consumer()
    event = {
        "event_type": "inference.completed",
        "request_id": "r1",
        "query_text": "secret question",
        "query": "secret question",
        "response_text": "secret answer",
        "context": "secret thread",
    }
    message = SimpleNamespace(
        value=json.dumps(event).encode(), key=None, topic="ic", partition=0, offset=3
    )

    await consumer._publish_dead_letter(
        "inference_completed", message, ValueError("bad"), stage="consume"
    )

    payload = consumer.dlq_producer.send_and_wait.await_args.kwargs["value"]
    assert payload["payload"]["request_id"] == "r1"
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_dict_rows_dead_letter_with_coordinates_and_ids():
    consumer = _consumer()
    row = {
        "request_id": "r-9",
        "query_text": "secret question",
        "created_at": datetime(2026, 4, 19, tzinfo=timezone.utc),
        "kafka_topic": "chat.requests",
        "kafka_partition": 2,
        "kafka_offset": 41,
    }

    assert await consumer._dead_letter_row("queries", row, DataError("bad"))

    kwargs = consumer.dlq_producer.send_and_wait.await_args.kwargs
    assert kwargs["key"] == "chat.requests:2:41"
    assert kwargs["topic"] == "chat.requests.dlq"
    payload = kwargs["value"]
    assert payload["source_offset"] == 41
    assert payload["payload"]["request_id"] == "r-9"
    assert payload["payload"]["row_type"] == "dict"
    assert payload["payload"]["created_at"].startswith("2026-04-19")
    assert "secret" not in json.dumps(payload)
