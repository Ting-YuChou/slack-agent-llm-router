"""A5: one undeliverable row must not stall a ClickHouse sink topic.

The real driver's exceptions live in ``clickhouse_connect.driver.exceptions``,
which CI does not install, so these tests raise look-alike classes with the
same names; the pipeline classifies errors by class name.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import (
    AlertEventEntry,
    ClickHouseManager,
    KafkaConsumerManager,
    QueryLogEntry,
)
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import UserTier


class DatabaseError(Exception):
    pass


class DataError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


QUERY_TP = TopicPartition("test-queries", 0)


@pytest.fixture
def pipeline_config():
    return {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {
            "queries": "test-queries",
            "metrics": "test-metrics",
            "alerts": "alerts",
        },
        "consumer": {
            "group_id": "test-group",
            "flush_retry_initial_backoff_seconds": 2,
            "flush_retry_max_backoff_seconds": 8,
        },
        "clickhouse": {"database": "test_db"},
    }


def _query_row(offset: int, *, bad: bool = False) -> QueryLogEntry:
    return QueryLogEntry(
        query_id=f"query-{offset}",
        timestamp=datetime.now(timezone.utc),
        user_id="user-1",
        user_tier="bogus-tier" if bad else UserTier.PREMIUM,
        query_text="secret question",
        query_type="general",
        selected_model="gpt-5",
        token_count_input=1,
        token_count_output=2,
        latency_ms=10,
        cost_usd=0.01,
        status="success",
        event_id=f"query-{offset}",
        kafka_topic="test-queries",
        kafka_partition=0,
        kafka_offset=offset,
    )


def _consumer(pipeline_config, *, dlq=True):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    consumer.dlq_producer = AsyncMock() if dlq else None
    for topic_key in ("queries", "metrics", "alerts"):
        consumer.consumers[topic_key] = AsyncMock()
    return consumer


def _buffer(consumer, rows, topic_key="queries", tp=QUERY_TP):
    for row in rows:
        consumer.batch_processors[topic_key].append(row)
        consumer.pending_commit_offsets[topic_key][tp] = row.kafka_offset + 1


def _rejecting_insert(error_factory, inserted):
    async def insert(batch):
        if any(row.user_tier == "bogus-tier" for row in batch):
            raise error_factory()
        inserted.extend(batch)

    return insert


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: DataError("Failed to write column 'user_tier'"),
        lambda: DatabaseError(
            "received ClickHouse error code 691\n Code: 691. DB::Exception: "
            "Unknown element 'bogus-tier' for enum. (UNKNOWN_ELEMENT_OF_ENUM)"
        ),
    ],
    ids=["DataError", "DatabaseError-enum"],
)
@pytest.mark.asyncio
async def test_permanent_row_error_dead_letters_that_row_and_commits_rest(
    pipeline_config, error_factory
):
    consumer = _consumer(pipeline_config)
    inserted = []
    consumer.clickhouse.batch_insert_query_logs = _rejecting_insert(
        error_factory, inserted
    )
    rows = [_query_row(0), _query_row(1), _query_row(2, bad=True), _query_row(3)]
    _buffer(consumer, rows)

    await consumer.flush_pending_batches()

    assert sorted(row.kafka_offset for row in inserted) == [0, 1, 3]
    dlq = consumer.dlq_producer.send_and_wait.await_args_list
    assert len(dlq) == 1
    kwargs = dlq[0].kwargs
    assert kwargs["topic"] == "test-queries.dlq"
    assert kwargs["key"] == "test-queries:0:2"
    assert kwargs["value"]["failure_stage"] == "persist"
    assert kwargs["value"]["source_offset"] == 2
    assert kwargs["value"]["error_type"] == type(error_factory()).__name__
    consumer.consumers["queries"].commit.assert_awaited_once_with({QUERY_TP: 4})
    assert consumer.batch_processors["queries"] == []
    assert consumer.pending_commit_offsets["queries"] == {}


@pytest.mark.asyncio
async def test_dead_lettered_row_payload_stays_private(pipeline_config):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = _rejecting_insert(
        lambda: DataError("bad"), []
    )
    _buffer(consumer, [_query_row(0, bad=True)])

    await consumer.flush_pending_batches()

    payload = consumer.dlq_producer.send_and_wait.await_args.kwargs["value"]
    assert "secret question" not in str(payload)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("clickhouse unavailable"),
        OperationalError("Error HTTPConnectionPool executing HTTP request"),
        DatabaseError("Code: 241. DB::Exception: Memory limit exceeded"),
    ],
    ids=["RuntimeError", "OperationalError", "DatabaseError-memory"],
)
@pytest.mark.asyncio
async def test_transient_error_keeps_batch_and_offsets_without_dlq(
    pipeline_config, error
):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(side_effect=error)
    rows = [_query_row(0), _query_row(1)]
    _buffer(consumer, rows)
    failures = PIPELINE_METRICS.clickhouse_flush_failures.labels(
        topic="queries", kind="transient"
    )
    before = failures._value.get()

    with pytest.raises(type(error)):
        await consumer.flush_pending_batches()

    consumer.dlq_producer.send_and_wait.assert_not_awaited()
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert consumer.batch_processors["queries"] == rows
    assert consumer.pending_commit_offsets["queries"] == {QUERY_TP: 2}
    assert consumer.clickhouse.batch_insert_query_logs.await_count == 1
    assert failures._value.get() == before + 1


@pytest.mark.asyncio
async def test_permanent_error_without_dlq_keeps_rows_and_offsets(pipeline_config):
    consumer = _consumer(pipeline_config, dlq=False)
    inserted = []
    consumer.clickhouse.batch_insert_query_logs = _rejecting_insert(
        lambda: DataError("bad"), inserted
    )
    bad = _query_row(1, bad=True)
    _buffer(consumer, [_query_row(0), bad])

    with pytest.raises(DataError):
        await consumer.flush_pending_batches()

    assert [row.kafka_offset for row in inserted] == [0]
    consumer.consumers["queries"].commit.assert_not_awaited()
    assert consumer.batch_processors["queries"] == [bad]
    assert consumer.pending_commit_offsets["queries"] == {QUERY_TP: 2}


@pytest.mark.asyncio
async def test_transient_failures_back_off_with_a_cap(pipeline_config):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(
        side_effect=RuntimeError("down")
    )
    _buffer(consumer, [_query_row(0)])
    delays = []
    with patch("src.llm_router_part3_pipeline.time.monotonic", return_value=100.0):
        for _ in range(4):
            with pytest.raises(RuntimeError):
                await consumer.flush_pending_batches()
            delays.append(consumer.batch_buffer_states["queries"].retry_delay_seconds)

    assert delays == [2, 4, 8, 8]
    assert consumer.batch_buffer_states["queries"].next_flush_at == 108.0
    assert (
        PIPELINE_METRICS.clickhouse_flush_retry_delay.labels(
            topic="queries"
        )._value.get()
        == 8
    )

    # The periodic flusher honours the backoff; explicit flushes do not.
    calls = consumer.clickhouse.batch_insert_query_logs.await_count
    with patch("src.llm_router_part3_pipeline.time.monotonic", return_value=104.0):
        await consumer.flush_pending_batches(respect_backoff=True)
    assert consumer.clickhouse.batch_insert_query_logs.await_count == calls

    consumer.clickhouse.batch_insert_query_logs = AsyncMock()
    with patch("src.llm_router_part3_pipeline.time.monotonic", return_value=109.0):
        await consumer.flush_pending_batches(respect_backoff=True)
    consumer.clickhouse.batch_insert_query_logs.assert_awaited_once()
    assert consumer.batch_buffer_states["queries"].retry_delay_seconds == 0
    assert consumer.batch_buffer_states["queries"].next_flush_at == 0.0


@pytest.mark.asyncio
async def test_poison_message_on_one_topic_is_not_blocked_by_another(
    pipeline_config,
):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(
        side_effect=RuntimeError("queries table down")
    )
    consumer.clickhouse.batch_insert_alert_events = AsyncMock()
    _buffer(consumer, [_query_row(0)])
    alert_tp = TopicPartition("alerts", 0)
    consumer.batch_processors["alerts"].append(
        AlertEventEntry(
            timestamp=datetime.now(timezone.utc),
            alert_type="x",
            severity="warning",
            description="earlier valid alert",
        )
    )
    consumer.pending_commit_offsets["alerts"][alert_tp] = 7
    poison = MagicMock(topic="alerts", partition=0, offset=7, value=b"{bad", key=None)

    await consumer._handle_processing_failure(
        "alerts", poison, ValueError("bad json"), stage="consume", batched=True
    )

    consumer.clickhouse.batch_insert_alert_events.assert_awaited_once()
    consumer.clickhouse.batch_insert_query_logs.assert_not_awaited()
    assert consumer.dlq_producer.send_and_wait.await_args.kwargs["key"] == (
        "alerts:0:7"
    )
    commits = consumer.consumers["alerts"].commit.await_args_list
    assert commits[-1].args[0] == {alert_tp: 8}
    assert consumer.batch_processors["queries"] != []


@pytest.mark.asyncio
async def test_poison_message_flush_failure_on_own_topic_still_raises(
    pipeline_config,
):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(
        side_effect=RuntimeError("queries table down")
    )
    _buffer(consumer, [_query_row(0)])
    poison = MagicMock(
        topic="test-queries", partition=0, offset=1, value=b"{", key=None
    )

    with pytest.raises(RuntimeError, match="queries table down"):
        await consumer._handle_processing_failure(
            "queries", poison, ValueError("bad"), stage="consume", batched=True
        )

    consumer.dlq_producer.send_and_wait.assert_not_awaited()
    consumer.consumers["queries"].commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_batched_topics_inherit_poison_handling(pipeline_config):
    consumer = _consumer(pipeline_config)
    inserted = []
    consumer._add_batched_topic("chat_example")
    consumer._batch_flush_targets = lambda: (
        ("chat_example", _rejecting_insert(lambda: DataError("bad"), inserted)),
    )
    consumer.consumers["chat_example"] = AsyncMock()
    tp = TopicPartition("test-queries", 0)
    _buffer(consumer, [_query_row(0), _query_row(1, bad=True)], "chat_example", tp)

    await consumer.flush_pending_batches()

    assert [row.kafka_offset for row in inserted] == [0]
    assert consumer.dlq_producer.send_and_wait.await_count == 1
    consumer.consumers["chat_example"].commit.assert_awaited_once_with({tp: 2})


@pytest.mark.asyncio
async def test_initialize_fails_when_schema_migration_fails(pipeline_config):
    with patch(
        "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
    ) as mock_get_client:
        client = MagicMock()
        client.query.return_value.result_rows = [[1]]

        def fail_alter(sql):
            if sql.startswith("ALTER TABLE test_db.query_logs ADD COLUMN"):
                raise RuntimeError("Code: 497. DB::Exception: Not enough privileges")
            return None

        client.command.side_effect = fail_alter
        mock_get_client.return_value = client
        manager = ClickHouseManager(pipeline_config["clickhouse"])

        with pytest.raises(RuntimeError, match="schema migration.*query_logs"):
            await manager.initialize()


CLICKHOUSE_HOST = os.environ.get("CHAT_TEST_CLICKHOUSE_HOST")


@pytest.mark.skipif(
    not CLICKHOUSE_HOST,
    reason="set CHAT_TEST_CLICKHOUSE_HOST to run against a real ClickHouse",
)
@pytest.mark.asyncio
async def test_real_driver_rejection_isolates_the_bad_row(pipeline_config):
    pytest.importorskip("clickhouse_connect")
    database = f"a5_poison_{uuid.uuid4().hex[:8]}"
    manager = ClickHouseManager(
        {
            "host": CLICKHOUSE_HOST,
            "port": int(os.environ.get("CHAT_TEST_CLICKHOUSE_PORT", "8123")),
            "username": os.environ.get("CHAT_TEST_CLICKHOUSE_USER", "default"),
            "password": os.environ.get("CHAT_TEST_CLICKHOUSE_PASSWORD", ""),
            "database": database,
        }
    )
    await manager.initialize()
    consumer = KafkaConsumerManager(pipeline_config, manager)
    consumer.dlq_producer = AsyncMock()
    consumer.consumers["queries"] = AsyncMock()
    bad = _query_row(1)
    bad.token_count_input = -5  # UInt32 column: the driver cannot pack it
    _buffer(consumer, [_query_row(0), bad, _query_row(2)])
    try:
        await consumer.flush_pending_batches()

        result = manager.client.query(
            f"SELECT kafka_offset FROM {database}.query_logs ORDER BY kafka_offset"
        )
        assert [row[0] for row in result.result_rows] == [0, 2]
        kwargs = consumer.dlq_producer.send_and_wait.await_args.kwargs
        assert kwargs["key"] == "test-queries:0:1"
        consumer.consumers["queries"].commit.assert_awaited_once_with({QUERY_TP: 3})
    finally:
        manager.client.command(f"DROP DATABASE IF EXISTS {database}")
