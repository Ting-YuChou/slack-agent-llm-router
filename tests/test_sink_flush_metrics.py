"""DP1: a ClickHouse flush records how many rows it wrote and how long it took."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiokafka.structs import TopicPartition

import src.llm_router_part3_pipeline as pipeline
from src.llm_router_part3_pipeline import KafkaConsumerManager, QueryLogEntry
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import UserTier

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
        "consumer": {"group_id": "test-group"},
        "clickhouse": {"database": "test_db"},
    }


def _query_row(offset: int) -> QueryLogEntry:
    return QueryLogEntry(
        query_id=f"query-{offset}",
        timestamp=datetime.now(timezone.utc),
        user_id="user-1",
        user_tier=UserTier.PREMIUM,
        query_text="question",
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


def _consumer(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    consumer.dlq_producer = AsyncMock()
    for topic_key in ("queries", "metrics", "alerts"):
        consumer.consumers[topic_key] = AsyncMock()
    return consumer


def _buffer(consumer, rows):
    for row in rows:
        consumer.batch_processors["queries"].append(row)
        consumer.pending_commit_offsets["queries"][QUERY_TP] = row.kafka_offset + 1


def _histogram(metric, processor="queries"):
    """(observations, sum) of one labelled histogram."""
    count = total = 0.0
    for family in metric.collect():
        for sample in family.samples:
            if sample.labels.get("processor") != processor:
                continue
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return count, total


@pytest.mark.asyncio
async def test_a_successful_flush_records_its_size_and_duration(
    pipeline_config, monkeypatch
):
    consumer = _consumer(pipeline_config)
    clock = {"now": 100.0}

    async def insert(batch):
        clock["now"] += 2.5

    # Only the insert moves the clock, so only the insert can be what is timed.
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: clock["now"])
    consumer.clickhouse.batch_insert_query_logs = insert
    _buffer(consumer, [_query_row(0), _query_row(1), _query_row(2)])
    sizes_before = _histogram(PIPELINE_METRICS.batch_size)
    times_before = _histogram(PIPELINE_METRICS.batch_processing_time)

    await consumer.flush_pending_batches()

    sizes = _histogram(PIPELINE_METRICS.batch_size)
    times = _histogram(PIPELINE_METRICS.batch_processing_time)
    assert sizes == (sizes_before[0] + 1, sizes_before[1] + 3)
    assert times[0] == times_before[0] + 1
    assert times[1] == pytest.approx(times_before[1] + 2.5)
    assert consumer.batch_processors["queries"] == []


@pytest.mark.asyncio
async def test_a_failed_flush_records_neither_and_restores_its_rows(pipeline_config):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock(
        side_effect=RuntimeError("clickhouse unavailable")
    )
    rows = [_query_row(0), _query_row(1)]
    _buffer(consumer, rows)
    sizes_before = _histogram(PIPELINE_METRICS.batch_size)
    times_before = _histogram(PIPELINE_METRICS.batch_processing_time)

    with pytest.raises(RuntimeError):
        await consumer.flush_pending_batches()

    assert _histogram(PIPELINE_METRICS.batch_size) == sizes_before
    assert _histogram(PIPELINE_METRICS.batch_processing_time) == times_before
    assert consumer.batch_processors["queries"] == rows
    assert consumer.pending_commit_offsets["queries"] == {QUERY_TP: 2}


@pytest.mark.asyncio
async def test_an_empty_flush_records_nothing(pipeline_config):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock()
    sizes_before = _histogram(PIPELINE_METRICS.batch_size)

    await consumer.flush_pending_batches()

    assert _histogram(PIPELINE_METRICS.batch_size) == sizes_before
    consumer.clickhouse.batch_insert_query_logs.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_bisected_batch_is_not_recorded_as_one_insert(pipeline_config):
    consumer = _consumer(pipeline_config)

    async def insert(batch):
        # One poison row: the whole batch is rejected, then written in pieces.
        if any(row.query_id == "query-1" for row in batch):
            raise ValueError("bad row")

    consumer.clickhouse.batch_insert_query_logs = insert
    _buffer(consumer, [_query_row(0), _query_row(1), _query_row(2)])
    sizes_before = _histogram(PIPELINE_METRICS.batch_size)
    times_before = _histogram(PIPELINE_METRICS.batch_processing_time)

    await consumer.flush_pending_batches()

    assert consumer.batch_processors["queries"] == []
    assert _histogram(PIPELINE_METRICS.batch_size) == sizes_before
    assert _histogram(PIPELINE_METRICS.batch_processing_time) == times_before


@pytest.mark.asyncio
async def test_a_failing_metric_never_strands_a_written_batch(
    pipeline_config, monkeypatch
):
    consumer = _consumer(pipeline_config)
    consumer.clickhouse.batch_insert_query_logs = AsyncMock()
    _buffer(consumer, [_query_row(0), _query_row(1)])

    def broken(**labels):
        raise ValueError("unknown label")

    monkeypatch.setattr(PIPELINE_METRICS.batch_size, "labels", broken)

    await consumer.flush_pending_batches()

    # The rows are in ClickHouse, so their offsets are committed and nothing
    # is left counted as in flight.
    assert consumer.batch_buffer_states["queries"].inflight_rows == 0
    assert consumer.pending_commit_offsets["queries"] == {}
    assert consumer.awaiting_commit_offsets["queries"] == {}
    consumer.consumers["queries"].commit.assert_awaited()


def test_flush_histograms_resolve_fast_inserts_and_full_buffers():
    def bounds(metric):
        return sorted(
            float(sample.labels["le"])
            for family in metric.collect()
            for sample in family.samples
            if sample.name.endswith("_bucket")
            and sample.labels.get("processor") == "queries"
        )

    PIPELINE_METRICS.batch_size.labels(processor="queries")
    PIPELINE_METRICS.batch_processing_time.labels(processor="queries")
    times = bounds(PIPELINE_METRICS.batch_processing_time)
    sizes = bounds(PIPELINE_METRICS.batch_size)
    # A normal insert takes 10-50 ms; a catch-up flush holds up to 5000 rows.
    assert [bound for bound in times if bound < 0.1] == [0.005, 0.01, 0.025, 0.05]
    assert 5000.0 in sizes and 2500.0 in sizes
