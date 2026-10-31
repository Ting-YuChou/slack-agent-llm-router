"""A12b: one declarative definition per ClickHouse table drives its CREATE,
its ALTERs and the keys its row builder may write."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analytics_schema import TABLES, Column, Table
from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    KafkaConsumerManager,
    MetricEntry,
    QueryLogEntry,
    build_chat_feedback_row,
    build_chat_request_row,
)

NOW = "2026-10-30T12:00:00+00:00"


def _consumer():
    return KafkaConsumerManager.__new__(KafkaConsumerManager)


def _query_log_row():
    return QueryLogEntry(
        query_id="q-1",
        timestamp=datetime.now(timezone.utc),
        user_id="U1",
        user_tier="free",
        query_text="hello",
        query_type="simple",
        selected_model="m",
        token_count_input=1,
        token_count_output=2,
        latency_ms=3,
        cost_usd=0.1,
        status="success",
    ).to_dict()


def _row_builders():
    consumer = _consumer()
    return {
        "query_logs": _query_log_row(),
        "system_metrics": MetricEntry(
            timestamp=datetime.now(timezone.utc),
            service="router",
            metric_name="latency",
            metric_value=1.0,
            labels={},
        ).to_dict(),
        "model_performance": consumer._build_model_performance_entry(
            {"model_name": "m", "window_start_ms": 1000, "window_end_ms": 2000}
        ).to_dict(),
        "alert_events": consumer._build_alert_event_entry(
            {"event_type": "alert", "emitted_at": NOW}
        ).to_dict(),
        "routing_policy_state_events": consumer._build_routing_policy_state_entry(
            {"scope_type": "user", "scope_key": "U1", "emitted_at": NOW}
        ).to_dict(),
        "chat_requests": build_chat_request_row(
            {"request_id": "r-1", "emitted_at": NOW}
        ),
        "chat_feedback": build_chat_feedback_row(
            {"request_id": "r-1", "rating": "up", "emitted_at": NOW}
        ),
        "routing_decisions": consumer._build_routing_decision_entry(
            {"request_id": "r-1", "decided_at": NOW}
        ).to_dict(),
    }


def test_tables_are_defined_in_creation_order():
    assert [table.name for table in TABLES] == [
        "query_logs",
        "system_metrics",
        "model_performance",
        "user_analytics",
        "alert_events",
        "routing_policy_state_events",
        "chat_requests",
        "chat_feedback",
        "routing_decisions",
    ]


@pytest.mark.parametrize("table_name", sorted(_row_builders()))
def test_row_builder_keys_match_the_table_columns(table_name):
    table = {table.name: table for table in TABLES}[table_name]
    row = _row_builders()[table_name]
    columns = {column.name: column for column in table.columns}

    assert set(row) - set(columns) == set()
    unfilled = [
        name
        for name, column in columns.items()
        if name not in row and column.default is None
    ]
    assert unfilled == []


def test_create_sql_lists_every_column_and_the_table_options():
    table = Table(
        name="demo",
        columns=(
            Column("ts", "DateTime64(3)"),
            Column("id", "String"),
            Column("note", "String", default="''"),
        ),
        engine="ReplacingMergeTree()",
        partition_by="toYYYYMMDD(ts)",
        order_by="(ts, id)",
        ttl="toDateTime(ts) + INTERVAL 7 DAY",
        settings="index_granularity = 8192",
    )

    sql = table.create_sql("db")

    assert sql.startswith("CREATE TABLE IF NOT EXISTS db.demo (")
    assert "note String DEFAULT ''" in sql
    assert "ENGINE = ReplacingMergeTree()" in sql
    assert "PARTITION BY toYYYYMMDD(ts)" in sql
    assert "ORDER BY (ts, id)" in sql
    assert "TTL toDateTime(ts) + INTERVAL 7 DAY" in sql
    assert "SETTINGS index_granularity = 8192" in sql


def test_alters_add_every_column_outside_the_sorting_and_partition_keys():
    table = Table(
        name="demo",
        columns=(
            Column("ts", "DateTime64(3)"),
            Column("id", "String", backfill="concat('x', toString(ts))"),
            Column("bucket", "String"),
            Column("note", "String", default="''"),
            Column("value", "UInt32"),
        ),
        engine="ReplacingMergeTree()",
        partition_by="toYYYYMM(ts)",
        order_by="(toStartOfHour(ts), bucket)",
    )

    assert table.alter_sql("db") == [
        # A key column only with an explicit backfill for older tables.
        "ALTER TABLE db.demo ADD COLUMN IF NOT EXISTS id String "
        "DEFAULT concat('x', toString(ts))",
        "ALTER TABLE db.demo ADD COLUMN IF NOT EXISTS note String DEFAULT ''",
        "ALTER TABLE db.demo ADD COLUMN IF NOT EXISTS value UInt32",
    ]


@pytest.mark.asyncio
async def test_create_tables_runs_the_generated_creates_then_the_alters():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [
        call.args[1]
        for call in manager._run_blocking.await_args_list
        if call.args[0] is manager.client.command
    ]
    # No column list came back, so every ALTER runs.
    expected = [table.create_sql("llm_router") for table in TABLES] + [
        sql for table in TABLES for sql in table.alter_sql("llm_router")
    ]
    assert statements == expected
    # Columns added after a table first shipped are still migrated.
    assert (
        "ALTER TABLE llm_router.chat_requests ADD COLUMN IF NOT EXISTS "
        "queue_wait_ms Nullable(UInt32)"
    ) in statements
    assert (
        "ALTER TABLE llm_router.query_logs ADD COLUMN IF NOT EXISTS "
        "event_id String DEFAULT query_id"
    ) in statements


def _manager_with_columns(existing, *, fail_alter=False):
    """A manager whose system.columns lists ``existing`` (table, column) pairs."""
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    calls = []

    async def run_blocking(func, sql, *args, **kwargs):
        calls.append((func, sql, kwargs))
        if func is manager.client.query:
            assert "system.columns" in sql
            assert kwargs["parameters"] == {"database": "llm_router"}
            return SimpleNamespace(result_rows=list(existing))
        if fail_alter and sql.startswith("ALTER"):
            raise RuntimeError("Code: 497. Not enough privileges")
        return None

    manager._run_blocking = run_blocking
    return manager, calls


def _all_columns():
    return [(table.name, column.name) for table in TABLES for column in table.columns]


@pytest.mark.asyncio
async def test_create_tables_alters_only_the_missing_columns():
    existing = [
        pair
        for pair in _all_columns()
        if pair not in {("chat_requests", "ttft_ms"), ("model_performance", "provider")}
    ]
    manager, calls = _manager_with_columns(existing)

    await manager._create_tables()

    alters = [sql for func, sql, _ in calls if sql.startswith("ALTER")]
    assert alters == [
        "ALTER TABLE llm_router.model_performance ADD COLUMN IF NOT EXISTS "
        "provider String DEFAULT ''",
        "ALTER TABLE llm_router.chat_requests ADD COLUMN IF NOT EXISTS "
        "ttft_ms Nullable(UInt32)",
    ]
    # One column lookup for the whole database.
    assert sum(1 for func, _, _ in calls if func is manager.client.query) == 1


@pytest.mark.asyncio
async def test_create_tables_issues_no_alters_when_nothing_is_missing():
    manager, calls = _manager_with_columns(_all_columns())

    await manager._create_tables()

    assert [sql for _, sql, _ in calls if sql.startswith("ALTER")] == []


@pytest.mark.asyncio
async def test_a_missing_column_alter_failure_is_still_fatal():
    existing = [
        pair for pair in _all_columns() if pair != ("query_logs", "kafka_offset")
    ]
    manager, _ = _manager_with_columns(existing, fail_alter=True)

    with pytest.raises(RuntimeError, match="schema migration.*query_logs"):
        await manager._create_tables()


@pytest.mark.asyncio
async def test_create_tables_issues_every_alter_when_the_column_lookup_fails():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    commands = []

    async def run_blocking(func, sql, *args, **kwargs):
        if func is manager.client.query:
            raise RuntimeError("no access to system.columns")
        commands.append(sql)

    manager._run_blocking = run_blocking

    await manager._create_tables()

    assert commands[len(TABLES) :] == [
        sql for table in TABLES for sql in table.alter_sql("llm_router")
    ]


# Every ALTER fc64233 issued, so older deployments still migrate.
FC64233_ALTERS = [
    "ALTER TABLE test_db.query_logs ADD COLUMN IF NOT EXISTS event_id String DEFAULT query_id",
    "ALTER TABLE test_db.query_logs ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
    "ALTER TABLE test_db.query_logs ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
    "ALTER TABLE test_db.query_logs ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
    "ALTER TABLE test_db.system_metrics ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(service, ':', metric_name, ':', toString(timestamp))",
    "ALTER TABLE test_db.system_metrics ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
    "ALTER TABLE test_db.system_metrics ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
    "ALTER TABLE test_db.system_metrics ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(model_name, ':', toString(timestamp))",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS provider String DEFAULT ''",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS window_start_ms UInt64 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS window_end_ms UInt64 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS window_size_seconds UInt32 DEFAULT 60",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS success_count UInt32 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS total_tokens UInt32 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS queries_per_second Float32 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS cache_hit_rate Float32 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS cached_count UInt32 DEFAULT 0",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
    "ALTER TABLE test_db.model_performance ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
    "ALTER TABLE test_db.alert_events ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(source_event_type, ':', alert_type, ':', request_id, ':', query_id, ':', model_name, ':', provider, ':', toString(timestamp))",
    "ALTER TABLE test_db.alert_events ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
    "ALTER TABLE test_db.alert_events ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
    "ALTER TABLE test_db.alert_events ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
    "ALTER TABLE test_db.routing_policy_state_events ADD COLUMN IF NOT EXISTS event_id String DEFAULT concat(scope_type, ':', scope_key, ':', toString(timestamp))",
    "ALTER TABLE test_db.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_topic String DEFAULT ''",
    "ALTER TABLE test_db.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_partition Int32 DEFAULT -1",
    "ALTER TABLE test_db.routing_policy_state_events ADD COLUMN IF NOT EXISTS kafka_offset Int64 DEFAULT -1",
    "ALTER TABLE test_db.routing_decisions ADD COLUMN IF NOT EXISTS fallback_from_model String DEFAULT ''",
]


def _normalized(sql):
    return " ".join(sql.split())


def test_every_alter_from_before_the_generated_schema_is_still_issued():
    generated = {
        _normalized(sql) for table in TABLES for sql in table.alter_sql("test_db")
    }

    missing = [sql for sql in FC64233_ALTERS if _normalized(sql) not in generated]
    assert missing == []
