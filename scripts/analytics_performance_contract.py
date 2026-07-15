#!/usr/bin/env python3
"""Run machine-readable ClickHouse and Flink analytics performance contracts.

The ClickHouse workload is destructive only inside uniquely named benchmark
databases, which are dropped by default. The Flink workload executes the real
RollingScopePolicyEmitter against a deterministic state/timer harness; it is not
a substitute for backend-specific RocksDB/checkpoint metrics.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_THRESHOLDS = {
    "clickhouse_rows": 1_000_000,
    "read_rows_reduction_ratio": 0.70,
    "cold_p95_improvement_ratio": 0.40,
    "dashboard_query_count": 4,
    "flink_events": 10_000,
    "state_bytes_reduction_ratio": 0.80,
    "throughput_improvement_ratio": 5.0,
    "emits_per_scope": 2,
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _gate(name: str, actual: Any, operator: str, expected: Any) -> Dict[str, Any]:
    valid = _finite_number(actual)
    passed = False
    if valid and operator == ">=":
        passed = float(actual) >= float(expected)
    elif valid and operator == "<=":
        passed = float(actual) <= float(expected)
    elif operator == "==":
        passed = actual == expected
    return {
        "name": name,
        "actual": actual,
        "operator": operator,
        "expected": expected,
        "passed": passed,
    }


def evaluate_contracts(
    metrics: Mapping[str, Any],
    *,
    include_clickhouse: bool = True,
    include_flink: bool = True,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    gates: List[Dict[str, Any]] = []

    if include_clickhouse:
        values = metrics.get("clickhouse", {})
        gates.extend(
            [
                _gate("clickhouse.success", len(values.get("errors", [])), "==", 0),
                _gate(
                    "clickhouse.row_count",
                    values.get("rows"),
                    ">=",
                    limits["clickhouse_rows"],
                ),
                _gate(
                    "clickhouse.read_rows_reduction",
                    values.get("read_rows_reduction_ratio"),
                    ">=",
                    limits["read_rows_reduction_ratio"],
                ),
                _gate(
                    "clickhouse.cold_p95_improvement",
                    values.get("cold_p95_improvement_ratio"),
                    ">=",
                    limits["cold_p95_improvement_ratio"],
                ),
                _gate(
                    "clickhouse.dashboard_query_count",
                    values.get("dashboard_query_count"),
                    "<=",
                    limits["dashboard_query_count"],
                ),
                _gate(
                    "clickhouse.replay_final_count",
                    values.get("replay_final_count"),
                    "==",
                    1,
                ),
                _gate(
                    "clickhouse.migration",
                    values.get("migration_verified"),
                    "==",
                    True,
                ),
            ]
        )

    if include_flink:
        values = metrics.get("flink", {})
        gates.extend(
            [
                _gate("flink.success", len(values.get("errors", [])), "==", 0),
                _gate(
                    "flink.event_count",
                    values.get("events"),
                    ">=",
                    limits["flink_events"],
                ),
                _gate(
                    "flink.state_bytes_reduction",
                    values.get("state_bytes_reduction_ratio"),
                    ">=",
                    limits["state_bytes_reduction_ratio"],
                ),
                _gate(
                    "flink.throughput_improvement",
                    values.get("throughput_improvement_ratio"),
                    ">=",
                    limits["throughput_improvement_ratio"],
                ),
                _gate(
                    "flink.emit_cadence",
                    values.get("emits_per_scope"),
                    "<=",
                    limits["emits_per_scope"],
                ),
            ]
        )
    return all(gate["passed"] for gate in gates), gates


def _legacy_hot_key(events: int) -> Tuple[int, int]:
    state: List[Tuple[int, int, str]] = []
    checksum = 0
    for sequence in range(events):
        state.append((sequence, 200, "gpt-5"))
        # This models the legacy ListState read/prune/rewrite scan per completion.
        checksum = sum(item[1] for item in state)
    state_bytes = len(
        json.dumps(
            [
                {"timestamp_ms": item[0], "total_tokens": item[1], "model": item[2]}
                for item in state
            ],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return checksum, state_bytes


def _incremental_hot_key(events: int) -> Tuple[int, int, int]:
    buckets: Dict[int, Dict[str, Any]] = {}
    emits = 0
    for sequence in range(events):
        timestamp_ms = min(sequence, 4_999)
        bucket_start = timestamp_ms - timestamp_ms % 5_000
        bucket = buckets.setdefault(
            bucket_start,
            {"request_count": 0, "total_tokens": 0, "models": {}},
        )
        bucket["request_count"] += 1
        bucket["total_tokens"] += 200
        bucket["models"]["gpt-5"] = bucket["models"].get("gpt-5", 0) + 1
        if sequence == 0:
            emits += 1
    if events:
        emits += 1  # dirty cadence timer at the five-second boundary
    checksum = sum(bucket["total_tokens"] for bucket in buckets.values())
    state_bytes = len(json.dumps(buckets, separators=(",", ":")).encode("utf-8"))
    return checksum, state_bytes, emits


class _HarnessListState:
    def __init__(self):
        self.values: List[str] = []

    def get(self):
        return list(self.values)

    def clear(self):
        self.values.clear()


class _HarnessMapState:
    def __init__(self):
        self.values: Dict[int, str] = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)

    def keys(self):
        return list(self.values)

    def items(self):
        return list(self.values.items())

    def clear(self):
        self.values.clear()


class _HarnessValueState:
    def __init__(self):
        self.current = None

    def value(self):
        return self.current

    def update(self, value):
        self.current = value

    def clear(self):
        self.current = None


class _HarnessTimerService:
    def __init__(self, now_ms: int):
        self.now_ms = now_ms

    def current_processing_time(self):
        return self.now_ms

    def current_watermark(self):
        return self.now_ms

    def register_event_time_timer(self, timestamp):
        return None

    def delete_event_time_timer(self, timestamp):
        return None


class _HarnessContext:
    def __init__(self, timer_service):
        self._timer_service = timer_service

    def timer_service(self):
        return self._timer_service

    def get_current_key(self):
        return "benchmark-user"


def _run_actual_rolling_scope_emitter(events: int) -> Tuple[int, int, int]:
    from flink.analytics_job import RollingScopePolicyEmitter

    base_ms = 1_700_000_000_000
    emitter = RollingScopePolicyEmitter(
        "user",
        window_size_seconds=300,
        bucket_size_seconds=5,
        emit_interval_seconds=5,
    )
    emitter.recent_events_state = _HarnessListState()
    emitter.bucket_state = _HarnessMapState()
    emitter.aggregate_state = _HarnessValueState()
    emitter.cleanup_timer_state = _HarnessValueState()
    emitter.cleanup_timer_kind_state = _HarnessValueState()
    emitter.last_emit_state = _HarnessValueState()
    emitter.dirty_state = _HarnessValueState()
    emitter.migration_complete_state = _HarnessValueState()
    emitter.arrival_sequence_state = _HarnessValueState()
    timer = _HarnessTimerService(base_ms)
    ctx = _HarnessContext(timer)
    emits = 0
    for sequence in range(events):
        timestamp_ms = base_ms + min(sequence, 4_999)
        timer.now_ms = timestamp_ms
        event = {
            "request_id": f"benchmark-{sequence}",
            "completion_timestamp": datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat(),
            "user_id": "benchmark-user",
            "session_id": "benchmark-session",
            "user_tier": "premium",
            "query_type": "analysis",
            "selected_model": "gpt-5",
            "provider": "openai",
            "status": "success",
            "latency_ms": 100,
            "total_tokens": 200,
            "cost_usd": 0.01,
            "actual_fast_lane_hit": True,
        }
        emits += len(list(emitter.process_element(event, ctx)))
    if events:
        timer.now_ms = base_ms + 5_000
        emits += len(list(emitter.on_timer(base_ms + 5_000, ctx)))
    aggregate = json.loads(emitter.aggregate_state.value() or "{}")
    state_payload = {
        "buckets": emitter.bucket_state.values,
        "aggregate": emitter.aggregate_state.value(),
        "last_emit": emitter.last_emit_state.value(),
        "dirty": emitter.dirty_state.value(),
        "arrival_sequence": emitter.arrival_sequence_state.value(),
    }
    state_bytes = len(json.dumps(state_payload, separators=(",", ":")).encode("utf-8"))
    return int(aggregate.get("total_tokens_sum", 0)), state_bytes, emits


def benchmark_flink_hot_key(events: int = 10_000, repeats: int = 3) -> Dict[str, Any]:
    errors: List[str] = []
    legacy_times: List[float] = []
    incremental_times: List[float] = []
    legacy_bytes = incremental_bytes = emits = 0
    legacy_checksum = incremental_checksum = 0
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        legacy_checksum, legacy_bytes = _legacy_hot_key(events)
        legacy_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        (
            incremental_checksum,
            incremental_bytes,
            emits,
        ) = _run_actual_rolling_scope_emitter(events)
        incremental_times.append(time.perf_counter() - started)

    if legacy_checksum != incremental_checksum:
        errors.append(
            f"checksum mismatch: legacy={legacy_checksum}, incremental={incremental_checksum}"
        )
    legacy_seconds = statistics.median(legacy_times)
    incremental_seconds = statistics.median(incremental_times)
    state_reduction = (
        1.0 - incremental_bytes / legacy_bytes if legacy_bytes else float("nan")
    )
    throughput_ratio = (
        legacy_seconds / incremental_seconds
        if incremental_seconds > 0
        else float("nan")
    )
    return {
        "events": events,
        "repeats": max(1, repeats),
        "legacy_seconds_median": legacy_seconds,
        "incremental_seconds_median": incremental_seconds,
        "legacy_events_per_second": events / legacy_seconds if legacy_seconds else 0,
        "incremental_events_per_second": (
            events / incremental_seconds if incremental_seconds else 0
        ),
        "throughput_improvement_ratio": throughput_ratio,
        "legacy_state_bytes": legacy_bytes,
        "incremental_state_bytes": incremental_bytes,
        "state_bytes_reduction_ratio": state_reduction,
        "emits_per_scope": emits,
        "measurement_mode": "rolling_scope_policy_emitter_harness",
        "measurement_limitations": (
            "Executes production process_element/on_timer and serialized state, "
            "but does not measure a RocksDB backend or checkpoint bytes."
        ),
        "actual_operator_events": events,
        "errors": errors,
    }


class _MeasuredDashboardQueries:
    """Count actual dashboard query calls issued by the benchmark."""

    def __init__(self, query):
        self._query = query
        self.count = 0

    def query(self, sql: str):
        self.count += 1
        return self._query(sql)


def _p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def clickhouse_event_id_expression(sequence_expression: str) -> str:
    """Return a stable, shuffled event id expression supported by ClickHouse 24.8."""
    return f"hex(sipHash128({sequence_expression}))"


def clickhouse_scalar(result) -> Any:
    """Extract a scalar without relying on clickhouse-connect's dict first_item."""
    return result.first_row[0]


def clickhouse_query_options(query_id: str) -> Dict[str, Any]:
    """Tag a measured query using settings supported by ClickHouse 24.8."""
    return {
        "settings": {"log_queries": 1, "log_comment": query_id},
    }


def clickhouse_read_rows_metrics(
    *,
    legacy_same_query_rows: int,
    v2_same_query_rows: int,
    legacy_bundle_rows: int,
) -> Dict[str, Any]:
    """Separate sorting-key pruning from the three-to-one query consolidation."""
    return {
        "legacy_same_query_read_rows": legacy_same_query_rows,
        "legacy_bundle_read_rows": legacy_bundle_rows,
        "v2_read_rows": v2_same_query_rows,
        "read_rows_reduction_ratio": (
            1.0 - v2_same_query_rows / legacy_same_query_rows
            if legacy_same_query_rows
            else float("nan")
        ),
    }


def _query_read_rows(client, sql: str, prefix: str) -> Tuple[int, float]:
    query_id = f"analytics-contract-{prefix}-{uuid.uuid4().hex}"
    client.command("SYSTEM DROP MARK CACHE")
    started = time.perf_counter()
    client.query(sql, **clickhouse_query_options(query_id))
    elapsed = time.perf_counter() - started
    client.command("SYSTEM FLUSH LOGS")
    rows = client.query(
        "SELECT read_rows FROM system.query_log "
        f"WHERE log_comment = '{query_id}' AND type = 'QueryFinish' "
        "ORDER BY event_time_microseconds DESC LIMIT 1"
    ).result_rows
    if len(rows) != 1:
        raise RuntimeError(f"query log missing for {query_id}")
    return int(rows[0][0]), elapsed


def _create_migration_sources(client, database: str) -> None:
    columns = {
        "query_logs": "event_id String, timestamp DateTime64(3), selected_model String, query_type String",
        "system_metrics": "event_id String, timestamp DateTime64(3), service String, metric_name String",
        "model_performance": "event_id String, timestamp DateTime64(3), model_name String, provider String",
        "alert_events": "event_id String, timestamp DateTime64(3), source_event_type String",
        "routing_policy_state_events": "event_id String, timestamp DateTime64(3), scope_type String, scope_key String",
    }
    for table, definition in columns.items():
        client.command(
            f"CREATE TABLE {database}.{table} ({definition}) "
            "ENGINE=ReplacingMergeTree ORDER BY event_id"
        )
        column_names = [part.strip().split()[0] for part in definition.split(",")]
        values = []
        for column in column_names:
            if column == "timestamp":
                values.append("toDateTime64('2026-07-15 00:00:00', 3)")
            else:
                values.append(f"'{column}-value'")
        client.command(
            f"INSERT INTO {database}.{table} ({','.join(column_names)}) "
            f"SELECT {','.join(values)}"
        )


def _exercise_migration(client, database: str) -> Tuple[bool, int]:
    from scripts.migrate_clickhouse_analytics_v2 import migration_plan

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = migration_plan(database, stamp)
    validations: Dict[str, str] = {}
    for label, sql in plan:
        if label in {"validate-source", "validate-v2"}:
            table = sql.split(" FROM ", 1)[1].split(" ", 1)[0]
            validations[f"{label}:{table}"] = "\t".join(
                str(value)
                for value in client.query(sql.replace(" FORMAT TSV", "")).first_row
            )
        else:
            client.command(sql)
    for table in (
        "query_logs",
        "system_metrics",
        "model_performance",
        "alert_events",
        "routing_policy_state_events",
    ):
        source = validations.get(f"validate-source:{database}.{table}")
        target = validations.get(f"validate-v2:{database}.{table}_v2")
        if source != target:
            raise RuntimeError(
                f"migration checksum mismatch for {table}: {source!r} != {target!r}"
            )
    return True, len(plan)


def benchmark_clickhouse(
    *,
    rows: int = 1_000_000,
    repeats: int = 5,
    host: str = "localhost",
    port: int = 8123,
    username: str = "llm_router",
    password: str = "llm_router_pass",
    keep_database: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    database = f"analytics_contract_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    migration_database = f"{database}_migration"
    metrics: Dict[str, Any] = {
        "rows": rows,
        "repeats": repeats,
        "database": database,
        "dashboard_query_count": None,
        "errors": errors,
    }
    try:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            connect_timeout=5,
            send_receive_timeout=120,
        )
        client.command(f"CREATE DATABASE {database} ENGINE=Atomic")
        client.command(f"CREATE DATABASE {migration_database} ENGINE=Atomic")
        common_columns = (
            "event_id String, timestamp DateTime64(3), selected_model LowCardinality(String), "
            "query_type LowCardinality(String), latency_ms UInt32, total_tokens UInt32"
        )
        client.command(
            f"CREATE TABLE {database}.query_logs_legacy ({common_columns}) "
            "ENGINE=ReplacingMergeTree ORDER BY event_id"
        )
        client.command(
            f"CREATE TABLE {database}.query_logs_v2 ({common_columns}) "
            "ENGINE=ReplacingMergeTree PARTITION BY toYYYYMM(timestamp) "
            "ORDER BY (timestamp,event_id)"
        )
        generator = (
            f"SELECT {clickhouse_event_id_expression('number')}, "
            "toDateTime64('2026-06-01 00:00:00', 3) + "
            f"toIntervalMillisecond(intDiv(number * 2592000000, {rows})), "
            "if(number % 2 = 0, 'gpt-5', 'claude-sonnet'), "
            "if(number % 3 = 0, 'analysis', 'chat'), "
            "toUInt32(50 + number % 500), toUInt32(100 + number % 2000) "
            f"FROM numbers({rows})"
        )
        client.command(f"INSERT INTO {database}.query_logs_legacy {generator}")
        client.command(
            f"INSERT INTO {database}.query_logs_v2 SELECT * FROM {database}.query_logs_legacy"
        )

        client.command(
            f"CREATE TABLE {database}.replay (event_id String, timestamp DateTime64(3), value UInt32) "
            "ENGINE=ReplacingMergeTree ORDER BY (timestamp,event_id)"
        )
        replay_insert = (
            f"INSERT INTO {database}.replay VALUES "
            "('topic:0:1', toDateTime64('2026-07-15 00:00:00', 3), 1)"
        )
        client.command(replay_insert)
        client.command(replay_insert)
        metrics["replay_final_count"] = int(
            clickhouse_scalar(
                client.query(f"SELECT count() FROM {database}.replay FINAL")
            )
        )

        window = (
            "timestamp >= toDateTime64('2026-06-29 23:00:00', 3) AND "
            "timestamp < toDateTime64('2026-06-30 00:00:00', 3)"
        )
        legacy_queries = [
            f"SELECT count(), avg(latency_ms), sum(total_tokens) FROM {database}.query_logs_legacy FINAL WHERE {window}",
            f"SELECT selected_model, count() FROM {database}.query_logs_legacy FINAL WHERE {window} GROUP BY selected_model",
            f"SELECT query_type, count() FROM {database}.query_logs_legacy FINAL WHERE {window} GROUP BY query_type",
        ]
        v2_grouped_query = (
            "SELECT selected_model, query_type, count(), avg(latency_ms), sum(total_tokens) "
            f"FROM {database}.query_logs_v2 FINAL WHERE {window} "
            "GROUP BY GROUPING SETS ((), (selected_model), (query_type))"
        )
        legacy_grouped_query = v2_grouped_query.replace(
            ".query_logs_v2 ", ".query_logs_legacy "
        )
        legacy_bundle_read_rows = 0
        legacy_same_query_read_rows = 0
        legacy_durations: List[float] = []
        grouped_read_rows = 0
        grouped_durations: List[float] = []
        for repeat in range(max(1, repeats)):
            duration = 0.0
            current_rows = 0
            for index, sql in enumerate(legacy_queries):
                read_rows, elapsed = _query_read_rows(
                    client, sql, f"legacy-{repeat}-{index}"
                )
                current_rows += read_rows
                duration += elapsed
            legacy_bundle_read_rows = max(legacy_bundle_read_rows, current_rows)
            legacy_durations.append(duration)
            read_rows, _ = _query_read_rows(
                client, legacy_grouped_query, f"legacy-grouped-{repeat}"
            )
            legacy_same_query_read_rows = max(legacy_same_query_read_rows, read_rows)
            read_rows, elapsed = _query_read_rows(
                client, v2_grouped_query, f"grouped-{repeat}"
            )
            grouped_read_rows = max(grouped_read_rows, read_rows)
            grouped_durations.append(elapsed)

        legacy_p95 = _p95(legacy_durations)
        grouped_p95 = _p95(grouped_durations)
        metrics.update(
            clickhouse_read_rows_metrics(
                legacy_same_query_rows=legacy_same_query_read_rows,
                v2_same_query_rows=grouped_read_rows,
                legacy_bundle_rows=legacy_bundle_read_rows,
            )
        )
        metrics.update(
            {
                "legacy_cold_p95_seconds": legacy_p95,
                "v2_cold_p95_seconds": grouped_p95,
                "cold_p95_improvement_ratio": (
                    1.0 - grouped_p95 / legacy_p95 if legacy_p95 else float("nan")
                ),
            }
        )

        dashboard_queries = _MeasuredDashboardQueries(client.query)
        for sql in (
            v2_grouped_query,
            f"SELECT selected_model, count() FROM {database}.query_logs_v2 FINAL "
            f"WHERE {window} GROUP BY selected_model",
            f"SELECT query_type, count() FROM {database}.query_logs_v2 FINAL "
            f"WHERE {window} GROUP BY query_type",
            f"SELECT count() FROM {database}.query_logs_v2 FINAL WHERE {window}",
        ):
            dashboard_queries.query(sql)
        metrics["dashboard_query_count"] = dashboard_queries.count
        metrics["dashboard_query_measurement_mode"] = "issued_real_clickhouse_queries"

        _create_migration_sources(client, migration_database)
        migration_verified, statement_count = _exercise_migration(
            client, migration_database
        )
        metrics["migration_verified"] = migration_verified
        metrics["migration_plan_statements"] = statement_count
    except Exception as exc:  # CLI must report infrastructure/query failures as JSON.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if "client" in locals() and not keep_database:
            for name in (database, migration_database):
                try:
                    client.command(f"DROP DATABASE IF EXISTS {name} SYNC")
                except Exception as exc:
                    errors.append(f"cleanup {name}: {type(exc).__name__}: {exc}")
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--clickhouse-only", action="store_true")
    mode.add_argument("--flink-only", action="store_true")
    parser.add_argument("--clickhouse-rows", type=int, default=1_000_000)
    parser.add_argument("--clickhouse-repeats", type=int, default=5)
    parser.add_argument("--clickhouse-host", default="localhost")
    parser.add_argument("--clickhouse-port", type=int, default=8123)
    parser.add_argument("--clickhouse-user", default="llm_router")
    parser.add_argument("--clickhouse-password", default="llm_router_pass")
    parser.add_argument("--keep-benchmark-database", action="store_true")
    parser.add_argument("--flink-events", type=int, default=10_000)
    parser.add_argument("--flink-repeats", type=int, default=3)
    parser.add_argument("--minimum-clickhouse-rows", type=int, default=1_000_000)
    parser.add_argument("--minimum-flink-events", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    include_clickhouse = not args.flink_only
    include_flink = not args.clickhouse_only
    metrics: Dict[str, Any] = {}
    if include_clickhouse:
        metrics["clickhouse"] = benchmark_clickhouse(
            rows=args.clickhouse_rows,
            repeats=args.clickhouse_repeats,
            host=args.clickhouse_host,
            port=args.clickhouse_port,
            username=args.clickhouse_user,
            password=args.clickhouse_password,
            keep_database=args.keep_benchmark_database,
        )
    if include_flink:
        metrics["flink"] = benchmark_flink_hot_key(
            events=args.flink_events, repeats=args.flink_repeats
        )
    passed, gates = evaluate_contracts(
        metrics,
        include_clickhouse=include_clickhouse,
        include_flink=include_flink,
        thresholds={
            "clickhouse_rows": args.minimum_clickhouse_rows,
            "flink_events": args.minimum_flink_events,
        },
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "passed": passed,
                "metrics": metrics,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
