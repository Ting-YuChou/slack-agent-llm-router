#!/usr/bin/env python3
"""Run machine-readable ClickHouse and Flink analytics performance contracts.

The ClickHouse workload is destructive only inside uniquely named benchmark
databases, which are dropped by default. The Flink workload executes the real
RollingScopePolicyEmitter against a deterministic state/timer harness; it is not
a substitute for backend-specific RocksDB/checkpoint metrics.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import threading
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
    "dashboard_cold_p95_improvement_ratio": 0.40,
    "dashboard_query_count": 4,
    "flink_events": 10_000,
    "state_bytes_reduction_ratio": 0.80,
    "throughput_improvement_ratio": 5.0,
    "emits_per_scope": 2,
}
MIN_DASHBOARD_LATENCY_SAMPLES = 20


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
                    "clickhouse.dashboard_cold_p95_improvement",
                    values.get("dashboard_cold_p95_improvement_ratio"),
                    ">=",
                    limits["dashboard_cold_p95_improvement_ratio"],
                ),
                _gate(
                    "clickhouse.dashboard_query_count",
                    values.get("dashboard_query_count"),
                    "<=",
                    limits["dashboard_query_count"],
                ),
                _gate(
                    "clickhouse.model_performance_seed_rows",
                    values.get("model_performance_seed_rows"),
                    "==",
                    1,
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


def _legacy_event(sequence: int) -> Dict[str, Any]:
    return {
        "event_timestamp_ms": min(sequence, 4_999),
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
        "route_to_fast_lane": False,
    }


def _legacy_hot_key(events: int) -> Tuple[int, int]:
    from flink.logic import build_routing_policy_state_event

    serialized_state: List[str] = []
    checksum = 0
    for sequence in range(events):
        event = _legacy_event(sequence)
        cutoff_ms = int(event["event_timestamp_ms"]) - 300_000
        # Match the old ListState hot path: deserialize every retained event,
        # prune the rolling window, append, then rewrite the complete state.
        state = [
            retained
            for retained in (json.loads(item) for item in serialized_state)
            if int(retained.get("event_timestamp_ms", 0) or 0) >= cutoff_ms
        ]
        state.append(event)
        serialized_state = [json.dumps(event) for event in state]
        build_routing_policy_state_event(
            scope_type="user",
            scope_key="benchmark-user",
            events=state,
            window_size_seconds=300,
        )
        checksum = sum(int(item.get("total_tokens", 0) or 0) for item in state)
    state_bytes = sum(len(item.encode("utf-8")) for item in serialized_state)
    return checksum, state_bytes


def _legacy_final_state(events: int) -> Tuple[int, int]:
    serialized_state = [
        json.dumps(_legacy_event(sequence)) for sequence in range(events)
    ]
    return events * 200, sum(len(item.encode("utf-8")) for item in serialized_state)


def _cumulative_list_state_work(events: int) -> int:
    return events * (events + 1) // 2


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
    legacy_measured_events = min(events, 1_000)
    measured_work = _cumulative_list_state_work(legacy_measured_events)
    requested_work = _cumulative_list_state_work(events)
    legacy_extrapolation_factor = (
        requested_work / measured_work if measured_work else 1.0
    )
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        _legacy_hot_key(legacy_measured_events)
        legacy_times.append(
            (time.perf_counter() - started) * legacy_extrapolation_factor
        )

        started = time.perf_counter()
        (
            incremental_checksum,
            incremental_bytes,
            emits,
        ) = _run_actual_rolling_scope_emitter(events)
        incremental_times.append(time.perf_counter() - started)

    legacy_checksum, legacy_bytes = _legacy_final_state(events)

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
        "legacy_measured_events": legacy_measured_events,
        "legacy_extrapolation_factor": legacy_extrapolation_factor,
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
            "Executes production process_element/on_timer for every requested event. "
            "The legacy JSON ListState and policy path is measured for at most 1,000 "
            "events and extrapolated by cumulative retained-entry work; neither path "
            "measures a RocksDB backend or checkpoint bytes."
        ),
        "actual_operator_events": events,
        "errors": errors,
    }


class _MeasuredDashboardQueries:
    """Count actual dashboard query calls issued by the benchmark."""

    def __init__(self, query):
        self._query = query
        self.count = 0
        self._lock = threading.Lock()

    def query(self, sql: str, *args, **kwargs):
        with self._lock:
            self.count += 1
        return self._query(sql, *args, **kwargs)


def _run_production_dashboard_bundle(
    client, database: str
) -> Tuple[int, Dict[str, Any]]:
    """Count queries issued by the production dashboard implementation."""
    from src.llm_router_part3_pipeline import KafkaIngestionPipeline

    pipeline = KafkaIngestionPipeline(
        {
            "clickhouse": {
                "database": database,
                "dashboard": {
                    "cache_ttl_seconds": 0,
                    "cache_max_entries": 1,
                    "max_concurrent_queries": 4,
                },
            }
        }
    )
    measured = _MeasuredDashboardQueries(client.query)
    pipeline.clickhouse_manager.client = measured
    bundle = asyncio.run(pipeline.get_dashboard_bundle(hours=1))
    return measured.count, bundle


def legacy_dashboard_queries(database: str, *, hours: int = 1) -> Dict[str, str]:
    """Return the six queries issued by the pre-bundle sequential dashboard flow."""
    window = f"timestamp >= now() - INTERVAL {max(int(hours), 0)} HOUR"
    return {
        "analytics_overall": f"""
            SELECT count(), sum(token_count_input + token_count_output),
                   sum(cost_usd), avg(latency_ms),
                   countIf(status = 'success') * 100.0 / count()
            FROM {database}.query_logs_legacy WHERE {window}
        """,
        "analytics_model_breakdown": f"""
            SELECT selected_model, count(), sum(cost_usd)
            FROM {database}.query_logs_legacy WHERE {window}
            GROUP BY selected_model
        """,
        "analytics_query_type_breakdown": f"""
            SELECT query_type, count()
            FROM {database}.query_logs_legacy WHERE {window}
            GROUP BY query_type
        """,
        "model_performance": f"""
            SELECT model_name, provider, sum(requests_count),
                   sum(success_count) * 100.0 / nullIf(sum(requests_count), 0),
                   sum(avg_latency_ms * requests_count) /
                       nullIf(sum(requests_count), 0),
                   sum(avg_tokens_per_second * requests_count) /
                       nullIf(sum(requests_count), 0),
                   sum(error_count), sum(total_cost_usd)
            FROM {database}.model_performance WHERE {window}
            GROUP BY model_name, provider
        """,
        "routing_guardrails": f"""
            SELECT timestamp, alert_type, severity, description, model_name,
                   provider, payload_json
            FROM {database}.alert_events WHERE {window}
              AND source_event_type = 'routing.guardrails'
            ORDER BY timestamp DESC LIMIT 50
        """,
        "routing_policy_state": f"""
            SELECT timestamp, scope_type, scope_key, user_id, session_id,
                   user_tier, hint_reason, recent_request_count,
                   recent_error_rate, avg_latency_ms, fast_lane_hit_rate,
                   dominant_query_type, query_complexity, requires_low_latency,
                   requires_high_reasoning, route_to_fast_lane,
                   burst_protection_active, enterprise_priority_active,
                   preferred_models, avoid_models, avoid_providers, payload_json
            FROM {database}.routing_policy_state_events WHERE {window}
            ORDER BY timestamp DESC LIMIT 50
        """,
    }


def _drop_dashboard_caches(client) -> None:
    """Apply the same ClickHouse cold-cache treatment before every sample."""
    client.command("SYSTEM DROP MARK CACHE")
    client.command("SYSTEM DROP UNCOMPRESSED CACHE")


def _time_legacy_dashboard_bundle(client, database: str) -> Tuple[int, float]:
    """Time the complete six-query legacy dashboard path sequentially."""
    _drop_dashboard_caches(client)
    queries = legacy_dashboard_queries(database)
    started = time.perf_counter()
    for sql in queries.values():
        client.query(sql)
    return len(queries), time.perf_counter() - started


def _time_production_dashboard_bundle(
    client, database: str
) -> Tuple[int, Dict[str, Any], float]:
    """Time the cache-disabled production bundle, including thread coordination."""
    _drop_dashboard_caches(client)
    started = time.perf_counter()
    count, bundle = _run_production_dashboard_bundle(client, database)
    return count, bundle, time.perf_counter() - started


def benchmark_dashboard_bundle_latency(
    client, database: str, *, repeats: int
) -> Dict[str, Any]:
    """Compare full legacy and production dashboard bundles with cold caches."""
    sample_count = max(MIN_DASHBOARD_LATENCY_SAMPLES, int(repeats))
    legacy_durations: List[float] = []
    production_durations: List[float] = []
    production_query_count = 0

    for sample_index in range(sample_count):
        if sample_index % 2 == 0:
            legacy_query_count, legacy_duration = _time_legacy_dashboard_bundle(
                client, database
            )
            (
                production_query_count,
                bundle,
                production_duration,
            ) = _time_production_dashboard_bundle(client, database)
        else:
            (
                production_query_count,
                bundle,
                production_duration,
            ) = _time_production_dashboard_bundle(client, database)
            legacy_query_count, legacy_duration = _time_legacy_dashboard_bundle(
                client, database
            )
        if legacy_query_count != 6:
            raise RuntimeError(
                f"legacy dashboard issued {legacy_query_count} queries instead of 6"
            )
        if production_query_count != 4:
            raise RuntimeError(
                "production dashboard must issue all four independent sections; "
                f"observed {production_query_count} queries"
            )
        if bundle.get("errors"):
            raise RuntimeError(
                f"production dashboard bundle errors: {bundle['errors']}"
            )
        legacy_durations.append(legacy_duration)
        production_durations.append(production_duration)

    if (
        len(legacy_durations) != sample_count
        or len(production_durations) != sample_count
        or not all(
            _finite_number(value) and value > 0
            for value in legacy_durations + production_durations
        )
    ):
        raise RuntimeError("dashboard bundle timing samples are missing or invalid")

    legacy_p95 = _p95(legacy_durations)
    production_p95 = _p95(production_durations)
    return {
        "dashboard_latency_samples": sample_count,
        "legacy_dashboard_query_count": 6,
        "dashboard_query_count": production_query_count,
        "legacy_dashboard_cold_p95_seconds": legacy_p95,
        "v2_dashboard_cold_p95_seconds": production_p95,
        "dashboard_cold_p95_improvement_ratio": 1.0 - production_p95 / legacy_p95,
        "dashboard_latency_measurement_mode": (
            "legacy_sequential_vs_production_get_dashboard_bundle_real_clickhouse"
        ),
    }


def _create_production_dashboard_tables(client, database: str) -> None:
    """Create the exact table schemas used by production dashboard methods."""
    from src.llm_router_part3_pipeline import ClickHouseManager

    manager = ClickHouseManager({"database": database})
    manager.client = client
    asyncio.run(manager._create_tables())


def seed_model_performance(client, database: str) -> int:
    """Seed and verify the non-empty section assumed by the six-query baseline."""
    event_id = "analytics-contract:model-performance"
    client.command(
        f"""
        INSERT INTO {database}.model_performance
            (event_id, timestamp, model_name, provider, requests_count,
             success_count, success_rate, avg_latency_ms,
             avg_tokens_per_second, error_count, total_cost_usd,
             gpu_utilization, memory_usage_gb)
        SELECT '{event_id}', now64(3), 'gpt-5', 'openai', 100,
               99, 99.0, 125.0, 40.0, 1, 0.25, 0.0, 0.0
        """
    )
    count = int(
        clickhouse_scalar(
            client.query(
                f"SELECT count() FROM {database}.model_performance FINAL "
                f"WHERE event_id = '{event_id}'"
            )
        )
    )
    if count != 1:
        raise RuntimeError(
            "model_performance seed must produce exactly one visible row; "
            f"observed {count}"
        )
    return count


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


def legacy_query_logs_ddl(database: str, columns: str) -> str:
    return (
        f"CREATE TABLE {database}.query_logs_legacy ({columns}) "
        "ENGINE=ReplacingMergeTree PARTITION BY toYYYYMM(timestamp) "
        "ORDER BY event_id"
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
            autogenerate_session_id=False,
        )
        client.command(f"CREATE DATABASE {database} ENGINE=Atomic")
        client.command(f"CREATE DATABASE {migration_database} ENGINE=Atomic")
        _create_production_dashboard_tables(client, database)
        metrics["model_performance_seed_rows"] = seed_model_performance(
            client, database
        )
        common_columns = (
            "event_id String, timestamp DateTime64(3), user_id String, "
            "selected_model LowCardinality(String), query_type LowCardinality(String), "
            "token_count_input UInt32, token_count_output UInt32, latency_ms UInt32, "
            "cost_usd Float64, status LowCardinality(String), total_tokens UInt32"
        )
        client.command(legacy_query_logs_ddl(database, common_columns))
        client.command(
            f"CREATE TABLE {database}.query_logs_v2 ({common_columns}) "
            "ENGINE=ReplacingMergeTree PARTITION BY toYYYYMM(timestamp) "
            "ORDER BY (timestamp,event_id)"
        )
        generator = (
            f"SELECT {clickhouse_event_id_expression('number')}, "
            "now64(3) - INTERVAL 30 DAY + "
            f"toIntervalMillisecond(intDiv(number * 2592000000, {rows})), "
            "'benchmark-user', "
            "if(number % 2 = 0, 'gpt-5', 'claude-sonnet'), "
            "if(number % 3 = 0, 'analysis', 'chat'), "
            "toUInt32(25 + number % 250), toUInt32(75 + number % 1750), "
            "toUInt32(50 + number % 500), toFloat64(number % 1000) / 100000.0, "
            "'success', toUInt32(100 + number % 2000) "
            f"FROM numbers({rows})"
        )
        client.command(f"INSERT INTO {database}.query_logs_legacy {generator}")
        client.command(
            f"INSERT INTO {database}.query_logs_v2 SELECT * FROM {database}.query_logs_legacy"
        )
        client.command(
            f"INSERT INTO {database}.query_logs "
            "(event_id, query_id, timestamp, user_id, user_tier, query_text, "
            "query_type, selected_model, token_count_input, token_count_output, "
            "latency_ms, cost_usd, status, error_message, context_compressed, "
            "compression_ratio, cached_response) "
            f"SELECT event_id, event_id, timestamp, user_id, 'premium', '', "
            "query_type, selected_model, token_count_input, token_count_output, latency_ms, cost_usd, "
            f"'success', '', false, 1.0, false FROM {database}.query_logs_v2"
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

        window = "timestamp >= now64(3) - INTERVAL 1 HOUR AND timestamp < now64(3)"
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
        grouped_read_rows = 0
        for repeat in range(max(1, repeats)):
            current_rows = 0
            for index, sql in enumerate(legacy_queries):
                read_rows, elapsed = _query_read_rows(
                    client, sql, f"legacy-{repeat}-{index}"
                )
                current_rows += read_rows
            legacy_bundle_read_rows = max(legacy_bundle_read_rows, current_rows)
            read_rows, _ = _query_read_rows(
                client, legacy_grouped_query, f"legacy-grouped-{repeat}"
            )
            legacy_same_query_read_rows = max(legacy_same_query_read_rows, read_rows)
            read_rows, elapsed = _query_read_rows(
                client, v2_grouped_query, f"grouped-{repeat}"
            )
            grouped_read_rows = max(grouped_read_rows, read_rows)
        metrics.update(
            clickhouse_read_rows_metrics(
                legacy_same_query_rows=legacy_same_query_read_rows,
                v2_same_query_rows=grouped_read_rows,
                legacy_bundle_rows=legacy_bundle_read_rows,
            )
        )
        metrics.update(
            benchmark_dashboard_bundle_latency(client, database, repeats=repeats)
        )

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
