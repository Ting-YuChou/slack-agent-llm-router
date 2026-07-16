#!/usr/bin/env python3
"""Side-by-side ClickHouse analytics table migration (dry-run by default)."""

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TableSpec:
    partition_by: str
    order_by: str
    ttl: str


TABLES: Dict[str, TableSpec] = {
    "query_logs": TableSpec(
        "toYYYYMM(timestamp)",
        "(timestamp, event_id)",
        "toDateTime(timestamp) + INTERVAL 90 DAY",
    ),
    "system_metrics": TableSpec(
        "toYYYYMMDD(timestamp)",
        "(timestamp, service, metric_name, event_id)",
        "toDateTime(timestamp) + INTERVAL 30 DAY",
    ),
    "model_performance": TableSpec(
        "toYYYYMMDD(timestamp)",
        "(timestamp, model_name, provider, event_id)",
        "toDateTime(timestamp) + INTERVAL 60 DAY",
    ),
    "alert_events": TableSpec(
        "toYYYYMMDD(timestamp)",
        "(source_event_type, timestamp, event_id)",
        "toDateTime(timestamp) + INTERVAL 30 DAY",
    ),
    "routing_policy_state_events": TableSpec(
        "toYYYYMMDD(timestamp)",
        "(timestamp, scope_type, scope_key, event_id)",
        "toDateTime(timestamp) + INTERVAL 30 DAY",
    ),
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def qualified(database: str, table: str) -> str:
    if not IDENTIFIER.fullmatch(database) or not IDENTIFIER.fullmatch(table):
        raise ValueError(f"Unsafe ClickHouse identifier: {database}.{table}")
    return f"{database}.{table}"


def validation_query(database: str, table: str) -> str:
    name = qualified(database, table)
    return (
        "SELECT uniqExact(event_id), count(), toString(min(timestamp)), "
        "toString(max(timestamp)), groupBitXor(cityHash64(event_id)), "
        "groupBitXor(cityHash64(toJSONString(tuple(*)))) "
        f"FROM {name} FINAL FORMAT TSV"
    )


def migration_plan(database: str, stamp: str) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    for table, spec in TABLES.items():
        live = qualified(database, table)
        v2 = qualified(database, f"{table}_v2")
        backup = qualified(database, f"{table}_backup_{stamp}")
        plan.extend(
            [
                (
                    "create-v2",
                    f"CREATE TABLE IF NOT EXISTS {v2} AS {live} "
                    "ENGINE = ReplacingMergeTree() "
                    f"PARTITION BY {spec.partition_by} ORDER BY {spec.order_by} "
                    f"TTL {spec.ttl}",
                ),
                ("backfill", f"INSERT INTO {v2} SELECT * FROM {live}"),
                (
                    "delta-copy-after-pause-and-flush",
                    f"INSERT INTO {v2} SELECT * FROM {live} "
                    f"WHERE event_id NOT IN (SELECT event_id FROM {v2})",
                ),
                ("validate-source", validation_query(database, table)),
                ("validate-v2", validation_query(database, f"{table}_v2")),
                (
                    "cutover-atomic-exchange",
                    f"EXCHANGE TABLES {live} AND {v2}",
                ),
                ("rename-old-to-backup", f"RENAME TABLE {v2} TO {backup}"),
            ]
        )
    return plan


class ClickHouseRunner:
    def __init__(self, command: str):
        self.command = shlex.split(command)

    def query(self, sql: str, *, capture: bool = False) -> str:
        result = subprocess.run(
            [*self.command, "--query", sql],
            check=True,
            capture_output=capture,
            text=True,
        )
        return result.stdout.strip() if capture else ""


def run_hook(command: str) -> None:
    subprocess.run(shlex.split(command), check=True)


def _normalize_sorting_key(value: str) -> str:
    return re.sub(r"[`()\s]", "", value or "")


def table_sorting_key(
    runner: ClickHouseRunner, database: str, table: str
) -> Optional[str]:
    qualified(database, table)
    value = runner.query(
        "SELECT sorting_key FROM system.tables "
        f"WHERE database = '{database}' AND name = '{table}' FORMAT TSVRaw",
        capture=True,
    ).strip()
    return value or None


def table_cutover_state(
    runner: ClickHouseRunner, database: str, table: str, stamp: str
) -> str:
    desired = _normalize_sorting_key(TABLES[table].order_by)
    live_key = table_sorting_key(runner, database, table)
    v2_key = table_sorting_key(runner, database, f"{table}_v2")
    backup_key = table_sorting_key(runner, database, f"{table}_backup_{stamp}")
    live_is_desired = _normalize_sorting_key(live_key or "") == desired
    v2_is_desired = _normalize_sorting_key(v2_key or "") == desired

    backup_is_old = backup_key is not None and not (
        _normalize_sorting_key(backup_key) == desired
    )
    v2_is_old = v2_key is not None and not v2_is_desired

    if live_is_desired and backup_is_old and v2_key is None:
        return "cutover-complete"
    if (
        live_key is not None
        and not live_is_desired
        and v2_is_desired
        and backup_key is None
    ):
        return "ready-for-cutover"
    if live_is_desired and v2_is_old and backup_key is None:
        return "exchanged-awaiting-backup"
    if (
        live_key is not None
        and not live_is_desired
        and v2_key is None
        and backup_key is None
    ):
        return "not-prepared"
    return "ambiguous"


def reconcile_table_cutover(
    runner: ClickHouseRunner, database: str, table: str, stamp: str
) -> str:
    state = table_cutover_state(runner, database, table, stamp)
    if state == "cutover-complete":
        return state
    if state not in {"ready-for-cutover", "exchanged-awaiting-backup"}:
        raise RuntimeError(
            f"Unsafe migration state for {database}.{table}: {state}; "
            "manual recovery required and ingestion must remain paused"
        )
    live = qualified(database, table)
    v2 = qualified(database, f"{table}_v2")
    backup = qualified(database, f"{table}_backup_{stamp}")
    if state == "ready-for-cutover":
        runner.query(f"EXCHANGE TABLES {live} AND {v2}")
        state = table_cutover_state(runner, database, table, stamp)
        if state != "exchanged-awaiting-backup":
            raise RuntimeError(
                f"Cutover exchange verification failed for {database}.{table}; "
                "manual recovery required and ingestion must remain paused"
            )
    runner.query(f"RENAME TABLE {v2} TO {backup}")
    state = table_cutover_state(runner, database, table, stamp)
    if state != "cutover-complete":
        raise RuntimeError(
            f"Cutover verification failed for {database}.{table}; "
            "manual recovery required and ingestion must remain paused"
        )
    return state


def resolve_migration_stamp(
    runner: ClickHouseRunner, database: str, proposed_stamp: str
) -> str:
    """Reuse the active cutover id when a prior invocation stopped mid-swap."""
    desired_live_tables = 0
    backup_stamps = set()
    for table, spec in TABLES.items():
        live_key = table_sorting_key(runner, database, table)
        if _normalize_sorting_key(live_key or "") == _normalize_sorting_key(
            spec.order_by
        ):
            desired_live_tables += 1
        names = runner.query(
            "SELECT name FROM system.tables "
            f"WHERE database = '{database}' AND startsWith(name, '{table}_backup_') "
            "FORMAT TSVRaw",
            capture=True,
        )
        for name in filter(None, names.splitlines()):
            prefix = f"{table}_backup_"
            if name.startswith(prefix):
                backup_stamps.add(name[len(prefix) :])
    if desired_live_tables and backup_stamps:
        return max(backup_stamps)
    return proposed_stamp


def _parse_backup_stamp(name: str, table: str) -> Optional[datetime]:
    prefix = f"{table}_backup_"
    if not name.startswith(prefix):
        return None
    try:
        return datetime.strptime(name[len(prefix) :], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def cleanup_expired_backups(
    runner: ClickHouseRunner, database: str, *, retention_days: int
) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for table in TABLES:
        names = runner.query(
            "SELECT name FROM system.tables "
            f"WHERE database = '{database}' AND startsWith(name, '{table}_backup_') "
            "FORMAT TSVRaw",
            capture=True,
        )
        for name in filter(None, names.splitlines()):
            created = _parse_backup_stamp(name.strip(), table)
            if created is not None and created < cutoff:
                runner.query(
                    f"DROP TABLE IF EXISTS {qualified(database, name.strip())}"
                )


def apply_migration(args, stamp: str) -> None:
    runner = ClickHouseRunner(args.clickhouse_command)
    stamp = resolve_migration_stamp(runner, args.database, stamp)
    plan = migration_plan(args.database, stamp)
    for table in TABLES:
        state = table_cutover_state(runner, args.database, table, stamp)
        if state == "cutover-complete":
            continue
        if state == "ambiguous":
            raise RuntimeError(
                f"Unsafe migration state for {args.database}.{table}: {state}; "
                "manual recovery required before pausing ingestion"
            )
        if state == "not-prepared":
            for label, sql in plan:
                if label in {"create-v2", "backfill"} and (f".{table}_v2" in sql):
                    runner.query(sql)

    run_hook(args.pause_command)
    run_hook(args.flush_command)
    for table in TABLES:
        state = table_cutover_state(runner, args.database, table, stamp)
        if state == "ready-for-cutover":
            runner.query(
                next(
                    sql
                    for label, sql in plan
                    if label == "delta-copy-after-pause-and-flush"
                    and f".{table}_v2 " in sql
                )
            )
            source = runner.query(validation_query(args.database, table), capture=True)
            target = runner.query(
                validation_query(args.database, f"{table}_v2"), capture=True
            )
            if source != target:
                raise RuntimeError(
                    f"Validation mismatch for {table}: source={source!r}, v2={target!r}"
                )
        reconcile_table_cutover(runner, args.database, table, stamp)
    incomplete = [
        table
        for table in TABLES
        if table_cutover_state(runner, args.database, table, stamp)
        != "cutover-complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"Migration incomplete for {', '.join(incomplete)}; "
            "manual recovery required and ingestion remains paused"
        )
    cleanup_expired_backups(
        runner, args.database, retention_days=args.backup_retention_days
    )
    run_hook(args.resume_command)


def rollback_plan(database: str, backup: str, stamp: str) -> Iterable[str]:
    if not any(backup.startswith(f"{table}_backup_") for table in TABLES):
        raise ValueError("Rollback backup must be a versioned analytics backup table")
    live_table = backup.split("_backup_", 1)[0]
    live = qualified(database, live_table)
    old = qualified(database, backup)
    backup_stamp = backup.split("_backup_", 1)[1]
    displaced = qualified(database, f"{live_table}_rollback_displaced_{backup_stamp}")
    yield f"EXCHANGE TABLES {live} AND {old}"
    yield f"RENAME TABLE {old} TO {displaced}"


def reconcile_rollback(
    runner: ClickHouseRunner, database: str, backup: str, stamp: str
) -> str:
    statements = list(rollback_plan(database, backup, stamp))
    live_table = backup.split("_backup_", 1)[0]
    backup_stamp = backup.split("_backup_", 1)[1]
    displaced_table = f"{live_table}_rollback_displaced_{backup_stamp}"
    desired = _normalize_sorting_key(TABLES[live_table].order_by)
    live_key = table_sorting_key(runner, database, live_table)
    backup_key = table_sorting_key(runner, database, backup)
    displaced_key = table_sorting_key(runner, database, displaced_table)

    live_is_desired = _normalize_sorting_key(live_key or "") == desired
    backup_is_desired = _normalize_sorting_key(backup_key or "") == desired
    displaced_is_desired = _normalize_sorting_key(displaced_key or "") == desired
    if (
        live_key is not None
        and not live_is_desired
        and backup_key is None
        and displaced_is_desired
    ):
        return "rollback-complete"
    ready = (
        live_is_desired
        and backup_key is not None
        and not backup_is_desired
        and displaced_key is None
    )
    exchanged = (
        live_key is not None
        and not live_is_desired
        and backup_is_desired
        and displaced_key is None
    )
    if not (ready or exchanged):
        raise RuntimeError(
            f"Unsafe rollback state for {database}.{live_table}; "
            "manual recovery required"
        )
    if ready:
        runner.query(statements[0])
        return reconcile_rollback(runner, database, backup, stamp)
    runner.query(statements[1])
    return reconcile_rollback(runner, database, backup, stamp)


def apply_rollback(args, backup: str, stamp: str) -> None:
    """Rollback without losing writes accepted after the original cutover."""
    if not any(backup.startswith(f"{table}_backup_") for table in TABLES):
        raise ValueError("Rollback backup must be a versioned analytics backup table")
    live_table = backup.split("_backup_", 1)[0]
    runner = ClickHouseRunner(args.clickhouse_command)
    run_hook(args.pause_command)
    run_hook(args.flush_command)

    desired = _normalize_sorting_key(TABLES[live_table].order_by)
    live_key = table_sorting_key(runner, args.database, live_table)
    backup_key = table_sorting_key(runner, args.database, backup)
    if (
        _normalize_sorting_key(live_key or "") == desired
        and backup_key is not None
        and _normalize_sorting_key(backup_key) != desired
    ):
        live = qualified(args.database, live_table)
        old = qualified(args.database, backup)
        runner.query(
            f"INSERT INTO {old} SELECT * FROM {live} "
            f"WHERE event_id NOT IN (SELECT event_id FROM {old})"
        )
        source = runner.query(validation_query(args.database, live_table), capture=True)
        target = runner.query(validation_query(args.database, backup), capture=True)
        if source != target:
            raise RuntimeError(
                f"Rollback validation mismatch for {live_table}: "
                f"live={source!r}, backup={target!r}"
            )

    if reconcile_rollback(runner, args.database, backup, stamp) != "rollback-complete":
        raise RuntimeError(
            f"Rollback incomplete for {args.database}.{live_table}; "
            "manual recovery required and ingestion remains paused"
        )
    run_hook(args.resume_command)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="llm_router")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", metavar="BACKUP_TABLE")
    parser.add_argument("--pause-command")
    parser.add_argument("--flush-command")
    parser.add_argument("--resume-command")
    parser.add_argument("--backup-retention-days", type=int, default=7)
    parser.add_argument(
        "--clickhouse-command",
        default=(
            "docker compose exec -T clickhouse clickhouse-client "
            "--user llm_router --password llm_router_pass"
        ),
    )
    args = parser.parse_args(argv)
    if args.apply:
        missing = [
            flag
            for flag, value in (
                ("--pause-command", args.pause_command),
                ("--flush-command", args.flush_command),
                ("--resume-command", args.resume_command),
            )
            if not value
        ]
        if missing:
            parser.error("--apply requires " + ", ".join(missing))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        if args.rollback:
            statements = list(rollback_plan(args.database, args.rollback, stamp))
            if args.apply:
                apply_rollback(args, args.rollback, stamp)
            else:
                print("DRY RUN: rollback; pass --apply to execute")
                for sql in statements:
                    print(sql)
            return 0

        if args.apply:
            apply_migration(args, stamp)
        else:
            print("DRY RUN: side-by-side migration; pass --apply to execute")
            print(
                "HOOK pause consumer; HOOK flush pending batches; HOOK resume consumer"
            )
            for label, sql in migration_plan(args.database, stamp):
                print(f"[{label}] {sql}")
        return 0
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
