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
                ("cutover", f"EXCHANGE TABLES {live} AND {v2}"),
                ("retain-backup-7-days", f"RENAME TABLE {v2} TO {backup}"),
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
    for label, sql in migration_plan(args.database, stamp):
        if label in {
            "delta-copy-after-pause-and-flush",
            "validate-source",
            "validate-v2",
            "cutover",
            "retain-backup-7-days",
        }:
            continue
        runner.query(sql)

    run_hook(args.pause_command)
    try:
        run_hook(args.flush_command)
        for table in TABLES:
            runner.query(
                next(
                    sql
                    for label, sql in migration_plan(args.database, stamp)
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

        for label, sql in migration_plan(args.database, stamp):
            if label in {"cutover", "retain-backup-7-days"}:
                runner.query(sql)
        cleanup_expired_backups(
            runner, args.database, retention_days=args.backup_retention_days
        )
    finally:
        run_hook(args.resume_command)


def rollback_plan(database: str, backup: str, stamp: str) -> Iterable[str]:
    if not any(backup.startswith(f"{table}_backup_") for table in TABLES):
        raise ValueError("Rollback backup must be a versioned analytics backup table")
    live_table = backup.split("_backup_", 1)[0]
    live = qualified(database, live_table)
    old = qualified(database, backup)
    displaced = qualified(database, f"{live_table}_rollback_displaced_{stamp}")
    yield f"EXCHANGE TABLES {live} AND {old}"
    yield f"RENAME TABLE {old} TO {displaced}"


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
    if args.apply and not args.rollback:
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
                runner = ClickHouseRunner(args.clickhouse_command)
                for sql in statements:
                    runner.query(sql)
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
