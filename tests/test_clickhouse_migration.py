import subprocess
import sys
from pathlib import Path

import pytest

from scripts import migrate_clickhouse_analytics_v2 as migration


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migrate_clickhouse_analytics_v2.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_defaults_to_dry_run_with_safe_side_by_side_plan():
    result = _run("--database", "analytics_test")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "analytics_test.query_logs_v2" in result.stdout
    assert "uniqExact(event_id)" in result.stdout
    assert "count()" in result.stdout and "FINAL" in result.stdout
    assert "groupBitXor(cityHash64(event_id))" in result.stdout
    assert "groupBitXor(cityHash64(toJSONString(tuple(*))))" in result.stdout
    assert "EXCHANGE TABLES" in result.stdout
    assert "RENAME TABLE" in result.stdout
    assert "backup" in result.stdout


def test_migration_apply_requires_pause_and_flush_hooks():
    result = _run("--apply")

    assert result.returncode != 0
    assert "--pause-command" in result.stderr
    assert "--flush-command" in result.stderr


def test_rollback_apply_requires_pause_flush_and_resume_hooks():
    result = _run(
        "--apply",
        "--rollback",
        "query_logs_backup_20260715T120000Z",
    )

    assert result.returncode != 0
    assert "--pause-command" in result.stderr
    assert "--flush-command" in result.stderr
    assert "--resume-command" in result.stderr


def test_rollback_plan_is_explicit_and_dry_run_by_default():
    result = _run(
        "--rollback",
        "query_logs_backup_20260715T120000Z",
        "--database",
        "analytics_test",
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "EXCHANGE TABLES analytics_test.query_logs AND" in result.stdout
    assert "analytics_test.query_logs_backup_20260715T120000Z" in result.stdout


class _Runner:
    def __init__(self, sorting_keys, fail_after=None):
        self.sorting_keys = dict(sorting_keys)
        self.commands = []
        self.fail_after = fail_after

    def query(self, sql, *, capture=False):
        self.commands.append(sql)
        if "FROM system.tables" in sql and "sorting_key" in sql:
            table = sql.split("name = '", 1)[1].split("'", 1)[0]
            return self.sorting_keys.get(table, "")
        if sql.startswith("EXCHANGE TABLES"):
            first, second = sql.removeprefix("EXCHANGE TABLES ").split(" AND ")
            first, second = first.split(".")[-1], second.split(".")[-1]
            self.sorting_keys[first], self.sorting_keys[second] = (
                self.sorting_keys[second],
                self.sorting_keys[first],
            )
        elif sql.startswith("RENAME TABLE"):
            source, target = sql.removeprefix("RENAME TABLE ").split(" TO ")
            self.sorting_keys[target.split(".")[-1]] = self.sorting_keys.pop(
                source.split(".")[-1]
            )
        if self.fail_after == sql:
            raise RuntimeError("injected crash")
        return ""


def test_cutover_resumes_after_crash_without_swapping_completed_table_back():
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner(
        {
            "query_logs": desired,
            "query_logs_backup_stamp": "event_id",
        }
    )

    state = migration.reconcile_table_cutover(
        runner, "analytics_test", "query_logs", "stamp"
    )

    assert state == "cutover-complete"
    assert not any(command.startswith("RENAME TABLE") for command in runner.commands)


def test_cutover_exchanges_then_renames_old_table_to_backup():
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner({"query_logs": "event_id", "query_logs_v2": desired})

    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    mutation_commands = [
        command
        for command in runner.commands
        if command.startswith(("EXCHANGE TABLES", "RENAME TABLE"))
    ]
    assert mutation_commands == [
        "EXCHANGE TABLES analytics_test.query_logs AND analytics_test.query_logs_v2",
        "RENAME TABLE analytics_test.query_logs_v2 TO "
        "analytics_test.query_logs_backup_stamp",
    ]

    runner.commands.clear()
    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    assert not any(
        command.startswith(("EXCHANGE TABLES", "RENAME TABLE"))
        for command in runner.commands
    )


def test_cutover_resumes_after_crash_between_exchange_and_backup_rename():
    desired = migration.TABLES["query_logs"].order_by
    exchange = (
        "EXCHANGE TABLES analytics_test.query_logs AND " "analytics_test.query_logs_v2"
    )
    runner = _Runner(
        {"query_logs": "event_id", "query_logs_v2": desired},
        fail_after=exchange,
    )

    with pytest.raises(RuntimeError, match="injected crash"):
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )

    assert runner.sorting_keys == {"query_logs": desired, "query_logs_v2": "event_id"}
    runner.fail_after = None
    runner.commands.clear()
    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    assert not any(command.startswith("EXCHANGE TABLES") for command in runner.commands)
    assert any(command.startswith("RENAME TABLE") for command in runner.commands)


def test_cutover_rerun_after_backup_rename_never_exchanges_back():
    desired = migration.TABLES["query_logs"].order_by
    rename = (
        "RENAME TABLE analytics_test.query_logs_v2 TO "
        "analytics_test.query_logs_backup_stamp"
    )
    runner = _Runner(
        {"query_logs": "event_id", "query_logs_v2": desired},
        fail_after=rename,
    )

    with pytest.raises(RuntimeError, match="injected crash"):
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )

    runner.fail_after = None
    runner.commands.clear()
    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    assert not any(command.startswith("EXCHANGE TABLES") for command in runner.commands)


def test_cutover_fails_closed_on_ambiguous_sorting_state():
    runner = _Runner({"query_logs": "unexpected", "query_logs_v2": "also_unexpected"})

    with pytest.raises(RuntimeError, match="manual recovery required"):
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )


def test_apply_failure_keeps_ingestion_paused(monkeypatch):
    hooks = []
    monkeypatch.setattr(migration, "run_hook", hooks.append)

    class _ApplyRunner:
        def query(self, sql, *, capture=False):
            if "sorting_key" in sql:
                table = sql.split("name = '", 1)[1].split("'", 1)[0]
                if table in migration.TABLES:
                    return "event_id"
                return ""
            if sql.startswith("SELECT uniqExact"):
                return "same"
            return ""

    monkeypatch.setattr(migration, "ClickHouseRunner", lambda command: _ApplyRunner())
    monkeypatch.setattr(
        migration,
        "reconcile_table_cutover",
        lambda *args: (_ for _ in ()).throw(RuntimeError("cutover failed")),
    )
    args = type(
        "Args",
        (),
        {
            "clickhouse_command": "ignored",
            "database": "analytics_test",
            "pause_command": "pause",
            "flush_command": "flush",
            "resume_command": "resume",
            "backup_retention_days": 7,
        },
    )()

    with pytest.raises(RuntimeError, match="cutover failed"):
        migration.apply_migration(args, "stamp")

    assert hooks == ["pause", "flush"]
    assert "resume" not in hooks


def test_restart_reuses_active_partial_cutover_stamp():
    class _StampRunner:
        def query(self, sql, *, capture=False):
            if "startsWith(name" in sql:
                return "query_logs_backup_20260715T120000Z\n"
            if "sorting_key" in sql:
                table = sql.split("name = '", 1)[1].split("'", 1)[0]
                if table == "query_logs":
                    return migration.TABLES["query_logs"].order_by
                if table == "system_metrics":
                    return "event_id"
            return ""

    assert (
        migration.resolve_migration_stamp(
            _StampRunner(), "analytics_test", "20260715T130000Z"
        )
        == "20260715T120000Z"
    )


def test_rollback_is_forward_only_and_restart_safe():
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner(
        {
            "query_logs": desired,
            "query_logs_backup_20260715T120000Z": "event_id",
        }
    )

    assert (
        migration.reconcile_rollback(
            runner,
            "analytics_test",
            "query_logs_backup_20260715T120000Z",
            "rollbackstamp",
        )
        == "rollback-complete"
    )
    assert runner.sorting_keys["query_logs"] == "event_id"

    runner.commands.clear()
    assert (
        migration.reconcile_rollback(
            runner,
            "analytics_test",
            "query_logs_backup_20260715T120000Z",
            "rollbackstamp",
        )
        == "rollback-complete"
    )
    assert not any(
        command.startswith(("EXCHANGE TABLES", "RENAME TABLE"))
        for command in runner.commands
    )


def test_rollback_resumes_after_crash_between_exchange_and_displaced_rename():
    desired = migration.TABLES["query_logs"].order_by
    backup = "query_logs_backup_20260715T120000Z"
    exchange = f"EXCHANGE TABLES analytics_test.query_logs AND analytics_test.{backup}"
    runner = _Runner({"query_logs": desired, backup: "event_id"}, fail_after=exchange)

    with pytest.raises(RuntimeError, match="injected crash"):
        migration.reconcile_rollback(runner, "analytics_test", backup, "rollbackstamp")

    runner.fail_after = None
    runner.commands.clear()
    assert (
        migration.reconcile_rollback(runner, "analytics_test", backup, "rollbackstamp")
        == "rollback-complete"
    )
    assert not any(command.startswith("EXCHANGE TABLES") for command in runner.commands)


def test_rollback_rerun_after_displaced_rename_never_exchanges_back():
    desired = migration.TABLES["query_logs"].order_by
    backup = "query_logs_backup_20260715T120000Z"
    rename = (
        f"RENAME TABLE analytics_test.{backup} TO "
        "analytics_test.query_logs_rollback_displaced_20260715T120000Z"
    )
    runner = _Runner({"query_logs": desired, backup: "event_id"}, fail_after=rename)

    with pytest.raises(RuntimeError, match="injected crash"):
        migration.reconcile_rollback(runner, "analytics_test", backup, "rollbackstamp")

    runner.fail_after = None
    runner.commands.clear()
    assert (
        migration.reconcile_rollback(runner, "analytics_test", backup, "rollbackstamp")
        == "rollback-complete"
    )
    assert not any(command.startswith("EXCHANGE TABLES") for command in runner.commands)


def test_rollback_restart_with_new_invocation_stamp_recognizes_complete():
    desired = migration.TABLES["query_logs"].order_by
    backup = "query_logs_backup_20260715T120000Z"
    runner = _Runner({"query_logs": desired, backup: "event_id"})

    assert (
        migration.reconcile_rollback(runner, "analytics_test", backup, "firststamp")
        == "rollback-complete"
    )
    runner.commands.clear()
    assert (
        migration.reconcile_rollback(runner, "analytics_test", backup, "secondstamp")
        == "rollback-complete"
    )
    assert not any(
        command.startswith(("EXCHANGE TABLES", "RENAME TABLE"))
        for command in runner.commands
    )


def test_apply_rollback_pauses_flushes_copies_delta_and_resumes_only_when_verified(
    monkeypatch,
):
    hooks = []
    desired = migration.TABLES["query_logs"].order_by
    backup = "query_logs_backup_20260715T120000Z"
    runner = _Runner({"query_logs": desired, backup: "event_id"})
    monkeypatch.setattr(migration, "run_hook", hooks.append)
    monkeypatch.setattr(migration, "ClickHouseRunner", lambda command: runner)
    monkeypatch.setattr(migration, "validation_query", lambda *args: "SELECT verify")
    original_query = runner.query

    def query(sql, *, capture=False):
        if sql == "SELECT verify":
            return "same"
        return original_query(sql, capture=capture)

    runner.query = query
    args = type(
        "Args",
        (),
        {
            "clickhouse_command": "ignored",
            "database": "analytics_test",
            "pause_command": "pause",
            "flush_command": "flush",
            "resume_command": "resume",
        },
    )()

    migration.apply_rollback(args, backup, "rollbackstamp")

    assert hooks == ["pause", "flush", "resume"]
    assert any(
        command.startswith(f"INSERT INTO analytics_test.{backup} SELECT *")
        for command in runner.commands
    )


def test_apply_rollback_failure_keeps_ingestion_paused(monkeypatch):
    hooks = []
    backup = "query_logs_backup_20260715T120000Z"
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner({"query_logs": desired, backup: "event_id"})
    monkeypatch.setattr(migration, "run_hook", hooks.append)
    monkeypatch.setattr(migration, "ClickHouseRunner", lambda command: runner)
    monkeypatch.setattr(
        migration,
        "validation_query",
        lambda database, table: f"SELECT verify {table}",
    )
    original_query = runner.query

    def query(sql, *, capture=False):
        if sql.startswith("SELECT verify"):
            return "live" if sql.endswith("query_logs") else "backup-mismatch"
        return original_query(sql, capture=capture)

    runner.query = query
    args = type(
        "Args",
        (),
        {
            "clickhouse_command": "ignored",
            "database": "analytics_test",
            "pause_command": "pause",
            "flush_command": "flush",
            "resume_command": "resume",
        },
    )()

    with pytest.raises(RuntimeError, match="Rollback validation mismatch"):
        migration.apply_rollback(args, backup, "rollbackstamp")

    assert hooks == ["pause", "flush"]


def test_rollback_rejects_backup_with_desired_sorting_key():
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner(
        {
            "query_logs": desired,
            "query_logs_backup_20260715T120000Z": desired,
        }
    )

    with pytest.raises(RuntimeError, match="manual recovery required"):
        migration.reconcile_rollback(
            runner,
            "analytics_test",
            "query_logs_backup_20260715T120000Z",
            "rollbackstamp",
        )
