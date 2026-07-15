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
    assert "RENAME TABLE" in result.stdout
    assert "backup" in result.stdout


def test_migration_apply_requires_pause_and_flush_hooks():
    result = _run("--apply")

    assert result.returncode != 0
    assert "--pause-command" in result.stderr
    assert "--flush-command" in result.stderr


def test_rollback_plan_is_explicit_and_dry_run_by_default():
    result = _run(
        "--rollback",
        "query_logs_backup_20260715T120000Z",
        "--database",
        "analytics_test",
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "RENAME TABLE analytics_test.query_logs" in result.stdout
    assert "analytics_test.query_logs_backup_20260715T120000Z" in result.stdout


class _Runner:
    def __init__(self, sorting_keys):
        self.sorting_keys = dict(sorting_keys)
        self.commands = []

    def query(self, sql, *, capture=False):
        self.commands.append(sql)
        if "FROM system.tables" in sql and "sorting_key" in sql:
            table = sql.split("name = '", 1)[1].split("'", 1)[0]
            return self.sorting_keys.get(table, "")
        if sql.startswith("RENAME TABLE"):
            first, second = sql.removeprefix("RENAME TABLE ").split(", ")
            old_live, backup = first.split(" TO ")
            v2, live = second.split(" TO ")
            old_key = self.sorting_keys.pop(old_live.split(".")[-1])
            new_key = self.sorting_keys.pop(v2.split(".")[-1])
            self.sorting_keys[backup.split(".")[-1]] = old_key
            self.sorting_keys[live.split(".")[-1]] = new_key
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


def test_cutover_atomically_renames_old_live_and_v2_and_is_restart_safe():
    desired = migration.TABLES["query_logs"].order_by
    runner = _Runner({"query_logs": "event_id", "query_logs_v2": desired})

    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    rename_commands = [
        command for command in runner.commands if command.startswith("RENAME TABLE")
    ]
    assert rename_commands == [
        "RENAME TABLE analytics_test.query_logs TO "
        "analytics_test.query_logs_backup_stamp, "
        "analytics_test.query_logs_v2 TO analytics_test.query_logs"
    ]

    runner.commands.clear()
    assert (
        migration.reconcile_table_cutover(
            runner, "analytics_test", "query_logs", "stamp"
        )
        == "cutover-complete"
    )
    assert not any(command.startswith("RENAME TABLE") for command in runner.commands)


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
    assert not any(command.startswith("RENAME TABLE") for command in runner.commands)


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
