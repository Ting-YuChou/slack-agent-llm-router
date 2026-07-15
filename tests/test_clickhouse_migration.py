import subprocess
import sys
from pathlib import Path


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
    assert "EXCHANGE TABLES analytics_test.query_logs" in result.stdout
    assert "analytics_test.query_logs_backup_20260715T120000Z" in result.stdout
