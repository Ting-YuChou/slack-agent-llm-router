"""scripts/migrate_clickhouse_alert_events.sh leaves today's ReplacingMergeTree
alert_events alone and creates the table from src/analytics_schema.py."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.analytics_schema import ALERT_EVENTS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migrate_clickhouse_alert_events.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the migration script needs bash"
)

# Records each --query and answers from FAKE_* variables.
FAKE_CLIENT = """#!/bin/sh
query=""
while [ $# -gt 0 ]; do
  case "$1" in
    --query) query="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\\n----\\n' "$query" >> "$LOG_DIR/queries"
case "$query" in
  EXISTS*) echo "$FAKE_EXISTS" ;;
  SHOW*) printf '%s\\n' "$FAKE_SHOW" ;;
  *system.columns*) printf '%s\\n' "$FAKE_COLUMNS" ;;
  *count*) echo 3 ;;
esac
"""

# What docker compose exec would run: the in-container script, as is.
FAKE_DOCKER = """#!/bin/sh
shift 4
exec "$@"
"""

CURRENT_SHOW = (
    "CREATE TABLE llm_router.alert_events\n(\n    `event_id` String,\n"
    "    `request_id` String,\n    `query_id` String,\n    `model_name` String,\n"
    "    `provider` String\n)\nENGINE = ReplacingMergeTree\n"
    "PARTITION BY toYYYYMMDD(timestamp)\n"
    "ORDER BY (source_event_type, timestamp, event_id)"
)
OLD_SHOW = (
    "CREATE TABLE llm_router.alert_events\n(\n    `timestamp` DateTime64(3),\n"
    "    `alert_type` String\n)\nENGINE = MergeTree\nORDER BY timestamp"
)


def _run(tmp_path, **fake):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("clickhouse-client", FAKE_CLIENT), ("docker", FAKE_DOCKER)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LOG_DIR": str(logs),
        "CLICKHOUSE_DATABASE": "llm_router",
        "CLICKHOUSE_USER": "u",
        "CLICKHOUSE_PASSWORD": "p",
        "PYTHON": sys.executable,
        **fake,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )
    queries_file = logs / "queries"
    queries = (
        [q.strip() for q in queries_file.read_text().split("\n----\n") if q.strip()]
        if queries_file.exists()
        else []
    )
    return result, queries


def test_a_current_replacing_merge_tree_table_is_left_alone(tmp_path):
    result, queries = _run(tmp_path, FAKE_EXISTS="1", FAKE_SHOW=CURRENT_SHOW)

    assert result.returncode == 0, result.stderr
    assert "already uses the current schema" in result.stdout
    assert not any(q.startswith(("RENAME", "CREATE", "INSERT")) for q in queries)


def test_a_missing_table_is_created_from_the_generated_definition(tmp_path):
    result, queries = _run(tmp_path, FAKE_EXISTS="0")

    assert result.returncode == 0, result.stderr
    creates = [q for q in queries if q.startswith("CREATE")]
    assert creates == [ALERT_EVENTS.create_sql("llm_router")]


def test_an_old_merge_tree_table_is_migrated_to_the_generated_definition(tmp_path):
    result, queries = _run(
        tmp_path,
        FAKE_EXISTS="1",
        FAKE_SHOW=OLD_SHOW,
        FAKE_COLUMNS="timestamp\nalert_type\nseverity\ndescription",
    )

    assert result.returncode == 0, result.stderr
    renames = [q for q in queries if q.startswith("RENAME")]
    assert len(renames) == 1
    creates = [q for q in queries if q.startswith("CREATE")]
    assert creates == [ALERT_EVENTS.create_sql("llm_router")]
    (insert,) = [q for q in queries if q.startswith("INSERT")]
    # Only the columns the old table has, plus a row id for the sorting key.
    assert "SELECT *" not in insert
    assert "(event_id, timestamp, alert_type, severity, description)" in insert
    assert "SELECT toString(generateUUIDv4()), timestamp," in insert


def test_an_old_table_with_the_backfill_columns_keeps_its_derived_ids():
    from scripts import alert_events_sql

    columns = [c.name for c in ALERT_EVENTS.columns if c.name != "event_id"]
    insert = alert_events_sql.copy_sql("db", "backup", columns)

    assert insert.startswith("INSERT INTO db.alert_events (event_id, timestamp,")
    assert f"SELECT {ALERT_EVENTS.columns[0].backfill}, timestamp," in insert
