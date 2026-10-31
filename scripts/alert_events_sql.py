#!/usr/bin/env python3
"""SQL for scripts/migrate_clickhouse_alert_events.sh, from src/analytics_schema.py.

    alert_events_sql.py create DATABASE
    alert_events_sql.py copy DATABASE BACKUP_TABLE COLUMN[,COLUMN...]
    alert_events_sql.py expected-count DATABASE BACKUP_TABLE COLUMN[,COLUMN...]

``create`` prints the router's CREATE for alert_events. ``copy`` prints an
INSERT that moves an older table's rows into it: the columns the old table
has, event_id from its backfill when that reads only columns the old table
has (otherwise a fresh id, so old rows stay apart), and defaults for the rest.
``expected-count`` prints how many rows the new table keeps once
ReplacingMergeTree collapses copied rows that share a sorting key.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics_schema import ALERT_EVENTS  # noqa: E402

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Functions the event_id backfill calls; every other identifier is a column.
_FUNCTIONS = frozenset({"concat", "toString"})


def create_sql(database: str) -> str:
    return ALERT_EVENTS.create_sql(database)


def _copied_values(source_columns: Sequence[str]) -> Dict[str, str]:
    """New column -> the expression that fills it from the old table."""
    source = set(source_columns)
    values: Dict[str, str] = {}
    for column in ALERT_EVENTS.columns:
        if column.name in source:
            values[column.name] = column.name
        elif column.name == "event_id":
            # The row id in the sorting key: without it, old rows sharing a
            # source and timestamp would collapse into one.
            backfill = column.backfill or ""
            if backfill and _FUNCTIONS.union(source) >= set(
                _IDENTIFIER.findall(backfill)
            ):
                values[column.name] = backfill
            else:
                values[column.name] = "toString(generateUUIDv4())"
        # Any other column the old table lacks takes its default.
    return values


def copy_sql(database: str, backup: str, source_columns: Sequence[str]) -> str:
    values = _copied_values(source_columns)
    return (
        f"INSERT INTO {database}.{ALERT_EVENTS.name} ({', '.join(values)}) "
        f"SELECT {', '.join(values.values())} FROM {database}.{backup}"
    )


def expected_count_sql(
    database: str, backup: str, source_columns: Sequence[str]
) -> str:
    """Distinct sorting keys among the copied rows: what FINAL will count.

    A sorting-key column the old table lacks is one default for every row,
    so it does not tell rows apart and is left out.
    """
    values = _copied_values(source_columns)
    key = [
        values[name]
        for name in _IDENTIFIER.findall(ALERT_EVENTS.order_by)
        if name in values
    ]
    return f"SELECT uniqExact({', '.join(key)}) FROM {database}.{backup}"


def main(argv: Sequence[str]) -> int:
    if len(argv) == 2 and argv[0] == "create":
        print(create_sql(argv[1]))
        return 0
    if len(argv) == 4 and argv[0] == "copy":
        columns = [name for name in argv[3].split(",") if name]
        print(copy_sql(argv[1], argv[2], columns))
        return 0
    if len(argv) == 4 and argv[0] == "expected-count":
        columns = [name for name in argv[3].split(",") if name]
        print(expected_count_sql(argv[1], argv[2], columns))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
