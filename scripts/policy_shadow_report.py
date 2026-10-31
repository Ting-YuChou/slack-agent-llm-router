#!/usr/bin/env python3
"""Report how often the policy cache would have changed routing (A1 shadow mode).

Reads ClickHouse routing_decisions over the last N hours and prints, overall,
by selected model and by user tier, how many decisions the shadow policy
agreed with, differed from, or skipped.

routing_decisions records only the shadow model: skipped_empty (the policy
would not change routing), unavailable (the policy cache could not be read)
and a failed shadow replay all leave it empty, so they share the "skipped"
column. The router's shadow_policy_decisions metric splits them. Rows written
outside shadow mode are skipped too.

Connection settings come from CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER
and CLICKHOUSE_PASSWORD; the password is never taken from the command line.
"""

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Report section -> the bucket expression; a fixed list, never user input.
DIMENSIONS = {
    "overall": "'all'",
    "by_model": "selected_model",
    "by_tier": "user_tier",
}

QUERY = """
SELECT
    {dimension} AS bucket,
    count() AS decisions,
    countIf(shadow_selected_model != '' AND shadow_selected_model = selected_model)
        AS agree,
    countIf(shadow_selected_model != '' AND shadow_selected_model != selected_model)
        AS differ,
    countIf(shadow_selected_model = '') AS skipped
FROM {{database:Identifier}}.routing_decisions FINAL
WHERE timestamp >= now() - INTERVAL {{hours:UInt32}} HOUR
GROUP BY bucket
ORDER BY decisions DESC, bucket
"""


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _summary(row: Sequence[Any]) -> Dict[str, Any]:
    bucket, decisions, agree, differ, skipped = row
    decisions, agree, differ, skipped = (
        int(decisions),
        int(agree),
        int(differ),
        int(skipped),
    )
    replayed = agree + differ
    return {
        "bucket": str(bucket),
        "decisions": decisions,
        "agree": agree,
        "differ": differ,
        "skipped": skipped,
        # Of the decisions the shadow policy replayed.
        "agree_rate": _rate(agree, replayed),
        "differ_rate": _rate(differ, replayed),
        # Of all decisions.
        "skipped_rate": _rate(skipped, decisions),
    }


def build_report(client: Any, *, hours: int, database: str) -> Dict[str, Any]:
    """One section per dimension, each a list of bucket summaries."""
    parameters = {"hours": int(hours), "database": database}
    report: Dict[str, Any] = {}
    for section, dimension in DIMENSIONS.items():
        result = client.query(QUERY.format(dimension=dimension), parameters=parameters)
        report[section] = [_summary(row) for row in result.result_rows]
    return report


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_report(report: Dict[str, Any], *, hours: int) -> str:
    lines = [
        f"Policy shadow report, last {hours} hours",
        "skipped = skipped_empty, unavailable or a failed replay "
        "(not split in routing_decisions)",
    ]
    titles = {
        "overall": "Overall",
        "by_model": "By selected model",
        "by_tier": "By user tier",
    }
    header = (
        f"{'bucket':<32} {'decisions':>9} {'agree':>7} {'differ':>7} "
        f"{'skipped':>7} {'agree%':>7} {'differ%':>8} {'skipped%':>9}"
    )
    for section, title in titles.items():
        lines.extend(["", title, header])
        rows: List[Dict[str, Any]] = report.get(section, [])
        if not rows:
            lines.append("(no decisions)")
        for row in rows:
            lines.append(
                f"{row['bucket'][:32]:<32} {row['decisions']:>9} "
                f"{row['agree']:>7} {row['differ']:>7} {row['skipped']:>7} "
                f"{_percent(row['agree_rate']):>7} "
                f"{_percent(row['differ_rate']):>8} "
                f"{_percent(row['skipped_rate']):>9}"
            )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument(
        "--database", default=os.getenv("CLICKHOUSE_DATABASE", "llm_router")
    )
    args = parser.parse_args(argv)
    if not IDENTIFIER.match(args.database):
        parser.error("--database must be a plain identifier")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )
    try:
        report = build_report(client, hours=args.hours, database=args.database)
    finally:
        client.close()
    print(format_report(report, hours=args.hours))
    return 0


if __name__ == "__main__":
    sys.exit(main())
