"""Export de-identified eval-task drafts from agent feedback, grades and CI repairs.

Selects runs the owner marked ``needs_changes``, runs where the grader and the
owner disagree, and CI-repair runs that failed. Each becomes one JSONL draft
keyed by the sha256 of its run id; user, Slack channel/thread and run ids are
never written, and comment and prompt text are redacted. A human turns drafts
into ``eval/tasks/*``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from scripts.agent_eval import redact_text
except ImportError:  # executed as ``python scripts/export_agent_feedback.py``
    from agent_eval import redact_text

REASONS = ("grader_disagreement", "owner_needs_changes", "ci_repair_failed")
FAILED_RUN_STATUSES = {"failed", "timed_out"}
# Slack mentions and ids (<@U…>, <#C…|name>, <!subteam^S…>) left in free text.
_SLACK_REF = re.compile(r"<[@#!][^>]*>")
_SLACK_ID = re.compile(r"\b[UWCDGS][A-Z0-9]{8,}\b")
# Runs joined with their latest owner feedback, grade, verification and the
# first user message. CI-repair children run under their repair id, which is
# also their task id.
QUERY = """
SELECT r.run_id AS run_id,
  JSONExtractString(r.payload_json, 'owner_user_id') AS user_id,
  r.status AS run_status,
  JSONExtractString(r.payload_json, 'task_id') = r.run_id AS is_ci_repair,
  f.verdict AS feedback_verdict, f.reason_category AS reason_category,
  f.comment AS comment, g.grader_status AS grader_status,
  v.verification_status AS verification_status,
  r.prompts_version AS prompts_version, p.prompt AS prompt,
  JSONExtract(r.payload_json, 'changed_files', 'Array(String)') AS changed_files,
  r.collected_at AS created_at
FROM agent_runs_latest r
LEFT JOIN (SELECT run_id, argMax(verdict, version) AS verdict,
    argMax(reason_category, version) AS reason_category,
    argMax(comment, version) AS comment
  FROM agent_feedback_latest GROUP BY run_id) f ON f.run_id = r.run_id
LEFT JOIN (SELECT run_id, argMax(status, (attempt, version)) AS grader_status
  FROM agent_grades_latest GROUP BY run_id) g ON g.run_id = r.run_id
LEFT JOIN (SELECT run_id, argMax(status, (attempt, version)) AS verification_status
  FROM agent_verifications_latest GROUP BY run_id) v ON v.run_id = r.run_id
LEFT JOIN (SELECT run_id,
    argMin(JSONExtractRaw(payload_json, 'entry', 'message', 'content'), sequence) AS prompt
  FROM agent_events_latest WHERE kind = 'session_entry'
    AND JSONExtractString(payload_json, 'entry', 'message', 'role') = 'user'
  GROUP BY run_id) p ON p.run_id = r.run_id
WHERE r.collected_at >= now() - INTERVAL {days:UInt32} DAY
ORDER BY r.collected_at
"""


def select_reason(row):
    """Return why ``row`` becomes a draft, or None when it is not selected."""
    owner = row.get("feedback_verdict") or ""
    grader = row.get("grader_status") or ""
    if (grader, owner) in {
        ("passed", "needs_changes"),
        ("needs_changes", "accepted"),
    }:
        return "grader_disagreement"
    if owner == "needs_changes":
        return "owner_needs_changes"
    if row.get("is_ci_repair") and (
        row.get("run_status") in FAILED_RUN_STATUSES
        or row.get("verification_status") == "failed"
    ):
        return "ci_repair_failed"
    return None


def _prompt_text(value):
    """Flatten a raw Pi message content (string or text blocks) to plain text."""
    if not value:
        return ""
    try:
        content = json.loads(value)
    except ValueError:
        return value
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return value


def _clean(text, identifiers):
    text = redact_text(text or "", identifiers)
    return _SLACK_ID.sub("[REDACTED]", _SLACK_REF.sub("[REDACTED]", text))


def draft(row):
    """Build the de-identified draft for ``row``; None when it is not selected."""
    reason = select_reason(row)
    if reason is None:
        return None
    # Scrub the row's own ids from free text; a very short id would only
    # mangle ordinary words.
    identifiers = [
        str(row[key])
        for key in ("run_id", "user_id", "channel_id", "thread_ts")
        if len(str(row.get(key) or "")) >= 6
    ]
    created_at = row.get("created_at") or ""
    return {
        "draft_id": hashlib.sha256(str(row["run_id"]).encode()).hexdigest(),
        "reason": reason,
        "reason_category": row.get("reason_category") or "",
        "comment": _clean(row.get("comment"), identifiers),
        "prompt": _clean(_prompt_text(row.get("prompt")), identifiers),
        "changed_files": [
            _clean(path, identifiers) for path in row.get("changed_files") or []
        ],
        "grader_status": row.get("grader_status") or "",
        "verification_status": row.get("verification_status") or "",
        "prompts_version": row.get("prompts_version") or "",
        "created_at": (
            created_at if isinstance(created_at, str) else created_at.isoformat()
        ),
    }


def load_rows(path):
    """Read joined rows (a JSON array of objects, as ``QUERY`` returns them)."""
    return json.loads(Path(path).read_text())


def query_rows(client, days):
    result = client.query(QUERY, parameters={"days": days})
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def export(rows, destination):
    """Write one JSONL draft per selected row; return how many were written."""
    drafts = [result for result in map(draft, rows) if result]
    lines = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drafts)
    if str(destination) == "-":
        sys.stdout.write(lines)
    else:
        Path(destination).write_text(lines)
    return len(drafts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="joined rows as JSON")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("output", help="JSONL destination, or - for stdout")
    args = parser.parse_args()
    if args.input:
        rows = load_rows(args.input)
    else:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            username=os.getenv("CLICKHOUSE_USER", "agent_grafana"),
            password=os.environ["CLICKHOUSE_PASSWORD"],
            database=os.getenv("CLICKHOUSE_DATABASE", "default"),
        )
        try:
            rows = query_rows(client, args.days)
        finally:
            client.close()
    count = export(rows, args.output)
    print(f"wrote {count} drafts", file=sys.stderr)


if __name__ == "__main__":
    main()
