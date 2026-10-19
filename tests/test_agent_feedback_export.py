import hashlib
import json

from scripts.export_agent_feedback import draft, export, load_rows, select_reason


def _row(run_id, **overrides):
    row = {
        "run_id": run_id,
        "user_id": "U0OWNER123",
        "channel_id": "C0CHANNEL99",
        "thread_ts": "1712345678.000100",
        "run_status": "completed",
        "is_ci_repair": False,
        "feedback_verdict": "accepted",
        "reason_category": "",
        "comment": "",
        "grader_status": "passed",
        "verification_status": "passed",
        "prompts_version": "pv-1",
        "prompt": "Fix the flaky retry test",
        "changed_files": ["src/retry.py"],
        "created_at": "2026-10-01T12:00:00Z",
    }
    row.update(overrides)
    return row


ROWS = [
    _row(
        "run-needs-changes",
        feedback_verdict="needs_changes",
        grader_status="needs_changes",
        reason_category="wrong_fix",
        comment="Still broken, see <@U0OWNER123> in <#C0CHANNEL99|agents>",
    ),
    _row("run-grader-too-lenient", feedback_verdict="needs_changes"),
    _row("run-grader-too-strict", grader_status="needs_changes"),
    _row(
        "run-ci-failed",
        is_ci_repair=True,
        run_status="failed",
        feedback_verdict="",
        grader_status="",
        verification_status="failed",
    ),
    _row(
        "run-secret",
        feedback_verdict="needs_changes",
        grader_status="needs_changes",
        comment="it leaked api_key=sk-abcdefghijklmnop1234 and Bearer abc.def",
        prompt="Use token ghp_abcdefghijklmnopqrstuvwxyz0123 to push",
    ),
    # Excluded: accepted and the grader agrees.
    _row("run-accepted"),
    # Excluded: a CI repair that succeeded.
    _row("run-ci-ok", is_ci_repair=True, feedback_verdict=""),
    # Excluded: accepted with no grader verdict is not a disagreement.
    _row("run-accepted-unknown-grade", grader_status="unknown"),
]


def _drafts(tmp_path):
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(ROWS))
    destination = tmp_path / "drafts.jsonl"
    export(load_rows(source), destination)
    lines = destination.read_text().splitlines()
    return [json.loads(line) for line in lines], destination.read_text()


def _hash(run_id):
    return hashlib.sha256(run_id.encode()).hexdigest()


def test_selects_each_reason_and_excludes_agreeing_accepted_runs(tmp_path):
    drafts, _ = _drafts(tmp_path)
    reasons = {draft["draft_id"]: draft["reason"] for draft in drafts}
    assert reasons == {
        _hash("run-needs-changes"): "owner_needs_changes",
        _hash("run-grader-too-lenient"): "grader_disagreement",
        _hash("run-grader-too-strict"): "grader_disagreement",
        _hash("run-ci-failed"): "ci_repair_failed",
        _hash("run-secret"): "owner_needs_changes",
    }


def test_select_reason_precedence():
    assert select_reason(_row("r")) is None
    assert (
        select_reason(_row("r", feedback_verdict="needs_changes", is_ci_repair=True))
        == "grader_disagreement"
    )
    assert (
        select_reason(
            _row(
                "r",
                feedback_verdict="needs_changes",
                grader_status="needs_changes",
                is_ci_repair=True,
                run_status="failed",
            )
        )
        == "owner_needs_changes"
    )
    assert (
        select_reason(_row("r", is_ci_repair=True, run_status="timed_out"))
        == "ci_repair_failed"
    )


def test_drafts_have_only_the_documented_fields(tmp_path):
    drafts, _ = _drafts(tmp_path)
    first = next(d for d in drafts if d["draft_id"] == _hash("run-needs-changes"))
    assert set(first) == {
        "draft_id",
        "reason",
        "reason_category",
        "comment",
        "prompt",
        "changed_files",
        "grader_status",
        "verification_status",
        "prompts_version",
        "created_at",
    }
    assert first["reason_category"] == "wrong_fix"
    assert first["prompt"] == "Fix the flaky retry test"
    assert first["changed_files"] == ["src/retry.py"]
    assert first["grader_status"] == "needs_changes"
    assert first["verification_status"] == "passed"
    assert first["prompts_version"] == "pv-1"
    assert first["created_at"] == "2026-10-01T12:00:00Z"


def test_drafts_are_deidentified(tmp_path):
    _, text = _drafts(tmp_path)
    for identifier in ["U0OWNER123", "C0CHANNEL99", "1712345678.000100"]:
        assert identifier not in text
    for row in ROWS:
        assert row["run_id"] not in text
    assert "user_id" not in text and "channel_id" not in text


def test_secrets_in_comment_and_prompt_are_redacted(tmp_path):
    drafts, text = _drafts(tmp_path)
    for secret in [
        "sk-abcdefghijklmnop1234",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "abc.def",
    ]:
        assert secret not in text
    leaked = next(d for d in drafts if d["draft_id"] == _hash("run-secret"))
    assert "[REDACTED]" in leaked["comment"]
    assert "[REDACTED]" in leaked["prompt"]


def test_structured_prompt_content_is_flattened():
    content = json.dumps([{"type": "text", "text": "first"}, {"type": "image"}])
    result = draft(_row("r", feedback_verdict="needs_changes", prompt=content))
    assert result["prompt"] == "first"
