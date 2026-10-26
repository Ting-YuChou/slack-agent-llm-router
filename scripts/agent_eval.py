"""Live, budgeted coding-agent eval on synthetic fixture repositories.

Each task under ``eval/tasks/<id>/`` has a ``task.json``, a ``fixture/`` tree
that becomes the agent's repository, and a hidden ``oracle/`` that the agent
never sees: after the run it is copied next to the committed tree (or the
baseline when nothing was committed) and executed in the verification image.

Results are written to ``eval/results/<model>__<prompt_version>__<checksum>.json``
and compared against ``eval/baseline.json``; any regression beyond tolerance,
any metric above its optional baseline ceiling, or any missing metric fails the
run (exit 1). ``--recompute-summary`` re-derives the summary of a committed
results file without a live run. ``--suite repair`` runs ``eval/repair-tasks``
against ``eval/repair-baseline.json`` instead.

Without an override flag a run uses the ``profile_overrides`` its baseline was
recorded with, so the two stay comparable; ``--no-baseline-overrides`` runs the
tasks' own profiles. A run whose overrides differ from the baseline's is warned
about, or failed with ``--strict-overrides``.

Beyond its ``expect``, a task can require run signals: ``requires_repair``,
``requires_run_checks`` (the writer ran run_checks and its first attempt passed),
``requires_degrade`` (a structured investigation degraded to an unresolved gap)
and ``mcp_servers`` (the run gets these servers and each must answer a call).
A ``delivery_decision`` of ``discard`` or ``expire`` has the fixture owner reject
the delivery card or let it lapse; the run must end with that decision's error
code, and the oracle judges the patch it kept.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

try:
    from scripts import validate_structured_workflow as topology
except ImportError:  # executed as ``python scripts/agent_eval.py``
    import validate_structured_workflow as topology

PROJECT = pathlib.Path(__file__).resolve().parents[1]
EXPECTATIONS = ("pass", "must_not_commit", "scope_violation", "no_diff")
TASK_MODES = ("implement", "discuss")
WORKFLOW_MODES = ("single", "structured")
ORACLE_DIR = ".eval-oracle"
# MCP servers the eval topology can give a run: stdio servers, which need no gateway.
EVAL_MCP_SERVERS = ("codegraph",)
# suite -> (tasks directory, baseline), relative to the project.
SUITES = {
    "tasks": ("eval/tasks", "eval/baseline.json"),
    "repair": ("eval/repair-tasks", "eval/repair-baseline.json"),
}
GRADER_SAMPLES = (1, 3)
GRADER_AGGREGATES = ("unanimous", "majority")
# metric -> (direction, default tolerance). "higher" and "lower_abs" use an
# absolute tolerance; "lower" a relative one (useless against a 0 reference).
CONTRACT_METRICS = {
    "pass_at_1": ("higher", 0.1),
    "grader_oracle_agreement": ("higher", 0.1),
    "cost_usd_p50": ("lower", 0.25),
    "wall_time_s_p50": ("lower", 0.5),
    "grader_unknown_rate": ("lower_abs", 0.1),
    # 12 runs make this noisy, hence the wide absolute tolerance.
    "first_attempt_pass_rate": ("higher", 0.15),
}
# The runtime grader emits passed | needs_changes | unknown; needs_changes (and
# a legacy failed) is a fail verdict. The hidden oracle emits passed | failed.
GRADER_VERDICTS = ("passed", "needs_changes", "failed")
ORACLE_VERDICTS = ("passed", "failed")
# Grader states meaning the grader ran, or was due to run, on a candidate.
GRADER_EXPECTED = ("passed", "needs_changes", "unknown")
ERROR_MESSAGE_LIMIT = 500
NO_ERROR_CODE = "no_error_code"
_CREDENTIAL_NAME = (
    r"[A-Za-z0-9_-]*(?:AUTHORIZATION|API[_-]?KEY|ACCESS[_-]?KEY|SECRET[_-]?KEY"
    r"|PRIVATE[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|GATEWAY[_-]?TOKEN"
    r"|SIGNING[_-]?SECRET|PASSWORD|PASSPHRASE|SECRET|TOKEN)"
)
# Mirrors agent-runtime/src/telemetry.ts redact(), plus common token shapes.
_REDACTIONS = (
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----[\s\S]*?"
            r"(?:-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----|$)"
        ),
        "[REDACTED]",
    ),
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]+", re.I), "Bearer [REDACTED]"),
    (
        re.compile(rf"(\b{_CREDENTIAL_NAME}[\"']?[\t ]*[:=][\t ]*)[^\r\n]+", re.I),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|xox[abposr]-[A-Za-z0-9-]{8,}"
            r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
        ),
        "[REDACTED]",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
        "[REDACTED]",
    ),
)


class TaskError(ValueError):
    pass


def redact_text(text, secrets=()):
    """Mask credentials in ``text``: known ``secrets`` first, then token shapes."""
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _bounded(text):
    if len(text) <= ERROR_MESSAGE_LIMIT:
        return text
    return text[: ERROR_MESSAGE_LIMIT - 1] + "\u2026"


def run_error(state, secrets=()):
    """Return the run's ``(error.code, error.message)``, redacted and bounded."""
    error = state.get("error")
    if not isinstance(error, dict):
        return None, None
    code = error.get("code") if isinstance(error.get("code"), str) else None
    message = error.get("message")
    if not isinstance(message, str):
        return code, None
    return code, _bounded(redact_text(message, secrets))


def _string(value):
    return value if isinstance(value, str) else None


def run_causes(state):
    """Return the run's ``(error.cause_code, grader_cause)``; ``None`` when absent."""
    error = state.get("error")
    cause_code = _string(error.get("cause_code")) if isinstance(error, dict) else None
    grader_cause = _string(state.get("grader_cause"))
    return cause_code, grader_cause and _bounded(redact_text(grader_cause))


def run_degrades(state):
    """Causes of the structured investigations that degraded to an unresolved gap."""
    return [
        _string(event.get("cause"))
        for event in state.get("events") or []
        if isinstance(event, dict) and event.get("type") == "investigation_degraded"
    ]


def run_attempts(state):
    """Return each attempt's gates and whether it reused the writer, or ``None``."""
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        return None
    return [
        {
            "verification": _string(attempt.get("verification_status")),
            "grader": _string(attempt.get("grader_status")),
            "reused": _reused(attempt),
        }
        for attempt in attempts
        if isinstance(attempt, dict)
    ]


def _reused(attempt):
    startup = attempt.get("startup")
    reused = startup.get("reused") if isinstance(startup, dict) else None
    return reused if isinstance(reused, bool) else None


def _files(root):
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def load_task(task_dir):
    task_dir = pathlib.Path(task_dir)
    try:
        task = json.loads((task_dir / "task.json").read_text())
    except (OSError, ValueError) as error:
        raise TaskError(f"{task_dir.name}: unreadable task.json ({error})")
    if task.get("id") != task_dir.name:
        raise TaskError(f"{task_dir.name}: task id must match its directory")
    if task.get("expect") not in EXPECTATIONS:
        raise TaskError(f"{task_dir.name}: expect must be one of {EXPECTATIONS}")
    if task.get("task_mode") not in TASK_MODES:
        raise TaskError(f"{task_dir.name}: task_mode must be one of {TASK_MODES}")
    if task.get("workflow_mode", "single") not in WORKFLOW_MODES:
        raise TaskError(
            f"{task_dir.name}: workflow_mode must be one of {WORKFLOW_MODES}"
        )
    for flag in ("requires_repair", "requires_run_checks", "requires_degrade"):
        if not isinstance(task.get(flag, False), bool):
            raise TaskError(f"{task_dir.name}: {flag} must be a boolean")
    if task.get("requires_degrade") and task.get("workflow_mode") != "structured":
        raise TaskError(
            f"{task_dir.name}: requires_degrade needs a structured workflow"
        )
    decision = task.get("delivery_decision", "deliver")
    if decision not in topology.DELIVERY_DECISIONS:
        raise TaskError(
            f"{task_dir.name}: delivery_decision must be one of "
            f"{topology.DELIVERY_DECISIONS}"
        )
    if decision != "deliver" and task["expect"] != "must_not_commit":
        raise TaskError(
            f"{task_dir.name}: a declined delivery must expect must_not_commit"
        )
    servers = task.get("mcp_servers", [])
    if not isinstance(servers, list) or not all(
        server in EVAL_MCP_SERVERS for server in servers
    ):
        raise TaskError(
            f"{task_dir.name}: mcp_servers may list only {EVAL_MCP_SERVERS}"
        )
    if servers and task["task_mode"] == "discuss":
        raise TaskError(f"{task_dir.name}: discuss runs get no mcp_servers")
    if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        raise TaskError(f"{task_dir.name}: prompt is required")
    profile = task.get("profile")
    if not isinstance(profile, dict) or not profile.get("allowedPaths"):
        raise TaskError(f"{task_dir.name}: profile.allowedPaths is required")
    argv = (task.get("oracle") or {}).get("argv")
    if not argv or not all(isinstance(part, str) for part in argv):
        raise TaskError(f"{task_dir.name}: oracle.argv is required")

    fixture_dir = task_dir / "fixture"
    oracle_dir = task_dir / "oracle"
    fixture_files = _files(fixture_dir) if fixture_dir.is_dir() else []
    if not fixture_files:
        raise TaskError(f"{task_dir.name}: fixture/ is missing or empty")
    oracle_files = _files(oracle_dir) if oracle_dir.is_dir() else []
    if not oracle_files:
        raise TaskError(f"{task_dir.name}: hidden oracle/ is missing or empty")
    # The oracle must stay hidden: reject it if it leaks into the fixture by
    # name, by content, or through the directory it is mounted at.
    fixture_names = {pathlib.PurePosixPath(name).name for name in fixture_files}
    fixture_digests = {
        hashlib.sha256((fixture_dir / name).read_bytes()).hexdigest()
        for name in fixture_files
    }
    for name in oracle_files:
        digest = hashlib.sha256((oracle_dir / name).read_bytes()).hexdigest()
        if (
            pathlib.PurePosixPath(name).name in fixture_names
            or digest in fixture_digests
        ):
            raise TaskError(
                f"{task_dir.name}: oracle file {name} leaks into the fixture"
            )
    if any(name.split("/")[0] == ORACLE_DIR for name in fixture_files):
        raise TaskError(f"{task_dir.name}: fixture must not contain {ORACLE_DIR}/")

    task.setdefault("workflow_mode", "single")
    task.update(
        fixture_dir=fixture_dir, oracle_dir=oracle_dir, oracle_files=oracle_files
    )
    return task


def load_tasks(tasks_dir, only=None):
    paths = sorted(path for path in pathlib.Path(tasks_dir).iterdir() if path.is_dir())
    if only:
        paths = [path for path in paths if path.name in only]
    if not paths:
        raise TaskError(f"no eval tasks found in {tasks_dir}")
    return [load_task(path) for path in paths]


def _in_scope(path, allowed_paths):
    for allowed in allowed_paths:
        prefix = allowed.rstrip("/")
        if prefix in ("", ".") or path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def score_run(task, outcome):
    """Return ``{"passed", "reason"}`` for an outcome against ``task["expect"]``.

    The outcome must also show every run signal the task requires (see the module
    docstring). A task with ``requires_repair`` also needs a repair to have
    happened, in the paused writer when its profile turns ``writerReuse`` on.
    """
    score = _score_expectation(task, outcome)
    if not score["passed"]:
        return score
    missing = _missing_signal(task, outcome)
    if missing:
        return {"passed": False, "reason": missing}
    if not task.get("requires_repair"):
        return score
    attempts = outcome.get("attempts") or []
    if len(attempts) < 2:
        return {"passed": False, "reason": "no repair happened"}
    if not task["profile"].get("writerReuse"):
        return {"passed": True, "reason": "repaired in a fresh writer"}
    if any(attempt.get("reused") is not True for attempt in attempts[1:]):
        return {"passed": False, "reason": "repair did not reuse the writer"}
    return {"passed": True, "reason": "repaired in a reused writer"}


def _missing_signal(task, outcome):
    """Why ``outcome`` lacks a run signal ``task`` requires; ``None`` if it has them all."""
    declined = topology.DECLINED_DELIVERY.get(task.get("delivery_decision", "deliver"))
    if declined and outcome.get("error_code") != declined:
        ended = outcome.get("error_code") or outcome.get("status")
        return f"run ended {ended}, not {declined}"
    if task.get("requires_run_checks"):
        if not outcome.get("writer_checks_calls"):
            return "the writer never ran run_checks"
        attempts = outcome.get("attempts") or []
        if not attempts or not _gates_passed(attempts[0]):
            return "run_checks did not catch the bug before final verification"
    if task.get("requires_degrade") and not outcome.get("degraded_investigations"):
        return "no investigation degraded"
    for server in task.get("mcp_servers", []):
        if not _mcp_answered(outcome, server):
            return f"no {server} MCP call succeeded"
    return None


def _mcp_answered(outcome, server):
    servers = (outcome.get("mcp_summary") or {}).get("servers") or {}
    entry = servers.get(server) or {}
    calls, failures = _number(entry.get("calls")), _number(entry.get("failures") or 0)
    return calls is not None and failures is not None and calls > failures


def _score_expectation(task, outcome):
    expect = task["expect"]
    commit = outcome.get("commit")
    changed = outcome.get("changed_files") or []
    status = outcome.get("status")
    out_of_scope = [
        path
        for path in changed
        if commit and not _in_scope(path, task["profile"]["allowedPaths"])
    ]

    def verdict(passed, reason):
        return {"passed": passed, "reason": reason}

    if expect == "pass":
        if status != "completed":
            return verdict(False, f"run ended {status}")
        if not commit:
            return verdict(False, "no commit")
        if out_of_scope:
            return verdict(False, "committed out of scope: " + ", ".join(out_of_scope))
        if outcome.get("oracle") != "passed":
            return verdict(False, f"hidden oracle {outcome.get('oracle')}")
        return verdict(True, "committed and hidden oracle passed")
    if expect == "must_not_commit":
        if commit:
            return verdict(False, f"unexpected commit {commit}")
        return verdict(True, f"no commit (run ended {status})")
    if expect == "scope_violation":
        if out_of_scope:
            return verdict(False, "committed out of scope: " + ", ".join(out_of_scope))
        return verdict(True, "no out-of-scope commit")
    if expect == "no_diff":
        if status != "completed":
            return verdict(False, f"run ended {status}")
        if commit or changed:
            return verdict(False, "discussion produced a diff")
        return verdict(True, "completed without a diff")
    raise TaskError(f"unknown expectation {expect}")


def _median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def _gates_passed(attempt):
    return attempt["verification"] == "passed" and attempt["grader"] == "passed"


def _attempt_metrics(records):
    """First-attempt pass rate (contract) and repair success rate (informational)."""
    histories = [record["attempts"] for record in records if record.get("attempts")]
    if not histories:
        # A contract metric: present but unknown, so the contract fails closed.
        return {"first_attempt_pass_rate": None}
    metrics = {
        "first_attempt_pass_rate": sum(
            1 for attempts in histories if _gates_passed(attempts[0])
        )
        / len(histories)
    }
    repaired = [
        attempts
        for attempts in histories
        if len(attempts) > 1 and not _gates_passed(attempts[0])
    ]
    if repaired:
        metrics["repair_success_rate"] = sum(
            1 for attempts in repaired if _gates_passed(attempts[-1])
        ) / len(repaired)
        # @k: a repaired run whose gates passed within its first k repairs.
        for k in range(1, max(len(attempts) for attempts in repaired)):
            metrics[f"repair_success_rate@{k}"] = sum(
                1
                for attempts in repaired
                if any(_gates_passed(attempt) for attempt in attempts[1 : k + 1])
            ) / len(repaired)
    return metrics


def _cost_by_role(records):
    """Total and per-run mean spend of each role, over the runs that report it."""
    reported = [
        record["cost_by_role"]
        for record in records
        if isinstance(record.get("cost_by_role"), dict)
    ]
    if not reported:
        return {}
    totals = {}
    for roles in reported:
        for role, cost in roles.items():
            if _number(cost) is not None and cost >= 0:
                totals[role] = totals.get(role, 0.0) + cost
    return {
        "cost_by_role": {
            "runs": len(reported),
            "total_usd": totals,
            "mean_usd": {role: total / len(reported) for role, total in totals.items()},
        }
    }


def _accepted_cost(records):
    """Every run's cost charged to the changes the oracle accepted and the run delivered."""
    accepted = sum(1 for record in records if record["passed"] and record.get("commit"))
    metrics = {"accepted_changes": accepted}
    if accepted:
        spent = sum(_number(record.get("cost_usd")) or 0.0 for record in records)
        metrics["cost_per_accepted_change_usd"] = spent / accepted
    return metrics


def _tool_error_rate(records):
    """Errors over finished calls of each writer tool, summed over the runs."""
    counts = {}
    for record in records:
        summary = record.get("tool_summary")
        for tool, entry in (summary if isinstance(summary, dict) else {}).items():
            if not isinstance(entry, dict):
                continue
            calls, errors = _number(entry.get("calls")), _number(entry.get("errors"))
            if calls is None or errors is None or not 0 <= errors <= calls or not calls:
                continue
            total = counts.setdefault(tool, {"calls": 0, "errors": 0})
            total["calls"] += calls
            total["errors"] += errors
    if not counts:
        return {}
    return {
        "tool_error_rate": {
            tool: {**total, "rate": total["errors"] / total["calls"]}
            for tool, total in counts.items()
        }
    }


def _harness_medians(records):
    """Informational medians of the runtime's harness signals; omitted without data."""
    sources = {
        "tool_util_p50": lambda r: (r.get("harness_metrics") or {}).get("tool_util"),
        "approvals_p50": lambda r: (r.get("approvals") or {}).get("requested"),
        "policy_blocks_p50": lambda r: (r.get("policy_summary") or {}).get("block"),
        "startup_s_p50": lambda r: (r.get("harness_metrics") or {}).get("startup_s"),
        "repair_startup_s_p50": lambda r: (r.get("harness_metrics") or {}).get(
            "repair_startup_s"
        ),
    }
    medians = {}
    for metric, read in sources.items():
        value = _median(read(record) for record in records)
        if value is not None:
            medians[metric] = value
    return medians


def aggregate(records):
    runs = len(records)
    passes = sum(1 for record in records if record["passed"])
    graded = [
        record
        for record in records
        if record.get("grader") in GRADER_VERDICTS
        and record.get("oracle") in ORACLE_VERDICTS
    ]
    grader_due = [
        record
        for record in records
        if record.get("verification") == "passed"
        or record.get("grader") in GRADER_EXPECTED
    ]
    grader_unknown = sum(1 for record in grader_due if record["grader"] == "unknown")
    agreeing = sum(
        1
        for record in graded
        if (record["grader"] == "passed") == (record["oracle"] == "passed")
    )
    tasks = {}
    for record in records:
        entry = tasks.setdefault(
            record["task_id"], {"runs": 0, "passes": 0, "failure_codes": {}}
        )
        entry["runs"] += 1
        entry["passes"] += 1 if record["passed"] else 0
        if not record["passed"]:
            code = record.get("error_code") or NO_ERROR_CODE
            entry["failure_codes"][code] = entry["failure_codes"].get(code, 0) + 1
    for entry in tasks.values():
        entry["pass_at_1"] = entry["passes"] / entry["runs"]
    walls = [record.get("wall_time_s") for record in records]
    return {
        "runs": runs,
        "pass_at_1": passes / runs if runs else None,
        "grader_oracle_agreement": agreeing / len(graded) if graded else None,
        "cost_usd_p50": _median(record.get("cost_usd") for record in records),
        "wall_time_s_p50": _median(walls),
        "wall_time_s_total": sum(
            value for value in walls if isinstance(value, (int, float))
        ),
        "grader_unknown_rate": (
            grader_unknown / len(grader_due) if grader_due else None
        ),
        **_attempt_metrics(records),
        **_harness_medians(records),
        **_accepted_cost(records),
        **_cost_by_role(records),
        **_tool_error_rate(records),
        "tasks": tasks,
    }


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def evaluate_contract(summary, baseline):
    """Compare ``summary`` to ``baseline``; fail closed on any missing metric.

    ``baseline["ceilings"]`` optionally caps metrics at an absolute maximum.
    """
    failures = []
    expected = (baseline or {}).get("metrics") or {}
    tolerances = (baseline or {}).get("tolerance") or {}
    ceilings = (baseline or {}).get("ceilings") or {}
    for metric, (direction, default_tolerance) in CONTRACT_METRICS.items():
        reference = _number(expected.get(metric))
        actual = _number(summary.get(metric))
        tolerance = _number(tolerances.get(metric, default_tolerance))
        if reference is None:
            failures.append(f"{metric}: missing from baseline")
            continue
        if actual is None:
            failures.append(f"{metric}: missing from eval summary")
            continue
        if tolerance is None or tolerance < 0:
            failures.append(f"{metric}: invalid tolerance")
            continue
        if direction == "higher" and actual < reference - tolerance:
            failures.append(
                f"{metric}: {actual:.3f} regressed below {reference:.3f} - {tolerance}"
            )
        if direction == "lower" and actual > reference * (1 + tolerance):
            failures.append(
                f"{metric}: {actual:.3f} regressed above {reference:.3f} x {1 + tolerance}"
            )
        if direction == "lower_abs" and actual > reference + tolerance:
            failures.append(
                f"{metric}: {actual:.3f} regressed above {reference:.3f} + {tolerance}"
            )
    for metric, value in ceilings.items():
        ceiling = _number(value)
        actual = _number(summary.get(metric))
        if ceiling is None or ceiling < 0:
            failures.append(f"{metric}: invalid ceiling")
        elif actual is None:
            missing = f"{metric}: missing from eval summary"
            if missing not in failures:
                failures.append(missing)
        elif actual > ceiling:
            failures.append(f"{metric}: {actual:.3f} exceeds ceiling {ceiling}")
    return failures


def result_key(model, prompt_version, profile_checksum):
    def clean(value):
        return re.sub(r"[^\w.-]", "_", str(value))

    return f"{clean(model)}__{clean(prompt_version)}__{clean(profile_checksum)[:12]}"


def profile_checksum(tasks):
    profiles = [
        [task["id"], task["profile"]] for task in sorted(tasks, key=lambda t: t["id"])
    ]
    return hashlib.sha256(json.dumps(profiles, sort_keys=True).encode()).hexdigest()


def default_model(project=PROJECT):
    source = (project / "agent-runtime/src/agent-model.ts").read_text()
    match = re.search(r'DEFAULT_AGENT_MODEL_REF\s*=\s*"([^"]+)"', source)
    return match[1] if match else "unknown"


def default_prompt_version(project=PROJECT):
    """Identify the prompts by the committed agent-runtime src + extensions trees."""
    try:
        trees = [
            topology.git(project, "rev-parse", f"HEAD:agent-runtime/{part}")
            for part in ("src", "extensions")
        ]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return hashlib.sha256("".join(trees).encode()).hexdigest()[:12]


def build_profile(task, image):
    profile = {
        "id": "eval-" + task["id"],
        "repository": "fixture/" + task["id"],
        "image": image,
        **task["profile"],
    }
    if task["workflow_mode"] == "structured":
        profile.setdefault(
            "structuredWorkflow",
            {
                "enabled": True,
                "concurrency": 2,
                "maxCostUsd": 5,
                "activeTimeoutMs": 1800000,
                "discoveryCostUsd": 1.5,
                "degradeUnknown": True,
            },
        )
    return profile


def runtime_env(task):
    """Runtime settings a task needs on top of the topology's: MCP servers, a delivery TTL."""
    env = topology.delivery_env(task.get("delivery_decision", "deliver"))
    if task.get("mcp_servers"):
        env.update(
            PI_AGENT_MCP_MODE="read_only",
            PI_AGENT_MCP_SERVERS=",".join(task["mcp_servers"]),
        )
    return env


def candidate_tree(booted, run_id, repo, baseline):
    """The tree a declined run's retained patch makes of ``baseline``, or ``None``.

    Nothing was committed, so this is the change the owner declined; the oracle
    judges it as it would a commit. ``repo``'s HEAD, index and files stay as they are.
    """
    try:
        patch = booted.download(f"/v1/runs/{run_id}/patch?user_id={topology.OWNER}")
        with tempfile.TemporaryDirectory(prefix="agent-eval-index-") as scratch:
            env = {**os.environ, "GIT_INDEX_FILE": str(pathlib.Path(scratch) / "index")}
            for argv, data in (
                (["read-tree", baseline], None),
                (["apply", "--cached"], patch),
                (["write-tree"], None),
            ):
                result = subprocess.run(
                    ["git", "-C", str(repo), *argv],
                    input=data,
                    env=env,
                    capture_output=True,
                    check=True,
                )
        return result.stdout.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def changed_files(repo, baseline, commit):
    if not commit:
        return []
    output = topology.git(repo, "diff", "--name-only", baseline, commit)
    return [line for line in output.splitlines() if line]


def run_oracle(task, repo, tree, image, timeout_s=120):
    """Run the hidden oracle against ``tree`` in the verification image."""
    with tempfile.TemporaryDirectory(prefix="agent-eval-oracle-") as scratch:
        work = pathlib.Path(scratch) / "tree"
        work.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", tree], capture_output=True, check=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(work)], input=archive, check=True)
        shutil.copytree(task["oracle_dir"], work / ORACLE_DIR)
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--tmpfs",
                    "/tmp",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--mount",
                    f"type=bind,src={work},dst=/workspace,readonly",
                    "--workdir",
                    "/workspace",
                    image,
                    *task["oracle"]["argv"],
                ],
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "error"
        return "passed" if result.returncode == 0 else "failed"


class LiveRunner:
    """Boot the validator topology per run and drive one task end to end."""

    def __init__(self, project=PROJECT, timeout_s=600):
        self.project = project
        self.timeout_s = timeout_s
        self.key = topology.read_openai_key(project)
        self.image = topology.verification_image_id()

    def __call__(self, task, attempt):
        root = topology.make_root(f"agent-eval-{task['id']}-")
        repo = root / "repo"
        shutil.copytree(task["fixture_dir"], repo)
        baseline = topology.init_fixture_repo(
            repo, remote=f"https://github.com/fixture/{task['id']}.git"
        )
        (root / "profile.json").write_text(json.dumps(build_profile(task, self.image)))
        decision = task.get("delivery_decision", "deliver")
        started = time.monotonic()
        with topology.Topology(
            self.project,
            root,
            repo,
            self.key,
            root / "profile.json",
            env=runtime_env(task),
        ) as booted:
            created = booted.request(
                "/v1/sessions",
                {
                    "team_id": "eval",
                    "channel_id": "eval",
                    "thread_ts": f"{attempt + 1}",
                    "user_id": topology.OWNER,
                    "task_mode": task["task_mode"],
                    "workflow_mode": task["workflow_mode"],
                    "prompt": task["prompt"],
                    "routing_text": task.get("routing_text", task["prompt"]),
                },
            )
            state, _ = topology.drive_run(
                booted, created["run_id"], timeout_s=self.timeout_s, delivery=decision
            )
            commit = state.get("commit")
            tree = commit or baseline
            if not commit and decision != "deliver":
                tree = candidate_tree(booted, created["run_id"], repo, baseline)
        wall_time_s = time.monotonic() - started
        error_code, error_message = run_error(state, secrets=[self.key])
        error_cause_code, grader_cause = run_causes(state)
        return {
            "run_id": created["run_id"],
            "fixture": str(root),
            "status": state["status"],
            "verification": state.get("verification_status"),
            "grader": state.get("grader_status"),
            "commit": commit,
            "changed_files": changed_files(repo, baseline, commit),
            "oracle": run_oracle(task, repo, tree, self.image) if tree else "error",
            "cost_usd": state.get("estimated_cost_usd"),
            "wall_time_s": wall_time_s,
            "error_code": error_code,
            "error_message": error_message,
            "error_cause_code": error_cause_code,
            "grader_cause": grader_cause,
            "prompts_version": state.get("prompts_version"),
            "approvals": state.get("approvals"),
            "writer_checks_calls": state.get("writer_checks_calls"),
            "policy_summary": state.get("policy_summary"),
            "harness_metrics": state.get("harness_metrics"),
            "cost_by_role": state.get("cost_by_role"),
            "tool_summary": state.get("tool_summary"),
            "attempts": run_attempts(state),
            "degraded_investigations": run_degrades(state),
            "mcp_summary": state.get("mcp_summary"),
        }


def execute(task, attempt, runner):
    try:
        outcome = runner(task, attempt)
    except Exception as error:  # a crashed run is a failed run, not a crashed eval
        message = _bounded(redact_text(f"{type(error).__name__}: {error}"))
        outcome = {
            "status": "error",
            "error_code": "runner_error",
            "error_message": message,
        }
        score = {"passed": False, "reason": "runner error: " + message}
    else:
        score = score_run(task, outcome)
    return {
        "task_id": task["id"],
        "attempt": attempt,
        "expect": task["expect"],
        **score,
        "status": outcome.get("status"),
        "verification": outcome.get("verification"),
        "grader": outcome.get("grader"),
        "oracle": outcome.get("oracle"),
        "commit": outcome.get("commit"),
        "changed_files": outcome.get("changed_files") or [],
        "cost_usd": outcome.get("cost_usd"),
        "wall_time_s": outcome.get("wall_time_s"),
        "run_id": outcome.get("run_id"),
        "error_code": outcome.get("error_code"),
        "error_message": outcome.get("error_message"),
        "error_cause_code": outcome.get("error_cause_code"),
        "grader_cause": outcome.get("grader_cause"),
        "prompts_version": outcome.get("prompts_version"),
        "approvals": outcome.get("approvals"),
        "writer_checks_calls": outcome.get("writer_checks_calls"),
        "policy_summary": outcome.get("policy_summary"),
        "harness_metrics": outcome.get("harness_metrics"),
        "cost_by_role": outcome.get("cost_by_role"),
        "tool_summary": outcome.get("tool_summary"),
        "attempts": outcome.get("attempts"),
        "degraded_investigations": outcome.get("degraded_investigations"),
        "mcp_summary": outcome.get("mcp_summary"),
    }


def parse_args(argv=None, require_live=True):
    parser = argparse.ArgumentParser(
        description="Live, budgeted coding-agent eval with hidden oracles; no Slack or GitHub publication."
    )
    parser.add_argument(
        "--live-openai",
        action="store_true",
        help="Explicitly allow model calls using OPENAI_API_KEY from .env.demo",
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="tasks",
        help="Task set to run and the baseline it is held to (default tasks)",
    )
    parser.add_argument(
        "--tasks-dir", type=pathlib.Path, help="Defaults to the suite's tasks"
    )
    parser.add_argument(
        "--task", action="append", help="Only run this task id (repeatable)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Stop starting new runs once this much has been spent; the eval then fails",
    )
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-run timeout")
    parser.add_argument(
        "--results-dir", type=pathlib.Path, default=PROJECT / "eval/results"
    )
    parser.add_argument(
        "--baseline", type=pathlib.Path, help="Defaults to the suite's baseline"
    )
    parser.add_argument("--model", default=None, help="Defaults to the runtime's model")
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Defaults to a hash of the committed agent-runtime src/extensions trees",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write this run's metrics to the baseline instead of comparing",
    )
    parser.add_argument(
        "--writer-reuse",
        action="store_true",
        help="Run every task with the profile's writerReuse on (paused writer reused for repairs)",
    )
    parser.add_argument(
        "--grader-samples",
        type=int,
        choices=GRADER_SAMPLES,
        help="Grade every task with this many grader samples (profile grader.samples)",
    )
    parser.add_argument(
        "--grader-aggregate",
        choices=GRADER_AGGREGATES,
        help="How the samples combine (default unanimous); needs --grader-samples",
    )
    parser.add_argument(
        "--no-baseline-overrides",
        action="store_true",
        help="Without override flags, run the tasks' own profiles rather than the "
        "profile_overrides the baseline was recorded with",
    )
    parser.add_argument(
        "--strict-overrides",
        action="store_true",
        help="Fail, rather than warn, when the run's profile overrides differ from "
        "the baseline's",
    )
    parser.add_argument(
        "--recompute-summary",
        type=pathlib.Path,
        metavar="RESULTS_JSON",
        help="Recompute the summary of a results file from its records; no live run",
    )
    args = parser.parse_args(argv)
    tasks_dir, baseline = SUITES[args.suite]
    args.tasks_dir = args.tasks_dir or PROJECT / tasks_dir
    args.baseline = args.baseline or PROJECT / baseline
    if args.recompute_summary:
        return args
    if require_live and not args.live_openai:
        parser.error("--live-openai is required; this command incurs model usage")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    if args.grader_aggregate and args.grader_samples is None:
        parser.error("--grader-aggregate needs --grader-samples")
    return args


def profile_overrides(args):
    """Profile fields the command line sets on every task (empty: the tasks' own profiles)."""
    overrides = {}
    if args.writer_reuse:
        overrides["writerReuse"] = True
    if args.grader_samples is not None:
        overrides["grader"] = {
            "samples": args.grader_samples,
            "aggregate": args.grader_aggregate or "unanimous",
        }
    return overrides


def baseline_overrides(baseline):
    """The baseline's ``profile_overrides``; ``ValueError`` unless the flags can set them."""
    overrides = baseline.get("profile_overrides") or {}
    flags = argparse.Namespace(
        writer_reuse=False, grader_samples=None, grader_aggregate=None
    )
    if isinstance(overrides, dict):
        flags.writer_reuse = overrides.get("writerReuse") is True
        grader = overrides.get("grader")
        if (
            isinstance(grader, dict)
            and type(grader.get("samples")) is int
            and grader["samples"] in GRADER_SAMPLES
            and grader.get("aggregate") in GRADER_AGGREGATES
        ):
            flags.grader_samples = grader["samples"]
            flags.grader_aggregate = grader["aggregate"]
    if profile_overrides(flags) != overrides:
        raise ValueError(
            f"profile_overrides {json.dumps(overrides, sort_keys=True)} "
            "are not ones the override flags can set"
        )
    return overrides


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _contract_failures(summary, args):
    if args.update_baseline:
        return [
            f"{metric}: missing from eval summary"
            for metric in CONTRACT_METRICS
            if _number(summary.get(metric)) is None
        ]
    return evaluate_contract(summary, _read_baseline(args.baseline))


def _read_baseline(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _finish(args, result, results_path):
    """Write ``result`` (and the baseline, if asked and clean); return the exit code."""
    summary, failures = result["summary"], result["contract_failures"]
    overrides = result.get("profile_overrides") or {}
    recorded = _read_baseline(args.baseline)
    if recorded and not args.update_baseline:
        if (recorded.get("profile_overrides") or {}) != overrides:
            mismatch = (
                "the baseline was recorded with profile overrides "
                f"{json.dumps(recorded.get('profile_overrides') or {}, sort_keys=True)} "
                f"but this run used {json.dumps(overrides, sort_keys=True)}; "
                "cost and wall time are not comparable"
            )
            if args.strict_overrides:
                failures.append("profile_overrides: " + mismatch)
            else:
                print("warning: " + mismatch, file=sys.stderr)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.update_baseline and not failures:
        baseline = {
            "key": result["key"],
            "updated_at": _now(),
            "repeats": result["repeats"],
            **({"profile_overrides": overrides} if overrides else {}),
            "metrics": {metric: summary[metric] for metric in CONTRACT_METRICS},
            "tolerance": {metric: spec[1] for metric, spec in CONTRACT_METRICS.items()},
        }
        # Ceilings are hand-set policy; carry them over rather than reset them.
        ceilings = recorded.get("ceilings")
        if ceilings:
            baseline["ceilings"] = ceilings
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(
        json.dumps(
            {"key": result["key"], "summary": summary, "contract_failures": failures},
            indent=2,
        )
    )
    return 1 if failures else 0


def recompute(args):
    """Re-derive the summary of a committed results file from its records."""
    path = args.recompute_summary
    try:
        result = json.loads(path.read_text())
        records = result["records"]
        previous = result.get("summary") or {}
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"unreadable results file {path}: {error}", file=sys.stderr)
        return 2
    summary = aggregate(records)
    summary["spent_usd"] = sum(_number(r.get("cost_usd")) or 0.0 for r in records)
    summary["budget_exhausted"] = bool(previous.get("budget_exhausted"))
    failures = _contract_failures(summary, args)
    if summary["budget_exhausted"]:
        failures.append("budget exhausted; eval incomplete")
    result.update(summary=summary, contract_failures=failures)
    return _finish(args, result, path)


def main(argv=None, runner=None):
    args = parse_args(argv, require_live=runner is None)
    if args.recompute_summary:
        return recompute(args)
    try:
        tasks = load_tasks(args.tasks_dir, args.task)
    except TaskError as error:
        print(f"invalid eval task: {error}", file=sys.stderr)
        return 2
    overrides = profile_overrides(args)
    if not overrides and not args.no_baseline_overrides:
        try:
            overrides = baseline_overrides(_read_baseline(args.baseline))
        except ValueError as error:
            print(f"invalid eval baseline {args.baseline}: {error}", file=sys.stderr)
            return 2
        if overrides:
            print(
                "using the baseline's profile overrides "
                + json.dumps(overrides, sort_keys=True),
                file=sys.stderr,
            )
    # Applied before the checksum, so an overridden run is keyed (and filed) apart from a plain one.
    tasks = [{**task, "profile": {**task["profile"], **overrides}} for task in tasks]
    if runner is None:
        runner = LiveRunner(timeout_s=args.timeout_s)
    model = args.model or default_model()
    prompt_version = args.prompt_version or default_prompt_version()
    checksum = profile_checksum(tasks)
    key = result_key(model, prompt_version, checksum)

    records = []
    spent = 0.0
    budget_exhausted = False
    for attempt in range(args.repeats):
        for task in tasks:
            if args.max_cost_usd is not None and spent >= args.max_cost_usd:
                budget_exhausted = True
                break
            record = execute(task, attempt, runner)
            spent += _number(record["cost_usd"]) or 0.0
            records.append(record)
            print(
                f"{task['id']}#{attempt}: {'PASS' if record['passed'] else 'FAIL'} ({record['reason']})",
                file=sys.stderr,
            )
        if budget_exhausted:
            break

    summary = aggregate(records)
    summary["spent_usd"] = spent
    summary["budget_exhausted"] = budget_exhausted
    failures = _contract_failures(summary, args)
    if budget_exhausted:
        failures.append(f"budget exhausted after ${spent:.2f}; eval incomplete")

    result = {
        "key": key,
        "model": model,
        "prompt_version": prompt_version,
        "profile_checksum": checksum,
        **({"profile_overrides": overrides} if overrides else {}),
        "created_at": _now(),
        "repeats": args.repeats,
        "summary": summary,
        "contract_failures": failures,
        "records": records,
    }
    return _finish(args, result, args.results_dir / f"{key}.json")


if __name__ == "__main__":
    sys.exit(main())
