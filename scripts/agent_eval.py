"""Live, budgeted coding-agent eval on synthetic fixture repositories.

Each task under ``eval/tasks/<id>/`` has a ``task.json``, a ``fixture/`` tree
that becomes the agent's repository, and a hidden ``oracle/`` that the agent
never sees: after the run it is copied next to the committed tree (or the
baseline when nothing was committed) and executed in the verification image.

Results are written to ``eval/results/<model>__<prompt_version>__<checksum>.json``
and compared against ``eval/baseline.json``; any regression beyond tolerance,
or any missing metric, fails the run (exit 1).
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
# metric -> (direction, default tolerance). Higher-is-better metrics use an
# absolute tolerance; lower-is-better metrics a relative one.
CONTRACT_METRICS = {
    "pass_at_1": ("higher", 0.1),
    "grader_oracle_agreement": ("higher", 0.1),
    "cost_usd_p50": ("lower", 0.25),
    "wall_time_s_p50": ("lower", 0.5),
}
GRADED = ("passed", "failed")
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
    """Return ``{"passed", "reason"}`` for an outcome against ``task["expect"]``."""
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


def aggregate(records):
    runs = len(records)
    passes = sum(1 for record in records if record["passed"])
    graded = [
        record
        for record in records
        if record.get("grader") in GRADED and record.get("oracle") in GRADED
    ]
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
        "tasks": tasks,
    }


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def evaluate_contract(summary, baseline):
    """Compare ``summary`` to ``baseline``; fail closed on any missing metric."""
    failures = []
    expected = (baseline or {}).get("metrics") or {}
    tolerances = (baseline or {}).get("tolerance") or {}
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
            },
        )
    return profile


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
        started = time.monotonic()
        with topology.Topology(
            self.project, root, repo, self.key, root / "profile.json"
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
                booted, created["run_id"], timeout_s=self.timeout_s
            )
        wall_time_s = time.monotonic() - started
        commit = state.get("commit")
        error_code, error_message = run_error(state, secrets=[self.key])
        return {
            "run_id": created["run_id"],
            "fixture": str(root),
            "status": state["status"],
            "verification": state.get("verification_status"),
            "grader": state.get("grader_status"),
            "commit": commit,
            "changed_files": changed_files(repo, baseline, commit),
            "oracle": run_oracle(task, repo, commit or baseline, self.image),
            "cost_usd": state.get("estimated_cost_usd"),
            "wall_time_s": wall_time_s,
            "error_code": error_code,
            "error_message": error_message,
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
        "--tasks-dir", type=pathlib.Path, default=PROJECT / "eval/tasks"
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
        "--baseline", type=pathlib.Path, default=PROJECT / "eval/baseline.json"
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
    args = parser.parse_args(argv)
    if require_live and not args.live_openai:
        parser.error("--live-openai is required; this command incurs model usage")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    return args


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main(argv=None, runner=None):
    args = parse_args(argv, require_live=runner is None)
    try:
        tasks = load_tasks(args.tasks_dir, args.task)
    except TaskError as error:
        print(f"invalid eval task: {error}", file=sys.stderr)
        return 2
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
    if args.update_baseline:
        failures = [
            f"{metric}: missing from eval summary"
            for metric in CONTRACT_METRICS
            if _number(summary.get(metric)) is None
        ]
    else:
        try:
            baseline = json.loads(args.baseline.read_text())
        except (OSError, ValueError):
            baseline = {}
        failures = evaluate_contract(summary, baseline)
    if budget_exhausted:
        failures.append(f"budget exhausted after ${spent:.2f}; eval incomplete")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "key": key,
        "model": model,
        "prompt_version": prompt_version,
        "profile_checksum": checksum,
        "created_at": _now(),
        "repeats": args.repeats,
        "summary": summary,
        "contract_failures": failures,
        "records": records,
    }
    (args.results_dir / f"{key}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.update_baseline and not failures:
        baseline = {
            "key": key,
            "updated_at": result["created_at"],
            "repeats": args.repeats,
            "metrics": {metric: summary[metric] for metric in CONTRACT_METRICS},
            "tolerance": {metric: spec[1] for metric, spec in CONTRACT_METRICS.items()},
        }
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(
        json.dumps(
            {"key": key, "summary": summary, "contract_failures": failures}, indent=2
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
