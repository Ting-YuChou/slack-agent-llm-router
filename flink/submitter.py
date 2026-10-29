"""Submit the router's Flink jobs, resuming each from its latest checkpoint.

Runs in the Flink image (compose service ``flink-jobs``)::

    python3 /workspace/flink/submitter.py \\
        --config /workspace/config/config.compose.yaml --watch

For each job that is not running on the cluster it looks up the newest
retained checkpoint of that job's previous runs and submits with ``-s``. The
YAML ``flink`` section and the Kafka topics are passed as ``FLINK_CONFIG_JSON``
so the jobs see the configured checkpoint interval and guardrail thresholds.
With ``--watch`` it re-checks periodically, which resubmits the jobs after a
JobManager restart (the session cluster has no HA store of its own).

Job ids of past runs are kept in ``<checkpoint root>/submitter-state.json``;
Flink writes checkpoints to ``<checkpoint root>/<job id>/chk-<n>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("flink.submitter")

WORKSPACE = Path(__file__).resolve().parents[1]
STATE_FILE = "submitter-state.json"
JOB_ID_HISTORY = 5
# `flink run -d` returns once the job is submitted; a hang past this is a
# failed attempt so the watch loop keeps going.
SUBMIT_TIMEOUT_SECONDS = float(os.environ.get("FLINK_SUBMIT_TIMEOUT_SECONDS", "300"))
ACTIVE_STATES = frozenset(
    {"CREATED", "INITIALIZING", "RUNNING", "RESTARTING", "RECONCILING"}
)
_JOB_ID = re.compile(r"Job has been submitted with JobID (\S+)")
_CHECKPOINT_DIR = re.compile(r"chk-(\d+)$")


@dataclass(frozen=True)
class FlinkJob:
    slug: str
    name: str
    script: str
    consumer_group: str


JOBS = (
    FlinkJob(
        slug="request-enrichment",
        name="LLM Router Request Enrichment Job",
        script="flink/job.py",
        consumer_group="flink-request-enrichment",
    ),
    FlinkJob(
        slug="inference-analytics",
        name="LLM Router Inference Analytics Job",
        script="flink/analytics_job.py",
        consumer_group="flink-inference-analytics",
    ),
)


def build_job_config(
    app_config: Mapping[str, Any],
    job: FlinkJob,
    *,
    parallelism: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the FLINK_CONFIG_JSON payload for one job from the app config."""
    kafka = dict(app_config.get("kafka", {}) or {})
    config: Dict[str, Any] = {
        "flink": dict(app_config.get("flink", {}) or {}),
        "kafka": {
            "bootstrap_servers": list(
                kafka.get("bootstrap_servers", ["kafka:29092"]) or ["kafka:29092"]
            ),
            "consumer_group": job.consumer_group,
            "topics": dict(kafka.get("topics", {}) or {}),
        },
    }
    if parallelism:
        config["parallelism"] = int(parallelism)
    return config


def _state_path(checkpoint_root: Path) -> Path:
    return Path(checkpoint_root) / STATE_FILE


def _read_state(checkpoint_root: Path) -> Dict[str, List[str]]:
    try:
        data = json.loads(_state_path(checkpoint_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return {str(k): [str(i) for i in v] for k, v in data.items() if isinstance(v, list)}


def recorded_job_ids(checkpoint_root: Path, slug: str) -> List[str]:
    """Return the job ids of past runs, newest first."""
    return _read_state(checkpoint_root).get(slug, [])


def record_job_id(checkpoint_root: Path, slug: str, job_id: str) -> None:
    state = _read_state(checkpoint_root)
    history = [job_id] + [i for i in state.get(slug, []) if i != job_id]
    state[slug] = history[:JOB_ID_HISTORY]
    path = _state_path(checkpoint_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def latest_checkpoint(checkpoint_root: Path, job_ids: Sequence[str]) -> Optional[Path]:
    """Return the newest completed checkpoint of the given runs, newest run first."""
    for job_id in job_ids:
        job_dir = Path(checkpoint_root) / job_id
        if not job_dir.is_dir():
            continue
        completed = []
        for child in job_dir.iterdir():
            match = _CHECKPOINT_DIR.search(child.name)
            if match and (child / "_metadata").is_file():
                completed.append((int(match.group(1)), child))
        if completed:
            return max(completed)[1]
    return None


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _run(
    command: Sequence[str],
    env: Mapping[str, str],
    timeout_seconds: float = SUBMIT_TIMEOUT_SECONDS,
):
    """Run a command; a hang past the timeout is reported as a failed attempt."""
    try:
        return subprocess.run(
            list(command),
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            list(command),
            returncode=124,
            stdout=stdout,
            stderr=f"timed out after {timeout_seconds}s",
        )


TIMEOUT_RETURNCODE = 124


def _submit(
    job: FlinkJob,
    job_config: Mapping[str, Any],
    *,
    jobmanager: str,
    restore_from: Optional[Path],
    run: Callable[..., Any],
) -> Optional[str]:
    """Submit one job and return its id, or None if the attempt failed."""
    return _submit_attempt(
        job, job_config, jobmanager=jobmanager, restore_from=restore_from, run=run
    )[0]


def _submit_attempt(
    job: FlinkJob,
    job_config: Mapping[str, Any],
    *,
    jobmanager: str,
    restore_from: Optional[Path],
    run: Callable[..., Any],
) -> Tuple[Optional[str], bool]:
    """Submit one job; return (job id or None, whether the client timed out)."""
    command = ["flink", "run", "-m", jobmanager, "-d"]
    if restore_from is not None:
        command += ["-s", str(restore_from)]
    command += ["-py", str(WORKSPACE / job.script)]
    env = {
        **os.environ,
        "FLINK_CONFIG_JSON": json.dumps(job_config, separators=(",", ":")),
    }
    result = run(command, env)
    returncode = getattr(result, "returncode", 1)
    match = _JOB_ID.search(getattr(result, "stdout", "") or "")
    if returncode != 0 or match is None:
        logger.error(
            "flink run for %s failed (exit %s): %s",
            job.slug,
            returncode,
            (getattr(result, "stderr", "") or "")[-2000:],
        )
        return None, returncode == TIMEOUT_RETURNCODE
    return match.group(1), False


def _active_job_id(
    fetch_json: Callable[[str], Any], jobmanager: str, name: str
) -> Optional[str]:
    """Return the id of an active job with this name, if the cluster has one."""
    try:
        overview = fetch_json(f"http://{jobmanager}/jobs/overview") or {}
    except Exception as exc:
        logger.warning("Could not list Flink jobs: %s", exc)
        return None
    for job in overview.get("jobs", []) or []:
        if job.get("name") == name and job.get("state") in ACTIVE_STATES:
            return str(job.get("jid"))
    return None


def reconcile(
    app_config: Mapping[str, Any],
    *,
    jobmanager: str,
    checkpoint_root: Path,
    parallelism: Optional[int] = None,
    fetch_json: Callable[[str], Any] = _fetch_json,
    run: Callable[..., Any] = _run,
) -> Dict[str, str]:
    """Make sure every job runs; return what happened to each job slug."""
    overview = fetch_json(f"http://{jobmanager}/jobs/overview") or {}
    active = {
        job.get("name")
        for job in overview.get("jobs", []) or []
        if job.get("state") in ACTIVE_STATES
    }
    outcome: Dict[str, str] = {}
    for job in JOBS:
        if job.name in active:
            outcome[job.slug] = "running"
            continue
        job_config = build_job_config(app_config, job, parallelism=parallelism)
        checkpoint = latest_checkpoint(
            checkpoint_root, recorded_job_ids(checkpoint_root, job.slug)
        )
        job_id, timed_out = _submit_attempt(
            job, job_config, jobmanager=jobmanager, restore_from=checkpoint, run=run
        )
        status = "restored" if checkpoint is not None else "submitted"
        if timed_out:
            # The client gave up, but the job may have been accepted. Never
            # start a stateless duplicate: adopt it if it runs, else retry the
            # same restore on the next pass.
            job_id = _active_job_id(fetch_json, jobmanager, job.name)
            status = "adopted"
        elif job_id is None and checkpoint is not None:
            # A checkpoint that no longer matches the job (or is damaged) must
            # not keep analytics down; start fresh and say so loudly.
            logger.warning(
                "Could not restore %s from %s; submitting without state",
                job.slug,
                checkpoint,
            )
            job_id, timed_out = _submit_attempt(
                job, job_config, jobmanager=jobmanager, restore_from=None, run=run
            )
            status = "submitted_fresh_after_restore_failure"
            if timed_out:
                job_id = _active_job_id(fetch_json, jobmanager, job.name)
                status = "adopted"
        if job_id is None:
            outcome[job.slug] = "failed"
            continue
        record_job_id(checkpoint_root, job.slug, job_id)
        logger.info("%s %s as %s", status, job.slug, job_id)
        outcome[job.slug] = status
    return outcome


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # PyYAML ships with PyFlink's apache-beam dependency

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _wait_for_jobmanager(
    jobmanager: str, timeout_seconds: float, fetch_json: Callable[[str], Any]
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fetch_json(f"http://{jobmanager}/overview")
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Flink JobManager {jobmanager} unreachable"
                ) from exc
            time.sleep(2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--jobmanager", default=os.environ.get("FLINK_JOBMANAGER", "localhost:8081")
    )
    parser.add_argument(
        "--checkpoint-root",
        default=os.environ.get("FLINK_CHECKPOINT_ROOT", "/opt/flink/checkpoints"),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=int(os.environ.get("FLINK_JOB_PARALLELISM", "0")) or None,
        help="override flink.parallelism (the local stack has one TaskManager)",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    app_config = _load_yaml(args.config)
    checkpoint_root = Path(args.checkpoint_root)
    _wait_for_jobmanager(args.jobmanager, args.timeout_seconds, _fetch_json)
    while True:
        try:
            outcome = reconcile(
                app_config,
                jobmanager=args.jobmanager,
                checkpoint_root=checkpoint_root,
                parallelism=args.parallelism,
            )
            print(json.dumps(outcome, sort_keys=True), flush=True)
            failed = "failed" in outcome.values()
        except Exception as exc:
            logger.error("Flink job reconcile failed: %s", exc)
            failed = True
        if not args.watch:
            return 1 if failed else 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    sys.exit(main())
