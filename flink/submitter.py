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
Flink writes checkpoints to ``<checkpoint root>/<job id>/chk-<n>``. Once a
job's newest run has completed a checkpoint, the directories of its older
runs beyond ``--keep-job-checkpoints`` (default 2) are deleted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger("flink.submitter")

WORKSPACE = Path(__file__).resolve().parents[1]
STATE_FILE = "submitter-state.json"
JOB_ID_HISTORY = 5
# Checkpoint directories of this many newest runs per job survive pruning.
DEFAULT_KEEP_JOB_CHECKPOINTS = 2
# `flink run -d` returns once the job is submitted; a hang past this is a
# failed attempt so the watch loop keeps going.
SUBMIT_TIMEOUT_SECONDS = float(os.environ.get("FLINK_SUBMIT_TIMEOUT_SECONDS", "300"))
ACTIVE_STATES = frozenset(
    {"CREATED", "INITIALIZING", "RUNNING", "RESTARTING", "RECONCILING"}
)
_JOB_ID = re.compile(r"Job has been submitted with JobID (\S+)")
_CHECKPOINT_DIR = re.compile(r"chk-(\d+)$")
# Flink names a job's checkpoint directory after its 32-hex JobID.
_JOB_DIR = re.compile(r"[0-9a-f]{32}")


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


def _has_completed_checkpoint(checkpoint_root: Path, job_id: str) -> bool:
    return latest_checkpoint(checkpoint_root, [job_id]) is not None


def prune_job_checkpoints(
    checkpoint_root: Path,
    keep: int = DEFAULT_KEEP_JOB_CHECKPOINTS,
    active_job_ids: Iterable[str] = (),
) -> List[Path]:
    """Delete checkpoint directories of superseded runs; return what was deleted.

    Each job keeps the directories of its ``keep`` newest recorded runs. A
    job's older runs go only once its newest run has completed a checkpoint,
    so a NO_CLAIM restore source survives until the new run no longer needs
    it. Directories of no recorded run go only once every job is in that
    state. Active jobs, ``savepoints/``, the state file and anything that is
    not a 32-hex job id directory are never touched. ``keep <= 0`` disables
    pruning.
    """
    if keep <= 0:
        return []
    root = Path(checkpoint_root)
    state = _read_state(root)
    histories = [state.get(job.slug, []) for job in JOBS]
    active = {str(job_id) for job_id in active_job_ids}
    protected = set(active)
    deletable = set()
    all_settled = True
    for history in histories:
        protected.update(history)
        if history and _has_completed_checkpoint(root, history[0]):
            deletable.update(history[keep:])
        else:
            all_settled = False

    try:
        children = list(root.iterdir())
    except FileNotFoundError:
        return []
    deleted: List[Path] = []
    for child in sorted(children):
        name = child.name
        if not _JOB_DIR.fullmatch(name) or child.is_symlink() or not child.is_dir():
            continue
        if name in active:
            continue
        if name in deletable or (all_settled and name not in protected):
            try:
                shutil.rmtree(child)
            except OSError as exc:
                logger.warning("Could not delete old checkpoints %s: %s", child, exc)
                continue
            deleted.append(child)
    if deleted:
        logger.info(
            "Pruned checkpoints of %d superseded Flink run(s): %s",
            len(deleted),
            ", ".join(path.name for path in deleted),
        )
    return deleted


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
    keep_job_checkpoints: int = DEFAULT_KEEP_JOB_CHECKPOINTS,
) -> Dict[str, str]:
    """Make sure every job runs; return what happened to each job slug.

    Afterwards, prune checkpoint directories of superseded runs (see
    ``prune_job_checkpoints``); a failed prune only logs.
    """
    overview = fetch_json(f"http://{jobmanager}/jobs/overview") or {}
    active_jobs = [
        job
        for job in overview.get("jobs", []) or []
        if job.get("state") in ACTIVE_STATES
    ]
    active = {job.get("name") for job in active_jobs}
    active_job_ids = {str(job.get("jid")) for job in active_jobs if job.get("jid")}
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
        active_job_ids.add(job_id)
        logger.info("%s %s as %s", status, job.slug, job_id)
        outcome[job.slug] = status
    try:
        prune_job_checkpoints(
            checkpoint_root, keep=keep_job_checkpoints, active_job_ids=active_job_ids
        )
    except Exception as exc:
        logger.warning("Flink checkpoint pruning failed: %s", exc)
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
    parser.add_argument(
        "--keep-job-checkpoints",
        type=int,
        default=int(
            os.environ.get(
                "FLINK_KEEP_JOB_CHECKPOINTS", str(DEFAULT_KEEP_JOB_CHECKPOINTS)
            )
        ),
        help="checkpoint directories of this many newest runs per job are kept "
        "(0 disables pruning)",
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
                keep_job_checkpoints=args.keep_job_checkpoints,
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
