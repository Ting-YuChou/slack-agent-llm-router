"""D2: the Flink submitter prunes checkpoint directories of superseded runs."""

import json
from types import SimpleNamespace

from flink import submitter

ANALYTICS = next(job for job in submitter.JOBS if job.slug == "inference-analytics")
ENRICHMENT = next(job for job in submitter.JOBS if job.slug == "request-enrichment")


def _jid(n: int) -> str:
    return f"{n:032x}"


def _run_dir(root, job_id, *checkpoints, complete=True):
    job_dir = root / job_id
    (job_dir / "shared").mkdir(parents=True)
    (job_dir / "taskowned").mkdir()
    for number in checkpoints:
        chk = job_dir / f"chk-{number}"
        chk.mkdir()
        if complete:
            (chk / "_metadata").write_bytes(b"x")
    return job_dir


def _history(root, slug, *job_ids):
    """Record runs oldest first, so the last one is the newest."""
    for job_id in job_ids:
        submitter.record_job_id(root, slug, job_id)


def _fixture(tmp_path, *, current_complete=True):
    """Analytics ran 4 times, enrichment twice; savepoints and state exist."""
    analytics = [_jid(1), _jid(2), _jid(3), _jid(4)]
    enrichment = [_jid(11), _jid(12)]
    for job_id in analytics[:-1] + enrichment:
        _run_dir(tmp_path, job_id, 5)
    _run_dir(tmp_path, analytics[-1], 1, complete=current_complete)
    _history(tmp_path, ANALYTICS.slug, *analytics)
    _history(tmp_path, ENRICHMENT.slug, *enrichment)
    savepoint = tmp_path / "savepoints" / "savepoint-abc"
    savepoint.mkdir(parents=True)
    (savepoint / "_metadata").write_bytes(b"x")
    return analytics, enrichment


def _job_dirs(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def test_prune_keeps_the_newest_runs_once_the_current_one_checkpointed(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    foreign = _run_dir(tmp_path, _jid(99), 7)
    state_before = (tmp_path / submitter.STATE_FILE).read_text()

    deleted = submitter.prune_job_checkpoints(tmp_path, keep=2)

    assert sorted(p.name for p in deleted) == sorted([analytics[0], analytics[1]])
    # A job id the submitter never recorded is not ours to delete.
    assert _job_dirs(tmp_path) == sorted(
        [analytics[2], analytics[3], *enrichment, foreign.name, "savepoints"]
    )
    assert (tmp_path / "savepoints" / "savepoint-abc" / "_metadata").is_file()
    assert (tmp_path / submitter.STATE_FILE).read_text() == state_before


def test_prune_waits_for_the_new_run_to_complete_a_checkpoint(tmp_path):
    analytics, enrichment = _fixture(tmp_path, current_complete=False)
    orphan = _run_dir(tmp_path, _jid(99), 7)

    deleted = submitter.prune_job_checkpoints(tmp_path, keep=1)

    # Nothing of analytics (its restore source must survive), and no orphans
    # while any job is still unsettled; enrichment is settled on its own.
    assert sorted(p.name for p in deleted) == [enrichment[0]]
    assert set(_job_dirs(tmp_path)) >= {*analytics, orphan.name}


def test_keep_one_drops_the_restore_source_after_the_first_checkpoint(tmp_path):
    analytics, enrichment = _fixture(tmp_path)

    submitter.prune_job_checkpoints(tmp_path, keep=1)

    assert _job_dirs(tmp_path) == sorted([analytics[-1], enrichment[-1], "savepoints"])


def test_prune_never_deletes_an_active_job_or_non_job_directories(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    (tmp_path / "not-a-job-id").mkdir()
    upper = tmp_path / _jid(0xABCDEF).upper()
    upper.mkdir()
    short = tmp_path / "abc123"
    short.mkdir()
    (tmp_path / _jid(97)).write_text("a file, not a directory")

    submitter.prune_job_checkpoints(tmp_path, keep=1, active_job_ids={analytics[0]})

    remaining = set(_job_dirs(tmp_path))
    assert analytics[0] in remaining
    assert {"not-a-job-id", upper.name, short.name} <= remaining
    assert (tmp_path / _jid(97)).is_file()
    assert analytics[1] not in remaining


def test_prune_is_a_noop_without_history(tmp_path):
    _run_dir(tmp_path, _jid(1), 3)
    _run_dir(tmp_path, _jid(2), 3)

    assert submitter.prune_job_checkpoints(tmp_path, keep=1) == []
    assert _job_dirs(tmp_path) == [_jid(1), _jid(2)]


def test_prune_disabled_with_keep_zero(tmp_path):
    analytics, _ = _fixture(tmp_path)

    assert submitter.prune_job_checkpoints(tmp_path, keep=0) == []
    assert analytics[0] in _job_dirs(tmp_path)


class _RunningCluster:
    def __init__(self, jobs):
        self.jobs = jobs
        self.commands = []

    def fetch_json(self, url):
        return {"jobs": list(self.jobs)}

    def run(self, command, env):
        json.loads(env["FLINK_CONFIG_JSON"])
        self.commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=f"Job has been submitted with JobID {_jid(50)}\n",
            stderr="",
        )


def test_reconcile_prunes_with_the_default_of_two_runs(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    cluster = _RunningCluster(
        [
            {"jid": analytics[-1], "name": ANALYTICS.name, "state": "RUNNING"},
            {"jid": enrichment[-1], "name": ENRICHMENT.name, "state": "RUNNING"},
        ]
    )

    result = submitter.reconcile(
        {"flink": {}, "kafka": {}},
        jobmanager="jm:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
    )

    assert result == {ANALYTICS.slug: "running", ENRICHMENT.slug: "running"}
    assert submitter.DEFAULT_KEEP_JOB_CHECKPOINTS == 2
    assert _job_dirs(tmp_path) == sorted(
        [analytics[2], analytics[3], *enrichment, "savepoints"]
    )


def test_reconcile_keeps_the_restore_source_of_a_fresh_restore(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    cluster = _RunningCluster(
        [{"jid": enrichment[-1], "name": ENRICHMENT.name, "state": "RUNNING"}]
    )

    result = submitter.reconcile(
        {"flink": {}, "kafka": {}},
        jobmanager="jm:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
        keep_job_checkpoints=1,
    )

    assert result[ANALYTICS.slug] == "restored"
    # The new run has no directory, let alone a checkpoint, yet.
    assert analytics[-1] in _job_dirs(tmp_path)
    assert submitter.recorded_job_ids(tmp_path, ANALYTICS.slug)[0] == _jid(50)


def test_reconcile_survives_a_failing_prune(tmp_path, monkeypatch):
    analytics, enrichment = _fixture(tmp_path)
    cluster = _RunningCluster(
        [
            {"jid": analytics[-1], "name": ANALYTICS.name, "state": "RUNNING"},
            {"jid": enrichment[-1], "name": ENRICHMENT.name, "state": "RUNNING"},
        ]
    )

    def broken_rmtree(path, *args, **kwargs):
        raise PermissionError("read-only volume")

    monkeypatch.setattr(submitter.shutil, "rmtree", broken_rmtree)

    result = submitter.reconcile(
        {"flink": {}, "kafka": {}},
        jobmanager="jm:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
    )

    assert result == {ANALYTICS.slug: "running", ENRICHMENT.slug: "running"}
    assert analytics[0] in _job_dirs(tmp_path)


# --- review round 1 -----------------------------------------------------------


def test_runs_evicted_from_the_history_are_still_pruned(tmp_path):
    runs = [_jid(n) for n in range(1, 9)]
    for job_id in runs[:-1]:
        _run_dir(tmp_path, job_id, 3)
    _run_dir(tmp_path, runs[-1], 1)
    _history(tmp_path, ANALYTICS.slug, *runs)
    assert len(submitter.recorded_job_ids(tmp_path, ANALYTICS.slug)) == (
        submitter.JOB_ID_HISTORY
    )
    assert submitter.evicted_job_ids(tmp_path, ANALYTICS.slug) == runs[:3][::-1]
    foreign = _run_dir(tmp_path, _jid(99), 7)

    submitter.prune_job_checkpoints(tmp_path, keep=2)

    assert _job_dirs(tmp_path) == sorted([runs[-2], runs[-1], foreign.name])


class _ScriptedCluster(_RunningCluster):
    """Answers each /jobs/overview call from a list, the last one repeating."""

    def __init__(self, overviews, returncode=0):
        super().__init__([])
        self.overviews = list(overviews)
        self.returncode = returncode

    def fetch_json(self, url):
        overview = (
            self.overviews.pop(0) if len(self.overviews) > 1 else (self.overviews[0])
        )
        if isinstance(overview, Exception):
            raise overview
        return {"jobs": list(overview)}

    def run(self, command, env):
        if self.returncode == 124:
            self.commands.append(command)
            return SimpleNamespace(returncode=124, stdout="", stderr="timed out")
        return super().run(command, env)


def _running(job, jid):
    return {"jid": jid, "name": job.name, "state": "RUNNING"}


def _reconcile_with(tmp_path, cluster, keep=1):
    return submitter.reconcile(
        {"flink": {}, "kafka": {}},
        jobmanager="jm:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
        keep_job_checkpoints=keep,
    )


def test_prune_uses_a_fresh_overview_of_running_jobs(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    initial = [_running(ANALYTICS, analytics[-1]), _running(ENRICHMENT, enrichment[-1])]
    # By the time pruning runs, an old enrichment run is running again.
    later = initial + [_running(ENRICHMENT, enrichment[0])]
    cluster = _ScriptedCluster([initial, later])

    _reconcile_with(tmp_path, cluster)

    remaining = set(_job_dirs(tmp_path))
    assert enrichment[0] in remaining
    assert analytics[0] not in remaining


def test_prune_is_skipped_when_the_fresh_overview_fails(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    initial = [_running(ANALYTICS, analytics[-1]), _running(ENRICHMENT, enrichment[-1])]
    cluster = _ScriptedCluster([initial, OSError("jobmanager went away")])

    result = _reconcile_with(tmp_path, cluster)

    assert result == {ANALYTICS.slug: "running", ENRICHMENT.slug: "running"}
    assert set(_job_dirs(tmp_path)) >= {*analytics, *enrichment}


def test_timed_out_submit_that_started_keeps_its_checkpoints(tmp_path):
    analytics, enrichment = _fixture(tmp_path)
    late = _jid(77)
    _run_dir(tmp_path, late, 1)
    enrichment_only = [_running(ENRICHMENT, enrichment[-1])]
    # The timed-out job shows up only after the submitter gave up on it.
    cluster = _ScriptedCluster(
        [
            enrichment_only,
            enrichment_only,
            enrichment_only + [_running(ANALYTICS, late)],
        ],
        returncode=124,
    )

    result = _reconcile_with(tmp_path, cluster)

    assert result[ANALYTICS.slug] == "failed"
    assert late in _job_dirs(tmp_path)


# --- review round 2 -----------------------------------------------------------


def _eight_runs(tmp_path, with_evicted_dirs=True):
    """Eight analytics runs: five in the history, three evicted from it."""
    runs = [_jid(n) for n in range(1, 9)]
    for index, job_id in enumerate(runs):
        if with_evicted_dirs or index >= 3:
            _run_dir(tmp_path, job_id, 3)
    _history(tmp_path, ANALYTICS.slug, *runs)
    return runs


def test_keep_beyond_the_history_size_also_keeps_evicted_runs(tmp_path):
    runs = _eight_runs(tmp_path)

    submitter.prune_job_checkpoints(tmp_path, keep=6)

    # The six newest survive: five in the history and the newest evicted one.
    assert _job_dirs(tmp_path) == sorted(runs[2:])
    assert submitter.evicted_job_ids(tmp_path, ANALYTICS.slug) == [runs[2]]


def test_evicted_ids_are_forgotten_once_their_directories_are_gone(tmp_path):
    runs = _eight_runs(tmp_path, with_evicted_dirs=False)
    history_before = submitter.recorded_job_ids(tmp_path, ANALYTICS.slug)

    assert submitter.prune_job_checkpoints(tmp_path, keep=2) != []

    assert submitter.evicted_job_ids(tmp_path, ANALYTICS.slug) == []
    assert submitter.recorded_job_ids(tmp_path, ANALYTICS.slug) == history_before
    assert _job_dirs(tmp_path) == sorted(runs[-2:])


def test_pruned_evicted_runs_are_dropped_from_the_state(tmp_path):
    runs = _eight_runs(tmp_path)

    submitter.prune_job_checkpoints(tmp_path, keep=2)

    assert submitter.evicted_job_ids(tmp_path, ANALYTICS.slug) == []
    assert _job_dirs(tmp_path) == sorted(runs[-2:])
