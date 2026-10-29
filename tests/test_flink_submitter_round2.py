"""A8 round 2: a timed-out restore never falls back to a stateless start."""

import json
from types import SimpleNamespace

from flink import submitter

ANALYTICS = next(job for job in submitter.JOBS if job.slug == "inference-analytics")
ENRICHMENT = next(job for job in submitter.JOBS if job.slug == "request-enrichment")


class _Cluster:
    def __init__(self, returncodes, appears_after_timeout):
        self.returncodes = list(returncodes)
        self.appears_after_timeout = appears_after_timeout
        self.jobs = [{"jid": "enrich", "name": ENRICHMENT.name, "state": "RUNNING"}]
        self.commands = []

    def fetch_json(self, url):
        return {"jobs": list(self.jobs)}

    def run(self, command, env):
        json.loads(env["FLINK_CONFIG_JSON"])
        self.commands.append(command)
        code = self.returncodes.pop(0)
        if code == 124:
            if self.appears_after_timeout:
                self.jobs.append(
                    {"jid": "late-jid", "name": ANALYTICS.name, "state": "RUNNING"}
                )
            return SimpleNamespace(returncode=124, stdout="", stderr="timed out")
        if code != 0:
            return SimpleNamespace(returncode=code, stdout="", stderr="rejected")
        return SimpleNamespace(
            returncode=0,
            stdout=f"Job has been submitted with JobID fresh-{len(self.commands)}\n",
            stderr="",
        )


def _with_checkpoint(tmp_path):
    submitter.record_job_id(tmp_path, ANALYTICS.slug, "prev")
    chk = tmp_path / "prev" / "chk-4"
    chk.mkdir(parents=True)
    (chk / "_metadata").write_bytes(b"x")
    return chk


def _reconcile(tmp_path, cluster):
    return submitter.reconcile(
        {"flink": {}, "kafka": {}},
        jobmanager="jm:8081",
        checkpoint_root=tmp_path,
        fetch_json=cluster.fetch_json,
        run=cluster.run,
    )


def test_timed_out_restore_adopts_the_job_that_started(tmp_path):
    _with_checkpoint(tmp_path)
    cluster = _Cluster([124], appears_after_timeout=True)

    result = _reconcile(tmp_path, cluster)

    assert result[ANALYTICS.slug] == "adopted"
    assert len(cluster.commands) == 1
    assert submitter.recorded_job_ids(tmp_path, ANALYTICS.slug)[0] == "late-jid"


def test_timed_out_restore_is_retried_later_not_started_fresh(tmp_path):
    checkpoint = _with_checkpoint(tmp_path)
    cluster = _Cluster([124], appears_after_timeout=False)

    result = _reconcile(tmp_path, cluster)

    assert result[ANALYTICS.slug] == "failed"
    assert len(cluster.commands) == 1
    assert "-s" in cluster.commands[0]
    assert submitter.recorded_job_ids(tmp_path, ANALYTICS.slug) == ["prev"]
    # Next pass restores from the same checkpoint again.
    cluster.returncodes = [0]
    assert _reconcile(tmp_path, cluster)[ANALYTICS.slug] == "restored"
    assert cluster.commands[-1][cluster.commands[-1].index("-s") + 1] == str(checkpoint)


def test_rejected_restore_still_falls_back_to_fresh(tmp_path):
    _with_checkpoint(tmp_path)
    cluster = _Cluster([1, 0], appears_after_timeout=False)

    result = _reconcile(tmp_path, cluster)

    assert result[ANALYTICS.slug] == "submitted_fresh_after_restore_failure"
    assert ["-s" in command for command in cluster.commands] == [True, False]
