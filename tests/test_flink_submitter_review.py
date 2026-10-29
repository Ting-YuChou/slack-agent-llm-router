"""A8 review: a hung `flink run` is a failed attempt, not a stuck submitter."""

import sys

from flink import submitter


def test_run_times_out_and_reports_failure():
    result = submitter._run(
        [sys.executable, "-c", "import time; time.sleep(5)"], {}, timeout_seconds=0.2
    )

    assert result.returncode != 0
    assert "timed out" in result.stderr


def test_submit_treats_a_timeout_as_a_failed_attempt(tmp_path):
    job = submitter.JOBS[0]

    def hung(command, env):
        return submitter._run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            env,
            timeout_seconds=0.2,
        )

    job_id = submitter._submit(
        job, {}, jobmanager="jm:8081", restore_from=None, run=hung
    )

    assert job_id is None


def test_default_run_has_a_bounded_timeout():
    assert 0 < submitter.SUBMIT_TIMEOUT_SECONDS <= 900
