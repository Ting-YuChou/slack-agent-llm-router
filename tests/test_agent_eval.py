import http.server
import json
import shutil
import subprocess
import threading
import uuid

import pytest

from scripts import agent_eval
from scripts import validate_structured_workflow as validator


def _acceptance_report(**overrides):
    report = {
        "status": "completed",
        "verification": "passed",
        "grader": "passed",
    }
    report.update(overrides)
    return report


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "failed"},
        {"status": "timed_out"},
        {"verification": "failed"},
        {"verification": "unverified"},
        {"grader": "failed"},
        {"grader": None},
        {"status": "starting"},
    ],
)
def test_validator_main_fails_when_run_or_gates_did_not_pass(monkeypatch, overrides):
    monkeypatch.setattr(
        validator, "run_acceptance", lambda args: _acceptance_report(**overrides)
    )

    assert validator.main(["--live-openai"]) == 1


def test_validator_main_succeeds_only_for_completed_and_passed_run(monkeypatch):
    monkeypatch.setattr(validator, "run_acceptance", lambda args: _acceptance_report())

    assert validator.main(["--live-openai"]) == 0


def test_validator_refuses_to_run_without_live_opt_in():
    with pytest.raises(SystemExit) as raised:
        validator.main([])

    assert raised.value.code == 2


# --- validator topology extraction -------------------------------------------


class _FakeTopology:
    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def request(self, path, body=None):
        self.calls.append((path, body))
        if path.endswith("/decisions") or path.endswith("/cancel"):
            return {}
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def test_drive_run_approves_each_pending_approval_once_until_terminal():
    approval = {"type": "approval", "approval_id": "a1"}
    topology = _FakeTopology(
        [
            {"status": "awaiting_approval", "events": [approval]},
            {"status": "awaiting_approval", "events": [approval]},
            {"status": "completed", "events": [approval]},
        ]
    )

    state, approvals = validator.drive_run(topology, "run-1", timeout_s=5, poll_s=0)

    assert state["status"] == "completed"
    assert approvals == {"a1"}
    decisions = [body for path, body in topology.calls if path.endswith("/decisions")]
    assert decisions == [
        {"user_id": "fixture-owner", "approval_id": "a1", "decision": "approve"}
    ]


def test_drive_run_cancels_a_run_that_outlives_the_deadline():
    topology = _FakeTopology([{"status": "running", "events": []}])

    state, _ = validator.drive_run(topology, "run-1", timeout_s=0, poll_s=0)

    assert ("/v1/runs/run-1/cancel", {"user_id": "fixture-owner"}) in topology.calls
    assert state["status"] == "running"


def test_collect_spans_ignores_malformed_otel_lines(tmp_path):
    (tmp_path / "otel.jsonl").write_text(
        "not json\n"
        + json.dumps(
            {
                "resourceSpans": [
                    {"scopeSpans": [{"spans": [{"name": "run", "traceId": "t"}]}]}
                ]
            }
        )
        + "\n"
    )

    assert validator.collect_spans(tmp_path) == [{"name": "run", "traceId": "t"}]
    assert validator.collect_spans(tmp_path / "missing") == []


def _inspect(*configs):
    return json.dumps([{"Name": "net", "IPAM": {"Config": list(configs)}}])


def test_bridge_gateway_returns_the_ipv4_gateway():
    output = _inspect(
        {"Subnet": "fd00::/64", "Gateway": "fd00::1"},
        {"Subnet": "172.20.0.0/16", "Gateway": "172.20.0.1"},
    )

    assert validator.bridge_gateway(output) == "172.20.0.1"


@pytest.mark.parametrize(
    "output",
    [
        "[]",
        _inspect(),
        _inspect({"Subnet": "fd00::/64", "Gateway": "fd00::1"}),
        json.dumps([{"Name": "net", "IPAM": {"Config": None}}]),
    ],
)
def test_bridge_gateway_fails_without_an_ipv4_gateway(output):
    with pytest.raises(RuntimeError):
        validator.bridge_gateway(output)


@pytest.mark.parametrize(
    "platform, docker_os",
    [
        ("darwin", "Docker Desktop"),
        ("darwin", "OrbStack"),
        ("linux", "Docker Desktop"),
        ("win32", "Docker Desktop"),
    ],
)
def test_host_binding_keeps_loopback_where_docker_forwards_the_host_alias(
    platform, docker_os
):
    assert validator.host_binding(platform, docker_os, "172.20.0.1") == (
        "127.0.0.1",
        [],
    )


def test_host_binding_uses_the_bridge_gateway_on_a_native_linux_daemon():
    assert validator.host_binding("linux", "Ubuntu 24.04.1 LTS", "172.20.0.1") == (
        "172.20.0.1",
        ["--add-host", "host.docker.internal:172.20.0.1"],
    )


def _docker_ready():
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True)
    return probe.returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="needs a running Docker daemon")
def test_container_on_topology_networks_reaches_host_runtime_through_alias():
    """Same network layout as Topology.start(): the gateway container starts on an
    --internal network, then joins an egress network, and calls the host runtime
    at http://host.docker.internal:<port>. No model calls."""
    suffix = uuid.uuid4().hex[:8]
    internal = "pi-structured-smoke-" + suffix
    external = internal + "-egress"
    container = "pi-structured-smoke-" + suffix

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"budget-rpc-ok"
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = None
    try:
        validator.docker("network", "create", "--internal", internal)
        validator.docker("network", "create", external)
        listen_host, add_host = validator.host_access(external)
        server = http.server.ThreadingHTTPServer((listen_host, 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        validator.docker(
            "create",
            "--name",
            container,
            "--network",
            internal,
            *add_host,
            "busybox:1.36",
            "wget",
            "-q",
            "-T",
            "10",
            "-O",
            "-",
            f"http://{validator.HOST_ALIAS}:{port}/",
        )
        validator.docker("network", "connect", external, container)
        result = subprocess.run(
            ["docker", "start", "-a", container],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "budget-rpc-ok"
    finally:
        if server:
            server.shutdown()
            server.server_close()
        for args in (
            ["rm", "-f", container],
            ["network", "rm", external],
            ["network", "rm", internal],
        ):
            subprocess.run(["docker", *args], capture_output=True)


# --- eval runner ---------------------------------------------------------------


def _write_task(root, task_id="t1", expect="pass", oracle=True, **fields):
    task_dir = root / task_id
    (task_dir / "fixture/src").mkdir(parents=True)
    (task_dir / "fixture/src/app.cjs").write_text("exports.x = 1;\n")
    if oracle:
        (task_dir / "oracle").mkdir()
        (task_dir / "oracle/hidden.test.cjs").write_text("// hidden assertions\n")
    task = {
        "id": task_id,
        "prompt": "Fix the bug.",
        "task_mode": "implement",
        "workflow_mode": "single",
        "expect": expect,
        "profile": {
            "allowedPaths": ["src/"],
            "baselinePaths": ["tests/"],
            "rubric": ["Fixes the bug"],
            "checks": [
                {"name": "unit", "argv": ["node", "--test"], "cwd": ".", "timeoutMs": 1}
            ],
        },
        "oracle": {"argv": ["node", "--test", ".eval-oracle/hidden.test.cjs"]},
    }
    task.update(fields)
    (task_dir / "task.json").write_text(json.dumps(task))
    return task_dir


def test_load_task_returns_task_with_fixture_and_oracle_paths(tmp_path):
    task = agent_eval.load_task(_write_task(tmp_path))

    assert task["id"] == "t1"
    assert task["fixture_dir"] == tmp_path / "t1/fixture"
    assert task["oracle_dir"] == tmp_path / "t1/oracle"
    assert task["oracle_files"] == ["hidden.test.cjs"]


def test_load_task_rejects_a_task_without_hidden_oracle(tmp_path):
    with pytest.raises(agent_eval.TaskError, match="oracle"):
        agent_eval.load_task(_write_task(tmp_path, oracle=False))


def test_load_task_rejects_an_empty_oracle_directory(tmp_path):
    task_dir = _write_task(tmp_path)
    (task_dir / "oracle/hidden.test.cjs").unlink()

    with pytest.raises(agent_eval.TaskError, match="oracle"):
        agent_eval.load_task(task_dir)


def test_load_task_rejects_oracle_files_leaked_into_the_fixture(tmp_path):
    task_dir = _write_task(tmp_path)
    (task_dir / "fixture/hidden.test.cjs").write_text("// hidden assertions\n")

    with pytest.raises(agent_eval.TaskError, match="fixture"):
        agent_eval.load_task(task_dir)


def test_load_task_rejects_oracle_content_copied_under_another_name(tmp_path):
    task_dir = _write_task(tmp_path)
    (task_dir / "fixture/src/copy.cjs").write_text("// hidden assertions\n")

    with pytest.raises(agent_eval.TaskError, match="fixture"):
        agent_eval.load_task(task_dir)


def test_load_task_rejects_unknown_expectation(tmp_path):
    with pytest.raises(agent_eval.TaskError, match="expect"):
        agent_eval.load_task(_write_task(tmp_path, expect="maybe"))


def test_load_task_rejects_task_id_that_does_not_match_directory(tmp_path):
    with pytest.raises(agent_eval.TaskError, match="id"):
        agent_eval.load_task(_write_task(tmp_path, id="other"))


def _task(expect):
    return {"id": "t1", "expect": expect, "profile": {"allowedPaths": ["src/"]}}


def _outcome(**overrides):
    outcome = {
        "status": "completed",
        "verification": "passed",
        "grader": "passed",
        "commit": "abc123",
        "changed_files": ["src/app.cjs"],
        "oracle": "passed",
    }
    outcome.update(overrides)
    return outcome


@pytest.mark.parametrize(
    "expect, overrides, passed",
    [
        ("pass", {}, True),
        ("pass", {"oracle": "failed"}, False),
        ("pass", {"oracle": None}, False),
        ("pass", {"status": "failed"}, False),
        ("pass", {"commit": None, "changed_files": []}, False),
        ("pass", {"changed_files": ["src/app.cjs", "tests/a.test.cjs"]}, False),
        ("must_not_commit", {"commit": None, "changed_files": []}, True),
        ("must_not_commit", {}, False),
        ("scope_violation", {"status": "failed", "commit": None}, True),
        ("scope_violation", {}, True),
        ("scope_violation", {"changed_files": ["vendor/lib.cjs"]}, False),
        ("no_diff", {"commit": None, "changed_files": []}, True),
        ("no_diff", {}, False),
        ("no_diff", {"status": "failed", "commit": None, "changed_files": []}, False),
    ],
)
def test_score_run_matches_outcome_against_expectation(expect, overrides, passed):
    score = agent_eval.score_run(_task(expect), _outcome(**overrides))

    assert score["passed"] is passed
    assert score["reason"]


def test_score_run_treats_allowed_path_as_directory_prefix_not_string_prefix():
    score = agent_eval.score_run(
        _task("scope_violation"), _outcome(changed_files=["src-other/x.cjs"])
    )

    assert score["passed"] is False


def _record(task_id="t1", passed=True, grader="passed", oracle="passed", **extra):
    record = {
        "task_id": task_id,
        "passed": passed,
        "grader": grader,
        "oracle": oracle,
        "cost_usd": 0.2,
        "wall_time_s": 60.0,
    }
    record.update(extra)
    return record


def test_aggregate_reports_pass_at_1_agreement_cost_and_wall_time():
    summary = agent_eval.aggregate(
        [
            _record(cost_usd=0.1, wall_time_s=10.0),
            _record(passed=False, grader="passed", oracle="failed", cost_usd=0.3),
            _record("t2", cost_usd=0.2, wall_time_s=30.0),
            _record("t2", passed=False, grader="failed", oracle="failed"),
        ]
    )

    assert summary["runs"] == 4
    assert summary["pass_at_1"] == 0.5
    assert summary["grader_oracle_agreement"] == 0.75
    assert summary["cost_usd_p50"] == 0.2
    assert summary["wall_time_s_p50"] == 45.0
    assert summary["wall_time_s_total"] == 160.0
    assert summary["tasks"]["t1"] == {
        "runs": 2,
        "passes": 1,
        "pass_at_1": 0.5,
        "failure_codes": {"no_error_code": 1},
    }


def test_aggregate_excludes_runs_without_both_verdicts_from_agreement():
    summary = agent_eval.aggregate(
        [
            _record(grader="not_applicable"),
            _record(oracle=None),
            _record(grader="failed", oracle="passed"),
        ]
    )

    assert summary["grader_oracle_agreement"] == 0.0


def test_aggregate_of_no_runs_leaves_metrics_missing():
    summary = agent_eval.aggregate([])

    assert summary["runs"] == 0
    for metric in agent_eval.CONTRACT_METRICS:
        assert summary[metric] is None


def test_aggregate_reports_a_per_task_histogram_of_failure_codes():
    summary = agent_eval.aggregate(
        [
            _record(passed=False, error_code="agent_tool_limit"),
            _record(passed=False, error_code="agent_tool_limit"),
            _record(passed=False, error_code="quality_gate_unavailable"),
            _record(passed=False),
            _record(error_code="scope_violation"),
            _record("t2"),
        ]
    )

    assert summary["tasks"]["t1"]["failure_codes"] == {
        "agent_tool_limit": 2,
        "quality_gate_unavailable": 1,
        "no_error_code": 1,
    }
    assert summary["tasks"]["t2"]["failure_codes"] == {}


def test_failure_codes_leave_the_contract_metrics_unchanged():
    records = [
        _record(passed=False, grader="failed", oracle="failed", cost_usd=0.3),
        _record(cost_usd=0.1, wall_time_s=10.0),
    ]
    coded = [dict(record, error_code="agent_tool_limit") for record in records]

    plain, with_codes = agent_eval.aggregate(records), agent_eval.aggregate(coded)

    for metric in agent_eval.CONTRACT_METRICS:
        assert with_codes[metric] == plain[metric]
    assert set(agent_eval.CONTRACT_METRICS) == {
        "pass_at_1",
        "grader_oracle_agreement",
        "cost_usd_p50",
        "wall_time_s_p50",
        "grader_unknown_rate",
        "first_attempt_pass_rate",
    }


# --- run error capture ---------------------------------------------------------


def test_run_error_reads_code_and_message_from_the_final_run_state():
    state = {
        "status": "failed",
        "error": {"code": "agent_tool_limit", "message": "Tool call limit reached"},
    }

    assert agent_eval.run_error(state) == (
        "agent_tool_limit",
        "Tool call limit reached",
    )


@pytest.mark.parametrize(
    "state",
    [
        {"status": "completed"},
        {"status": "failed", "error": None},
        {"status": "failed", "error": "boom"},
        {"status": "failed", "error": {"code": 7, "message": ["x"]}},
    ],
)
def test_run_error_is_empty_without_a_well_formed_error(state):
    assert agent_eval.run_error(state) == (None, None)


def test_run_error_message_is_redacted_and_bounded():
    secret = "sk-proj-" + "A1b2C3d4" * 6
    message = (
        f"upstream rejected {secret}; Authorization: Bearer abc.def-ghi "
        "OPENAI_API_KEY=hunter2\n" + "x" * 5000
    )

    code, redacted = agent_eval.run_error(
        {"error": {"code": "upstream_unavailable", "message": message}},
        secrets=["hunter2"],
    )

    assert code == "upstream_unavailable"
    assert len(redacted) <= agent_eval.ERROR_MESSAGE_LIMIT
    assert secret not in redacted
    assert "abc.def-ghi" not in redacted
    assert "hunter2" not in redacted
    assert "[REDACTED]" in redacted
    assert redacted.startswith("upstream rejected")


def test_redact_masks_known_token_shapes_without_explicit_secrets():
    text = agent_eval.redact_text(
        "slack xoxb-1234-5678-abcdefgh github ghp_" + "a" * 36 + " ok"
    )

    assert "xoxb-1234" not in text
    assert "ghp_" not in text
    assert text.endswith(" ok")


def test_execute_records_the_run_error_code_and_message():
    def runner(task, attempt):
        return _outcome(
            status="failed",
            commit=None,
            changed_files=[],
            oracle="failed",
            error_code="quality_gate_unavailable",
            error_message="Quality gate could not run",
        )

    record = agent_eval.execute(_task("pass"), 0, runner)

    assert record["passed"] is False
    assert record["error_code"] == "quality_gate_unavailable"
    assert record["error_message"] == "Quality gate could not run"


def test_execute_records_no_error_for_a_clean_run():
    record = agent_eval.execute(_task("pass"), 0, lambda task, attempt: _outcome())

    assert record["error_code"] is None
    assert record["error_message"] is None


def test_execute_records_the_prompts_version_the_run_was_built_with():
    record = agent_eval.execute(
        _task("pass"),
        0,
        lambda task, attempt: _outcome(prompts_version="6da0aeafa863"),
    )

    assert record["prompts_version"] == "6da0aeafa863"
    unknown = agent_eval.execute(_task("pass"), 0, lambda task, attempt: _outcome())
    assert unknown["prompts_version"] is None


def test_execute_records_approvals_checks_and_policy_from_the_run():
    approvals = {
        "requested": 1,
        "approved": 1,
        "rejected": 0,
        "expired": 0,
        "wait_ms": 900,
    }
    policy = {"allow": 3, "block": 1, "approval": 1, "approved": 1, "by_reason": {}}
    record = agent_eval.execute(
        _task("pass"),
        0,
        lambda task, attempt: _outcome(
            approvals=approvals, writer_checks_calls=2, policy_summary=policy
        ),
    )

    assert record["approvals"] == approvals
    assert record["writer_checks_calls"] == 2
    assert record["policy_summary"] == policy
    bare = agent_eval.execute(_task("pass"), 0, lambda task, attempt: _outcome())
    assert bare["approvals"] is None
    assert bare["writer_checks_calls"] is None
    assert bare["policy_summary"] is None


def test_execute_records_a_crashed_runner_as_runner_error():
    def runner(task, attempt):
        raise RuntimeError("docker exploded with Bearer leaked.token")

    record = agent_eval.execute(_task("pass"), 0, runner)

    assert record["error_code"] == "runner_error"
    assert "docker exploded" in record["error_message"]
    assert "leaked.token" not in record["error_message"]
    assert "leaked.token" not in record["reason"]


def _baseline(**metrics):
    values = {
        "pass_at_1": 0.75,
        "grader_oracle_agreement": 0.9,
        "cost_usd_p50": 0.4,
        "wall_time_s_p50": 300.0,
        "grader_unknown_rate": 0.2,
        "first_attempt_pass_rate": 0.75,
    }
    values.update(metrics)
    return {"metrics": values}


def _summary(**metrics):
    summary = dict(_baseline()["metrics"])
    summary.update(metrics)
    return summary


def test_evaluate_contract_accepts_a_run_within_tolerance():
    assert agent_eval.evaluate_contract(_summary(), _baseline()) == []
    assert (
        agent_eval.evaluate_contract(
            _summary(pass_at_1=0.7, cost_usd_p50=0.45), _baseline()
        )
        == []
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {"pass_at_1": 0.5},
        {"grader_oracle_agreement": 0.6},
        {"cost_usd_p50": 0.8},
        {"wall_time_s_p50": 900.0},
        {"grader_unknown_rate": 0.35},
        {"first_attempt_pass_rate": 0.55},
    ],
)
def test_evaluate_contract_flags_regressions_beyond_tolerance(metrics):
    failures = agent_eval.evaluate_contract(_summary(**metrics), _baseline())

    assert len(failures) == 1
    assert next(iter(metrics)) in failures[0]


@pytest.mark.parametrize("metric", ["pass_at_1", "cost_usd_p50"])
def test_evaluate_contract_fails_closed_when_summary_metric_is_missing(metric):
    summary = _summary()
    del summary[metric]

    assert any(
        metric in failure
        for failure in agent_eval.evaluate_contract(summary, _baseline())
    )
    assert agent_eval.evaluate_contract(_summary(**{metric: None}), _baseline())


def test_evaluate_contract_fails_closed_when_baseline_metric_is_missing():
    baseline = _baseline()
    del baseline["metrics"]["grader_oracle_agreement"]

    failures = agent_eval.evaluate_contract(_summary(), baseline)

    assert any("grader_oracle_agreement" in failure for failure in failures)
    assert agent_eval.evaluate_contract(_summary(), {})


def test_first_attempt_pass_rate_is_a_contract_metric_with_room_for_noise():
    assert (
        agent_eval.evaluate_contract(
            _summary(first_attempt_pass_rate=0.62), _baseline()
        )
        == []
    )
    baseline = _baseline()
    del baseline["metrics"]["first_attempt_pass_rate"]
    assert agent_eval.evaluate_contract(_summary(), baseline) == [
        "first_attempt_pass_rate: missing from baseline"
    ]


def test_aggregate_reports_harness_signals_as_informational_medians():
    def record(tool_util, approvals, blocks):
        return {
            **_record(passed=True),
            "harness_metrics": {"tool_util": tool_util},
            "approvals": {"requested": approvals},
            "policy_summary": {"block": blocks},
        }

    summary = agent_eval.aggregate(
        [record(0.25, 1, 0), record(0.5, 1, 2), record(0.75, 3, 1)]
    )

    assert summary["tool_util_p50"] == 0.5
    assert summary["approvals_p50"] == 1
    assert summary["policy_blocks_p50"] == 1
    assert not set(agent_eval.CONTRACT_METRICS) & {
        "tool_util_p50",
        "approvals_p50",
        "policy_blocks_p50",
    }
    bare = agent_eval.aggregate([_record(passed=True)])
    assert "tool_util_p50" not in bare


def test_evaluate_contract_honours_baseline_tolerance_override():
    baseline = _baseline()
    baseline["tolerance"] = {"pass_at_1": 0.0}

    assert agent_eval.evaluate_contract(_summary(pass_at_1=0.74), baseline)


def test_result_key_combines_model_prompt_version_and_profile_checksum():
    key = agent_eval.result_key("openai/gpt-5.6-luna", "v3", "a" * 64)

    assert key == "openai_gpt-5.6-luna__v3__" + "a" * 12


# --- eval runner main ----------------------------------------------------------


def _eval_args(tmp_path, *extra):
    return [
        "--tasks-dir",
        str(tmp_path / "tasks"),
        "--results-dir",
        str(tmp_path / "results"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--model",
        "m",
        "--prompt-version",
        "p",
        *extra,
    ]


def _stub_runner(outcome=None, cost=0.1):
    calls = []

    def runner(task, attempt):
        calls.append((task["id"], attempt))
        result = _outcome(
            cost_usd=cost,
            wall_time_s=1.0,
            attempts=[{"attempt": 1, "verification": "passed", "grader": "passed"}],
        )
        result.update(outcome or {})
        return result

    runner.calls = calls
    return runner


def _write_baseline(tmp_path, **metrics):
    baseline = _baseline(
        pass_at_1=1.0,
        grader_oracle_agreement=1.0,
        cost_usd_p50=0.1,
        wall_time_s_p50=1.0,
        grader_unknown_rate=0.0,
        first_attempt_pass_rate=1.0,
    )
    baseline["metrics"].update(metrics)
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))


def test_eval_main_exits_zero_and_writes_keyed_results_when_contract_holds(tmp_path):
    _write_task(tmp_path / "tasks")
    _write_baseline(tmp_path)
    runner = _stub_runner()

    code = agent_eval.main(_eval_args(tmp_path, "--repeats", "2"), runner=runner)

    assert code == 0
    assert runner.calls == [("t1", 0), ("t1", 1)]
    [result_file] = (tmp_path / "results").glob("*.json")
    assert result_file.name.startswith("m__p__")
    result = json.loads(result_file.read_text())
    assert result["summary"]["runs"] == 2
    assert result["contract_failures"] == []
    assert [record["task_id"] for record in result["records"]] == ["t1", "t1"]


def test_eval_main_exits_one_when_a_run_regresses(tmp_path):
    _write_task(tmp_path / "tasks")
    _write_baseline(tmp_path)

    code = agent_eval.main(
        _eval_args(tmp_path), runner=_stub_runner({"oracle": "failed"})
    )

    assert code == 1


def test_eval_main_exits_one_without_a_baseline(tmp_path):
    _write_task(tmp_path / "tasks")

    assert agent_eval.main(_eval_args(tmp_path), runner=_stub_runner()) == 1


def test_eval_main_counts_a_crashed_run_as_failed(tmp_path):
    _write_task(tmp_path / "tasks")
    _write_baseline(tmp_path)

    def runner(task, attempt):
        raise RuntimeError("docker exploded")

    assert agent_eval.main(_eval_args(tmp_path), runner=runner) == 1
    [result_file] = (tmp_path / "results").glob("*.json")
    [record] = json.loads(result_file.read_text())["records"]
    assert record["passed"] is False
    assert "docker exploded" in record["reason"]


def test_eval_main_stops_and_fails_once_the_budget_is_spent(tmp_path):
    _write_task(tmp_path / "tasks")
    _write_baseline(tmp_path)
    runner = _stub_runner(cost=0.6)

    code = agent_eval.main(
        _eval_args(tmp_path, "--repeats", "3", "--max-cost-usd", "1"), runner=runner
    )

    assert code == 1
    assert len(runner.calls) == 2


def test_eval_main_update_baseline_writes_summary_metrics(tmp_path):
    _write_task(tmp_path / "tasks")

    code = agent_eval.main(
        _eval_args(tmp_path, "--update-baseline"), runner=_stub_runner()
    )

    assert code == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["metrics"]["pass_at_1"] == 1.0
    assert baseline["key"].startswith("m__p__")


def test_eval_main_rejects_invalid_task_before_spending(tmp_path):
    _write_task(tmp_path / "tasks", oracle=False)
    runner = _stub_runner()

    assert agent_eval.main(_eval_args(tmp_path), runner=runner) == 2
    assert runner.calls == []


def test_eval_main_requires_live_opt_in_without_injected_runner(tmp_path):
    with pytest.raises(SystemExit) as raised:
        agent_eval.main(_eval_args(tmp_path))

    assert raised.value.code == 2


def test_repository_eval_tasks_are_valid_and_match_the_plan():
    tasks = {
        path.name: agent_eval.load_task(path)
        for path in sorted((agent_eval.PROJECT / "eval/tasks").iterdir())
    }

    assert {name: task["expect"] for name, task in tasks.items()} == {
        "two-bug-fix": "pass",
        "test-weakening-bait": "pass",
        "scope-bait": "scope_violation",
        "discuss-no-diff": "no_diff",
    }
    assert tasks["discuss-no-diff"]["task_mode"] == "discuss"
    assert "tests/" in tasks["test-weakening-bait"]["profile"]["allowedPaths"]
    assert not any(
        agent_eval._in_scope("vendor/format.cjs", [path])
        for path in tasks["scope-bait"]["profile"]["allowedPaths"]
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_run_oracle_judges_the_committed_tree_without_exposing_the_oracle(
    tmp_path, monkeypatch
):
    task = agent_eval.load_task(agent_eval.PROJECT / "eval/tasks/two-bug-fix")
    repo = tmp_path / "repo"
    shutil.copytree(task["fixture_dir"], repo)
    baseline = validator.init_fixture_repo(repo)
    (repo / "src/counter.cjs").write_text("exports.increment = n => n + 1;\n")
    (repo / "src/label.cjs").write_text("exports.label = n => 'count:' + n;\n")
    validator.git(
        repo, "-c", "user.name=T", "-c", "user.email=t@e", "commit", "-qam", "fix"
    )
    commit = validator.git(repo, "rev-parse", "HEAD")
    real_run = subprocess.run
    docker_calls = []

    def fake_run(argv, **kwargs):
        if argv[0] != "docker":
            return real_run(argv, **kwargs)
        docker_calls.append(argv)
        mount = next(part for part in argv if part.startswith("type=bind"))
        source = mount.split("src=")[1].split(",")[0]
        image_index = argv.index("image-id")
        return real_run(argv[image_index + 1 :], cwd=source, **kwargs)

    monkeypatch.setattr(agent_eval.subprocess, "run", fake_run)

    assert agent_eval.run_oracle(task, repo, commit, "image-id") == "passed"
    assert agent_eval.run_oracle(task, repo, baseline, "image-id") == "failed"
    assert "--network" in docker_calls[0] and "none" in docker_calls[0]
    assert not (repo / agent_eval.ORACLE_DIR).exists()
    assert agent_eval.changed_files(repo, baseline, commit) == [
        "src/counter.cjs",
        "src/label.cjs",
    ]
    assert agent_eval.changed_files(repo, baseline, None) == []


def test_live_runner_takes_the_error_from_the_final_run_state(tmp_path, monkeypatch):
    task = {
        "id": "t1",
        "fixture_dir": tmp_path / "fixture",
        "task_mode": "implement",
        "workflow_mode": "single",
        "prompt": "fix it",
        "profile": {"allowedPaths": ["src/"]},
    }
    task["fixture_dir"].mkdir()
    (task["fixture_dir"] / "a.txt").write_text("a")
    final = {
        "status": "failed",
        "verification_status": "unknown",
        "grader_status": "unknown",
        "estimated_cost_usd": 0.05,
        "prompts_version": "6da0aeafa863",
        "error": {
            "code": "agent_tool_limit",
            "message": "Tool limit hit using sk-live-secret-key",
            "cause_code": "tool_budget",
        },
        "grader_cause": "grader_timeout",
        "attempts": [
            {
                "attempt": 1,
                "execution_id": "e1",
                "verification_status": "failed",
                "grader_status": "not_run",
            },
            {"attempt": 2, "execution_id": "e2"},
        ],
    }

    class FakeTopology:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def request(self, path, body=None):
            return {"run_id": "run-1"}

    monkeypatch.setattr(agent_eval.topology, "make_root", lambda prefix: tmp_path)
    monkeypatch.setattr(
        agent_eval.topology, "init_fixture_repo", lambda repo, remote: "base"
    )
    monkeypatch.setattr(agent_eval.topology, "Topology", FakeTopology)
    monkeypatch.setattr(
        agent_eval.topology,
        "drive_run",
        lambda booted, run_id, timeout_s: (final, set()),
    )
    monkeypatch.setattr(agent_eval, "run_oracle", lambda *args: "failed")
    runner = object.__new__(agent_eval.LiveRunner)
    runner.project, runner.timeout_s = tmp_path, 1
    runner.key, runner.image = "sk-live-secret-key", "image-id"

    outcome = runner(task, 0)

    assert outcome["error_code"] == "agent_tool_limit"
    assert outcome["error_message"] == "Tool limit hit using [REDACTED]"
    assert outcome["error_cause_code"] == "tool_budget"
    assert outcome["grader_cause"] == "grader_timeout"
    assert outcome["prompts_version"] == "6da0aeafa863"
    assert outcome["attempts"] == [
        {"verification": "failed", "grader": "not_run"},
        {"verification": None, "grader": None},
    ]


# --- grader health and recomputation (EV1) -------------------------------------

LIVE_RESULTS = (
    agent_eval.PROJECT
    / "eval/results/openai_gpt-5.6-luna__158a4739d12c__035297f99f3f.json"
)


# Phase 2 records carry per-attempt gates, so they can set every contract metric.
ATTEMPT_RESULTS = (
    agent_eval.PROJECT
    / "eval/results/openai_gpt-5.6-luna__a4f39169832a__035297f99f3f.json"
)


def _live_records():
    return json.loads(LIVE_RESULTS.read_text())["records"]


def test_aggregate_of_the_committed_live_run_reports_grader_unknown_rate():
    summary = agent_eval.aggregate(_live_records())

    # 9 implement runs where the grader ran or should have run; 5 said unknown.
    assert summary["grader_unknown_rate"] == pytest.approx(5 / 9)
    assert summary["grader_oracle_agreement"] == 1.0


def test_grader_unknown_rate_counts_only_runs_where_the_grader_should_run():
    summary = agent_eval.aggregate(
        [
            _record(verification="not_applicable", grader="not_applicable"),
            _record(verification="failed", grader="not_run"),
            _record(verification="passed", grader="unknown"),
            _record(verification="unknown", grader="unknown"),
            _record(verification="passed", grader="needs_changes"),
            _record(verification="passed", grader="passed"),
        ]
    )

    assert summary["grader_unknown_rate"] == 0.5


def test_grader_unknown_rate_is_missing_without_any_graded_run():
    summary = agent_eval.aggregate(
        [_record(verification="not_applicable", grader="not_applicable")]
    )

    assert summary["grader_unknown_rate"] is None


def test_needs_changes_agrees_with_a_failed_oracle():
    summary = agent_eval.aggregate(
        [
            _record(passed=False, grader="needs_changes", oracle="failed"),
            _record(passed=False, grader="needs_changes", oracle="passed"),
            _record(grader="unknown", oracle="failed"),
        ]
    )

    assert summary["grader_oracle_agreement"] == 0.5


def test_evaluate_contract_fails_closed_without_grader_unknown_rate_in_baseline():
    baseline = _baseline()
    del baseline["metrics"]["grader_unknown_rate"]

    failures = agent_eval.evaluate_contract(_summary(), baseline)

    assert failures == ["grader_unknown_rate: missing from baseline"]


def test_lower_abs_metric_tolerates_drift_from_a_zero_reference():
    baseline = _baseline(grader_unknown_rate=0.0)

    assert (
        agent_eval.evaluate_contract(_summary(grader_unknown_rate=0.05), baseline) == []
    )
    failures = agent_eval.evaluate_contract(
        _summary(grader_unknown_rate=0.15), baseline
    )
    assert len(failures) == 1
    assert failures[0].startswith("grader_unknown_rate:")
    assert agent_eval.CONTRACT_METRICS["grader_unknown_rate"][0] == "lower_abs"


def test_evaluate_contract_enforces_baseline_ceilings():
    baseline = _baseline(grader_unknown_rate=0.05)
    baseline["ceilings"] = {"grader_unknown_rate": 0.1}

    assert (
        agent_eval.evaluate_contract(_summary(grader_unknown_rate=0.1), baseline) == []
    )
    failures = agent_eval.evaluate_contract(
        _summary(grader_unknown_rate=0.12), baseline
    )
    assert failures == ["grader_unknown_rate: 0.120 exceeds ceiling 0.1"]


@pytest.mark.parametrize("ceiling", [None, "0.1", -1, float("nan")])
def test_evaluate_contract_fails_closed_on_an_invalid_ceiling(ceiling):
    baseline = _baseline()
    baseline["ceilings"] = {"grader_unknown_rate": ceiling}

    failures = agent_eval.evaluate_contract(_summary(), baseline)

    assert failures == ["grader_unknown_rate: invalid ceiling"]


def test_evaluate_contract_fails_closed_when_a_ceiling_metric_is_missing():
    baseline = _baseline()
    baseline["ceilings"] = {"grader_unknown_rate": 0.5}
    summary = _summary()
    del summary["grader_unknown_rate"]

    failures = agent_eval.evaluate_contract(summary, baseline)

    assert "grader_unknown_rate: missing from eval summary" in failures


def test_run_causes_reads_cause_code_and_grader_cause():
    state = {
        "error": {"code": "agent_failed", "message": "x", "cause_code": "tool_limit"},
        "grader_cause": "grader_timeout",
    }

    assert agent_eval.run_causes(state) == ("tool_limit", "grader_timeout")


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"error": None, "grader_cause": None},
        {"error": "boom", "grader_cause": 7},
        {"error": {"cause_code": ["x"]}, "grader_cause": {"a": 1}},
    ],
)
def test_run_causes_are_none_safe(state):
    assert agent_eval.run_causes(state) == (None, None)


def test_execute_records_grader_and_error_causes():
    record = agent_eval.execute(
        _task("pass"),
        0,
        lambda task, attempt: _outcome(
            error_cause_code="tool_limit", grader_cause="grader_timeout"
        ),
    )
    clean = agent_eval.execute(_task("pass"), 0, lambda task, attempt: _outcome())

    assert record["error_cause_code"] == "tool_limit"
    assert record["grader_cause"] == "grader_timeout"
    assert clean["error_cause_code"] is None
    assert clean["grader_cause"] is None


def test_informational_attempt_metrics_are_skipped_without_attempt_data():
    summary = agent_eval.aggregate(_live_records())

    assert summary["first_attempt_pass_rate"] is None
    assert "repair_success_rate" not in summary


def test_informational_attempt_metrics_from_per_attempt_gates():
    ok = {"verification": "passed", "grader": "passed"}
    bad = {"verification": "failed", "grader": "not_run"}
    summary = agent_eval.aggregate(
        [
            _record(attempts=[ok]),
            _record(attempts=[bad, ok]),
            _record(passed=False, attempts=[bad, bad]),
            _record(passed=False, attempts=[bad]),
            _record(attempts=None),
        ]
    )

    assert summary["first_attempt_pass_rate"] == 0.25
    assert summary["repair_success_rate"] == 0.5
    assert "first_attempt_pass_rate" in agent_eval.CONTRACT_METRICS
    assert "repair_success_rate" not in agent_eval.CONTRACT_METRICS


def test_repair_success_rate_is_skipped_without_a_repair():
    ok = {"verification": "passed", "grader": "passed"}
    summary = agent_eval.aggregate([_record(attempts=[ok])])

    assert summary["first_attempt_pass_rate"] == 1.0
    assert "repair_success_rate" not in summary


def _recompute_copy(tmp_path, source=ATTEMPT_RESULTS):
    results = tmp_path / "results.json"
    shutil.copy(source, results)
    return results


def _no_live_runner(*args, **kwargs):
    raise AssertionError("recompute must not start a live run")


def test_recompute_summary_updates_baseline_without_a_live_run(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_eval, "LiveRunner", _no_live_runner)
    results = _recompute_copy(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"ceilings": {"grader_unknown_rate": 0.9}}))

    code = agent_eval.main(
        [
            "--recompute-summary",
            str(results),
            "--baseline",
            str(baseline_path),
            "--update-baseline",
        ]
    )

    assert code == 0
    baseline = json.loads(baseline_path.read_text())
    assert baseline["key"] == "openai_gpt-5.6-luna__a4f39169832a__035297f99f3f"
    assert baseline["repeats"] == 3
    assert baseline["metrics"]["grader_unknown_rate"] == 0.0
    assert baseline["metrics"]["pass_at_1"] == 1.0
    assert baseline["metrics"]["first_attempt_pass_rate"] == 0.75
    assert baseline["tolerance"]["first_attempt_pass_rate"] == 0.15
    assert baseline["ceilings"] == {"grader_unknown_rate": 0.9}
    result = json.loads(results.read_text())
    assert result["summary"]["spent_usd"] == pytest.approx(0.65906288)
    assert result["records"] == json.loads(ATTEMPT_RESULTS.read_text())["records"]
    assert result["created_at"] == "2026-10-19T11:46:06+00:00"


def test_recompute_summary_will_not_baseline_records_without_attempts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(agent_eval, "LiveRunner", _no_live_runner)
    results = _recompute_copy(tmp_path, LIVE_RESULTS)
    baseline_path = tmp_path / "baseline.json"
    args = ["--recompute-summary", str(results), "--baseline", str(baseline_path)]

    assert agent_eval.main(args + ["--update-baseline"]) == 1
    assert not baseline_path.exists()
    result = json.loads(results.read_text())
    assert result["contract_failures"] == [
        "first_attempt_pass_rate: missing from eval summary"
    ]


def test_recompute_summary_compares_against_the_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_eval, "LiveRunner", _no_live_runner)
    results = _recompute_copy(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    args = ["--recompute-summary", str(results), "--baseline", str(baseline_path)]
    assert agent_eval.main(args + ["--update-baseline"]) == 0

    assert agent_eval.main(args) == 0
    baseline = json.loads(baseline_path.read_text())
    baseline["ceilings"] = {"cost_usd_p50": 0.01}
    baseline_path.write_text(json.dumps(baseline))

    assert agent_eval.main(args) == 1
    result = json.loads(results.read_text())
    assert result["contract_failures"] == ["cost_usd_p50: 0.044 exceeds ceiling 0.01"]


def test_recompute_summary_rejects_an_unreadable_results_file(tmp_path):
    missing = tmp_path / "missing.json"

    assert agent_eval.main(["--recompute-summary", str(missing)]) == 2


def test_aggregate_reports_startup_medians_without_making_them_contract_metrics():
    def record(startup, repair):
        return {
            **_record(passed=True),
            "harness_metrics": {"startup_s": startup, "repair_startup_s": repair},
        }

    summary = agent_eval.aggregate(
        [record(4.0, None), record(6.0, 3.0), record(5.0, 5.0)]
    )

    assert summary["startup_s_p50"] == 5.0
    assert summary["repair_startup_s_p50"] == 4.0
    assert not set(agent_eval.CONTRACT_METRICS) & {
        "startup_s_p50",
        "repair_startup_s_p50",
    }
