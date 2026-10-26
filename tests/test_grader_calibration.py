import contextlib
import io
import json
import pathlib
import shutil

import pytest

from scripts import grader_calibration as calibration

CASES = {"good": "pass", "bad": "fail", "worse": "fail"}


def _record(case, status, cost=0.01, cause=None):
    record = {"case": case, "status": status, "cost_usd": cost}
    if cause is not None:
        record["cause"] = {"code": cause, "message": "why"}
    return record


def _records():
    return [
        _record("good", "passed"),
        _record("good", "passed"),
        _record("good", "needs_changes"),
        _record("bad", "passed"),
        _record("bad", "needs_changes"),
        _record("bad", "needs_changes"),
        _record("worse", "needs_changes"),
        _record("worse", "unknown", cause="grader_invalid_submission"),
        _record("worse", "unknown", cost=0.0, cause="grader_execution_failed"),
        _record("worse", "unknown", cause="grader_invalid_submission"),
    ]


def test_every_committed_case_loads():
    cases = calibration.load_cases()
    assert len(cases) >= 10
    assert {case["expected"] for case in cases} == {"pass", "fail"}
    assert sum(case["category"] == "control" for case in cases) >= 2
    for case in cases:
        assert case["changed_paths"], case["id"]


def test_aggregate_computes_rates_over_valid_verdicts():
    summary = calibration.aggregate(_records(), CASES)

    assert summary["records"] == 10
    assert summary["cases"] == 3
    # 1 of the 4 valid verdicts on expected-fail cases is a pass.
    assert summary["false_pass_rate"] == pytest.approx(1 / 4)
    # 1 of the 3 valid verdicts on the expected-pass case is a fail.
    assert summary["false_fail_rate"] == pytest.approx(1 / 3)
    assert summary["invalid_rate"] == pytest.approx(3 / 10)
    assert summary["invalid_rate_by_cause"] == {
        "grader_invalid_submission": pytest.approx(2 / 10),
        "grader_execution_failed": pytest.approx(1 / 10),
    }
    # Modal share per case: good 2/3, bad 2/3, worse 3/4 (unknown is an outcome).
    assert summary["self_consistency"] == pytest.approx((2 / 3 + 2 / 3 + 3 / 4) / 3)
    assert summary["cost_usd"] == pytest.approx(0.09)


def test_aggregate_buckets_unknown_without_cause_and_accepts_string_causes():
    summary = calibration.aggregate(
        [
            _record("good", "unknown"),
            {"case": "bad", "status": "unknown", "cause": "insufficient_evidence"},
        ],
        CASES,
    )
    assert summary["invalid_rate_by_cause"] == {
        "no_cause": pytest.approx(0.5),
        "insufficient_evidence": pytest.approx(0.5),
    }
    # No valid verdicts and no repeats: the rates are undefined, not zero.
    assert summary["false_pass_rate"] is None
    assert summary["false_fail_rate"] is None
    assert summary["self_consistency"] is None
    # A record without a cost makes the total unknown rather than understated.
    assert summary["cost_usd"] is None


@pytest.mark.parametrize(
    "record",
    [
        {"case": "missing", "status": "passed", "cost_usd": 0},
        {"case": "good", "status": "failed", "cost_usd": 0},
        {"case": "good", "cost_usd": 0},
        "passed",
    ],
)
def test_aggregate_rejects_malformed_records(record):
    with pytest.raises(ValueError):
        calibration.aggregate([record], CASES)


def test_aggregate_rejects_empty_records():
    with pytest.raises(ValueError):
        calibration.aggregate([], CASES)


def _summary(**overrides):
    summary = {
        "false_pass_rate": 0.1,
        "false_fail_rate": 0.1,
        "self_consistency": 0.9,
        "invalid_rate": 0.05,
        "cost_usd": 1.0,
    }
    summary.update(overrides)
    return summary


def _baseline(**metrics):
    return {"metrics": _summary(**metrics)}


def test_contract_passes_at_baseline():
    assert calibration.evaluate_contract(_summary(), _baseline()) == []


@pytest.mark.parametrize("baseline", [None, {}, {"metrics": {}}])
def test_contract_fails_closed_without_baseline(baseline):
    failures = calibration.evaluate_contract(_summary(), baseline)
    assert len(failures) == len(calibration.CONTRACT_METRICS)
    assert all("missing from baseline" in failure for failure in failures)


def test_contract_fails_closed_when_metric_missing_or_undefined():
    summary = _summary(false_pass_rate=None)
    del summary["cost_usd"]
    failures = calibration.evaluate_contract(summary, _baseline())
    assert failures == [
        "false_pass_rate: missing from calibration summary",
        "cost_usd: missing from calibration summary",
    ]


def test_contract_flags_regressions_beyond_tolerance():
    summary = _summary(
        false_pass_rate=0.2,
        false_fail_rate=0.25,
        self_consistency=0.7,
        invalid_rate=0.2,
        cost_usd=1.3,
    )
    failures = calibration.evaluate_contract(summary, _baseline())
    assert [failure.split(":")[0] for failure in failures] == list(
        calibration.CONTRACT_METRICS
    )


def test_contract_respects_tolerance_overrides_and_ceilings():
    baseline = _baseline()
    baseline["tolerance"] = {"false_pass_rate": 0.2, "cost_usd": -1}
    baseline["ceilings"] = {"false_pass_rate": 0.25, "invalid_rate": "x"}
    failures = calibration.evaluate_contract(_summary(false_pass_rate=0.28), baseline)
    assert failures == [
        "cost_usd: invalid tolerance",
        "false_pass_rate: 0.280 exceeds ceiling 0.25",
        "invalid_rate: invalid ceiling",
    ]


def _copy_case(tmp_path, name="control-minimal-fix"):
    target = tmp_path / name
    shutil.copytree(calibration.CASES_DIR / name, target)
    return target


def _rewrite_case(case_dir, **changes):
    path = case_dir / "case.json"
    case = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            case.pop(key, None)
        else:
            case[key] = value
    path.write_text(json.dumps(case))


def test_load_case_returns_the_case(tmp_path):
    case = calibration.load_case(_copy_case(tmp_path))
    assert case["id"] == "control-minimal-fix"
    assert case["expected"] == "pass"
    assert case["changed_paths"] == ["src/counter.cjs", "src/label.cjs"]
    assert case["patch"].startswith("diff --git")
    assert case["base"].is_dir()


@pytest.mark.parametrize(
    "changes",
    [
        {"expected": "needs_changes"},
        {"expected": None},
        {"category": "vibes"},
        {"category": "control", "expected": "fail"},
        {"category": "scope_creep", "expected": "pass"},
        {"rubric": []},
        {"rubric": ["a", "a"]},
        {"allowedPaths": "src/"},
        {"allowedPaths": ["tests/"]},
        {"baselinePaths": ["src/"]},
        {"verification": {"status": "failed", "checks": []}},
        {"verification": {"status": "passed"}},
        {"verification": None},
    ],
)
def test_load_case_rejects_invalid_case_json(tmp_path, changes):
    case_dir = _copy_case(tmp_path)
    _rewrite_case(case_dir, **changes)
    with pytest.raises(calibration.CaseError):
        calibration.load_case(case_dir)


def test_load_case_rejects_a_patch_that_does_not_apply(tmp_path):
    case_dir = _copy_case(tmp_path)
    (case_dir / "base/src/counter.cjs").write_text("exports.increment = n => n;\n")
    with pytest.raises(calibration.CaseError, match="does not apply"):
        calibration.load_case(case_dir)


@pytest.mark.parametrize("missing", ["case.json", "candidate.patch", "base"])
def test_load_case_rejects_missing_parts(tmp_path, missing):
    case_dir = _copy_case(tmp_path)
    target = case_dir / missing
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    with pytest.raises(calibration.CaseError):
        calibration.load_case(case_dir)


def test_load_case_does_not_modify_base(tmp_path):
    case_dir = _copy_case(tmp_path)
    before = (case_dir / "base/src/counter.cjs").read_text()
    calibration.load_case(case_dir)
    assert (case_dir / "base/src/counter.cjs").read_text() == before
    assert not (case_dir / "base/.git").exists()


def _main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = calibration.main(argv)
    return code, out.getvalue()


def _run_line(**overrides):
    run = {
        "type": "run",
        "complete": True,
        "records": 2,
        "expected_records": 2,
        "cost_usd": 0.02,
    }
    run.update(overrides)
    return run


def _write_jsonl(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_read_run_splits_cli_grades_from_the_run_line(tmp_path):
    grade = {"type": "grade", **_record("good", "passed"), "repeat": 1}
    path = _write_jsonl(tmp_path / "out.jsonl", [grade, grade, _run_line()])
    records, run = calibration.read_run(path)
    assert records == [grade, grade]
    assert run["complete"] is True
    assert calibration.run_failures(run, records) == []


def test_read_run_treats_a_missing_run_line_as_incomplete(tmp_path):
    path = _write_jsonl(tmp_path / "out.jsonl", [_record("good", "passed")])
    records, run = calibration.read_run(path)
    assert records == [_record("good", "passed")]
    assert run is None
    assert calibration.run_failures(run, records) == [
        "calibration incomplete: no run summary line"
    ]


def test_run_failures_flag_an_incomplete_or_short_run():
    records = [_record("good", "passed")]
    assert calibration.run_failures(
        _run_line(complete=False, reason="max_cost_usd_exceeded", records=1), records
    ) == ["calibration incomplete: max_cost_usd_exceeded (1 of 2 records)"]
    assert calibration.run_failures(_run_line(), records) == [
        "calibration incomplete: 1 grade lines but the run reports 2"
    ]


def test_main_fails_an_incomplete_run_even_at_baseline(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    grade = {"type": "grade", **_record("control-minimal-fix", "passed")}
    out = _write_jsonl(
        tmp_path / "out.jsonl",
        [grade, _run_line(complete=False, reason="max_cost_usd_exceeded", records=1)],
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline()))
    code, stdout = _main(
        ["--cases", str(cases), "--records", str(out), "--baseline", str(baseline)]
    )
    report = json.loads(stdout)
    assert code == 1
    assert report["failures"][0].startswith("calibration incomplete")


def test_update_baseline_writes_metrics_and_keeps_ceilings(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    _copy_case(cases, "weakened-test")
    good = {"type": "grade", **_record("control-minimal-fix", "passed")}
    bad = {"type": "grade", **_record("weakened-test", "needs_changes")}
    out = _write_jsonl(
        tmp_path / "out.jsonl", [good, good, bad, bad, _run_line(records=4)]
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"ceilings": {"invalid_rate": 0.1}}))
    code, stdout = _main(
        [
            "--cases",
            str(cases),
            "--records",
            str(out),
            "--baseline",
            str(baseline),
            "--update-baseline",
        ]
    )
    written = json.loads(baseline.read_text())
    assert code == 0
    assert set(written["metrics"]) == set(calibration.CONTRACT_METRICS)
    assert written["metrics"]["false_fail_rate"] == 0
    assert written["ceilings"] == {"invalid_rate": 0.1}
    assert written["tolerance"]["false_pass_rate"] == 0.05

    # A first baseline starts with the default contract ceilings.
    baseline.unlink()
    code, _ = _main(
        [
            "--cases",
            str(cases),
            "--records",
            str(out),
            "--baseline",
            str(baseline),
            "--update-baseline",
        ]
    )
    assert code == 0
    assert json.loads(baseline.read_text())["ceilings"] == {
        "invalid_rate": 0.10,
        "false_pass_rate": 0.2,
    }


def test_update_baseline_refuses_undefined_metrics(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    # One record per case: self_consistency is undefined and must not be written.
    grade = {"type": "grade", **_record("control-minimal-fix", "passed")}
    out = _write_jsonl(tmp_path / "out.jsonl", [grade, _run_line(records=1)])
    baseline = tmp_path / "baseline.json"
    code, stdout = _main(
        [
            "--cases",
            str(cases),
            "--records",
            str(out),
            "--baseline",
            str(baseline),
            "--update-baseline",
        ]
    )
    report = json.loads(stdout)
    assert code == 1
    assert "self_consistency: missing from calibration summary" in report["failures"]
    assert not baseline.exists()


def test_cli_command_keeps_the_signing_secret_out_of_argv(tmp_path):
    gateway = {
        "gateway_url": "http://model-gateway:8080",
        "network": "pi-net",
        "image": "slack-pi-agent:structured-test",
        "budget_ledger": "/state/budget/budget.sqlite",
        "signing_secret": "s3cret",
    }
    argv, env = calibration.cli_command(
        cases=tmp_path / "cases",
        out=tmp_path / "out.jsonl",
        repeats=3,
        max_cost_usd=2.0,
        grader_model="openai/gpt-5.6-sol",
        gateway=gateway,
        base_env={"PATH": "/bin", "OPENAI_API_KEY": "sk-live"},
    )
    assert argv[0] == "node"
    assert argv[1].endswith("agent-runtime/dist/src/grader-calibration-cli.js")
    for flag, value in [
        ("--repeats", "3"),
        ("--max-cost-usd", "2.0"),
        ("--grader-model", "openai/gpt-5.6-sol"),
        ("--gateway-url", "http://model-gateway:8080"),
        ("--network", "pi-net"),
        ("--image", "slack-pi-agent:structured-test"),
        ("--budget-ledger", "/state/budget/budget.sqlite"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "s3cret" not in argv
    assert env["MODEL_GATEWAY_SIGNING_SECRET"] == "s3cret"
    # The grader reaches models only through the gateway; the CLI never sees the key.
    assert "OPENAI_API_KEY" not in env


def test_live_mode_requires_explicit_opt_in():
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        calibration.main(["--repeats", "2"])
    assert "--live-openai" in err.getvalue()


def test_topology_exposes_the_grader_gateway_once_started(tmp_path):
    topology = calibration.topology
    booted = topology.Topology(
        calibration.PROJECT, tmp_path, tmp_path / "repo", "key", tmp_path / "p.json"
    )
    with pytest.raises(RuntimeError):
        booted.grader_gateway()
    booted.signing = "s3cret"
    gateway = booted.grader_gateway()
    assert gateway == {
        "gateway_url": "http://model-gateway:8080",
        "network": booted.internal,
        "image": topology.AGENT_IMAGE,
        "budget_ledger": str(tmp_path / "state/budget/budget.sqlite"),
        "signing_secret": "s3cret",
    }


def test_live_run_boots_the_topology_and_scores_the_cli_output(tmp_path, monkeypatch):
    cases = tmp_path / "cases"
    _copy_case(cases)
    gateway = {
        "gateway_url": "http://model-gateway:8080",
        "network": "pi-net",
        "image": "agent",
        "budget_ledger": str(tmp_path / "ledger.sqlite"),
        "signing_secret": "s3cret",
    }
    booted_with = {}

    class FakeTopology:
        def __init__(self, project, root, repo, key, profile_path):
            booted_with.update(repo=repo, key=key, profile=profile_path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grader_gateway(self):
            return gateway

    def fake_run(argv, env):
        assert env["MODEL_GATEWAY_SIGNING_SECRET"] == "s3cret"
        out = pathlib.Path(argv[argv.index("--out") + 1])
        grade = {"type": "grade", **_record("control-minimal-fix", "passed")}
        _write_jsonl(out, [grade, grade, _run_line()])

    monkeypatch.setattr(calibration.topology, "Topology", FakeTopology)
    monkeypatch.setattr(calibration.topology, "read_openai_key", lambda: "sk-test")
    monkeypatch.setattr(calibration.topology, "verification_image_id", lambda: "img")
    monkeypatch.setattr(calibration, "_run_cli", fake_run)
    out = tmp_path / "results/grades.jsonl"
    code, stdout = _main(
        [
            "--live-openai",
            "--cases",
            str(cases),
            "--repeats",
            "2",
            "--out",
            str(out),
            "--baseline",
            str(tmp_path / "missing.json"),
        ]
    )
    report = json.loads(stdout)
    assert booted_with["key"] == "sk-test"
    assert (booted_with["repo"] / "src/counter.cjs").exists()
    assert json.loads(booted_with["profile"].read_text())["image"] == "img"
    assert report["summary"]["records"] == 2
    # No baseline yet: the live run still fails closed.
    assert code == 1
    assert all("missing from baseline" in failure for failure in report["failures"])


def test_live_profile_repository_matches_the_fixture_remote():
    # The runtime refuses to boot when the trusted profile's repository is not
    # the checkout's origin, so the two must name the same repository.
    case = calibration.load_case(next(calibration.CASES_DIR.iterdir()))
    profile = calibration._live_profile(case, "sha256:" + "0" * 64)

    remote = calibration.FIXTURE_REMOTE
    assert remote.startswith("https://github.com/") and remote.endswith(".git")
    assert profile["repository"] == remote[len("https://github.com/") : -len(".git")]


def test_update_baseline_refuses_a_summary_over_a_ceiling(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    _copy_case(cases, "weakened-test")
    good = {"type": "grade", **_record("control-minimal-fix", "passed")}
    refused = {
        "type": "grade",
        **_record("weakened-test", "unknown", cost=0.0, cause="grader_did_not_submit"),
    }
    out = _write_jsonl(
        tmp_path / "out.jsonl", [good, good, refused, refused, _run_line(records=4)]
    )
    baseline = tmp_path / "baseline.json"
    code, stdout = _main(
        [
            "--cases",
            str(cases),
            "--records",
            str(out),
            "--baseline",
            str(baseline),
            "--update-baseline",
        ]
    )
    report = json.loads(stdout)
    assert code == 1
    assert any("exceeds ceiling" in failure for failure in report["failures"])
    assert not baseline.exists()


def test_topology_prices_match_the_model_catalog_with_cache_reads():
    # The gateway ledger is the run's hard spend cap; prices above the
    # catalog (or cached input at the full rate) exhaust it far below the
    # real spend, as the first live calibration showed.
    from scripts import validate_structured_workflow as topology

    catalog = {
        "openai/gpt-5.6-luna": {"input": 0.2, "output": 1.2, "cacheRead": 0.02},
        "openai/gpt-5.6-sol": {"input": 4, "output": 20, "cacheRead": 0.4},
    }
    assert topology.MODEL_PRICES == catalog


def _sampled(case, status, samples=3, aggregate="unanimous", cost=0.03):
    return {
        "type": "grade",
        **_record(case, status, cost=cost),
        "samples": samples,
        "aggregate": aggregate,
    }


def test_aggregate_reports_the_sampling_and_cost_per_grade():
    summary = calibration.aggregate(
        [
            _sampled("good", "passed", cost=0.02),
            _sampled("bad", "needs_changes", cost=0.03),
            _sampled("worse", "needs_changes", cost=0.04),
            _sampled("worse", "passed", cost=0.03),
        ],
        CASES,
    )
    assert summary["samples"] == 3
    assert summary["aggregate"] == "unanimous"
    # The rates are over the aggregated verdicts, one per grade record.
    assert summary["false_pass_rate"] == pytest.approx(1 / 3)
    assert summary["false_fail_rate"] == 0
    assert summary["cost_usd"] == pytest.approx(0.12)
    assert summary["cost_per_grade"] == pytest.approx(0.03)


def test_aggregate_treats_records_without_sampling_as_one_unanimous_sample():
    summary = calibration.aggregate(_records(), CASES)
    assert summary["samples"] == 1
    assert summary["aggregate"] == "unanimous"
    assert summary["cost_per_grade"] == pytest.approx(0.09 / 10)
    unpriced = calibration.aggregate([{"case": "good", "status": "passed"}], CASES)
    assert unpriced["cost_per_grade"] is None


@pytest.mark.parametrize(
    "records",
    [
        [_sampled("good", "passed"), _record("good", "passed")],
        [_sampled("good", "passed"), _sampled("good", "passed", aggregate="majority")],
        [_sampled("good", "passed", samples=2)],
        [_sampled("good", "passed", aggregate="any")],
    ],
)
def test_aggregate_rejects_mixed_or_unknown_sampling(records):
    with pytest.raises(ValueError):
        calibration.aggregate(records, CASES)


def test_cli_command_passes_the_sampling_to_the_cli(tmp_path):
    gateway = {
        "gateway_url": "http://model-gateway:8080",
        "network": "pi-net",
        "image": "agent",
        "budget_ledger": "/state/budget/budget.sqlite",
        "signing_secret": "s3cret",
    }
    common = dict(
        cases=tmp_path / "cases",
        out=tmp_path / "out.jsonl",
        repeats=3,
        max_cost_usd=2.0,
        grader_model="openai/gpt-5.6-sol",
        gateway=gateway,
        base_env={},
    )
    argv, _ = calibration.cli_command(**common)
    assert argv[argv.index("--samples") + 1] == "1"
    assert argv[argv.index("--aggregate") + 1] == "unanimous"
    argv, _ = calibration.cli_command(**common, samples=3, aggregate="majority")
    assert argv[argv.index("--samples") + 1] == "3"
    assert argv[argv.index("--aggregate") + 1] == "majority"


def test_parse_args_sampling_defaults_and_choices(tmp_path):
    # Without a recorded sampling in the baseline the live default is one sample.
    no_sampling = ["--baseline", str(tmp_path / "missing.json")]
    live = calibration.parse_args(["--live-openai", *no_sampling])
    assert (live.samples, live.aggregate) == (1, "unanimous")
    sampled = calibration.parse_args(
        ["--live-openai", "--samples", "3", "--aggregate", "majority"]
    )
    assert (sampled.samples, sampled.aggregate) == (3, "majority")
    # Offline the setting is what the records must have been graded with.
    offline = calibration.parse_args(["--records", "r.jsonl", "--samples", "3"])
    assert (offline.samples, offline.aggregate) == (3, None)
    for bad in (["--samples", "2"], ["--aggregate", "any"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            calibration.parse_args(["--live-openai", *bad])


def test_main_fails_records_graded_with_a_different_sampling(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    grade = _sampled("control-minimal-fix", "passed", aggregate="majority")
    out = _write_jsonl(tmp_path / "out.jsonl", [grade, grade, _run_line()])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline()))
    argv = ["--cases", str(cases), "--records", str(out), "--baseline", str(baseline)]

    code, stdout = _main([*argv, "--samples", "3", "--aggregate", "unanimous"])
    report = json.loads(stdout)
    assert code == 1
    assert report["failures"][0] == (
        "sampling mismatch: records use samples=3 aggregate=majority, "
        "expected samples=3 aggregate=unanimous"
    )
    _, stdout = _main([*argv, "--samples", "3", "--aggregate", "majority"])
    assert not any("sampling" in f for f in json.loads(stdout)["failures"])


def test_update_baseline_records_the_sampling_it_was_measured_with(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    _copy_case(cases, "weakened-test")
    good = _sampled("control-minimal-fix", "passed")
    bad = _sampled("weakened-test", "needs_changes")
    out = _write_jsonl(
        tmp_path / "out.jsonl", [good, good, bad, bad, _run_line(records=4)]
    )
    baseline = tmp_path / "baseline.json"
    code, _ = _main(
        [
            "--cases",
            str(cases),
            "--records",
            str(out),
            "--baseline",
            str(baseline),
            "--update-baseline",
        ]
    )
    assert code == 0
    written = json.loads(baseline.read_text())
    assert written["sampling"] == {"samples": 3, "aggregate": "unanimous"}
    assert set(written["metrics"]) == set(calibration.CONTRACT_METRICS)


def _sampled_baseline(tmp_path, samples=3, aggregate="majority"):
    baseline = tmp_path / "baseline.json"
    recorded = _baseline()
    recorded["sampling"] = {"samples": samples, "aggregate": aggregate}
    baseline.write_text(json.dumps(recorded))
    return baseline


def test_sampling_defaults_to_the_grader_baseline(tmp_path):
    baseline = ["--baseline", str(_sampled_baseline(tmp_path))]
    live = calibration.parse_args(["--live-openai", *baseline])
    assert (live.samples, live.aggregate) == (3, "majority")
    offline = calibration.parse_args(["--records", "r.jsonl", *baseline])
    assert (offline.samples, offline.aggregate) == (3, "majority")
    # Either flag replaces the baseline's sampling as a whole.
    explicit = calibration.parse_args(["--live-openai", *baseline, "--samples", "1"])
    assert (explicit.samples, explicit.aggregate) == (1, "unanimous")
    explicit = calibration.parse_args(["--records", "r", *baseline, "--samples", "1"])
    assert (explicit.samples, explicit.aggregate) == (1, None)
    explicit = calibration.parse_args(
        ["--live-openai", *baseline, "--aggregate", "unanimous"]
    )
    assert (explicit.samples, explicit.aggregate) == (1, "unanimous")


def test_the_committed_grader_baseline_sets_the_default_sampling():
    recorded = json.loads(calibration.BASELINE.read_text())["sampling"]
    live = calibration.parse_args(["--live-openai"])
    assert (live.samples, live.aggregate) == (
        recorded["samples"],
        recorded["aggregate"],
    )


@pytest.mark.parametrize(
    "sampling",
    [
        {"samples": 2, "aggregate": "unanimous"},
        {"samples": True, "aggregate": "unanimous"},
        {"samples": 3, "aggregate": "any"},
        {"samples": 3},
        "3",
    ],
)
def test_parse_args_refuses_an_unknown_baseline_sampling(tmp_path, sampling):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({**_baseline(), "sampling": sampling}))
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        calibration.parse_args(["--live-openai", "--baseline", str(baseline)])
    assert "sampling" in err.getvalue()


def test_main_holds_records_to_the_baseline_sampling_by_default(tmp_path):
    cases = tmp_path / "cases"
    _copy_case(cases)
    single = {"type": "grade", **_record("control-minimal-fix", "passed")}
    out = _write_jsonl(tmp_path / "out.jsonl", [single, single, _run_line()])
    argv = ["--cases", str(cases), "--records", str(out)]
    baseline = _sampled_baseline(tmp_path, samples=3, aggregate="unanimous")

    code, stdout = _main([*argv, "--baseline", str(baseline)])
    assert code == 1
    assert json.loads(stdout)["failures"][0] == (
        "sampling mismatch: records use samples=1 aggregate=unanimous, "
        "expected samples=3 aggregate=unanimous"
    )
    _, stdout = _main([*argv, "--baseline", str(baseline), "--samples", "1"])
    assert not any("sampling" in f for f in json.loads(stdout)["failures"])
