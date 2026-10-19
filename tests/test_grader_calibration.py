import json
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
