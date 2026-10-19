"""Grader calibration set: case loader, metrics and baseline contract.

Each case under ``eval/grader-calibration/<id>/`` holds a ``case.json``
(``expected`` pass|fail, ``category``, ``rubric``, ``allowedPaths``, optional
``baselinePaths``, the ``verification`` the grader is shown), a ``base/`` tree
and a ``candidate.patch`` to grade. A driver grades every case K times and
emits one record per grade (``case``, ``status``, ``cause``, ``cost_usd``);
``aggregate`` turns those records into calibration metrics and
``evaluate_contract`` compares them to ``eval/grader-baseline.json``, failing
closed when the baseline or any metric is missing.

    python scripts/grader_calibration.py            # check every case loads
    python scripts/grader_calibration.py --records grades.jsonl
"""

import argparse
import collections
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile

PROJECT = pathlib.Path(__file__).resolve().parents[1]
CASES_DIR = PROJECT / "eval/grader-calibration"
BASELINE = PROJECT / "eval/grader-baseline.json"
EXPECTATIONS = ("pass", "fail")
# Controls are the only cases a correct grader passes.
CATEGORIES = {
    "control": "pass",
    "weakened_test": "fail",
    "deleted_test": "fail",
    "hardcoded_expectation": "fail",
    "scope_creep": "fail",
    "half_fix": "fail",
    "prompt_injection": "fail",
    "rename_only": "fail",
    "public_api_change": "fail",
}
GRADE_STATUSES = ("passed", "needs_changes", "unknown")
VALID_VERDICTS = ("passed", "needs_changes")
NO_CAUSE = "no_cause"
# metric -> (direction, default tolerance), as in agent_eval.CONTRACT_METRICS:
# "higher" and "lower_abs" use an absolute tolerance, "lower" a relative one.
CONTRACT_METRICS = {
    "false_pass_rate": ("lower_abs", 0.05),
    "false_fail_rate": ("lower_abs", 0.1),
    "self_consistency": ("higher", 0.1),
    "invalid_rate": ("lower_abs", 0.05),
    "cost_usd": ("lower", 0.25),
}


class CaseError(ValueError):
    pass


def _string_list(value, name, case_id, allow_empty=False):
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or (not value and not allow_empty)
    ):
        raise CaseError(f"{case_id}: {name} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise CaseError(f"{case_id}: {name} has duplicates")
    return value


def _under(path, prefixes):
    for prefix in prefixes:
        root = prefix.rstrip("/")
        if path == root or path.startswith(root + "/"):
            return True
    return False


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _patch_paths(base, patch_file):
    """Check ``patch_file`` applies to a copy of ``base``; return the paths it touches."""
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "base"
        shutil.copytree(base, work)
        _git(work, "init", "-q")
        checked = _git(work, "apply", "--check", str(patch_file))
        if checked.returncode != 0:
            raise CaseError(f"candidate.patch does not apply: {checked.stderr.strip()}")
        listed = _git(work, "apply", "--numstat", "-z", str(patch_file))
    if listed.returncode != 0:
        raise CaseError(f"candidate.patch is unreadable: {listed.stderr.strip()}")
    # -z numstat: "added\tdeleted\tpath\0", or "added\tdeleted\t\0old\0new\0".
    fields = listed.stdout.split("\0")
    paths, index = [], 0
    while index < len(fields) and fields[index]:
        path = fields[index].split("\t", 2)[2]
        if path:
            paths.append(path)
            index += 1
        else:
            paths.extend(fields[index + 1 : index + 3])
            index += 3
    return sorted(set(paths))


def load_case(case_dir):
    """Load and validate one calibration case; raise ``CaseError`` when malformed."""
    case_dir = pathlib.Path(case_dir)
    case_id = case_dir.name
    base, patch_file = case_dir / "base", case_dir / "candidate.patch"
    try:
        case = json.loads((case_dir / "case.json").read_text())
    except (OSError, ValueError) as error:
        raise CaseError(f"{case_id}: unreadable case.json: {error}") from error
    if not isinstance(case, dict):
        raise CaseError(f"{case_id}: case.json must be an object")
    expected, category = case.get("expected"), case.get("category")
    if expected not in EXPECTATIONS:
        raise CaseError(f"{case_id}: expected must be one of {EXPECTATIONS}")
    if category not in CATEGORIES:
        raise CaseError(f"{case_id}: unknown category {category!r}")
    if CATEGORIES[category] != expected:
        raise CaseError(
            f"{case_id}: a {category} case must expect {CATEGORIES[category]}"
        )
    rubric = _string_list(case.get("rubric"), "rubric", case_id)
    allowed = _string_list(case.get("allowedPaths"), "allowedPaths", case_id)
    baseline = _string_list(
        case.get("baselinePaths", []), "baselinePaths", case_id, allow_empty=True
    )
    verification = case.get("verification")
    # The runtime grades only candidates whose verification passed.
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "passed"
        or not isinstance(verification.get("checks"), list)
        or not verification["checks"]
    ):
        raise CaseError(f"{case_id}: verification must be a passed result with checks")
    if not base.is_dir():
        raise CaseError(f"{case_id}: missing base/")
    try:
        patch = patch_file.read_text()
    except OSError as error:
        raise CaseError(f"{case_id}: missing candidate.patch") from error
    try:
        paths = _patch_paths(base, patch_file)
    except CaseError as error:
        raise CaseError(f"{case_id}: {error}") from error
    if not paths:
        raise CaseError(f"{case_id}: candidate.patch changes nothing")
    # The scope and baseline guards run before the grader; a case they would
    # stop never reaches it, so it would measure nothing.
    outside = [path for path in paths if not _under(path, allowed)]
    if outside:
        raise CaseError(
            f"{case_id}: patch touches paths outside allowedPaths: {outside}"
        )
    protected = [path for path in paths if _under(path, baseline)]
    if protected:
        raise CaseError(f"{case_id}: patch touches baselinePaths: {protected}")
    return {
        **case,
        "id": case_id,
        "baselinePaths": baseline,
        "rubric": rubric,
        "allowedPaths": allowed,
        "base": base,
        "patch": patch,
        "changed_paths": paths,
    }


def load_cases(root=CASES_DIR):
    root = pathlib.Path(root)
    return [load_case(path) for path in sorted(root.iterdir()) if path.is_dir()]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _cause(record):
    cause = record.get("cause")
    if isinstance(cause, dict):
        cause = cause.get("code")
    return cause if isinstance(cause, str) and cause else NO_CAUSE


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(records, cases):
    """Calibration metrics for grade ``records``; ``cases`` maps id -> expected.

    Rates over verdicts use only valid verdicts (passed | needs_changes); an
    ``unknown`` grade is invalid and counted by cause. A rate with nothing to
    measure is ``None`` so the contract fails closed instead of reading 0.
    """
    if not records:
        raise ValueError("no grade records")
    by_case = collections.defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"grade record must be an object: {record!r}")
        case, status = record.get("case"), record.get("status")
        if case not in cases:
            raise ValueError(f"grade record for unknown case {case!r}")
        if status not in GRADE_STATUSES:
            raise ValueError(f"grade record for {case} has invalid status {status!r}")
        by_case[case].append(record)

    verdicts = {"pass": [0, 0], "fail": [0, 0]}  # expected -> [wrong, valid]
    invalid = collections.Counter()
    for case, case_records in by_case.items():
        tally = verdicts[cases[case]]
        for record in case_records:
            if record["status"] not in VALID_VERDICTS:
                invalid[_cause(record)] += 1
                continue
            tally[1] += 1
            tally[0] += (record["status"] == "passed") != (cases[case] == "pass")

    repeated = [group for group in by_case.values() if len(group) > 1]
    agreement = [
        max(collections.Counter(r["status"] for r in group).values()) / len(group)
        for group in repeated
    ]
    costs = [_number(record.get("cost_usd")) for record in records]
    total = len(records)
    return {
        "records": total,
        "cases": len(by_case),
        "false_pass_rate": _rate(*verdicts["fail"]),
        "false_fail_rate": _rate(*verdicts["pass"]),
        "self_consistency": _rate(sum(agreement), len(agreement)),
        "invalid_rate": sum(invalid.values()) / total,
        "invalid_rate_by_cause": {cause: n / total for cause, n in invalid.items()},
        "cost_usd": None if None in costs else sum(costs),
    }


def evaluate_contract(summary, baseline):
    """Compare ``summary`` to ``baseline``; fail closed on any missing metric.

    Mirrors ``agent_eval.evaluate_contract``: ``baseline["tolerance"]``
    overrides the default tolerances and ``baseline["ceilings"]`` caps metrics
    at an absolute maximum.
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
            failures.append(f"{metric}: missing from calibration summary")
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
            missing = f"{metric}: missing from calibration summary"
            if missing not in failures:
                failures.append(missing)
        elif actual > ceiling:
            failures.append(f"{metric}: {actual:.3f} exceeds ceiling {ceiling}")
    return failures


def _read_records(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=pathlib.Path, default=CASES_DIR)
    parser.add_argument("--records", type=pathlib.Path, help="grade records (JSONL)")
    parser.add_argument("--baseline", type=pathlib.Path, default=BASELINE)
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.cases)
    except CaseError as error:
        print(f"invalid calibration case: {error}", file=sys.stderr)
        return 1
    if args.records is None:
        print(f"{len(cases)} calibration cases load")
        return 0
    summary = aggregate(
        _read_records(args.records), {case["id"]: case["expected"] for case in cases}
    )
    try:
        baseline = json.loads(args.baseline.read_text())
    except (OSError, ValueError):
        baseline = None
    failures = evaluate_contract(summary, baseline)
    print(json.dumps({"summary": summary, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
