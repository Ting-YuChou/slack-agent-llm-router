import pytest

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
