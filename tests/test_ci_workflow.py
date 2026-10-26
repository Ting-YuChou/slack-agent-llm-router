"""Guards against CI drifting away from the agent harness it is meant to cover."""

import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKER_SUITES = (
    "docker-integration",
    "run-checks-rpc",
    "verification-docker",
    "full-loop-docker",
)
DOCKERFILES = (
    "agent-runtime/Dockerfile.agent",
    "agent-runtime/Dockerfile.gateway",
    "agent-runtime/Dockerfile.verification",
)
INTEGRATION_TESTS = (
    "tests/test_agent_observability_integration.py",
    "tests/test_agent_mcp_reader_integration.py",
)
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _run_text(job):
    return "\n".join(
        step.get("run", "") + json.dumps(step.get("with", {})) for step in job["steps"]
    )


def test_docker_job_builds_every_harness_image_and_runs_every_opt_in_suite():
    workflow = _workflow("agent-docker.yml")
    job = workflow["jobs"]["agent-runtime-docker"]
    text = _run_text(job) + json.dumps(job.get("env", {}))
    for dockerfile in DOCKERFILES:
        assert dockerfile in text, f"{dockerfile} is not built"
    for suite in DOCKER_SUITES:
        assert f"dist/test/{suite}.test.js" in text, f"{suite} does not run"
    env = job["env"]
    assert env["PI_AGENT_DOCKER_INTEGRATION"] == "1"
    assert "PI_AGENT_TEST_AGENT_IMAGE" in env
    # The verification suites need a digest; the job resolves it after building.
    assert "PI_AGENT_VERIFICATION_TEST_IMAGE" in text


def test_docker_job_pulls_the_digest_pinned_fake_gateway_image():
    images = (ROOT / "agent-runtime/test/support/docker-images.ts").read_text()
    image = re.search(r'FAKE_GATEWAY_IMAGE = "([^"]+)"', images).group(1)
    assert DIGEST.search(image), f"{image} is not pinned by digest"
    job = _workflow("agent-docker.yml")["jobs"]["agent-runtime-docker"]
    assert f"docker pull {image}" in _run_text(job)
    for suite in DOCKER_SUITES:
        source = (ROOT / f"agent-runtime/test/{suite}.test.ts").read_text()
        assert not re.search(r"[\"']node:\d", source), f"{suite} runs a tag-only image"


def test_integration_job_runs_the_opt_in_python_tests_against_pinned_services():
    job = _workflow("tests.yml")["jobs"]["agent-integration"]
    env = job["env"]
    for path in INTEGRATION_TESTS:
        source = (ROOT / path).read_text()
        # Every variable the test reads, its opt-in flag included, is set by the job.
        names = set(re.findall(r'os\.environ(?:\[|\.get\()"(AGENT_\w+)"', source))
        assert "AGENT_OBSERVABILITY_INTEGRATION" in names, f"{path} is not gated"
        assert names <= set(env), f"{path} needs {sorted(names - set(env))}"
    assert env["AGENT_OBSERVABILITY_INTEGRATION"] == "1"
    assert {"kafka", "clickhouse"} <= set(job["services"])
    for name, service in job["services"].items():
        assert DIGEST.search(service["image"]), f"service {name} is not pinned"
    text = _run_text(job)
    for path in INTEGRATION_TESTS:
        assert path in text, f"{path} does not run"
    for image in re.findall(r"^\s*([a-z0-9./-]+:[\w.-]+(?:@\S+)?) \\$", text, re.M):
        assert DIGEST.search(image), f"{image} is not pinned by digest"
    assert "otel/opentelemetry-collector-contrib" in text
    # The tests skip themselves without their services, so a skip fails the job.
    assert re.search(r"grep -qE \"\[0-9\]\+ skipped\".*exit 1", text)
    assert job["defaults"]["run"]["shell"] == "bash", "pytest | tee needs pipefail"


def test_docker_job_runs_on_harness_changes():
    # PyYAML parses the bare `on` key as True.
    triggers = _workflow("agent-docker.yml")[True]
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        for path in (
            "agent-runtime/**",
            "config/verification/**",
            ".github/workflows/agent-docker.yml",
        ):
            assert path in paths, f"{event} ignores {path}"
    assert triggers["push"]["branches"] == ["main"]


def test_example_profile_ci_jobs_exist():
    jobs = _workflow("tests.yml")["jobs"]
    profile = json.loads(
        (ROOT / "config/verification/slack-router.example.json").read_text()
    )
    named = {check["ciJob"] for check in profile["checks"] if "ciJob" in check}
    assert named, "the example profile maps checks to CI jobs"
    assert named <= set(jobs), f"unknown CI jobs: {sorted(named - set(jobs))}"


def test_grader_calibration_is_a_manual_budgeted_live_run():
    workflow = _workflow("grader-calibration.yml")
    # PyYAML reads the bare `on` key as True.
    triggers = workflow[True]
    assert list(triggers) == ["workflow_dispatch"]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["repeats"]["default"] == "3"
    assert inputs["max_cost_usd"]["default"] == "2"
    assert {"model", "update_baseline"} <= set(inputs)
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["calibrate"]
    text = _run_text(job)
    for dockerfile in DOCKERFILES:
        assert dockerfile in text
    assert "secrets.OPENAI_API_KEY" in json.dumps(job)
    assert "scripts/grader_calibration.py" in text
    for flag in ("--live-openai", "--max-cost-usd", "--update-baseline"):
        assert flag in text
    assert "set -o pipefail" in text
    upload = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
    assert upload["if"] == "always()"
    assert "eval/grader-baseline.json" in upload["with"]["path"]


def test_grader_calibration_sampling_defaults_to_the_baseline():
    inputs = _workflow("grader-calibration.yml")[True]["workflow_dispatch"]["inputs"]
    # Blank inputs leave the sampling to eval/grader-baseline.json.
    assert inputs["samples"]["default"] == ""
    assert inputs["aggregate"]["default"] == ""
    text = _run_text(_workflow("grader-calibration.yml")["jobs"]["calibrate"])
    assert '--samples "$SAMPLES"' in text
    assert '--aggregate "$AGGREGATE"' in text


def test_agent_eval_picks_a_suite_and_holds_runs_to_its_baseline_overrides():
    workflow = _workflow("agent-eval.yml")
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["suite"]["type"] == "choice"
    assert inputs["suite"]["options"] == ["tasks", "repair"]
    assert inputs["suite"]["default"] == "tasks"
    job = workflow["jobs"]["eval"]
    text = _run_text(job)
    assert '--suite "$SUITE"' in text
    assert "--strict-overrides" in text
    step = next(s for s in job["steps"] if "scripts/agent_eval.py" in s.get("run", ""))
    assert step["env"]["SUITE"] == "${{ inputs.suite }}"
    upload = next(s for s in job["steps"] if "upload-artifact" in s.get("uses", ""))
    for baseline in ("eval/baseline.json", "eval/repair-baseline.json"):
        assert baseline in upload["with"]["path"]


def test_npm_audit_gate_runs_in_ci_not_in_image_builds():
    # An audit inside `docker build` needs the registry and fails rebuilds of an
    # unchanged tree whenever a new advisory lands, so the gate lives in CI.
    for dockerfile in (ROOT / "agent-runtime").glob("Dockerfile.*"):
        assert "npm audit" not in dockerfile.read_text(), f"{dockerfile.name} audits"
    job = _workflow("tests.yml")["jobs"]["agent-runtime-audit"]
    runs = [step.get("run", "") for step in job["steps"]]
    audit = "npm --prefix agent-runtime audit --omit=dev --audit-level=high"
    assert audit in runs
    install = next(i for i, run in enumerate(runs) if "npm" in run and " ci" in run)
    assert install < runs.index(audit), "the audit runs before npm ci"
