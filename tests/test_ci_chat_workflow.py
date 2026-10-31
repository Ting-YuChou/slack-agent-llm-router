"""X9c: CI holds the chat path to its own coverage floor and runs its integration tests."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
AGENT_MODULES = ("src/agent_runtime.py", "src/agent_analytics.py")
CHAT_MODULES = (
    "src/llm_router_part1_router.py",
    "src/llm_router_part2_inference.py",
    "src/llm_router_part3_pipeline.py",
    "src/llm_router_part3_policy.py",
    "src/provider_scheduler.py",
    "src/admission.py",
    "src/query_classifier.py",
    "src/analytics_schema.py",
    "src/event_version.py",
    "src/routing_rules.py",
    "slack/*",
)
CHAT_INTEGRATION_TESTS = (
    "tests/test_integration_pipeline.py",
    "tests/test_analytics_dedupe.py",
    "tests/test_clickhouse_poison_batches.py",
    "tests/test_policy_pause_poll_kafka.py",
)


def _jobs():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    return yaml.safe_load(workflow)["jobs"]


def _runs(job):
    return [step.get("run", "") for step in job["steps"]]


def _include(command):
    match = re.search(r'--include="([^"]+)"', command)
    assert match, f"no --include in {command!r}"
    return set(match.group(1).split(","))


def _floor(command):
    match = re.search(r"--fail-under=(\d+)", command)
    assert match, f"no --fail-under in {command!r}"
    return int(match.group(1))


def test_pytest_job_holds_agent_and_chat_modules_to_separate_floors():
    runs = _runs(_jobs()["pytest"])
    pytest_run = next(run for run in runs if "-m pytest tests" in run)
    for source in ("--cov=src", "--cov=slack"):
        assert source in pytest_run.split(), f"{source} is not measured"

    reports = [run for run in runs if "coverage report" in run]
    agent = [run for run in reports if "src/agent_runtime.py" in run]
    chat = [run for run in reports if "src/llm_router_part2_inference.py" in run]
    assert len(agent) == 1 and len(chat) == 1, reports

    # The agent floor keeps its meaning: src/agent_* only, never the chat modules.
    assert _include(agent[0]) == set(AGENT_MODULES)
    assert _floor(agent[0]) == 77
    assert _include(chat[0]) == set(CHAT_MODULES)
    assert _floor(chat[0]) >= 85


def test_chat_integration_job_runs_the_real_service_tests_against_pinned_images():
    jobs = _jobs()
    job = jobs["chat-integration"]
    assert {"kafka", "clickhouse", "redis"} <= set(job["services"])
    for name, service in job["services"].items():
        assert DIGEST.search(service["image"]), f"service {name} is not pinned"
    agent_services = jobs["agent-integration"]["services"]
    for name in ("kafka", "clickhouse"):
        assert job["services"][name]["image"] == agent_services[name]["image"]

    env = job["env"]
    text = "\n".join(_runs(job))
    for path in CHAT_INTEGRATION_TESTS:
        assert path in text, f"{path} does not run"
        source = (ROOT / path).read_text()
        # Every opt-in variable the test reads is set by the job.
        names = set(re.findall(r'os\.environ(?:\[|\.get\()"(CHAT_TEST_\w+)"', source))
        assert names <= set(env), f"{path} needs {sorted(names - set(env))}"
    # scripts/integration_smoke.py reads the ClickHouse password from the env.
    assert (
        env["CLICKHOUSE_PASSWORD"]
        == job["services"]["clickhouse"]["env"]["CLICKHOUSE_PASSWORD"]
    )
    assert "aiokafka|clickhouse-connect" in text
    # The tests skip themselves without their services, so a skip fails the job.
    assert re.search(r"grep -qE \"\[0-9\]\+ skipped\".*exit 1", text)
    assert job["defaults"]["run"]["shell"] == "bash", "pytest | tee needs pipefail"
