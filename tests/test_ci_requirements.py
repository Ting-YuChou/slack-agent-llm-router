from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_redis_runtime_dependency():
    requirements = (ROOT / "requirements-ci.txt").read_text().splitlines()

    assert any(line.startswith("redis==") for line in requirements)


def test_pytest_job_provisions_redis_service():
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())

    redis_service = workflow["jobs"]["pytest"]["services"]["redis"]
    assert redis_service["image"].startswith("redis:")
    assert "6379:6379" in redis_service["ports"]
    assert "redis-cli ping" in redis_service["options"]
