from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


ROOT = Path(__file__).resolve().parents[1]
CI_LOCK = ROOT / "agent-runtime/requirements-ci.lock"
# A pinned entry in a uv-exported lock: `name==version [; marker] \`, hashes follow.
LOCK_PIN = re.compile(r"^([A-Za-z0-9][\w.-]*)==([^\s;]+)[^\n]*\\$", re.M)


def test_ci_installs_redis_runtime_dependency():
    requirements = (ROOT / "requirements-ci.txt").read_text().splitlines()

    assert any(line.startswith("redis==") for line in requirements)


def test_pytest_job_provisions_redis_service():
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())

    redis_service = workflow["jobs"]["pytest"]["services"]["redis"]
    assert redis_service["image"].startswith("redis:")
    assert "6379:6379" in redis_service["ports"]
    assert "redis-cli ping" in redis_service["options"]


def _pip_installs(dockerfile):
    # Join continued lines so a split `pip install` reads as one command.
    text = dockerfile.read_text().replace("\\\n", " ")
    copies = {
        dest: source
        for source, dest in re.findall(r"^COPY\s+(\S+)\s+(\S+)\s*$", text, re.M)
    }
    return copies, re.findall(r"\bpip3? install\b[^&;|\n]*", text)


def test_agent_runtime_images_install_python_packages_by_hash():
    installing = []
    for dockerfile in sorted((ROOT / "agent-runtime").glob("Dockerfile.*")):
        copies, installs = _pip_installs(dockerfile)
        for command in installs:
            installing.append(dockerfile.name)
            args = command.split()
            assert "--require-hashes" in args, f"{dockerfile.name}: {command}"
            locks = [args[i + 1] for i, arg in enumerate(args) if arg == "-r"]
            assert locks, f"{dockerfile.name} installs without a lock: {command}"
            for lock in locks:
                assert lock.endswith(".lock"), f"{dockerfile.name} installs {lock}"
                source = copies.get(lock)
                assert source, f"{dockerfile.name} never copies {lock}"
                # Dockerfile.verification builds from the repository root, the others
                # from agent-runtime/.
                assert any(
                    (base / source).is_file() for base in (ROOT, ROOT / "agent-runtime")
                ), f"{dockerfile.name} copies a missing {source}"
    assert "Dockerfile.verification" in installing


def test_verification_lock_pins_every_ci_requirement():
    pins = {
        canonicalize_name(name): version
        for name, version in LOCK_PIN.findall(CI_LOCK.read_text())
    }
    for line in (ROOT / "requirements-ci.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        # Regenerate the lock with the command in its header.
        assert name in pins, f"{CI_LOCK.name} does not lock {name}"
        assert requirement.specifier.contains(
            pins[name]
        ), f"{CI_LOCK.name} locks {name}=={pins[name]}, outside {requirement}"
