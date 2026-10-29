import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.y*ml"))
REQUIRED_SECRETS = ("CLICKHOUSE_PASSWORD", "LLM_ROUTER_API_KEYS", "REDIS_PASSWORD")
APP_SERVICES = ("api", "workers", "rag-workers")
FLINK_SERVICES = ("flink-jobmanager", "flink-taskmanager")


class _ComposeLoader(yaml.SafeLoader):
    """Safe loader that reads compose tags such as `!override` as plain values."""


def _construct_tagged(loader, _tag_suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _construct_tagged)


def _load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)


def _base():
    return _load(ROOT / "docker-compose.yml")


def _required(name: str) -> re.Pattern:
    return re.compile(rf"^\$\{{{name}:\?[^}}]+\}}$")


def _port_host_ip(port) -> str:
    if isinstance(port, dict):
        return str(port.get("host_ip") or "")
    text = str(port)
    parts = text.split(":")
    return parts[0] if len(parts) == 3 else ""


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda path: path.name)
def test_every_published_port_is_bound_to_loopback(path):
    services = (_load(path) or {}).get("services", {}) or {}
    published = [
        (name, port)
        for name, service in services.items()
        for port in (service or {}).get("ports", []) or []
    ]

    exposed = [
        (name, port) for name, port in published if _port_host_ip(port) != "127.0.0.1"
    ]
    assert exposed == []


def test_base_compose_publishes_ports_for_the_local_stack():
    services = _base()["services"]

    assert services["redis"]["ports"] == ["127.0.0.1:6379:6379"]
    assert services["kafka"]["ports"] == ["127.0.0.1:9092:9092"]
    assert services["clickhouse"]["ports"] == [
        "127.0.0.1:8123:8123",
        "127.0.0.1:9000:9000",
    ]
    assert services["api"]["ports"] == ["127.0.0.1:8080:8080"]
    assert services["flink-jobmanager"]["ports"] == ["127.0.0.1:8081:8081"]


@pytest.mark.parametrize(
    "path",
    [ROOT / "docker-compose.yml", ROOT / "config" / "config.compose.yaml"],
    ids=lambda path: path.name,
)
def test_no_literal_default_credentials(path):
    text = path.read_text(encoding="utf-8")

    assert "llm_router_pass" not in text
    assert "dev-api-key" not in text


@pytest.mark.parametrize("service_name", APP_SERVICES)
@pytest.mark.parametrize("secret", REQUIRED_SECRETS)
def test_app_services_require_each_secret(service_name, secret):
    environment = _base()["services"][service_name]["environment"]

    assert _required(secret).match(str(environment[secret])), environment[secret]


def test_clickhouse_requires_a_password_and_its_healthcheck_uses_it():
    clickhouse = _base()["services"]["clickhouse"]

    assert _required("CLICKHOUSE_PASSWORD").match(
        clickhouse["environment"]["CLICKHOUSE_PASSWORD"]
    )
    healthcheck = " ".join(clickhouse["healthcheck"]["test"])
    assert "$${CLICKHOUSE_PASSWORD}" in healthcheck


def test_compose_config_reads_the_clickhouse_password_from_the_environment():
    config = _load(ROOT / "config" / "config.compose.yaml")

    assert config["clickhouse"]["password"] == ""


@pytest.mark.parametrize("service_name", ["redis", "redis-stack"])
def test_redis_services_require_the_password(service_name):
    service = _base()["services"][service_name]
    environment = service.get("environment", {})
    startup = " ".join(
        [
            " ".join(service.get("command", []) or []),
            str(environment.get("REDIS_ARGS", "")),
        ]
    )
    healthcheck = " ".join(service["healthcheck"]["test"])

    assert _required("REDIS_PASSWORD").match(str(environment["REDIS_PASSWORD"]))
    assert "--requirepass" in startup
    assert "REDISCLI_AUTH" in healthcheck
    assert "PONG" in healthcheck


def test_redis_keeps_append_only_persistence():
    command = " ".join(_base()["services"]["redis"]["command"])

    assert "--appendonly yes" in command


@pytest.mark.parametrize("service_name", FLINK_SERVICES)
def test_flink_mounts_only_its_code_read_only(service_name):
    service = _base()["services"][service_name]
    volumes = [str(volume) for volume in service["volumes"]]

    assert volumes == [
        "./flink:/workspace/flink:ro",
        "./src:/workspace/src:ro",
        "flink_checkpoints:/opt/flink/checkpoints",
    ]
    assert service["environment"]["PYTHONPATH"] == "/workspace"


def _compose_config(env_overrides, tmp_path):
    env = {
        key: value for key, value in os.environ.items() if key not in REQUIRED_SECRETS
    }
    env.update(env_overrides)
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("")
    return subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(empty_env_file),
            "--profile",
            "rag",
            "config",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not found")
def test_docker_compose_refuses_to_render_without_the_secrets(tmp_path):
    result = _compose_config({}, tmp_path)

    assert result.returncode != 0
    assert any(secret in result.stderr for secret in REQUIRED_SECRETS)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not found")
def test_docker_compose_renders_with_the_secrets(tmp_path):
    result = _compose_config(
        {secret: f"test-{secret.lower()}" for secret in REQUIRED_SECRETS}, tmp_path
    )

    assert result.returncode == 0, result.stderr
    resolved = yaml.safe_load(result.stdout)
    assert (
        resolved["services"]["api"]["environment"]["REDIS_PASSWORD"]
        == "test-redis_password"
    )
