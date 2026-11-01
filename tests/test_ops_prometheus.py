"""O1: the compose stack runs Prometheus with the router alert rules loaded."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import main

ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_DIR = ROOT / "docker" / "observability" / "prometheus"
RULES_FILE = "llm-router-alerts.yml"
REQUIRED_SECRETS = ("CLICKHOUSE_PASSWORD", "LLM_ROUTER_API_KEYS", "REDIS_PASSWORD")


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _prometheus_service():
    return _compose()["services"]["prometheus"]


def _prometheus_config():
    return yaml.safe_load(
        (PROMETHEUS_DIR / "prometheus.yml").read_text(encoding="utf-8")
    )


def _compose_app_config():
    return yaml.safe_load(
        (ROOT / "config" / "config.compose.yaml").read_text(encoding="utf-8")
    )


def _mounts():
    mounts = {}
    for volume in _prometheus_service()["volumes"]:
        source, target, mode = str(volume).split(":")
        mounts[source] = (target, mode)
    return mounts


def test_prometheus_runs_only_under_the_observability_profile():
    assert _prometheus_service()["profiles"] == ["observability"]


def test_prometheus_image_is_pinned():
    image = _prometheus_service()["image"]

    name, _, tag = image.partition(":")
    assert name == "prom/prometheus"
    assert tag.startswith("v") and tag != "latest"


def test_prometheus_port_is_bound_to_loopback():
    assert _prometheus_service()["ports"] == ["127.0.0.1:9090:9090"]


def test_prometheus_mounts_its_config_and_rules_read_only():
    mounts = _mounts()

    config_target, config_mode = mounts[
        "./docker/observability/prometheus/prometheus.yml"
    ]
    rules_target, rules_mode = mounts[f"./docker/observability/prometheus/{RULES_FILE}"]
    assert (config_mode, rules_mode) == ("ro", "ro")
    assert config_target == "/etc/prometheus/prometheus.yml"
    assert rules_target.endswith(f"/{RULES_FILE}")


def test_prometheus_config_loads_the_router_rules_file():
    rules_target, _ = _mounts()[f"./docker/observability/prometheus/{RULES_FILE}"]

    assert rules_target in _prometheus_config()["rule_files"]


def test_prometheus_scrapes_the_workers_metrics_port():
    port = _compose_app_config()["monitoring"]["prometheus_port"]
    jobs = {job["job_name"]: job for job in _prometheus_config()["scrape_configs"]}

    targets = [
        target
        for static in jobs["llm-router-workers"]["static_configs"]
        for target in static["targets"]
    ]
    assert targets == [f"workers:{port}"]


def test_compose_enables_the_workers_metrics_exporter_without_monitoring():
    monitoring = _compose_app_config()["monitoring"]

    assert monitoring["prometheus"]["enabled"] is True


def _write_config(tmp_path: Path, monitoring) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api": {"host": "127.0.0.1", "port": 8080},
                "logging": {"level": "INFO", "file": str(tmp_path / "router.log")},
                "monitoring": monitoring,
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("monitoring", "expected"),
    [
        ({"enabled": False, "prometheus": {"enabled": True}}, [9123]),
        ({"enabled": True, "prometheus": {"enabled": False}}, []),
        ({"enabled": True}, [9123]),
        ({"enabled": False}, []),
        ({"enabled": False, "prometheus": {"scrape_interval": "15s"}}, []),
    ],
    ids=[
        "exporter-on-monitoring-off",
        "exporter-off-monitoring-on",
        "falls-back-to-monitoring-on",
        "falls-back-to-monitoring-off",
        "exporter-unset",
    ],
)
def test_workers_metrics_server_follows_its_own_flag(
    tmp_path, monkeypatch, monitoring, expected
):
    started = []
    monkeypatch.setattr(main, "start_http_server", started.append)
    platform = main.LLMRouterPlatform(
        config_path=str(
            _write_config(tmp_path, {"prometheus_port": 9123, **monitoring})
        )
    )

    platform._start_metrics_server()

    assert started == expected


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not found")
def test_observability_profile_renders(tmp_path):
    env = {
        key: value for key, value in os.environ.items() if key not in REQUIRED_SECRETS
    }
    env.update({secret: f"test-{secret.lower()}" for secret in REQUIRED_SECRETS})
    empty_env_file = tmp_path / "empty.env"
    empty_env_file.write_text("")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(empty_env_file),
            "--profile",
            "observability",
            "config",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    services = yaml.safe_load(result.stdout)["services"]
    assert "prometheus" in services
    assert services["prometheus"]["ports"][0]["host_ip"] == "127.0.0.1"
