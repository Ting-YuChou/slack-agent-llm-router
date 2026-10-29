"""Operator scripts and the policy cache authenticate against the compose stack."""

import asyncio
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class _StopAfterConnect(Exception):
    pass


def _capturing_client(captured):
    def factory(*args, **kwargs):
        captured.append({"args": args, **kwargs})
        raise _StopAfterConnect()

    return factory


def test_scripts_have_no_literal_default_credentials():
    offenders = [
        f"{path.relative_to(ROOT)}: {literal}"
        for path in sorted(SCRIPTS.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
        for literal in ("llm_router_pass", "dev-api-key")
        if literal in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_integration_smoke_reads_the_clickhouse_password_from_the_environment(
    monkeypatch,
):
    from scripts import integration_smoke

    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "from-env")
    assert integration_smoke.parse_args([]).clickhouse_password == "from-env"
    assert (
        integration_smoke.parse_args(
            ["--clickhouse-password", "from-flag"]
        ).clickhouse_password
        == "from-flag"
    )

    monkeypatch.delenv("CLICKHOUSE_PASSWORD")
    with pytest.raises(SystemExit):
        integration_smoke.parse_args([])


def test_analytics_contract_reads_the_clickhouse_password_from_the_environment(
    monkeypatch,
):
    from scripts import analytics_performance_contract as contract

    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "from-env")
    assert contract.parse_args([]).clickhouse_password == "from-env"
    assert (
        contract.parse_args(["--clickhouse-password", "x"]).clickhouse_password == "x"
    )

    monkeypatch.delenv("CLICKHOUSE_PASSWORD")
    with pytest.raises(SystemExit):
        contract.parse_args([])
    assert contract.parse_args(["--flink-only"]).flink_only is True


def test_analytics_contract_benchmark_defaults_to_the_environment_password(
    monkeypatch,
):
    from scripts.analytics_performance_contract import benchmark_clickhouse

    captured = []
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "from-env")
    monkeypatch.setitem(
        sys.modules,
        "clickhouse_connect",
        Namespace(get_client=_capturing_client(captured)),
    )

    benchmark_clickhouse(rows=1, repeats=1)

    assert captured[0]["password"] == "from-env"


def test_clickhouse_migration_default_command_uses_the_container_password(
    tmp_path,
):
    from scripts import migrate_clickhouse_analytics_v2 as migration

    command = shlex.split(migration.parse_args([]).clickhouse_command)
    assert command[:5] == ["docker", "compose", "exec", "-T", "clickhouse"]

    fake_client = tmp_path / "clickhouse-client"
    fake_client.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_client.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CLICKHOUSE_USER": "llm_router",
        "CLICKHOUSE_PASSWORD": "container-secret",
    }
    # Run what `docker compose exec` would run inside the container.
    result = subprocess.run(
        [*command[5:], "--query", "SELECT 1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "--user",
        "llm_router",
        "--password",
        "container-secret",
        "--query",
        "SELECT 1",
    ]


def test_alert_events_migration_requires_the_clickhouse_password(tmp_path):
    calls = tmp_path / "docker-calls"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    fake_docker.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items() if key != "CLICKHOUSE_PASSWORD"
    }
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "migrate_clickhouse_alert_events.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "CLICKHOUSE_PASSWORD" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    "module_name, argv",
    [
        ("cleanup_v1_provider_scheduler_keys", ["cleanup"]),
        ("loadtest_redis_control_plane", ["loadtest"]),
    ],
)
def test_redis_url_scripts_send_the_environment_password(
    monkeypatch, module_name, argv
):
    module = __import__(f"scripts.{module_name}", fromlist=["main"])
    captured = []
    monkeypatch.setenv("REDIS_PASSWORD", "from-env")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module.redis.Redis, "from_url", _capturing_client(captured))

    with pytest.raises(_StopAfterConnect):
        asyncio.run(module.main())

    assert captured[0]["password"] == "from-env"


def test_redis_url_scripts_without_a_password_stay_unauthenticated(monkeypatch):
    from scripts import cleanup_v1_provider_scheduler_keys as cleanup

    captured = []
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["cleanup"])
    monkeypatch.setattr(cleanup.redis.Redis, "from_url", _capturing_client(captured))

    with pytest.raises(_StopAfterConnect):
        asyncio.run(cleanup.main())

    assert captured[0]["password"] is None


def test_loadtest_monitor_connection_sends_the_environment_password(monkeypatch):
    from scripts import loadtest_redis_control_plane as loadtest

    captured = []
    monkeypatch.setenv("REDIS_PASSWORD", "from-env")
    monkeypatch.setattr(loadtest.redis.Redis, "from_url", _capturing_client(captured))

    async def operation():
        return None

    with pytest.raises(_StopAfterConnect):
        asyncio.run(loadtest.capture_redis_commands("redis://localhost/15", operation))

    assert captured[0]["password"] == "from-env"


def test_rag_benchmark_sends_the_environment_password(monkeypatch):
    import redis.asyncio

    from scripts import rag_performance_contract as rag_contract

    captured = []
    monkeypatch.setenv("REDIS_PASSWORD", "from-env")
    monkeypatch.setattr(redis.asyncio, "Redis", _capturing_client(captured))

    with pytest.raises(_StopAfterConnect):
        asyncio.run(rag_contract._redis_benchmark(1, "localhost", 6380))

    assert captured[0]["password"] == "from-env"


def test_rag_index_migration_sends_the_configured_password(monkeypatch, tmp_path):
    import redis.asyncio

    from scripts import migrate_rag_index_v2 as rag_migration

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"rag": {"redis": {"host": "redis-stack", "port": 6379}}})
    )
    captured = []
    monkeypatch.setenv("REDIS_PASSWORD", "from-env")
    monkeypatch.setattr(redis.asyncio, "Redis", _capturing_client(captured))
    args = Namespace(
        config=str(config),
        batch_size=1,
        apply=False,
        rollback=False,
        pause_command=None,
        resume_command=None,
        switch_command=None,
        rollback_command=None,
    )

    with pytest.raises(_StopAfterConnect):
        asyncio.run(rag_migration.migrate(args))

    assert captured[0]["password"] == "from-env"


class _FakePolicyRedis:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakePolicyRedis.instances.append(self)

    async def ping(self):
        return True


@pytest.mark.asyncio
async def test_policy_cache_authenticates_with_password_env(monkeypatch):
    from src import llm_router_part3_policy as policy

    _FakePolicyRedis.instances.clear()
    monkeypatch.setenv("POLICY_TEST_REDIS_PASSWORD", "policy-secret")
    monkeypatch.setattr(policy.redis, "Redis", _FakePolicyRedis)
    cache = policy.RoutingPolicyCache(
        {
            "enabled": True,
            "redis": {"host": "redis", "password_env": "POLICY_TEST_REDIS_PASSWORD"},
        }
    )

    await cache.initialize()

    assert cache.enabled is True
    assert _FakePolicyRedis.instances[0].kwargs["password"] == "policy-secret"


@pytest.mark.asyncio
async def test_policy_cache_without_password_env_connects_without_auth(monkeypatch):
    from src import llm_router_part3_policy as policy

    _FakePolicyRedis.instances.clear()
    monkeypatch.setattr(policy.redis, "Redis", _FakePolicyRedis)
    cache = policy.RoutingPolicyCache({"enabled": True, "redis": {"host": "redis"}})

    await cache.initialize()

    assert _FakePolicyRedis.instances[0].kwargs.get("password") is None


def test_compose_policy_cache_uses_the_redis_password():
    config = yaml.safe_load(
        (ROOT / "config" / "config.compose.yaml").read_text(encoding="utf-8")
    )

    assert config["policy_cache"]["redis"]["password_env"] == "REDIS_PASSWORD"
