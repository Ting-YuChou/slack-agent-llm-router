"""Operator scripts and the policy cache authenticate against the compose stack."""

import asyncio
import json
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


CONTAINER_PASSWORD = "it's a $ecret; --password=x"
FAKE_CLICKHOUSE_CLIENT = """#!/bin/sh
printf '%s\\n' "$@" >> "$LOG_DIR/argv"
config=""
query=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config-file) config="$2"; shift 2 ;;
    --query) query="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\\n' "$config" >> "$LOG_DIR/config-paths"
cat "$config" > "$LOG_DIR/config"
case "$query" in
  EXISTS*) echo 1 ;;
  SHOW*) echo "ENGINE = MergeTree() request_id query_id model_name provider" ;;
esac
exit "${FAKE_EXIT:-0}"
"""


def _install_fake_clickhouse_client(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    client = bin_dir / "clickhouse-client"
    client.write_text(FAKE_CLICKHOUSE_CLIENT)
    client.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    return bin_dir, logs


def _assert_password_only_in_a_removed_config_file(logs):
    argv = (logs / "argv").read_text()
    assert "--password" not in argv
    assert "ecret" not in argv
    assert "--config-file" in argv
    config = yaml.safe_load((logs / "config").read_text())
    assert config == {"user": "llm_router", "password": CONTAINER_PASSWORD}
    paths = (logs / "config-paths").read_text().split()
    assert paths
    assert not any(Path(config_path).exists() for config_path in paths)


def test_clickhouse_migration_default_command_keeps_the_password_off_argv(
    tmp_path,
):
    from scripts import migrate_clickhouse_analytics_v2 as migration

    command = shlex.split(migration.parse_args([]).clickhouse_command)
    assert command[:5] == ["docker", "compose", "exec", "-T", "clickhouse"]
    assert "--password" not in " ".join(command)

    bin_dir, logs = _install_fake_clickhouse_client(tmp_path)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LOG_DIR": str(logs),
        "CLICKHOUSE_USER": "llm_router",
        "CLICKHOUSE_PASSWORD": CONTAINER_PASSWORD,
    }
    # Run what `docker compose exec` would run inside the container.
    result = subprocess.run(
        [*command[5:], "--query", "SELECT 1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (logs / "argv").read_text().splitlines()[-2:] == ["--query", "SELECT 1"]
    _assert_password_only_in_a_removed_config_file(logs)

    failed = subprocess.run(
        [*command[5:], "--query", "SELECT 1"],
        env={**env, "FAKE_EXIT": "3"},
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 3


def test_alert_events_migration_keeps_the_password_off_argv(tmp_path):
    bin_dir, logs = _install_fake_clickhouse_client(tmp_path)
    fake_docker = bin_dir / "docker"
    # Record the host command line, then run the rest as the container would,
    # with the credentials compose gives the ClickHouse container.
    fake_docker.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$@" >> "$LOG_DIR/docker-argv"\n'
        "shift 4\n"
        "export CLICKHOUSE_USER=llm_router\n"
        'export CLICKHOUSE_PASSWORD="$CONTAINER_PASSWORD"\n'
        'exec "$@"\n'
    )
    fake_docker.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CLICKHOUSE_")
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        LOG_DIR=str(logs),
        CONTAINER_PASSWORD=CONTAINER_PASSWORD,
    )

    result = subprocess.run(
        ["bash", str(SCRIPTS / "migrate_clickhouse_alert_events.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "already uses the current schema" in result.stdout
    docker_argv = (logs / "docker-argv").read_text()
    assert docker_argv.startswith("compose\nexec\n-T\nclickhouse\n")
    assert "--password" not in docker_argv
    assert "ecret" not in docker_argv
    _assert_password_only_in_a_removed_config_file(logs)


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


@pytest.mark.asyncio
async def test_loadtest_result_masks_the_password_in_the_redis_url(monkeypatch):
    from scripts import loadtest_redis_control_plane as loadtest

    class _FakeClient:
        async def ping(self):
            return True

        async def aclose(self):
            return None

    async def fake_run_mode(**kwargs):
        return {}

    monkeypatch.setattr(
        loadtest.redis.Redis, "from_url", lambda *args, **kwargs: _FakeClient()
    )
    monkeypatch.setattr(loadtest, "run_mode", fake_run_mode)
    monkeypatch.setattr(loadtest, "evaluate_gates", lambda *args: {"passed": True})
    args = Namespace(
        redis_url="redis://bench:s3cret-p%40ss@localhost:6379/15",
        requests=1,
        waiters=1,
        hold_ms=0.0,
        wait_timeout_ms=1,
        source_root=None,
        baseline=None,
    )

    result = await loadtest.run_benchmark(args)

    assert result["redis_url"] == "redis://bench:***@localhost:6379/15"
    assert "s3cret" not in json.dumps(result)


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379/15",
        "redis://bench@localhost:6379/15",
        "unix:///var/run/redis.sock?db=15",
    ],
)
def test_loadtest_shows_a_redis_url_without_a_password_unchanged(url):
    from scripts import loadtest_redis_control_plane as loadtest

    assert loadtest._display_url(url) == url


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "redis://localhost:6379/15?password=s3cret-pw",
            "redis://localhost:6379/15?password=***",
        ),
        (
            "unix:///run/redis.sock?db=15&password=s3cret-pw&x=1",
            "unix:///run/redis.sock?db=15&password=***&x=1",
        ),
        (
            "redis://u:s3cret-pw@localhost/0?password=s3cret-pw",
            "redis://u:***@localhost/0?password=***",
        ),
    ],
)
def test_loadtest_masks_a_password_query_parameter(url, expected):
    from scripts import loadtest_redis_control_plane as loadtest

    assert loadtest._display_url(url) == expected


def test_no_script_passes_a_password_to_clickhouse_client():
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(SCRIPTS.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "clickhouse-client" in line and "--password" in line
    ]

    assert offenders == []


def test_demo_applies_the_mcp_views_with_the_in_container_client_script(tmp_path):
    from scripts import migrate_clickhouse_analytics_v2 as migration

    launcher = (SCRIPTS / "run_slack_demo.sh").read_text(encoding="utf-8")
    assert (
        'docker exec -i "${clickhouse_container}" sh -c '
        '"${clickhouse_client_script}" sh --multiquery'
    ) in launcher
    shared = (SCRIPTS / "clickhouse_client_in_container.sh").read_text(encoding="utf-8")
    assert migration.CLICKHOUSE_CLIENT_SCRIPT == shared
    assert "clickhouse_client_in_container.sh" in launcher
    assert "clickhouse_client_in_container.sh" in (
        SCRIPTS / "migrate_clickhouse_alert_events.sh"
    ).read_text(encoding="utf-8")


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
