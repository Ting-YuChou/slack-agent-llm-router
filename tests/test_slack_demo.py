import asyncio
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from slack.bot_real import SlackBot
from src.utils.schema import PlatformConfig


ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_is_single_provider_and_needs_no_infrastructure():
    raw_config = yaml.safe_load(
        (ROOT / "config" / "config.demo.yaml").read_text(encoding="utf-8")
    )
    config = PlatformConfig.model_validate(raw_config)

    assert config.router.default_model == "gpt-5"
    assert config.router.fast_lane_models == ["gpt-5"]
    assert set(config.router.models) == {"gpt-5"}
    assert config.router.models["gpt-5"].provider == "openai"

    assert config.inference.openai.enabled is True
    assert config.inference.anthropic.enabled is False
    assert config.inference.vllm.enabled is False
    assert config.inference.cache.enabled is False
    assert config.inference.scheduler.enabled is False
    assert config.api.rate_limiting.enabled is False

    assert config.slack.enabled is True
    assert config.slack.bot_token_env == "SLACK_BOT_TOKEN"
    assert config.slack.app_token_env == "SLACK_APP_TOKEN"
    assert config.slack.state_backend == "file"
    assert config.slack.state_file == ".pi-agent-runtime/slack_state.json"
    assert config.slack.memory.enabled is False
    assert config.slack.agent_context.enabled is True
    assert config.slack.agent_context.max_thread_messages == 20
    assert config.slack.agent_context.max_message_chars == 4000
    assert config.slack.agent_context.max_resources == 10
    assert config.slack.agent_context.max_resource_bytes == 262144
    assert config.slack.agent_context.max_resource_chars == 8000
    assert config.slack.agent_context.max_total_chars == 12000
    assert config.slack.agent_context.timeout_seconds == 10
    assert config.slack.agent_context.include_bot_messages is False
    assert config.slack.agent_context.fail_open is True

    assert config.kafka.enabled is False
    assert config.clickhouse.enabled is False
    assert config.flink.enabled is False
    assert config.pipeline.enabled is False
    assert config.rag.enabled is False
    assert config.tools.web_search.enabled is False
    assert config.agent.enabled is True
    assert config.agent.base_url == "http://127.0.0.1:3001"
    assert config.agent.token_env == "AGENT_RUNTIME_TOKEN"
    assert config.agent.connect_timeout_seconds == 2
    assert config.agent.request_timeout_seconds == 910
    assert config.agent.feedback_enabled is True


def test_demo_manifest_declares_socket_mode_command_and_events():
    manifest = yaml.safe_load(
        (ROOT / "slack" / "app-manifest.demo.yaml").read_text(encoding="utf-8")
    )

    assert manifest["settings"]["socket_mode_enabled"] is True
    assert manifest["settings"]["interactivity"]["is_enabled"] is True
    assert manifest["features"]["slash_commands"] == [
        {
            "command": "/llm",
            "description": "Ask the single-model LLM Router demo",
            "usage_hint": "help | agent <task|status|stop|close> | your question",
            "should_escape": False,
        }
    ]
    assert manifest["features"]["shortcuts"] == [
        {
            "name": "Run Pi Agent",
            "type": "message",
            "callback_id": "run_pi_agent_from_thread",
            "description": "Start a Pi coding agent with this Slack thread as context",
        }
    ]
    assert set(manifest["settings"]["event_subscriptions"]["bot_events"]) == {
        "app_mention",
        "message.channels",
    }
    assert {
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "commands",
        "files:read",
        "lists:read",
    }.issubset(set(manifest["oauth_config"]["scopes"]["bot"]))


# Bot-token scopes each Slack Web API method needs; an empty set means none.
WEB_API_METHOD_SCOPES = {
    "auth_test": set(),
    "chat_postEphemeral": {"chat:write"},
    "chat_postMessage": {"chat:write"},
    "chat_update": {"chat:write"},
    "conversations_info": {"channels:read"},
    "conversations_list": {"channels:read"},
    "files_upload_v2": {"files:write"},
    "views_open": set(),
}


def test_demo_manifest_grants_every_scope_the_bot_calls_need():
    source = (ROOT / "slack" / "bot_real.py").read_text(encoding="utf-8")
    called = set(re.findall(r"\bweb_client\.([A-Za-z_][A-Za-z0-9_]*)\(", source))
    manifest = yaml.safe_load(
        (ROOT / "slack" / "app-manifest.demo.yaml").read_text(encoding="utf-8")
    )
    granted = set(manifest["oauth_config"]["scopes"]["bot"])

    assert "files_upload_v2" in called
    unmapped = called - set(WEB_API_METHOD_SCOPES)
    assert not unmapped, f"add the required scopes for {sorted(unmapped)}"
    missing = {
        method: sorted(WEB_API_METHOD_SCOPES[method] - granted)
        for method in sorted(called)
        if WEB_API_METHOD_SCOPES[method] - granted
    }
    assert missing == {}


def test_demo_state_file_is_gitignored():
    raw_config = yaml.safe_load(
        (ROOT / "config" / "config.demo.yaml").read_text(encoding="utf-8")
    )
    state_file = raw_config["slack"]["state_file"]

    result = subprocess.run(
        ["git", "check-ignore", "--quiet", state_file],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0


def demo_bot(tmp_path, **services):
    raw_config = yaml.safe_load(
        (ROOT / "config" / "config.demo.yaml").read_text(encoding="utf-8")
    )
    config = PlatformConfig.model_validate(raw_config)
    slack_config = config.slack.model_dump()
    slack_config["state_file"] = str(tmp_path / Path(config.slack.state_file).name)
    return SlackBot(
        slack_config,
        SimpleNamespace(),
        services=services,
        agent_config=config.agent.model_dump(),
    )


@pytest.mark.asyncio
async def test_demo_bot_restart_restores_agent_runs_ci_jobs_and_monitors(tmp_path):
    first = demo_bot(tmp_path)
    tracked = first._track_agent_run(
        {"session_id": "S1", "run_id": "R1"},
        team_id="T1",
        channel_id="C1",
        thread_ts="1.0",
        owner_user_id="U1",
        mode="implement",
    )
    await first._persist_agent_run(tracked)
    assert await first._claim_ci_job("J1") is True

    second = demo_bot(tmp_path, agent_runtime=SimpleNamespace())
    second._monitor_agent_run = AsyncMock()
    await second._restore_state()
    second._reconcile_agent_runs()
    while second.background_tasks:
        await asyncio.gather(*list(second.background_tasks))

    assert second.agent_runs["R1"]["session_id"] == "S1"
    assert second._is_agent_thread("C1", "1.0")
    assert list(second.posted_ci_jobs) == ["J1"]
    assert await second._claim_ci_job("J1") is False
    second._monitor_agent_run.assert_awaited_once()
    assert second._monitor_agent_run.await_args.args[0]["run_id"] == "R1"


def test_demo_runner_starts_agent_sidecar_then_worker_runtime(tmp_path):
    fake_repo = tmp_path / "clean-repo"
    launcher = fake_repo / "scripts" / "run_slack_demo.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'agent-keys:%s:%s openai:%s openrouter:%s\\n' "
        '"${ANTHROPIC_API_KEY-unset}" "${OPENCODE_API_KEY-unset}" '
        '"${OPENAI_API_KEY-unset}" "${OPENROUTER_API_KEY-unset}"\n'
        "printf '%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_node = tmp_path / "node"
    fake_node.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf 'v22.22.3\\n'
else
  printf 'agent-runtime node\\n'
  printf 'openrouter:%s\\n' "${OPENROUTER_API_KEY:+present}"
fi
""",
        encoding="utf-8",
    )
    fake_node.chmod(0o755)
    fake_npm = tmp_path / "npm"
    fake_npm.write_text(
        "#!/usr/bin/env bash\nprintf 'npm %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '{\"status\":\"healthy\"}'\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PYTHON": str(fake_python),
        "DEMO_ENV_FILE": str(tmp_path / "does-not-exist"),
        "OPENAI_API_KEY": "test-openai",
        "ANTHROPIC_API_KEY": "test-anthropic",
        "OPENCODE_API_KEY": "test-opencode",
        "OPENROUTER_API_KEY": "test-openrouter",
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "AGENT_RUNTIME_TOKEN": "runtime-test",
        "MODEL_GATEWAY_SIGNING_SECRET": "gateway-test",
        "DEMO_TEST_MODE": "1",
    }
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=fake_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "npm --prefix" in result.stdout
    assert "agent-runtime node" in result.stdout
    assert "openrouter:present" in result.stdout
    assert "agent-keys:unset:unset openai:test-openai openrouter:unset" in result.stdout
    assert result.stdout.strip().endswith(
        "main.py start-workers --config config/config.demo.yaml"
    )


def test_demo_runner_reports_missing_secrets_without_printing_values(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"}
    }
    env["DEMO_ENV_FILE"] = str(tmp_path / "does-not-exist")

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_slack_demo.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "OPENAI_API_KEY" in result.stderr
    assert "SLACK_BOT_TOKEN" in result.stderr
    assert "SLACK_APP_TOKEN" in result.stderr


def test_demo_runner_signal_stops_agent_and_python_children(tmp_path):
    node_pid_file = tmp_path / "node.pid"
    python_pid_file = tmp_path / "python.pid"
    fake_node = tmp_path / "node"
    fake_node.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf 'v22.22.3\\n'
  exit 0
fi
printf '%s\\n' "$$" > "${FAKE_NODE_PID_FILE}"
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
        encoding="utf-8",
    )
    fake_node.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$$" > "${FAKE_PYTHON_PID_FILE}"
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_npm = tmp_path / "npm"
    fake_npm.write_text(
        """#!/usr/bin/env bash
if [[ " $* " == *" start "* ]]; then
  printf '%s\\n' "$$" > "${FAKE_NODE_PID_FILE}"
  trap 'exit 0' TERM INT
  while true; do sleep 0.1; done
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_npm.chmod(0o755)
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '{\"status\":\"healthy\"}'\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PYTHON": str(fake_python),
        "DEMO_ENV_FILE": str(tmp_path / "does-not-exist"),
        "OPENAI_API_KEY": "test-openai",
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "AGENT_RUNTIME_TOKEN": "runtime-test",
        "MODEL_GATEWAY_SIGNING_SECRET": "gateway-test",
        "DEMO_TEST_MODE": "1",
        "FAKE_NODE_PID_FILE": str(node_pid_file),
        "FAKE_PYTHON_PID_FILE": str(python_pid_file),
    }
    launcher = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / "run_slack_demo.sh")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    child_pids = []
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if node_pid_file.exists() and python_pid_file.exists():
                break
            time.sleep(0.02)
        assert node_pid_file.exists()
        assert python_pid_file.exists()
        child_pids = [
            int(node_pid_file.read_text().strip()),
            int(python_pid_file.read_text().strip()),
        ]

        launcher.terminate()
        launcher.wait(timeout=3)

        for child_pid in child_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=3)
        for child_pid in child_pids:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_demo_env_and_makefile_include_agent_setup():
    env_example = (ROOT / "config" / "demo.env.example").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "AGENT_RUNTIME_TOKEN=" in env_example
    assert "MODEL_GATEWAY_SIGNING_SECRET=" in env_example
    assert "MCP_GATEWAY_SIGNING_SECRET=" in env_example
    assert "PI_AGENT_JEV_MODE=on" in env_example
    assert "PI_AGENT_JEV_CLASSIFIER_MODE=on" in env_example
    assert "PI_AGENT_MCP_MODE=off" in env_example
    assert "PI_AGENT_MCP_SERVERS=github" in env_example
    assert "PI_AGENT_TELEMETRY_ENABLED=true" in env_example
    assert "PI_AGENT_OTEL_ENABLED=true" in env_example
    assert "PI_AGENT_MCP_GITHUB_GATEWAY_URL=http://mcp-github:8090/mcp" in env_example
    assert (
        "PI_AGENT_MCP_CLICKHOUSE_GATEWAY_URL=http://mcp-clickhouse-gateway:8090/mcp"
        in env_example
    )
    assert (
        "PI_AGENT_MCP_CONTEXT7_GATEWAY_URL=http://mcp-context7:8090/mcp" in env_example
    )
    assert "PI_AGENT_GITHUB_REPOSITORIES=" in env_example
    assert "ANTHROPIC_API_KEY=" in env_example
    assert "OPENCODE_API_KEY=" in env_example
    assert "demo-agent-install:" in makefile
    assert "npm --prefix agent-runtime ci --ignore-scripts" in makefile
    assert "demo-agent-images:" in makefile
    assert "run lock-image -- slack-pi-agent:1.0.1" in makefile
    assert (
        "github/github-mcp-server@sha256:7aaeeec9ae4fe9a736d100c1ff0798f3c219b5009e05f5d3945fcacb13cc196b"
        in makefile
    )


def test_demo_gateway_uses_the_hostname_required_by_agent_policy():
    launcher = (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8")
    gateway_block = launcher.split('--name "${gateway_container}"', 1)[1].split(
        "gateway_started=true", 1
    )[0]

    assert "--network-alias model-gateway" in launcher
    assert "--env ANTHROPIC_API_KEY" in launcher
    assert "--env OPENCODE_API_KEY" in launcher
    assert "--env OPENROUTER_API_KEY" in gateway_block
    assert "PI_AGENT_CONFIGURED_PROVIDERS" in launcher
    assert "env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u OPENCODE_API_KEY" in launcher
    assert (
        "-u GITHUB_APP_ID -u GITHUB_APP_INSTALLATION_ID -u GITHUB_APP_PRIVATE_KEY_PATH"
        in launcher
    )
    assert (
        "-u MCP_GATEWAY_SIGNING_SECRET -u GITHUB_APP_ID -u GITHUB_APP_INSTALLATION_ID"
        in launcher
    )
    assert "-u GITHUB_APP_PRIVATE_KEY_PATH -u PI_AGENT_GITHUB_REPOSITORIES" in launcher


def test_demo_mcp_gateway_has_dual_networks_and_cleanup():
    launcher = (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8")

    assert "--network-alias mcp-github" in launcher
    assert "--network-alias mcp-clickhouse-gateway" in launcher
    assert "--network-alias mcp-context7" in launcher
    assert (
        'docker network connect "${mcp_egress_network}" "${mcp_gateway_container}"'
        in launcher
    )
    assert "GITHUB_READ_ONLY=1" in launcher
    assert "GITHUB_LOCKDOWN_MODE=1" in launcher
    assert (
        "GITHUB_TOOLSETS=repos,issues,pull_requests,actions,code_security,dependabot,secret_protection"
        in launcher
    )
    assert (
        "GITHUB_TOOLS=get_file_contents,search_code,issue_read,pull_request_read,list_issues,list_pull_requests,actions_get,actions_list,get_job_logs"
        in launcher
    )
    assert (
        '--mount "type=bind,src=${GITHUB_APP_PRIVATE_KEY_PATH},dst=/run/secrets/github-app.pem,readonly"'
        in launcher
    )
    assert "--env GITHUB_APP_PRIVATE_KEY " not in launcher
    assert 'docker stop "${mcp_gateway_container}"' in launcher
    assert 'docker stop "${github_mcp_container}"' in launcher
    assert 'docker stop "${clickhouse_mcp_container}"' in launcher
    assert 'docker stop "${clickhouse_gateway_container}"' in launcher
    assert 'docker stop "${context7_gateway_container}"' in launcher
    assert 'docker network rm "${mcp_egress_network}"' in launcher


def test_demo_mcp_gateways_ask_the_host_whether_a_run_is_still_active():
    launcher = (ROOT / "scripts/run_slack_demo.sh").read_text()
    revocation = (
        '--env "PI_AGENT_REVOCATION_RPC_URL='
        'http://host.docker.internal:${PI_AGENT_PORT:-3001}"'
    )
    for secret in ("github", "clickhouse", "context7"):
        launch = launcher.split(f"${{{secret}_mcp_signing_secret}}", 1)[1]
        assert revocation in launch.split('"${mcp_gateway_image}"', 1)[0], secret


def test_demo_applies_agent_mcp_views_before_starting_clickhouse_mcp():
    launcher = (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8")
    clickhouse_block = launcher.split(
        'if [[ "${clickhouse_mcp_enabled}" == true ]]; then\n    observability_network=',
        1,
    )[1].split("clickhouse_mcp_started=true", 1)[0]

    views = clickhouse_block.index(
        '<"${repo_root}/docker/observability/agent-mcp-views.sql"'
    )
    assert 'docker exec -i "${clickhouse_container}"' in clickhouse_block
    assert "--multiquery" in clickhouse_block
    assert views < clickhouse_block.index('--name "${clickhouse_mcp_container}"')
