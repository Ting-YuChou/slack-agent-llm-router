import os
from pathlib import Path
import subprocess
import time

import pytest
import yaml

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
    assert config.slack.state_backend == "memory"
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
        "printf 'agent-keys:%s:%s openai:%s\\n' "
        '"${ANTHROPIC_API_KEY-unset}" "${OPENCODE_API_KEY-unset}" '
        '"${OPENAI_API_KEY-unset}"\n'
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
    assert "agent-keys:unset:unset openai:test-openai" in result.stdout
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
    assert "ANTHROPIC_API_KEY=" in env_example
    assert "OPENCODE_API_KEY=" in env_example
    assert "demo-agent-install:" in makefile
    assert "npm --prefix agent-runtime ci --ignore-scripts" in makefile
    assert "demo-agent-images:" in makefile
    assert "run lock-image -- slack-pi-agent:0.83.0" in makefile


def test_demo_gateway_uses_the_hostname_required_by_agent_policy():
    launcher = (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8")

    assert "--network-alias model-gateway" in launcher
    assert "--env ANTHROPIC_API_KEY" in launcher
    assert "--env OPENCODE_API_KEY" in launcher
    assert "PI_AGENT_CONFIGURED_PROVIDERS" in launcher
    assert "env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u OPENCODE_API_KEY" in launcher
