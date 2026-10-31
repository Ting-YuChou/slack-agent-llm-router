import asyncio
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from slack.bot_real import SlackBot, SlackMessageHandler
from src.utils.schema import PlatformConfig


ROOT = Path(__file__).resolve().parents[1]
SETUP_GUIDE = ROOT / "docs" / "slack-setup.md"


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


DEMO_LOCKS = Path(".pi-agent-runtime") / "locks"
FAKE_DIGEST = "sha256:" + "a" * 64


def fake_executable(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_demo_lock_copies_are_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(DEMO_LOCKS / "plugins.lock.json")],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_demo_agent_images_locks_copies_and_leaves_tracked_locks_unchanged(tmp_path):
    fake_repo = tmp_path / "repo"
    runtime = fake_repo / "agent-runtime"
    runtime.mkdir(parents=True)
    for name in ("plugins.lock.json", "skills.lock.json"):
        shutil.copy(ROOT / "agent-runtime" / name, runtime / name)
    for name in ("plugins", "skills"):
        shutil.copytree(ROOT / "agent-runtime" / name, runtime / name)
    tracked = {
        name: (runtime / name).read_bytes()
        for name in ("plugins.lock.json", "skills.lock.json")
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_executable(bin_dir / "docker", "exit 0\n")
    # Mimics lock-image.mjs: it rewrites PI_AGENT_PLUGIN_LOCK, or the tracked lock without it.
    fake_executable(
        bin_dir / "npm",
        """if [[ " $* " == *" lock-image "* ]]; then
  lock="${PI_AGENT_PLUGIN_LOCK:-agent-runtime/plugins.lock.json}"
  printf '%s\\n' "${lock}" > "${FAKE_LOCK_LOG}"
  printf '{"schema_version": 1, "image_digest": "%s", "plugins": []}\\n' "${FAKE_DIGEST}" > "${lock}"
fi
""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DIGEST": FAKE_DIGEST,
        "FAKE_LOCK_LOG": str(tmp_path / "lock.log"),
    }

    result = subprocess.run(
        ["make", "-f", str(ROOT / "Makefile"), "demo-agent-images"],
        cwd=fake_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    locks = fake_repo / DEMO_LOCKS
    assert (tmp_path / "lock.log").read_text().strip() == str(
        locks / "plugins.lock.json"
    )
    for name, content in tracked.items():
        assert (runtime / name).read_bytes() == content
    assert FAKE_DIGEST in (locks / "plugins.lock.json").read_text()
    assert (locks / "skills.lock.json").read_bytes() == tracked["skills.lock.json"]
    for name in ("plugins", "skills"):
        for source in (runtime / name).rglob("*"):
            if source.is_file():
                copy = locks / source.relative_to(runtime)
                assert copy.read_bytes() == source.read_bytes()


def demo_launcher_repo(tmp_path, *, digest=FAKE_DIGEST, with_locks=True):
    """A repository holding only the launcher, with docker, node, npm, curl and python faked."""
    fake_repo = tmp_path / "repo"
    launcher = fake_repo / "scripts" / "run_slack_demo.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "run_slack_demo.sh", launcher)
    (fake_repo / "agent-runtime" / "node_modules").mkdir(parents=True)
    if with_locks:
        locks = fake_repo / DEMO_LOCKS
        locks.mkdir(parents=True)
        (locks / "plugins.lock.json").write_text(
            f'{{\n  "schema_version": 1,\n  "image_digest": "{digest}",\n'
            '  "plugins": []\n}\n',
            encoding="utf-8",
        )
        (locks / "skills.lock.json").write_text(
            '{\n  "schema_version": 1,\n  "skills": []\n}\n', encoding="utf-8"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    fake_executable(
        bin_dir / "docker",
        """printf 'docker %s\\n' "$*" >> "${FAKE_COMMAND_LOG}"
if [[ "${1:-}" == image && "${2:-}" == inspect ]]; then
  image="${*: -1}"
  if [[ " ${FAKE_MISSING_IMAGES:-} " == *" ${image} "* ]]; then exit 1; fi
  if [[ " $* " == *"{{.Id}}"* ]]; then printf '%s\\n' "${FAKE_IMAGE_DIGEST}"; fi
fi
exit 0
""",
    )
    fake_executable(
        bin_dir / "node",
        """if [[ "${1:-}" == "--version" ]]; then printf 'v22.22.3\\n'; exit 0; fi
if [[ "${1:-}" == "--input-type=module" ]]; then
  printf 'node image-check %s %s %s\\n' "${DEMO_RUNTIME_DIR:-}" "${DEMO_IMAGE_DIGEST:-}" "${DEMO_PLUGIN_LOCK:-}" >> "${FAKE_COMMAND_LOG}"
  if [[ -n "${FAKE_IMAGE_CHECK_ERROR:-}" ]]; then printf '%s\\n' "${FAKE_IMAGE_CHECK_ERROR}" >&2; exit 1; fi
  exit 0
fi
printf 'node %s\\n' "$*" >> "${FAKE_COMMAND_LOG}"
printf 'plugin-lock:%s\\nskill-lock:%s\\n' "${PI_AGENT_PLUGIN_LOCK-unset}" "${PI_AGENT_SKILL_LOCK-unset}"
""",
    )
    fake_executable(bin_dir / "npm", "exit 0\n")
    fake_executable(bin_dir / "curl", "printf '%s\\n' '{\"status\":\"healthy\"}'\n")
    fake_executable(
        bin_dir / "python",
        """printf 'python %s\\n' "$*" >> "${FAKE_COMMAND_LOG}"\n""",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON": str(bin_dir / "python"),
        "DEMO_ENV_FILE": str(tmp_path / "does-not-exist"),
        "OPENAI_API_KEY": "secret-openai-value",
        "SLACK_BOT_TOKEN": "xoxb-secret-value",
        "SLACK_APP_TOKEN": "xapp-secret-value",
        "AGENT_RUNTIME_TOKEN": "runtime-test",
        "MODEL_GATEWAY_SIGNING_SECRET": "gateway-test",
        "FAKE_COMMAND_LOG": str(log),
        "FAKE_IMAGE_DIGEST": FAKE_DIGEST,
    }
    for name in ("PI_AGENT_PLUGIN_LOCK", "PI_AGENT_SKILL_LOCK", "DEMO_TEST_MODE"):
        env.pop(name, None)
    return fake_repo, env, log


def run_demo_launcher(fake_repo, env, *args):
    return subprocess.run(
        ["bash", str(fake_repo / "scripts" / "run_slack_demo.sh"), *args],
        cwd=fake_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_demo_runner_points_the_runtime_at_the_lock_copies(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)

    result = run_demo_launcher(fake_repo, env)

    assert result.returncode == 0, result.stderr
    locks = fake_repo / DEMO_LOCKS
    assert f"plugin-lock:{locks / 'plugins.lock.json'}" in result.stdout
    assert f"skill-lock:{locks / 'skills.lock.json'}" in result.stdout
    assert "start-workers" in log.read_text()


@pytest.mark.parametrize("missing", ["plugins.lock.json", "skills.lock.json"])
def test_demo_runner_refuses_a_missing_lock_copy(tmp_path, missing):
    fake_repo, env, log = demo_launcher_repo(tmp_path)
    (fake_repo / DEMO_LOCKS / missing).unlink()

    result = run_demo_launcher(fake_repo, env)

    assert result.returncode == 2
    assert missing in result.stderr
    assert "make demo-agent-images" in result.stderr
    assert "docker run" not in log.read_text()
    assert "server.js" not in log.read_text()


def test_demo_runner_refuses_a_lock_copy_for_another_image(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path, digest="sha256:" + "b" * 64)

    result = run_demo_launcher(fake_repo, env)

    assert result.returncode == 2
    assert "make demo-agent-images" in result.stderr
    assert "docker run" not in log.read_text()
    assert "server.js" not in log.read_text()


def test_demo_check_passes_without_starting_anything(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)

    result = run_demo_launcher(fake_repo, env, "--check")

    assert result.returncode == 0, result.stderr
    assert "Demo preflight passed" in result.stdout
    commands = log.read_text()
    runtime = fake_repo / "agent-runtime"
    lock = fake_repo / DEMO_LOCKS / "plugins.lock.json"
    assert f"node image-check {runtime} {FAKE_DIGEST} {lock}" in commands
    assert "docker run" not in commands
    assert "docker network create" not in commands
    assert "docker ps" not in commands
    assert "server.js" not in commands
    assert "python" not in commands


def test_demo_check_reports_an_image_built_from_another_tree(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)
    env[
        "FAKE_IMAGE_CHECK_ERROR"
    ] = "Agent image was built from a different agent-runtime tree (changed: src/x.ts)"

    result = run_demo_launcher(fake_repo, env, "--check")

    assert result.returncode == 2
    assert "different agent-runtime tree (changed: src/x.ts)" in result.stderr
    assert "make demo-agent-images" in result.stderr
    assert "Demo preflight passed" not in result.stdout


def test_demo_check_names_missing_settings_without_printing_values(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)
    del env["SLACK_APP_TOKEN"]

    result = run_demo_launcher(fake_repo, env, "--check")

    assert result.returncode == 2
    assert "SLACK_APP_TOKEN" in result.stderr
    output = result.stdout + result.stderr
    for value in ("secret-openai-value", "xoxb-secret-value"):
        assert value not in output


def test_demo_check_reports_a_missing_image(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)
    env["FAKE_MISSING_IMAGES"] = "slack-pi-model-gateway:0.1.0"

    result = run_demo_launcher(fake_repo, env, "--check")

    assert result.returncode == 2
    assert "slack-pi-model-gateway:0.1.0" in result.stderr
    assert "make demo-agent-images" in result.stderr


def test_demo_runner_rejects_unknown_arguments(tmp_path):
    fake_repo, env, log = demo_launcher_repo(tmp_path)

    result = run_demo_launcher(fake_repo, env, "--chek")

    assert result.returncode == 2
    assert "--check" in result.stderr
    assert not log.exists()


def test_setup_guide_lists_every_bot_scope_the_manifest_requests():
    manifest = yaml.safe_load(
        (ROOT / "slack" / "app-manifest.demo.yaml").read_text(encoding="utf-8")
    )
    guide = SETUP_GUIDE.read_text(encoding="utf-8")

    scopes = manifest["oauth_config"]["scopes"]["bot"]
    assert "files:write" in scopes
    assert [scope for scope in scopes if f"`{scope}`" not in guide] == []


def demo_required_env_vars():
    """Every variable whose absence makes run_slack_demo.sh exit, optional MCP ones included."""
    launcher = (ROOT / "scripts" / "run_slack_demo.sh").read_text(encoding="utf-8")
    names = set()
    for group in re.findall(r"^\s*for name in ([A-Z0-9_ ]+); do$", launcher, re.M):
        names.update(group.split())
    names.update(
        re.findall(
            r'-z "\$\{([A-Z][A-Z0-9_]*):-\}" \]\]; then printf \'[^\']* is required',
            launcher,
        )
    )
    return names


def test_setup_guide_lists_every_env_var_the_launcher_requires():
    guide = SETUP_GUIDE.read_text(encoding="utf-8")

    required = demo_required_env_vars()
    assert {
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "GITHUB_APP_ID",
        "CONTEXT7_API_KEY",
    } <= required
    missing = sorted(
        name
        for name in required
        if not re.search(rf"(?<![A-Z0-9_]){name}(?![A-Z0-9_])", guide)
    )
    assert missing == []


@pytest.mark.asyncio
async def test_setup_guide_lists_every_agent_subcommand_in_the_help():
    handler = SlackMessageHandler(
        SimpleNamespace(
            agent_runtime_client=object(), _format_tier_limit_summary=lambda: ""
        )
    )
    help_text = await handler._handle_help_command([], "U1", "C1", None)
    guide = SETUP_GUIDE.read_text(encoding="utf-8")

    subcommands = [
        option
        for usage in re.findall(r"`/llm agent ([^`]+)`", help_text)
        for option in usage.split()[0].split("|")
    ]
    assert {"<task>", "--discuss", "--implement", "status", "close"} <= set(subcommands)
    assert [
        option for option in subcommands if f"/llm agent {option}" not in guide
    ] == []


@pytest.mark.asyncio
async def test_channel_allowlist_lookup_only_asks_for_public_channels():
    # The demo app has channels:read but no groups:read; asking for private
    # channels fails with missing_scope and leaves the allowlist unresolved.
    bot = SlackBot({"channels": ["ai-testing"]}, inference_engine=SimpleNamespace())
    bot.web_client = SimpleNamespace(
        conversations_list=AsyncMock(
            return_value={
                "channels": [{"id": "C123", "name": "ai-testing"}],
                "response_metadata": {},
            }
        )
    )

    await bot._resolve_allowed_channels()

    assert bot.web_client.conversations_list.await_args.kwargs["types"] == (
        "public_channel"
    )
    assert bot.allowed_channel_ids == {"C123"}
    assert bot.allowed_channel_names == set()


def test_setup_guide_says_result_feedback_needs_telemetry():
    guide = SETUP_GUIDE.read_text(encoding="utf-8")
    feedback = guide.split("### 結果回饋按鈕", 1)[1].split("\n## ", 1)[0]
    assert "PI_AGENT_TELEMETRY_ENABLED" in feedback
