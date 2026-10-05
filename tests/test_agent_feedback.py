import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.agent_runtime import AgentRuntimeClient, AgentRuntimeUnauthorized
from slack.bot_real import SlackBot


@pytest.mark.asyncio
async def test_feedback_client_uses_authenticated_endpoint(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_TOKEN", "runtime-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"status": "accepted"})

    runtime = AgentRuntimeClient({}, transport=httpx.MockTransport(handler))
    try:
        await runtime.feedback("run1", "user1", "needs_changes", "click1")
    finally:
        await runtime.close()
    assert requests[0].url.path == "/v1/runs/run1/feedback"
    assert requests[0].headers["authorization"] == "Bearer runtime-secret"
    assert json.loads(requests[0].content) == {
        "user_id": "user1",
        "verdict": "needs_changes",
        "feedback_id": "click1",
    }


@pytest.mark.asyncio
async def test_slack_feedback_does_not_depend_on_active_run_cache_or_call_approval():
    runtime = SimpleNamespace(feedback=AsyncMock(), decide=AsyncMock())
    bot = SimpleNamespace(
        agent_runtime_client=runtime,
        web_client=SimpleNamespace(chat_postEphemeral=AsyncMock()),
    )
    payload = {
        "user": {"id": "owner"},
        "channel": {"id": "channel"},
        "actions": [
            {
                "action_id": "pi_agent_accept_result",
                "value": json.dumps({"run_id": "old-run"}),
                "action_ts": "100.1",
            }
        ],
    }
    await SlackBot._handle_agent_interactive(bot, payload)
    runtime.feedback.assert_awaited_once_with("old-run", "owner", "accepted", "100.1")
    runtime.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_feedback_reports_runtime_owner_rejection():
    runtime = SimpleNamespace(
        feedback=AsyncMock(side_effect=AgentRuntimeUnauthorized())
    )
    bot = SimpleNamespace(
        agent_runtime_client=runtime,
        web_client=SimpleNamespace(chat_postEphemeral=AsyncMock()),
    )
    await SlackBot._handle_agent_interactive(
        bot,
        {
            "user": {"id": "other"},
            "channel": {"id": "channel"},
            "actions": [
                {
                    "action_id": "pi_agent_needs_changes",
                    "value": json.dumps({"run_id": "r"}),
                    "action_ts": "101.1",
                }
            ],
        },
    )
    assert "owner" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_feedback_buttons_preserve_long_result_with_valid_slack_sections():
    answer = "中文" * 2000
    bot = SimpleNamespace(
        agent_runtime_client=SimpleNamespace(
            wait_for_update=AsyncMock(
                return_value={"run_id": "r", "status": "completed"}
            )
        ),
        web_client=SimpleNamespace(chat_postMessage=AsyncMock()),
        config={
            "agent": {"feedback_enabled": True},
            "response_settings": {"max_response_length": 5000},
        },
        message_handler=SimpleNamespace(
            _format_agent_response=Mock(return_value=answer)
        ),
    )
    await SlackBot._monitor_agent_run(
        bot, {"run_id": "r", "channel_id": "c", "thread_ts": "t"}
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert all(len(text) <= 3000 for text in sections)
    assert "".join(sections) == answer
    assert len(blocks[-1]["elements"]) == 2
