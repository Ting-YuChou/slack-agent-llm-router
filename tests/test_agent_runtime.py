import json
import os

import httpx
import pytest

from src.agent_runtime import (
    AgentRuntimeBusy,
    AgentRuntimeClient,
    AgentRuntimeConflict,
    AgentRuntimeTimeout,
    AgentRuntimeUnauthorized,
    AgentRuntimeUnavailable,
)


def client(handler, monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_TOKEN", "runtime-token")
    return AgentRuntimeClient(
        {"base_url": "http://agent.test:3001", "token_env": "AGENT_RUNTIME_TOKEN"},
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_create_session_posts_thread_identity_with_auth_and_returns_202(
    monkeypatch,
):
    captured = {}

    def handler(request):
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    result = await runtime.create_session("T1", "C1", "1.0", "U1", "fix tests")
    await runtime.close()

    assert captured["auth"] == "Bearer runtime-token"
    assert captured["body"] == {
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "1.0",
        "user_id": "U1",
        "prompt": "fix tests",
    }
    assert result == {"session_id": "S1", "run_id": "R1", "status": "starting"}


@pytest.mark.asyncio
async def test_prompt_status_events_decision_cancel_and_close_contract(monkeypatch):
    calls = []

    def handler(request):
        calls.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content) if request.content else None,
            )
        )
        if request.url.path.endswith("/prompts"):
            return httpx.Response(
                202, json={"session_id": "S1", "run_id": "R2", "status": "starting"}
            )
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='id: 1\nevent: approval\ndata: {"type":"approval","approval_id":"A1"}\n\n',
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "run_id": "R2",
                    "session_id": "S1",
                    "status": "running",
                    "answer": "",
                    "events": [],
                },
            )
        return httpx.Response(202, json={"status": "accepted"})

    runtime = client(handler, monkeypatch)
    await runtime.prompt("S1", "continue", "U1")
    status = await runtime.get_run("R2")
    events = await runtime.get_events("R2")
    await runtime.decide("R2", "A1", "U1", "approve")
    await runtime.cancel("R2", "U1")
    await runtime.close_session("S1", "U1")
    await runtime.close()

    assert status["status"] == "running"
    assert events == [{"type": "approval", "approval_id": "A1", "event_id": 1}]
    assert [path for _method, path, _body in calls] == [
        "/v1/sessions/S1/prompts",
        "/v1/runs/R2",
        "/v1/runs/R2/events",
        "/v1/runs/R2/decisions",
        "/v1/runs/R2/cancel",
        "/v1/sessions/S1/close",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "unauthorized", AgentRuntimeUnauthorized),
        (409, "session_busy", AgentRuntimeConflict),
        (429, "runtime_busy", AgentRuntimeBusy),
        (504, "run_timeout", AgentRuntimeTimeout),
    ],
)
async def test_stable_error_mapping(monkeypatch, status, code, expected):
    runtime = client(
        lambda _request: httpx.Response(
            status, json={"error": {"code": code, "message": "safe"}}
        ),
        monkeypatch,
    )
    with pytest.raises(expected):
        await runtime.create_session("T", "C", "1", "U", "task")
    await runtime.close()


@pytest.mark.asyncio
async def test_transport_failure_and_health_are_sanitized(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("secret address", request=request)

    runtime = client(handler, monkeypatch)
    assert await runtime.health() is False
    with pytest.raises(AgentRuntimeUnavailable, match="unavailable"):
        await runtime.get_run("R1")
    await runtime.close()


@pytest.mark.asyncio
async def test_lookup_restores_thread_session_after_slack_restart(monkeypatch):
    def handler(request):
        assert request.url.params["team_id"] == "T1"
        return httpx.Response(
            200,
            json={
                "found": True,
                "session": {
                    "session_id": "S1",
                    "run_id": None,
                    "owner_user_id": "U1",
                    "branch": "pi-agent/b",
                    "active": False,
                },
            },
        )

    runtime = client(handler, monkeypatch)
    session = await runtime.lookup_session("T1", "C1", "1.0")
    await runtime.close()

    assert session["session_id"] == "S1"
