import json

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
async def test_create_session_can_select_an_allowlisted_agent_model(monkeypatch):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    await runtime.create_session(
        "T1",
        "C1",
        "1.0",
        "U1",
        "fix tests",
        model="anthropic/claude-sonnet-4-6",
    )
    await runtime.close()

    assert captured["body"]["model"] == "anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_health_details_reports_provider_readiness(monkeypatch):
    payload = {
        "status": "healthy",
        "models": [
            {"ref": "openai/gpt-5.6-luna", "configured": True},
            {"ref": "anthropic/claude-sonnet-4-6", "configured": False},
        ],
    }
    runtime = client(lambda _request: httpx.Response(200, json=payload), monkeypatch)

    assert await runtime.health_details() == payload
    assert runtime.cached_configured_models() == ["openai/gpt-5.6-luna"]
    assert await runtime.health() is True
    await runtime.close()


@pytest.mark.asyncio
async def test_failed_health_refresh_keeps_last_configured_model_cache(monkeypatch):
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "models": [
                        {"ref": "openai/gpt-5.6-luna", "configured": True},
                        {"ref": "anthropic/claude-sonnet-4-6", "configured": False},
                    ],
                },
            ),
            httpx.Response(503),
        ]
    )
    runtime = client(lambda _request: next(responses), monkeypatch)

    await runtime.health_details()
    assert await runtime.health_details() is None
    assert runtime.cached_configured_models() == ["openai/gpt-5.6-luna"]
    await runtime.close()


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
    with pytest.raises(expected) as exc_info:
        await runtime.create_session("T", "C", "1", "U", "task")
    assert exc_info.value.code == code
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


@pytest.mark.asyncio
async def test_tree_stats_compact_and_fork_contracts_are_validated(monkeypatch):
    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, dict(request.url.params), body))
        if request.url.path.endswith("/tree"):
            return httpx.Response(
                200,
                json={
                    "session_id": "S1",
                    "active_leaf_id": "leaf-1",
                    "turns": [
                        {
                            "number": 1,
                            "run_id": "R1",
                            "status": "completed",
                            "task_preview": "fix",
                            "forkable": True,
                        }
                    ],
                    "lineage": {},
                    "truncated": False,
                },
            )
        if request.url.path.endswith("/stats"):
            return httpx.Response(
                200,
                json={
                    "session_id": "S1",
                    "total_messages": 2,
                    "auto_compaction_enabled": True,
                    "compaction_count": 0,
                },
            )
        if request.url.path.endswith("/compact"):
            return httpx.Response(
                202,
                json={"session_id": "S1", "run_id": "RC", "status": "starting"},
            )
        return httpx.Response(
            202,
            json={
                "session_id": "S2",
                "branch": "pi-agent/child",
                "baseline_commit": "abc",
                "parent_session_id": "S1",
                "fork_source_run_id": "R1",
                "model_ref": "openai/gpt-5.6-luna",
            },
        )

    runtime = client(handler, monkeypatch)
    tree = await runtime.get_tree("S1", "U1")
    stats = await runtime.get_stats("S1", "U1")
    compact = await runtime.compact("S1", "U1", "Keep decisions")
    fork = await runtime.fork_session(
        "S1",
        "U1",
        "R1",
        team_id="T1",
        channel_id="C1",
        thread_ts="2",
    )
    await runtime.close()

    assert tree["turns"][0]["run_id"] == "R1"
    assert stats["auto_compaction_enabled"] is True
    assert compact["run_id"] == "RC"
    assert fork["session_id"] == "S2"
    assert calls == [
        ("GET", "/v1/sessions/S1/tree", {"user_id": "U1"}, None),
        ("GET", "/v1/sessions/S1/stats", {"user_id": "U1"}, None),
        (
            "POST",
            "/v1/sessions/S1/compact",
            {},
            {"user_id": "U1", "custom_instructions": "Keep decisions"},
        ),
        (
            "POST",
            "/v1/sessions/S1/forks",
            {},
            {
                "user_id": "U1",
                "source_run_id": "R1",
                "team_id": "T1",
                "channel_id": "C1",
                "thread_ts": "2",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_create_session_sends_a_separate_safe_display_prompt(monkeypatch):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    await runtime.create_session(
        "T1",
        "C1",
        "1",
        "U1",
        "untrusted Slack context plus task",
        display_prompt="implement the todo",
    )
    await runtime.close()

    assert captured["prompt"] == "untrusted Slack context plus task"
    assert captured["display_prompt"] == "implement the todo"
