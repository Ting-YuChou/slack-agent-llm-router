import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from src.agent_runtime import (
    AgentRuntimeBusy,
    AgentRuntimeClient,
    AgentRuntimeConflict,
    AgentRuntimeError,
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


def test_live_acceptance_driver_requires_explicit_model_usage_opt_in():
    script = (
        Path(__file__).resolve().parents[1] / "scripts/validate_structured_workflow.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--live-openai is required" in result.stderr


@pytest.mark.asyncio
async def test_structured_mode_is_explicit_for_create_prompt_ci_and_resume(monkeypatch):
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            202,
            json={
                "session_id": "S",
                "run_id": "R",
                "status": "awaiting_approval"
                if request.url.path == "/v1/ci-repairs"
                else "starting",
            },
        )

    runtime = client(handler, monkeypatch)
    try:
        await runtime.create_session(
            "T", "C", "1", "U", "fix", workflow_mode="structured"
        )
        await runtime.prompt("S", "fix", "U", workflow_mode="structured")
        await runtime.repair_ci(
            "7", "U", team_id="T", channel_id="C", workflow_mode="structured"
        )
        await runtime.resume("R", "U")
    finally:
        await runtime.close()
    assert all(body["workflow_mode"] == "structured" for _, body in requests[:3])
    assert requests[-1] == ("/v1/runs/R/resume", {"user_id": "U"})


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
async def test_create_session_sends_routing_text_separately_from_full_prompt(
    monkeypatch,
):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    await runtime.create_session(
        "T1", "C1", "1.0", "U1", "context plus task", routing_text="fix the test"
    )
    await runtime.close()
    assert captured["body"]["prompt"] == "context plus task"
    assert captured["body"]["routing_text"] == "fix the test"


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


@pytest.mark.asyncio
async def test_ci_repair_accepts_initial_owner_approval_without_starting_a_writer(
    monkeypatch,
):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "run_id": "ci",
                "session_id": "ci",
                "status": "awaiting_approval",
                "approval_id": "approval",
            },
        )

    runtime = client(handler, monkeypatch)
    try:
        result = await runtime.repair_ci(
            "https://github.com/org/repo/pull/7",
            "owner",
            team_id="team",
            channel_id="channel",
        )
        assert result["status"] == "awaiting_approval"
        assert captured["path"] == "/v1/ci-repairs"
        assert captured["body"]["user_id"] == "owner"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_discussion_mode_is_per_request_for_create_and_followup(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            202, json={"run_id": "R", "session_id": "S", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    try:
        await runtime.create_session("T", "C", "1", "U", "explain", task_mode="discuss")
        await runtime.prompt("S", "explain more", "U", task_mode="discuss")
        await runtime.prompt("S", "fix", "U")
    finally:
        await runtime.close()
    assert bodies[0]["task_mode"] == "discuss"
    assert bodies[1]["task_mode"] == "discuss"
    assert "task_mode" not in bodies[2]


def _run_payload(status):
    return {"run_id": "R", "session_id": "S", "status": status}


def _fake_clock(runtime, monkeypatch):
    """Replace sleep and the monotonic clock so backoff runs instantly."""
    import src.agent_runtime as agent_runtime_module

    now = {"t": 0.0}
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(agent_runtime_module.asyncio, "sleep", fake_sleep)
    runtime._monotonic = lambda: now["t"]
    return sleeps


@pytest.mark.asyncio
async def test_wait_for_update_retries_transient_5xx_until_terminal(monkeypatch):
    responses = iter(
        [
            httpx.Response(503, json={"error": {"code": "unavailable"}}),
            httpx.Response(503, json={"error": {"code": "unavailable"}}),
            httpx.Response(200, json=_run_payload("completed")),
        ]
    )
    runtime = client(lambda request: next(responses), monkeypatch)
    sleeps = _fake_clock(runtime, monkeypatch)
    try:
        run = await runtime.wait_for_update("R")
    finally:
        await runtime.close()
    assert run["status"] == "completed"
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_wait_for_update_retries_request_timeouts(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_run_payload("awaiting_approval"))

    runtime = client(handler, monkeypatch)
    _fake_clock(runtime, monkeypatch)
    try:
        run = await runtime.wait_for_update("R")
    finally:
        await runtime.close()
    assert run["status"] == "awaiting_approval"
    assert calls["n"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 404])
async def test_wait_for_update_does_not_retry_auth_or_missing_run(monkeypatch, status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"code": "not_found"}})

    runtime = client(handler, monkeypatch)
    sleeps = _fake_clock(runtime, monkeypatch)
    try:
        with pytest.raises(AgentRuntimeError) as raised:
            await runtime.wait_for_update("R")
    finally:
        await runtime.close()
    assert calls["n"] == 1
    assert sleeps == []
    assert raised.value.status == status


@pytest.mark.asyncio
async def test_wait_for_update_backoff_is_capped_and_gives_up_after_deadline(
    monkeypatch,
):
    runtime = client(
        lambda request: httpx.Response(
            502, json={"error": {"code": "bad_gateway", "message": "down"}}
        ),
        monkeypatch,
    )
    sleeps = _fake_clock(runtime, monkeypatch)
    try:
        with pytest.raises(AgentRuntimeError) as raised:
            await runtime.wait_for_update("R", max_error_seconds=120)
    finally:
        await runtime.close()
    assert sleeps[:6] == [1, 2, 4, 8, 16, 30]
    assert max(sleeps) == 30
    assert sum(sleeps) <= 120
    assert raised.value.status == 502
    assert raised.value.code == "bad_gateway"


@pytest.mark.asyncio
async def test_wait_for_update_resets_backoff_after_a_successful_poll(monkeypatch):
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json=_run_payload("running")),
            httpx.Response(503),
            httpx.Response(200, json=_run_payload("completed")),
        ]
    )
    runtime = client(lambda request: next(responses), monkeypatch)
    sleeps = _fake_clock(runtime, monkeypatch)
    try:
        await runtime.wait_for_update("R", poll_seconds=0.5)
    finally:
        await runtime.close()
    assert sleeps == [1, 2, 0.5, 1]


@pytest.mark.asyncio
async def test_mutating_calls_send_idempotency_keys(monkeypatch):
    keys = []

    def handler(request):
        keys.append((request.url.path, request.headers.get("idempotency-key")))
        if request.url.path.endswith("/decisions"):
            return httpx.Response(202, json={"status": "accepted"})
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    await runtime.create_session("T", "C", "1", "U", "task")
    await runtime.create_session("T", "C", "1", "U", "task")
    await runtime.create_session("T", "C", "1", "U", "task", idempotency_key="k-1")
    await runtime.prompt("S1", "continue", "U")
    await runtime.decide("R1", "A1", "U", "approve")
    await runtime.decide("R1", "A1", "U", "approve")
    await runtime.close()

    assert all(key for _path, key in keys)
    create_keys = [key for path, key in keys if path == "/v1/sessions"]
    assert create_keys[0] != create_keys[1]
    assert create_keys[2] == "k-1"
    decide_keys = [key for path, key in keys if path.endswith("/decisions")]
    assert decide_keys[0] == decide_keys[1]


@pytest.mark.asyncio
async def test_control_calls_use_short_timeouts_but_create_keeps_long_one(
    monkeypatch,
):
    timeouts = {}

    def handler(request):
        timeouts[(request.method, request.url.path)] = request.extensions["timeout"][
            "read"
        ]
        if request.url.path == "/v1/sessions/lookup":
            return httpx.Response(200, json={"found": False})
        if request.url.path == "/v1/sessions":
            return httpx.Response(
                202, json={"session_id": "S1", "run_id": "R1", "status": "starting"}
            )
        if request.method == "GET":
            return httpx.Response(
                200, json={"run_id": "R1", "session_id": "S1", "status": "running"}
            )
        return httpx.Response(202, json={"status": "accepted"})

    runtime = client(handler, monkeypatch)
    await runtime.create_session("T", "C", "1", "U", "task")
    await runtime.get_run("R1")
    await runtime.lookup_session("T", "C", "1")
    await runtime.decide("R1", "A1", "U", "approve")
    await runtime.cancel("R1", "U")
    await runtime.close()

    assert timeouts[("POST", "/v1/sessions")] == 910
    for key in (
        ("GET", "/v1/runs/R1"),
        ("GET", "/v1/sessions/lookup"),
        ("POST", "/v1/runs/R1/decisions"),
        ("POST", "/v1/runs/R1/cancel"),
    ):
        assert timeouts[key] == 10


def _sse(*events):
    return "".join(
        f"id: {event_id}\nevent: {payload['type']}\ndata: {json.dumps(payload)}\n\n"
        for event_id, payload in events
    )


@pytest.mark.asyncio
async def test_wait_for_update_follows_the_event_cursor_between_run_reads(
    monkeypatch,
):
    started = [
        {"type": "status", "status": "starting"},
        {"type": "status", "status": "running"},
    ]
    runs = iter(
        [
            {
                **_run_payload("running"),
                "phase": "implementing",
                "tool_count": 1,
                "events": started,
            },
            {**_run_payload("completed"), "phase": "completed", "events": []},
        ]
    )
    pages = iter(
        [
            httpx.Response(
                200,
                text=_sse((3, {"type": "tool", "phase": "start", "tool": "read"})),
            ),
            httpx.Response(503),
            httpx.Response(
                200,
                text=_sse(
                    (4, {"type": "tool", "phase": "end", "tool": "read"}),
                    (5, {"type": "tool", "phase": "start", "tool": "edit"}),
                ),
            ),
            httpx.Response(
                200, text=_sse((6, {"type": "status", "status": "completed"}))
            ),
        ]
    )
    requests = []

    def handler(request):
        if request.url.path.endswith("/events"):
            requests.append(("events", request.url.params["after"]))
            return next(pages)
        requests.append(("run", None))
        return httpx.Response(200, json=next(runs))

    progress = []

    async def on_progress(run):
        progress.append((run["status"], run["tool_count"], run.get("last_tool")))

    runtime = client(handler, monkeypatch)
    sleeps = _fake_clock(runtime, monkeypatch)
    try:
        run = await runtime.wait_for_update(
            "R", poll_seconds=0.5, on_progress=on_progress
        )
    finally:
        await runtime.close()

    assert run["status"] == "completed"
    # The full run is read only at start and after a status event.
    assert requests == [
        ("run", None),
        ("events", "2"),
        ("events", "3"),
        ("events", "3"),
        ("events", "5"),
        ("run", None),
    ]
    assert progress == [
        ("running", 1, None),
        ("running", 2, "read"),
        ("running", 3, "edit"),
    ]
    assert sleeps == [0.5, 0.5, 1, 0.5]


@pytest.mark.asyncio
async def test_wait_for_update_keeps_an_absolute_cursor_across_the_event_cap(
    monkeypatch,
):
    # The runtime keeps the newest events of a run and counts the dropped ones.
    runs = iter(
        [
            {
                **_run_payload("running"),
                "events": [{"type": "status", "status": "running"}] * 3,
                "events_truncated": 10,
            },
            {**_run_payload("completed"), "events": []},
        ]
    )
    pages = iter(
        [
            # Events 14-19 were dropped by the cap before this poll read them.
            httpx.Response(
                200,
                text=_sse((20, {"type": "tool", "phase": "start", "tool": "read"})),
            ),
            httpx.Response(200, text=""),
            httpx.Response(
                200, text=_sse((21, {"type": "status", "status": "completed"}))
            ),
        ]
    )
    requests = []

    def handler(request):
        if request.url.path.endswith("/events"):
            requests.append(request.url.params["after"])
            return next(pages)
        return httpx.Response(200, json=next(runs))

    async def on_progress(run):
        return None

    runtime = client(handler, monkeypatch)
    _fake_clock(runtime, monkeypatch)
    try:
        run = await runtime.wait_for_update(
            "R", poll_seconds=0.5, on_progress=on_progress
        )
    finally:
        await runtime.close()

    assert run["status"] == "completed"
    assert requests == ["13", "20", "20"]


@pytest.mark.asyncio
async def test_get_run_patch_reads_owner_patch_bytes_and_truncation(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.params.get("user_id") == "other":
            return httpx.Response(
                403, json={"error": {"code": "patch_forbidden", "message": "no"}}
            )
        if request.url.path == "/v1/runs/none/patch":
            return httpx.Response(
                404, json={"error": {"code": "patch_not_found", "message": "no"}}
            )
        return httpx.Response(
            200,
            content=b"diff --git a/a b/a\n\xff\n",
            headers={
                "content-type": "text/x-diff; charset=utf-8",
                "x-patch-truncated": "true",
                "x-patch-bytes": "5002",
            },
        )

    runtime = client(handler, monkeypatch)
    try:
        patch = await runtime.get_run_patch("run/1", "U1")
        with pytest.raises(AgentRuntimeError) as forbidden:
            await runtime.get_run_patch("r", "other")
        with pytest.raises(AgentRuntimeError) as missing:
            await runtime.get_run_patch("none", "U1")
    finally:
        await runtime.close()
    assert patch == {
        "patch": b"diff --git a/a b/a\n\xff\n",
        "truncated": True,
        "bytes": 5002,
    }
    assert requests[0].method == "GET"
    assert requests[0].url.raw_path == b"/v1/runs/run%2F1/patch?user_id=U1"
    assert requests[0].headers["authorization"] == "Bearer runtime-token"
    assert (forbidden.value.status, forbidden.value.code) == (403, "patch_forbidden")
    assert (missing.value.status, missing.value.code) == (404, "patch_not_found")


@pytest.mark.asyncio
async def test_resume_from_patch_posts_owner_with_idempotency_key(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if json.loads(request.content)["user_id"] == "busy":
            return httpx.Response(
                409, json={"error": {"code": "patch_conflict", "message": "no"}}
            )
        return httpx.Response(
            202, json={"session_id": "S1", "run_id": "R2", "status": "starting"}
        )

    runtime = client(handler, monkeypatch)
    try:
        accepted = await runtime.resume_from_patch("R1", "U1", "slack-key-1")
        with pytest.raises(AgentRuntimeConflict) as conflict:
            await runtime.resume_from_patch("R1", "busy", "slack-key-2")
    finally:
        await runtime.close()
    assert accepted == {"session_id": "S1", "run_id": "R2", "status": "starting"}
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/runs/R1/resume-from-patch"
    assert json.loads(requests[0].content) == {"user_id": "U1"}
    assert requests[0].headers["idempotency-key"] == "slack-key-1"
    assert conflict.value.code == "patch_conflict"
