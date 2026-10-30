"""Round-1 review fixes for X1: cache key, log context, body id, surfaces."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from slack.bot_real import SlackQueryTrace
from src.llm_router_part2_inference import ResponseCache
from src.llm_router_part3_pipeline import REQUEST_SURFACES
from src.utils.logger import (
    request_id_context,
    session_id_context,
    set_request_context,
    user_id_context,
)
from src.utils.schema import QueryRequest

from test_inference_resilience import _chain_engine, _request, _response, _router
from test_main import (  # noqa: F401
    DummyAdmissionController,
    _admission_platform,
    patched_platform_deps,
)


# The cache key ignores the Slack trace keys


def _slack_request(message_ts, event_id, surface):
    return QueryRequest(
        query="what is 2+2?",
        user_id="U1",
        metadata={
            "source": "slack",
            "surface": surface,
            "slack": {
                "team_id": "T1",
                "channel_id": "D1",
                "thread_ts": None,
                "message_ts": message_ts,
                "event_id": event_id,
            },
        },
    )


def test_the_slack_trace_keys_do_not_fragment_the_cache_key():
    cache = ResponseCache({"enabled": False})

    first = cache.generate_cache_key(_slack_request("1.0", "Ev1", "dm"), "gpt-5")
    second = cache.generate_cache_key(_slack_request("2.0", "Ev2", "mention"), "gpt-5")

    assert first == second


# The log context belongs to one request


@pytest.mark.asyncio
async def test_the_log_context_is_reset_after_the_query():
    engine = _chain_engine(_router(), None)
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=_response("gpt-5")
    )
    seen = []
    decision = engine.router.route_query.return_value

    async def route_query(_request):
        seen.append(
            (
                request_id_context.get(),
                user_id_context.get(),
                session_id_context.get(),
            )
        )
        return decision

    engine.router.route_query = AsyncMock(side_effect=route_query)

    await engine.process_query(_request(request_id="first", session_id="s1"))
    assert request_id_context.get() is None
    await engine.process_query(_request(request_id="second"))

    assert seen == [("first", "u1", "s1"), ("second", "u1", None)]
    assert request_id_context.get() is None
    assert session_id_context.get() is None


def test_set_request_context_clears_missing_values():
    set_request_context("r1", "u1", "s1")
    set_request_context("r2", None, None)

    try:
        assert request_id_context.get() == "r2"
        assert user_id_context.get() is None
        assert session_id_context.get() is None
    finally:
        set_request_context(None, None, None)


# The body request_id keeps its old contract


def test_a_body_request_id_with_any_charset_is_still_accepted():
    request = QueryRequest(query="hi", user_id="u1", request_id="job/123")

    assert request.request_id == "job/123"


def test_route_accepts_a_body_id_it_accepted_before(tmp_path, patched_platform_deps):
    platform = _admission_platform(tmp_path)
    platform.services["admission"] = DummyAdmissionController()
    inference = platform.services["inference"]

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post(
            "/route",
            json={"query": "hello", "user_id": "u1", "request_id": "job/123"},
        )

    assert response.status_code == 200
    assert inference.processed_requests[0].request_id == "job/123"


# Surfaces


@pytest.mark.parametrize(
    ("event", "surface"),
    [
        ({"type": "app_mention", "ts": "1.0"}, "mention"),
        ({"type": "app_mention", "ts": "2.0", "thread_ts": "1.0"}, "thread"),
        ({"type": "app_mention", "ts": "1.0", "thread_ts": "1.0"}, "mention"),
        ({"type": "message", "ts": "2.0", "thread_ts": "1.0"}, "thread"),
        ({"type": "message", "ts": "1.0", "channel_type": "im"}, "dm"),
        (
            {"type": "message", "ts": "2.0", "thread_ts": "1.0", "channel_type": "im"},
            "dm",
        ),
    ],
)
def test_the_surface_follows_the_event(event, surface):
    trace = SlackQueryTrace.for_message({"channel": "C1", **event}, team_id="T1")

    assert trace.surface == surface


def test_dm_is_a_known_surface_in_events_and_rows():
    assert "dm" in REQUEST_SURFACES
