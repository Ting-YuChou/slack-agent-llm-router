"""X1: one request id from the edge (/route or Slack) to logs and events."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from slack.bot_real import SlackBot
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.utils.logger import request_id_context
from src.utils.schema import InferenceResponse, QueryRequest

from test_inference_resilience import _ENTERPRISE, _chain_engine, _request, _response
from test_inference_resilience import _router
from test_main import (  # noqa: F401
    DummyEventProducer,
    DummyInferenceEngine,
    DummyRouter,
    _write_config,
    patched_platform_deps,
)


def _route_platform(tmp_path, producer=None):
    platform = main.LLMRouterPlatform(config_path=str(_write_config(tmp_path)))
    inference = DummyInferenceEngine({}, DummyRouter({}), event_producer=producer)
    platform.services["inference"] = inference
    if producer is not None:
        platform.services["event_producer"] = producer
    return platform, inference


# /route


def test_route_echoes_a_valid_x_request_id(tmp_path, patched_platform_deps):
    platform, inference = _route_platform(tmp_path)

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post(
            "/route",
            json={"query": "hello", "user_id": "u1"},
            headers={"X-Request-ID": "abc"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "abc"
    assert response.json()["request_id"] == "abc"
    assert inference.processed_requests[0].request_id == "abc"


def test_route_without_header_returns_the_generated_id(tmp_path, patched_platform_deps):
    platform, inference = _route_platform(tmp_path)

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post("/route", json={"query": "hello", "user_id": "u1"})

    request_id = inference.processed_requests[0].request_id
    assert request_id
    assert response.headers["x-request-id"] == request_id
    assert response.json()["request_id"] == request_id


@pytest.mark.parametrize("bad_id", ["has space", "semi;colon", "x" * 129, "a/b"])
def test_route_rejects_an_invalid_x_request_id(tmp_path, patched_platform_deps, bad_id):
    platform, inference = _route_platform(tmp_path)

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post(
            "/route",
            json={"query": "hello", "user_id": "u1"},
            headers={"X-Request-ID": bad_id},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request_id"
    assert "x-request-id" not in response.headers
    assert inference.processed_requests == []


@pytest.mark.parametrize("bad_id", ["has space", "a/b", "line\nbreak", ""])
def test_query_request_id_charset_is_checked(bad_id):
    with pytest.raises(ValidationError):
        QueryRequest(query="hi", user_id="u1", request_id=bad_id)

    QueryRequest(query="hi", user_id="u1", request_id="slash:1.2.abc-DEF_9")


def test_route_rejection_still_carries_the_request_id_header(
    tmp_path, patched_platform_deps
):
    platform, inference = _route_platform(tmp_path)
    inference.response = inference.response.model_copy(update={"error": "upstream"})

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post(
            "/route",
            json={"query": "hello", "user_id": "u1"},
            headers={"X-Request-ID": "err-1"},
        )

    assert response.status_code == 502
    assert response.headers["x-request-id"] == "err-1"


# Inference


@pytest.mark.asyncio
async def test_process_query_returns_and_publishes_the_request_id():
    engine = _chain_engine(_router(), None)
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=_response("gpt-5")
    )
    request = _request(request_id="abc", **_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.request_id == "abc"
    (
        _req,
        published,
        decision,
    ) = engine.event_producer.produce_inference_completed.await_args.args
    event = build_inference_completed_event(request, published, decision)
    assert event["request_id"] == "abc"


@pytest.mark.asyncio
async def test_a_cache_hit_answers_with_this_requests_id():
    engine = _chain_engine(_router(), None)
    cached = _response("gpt-5").model_dump(mode="json")
    cached["request_id"] = "the-original-request"
    engine.cache.get_cached_response = AsyncMock(return_value=cached)

    response = await engine.process_query(_request(request_id="second-request"))

    assert response.cached is True
    assert response.request_id == "second-request"


@pytest.mark.asyncio
async def test_the_cached_payload_leaves_the_request_id_out():
    engine = _chain_engine(_router(), None)
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=_response("gpt-5")
    )
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()

    await engine.process_query(_request(request_id="first-request"))

    engine.cache.cache_response.assert_awaited_once()
    payload = engine.cache.cache_response.await_args.args[1]
    assert "request_id" not in payload


@pytest.mark.asyncio
async def test_logs_carry_the_request_id_while_the_query_runs():
    engine = _chain_engine(_router(), None)
    seen = []
    decision = engine.router.route_query.return_value

    async def route_query(_request):
        seen.append(request_id_context.get())
        return decision

    engine.router.route_query = AsyncMock(side_effect=route_query)
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=_response("gpt-5")
    )

    await engine.process_query(_request(request_id="log-me"))

    assert seen == ["log-me"]


# inference.completed source and surface


def _completed_event(metadata):
    request = QueryRequest(query="hi", user_id="u1", metadata=metadata)
    response = InferenceResponse(
        response_text="ok",
        model_name="gpt-5",
        provider="openai",
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
    )
    return build_inference_completed_event(request, response)


def test_inference_completed_names_an_api_request_as_api():
    event = _completed_event({})

    assert event["source"] == "api"
    assert event["surface"] == "api"


@pytest.mark.parametrize("surface", ["mention", "thread", "slash"])
def test_inference_completed_names_the_slack_surface(surface):
    event = _completed_event({"source": "slack", "surface": surface})

    assert event["source"] == "slack"
    assert event["surface"] == surface


# Slack


def _slack_bot(inference_response_factory):
    seen = []

    async def process_query(request):
        seen.append(request)
        return inference_response_factory(response_text="four")

    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=SimpleNamespace(
            process_query=AsyncMock(side_effect=process_query)
        ),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
        chat_update=AsyncMock(return_value={"ok": True}),
    )
    return bot, seen


def _expected_slack_id(team, channel, ts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slack:{team}:{channel}:{ts}"))


def _mention(ts="100.0", **fields):
    return {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": "what is 2+2?",
        "ts": ts,
        "_team_id": "T1",
        **fields,
    }


@pytest.mark.asyncio
async def test_a_redelivered_slack_event_keeps_its_request_id(
    inference_response_factory,
):
    bot, seen = _slack_bot(inference_response_factory)

    await bot._process_message(_mention(_event_id="Ev1"))
    await bot._process_message(_mention(_event_id="Ev2"))

    first, second = seen
    assert first.request_id == second.request_id
    assert first.request_id == _expected_slack_id("T1", "C1", "100.0")


@pytest.mark.asyncio
async def test_a_mention_carries_its_slack_trace_metadata(
    inference_response_factory,
):
    bot, seen = _slack_bot(inference_response_factory)

    await bot._process_message(_mention(_event_id="Ev1"))

    (request,) = seen
    assert request.metadata["source"] == "slack"
    assert request.metadata["surface"] == "mention"
    assert request.metadata["slack"]["message_ts"] == "100.0"
    assert request.metadata["slack"]["event_id"] == "Ev1"


@pytest.mark.asyncio
async def test_a_thread_follow_up_is_the_thread_surface(inference_response_factory):
    bot, seen = _slack_bot(inference_response_factory)

    await bot._process_message(
        {
            "type": "message",
            "channel": "C1",
            "user": "U1",
            "text": "and 3+3?",
            "ts": "101.0",
            "thread_ts": "100.0",
            "_team_id": "T1",
        }
    )

    (request,) = seen
    assert request.metadata["surface"] == "thread"
    assert request.request_id == _expected_slack_id("T1", "C1", "101.0")


@pytest.mark.asyncio
async def test_a_slash_query_derives_its_id_from_the_trigger(
    inference_response_factory,
):
    bot, seen = _slack_bot(inference_response_factory)

    await bot._handle_slash_command(
        {
            "channel_id": "C1",
            "user_id": "U1",
            "team_id": "T1",
            "text": "what is 2+2?",
            "trigger_id": "13345.7384.abc",
        }
    )

    (request,) = seen
    assert request.request_id == "slash:13345.7384.abc"
    assert request.metadata["surface"] == "slash"


@pytest.mark.asyncio
async def test_the_socket_handler_keeps_the_envelope_event_id():
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    queued = []
    bot._enqueue_work = AsyncMock(side_effect=queued.append)
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    request = SimpleNamespace(
        envelope_id="E1",
        type="events_api",
        payload={
            "team_id": "T1",
            "event_id": "Ev123",
            "event": {"type": "app_mention", "channel": "C1", "ts": "1.0"},
        },
    )

    await bot._handle_socket_mode_request(socket_client, request)

    (item,) = queued
    assert item.payload["_event_id"] == "Ev123"
    assert item.payload["_team_id"] == "T1"
