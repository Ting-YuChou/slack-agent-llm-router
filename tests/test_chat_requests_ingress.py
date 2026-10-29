"""X2: Slack and /route share one ingress; inference.completed lands in
chat_requests."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from slack.bot_real import SlackBot
from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    KafkaConsumerManager,
    build_chat_request_row,
    build_inference_completed_event,
    build_request_raw_event,
)
from src.utils.schema import QueryRequest

from test_main import (  # noqa: F401
    DummyAdmissionController,
    DummyEventProducer,
    _admission_platform,
    _write_config,
    patched_platform_deps,
)


def _ingress_platform(tmp_path):
    platform = _admission_platform(tmp_path)
    producer = DummyEventProducer({})
    admission = DummyAdmissionController()
    platform.services["event_producer"] = producer
    platform.services["admission"] = admission
    return platform, producer, admission


def _slack_bot(platform, ingress):
    services = {**platform.services, "chat_ingress": ingress}
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=platform.services["inference"],
        services=services,
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
        chat_update=AsyncMock(return_value={"ok": True}),
    )
    return bot


def _mention():
    return {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": "what is 2+2?",
        "ts": "100.0",
        "_team_id": "T1",
    }


# Shared ingress


@pytest.mark.asyncio
async def test_a_slack_query_goes_through_the_shared_ingress_once(
    tmp_path, patched_platform_deps
):
    platform, producer, admission = _ingress_platform(tmp_path)
    bot = _slack_bot(platform, main.ChatIngress(platform))

    await bot._process_message(_mention())

    (request,) = producer.request_events
    assert request.metadata["source"] == "slack"
    assert len(platform.services["inference"].processed_requests) == 1
    # Slack keeps its own per-user limiter: no route admission.
    assert admission.route_calls == []
    assert [event["metric_name"] for event in producer.metric_events] == [
        "requests_total",
        "request_duration_seconds",
    ]
    labels = producer.metric_events[0]["labels"]
    assert labels == {"endpoint": "slack", "method": "mention", "status": "200"}


@pytest.mark.asyncio
async def test_a_failed_slack_query_is_counted_with_the_route_error_mapping(
    tmp_path, patched_platform_deps
):
    platform, producer, _admission = _ingress_platform(tmp_path)
    inference = platform.services["inference"]
    inference.response = inference.response.model_copy(
        update={"error": "inference_failed", "error_code": "timeout"}
    )
    bot = _slack_bot(platform, main.ChatIngress(platform))

    await bot._process_message(_mention())

    names = [event["metric_name"] for event in producer.metric_events]
    assert names == ["requests_total", "request_duration_seconds", "errors_total"]
    assert producer.metric_events[0]["labels"]["status"] == "504"


def test_route_still_takes_route_admission(tmp_path, patched_platform_deps):
    platform, producer, admission = _ingress_platform(tmp_path)

    with TestClient(platform._create_fastapi_app()) as client:
        response = client.post("/route", json={"query": "hello", "user_id": "u1"})

    assert response.status_code == 200
    assert len(admission.route_calls) == 1
    assert len(producer.request_events) == 1


@pytest.mark.asyncio
async def test_platform_hands_the_ingress_to_the_slack_bot(
    tmp_path, patched_platform_deps
):
    config_path = _write_config(tmp_path, overrides={"slack": {"enabled": True}})
    platform = main.LLMRouterPlatform(config_path=str(config_path))

    await platform._initialize_services(include_api=True, include_background=True)

    ingress = platform.services["slack_bot"].services["chat_ingress"]
    assert isinstance(ingress, main.ChatIngress)


def test_slack_and_api_requests_raw_events_have_one_shape():
    api = QueryRequest(query="hi", user_id="u1", context="secret thread text")
    slack = QueryRequest(
        query="hi",
        user_id="U1",
        context="secret thread text",
        metadata={"source": "slack", "surface": "mention", "slack": {}},
    )

    api_event = build_request_raw_event(api)
    slack_event = build_request_raw_event(slack)

    assert set(api_event) == set(slack_event)
    assert "context" not in slack_event
    assert "secret thread text" not in str(slack_event)


# inference.completed new fields


def _completed(metadata=None, decision=None, **response_fields):
    from src.utils.schema import InferenceResponse

    request = QueryRequest(
        query="private question",
        user_id="U1",
        request_id="req-1",
        context="private context",
        metadata=metadata or {},
    )
    response = InferenceResponse(
        response_text="ok",
        model_name="gpt-5",
        provider="openai",
        token_count_input=10,
        token_count_output=20,
        total_tokens=30,
        latency_ms=1200,
        tokens_per_second=1.0,
        cost_usd=0.01,
        reasoning_tokens=7,
        **response_fields,
    )
    return build_inference_completed_event(request, response, decision)


def test_inference_completed_carries_the_latency_breakdown():
    event = _completed(decision=SimpleNamespace(routing_time_ms=12))

    assert event["routing_ms"] == 12
    assert event["queue_wait_ms"] is None
    assert event["ttft_ms"] is None


def test_inference_completed_hashes_the_slack_thread_key():
    event = _completed(
        metadata={
            "source": "slack",
            "surface": "thread",
            "slack": {"team_id": "T1", "channel_id": "C1", "thread_ts": "100.0"},
        }
    )

    expected = hashlib.sha256(b"slack:T1:C1:100.0").hexdigest()[:32]
    assert event["thread_key_hash"] == expected
    assert "100.0" not in event["thread_key_hash"]


def test_an_api_request_has_no_thread_key():
    assert _completed()["thread_key_hash"] is None


# chat_requests


def test_chat_request_row_keeps_the_fields_and_no_text():
    event = _completed(
        metadata={
            "source": "slack",
            "surface": "mention",
            "slack": {"team_id": "T1", "channel_id": "C1", "thread_ts": "100.0"},
        },
        decision=SimpleNamespace(routing_time_ms=12, query_type="general"),
        fallback_from_model="claude-sonnet-4-6",
        attempts=2,
        finish_reason="length",
        error_type=None,
    )

    row = build_chat_request_row(
        event,
        {
            "event_id": "inference.completed:0:5",
            "kafka_topic": "inference.completed",
            "kafka_partition": 0,
            "kafka_offset": 5,
        },
    )

    assert row["request_id"] == "req-1"
    assert row["model"] == "gpt-5"
    assert row["provider"] == "openai"
    assert row["tier"] == "free"
    assert row["status"] == "success"
    assert row["source"] == "slack"
    assert row["surface"] == "mention"
    assert row["routing_ms"] == 12
    assert row["queue_wait_ms"] is None
    assert row["ttft_ms"] is None
    assert row["fallback_from_model"] == "claude-sonnet-4-6"
    assert row["attempts"] == 2
    assert row["finish_reason"] == "length"
    assert row["error_type"] == ""
    assert row["reasoning_tokens"] == 7
    assert row["thread_key_hash"] == event["thread_key_hash"]
    assert row["version"] > 0
    flattened = str(row)
    assert "private question" not in flattened
    assert "private context" not in flattened
    assert "query_text" not in row


@pytest.mark.asyncio
async def test_chat_requests_table_is_created_after_the_existing_tables():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [call.args[1] for call in manager._run_blocking.await_args_list]
    (ddl,) = [sql for sql in statements if "llm_router.chat_requests (" in sql]
    assert statements.index(ddl) >= 6
    assert "ENGINE = ReplacingMergeTree(version)" in ddl
    assert "ORDER BY (toStartOfHour(ts), request_id)" in ddl
    for column in ("model", "provider", "tier", "status"):
        assert f"{column} LowCardinality(String)" in ddl
    for column in ("routing_ms", "queue_wait_ms", "ttft_ms"):
        assert f"{column} Nullable(UInt32)" in ddl
    assert "query_text" not in ddl
    assert "context" not in ddl


@pytest.mark.asyncio
async def test_inference_completed_is_batched_into_chat_requests():
    config = {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {"inference_completed": "inference.completed"},
        "consumer": {},
    }
    clickhouse = MagicMock()
    clickhouse.batch_insert_chat_requests = AsyncMock()
    consumer = KafkaConsumerManager(config, clickhouse)
    event = _completed()
    message = SimpleNamespace(
        topic="inference.completed", partition=0, offset=5, value=event
    )

    class _Consumer:
        def __aiter__(self):
            async def messages():
                yield message

            return messages()

    await consumer._run_consumer("inference_completed", _Consumer())
    await consumer.flush_pending_batches()

    (rows,) = clickhouse.batch_insert_chat_requests.await_args.args
    assert [row["request_id"] for row in rows] == ["req-1"]
    assert rows[0]["event_id"] == "inference.completed:0:5"
