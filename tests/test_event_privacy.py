import hashlib
import importlib
import json
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flink.logic import (
    build_alert_event,
    build_fast_lane_hint_event,
    build_request_enriched_event,
    build_routing_guardrail_event,
    validate_request_event,
)
from src.llm_router_part3_pipeline import (
    KafkaConsumerManager,
    KafkaProducerManager,
    QueryLogEntry,
    build_inference_completed_event,
    build_request_raw_event,
)
from src.routing_features import estimate_query_tokens
from src.utils.schema import InferenceResponse, QueryRequest, UserTier

SLACK_TOKEN = "xoxb-1234567890-0987654321-AbCdEfGhIjKl"
OPENAI_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
PRIVATE_KEY_BODY = "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6"
SECRET_QUERY = (
    f"Why does Bearer {JWT} fail? My key is {OPENAI_KEY} and the bot uses "
    f"{SLACK_TOKEN}.\n-----BEGIN PRIVATE KEY-----\n{PRIVATE_KEY_BODY}\n"
    "-----END PRIVATE KEY-----"
)
SECRETS = (JWT, OPENAI_KEY, SLACK_TOKEN, PRIVATE_KEY_BODY)
OTHER_MEMBERS_THREAD = "Alice: the merger closes Friday, keep it quiet. " * 120


@pytest.fixture
def pipeline_config():
    return {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {
            "queries": "test-queries",
            "requests_raw": "requests.raw",
            "inference_completed": "inference.completed",
            "alerts": "alerts",
            "routing_guardrails": "routing.guardrails",
        },
        "consumer": {"group_id": "test-group", "auto_offset_reset": "latest"},
    }


def _request(**overrides):
    values = {
        "query": SECRET_QUERY,
        "user_id": "u1",
        "user_tier": UserTier.FREE,
        "context": OTHER_MEMBERS_THREAD,
    }
    values.update(overrides)
    return QueryRequest(**values)


def _response(**overrides):
    values = {
        "response_text": "world",
        "model_name": "gpt-5",
        "provider": "openai",
        "token_count_input": 3,
        "token_count_output": 5,
        "total_tokens": 8,
        "latency_ms": 12,
        "tokens_per_second": 120.0,
        "cost_usd": 0.01,
    }
    values.update(overrides)
    return InferenceResponse(**values)


def _assert_no_secrets(serialized: str):
    for secret in SECRETS:
        assert secret not in serialized


def test_requests_raw_carries_context_size_and_hash_not_text():
    request = _request()

    event = build_request_raw_event(request)

    assert "context" not in event
    assert event["context_chars"] == len(OTHER_MEMBERS_THREAD)
    assert (
        event["context_sha256"]
        == hashlib.sha256(OTHER_MEMBERS_THREAD.encode("utf-8")).hexdigest()
    )
    assert event["query_token_estimate"] == estimate_query_tokens(
        request.query, OTHER_MEMBERS_THREAD
    )
    assert "merger" not in json.dumps(event, ensure_ascii=False)


def test_requests_raw_without_context_has_zero_size_and_no_hash():
    event = build_request_raw_event(_request(context=None, query="hello"))

    assert "context" not in event
    assert event["context_chars"] == 0
    assert event["context_sha256"] is None
    assert event["query_text"] == "hello"


def test_requests_raw_redacts_credentials_in_query_and_metadata():
    event = build_request_raw_event(
        _request(metadata={"api_key": "plain-secret", "note": f"use {SLACK_TOKEN}"})
    )

    serialized = json.dumps(event, ensure_ascii=False)
    _assert_no_secrets(serialized)
    assert "plain-secret" not in serialized
    assert event["query_text"].startswith("Why does Bearer [REDACTED] fail?")


def test_inference_completed_redacts_query_text():
    event = build_inference_completed_event(_request(), _response())

    _assert_no_secrets(json.dumps(event, ensure_ascii=False))
    assert "Why does Bearer [REDACTED] fail?" in event["query_text"]
    assert "context" not in event


def test_query_log_entries_redact_query_text():
    entry = QueryLogEntry(
        query_id="q1",
        timestamp=None,
        user_id="u1",
        user_tier="free",
        query_text=SECRET_QUERY,
        query_type="general",
        selected_model="gpt-5",
        token_count_input=1,
        token_count_output=1,
        latency_ms=1,
        cost_usd=0.0,
        status="success",
    )

    _assert_no_secrets(entry.to_dict()["query_text"])


@pytest.mark.asyncio
async def test_produced_query_log_is_redacted(pipeline_config):
    manager = KafkaProducerManager(pipeline_config)
    manager._accepting = True

    await manager.produce_query_log(_request(), _response(), SimpleNamespace())

    envelope = manager._delivery_queue.get_nowait()
    _assert_no_secrets(envelope.value["query_text"])


def test_cjk_events_serialize_as_utf8_not_unicode_escapes(pipeline_config):
    query = "請總結這段對話的重點並列出風險" * 700  # 10,500 CJK characters
    manager = KafkaProducerManager(pipeline_config)

    encoded = manager._serialize_message(
        build_request_raw_event(_request(query=query, context=None))
    )

    assert json.loads(encoded.decode("utf-8"))["query_text"] == query
    # UTF-8 is 3 bytes per CJK character; `\uXXXX` escapes would be 6.
    assert len(encoded) < 3 * len(query) + 2048


def test_serializer_still_handles_lone_surrogates(pipeline_config):
    manager = KafkaProducerManager(pipeline_config)

    encoded = manager._serialize_message({"query_text": "bad \ud800 text"})

    assert json.loads(encoded.decode("utf-8"))["query_text"] == "bad \ud800 text"


def test_dlq_serializer_writes_utf8(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())

    encoded = consumer._serialize_message({"payload": "中文內容"})

    assert "中文內容".encode("utf-8") in encoded


@pytest.mark.asyncio
async def test_dead_letter_payload_is_redacted_and_drops_context(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    consumer.dlq_producer = AsyncMock()
    poison = {
        "query_id": "q1",
        "query_text": SECRET_QUERY,
        "context": OTHER_MEMBERS_THREAD,
    }
    message = SimpleNamespace(
        value=json.dumps(poison).encode("utf-8"),
        key=b"u1",
        topic="test-queries",
        partition=0,
        offset=7,
    )

    await consumer._publish_dead_letter(
        "queries", message, ValueError(f"bad row {OPENAI_KEY}"), stage="consume"
    )

    payload = consumer.dlq_producer.send_and_wait.await_args.kwargs["value"]
    serialized = json.dumps(payload, ensure_ascii=False)
    _assert_no_secrets(serialized)
    assert "merger" not in serialized
    assert payload["payload"]["query_id"] == "q1"
    assert payload["payload"]["context_chars"] == len(OTHER_MEMBERS_THREAD)


def test_alert_payload_json_keeps_only_ids_and_error(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    alert = build_alert_event(
        {
            "request_id": "r1",
            "user_id": "u1",
            "query_text": SECRET_QUERY,
            "context": OTHER_MEMBERS_THREAD,
            "metadata": {"note": "private"},
        },
        alert_type="enrichment_error",
        description="Missing required request fields: user_tier",
        severity="error",
    )
    # An alert produced before this change still embeds the raw request.
    alert["original_payload"] = {"query_text": SECRET_QUERY}

    entry = consumer._build_alert_event_entry(alert)

    payload = json.loads(entry.payload_json)
    assert payload["request_id"] == "r1"
    assert payload["alert_type"] == "enrichment_error"
    assert payload["severity"] == "error"
    assert payload["description"] == "Missing required request fields: user_tier"
    for raw_field in ("original_payload", "query_text", "context", "metadata"):
        assert raw_field not in payload
    _assert_no_secrets(entry.payload_json)
    assert "Why does" not in entry.payload_json


def test_guardrail_payload_json_keeps_its_scope(pipeline_config):
    consumer = KafkaConsumerManager(pipeline_config, MagicMock())
    guardrail = build_routing_guardrail_event(
        {"model_name": "gpt-5", "provider": "openai", "error_rate": 0.5},
        scope_type="model",
        scope_key="gpt-5",
        trigger_type="high_error_rate",
        description="error rate 0.5",
    )

    payload = json.loads(consumer._build_alert_event_entry(guardrail).payload_json)

    assert payload["scope_type"] == "model"
    assert payload["scope_key"] == "gpt-5"
    assert payload["guardrail_action"] == "avoid"


def test_flink_alert_does_not_embed_the_raw_request():
    alert = build_alert_event(
        {"request_id": "r1", "query_text": SECRET_QUERY, "context": "private"},
        alert_type="enrichment_error",
        description="boom",
    )

    assert "original_payload" not in alert
    assert alert["request_id"] == "r1"
    serialized = json.dumps(alert)
    assert "Why does" not in serialized
    assert "private" not in serialized


def test_flink_alert_for_invalid_json_does_not_embed_the_message():
    alert = build_alert_event(
        '{"query_text": "secret question',
        alert_type="invalid_json",
        description="Could not decode requests.raw payload as JSON",
    )

    assert "secret question" not in json.dumps(alert)


def test_flink_accepts_requests_without_context():
    event = build_request_raw_event(_request(context=None, query="hello"))

    assert validate_request_event(event)["request_id"] == event["request_id"]


def test_flink_long_context_comes_from_context_chars():
    event = build_request_enriched_event(
        {
            "request_id": "r1",
            "query_text": "Summarize this thread",
            "user_tier": "free",
            "user_id": "u1",
            "context_chars": 5000,
        }
    )

    assert event["long_context"] is True


def test_flink_features_match_between_new_and_legacy_raw_events():
    request = _request(query="Summarize this thread for me", context="word " * 600)
    new_event = build_request_raw_event(request)
    legacy_event = dict(new_event, context=request.context)
    for key in ("context_chars", "context_sha256", "query_token_estimate"):
        legacy_event.pop(key)

    feature_keys = ("long_context", "query_complexity", "query_token_estimate")
    new_features = build_request_enriched_event(new_event)
    legacy_features = build_request_enriched_event(legacy_event)

    assert {key: new_features[key] for key in feature_keys} == {
        key: legacy_features[key] for key in feature_keys
    }


def test_enriched_events_and_hints_from_legacy_events_drop_context():
    legacy = {
        "request_id": "r1",
        "query_text": "Summarize this thread",
        "user_tier": "free",
        "user_id": "u1",
        "priority": 4,
        "context": OTHER_MEMBERS_THREAD,
    }

    enriched = build_request_enriched_event(legacy)
    hint = build_fast_lane_hint_event(legacy)

    assert "context" not in enriched
    assert enriched["context_chars"] == len(OTHER_MEMBERS_THREAD)
    assert enriched["long_context"] is True
    assert "context" not in hint
    assert "merger" not in json.dumps(enriched) + json.dumps(hint)


def _import_flink_job(monkeypatch):
    class _Function:
        pass

    stubs = {
        "pyflink": {},
        "pyflink.common": {},
        "pyflink.common.serialization": {"SimpleStringSchema": object},
        "pyflink.common.typeinfo": {"Types": SimpleNamespace()},
        "pyflink.datastream": {"StreamExecutionEnvironment": object},
        "pyflink.datastream.connectors": {},
        "pyflink.datastream.connectors.kafka": {
            "FlinkKafkaConsumer": object,
            "FlinkKafkaProducer": object,
        },
        "pyflink.datastream.functions": {
            "FilterFunction": _Function,
            "MapFunction": _Function,
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "flink.job", raising=False)
    return importlib.import_module("flink.job")


def test_flink_job_does_not_log_raw_messages(monkeypatch, caplog):
    job = _import_flink_job(monkeypatch)
    mapper = job.QueryEnrichmentMapper()

    with caplog.at_level(logging.ERROR, logger="flink.job"):
        invalid = mapper.map('{"query_text": "' + SECRET_QUERY)
        failed = mapper.map(json.dumps({"request_id": "r1", "query_text": "x"}))

    assert invalid["alert_type"] == "invalid_json"
    assert failed["alert_type"] == "enrichment_error"
    assert "Why does" not in caplog.text
    _assert_no_secrets(caplog.text)
    assert "r1" in caplog.text


def test_flink_sink_serializer_keeps_cjk_as_utf8():
    from flink.logic import serialize_event

    query = "請總結這段對話" * 100

    assert query in serialize_event({"query_text": query})
    assert json.loads(serialize_event({"query_text": "bad \ud800"})) == {
        "query_text": "bad \ud800"
    }
