"""A12b: consumers check event_version instead of trusting it."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiokafka.structs import TopicPartition

from flink import analytics_job
from flink import logic as flink_logic
from src.event_version import is_supported_event_version
from src.llm_router_part3_pipeline import KafkaConsumerManager
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache


@pytest.mark.parametrize(
    "value", [None, "1.0", "1", "1.7", 1, 1.0, " 1.0 "]  # legacy and major 1
)
def test_supported_versions(value):
    assert is_supported_event_version(value) is True


@pytest.mark.parametrize("value", ["2.0", 2, "0.9", "", "v1", "abc", True, [1]])
def test_unsupported_versions(value):
    assert is_supported_event_version(value) is False


class _Consumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


class _DlqProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, key=None, value=None):
        self.sent.append((topic, value))


def _message(topic, value, offset=3):
    return SimpleNamespace(value=value, topic=topic, partition=0, offset=offset)


def _sink():
    manager = KafkaConsumerManager(
        {
            "topics": {
                "requests_enriched": "requests.enriched",
                "inference_completed": "inference.completed",
            },
            "consumer": {"enable_auto_commit": False},
        },
        MagicMock(),
    )
    manager.dlq_producer = _DlqProducer()
    return manager


@pytest.mark.asyncio
async def test_observer_skips_unsupported_version_and_commits():
    manager = _sink()
    seen = []
    manager.register_observer("requests_enriched", seen.append)
    message = _message(
        "requests.enriched",
        {
            "event_type": "requests.enriched",
            "event_version": "2.0",
            "request_id": "r1",
            "query_text": "private question",
        },
    )
    consumer = _Consumer([message])
    manager.consumers["requests_enriched"] = consumer

    await manager._consume_requests_enriched(consumer)

    # Observer-only: skipped, not dead-lettered (D4 review); the sink loops
    # dead-letter (test_sink_batched_loop_dead_letters_unsupported_version).
    assert seen == []
    assert manager.dlq_producer.sent == []
    consumer.commit.assert_awaited_once_with(
        {TopicPartition("requests.enriched", 0): 4}
    )


@pytest.mark.asyncio
async def test_sink_batched_loop_dead_letters_unsupported_version():
    manager = _sink()
    manager.flush_topic = AsyncMock()
    message = _message(
        "inference.completed",
        {"event_type": "inference.completed", "event_version": 2},
    )
    consumer = _Consumer([message])
    manager.consumers["inference_completed"] = consumer

    await manager._consume_inference_completed(consumer)

    assert manager.batch_processors["inference_completed"] == []
    ((_topic, payload),) = manager.dlq_producer.sent
    assert payload["failure_reason"] == "unsupported_event_version"
    consumer.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_sink_accepts_legacy_and_current_versions():
    manager = _sink()
    seen = []
    manager.register_observer("requests_enriched", lambda event: seen.append(event))
    consumer = _Consumer(
        [
            _message("requests.enriched", {"request_id": "legacy"}, offset=1),
            _message(
                "requests.enriched",
                {"request_id": "current", "event_version": "1.0"},
                offset=2,
            ),
        ]
    )

    await manager._consume_requests_enriched(consumer)

    assert [event["request_id"] for event in seen] == ["legacy", "current"]
    assert manager.dlq_producer.sent == []


def _materializer():
    cache = RoutingPolicyCache(
        {"enabled": False, "consumer": {"enable_auto_commit": False}}
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    materializer.dlq_producer = _DlqProducer()
    return materializer


@pytest.mark.asyncio
async def test_materializer_dead_letters_unsupported_version_and_commits():
    materializer = _materializer()
    materialize = AsyncMock()
    materializer.policy_cache.materialize_routing_policy_state = materialize
    message = _message(
        "routing.policy_state",
        {"scope_type": "user", "scope_key": "u1", "event_version": "3.1"},
    )
    consumer = _Consumer([message])

    await materializer._consume_routing_policy_state(consumer)

    materialize.assert_not_awaited()
    ((topic, payload),) = materializer.dlq_producer.sent
    assert topic == "routing.policy_state.dlq"
    assert payload["failure_reason"] == "unsupported_event_version"
    consumer.commit.assert_awaited_once_with(
        {TopicPartition("routing.policy_state", 0): 4}
    )


@pytest.mark.asyncio
async def test_materializer_accepts_legacy_version():
    materializer = _materializer()
    materialize = AsyncMock()
    materializer.policy_cache.materialize_routing_policy_state = materialize
    consumer = _Consumer(
        [_message("routing.policy_state", {"scope_type": "user", "scope_key": "u1"})]
    )

    await materializer._consume_routing_policy_state(consumer)

    materialize.assert_awaited_once()
    assert materializer.dlq_producer.sent == []


class _Counter:
    def __init__(self):
        self.count = 0

    def inc(self, value=1):
        self.count += value


def test_flink_logic_flags_unsupported_versions():
    assert flink_logic.unsupported_event_version_drop({"event_version": "1.0"}) is None
    assert flink_logic.unsupported_event_version_drop({}) is None
    dropped = flink_logic.unsupported_event_version_drop(
        {"event_version": "2.0", "query_text": "secret"}
    )
    assert dropped == {"event_type": flink_logic.DROPPED_EVENT}


def test_analytics_mapper_drops_and_counts_unsupported_version():
    mapper = analytics_job.InferenceCompletedMapper()
    counter = _Counter()
    mapper.unsupported_version_counter = counter
    value = json.dumps(
        {
            "event_version": "2.0",
            "request_id": "r1",
            "selected_model": "m",
            "provider": "p",
            "status": "success",
            "latency_ms": 1,
        }
    )

    result = mapper.map(value)

    assert result["event_type"] == flink_logic.DROPPED_EVENT
    assert result["event_type"] not in (
        flink_logic.ALERT_EVENT,
        flink_logic.INFERENCE_COMPLETED_EVENT,
    )
    assert counter.count == 1
    assert mapper.dropped_unsupported_version == 1


def test_analytics_mapper_still_validates_current_version():
    mapper = analytics_job.InferenceCompletedMapper()
    value = json.dumps(
        {
            "event_version": "1.0",
            "request_id": "r1",
            "selected_model": "m",
            "provider": "p",
            "status": "success",
            "latency_ms": 1,
        }
    )

    assert mapper.map(value)["event_type"] == flink_logic.INFERENCE_COMPLETED_EVENT
    assert mapper.dropped_unsupported_version == 0


def test_enrichment_mapper_drops_and_counts_unsupported_version(monkeypatch):
    from tests.test_flink_state import _import_enrichment_job

    job = _import_enrichment_job(monkeypatch)
    mapper = job.QueryEnrichmentMapper()
    counter = _Counter()
    mapper.unsupported_version_counter = counter
    value = json.dumps(
        {
            "event_version": "9",
            "request_id": "r1",
            "query_text": "hello",
            "user_id": "u",
            "user_tier": "free",
        }
    )

    result = mapper.map(value)

    assert result == {"event_type": flink_logic.DROPPED_EVENT}
    assert counter.count == 1
    assert mapper.dropped_unsupported_version == 1
