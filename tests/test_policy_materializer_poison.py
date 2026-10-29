"""A malformed policy event is skipped or dead-lettered, never retried (A7)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiokafka.structs import TopicPartition

from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import (
    PolicyCacheUnavailableError,
    PolicyMaterializer,
    RoutingPolicyCache,
)
from src.utils.metrics import metric_value_for_labels


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
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    async def send_and_wait(self, topic, key=None, value=None):
        if self.fail:
            raise RuntimeError("dlq down")
        self.sent.append((topic, key, value))


def _materializer(cache=None):
    cache = cache or RoutingPolicyCache(
        {"enabled": False, "consumer": {"enable_auto_commit": False}}
    )
    materializer = PolicyMaterializer(
        {"bootstrap_servers": ["localhost:9092"], "topics": {}}, cache
    )
    materializer.running = True
    return materializer


def _message(topic, value, offset=7):
    return SimpleNamespace(value=value, topic=topic, partition=0, offset=offset)


def _invalid_count(topic_key, destination):
    return metric_value_for_labels(
        policy.POLICY_MATERIALIZER_INVALID_MESSAGES,
        {"topic": topic_key, "destination": destination},
    )


POISON = [
    pytest.param(
        "_consume_routing_policy_state",
        "routing_policy_state",
        "routing.policy_state",
        {"scope_type": "user", "scope_key": "u1", "recent_request_count": [3]},
        id="policy-state-list-count",
    ),
    pytest.param(
        "_consume_routing_policy_state",
        "routing_policy_state",
        "routing.policy_state",
        {"scope_type": "user", "scope_key": "u1", "query_type_breakdown": [1]},
        id="policy-state-list-breakdown",
    ),
    pytest.param(
        "_consume_routing_policy_state",
        "routing_policy_state",
        "routing.policy_state",
        b"[3]",
        id="policy-state-non-object-json",
    ),
    pytest.param(
        "_consume_requests_enriched",
        "requests_enriched",
        "requests.enriched",
        {"request_id": "r1", "preferred_models": 7},
        id="enriched-scalar-list-field",
    ),
    pytest.param(
        "_consume_requests_enriched",
        "requests_enriched",
        "requests.enriched",
        "[1, 2]",
        id="enriched-non-object-json",
    ),
    pytest.param(
        "_consume_fast_lane_hints",
        "fast_lane_hints",
        "fast_lane_hints",
        {"request_id": "r1", "avoid_models": 5},
        id="fast-lane-scalar-list-field",
    ),
    pytest.param(
        "_consume_routing_guardrails",
        "routing_guardrails",
        "routing.guardrails",
        b'"just a string"',
        id="guardrail-non-object-json",
    ),
    pytest.param(
        "_consume_routing_guardrails",
        "routing_guardrails",
        "routing.guardrails",
        {"scope_type": "model", "scope_key": ["gpt-5"]},
        id="guardrail-list-scope-key",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("consume", "topic_key", "topic", "value"), POISON)
async def test_poison_message_is_committed_and_counted(
    consume, topic_key, topic, value
):
    materializer = _materializer()
    consumer = _Consumer([_message(topic, value)])
    before = _invalid_count(topic_key, "skipped")

    await getattr(materializer, consume)(consumer)

    consumer.commit.assert_awaited_once_with({TopicPartition(topic, 0): 8})
    assert _invalid_count(topic_key, "skipped") == before + 1


@pytest.mark.asyncio
async def test_poison_message_goes_to_the_dlq_when_one_is_configured():
    materializer = _materializer()
    materializer.dlq_producer = _DlqProducer()
    value = {"scope_type": "user", "scope_key": "u1", "recent_request_count": [3]}
    consumer = _Consumer([_message("routing.policy_state", value, offset=41)])
    before = _invalid_count("routing_policy_state", "dlq")

    await materializer._consume_routing_policy_state(consumer)

    ((dlq_topic, _key, payload),) = materializer.dlq_producer.sent
    assert dlq_topic == "routing.policy_state.dlq"
    assert payload["source_topic"] == "routing.policy_state"
    assert payload["source_offset"] == 41
    assert payload["error_type"] == "PolicyEventInvalid"
    assert "recent_request_count" in payload["error_message"]
    # Field names only: no values reach the DLQ.
    assert payload["payload_keys"] == sorted(value)
    assert "u1" not in json.dumps(payload)
    consumer.commit.assert_awaited_once_with(
        {TopicPartition("routing.policy_state", 0): 42}
    )
    assert _invalid_count("routing_policy_state", "dlq") == before + 1


@pytest.mark.asyncio
async def test_a_failed_dlq_send_still_moves_past_the_poison_message():
    materializer = _materializer()
    materializer.dlq_producer = _DlqProducer(fail=True)
    consumer = _Consumer([_message("routing.guardrails", b"[1]")])
    before = _invalid_count("routing_guardrails", "dlq_failed")

    await materializer._consume_routing_guardrails(consumer)

    consumer.commit.assert_awaited_once()
    assert _invalid_count("routing_guardrails", "dlq_failed") == before + 1


@pytest.mark.asyncio
async def test_an_unavailable_cache_still_holds_the_offset():
    async def unavailable(_event):
        # The materializer is stopped while it waits for Redis.
        materializer.running = False
        raise PolicyCacheUnavailableError("cooldown")

    cache = SimpleNamespace(
        config={"consumer": {"enable_auto_commit": False}},
        enabled=True,
        materialize_routing_guardrail=unavailable,
    )
    materializer = _materializer(cache)
    consumer = _Consumer(
        [_message("routing.guardrails", {"scope_type": "model", "scope_key": "m"})]
    )

    await materializer._consume_routing_guardrails(consumer)

    consumer.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_creates_the_dlq_producer_when_enabled(monkeypatch):
    created = []

    class _Producer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        async def start(self):
            return None

        async def stop(self):
            return None

    class _KafkaConsumer:
        def __init__(self, topic, **kwargs):
            self.topic = topic

        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(policy, "AIOKafkaProducer", _Producer)
    monkeypatch.setattr(policy, "AIOKafkaConsumer", _KafkaConsumer)
    cache = RoutingPolicyCache({"enabled": True})
    cache.initialize = AsyncMock()
    materializer = PolicyMaterializer(
        {
            "bootstrap_servers": ["kafka:9092"],
            "topics": {"routing_guardrails": "routing.guardrails"},
            "dlq": {"enabled": True, "topic_suffix": ".dlq"},
        },
        cache,
    )

    await materializer.initialize()
    await materializer.shutdown()

    assert len(created) == 1
    assert created[0].kwargs["bootstrap_servers"] == ["kafka:9092"]
    assert materializer.dlq_producer is None


def test_a_materializer_without_a_dlq_has_none():
    assert _materializer().dlq_producer is None
