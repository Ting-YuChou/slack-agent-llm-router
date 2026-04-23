from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiokafka.structs import TopicPartition

from src.llm_router_part3_policy import PolicyMaterializer


class AsyncMessageConsumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consume_method", "cache_method", "topic_name", "payload"),
    [
        (
            "_consume_requests_enriched",
            "materialize_request_enriched",
            "requests.enriched",
            {"event_type": "requests.enriched", "request_id": "req-1"},
        ),
        (
            "_consume_fast_lane_hints",
            "materialize_fast_lane_hint",
            "fast_lane_hints",
            {"event_type": "fast_lane_hints", "request_id": "req-1"},
        ),
        (
            "_consume_routing_guardrails",
            "materialize_routing_guardrail",
            "routing.guardrails",
            {"event_type": "routing.guardrails", "scope_key": "gpt-5"},
        ),
        (
            "_consume_routing_policy_state",
            "materialize_routing_policy_state",
            "routing.policy_state",
            {"event_type": "routing.policy_state", "scope_key": "session-1"},
        ),
    ],
)
async def test_policy_materializer_commits_offsets_after_successful_materialization(
    consume_method, cache_method, topic_name, payload
):
    policy_cache = SimpleNamespace(
        config={"consumer": {"enable_auto_commit": False}},
        enabled=True,
        materialize_request_enriched=AsyncMock(),
        materialize_fast_lane_hint=AsyncMock(),
        materialize_routing_guardrail=AsyncMock(),
        materialize_routing_policy_state=AsyncMock(),
    )
    materializer = PolicyMaterializer(
        {"bootstrap_servers": ["localhost:9092"], "consumer": {"enable_auto_commit": False}},
        policy_cache,
    )
    materializer.running = True
    message = SimpleNamespace(
        value=payload,
        topic=topic_name,
        partition=0,
        offset=7,
    )
    consumer = AsyncMessageConsumer([message])

    await getattr(materializer, consume_method)(consumer)

    getattr(policy_cache, cache_method).assert_awaited_once_with(payload)
    consumer.commit.assert_awaited_once_with({TopicPartition(topic_name, 0): 8})


@pytest.mark.asyncio
async def test_policy_materializer_skips_invalid_json_with_manual_commit():
    policy_cache = SimpleNamespace(
        config={"consumer": {"enable_auto_commit": False}},
        enabled=True,
        materialize_request_enriched=AsyncMock(),
        materialize_fast_lane_hint=AsyncMock(),
        materialize_routing_guardrail=AsyncMock(),
        materialize_routing_policy_state=AsyncMock(),
    )
    materializer = PolicyMaterializer(
        {
            "bootstrap_servers": ["localhost:9092"],
            "consumer": {"enable_auto_commit": False},
        },
        policy_cache,
    )
    materializer.running = True
    message = SimpleNamespace(
        value=b"{not-json",
        topic="requests.enriched",
        partition=0,
        offset=3,
    )
    consumer = AsyncMessageConsumer([message])

    await materializer._consume_requests_enriched(consumer)

    policy_cache.materialize_request_enriched.assert_not_awaited()
    consumer.commit.assert_awaited_once_with({TopicPartition("requests.enriched", 0): 4})
