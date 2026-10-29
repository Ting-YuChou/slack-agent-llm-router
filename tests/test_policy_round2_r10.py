"""Round-2 review: group membership, read-timeout escalation, Redis errors (R10)."""

import json
from types import SimpleNamespace

import pytest
from aiokafka.structs import TopicPartition
from redis.exceptions import ResponseError

from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import (
    PolicyCacheUnavailableError,
    PolicyMaterializer,
    RoutingPolicyCache,
)
from src.utils.metrics import metric_value_for_labels

TP = TopicPartition("routing.guardrails", 0)


class _GroupConsumer:
    """Records the calls that keep a consumer in its group while it waits."""

    def __init__(self, messages, buffered=None, commit_error=None):
        self._messages = list(messages)
        self.calls = []
        self.commits = []
        self.buffered = dict(buffered or {})
        self.commit_error = commit_error

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()

    def assignment(self):
        return {TP}

    def pause(self, *partitions):
        self.calls.append(("pause", set(partitions)))

    def resume(self, *partitions):
        self.calls.append(("resume", set(partitions)))

    async def getmany(self, *partitions, timeout_ms=0, max_records=None):
        self.calls.append(("getmany", timeout_ms))
        records, self.buffered = self.buffered, {}
        return records

    def seek(self, partition, offset):
        self.calls.append(("seek", partition, offset))

    async def commit(self, offsets=None):
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(offsets)


def _message(offset=3):
    return SimpleNamespace(
        value={"scope_type": "model", "scope_key": "m"},
        topic=TP.topic,
        partition=TP.partition,
        offset=offset,
    )


def _materializer(materialize):
    cache = SimpleNamespace(
        config={
            "consumer": {
                "enable_auto_commit": False,
                "redis_retry_initial_backoff_seconds": 0.001,
            }
        },
        enabled=True,
        materialize_routing_guardrail=materialize,
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    return materializer


def _failing_then_ok(failures):
    calls = []

    async def materialize(_event):
        calls.append(1)
        if len(calls) <= failures:
            raise PolicyCacheUnavailableError("cooldown")

    return materialize, calls


# 1. The consumer keeps polling, paused, while it waits on Redis.


@pytest.mark.asyncio
async def test_waiting_on_redis_pauses_and_keeps_polling():
    materialize, calls = _failing_then_ok(2)
    materializer = _materializer(materialize)
    consumer = _GroupConsumer([_message()])

    await materializer._consume_routing_guardrails(consumer)

    kinds = [call[0] for call in consumer.calls]
    assert kinds[0] == "pause"
    assert kinds.count("getmany") == 2
    assert kinds[-1] == "resume"
    assert consumer.calls[0][1] == {TP}
    assert consumer.calls[-1][1] == {TP}
    assert len(calls) == 3
    assert consumer.commits == [{TP: 4}]


@pytest.mark.asyncio
async def test_records_returned_while_paused_are_rewound():
    materialize, _calls = _failing_then_ok(1)
    materializer = _materializer(materialize)
    buffered = {TP: [SimpleNamespace(offset=4), SimpleNamespace(offset=5)]}
    consumer = _GroupConsumer([_message()], buffered=buffered)

    await materializer._consume_routing_guardrails(consumer)

    assert ("seek", TP, 4) in consumer.calls


@pytest.mark.asyncio
async def test_a_commit_lost_to_a_rebalance_does_not_restart_the_consumer():
    CommitFailedError = getattr(
        pytest.importorskip("aiokafka.errors"), "CommitFailedError", None
    )
    if CommitFailedError is None:
        pytest.skip("aiokafka is stubbed")

    materialize, _calls = _failing_then_ok(1)
    materializer = _materializer(materialize)
    consumer = _GroupConsumer([_message()], commit_error=CommitFailedError("gone"))

    # Returns normally: the message is redelivered and rewritten safely.
    await materializer._consume_routing_guardrails(consumer)


# 2. Consecutive router deadline misses escalate to a cooldown.


class _Redis:
    def __init__(self):
        self.store = {}

    def pipeline(self, transaction=False):
        class _Pipeline:
            def __init__(self):
                self.count = 0

            def get(self, _key):
                self.count += 1
                return self

            def smembers(self, _key):
                self.count += 1
                return self

            async def execute(self):
                return [None] * self.count

        return _Pipeline()

    async def setex(self, key, ttl, value):
        raise ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )

    async def get(self, _key):
        raise ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )


def _cache():
    cache = RoutingPolicyCache({"enabled": True, "read_timeout_failure_threshold": 5})
    cache.redis_client = _Redis()
    cache._initialized = True
    return cache


def test_five_read_deadline_misses_in_a_row_start_the_cooldown():
    cache = _cache()

    for _ in range(5):
        cache.record_read_timeout()

    assert cache.is_degraded() is True


@pytest.mark.asyncio
async def test_a_successful_read_resets_the_miss_count():
    cache = _cache()

    for _ in range(4):
        cache.record_read_timeout()
    await cache.get_effective_policy("r1", "u1")
    for _ in range(4):
        cache.record_read_timeout()

    assert cache.is_degraded() is False


# 3. A deterministic Redis error is per message, not an outage.


@pytest.mark.asyncio
async def test_a_wrongtype_write_is_rejected_without_a_cooldown():
    cache = _cache()

    with pytest.raises(policy.PolicyCacheWriteRejected):
        await cache.materialize_request_enriched(
            {"request_id": "r1", "emitted_at": None}
        )

    assert cache.is_degraded() is False


@pytest.mark.asyncio
async def test_a_wrongtype_read_fails_one_request_without_a_cooldown():
    cache = _cache()

    assert await cache.get_session_policy("s1") is None
    assert cache.is_degraded() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    ["LOADING Redis is loading the dataset in memory", "BUSY script running"],
)
async def test_transient_replies_still_start_the_cooldown(reply):
    cache = _cache()

    async def setex(key, ttl, value):
        raise ResponseError(reply)

    cache.redis_client.setex = setex

    with pytest.raises(PolicyCacheUnavailableError):
        await cache.materialize_request_enriched({"request_id": "r1"})

    assert cache.is_degraded() is True


@pytest.mark.asyncio
async def test_a_rejected_write_is_skipped_and_committed():
    cache = _cache()
    cache.config["consumer"] = {"enable_auto_commit": False}
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    consumer = _GroupConsumer(
        [
            SimpleNamespace(
                value=json.dumps({"request_id": "r1"}),
                topic="requests.enriched",
                partition=0,
                offset=9,
            )
        ]
    )
    before = metric_value_for_labels(
        policy.POLICY_MATERIALIZER_INVALID_MESSAGES,
        {"topic": "requests_enriched", "destination": "skipped"},
    )

    await materializer._consume_requests_enriched(consumer)

    assert consumer.commits == [{TopicPartition("requests.enriched", 0): 10}]
    assert (
        metric_value_for_labels(
            policy.POLICY_MATERIALIZER_INVALID_MESSAGES,
            {"topic": "requests_enriched", "destination": "skipped"},
        )
        == before + 1
    )
