"""P1: the materializer's pause-and-poll wait against a real Kafka broker.

Opt-in: set CHAT_TEST_KAFKA_BROKERS (for example ``localhost:19092``). Each
run uses its own topic and consumer group.
"""

import asyncio
import json
import os
import uuid

import pytest

from src.llm_router_part3_policy import (
    PolicyCacheUnavailableError,
    PolicyMaterializer,
    RoutingPolicyCache,
)

BROKERS = os.environ.get("CHAT_TEST_KAFKA_BROKERS")

pytestmark = pytest.mark.skipif(
    not BROKERS, reason="set CHAT_TEST_KAFKA_BROKERS to run against a real Kafka"
)

MESSAGES = 8
# Sequence number -> how many times the cache is unavailable for it.
FAILURES = {0: 3, 4: 2}


class _FlakyCache:
    """Raises PolicyCacheUnavailableError FAILURES[seq] times, then succeeds."""

    def __init__(self):
        self.failures_left = dict(FAILURES)
        self.materialized = []

    async def materialize(self, payload):
        seq = payload["seq"]
        if self.failures_left.get(seq, 0) > 0:
            self.failures_left[seq] -= 1
            raise PolicyCacheUnavailableError("redis down")
        self.materialized.append(seq)


class _RecordingConsumer:
    """Wraps a real AIOKafkaConsumer and records pause, poll, seek and resume."""

    def __init__(self, consumer):
        self._consumer = consumer
        self.events = []
        self.sought = []

    def __getattr__(self, name):
        return getattr(self._consumer, name)

    def __aiter__(self):
        return self._consumer.__aiter__()

    def pause(self, *partitions):
        self.events.append("pause")
        return self._consumer.pause(*partitions)

    def resume(self, *partitions):
        self.events.append("resume")
        return self._consumer.resume(*partitions)

    async def getmany(self, *partitions, **kwargs):
        if self._consumer.paused():
            self.events.append("poll")
        return await self._consumer.getmany(*partitions, **kwargs)

    def seek(self, partition, offset):
        self.sought.append(offset)
        return self._consumer.seek(partition, offset)


async def _create_topic(topic):
    admin_module = pytest.importorskip("aiokafka.admin")
    admin = admin_module.AIOKafkaAdminClient(bootstrap_servers=BROKERS)
    await admin.start()
    try:
        await admin.create_topics(
            [admin_module.NewTopic(topic, num_partitions=1, replication_factor=1)]
        )
    finally:
        await admin.close()


async def _produce(topic, count):
    aiokafka = pytest.importorskip("aiokafka")
    producer = aiokafka.AIOKafkaProducer(bootstrap_servers=BROKERS)
    await producer.start()
    try:
        for seq in range(count):
            await producer.send_and_wait(
                topic,
                json.dumps({"seq": seq, "event_version": "1.0"}).encode(),
                partition=0,
            )
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_pause_and_poll_keeps_every_record_once_and_commits():
    aiokafka = pytest.importorskip("aiokafka")
    structs = pytest.importorskip("aiokafka.structs")
    run = uuid.uuid4().hex[:10]
    topic = f"p1-policy-state-{run}"
    group = f"p1-materializer-{run}"
    await _create_topic(topic)
    await _produce(topic, MESSAGES)

    cache = RoutingPolicyCache(
        {
            "enabled": False,
            "consumer": {
                "group_id": group,
                "auto_offset_reset": "earliest",
                "enable_auto_commit": False,
                "redis_retry_initial_backoff_seconds": 0.2,
                "redis_retry_max_backoff_seconds": 0.4,
            },
        }
    )
    materializer = PolicyMaterializer(
        {"bootstrap_servers": BROKERS, "topics": {"routing_policy_state": topic}},
        cache,
    )
    flaky = _FlakyCache()
    cache.materialize_routing_policy_state = flaky.materialize
    config = dict(materializer.consumer_config)
    config["bootstrap_servers"] = BROKERS
    real = aiokafka.AIOKafkaConsumer(topic, **config)
    await real.start()
    consumer = _RecordingConsumer(real)
    partition = structs.TopicPartition(topic, 0)
    materializer.running = True
    task = asyncio.create_task(materializer._consume_routing_policy_state(consumer))

    async def _all_committed():
        while (
            len(flaky.materialized) < MESSAGES
            or (await real.committed(partition)) != MESSAGES
        ):
            assert not task.done(), task.exception()
            await asyncio.sleep(0.1)

    try:
        try:
            await asyncio.wait_for(_all_committed(), timeout=60)
        finally:
            materializer.running = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        # No loss and no duplicates, in order, despite the paused waits.
        assert flaky.materialized == list(range(MESSAGES))
        assert flaky.failures_left == {seq: 0 for seq in FAILURES}
        # Each outage paused once, polled while paused, then resumed.
        assert consumer.events.count("pause") == len(FAILURES)
        assert consumer.events.count("resume") == len(FAILURES)
        for _ in FAILURES:
            first_pause = consumer.events.index("pause")
            resume = consumer.events.index("resume")
            assert "poll" in consumer.events[first_pause:resume]
            consumer.events = consumer.events[resume + 1 :]
        # Any record a paused poll returned was sought back, never skipped.
        assert all(0 <= offset < MESSAGES for offset in consumer.sought)
        assert not real.paused()
        assert await real.committed(partition) == MESSAGES
    finally:
        await real.stop()

    # A fresh member of the group starts after the last committed record.
    await _produce(topic, 1)
    again = aiokafka.AIOKafkaConsumer(topic, **config)
    await again.start()
    try:
        record = await asyncio.wait_for(again.getone(), timeout=30)
        assert json.loads(record.value)["seq"] == 0
        assert record.offset == MESSAGES
    finally:
        await again.stop()
