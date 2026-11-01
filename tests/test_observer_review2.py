"""D4 round-2 review: observers never commit, and observers registered
during startup are not lost."""

import asyncio
import json
import logging
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import KafkaConsumerManager

BROKERS = os.environ.get("CHAT_TEST_KAFKA_BROKERS")


def _manager(topics=None, **consumer):
    return KafkaConsumerManager(
        {
            "topics": topics or {"queries": "r2.q", "alerts": "r2.alerts"},
            "consumer": {
                "group_id": "sink-group",
                "enable_auto_commit": False,
                **consumer,
            },
            "dlq": {"enabled": False},
        },
        MagicMock(),
    )


class _Messages:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


def _event(topic, offset):
    return SimpleNamespace(
        value={"request_id": f"r{offset}"},
        topic=topic,
        partition=0,
        offset=offset,
        key=None,
    )


# --- 1: observers never commit ------------------------------------------


@pytest.mark.parametrize("setting", [None, True, False])
def test_observer_config_never_auto_commits(setting):
    extra = {} if setting is None else {"observer_enable_auto_commit": setting}
    config = _manager(**extra).observer_consumer_config

    assert config["enable_auto_commit"] is False
    assert config["auto_offset_reset"] == "latest"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topic_key, consumer_key",
    [
        ("alerts", "alerts_observer"),
        ("requests_enriched", "requests_enriched"),
    ],
)
async def test_started_observer_consumers_never_commit(topic_key, consumer_key):
    manager = _manager(
        topics={topic_key: f"r2.{topic_key}"}, observer_enable_auto_commit=True
    )
    seen = []
    manager.register_observer(topic_key, lambda event: seen.append(event))
    messages = _Messages([_event(f"r2.{topic_key}", 1), _event(f"r2.{topic_key}", 2)])
    manager.consumers[consumer_key] = messages
    manager.observer_consumer_keys[consumer_key] = topic_key

    await manager._run_consumer(consumer_key, messages)

    assert len(seen) == 2
    messages.commit.assert_not_awaited()


@pytest.mark.skipif(not BROKERS, reason="set CHAT_TEST_KAFKA_BROKERS")
@pytest.mark.asyncio
async def test_reused_observer_group_has_no_committed_offsets():
    aiokafka = pytest.importorskip("aiokafka")
    admin_module = pytest.importorskip("aiokafka.admin")
    run = uuid.uuid4().hex[:10]
    topic = f"r2-alerts-{run}"
    admin = admin_module.AIOKafkaAdminClient(bootstrap_servers=BROKERS)
    await admin.start()
    try:
        await admin.create_topics(
            [admin_module.NewTopic(topic, num_partitions=1, replication_factor=1)]
        )
    finally:
        await admin.close()

    manager = KafkaConsumerManager(
        {
            "bootstrap_servers": BROKERS,
            "topics": {"alerts": topic},
            "consumer": {"observer_group_prefix": f"r2-obs-{run}"},
            "dlq": {"enabled": False},
        },
        MagicMock(),
    )
    seen = []
    manager.register_observer("alerts", lambda event: seen.append(event["n"]))
    await manager._start_observer_consumers()
    consumer = manager.consumers["alerts_observer"]
    partition = TopicPartition(topic, 0)
    task = None
    try:
        while not consumer.assignment():
            await asyncio.sleep(0.1)
        await consumer.position(partition)  # resolve "latest" before producing
        producer = aiokafka.AIOKafkaProducer(bootstrap_servers=BROKERS)
        await producer.start()
        try:
            for n in range(3):
                await producer.send_and_wait(topic, json.dumps({"n": n}).encode())
        finally:
            await producer.stop()
        manager.running = True
        task = asyncio.create_task(manager._run_consumer("alerts_observer", consumer))

        async def _seen_all():
            while len(seen) < 3:
                await asyncio.sleep(0.1)

        await asyncio.wait_for(_seen_all(), timeout=30)
    finally:
        manager.running = False
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await consumer.stop()

    assert seen == [0, 1, 2]
    # A restart reuses the same group id (same host, PID 1 in a container):
    # nothing was committed, so it starts at latest instead of replaying.
    again = aiokafka.AIOKafkaConsumer(
        bootstrap_servers=BROKERS,
        group_id=manager.observer_consumer_config["group_id"],
        enable_auto_commit=False,
    )
    await again.start()
    try:
        assert await again.committed(partition) is None
    finally:
        await again.stop()


# --- 5: observers registered during startup -------------------------------


class _TaskGroup:
    def __init__(self, fail=False):
        self.tasks = []
        self.fail = fail

    def create_task(self, coro, name=None):
        if self.fail:
            coro.close()
            raise RuntimeError("TaskGroup is shutting down")
        self.tasks.append((name, coro))


def _close(task_group):
    for _name, coro in task_group.tasks:
        coro.close()


def test_registration_during_startup_is_queued_then_started():
    manager = _manager()
    manager.running = True  # start_consuming() has begun, no task group yet

    manager.register_observer("alerts", lambda event: None)
    manager.register_observer("alerts", lambda event: None)

    task_group = _TaskGroup()
    manager._attach_task_group(task_group)
    try:
        assert [name for name, _ in task_group.tasks] == [
            "kafka_consumer_alerts_observer"
        ]
        assert manager._task_group is task_group
    finally:
        _close(task_group)


def test_registration_after_shutdown_starts_nothing():
    manager = _manager()
    manager.register_observer("alerts", lambda event: None)
    task_group = _TaskGroup()

    manager._attach_task_group(task_group)

    # Not running: start_consuming() starts registered observers itself.
    assert task_group.tasks == []


def test_task_group_shutting_down_is_a_warning(caplog):
    manager = _manager()
    manager.running = True
    manager._task_group = _TaskGroup(fail=True)

    with caplog.at_level(logging.WARNING):
        manager.register_observer("alerts", lambda event: None)

    assert "alerts" in caplog.text
    assert not manager._pending_observer_starts
    # A later registration may try again.
    manager._task_group = _TaskGroup()
    manager.register_observer("alerts", lambda event: None)
    assert len(manager._task_group.tasks) == 1
    _close(manager._task_group)
