"""D4 review: per-instance observer groups, auto-commit, a skip-only
requests_enriched path and observers registered after start."""

import logging
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import KafkaConsumerManager

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config.yaml", ROOT / "config" / "config.compose.yaml"]
TOPICS = {
    "queries": "llm-queries",
    "requests_enriched": "requests.enriched",
    "alerts": "alerts",
}


def _manager(**consumer):
    return KafkaConsumerManager(
        {
            "topics": dict(TOPICS),
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
    def __init__(self, messages, commit=None):
        self._messages = list(messages)
        self.commit = commit or AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


def _event(topic, offset, value=None):
    return SimpleNamespace(
        value=value if value is not None else {"request_id": f"r{offset}"},
        topic=topic,
        partition=0,
        offset=offset,
        key=None,
    )


# --- 4: one observer group per instance, auto-committed ---------------------


def test_observer_group_is_per_instance_and_auto_commits():
    config = _manager().observer_consumer_config

    assert config["group_id"].startswith("llm-router-observers-")
    assert str(os.getpid()) in config["group_id"]
    assert socket.gethostname().split(".")[0] in config["group_id"]
    assert config["auto_offset_reset"] == "latest"
    assert config["enable_auto_commit"] is True


def test_observer_group_prefix_is_configurable():
    config = _manager(observer_group_prefix="obs").observer_consumer_config

    assert config["group_id"].startswith("obs-")
    assert config["group_id"].endswith(f"-{os.getpid()}")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_shipped_configs_set_the_observer_prefix(path):
    kafka = yaml.safe_load(path.read_text("utf-8"))["kafka"]

    assert kafka["consumer"]["observer_group_prefix"] == "llm-router-observers"
    assert "observer_group_id" not in kafka["consumer"]


# --- 5: auto-committed observers never commit by hand -----------------------


@pytest.mark.asyncio
async def test_auto_committed_observer_does_not_commit_by_hand():
    manager = _manager()
    seen = []
    manager.register_observer("alerts", lambda event: seen.append(event["request_id"]))
    messages = _Messages([_event("alerts", 1), _event("alerts", 2)])
    manager.consumers["alerts_observer"] = messages
    manager.observer_consumer_keys["alerts_observer"] = "alerts"

    await manager._run_consumer("alerts_observer", messages)

    assert seen == ["r1", "r2"]
    messages.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_commit_does_not_stop_the_observer_loop():
    from aiokafka.errors import CommitFailedError

    manager = _manager(observer_enable_auto_commit=False)
    seen = []
    manager.register_observer("alerts", lambda event: seen.append(event["request_id"]))
    commit = AsyncMock(side_effect=[CommitFailedError("rebalanced"), None])
    messages = _Messages([_event("alerts", 1), _event("alerts", 2)], commit=commit)
    manager.consumers["alerts_observer"] = messages
    manager.observer_consumer_keys["alerts_observer"] = "alerts"

    await manager._run_consumer("alerts_observer", messages)

    assert seen == ["r1", "r2"]
    assert commit.await_count == 2


# --- 6: requests_enriched skips unreadable events, never dead-letters -------


@pytest.mark.asyncio
@pytest.mark.parametrize("dlq_on", [True, False])
async def test_requests_enriched_skips_unreadable_events(dlq_on):
    manager = _manager()
    manager.dlq_enabled = dlq_on
    manager.dlq_producer = AsyncMock() if dlq_on else None
    seen = []
    manager.register_observer(
        "requests_enriched", lambda event: seen.append(event["request_id"])
    )
    messages = _Messages(
        [
            _event("requests.enriched", 3, value=b"{bad"),
            _event("requests.enriched", 4),
        ]
    )
    manager.consumers["requests_enriched"] = messages

    await manager._consume_requests_enriched(messages)

    assert seen == ["r4"]
    if dlq_on:
        manager.dlq_producer.send_and_wait.assert_not_awaited()
        manager.dlq_producer.send.assert_not_awaited()
    # Sink-group commit settings here (no observer consumer was started), so
    # both records are committed, the unreadable one included.
    assert messages.commit.await_args_list[-1].args == (
        {TopicPartition("requests.enriched", 0): 5},
    )


# --- 8: observers registered after start get a consumer ---------------------


class _TaskGroup:
    def __init__(self):
        self.tasks = []

    def create_task(self, coro, name=None):
        self.tasks.append((name, coro))


class _FakeKafkaConsumer:
    created: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.topics = []
        _FakeKafkaConsumer.created.append(self)

    def subscribe(self, topics, listener=None):
        self.topics = list(topics)

    async def start(self):
        return None

    async def stop(self):
        return None


@pytest.mark.asyncio
async def test_observer_registered_while_running_starts_its_consumer(monkeypatch):
    _FakeKafkaConsumer.created = []
    monkeypatch.setattr(
        "src.llm_router_part3_pipeline.AIOKafkaConsumer", _FakeKafkaConsumer
    )
    manager = _manager()
    manager.running = True
    manager._task_group = _TaskGroup()
    manager._supervise_consumer = AsyncMock()

    manager.register_observer("alerts", lambda event: None)
    manager.register_observer("alerts", lambda event: None)

    ((name, coro),) = manager._task_group.tasks
    assert name == "kafka_consumer_alerts_observer"
    await coro
    (created,) = _FakeKafkaConsumer.created
    assert created.topics == ["alerts"]
    assert manager.observer_consumer_keys == {"alerts_observer": "alerts"}
    manager._supervise_consumer.assert_awaited_once_with(
        "alerts_observer", manager.consumers["alerts_observer"]
    )


def test_observer_registered_while_running_without_a_task_group_warns(caplog):
    manager = _manager()
    manager.running = True

    with caplog.at_level(logging.WARNING):
        manager.register_observer("alerts", lambda event: None)

    assert "alerts" in caplog.text
    assert "observer" in caplog.text.lower()


def test_observer_registered_before_start_waits_for_start():
    manager = _manager()
    manager._task_group = _TaskGroup()

    manager.register_observer("alerts", lambda event: None)

    assert manager._task_group.tasks == []
