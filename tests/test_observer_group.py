"""D4: observers read from their own consumer group, starting at latest."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import KafkaConsumerManager
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config.yaml", ROOT / "config" / "config.compose.yaml"]
TOPICS = {
    "queries": "llm-queries",
    "requests_enriched": "requests.enriched",
    "analytics_model_metrics_1m": "analytics.model_metrics_1m",
    "routing_guardrails": "routing.guardrails",
    "routing_policy_state": "routing.policy_state",
    "alerts": "alerts",
}


class _FakeKafkaConsumer:
    created: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.topics = []
        self.commit = AsyncMock()
        _FakeKafkaConsumer.created.append(self)

    def subscribe(self, topics, listener=None):
        self.topics = list(topics)

    async def start(self):
        return None

    async def stop(self):
        return None


@pytest.fixture
def fake_kafka(monkeypatch):
    _FakeKafkaConsumer.created = []
    monkeypatch.setattr(
        "src.llm_router_part3_pipeline.AIOKafkaConsumer", _FakeKafkaConsumer
    )
    return _FakeKafkaConsumer


def _manager(**consumer):
    return KafkaConsumerManager(
        {
            "topics": dict(TOPICS),
            "consumer": {
                "group_id": "sink-group",
                "auto_offset_reset": "latest",
                "sink_auto_offset_reset": "earliest",
                "enable_auto_commit": False,
                **consumer,
            },
            "dlq": {"enabled": False},
        },
        MagicMock(),
    )


def test_observer_group_defaults_to_latest_and_sink_is_unchanged():
    manager = _manager()

    assert manager.observer_consumer_config["group_id"].startswith(
        "llm-router-observers-"
    )
    assert manager.observer_consumer_config["auto_offset_reset"] == "latest"
    assert manager.consumer_config["group_id"] == "sink-group"
    assert manager.consumer_config["auto_offset_reset"] == "earliest"


def test_observer_group_is_configurable():
    manager = _manager(
        observer_group_prefix="obs", observer_auto_offset_reset="earliest"
    )

    assert manager.observer_consumer_config["group_id"].startswith("obs-")
    assert manager.observer_consumer_config["auto_offset_reset"] == "earliest"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_shipped_configs_name_the_observer_group(path):
    kafka = yaml.safe_load(path.read_text("utf-8"))["kafka"]
    manager = KafkaConsumerManager(kafka, MagicMock())

    assert manager.observer_consumer_config["group_id"].startswith(
        "llm-router-observers-"
    )
    assert manager.observer_consumer_config["auto_offset_reset"] == "latest"
    assert manager.consumer_config["group_id"] != (
        manager.observer_consumer_config["group_id"]
    )


@pytest.mark.asyncio
async def test_sink_group_has_no_requests_enriched_consumer(fake_kafka):
    manager = _manager()

    await manager.initialize()

    assert "requests_enriched" not in manager.consumers
    assert set(manager.consumers) == set(TOPICS) - {"requests_enriched"}
    assert {c.kwargs["group_id"] for c in fake_kafka.created} == {"sink-group"}


@pytest.mark.asyncio
async def test_no_observers_means_no_observer_consumers(fake_kafka):
    manager = _manager()
    await manager.initialize()
    sink_count = len(fake_kafka.created)

    await manager._start_observer_consumers()

    assert len(fake_kafka.created) == sink_count
    assert not manager.observer_consumer_keys


@pytest.mark.asyncio
async def test_observer_consumers_start_only_for_observed_topics(fake_kafka):
    manager = _manager()
    await manager.initialize()
    manager.register_observer("requests_enriched", lambda event: None)
    manager.register_observer("alerts", lambda event: None)
    fake_kafka.created = []

    await manager._start_observer_consumers()

    by_topic = {c.topics[0]: c.kwargs for c in fake_kafka.created}
    assert set(by_topic) == {"requests.enriched", "alerts"}
    for kwargs in by_topic.values():
        assert kwargs["group_id"] == manager.observer_consumer_config["group_id"]
        assert kwargs["auto_offset_reset"] == "latest"
    assert manager.observer_consumer_keys == {
        "requests_enriched": "requests_enriched",
        "alerts_observer": "alerts",
    }
    # The sink consumer for alerts is still there, in the sink group.
    assert manager.consumers["alerts"].kwargs["group_id"] == "sink-group"
    assert manager.consumers["alerts_observer"].kwargs["group_id"] == (
        manager.observer_consumer_config["group_id"]
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


def _alert(offset=7, value=None):
    return SimpleNamespace(
        value=value
        if value is not None
        else {
            "event_type": "alerts",
            "alert_type": "latency",
            "severity": "warning",
            "description": "slow",
            "emitted_at": "2026-10-30T00:00:00+00:00",
        },
        topic="alerts",
        partition=0,
        offset=offset,
        key=None,
    )


@pytest.mark.asyncio
async def test_sink_loop_no_longer_notifies_observers():
    manager = _manager()
    seen = []
    manager.register_observer("alerts", seen.append)
    messages = _Messages([_alert()])
    manager.consumers["alerts"] = messages

    await manager._consume_alerts(messages)

    assert seen == []
    assert len(manager.batch_processors["alerts"]) == 1


@pytest.mark.asyncio
async def test_observer_loop_notifies_then_commits():
    manager = _manager(observer_enable_auto_commit=False)
    seen = []
    manager.register_observer("alerts", lambda event: seen.append(event["alert_type"]))
    messages = _Messages([_alert()])
    manager.consumers["alerts_observer"] = messages
    manager.observer_consumer_keys["alerts_observer"] = "alerts"

    await manager._run_consumer("alerts_observer", messages)

    assert seen == ["latency"]
    assert manager.batch_processors["alerts"] == []
    messages.commit.assert_awaited_once_with({TopicPartition("alerts", 0): 8})


@pytest.mark.asyncio
async def test_observer_loop_skips_poison_without_dead_lettering():
    manager = _manager(observer_enable_auto_commit=False)
    manager.dlq_producer = AsyncMock()
    manager.register_observer("alerts", lambda event: None)
    messages = _Messages([_alert(value=b"{bad")])
    manager.consumers["alerts_observer"] = messages
    manager.observer_consumer_keys["alerts_observer"] = "alerts"

    await manager._run_consumer("alerts_observer", messages)

    # The sink group dead-letters the same record; the observer only skips it.
    manager.dlq_producer.send_and_wait.assert_not_awaited()
    manager.dlq_producer.send.assert_not_awaited()
    messages.commit.assert_awaited_once_with({TopicPartition("alerts", 0): 8})


@pytest.mark.asyncio
async def test_observer_restart_uses_the_observer_group(fake_kafka):
    manager = _manager()
    manager.register_observer("routing_guardrails", lambda event: None)
    await manager._start_observer_consumers()
    fake_kafka.created = []

    await manager._restart_consumer("routing_guardrails_observer")

    (restarted,) = fake_kafka.created
    assert restarted.kwargs["group_id"] == (
        manager.observer_consumer_config["group_id"]
    )
    assert restarted.topics == ["routing.guardrails"]


def test_materializer_group_falls_back_to_its_own_group():
    cache = RoutingPolicyCache({"enabled": False})
    materializer = PolicyMaterializer(
        {"consumer": {"group_id": "llm-router-consumer"}}, cache
    )

    assert materializer.consumer_config["group_id"] == (
        "llm-router-policy-materializer"
    )


def test_materializer_group_still_honours_its_own_setting():
    cache = RoutingPolicyCache({"enabled": False, "consumer": {"group_id": "mine"}})
    materializer = PolicyMaterializer(
        {"consumer": {"group_id": "llm-router-consumer"}}, cache
    )

    assert materializer.consumer_config["group_id"] == "mine"
