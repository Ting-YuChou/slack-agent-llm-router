"""A9 review: sink-only earliest offsets, subscribe-before-start restarts,
config-preserving topic updates and one shared DLQ resolution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from src import kafka_topics
from src.kafka_topics import TopicSpec, topic_specs
from src.llm_router_part3_pipeline import KafkaConsumerManager

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config.yaml", ROOT / "config" / "config.compose.yaml"]


def _kafka(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["kafka"]


# --- item 1: only the ClickHouse sink replays from earliest -----------------


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_shared_consumer_default_stays_latest_and_only_the_sink_uses_earliest(path):
    kafka = _kafka(path)

    # The policy materializer and other consumers fall back to this key.
    assert kafka["consumer"]["auto_offset_reset"] == "latest"
    assert kafka["consumer"]["sink_auto_offset_reset"] == "earliest"
    manager = KafkaConsumerManager(kafka, MagicMock())
    assert manager.consumer_config["auto_offset_reset"] == "earliest"


def test_sink_falls_back_to_the_shared_setting_when_not_configured():
    manager = KafkaConsumerManager(
        {"consumer": {"auto_offset_reset": "latest"}}, MagicMock()
    )
    assert manager.consumer_config["auto_offset_reset"] == "latest"


# --- item 5: restarts subscribe before start --------------------------------


class _RecordingConsumer:
    instances = []

    def __init__(self, *topics, **kwargs):
        self.topics = topics
        self.kwargs = kwargs
        self.calls = []
        _RecordingConsumer.instances.append(self)

    def subscribe(self, topics=None, listener=None):
        self.calls.append(("subscribe", tuple(topics)))

    async def start(self):
        self.calls.append(("start",))

    async def stop(self):
        self.calls.append(("stop",))

    def assignment(self):
        return set()


@pytest.mark.parametrize("path", ["initialize", "restart"])
@pytest.mark.asyncio
async def test_consumers_subscribe_with_the_listener_before_start(monkeypatch, path):
    import src.llm_router_part3_pipeline as pipeline

    _RecordingConsumer.instances = []
    monkeypatch.setattr(pipeline, "AIOKafkaConsumer", _RecordingConsumer)
    manager = KafkaConsumerManager(
        {"topics": {"queries": "llm-queries"}, "dlq": {"enabled": False}},
        MagicMock(),
    )
    await manager.initialize()
    if path == "restart":
        await manager._restart_consumer("queries")

    consumer = _RecordingConsumer.instances[-1]
    assert consumer.topics == ()
    assert consumer.calls[:2] == [("subscribe", ("llm-queries",)), ("start",)]
    assert manager.consumers["queries"] is consumer


# --- item 7: topic updates keep operator-set configs -------------------------


def _describe_entry(name, value, source):
    # DescribeConfigs v1 entry: name, value, read_only, is_default, is_sensitive,
    # synonyms[(name, value, source)]; source 1 = DYNAMIC_TOPIC_CONFIG.
    return (name, value, False, source != 1, False, [(name, value, source)])


class _AdminClient:
    def __init__(self, entries):
        self.entries = entries
        self.altered = []

    async def describe_configs(self, resources, include_synonyms=False):
        return [
            SimpleNamespace(
                resources=[
                    (0, None, 2, resource.name, self.entries) for resource in resources
                ]
            )
        ]

    async def alter_configs(self, resources):
        self.altered.extend(resources)
        return [SimpleNamespace(resources=[(0, None, 2, r.name) for r in resources])]


@pytest.mark.asyncio
async def test_apply_configs_merges_existing_dynamic_configs():
    pytest.importorskip("aiokafka.admin")
    client = _AdminClient(
        [
            _describe_entry("retention.ms", "1000", 1),
            _describe_entry("max.message.bytes", "2097152", 1),
            _describe_entry("cleanup.policy", "delete", 5),
        ]
    )
    spec = TopicSpec("t", 1, 1, {"cleanup.policy": "delete", "retention.ms": "5000"})

    await kafka_topics.AIOKafkaTopicAdmin(client).apply_configs([spec])

    [resource] = client.altered
    assert resource.configs == {
        "max.message.bytes": "2097152",
        "cleanup.policy": "delete",
        "retention.ms": "5000",
    }


@pytest.mark.asyncio
async def test_apply_configs_skips_topics_already_matching():
    pytest.importorskip("aiokafka.admin")
    client = _AdminClient(
        [
            _describe_entry("retention.ms", "5000", 1),
            _describe_entry("cleanup.policy", "delete", 1),
            _describe_entry("max.message.bytes", "2097152", 1),
        ]
    )
    spec = TopicSpec("t", 1, 1, {"cleanup.policy": "delete", "retention.ms": "5000"})

    await kafka_topics.AIOKafkaTopicAdmin(client).apply_configs([spec])

    assert client.altered == []


# --- item 9: one DLQ resolution for the provisioner and the consumer ---------


@pytest.mark.parametrize(
    "kafka",
    [
        {"topics": {"queries": "q"}, "consumer": {"dlq_enabled": False}},
        {"topics": {"queries": "q"}, "dlq": {"enabled": False}},
    ],
    ids=["consumer.dlq_enabled", "dlq.enabled"],
)
def test_disabled_dlq_is_not_provisioned_and_not_used(kafka):
    assert [spec.name for spec in topic_specs(kafka)] == ["q"]
    assert KafkaConsumerManager(kafka, MagicMock()).dlq_enabled is False


@pytest.mark.parametrize(
    "kafka, expected",
    [
        (
            {"topics": {"queries": "q"}, "dead_letter_topics": {"queries": "dead"}},
            "dead",
        ),
        ({"topics": {"queries": "q", "queries_dlq": "q-dead"}}, "q-dead"),
        ({"topics": {"queries": "q"}, "dlq": {"topic_suffix": ".bad"}}, "q.bad"),
    ],
    ids=["override", "topic-key", "suffix"],
)
def test_provisioner_and_consumer_agree_on_the_dlq_topic(kafka, expected):
    provisioned = {spec.name for spec in topic_specs(kafka)}
    consumer = KafkaConsumerManager(kafka, MagicMock())
    message = SimpleNamespace(topic="q")

    assert consumer._dead_letter_topic("queries", message) == expected
    assert expected in provisioned
    assert kafka_topics.dead_letter_topic_name(kafka, "queries", "q") == expected
