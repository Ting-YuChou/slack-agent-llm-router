"""A9: chat topics are provisioned explicitly and Kafka keeps its data."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from src import kafka_topics
from src.kafka_topics import TopicSpec, provision_topics, topic_specs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config" / "config.yaml", ROOT / "config" / "config.compose.yaml"]
KAFKA_CLIENTS = ("api", "workers", "flink-jobmanager")


def _kafka(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["kafka"]


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


class FakeAdmin:
    def __init__(self, existing=()):
        self.topics = {name: None for name in existing}
        self.created = []
        self.configured = []

    async def existing_topics(self):
        return list(self.topics)

    async def create_topics(self, specs):
        for spec in specs:
            self.topics[spec.name] = spec
        self.created.extend(specs)

    async def apply_configs(self, specs):
        for spec in specs:
            self.topics[spec.name] = spec
        self.configured.extend(specs)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_every_configured_topic_and_its_dlq_is_provisioned(path):
    kafka = _kafka(path)
    specs = {spec.name: spec for spec in topic_specs(kafka)}

    expected = set(kafka["topics"].values())
    expected |= {f"{name}.dlq" for name in kafka["topics"].values()}
    assert set(specs) == expected
    for spec in specs.values():
        assert spec.partitions >= 1
        assert spec.replication_factor >= 1
        assert int(spec.configs["retention.ms"]) > 0
        assert spec.configs["cleanup.policy"] in {"delete", "compact"}


def test_new_topic_keys_are_picked_up_from_config():
    specs = topic_specs({"topics": {"chat_feedback": "chat.feedback"}})

    assert [spec.name for spec in specs] == ["chat.feedback", "chat.feedback.dlq"]
    assert specs[1].configs["retention.ms"] == str(
        kafka_topics.DEFAULT_DLQ_RETENTION_MS
    )


def test_provisioning_settings_override_defaults():
    specs = topic_specs(
        {
            "topics": {"queries": "llm-queries"},
            "dlq": {"topic_suffix": ".dead"},
            "provisioning": {
                "partitions": 3,
                "retention_ms": 1000,
                "topics": {"queries": {"partitions": 6, "configs": {"x.y": 1}}},
            },
        }
    )

    queries, dlq = specs
    assert (queries.name, queries.partitions) == ("llm-queries", 6)
    assert queries.configs == {
        "cleanup.policy": "delete",
        "retention.ms": "1000",
        "x.y": "1",
    }
    assert (dlq.name, dlq.partitions) == ("llm-queries.dead", 3)


def test_policy_state_topic_has_short_retention_and_no_keyless_compaction():
    kafka = _kafka(ROOT / "config" / "config.yaml")
    spec = next(
        spec for spec in topic_specs(kafka) if spec.topic_key == "routing_policy_state"
    )

    assert spec.name == "routing.policy_state"
    assert int(spec.configs["retention.ms"]) <= 24 * 3600 * 1000
    # The Flink FlinkKafkaProducer sink cannot set record keys, and a compacted
    # topic rejects keyless records, so compaction must wait for a keyed sink.
    assert spec.configs["cleanup.policy"] == "delete"


@pytest.mark.asyncio
async def test_provisioning_is_idempotent():
    specs = topic_specs(_kafka(ROOT / "config" / "config.compose.yaml"))
    admin = FakeAdmin(existing=["llm-queries"])

    first = await provision_topics(admin, specs)
    second = await provision_topics(admin, specs)

    assert "llm-queries" not in first["created"]
    assert set(first["created"]) | {"llm-queries"} == {spec.name for spec in specs}
    assert second["created"] == []
    assert set(second["updated"]) == {spec.name for spec in specs}
    assert len(admin.created) == len(specs) - 1


@pytest.mark.asyncio
async def test_aiokafka_admin_tolerates_already_exists_and_raises_other_errors():
    pytest.importorskip("aiokafka.admin")

    class Client:
        def __init__(self, topic_errors, resources):
            self.topic_errors = topic_errors
            self.resources = resources
            self.new_topics = []
            self.config_resources = []

        async def describe_configs(self, resources, include_synonyms=False):
            return [SimpleNamespace(resources=[])]

        async def create_topics(self, new_topics):
            self.new_topics.extend(new_topics)
            return SimpleNamespace(topic_errors=self.topic_errors)

        async def alter_configs(self, resources):
            self.config_resources.extend(resources)
            return [SimpleNamespace(resources=self.resources)]

    spec = TopicSpec("a", 2, 1, {"retention.ms": "5"})
    ok = Client([("a", 0, None), ("b", 36, "exists")], [(0, None, 2, "a")])
    admin = kafka_topics.AIOKafkaTopicAdmin(ok)
    await admin.create_topics([spec])
    await admin.apply_configs([spec])
    assert ok.new_topics[0].num_partitions == 2
    assert ok.new_topics[0].topic_configs == {"retention.ms": "5"}
    assert ok.config_resources[0].configs == {"retention.ms": "5"}

    bad = kafka_topics.AIOKafkaTopicAdmin(
        Client([("a", 37, "bad partitions")], [(40, "denied", 2, "a")])
    )
    with pytest.raises(RuntimeError, match="a: error 37"):
        await bad.create_topics([spec])
    with pytest.raises(RuntimeError, match="a: error 40"):
        await bad.apply_configs([spec])


def test_kafka_disables_auto_create_and_keeps_its_data():
    compose = _compose()
    kafka = compose["services"]["kafka"]

    assert kafka["environment"]["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"
    assert kafka["environment"]["KAFKA_LOG_DIRS"] == "/var/lib/kafka/data"
    assert "kafka_data:/var/lib/kafka/data" in kafka["volumes"]
    assert "kafka_data" in compose["volumes"]


def test_chat_topics_service_provisions_before_kafka_clients():
    services = _compose()["services"]
    topics = services["chat-topics"]

    assert topics["command"][:3] == ["python", "-m", "src.kafka_topics"]
    assert "config/config.compose.yaml" in topics["command"]
    assert topics["build"]["dockerfile"] == "docker/Dockerfile"
    assert topics["depends_on"]["kafka"]["condition"] == "service_healthy"
    assert topics["restart"] == "no"
    assert topics.get("ports") is None
    for name in KAFKA_CLIENTS:
        depends_on = services[name]["depends_on"]
        assert depends_on["chat-topics"]["condition"] == (
            "service_completed_successfully"
        ), name


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_clickhouse_sink_group_replays_from_earliest(path):
    assert _kafka(path)["consumer"]["sink_auto_offset_reset"] == "earliest"


@pytest.mark.asyncio
async def test_integration_smoke_provisions_topics_before_the_pipeline(monkeypatch):
    from scripts import integration_smoke

    calls = []

    async def fake_provision(admin, specs):
        calls.append(("provision", sorted(spec.name for spec in specs)))
        return {"created": [], "updated": []}

    class StopPipeline(Exception):
        pass

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            calls.append(("initialize", None))
            raise StopPipeline()

    async def fake_admin(bootstrap_servers, timeout_seconds):
        calls.append(("admin", bootstrap_servers))
        return SimpleNamespace(close=lambda: asyncio.sleep(0))

    monkeypatch.setattr(integration_smoke, "provision_topics", fake_provision)
    monkeypatch.setattr(integration_smoke, "start_topic_admin", fake_admin)
    monkeypatch.setattr(integration_smoke, "KafkaIngestionPipeline", FakePipeline)
    monkeypatch.setattr(
        integration_smoke,
        "clickhouse_connect",
        SimpleNamespace(get_client=MagicMock(side_effect=RuntimeError("offline"))),
    )
    args = integration_smoke.parse_args(
        ["--clickhouse-password", "test-only", "--kafka-bootstrap", "k:1"]
    )

    with pytest.raises(StopPipeline):
        await integration_smoke.run_smoke(args)

    assert [call[0] for call in calls] == ["admin", "provision", "initialize"]
    assert calls[0][1] == "k:1"
    provisioned = calls[1][1]
    assert any(name.startswith("smoke-queries-") for name in provisioned)
    assert any(name.endswith(".dlq") for name in provisioned)


@pytest.mark.asyncio
async def test_backpressure_listener_is_accepted_by_a_real_consumer():
    aiokafka = pytest.importorskip("aiokafka")
    if not hasattr(aiokafka, "ConsumerRebalanceListener"):
        pytest.skip("aiokafka stub without ConsumerRebalanceListener")
    from src.llm_router_part3_pipeline import KafkaConsumerManager

    manager = KafkaConsumerManager({"topics": {"queries": "llm-queries"}}, MagicMock())
    consumer = aiokafka.AIOKafkaConsumer(bootstrap_servers="localhost:1")
    try:
        # The integration smoke failed here: subscribe() type-checks listeners.
        manager._install_rebalance_listener("queries", "llm-queries", consumer)
    finally:
        await consumer.stop()

    listener = manager._rebalance_listeners["queries"]
    assert isinstance(listener, aiokafka.ConsumerRebalanceListener)
