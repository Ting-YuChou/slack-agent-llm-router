"""D7: the policy materializer resolves its DLQ like the sink and provisioning."""

from pathlib import Path

import pytest
import yaml

from src import llm_router_part3_policy as policy
from src.kafka_topics import topic_specs
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_TOPICS = {
    "requests_enriched": "requests.enriched",
    "fast_lane_hints": "fast_lane_hints",
    "routing_guardrails": "routing.guardrails",
    "routing_policy_state": "routing.policy_state",
}


def _materializer(kafka, **cache_consumer):
    cache = RoutingPolicyCache({"enabled": False, "consumer": cache_consumer})
    return PolicyMaterializer(kafka, cache)


def test_dlq_defaults_to_on_like_the_shared_helper():
    assert _materializer({"topics": dict(MATERIALIZER_TOPICS)}).dlq_enabled is True


@pytest.mark.parametrize(
    "kafka",
    [
        {"dlq": {"enabled": False}},
        {"consumer": {"dlq_enabled": False}},
        {"dlq": {"enabled": False}, "consumer": {"dlq_enabled": True}},
    ],
    ids=["dlq.enabled", "consumer.dlq_enabled", "dlq.enabled-wins"],
)
def test_kafka_settings_disable_the_dlq(kafka):
    assert _materializer({"topics": {}, **kafka}).dlq_enabled is False


def test_policy_override_can_turn_the_dlq_off():
    materializer = _materializer({"topics": {}}, dlq_enabled=False)
    assert materializer.dlq_enabled is False


def test_policy_override_cannot_enable_an_unprovisioned_dlq():
    materializer = _materializer(
        {"topics": {}, "dlq": {"enabled": False}}, dlq_enabled=True
    )
    assert materializer.dlq_enabled is False


@pytest.mark.parametrize(
    "kafka, expected",
    [
        ({"dead_letter_topics": {"routing_policy_state": "ops.dlq"}}, "ops.dlq"),
        (
            {"topics": {"routing_policy_state_dlq": "policy-dead"}},
            "policy-dead",
        ),
        ({"dlq": {"topic_suffix": "-bad"}}, "routing.policy_state-bad"),
        ({}, "routing.policy_state.dlq"),
    ],
)
def test_dlq_topic_uses_the_shared_resolution(kafka, expected):
    config = {"topics": dict(MATERIALIZER_TOPICS), **kafka}
    config["topics"].update(kafka.get("topics", {}))
    materializer = _materializer(config)

    assert (
        materializer._dead_letter_topic("routing_policy_state", "routing.policy_state")
        == expected
    )


@pytest.mark.asyncio
async def test_dlq_producer_is_idempotent(monkeypatch):
    created = []

    class _Producer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        async def start(self):
            return None

    monkeypatch.setattr(policy, "AIOKafkaProducer", _Producer)
    materializer = _materializer({"topics": {}})

    await materializer._start_dlq_producer()

    (producer,) = created
    assert producer.kwargs["acks"] == "all"
    assert producer.kwargs["enable_idempotence"] is True


def _shipped(name):
    config = yaml.safe_load((ROOT / "config" / name).read_text("utf-8"))
    return config["kafka"], config.get("policy_cache", {}).get("consumer", {})


INVARIANT_CASES = [
    ({"topics": dict(MATERIALIZER_TOPICS)}, {}),
    ({"topics": dict(MATERIALIZER_TOPICS), "dlq": {"enabled": False}}, {}),
    ({"topics": dict(MATERIALIZER_TOPICS), "consumer": {"dlq_enabled": False}}, {}),
    (
        {"topics": dict(MATERIALIZER_TOPICS), "dlq": {"enabled": False}},
        {"dlq_enabled": True},
    ),
    (
        {
            "topics": dict(MATERIALIZER_TOPICS),
            "dead_letter_topics": {"fast_lane_hints": "hints.dead"},
            "dlq": {"topic_suffix": "-x"},
        },
        {},
    ),
    (
        {
            "topics": {
                **MATERIALIZER_TOPICS,
                "requests_enriched_dlq": "enriched.dead",
            }
        },
        {},
    ),
    _shipped("config.yaml"),
    _shipped("config.compose.yaml"),
]


@pytest.mark.parametrize("kafka, cache_consumer", INVARIANT_CASES)
def test_every_dlq_the_materializer_uses_is_provisioned(kafka, cache_consumer):
    materializer = _materializer(kafka, **cache_consumer)
    provisioned = {spec.name for spec in topic_specs(kafka)}

    if not materializer.dlq_enabled:
        return
    for topic_key in MATERIALIZER_TOPICS:
        source_topic = (kafka.get("topics") or {}).get(topic_key)
        if not source_topic:
            continue
        dlq_topic = materializer._dead_letter_topic(topic_key, source_topic)
        assert dlq_topic in provisioned, (topic_key, dlq_topic)
