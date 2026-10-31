"""O2 review: lag alerts are sink-only, revoked partitions drop their lag
series, and lag is refreshed from the broker between messages."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import (
    KafkaConsumerManager,
    _BatchBackpressureRebalanceListener,
)
from src.utils.metrics import PIPELINE_METRICS, metric_value_for_labels

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "docker" / "observability" / "prometheus" / "llm-router-alerts.yml"
OBSERVER_KEYS = ["requests_enriched", "alerts_observer", "routing_guardrails_observer"]
SINK_KEYS = ["queries", "alerts", "routing_guardrails", "inference_completed"]


def _rules():
    document = yaml.safe_load(RULES_PATH.read_text())
    return {rule["alert"]: rule for g in document["groups"] for rule in g["rules"]}


def _excluded_topic_keys(expr):
    match = re.search(r'topic_key\s*!~\s*"([^"]+)"', expr)
    assert match, expr
    pattern = re.compile(match.group(1))
    return lambda key: pattern.fullmatch(key) is not None


def test_lag_high_is_sink_only_and_per_consumer():
    expr = _rules()["LLMRouterKafkaConsumerLagHigh"]["expr"]

    excluded = _excluded_topic_keys(expr)
    assert all(excluded(key) for key in OBSERVER_KEYS)
    assert not any(excluded(key) for key in SINK_KEYS)
    assert re.search(r"by\s*\(\s*topic\s*,\s*topic_key\s*\)", expr), expr


def test_stall_rule_comment_names_every_observer_key_form():
    text = RULES_PATH.read_text()
    block = text[text.index("alert: LLMRouterKafkaConsumerStalled") :]
    block = block[: block.index("expr:")]
    assert "requests_enriched" in block


def _lag(topic, partition, topic_key):
    return metric_value_for_labels(
        PIPELINE_METRICS.consumer_lag,
        {"topic": topic, "partition": str(partition), "topic_key": topic_key},
    )


def _lag_series(topic_key):
    return [
        sample
        for metric in PIPELINE_METRICS.consumer_lag.collect()
        for sample in metric.samples
        if sample.labels.get("topic_key") == topic_key
    ]


@pytest.mark.asyncio
async def test_revoked_partitions_drop_their_lag_series():
    manager = KafkaConsumerManager({"topics": {"queries": "lagr.q"}}, MagicMock())
    consumer = SimpleNamespace(highwater=lambda _tp: 10)
    for partition in (0, 1):
        manager._record_successful_consume(
            "lagr_queries",
            consumer,
            SimpleNamespace(topic="lagr.q", partition=partition, offset=3),
        )
    listener = _BatchBackpressureRebalanceListener(manager, "lagr_queries", consumer)

    await listener.on_partitions_revoked({TopicPartition("lagr.q", 0)})

    remaining = {s.labels["partition"] for s in _lag_series("lagr_queries")}
    assert remaining == {"1"}
    # Revoking a partition without a series is harmless.
    await listener.on_partitions_revoked({TopicPartition("lagr.q", 7)})


@pytest.mark.asyncio
async def test_observer_consumers_get_the_rebalance_listener(monkeypatch):
    listeners = []

    class _Consumer:
        def __init__(self, **_kwargs):
            pass

        def subscribe(self, topics, listener=None):
            listeners.append(listener)

        async def start(self):
            return None

    monkeypatch.setattr("src.llm_router_part3_pipeline.AIOKafkaConsumer", _Consumer)
    manager = KafkaConsumerManager({"topics": {"alerts": "alerts"}}, MagicMock())
    manager.register_observer("alerts", lambda event: None)

    await manager._start_observer_consumers()

    (listener,) = listeners
    assert isinstance(listener, _BatchBackpressureRebalanceListener)
    assert listener.topic_key == "alerts_observer"


class _LagConsumer:
    """Assigned partitions with a broker end offset and a stuck position."""

    def __init__(self, ends, positions):
        self.ends = ends
        self.positions = positions

    def assignment(self):
        return set(self.ends)

    async def end_offsets(self, partitions):
        return {tp: self.ends[tp] for tp in partitions}

    async def position(self, tp):
        return self.positions[tp]


@pytest.mark.asyncio
async def test_lag_refresh_shows_a_hung_consumer_falling_behind():
    manager = KafkaConsumerManager({"topics": {}}, MagicMock())
    tp = TopicPartition("lagh.q", 0)
    consumer = _LagConsumer({tp: 10}, {tp: 10})
    manager.consumers["lagh_queries"] = consumer

    await manager._refresh_consumer_lag()
    assert _lag("lagh.q", 0, "lagh_queries") == 0

    consumer.ends[tp] = 25  # new records; the consumer never moved
    await manager._refresh_consumer_lag()
    assert _lag("lagh.q", 0, "lagh_queries") == 15


@pytest.mark.asyncio
async def test_lag_refresh_skips_failing_consumers():
    manager = KafkaConsumerManager({"topics": {}}, MagicMock())
    tp = TopicPartition("lagf.q", 0)

    class _Broken(_LagConsumer):
        async def end_offsets(self, partitions):
            raise RuntimeError("broker down")

    manager.consumers["lagf_broken"] = _Broken({tp: 5}, {tp: 0})
    manager.consumers["lagf_ok"] = _LagConsumer(
        {TopicPartition("lagf.ok", 0): 4}, {TopicPartition("lagf.ok", 0): 1}
    )
    manager.consumers["lagf_fake"] = SimpleNamespace()  # no assignment()

    await manager._refresh_consumer_lag()

    assert _lag("lagf.ok", 0, "lagf_ok") == 3
    assert _lag_series("lagf_broken") == []


def test_lag_refresh_interval_is_configurable():
    manager = KafkaConsumerManager(
        {"consumer": {"lag_refresh_interval_seconds": 7}}, MagicMock()
    )
    assert manager.lag_refresh_interval_seconds == 7
    assert KafkaConsumerManager({}, MagicMock()).lag_refresh_interval_seconds == 30
