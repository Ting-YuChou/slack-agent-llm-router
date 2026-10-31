"""O2 round-2 review: a restart drops the old consumer's lag series, and a
consumer that never consumed still has a last-consumed series for the stall
rule to join."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part3_pipeline import KafkaConsumerManager
from src.utils.metrics import PIPELINE_METRICS, metric_value_for_labels


def _manager(topics):
    return KafkaConsumerManager(
        {
            "topics": topics,
            "consumer": {"group_id": "sink-group", "enable_auto_commit": False},
            "dlq": {"enabled": False},
        },
        MagicMock(),
    )


# --- 2: a restart drops the old consumer's lag series ----------------------


def _lag_partitions(topic_key):
    return {
        sample.labels["partition"]
        for metric in PIPELINE_METRICS.consumer_lag.collect()
        for sample in metric.samples
        if sample.labels.get("topic_key") == topic_key
    }


@pytest.mark.asyncio
async def test_restart_drops_the_old_consumers_lag_series(monkeypatch):
    class _Consumer:
        def __init__(self, **_kwargs):
            pass

        def subscribe(self, topics, listener=None):
            pass

        async def start(self):
            return None

    monkeypatch.setattr("src.llm_router_part3_pipeline.AIOKafkaConsumer", _Consumer)
    manager = _manager({"queries": "r2.restart"})
    old = SimpleNamespace(highwater=lambda _tp: 9, stop=AsyncMock())
    for partition in (0, 1):
        manager._record_successful_consume(
            "queries",
            old,
            SimpleNamespace(topic="r2.restart", partition=partition, offset=2),
        )
    manager.consumers["queries"] = old
    manager.topic_names_by_key["queries"] = "r2.restart"
    assert _lag_partitions("queries") >= {"0", "1"}

    await manager._restart_consumer("queries")

    old.stop.assert_awaited_once()
    assert _lag_partitions("queries") == set()


# --- 3: a consumer that never consumed still has a last-consumed series -----


def _last_consumed(topic, topic_key):
    return metric_value_for_labels(
        PIPELINE_METRICS.last_consumed_timestamp,
        {"topic": topic, "topic_key": topic_key},
    )


@pytest.mark.asyncio
async def test_consumer_start_seeds_last_consumed(monkeypatch):
    class _Consumer:
        def __init__(self, **_kwargs):
            pass

        def subscribe(self, topics, listener=None):
            pass

        async def start(self):
            return None

    monkeypatch.setattr("src.llm_router_part3_pipeline.AIOKafkaConsumer", _Consumer)
    manager = _manager({"queries": "r2.seed.q", "alerts": "r2.seed.a"})
    before = time.time()

    await manager.initialize()
    manager.register_observer("alerts", lambda event: None)
    await manager._start_observer_consumers()

    for topic, key in (
        ("r2.seed.q", "queries"),
        ("r2.seed.a", "alerts"),
        ("r2.seed.a", "alerts_observer"),
    ):
        assert _last_consumed(topic, key) >= before, key
    # Seeding is not a consume: health still reports nothing consumed.
    assert manager.last_consumed_at == {}
