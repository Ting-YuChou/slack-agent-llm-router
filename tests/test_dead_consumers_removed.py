"""The llm-responses and llm-errors topics had no producer callers and no
consumer work; they are gone from the code and the shipped configs."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src import kafka_topics
from src.llm_router_part3_pipeline import (
    KafkaConsumerManager,
    KafkaIngestionPipeline,
    KafkaProducerManager,
)

ROOT = Path(__file__).resolve().parents[1]
DEAD_KEYS = ("responses", "errors")


def test_dead_producers_and_consumers_are_removed():
    for name in ("produce_response_log", "produce_error"):
        assert not hasattr(KafkaProducerManager, name)
    for name in ("_consume_responses", "_consume_errors"):
        assert not hasattr(KafkaConsumerManager, name)
    assert not hasattr(KafkaIngestionPipeline, "log_error")


@pytest.mark.parametrize("config_name", ["config.yaml", "config.compose.yaml"])
def test_shipped_configs_do_not_provision_dead_topics(config_name):
    config = yaml.safe_load((ROOT / "config" / config_name).read_text("utf-8"))
    kafka = config["kafka"]

    for key in DEAD_KEYS:
        assert key not in kafka["topics"]
    names = {spec.name for spec in kafka_topics.topic_specs(kafka)}
    assert not names & {"llm-responses", "llm-errors"}
    assert not any(name.startswith(("llm-responses", "llm-errors")) for name in names)


@pytest.mark.asyncio
async def test_initialize_ignores_leftover_dead_topic_keys(monkeypatch):
    created = []

    class _Consumer:
        def __init__(self, **_kwargs):
            pass

        def subscribe(self, topics, listener=None):
            created.extend(topics)

        async def start(self):
            return None

    monkeypatch.setattr(
        "src.llm_router_part3_pipeline.AIOKafkaConsumer", _Consumer, raising=True
    )
    manager = KafkaConsumerManager(
        {
            "topics": {
                "queries": "q",
                "responses": "old-responses",
                "errors": "old-errors",
            },
            "dlq": {"enabled": False},
        },
        MagicMock(),
    )

    await manager.initialize()

    assert set(manager.consumers) == {"queries"}
    assert created == ["q"]
