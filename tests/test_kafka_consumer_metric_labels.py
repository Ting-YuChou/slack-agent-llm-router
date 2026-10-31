"""O2: the consumer lag and last-consumed gauges share a topic_key label."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.llm_router_part3_pipeline import KafkaConsumerManager
from src.utils.metrics import PIPELINE_METRICS, metric_value_for_labels


def _consumer(highwater):
    return SimpleNamespace(highwater=lambda _tp: highwater)


def _message(offset):
    return SimpleNamespace(topic="o2.alerts", partition=1, offset=offset)


def test_label_sets_carry_topic_key():
    assert "topic_key" in PIPELINE_METRICS.consumer_lag._labelnames
    assert {"topic", "partition"} <= set(PIPELINE_METRICS.consumer_lag._labelnames)
    assert set(PIPELINE_METRICS.last_consumed_timestamp._labelnames) == {
        "topic",
        "topic_key",
    }


def test_successful_consume_labels_lag_and_timestamp_with_topic_key():
    manager = KafkaConsumerManager({"topics": {"alerts": "o2.alerts"}}, MagicMock())

    manager._record_successful_consume("alerts", _consumer(10), _message(4))

    assert (
        metric_value_for_labels(
            PIPELINE_METRICS.consumer_lag,
            {"topic": "o2.alerts", "partition": "1", "topic_key": "alerts"},
        )
        == 5
    )
    assert (
        metric_value_for_labels(
            PIPELINE_METRICS.last_consumed_timestamp,
            {"topic": "o2.alerts", "topic_key": "alerts"},
        )
        > 0
    )


def test_observer_and_sink_lag_do_not_overwrite_each_other():
    manager = KafkaConsumerManager({"topics": {"alerts": "o2.alerts"}}, MagicMock())

    manager._record_successful_consume("alerts", _consumer(10), _message(8))
    manager._record_successful_consume("alerts_observer", _consumer(10), _message(2))

    def lag(topic_key):
        return metric_value_for_labels(
            PIPELINE_METRICS.consumer_lag,
            {"topic": "o2.alerts", "partition": "1", "topic_key": topic_key},
        )

    assert (lag("alerts"), lag("alerts_observer")) == (1, 7)
