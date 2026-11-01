"""A12b review: version parsing never raises, rejections are value-free, and
the supported major follows the stamped schema version."""

import json
import logging
from types import SimpleNamespace

import pytest

from flink import analytics_job
from flink import logic as flink_logic
from src import event_version
from src import llm_router_part3_pipeline as pipeline
from src.event_version import UnsupportedEventVersionError, is_supported_event_version
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache


@pytest.mark.parametrize(
    "value",
    [
        # superscript two: str.isdigit() is True, int() raises
        pytest.param("1²", id="superscript"),
        # Arabic-Indic one: a Unicode decimal digit
        pytest.param("١", id="arabic-indic"),
        # past int()'s default digit limit
        pytest.param("1" * 5000, id="5000-digits"),
        pytest.param("9" * 40, id="40-digits"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_odd_versions_are_rejected_without_raising(value):
    assert is_supported_event_version(value) is False


def test_analytics_mapper_survives_an_odd_version():
    mapper = analytics_job.InferenceCompletedMapper()

    result = mapper.map(json.dumps({"event_version": "1²", "request_id": "r"}))

    assert result == {"event_type": flink_logic.DROPPED_EVENT}


def test_rejection_message_carries_no_version_value():
    secret = "sk-secretvalue-1234567890"
    error = UnsupportedEventVersionError(secret)

    assert secret not in str(error)
    assert "secretvalue" not in str(error)
    assert str(error).startswith("unsupported_event_version")


class _DlqProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, key=None, value=None):
        self.sent.append((topic, value))


class _Consumer:
    def __init__(self, messages):
        self._messages = list(messages)

        async def commit(*_args, **_kwargs):
            return None

        self.commit = commit

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


@pytest.mark.asyncio
async def test_materializer_dlq_message_carries_no_version_value():
    cache = RoutingPolicyCache(
        {"enabled": False, "consumer": {"enable_auto_commit": False}}
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    materializer.dlq_producer = _DlqProducer()
    secret = "private-marker-7781"
    message = SimpleNamespace(
        value={"scope_type": "user", "scope_key": "u", "event_version": secret},
        topic="routing.policy_state",
        partition=0,
        offset=1,
    )

    await materializer._consume_routing_policy_state(_Consumer([message]))

    ((_topic, payload),) = materializer.dlq_producer.sent
    assert "private-marker-7781" not in json.dumps(payload)
    assert payload["failure_reason"] == "unsupported_event_version"


def test_flink_drop_log_carries_no_version_value(caplog):
    mapper = analytics_job.InferenceCompletedMapper()
    secret = "2.0-sk-abcdefghijklmnop"

    with caplog.at_level(logging.WARNING):
        mapper.map(json.dumps({"event_version": secret, "request_id": "r"}))

    assert mapper.dropped_unsupported_version == 1
    assert "sk-abcdefghijklmnop" not in caplog.text


def test_supported_major_follows_the_stamped_schema_version():
    major = int(event_version.EVENT_SCHEMA_VERSION.split(".")[0])

    assert event_version.SUPPORTED_EVENT_MAJOR_VERSION == major
    assert pipeline.EVENT_SCHEMA_VERSION == event_version.EVENT_SCHEMA_VERSION
    assert flink_logic.EVENT_SCHEMA_VERSION == event_version.EVENT_SCHEMA_VERSION
    assert is_supported_event_version(event_version.EVENT_SCHEMA_VERSION)
