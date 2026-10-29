"""Round-2 review: no payload values in DLQ records or logs (A7)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src import llm_router_part3_policy as policy
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache

SECRET = "patient-record-4471"


class _Consumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


class _DlqProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, key=None, value=None):
        self.sent.append(value)


@pytest.mark.asyncio
async def test_dlq_record_and_log_keep_only_the_field_and_error_type(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        policy.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    cache = RoutingPolicyCache(
        {"enabled": False, "consumer": {"enable_auto_commit": False}}
    )
    materializer = PolicyMaterializer({"topics": {}}, cache)
    materializer.running = True
    materializer.dlq_producer = _DlqProducer()
    message = SimpleNamespace(
        value={"scope_type": "user", "scope_key": "u1", "recent_error_rate": SECRET},
        topic="routing.policy_state",
        partition=0,
        offset=1,
    )

    await materializer._consume_routing_policy_state(_Consumer([message]))

    (record,) = materializer.dlq_producer.sent
    assert SECRET not in record["error_message"]
    assert "recent_error_rate" in record["error_message"]
    assert record["error_type"] == "PolicyEventInvalid"
    logged = " ".join(str(part) for args in warnings for part in args)
    assert "recent_error_rate" in logged
    assert SECRET not in logged
