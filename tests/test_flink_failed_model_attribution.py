"""Model failure attribution in the Flink routing policy.

A model is avoided for its own failures only: one the caller, policy or our
own code caused must not count against the model that was routed to.
"""

from datetime import datetime

import pytest

from flink import logic

_LEGACY = object()


def _failure(error_type=_LEGACY, **overrides):
    event = {
        "event_type": "inference.completed",
        "request_id": "request-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "user_tier": "premium",
        "query_type": "general",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "error",
        "latency_ms": 100,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    if error_type is not _LEGACY:
        event["error_type"] = error_type
    event.update(overrides)
    return event


@pytest.mark.parametrize(
    ("error_type", "counted"),
    [
        ("provider_error", True),
        ("timeout", True),
        ("rate_limited", True),
        ("user_or_policy", False),
        ("internal", False),
        (None, True),
        (_LEGACY, True),
    ],
)
def test_routing_policy_counts_only_model_attributable_failures(error_type, counted):
    aggregate = logic.add_routing_policy_event(
        logic._empty_routing_policy_aggregate(),
        _failure(error_type=error_type),
    )

    assert aggregate["failed_model_counts"] == ({"gpt-5": 1} if counted else {})
    assert aggregate["model_all_counts"] == {"gpt-5": 1}
    assert aggregate.get("model_success_counts", {}) == {}


@pytest.mark.parametrize(
    ("error_type", "avoided"),
    [
        ("provider_error", True),
        ("timeout", True),
        ("user_or_policy", False),
        ("internal", False),
        (_LEGACY, True),
    ],
)
def test_routing_policy_avoids_a_model_only_for_its_own_failures(error_type, avoided):
    policy = logic.build_routing_policy_state_event(
        scope_type="user",
        scope_key="user-1",
        events=[
            _failure(error_type=error_type, request_id=f"request-{index}")
            for index in range(3)
        ],
        timestamp=datetime(2026, 10, 28, 12, 0, 0),
    )

    assert ("gpt-5" in policy["avoid_models"]) is avoided
