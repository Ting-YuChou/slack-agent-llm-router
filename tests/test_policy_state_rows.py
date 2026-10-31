"""D5: a replayed routing.policy_state event lands on the same ClickHouse row;
D6: alert rows carry emitted_at so the latest queries can order same-window
rows."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part3_pipeline import ClickHouseManager, KafkaConsumerManager

LATEST = "2026-10-30T11:59:30+00:00"


def _consumer():
    return KafkaConsumerManager.__new__(KafkaConsumerManager)


def _state(**overrides):
    payload = {
        "event_type": "routing.policy_state",
        "event_version": 1,
        "emitted_at": "2026-10-30T12:00:00+00:00",
        "latest_event_at": LATEST,
        "scope_type": "user",
        "scope_key": "U1",
        "user_id": "U1",
        "recent_request_count": 4,
        "recent_error_rate": 0.25,
        "avoid_models": ["m1"],
    }
    payload.update(overrides)
    return payload


def _row(payload, offset):
    consumer = _consumer()
    message = SimpleNamespace(topic="routing.policy_state", partition=0, offset=offset)
    return consumer._build_routing_policy_state_entry(
        payload, kafka_metadata=consumer._kafka_event_metadata(message)
    ).to_dict()


def test_a_replay_keeps_the_row_id_and_timestamp():
    first = _row(_state(), offset=10)
    replay = _row(_state(emitted_at="2026-10-30T13:30:00+00:00"), offset=99)

    assert replay["event_id"] == first["event_id"]
    assert replay["timestamp"] == first["timestamp"]
    assert first["timestamp"].isoformat() == LATEST
    assert first["event_id"].startswith(f"policy_state:user:U1:{LATEST}:")
    # The row still records where it was read from.
    assert replay["kafka_offset"] == 99


def test_a_different_state_at_the_same_event_time_keeps_its_own_row():
    first = _row(_state(), offset=10)
    expired = _row(_state(recent_request_count=3), offset=11)

    assert expired["event_id"] != first["event_id"]
    assert expired["timestamp"] == first["timestamp"]


def test_a_state_without_latest_event_at_falls_back_to_emitted_at():
    row = _row(_state(latest_event_at=None), offset=10)

    assert row["timestamp"].isoformat() == "2026-10-30T12:00:00+00:00"
    assert row["event_id"].startswith("policy_state:user:U1:2026-10-30T12:00:00+00:00:")


# D6: avoid and clear of one window have the same timestamp; emitted_at
# orders them.


def _guardrail(action, emitted_at):
    return {
        "event_type": "routing.guardrails",
        "emitted_at": emitted_at,
        "scope_type": "model",
        "scope_key": "m1",
        "trigger_type": "high_error_rate",
        "guardrail_action": action,
        "window_start_ms": 1_776_556_800_000,
        "window_end_ms": 1_776_556_860_000,
    }


def test_avoid_and_clear_of_one_window_carry_their_emitted_at():
    consumer = _consumer()
    avoid = consumer._build_alert_event_entry(
        _guardrail("avoid", "2026-10-30T12:00:01+00:00")
    ).to_dict()
    clear = consumer._build_alert_event_entry(
        _guardrail("clear", "2026-10-30T12:00:02+00:00")
    ).to_dict()

    assert avoid["timestamp"] == clear["timestamp"]
    assert clear["emitted_at"] > avoid["emitted_at"]
    assert clear["emitted_at"].isoformat() == "2026-10-30T12:00:02+00:00"


def test_an_alert_without_emitted_at_gets_its_ingest_time():
    before = datetime.now(timezone.utc)
    entry = _consumer()._build_alert_event_entry(_guardrail("avoid", None))

    assert entry.to_dict()["emitted_at"] >= before


def test_a_policy_state_row_carries_its_emitted_at():
    row = _row(_state(), offset=10)

    assert row["emitted_at"].isoformat() == "2026-10-30T12:00:00+00:00"


@pytest.mark.asyncio
async def test_alert_and_policy_state_tables_get_the_emitted_at_column():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [call.args[1] for call in manager._run_blocking.await_args_list]
    for table in ("alert_events", "routing_policy_state_events"):
        assert (
            f"ALTER TABLE llm_router.{table} ADD COLUMN IF NOT EXISTS "
            "emitted_at DateTime64(3) DEFAULT timestamp"
        ) in statements
    # The engine and sorting key stay as they are.
    (alerts,) = [s for s in statements if "llm_router.alert_events (" in s]
    assert "ENGINE = ReplacingMergeTree()" in alerts
    assert "ORDER BY (source_event_type, timestamp, event_id)" in alerts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["get_routing_guardrails", "get_routing_policy_state_events"]
)
async def test_latest_queries_break_timestamp_ties_by_emitted_at(method):
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=SimpleNamespace(result_rows=[]))

    await getattr(manager, method)(hours=6, raise_errors=True)

    sql = manager._run_blocking.await_args.args[1]
    assert "ORDER BY timestamp DESC, emitted_at DESC" in sql
