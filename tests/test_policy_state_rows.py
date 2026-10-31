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
    # emitted_at changes on replay, so it is not part of the id.
    assert row["event_id"].startswith("policy_state:user:U1::")
    assert "2026-10-30T12:00:00" not in row["event_id"]


@pytest.mark.parametrize("latest_event_at", [None, "not-a-time"])
def test_a_replay_without_a_usable_latest_event_at_keeps_its_id(latest_event_at):
    first = _row(_state(latest_event_at=latest_event_at), offset=10)
    replay = _row(
        _state(
            latest_event_at=latest_event_at,
            emitted_at="2026-10-30T13:30:00+00:00",
        ),
        offset=99,
    )

    assert replay["event_id"] == first["event_id"]
    assert first["event_id"].startswith("policy_state:user:U1::")


def test_an_unparseable_latest_event_at_falls_back_to_emitted_at():
    row = _row(_state(latest_event_at="not-a-time"), offset=10)

    assert row["timestamp"].isoformat() == "2026-10-30T12:00:00+00:00"


@pytest.mark.asyncio
async def test_policy_state_query_filters_on_emitted_at():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=SimpleNamespace(result_rows=[]))

    await manager.get_routing_policy_state_events(hours=6, raise_errors=True)

    sql = " ".join(manager._run_blocking.await_args.args[1].split())
    # timestamp is the state's event time; a state emitted now for a scope
    # with old events is still recent.
    assert "WHERE emitted_at >= now() - INTERVAL {hours:UInt32} HOUR" in sql


async def _policy_state_query(hours=6):
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=SimpleNamespace(result_rows=[]))
    await manager.get_routing_policy_state_events(hours=hours, raise_errors=True)
    call = manager._run_blocking.await_args
    return " ".join(call.args[1].split()), call.kwargs["parameters"]


@pytest.mark.asyncio
async def test_policy_state_query_keeps_a_coarse_event_time_bound():
    sql, parameters = await _policy_state_query(hours=6)

    # Partition pruning on the sorting key: a state covers events at most
    # POLICY_STATE_MAX_EVENT_LAG_DAYS older than when it was emitted.
    assert (
        "AND timestamp >= now() - INTERVAL {hours:UInt32} HOUR - INTERVAL 7 DAY"
    ) in sql
    assert parameters == {"hours": 6}


@pytest.mark.asyncio
async def test_policy_state_query_lists_the_most_recently_emitted_first():
    sql, _ = await _policy_state_query()

    # A fresh state for a quiet scope (old event time) is not pushed past
    # LIMIT 50 by busier scopes.
    assert "ORDER BY emitted_at DESC, timestamp DESC, kafka_offset DESC" in sql


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
async def test_latest_guardrail_query_breaks_timestamp_ties_by_emitted_at():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=SimpleNamespace(result_rows=[]))

    await manager.get_routing_guardrails(hours=6, raise_errors=True)

    sql = manager._run_blocking.await_args.args[1]
    assert "ORDER BY timestamp DESC, emitted_at DESC" in sql
    # One Flink call can emit a clear and an avoid with one emitted_at; the
    # later offset is the later event.
    assert "ORDER BY timestamp DESC, emitted_at DESC, kafka_offset DESC" in sql


def test_emitted_at_has_the_same_default_in_fresh_and_migrated_tables():
    from src.analytics_schema import TABLES

    for table in TABLES:
        if table.name not in {"alert_events", "routing_policy_state_events"}:
            continue
        (create_line,) = [
            line.strip().rstrip(",")
            for line in table.create_sql("db").splitlines()
            if line.strip().startswith("emitted_at ")
        ]
        (alter,) = [sql for sql in table.alter_sql("db") if " emitted_at " in sql]
        assert create_line == "emitted_at DateTime64(3) DEFAULT timestamp"
        assert alter.endswith(create_line)


def test_alert_entries_built_directly_get_an_emitted_at():
    from src.llm_router_part3_pipeline import AlertEventEntry, RoutingPolicyStateEntry

    before = datetime.now(timezone.utc)
    stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    alert = AlertEventEntry(
        timestamp=stamp, alert_type="a", severity="s", description="d"
    )
    state = RoutingPolicyStateEntry(timestamp=stamp, scope_type="u", scope_key="k")

    assert alert.emitted_at >= before
    assert state.emitted_at >= before
    assert alert.to_dict()["emitted_at"] == alert.emitted_at
