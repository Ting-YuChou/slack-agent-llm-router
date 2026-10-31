"""D5: a replayed routing.policy_state event lands on the same ClickHouse row;
D6: alert rows carry emitted_at so the latest queries can order same-window
rows."""

from types import SimpleNamespace

from src.llm_router_part3_pipeline import KafkaConsumerManager

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
