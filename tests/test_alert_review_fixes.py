"""Round-1 review fixes for A10 alerting."""

from datetime import datetime, timedelta, timezone

import pytest

from flink.logic import build_routing_guardrail_event
from src.llm_router_part4_monitor import AlertManager, AlertRule, MonitoringService


class _RecordingHandler:
    def __init__(self):
        self.calls = []

    async def send_alert(self, payload):
        self.calls.append(payload)


# --- A10 review 3 and 4: rule state machine ---


def _rule(**overrides):
    values = dict(
        name="high_error_rate",
        condition="error_rate > threshold",
        threshold=0.05,
        duration=3600,
        severity="critical",
        description="Error rate too high",
    )
    values.update(overrides)
    return AlertRule(**values)


@pytest.mark.asyncio
async def test_rebreach_after_resolve_notifies_again():
    manager = AlertManager({})
    # Round 2 added resolve hysteresis; resolve on the first healthy check.
    manager.alert_rules = [_rule(resolve_after_seconds=0)]
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    start = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)

    await manager.check_alerts({"error_rate": 0.2}, now=start)
    await manager.check_alerts({"error_rate": 0.0}, now=start + timedelta(minutes=1))
    await manager.check_alerts({"error_rate": 0.2}, now=start + timedelta(minutes=2))

    assert [c["status"] for c in handler.calls] == ["firing", "resolved", "firing"]


@pytest.mark.asyncio
async def test_evaluation_error_does_not_resolve_or_reset_pending():
    manager = AlertManager({})
    manager.alert_rules = [_rule(pending_seconds=60)]
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    start = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)

    await manager.check_alerts({"error_rate": 0.2}, now=start)
    await manager.check_alerts({"error_rate": 0.2}, now=start + timedelta(seconds=61))
    assert [c["status"] for c in handler.calls] == ["firing"]

    # A value the condition cannot compare raises inside evaluation.
    await manager.check_alerts({"error_rate": "n/a"}, now=start + timedelta(seconds=90))
    assert [c["status"] for c in handler.calls] == ["firing"]
    assert manager.alert_rules[0].firing is True

    # Pending progress also survives an unknown evaluation.
    rule = _rule(name="other", pending_seconds=60)
    manager.alert_rules = [rule]
    await manager.check_alerts({"error_rate": 0.2}, now=start)
    await manager.check_alerts({"error_rate": "n/a"}, now=start + timedelta(seconds=30))
    assert rule.pending_since == start


def test_evaluate_condition_still_returns_bool_on_error():
    manager = AlertManager({})
    assert manager._evaluate_condition(_rule(), {"error_rate": "n/a"}) is False


# --- A10 review 5: escalation inside one incident notifies ---


def _guardrail(**overrides):
    event = {
        "event_type": "routing.guardrails",
        "trigger_type": "high_latency",
        "severity": "warning",
        "description": "latency",
        "scope_type": "model",
        "scope_key": "gpt-5",
        "guardrail_action": "warn",
        "model_name": "gpt-5",
        "provider": "openai",
        "window_start_ms": 1_710_000_000_000,
        "window_end_ms": 1_710_000_060_000,
    }
    event.update(overrides)
    return event


@pytest.mark.asyncio
async def test_warn_to_avoid_escalation_notifies_within_cooldown():
    manager = AlertManager({"alerts": {"dedupe_cooldown_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_guardrail())
    await manager.record_external_alert(
        _guardrail(window_start_ms=1_710_000_060_000, window_end_ms=1_710_000_120_000)
    )
    await manager.record_external_alert(
        _guardrail(
            guardrail_action="avoid",
            severity="error",
            window_start_ms=1_710_000_120_000,
            window_end_ms=1_710_000_180_000,
        )
    )
    # Back to warn, or another avoid, inside the cooldown stays quiet.
    await manager.record_external_alert(
        _guardrail(
            guardrail_action="avoid",
            severity="error",
            window_start_ms=1_710_000_180_000,
            window_end_ms=1_710_000_240_000,
        )
    )

    assert [(c["guardrail_action"], c["severity"]) for c in handler.calls] == [
        ("warn", "warning"),
        ("avoid", "error"),
    ]


# --- A10 review 6: stale-stream alert is opt-in and waits for a first window ---


def test_stream_stale_rule_is_disabled_by_default():
    manager = AlertManager({})
    rule = next(r for r in manager.alert_rules if r.name == "stream_metrics_stale")
    assert rule.enabled is False


@pytest.mark.asyncio
async def test_enabled_stream_stale_does_not_fire_before_first_window():
    service = MonitoringService(
        {
            "stream_metrics_staleness_seconds": 300,
            "alerts": {"stream_metrics_stale_enabled": True},
        }
    )
    handler = _RecordingHandler()
    service.alert_manager.notification_handlers = {"test": handler}
    collector = service.metrics_collector
    collector._clock = lambda: collector._started_at + 10_000

    metrics = service._flatten_metrics(collector.stream_freshness_metrics())
    await service.alert_manager.check_alerts(metrics)

    assert handler.calls == []


# --- A10 review 7: suppressed incidents stay fresh in LRU order ---


@pytest.mark.asyncio
async def test_suppressed_incident_is_refreshed_in_lru():
    manager = AlertManager({"alerts": {"dedupe_max_entries": 2}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_guardrail(scope_key="a"))
    await manager.record_external_alert(_guardrail(scope_key="b"))
    # "a" repeats in a new window (suppressed), then "c" arrives.
    await manager.record_external_alert(
        _guardrail(scope_key="a", window_start_ms=1_710_000_060_000)
    )
    await manager.record_external_alert(_guardrail(scope_key="c"))
    # "a" is still active, so another new window of it stays quiet.
    await manager.record_external_alert(
        _guardrail(scope_key="a", window_start_ms=1_710_000_120_000)
    )

    assert [c["scope_key"] for c in handler.calls] == ["a", "b", "c"]


# --- A10 review 9: the policy track's clear event resolves the incident ---


@pytest.mark.asyncio
async def test_clear_event_shaped_like_flink_resolves_incident():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    metric_event = {
        "model_name": "gpt-5",
        "provider": "openai",
        "window_start_ms": 1_710_000_000_000,
        "window_end_ms": 1_710_000_060_000,
    }
    avoid = build_routing_guardrail_event(
        metric_event,
        scope_type="model",
        scope_key="gpt-5",
        trigger_type="high_error_rate",
        description="error rate",
        guardrail_action="avoid",
        severity="error",
    )
    later = {**metric_event, "window_start_ms": 1_710_000_300_000}
    clear = build_routing_guardrail_event(
        later,
        scope_type="model",
        scope_key="gpt-5",
        trigger_type="high_error_rate",
        description="Model gpt-5 recovered after 3 healthy windows",
        guardrail_action="clear",
        severity="info",
    )

    await manager.record_external_alert(avoid)
    await manager.record_external_alert(clear)

    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]
    assert handler.calls[-1]["scope_key"] == "gpt-5"


# --- Cross-branch A: the pending warn and the real avoid share one incident ---


def _flink_avoid(window_start_ms, trigger_type="high_error_rate"):
    return build_routing_guardrail_event(
        {
            "model_name": "gpt-5",
            "provider": "openai",
            "window_start_ms": window_start_ms,
            "window_end_ms": window_start_ms + 60_000,
        },
        scope_type="model",
        scope_key="gpt-5",
        trigger_type=trigger_type,
        description="error rate",
        guardrail_action="avoid",
        severity="error",
    )


@pytest.mark.asyncio
async def test_pending_warn_then_avoid_like_flink_notifies_both():
    manager = AlertManager({"alerts": {"dedupe_cooldown_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    # flink/logic.py downgrades a not-yet-entered avoid to a pending warn.
    pending = {
        **_flink_avoid(1_710_000_000_000),
        "guardrail_action": "warn",
        "severity": "warning",
        "hysteresis": "pending",
    }

    await manager.record_external_alert(pending)
    await manager.record_external_alert(_flink_avoid(1_710_000_060_000))
    await manager.record_external_alert(_flink_avoid(1_710_000_120_000))

    assert [c["guardrail_action"] for c in handler.calls] == ["warn", "avoid"]


# --- Cross-branch B: a clear resolves every incident of the scope ---


@pytest.mark.asyncio
async def test_clear_resolves_all_triggers_for_the_scope():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_flink_avoid(1_710_000_000_000))
    await manager.record_external_alert(
        _flink_avoid(1_710_000_000_000, trigger_type="high_latency")
    )
    await manager.record_external_alert(
        _guardrail(scope_key="claude-sonnet", trigger_type="high_latency")
    )
    clear = {
        **_flink_avoid(1_710_000_300_000),
        "guardrail_action": "clear",
        "severity": "info",
    }
    await manager.record_external_alert(clear)

    resolved = [
        (c["rule_name"], c["scope_key"])
        for c in handler.calls
        if c["status"] == "resolved"
    ]
    assert sorted(resolved) == [
        ("high_error_rate", "gpt-5"),
        ("high_latency", "gpt-5"),
    ]


# --- Cross-branch C: replayed old events are recorded but not notified ---


@pytest.mark.asyncio
async def test_events_older_than_max_age_are_recorded_without_notifying():
    manager = AlertManager({"alerts": {"max_event_age_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=2)).isoformat()
    fresh = (now - timedelta(seconds=30)).isoformat()

    await manager.record_external_alert(_guardrail(scope_key="a", emitted_at=old))
    await manager.record_external_alert(
        {
            "alert_type": "anomaly_detected",
            "anomaly_type": "high_volume",
            "model_name": "m",
            "emitted_at": old,
        }
    )
    await manager.record_external_alert(
        _guardrail(
            scope_key="a",
            emitted_at=fresh,
            window_start_ms=1_710_000_060_000,
        )
    )

    assert len(manager.alert_history) == 3
    assert [c["scope_key"] for c in handler.calls] == ["a"]


@pytest.mark.asyncio
async def test_window_end_marks_age_when_emitted_at_is_missing():
    manager = AlertManager({"alerts": {"max_event_age_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    old_ms = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)

    await manager.record_external_alert(
        _guardrail(window_start_ms=old_ms - 60_000, window_end_ms=old_ms)
    )

    assert handler.calls == []
    assert len(manager.alert_history) == 1


@pytest.mark.asyncio
async def test_max_event_age_is_off_by_default():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    await manager.record_external_alert(_guardrail(emitted_at="2026-04-08T12:00:00Z"))
    assert len(handler.calls) == 1
