"""Round-2 review fixes for A10 alerting."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from flink.logic import build_routing_guardrail_event
from src.llm_router_part4_monitor import (
    STALE_ALERT_EVENTS,
    AlertManager,
    AlertRule,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = (
    REPO_ROOT / "docker" / "observability" / "prometheus" / "llm-router-alerts.yml"
)
START = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


class _RecordingHandler:
    def __init__(self):
        self.calls = []

    async def send_alert(self, payload):
        self.calls.append(payload)


def _manager(config=None, **rule_overrides):
    manager = AlertManager(config or {})
    values = dict(
        name="high_error_rate",
        condition="error_rate > threshold",
        threshold=0.05,
        duration=300,
        severity="critical",
        description="Error rate too high",
    )
    values.update(rule_overrides)
    manager.alert_rules = [AlertRule(**values)]
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    return manager, handler


async def _at(manager, seconds, error_rate):
    await manager.check_alerts(
        {"error_rate": error_rate}, now=START + timedelta(seconds=seconds)
    )


# --- 1: no rule joins the logical-key and topic-name label spaces ---


def test_no_rule_joins_consumed_timestamp_with_lag():
    document = yaml.safe_load(RULES_PATH.read_text())
    names = [r["alert"] for g in document["groups"] for r in g["rules"]]
    assert "LLMRouterKafkaConsumerStalled" not in names
    for group in document["groups"]:
        for rule in group["rules"]:
            expr = rule["expr"]
            assert not (
                "llm_router_kafka_last_consumed_timestamp_seconds" in expr
                and "llm_router_kafka_consumer_lag" in expr
            ), rule["alert"]


# --- 2: resolve hysteresis ---


@pytest.mark.asyncio
async def test_flapping_metric_fires_once_and_never_resolves():
    # duration stays the re-notify cooldown, so keep the run inside it; each
    # 60s recovery is shorter than resolve_after_seconds.
    manager, handler = _manager(duration=3600, resolve_after_seconds=120)

    for minute in range(20):
        await _at(manager, minute * 60, 0.2 if minute % 2 == 0 else 0.0)

    assert [c["status"] for c in handler.calls] == ["firing"]


@pytest.mark.asyncio
async def test_sustained_recovery_resolves_then_new_breach_notifies():
    manager, handler = _manager()

    await _at(manager, 0, 0.2)
    await _at(manager, 60, 0.0)
    await _at(manager, 200, 0.0)
    assert [c["status"] for c in handler.calls] == ["firing"]

    await _at(manager, 361, 0.0)  # false for 301s >= duration (300)
    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]

    await _at(manager, 400, 0.2)
    assert [c["status"] for c in handler.calls] == ["firing", "resolved", "firing"]


@pytest.mark.asyncio
async def test_resolve_after_seconds_is_configurable():
    manager = AlertManager({"alerts": {"resolve_after_seconds": 30}})
    assert all(rule.resolve_after_seconds == 30 for rule in manager.alert_rules)

    manager, handler = _manager(resolve_after_seconds=30)
    await _at(manager, 0, 0.2)
    await _at(manager, 10, 0.0)
    await _at(manager, 41, 0.0)
    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]


@pytest.mark.asyncio
async def test_resolve_after_defaults_to_300_without_duration():
    manager, handler = _manager(duration=0)
    await _at(manager, 0, 0.2)
    await _at(manager, 1, 0.0)
    await _at(manager, 299, 0.0)
    assert [c["status"] for c in handler.calls] == ["firing"]
    await _at(manager, 302, 0.0)
    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]


# --- 3: a clear resolves avoid incidents and its own trigger only ---


def _event(trigger_type, action, window_start_ms, severity=None):
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
        description=trigger_type,
        guardrail_action=action,
        severity=severity or ("error" if action == "avoid" else "warning"),
    )


@pytest.mark.asyncio
async def test_clear_leaves_warn_only_incidents_open():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_event("high_error_rate", "avoid", 0))
    await manager.record_external_alert(_event("high_latency", "avoid", 0))
    await manager.record_external_alert(_event("cost_spike", "warn", 0))
    await manager.record_external_alert(_event("throughput_drop", "warn", 0))
    await manager.record_external_alert(
        _event("high_error_rate", "clear", 300_000, severity="info")
    )

    resolved = sorted(
        c["rule_name"] for c in handler.calls if c["status"] == "resolved"
    )
    assert resolved == ["high_error_rate", "high_latency"]

    # The warn incident is still active, so its next window stays quiet.
    before = len(handler.calls)
    await manager.record_external_alert(_event("cost_spike", "warn", 60_000))
    assert len(handler.calls) == before


@pytest.mark.asyncio
async def test_clear_resolves_its_own_trigger_even_when_it_only_warned():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_event("high_error_rate", "warn", 0))
    await manager.record_external_alert(
        _event("high_error_rate", "clear", 300_000, severity="info")
    )

    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]


# --- 4: late clears still resolve; stale firings are counted and logged ---


def _old_ms(hours):
    then = datetime.now(timezone.utc) - timedelta(hours=hours)
    return int(then.timestamp() * 1000)


@pytest.mark.asyncio
async def test_late_clear_still_resolves():
    manager = AlertManager({"alerts": {"max_event_age_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    fresh = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    await manager.record_external_alert(
        {**_event("high_error_rate", "avoid", _old_ms(0)), "emitted_at": fresh}
    )
    await manager.record_external_alert(
        {
            **_event("high_error_rate", "clear", _old_ms(2), severity="info"),
            "emitted_at": old,
        }
    )

    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]


@pytest.mark.asyncio
async def test_stale_firing_events_are_counted_and_logged(caplog):
    manager = AlertManager({"alerts": {"max_event_age_seconds": 900}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    before = STALE_ALERT_EVENTS._value.get()

    with caplog.at_level(logging.WARNING, logger="src.llm_router_part4_monitor"):
        for index in range(5):
            await manager.record_external_alert(
                {
                    **_event("high_error_rate", "avoid", index * 60_000),
                    "emitted_at": old,
                }
            )

    assert handler.calls == []
    assert manager.stale_event_count == 5
    assert STALE_ALERT_EVENTS._value.get() - before == 5
    stale_logs = [r for r in caplog.records if "stale" in r.getMessage().lower()]
    assert len(stale_logs) == 1


# --- Cross-branch: avoid(W) -> clear(W) -> avoid(W) pages again ---


@pytest.mark.asyncio
async def test_avoid_after_retraction_clear_in_same_window_notifies():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    window = 1_710_000_000_000

    await manager.record_external_alert(_event("provider_failure", "avoid", window))
    await manager.record_external_alert(
        _event("provider_failure", "clear", window, severity="info")
    )
    await manager.record_external_alert(_event("provider_failure", "avoid", window))

    assert [c["status"] for c in handler.calls] == ["firing", "resolved", "firing"]
