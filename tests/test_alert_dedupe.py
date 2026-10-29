"""A10: chat alerting dedupes, sends in the background, and alerts on silence."""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from src.llm_router_part4_monitor import (
    AlertManager,
    AlertRule,
    MetricsCollector,
    MonitoringService,
    SlackNotificationHandler,
    WebhookNotificationHandler,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = (
    REPO_ROOT / "docker" / "observability" / "prometheus" / "llm-router-alerts.yml"
)


class _RecordingHandler:
    def __init__(self):
        self.calls = []

    async def send_alert(self, payload):
        self.calls.append(payload)


def _guardrail(**overrides):
    event = {
        "event_type": "routing.guardrails",
        "trigger_type": "high_error_rate",
        "severity": "error",
        "description": "Model error rate exceeded the guardrail threshold",
        "scope_type": "model",
        "scope_key": "gpt-5",
        "guardrail_action": "avoid",
        "model_name": "gpt-5",
        "provider": "openai",
        "window_start_ms": 1_710_000_000_000,
        "window_end_ms": 1_710_000_060_000,
        "emitted_at": "2026-04-08T12:00:00Z",
    }
    event.update(overrides)
    return event


@pytest.mark.asyncio
async def test_identical_guardrails_notify_once():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    for _ in range(100):
        await manager.record_external_alert(_guardrail())

    assert len(handler.calls) == 1
    assert len(manager.alert_history) == 1
    assert manager.suppressed_alert_count == 99


@pytest.mark.asyncio
async def test_new_window_of_same_guardrail_is_suppressed_during_cooldown():
    manager = AlertManager({"alerts": {"dedupe_cooldown_seconds": 600}})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_guardrail())
    await manager.record_external_alert(
        _guardrail(window_start_ms=1_710_000_060_000, window_end_ms=1_710_000_120_000)
    )
    await manager.record_external_alert(_guardrail(scope_key="claude-sonnet"))

    assert [call["scope_key"] for call in handler.calls] == ["gpt-5", "claude-sonnet"]


@pytest.mark.asyncio
async def test_guardrail_clear_resolves_instead_of_alerting():
    manager = AlertManager({})
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}

    await manager.record_external_alert(_guardrail(guardrail_action="clear"))
    assert handler.calls == []

    await manager.record_external_alert(_guardrail())
    await manager.record_external_alert(
        _guardrail(guardrail_action="clear", window_start_ms=1_710_000_120_000)
    )
    await manager.record_external_alert(
        _guardrail(guardrail_action="clear", window_start_ms=1_710_000_180_000)
    )

    assert [call.get("status") for call in handler.calls] == ["firing", "resolved"]

    # After a resolve the same scope can alert again straight away.
    await manager.record_external_alert(_guardrail(window_start_ms=1_710_000_240_000))
    assert [call.get("status") for call in handler.calls] == [
        "firing",
        "resolved",
        "firing",
    ]


@pytest.mark.asyncio
async def test_slow_handler_does_not_block_record_external_alert():
    manager = AlertManager({"alerts": {"notification_timeout_seconds": 5}})
    release = asyncio.Event()
    started = asyncio.Event()
    sent = []

    class SlowHandler:
        async def send_alert(self, payload):
            started.set()
            await release.wait()
            sent.append(payload)

    manager.notification_handlers = {"slow": SlowHandler()}
    manager.start_sender()
    try:
        await asyncio.wait_for(manager.record_external_alert(_guardrail()), timeout=0.5)
        await asyncio.wait_for(
            manager.record_external_alert(_guardrail(scope_key="other")),
            timeout=0.5,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        assert sent == []
        release.set()
        await asyncio.wait_for(manager.drain_notifications(), timeout=1)
        assert [p["scope_key"] for p in sent] == ["gpt-5", "other"]
    finally:
        await manager.stop_sender()


@pytest.mark.asyncio
async def test_full_notification_queue_drops_instead_of_blocking():
    manager = AlertManager({"alerts": {"notification_queue_size": 1}})
    release = asyncio.Event()

    class SlowHandler:
        async def send_alert(self, payload):
            await release.wait()

    manager.notification_handlers = {"slow": SlowHandler()}
    manager.start_sender()
    try:
        for index in range(5):
            await asyncio.wait_for(
                manager.record_external_alert(_guardrail(scope_key=f"m{index}")),
                timeout=0.5,
            )
        assert manager.dropped_notification_count >= 1
    finally:
        release.set()
        await manager.stop_sender()


@pytest.mark.asyncio
async def test_hung_handler_times_out():
    manager = AlertManager({"alerts": {"notification_timeout_seconds": 0.05}})

    class HungHandler:
        async def send_alert(self, payload):
            await asyncio.Event().wait()

    manager.notification_handlers = {"hung": HungHandler()}

    await asyncio.wait_for(manager.record_external_alert(_guardrail()), timeout=1)


def _rule(pending_seconds=0, duration=3600):
    return AlertRule(
        name="high_error_rate",
        condition="error_rate > threshold",
        threshold=0.05,
        duration=duration,
        severity="critical",
        description="Error rate too high",
        pending_seconds=pending_seconds,
    )


@pytest.mark.asyncio
async def test_pending_period_delays_notification_and_recovery_resolves():
    manager = AlertManager({})
    manager.alert_rules = [_rule(pending_seconds=120)]
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    start = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)

    await manager.check_alerts({"error_rate": 0.2}, now=start)
    await manager.check_alerts({"error_rate": 0.2}, now=start + timedelta(seconds=60))
    assert handler.calls == []

    await manager.check_alerts({"error_rate": 0.2}, now=start + timedelta(seconds=121))
    assert [c["status"] for c in handler.calls] == ["firing"]

    await manager.check_alerts({"error_rate": 0.0}, now=start + timedelta(seconds=180))
    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]
    assert handler.calls[-1]["rule_name"] == "high_error_rate"

    # A second, short breach that recovers inside the pending period is silent.
    await manager.check_alerts({"error_rate": 0.2}, now=start + timedelta(seconds=200))
    await manager.check_alerts({"error_rate": 0.0}, now=start + timedelta(seconds=230))
    assert len(handler.calls) == 2


@pytest.mark.asyncio
async def test_pending_seconds_defaults_to_zero():
    assert _rule().pending_seconds == 0
    rule = AlertRule(
        name="x",
        condition="error_rate > threshold",
        threshold=1,
        duration=60,
        severity="warning",
        description="x",
    )
    assert rule.pending_seconds == 0


def _stale_rule_names(manager):
    return [rule.name for rule in manager.alert_rules]


@pytest.mark.asyncio
async def test_stale_stream_metrics_raise_an_alert():
    service = MonitoringService({"stream_metrics_staleness_seconds": 300})
    handler = _RecordingHandler()
    service.alert_manager.notification_handlers = {"test": handler}
    assert "stream_metrics_stale" in _stale_rule_names(service.alert_manager)
    collector = service.metrics_collector

    # The event says April 2026; freshness uses the monitor's receive time.
    received = 1_000.0
    collector._clock = lambda: received
    collector.record_stream_model_metrics(
        {
            "model_name": "gpt-5",
            "provider": "openai",
            "request_count": 4,
            "emitted_at": "2026-04-08T12:00:00Z",
        }
    )

    collector._clock = lambda: received + 60
    metrics = service._flatten_metrics(collector.stream_freshness_metrics())
    await service.alert_manager.check_alerts(metrics)
    assert handler.calls == []

    collector._clock = lambda: received + 301
    metrics = service._flatten_metrics(collector.stream_freshness_metrics())
    await service.alert_manager.check_alerts(metrics)
    assert [c["rule_name"] for c in handler.calls] == ["stream_metrics_stale"]


@pytest.mark.asyncio
async def test_stream_metrics_never_received_is_stale_after_threshold():
    collector = MetricsCollector({"stream_metrics_staleness_seconds": 300})
    collector._clock = lambda: collector._started_at + 301

    freshness = collector.stream_freshness_metrics()

    assert freshness["streaming"]["metrics_age_seconds"] >= 301


@pytest.mark.asyncio
async def test_collected_metrics_include_stream_age():
    collector = MetricsCollector({})
    metrics = await collector.collect_all_metrics()
    assert "metrics_age_seconds" in metrics["streaming"]


@pytest.mark.asyncio
async def test_business_metrics_have_no_fabricated_values():
    collector = MetricsCollector({})
    business = await collector._collect_business_metrics()
    assert "user_satisfaction_score" not in business
    assert "model_availability" not in business


@pytest.mark.asyncio
async def test_notification_handlers_share_one_client_with_timeout(monkeypatch):
    created = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.posts = 0
            self.closed = False
            created.append(self)

        async def post(self, *args, **kwargs):
            self.posts += 1
            return FakeResponse()

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("src.llm_router_part4_monitor.httpx.AsyncClient", FakeClient)
    payload = {
        "rule_name": "x",
        "severity": "warning",
        "description": "d",
        "current_value": 1,
        "threshold": 1,
        "timestamp": "t",
        "trigger_count": 1,
    }
    slack = SlackNotificationHandler({"webhook_url": "http://hooks.test/slack"})
    webhook = WebhookNotificationHandler({"url": "http://hooks.test/webhook"})
    for _ in range(3):
        await slack.send_alert(payload)
        await webhook.send_alert(payload)

    assert len(created) == 2
    assert [client.posts for client in created] == [3, 3]
    assert all(client.kwargs.get("timeout") is not None for client in created)

    await slack.aclose()
    await webhook.aclose()
    assert all(client.closed for client in created)


@pytest.mark.asyncio
async def test_monitoring_shutdown_stops_sender():
    service = MonitoringService({})
    service.alert_manager.start_sender()
    assert service.alert_manager.sender_running is True
    await service.shutdown()
    assert service.alert_manager.sender_running is False


def _defined_metric_names():
    source = (REPO_ROOT / "src" / "utils" / "metrics.py").read_text()
    names = set(re.findall(r'"(llm_router_[a-z0-9_]+)"', source))
    expanded = set(names)
    for name in names:
        for suffix in ("_bucket", "_sum", "_count", "_created"):
            expanded.add(name + suffix)
        if not name.endswith("_total"):
            expanded.add(name + "_total")
    return expanded


def test_prometheus_rule_file_parses_and_references_known_metrics():
    document = yaml.safe_load(RULES_PATH.read_text())
    rules = [rule for group in document["groups"] for rule in group["rules"]]
    assert rules
    defined = _defined_metric_names()
    referenced = set()
    for rule in rules:
        assert rule["alert"]
        assert rule["expr"]
        assert rule["labels"]["severity"] in {"critical", "warning", "info"}
        assert rule["annotations"]["summary"]
        referenced.update(re.findall(r"llm_router_[a-z0-9_]+", rule["expr"]))
    assert any(name.startswith("llm_router_kafka_") for name in referenced)
    assert referenced - defined == set()


def test_prometheus_rules_cover_kafka_failure_modes():
    document = yaml.safe_load(RULES_PATH.read_text())
    exprs = " ".join(
        rule["expr"] for group in document["groups"] for rule in group["rules"]
    )
    for metric in (
        "llm_router_kafka_producer_queue_dropped_total",
        "llm_router_kafka_dead_letter_messages_total",
        "llm_router_kafka_consumer_lag",
        "llm_router_kafka_consumer_paused_partitions",
    ):
        assert metric in exprs
