from datetime import datetime
from types import SimpleNamespace

import pytest

from src.llm_router_part4_monitor import (
    AlertManager,
    AlertRule,
    SystemHealth,
    SystemMonitor,
)


class TestSystemMonitor:
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, monkeypatch, fixed_datetime):
        monitor = SystemMonitor({"collection_interval": 1})

        monkeypatch.setattr(
            "src.llm_router_part4_monitor.psutil.cpu_percent",
            lambda interval=None: 12.5,
        )
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.psutil.virtual_memory",
            lambda: SimpleNamespace(percent=55.0),
        )
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.psutil.disk_usage",
            lambda _path: SimpleNamespace(used=200, total=400),
        )
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.psutil.net_io_counters",
            lambda: SimpleNamespace(bytes_sent=1000, bytes_recv=2000),
        )
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.psutil.pids", lambda: [1, 2, 3]
        )
        monkeypatch.setattr("src.llm_router_part4_monitor.psutil.boot_time", lambda: 10)
        monkeypatch.setattr("src.llm_router_part4_monitor.time.time", lambda: 110)
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.GPUtil.getGPUs",
            lambda: [SimpleNamespace(load=0.25, memoryUtil=0.5)],
        )
        monkeypatch.setattr(
            "src.llm_router_part4_monitor.datetime",
            SimpleNamespace(now=lambda: fixed_datetime),
        )

        health = await monitor._collect_system_metrics()

        assert isinstance(health, SystemHealth)
        assert health.cpu_usage == 12.5
        assert health.memory_usage == 55.0
        assert health.disk_usage == 50.0
        assert health.gpu_usage == [25.0]
        assert health.process_count == 3
        assert health.uptime == 100

    @pytest.mark.asyncio
    async def test_update_metrics_sets_gauges(self):
        monitor = SystemMonitor({"collection_interval": 1})
        health = SystemHealth(
            cpu_usage=22.0,
            memory_usage=44.0,
            disk_usage=66.0,
            gpu_usage=[10.0],
            gpu_memory=[30.0],
            network_io={"bytes_sent": 123, "bytes_recv": 456},
            process_count=7,
            uptime=99.0,
            timestamp=datetime(2026, 3, 12, 10, 0, 0),
        )

        await monitor._update_metrics(health)

        assert monitor.cpu_usage._value.get() == 22.0
        assert monitor.memory_usage._value.get() == 44.0
        assert monitor.process_count._value.get() == 7
        assert monitor.uptime._value.get() == 99.0


class TestAlertManager:
    def test_evaluate_condition_supports_alias_metric_names(self):
        manager = AlertManager({"alerts": {"latency_threshold_ms": 5000}})
        rule = AlertRule(
            name="high_latency",
            condition="avg_latency_ms > threshold",
            threshold=5000,
            duration=60,
            severity="warning",
            description="Latency too high",
        )

        assert (
            manager._evaluate_condition(rule, {"application.avg_latency_ms": 8000})
            is True
        )
        assert manager._evaluate_condition(rule, {"avg_latency_ms": 1000}) is False

    @pytest.mark.asyncio
    async def test_check_alerts_records_history_and_notifies_once(self):
        manager = AlertManager({})
        manager.alert_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=0.05,
                duration=3600,
                severity="critical",
                description="Error rate too high",
            )
        ]

        calls = []

        class Handler:
            async def send_alert(self, payload):
                calls.append(payload)

        manager.notification_handlers = {"test": Handler()}

        await manager.check_alerts({"error_rate": 0.2})
        await manager.check_alerts({"error_rate": 0.2})

        assert len(calls) == 1
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0]["rule_name"] == "high_error_rate"
