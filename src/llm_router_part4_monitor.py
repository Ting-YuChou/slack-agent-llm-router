"""
LLM Router Part 4: Monitoring Service - Prometheus, Grafana, and Slack Integration
Comprehensive observability, alerting, and performance monitoring
"""

import asyncio
from collections import Counter as CollectionCounter
import json
import logging
import time
import psutil
import GPUtil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import yaml

from src.utils.logger import setup_logging
from src.utils.metrics import (
    SYSTEM_METRICS,
    ROUTER_METRICS,
    INFERENCE_METRICS,
    PIPELINE_METRICS,
)

logger = logging.getLogger(__name__)


def _parse_event_timestamp(value: Any) -> datetime:
    """Parse event timestamps emitted by the streaming pipeline."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug(f"Failed to parse event timestamp: {value}")
    return datetime.now()


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str  # critical, warning, info
    description: str
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SystemHealth:
    """System health status"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: List[float]
    gpu_memory: List[float]
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    timestamp: datetime


class SystemMonitor:
    """System resource monitoring"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_interval = config.get("collection_interval", 15)  # seconds
        self.metrics_registry = CollectorRegistry()
        self.running = False

        # Initialize metrics
        self._init_system_metrics()

    def _init_system_metrics(self):
        """Initialize system monitoring metrics"""
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "CPU Usage Percentage",
            registry=self.metrics_registry,
        )
        self.memory_usage = Gauge(
            "system_memory_usage_percent",
            "Memory Usage Percentage",
            registry=self.metrics_registry,
        )
        self.disk_usage = Gauge(
            "system_disk_usage_percent",
            "Disk Usage Percentage",
            ["mount_point"],
            registry=self.metrics_registry,
        )
        self.gpu_usage = Gauge(
            "system_gpu_usage_percent",
            "GPU Usage Percentage",
            ["gpu_id"],
            registry=self.metrics_registry,
        )
        self.gpu_memory = Gauge(
            "system_gpu_memory_usage_percent",
            "GPU Memory Usage Percentage",
            ["gpu_id"],
            registry=self.metrics_registry,
        )
        self.network_io = Gauge(
            "system_network_io_bytes",
            "Network I/O Bytes",
            ["direction"],
            registry=self.metrics_registry,
        )
        self.process_count = Gauge(
            "system_process_count",
            "Number of Running Processes",
            registry=self.metrics_registry,
        )
        self.uptime = Gauge(
            "system_uptime_seconds",
            "System Uptime in Seconds",
            registry=self.metrics_registry,
        )

    async def start_monitoring(self):
        """Start system monitoring loop"""
        self.running = True
        logger.info("System monitoring started")

        while self.running:
            try:
                health = await self._collect_system_metrics()
                await self._update_metrics(health)
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self) -> SystemHealth:
        """Collect system metrics"""
        # CPU usage (non-blocking sample)
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
        }

        # Process count
        process_count = len(psutil.pids())

        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # GPU metrics (if available)
        gpu_usage = []
        gpu_memory = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_usage.append(gpu.load * 100)
                gpu_memory.append(gpu.memoryUtil * 100)
        except:
            # GPU monitoring not available
            pass

        return SystemHealth(
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_usage=disk_percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_io=network_io,
            process_count=process_count,
            uptime=uptime,
            timestamp=datetime.now(),
        )

    async def _update_metrics(self, health: SystemHealth):
        """Update Prometheus metrics"""
        self.cpu_usage.set(health.cpu_usage)
        self.memory_usage.set(health.memory_usage)
        self.disk_usage.labels(mount_point="/").set(health.disk_usage)
        self.process_count.set(health.process_count)
        self.uptime.set(health.uptime)

        # Network I/O
        self.network_io.labels(direction="sent").set(health.network_io["bytes_sent"])
        self.network_io.labels(direction="received").set(
            health.network_io["bytes_recv"]
        )

        # GPU metrics
        for i, usage in enumerate(health.gpu_usage):
            self.gpu_usage.labels(gpu_id=str(i)).set(usage)

        for i, memory in enumerate(health.gpu_memory):
            self.gpu_memory.labels(gpu_id=str(i)).set(memory)

    async def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health asynchronously"""
        try:
            return await self._collect_system_metrics()
        except Exception as exc:
            logger.debug(f"Failed to collect system health: {exc}")
            return None

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        logger.info("System monitoring stopped")


class AlertManager:
    """Alert management and notification system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
        self.notification_handlers = {}
        self.alert_history = []
        self.max_history_size = 1000

        # Load alert rules
        self._load_alert_rules()

        # Initialize notification handlers
        self._init_notification_handlers()

    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        rules_config = self.config.get("alerts", {})

        # Default alert rules
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=rules_config.get("error_rate_threshold", 0.05),
                duration=300,  # 5 minutes
                severity="critical",
                description="Error rate is above acceptable threshold",
            ),
            AlertRule(
                name="high_latency",
                condition="avg_latency_ms > threshold",
                threshold=rules_config.get("latency_threshold_ms", 5000),
                duration=300,
                severity="warning",
                description="Average latency is high",
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage > threshold",
                threshold=rules_config.get("memory_usage_threshold", 0.9),
                duration=600,  # 10 minutes
                severity="warning",
                description="System memory usage is high",
            ),
            AlertRule(
                name="high_disk_usage",
                condition="disk_usage > threshold",
                threshold=rules_config.get("disk_usage_threshold", 0.8),
                duration=1800,  # 30 minutes
                severity="warning",
                description="Disk usage is high",
            ),
            AlertRule(
                name="model_unavailable",
                condition="model_health_status == 'unhealthy'",
                threshold=1,
                duration=60,
                severity="critical",
                description="Model is unavailable",
            ),
        ]

        self.alert_rules.extend(default_rules)
        logger.info(f"Loaded {len(self.alert_rules)} alert rules")

    def _init_notification_handlers(self):
        """Initialize notification handlers"""
        # Slack notification handler
        if self.config.get("slack", {}).get("enabled"):
            self.notification_handlers["slack"] = SlackNotificationHandler(
                self.config["slack"]
            )

        # Email notification handler (placeholder)
        if self.config.get("email", {}).get("enabled"):
            self.notification_handlers["email"] = EmailNotificationHandler(
                self.config["email"]
            )

        # Webhook notification handler
        if self.config.get("webhook", {}).get("enabled"):
            self.notification_handlers["webhook"] = WebhookNotificationHandler(
                self.config["webhook"]
            )

    async def check_alerts(self, metrics: Dict[str, float]):
        """Check all alert rules against current metrics"""
        current_time = datetime.now()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Evaluate alert condition
                triggered = self._evaluate_condition(rule, metrics)

                if triggered:
                    # Check if we should send notification (debouncing)
                    if (
                        not rule.last_triggered
                        or (current_time - rule.last_triggered).total_seconds()
                        > rule.duration
                    ):
                        await self._trigger_alert(rule, metrics)
                        rule.last_triggered = current_time
                        rule.trigger_count += 1

            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """Evaluate if alert condition is met"""
        try:
            # Simple condition evaluation
            # In production, use a proper expression parser
            condition = rule.condition.replace("threshold", str(rule.threshold))

            # Map metric names to values
            context = {
                "error_rate": self._get_metric_value(
                    metrics, ["application.error_rate", "error_rate"], 0
                ),
                "avg_latency_ms": self._get_metric_value(
                    metrics, ["application.avg_latency_ms", "avg_latency_ms"], 0
                ),
                "memory_usage": self._get_metric_value(
                    metrics, ["system.memory_usage", "memory_usage"], 0
                ),
                "disk_usage": self._get_metric_value(
                    metrics, ["system.disk_usage", "disk_usage"], 0
                ),
                "cpu_usage": self._get_metric_value(
                    metrics, ["system.cpu_usage", "cpu_usage"], 0
                ),
                "model_health_status": self._get_metric_value(
                    metrics,
                    ["application.model_health_status", "model_health_status"],
                    "healthy",
                ),
            }

            # Evaluate condition
            return eval(condition, {"__builtins__": {}}, context)

        except Exception as e:
            logger.error(f"Failed to evaluate condition for rule {rule.name}: {e}")
            return False

    @staticmethod
    def _get_metric_value(
        metrics: Dict[str, Any], keys: List[str], default: Any
    ) -> Any:
        """Return first available metric value for given keys"""
        for key in keys:
            if key in metrics:
                return metrics[key]
        return default

    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, float]):
        """Trigger alert and send notifications"""
        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity,
            "description": rule.description,
            "threshold": rule.threshold,
            "current_value": self._get_current_value(rule, metrics),
            "timestamp": datetime.now().isoformat(),
            "trigger_count": rule.trigger_count,
        }

        self._append_alert_to_history(alert_data)
        await self._dispatch_alert(alert_data)

    def _append_alert_to_history(self, alert_data: Dict[str, Any]):
        """Persist an alert in the in-memory alert history."""
        self.alert_history.append(alert_data)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)

    async def _dispatch_alert(self, alert_data: Dict[str, Any]):
        """Send an alert to all configured notification handlers."""
        for handler_name, handler in self.notification_handlers.items():
            try:
                await handler.send_alert(alert_data)
                logger.info(
                    f"Alert sent via {handler_name}: {alert_data.get('rule_name')}"
                )
            except Exception as e:
                logger.error(f"Failed to send alert via {handler_name}: {e}")

    async def record_external_alert(self, alert_data: Dict[str, Any]):
        """Record and optionally notify on alert events emitted by the stream layer."""
        metrics = alert_data.get("metrics", {}) if isinstance(alert_data, dict) else {}
        anomaly_type = alert_data.get("anomaly_type")
        event_type = alert_data.get("event_type")
        trigger_type = alert_data.get("trigger_type")
        metric_name = anomaly_type or trigger_type or alert_data.get("alert_type")
        current_value = None

        if anomaly_type == "high_latency":
            current_value = metrics.get("avg_latency_ms")
        elif anomaly_type == "high_volume":
            current_value = metrics.get("queries_per_second")
        elif anomaly_type == "high_error_rate":
            current_value = metrics.get("error_rate")

        normalized_alert = {
            "rule_name": anomaly_type
            or trigger_type
            or alert_data.get("alert_type", "external_alert"),
            "severity": alert_data.get("severity", "warning"),
            "description": alert_data.get("description", "External alert"),
            "threshold": alert_data.get("threshold"),
            "current_value": current_value,
            "timestamp": alert_data.get("emitted_at", datetime.now().isoformat()),
            "trigger_count": 1,
            "metric_name": metric_name,
            "alert_type": alert_data.get("alert_type"),
            "event_type": event_type,
            "source": (
                "routing_guardrails"
                if event_type == "routing.guardrails"
                else "stream_pipeline"
            ),
            "model_name": alert_data.get("model_name"),
            "provider": alert_data.get("provider"),
            "scope_type": alert_data.get("scope_type"),
            "scope_key": alert_data.get("scope_key"),
            "guardrail_action": alert_data.get("guardrail_action"),
        }

        self._append_alert_to_history(normalized_alert)
        await self._dispatch_alert(normalized_alert)

    def _get_current_value(self, rule: AlertRule, metrics: Dict[str, float]) -> float:
        """Get current value for the metric being alerted on"""
        if "error_rate" in rule.condition:
            return metrics.get("error_rate", 0)
        elif "latency" in rule.condition:
            return metrics.get("avg_latency_ms", 0)
        elif "memory" in rule.condition:
            return metrics.get("memory_usage", 0)
        elif "disk" in rule.condition:
            return metrics.get("disk_usage", 0)
        else:
            return 0

    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        return {
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules if r.enabled]),
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "notification_handlers": list(self.notification_handlers.keys()),
        }


class SlackNotificationHandler:
    """Slack notification handler"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "LLM Router Monitor")

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        # Format alert message
        severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}

        emoji = severity_emoji.get(alert_data["severity"], "📊")

        message = {
            "channel": self.channel,
            "username": self.username,
            "text": f"{emoji} *{alert_data['rule_name'].upper()}*",
            "attachments": [
                {
                    "color": "danger"
                    if alert_data["severity"] == "critical"
                    else "warning",
                    "fields": [
                        {
                            "title": "Description",
                            "value": alert_data["description"],
                            "short": False,
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert_data["current_value"]),
                            "short": True,
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert_data["threshold"]),
                            "short": True,
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_data["timestamp"],
                            "short": True,
                        },
                        {
                            "title": "Trigger Count",
                            "value": str(alert_data["trigger_count"]),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.webhook_url, json=message)
            response.raise_for_status()


class EmailNotificationHandler:
    """Email notification handler (placeholder)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email"""
        # Placeholder for email implementation
        logger.info(f"Email alert would be sent: {alert_data['rule_name']}")


class WebhookNotificationHandler:
    """Generic webhook notification handler"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get("url")
        self.headers = config.get("headers", {})

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert via webhook"""
        if not self.webhook_url:
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.webhook_url, json=alert_data, headers=self.headers
            )
            response.raise_for_status()


class MetricsCollector:
    """Collects and aggregates metrics from various sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_interval = config.get("collection_interval", 30)
        self.metrics_history = []
        self.max_history_size = 1000
        self.stream_metrics_history = []
        self.latest_stream_metrics_by_model: Dict[str, Dict[str, Any]] = {}
        self.routing_feature_history = []
        self.routing_guardrail_history = []
        self.stream_metrics_staleness_seconds = config.get(
            "stream_metrics_staleness_seconds", 300
        )

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all sources"""
        system_metrics = await self._collect_system_metrics()
        application_metrics = await self._collect_application_metrics()
        business_metrics = await self._collect_business_metrics()
        streaming_metrics = self._collect_streaming_metrics()
        routing_metrics = self.get_routing_feature_summary(hours=1)
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "application": application_metrics,
            "business": business_metrics,
            "streaming": streaming_metrics,
            "routing": routing_metrics,
        }

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

        return metrics

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU, Memory, Disk usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_usage": cpu_percent / 100,
                "memory_usage": memory.percent / 100,
                "disk_usage": (disk.used / disk.total),
                "load_average": psutil.getloadavg()[0]
                if hasattr(psutil, "getloadavg")
                else 0,
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-level metrics"""
        try:
            # Get metrics from Prometheus collectors
            router_metrics = self._extract_prometheus_metrics(ROUTER_METRICS)
            inference_metrics = self._extract_prometheus_metrics(INFERENCE_METRICS)
            pipeline_metrics = self._extract_prometheus_metrics(PIPELINE_METRICS)
            streaming_metrics = self._collect_streaming_metrics()
            application_metrics = {
                **router_metrics,
                **inference_metrics,
                **pipeline_metrics,
            }

            if streaming_metrics.get("request_count", 0) > 0:
                application_metrics.update(
                    {
                        "avg_latency_ms": streaming_metrics.get("avg_latency_ms", 0.0),
                        "error_rate": streaming_metrics.get("error_rate", 0.0),
                        "queries_per_second": streaming_metrics.get(
                            "queries_per_second", 0.0
                        ),
                        "cache_hit_rate": streaming_metrics.get("cache_hit_rate", 0.0),
                    }
                )

            return application_metrics
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}

    async def _collect_business_metrics(self) -> Dict[str, float]:
        """Collect business-level metrics"""
        try:
            # Calculate business metrics from recent data
            # This would typically query the database
            return {
                "queries_per_minute": 0,  # TODO: Calculate from recent data
                "average_cost_per_query": 0,
                "user_satisfaction_score": 0.95,  # Placeholder
                "model_availability": 1.0,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
            return {}

    def _extract_prometheus_metrics(self, metrics_object) -> Dict[str, float]:
        """Extract values from Prometheus metrics"""
        extracted = {}

        # This is a simplified extraction
        # In practice, you'd iterate through the metrics registry
        try:
            for attr_name in dir(metrics_object):
                attr = getattr(metrics_object, attr_name)
                if hasattr(attr, "_value"):
                    extracted[attr_name] = attr._value._value
        except Exception as e:
            logger.debug(f"Metrics extraction error: {e}")

        return extracted

    def record_stream_model_metrics(self, metric_event: Dict[str, Any]):
        """Store a consumed analytics.model_metrics_1m event for dashboard use."""
        event = dict(metric_event)
        event.setdefault(
            "timestamp", event.get("emitted_at", datetime.now().isoformat())
        )
        model_name = str(event.get("model_name", "unknown"))

        self.stream_metrics_history.append(event)
        if len(self.stream_metrics_history) > self.max_history_size:
            self.stream_metrics_history.pop(0)

        self.latest_stream_metrics_by_model[model_name] = event

    def _recent_stream_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Return stream metrics still within the configured freshness window."""
        cutoff = datetime.now() - timedelta(hours=hours)

        return [
            metric
            for metric in self.stream_metrics_history
            if _parse_event_timestamp(metric.get("timestamp")) >= cutoff
        ]

    def _fresh_stream_metrics(self) -> List[Dict[str, Any]]:
        """Return the latest non-stale stream window for each model."""
        cutoff = datetime.now() - timedelta(
            seconds=self.stream_metrics_staleness_seconds
        )
        return [
            metric
            for metric in self.latest_stream_metrics_by_model.values()
            if _parse_event_timestamp(metric.get("timestamp")) >= cutoff
        ]

    def _collect_streaming_metrics(self) -> Dict[str, Any]:
        """Build a lightweight aggregate view from recent Flink model windows."""
        fresh_metrics = self._fresh_stream_metrics()
        if not fresh_metrics:
            return {
                "models": {},
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
                "queries_per_second": 0.0,
                "cache_hit_rate": 0.0,
            }

        total_requests = sum(int(m.get("request_count", 0) or 0) for m in fresh_metrics)
        total_errors = sum(int(m.get("error_count", 0) or 0) for m in fresh_metrics)
        total_cached = sum(int(m.get("cached_count", 0) or 0) for m in fresh_metrics)
        weighted_latency = sum(
            float(m.get("avg_latency_ms", 0.0) or 0.0)
            * int(m.get("request_count", 0) or 0)
            for m in fresh_metrics
        )
        total_qps = sum(
            float(m.get("queries_per_second", 0.0) or 0.0) for m in fresh_metrics
        )

        latest_models = {}
        for metric in fresh_metrics:
            model_name = str(metric.get("model_name", "unknown"))
            latest_models[model_name] = {
                "provider": metric.get("provider"),
                "request_count": int(metric.get("request_count", 0) or 0),
                "avg_latency_ms": float(metric.get("avg_latency_ms", 0.0) or 0.0),
                "error_rate": float(metric.get("error_rate", 0.0) or 0.0),
                "queries_per_second": float(
                    metric.get("queries_per_second", 0.0) or 0.0
                ),
                "cache_hit_rate": float(metric.get("cache_hit_rate", 0.0) or 0.0),
                "window_end_ms": int(metric.get("window_end_ms", 0) or 0),
            }

        return {
            "models": latest_models,
            "request_count": total_requests,
            "avg_latency_ms": (
                weighted_latency / total_requests if total_requests > 0 else 0.0
            ),
            "error_rate": (
                total_errors / total_requests if total_requests > 0 else 0.0
            ),
            "queries_per_second": total_qps,
            "cache_hit_rate": (
                total_cached / total_requests if total_requests > 0 else 0.0
            ),
        }

    def get_stream_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Expose a dashboard-friendly summary of recent Flink model windows."""
        recent_metrics = self._recent_stream_metrics(hours=hours)
        summary = self._collect_streaming_metrics()
        summary["period_hours"] = hours
        summary["sample_count"] = len(recent_metrics)
        summary["total_requests_in_period"] = sum(
            int(metric.get("request_count", 0) or 0) for metric in recent_metrics
        )
        return summary

    def record_request_routing_features(self, routing_event: Dict[str, Any]):
        """Store a consumed requests.enriched event for dashboard use."""
        event = dict(routing_event)
        event.setdefault("timestamp", event.get("emitted_at", datetime.now().isoformat()))
        self.routing_feature_history.append(event)
        if len(self.routing_feature_history) > self.max_history_size:
            self.routing_feature_history.pop(0)

    def _recent_request_routing_features(self, hours: int = 1) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event
            for event in self.routing_feature_history
            if _parse_event_timestamp(event.get("timestamp")) >= cutoff
        ]

    def get_routing_feature_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Summarize recent request-side Flink routing features."""
        recent_events = self._recent_request_routing_features(hours=hours)
        if not recent_events:
            return {
                "period_hours": hours,
                "request_count": 0,
                "fast_lane_count": 0,
                "requires_high_reasoning_count": 0,
                "query_type_breakdown": {},
                "query_complexity_breakdown": {},
                "session_hotness_breakdown": {},
                "top_preferred_models": {},
                "top_avoid_models": {},
                "top_avoid_providers": {},
                "recent_requests": [],
            }

        query_type_counts = CollectionCounter(
            str(event.get("query_type", "unknown")) for event in recent_events
        )
        query_complexity_counts = CollectionCounter(
            str(event.get("query_complexity", "unknown")) for event in recent_events
        )
        session_hotness_counts = CollectionCounter(
            str(event.get("session_hotness", "unknown")) for event in recent_events
        )
        preferred_model_counts = CollectionCounter()
        avoid_model_counts = CollectionCounter()
        avoid_provider_counts = CollectionCounter()

        for event in recent_events:
            preferred_model_counts.update(list(event.get("preferred_models", []) or []))
            avoid_model_counts.update(list(event.get("avoid_models", []) or []))
            avoid_provider_counts.update(list(event.get("avoid_providers", []) or []))

        recent_requests = []
        for event in recent_events[-10:]:
            recent_requests.append(
                {
                    "request_id": event.get("request_id"),
                    "query_type": event.get("query_type"),
                    "query_complexity": event.get("query_complexity"),
                    "priority": event.get("priority"),
                    "route_to_fast_lane": bool(event.get("route_to_fast_lane", False)),
                    "requires_low_latency": bool(
                        event.get("requires_low_latency", False)
                    ),
                    "requires_high_reasoning": bool(
                        event.get("requires_high_reasoning", False)
                    ),
                    "session_hotness": event.get("session_hotness"),
                    "cost_sensitivity": event.get("cost_sensitivity"),
                    "error_sensitivity": event.get("error_sensitivity"),
                    "timestamp": event.get("timestamp"),
                }
            )

        return {
            "period_hours": hours,
            "request_count": len(recent_events),
            "fast_lane_count": sum(
                1 for event in recent_events if event.get("route_to_fast_lane")
            ),
            "requires_high_reasoning_count": sum(
                1 for event in recent_events if event.get("requires_high_reasoning")
            ),
            "query_type_breakdown": dict(query_type_counts),
            "query_complexity_breakdown": dict(query_complexity_counts),
            "session_hotness_breakdown": dict(session_hotness_counts),
            "top_preferred_models": dict(preferred_model_counts.most_common(5)),
            "top_avoid_models": dict(avoid_model_counts.most_common(5)),
            "top_avoid_providers": dict(avoid_provider_counts.most_common(5)),
            "recent_requests": recent_requests,
        }

    def record_routing_guardrail(self, guardrail_event: Dict[str, Any]):
        """Store a consumed routing.guardrails event for dashboard use."""
        event = dict(guardrail_event)
        event.setdefault("timestamp", event.get("emitted_at", datetime.now().isoformat()))
        self.routing_guardrail_history.append(event)
        if len(self.routing_guardrail_history) > self.max_history_size:
            self.routing_guardrail_history.pop(0)

    def _recent_routing_guardrails(self, hours: int = 1) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event
            for event in self.routing_guardrail_history
            if _parse_event_timestamp(event.get("timestamp")) >= cutoff
        ]

    def get_routing_guardrail_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Summarize recent routing guardrails for dashboard use."""
        recent_events = self._recent_routing_guardrails(hours=hours)
        if not recent_events:
            return {
                "period_hours": hours,
                "guardrail_count": 0,
                "scope_breakdown": {},
                "trigger_breakdown": {},
                "recent_guardrails": [],
            }

        scope_counts = CollectionCounter(
            str(event.get("scope_type", "unknown")) for event in recent_events
        )
        trigger_counts = CollectionCounter(
            str(event.get("trigger_type", "unknown")) for event in recent_events
        )
        recent_guardrails = []
        for event in recent_events[-10:]:
            recent_guardrails.append(
                {
                    "scope_type": event.get("scope_type"),
                    "scope_key": event.get("scope_key"),
                    "trigger_type": event.get("trigger_type"),
                    "guardrail_action": event.get("guardrail_action"),
                    "severity": event.get("severity"),
                    "description": event.get("description"),
                    "model_name": event.get("model_name"),
                    "provider": event.get("provider"),
                    "timestamp": event.get("timestamp"),
                }
            )

        return {
            "period_hours": hours,
            "guardrail_count": len(recent_events),
            "scope_breakdown": dict(scope_counts),
            "trigger_breakdown": dict(trigger_counts),
            "recent_guardrails": recent_guardrails,
        }

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

        if not recent_metrics:
            return {}

        # Calculate averages and trends
        summary = {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "avg_cpu_usage": self._average_metric(recent_metrics, "system.cpu_usage"),
            "avg_memory_usage": self._average_metric(
                recent_metrics, "system.memory_usage"
            ),
            "avg_latency": self._average_metric(
                recent_metrics, "application.avg_latency_ms"
            ),
            "error_rate": self._average_metric(
                recent_metrics, "application.error_rate"
            ),
        }

        return summary

    def _average_metric(self, metrics_list: List[Dict], metric_path: str) -> float:
        """Calculate average for a nested metric"""
        values = []
        for metrics in metrics_list:
            try:
                keys = metric_path.split(".")
                value = metrics
                for key in keys:
                    value = value[key]
                if isinstance(value, (int, float)):
                    values.append(value)
            except (KeyError, TypeError):
                continue

        return sum(values) / len(values) if values else 0


class PerformanceProfiler:
    """Performance profiling and optimization recommendations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profiling_enabled = config.get("profiling_enabled", False)
        self.performance_data = []

    async def profile_inference_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile a single inference request"""
        if not self.profiling_enabled:
            return {}

        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_data.get("request_id"),
            "model_name": request_data.get("model_name"),
            "token_count": request_data.get("token_count", 0),
            "latency_ms": request_data.get("latency_ms", 0),
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage_percent": psutil.cpu_percent(),
        }

        self.performance_data.append(profile_data)

        # Keep only recent data
        if len(self.performance_data) > 10000:
            self.performance_data = self.performance_data[-5000:]

        return profile_data

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0

    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        if not self.performance_data:
            return {"recommendations": []}

        recommendations = []

        # Analyze latency patterns
        recent_data = self.performance_data[-1000:]  # Last 1000 requests
        avg_latency = sum(d["latency_ms"] for d in recent_data) / len(recent_data)

        if avg_latency > 2000:  # 2 seconds
            recommendations.append(
                {
                    "type": "latency",
                    "priority": "high",
                    "description": f"Average latency is {avg_latency:.0f}ms, consider model optimization",
                    "suggestions": [
                        "Enable response caching",
                        "Use smaller models for simple queries",
                        "Implement request batching",
                    ],
                }
            )

        # Analyze memory usage
        avg_memory = sum(d["memory_usage_mb"] for d in recent_data) / len(recent_data)
        if avg_memory > 8000:  # 8GB
            recommendations.append(
                {
                    "type": "memory",
                    "priority": "medium",
                    "description": f"High memory usage detected: {avg_memory:.0f}MB",
                    "suggestions": [
                        "Implement context compression",
                        "Reduce batch sizes",
                        "Consider model quantization",
                    ],
                }
            )

        # Analyze model usage patterns
        model_usage = {}
        for data in recent_data:
            model = data["model_name"]
            model_usage[model] = model_usage.get(model, 0) + 1

        if len(model_usage) > 1:
            most_used = max(model_usage.keys(), key=model_usage.get)
            usage_percent = model_usage[most_used] / len(recent_data) * 100

            if usage_percent > 80:
                recommendations.append(
                    {
                        "type": "routing",
                        "priority": "low",
                        "description": f"Model {most_used} handles {usage_percent:.1f}% of requests",
                        "suggestions": [
                            "Review routing rules",
                            "Consider load balancing",
                            "Evaluate model performance differences",
                        ],
                    }
                )

        return {
            "analysis_period": "last_1000_requests",
            "avg_latency_ms": avg_latency,
            "avg_memory_mb": avg_memory,
            "model_distribution": model_usage,
            "recommendations": recommendations,
        }


class MonitoringService:
    """Main monitoring service coordinator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_monitor = SystemMonitor(config)
        self.alert_manager = AlertManager(config)
        self.metrics_collector = MetricsCollector(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.running = False

        # Monitoring intervals
        self.health_check_interval = config.get("health_checks", {}).get(
            "interval_seconds", 30
        )
        self.metrics_collection_interval = config.get("metrics_collection_interval", 60)

    async def initialize(self):
        """Initialize monitoring service"""
        logger.info("Initializing monitoring service...")
        # Monitoring service is ready immediately
        logger.info("Monitoring service initialized successfully")

    async def start(self):
        """Start all monitoring components"""
        logger.info("Starting monitoring service...")
        self.running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self.system_monitor.start_monitoring()),
            asyncio.create_task(self._alert_checking_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
        ]

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()

                # Log health status
                if not health_status["overall_healthy"]:
                    logger.warning(f"System health issues detected: {health_status}")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _alert_checking_loop(self):
        """Alert checking loop"""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = await self.metrics_collector.collect_all_metrics()

                # Flatten metrics for alert checking
                flat_metrics = self._flatten_metrics(current_metrics)

                # Check alerts
                await self.alert_manager.check_alerts(flat_metrics)

                await asyncio.sleep(60)  # Check alerts every minute

            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.running:
            try:
                # Collect and store metrics
                await self.metrics_collector.collect_all_metrics()

                await asyncio.sleep(self.metrics_collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.metrics_collection_interval)

    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "components": {},
        }

        # System health
        system_health = await self.system_monitor.get_current_health()
        if system_health:
            health_status["components"]["system"] = {
                "healthy": system_health.cpu_usage < 90
                and system_health.memory_usage < 90,
                "cpu_usage": system_health.cpu_usage,
                "memory_usage": system_health.memory_usage,
                "disk_usage": system_health.disk_usage,
            }

            if not health_status["components"]["system"]["healthy"]:
                health_status["overall_healthy"] = False

        # Add other component health checks here
        # (router, inference engine, pipeline, etc.)

        return health_status

    def _flatten_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested metrics dictionary"""
        flat = {}

        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, (int, float)):
                    flat[new_key] = value

        flatten_dict(metrics)
        return flat

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return await self.metrics_collector.collect_all_metrics()

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report"""
        return await self.performance_profiler.generate_optimization_report()

    async def ingest_stream_model_metrics(self, metric_event: Dict[str, Any]):
        """Consume windowed analytics events forwarded by the pipeline."""
        self.metrics_collector.record_stream_model_metrics(metric_event)

    async def ingest_stream_request_enriched(self, routing_event: Dict[str, Any]):
        """Consume request enrichment events forwarded by the pipeline."""
        self.metrics_collector.record_request_routing_features(routing_event)

    async def ingest_stream_routing_guardrail(self, guardrail_event: Dict[str, Any]):
        """Consume routing guardrail events forwarded by the pipeline."""
        self.metrics_collector.record_routing_guardrail(guardrail_event)
        await self.alert_manager.record_external_alert(guardrail_event)

    async def ingest_stream_alert(self, alert_event: Dict[str, Any]):
        """Consume alert events forwarded by the pipeline."""
        await self.alert_manager.record_external_alert(alert_event)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        current_metrics = await self.metrics_collector.collect_all_metrics()
        metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
        stream_analytics = self.metrics_collector.get_stream_metrics_summary(hours=1)
        routing_features = self.metrics_collector.get_routing_feature_summary(hours=1)
        routing_guardrails = self.metrics_collector.get_routing_guardrail_summary(hours=1)
        alert_status = self.alert_manager.get_alert_status()
        performance_report = (
            await self.performance_profiler.generate_optimization_report()
        )

        return {
            "current_metrics": current_metrics,
            "metrics_summary": metrics_summary,
            "stream_analytics": stream_analytics,
            "routing_features": routing_features,
            "routing_guardrails": routing_guardrails,
            "alert_status": alert_status,
            "performance_report": performance_report,
            "timestamp": datetime.now().isoformat(),
        }

    def is_healthy(self) -> bool:
        """Check if monitoring service is healthy"""
        return self.running

    async def shutdown(self):
        """Shutdown monitoring service"""
        logger.info("Shutting down monitoring service...")
        self.running = False
        self.system_monitor.stop_monitoring()
        logger.info("Monitoring service shutdown complete")
