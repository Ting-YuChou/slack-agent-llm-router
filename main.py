#!/usr/bin/env python3
"""
LLM Router & Execution Platform - Main Entry Point
Production-grade multi-model deployment system for Slack AI

This is the unified entry point that orchestrates all system components:
- Model Router Service
- Inference Engine
- Kafka Pipeline
- Monitoring Stack
- Slack Bot Integration
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server

# Import all our components
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache
from src.llm_router_part3_pipeline import KafkaIngestionPipeline, KafkaProducerManager
from src.llm_router_part4_monitor import MonitoringService
from src.utils.logger import setup_logging
from src.utils.metrics import (
    INFERENCE_METRICS,
    ROUTER_METRICS,
    SYSTEM_METRICS,
    USER_METRICS,
)
from src.utils.schema import QueryRequest
from slack.bot import SlackBot


DEFAULT_CONFIG_PATH = "config/config.yaml"
CONFIG_ENV_VAR = "LLM_ROUTER_CONFIG"


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """Resolve config path from explicit input or environment."""
    return config_path or os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)


def get_api_worker_count(config: Dict[str, Any], dev: bool = False) -> int:
    """Return the configured API worker count."""
    if dev:
        return 1

    workers = config.get("performance", {}).get("workers", {}).get("api_workers", 1)
    try:
        return max(1, int(workers))
    except (TypeError, ValueError):
        return 1


class LLMRouterPlatform:
    """Main orchestrator for the LLM Router Platform"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = resolve_config_path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        self.services: Dict[str, Any] = {}
        self.processes: List[mp.Process] = []
        self.shutdown_event = asyncio.Event()

        # Setup logging
        setup_logging(
            log_level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file", "logs/llm_router.log")
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.getLogger(__name__).error(
                f"Configuration file not found: {self.config_path}"
            )
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.getLogger(__name__).error(f"Error parsing configuration: {e}")
            sys.exit(1)

    def _service_enabled(self, section: str) -> bool:
        """Return whether a top-level config section is enabled."""
        return bool(self.config.get(section, {}).get("enabled", False))

    def _build_pipeline_config(self) -> Dict[str, Any]:
        """Build pipeline config from top-level Kafka and ClickHouse sections."""
        pipeline_config = dict(self.config.get("kafka", {}))
        pipeline_config["clickhouse"] = dict(self.config.get("clickhouse", {}))
        pipeline_config["pipeline"] = dict(self.config.get("pipeline", {}))
        return pipeline_config

    def _build_event_producer_config(self) -> Dict[str, Any]:
        """Build producer-only Kafka config for API-side event emission."""
        return dict(self.config.get("kafka", {}))

    def _event_streaming_enabled(self) -> bool:
        """Return whether request/completion event streaming is enabled."""
        return self._service_enabled("pipeline") or self._service_enabled("flink")

    def _policy_cache_enabled(self) -> bool:
        """Return whether Flink routing hints should be materialized into shared cache."""
        return bool(self.config.get("policy_cache", {}).get("enabled", False))

    def _build_policy_cache_config(self) -> Dict[str, Any]:
        """Build shared policy cache config."""
        return dict(self.config.get("policy_cache", {}))

    async def _initialize_core_services(self):
        """Initialize the router and inference services required by the API."""
        if self._event_streaming_enabled():
            self.services["event_producer"] = KafkaProducerManager(
                config=self._build_event_producer_config()
            )
            await self.services["event_producer"].initialize()

        if self._policy_cache_enabled():
            self.services["policy_cache"] = RoutingPolicyCache(
                config=self._build_policy_cache_config()
            )
            await self.services["policy_cache"].initialize()

        self.services["router"] = ModelRouter(
            config=self.config.get("router", {}),
            policy_cache=self.services.get("policy_cache"),
        )
        await self.services["router"].initialize()

        self.services["inference"] = InferenceEngine(
            config=self.config.get("inference", {}),
            router=self.services["router"],
            event_producer=self.services.get("event_producer"),
        )
        await self.services["inference"].initialize()

    async def _initialize_background_services(self):
        """Initialize enabled background services."""
        if self._service_enabled("pipeline"):
            self.services["pipeline"] = KafkaIngestionPipeline(
                config=self._build_pipeline_config()
            )
            await self.services["pipeline"].initialize()

        if self._policy_cache_enabled():
            if "policy_cache" not in self.services:
                self.services["policy_cache"] = RoutingPolicyCache(
                    config=self._build_policy_cache_config()
                )
                await self.services["policy_cache"].initialize()

            self.services["policy_materializer"] = PolicyMaterializer(
                kafka_config=dict(self.config.get("kafka", {})),
                policy_cache=self.services["policy_cache"],
            )
            await self.services["policy_materializer"].initialize()

        if self._service_enabled("monitoring"):
            self.services["monitoring"] = MonitoringService(
                config=self.config.get("monitoring", {})
            )
            await self.services["monitoring"].initialize()

        if self._service_enabled("slack"):
            self.services["slack_bot"] = SlackBot(
                config=self.config.get("slack", {}),
                inference_engine=self.services["inference"],
                router=self.services.get("router"),
                monitoring_service=self.services.get("monitoring"),
                analytics_service=self.services.get("pipeline"),
            )
            await self.services["slack_bot"].initialize()

        pipeline_service = self.services.get("pipeline")
        monitoring_service = self.services.get("monitoring")
        attach_monitoring = getattr(
            pipeline_service, "attach_monitoring_service", None
        )
        if (
            pipeline_service is not None
            and monitoring_service is not None
            and callable(attach_monitoring)
        ):
            attach_monitoring(monitoring_service)

    async def _initialize_services(
        self, include_api: bool = True, include_background: bool = True
    ):
        """Initialize platform services for the selected runtime mode."""
        self.logger.info("Initializing LLM Router Platform services...")

        needs_core_services = include_api or (
            include_background and self._service_enabled("slack")
        )
        if needs_core_services:
            await self._initialize_core_services()

        if include_background:
            await self._initialize_background_services()
        self.logger.info("Service initialization complete")

    def _start_metrics_server(self):
        """Start the Prometheus metrics server when monitoring is enabled."""
        if not self._service_enabled("monitoring"):
            return

        metrics_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
        start_http_server(metrics_port)
        self.logger.info(f"Prometheus metrics server started on port {metrics_port}")

    async def _start_services(
        self, include_api: bool = True, include_background: bool = True
    ):
        """Start selected platform services."""
        self.logger.info("Starting platform services...")

        tasks = []

        if include_background:
            self._start_metrics_server()

            if "pipeline" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["pipeline"].start(),
                        name="kafka_pipeline"
                    )
                )

            if "policy_materializer" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["policy_materializer"].start(),
                        name="policy_materializer",
                    )
                )

            if "monitoring" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["monitoring"].start(),
                        name="monitoring_service"
                    )
                )

            if "slack_bot" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["slack_bot"].start(),
                        name="slack_bot"
                    )
                )

        if include_api:
            api_config = self.config.get("api", {})
            tasks.append(
                asyncio.create_task(
                    self._start_api_server(api_config),
                    name="api_server"
                )
            )

        if not tasks:
            self.logger.warning("No services selected to start; exiting")
            return

        self.logger.info("Selected services started")

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            await self._shutdown_services()

    async def _start_api_server(self, api_config: Dict):
        """Start the FastAPI server"""
        app = self._create_fastapi_app()

        config = uvicorn.Config(
            app,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            loop="asyncio",
            log_level=api_config.get("log_level", "info")
        )

        server = uvicorn.Server(config)
        await server.serve()

    def _record_api_metrics(
        self, endpoint: str, method: str, status_code: int,
        duration_seconds: float, error_type: Optional[str] = None
    ):
        """Record request-level API metrics."""
        requests_total = getattr(SYSTEM_METRICS, "requests_total", None)
        request_duration = getattr(SYSTEM_METRICS, "request_duration", None)

        if requests_total is not None:
            requests_total.labels(
                endpoint=endpoint,
                method=method,
                status=str(status_code),
            ).inc()

        if request_duration is not None:
            request_duration.labels(
                endpoint=endpoint,
                method=method,
            ).observe(duration_seconds)

        if error_type:
            errors_total = getattr(SYSTEM_METRICS, "errors_total", None)
            if errors_total is not None:
                errors_total.labels(
                    component="api",
                    error_type=error_type,
                ).inc()

    def _get_service_status(self) -> Dict[str, bool]:
        """Return health status for initialized services."""
        service_status = {}
        for name, service in self.services.items():
            service_status[name] = getattr(service, "is_healthy", lambda: True)()
        return service_status

    def _build_health_payload(self) -> Dict[str, Any]:
        """Build health response payload."""
        service_status = self._get_service_status()
        return {
            "status": "healthy" if all(service_status.values()) else "unhealthy",
            "services": service_status,
            "timestamp": time.time(),
        }

    async def _publish_request_raw_event(self, query_request: QueryRequest):
        """Best-effort publication of pre-inference API events."""
        producer = self.services.get("event_producer")
        if producer is None:
            return

        publish_method = getattr(producer, "produce_request_raw", None)
        if publish_method is None:
            return

        try:
            await publish_method(query_request)
        except Exception as exc:
            self.logger.warning(f"Failed to publish requests.raw event: {exc}")

    def _iter_metric_samples(self, metric):
        """Iterate over metric children with their label values."""
        if metric is None:
            return

        label_names = tuple(getattr(metric, "_labelnames", ()) or ())
        metric_children = getattr(metric, "_metrics", None)

        if metric_children:
            for label_values, child in metric_children.items():
                if not isinstance(label_values, tuple):
                    label_values = (label_values,)
                yield dict(zip(label_names, label_values)), child
            return

        if hasattr(metric, "_value"):
            yield {}, metric

    def _child_value(self, child, attr_name: str = "_value") -> float:
        """Safely read a numeric Prometheus child metric value."""
        if child is None:
            return 0.0

        value_holder = getattr(child, attr_name, None)
        raw_value = getattr(value_holder, "_value", 0.0)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0

    def _sum_metric_values(self, metric) -> float:
        """Sum all child values for a metric."""
        total = 0.0
        for _labels, child in self._iter_metric_samples(metric) or []:
            total += self._child_value(child)
        return total

    def _histogram_average(self, histogram) -> float:
        """Compute histogram average across all labeled children."""
        total_sum = 0.0
        total_count = 0.0
        for _labels, child in self._iter_metric_samples(histogram) or []:
            total_sum += self._child_value(child, "_sum")
            total_count += self._child_value(child, "_count")
        return total_sum / total_count if total_count > 0 else 0.0

    def _sum_metric_by_label(self, metric, label_name: str) -> Dict[str, float]:
        """Aggregate a metric by label value."""
        totals: Dict[str, float] = {}
        for labels, child in self._iter_metric_samples(metric) or []:
            label_value = labels.get(label_name, "unknown")
            totals[label_value] = totals.get(label_value, 0.0) + self._child_value(child)
        return totals

    def _build_business_metrics(self) -> Dict[str, Any]:
        """Build business metrics directly from Prometheus collectors."""
        total_requests = self._sum_metric_values(getattr(SYSTEM_METRICS, "requests_total", None))
        total_errors = self._sum_metric_values(getattr(SYSTEM_METRICS, "errors_total", None))
        total_cost = self._sum_metric_values(getattr(INFERENCE_METRICS, "cost_total", None))
        cache_hits = self._sum_metric_values(getattr(INFERENCE_METRICS, "cache_hits", None))
        cache_misses = self._sum_metric_values(getattr(INFERENCE_METRICS, "cache_misses", None))
        total_cache_requests = cache_hits + cache_misses

        return {
            "total_requests_24h": total_requests,
            "total_cost_24h": total_cost,
            "average_response_time": self._histogram_average(
                getattr(SYSTEM_METRICS, "request_duration", None)
            ),
            "cache_hit_rate": (
                cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
            ),
            "error_rate": (total_errors / total_requests if total_requests > 0 else 0.0),
            "model_distribution": self._sum_metric_by_label(
                getattr(ROUTER_METRICS, "routing_decisions", None),
                "model",
            ),
            "user_tier_distribution": self._sum_metric_by_label(
                getattr(USER_METRICS, "users_by_tier", None),
                "user_tier",
            ),
        }

    def _build_system_snapshot(self) -> Dict[str, Any]:
        """Build a minimal system snapshot for the dashboard."""
        return {
            "cpu_usage": self._sum_metric_values(getattr(SYSTEM_METRICS, "cpu_usage", None)),
            "memory_usage_percent": self._sum_metric_values(
                getattr(SYSTEM_METRICS, "memory_usage_percent", None)
            ),
            "active_requests": self._sum_metric_values(
                getattr(SYSTEM_METRICS, "active_requests", None)
            ),
            "total_requests": self._sum_metric_values(
                getattr(SYSTEM_METRICS, "requests_total", None)
            ),
            "total_errors": self._sum_metric_values(
                getattr(SYSTEM_METRICS, "errors_total", None)
            ),
        }

    def _build_inference_snapshot(
        self, business_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a minimal inference snapshot for the dashboard."""
        return {
            "total_requests": self._sum_metric_values(
                getattr(INFERENCE_METRICS, "requests_total", None)
            ),
            "total_cost": business_metrics.get("total_cost_24h", 0.0),
            "cache_hit_rate": business_metrics.get("cache_hit_rate", 0.0),
            "error_rate": business_metrics.get("error_rate", 0.0),
        }

    def _normalize_model_performance(
        self, model_data: List[Dict[str, Any]], source: str
    ) -> List[Dict[str, Any]]:
        """Normalize model performance payload for the dashboard."""
        normalized_models = []
        for model in model_data:
            normalized_models.append(
                {
                    "model_name": model.get("model_name", "unknown"),
                    "provider": model.get("provider"),
                    "requests": int(model.get("requests", 0) or 0),
                    "success_rate": model.get("success_rate"),
                    "avg_latency_ms": model.get("avg_latency_ms"),
                    "tokens_per_second": model.get("tokens_per_second"),
                    "error_count": int(model.get("error_count", 0) or 0),
                    "total_cost": float(model.get("total_cost", 0.0) or 0.0),
                    "source": source,
                }
            )
        return normalized_models

    def _fallback_model_performance(self) -> List[Dict[str, Any]]:
        """Build model performance from in-memory Prometheus counters."""
        request_totals = self._sum_metric_by_label(
            getattr(ROUTER_METRICS, "routing_decisions", None),
            "model",
        )
        cost_totals = self._sum_metric_by_label(
            getattr(INFERENCE_METRICS, "cost_total", None),
            "model",
        )
        error_totals = self._sum_metric_by_label(
            getattr(INFERENCE_METRICS, "errors_total", None),
            "model",
        )
        models = sorted(set(request_totals) | set(cost_totals) | set(error_totals))
        model_data = []
        for model_name in models:
            model_data.append(
                {
                    "model_name": model_name,
                    "requests": int(request_totals.get(model_name, 0) or 0),
                    "success_rate": None,
                    "avg_latency_ms": None,
                    "tokens_per_second": None,
                    "error_count": int(error_totals.get(model_name, 0) or 0),
                    "total_cost": float(cost_totals.get(model_name, 0.0) or 0.0),
                }
            )
        return self._normalize_model_performance(model_data, source="in_memory_metrics")

    def _build_threshold_alerts(self) -> List[Dict[str, Any]]:
        """Convert threshold alerts to a dashboard-friendly format."""
        current_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        alerts = []
        business_metrics = self._build_business_metrics()
        thresholds = [
            (
                "cpu_usage",
                self._sum_metric_values(getattr(SYSTEM_METRICS, "cpu_usage", None)),
                80.0,
                95.0,
            ),
            (
                "memory_usage",
                self._sum_metric_values(
                    getattr(SYSTEM_METRICS, "memory_usage_percent", None)
                ),
                80.0,
                95.0,
            ),
            ("error_rate", business_metrics.get("error_rate", 0.0), 0.05, 0.10),
        ]

        for metric_key, value, warning_threshold, critical_threshold in thresholds:
            severity = None
            threshold = None
            if value >= critical_threshold:
                severity = "critical"
                threshold = critical_threshold
            elif value >= warning_threshold:
                severity = "warning"
                threshold = warning_threshold

            if severity is None:
                continue

            metric_name = metric_key.replace("_", " ").title()
            alerts.append(
                {
                    "source": "thresholds",
                    "severity": severity,
                    "title": f"{metric_name} threshold exceeded",
                    "description": (
                        f"{metric_name} is {value:.2f} "
                        f"(threshold: {threshold:.2f})"
                    )
                    if isinstance(value, (int, float))
                    and isinstance(threshold, (int, float))
                    else f"{metric_name} threshold exceeded",
                    "metric": metric_key,
                    "value": value,
                    "threshold": threshold,
                    "timestamp": current_time,
                }
            )
        return alerts

    def _normalize_monitoring_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize monitoring alerts from the monitoring service."""
        normalized = []
        for alert in alerts:
            normalized.append(
                {
                    "source": alert.get("source", "monitoring_service"),
                    "severity": alert.get("severity", "warning"),
                    "title": alert.get("rule_name", "Monitoring alert"),
                    "description": alert.get("description", "Alert triggered"),
                    "metric": alert.get("metric_name"),
                    "value": alert.get("current_value"),
                    "threshold": alert.get("threshold"),
                    "timestamp": alert.get("timestamp"),
                    "trigger_count": alert.get("trigger_count"),
                }
            )
        return normalized

    async def _build_dashboard_payload(self, hours: int = 24) -> Dict[str, Any]:
        """Build a dashboard-friendly live snapshot."""
        business_metrics = self._build_business_metrics()
        system_report = self._build_system_snapshot()
        user_report = {
            "users": {
                "by_tier": dict(business_metrics.get("user_tier_distribution", {})),
            }
        }
        hours = max(1, min(int(hours), 24 * 7))

        overview = {
            "total_requests": int(business_metrics.get("total_requests_24h", 0) or 0),
            "avg_latency_ms": (
                float(business_metrics.get("average_response_time", 0.0) or 0.0)
                * 1000
            ),
            "success_rate": max(
                0.0,
                1.0 - float(business_metrics.get("error_rate", 0.0) or 0.0),
            ),
            "total_cost": float(business_metrics.get("total_cost_24h", 0.0) or 0.0),
            "cache_hit_rate": float(business_metrics.get("cache_hit_rate", 0.0) or 0.0),
            "error_rate": float(business_metrics.get("error_rate", 0.0) or 0.0),
        }

        analytics = {
            "total_queries": overview["total_requests"],
            "total_cost": overview["total_cost"],
            "avg_latency_ms": overview["avg_latency_ms"],
            "success_rate": overview["success_rate"],
            "query_type_breakdown": {},
            "model_request_distribution": {},
            "model_cost_breakdown": {},
            "user_tier_distribution": dict(
                business_metrics.get("user_tier_distribution", {})
            ),
        }
        model_performance = self._fallback_model_performance()
        alerts = self._build_threshold_alerts()

        sources = {
            "overview": "in_memory_metrics",
            "analytics": "in_memory_metrics",
            "model_performance": "in_memory_metrics",
            "alerts": "thresholds",
            "logs": "structured_log_file",
        }

        for model in model_performance:
            model_name = model["model_name"]
            analytics["model_request_distribution"][model_name] = model["requests"]
            analytics["model_cost_breakdown"][model_name] = model["total_cost"]

        monitoring_dashboard = None
        if "monitoring" in self.services:
            monitoring_dashboard = await self.services["monitoring"].get_dashboard_data()
            recent_alerts = monitoring_dashboard.get("alert_status", {}).get(
                "recent_alerts", []
            )
            alerts.extend(self._normalize_monitoring_alerts(recent_alerts))
            sources["alerts"] = "thresholds+monitoring_service"

        if "pipeline" in self.services:
            pipeline_analytics = await self.services["pipeline"].get_query_analytics(
                hours=hours
            )
            pipeline_models = await self.services["pipeline"].get_model_performance(
                hours=hours
            )

            if pipeline_analytics:
                analytics.update(
                    {
                        "total_queries": int(
                            pipeline_analytics.get("total_queries", 0) or 0
                        ),
                        "total_tokens": int(
                            pipeline_analytics.get("total_tokens", 0) or 0
                        ),
                        "total_cost": float(
                            pipeline_analytics.get("total_cost", 0.0) or 0.0
                        ),
                        "avg_latency_ms": float(
                            pipeline_analytics.get("avg_latency", 0.0) or 0.0
                        ),
                        "success_rate": float(
                            pipeline_analytics.get("success_rate", 0.0) or 0.0
                        )
                        / 100.0,
                        "query_type_breakdown": dict(
                            pipeline_analytics.get("query_type_breakdown", {})
                        ),
                    }
                )
                model_breakdown = pipeline_analytics.get("model_breakdown", {})
                analytics["model_request_distribution"] = {
                    model_name: int(stats.get("queries", 0) or 0)
                    for model_name, stats in model_breakdown.items()
                }
                analytics["model_cost_breakdown"] = {
                    model_name: float(stats.get("cost", 0.0) or 0.0)
                    for model_name, stats in model_breakdown.items()
                }
                sources["analytics"] = "clickhouse"

            if pipeline_models:
                model_performance = self._normalize_model_performance(
                    pipeline_models, source="clickhouse"
                )
                sources["model_performance"] = "clickhouse"

        health = self._build_health_payload()
        return {
            "timestamp": time.time(),
            "time_window_hours": hours,
            "health": health,
            "overview": overview,
            "system": system_report,
            "inference": self._build_inference_snapshot(business_metrics),
            "user_report": user_report,
            "analytics": analytics,
            "model_performance": model_performance,
            "alerts": alerts,
            "sources": sources,
            "capabilities": {
                "pipeline_analytics": "pipeline" in self.services,
                "monitoring": "monitoring" in self.services,
                "logs": Path(
                    self.config.get("logging", {}).get("file", "logs/llm_router.log")
                ).exists(),
            },
            "monitoring": monitoring_dashboard,
        }

    def _parse_log_entry(self, raw_line: str) -> Optional[Dict[str, Any]]:
        """Parse a single structured log line."""
        if not raw_line.strip():
            return None

        try:
            record = json.loads(raw_line)
            logger_name = str(record.get("logger", "app"))
            return {
                "timestamp": record.get("timestamp"),
                "level": str(record.get("level", "INFO")).upper(),
                "component": logger_name.split(".")[-1] if logger_name else "app",
                "logger": logger_name,
                "message": record.get("message", ""),
                "request_id": record.get("request_id"),
                "file": record.get("file"),
            }
        except json.JSONDecodeError:
            return {
                "timestamp": None,
                "level": "INFO",
                "component": "app",
                "logger": "raw",
                "message": raw_line.strip(),
                "request_id": None,
                "file": None,
            }

    def _get_recent_logs(
        self,
        limit: int = 50,
        level: Optional[str] = None,
        component: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent structured log entries with optional filtering."""
        log_path = Path(self.config.get("logging", {}).get("file", "logs/llm_router.log"))
        if not log_path.exists():
            return []

        try:
            raw_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []

        filtered_logs = []
        max_logs = max(1, min(int(limit), 500))
        normalized_level = (level or "ALL").upper()
        normalized_component = (component or "ALL").lower()

        for raw_line in reversed(raw_lines):
            entry = self._parse_log_entry(raw_line)
            if not entry:
                continue

            if normalized_level != "ALL" and entry["level"] != normalized_level:
                continue

            if normalized_component != "all":
                component_value = entry["component"].lower()
                logger_value = entry["logger"].lower()
                if (
                    normalized_component not in component_value
                    and normalized_component not in logger_value
                ):
                    continue

            filtered_logs.append(entry)
            if len(filtered_logs) >= max_logs:
                break

        return filtered_logs

    def _create_fastapi_app(self, lifespan=None) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        app = FastAPI(
            title="LLM Router & Execution Platform",
            description="Production-grade multi-model deployment system",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return self._build_health_payload()

        # Model routing endpoint
        @app.post("/route")
        async def route_query(query_data: QueryRequest):
            endpoint = "/route"
            method = "POST"
            start_time = time.perf_counter()
            status_code = 500
            active_requests = getattr(SYSTEM_METRICS, "active_requests", None)
            if active_requests is not None:
                active_requests.labels(endpoint=endpoint).inc()

            try:
                await self._publish_request_raw_event(query_data)
                result = await self.services["inference"].process_query(query_data)
                status_code = 502 if result.error else 200
                return JSONResponse(
                    status_code=status_code,
                    content=jsonable_encoder(result),
                )
            except Exception as e:
                status_code = 500
                self.logger.error(f"Query processing error: {e}")
                return JSONResponse(
                    status_code=status_code,
                    content={"error": str(e)},
                )
            finally:
                self._record_api_metrics(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    duration_seconds=time.perf_counter() - start_time,
                    error_type="RouteError" if status_code >= 500 else None,
                )
                if active_requests is not None:
                    active_requests.labels(endpoint=endpoint).dec()

        # Metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            if "monitoring" not in self.services:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Monitoring service is disabled"},
                )
            return await self.services["monitoring"].get_system_metrics()

        @app.get("/dashboard")
        async def get_dashboard(hours: int = 24):
            return await self._build_dashboard_payload(hours=hours)

        @app.get("/dashboard/logs")
        async def get_dashboard_logs(
            limit: int = 50,
            level: Optional[str] = None,
            component: Optional[str] = None,
        ):
            logs = self._get_recent_logs(limit=limit, level=level, component=component)
            return {
                "logs": logs,
                "count": len(logs),
                "timestamp": time.time(),
            }

        return app

    async def _shutdown_services(self):
        """Gracefully shutdown all services"""
        self.logger.info("Shutting down platform services...")

        for name, service in self.services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                self.logger.info(f"Service {name} shutdown successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down service {name}: {e}")

        self.logger.info("Platform shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self._shutdown_services())
        sys.exit(0)

    async def run(self, include_api: bool = True, include_background: bool = True):
        """Run the selected platform services."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Initialize and start services
            await self._initialize_services(
                include_api=include_api,
                include_background=include_background,
            )
            await self._start_services(
                include_api=include_api,
                include_background=include_background,
            )

        except Exception as e:
            self.logger.error(f"Platform startup error: {e}")
            await self._shutdown_services()
            sys.exit(1)


@click.group()
def cli():
    """LLM Router Platform CLI"""
    pass


def create_app() -> FastAPI:
    """Create an API-only FastAPI app for Uvicorn factory mode."""
    platform = LLMRouterPlatform()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await platform._initialize_services(
            include_api=True,
            include_background=False,
        )
        yield
        await platform._shutdown_services()

    return platform._create_fastapi_app(lifespan=lifespan)


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
@click.option("--dev", is_flag=True, help="Run in development mode")
def start(config: str, dev: bool):
    """Start the LLM Router Platform"""
    if dev:
        os.environ[CONFIG_ENV_VAR] = config
        uvicorn.run(
            "main:create_app",
            host="0.0.0.0",
            port=8080,
            factory=True,
            reload=True,
            log_level="debug"
        )
    else:
        platform = LLMRouterPlatform(config_path=config)
        asyncio.run(platform.run(include_api=True, include_background=True))


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
@click.option("--dev", is_flag=True, help="Run API-only development mode")
def start_api(config: str, dev: bool):
    """Start only the API services."""
    os.environ[CONFIG_ENV_VAR] = config
    platform = LLMRouterPlatform(config_path=config)
    api_config = platform.config.get("api", {})
    worker_count = get_api_worker_count(platform.config, dev=dev)

    if dev:
        uvicorn.run(
            "main:create_app",
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            factory=True,
            reload=True,
            log_level=api_config.get("log_level", "debug"),
        )
        return

    uvicorn.run(
        "main:create_app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8080),
        factory=True,
        workers=worker_count,
        log_level=api_config.get("log_level", "info"),
    )


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG_PATH, help="Configuration file path")
def start_workers(config: str):
    """Start only enabled background services."""
    platform = LLMRouterPlatform(config_path=config)
    asyncio.run(platform.run(include_api=False, include_background=True))


@cli.command()
def setup():
    """Setup initial project structure and dependencies"""
    from src.llm_router_part0_setup import setup_project_environment
    setup_project_environment()


@cli.command()
@click.option("--service", help="Specific service to check (router, inference, pipeline, monitoring)")
def health(service: Optional[str]):
    """Check system health"""
    # Implementation for health checks
    click.echo(f"Health check for service: {service or 'all'}")


@cli.command()
@click.option("--output", default="docker-compose.yml", help="Output file for Docker Compose")
def deploy(output: str):
    """Generate deployment configurations"""
    from src.llm_router_part5_deploy import generate_deployment_configs
    generate_deployment_configs(output)


if __name__ == "__main__":
    cli()
