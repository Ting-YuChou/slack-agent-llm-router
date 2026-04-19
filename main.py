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
import secrets
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
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from prometheus_client import start_http_server

# Import all our components
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_policy import PolicyMaterializer, RoutingPolicyCache
from src.llm_router_part3_pipeline import KafkaIngestionPipeline, KafkaProducerManager
from src.llm_router_part4_monitor import MonitoringService
from src.utils.logger import security_logger, setup_logging
from src.utils.metrics import (
    INFERENCE_METRICS,
    ROUTER_METRICS,
    SYSTEM_METRICS,
    USER_METRICS,
    histogram_average,
    sum_metric_by_label,
    sum_metric_values,
)
from src.utils.schema import PlatformConfig, QueryRequest
from slack.bot import SlackBot


DEFAULT_CONFIG_PATH = "config/config.yaml"
CONFIG_ENV_VAR = "LLM_ROUTER_CONFIG"
DEFAULT_API_KEY_ENV_VAR = "LLM_ROUTER_API_KEYS"
PUBLIC_ENDPOINTS = {"/live", "/ready", "/health"}


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
            log_file=self.config.get("logging", {}).get("file", "logs/llm_router.log"),
        )
        self.logger = logging.getLogger(__name__)
        self._log_configuration_warnings()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            validated = PlatformConfig.model_validate(raw_config)
            return validated.model_dump(mode="json")
        except FileNotFoundError:
            logging.getLogger(__name__).error(
                f"Configuration file not found: {self.config_path}"
            )
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.getLogger(__name__).error(f"Error parsing configuration: {e}")
            sys.exit(1)
        except ValidationError as e:
            logging.getLogger(__name__).error(
                "Invalid configuration in %s: %s", self.config_path, e
            )
            sys.exit(1)

    def _log_configuration_warnings(self):
        """Log high-risk runtime configuration warnings."""
        security_config = self._security_config()
        cors_config = dict(security_config.get("cors", {}))
        clickhouse_config = dict(self.config.get("clickhouse", {}))
        monitoring_config = dict(self.config.get("monitoring", {}))
        slack_config = dict(self.config.get("slack", {}))

        if cors_config.get("enabled", False) and cors_config.get("allow_origins") == [
            "*"
        ]:
            self.logger.warning(
                "CORS is configured with a wildcard origin; restrict allow_origins before production rollout."
            )

        if self._api_key_auth_enabled() and not self._configured_api_keys():
            self.logger.warning(
                "API key authentication is enabled but no keys were loaded from %s.",
                self._api_key_env_var_name(),
            )

        if clickhouse_config.get("enabled", False) and clickhouse_config.get(
            "password", ""
        ) in {"", "llm_router_pass"}:
            self.logger.warning(
                "ClickHouse is enabled with an empty or default password; move credentials to a secret-backed environment."
            )

        grafana_password = (monitoring_config.get("grafana", {}) or {}).get(
            "admin_password"
        )
        if monitoring_config.get("enabled", False) and grafana_password == "admin":
            self.logger.warning(
                "Monitoring is enabled with the default Grafana admin password."
            )

        if (
            slack_config.get("enabled", False)
            and slack_config.get("state_backend", "memory") == "memory"
        ):
            self.logger.warning(
                "Slack bot state is configured for in-memory storage; this is not horizontally scalable."
            )

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
                services=self.services,
                router=self.services.get("router"),
                monitoring_service=self.services.get("monitoring"),
                analytics_service=self.services.get("pipeline"),
            )
            await self.services["slack_bot"].initialize()

        pipeline_service = self.services.get("pipeline")
        monitoring_service = self.services.get("monitoring")
        attach_monitoring = getattr(pipeline_service, "attach_monitoring_service", None)
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
                        self.services["pipeline"].start(), name="kafka_pipeline"
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
                        self.services["monitoring"].start(), name="monitoring_service"
                    )
                )

            if "slack_bot" in self.services:
                tasks.append(
                    asyncio.create_task(
                        self.services["slack_bot"].start(), name="slack_bot"
                    )
                )

        if include_api:
            api_config = self.config.get("api", {})
            tasks.append(
                asyncio.create_task(
                    self._start_api_server(api_config), name="api_server"
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
            log_level=api_config.get("log_level", "info"),
        )

        server = uvicorn.Server(config)
        await server.serve()

    def _record_api_metrics(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
        error_type: Optional[str] = None,
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

    def _security_config(self) -> Dict[str, Any]:
        """Return security configuration."""
        return dict(self.config.get("security", {}))

    def _api_keys_config(self) -> Dict[str, Any]:
        """Return API key authentication config."""
        return dict(self._security_config().get("api_keys", {}))

    def _api_key_auth_enabled(self) -> bool:
        """Return whether API key auth is enabled."""
        return bool(self._api_keys_config().get("enabled", False))

    def _api_key_header_name(self) -> str:
        """Return the configured API key header name."""
        return str(self._api_keys_config().get("header_name", "X-API-Key"))

    def _api_key_env_var_name(self) -> str:
        """Return the environment variable containing accepted API keys."""
        return str(self._api_keys_config().get("env_var", DEFAULT_API_KEY_ENV_VAR))

    def _configured_api_keys(self) -> List[str]:
        """Load accepted API keys from the configured environment variable."""
        raw_value = os.getenv(self._api_key_env_var_name(), "")
        normalized = raw_value.replace("\n", ",")
        return [key.strip() for key in normalized.split(",") if key.strip()]

    def _api_key_auth_configured(self) -> bool:
        """Return whether auth is either disabled or fully configured."""
        return not self._api_key_auth_enabled() or bool(self._configured_api_keys())

    def _request_path_is_public(self, path: str) -> bool:
        """Return whether the request path is publicly accessible."""
        return path in PUBLIC_ENDPOINTS

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract an API key from supported request headers."""
        header_name = self._api_key_header_name()
        header_value = request.headers.get(header_name)
        if header_value:
            return header_value.strip()

        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()

        return None

    def _is_valid_api_key(self, provided_key: Optional[str]) -> bool:
        """Return whether the provided API key matches a configured key."""
        if not provided_key:
            return False

        for expected_key in self._configured_api_keys():
            if secrets.compare_digest(provided_key, expected_key):
                return True
        return False

    def _build_liveness_payload(self) -> Dict[str, Any]:
        """Build process liveness payload."""
        return {
            "status": "live",
            "timestamp": time.time(),
        }

    def _get_service_status(self) -> Dict[str, bool]:
        """Return health status for initialized services."""
        service_status = {}
        for name, service in self.services.items():
            service_status[name] = getattr(service, "is_healthy", lambda: True)()
        return service_status

    def _build_readiness_payload(self) -> Dict[str, Any]:
        """Build readiness response payload."""
        service_status = self._get_service_status()
        required_services = ["router", "inference"]
        missing_services = [
            service_name
            for service_name in required_services
            if service_name not in self.services
        ]
        unhealthy_services = [
            service_name
            for service_name in required_services
            if service_name in service_status and not service_status[service_name]
        ]
        api_key_auth_configured = self._api_key_auth_configured()
        ready = (
            not missing_services and not unhealthy_services and api_key_auth_configured
        )
        return {
            "status": "ready" if ready else "not_ready",
            "services": service_status,
            "required_services": required_services,
            "missing_services": missing_services,
            "unhealthy_services": unhealthy_services,
            "security": {
                "api_key_auth_enabled": self._api_key_auth_enabled(),
                "api_key_header_name": self._api_key_header_name(),
                "api_keys_configured": api_key_auth_configured,
            },
            "timestamp": time.time(),
        }

    def _build_health_payload(self) -> Dict[str, Any]:
        """Build backward-compatible health payload."""
        return self._build_readiness_payload()

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

    def _sum_metric_values(self, metric) -> float:
        """Sum all collected values for a counter or gauge metric."""
        return sum_metric_values(metric)

    def _histogram_average(self, histogram) -> float:
        """Compute histogram average using collected public samples."""
        return histogram_average(histogram)

    def _sum_metric_by_label(self, metric, label_name: str) -> Dict[str, float]:
        """Aggregate a metric by label value using collected public samples."""
        return sum_metric_by_label(metric, label_name)

    def _process_local_metrics_authoritative(self) -> bool:
        """Return whether process-local metrics represent the full serving instance."""
        return mp.current_process().name == "MainProcess"

    def _build_business_metrics(self) -> Dict[str, Any]:
        """Build business metrics directly from Prometheus collectors."""
        total_requests = self._sum_metric_values(
            getattr(SYSTEM_METRICS, "requests_total", None)
        )
        total_errors = self._sum_metric_values(
            getattr(SYSTEM_METRICS, "errors_total", None)
        )
        total_cost = self._sum_metric_values(
            getattr(INFERENCE_METRICS, "cost_total", None)
        )
        cache_hits = self._sum_metric_values(
            getattr(INFERENCE_METRICS, "cache_hits", None)
        )
        cache_misses = self._sum_metric_values(
            getattr(INFERENCE_METRICS, "cache_misses", None)
        )
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
            "error_rate": (
                total_errors / total_requests if total_requests > 0 else 0.0
            ),
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
            "cpu_usage": self._sum_metric_values(
                getattr(SYSTEM_METRICS, "cpu_usage", None)
            ),
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
        return self._normalize_model_performance(
            model_data, source="process_local_metrics"
        )

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
                        f"{metric_name} is {value:.2f} " f"(threshold: {threshold:.2f})"
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

    def _normalize_monitoring_alerts(
        self, alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        process_local_authoritative = self._process_local_metrics_authoritative()
        observability_warnings: List[str] = []
        user_report = {
            "users": {
                "by_tier": dict(business_metrics.get("user_tier_distribution", {})),
            }
        }
        hours = max(1, min(int(hours), 24 * 7))

        overview = {
            "total_requests": int(business_metrics.get("total_requests_24h", 0) or 0),
            "avg_latency_ms": (
                float(business_metrics.get("average_response_time", 0.0) or 0.0) * 1000
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
        alerts = self._build_threshold_alerts() if process_local_authoritative else []
        routing_features = {}
        routing_guardrails = {
            "guardrail_count": 0,
            "scope_breakdown": {},
            "trigger_breakdown": {},
            "recent_guardrails": [],
            "persisted_guardrails": [],
        }
        routing_policy_state = {
            "state_count": 0,
            "scope_breakdown": {},
            "query_complexity_breakdown": {},
            "dominant_query_type_breakdown": {},
            "burst_protection_count": 0,
            "enterprise_priority_count": 0,
            "route_to_fast_lane_count": 0,
            "top_preferred_models": {},
            "top_avoid_models": {},
            "top_avoid_providers": {},
            "recent_states": [],
            "persisted_states": [],
        }

        sources = {
            "overview": "process_local_metrics",
            "analytics": "process_local_metrics",
            "model_performance": "process_local_metrics",
            "inference": "process_local_metrics",
            "system": "process_local_metrics",
            "alerts": "process_local_thresholds",
            "routing_features": "unavailable",
            "routing_guardrails": "unavailable",
            "routing_policy_state": "unavailable",
            "logs": "structured_log_file",
        }
        source_authority = {
            "overview": process_local_authoritative,
            "analytics": process_local_authoritative,
            "model_performance": process_local_authoritative,
            "inference": process_local_authoritative,
            "system": process_local_authoritative,
            "alerts": process_local_authoritative,
            "logs": True,
        }

        if not process_local_authoritative:
            observability_warnings.append(
                "Dashboard metrics are process-local because the API is running in a worker process without pipeline-backed aggregation."
            )
            sources["alerts"] = "disabled_non_authoritative_process_local"

        for model in model_performance:
            model_name = model["model_name"]
            analytics["model_request_distribution"][model_name] = model["requests"]
            analytics["model_cost_breakdown"][model_name] = model["total_cost"]

        monitoring_dashboard = None
        if "monitoring" in self.services:
            monitoring_dashboard = await self.services[
                "monitoring"
            ].get_dashboard_data()
            recent_alerts = monitoring_dashboard.get("alert_status", {}).get(
                "recent_alerts", []
            )
            alerts.extend(self._normalize_monitoring_alerts(recent_alerts))
            sources["alerts"] = "thresholds+monitoring_service"
            source_authority["alerts"] = process_local_authoritative
            routing_features = dict(
                monitoring_dashboard.get("routing_features", {}) or {}
            )
            sources["routing_features"] = "monitoring_service"
            routing_guardrails.update(
                dict(monitoring_dashboard.get("routing_guardrails", {}) or {})
            )
            sources["routing_guardrails"] = "monitoring_service"
            routing_policy_state.update(
                dict(monitoring_dashboard.get("routing_policy_state", {}) or {})
            )
            sources["routing_policy_state"] = "monitoring_service"

        if "pipeline" in self.services:
            pipeline_analytics = await self.services["pipeline"].get_query_analytics(
                hours=hours
            )
            pipeline_models = await self.services["pipeline"].get_model_performance(
                hours=hours
            )
            get_routing_guardrails = getattr(
                self.services["pipeline"], "get_routing_guardrails", None
            )
            pipeline_guardrails = (
                await get_routing_guardrails(hours=hours)
                if callable(get_routing_guardrails)
                else []
            )
            get_routing_policy_state_events = getattr(
                self.services["pipeline"], "get_routing_policy_state_events", None
            )
            pipeline_policy_states = (
                await get_routing_policy_state_events(hours=hours)
                if callable(get_routing_policy_state_events)
                else []
            )

            if pipeline_analytics:
                overview.update(
                    {
                        "total_requests": int(
                            pipeline_analytics.get("total_queries", 0) or 0
                        ),
                        "avg_latency_ms": float(
                            pipeline_analytics.get("avg_latency", 0.0) or 0.0
                        ),
                        "success_rate": float(
                            pipeline_analytics.get("success_rate", 0.0) or 0.0
                        )
                        / 100.0,
                        "total_cost": float(
                            pipeline_analytics.get("total_cost", 0.0) or 0.0
                        ),
                        "cache_hit_rate": None,
                        "error_rate": max(
                            0.0,
                            1.0
                            - (
                                float(
                                    pipeline_analytics.get("success_rate", 0.0) or 0.0
                                )
                                / 100.0
                            ),
                        ),
                    }
                )
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
                sources["overview"] = "clickhouse"
                sources["analytics"] = "clickhouse"
                sources["inference"] = "clickhouse"
                source_authority["overview"] = True
                source_authority["analytics"] = True
                source_authority["inference"] = True

            if pipeline_models:
                model_performance = self._normalize_model_performance(
                    pipeline_models, source="clickhouse"
                )
                sources["model_performance"] = "clickhouse"
                source_authority["model_performance"] = True

            source_authority["alerts"] = bool(
                "monitoring" in self.services or process_local_authoritative
            )

            if pipeline_guardrails:
                routing_guardrails["persisted_guardrails"] = pipeline_guardrails
                if sources["routing_guardrails"] == "monitoring_service":
                    sources["routing_guardrails"] = "monitoring_service+clickhouse"
                else:
                    sources["routing_guardrails"] = "clickhouse"

            if pipeline_policy_states:
                routing_policy_state["persisted_states"] = pipeline_policy_states
                if sources["routing_policy_state"] == "monitoring_service":
                    sources["routing_policy_state"] = "monitoring_service+clickhouse"
                else:
                    sources["routing_policy_state"] = "clickhouse"

        inference_snapshot = self._build_inference_snapshot(business_metrics)
        if sources["inference"] == "clickhouse":
            inference_snapshot = {
                "total_requests": analytics.get("total_queries", 0),
                "total_cost": analytics.get("total_cost", 0.0),
                "cache_hit_rate": overview.get("cache_hit_rate"),
                "error_rate": overview.get("error_rate", 0.0),
            }

        health = self._build_health_payload()
        return {
            "timestamp": time.time(),
            "time_window_hours": hours,
            "health": health,
            "overview": overview,
            "system": system_report,
            "inference": inference_snapshot,
            "user_report": user_report,
            "analytics": analytics,
            "model_performance": model_performance,
            "alerts": alerts,
            "routing_features": routing_features,
            "routing_guardrails": routing_guardrails,
            "routing_policy_state": routing_policy_state,
            "sources": sources,
            "source_authority": source_authority,
            "observability": {
                "authoritative": all(
                    source_authority[key]
                    for key in ["overview", "analytics", "model_performance"]
                ),
                "metrics_scope": (
                    "durable_backend"
                    if "pipeline" in self.services
                    else "process_local"
                ),
                "warnings": observability_warnings,
            },
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
        log_path = Path(
            self.config.get("logging", {}).get("file", "logs/llm_router.log")
        )
        if not log_path.exists():
            return []

        try:
            raw_lines = log_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
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

        cors_config = dict(self._security_config().get("cors", {}))
        if cors_config.get("enabled", False):
            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(cors_config.get("allow_origins", ["*"])),
                allow_credentials=bool(cors_config.get("allow_credentials", False)),
                allow_methods=list(cors_config.get("allow_methods", ["GET", "POST"])),
                allow_headers=list(cors_config.get("allow_headers", ["*"])),
            )

        @app.middleware("http")
        async def enforce_api_auth(request: Request, call_next):
            if request.method == "OPTIONS" or self._request_path_is_public(
                request.url.path
            ):
                return await call_next(request)

            if not self._api_key_auth_enabled():
                return await call_next(request)

            source_ip = request.client.host if request.client else None
            if not self._api_key_auth_configured():
                security_logger.log_authentication_attempt(
                    user_id="anonymous",
                    success=False,
                    source_ip=source_ip,
                    path=request.url.path,
                    method=request.method,
                    reason="api_keys_not_configured",
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "auth_unavailable",
                        "message": "API authentication is enabled but no API keys are configured",
                    },
                )

            if self._is_valid_api_key(self._extract_api_key(request)):
                return await call_next(request)

            security_logger.log_authentication_attempt(
                user_id="anonymous",
                success=False,
                source_ip=source_ip,
                path=request.url.path,
                method=request.method,
                reason="invalid_or_missing_api_key",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "A valid API key is required",
                },
            )

        @app.get("/live")
        async def live_check():
            return self._build_liveness_payload()

        @app.get("/ready")
        async def ready_check():
            readiness = self._build_readiness_payload()
            status_code = 200 if readiness["status"] == "ready" else 503
            return JSONResponse(status_code=status_code, content=readiness)

        @app.get("/health")
        async def health_check():
            health = self._build_health_payload()
            status_code = 200 if health["status"] == "ready" else 503
            return JSONResponse(status_code=status_code, content=health)

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
            except Exception:
                status_code = 500
                self.logger.exception("Query processing error")
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "error": "internal_server_error",
                        "message": "Request processing failed",
                    },
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
                if hasattr(service, "shutdown"):
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
            log_level="debug",
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
@click.option(
    "--service",
    help="Specific service to check (router, inference, pipeline, monitoring)",
)
def health(service: Optional[str]):
    """Check system health"""
    # Implementation for health checks
    click.echo(f"Health check for service: {service or 'all'}")


@cli.command()
@click.option(
    "--output", default="docker-compose.yml", help="Output file for Docker Compose"
)
def deploy(output: str):
    """Generate deployment configurations"""
    from src.llm_router_part5_deploy import generate_deployment_configs

    generate_deployment_configs(output)


if __name__ == "__main__":
    cli()
