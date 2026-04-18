import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from src.utils.schema import InferenceResponse


def _write_config(tmp_path: Path, overrides=None) -> Path:
    config = {
        "api": {"host": "127.0.0.1", "port": 8080, "log_level": "info"},
        "logging": {"level": "INFO", "file": str(tmp_path / "llm_router.log")},
        "router": {"default_model": "gpt-5", "routing_strategy": "intelligent"},
        "inference": {},
        "pipeline": {"enabled": False},
        "kafka": {
            "enabled": False,
            "bootstrap_servers": ["localhost:9092"],
            "topics": {
                "requests_raw": "requests.raw",
                "inference_completed": "inference.completed",
                "requests_enriched": "requests.enriched",
                "fast_lane_hints": "fast_lane_hints",
                "routing_guardrails": "routing.guardrails",
                "alerts": "alerts",
            },
        },
        "clickhouse": {"enabled": False, "database": "llm_router"},
        "monitoring": {"enabled": False},
        "slack": {"enabled": False},
        "flink": {"enabled": False},
        "policy_cache": {"enabled": False},
        "security": {
            "api_keys": {
                "enabled": False,
                "header_name": "X-API-Key",
                "env_var": "LLM_ROUTER_API_KEYS",
            },
            "cors": {"enabled": False},
        },
    }

    if overrides:
        config.update(overrides)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class DummyRouter:
    def __init__(self, config, policy_cache=None):
        self.config = config
        self.policy_cache = policy_cache

    async def initialize(self):
        return None

    def is_healthy(self):
        return True


class DummyInferenceEngine:
    def __init__(self, config, router, event_producer=None):
        self.config = config
        self.router = router
        self.event_producer = event_producer
        self.response = InferenceResponse(
            response_text="ok",
            model_name="gpt-5",
            provider="openai",
            token_count_input=1,
            token_count_output=1,
            total_tokens=2,
            latency_ms=10,
            tokens_per_second=100.0,
            cost_usd=0.01,
        )

    async def initialize(self):
        return None

    async def process_query(self, request):
        return self.response

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummyPipeline:
    def __init__(self, config):
        self.config = config
        self.attached_monitoring = None

    async def initialize(self):
        return None

    async def start(self):
        return None

    def attach_monitoring_service(self, monitoring_service):
        self.attached_monitoring = monitoring_service

    async def get_query_analytics(self, user_id=None, hours=24):
        return {
            "total_queries": 12,
            "total_tokens": 240,
            "total_cost": 1.75,
            "avg_latency": 321.0,
            "success_rate": 99.0,
            "model_breakdown": {
                "gpt-5": {"queries": 7, "cost": 1.25},
                "mistral-7b": {"queries": 5, "cost": 0.5},
            },
            "query_type_breakdown": {"general": 8, "analysis": 4},
        }

    async def get_model_performance(self, hours=24):
        return [
            {
                "model_name": "gpt-5",
                "requests": 7,
                "success_rate": 99.5,
                "avg_latency_ms": 412.0,
                "tokens_per_second": 155.0,
                "error_count": 0,
                "total_cost": 1.25,
            }
        ]

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummyMonitoring:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def get_system_metrics(self):
        return {"status": "ok"}

    async def get_dashboard_data(self):
        return {
            "alert_status": {
                "recent_alerts": [
                    {
                        "rule_name": "high_latency",
                        "severity": "warning",
                        "description": "Latency exceeded threshold",
                        "current_value": 2500,
                        "threshold": 2000,
                        "timestamp": "2026-04-04T00:00:00Z",
                        "trigger_count": 2,
                    }
                ]
            }
        }

    async def ingest_stream_model_metrics(self, metric_event):
        return metric_event

    async def ingest_stream_alert(self, alert_event):
        return alert_event

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummySlackBot:
    def __init__(
        self,
        config,
        inference_engine,
        services=None,
        router=None,
        monitoring_service=None,
        analytics_service=None,
    ):
        self.config = config
        self.inference_engine = inference_engine
        resolved_services = services or {}
        self.services = resolved_services
        self.router = router or resolved_services.get("router")
        self.monitoring_service = monitoring_service or resolved_services.get(
            "monitoring"
        )
        self.analytics_service = analytics_service or resolved_services.get("pipeline")

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def shutdown(self):
        return None


class DummyEventProducer:
    def __init__(self, config):
        self.config = config
        self.request_events = []
        self.completion_events = []

    async def initialize(self):
        return None

    async def produce_request_raw(self, request):
        self.request_events.append(request)

    async def produce_inference_completed(
        self, request, response, routing_decision=None
    ):
        self.completion_events.append((request, response, routing_decision))

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummyPolicyCache:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return None

    async def get_effective_policy(self, request_id, user_id):
        return {}

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummyPolicyMaterializer:
    def __init__(self, kafka_config, policy_cache):
        self.kafka_config = kafka_config
        self.policy_cache = policy_cache

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


@pytest.fixture
def patched_platform_deps(monkeypatch):
    monkeypatch.setattr(main, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main, "ModelRouter", DummyRouter)
    monkeypatch.setattr(main, "InferenceEngine", DummyInferenceEngine)
    monkeypatch.setattr(main, "KafkaIngestionPipeline", DummyPipeline)
    monkeypatch.setattr(main, "KafkaProducerManager", DummyEventProducer)
    monkeypatch.setattr(main, "RoutingPolicyCache", DummyPolicyCache)
    monkeypatch.setattr(main, "PolicyMaterializer", DummyPolicyMaterializer)
    monkeypatch.setattr(main, "MonitoringService", DummyMonitoring)
    monkeypatch.setattr(main, "SlackBot", DummySlackBot)


class TestPlatformInitialization:
    def test_get_api_worker_count_respects_dev_and_config(self):
        config = {"performance": {"workers": {"api_workers": 4}}}

        assert main.get_api_worker_count(config, dev=False) == 4
        assert main.get_api_worker_count(config, dev=True) == 1

    def test_invalid_config_exits_on_schema_validation(self, tmp_path):
        config_path = _write_config(tmp_path, overrides={"api": {"port": 70000}})

        with pytest.raises(SystemExit):
            main.LLMRouterPlatform(config_path=str(config_path))

    @pytest.mark.asyncio
    async def test_initialize_services_skips_disabled_background_services(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=True)

        assert set(platform.services) == {"router", "inference"}

    @pytest.mark.asyncio
    async def test_initialize_services_merges_pipeline_config(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "pipeline": {"enabled": True},
                "kafka": {
                    "enabled": True,
                    "bootstrap_servers": ["kafka:9092"],
                    "topics": {"queries": "queries"},
                },
                "clickhouse": {
                    "enabled": True,
                    "database": "analytics",
                    "host": "clickhouse",
                },
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=True)

        pipeline = platform.services["pipeline"]
        assert pipeline.config["bootstrap_servers"] == ["kafka:9092"]
        assert pipeline.config["clickhouse"]["database"] == "analytics"
        assert pipeline.config["clickhouse"]["host"] == "clickhouse"

    @pytest.mark.asyncio
    async def test_initialize_services_wires_pipeline_to_monitoring(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "pipeline": {"enabled": True},
                "monitoring": {"enabled": True},
                "kafka": {"enabled": True, "bootstrap_servers": ["kafka:9092"]},
                "clickhouse": {"enabled": True, "database": "analytics"},
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=True)

        assert (
            platform.services["pipeline"].attached_monitoring
            is platform.services["monitoring"]
        )

    @pytest.mark.asyncio
    async def test_initialize_services_adds_event_producer_when_flink_enabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "flink": {"enabled": True},
                "kafka": {"enabled": True, "bootstrap_servers": ["kafka:9092"]},
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=False)

        assert "event_producer" in platform.services
        assert (
            platform.services["inference"].event_producer
            is platform.services["event_producer"]
        )

    @pytest.mark.asyncio
    async def test_initialize_services_adds_policy_cache_when_enabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={"policy_cache": {"enabled": True}},
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=False)

        assert "policy_cache" in platform.services
        assert (
            platform.services["router"].policy_cache
            is platform.services["policy_cache"]
        )

    @pytest.mark.asyncio
    async def test_workers_only_mode_skips_core_services_when_not_needed(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={"monitoring": {"enabled": True}},
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=False, include_background=True)

        assert "router" not in platform.services
        assert "inference" not in platform.services
        assert "monitoring" in platform.services

    @pytest.mark.asyncio
    async def test_background_services_add_policy_materializer_when_enabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "policy_cache": {"enabled": True},
                "kafka": {
                    "enabled": True,
                    "bootstrap_servers": ["kafka:9092"],
                    "topics": {
                        "requests_enriched": "requests.enriched",
                        "fast_lane_hints": "fast_lane_hints",
                    },
                },
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=False, include_background=True)

        assert "policy_materializer" in platform.services
        assert (
            platform.services["policy_materializer"].policy_cache
            is platform.services["policy_cache"]
        )

    @pytest.mark.asyncio
    async def test_initialize_services_passes_service_registry_to_slack_bot(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={"slack": {"enabled": True}},
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))

        await platform._initialize_services(include_api=True, include_background=True)

        assert "slack_bot" in platform.services
        assert platform.services["slack_bot"].services is platform.services


class TestApiApp:
    def test_live_endpoint_is_public_and_ready_endpoint_reports_missing_services(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {"enabled": False},
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            live_response = client.get("/live")
            ready_response = client.get("/ready")
            health_response = client.get("/health")

        assert live_response.status_code == 200
        assert live_response.json()["status"] == "live"

        assert ready_response.status_code == 503
        ready_body = ready_response.json()
        assert ready_body["status"] == "not_ready"
        assert ready_body["missing_services"] == ["router", "inference"]
        assert ready_body["security"]["api_keys_configured"] is False

        assert health_response.status_code == 503
        assert health_response.json()["status"] == "not_ready"

    def test_ready_endpoint_returns_200_when_core_services_and_auth_are_configured(
        self, tmp_path, monkeypatch, patched_platform_deps
    ):
        monkeypatch.setenv("LLM_ROUTER_API_KEYS", "test-key")
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {"enabled": False},
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["router"] = DummyRouter({})
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["security"]["api_keys_configured"] is True

    def test_route_endpoint_requires_api_key_when_enabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {"enabled": False},
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello", "user_id": "u1"})

        assert response.status_code == 503
        assert response.json()["error"] == "auth_unavailable"

    def test_route_endpoint_accepts_configured_api_key(
        self, tmp_path, monkeypatch, patched_platform_deps
    ):
        monkeypatch.setenv("LLM_ROUTER_API_KEYS", "test-key")
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {"enabled": False},
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post(
                "/route",
                json={"query": "hello", "user_id": "u1"},
                headers={"X-API-Key": "test-key"},
            )

        assert response.status_code == 200
        assert response.json()["response_text"] == "ok"

    @pytest.mark.parametrize("path", ["/dashboard", "/dashboard/logs", "/metrics"])
    def test_management_endpoints_require_api_key_when_enabled(
        self, path, tmp_path, monkeypatch, patched_platform_deps
    ):
        monkeypatch.setenv("LLM_ROUTER_API_KEYS", "test-key")
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {"enabled": False},
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get(path)

        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    def test_route_endpoint_returns_200_for_success(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello", "user_id": "u1"})

        assert response.status_code == 200
        assert response.json()["response_text"] == "ok"

    def test_route_endpoint_returns_502_for_engine_error_payload(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        inference = DummyInferenceEngine({}, DummyRouter({}))
        inference.response = inference.response.model_copy(update={"error": "upstream"})
        platform.services["inference"] = inference
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello", "user_id": "u1"})

        assert response.status_code == 502
        assert response.json()["error"] == "upstream"

    def test_route_endpoint_returns_sanitized_error_for_unhandled_exception(
        self, tmp_path, patched_platform_deps
    ):
        class ExplodingInferenceEngine(DummyInferenceEngine):
            async def process_query(self, request):
                raise RuntimeError("database password leaked")

        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = ExplodingInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello", "user_id": "u1"})

        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "internal_server_error"
        assert body["message"] == "Request processing failed"
        assert "database password leaked" not in response.text

    def test_route_endpoint_validates_request_body(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello"})

        assert response.status_code == 422

    def test_route_endpoint_publishes_request_raw_when_event_producer_enabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        producer = DummyEventProducer({})
        platform.services["event_producer"] = producer
        platform.services["inference"] = DummyInferenceEngine(
            {},
            DummyRouter({}),
            event_producer=producer,
        )
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.post("/route", json={"query": "hello", "user_id": "u1"})

        assert response.status_code == 200
        assert len(producer.request_events) == 1
        assert producer.request_events[0].query == "hello"

    def test_metrics_endpoint_returns_503_when_monitoring_disabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 503

    def test_dashboard_endpoint_returns_live_bundle_without_optional_services(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/dashboard")

        assert response.status_code == 200
        body = response.json()
        assert body["time_window_hours"] == 24
        assert body["sources"]["analytics"] == "process_local_metrics"
        assert body["source_authority"]["analytics"] is True
        assert body["observability"]["authoritative"] is True
        assert body["capabilities"]["pipeline_analytics"] is False
        assert "overview" in body
        assert "alerts" in body

    def test_dashboard_endpoint_marks_process_local_metrics_non_authoritative_in_worker(
        self, tmp_path, monkeypatch, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()
        monkeypatch.setattr(
            main.mp, "current_process", lambda: SimpleNamespace(name="SpawnProcess-2")
        )

        with TestClient(app) as client:
            response = client.get("/dashboard")

        assert response.status_code == 200
        body = response.json()
        assert body["source_authority"]["analytics"] is False
        assert body["sources"]["alerts"] == "disabled_non_authoritative_process_local"
        assert body["observability"]["authoritative"] is False
        assert body["observability"]["metrics_scope"] == "process_local"
        assert body["alerts"] == []

    def test_dashboard_endpoint_prefers_pipeline_and_monitoring_data(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["pipeline"] = DummyPipeline({})
        platform.services["monitoring"] = DummyMonitoring({})
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/dashboard", params={"hours": 6})

        assert response.status_code == 200
        body = response.json()
        assert body["time_window_hours"] == 6
        assert body["sources"]["overview"] == "clickhouse"
        assert body["sources"]["analytics"] == "clickhouse"
        assert body["sources"]["model_performance"] == "clickhouse"
        assert body["sources"]["inference"] == "clickhouse"
        assert body["source_authority"]["analytics"] is True
        assert body["observability"]["authoritative"] is True
        assert body["capabilities"]["pipeline_analytics"] is True
        assert body["analytics"]["query_type_breakdown"]["general"] == 8
        assert body["overview"]["total_requests"] == 12
        assert body["model_performance"][0]["model_name"] == "gpt-5"
        assert any(alert["source"] == "monitoring_service" for alert in body["alerts"])

    def test_dashboard_logs_endpoint_reads_structured_logs(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        log_path = tmp_path / "llm_router.log"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-04T10:00:00Z",
                            "level": "INFO",
                            "logger": "service.router",
                            "message": "Routing completed",
                            "request_id": "req-1",
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-04T10:01:00Z",
                            "level": "ERROR",
                            "logger": "service.pipeline",
                            "message": "Batch insert failed",
                            "request_id": "req-2",
                        }
                    ),
                ]
            )
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get(
                "/dashboard/logs",
                params={"level": "ERROR", "component": "pipeline", "limit": 10},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        assert body["logs"][0]["component"] == "pipeline"
        assert body["logs"][0]["request_id"] == "req-2"

    def test_dashboard_logs_endpoint_returns_recent_logs_without_filters(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        log_path = tmp_path / "llm_router.log"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-04T10:00:00Z",
                            "level": "INFO",
                            "logger": "service.router",
                            "message": "Routing completed",
                            "request_id": "req-1",
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-04T10:01:00Z",
                            "level": "ERROR",
                            "logger": "service.pipeline",
                            "message": "Batch insert failed",
                            "request_id": "req-2",
                        }
                    ),
                ]
            )
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/dashboard/logs", params={"limit": 10})

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert body["logs"][0]["request_id"] == "req-2"
        assert body["logs"][1]["request_id"] == "req-1"

    def test_cors_preflight_respects_security_cors_config(
        self, tmp_path, monkeypatch, patched_platform_deps
    ):
        monkeypatch.setenv("LLM_ROUTER_API_KEYS", "test-key")
        config_path = _write_config(
            tmp_path,
            overrides={
                "security": {
                    "api_keys": {
                        "enabled": True,
                        "header_name": "X-API-Key",
                        "env_var": "LLM_ROUTER_API_KEYS",
                    },
                    "cors": {
                        "enabled": True,
                        "allow_origins": ["https://example.com"],
                        "allow_methods": ["POST"],
                        "allow_headers": ["X-API-Key", "Content-Type"],
                    },
                }
            },
        )
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        platform.services["inference"] = DummyInferenceEngine({}, DummyRouter({}))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.options(
                "/route",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-API-Key, Content-Type",
                },
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://example.com"
