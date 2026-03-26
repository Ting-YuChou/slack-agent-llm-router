from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from src.utils.schema import InferenceResponse


def _write_config(tmp_path: Path, overrides=None) -> Path:
    config = {
        "api": {"host": "127.0.0.1", "port": 8080, "log_level": "info"},
        "logging": {"level": "INFO", "file": str(tmp_path / "llm_router.log")},
        "router": {"default_model": "mistral-7b", "routing_strategy": "intelligent"},
        "inference": {},
        "pipeline": {"enabled": False},
        "kafka": {"enabled": False, "bootstrap_servers": ["localhost:9092"]},
        "clickhouse": {"enabled": False, "database": "llm_router"},
        "monitoring": {"enabled": False},
        "slack": {"enabled": False},
    }

    if overrides:
        config.update(overrides)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class DummyRouter:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return None


class DummyInferenceEngine:
    def __init__(self, config, router):
        self.config = config
        self.router = router
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


class DummyPipeline:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def shutdown(self):
        return None


class DummyMonitoring:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def get_system_metrics(self):
        return {"status": "ok"}

    async def shutdown(self):
        return None

    def is_healthy(self):
        return True


class DummySlackBot:
    def __init__(self, config, inference_engine):
        self.config = config
        self.inference_engine = inference_engine

    async def initialize(self):
        return None

    async def start(self):
        return None

    async def shutdown(self):
        return None


@pytest.fixture
def patched_platform_deps(monkeypatch):
    monkeypatch.setattr(main, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main, "ModelRouter", DummyRouter)
    monkeypatch.setattr(main, "InferenceEngine", DummyInferenceEngine)
    monkeypatch.setattr(main, "KafkaIngestionPipeline", DummyPipeline)
    monkeypatch.setattr(main, "MonitoringService", DummyMonitoring)
    monkeypatch.setattr(main, "SlackBot", DummySlackBot)


class TestPlatformInitialization:
    def test_get_api_worker_count_respects_dev_and_config(self):
        config = {"performance": {"workers": {"api_workers": 4}}}

        assert main.get_api_worker_count(config, dev=False) == 4
        assert main.get_api_worker_count(config, dev=True) == 1

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


class TestApiApp:
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

    def test_metrics_endpoint_returns_503_when_monitoring_disabled(
        self, tmp_path, patched_platform_deps
    ):
        config_path = _write_config(tmp_path)
        platform = main.LLMRouterPlatform(config_path=str(config_path))
        app = platform._create_fastapi_app()

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 503
