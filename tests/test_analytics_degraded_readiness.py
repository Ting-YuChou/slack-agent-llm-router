"""A11: best-effort analytics must not block chat startup or readiness."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main
from src.llm_router_part3_pipeline import ClickHouseManager, KafkaIngestionPipeline


def _write_config(tmp_path: Path) -> Path:
    config = {
        "api": {"host": "127.0.0.1", "port": 8080, "log_level": "info"},
        "logging": {"level": "INFO", "file": str(tmp_path / "llm_router.log")},
        "router": {"default_model": "gpt-5"},
        "inference": {},
        "pipeline": {"enabled": False},
        "kafka": {
            "enabled": True,
            "bootstrap_servers": ["kafka:9092"],
            "producer": {
                "startup_timeout_seconds": 1,
                "startup_retry_initial_seconds": 0.01,
                "startup_retry_max_seconds": 0.02,
            },
        },
        "clickhouse": {"enabled": False},
        "monitoring": {"enabled": False},
        "slack": {"enabled": False},
        "flink": {"enabled": True},
        "policy_cache": {"enabled": False},
        "security": {"api_keys": {"enabled": False}, "cors": {"enabled": False}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class _Router:
    def __init__(self, config, policy_cache=None):
        self.config = config

    async def initialize(self):
        return None

    def is_healthy(self):
        return True


class _Inference:
    def __init__(self, config, router, event_producer=None, **_kwargs):
        self.event_producer = event_producer

    async def initialize(self):
        return None

    def is_healthy(self):
        return True


class _FlakyProducer:
    """Kafka producer whose first ``failures`` initialize calls fail."""

    failures = 2

    def __init__(self, config):
        self.config = config
        self.initialize_calls = 0
        self.shutdown_calls = 0
        self.started = False

    async def initialize(self):
        self.initialize_calls += 1
        if self.initialize_calls <= self.failures:
            raise ConnectionError("kafka unreachable")
        self.started = True

    def is_healthy(self):
        return True

    async def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main, "ModelRouter", _Router)
    monkeypatch.setattr(main, "InferenceEngine", _Inference)
    monkeypatch.setattr(main, "KafkaProducerManager", _FlakyProducer)
    return main.LLMRouterPlatform(config_path=str(_write_config(tmp_path)))


async def _wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_kafka_failure_at_startup_keeps_api_ready_and_retries(platform):
    await platform._initialize_services(include_api=True, include_background=False)

    producer = platform.services["event_producer"]
    assert producer.started is False
    assert platform.services["inference"].event_producer is producer

    readiness = platform._build_readiness_payload()
    assert readiness["status"] == "ready"
    assert "event_producer" not in readiness["required_services"]
    assert readiness["degraded_services"] == ["event_producer"]
    assert readiness["analytics"]["status"] == "degraded"

    await _wait_for(lambda: producer.started)

    readiness = platform._build_readiness_payload()
    assert readiness["status"] == "ready"
    assert readiness["degraded_services"] == []
    assert readiness["analytics"]["status"] == "ok"
    assert producer.initialize_calls == 3
    await platform._shutdown_services()


@pytest.mark.asyncio
async def test_unhealthy_event_producer_is_degraded_not_unready(platform):
    _FlakyProducer.failures = 0
    try:
        await platform._initialize_services(include_api=True, include_background=False)
    finally:
        _FlakyProducer.failures = 2
    platform.services["event_producer"].is_healthy = lambda: False

    readiness = platform._build_readiness_payload()

    assert readiness["status"] == "ready"
    assert readiness["unhealthy_services"] == []
    assert readiness["degraded_services"] == ["event_producer"]
    await platform._shutdown_services()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_producer_retry(platform):
    _FlakyProducer.failures = 10_000
    try:
        await platform._initialize_services(include_api=True, include_background=False)
        await platform._shutdown_services()
        producer = platform.services["event_producer"]
        calls = producer.initialize_calls
        await asyncio.sleep(0.05)
        assert producer.initialize_calls == calls
    finally:
        _FlakyProducer.failures = 2


def test_services_reporting_degraded_are_listed_without_failing_readiness():
    platform = SimpleNamespace(
        services={
            "router": SimpleNamespace(is_healthy=lambda: True),
            "inference": SimpleNamespace(is_healthy=lambda: True),
            "policy_cache": SimpleNamespace(
                is_healthy=lambda: True, is_degraded=lambda: True
            ),
        },
        _api_key_auth_configured=lambda: True,
        _api_key_auth_enabled=lambda: False,
        _api_key_header_name=lambda: "X-API-Key",
    )
    platform._get_service_status = lambda: main.LLMRouterPlatform._get_service_status(
        platform
    )

    payload = main.LLMRouterPlatform._build_readiness_payload(platform)

    assert payload["status"] == "ready"
    assert payload["degraded_services"] == ["policy_cache"]


class _PingClient:
    def __init__(self, ok=True, error=None):
        self.ok = ok
        self.error = error
        self.pings = 0

    def ping(self):
        self.pings += 1
        if self.error is not None:
            raise self.error
        return self.ok


@pytest.mark.asyncio
async def test_clickhouse_ping_failure_marks_clickhouse_unhealthy():
    pipeline = KafkaIngestionPipeline({"clickhouse": {}})
    pipeline.running = True
    pipeline.producer_manager.producer = object()
    pipeline.consumer_manager = SimpleNamespace(consumers={"queries": object()})
    client = _PingClient(error=ConnectionError("clickhouse down"))
    pipeline.clickhouse_manager.client = client

    assert pipeline.get_health_status()["clickhouse_healthy"] is True
    assert await pipeline.clickhouse_manager.ping() is False

    assert pipeline.get_health_status()["clickhouse_healthy"] is False
    # Review round 1: a failed ping degrades analytics, not readiness.
    assert pipeline.is_degraded() is True

    client.error = None
    assert await pipeline.clickhouse_manager.ping() is True
    assert pipeline.get_health_status()["clickhouse_healthy"] is True


@pytest.mark.asyncio
async def test_clickhouse_ping_returning_false_is_unhealthy():
    manager = ClickHouseManager({})
    manager.client = _PingClient(ok=False)

    assert await manager.ping() is False
    assert manager.is_connection_healthy() is False


@pytest.mark.asyncio
async def test_clickhouse_health_ping_loop_runs_while_pipeline_runs():
    manager = ClickHouseManager({"health_ping_interval_seconds": 0.01})
    client = _PingClient()
    manager.client = client

    manager.start_health_pings()
    await _wait_for(lambda: client.pings >= 2)
    await manager.stop_health_pings()
    pings = client.pings
    await asyncio.sleep(0.03)

    assert client.pings == pings
