"""Round-1 review fixes for A11 readiness."""

from types import SimpleNamespace

import pytest

import main
from src.llm_router_part3_pipeline import KafkaIngestionPipeline


class _FailingPingClient:
    def ping(self):
        raise ConnectionError("clickhouse down")


def _readiness(services):
    platform = SimpleNamespace(
        services=services,
        _api_key_auth_configured=lambda: True,
        _api_key_auth_enabled=lambda: False,
        _api_key_header_name=lambda: "X-API-Key",
    )
    platform._get_service_status = lambda: main.LLMRouterPlatform._get_service_status(
        platform
    )
    return main.LLMRouterPlatform._build_readiness_payload(platform)


def _healthy():
    return SimpleNamespace(is_healthy=lambda: True)


# --- A11 review 1: a ClickHouse ping failure degrades, it does not unready ---


@pytest.mark.asyncio
async def test_clickhouse_ping_failure_degrades_pipeline_without_failing_ready():
    pipeline = KafkaIngestionPipeline({"clickhouse": {}})
    pipeline.running = True
    pipeline.producer_manager.producer = object()
    pipeline.consumer_manager = SimpleNamespace(consumers={"queries": object()})
    pipeline.clickhouse_manager.client = _FailingPingClient()

    assert await pipeline.clickhouse_manager.ping() is False

    assert pipeline.get_health_status()["clickhouse_healthy"] is False
    assert pipeline.is_healthy() is True
    assert pipeline.is_degraded() is True

    payload = _readiness(
        {"router": _healthy(), "inference": _healthy(), "pipeline": pipeline}
    )
    assert payload["status"] == "ready"
    assert payload["unhealthy_services"] == []
    assert "pipeline" in payload["degraded_services"]
    assert payload["analytics"]["status"] == "degraded"


def test_pipeline_without_ping_is_not_degraded():
    pipeline = KafkaIngestionPipeline({"clickhouse": {}})
    pipeline.clickhouse_manager.client = object()
    assert pipeline.is_degraded() is False


# --- A11 review 2: policy_materializer degrades, it does not gate readiness ---


def test_unhealthy_policy_materializer_is_degraded_not_required():
    payload = _readiness(
        {
            "router": _healthy(),
            "inference": _healthy(),
            "policy_materializer": SimpleNamespace(is_healthy=lambda: False),
        }
    )
    assert payload["status"] == "ready"
    assert "policy_materializer" not in payload["required_services"]
    assert payload["degraded_services"] == ["policy_materializer"]
    assert payload["analytics"]["status"] == "degraded"
