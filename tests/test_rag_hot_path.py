import asyncio
from pathlib import Path

import pytest
import yaml

import main
from src.rag.service import RagService


class _HeartbeatRedis:
    def __init__(self):
        self.calls = []

    async def set(self, key, value, *, ex):
        self.calls.append((key, value, ex))

    async def get(self, key):
        return None

    async def setex(self, key, ttl, value):
        return True


class _HeartbeatStore:
    def __init__(self, client):
        self.client = client

    async def initialize(self):
        return None

    async def shutdown(self):
        return None


def test_rag_environment_overrides_enable_service_and_queue(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "logging": {"file": str(tmp_path / "router.log")},
                "rag": {
                    "enabled": False,
                    "ingestion_queue": {"enabled": False},
                },
            }
        )
    )
    monkeypatch.setenv("LLM_ROUTER_RAG_ENABLED", "true")
    monkeypatch.setenv("LLM_ROUTER_RAG_QUEUE_ENABLED", "true")

    platform = main.LLMRouterPlatform(str(config_path))

    assert platform.config["rag"]["enabled"] is True
    assert platform.config["rag"]["ingestion_queue"]["enabled"] is True


def test_rag_compose_overlay_enables_api_and_worker():
    overlay = yaml.safe_load(Path("docker-compose.rag.yml").read_text())

    for service_name in ("api", "rag-workers"):
        environment = overlay["services"][service_name]["environment"]
        assert environment["LLM_ROUTER_RAG_ENABLED"] == "true"
        assert environment["LLM_ROUTER_RAG_QUEUE_ENABLED"] == "true"
        assert "redis-stack" in overlay["services"][service_name]["depends_on"]
    assert "healthcheck" in overlay["services"]["rag-workers"]


@pytest.mark.asyncio
async def test_worker_heartbeat_is_refreshed_with_ttl(tmp_path):
    redis = _HeartbeatRedis()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "ingestion_queue": {
                "enabled": True,
                "heartbeat_interval_seconds": 0.01,
                "heartbeat_ttl_seconds": 1,
            },
            "storage": {"staging_dir": str(tmp_path)},
        },
        vector_store=_HeartbeatStore(redis),
    )

    task = asyncio.create_task(service._worker_heartbeat_loop("worker-a"))
    await asyncio.sleep(0.025)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(redis.calls) >= 2
    assert all(call[0].endswith(":heartbeat:worker-a") for call in redis.calls)
    assert all(call[2] == 1 for call in redis.calls)
