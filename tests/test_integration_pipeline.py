from pathlib import Path
import asyncio
import socket
import subprocess
import sys
import uuid

import pytest

from src.llm_router_part3_pipeline import KafkaIngestionPipeline


ROOT = Path(__file__).resolve().parents[1]


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.integration
def test_kafka_messages_are_flushed_to_clickhouse():
    if not _port_open("localhost", 9092) or not _port_open("localhost", 8123):
        pytest.skip(
            "Kafka or ClickHouse is not running. Start `docker compose up -d kafka clickhouse` first."
        )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "integration_smoke.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"integration_smoke.py failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Smoke test passed" in result.stdout


@pytest.mark.integration
def test_dashboard_bundle_runs_four_real_clickhouse_queries_concurrently():
    if not _port_open("localhost", 8123):
        pytest.skip("ClickHouse is not running")

    database = f"dashboard_session_test_{uuid.uuid4().hex[:8]}"

    async def exercise():
        pipeline = KafkaIngestionPipeline(
            {
                "bootstrap_servers": ["localhost:9092"],
                "clickhouse": {
                    "host": "localhost",
                    "port": 8123,
                    "username": "llm_router",
                    "password": "llm_router_pass",
                    "database": database,
                    "dashboard": {
                        "cache_ttl_seconds": 0,
                        "max_concurrent_queries": 4,
                    },
                },
            }
        )
        manager = pipeline.clickhouse_manager
        await manager.initialize()
        try:
            bundle = await pipeline.get_dashboard_bundle(hours=1)
            assert bundle["errors"] == {}
            assert set(bundle) == {
                "analytics",
                "model_performance",
                "routing_guardrails",
                "routing_policy_state",
                "errors",
            }
        finally:
            await manager._run_blocking(
                manager.client.command, f"DROP DATABASE IF EXISTS {database} SYNC"
            )
            manager.shutdown()

    asyncio.run(exercise())
