from pathlib import Path
import socket
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.integration
def test_kafka_messages_are_flushed_to_clickhouse():
    if not _port_open("localhost", 9092) or not _port_open("localhost", 8123):
        pytest.skip("Kafka or ClickHouse is not running. Start `docker compose up -d kafka clickhouse` first.")

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
