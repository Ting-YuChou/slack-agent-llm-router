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
def test_flink_runtime_processes_query_from_kafka_back_to_kafka():
    if not _port_open("localhost", 9092) or not _port_open("localhost", 8081):
        pytest.skip(
            "Kafka or Flink is not running. Start `docker compose up -d kafka flink-jobmanager flink-taskmanager` first."
        )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "flink_runtime_smoke.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"flink_runtime_smoke.py failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Flink runtime smoke test passed" in result.stdout
