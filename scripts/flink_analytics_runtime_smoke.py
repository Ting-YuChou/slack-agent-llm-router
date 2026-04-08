#!/usr/bin/env python3
"""
Runtime smoke test for inference.completed -> analytics.model_metrics_1m using Dockerized Flink.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

ROOT = Path(__file__).resolve().parents[1]


def run_compose(args):
    result = subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose {' '.join(args)} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def create_topics(
    inference_completed_topic: str,
    analytics_metrics_topic: str,
    alerts_topic: str,
):
    for topic in (
        inference_completed_topic,
        analytics_metrics_topic,
        alerts_topic,
    ):
        run_compose(
            [
                "exec",
                "-T",
                "kafka",
                "kafka-topics",
                "--create",
                "--if-not-exists",
                "--bootstrap-server",
                "localhost:29092",
                "--topic",
                topic,
                "--partitions",
                "1",
                "--replication-factor",
                "1",
            ]
        )


def submit_flink_job(config: dict) -> str:
    config_json = json.dumps(config, separators=(",", ":"))
    result = run_compose(
        [
            "exec",
            "-T",
            "-e",
            f"FLINK_CONFIG_JSON={config_json}",
            "flink-jobmanager",
            "flink",
            "run",
            "-d",
            "-py",
            "/workspace/flink/analytics_job.py",
        ]
    )
    for line in result.stdout.splitlines():
        if "Job has been submitted with JobID" in line:
            return line.rsplit(" ", 1)[-1].strip()
    raise RuntimeError(f"Could not parse Flink JobID from output:\n{result.stdout}")


def cancel_flink_job(job_id: str):
    run_compose(["exec", "-T", "flink-jobmanager", "flink", "cancel", job_id])


async def wait_for_metrics_event(topic: str, group_id: str, timeout: float) -> dict:
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda data: json.loads(data.decode("utf-8")),
    )
    await consumer.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            batch = await consumer.getmany(timeout_ms=1000)
            for records in batch.values():
                for record in records:
                    value = record.value
                    if value.get("event_type") == "analytics.model_metrics_1m":
                        return value
        raise TimeoutError(
            f"Timed out waiting for analytics.model_metrics_1m event on topic {topic}"
        )
    finally:
        await consumer.stop()


async def produce_completion_events(topic: str, suffix: str):
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda data: json.dumps(data).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        compression_type="gzip",
    )
    await producer.start()
    try:
        for index in range(3):
            request_id = f"analytics-runtime-{suffix}-{index}"
            await producer.send_and_wait(
                topic,
                key=request_id,
                value={
                    "event_type": "inference.completed",
                    "event_version": "1.0",
                    "emitted_at": "2026-04-08T12:00:00+00:00",
                    "request_id": request_id,
                    "query_id": request_id,
                    "user_id": "runtime-user",
                    "user_tier": "free",
                    "selected_model": "mistral-7b",
                    "provider": "vllm",
                    "status": "success",
                    "latency_ms": 200 + (index * 50),
                    "token_count_input": 10,
                    "token_count_output": 20,
                    "total_tokens": 30,
                    "tokens_per_second": 100.0 + (index * 10),
                    "cost_usd": 0.0,
                    "cached_response": False,
                },
            )
    finally:
        await producer.stop()


async def run_smoke(timeout_seconds: float):
    suffix = uuid.uuid4().hex[:8]
    inference_completed_topic = f"flink-runtime-inference-completed-{suffix}"
    analytics_metrics_topic = f"flink-runtime-model-metrics-{suffix}"
    alerts_topic = f"flink-runtime-analytics-alerts-{suffix}"
    group_id = f"flink-runtime-analytics-group-{suffix}"
    config = {
        "parallelism": 1,
        "checkpoint_interval_ms": 10000,
        "window_size_seconds": 5,
        "anomaly_threshold_multiplier": 2.0,
        "kafka": {
            "bootstrap_servers": ["kafka:29092"],
            "consumer_group": group_id,
            "auto_offset_reset": "earliest",
            "topics": {
                "inference_completed": inference_completed_topic,
                "analytics_model_metrics_1m": analytics_metrics_topic,
                "alerts": alerts_topic,
            },
        },
    }

    create_topics(inference_completed_topic, analytics_metrics_topic, alerts_topic)
    job_id = submit_flink_job(config)
    try:
        await asyncio.sleep(8)
        consumer_task = asyncio.create_task(
            wait_for_metrics_event(
                analytics_metrics_topic,
                f"{group_id}-consumer",
                timeout_seconds,
            )
        )
        await produce_completion_events(inference_completed_topic, suffix)
        result = await consumer_task
        print(
            f"Flink analytics runtime smoke test passed: metrics topic {analytics_metrics_topic} "
            f"received aggregate for model={result.get('model_name')}"
        )
    finally:
        cancel_flink_job(job_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference.completed -> analytics.model_metrics_1m smoke test"
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_smoke(parse_args().timeout_seconds))
