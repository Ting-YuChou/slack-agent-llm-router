#!/usr/bin/env python3
"""
Runtime smoke test for Kafka -> Flink -> Kafka using Dockerized Flink.
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


def create_topics(query_topic: str, processed_topic: str, alerts_topic: str):
    for topic in (query_topic, processed_topic, alerts_topic):
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
            "/workspace/flink/job.py",
        ]
    )
    for line in result.stdout.splitlines():
        if "Job has been submitted with JobID" in line:
            return line.rsplit(" ", 1)[-1].strip()
    raise RuntimeError(f"Could not parse Flink JobID from output:\n{result.stdout}")


def cancel_flink_job(job_id: str):
    run_compose(["exec", "-T", "flink-jobmanager", "flink", "cancel", job_id])


async def wait_for_processed_message(topic: str, group_id: str, timeout: float) -> dict:
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
                    if value.get("fast_lane_processed") is True:
                        return value
        raise TimeoutError(f"Timed out waiting for fast-lane event on topic {topic}")
    finally:
        await consumer.stop()


async def produce_query(topic: str, suffix: str):
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda data: json.dumps(data).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        compression_type="gzip",
    )
    await producer.start()
    try:
        await producer.send_and_wait(
            topic,
            key=f"flink-runtime-{suffix}",
            value={
                "query_id": f"flink-runtime-{suffix}",
                "timestamp": "2026-03-13T12:00:00+00:00",
                "user_id": "runtime-user",
                "user_tier": "free",
                "query_text": "Critical production outage, respond ASAP",
                "query_type": "analysis",
                "selected_model": "gpt-5",
                "token_count_input": 10,
                "token_count_output": 5,
                "latency_ms": 100,
                "cost_usd": 0.001,
                "status": "success",
            },
        )
    finally:
        await producer.stop()


async def run_smoke(timeout_seconds: float):
    suffix = uuid.uuid4().hex[:8]
    query_topic = f"flink-runtime-queries-{suffix}"
    processed_topic = f"flink-runtime-processed-{suffix}"
    alerts_topic = f"flink-runtime-alerts-{suffix}"
    group_id = f"flink-runtime-group-{suffix}"
    config = {
        "parallelism": 1,
        "checkpoint_interval_ms": 10000,
        "kafka": {
            "bootstrap_servers": ["kafka:29092"],
            "consumer_group": group_id,
            "auto_offset_reset": "earliest",
            "topics": {
                "queries": query_topic,
                "processed": processed_topic,
                "alerts": alerts_topic,
            },
        },
    }

    create_topics(query_topic, processed_topic, alerts_topic)
    job_id = submit_flink_job(config)
    try:
        await asyncio.sleep(8)
        consumer_task = asyncio.create_task(
            wait_for_processed_message(processed_topic, f"{group_id}-consumer", timeout_seconds)
        )
        await produce_query(query_topic, suffix)
        result = await consumer_task
        print(
            f"Flink runtime smoke test passed: processed topic {processed_topic} "
            f"received fast-lane event for query_id={result.get('query_id')}"
        )
    finally:
        cancel_flink_job(job_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kafka -> Flink -> Kafka runtime smoke test")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_smoke(parse_args().timeout_seconds))
