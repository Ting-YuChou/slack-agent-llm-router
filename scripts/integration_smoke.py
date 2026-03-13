#!/usr/bin/env python3
"""
End-to-end smoke test for the Kafka -> ClickHouse ingestion pipeline.
"""

import argparse
import asyncio
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import clickhouse_connect

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_router_part3_pipeline import KafkaIngestionPipeline
from src.utils.schema import InferenceResponse, QueryRequest, QueryType, UserTier


def build_config(args, suffix: str):
    return {
        "bootstrap_servers": [args.kafka_bootstrap],
        "topics": {
            "queries": f"smoke-queries-{suffix}",
            "metrics": f"smoke-metrics-{suffix}",
        },
        "consumer": {
            "group_id": f"smoke-consumer-{suffix}",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": True,
            "max_poll_records": 50,
        },
        "clickhouse": {
            "host": args.clickhouse_host,
            "port": args.clickhouse_port,
            "database": f"llm_router_smoke_{suffix}",
            "username": args.clickhouse_user,
            "password": args.clickhouse_password,
        },
    }


async def wait_for(condition, timeout: float, interval: float = 0.5):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    return False


async def run_smoke(args):
    suffix = uuid.uuid4().hex[:8]
    config = build_config(args, suffix)
    pipeline = KafkaIngestionPipeline(config)
    start_task = None

    try:
        await pipeline.initialize()
        start_task = asyncio.create_task(pipeline.start())
        await asyncio.sleep(2)

        request = QueryRequest(
            query="Run an integration smoke test for Kafka and ClickHouse.",
            user_id="integration-smoke-user",
            user_tier=UserTier.PREMIUM,
            context="Synthetic test traffic",
            max_tokens=64,
            temperature=0.1,
        )
        response = InferenceResponse(
            response_text="Smoke test response",
            model_name="gpt-5",
            provider="openai",
            token_count_input=32,
            token_count_output=18,
            total_tokens=50,
            latency_ms=250,
            tokens_per_second=72.0,
            cost_usd=0.005,
        )
        decision = SimpleNamespace(query_type=QueryType.ANALYSIS)

        await pipeline.log_query(request, response, decision)
        await pipeline.log_metric(
            service="integration-smoke",
            metric_name="pipeline.smoke",
            metric_value=1.0,
            labels={"suite": "smoke", "suffix": suffix},
        )

        pending_ready = await wait_for(
            lambda: (
                bool(pipeline.consumer_manager.batch_processors["queries"])
                and bool(pipeline.consumer_manager.batch_processors["metrics"])
            ),
            timeout=args.timeout_seconds,
        )
        if not pending_ready:
            raise TimeoutError("Kafka consumers did not receive test messages in time")

        await pipeline.flush()
        await asyncio.sleep(1)

        query_count = pipeline.clickhouse_manager.client.query(
            "SELECT count() FROM query_logs"
        ).result_rows[0][0]
        metric_count = pipeline.clickhouse_manager.client.query(
            "SELECT count() FROM system_metrics"
        ).result_rows[0][0]

        if query_count < 1 or metric_count < 1:
            raise RuntimeError(
                f"Expected ClickHouse rows, got query_logs={query_count}, system_metrics={metric_count}"
            )

        print(
            f"Smoke test passed for database {config['clickhouse']['database']}: "
            f"query_logs={query_count}, system_metrics={metric_count}"
        )
    finally:
        if start_task is not None:
            await pipeline.shutdown()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(start_task, timeout=10)
            if not start_task.done():
                start_task.cancel()
                with suppress(asyncio.CancelledError):
                    await start_task
        with suppress(Exception):
            admin_client = clickhouse_connect.get_client(
                host=config["clickhouse"]["host"],
                port=config["clickhouse"]["port"],
                username=config["clickhouse"]["username"],
                password=config["clickhouse"]["password"],
                database="default",
            )
            try:
                admin_client.command(f"DROP DATABASE IF EXISTS {config['clickhouse']['database']}")
            finally:
                admin_client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kafka -> ClickHouse smoke test")
    parser.add_argument("--kafka-bootstrap", default="localhost:9092")
    parser.add_argument("--clickhouse-host", default="localhost")
    parser.add_argument("--clickhouse-port", type=int, default=8123)
    parser.add_argument("--clickhouse-user", default="llm_router")
    parser.add_argument("--clickhouse-password", default="llm_router_pass")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_smoke(parse_args()))
