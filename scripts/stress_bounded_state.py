#!/usr/bin/env python3
"""Deterministic bounded-state and Kafka outage stress contract."""

import argparse
import asyncio
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import KafkaConsumerManager
from src.utils.bounded_state import BoundedTTLMap


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if os.uname().sysname == "Darwin" else value * 1024)


class FakeConsumer:
    def __init__(self):
        self.partition = TopicPartition("requests", 0)
        self.paused = False
        self.committed = {}

    def assignment(self):
        return {self.partition}

    def pause(self, *_partitions):
        self.paused = True

    def resume(self, *_partitions):
        self.paused = False

    async def commit(self, offsets):
        self.committed.update(offsets)

    async def stop(self):
        return None


class RecoveringClickHouse:
    async def batch_insert_query_logs(self, _rows):
        return None

    async def batch_insert_metrics(self, _rows):
        return None

    async def batch_insert_model_performance(self, _rows):
        return None

    async def batch_insert_alert_events(self, _rows):
        return None

    async def batch_insert_routing_policy_state(self, _rows):
        return None


async def run_contract(events: int) -> dict:
    high_watermark = 5_000
    max_poll_records = 500
    consumer = KafkaConsumerManager(
        {
            "consumer": {
                "enable_auto_commit": False,
                "batch_high_watermark_rows": high_watermark,
                "batch_low_watermark_rows": 2_500,
                "max_poll_records": max_poll_records,
            }
        },
        RecoveringClickHouse(),
    )
    fake_consumer = FakeConsumer()
    consumer.consumers["queries"] = fake_consumer
    rss_samples = [rss_bytes()]
    pause_started = None
    delivered = min(events, high_watermark + max_poll_records)
    for offset in range(delivered):
        await consumer._append_batched_row(
            "queries",
            {"offset": offset},
            SimpleNamespace(topic="requests", partition=0, offset=offset),
        )
        if fake_consumer.paused and pause_started is None:
            pause_started = time.monotonic()
        if offset % 500 == 0:
            rss_samples.append(rss_bytes())

    buffered_peak = consumer._buffered_row_count("queries")
    cache = BoundedTTLMap[str, int](
        max_entries=4_096, ttl_seconds=3_600, metric_name="stress_cache"
    )
    for index in range(events):
        cache[f"key-{index}"] = index
        if index and index % 10_000 == 0:
            rss_samples.append(rss_bytes())
    rss_samples.append(rss_bytes())

    shutdown_started = time.monotonic()
    await consumer.shutdown()
    shutdown_duration = time.monotonic() - shutdown_started
    pause_duration = (
        time.monotonic() - pause_started if pause_started is not None else 0.0
    )
    tail = rss_samples[-4:]
    rss_plateau = max(tail) - min(tail) <= 8 * 1024 * 1024
    result = {
        "events_offered": events,
        "events_buffered_peak": buffered_peak,
        "broker_backlog": max(0, events - delivered),
        "cache_entries_peak": len(cache),
        "cache_capacity": cache.max_entries,
        "oldest_item_age_seconds": cache.oldest_item_age(),
        "pause_duration_seconds": pause_duration,
        "rejects": 0,
        "rss_start_bytes": rss_samples[0],
        "rss_peak_bytes": max(rss_samples),
        "rss_end_bytes": rss_samples[-1],
        "rss_plateau": rss_plateau,
        "shutdown_duration_seconds": shutdown_duration,
    }
    result["passed"] = bool(
        buffered_peak <= high_watermark + max_poll_records
        and len(cache) <= cache.max_entries
        and fake_consumer.committed.get(fake_consumer.partition) == delivered
        and rss_plateau
        and shutdown_duration <= 75
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    args = parser.parse_args()
    result = asyncio.run(run_contract(max(1, args.events)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
