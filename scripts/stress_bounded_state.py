#!/usr/bin/env python3
"""Exercise production buffering against a blocked ClickHouse insert."""

import argparse
import asyncio
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
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


class FakeBrokerConsumer:
    """A max-poll sized broker fixture that stops fetching while paused."""

    def __init__(self, events: int, max_poll_records: int):
        self.partition = TopicPartition("requests", 0)
        self.events = events
        self.max_poll_records = max_poll_records
        self.delivered = 0
        self._poll_remaining = 0
        self.paused = False
        self.pause_observed = asyncio.Event()
        self.first_batch_ready = asyncio.Event()
        self.committed = {}

    def assignment(self):
        return {self.partition}

    def pause(self, *_partitions):
        self.paused = True
        self.pause_observed.set()

    def resume(self, *_partitions):
        self.paused = False
        self.pause_observed.clear()

    async def commit(self, offsets):
        self.committed.update(offsets)

    async def stop(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.delivered >= self.events:
            raise StopAsyncIteration
        if self._poll_remaining == 0:
            while self.paused:
                await asyncio.sleep(0.001)
            self._poll_remaining = min(
                self.max_poll_records, self.events - self.delivered
            )
        offset = self.delivered
        self.delivered += 1
        self._poll_remaining -= 1
        if self.delivered >= 100:
            self.first_batch_ready.set()
        await asyncio.sleep(0)
        value = {
            "query_id": f"query-{offset}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": "stress-user",
            "user_tier": "free",
            "query_text": "bounded-state stress",
            "query_type": "general",
            "selected_model": "fake",
            "token_count_input": 1,
            "token_count_output": 1,
            "latency_ms": 1.0,
            "cost_usd": 0.0,
            "status": "success",
        }
        return SimpleNamespace(
            topic="requests", partition=0, offset=offset, value=value, key=None
        )


class BlockedClickHouse:
    def __init__(self):
        self.insert_started = asyncio.Event()
        self.release_insert = asyncio.Event()
        self.outage = True
        self.inserted_offsets = []

    async def batch_insert_query_logs(self, _rows):
        if self.outage:
            self.insert_started.set()
            await self.release_insert.wait()
            raise RuntimeError("injected ClickHouse outage")
        self.inserted_offsets.extend(row.kafka_offset for row in _rows)

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
    clickhouse = BlockedClickHouse()
    manager = KafkaConsumerManager(
        {
            "consumer": {
                "enable_auto_commit": False,
                "batch_high_watermark_rows": high_watermark,
                "batch_low_watermark_rows": 2_500,
                "max_poll_records": max_poll_records,
            }
        },
        clickhouse,
    )
    broker = FakeBrokerConsumer(events, max_poll_records)
    manager.consumers["queries"] = broker
    rss_samples = [rss_bytes()]

    consume_task = asyncio.create_task(manager._consume_queries(broker))
    await asyncio.wait_for(broker.first_batch_ready.wait(), timeout=5)
    flush_task = asyncio.create_task(
        manager._flush_batch_topic("queries", clickhouse.batch_insert_query_logs)
    )
    await asyncio.wait_for(clickhouse.insert_started.wait(), timeout=5)
    pause_started = time.monotonic()
    await asyncio.wait_for(broker.pause_observed.wait(), timeout=5)
    await asyncio.sleep(0.05)
    rss_samples.append(rss_bytes())
    buffered_peak = manager._buffered_row_count("queries")
    delivered_at_first_pause = broker.delivered

    clickhouse.release_insert.set()
    failed_flush = await asyncio.gather(flush_task, return_exceptions=True)
    insert_failed = isinstance(failed_flush[0], RuntimeError)
    restored_rows = manager._buffered_row_count("queries")

    cache = BoundedTTLMap[str, int](
        max_entries=4_096, ttl_seconds=3_600, metric_name="stress_cache"
    )
    for index in range(events):
        cache[f"key-{index}"] = index
        if index and index % 10_000 == 0:
            rss_samples.append(rss_bytes())
    rss_samples.append(rss_bytes())

    clickhouse.outage = False
    await manager._flush_batch_topic("queries", clickhouse.batch_insert_query_logs)
    while not consume_task.done():
        pause_wait = asyncio.create_task(broker.pause_observed.wait())
        done, _ = await asyncio.wait(
            {consume_task, pause_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if pause_wait in done:
            await manager._flush_batch_topic(
                "queries", clickhouse.batch_insert_query_logs
            )
        else:
            pause_wait.cancel()
            await asyncio.gather(pause_wait, return_exceptions=True)
    await consume_task
    await manager._flush_batch_topic("queries", clickhouse.batch_insert_query_logs)
    shutdown_started = time.monotonic()
    await manager.shutdown()
    shutdown_duration = time.monotonic() - shutdown_started
    tail = rss_samples[-4:]
    rss_plateau = max(tail) - min(tail) <= 8 * 1024 * 1024
    result = {
        "events_offered": events,
        "events_delivered_by_broker": broker.delivered,
        "events_buffered_peak": buffered_peak,
        "rows_restored_after_insert_failure": restored_rows,
        "broker_backlog": events - broker.delivered,
        "events_inserted": len(clickhouse.inserted_offsets),
        "insertion_order_preserved": clickhouse.inserted_offsets == list(range(events)),
        "cache_entries_peak": len(cache),
        "cache_capacity": cache.max_entries,
        "oldest_item_age_seconds": cache.oldest_item_age(),
        "pause_duration_seconds": time.monotonic() - pause_started,
        "rejects": 0,
        "rss_start_bytes": rss_samples[0],
        "rss_peak_bytes": max(rss_samples),
        "rss_end_bytes": rss_samples[-1],
        "rss_plateau": rss_plateau,
        "shutdown_duration_seconds": shutdown_duration,
    }
    result["passed"] = bool(
        insert_failed
        and buffered_peak <= high_watermark + max_poll_records
        and restored_rows == delivered_at_first_pause
        and broker.delivered == events
        and clickhouse.inserted_offsets == list(range(events))
        and len(cache) <= cache.max_entries
        and broker.committed.get(broker.partition) == events
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
