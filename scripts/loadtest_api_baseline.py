#!/usr/bin/env python3
"""Simple API load-test baseline for the LLM Router route endpoint."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/route")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--mode",
        choices=("hot", "unique"),
        default="hot",
        help="hot reuses the same user/query to measure coalescing and cache-hit paths",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_payload(index: int, mode: str) -> Dict[str, object]:
    if mode == "hot":
        return {
            "query": "Summarize the deployment plan in one sentence.",
            "user_id": "loadtest-user",
            "temperature": 0.1,
            "max_tokens": 64,
        }

    return {
        "query": f"Summarize request {index} in one sentence.",
        "user_id": f"loadtest-user-{index % 100}",
        "temperature": 0.1,
        "max_tokens": 64,
    }


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


async def run_loadtest(args: argparse.Namespace) -> Dict[str, object]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(args.requests):
        queue.put_nowait(index)

    latencies_ms: List[float] = []
    status_counts: Dict[int, int] = {}
    network_errors = 0
    lock = asyncio.Lock()
    limits = httpx.Limits(
        max_connections=max(args.concurrency, 100),
        max_keepalive_connections=max(min(args.concurrency, 100), 20),
    )

    async def worker(client: httpx.AsyncClient):
        nonlocal network_errors
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            payload = build_payload(index, args.mode)
            started = time.perf_counter()
            try:
                response = await client.post(args.endpoint, json=payload)
                latency_ms = (time.perf_counter() - started) * 1000
                async with lock:
                    latencies_ms.append(latency_ms)
                    status_counts[response.status_code] = (
                        status_counts.get(response.status_code, 0) + 1
                    )
            except Exception:
                async with lock:
                    network_errors += 1
            finally:
                queue.task_done()

    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
        limits=limits,
    ) as client:
        workers = [
            asyncio.create_task(worker(client)) for _ in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
    elapsed = time.perf_counter() - started

    sorted_latencies = sorted(latencies_ms)
    summary = {
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "mode": args.mode,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 4),
        "achieved_rps": round(args.requests / elapsed, 2) if elapsed > 0 else 0.0,
        "network_errors": network_errors,
        "status_counts": status_counts,
        "latency_ms": {
            "avg": round(sum(sorted_latencies) / len(sorted_latencies), 2)
            if sorted_latencies
            else 0.0,
            "p50": round(percentile(sorted_latencies, 0.50), 2),
            "p95": round(percentile(sorted_latencies, 0.95), 2),
            "p99": round(percentile(sorted_latencies, 0.99), 2),
            "max": round(sorted_latencies[-1], 2) if sorted_latencies else 0.0,
        },
    }
    return summary


async def main():
    args = parse_args()
    summary = await run_loadtest(args)
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)

    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    asyncio.run(main())
