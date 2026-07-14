#!/usr/bin/env python3
"""Benchmark Redis admission/provider-scheduler control-plane overhead.

For a legacy baseline, create an origin/main worktree and run this same script
from that source root, then compare it from the feature worktree::

  git worktree add /tmp/router-main origin/main
  python scripts/loadtest_redis_control_plane.py --source-root /tmp/router-main \
    --output /tmp/redis-control-plane-main.json
  python scripts/loadtest_redis_control_plane.py \
    --baseline /tmp/redis-control-plane-main.json

Use a dedicated Redis database/server: commandstats are server-wide counters.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import redis.asyncio as redis


QUEUE_ZSET_COMMANDS = {
    "zadd",
    "zcard",
    "zrange",
    "zrangebyscore",
    "zrem",
    "zremrangebyscore",
    "zscore",
}


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower))


def _calls(stats: Dict[str, Any], command: str) -> int:
    return int(stats.get(f"cmdstat_{command}", {}).get("calls", 0) or 0)


def command_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    commands = sorted(
        {
            key.removeprefix("cmdstat_")
            for key in {*before, *after}
            if key.startswith("cmdstat_")
        }
    )
    by_command = {
        command: _calls(after, command) - _calls(before, command)
        for command in commands
        if _calls(after, command) - _calls(before, command) != 0
    }
    return {
        "total": sum(by_command.values()),
        "queue_zset": sum(
            count
            for command, count in by_command.items()
            if command in QUEUE_ZSET_COMMANDS
        ),
        "by_command": by_command,
    }


def monitored_command_summary(
    commands: Iterable[Dict[str, Any]],
    *,
    key_prefixes: Iterable[str],
    queue_prefix: str,
    elapsed_s: float,
) -> Dict[str, Any]:
    """Summarize MONITOR events, including commands invoked from Lua."""
    prefixes = tuple(key_prefixes)
    by_command: Dict[str, int] = {}
    queue_zset = 0
    for event in commands:
        raw = str(event.get("command", ""))
        if not any(prefix in raw for prefix in prefixes):
            continue
        command = raw.split(" ", 1)[0].lower()
        by_command[command] = by_command.get(command, 0) + 1
        if command in QUEUE_ZSET_COMMANDS and queue_prefix in raw:
            queue_zset += 1
    total = sum(by_command.values())
    return {
        "total": total,
        "queue_zset": queue_zset,
        "by_command": dict(sorted(by_command.items())),
        "commands_per_second": round(total / elapsed_s, 6) if elapsed_s else 0.0,
    }


_WORKLOAD_FIELDS = (
    "mode",
    "requests",
    "waiters",
    "active_limit",
    "hold_ms",
    "wait_timeout_ms",
)


def workload_signature(mode_result: Dict[str, Any]) -> Dict[str, Any]:
    return {field: mode_result[field] for field in _WORKLOAD_FIELDS}


def _mode_succeeded(result: Dict[str, Any]) -> bool:
    return (
        int(result["successes"]) == int(result["requests"])
        and int(result["rejections"]) == 0
        and int(result["errors"]) == 0
    )


async def capture_redis_commands(
    redis_url: str, operation: Callable[[], Awaitable[Any]]
) -> List[Dict[str, Any]]:
    """Capture one operation with a dedicated Redis MONITOR connection."""
    events: List[Dict[str, Any]] = []
    monitor_client = redis.Redis.from_url(redis_url, decode_responses=True)
    monitor_context = monitor_client.monitor()
    collector: Optional[asyncio.Task] = None
    entered = False
    try:
        monitor = await asyncio.wait_for(monitor_context.__aenter__(), timeout=2.0)
        entered = True

        async def collect() -> None:
            async for event in monitor.listen():
                events.append(event)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        await operation()
        await asyncio.sleep(0.01)
    finally:
        if collector is not None:
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)
        if entered:
            await asyncio.wait_for(monitor_context.__aexit__(None, None, None), 2.0)
        await monitor_client.aclose()
    return events


def evaluate_gates(
    current: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    modes = current["modes"]
    uncontended_zset = int(modes["uncontended"]["commands"]["queue_zset"])
    checks: Dict[str, Any] = {
        "uncontended_queue_zset_zero": {
            "passed": uncontended_zset == 0,
            "actual": uncontended_zset,
            "required": 0,
        }
    }
    workload_results = list(modes.values())
    checks["all_requests_succeeded"] = {
        "passed": all(_mode_succeeded(result) for result in workload_results),
        "modes": {
            result["mode"]: {
                field: result[field]
                for field in ("requests", "successes", "rejections", "errors")
            }
            for result in workload_results
        },
    }
    if baseline is not None:
        current_contended = modes["contended"]
        baseline_contended = baseline["modes"]["contended"]
        baseline_results = list(baseline["modes"].values())
        checks["baseline_all_requests_succeeded"] = {
            "passed": all(_mode_succeeded(result) for result in baseline_results),
            "modes": {
                result["mode"]: {
                    field: result[field]
                    for field in ("requests", "successes", "rejections", "errors")
                }
                for result in baseline_results
            },
        }
        checks["baseline_workload_compatible"] = {
            "passed": workload_signature(current_contended)
            == workload_signature(baseline_contended),
            "current": workload_signature(current_contended),
            "baseline": workload_signature(baseline_contended),
        }
        current_cpr = current_contended["commands"]["total"] / max(
            1, current_contended["requests"]
        )
        baseline_cpr = baseline_contended["commands"]["total"] / max(
            1, baseline_contended["requests"]
        )
        reduction = 1.0 - (current_cpr / baseline_cpr) if baseline_cpr > 0 else 0.0
        checks["contended_command_reduction"] = {
            "passed": baseline_cpr > 0 and reduction >= 0.70,
            "reduction": round(reduction, 6),
            "current_commands_per_request": round(current_cpr, 6),
            "baseline_commands_per_request": round(baseline_cpr, 6),
            "required_reduction": 0.70,
        }
        current_cps = float(current_contended["commands"]["commands_per_second"])
        baseline_cps = float(baseline_contended["commands"]["commands_per_second"])
        cps_reduction = 1.0 - (current_cps / baseline_cps) if baseline_cps > 0 else 0.0
        checks["contended_commands_per_second_reduction"] = {
            "passed": baseline_cps > 0 and cps_reduction >= 0.70,
            "reduction": round(cps_reduction, 6),
            "current_commands_per_second": round(current_cps, 6),
            "baseline_commands_per_second": round(baseline_cps, 6),
            "required_reduction": 0.70,
        }
        current_p95 = float(current_contended["slot_release_to_admit_ms"]["p95"])
        baseline_p95 = float(baseline_contended["slot_release_to_admit_ms"]["p95"])
        checks["slot_release_p95"] = {
            "passed": current_p95 <= baseline_p95 * 1.10,
            "current_ms": current_p95,
            "baseline_ms": baseline_p95,
            "maximum_ms": round(baseline_p95 * 1.10, 6),
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def resolve_source_root(source_root: Optional[Path]) -> Path:
    """Return an explicit legacy root or this script's repository root."""
    return (
        source_root.resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )


def _load_router_types(source_root: Optional[Path]):
    root = str(resolve_source_root(source_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    admission = importlib.import_module("src.admission")
    scheduler = importlib.import_module("src.provider_scheduler")
    schema = importlib.import_module("src.utils.schema")
    return (
        admission.RedisAdmissionController,
        scheduler.ProviderCapacityScheduler,
        schema.QueryRequest,
    )


async def _queue_stale_count(client: Any, prefix: str) -> int:
    total = 0
    async for key in client.scan_iter(match=f"{prefix}:v2:queue:*:expiry"):
        total += int(await client.zcard(key))
    return total


async def run_mode(
    *,
    client: Any,
    mode: str,
    requests: int,
    waiters: int,
    hold_ms: float,
    wait_timeout_ms: int,
    source_root: Optional[Path],
    redis_url: str,
) -> Dict[str, Any]:
    (
        RedisAdmissionController,
        ProviderCapacityScheduler,
        QueryRequest,
    ) = _load_router_types(source_root)
    run_id = uuid.uuid4().hex
    admission_prefix = f"bench:admission:{run_id}"
    scheduler_prefix = f"bench:scheduler:{run_id}"
    active_limit = 1 if mode == "contended" else max(1, waiters)
    controller = RedisAdmissionController(
        {
            "enabled": True,
            "failure_mode": "closed",
            "redis": {"key_prefix": admission_prefix},
            "queue": {"enabled": False, "control_plane_version": "v2"},
            "global_limits": {"active_requests": 0},
            "providers": {"openai": {"active_requests": active_limit}},
            "models": {"gpt-5": {"active_requests": active_limit}},
        },
        redis_client=client,
    )
    scheduler = ProviderCapacityScheduler(
        {
            "enabled": True,
            "queue_enabled": mode == "contended",
            "wait_timeout_ms": wait_timeout_ms,
            "poll_interval_ms": 25,
            "max_poll_interval_ms": 250,
            "poll_jitter_ratio": 0.2,
            "queue_lease_grace_ms": 1000,
            "control_plane_version": "v2",
            "failure_mode": "closed",
            "key_prefix": scheduler_prefix,
            "retry": {"max_attempts_per_request": 1},
            "circuit_breaker": {"enabled": False},
        },
        admission_controller=controller,
        redis_client=client,
    )

    latencies: List[float] = []
    queue_waits: List[float] = []
    release_to_admit: List[float] = []
    completion_order: List[int] = []
    successes = 0
    rejections = 0
    errors = 0
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, waiters))
    last_release_started: Optional[float] = None

    class FakeResponse:
        total_tokens = 1

    async def run_one(index: int) -> None:
        nonlocal successes, rejections, errors, last_release_started
        async with semaphore:
            request = QueryRequest(
                query=f"fixed fake request {index}",
                user_id=f"bench-{index}",
                max_tokens=1,
            )
            started = time.perf_counter()

            async def fake_provider():
                if hold_ms > 0:
                    await asyncio.sleep(hold_ms / 1000.0)
                return FakeResponse()

            try:
                lease = await scheduler.acquire(
                    request=request,
                    model_name="gpt-5",
                    provider="openai",
                    estimated_input_tokens=1,
                )
                admitted_at = time.perf_counter()
                if last_release_started is not None:
                    release_to_admit.append(
                        max(0.0, (admitted_at - last_release_started) * 1000.0)
                    )
                queue_waits.append(lease.wait_seconds * 1000.0)
                response = await fake_provider()
                last_release_started = time.perf_counter()
                await scheduler.release(lease, actual_tokens=response.total_tokens)
                async with lock:
                    successes += 1
                    completion_order.append(index)
            except Exception as exc:
                async with lock:
                    if exc.__class__.__name__ == "AdmissionRejectedError":
                        rejections += 1
                    else:
                        errors += 1
            finally:
                async with lock:
                    latencies.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()

    async def workload() -> None:
        await asyncio.gather(*(run_one(index) for index in range(requests)))

    events = await capture_redis_commands(redis_url, workload)
    elapsed = time.perf_counter() - started
    commands = monitored_command_summary(
        events,
        key_prefixes=(admission_prefix, scheduler_prefix),
        queue_prefix=f"{scheduler_prefix}:v2:queue:",
        elapsed_s=elapsed,
    )
    inversions = sum(
        1 for left, right in zip(completion_order, completion_order[1:]) if right < left
    )
    result = {
        "mode": mode,
        "requests": requests,
        "waiters": waiters,
        "active_limit": active_limit,
        "hold_ms": hold_ms,
        "wait_timeout_ms": wait_timeout_ms,
        "elapsed_s": round(elapsed, 6),
        "rps": round(requests / elapsed, 3) if elapsed else 0.0,
        "successes": successes,
        "rejections": rejections,
        "errors": errors,
        "commands": commands,
        "commands_per_request": round(commands["total"] / max(1, requests), 6),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
        },
        "queue_wait_ms": {
            "p50": round(percentile(queue_waits, 0.50), 3),
            "p95": round(percentile(queue_waits, 0.95), 3),
            "p99": round(percentile(queue_waits, 0.99), 3),
        },
        "slot_release_to_admit_ms": {
            "p50": round(percentile(release_to_admit, 0.50), 3),
            "p95": round(percentile(release_to_admit, 0.95), 3),
            "p99": round(percentile(release_to_admit, 0.99), 3),
        },
        "fairness": {
            "adjacent_inversions": inversions,
            "completion_count": len(completion_order),
        },
        "stale_count": await _queue_stale_count(client, scheduler_prefix),
    }
    async for key in client.scan_iter(match=f"bench:*:{run_id}*"):
        await client.delete(key)
    return result


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    try:
        await client.ping()
        modes = {}
        for mode in ("uncontended", "contended"):
            modes[mode] = await run_mode(
                client=client,
                mode=mode,
                requests=args.requests,
                waiters=1 if mode == "uncontended" else args.waiters,
                hold_ms=0.0 if mode == "uncontended" else args.hold_ms,
                wait_timeout_ms=args.wait_timeout_ms,
                source_root=args.source_root,
                redis_url=args.redis_url,
            )
    finally:
        await client.aclose()
    result: Dict[str, Any] = {
        "redis_url": args.redis_url,
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "modes": modes,
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    result["gates"] = evaluate_gates(result, baseline)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument(
        "--waiters",
        type=int,
        default=1000,
        help="maximum simultaneous calls; default creates 1000 Redis waiters",
    )
    parser.add_argument("--hold-ms", type=float, default=1.0)
    parser.add_argument("--wait-timeout-ms", type=int, default=300000)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.requests < 1 or args.waiters < 1:
        raise SystemExit("--requests and --waiters must be positive")
    result = await run_benchmark(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")
    if not result["gates"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
