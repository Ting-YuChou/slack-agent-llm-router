import uuid

import pytest
import redis.asyncio as redis

from scripts.loadtest_redis_control_plane import (
    command_delta,
    capture_redis_commands,
    evaluate_gates,
    monitored_command_summary,
    percentile,
    resolve_source_root,
    workload_signature,
)


def test_percentile_interpolates_and_handles_empty_samples():
    assert percentile([], 0.95) == 0.0
    assert percentile([10, 20, 30, 40], 0.50) == 25.0
    assert percentile([10, 20, 30, 40], 0.95) == pytest.approx(38.5)


def test_default_source_root_is_repository_root():
    root = resolve_source_root(None)

    assert (root / "src" / "admission.py").is_file()


def test_command_delta_reports_total_and_queue_sorted_set_commands():
    before = {
        "cmdstat_evalsha": {"calls": 10},
        "cmdstat_zadd": {"calls": 2},
        "cmdstat_zrange": {"calls": 4},
    }
    after = {
        "cmdstat_evalsha": {"calls": 15},
        "cmdstat_zadd": {"calls": 5},
        "cmdstat_zrange": {"calls": 6},
        "cmdstat_zrem": {"calls": 1},
    }

    delta = command_delta(before, after)

    assert delta["total"] == 11
    assert delta["queue_zset"] == 6
    assert delta["by_command"] == {
        "evalsha": 5,
        "zadd": 3,
        "zrange": 2,
        "zrem": 1,
    }


def test_monitor_summary_counts_lua_internal_queue_commands():
    commands = [
        {
            "command": "EVAL admission:atomic_admit 4 bench:scheduler:x:v2:queue:p:m:order"
        },
        {"command": "ZADD bench:scheduler:x:v2:queue:p:m:order 1 member"},
        {"command": "ZRANGE bench:scheduler:x:v2:queue:p:m:order 0 0"},
        {"command": "GET unrelated:key"},
    ]

    summary = monitored_command_summary(
        commands,
        key_prefixes=("bench:admission:x", "bench:scheduler:x"),
        queue_prefix="bench:scheduler:x:v2:queue:",
        elapsed_s=2.0,
    )

    assert summary["total"] == 3
    assert summary["queue_zset"] == 2
    assert summary["by_command"] == {"eval": 1, "zadd": 1, "zrange": 1}
    assert summary["commands_per_second"] == 1.5


@pytest.mark.asyncio
async def test_monitor_captures_sorted_set_command_executed_inside_lua():
    redis_url = "redis://localhost:6379/15"
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    key = f"bench:monitor-test:{uuid.uuid4().hex}:v2:queue:x:order"
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"real Redis unavailable at {redis_url}: {exc}")

    async def execute_lua():
        await client.eval("return redis.call('ZADD', KEYS[1], 1, 'member')", 1, key)

    try:
        events = await capture_redis_commands(redis_url, execute_lua)
    finally:
        await client.delete(key)
        await client.aclose()

    summary = monitored_command_summary(
        events,
        key_prefixes=("bench:monitor-test:",),
        queue_prefix="bench:monitor-test:",
        elapsed_s=1.0,
    )
    assert summary["queue_zset"] == 1
    assert summary["by_command"]["zadd"] == 1


def test_workload_signature_locks_all_comparison_parameters():
    mode = {
        "mode": "contended",
        "requests": 1000,
        "waiters": 1000,
        "active_limit": 1,
        "hold_ms": 1.0,
        "wait_timeout_ms": 300000,
    }

    assert workload_signature(mode) == mode


def test_baseline_gates_enforce_reduction_and_slot_release_latency():
    current = {
        "modes": {
            "uncontended": {
                "mode": "uncontended",
                "requests": 1000,
                "waiters": 1,
                "active_limit": 1,
                "hold_ms": 0.0,
                "wait_timeout_ms": 300000,
                "successes": 1000,
                "rejections": 0,
                "errors": 0,
                "commands": {"queue_zset": 0},
            },
            "contended": {
                "mode": "contended",
                "requests": 1000,
                "waiters": 1000,
                "active_limit": 1,
                "hold_ms": 1.0,
                "wait_timeout_ms": 300000,
                "successes": 1000,
                "rejections": 0,
                "errors": 0,
                "commands": {"total": 2000, "commands_per_second": 250},
                "slot_release_to_admit_ms": {"p95": 105},
            },
        }
    }
    baseline = {
        "modes": {
            "contended": {
                "mode": "contended",
                "requests": 1000,
                "waiters": 1000,
                "active_limit": 1,
                "hold_ms": 1.0,
                "wait_timeout_ms": 300000,
                "successes": 1000,
                "rejections": 0,
                "errors": 0,
                "commands": {"total": 8000, "commands_per_second": 1000},
                "slot_release_to_admit_ms": {"p95": 100},
            }
        }
    }

    gates = evaluate_gates(current, baseline)

    assert gates["passed"] is True
    assert gates["checks"]["uncontended_queue_zset_zero"]["passed"] is True
    assert gates["checks"]["contended_command_reduction"]["reduction"] == 0.75
    assert (
        gates["checks"]["contended_commands_per_second_reduction"]["reduction"] == 0.75
    )
    assert gates["checks"]["slot_release_p95"]["passed"] is True


def test_baseline_gates_fail_without_seventy_percent_reduction_or_p95_budget():
    current = {
        "modes": {
            "uncontended": {
                "mode": "uncontended",
                "requests": 100,
                "waiters": 1,
                "active_limit": 1,
                "hold_ms": 0.0,
                "wait_timeout_ms": 300000,
                "successes": 99,
                "rejections": 1,
                "errors": 0,
                "commands": {"queue_zset": 1},
            },
            "contended": {
                "mode": "contended",
                "requests": 100,
                "waiters": 100,
                "active_limit": 1,
                "hold_ms": 1.0,
                "wait_timeout_ms": 300000,
                "successes": 100,
                "rejections": 0,
                "errors": 0,
                "commands": {"total": 400, "commands_per_second": 400},
                "slot_release_to_admit_ms": {"p95": 112},
            },
        }
    }
    baseline = {
        "modes": {
            "contended": {
                "mode": "contended",
                "requests": 100,
                "waiters": 100,
                "active_limit": 1,
                "hold_ms": 1.0,
                "wait_timeout_ms": 300000,
                "successes": 100,
                "rejections": 0,
                "errors": 0,
                "commands": {"total": 1000, "commands_per_second": 1000},
                "slot_release_to_admit_ms": {"p95": 100},
            }
        }
    }

    gates = evaluate_gates(current, baseline)

    assert gates["passed"] is False
    assert gates["checks"]["all_requests_succeeded"]["passed"] is False
    assert gates["checks"]["contended_command_reduction"]["passed"] is False
    assert gates["checks"]["contended_commands_per_second_reduction"]["passed"] is False
    assert gates["checks"]["slot_release_p95"]["passed"] is False


def test_commands_per_second_gate_cannot_be_bypassed_by_a_slower_run():
    mode = {
        "mode": "contended",
        "requests": 100,
        "waiters": 100,
        "active_limit": 1,
        "hold_ms": 1.0,
        "wait_timeout_ms": 300000,
        "successes": 100,
        "rejections": 0,
        "errors": 0,
        "commands": {"total": 200, "commands_per_second": 800},
        "slot_release_to_admit_ms": {"p95": 100},
    }
    current = {
        "modes": {
            "uncontended": {
                **mode,
                "mode": "uncontended",
                "waiters": 1,
                "hold_ms": 0.0,
                "commands": {"queue_zset": 0},
            },
            "contended": mode,
        }
    }
    baseline_mode = {
        **mode,
        "commands": {"total": 1000, "commands_per_second": 1000},
    }

    gates = evaluate_gates(current, {"modes": {"contended": baseline_mode}})

    assert gates["checks"]["contended_command_reduction"]["passed"] is True
    assert gates["checks"]["contended_commands_per_second_reduction"]["passed"] is False
    assert gates["passed"] is False


def test_baseline_gates_reject_an_unsuccessful_baseline_workload():
    mode = {
        "mode": "contended",
        "requests": 10,
        "waiters": 10,
        "active_limit": 1,
        "hold_ms": 1.0,
        "wait_timeout_ms": 300000,
        "successes": 10,
        "rejections": 0,
        "errors": 0,
        "commands": {"total": 10, "commands_per_second": 10},
        "slot_release_to_admit_ms": {"p95": 1.0},
    }
    current = {
        "modes": {
            "uncontended": {
                **mode,
                "mode": "uncontended",
                "waiters": 1,
                "hold_ms": 0.0,
                "commands": {"queue_zset": 0, "total": 10},
            },
            "contended": mode,
        }
    }
    baseline_mode = {**mode, "successes": 9, "rejections": 1}

    gates = evaluate_gates(current, {"modes": {"contended": baseline_mode}})

    assert gates["passed"] is False
    assert gates["checks"]["baseline_all_requests_succeeded"]["passed"] is False
