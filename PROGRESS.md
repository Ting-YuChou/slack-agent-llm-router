# Project Progress

## 2026-07-15 — Analytics performance contracts and runtime verification

- Added: a machine-readable ClickHouse/Flink performance-contract runner with hard success/error gates, isolated benchmark databases, replay-before-merge verification, migration checksum execution, and deterministic hot-key state/throughput comparison.
- Runtime: made `/workspace` importable in both Flink containers and corrected uninitialized event-time watermark handling so buffered startup events are not misclassified as late.
- Measured: on a real 1M-row ClickHouse 24.8 workload, the same one-hour `GROUPING SETS` query reduced `read_rows` from 1,081,920 to 24,576 (97.73%); cold bundle p95 improved 88.02%; replay `FINAL` count was exactly one; all migration checks passed; the benchmark measured four issued queries instead of assuming the count. The 10k production `RollingScopePolicyEmitter` harness reduced serialized state by 99.78%, improved throughput by 5.24x, and emitted twice; this harness does not claim RocksDB/checkpoint-byte coverage.
- Review fixes: disabled generated ClickHouse sessions for safe concurrent dashboard reads; made migration cutover forward-only and restart-aware with atomic multi-rename and fail-closed ambiguity handling; preserved legacy arrival-order identity semantics for out-of-order Flink events.
- Verification: focused analytics tests passed (`88 passed`) plus a real concurrent ClickHouse bundle test; the real Flink analytics runtime smoke passed after the review fixes with model metrics, model/provider guardrails, and user/session policy outputs; touched-file Black and Flake8 checks and `git diff --check` passed.

## 2026-07-15 — Incremental Flink rolling-policy state

- Changed: replaced per-event five-minute `ListState` rewrites with five-second event-time `MapState` buckets plus an incremental rolling aggregate; split event-list compatibility from aggregate policy construction; limited dirty hot-key emission to one update per five seconds after the initial emission; added 60-second watermark idleness.
- Migration: preserved the original ListState descriptor for savepoint restoration and lazily migrates only in-window events; legacy state is cleared only after v2 state is fully written, and partial writes are safely overwritten on retry.
- Verification: focused Flink, schema, late-event, cleanup, emit-cadence, and migration tests passed (`56 passed`); touched-file Black and Flake8 checks, compileall, and `git diff --check` passed.
- Follow-up: the real PyFlink savepoint restore and Kafka/Flink runtime smoke remain for the performance-contract phase because the local runtime was not running.

## 2026-07-15 — ClickHouse analytics read-path hardening

- Changed: moved analytics tables to time-first `ReplacingMergeTree` sorting; made dashboard reads replay-safe with `FINAL`; combined query analytics with `GROUPING SETS`; added a four-query concurrent dashboard bundle and a bounded five-second single-flight cache; fetched pipeline and monitoring bundles concurrently.
- Migration: added a dry-run-by-default side-by-side v2 migration with pause/flush/resume hooks, delta copy, exact-count/time-range/event/full-row checksum validation, atomic table exchange, seven-day backup retention, and explicit rollback planning.
- Verification: focused dashboard, schema, and migration regression suite passed (`129 passed`); touched-file Black and Flake8 checks, compileall, migration dry-run, and `git diff --check` passed.
- Follow-up: real ClickHouse replay, migration, pruning, and latency benchmarks remain for the analytics performance-contract phase because no Docker services were running during this phase.

## 2026-07-12 — Kafka and provider hot-path hardening

- Changed: made Kafka consumer batch persistence race-safe; made Redis policy materialization fail before offset commit; moved Kafka delivery/ack/retry into a bounded background dispatcher with explicit drain, abandonment, and producer-close deadlines; replaced pseudo batching with zero-wait single-flight ahead of provider scheduling; centralized retries under a shared 60-second deadline and transport-attempt budget.
- Key files: `src/llm_router_part3_pipeline.py`, `src/llm_router_part3_policy.py`, `src/llm_router_part2_inference.py`, `src/provider_scheduler.py`, `src/utils/schema.py`, and matching tests/config.
- Verification: `python -m pytest tests -q` (`351 passed, 7 skipped`); Black, Flake8, mypy, compileall, and `git diff --check` passed. Kafka/policy and provider hot-path changes passed independent reviews.
- Follow-up: live hot/unique `/route` load comparison was not run because no OpenAI, Anthropic, or local vLLM endpoint credentials/runtime were available. Deterministic tests cover delayed Kafka acknowledgements, queue overflow/shutdown, 20-way identical-request coalescing, shared retry budget, deadline, and cancellation behavior.
