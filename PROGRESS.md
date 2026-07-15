# Project Progress

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
