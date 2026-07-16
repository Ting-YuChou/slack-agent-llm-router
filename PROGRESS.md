# Project Progress

## 2026-07-16 — Operational RAG worker profile

- Changed: added explicit Docker environment overrides for RAG and its ingestion queue, a Compose overlay that enables both API and worker against Redis Stack, and a TTL-backed worker heartbeat used by the container healthcheck.
- Verification: new RED/GREEN tests cover environment wiring, overlay dependencies, and repeated heartbeat refresh; RAG/API/schema regression passed (`86 passed`), along with touched-file Black and `git diff --check`.
- Follow-up: embedding batching, atomic generation indexing, parallel retrieval, and streaming uploads remain in the next phases.

## 2026-07-14 — Redis control-plane performance gates and v1 cleanup

- Changed: added a real-Redis control-plane benchmark, dry-run-first v1 provider queue cleanup, and a per-process/provider polling coordinator that uses Redis-assigned queue scores while waking only the next local waiter. Redis transport timeouts no longer count healthy client/server queue delay as a socket blackhole.
- Key files: `src/admission.py`, `src/provider_scheduler.py`, `scripts/loadtest_redis_control_plane.py`, `scripts/cleanup_v1_provider_scheduler_keys.py`, `Makefile`, and matching tests.
- Verification: paired 1,000-waiter Redis benchmark passed with 1,000/1,000 successful requests and zero stale entries; commands/request fell from 494.63 to 49.23 (90.05%), commands/sec from 16,361.62 to 1,575.06 (90.37%), and slot-release p95 from 62.021 ms to 34.934 ms. Full suite passed with real Redis (`420 passed, 7 skipped` before final review fixes); Black, Flake8, mypy, compileall, and `git diff --check` passed. Independent task and whole-branch reviews approved after fixing cleanup races, explicit commands/sec gating, post-response circuit persistence semantics, and stale probe epoch handling.
- Follow-up: publish the reviewed branch when ready.

## 2026-07-12 — Resilient Redis admission connections

- Changed: added bounded Redis connect/read timeouts, a bounded connection pool, controller-level operation deadlines, and cooldown-gated single-flight recovery so transient failures recover without restarting workers.
- Key files: `src/admission.py`, `src/utils/schema.py`, `src/utils/metrics.py`, API rate-limit config, and admission/schema tests.
- Verification: `python -m pytest tests/test_admission.py tests/test_schema.py -q` (`57 passed`); Black, Flake8, mypy, and `git diff --check` passed; independent task review approved.
- Follow-up: atomic fast admission, v2 queue leases, and circuit transitions remain in the next tasks.

## 2026-07-12 — Atomic admission before queueing

- Changed: unified API/provider fast admission and queued admission in one Lua primitive; added v2 order/expiry/depth/sequence state, stale lease pruning, deterministic FIFO/priority ordering, adaptive jittered polling, and atomic terminal cleanup. Provider scheduling now bypasses the admission global queue.
- Key files: `src/admission.py`, `src/provider_scheduler.py`, scheduler/admission schema and metrics, scoped tests, and `tests/test_admission_redis_integration.py`.
- Verification: scoped suite (`81 passed`) and real Redis integration (`13 passed`); Black, Flake8, mypy, compileall, and `git diff --check` passed; independent review approved after two fix/re-review rounds.
- Follow-up: atomic circuit transitions, v1 key cleanup tooling, and performance gates remain.

## 2026-07-12 — Atomic provider circuit transitions

- Changed: replaced circuit read/modify/write operations with an epoch-aware Redis Lua state machine for acquire, success, failure, and probe release; streaming and non-streaming paths now propagate attempt timestamps and permits consistently.
- Key files: `src/provider_scheduler.py`, `src/llm_router_part2_inference.py`, scheduler schema, and scheduler/inference/real-Redis tests.
- Verification: scoped suite (`133 passed`) and real Redis integration (`15 passed`); Black, Flake8, mypy, compileall, and `git diff --check` passed; independent review approved after streaming, fallback-release, closed-mode, and probe-cap fixes.
- Follow-up: v1 cleanup tooling and Redis control-plane performance gates remain.

## 2026-07-15 — Analytics performance contracts and runtime verification

- Added: a machine-readable ClickHouse/Flink performance-contract runner with hard success/error gates, isolated benchmark databases, replay-before-merge verification, migration checksum execution, and deterministic hot-key state/throughput comparison.
- Runtime: made `/workspace` importable in both Flink containers and corrected uninitialized event-time watermark handling so buffered startup events are not misclassified as late.
- Measured: on a real 1M-row ClickHouse 24.8 workload with the legacy monthly partitioning preserved, the same one-hour `GROUPING SETS` query reduced `read_rows` from 581,479 to 8,039 (98.62%). Across 20 consistently cold, alternating-order samples with a verified non-empty model-performance section, the complete pre-change six-query sequential dashboard p95 was 253.52 ms and the cache-disabled production `KafkaIngestionPipeline.get_dashboard_bundle` four-section p95 was 23.81 ms (90.61% improvement); replay `FINAL` count was exactly one and all migration checks passed. The production `RollingScopePolicyEmitter` processed all 10k events in 235 ms, reduced serialized state by 99.96%, and emitted twice. Its legacy comparison faithfully executes JSON ListState decode/prune/rewrite and policy construction on a bounded 1k-event sample, then extrapolates the exact cumulative retained-entry work to 10k (1,247x measured-model improvement); neither path claims RocksDB/checkpoint-byte coverage.
- Review fixes: disabled generated ClickHouse sessions for safe concurrent dashboard reads; made migration cutover forward-only and restart-aware with an atomic `EXCHANGE TABLES` followed by resumable backup rename and fail-closed ambiguity handling; preserved legacy arrival-order identity semantics for out-of-order Flink events.
- Verification: focused analytics tests passed (`88 passed`) plus the benchmark-truthfulness regression suite (`21 passed`) and a real concurrent ClickHouse bundle test; the real Flink analytics runtime smoke passed after the review fixes with model metrics, model/provider guardrails, and user/session policy outputs; touched-file Black and Flake8 checks and `git diff --check` passed.

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
