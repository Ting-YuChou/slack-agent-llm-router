# Project Progress

## 2026-07-18 — Bounded local state and Slack work admission

- Added: a shared monotonic `BoundedTTLMap` with TTL-first pruning, LRU capacity eviction, response-safe copies at call sites, eviction callbacks, and low-cardinality entry/capacity/age/eviction metrics.
- Bounded: policy L1 (4,096), RAG jobs (1,000), RAG batches (100), web-search cache (512), and web-search limiter users (10,000). Active limiter users are never evicted to reset quota; new users receive a distinct capacity denial.
- Slack: replaced one-task-per-envelope fan-out with 16 fixed workers and a 256-item FIFO. Overflow is ACKed then receives a bounded-time visible busy reply; queue/running/reject/reply metrics were added.
- RAG Redis hygiene: batch membership now shares the batch TTL and dead-letter writes use bounded approximate `MAXLEN 10000`.
- Verification: bounded-state, policy, web-search, RAG, schema, and non-integration Slack regressions passed (`127 passed, 3 deselected`); the three deselected tests require the external Redis fixture already noted in baseline.

## 2026-07-18 — Lossless Kafka consumer backpressure

- Changed: introduced per-topic `BatchBufferState` with pending, in-flight, and awaiting-commit state; Kafka partitions pause at a 5,000-row high watermark and resume at 2,500 rows, including newly assigned partitions while paused.
- Isolation: ClickHouse flushes run with a four-topic concurrency bound, inspect every result before reporting failures, preserve row order on insert failure, and retry commit-only failures without duplicate inserts.
- Bounded I/O: ClickHouse connections now default to a 2-second connect timeout and 10-second send/receive timeout.
- Verification: RED/GREEN backpressure, in-flight accounting, concurrent-topic flush, durability regression, and schema tests passed (`93 passed`); touched-file Black, Flake8, and `git diff --check` passed.

## 2026-07-16 — RAG hot-path review hardening

- Correctness: embedding batch failures now retry through the scalar provider and fail ingestion before cutover if any required vector remains unavailable. Redis stages chunks plus table, row, figure, and visual resources under one document hash tag, validates every staging key, and switches them in one Lua execution; stale generations return a distinct result and cannot mutate the active document.
- Resilience: one monotonic retrieval deadline now covers query embeddings, Redis branches, and reranking, with unconditional branch-task cleanup. JSON batches enforce the decoded 10 MB limit before creating any job and return HTTP 413. Multipart publication is cancellation-safe, and worker heartbeat failures are retried then propagated to the Compose restart supervisor.
- Rollout: `docker compose --profile rag up` now auto-loads the RAG API/worker wiring and healthcheck. V1 and v2 RediSearch hashes use disjoint roots so rolling deployment cannot double-index a chunk. The v2 key migration is dry-run by default and uses persisted copy, validation, pause, delta-copy, switch, resume, and rollback phases; rollback reverse-copies and validates post-cutover v2 writes into v1 while paused before switching. Apply requires explicit idempotent operator hooks and failed rollback remains paused.
- Measured production paths: for 1,000 chunks, embedding requests fell from 1,000 to 32 (96.8%), observed Redis waits from 1,006 to 22 (97.81%), and complete `RagService.ingest_document` time from 1.586 s to 0.141 s (11.25x throughput). Production `RagService.retrieve` p95 fell from 34.86 ms to 22.25 ms (36.18%) under a disclosed fixed 10 ms Redis RTT, preserving result order. A real uvicorn `/rag/documents` request streamed by curl increased sampled server RSS by 4.73 MB for a 100 MB upload. All gates completed with zero errors.
- Verification: deterministic stale-writer and sidecar-fault tests, real Redis generation/migration tests, migration dry-run, and the performance contract passed. The complete suite passed with local Redis services (`534 passed, 4 skipped`); touched-file Black and Flake8, configured mypy, compileall, Compose config, and `git diff --check` passed. The production worker image built successfully, remained healthy under the exact RAG profile, completed a real text/hash ingestion job, promoted one forced failure through retry, and wrote exactly one dead-letter event after attempt two.

## 2026-07-16 — RAG hot-path performance contracts

- Added: a fail-closed 1,000-chunk benchmark that directly exercises production embedding batching, Redis generation indexing, retrieval branch orchestration, and 100 MB streaming staging. Contract tests reject regressions below the agreed request/wait/throughput/p95/RSS thresholds.
- Measured: embedding calls fell from 1,000 to 32 (96.8%); Redis network waits fell from 3,000 to 17 (99.43%); combined ingestion throughput improved 15.06x; retrieval p95 fell from 36.51 ms to 12.54 ms (65.65%); the 100 MB streaming upload increased peak RSS by 10.13 MB. The workload completed with zero errors.
- Real Redis: all four Redis Stack integration scenarios passed after correcting exact `RETURN` projections and stale table/figure/visual sidecar cleanup (`21 passed` with the hot-path suite).
- Runtime note: the production RAG worker image build was started but intentionally stopped while its first-time Docling/Torch dependency layer was downloading a 427 MB Torch wheel plus accelerator packages; Compose runtime health and live job retry/dead-letter still require that heavyweight build to finish.
- Verification: full test suite passed with local Redis services (`514 passed, 4 skipped`); full Black and Flake8 checks, configured mypy, compileall, migration dry-run, and `git diff --check` passed.

## 2026-07-16 — Streaming RAG upload staging

- Changed: multipart files are read in 1 MiB chunks into a same-directory `.part` file, with incremental SHA-256/size accounting, fsync, and atomic rename. Cancellation, read/write failure, and the 100 MB limit remove the partial file. JSON base64 is capped at 10 MB decoded before decode/allocation proceeds.
- API: oversized uploads return HTTP 413 with `rag_payload_too_large`; both queued and queue-disabled background ingestion receive a durable `storage_ref` instead of retaining a multipart body in request memory.
- Verification: RED/GREEN tests cover fixed-size reads, byte fidelity, atomic publication, overflow cleanup, decoded base64 boundaries, schema preservation, API 413 behavior, and queue-disabled storage-ref handoff. RAG/API/schema scoped regression passed (`126 passed`).
- Follow-up: real 100 MB RSS measurement, real Redis fault injection, compose worker smoke, and full-suite verification remain.

## 2026-07-16 — Parallel RAG retrieval and bounded result payloads

- Changed: text and visual query embeddings now start together; Redis keyword, vector, and visual branches run with a three-branch bound and one shared 30-second deadline. Branch failures degrade only that source and merge order remains keyword/vector/visual deterministic.
- Payload: keyword and vector `FT.SEARCH` projections no longer return or deserialize the unused embedding vector, removing 1024-float payloads from the default 30-candidate read path.
- Verification: RED/GREEN concurrency tests require both embedding paths and all three retrieval branches to be in flight together, cover isolated branch failure, and inspect production Redis commands for omitted embedding fields. Scoped RAG regression passed (`39 passed`).
- Follow-up: streaming upload staging and end-to-end performance contracts remain.

## 2026-07-16 — Batched RAG embedding and atomic generation indexing

- Changed: added ordered multi-input embedding calls for OpenAI and local HTTP providers, bounded 32-item/2-concurrent batching with scalar-provider compatibility, and a reusable local HTTP client. Redis ingestion now stages chunks in 64-item pipelines and performs one verified Lua cutover without deleting the prior document first; generation keys expire after 24 hours and documents above the 2,000-chunk atomic bound fail before writes.
- Rollout: RediSearch index names are versioned under `v2`; the dry-run-first migration command creates and validates the new index while retaining the legacy index and existing chunk hashes.
- Verification: RED/GREEN tests cover 1,000-chunk batching/order, isolated batch failure, provider wire format/client reuse, pipeline count, staging-only writes, single atomic cutover, atomic size limit, and validated config. RAG/schema scoped regression passed (`77 passed`).
- Follow-up: retrieval parallelism, upload streaming, real Redis fault injection, and performance contracts remain.

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
