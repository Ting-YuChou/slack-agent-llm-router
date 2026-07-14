# Project Progress

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

## 2026-07-12 — Kafka and provider hot-path hardening

- Changed: made Kafka consumer batch persistence race-safe; made Redis policy materialization fail before offset commit; moved Kafka delivery/ack/retry into a bounded background dispatcher with explicit drain, abandonment, and producer-close deadlines; replaced pseudo batching with zero-wait single-flight ahead of provider scheduling; centralized retries under a shared 60-second deadline and transport-attempt budget.
- Key files: `src/llm_router_part3_pipeline.py`, `src/llm_router_part3_policy.py`, `src/llm_router_part2_inference.py`, `src/provider_scheduler.py`, `src/utils/schema.py`, and matching tests/config.
- Verification: `python -m pytest tests -q` (`351 passed, 7 skipped`); Black, Flake8, mypy, compileall, and `git diff --check` passed. Kafka/policy and provider hot-path changes passed independent reviews.
- Follow-up: live hot/unique `/route` load comparison was not run because no OpenAI, Anthropic, or local vLLM endpoint credentials/runtime were available. Deterministic tests cover delayed Kafka acknowledgements, queue overflow/shutdown, 20-way identical-request coalescing, shared retry budget, deadline, and cancellation behavior.
