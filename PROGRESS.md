# Project Progress

## 2026-07-12 — Resilient Redis admission connections

- Changed: added bounded Redis connect/read timeouts, a bounded connection pool, controller-level operation deadlines, and cooldown-gated single-flight recovery so transient failures recover without restarting workers.
- Key files: `src/admission.py`, `src/utils/schema.py`, `src/utils/metrics.py`, API rate-limit config, and admission/schema tests.
- Verification: `python -m pytest tests/test_admission.py tests/test_schema.py -q` (`57 passed`); Black, Flake8, mypy, and `git diff --check` passed; independent task review approved.
- Follow-up: atomic fast admission, v2 queue leases, and circuit transitions remain in the next tasks.

## 2026-07-12 — Kafka and provider hot-path hardening

- Changed: made Kafka consumer batch persistence race-safe; made Redis policy materialization fail before offset commit; moved Kafka delivery/ack/retry into a bounded background dispatcher with explicit drain, abandonment, and producer-close deadlines; replaced pseudo batching with zero-wait single-flight ahead of provider scheduling; centralized retries under a shared 60-second deadline and transport-attempt budget.
- Key files: `src/llm_router_part3_pipeline.py`, `src/llm_router_part3_policy.py`, `src/llm_router_part2_inference.py`, `src/provider_scheduler.py`, `src/utils/schema.py`, and matching tests/config.
- Verification: `python -m pytest tests -q` (`351 passed, 7 skipped`); Black, Flake8, mypy, compileall, and `git diff --check` passed. Kafka/policy and provider hot-path changes passed independent reviews.
- Follow-up: live hot/unique `/route` load comparison was not run because no OpenAI, Anthropic, or local vLLM endpoint credentials/runtime were available. Deterministic tests cover delayed Kafka acknowledgements, queue overflow/shutdown, 20-way identical-request coalescing, shared retry budget, deadline, and cancellation behavior.
