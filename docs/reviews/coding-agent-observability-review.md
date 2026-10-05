# Coding-agent observability code review

Review date: 2026-10-04
Comparison point: `bddeb519d8703294ed727453c2b45fb40455ddc2`
Scope: coding-agent telemetry, Pi capture and recovery, Kafka to ClickHouse analytics, OTEL, feedback, and the observability Compose overlay.

## Standards review

No project-specific `AGENTS.md`, `CONTRIBUTING.md`, or coding-standard file was present for this scope. The review therefore used the repository's existing patterns plus the review baseline for correctness, security boundaries, data integrity, resource bounds, and testability.

All actionable findings identified during review were repaired:

- **P1: host file read/write through session symlinks.** Pi recovery followed a container-writable cursor and JSONL symlinks. Cursor state is now stored in a host-only, per-run sibling directory; recovery opens files with `O_NOFOLLOW | O_NONBLOCK`, verifies regular descriptors, and rejects mismatched cursor identity.
- **P1: monitoring failure could terminate the agent runtime.** Deep JSON redaction and serialization happened outside the fail-open boundary, and Pi capture did not consume rejected telemetry promises. The entire record path now returns `false` on failure, capture consumes asynchronous failures, and traversal has a nesting limit.
- **P1: credential leakage.** Structural and streamed masking missed provider-prefixed keys, Basic authorization, escaped or unterminated quoted values, AWS/private keys, and multiword assignments. A shared credential-key predicate and line-oriented opaque-output masking now cover those forms; PEM private-key blocks are masked across chunks. Official Pi usage fields remain intact.
- **P2: unbounded recovery.** Recovery loaded whole JSONL files and enumerated every directory entry. It now streams verified descriptors with shared limits of 128 entries, 512 MiB, ten seconds, 32 MiB per line, and 16 MiB for the cursor.
- **P2: late finalizer writes and retained run state.** Cancellation now blocks late RPC, snapshot, entry, bash, and capture-status writes. Run-scoped secrets and maps are released after terminal delivery settles; feedback uses a monotonic scalar cursor without recreating per-run state.

No unresolved standards finding remains from the reviewed scope. The final reviewer follow-up could not run because its account reached a usage limit; every concrete finding it had already reported was reproduced locally and covered by regression tests.

## Spec review

The implementation now matches the approved plan in `docs/plans/coding-agent-observability.md`, including preservation of Pi 1.0.1 native and future usage fields, runtime-side collection, ClickHouse projections, durable outbox delivery, OTEL, feedback, and quality/test-result analytics.

The review found and fixed these spec gaps:

- **P1: capture completeness changed after restart.** Completeness and reasons now persist with terminal runs. Newly interrupted runs retain recovery reasons, older terminal records without evidence remain `unknown`, and recovered interrupted runs are not scanned again on every restart.
- **P1: raw telemetry stores were exposed with defaults.** Kafka and ClickHouse host ports are loopback-only in the overlay. The ClickHouse writer password is required and shared chat analytics reads the same environment override. Grafana continues to use a separate read-only account.
- **P2: malformed events blocked partitions.** `KeyError` was caught as `LookupError` and treated as missing content for five minutes. Only `ContentPendingError` now enters the dependency retry path; malformed records go directly to the DLQ and resolved pending state is cleared.
- **P2: Kafka retention was implicit.** The three agent topics are explicitly provisioned with 30-day time retention before the analytics worker starts. This defines the supported analytics outage window.
- **P2: terminal outbox failure was misreported.** A failed terminal envelope persists `capture_complete=false` and its reason before telemetry state is released, including cleanup if persistence itself fails.

## Verification

- Node runtime: `144 passed`, `3 skipped` (`npm test --prefix agent-runtime`). The skipped tests require a real Docker/Pi environment.
- Python and Slack scoped suite: `163 passed`, `1 skipped`. The skipped test is the opt-in real SQLite to Kafka to ClickHouse integration test.
- Black check: 8 relevant Python files unchanged.
- Compose merge/config: passed with explicit placeholder passwords.
- Generated ClickHouse schema: exact match with `docker/observability/agent-tables.sql`.
- Shell syntax and `git diff --check`: passed.

## Remaining deployment validation

The local Docker daemon was unavailable, so this review did not execute a real Pi to Kafka to ClickHouse path, Collector config validation inside its image, Grafana rendering, or restart/fault injection. These are deployment-readiness checks rather than known code defects. Kafka retention is deliberately bounded at 30 days; outages beyond that window, broker disk exhaustion, or volume loss can still lose analytics data. Cross-host Kafka deployment also requires SASL/TLS and network ACLs because the Compose listener is plaintext and intended for a trusted local host.
