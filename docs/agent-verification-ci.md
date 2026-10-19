# Coding-agent verification and CI repair

The workflow keeps Pi 1.0.1. A repository opts in by setting an external trusted verification profile. Without a profile, existing coding sessions retain their previous behavior; the CI repair endpoint explicitly rejects requests. Enabling CI repair with incomplete configuration fails startup.

## Execution and trust boundaries

1. `/llm agent repair-ci <PR_URL|RUN_ID>` or a signed `workflow_run.completed` failure enters the control inbox.
2. The host checks the repository, open same-repository PR, workflow allowlist, current head, run attempt and failed jobs. Logs are bounded, redacted diagnostic data.
3. An owner-bound approval expires after five minutes. No reproduction or writer starts before approval.
4. The verifier reproduces failure from the locked tested commit using the profile's fixed commands. A failure must match a configured `ciJob`; unrelated failures and unavailable environments stop the workflow.
5. The writer starts from the original PR head. The session stays busy through verification and grading. Initial implementation plus two repairs share the active-time and model budgets.
6. Every candidate is copied into a private immutable snapshot. Trusted baseline tests/configuration are overlaid in a separate verification pass; candidate tests run separately. For merge-based CI, the patch is also applied to the locked synthetic merge commit and checked there. An apply conflict prevents delivery.
7. The grader starts in a fresh directory and session with Sol high. It has no writer messages, context-file discovery, MCP, mutation or shell tools. Only snapshot reads and `submit_grade` are available. The host checks candidate identity, rubric completeness, evidence and verdict consistency.
8. Passing candidates are checked again before local commit. Failed/unknown candidates retain patch and structured evidence. Cancellation stops writer, verifier and grader before archiving and rollback. If shutdown or evidence preservation fails, the worktree stays isolated.
9. A separate owner approval publishes a deterministic repair branch and draft PR whose base is the original PR branch. Publisher credentials stay on the host. Remote branch/PR lookup makes retries recoverable; a changed PR/CI identity requires a new verified workflow.

`status` retains the existing API contract. `phase`, `execution_status`, `verification_status`, `grader_status`, `publication_status`, attempts and timing provide separate workflow results. Owner feedback remains independent.

## Required setup

Build the verification image from the repository root:

```sh
docker build -f agent-runtime/Dockerfile.verification -t slack-pi-verification:1 .
docker image inspect --format '{{.Id}}' slack-pi-verification:1
mkdir -p "$HOME/.config/slack-pi"
cp config/verification/slack-router.example.json "$HOME/.config/slack-pi/verification.json"
chmod 600 "$HOME/.config/slack-pi/verification.json"
```

Edit that external copy: actual repository slug, immutable verification image digest, Slack team/channel/owner IDs, workflow names, and `ciJob` names matching GitHub job names. The checked-in example is a template and cannot be activated directly. `ciCheckout` must match the workflow's checkout behavior (`merge` for the sample pull-request workflow; `head` only when CI explicitly checks out the PR head). Candidate snapshots cannot replace the profile. Its checksum is recorded on verification, grading and execution events.

Set these in the local demo environment:

```dotenv
PI_AGENT_VERIFICATION_PROFILE=/absolute/path/outside/repository/verification.json
PI_AGENT_CI_REPAIR_ENABLED=true
PI_AGENT_CI_WEBHOOK_ENABLED=true
PI_AGENT_CI_WEBHOOK_PORT=3003
GITHUB_WEBHOOK_SECRET=<random webhook signing secret>
GITHUB_PUBLISHER_APP_ID=<separate write App>
GITHUB_PUBLISHER_APP_INSTALLATION_ID=<installation for allowlisted repository>
GITHUB_PUBLISHER_APP_PRIVATE_KEY_PATH=/absolute/private/path/publisher.pem
PI_AGENT_MODEL_PRICES_JSON={"openai/gpt-5.6-luna":{"input":<USD_per_million>,"output":<USD_per_million>},"openai/gpt-5.6-sol":{"input":<USD_per_million>,"output":<USD_per_million>},"openrouter/typesafe/jev-1.13":{"input":<USD_per_million>,"output":<USD_per_million>}}
```

Without `PI_AGENT_VERIFICATION_PROFILE`, implement runs are committed with `verification_status: "unverified"` and a `Pi-Verification: unverified` commit trailer, and Slack labels them as unverified. Production deployments should set `PI_AGENT_REQUIRE_VERIFICATION=1`, which rejects implement runs with `verification_profile_required` (HTTP 422) unless a trusted profile is configured; discuss mode stays available.

Replace every placeholder; the prices example is deliberately not valid JSON until actual rates are supplied. Rates should conservatively cover cache-write premiums when applicable. Zero output price is accepted. No live price is hard-coded.

Keep the existing read-only `GITHUB_APP_*` configuration and provider/Slack/Kafka/OTEL settings. Create a **different** publisher App, installed only on the allowed repository, with Contents and Pull requests write permissions. Do not grant those permissions to the MCP App. The runtime requires configured OpenAI Sol for grading.

The host owns the budget SQLite database. The Docker gateway uses authenticated budget RPC through `PI_AGENT_BUDGET_RPC_URL=http://host.docker.internal:3001`; the launcher adjusts this to `PI_AGENT_PORT`. Neither Pi nor the gateway mounts the host database. Do not share SQLite WAL files across the macOS/Docker VM boundary. Budgeted requests fail closed without host admission or model prices. See [structured workflows](agent-structured-workflow.md) for the opt-in multi-agent configuration.

Expose only the webhook listener through an HTTPS reverse proxy and configure the GitHub App webhook to `<public HTTPS origin>/webhooks/github`, with the identical signing secret and workflow-run events. Keep runtime port 3001 private. The listener itself binds loopback port 3003. Signed deliveries are persisted before ACK; delivery and run-attempt/head identities deduplicate replay. Slack polls the runtime for webhook-created approvals.

Rebuild the existing agent/gateway images after source changes and refresh the existing agent plugin image lock using the repository's `make demo-agent-images` procedure. Then start the demo launcher with the chosen environment file. Missing profiles, owners, providers and publisher credentials must be resolved before attempting a real repair.

## Accounting and observability

Control state uses its own SQLite database. Telemetry continues through redaction, SQLite outbox, Kafka and ClickHouse. New tables are `agent_executions`, `agent_verifications`, `agent_grades` and `agent_deliveries`, with deduplicated `*_latest` views. Metadata retention is 90 days. Snapshot/patch/log directories are private raw evidence. A startup/daily cleanup applies 30-day retention and preserves paths mounted by active or stopped agent containers. If Docker state cannot be read, evidence is retained.

Every implementation and grader has its own execution ID, official Pi entries and start/end snapshots. `agent_workflow_usage` aggregates execution reconciliations instead of summing overlapping session totals. Unknown/future Pi fields remain in the raw payload. Missing usage remains unknown.

The gateway reserves a conservative amount **before** each request, caps generative output at 4096 tokens and retains reservations when usage is unavailable. Jev routing and in-run decisions participate in the same ledger. Its fixed decision schema is bounded separately; no unsupported token-limit parameter is injected into the Decisions API. See the [official Decisions API schema](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

Gateway dollar figures derived from token counts/rates are estimates; official Pi costs remain the analysis ledger. In-flight requests, unknown usage and provider billing differences prevent an absolute $1 billing guarantee. Reservation summaries and unknown-request counts are emitted separately.

Startup timing distinguishes Docker CLI spawn, inspected container running, verified RPC response, configuration, first accepted model request and first nonempty output. CI approval/reproduction/publication and verification/grading have separate spans. The dashboard shows wall time, active time and human waiting; overlapping spans must not be added together.

## Validation and operating limits

Local suites cover timing boundaries, candidate hashes, fresh grader settings, bounded repairs, no commit on unknown results, owner approvals, durable inbox replay, budget reservations and recovery after PR creation times out. Node tests are included in GitHub CI. Tests that explicitly require Docker or Kafka remain opt-in.

### Local validation on 2026-10-14

- `cd agent-runtime && npm test`: 192 tests, 189 passed, three live Pi integration tests skipped, zero failures. This also builds the runtime and extensions.
- Python agent analytics/runtime and Slack suites: 123 passed, using a dedicated temporary Redis Unix socket. That Redis was stopped and its temporary directory removed after the run.
- Scoped Black/flake8, configured mypy, an additional check of `src/agent_runtime.py`, shell syntax and `git diff --check` passed.
- HTTP tests use actual loopback listeners, SQLite control state and the CI coordinator. They check signed delivery persistence and deduplication, owner/context rejection, distinct single-use approvals, and publication retry retaining the verified commit. External GitHub and execution responses are simulated; these tests do not exercise a real Pi repair.
- Timing regression tests verify publication duration, both owner waits, tool approval waits after reproduction, late expiry observation and frozen terminal totals. Session retention also checks recently appended contexts inside old directories.

The Docker daemon was unavailable, so the verification image was not built or run. The local demo configuration did not contain the new trusted profile, Publisher App credentials, model price configuration or CI enablement. No real Slack approval, GitHub publication or Kafka/ClickHouse acceptance was performed. Existing read-only GitHub App credentials were not reused for publication.

For final acceptance, use an isolated test PR with a reproducible allowlisted failure. Exercise both Slack approvals, inspect the resulting draft PR base/head, and query execution usage, verification, grades, delivery and OTEL trace in ClickHouse. Retain the candidate identities, test results, grade artifacts, trace ID and PR URL. Stop only resources created for this acceptance run.

Until that real flow succeeds, local test results are not proof of live Docker, GitHub, Slack, Kafka or ClickHouse integration. The initial implementation intentionally does not resume interrupted repair attempts after restart; publication retries retain the verified local commit and require a fresh owner approval.

Publication failures keep the verified commit. The owner can use `/llm agent publish-repair <RUN_ID>` in the configured channel to request a new five-minute publication approval; this does not rerun Pi.
