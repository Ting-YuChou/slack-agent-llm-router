# Structured coding-agent workflows

This opt-in mode adds two waves of independent, read-only investigation before the existing single writer, verifier and independent grader. It keeps Pi 1.0.1. It does not execute generated JavaScript, run multiple writers or grant publication authority to child agents.

## Enable and invoke

Build the updated agent, gateway and verification images. Copy the example verification profile outside the repository, configure its actual repository and verification image digest, and add:

```json
{
  "structuredWorkflow": {
    "enabled": true,
    "concurrency": 2,
    "maxCostUsd": 5,
    "activeTimeoutMs": 1800000,
    "discoveryCostUsd": 1.5
  }
}
```

The external profile's single-mode limits remain independent: $1 and 15 minutes by default. Structured concurrency can be set from one to four; the default is two, with a host-wide pool of four agent containers. Planner and grader require configured Sol high; investigators use Luna max. The writer retains its selected model or existing Jev route. All configured model prices must be present.

- `/llm agent --workflow <task>` enables structured mode for that task.
- `/llm agent repair-ci --workflow <PR_URL|RUN_ID>` uses it after existing CI authorization and reproduction.
- Session creation, follow-up prompts and CI repair accept `workflow_mode: "structured"`. Omission means `single`; the setting is not silently inherited by another task.
- `/llm agent resume <RUN_ID>` or owner-authenticated `POST /v1/runs/:id/resume` resumes an interrupted read-only stage.
- `GET /v1/runs/:id/tasks` returns task states, attempts and results. Existing run, events, decisions and cancellation APIs retain their contracts.

Structured requests without enabled trusted configuration fail explicitly. GitHub webhooks continue to create single-mode repairs. Existing publication approval remains mandatory for CI repair PRs.

## Discussion versus implementation

Task authority is independent of workflow scheduling. Session creation and follow-up prompt APIs accept `task_mode: "discuss" | "implement"`. Omission defaults to `implement` for compatibility. Each new prompt chooses its own mode; the previous prompt's mode is never inherited. Jev labels affect model routing, not permission to modify files.

- `/llm agent --discuss <question>` reads and explains the project without verification or commit.
- `/llm agent --discuss --workflow <question>` runs the structured investigations and a read-only final responder. It still skips verifier, grader and commit.
- `/llm agent --implement <task>` retains the existing writer and configured quality gates. Omitting both flags has the same behavior.

The host mounts the discussion workspace read-only, removes Git metadata mounts, and exposes only `read`, `grep`, `find` and `ls`. Discussion uses trusted model-gateway and discussion-policy extensions; writer plugins, repo context files, skills, MCP, Codemode and shell are unavailable. A separate read policy rejects protected paths, escaping symlinks and tools outside the allowlist. Existing session conversation context can continue, but it cannot restore tool privileges. Rebuild the agent image before enabling these requests.

At completion the host also checks for unexpected workspace changes; any change blocks delivery. Successful discussion reports `verification_status`, `grader_status` and `publication_status` as `not_applicable`, with no commit/cherry-pick field. It retains execution usage, cost admission and traces when the trusted quality profile is configured. CI repair always requires `implement`; a discussion request to that endpoint is rejected. Starting an implementation after discussion requires a new prompt and existing write approvals.

## Execution and authority

The host locks the requirement, baseline and profile checksum, and marks the parent session busy before investigation. A planner submits a validated graph through `submit_plan`. Supported investigator roles are `code_explorer`, `ci_investigator` and `test_analyst`; workers submit findings through `submit_investigation`. The host rejects cycles, unknown dependencies, escaping paths, unsupported roles and more than eight investigation tasks across both waves.

The second planner sees completed results and can add another wave. Each task waits for its dependencies. Invalid output and execution failures get at most one retry with a new context; unresolved required work blocks delivery. Planner executions have two-minute deadlines; investigators have three-minute deadlines and a 20-tool-call limit. Parent time and cost limits also apply.

Every planner/investigator has a fresh directory and context, a read-only baseline snapshot, no repository context-file discovery and no writer plugins or skills. File reads are restricted to task paths. Shell, editing, writing and child spawning are unavailable. A configured CI investigator can use the trusted GitHub MCP's direct read tools, with a repository-scoped, four-call token. It does not receive Codemode, other MCP namespaces or GitHub credentials. Other roles use snapshot reads in v1.

All workers must stop before the writer starts. Investigation output is labeled untrusted evidence. The grader still receives only the original requirement, fixed rubric, candidate identity/snapshot, diff and verifier output; it never receives writer or planner conversations. Passing investigation is not passing verification.

## Budgets, cancellation and restart

Only the host opens the budget SQLite database. Docker gateways use authenticated `/internal/budget/{reserve,settle,first-model}` operations, configured with `PI_AGENT_BUDGET_RPC_URL=http://host.docker.internal:3001` and the existing gateway signing secret. Adjust the port to `PI_AGENT_PORT` when needed. The demo launcher configures this automatically. Gateways cannot create or increase budgets. A same-OS standalone gateway can still use `PI_AGENT_BUDGET_FILE`; do not share a WAL database across the macOS/Docker VM boundary.

Planner, investigations, writer, repairs, grader and Jev charge one parent budget. The $1.50 discovery cap is shared across both waves and retries. Unreported usage remains reserved. Failure to reserve prevents upstream model dispatch; failure to settle retains the reservation. The $5 limit is admission control, not a provider billing guarantee.

Control state is separate from telemetry. Node claims and terminal updates are fenced by execution IDs. One host coordinator owns startup/recovery, preventing another runtime from stopping an active runtime's containers. Cancellation stops new dispatch, confirms child shutdown, preserves evidence and then rolls back; an unconfirmed stop retains the isolated worktree.

Read-only graceful shutdown records `interrupted` without closing the parent budget. Resume requires the owner, an available idle session, unchanged requirement/baseline/profile/tools/models and retained artifacts. Completed nodes are reused; unfinished work gets a new execution ID. Active time and cost continue from persisted values, and downtime is recorded separately. Existing session expiry still applies. Once writer startup is recorded, resume is refused and existing interruption/rollback behavior applies. Publication retries remain independent.

## Analytics and validation

Official Pi entries and unknown usage fields remain intact for every execution. Domain events use the existing redacted SQLite outbox, Kafka and ClickHouse path. Added tables are `agent_workflow_nodes`, `agent_workflow_revisions` and `agent_workflow_dependencies`, with deduplicated views and 90-day metadata retention. Full contexts/results remain on the 30-day content path. Control result content is pruned with evidence retention.

OTEL links all child executions to the workflow. A resumed workflow retains its stored trace identity. Grafana separates wall, active, approval wait, suspension, admission queues and cumulative agent intervals. `agent_workflow_execution_durations` includes every role; missing timing remains unknown. Parallel intervals must not be summed to infer wall time.

Local tests cover graph validation, role models and isolation, dependency order, two-wave discovery, retries, fencing, budget RPC authentication and stage caps, cancellation before writer, graceful interruption and resume, Slack/API compatibility, and analytics projection. The opt-in `PI_AGENT_VERIFICATION_TEST_IMAGE=<immutable digest> node --test dist/test/verification-docker.test.js` additionally runs real credential-free verification, trusted-test restoration, task-local Redis and snapshot write protection.

Real model/API, GitHub, Slack and Kafka acceptance must be reported separately from local tests. Use an isolated repository/PR and retain run IDs, task results, candidate hashes, official usage, traces and approvals. Stop temporary resources when finished.

## Isolated acceptance recorded on 2026-10-15

The [sanitized evidence report](validation/structured-workflow-2026-10-15.json) records a real API/live-model run against a temporary two-module repository. Parallel investigations overlapped for 30,116 ms. The runtime was restarted before writer startup; one completed worker was reused under the same run and trace IDs. The writer then passed Docker verification and an independent grader and created a commit in the temporary repository. The second planner needed no additional tasks; dynamic second-wave additions are covered separately by coordinator tests.

Wall time was 119,738 ms, including 546 ms of simulated owner waiting and 952 ms of interruption. The runtime reported $0.11998477 in known model cost. Its OTEL SDK exported 188 spans to an isolated local receiver, retaining one trace across restart. This does not validate a deployed Collector backend or Grafana UI, and there is no measured speedup claim.

A private outbox copy went through the production Kafka publisher and analytics consumer into real ClickHouse: 5,144 canonical events and 27 usage rows. Exact replay retained 27 usage rows. Six executions reconciled and two interrupted executions remained incomplete; the analytical workflow total correctly remained null rather than treating missing cost as zero.

Reproduce after building the runtime and the `slack-pi-agent:structured-test`, `slack-pi-model-gateway:structured-test` and `slack-pi-verification:structured-test` images, with Docker running:

```sh
npm --prefix agent-runtime run build
.venv/bin/python scripts/validate_structured_workflow.py --live-openai --exercise-resume
.venv/bin/python scripts/validate_agent_analytics_pipeline.py --docker --outbox /absolute/path/from/acceptance/state/analytics/outbox.sqlite
```

The first command using `--live-openai` explicitly permits paid provider requests and reads the locally configured credential without printing it. Both drivers create private evidence directories and remove their own temporary containers and networks. The analytics driver clones the outbox; it does not acknowledge or mutate the original. Preserve the printed evidence paths privately because full contexts and source snapshots can be sensitive.

Real CI failure → Slack owner approval → draft PR acceptance is still pending a separate Publisher App, allowlisted test PR and Slack team/channel/owner mapping. The synthetic owner approvals above are not evidence of real Slack authorization or GitHub publication. Do not elevate the existing read-only MCP App to complete this setup.
