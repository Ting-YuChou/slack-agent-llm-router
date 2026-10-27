import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { InMemorySpanExporter, SimpleSpanProcessor, type ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { AgentTracing, type TraceSignal } from "../src/tracing.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import { POLICY_ENTRY, parsePolicyDecision } from "../src/policy-decision.js";
import { DiffLimitError } from "../src/worktree-manager.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { waitFor } from "./support/wait.js";

// Read-only roles run with cwd outside the snapshot (container-runtime mounts the session state here).
const SESSION_CWD = "/var/lib/pi-session";

/** Loads a read-only extension with a fake Pi API that keeps the session entries it appends. */
async function loadReadOnly(name: string, env: Record<string, string>) {
  const previous = Object.fromEntries(Object.keys(env).map(key => [key, process.env[key]]));
  Object.assign(process.env, env);
  try {
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname, `../../extensions/${name}.ts`)).href);
    const handlers = new Map<string, (...args: any[]) => any>();
    const entries: Array<{ customType: string; data: any }> = [];
    extension.default({
      on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler),
      registerTool: () => undefined,
      appendEntry: (customType: string, data: unknown) => { entries.push({ customType, data }); },
    });
    const ctx = { cwd: SESSION_CWD, abort: () => undefined };
    const call = (toolName: string, input: Record<string, unknown>) => handlers.get("tool_call")!({ toolName, input }, ctx);
    return { call, entries, handlers };
  } finally {
    for (const [key, value] of Object.entries(previous)) if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
}

async function withSnapshot(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), "trace-decisions-"));
  try {
    await mkdir(path.join(root, "src"));
    await writeFile(path.join(root, "src", "a.ts"), "export const a = 1");
    await writeFile(path.join(root, "README.md"), "readme");
    await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

/** Each entry is a policy decision the host accepts unchanged; returns `tool:decision:reason` per entry. */
function decisions(entries: Array<{ customType: string; data: any }>): string[] {
  return entries.map(entry => {
    assert.equal(entry.customType, POLICY_ENTRY);
    assert.deepEqual(parsePolicyDecision(entry.data), entry.data, "the host parses the entry as it was written");
    return `${entry.data.tool}:${entry.data.decision}:${entry.data.reason}`;
  });
}

test("the discussion extension records each block as a policy decision, and allowed reads record none", () => withSnapshot(async root => {
  const { call, entries } = await loadReadOnly("discussion", { PI_AGENT_WORKTREE: root });
  assert.equal(await call("read", { path: path.join(root, "src", "a.ts") }), undefined);
  assert.deepEqual(entries, []);
  assert.equal((await call("bash", { command: "rm -rf /" })).block, true);
  assert.equal((await call("read", { path: "/etc/passwd" })).block, true);
  for (let i = 0; i < 37; i++) await call("read", { path: path.join(root, "src", "a.ts") });
  assert.equal((await call("read", { path: path.join(root, "src", "a.ts") })).block, true);
  assert.deepEqual(decisions(entries), ["bash:block:read_only", "read:block:outside_workspace", "read:block:tool_limit"]);
}));

test("the grader extension records each block as a policy decision", () => withSnapshot(async root => {
  const { call, entries } = await loadReadOnly("grader", { PI_AGENT_WORKTREE: root, PI_AGENT_GRADER_JSON: JSON.stringify({ candidate_hash: "h", rubric: [] }) });
  assert.equal(await call("read", { path: "src/a.ts" }), undefined);
  assert.equal((await call("write", { path: "src/a.ts" })).block, true);
  assert.equal((await call("read", { path: "../escape" })).block, true);
  for (let i = 0; i < 37; i++) await call("read", { path: "src/a.ts" });
  assert.equal((await call("read", { path: "src/a.ts" })).block, true);
  assert.deepEqual(decisions(entries), ["write:block:read_only", "read:block:outside_snapshot", "read:block:tool_limit"]);
}));

test("the workflow-agent extension records each block as a policy decision", () => withSnapshot(async root => {
  const { call, entries } = await loadReadOnly("workflow-agent", { PI_AGENT_WORKTREE: root, PI_AGENT_WORKFLOW_JSON: JSON.stringify({ role: "code_explorer", paths: ["src"] }) });
  assert.equal(await call("read", { path: "src/a.ts" }), undefined);
  assert.equal((await call("edit", { path: "src/a.ts" })).block, true);
  assert.equal((await call("read", { path: "README.md" })).block, true);
  for (let i = 0; i < 17; i++) await call("read", { path: "src/a.ts" });
  assert.equal((await call("read", { path: "src/a.ts" })).block, true);
  assert.deepEqual(decisions(entries), ["edit:block:restricted_tool", "read:block:outside_scope", "read:block:tool_limit"]);
}));

test("a read-only extension whose entry cannot be appended still blocks", () => withSnapshot(async root => {
  const previous = process.env.PI_AGENT_WORKTREE;
  process.env.PI_AGENT_WORKTREE = root;
  const handlers = new Map<string, (...args: any[]) => any>();
  try {
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname, "../../extensions/discussion.ts")).href);
    extension.default({ on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler), appendEntry: () => { throw new Error("session closed"); } });
  } finally {
    if (previous === undefined) delete process.env.PI_AGENT_WORKTREE; else process.env.PI_AGENT_WORKTREE = previous;
  }
  const result = await handlers.get("tool_call")!({ toolName: "write", input: {} }, { cwd: SESSION_CWD, abort: () => undefined });
  assert.equal(result.block, true);
}));

async function traced(run: (tracing: AgentTracing) => void | Promise<void>): Promise<ReadableSpan[]> {
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  try {
    await run(tracing);
    await tracing.flush();
    return exporter.getFinishedSpans();
  } finally { await tracing.close(); }
}

const parentOf = (span: ReadableSpan) => (span as unknown as { parentSpanContext?: { spanId: string } }).parentSpanContext?.spanId;

test("policy entries from any execution become events on its run span; malformed entries are dropped", async () => {
  const spans = await traced(tracing => {
    tracing.startRun("grader-1", "grader-1", undefined);
    const entry = (data: unknown) => tracing.rpc("grader-1", { type: "entry_appended", entry: { type: "custom", customType: POLICY_ENTRY, data } });
    entry({ tool: "bash", decision: "approval", reason: "write_command", approved: false, argv0: "npm", cmd_hash: "0123456789abcdef", latency_ms: 1200 });
    entry({ tool: "read", decision: "block", reason: "outside_snapshot" });
    entry({ tool: "read", decision: "maybe", reason: "outside_snapshot" });
    entry({ tool: "read", decision: "block", reason: "Free text with $ecrets" });
    tracing.rpc("grader-1", { type: "entry_appended", entry: { type: "custom", customType: "other", data: { tool: "read", decision: "block", reason: "x" } } });
    tracing.endRun("grader-1", "completed");
  });
  const root = spans.find(span => span.name === "agent.run")!;
  const events = root.events.filter(event => event.name === "agent.policy_decision");
  assert.equal(events.length, 2);
  assert.deepEqual(events[0].attributes, {
    "agent.policy.tool": "bash", "agent.policy.decision": "approval", "agent.policy.reason": "write_command",
    "agent.policy.approved": false, "agent.policy.argv0": "npm", "agent.policy.cmd_hash": "0123456789abcdef", "agent.policy.latency_ms": 1200,
  });
  assert.deepEqual(events[1].attributes, { "agent.policy.tool": "read", "agent.policy.decision": "block", "agent.policy.reason": "outside_snapshot" });
});

test("the run span carries the route, the execution outcome, the cause and the last repair reason", async () => {
  const spans = await traced(tracing => {
    tracing.startRun("r", "s");
    tracing.describeRun("r", { model: "openai/gpt-5.6-luna", reasoning_effort: "high", routing: { modelRef: "openai/gpt-5.6-luna", effort: "high", source: "jev", label: "read_only", probability: 0.93 } });
    tracing.signal("r", { type: "repair", attempt: 2, reason: "verification_failed" });
    tracing.signal("r", { type: "repair", attempt: 3, reason: "grader_needs_changes" });
    tracing.describeRun("r", { execution_outcome: "limit_reached" });
    tracing.endRun("r", "failed", { code: "verification_failed", cause_code: "insufficient_evidence" });
  });
  const root = spans.find(span => span.name === "agent.run")!;
  assert.equal(root.attributes["gen_ai.request.model"], "openai/gpt-5.6-luna");
  assert.equal(root.attributes["gen_ai.request.reasoning_effort"], "high");
  assert.equal(root.attributes["agent.route.label"], "read_only");
  assert.equal(root.attributes["agent.route.probability"], 0.93);
  assert.equal(root.attributes["agent.route.source"], "jev");
  assert.equal(root.attributes["agent.outcome"], "limit_reached");
  assert.equal(root.attributes["agent.error.cause_code"], "insufficient_evidence");
  assert.equal(root.attributes["agent.repair.reason"], "grader_needs_changes");
  assert.equal(root.attributes["agent.repair.count"], 2);
});

test("phases and repair attempts are spans under the run span, and limit hits are events on it", async () => {
  const spans = await traced(tracing => {
    tracing.startRun("r", "s");
    const signals: TraceSignal[] = [
      { type: "phase", phase: "verifying" },
      { type: "limit", kind: "diff_budget", code: "diff_limit_exceeded", action: "repair" },
      { type: "repair", attempt: 2, reason: "diff_limit" },
      { type: "phase", phase: "repairing" },
      { type: "phase", phase: "verifying" },
      { type: "limit", kind: "budget", code: "workflow_budget_exhausted", action: "fail" },
    ];
    for (const signal of signals) tracing.signal("r", signal);
    tracing.signal("other-run", { type: "phase", phase: "verifying" });
    tracing.endRun("r", "failed", { code: "workflow_budget_exhausted" });
  });
  const root = spans.find(span => span.name === "agent.run")!;
  const phases = spans.filter(span => span.name === "agent.phase");
  assert.deepEqual(phases.map(span => span.attributes["agent.phase"]), ["verifying", "repairing", "verifying"]);
  assert.ok(phases.every(span => parentOf(span) === root.spanContext().spanId), "each phase is a child of the run span");
  assert.ok(phases.every(span => span.attributes["capture.incomplete"] === undefined), "phases end cleanly, the last one at the run's end");
  assert.deepEqual(phases.map(span => span.status.code), [0, 0, 2], "the phase the run failed in is marked failed");
  const repairs = spans.filter(span => span.name === "agent.repair");
  assert.equal(repairs.length, 1);
  assert.equal(parentOf(repairs[0]), root.spanContext().spanId);
  assert.deepEqual([repairs[0].attributes["agent.repair.attempt"], repairs[0].attributes["agent.repair.reason"]], [2, "diff_limit"]);
  const limits = root.events.filter(event => event.name === "agent.limit").map(event => event.attributes);
  assert.deepEqual(limits, [
    { "agent.limit.kind": "diff_budget", "agent.limit.code": "diff_limit_exceeded", "agent.limit.action": "repair" },
    { "agent.limit.kind": "budget", "agent.limit.code": "workflow_budget_exhausted", "agent.limit.action": "fail" },
  ]);
});

test("a new repair attempt ends the previous repair span", async () => {
  const spans = await traced(tracing => {
    tracing.startRun("r", "s");
    tracing.signal("r", { type: "repair", attempt: 2, reason: "verification_failed" });
    tracing.signal("r", { type: "repair", attempt: 3, reason: "scope_violation" });
    tracing.endRun("r", "completed");
  });
  const repairs = spans.filter(span => span.name === "agent.repair");
  assert.deepEqual(repairs.map(span => [span.attributes["agent.repair.attempt"], span.attributes["agent.repair.reason"], span.status.code]), [[2, "verification_failed", 0], [3, "scope_violation", 0]]);
  assert.ok(repairs.every(span => span.attributes["capture.incomplete"] === undefined));
});

const SETTLED = { type: "settled", outcome: "completed" } as const;

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompts: string[] = [];
  prompt(_runId: string, prompt: string) { this.prompts.push(prompt); }
  decide() {}
  respondUi() {}
  async abort() {}
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function traceFixture(quality: Record<string, unknown> | undefined, extra: Record<string, unknown> = {}) {
  const process = new FakeProcess();
  const signals: Array<{ runId: string; signal: TraceSignal }> = [];
  const calls = { inspectError: undefined as Error | undefined, rollbacks: 0, commits: 0 };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "pi-agent/run", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => { if (calls.inspectError) throw calls.inspectError; return { files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }; },
      commit: async () => { calls.commits += 1; return "b".repeat(40); },
      rollback: async () => { calls.rollbacks += 1; },
      remove: async () => undefined,
    },
    startProcess: async (_session: unknown, onEvent: (event: PublicRunEvent) => void) => { process.handler = onEvent; return process; },
    maxActiveSessions: 2, deadlineMs: 10_000, quality,
    observeTrace: (runId: string, signal: TraceSignal) => signals.push({ runId, signal }),
    ...extra,
  } as any);
  return { orchestrator, process, signals, calls };
}

const gate = (evaluate: () => Promise<unknown>, extra: Record<string, unknown> = {}) => ({ execution: () => {}, assertUnchanged: async () => {}, maxCostUsd: 5, evaluate, ...extra });
const PASS = { action: "pass", candidate: { hash: "p", path: "/x" }, verification: { status: "passed", checks: [] }, grade: { status: "passed", candidate_hash: "p", criteria: [], findings: [] } };

async function finished(f: ReturnType<typeof traceFixture>, runId: string) {
  return waitFor(() => f.orchestrator.getRun(runId), run => ["completed", "failed", "timed_out", "cancelled"].includes(run.status) && f.calls.commits + f.calls.rollbacks > 0,
    run => `run never settled; status ${run.status}`);
}

test("each repair reaches the trace with its reason, and every phase the run moves through opens a phase span", async () => {
  const results = [
    { action: "repair", repair_prompt: "fix tests", candidate: { hash: "a", path: "/x" }, verification: { status: "failed", checks: [] } },
    { action: "repair", repair_prompt: "address findings", candidate: { hash: "b", path: "/x" }, verification: { status: "passed", checks: [] }, grade: { status: "needs_changes", candidate_hash: "b", criteria: [], findings: [] } },
    { action: "repair", repair_prompt: "revert docs", candidate: { hash: "", path: "/x" }, verification: { status: "failed", checks: [] }, failure: { code: "scope_violation", files: ["docs/x.md"] } },
    PASS,
  ];
  const f = traceFixture(gate(async () => results.shift()));
  const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-repairs", user_id: "U", prompt: "fix" });
  for (let attempt = 1; attempt <= 4; attempt++) {
    await waitFor(() => f.process.prompts.length, count => count >= attempt, () => `attempt ${attempt} was never prompted`);
    f.process.emit(SETTLED);
  }
  const run = await finished(f, accepted.run_id);
  assert.equal(run.status, "completed", JSON.stringify(run.error));
  assert.ok(f.signals.every(entry => entry.runId === accepted.run_id));
  const of = (type: TraceSignal["type"]) => f.signals.map(entry => entry.signal).filter(signal => signal.type === type);
  assert.deepEqual(of("repair"), [
    { type: "repair", attempt: 2, reason: "verification_failed" },
    { type: "repair", attempt: 3, reason: "grader_needs_changes" },
    { type: "repair", attempt: 4, reason: "scope_violation" },
  ]);
  assert.deepEqual(of("phase").map(signal => (signal as { phase: string }).phase),
    ["verifying", "repairing", "verifying", "repairing", "verifying", "repairing", "verifying", "finalizing"]);
  assert.deepEqual(of("limit"), []);
});

test("a diff over the limits that becomes a repair is a limit hit and a diff_limit repair on the trace", async () => {
  const f = traceFixture(gate(async () => PASS, { maxRepairs: 1 }));
  f.calls.inspectError = new DiffLimitError("Changed file limit exceeded (51/50)");
  const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-diff", user_id: "U", prompt: "fix" });
  await waitFor(() => f.process.prompts.length, count => count > 0, () => "the writer was never prompted");
  f.process.emit(SETTLED);
  await waitFor(() => f.process.prompts.length, count => count > 1, () => "the diff limit was never sent back as a repair");
  f.calls.inspectError = undefined;
  f.process.emit(SETTLED);
  await finished(f, accepted.run_id);
  const signals = f.signals.map(entry => entry.signal).filter(signal => signal.type !== "phase");
  assert.deepEqual(signals, [
    { type: "limit", kind: "diff_budget", code: "diff_limit_exceeded", action: "repair" },
    { type: "repair", attempt: 2, reason: "diff_limit" },
  ]);
});

test("runs that fail on a limit record the limit hit; other failures record none", async () => {
  const budget = traceFixture(gate(async () => { throw new Error("workflow_budget_exhausted"); }));
  const spent = await budget.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-budget", user_id: "U", prompt: "fix" });
  await waitFor(() => budget.process.prompts.length, count => count > 0, () => "the writer was never prompted");
  budget.process.emit(SETTLED);
  assert.equal((await finished(budget, spent.run_id)).error?.code, "workflow_budget_exhausted");
  assert.deepEqual(budget.signals.map(entry => entry.signal).filter(signal => signal.type === "limit"),
    [{ type: "limit", kind: "budget", code: "workflow_budget_exhausted", action: "fail" }]);

  const turns = traceFixture(undefined);
  const limited = await turns.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-turns", user_id: "U", prompt: "fix" });
  turns.process.emit({ type: "settled", outcome: "limit_reached", limit: "turns" } as PublicRunEvent);
  assert.equal((await finished(turns, limited.run_id)).error?.code, "agent_turn_limit");
  assert.deepEqual(turns.signals.map(entry => entry.signal).filter(signal => signal.type === "limit"),
    [{ type: "limit", kind: "turns", code: "agent_turn_limit", action: "fail" }]);

  let usage = 10;
  const quota = traceFixture(undefined, { diskQuota: { bytes: 100, intervalMs: 5, usage: async () => usage } });
  const grown = await quota.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-quota", user_id: "U", prompt: "fix" });
  usage = 101;
  assert.equal((await finished(quota, grown.run_id)).error?.code, "disk_quota_exceeded");
  assert.deepEqual(quota.signals.map(entry => entry.signal).filter(signal => signal.type === "limit"),
    [{ type: "limit", kind: "disk_quota", code: "disk_quota_exceeded", action: "fail" }]);

  const other = traceFixture(gate(async () => { throw new Error("docker daemon unreachable"); }));
  const broken = await other.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trace-other", user_id: "U", prompt: "fix" });
  await waitFor(() => other.process.prompts.length, count => count > 0, () => "the writer was never prompted");
  other.process.emit(SETTLED);
  assert.equal((await finished(other, broken.run_id)).error?.code, "quality_gate_unavailable");
  assert.deepEqual(other.signals.map(entry => entry.signal).filter(signal => signal.type === "limit"), []);
});
