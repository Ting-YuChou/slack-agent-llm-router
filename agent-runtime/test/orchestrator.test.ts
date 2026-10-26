import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator, RuntimeConflictError, RuntimeForbiddenError, publicRun } from "../src/orchestrator.js";
import { HarnessError } from "../src/error-taxonomy.js";
import { DiffLimitError, ProtectedFileError } from "../src/worktree-manager.js";
import { limitsForRoute, parseAgentLimitConfig } from "../src/agent-limits.js";
import type { RouteDecision } from "../src/jev-router.js";
import { promptTelemetry } from "../src/prompts.js";
import { piTurn, runScenario, settledRecord } from "./support/scenario.js";
import { waitFor } from "./support/wait.js";

const SETTLED = { type: "settled", outcome: "completed" } as const;

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompts: Array<{ runId: string; prompt: string; route?: { modelRef: string; effort: string }; taskMode?: string }> = [];
  decisions: Array<{ id: string; approved: boolean }> = [];
  aborted: string[] = [];
  prompt(runId: string, prompt: string, route?: { modelRef: string; effort: string }, taskMode?: string) { this.prompts.push({ runId, prompt, route, taskMode }); }
  decide(id: string, approved: boolean) { this.decisions.push({ id, approved }); }
  responses: Array<{ id: string; response: unknown }> = [];
  respondUi(id: string, response: unknown) { this.responses.push({ id, response }); }
  async abort(runId: string) { this.aborted.push(runId); }
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function fixture(router?: { route(text?: string): Promise<any> }, quality?: any, stateStore?: any, extra: Record<string, unknown> = {}) {
  const process = new FakeProcess();
  const calls = {
    rollback: [] as string[],
    commits: 0,
    diff: { files: ["src/a.ts"], bytes: 10, stat: "1 file changed" },
    commitImpl: async () => "b".repeat(40),
    commitMessages: [] as string[],
    inspectError: undefined as Error | undefined,
    rollbackError: undefined as Error | undefined,
    startedModels: [] as string[],
    applied: [] as Array<{ path: string; file: string }>,
    applyError: undefined as Error | undefined,
  };
  const worktrees = {
    create: async () => ({ path: "/managed/w1", branch: "pi-agent/20260801-run", baselineCommit: "a".repeat(40) }),
    inspectDiff: async () => { if (calls.inspectError) throw calls.inspectError; return calls.diff; },
    commit: async (_path: string, message: string) => { calls.commits += 1; calls.commitMessages.push(message); return calls.commitImpl(); },
    rollback: async (_path: string, baseline: string) => { calls.rollback.push(baseline); if (calls.rollbackError) throw calls.rollbackError; },
    remove: async () => undefined,
    applyPatch: async (path: string, file: string) => { if (calls.applyError) throw calls.applyError; calls.applied.push({ path, file }); },
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees,
    startProcess: async (session: { modelRef: string }, onEvent: (event: PublicRunEvent) => void) => {
      calls.startedModels.push(session.modelRef);
      process.handler = onEvent;
      return process;
    },
    maxActiveSessions: 2,
    deadlineMs: 10_000,
    router,
    quality, stateStore,
    mcpRepository: "acme/widgets",
    ...extra,
  } as any);
  return { orchestrator, process, calls };
}

test("reproduction time cannot erase human tool-approval waiting from workflow timing", async t => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const { orchestrator, process } = fixture(undefined, {
    elapsedMs: () => 1000, execution: () => {}, assertUnchanged: async () => {},
    evaluate: async () => ({ action: "fail", candidate: { hash: "h", path: "/s" }, verification: { status: "unknown", checks: [] } }),
  });
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "timing", user_id: "U", prompt: "fix" });
  t.mock.timers.tick(100);
  process.emit({ type: "approval", approval_id: "a", title: "Run check", detail: "check" });
  await waitFor(() => orchestrator.getRun(created.run_id).status, status => status === "awaiting_approval", status => `the approval was never recorded; run is ${status}`);
  t.mock.timers.tick(500);
  const approval = orchestrator.getRun(created.run_id).events.find(event => event.type === "approval") as any;
  await orchestrator.decide(created.run_id, approval.approval_id, "U", "approve");
  process.emit(SETTLED);
  await terminalRun(orchestrator, created.run_id);
  assert.equal(orchestrator.getRun(created.run_id).timing?.human_wait_ms, 500);
  assert.deepEqual(orchestrator.getRun(created.run_id).approvals, { requested: 1, approved: 1, rejected: 0, expired: 0, wait_ms: 500 });
});

test("quality failures repair without committing and unknown never commits", async () => {
  let evaluations = 0;
  const { orchestrator, process, calls } = fixture(undefined, {
    evaluate: async () => (++evaluations === 1 ? { action: "repair", repair_prompt: "fix regression", verification: { status: "failed", checks: [] }, candidate: { hash: "a", path: "/x" } } : { action: "fail", verification: { status: "unknown", checks: [] }, candidate: { hash: "b", path: "/x" } }),
    assertUnchanged: async () => {}, execution: () => {},
  });
  const r = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "quality", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await waitFor(() => process.prompts.length, count => count > 1, () => "the repair was never prompted");
  assert.equal(calls.commits, 0); assert.equal(process.prompts.length, 2);
  assert.notEqual(process.prompts[0].runId, process.prompts[1].runId);
  process.emit(SETTLED); await settledRun(orchestrator, r.run_id, calls);
  assert.equal(calls.commits, 0); assert.equal(orchestrator.getRun(r.run_id).status, "failed");
  assert.equal(calls.rollback.length, 1);
});

test("every phase a running run moves through is logged as an event, so followers of the event log see it", async () => {
  let evaluations = 0;
  const { orchestrator, process } = fixture(undefined, {
    evaluate: async () => (++evaluations === 1
      ? { action: "repair", repair_prompt: "fix regression", verification: { status: "failed", checks: [] }, candidate: { hash: "a", path: "/x" } }
      : { action: "pass", verification: { status: "passed", checks: [] }, grade: { status: "passed" }, candidate: { hash: "b", path: "/x" } }),
    assertUnchanged: async () => {}, execution: () => {},
  });
  const r = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "phases", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await waitFor(() => process.prompts.length, count => count > 1, () => "the repair was never prompted");
  process.emit(SETTLED); await waitFor(() => orchestrator.getRun(r.run_id).status, (status) => status === "completed", (status) => `run is ${status}`);
  const phases = orchestrator.listEvents(r.run_id).filter((event) => event.type === "phase").map((event) => event.phase);
  assert.deepEqual(phases, ["verifying", "repairing", "verifying", "finalizing"]);
});

test('structured discovery holds the session busy and finishes before the sole writer starts', async () => {
  let release!: (value: string) => void;
  const { orchestrator, process, calls } = fixture(undefined, {
    structured: { enabled: true, maxCostUsd: 5, activeTimeoutMs: 1_800_000 },
    discover: () => new Promise<string>(resolve => { release = resolve; }), writerStarted: () => {},
    execution: () => {}, evaluate: async () => ({ action: 'fail', candidate: { hash: 'h', path: '/s' }, verification: { status: 'unknown', checks: [] } }),
    assertUnchanged: async () => {},
  });
  const run = await orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'structured', user_id: 'U', prompt: 'fix', workflow_mode: 'structured' } as any);
  assert.equal(process.prompts.length, 0); assert.equal(calls.commits, 0);
  await assert.rejects(orchestrator.prompt(run.session_id, 'other', 'U'), RuntimeConflictError);
  release('Evidence: counter reset fails'); await prompted(process);
  assert.equal(process.prompts.length, 1); assert.match(process.prompts[0].prompt, /counter reset fails/);
  process.emit(SETTLED); await terminalRun(orchestrator, run.run_id);
});

test('a failed discovery reports why it failed', async () => {
  const { HarnessError } = await import('../src/error-taxonomy.js');
  const { orchestrator } = fixture(undefined, {
    structured: { enabled: true, maxCostUsd: 5, activeTimeoutMs: 1_800_000 },
    discover: async () => { throw new Error('investigation_attempts_exhausted', { cause: new HarnessError('investigation_timeout', 'investigation_timeout') }); },
    writerStarted: () => {}, execution: () => {}, assertUnchanged: async () => {},
    evaluate: async () => ({ action: 'fail', candidate: { hash: 'h', path: '/s' }, verification: { status: 'unknown', checks: [] } }),
  });
  const run = await orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'discovery-cause', user_id: 'U', prompt: 'fix', workflow_mode: 'structured' } as any);
  await waitFor(() => orchestrator.getRun(run.run_id).status, status => status === 'failed', status => `run is ${status}`);
  assert.deepEqual(orchestrator.getRun(run.run_id).error, { code: 'investigation_unknown', message: 'Required investigation could not complete; evidence retained', cause_code: 'investigation_timeout' });
});

test('a degraded investigation becomes a run event and the writer still starts', async () => {
  const degraded = { type: 'investigation_degraded', node_id: 'a', role: 'code_explorer', wave: 1, attempt: 2, cause: 'investigation_timeout' };
  const { orchestrator, process } = fixture(undefined, {
    structured: { enabled: true, maxCostUsd: 5, activeTimeoutMs: 1_800_000 },
    discover: async (input: any) => { input.degraded(degraded); return '## Open questions and unresolved\n- a (code_explorer) did not complete (investigation_timeout)'; },
    writerStarted: () => {}, execution: () => {}, assertUnchanged: async () => {},
    evaluate: async () => ({ action: 'fail', candidate: { hash: 'h', path: '/s' }, verification: { status: 'unknown', checks: [] } }),
  });
  const run = await orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'degraded', user_id: 'U', prompt: 'fix', workflow_mode: 'structured' } as any);
  await prompted(process);
  assert.deepEqual(orchestrator.getRun(run.run_id).events.filter(event => event.type === 'investigation_degraded'), [degraded]);
  assert.equal(process.prompts.length, 1); assert.match(process.prompts[0].prompt, /did not complete \(investigation_timeout\)/);
  assert.notEqual(orchestrator.getRun(run.run_id).status, 'failed');
  process.emit(SETTLED); await terminalRun(orchestrator, run.run_id);
});

test('structured requests fail closed without an enabled trusted profile', async () => {
  const { orchestrator, process } = fixture();
  await assert.rejects(orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'no-profile', user_id: 'U', prompt: 'fix', workflow_mode: 'structured' } as any), /structured|profile/i);
  assert.equal(process.prompts.length, 0);
});

test('cancelling discovery prevents a late writer and archives before rollback', async () => {
  let saved = false;
  const { orchestrator,process,calls } = fixture(undefined,{ structured: { enabled: true,maxCostUsd: 5,activeTimeoutMs: 1_800_000 }, execution: () => {},
    discover: (input: any) => new Promise((_resolve,reject) => input.signal.addEventListener('abort',() => reject(new Error('aborted')),{ once: true })),
    stop: async () => {}, archive: async () => { assert.equal(calls.rollback.length,0); saved = true; },
  });
  const run = await orchestrator.createSession({ team_id: 'T',channel_id: 'C',thread_ts: 'cancel-discovery',user_id: 'U',prompt: 'fix',workflow_mode: 'structured' } as any);
  await orchestrator.cancel(run.run_id,'U'); await terminalRun(orchestrator, run.run_id);
  assert.equal(process.prompts.length,0); assert.equal(saved,true); assert.equal(calls.rollback.length,1);
});

test('graceful shutdown interrupts read-only workflow without closing its resumable parent budget', async () => {
  let closed = 0;
  const { orchestrator,calls } = fixture(undefined,{ structured: { enabled: true,maxCostUsd: 5,activeTimeoutMs: 1_800_000 }, canResume: () => true,
    discover: (input: any) => new Promise((_resolve,reject) => input.signal.addEventListener('abort',() => reject(new Error('aborted')),{ once: true })),
    close: () => { closed++; }, stopDiscovery: async () => {},
  });
  const run = await orchestrator.createSession({ team_id: 'T',channel_id: 'C',thread_ts: 'shutdown-discovery',user_id: 'U',prompt: 'fix',workflow_mode: 'structured' } as any);
  await orchestrator.shutdown();
  assert.equal(orchestrator.getRun(run.run_id).status,'interrupted');
  assert.equal(closed,0); assert.equal(calls.rollback.length,0);
});

test('owner can resume an interrupted read-only workflow without resetting its elapsed budget', async () => {
  const baseline = 'a'.repeat(40);
  const stored: any = { run_id: 'R', session_id: 'S', owner_user_id: 'U', status: 'interrupted', workflow_mode: 'structured', workflow_requirement: 'fix', workflow_prompt: 'fix', baseline_commit: baseline,
    routing: { modelRef: 'openai/gpt-5.6-luna', effort: 'max', source: 'off' }, created_at: new Date(Date.now()-3000).toISOString(), updated_at: new Date().toISOString(), answer: '', events: [], attempts: [], estimated_cost_usd: .2 };
  const { orchestrator, process } = fixture(undefined, { structured: { enabled: true, maxCostUsd: 5, activeTimeoutMs: 1_800_000 }, canResume: () => true, resumeElapsed: () => 2000, tasks: () => [{ wave: 1, state: 'succeeded' }], discover: async () => 'cached evidence', writerStarted: () => {}, execution: () => {}, assertUnchanged: async () => {} }, {
    load: async () => ({ schema_version: 1, runs: [stored], sessions: [{ id: 'S', key: 'T:C:1', ownerUserId: 'U', modelRef: 'openai/gpt-5.6-luna', worktreePath: '/managed/w1', branch: 'b', baselineCommit: baseline, closed: false, lastActivity: Date.now() }] }),
    takeExpiredSessions: () => [], save: async () => {},
  });
  await orchestrator.restore();
  await assert.rejects(orchestrator.resume('R','other'),RuntimeForbiddenError);
  await orchestrator.resume('R','U'); await prompted(process);
  assert.equal(process.prompts.length,1); assert.match(process.prompts[0].prompt,/cached evidence/);
  assert.ok(orchestrator.getRun('R').timing!.active_ms >= 2000);
  await orchestrator.cancel('R','U');
});

test("model spend is recorded per role and sums to the run's estimated cost", async () => {
  let evaluations = 0;
  const route = { modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev", label: "simple", jevCostUsd: 0.0625 };
  const { orchestrator, process } = fixture({ route: async () => route }, {
    evaluate: async () => (++evaluations === 1
      ? { action: "repair", repair_prompt: "fix regression", verification: { status: "failed", checks: [] }, candidate: { hash: "a", path: "/x" } }
      : { action: "pass", verification: { status: "passed", checks: [] }, grade: { status: "passed" }, candidate: { hash: "b", path: "/x" } }),
    assertUnchanged: async () => {}, execution: () => {},
  });
  const r = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "cost-roles", user_id: "U", prompt: "fix", routing_text: "fix" } as any);
  await prompted(process);
  process.emit({ type: "usage", cost_usd: 0.25 });
  await orchestrator.qualityPhase(r.run_id, "grading", 0.125, "grader");
  await orchestrator.qualityPhase(r.run_id, "verifying");
  process.emit(SETTLED); await waitFor(() => process.prompts.length, count => count > 1, () => "the repair was never prompted");
  process.emit({ type: "usage", cost_usd: 0.5 });
  process.emit(SETTLED); await waitFor(() => orchestrator.getRun(r.run_id).status, status => status === "completed", status => `run is ${status}`);
  const run = orchestrator.getRun(r.run_id);
  assert.deepEqual(run.cost_by_role, { router: 0.0625, writer: 0.25, grader: 0.125, repair: 0.5 });
  assert.equal(run.estimated_cost_usd, 0.9375);
});

test("each finished tool call is counted per tool, with its errors", async () => {
  const { orchestrator, process } = fixture();
  const r = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "tool-errors", user_id: "U", prompt: "fix" });
  const call = (id: string, tool: string, is_error?: boolean) => {
    process.emit({ type: "tool", phase: "start", tool_call_id: id, tool });
    process.emit({ type: "tool", phase: "end", tool_call_id: id, tool, ...(is_error === undefined ? {} : { is_error }) });
  };
  call("1", "bash", true); call("2", "bash", false); call("3", "edit"); call("4", "mcp__codegraph__search", true);
  call("5", "__proto__", true);
  process.emit({ type: "tool", phase: "start", tool_call_id: "6", tool: "read" });
  process.emit(SETTLED); await terminalRun(orchestrator, r.run_id);
  const summary = orchestrator.getRun(r.run_id).tool_summary!;
  assert.deepEqual(Object.keys(summary), ["bash", "edit", "mcp__codegraph__search", "__proto__"]);
  assert.deepEqual([summary.bash, summary.edit, summary.mcp__codegraph__search], [{ calls: 2, errors: 1 }, { calls: 1, errors: 0 }, { calls: 1, errors: 1 }]);
  assert.deepEqual(Object.getOwnPropertyDescriptor(summary, "__proto__")?.value, { calls: 1, errors: 1 });
  assert.equal(Object.getPrototypeOf(summary), Object.prototype);
  assert.equal(({} as Record<string, unknown>).calls, undefined);
  assert.deepEqual(JSON.parse(JSON.stringify(orchestrator.getRun(r.run_id))).tool_summary.__proto__, { calls: 1, errors: 1 });
});

test("automatic session selects Sol for a complex run and Luna for its follow-up", async () => {
  const decisions = [
    { modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev", label: "complex" },
    { modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev", label: "read_only" },
  ];
  const { orchestrator, process } = fixture({ route: async () => decisions.shift()! });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "route", user_id: "U", prompt: "full context", routing_text: "refactor everything" } as any);
  assert.equal(orchestrator.getRun(first.run_id).model, "gpt-5.6-sol");
  assert.equal(process.prompts[0].route?.effort, "high");
  process.emit({ type: "usage", cost_usd: 0.42 });
  assert.equal(orchestrator.getRun(first.run_id).estimated_cost_usd, 0.42);
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  const second = await (orchestrator as any).prompt(first.session_id, "inspect result", "U", "inspect result");
  assert.equal(orchestrator.getRun(second.run_id).model, "gpt-5.6-luna");
  assert.equal(process.prompts[1].route?.effort, "low");
});

test("a run records the prompt version and the writer prompts it runs with", async () => {
  const { orchestrator } = fixture();
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "prompts", user_id: "U", prompt: "fix" });
  const run = orchestrator.getRun(created.run_id);
  assert.deepEqual({ prompts_version: run.prompts_version, prompts: run.prompts }, promptTelemetry("implementation"));
  assert.match(run.prompts_version ?? "", /^[0-9a-f]{12}$/);
});

test("a run records the turn and tool-call limits its route was started with", async () => {
  const { orchestrator, process } = fixture({ route: async () => ({ modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev", label: "complex" }) });
  const config = parseAgentLimitConfig({});
  (process as any).limitsFor = (route: RouteDecision) => limitsForRoute(route, config);
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "route-limits", user_id: "U", prompt: "refactor", routing_text: "refactor everything" } as any);
  assert.deepEqual(orchestrator.getRun(created.run_id).limits, { maxTurns: 30, maxToolCalls: 60 });
});

test("explicit Sol session ignores Jev and keeps Sol max on follow-up", async () => {
  let calls = 0;
  const { orchestrator, process } = fixture({ route: async () => { calls++; throw new Error("should not classify"); } });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "manual-sol", user_id: "U", prompt: "fix", model: "openai/gpt-5.6-sol", routing_text: "fix" } as any);
  assert.equal(process.prompts[0].route?.effort, "max");
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  await (orchestrator as any).prompt(first.session_id, "continue", "U", "continue");
  assert.equal(process.prompts[1].route?.modelRef, "openai/gpt-5.6-sol");
  assert.equal(calls, 0);
});

test("session API rejects non-string routing text", async () => {
  const { orchestrator } = fixture();
  await assert.rejects(orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "bad-route", user_id: "U", prompt: "fix", routing_text: { secret: "value" } } as any),
    (error: unknown) => (error as { statusCode?: number }).statusCode === 400);
});

test("a Jev client failure cannot prevent the default Luna run", async () => {
  const { orchestrator } = fixture({ route: async () => { throw new Error("Jev unavailable"); } });
  const result = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "fallback", user_id: "U", prompt: "inspect", routing_text: "inspect" });
  const run = orchestrator.getRun(result.run_id);
  assert.equal(run.model, "gpt-5.6-luna");
  assert.equal(run.reasoning_effort, "max");
});

test("a new session binds its allowlisted model and the same thread cannot switch providers", async () => {
  const { orchestrator, process, calls } = fixture();
  const first = await orchestrator.createSession({
    team_id: "T1",
    channel_id: "C1",
    thread_ts: "model-thread",
    user_id: "U1",
    prompt: "fix it",
    model: "anthropic/claude-sonnet-4-6",
  });

  assert.deepEqual(calls.startedModels, ["anthropic/claude-sonnet-4-6"]);
  assert.equal(orchestrator.getRun(first.run_id).model, "claude-sonnet-4-6");
  assert.equal(orchestrator.getRun(first.run_id).provider, "anthropic");
  assert.equal(orchestrator.lookupSession({ team_id: "T1", channel_id: "C1", thread_ts: "model-thread" })?.model_ref,
    "anthropic/claude-sonnet-4-6");

  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  await assert.rejects(orchestrator.createSession({
    team_id: "T1",
    channel_id: "C1",
    thread_ts: "model-thread",
    user_id: "U1",
    prompt: "continue",
    model: "openai/gpt-5.6-luna",
  }), /cannot switch/i);
});

test("a model without a configured provider key is rejected before creating a worktree", async () => {
  let creates = 0;
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => { creates += 1; return { path: "/w", branch: "b", baselineCommit: "base" }; },
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    startProcess: async () => new FakeProcess(),
    availableModelRefs: ["openai/gpt-5.6-luna"],
  });

  await assert.rejects(orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "missing-key", user_id: "U", prompt: "fix",
    model: "anthropic/claude-sonnet-4-6",
  }), (error: unknown) => {
    assert.equal((error as { code?: string }).code, "provider_not_configured");
    assert.equal((error as { statusCode?: number }).statusCode, 503);
    return true;
  });
  assert.equal(creates, 0);
});

test("creates an async thread session and commits only after Pi settles", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T1", channel_id: "C1", thread_ts: "1.0", user_id: "U1", prompt: "fix it" });

  assert.equal(accepted.status, "starting");
  assert.equal(process.prompts[0].prompt, "fix it");
  process.emit({ type: "answer", text: "Fixed" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, accepted.run_id);
  const run = orchestrator.getRun(accepted.run_id);

  assert.equal(run.status, "completed");
  assert.equal(run.answer, "Fixed");
  assert.deepEqual(run.changed_files, ["src/a.ts"]);
  assert.equal(run.commit, "b".repeat(40));
  assert.equal(calls.commits, 1);
});

test("counts Pi model turns without exposing thinking content", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "turns", user_id: "U", prompt: "inspect" });

  process.emit({ type: "turn" });
  process.emit({ type: "turn" });

  assert.equal(orchestrator.getRun(accepted.run_id).turn_count, 2);
});

test("run audit stores sanitized GitHub MCP metadata and clears unfinished call timing", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "mcp", user_id: "U", prompt: "inspect" });
  process.emit({ type: "tool", phase: "start", tool_call_id: "m1", tool: "mcp__github__get_file_contents" });
  process.emit({ type: "tool", phase: "end", tool_call_id: "m1", tool: "mcp__github__get_file_contents", is_error: false, result_bytes: 123 });
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.mcp_server, "github");
  assert.equal(run.mcp_tool, "get_file_contents");
  assert.equal(run.mcp_repository, "acme/widgets");
  assert.equal(run.mcp_success, true);
  assert.equal(run.mcp_error_code, null);
  assert.equal(run.mcp_result_bytes, 123);
  assert.equal(typeof run.mcp_latency_ms, "number");
  assert.deepEqual(run.mcp_summary?.total_calls, 1);
  assert.deepEqual(run.mcp_summary?.total_result_bytes, 123);
  assert.deepEqual(run.mcp_summary?.servers.github.calls, 1);
  assert.equal(JSON.stringify(run.events).includes("arguments"), false);
  assert.deepEqual(run.events.slice(-2).map((event) => event.type), ["mcp", "mcp"]);
  process.emit({ type: "tool", phase: "start", tool_call_id: "m2", tool: "mcp__github__search_code" });
  process.emit({ type: "tool", phase: "end", tool_call_id: "m2", tool: "mcp__github__search_code", is_error: true, result_bytes: 5 });
  process.emit({ type: "tool", phase: "start", tool_call_id: "m3", tool: "mcp__context7__query_docs" });
  process.emit({ type: "tool", phase: "end", tool_call_id: "m3", tool: "mcp__context7__query_docs", is_error: false, result_bytes: 10 });
  assert.equal(orchestrator.getRun(run.run_id).mcp_summary?.total_calls, 3);
  assert.equal(orchestrator.getRun(run.run_id).mcp_summary?.failures, 1);
  assert.equal(orchestrator.getRun(run.run_id).mcp_summary?.servers.context7.calls, 1);
  process.emit({ type: "tool", phase: "start", tool_call_id: "m4", tool: "mcp__github__search_code" });
  assert.equal((orchestrator as any).mcpToolStarts.size, 1);
  process.emit({ type: "error", code: "container_exited", message: "stopped" });
  await terminalRun(orchestrator, accepted.run_id);
  assert.equal((orchestrator as any).mcpToolStarts.size, 0);
});

test("approval is one-time, owner-bound, and resumes the exact Pi UI request", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T1", channel_id: "C1", thread_ts: "1.0", user_id: "U1", prompt: "edit" });
  process.emit({ type: "approval", approval_id: "rpc-ui-1", title: "Allow edit?", detail: "src/a.ts" });
  const run = orchestrator.getRun(accepted.run_id);
  const publicApproval = run.events.find((event: any) => event.type === "approval") as any;

  assert.equal(run.status, "awaiting_approval");
  await assert.rejects(orchestrator.decide(accepted.run_id, publicApproval.approval_id, "U2", "approve"), RuntimeForbiddenError);
  await orchestrator.decide(accepted.run_id, publicApproval.approval_id, "U1", "approve");
  assert.deepEqual(process.decisions, [{ id: "rpc-ui-1", approved: true }]);
  await assert.rejects(orchestrator.decide(accepted.run_id, publicApproval.approval_id, "U1", "approve"), RuntimeConflictError);
});

test("reject and cancel abort Pi and rollback to the prompt baseline", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T1", channel_id: "C1", thread_ts: "1.0", user_id: "U1", prompt: "edit" });
  process.emit({ type: "approval", approval_id: "rpc-ui-1", title: "Allow?", detail: "edit" });
  const approval = orchestrator.getRun(accepted.run_id).events.find((event: any) => event.type === "approval") as any;
  await orchestrator.decide(accepted.run_id, approval.approval_id, "U1", "reject");

  assert.equal(orchestrator.getRun(accepted.run_id).status, "rejected");
  assert.equal(orchestrator.getRun(accepted.run_id).approvals?.rejected, 1);
  assert.deepEqual(process.decisions, [{ id: "rpc-ui-1", approved: false }]);
  assert.deepEqual(calls.rollback, ["a".repeat(40)]);
});

test("same thread resumes its session and a busy session rejects overlapping prompts", async () => {
  const { orchestrator, process } = fixture();
  const first = await orchestrator.createSession({ team_id: "T1", channel_id: "C1", thread_ts: "1.0", user_id: "U1", prompt: "first" });
  await assert.rejects(orchestrator.prompt(first.session_id, "second", "U1"), RuntimeConflictError);
  process.emit({ type: "answer", text: "done" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  const second = await orchestrator.prompt(first.session_id, "second", "U1");

  assert.equal(second.session_id, first.session_id);
  assert.equal(process.prompts.length, 2);
});

test("abort fully settles before rollback", async () => {
  const order: string[] = [];
  let releaseAbort!: () => void;
  const abortSettled = new Promise<void>((resolve) => { releaseAbort = resolve; });
  const process = new FakeProcess();
  process.abort = async (runId: string) => {
    process.aborted.push(runId);
    order.push("abort-start");
    await abortSettled;
    order.push("abort-done");
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "b", baselineCommit: "a" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "b",
      rollback: async () => { order.push("rollback"); },
      remove: async () => undefined,
    },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "1", user_id: "U", prompt: "edit" });
  const cancelling = orchestrator.cancel(accepted.run_id, "U");
  await waitFor(() => order.length, count => count > 0, () => "the abort never started");
  assert.deepEqual(order, ["abort-start"]);
  releaseAbort();
  await cancelling;
  assert.deepEqual(order, ["abort-start", "abort-done", "rollback"]);
});

test("rollback is refused when container stop cannot be confirmed", async () => {
  const { orchestrator, process, calls } = fixture();
  process.abort = async () => { throw new Error("docker daemon unavailable"); };
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "stop-failure", user_id: "U", prompt: "edit" });

  await orchestrator.cancel(accepted.run_id, "U");

  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "container_stop_failed");
  assert.deepEqual(calls.rollback, []);
});

test("a settled run owns finalization and cannot be cancelled during commit", async () => {
  const { orchestrator, process, calls } = fixture();
  let releaseCommit!: () => void;
  const commitGate = new Promise<void>((resolve) => { releaseCommit = resolve; });
  calls.commits = 0;
  calls.commitImpl = async () => { await commitGate; return "b".repeat(40); };
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "2", user_id: "U", prompt: "fix" });
  process.emit(SETTLED);
  await assert.rejects(orchestrator.cancel(accepted.run_id, "U"), RuntimeConflictError);
  releaseCommit();
  await terminalRun(orchestrator, accepted.run_id);
  assert.equal(orchestrator.getRun(accepted.run_id).status, "completed");
});

test("approval cannot cross runs or replay after cancellation", async () => {
  const { orchestrator, process } = fixture();
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "3", user_id: "U", prompt: "first" });
  process.emit({ type: "approval", approval_id: "rpc-old", title: "Allow?", detail: "old" });
  const oldApproval = orchestrator.getRun(first.run_id).events.find((event: any) => event.type === "approval") as any;
  await orchestrator.cancel(first.run_id, "U");
  await assert.rejects(orchestrator.decide(first.run_id, oldApproval.approval_id, "U", "approve"), RuntimeConflictError);

  const second = await orchestrator.prompt(first.session_id, "second", "U");
  process.emit({ type: "approval", approval_id: "rpc-new", title: "Allow?", detail: "new" });
  await assert.rejects(orchestrator.decide(second.run_id, oldApproval.approval_id, "U", "approve"), RuntimeConflictError);
});

test("restore rolls back the latest interrupted prompt before starting Pi", async () => {
  const process = new FakeProcess();
  const order: string[] = [];
  const state = {
    schema_version: 1 as const,
    sessions: [{ id: "S", key: "T:C:4", ownerUserId: "U", worktreePath: "/managed/w", branch: "b", baselineCommit: "base", closed: false, lastActivity: Date.now() }],
    runs: [{ run_id: "R", session_id: "S", status: "interrupted" as const, answer: "", owner_user_id: "U", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:01Z", tool_count: 1, turn_count: 1, events: [] }],
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "", branch: "", baselineCommit: "" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "",
      rollback: async (_path, baseline) => { order.push(`rollback:${baseline}`); },
      remove: async () => undefined,
    },
    stateStore: {
      load: async () => state,
      save: async () => undefined,
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => { order.push("start"); process.handler = onEvent; return process; },
  });
  await orchestrator.restore();
  assert.deepEqual(order, ["rollback:base", "start"]);
});

test("legacy restored sessions stay fixed at their saved model and max effort", async () => {
  const process = new FakeProcess();
  const state = {
    schema_version: 1 as const,
    sessions: [{ id: "legacy", key: "T:C:legacy", ownerUserId: "U", modelRef: "openai/gpt-5.6-sol", worktreePath: "/managed/w", branch: "b", baselineCommit: "base", closed: false, lastActivity: Date.now() }],
    runs: [],
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "", branch: "", baselineCommit: "" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "",
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    stateStore: { load: async () => state, save: async () => undefined, takeExpiredSessions: () => [] } as any,
    router: { route: async () => ({ modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev" }) },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  await orchestrator.restore();
  await orchestrator.prompt("legacy", "continue", "U", "inspect only");
  assert.deepEqual(process.prompts[0]?.route, { modelRef: "openai/gpt-5.6-sol", effort: "max", source: "explicit" });
});

test("restored automatic sessions reroute each follow-up", async () => {
  const process = new FakeProcess();
  const state = {
    schema_version: 1 as const,
    sessions: [{ id: "auto", key: "T:C:auto", ownerUserId: "U", modelRef: "openai/gpt-5.6-sol", autoRouting: true, worktreePath: "/managed/w", branch: "b", baselineCommit: "base", closed: false, lastActivity: Date.now() }],
    runs: [],
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "", branch: "", baselineCommit: "" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "",
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    stateStore: { load: async () => state, save: async () => undefined, takeExpiredSessions: () => [] } as any,
    router: { route: async () => ({ modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev" }) },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  await orchestrator.restore();
  await orchestrator.prompt("auto", "continue", "U", "inspect only");
  assert.deepEqual(process.prompts[0]?.route, { modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev" });
});

test("session close is rejected while a routed follow-up is pending", async () => {
  let releaseRoute!: (decision: any) => void;
  let routeCalls = 0;
  const { orchestrator, process } = fixture({ route: async () => {
    routeCalls += 1;
    if (routeCalls === 1) return { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "fallback" };
    return new Promise((resolve) => { releaseRoute = resolve; });
  } });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "close-race", user_id: "U", prompt: "first", routing_text: "first" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);

  const pendingPrompt = orchestrator.prompt(first.session_id, "follow up", "U", "follow up");
  await waitFor(() => routeCalls, count => count > 1, () => "the follow-up never reached the router");
  await assert.rejects(orchestrator.closeSession(first.session_id, "U"), RuntimeConflictError);
  releaseRoute({ modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev" });
  await pendingPrompt;
});

test("routed follow-up is rejected while session close is pending", async () => {
  const { orchestrator, process } = fixture();
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "prompt-race", user_id: "U", prompt: "first" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  let releaseClose!: () => void;
  const closeGate = new Promise<void>((resolve) => { releaseClose = resolve; });
  let closeStarted = false;
  process.close = async () => { closeStarted = true; await closeGate; };

  const pendingClose = orchestrator.closeSession(first.session_id, "U");
  await waitFor(() => closeStarted, Boolean, () => "the session close never reached the process");
  await assert.rejects(orchestrator.prompt(first.session_id, "follow up", "U", "follow up"), RuntimeConflictError);
  releaseClose();
  await pendingClose;
});

test("parallel first prompts in the same Slack thread create only one session", async () => {
  const process = new FakeProcess();
  let releaseCreate!: () => void;
  const createGate = new Promise<void>((resolve) => { releaseCreate = resolve; });
  let creates = 0;
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => { creates += 1; await createGate; return { path: "/w", branch: "b", baselineCommit: "base" }; },
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  const input = { team_id: "T", channel_id: "C", thread_ts: "parallel", user_id: "U", prompt: "fix" };

  const first = orchestrator.createSession(input);
  await assert.rejects(orchestrator.createSession(input), RuntimeConflictError);
  assert.equal(creates, 1);
  releaseCreate();
  await first;
});

test("terminal capture completeness is persisted and retained on restore", async () => {
  let state: any = {schema_version:1,sessions:[],runs:[]};
  const store: any = {load:async()=>structuredClone(state),save:async(value:any)=>{state=structuredClone(value);},takeExpiredSessions:()=>[]};
  const {orchestrator,process} = fixture();
  (orchestrator as any).stateStore = store;
  (orchestrator as any).captureState = () => ({complete:false,reasons:["final_snapshot_failed"]});
  const first = await orchestrator.createSession({team_id:"T",channel_id:"C",thread_ts:"capture",user_id:"U",prompt:"fix"} as any);
  process.emit(SETTLED);
  await terminalRun(orchestrator,first.run_id);
  assert.equal(state.runs[0].capture_complete,false);
  state.sessions = [];
  const restored = fixture().orchestrator;
  (restored as any).stateStore = store;
  (restored as any).captureState = () => ({complete:true,reasons:[]});
  await restored.restore();
  assert.equal((restored.getRun(first.run_id) as any).capture_complete,false);
  assert.deepEqual((restored.getRun(first.run_id) as any).capture_reasons,["final_snapshot_failed"]);
});

test("restored newly interrupted runs retain known incomplete capture reasons", async () => {
  const {orchestrator} = fixture();
  const observed: any[] = [];
  const state = {schema_version:1,sessions:[],runs:[{run_id:"r",session_id:"s",status:"interrupted",events:[]}]};
  (orchestrator as any).stateStore = {load:async()=>state,takeExpiredSessions:()=>[],save:async()=>{}};
  (orchestrator as any).captureState = () => ({complete:false,reasons:["container_stopped_before_final_snapshot"]});
  (orchestrator as any).observeRun = (run:any)=>observed.push(run);
  await orchestrator.restore();
  assert.equal(observed[0].capture_complete,false);
  assert.deepEqual(observed[0].capture_reasons,["container_stopped_before_final_snapshot"]);
});

test("failed terminal observability writes persist an incomplete run and stay terminal", async () => {
  let state: any;
  const {orchestrator,process} = fixture();
  (orchestrator as any).stateStore = {save:async (value:any)=>{state=structuredClone(value);}};
  (orchestrator as any).captureState = () => ({complete:true,reasons:[]});
  const created = await orchestrator.createSession({team_id:"T",channel_id:"C",thread_ts:"failed-telemetry",user_id:"U",prompt:"fix"} as any);
  process.emit(SETTLED);
  await terminalRun(orchestrator,created.run_id);
  assert.equal(state.runs[0].capture_complete,true);
  await (orchestrator as any).markCaptureIncomplete(created.run_id,"terminal_outbox_write_failed");
  assert.equal(state.runs[0].capture_complete,false);
  assert.deepEqual(state.runs[0].capture_reasons,["terminal_outbox_write_failed"]);
  assert.equal(state.runs[0].status,"completed");
});

test("quality gate holds the session busy until candidate verification and grading finish", async () => {
  let finish!: (result: any) => void; let unchanged = 0;
  const { orchestrator, process, calls } = fixture(undefined, {
    evaluate: () => new Promise(resolve => { finish = resolve; }),
    assertUnchanged: async () => { unchanged++; }, execution: () => {},
  });
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "gated", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await waitFor(() => finish, Boolean, () => "the quality gate never started");
  await assert.rejects(orchestrator.prompt(run.session_id, "another", "U"), RuntimeConflictError);
  assert.equal(calls.commits, 0);
  finish({ action: "pass", candidate: { hash: "hash", path: "/snapshot" }, verification: { status: "passed", checks: [] }, grade: { status: "passed", findings: [] } });
  await terminalRun(orchestrator, run.run_id);
  assert.equal(calls.commits, 1); assert.equal(unchanged, 1);
  assert.equal(orchestrator.getRun(run.run_id).verification_status, "passed");
});

test("unconfirmed verifier/grader stop prevents rollback", async () => {
  let archived = false;
  const { orchestrator, process, calls } = fixture(undefined, {
    evaluate: async () => { throw new Error("unused"); }, assertUnchanged: async () => {}, execution: () => {},
    stop: async () => { throw new Error("stop failed"); }, archive: async () => { archived = true; },
  });
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "stop-quality", user_id: "U", prompt: "fix" });
  await orchestrator.cancel(run.run_id, "U");
  assert.equal(process.aborted.length, 1); assert.equal(archived, false); assert.equal(calls.rollback.length, 0);
});

test("failed candidate is archived after workers stop and before rollback", async () => {
  const sequence: string[] = [];
  const { orchestrator, process, calls } = fixture(undefined, {
    evaluate: async () => ({ action: "fail", candidate: { hash: "hash", path: "/snapshot" }, verification: { status: "unknown", checks: [] } }),
    assertUnchanged: async () => {}, execution: () => {},
    stop: async () => { assert.equal(process.aborted.length, 1); sequence.push("stopped"); },
    archive: async () => { assert.equal(calls.rollback.length, 0); sequence.push("archived"); },
  });
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "archive", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await terminalRun(orchestrator, run.run_id);
  assert.deepEqual(sequence, ["stopped", "archived"]); assert.equal(calls.rollback.length, 1);
  assert.equal(orchestrator.getRun(run.run_id).status, "failed");
});

test("cancelling while initial state persists cannot launch a late writer", async () => {
  let release!: () => void; let saves = 0;
  const blocked = new Promise<void>(resolve => { release = resolve; });
  const { orchestrator, process } = fixture(undefined, undefined, { save: () => ++saves === 1 ? blocked : Promise.resolve() });
  const input = { team_id: "T", channel_id: "C", thread_ts: "persist-cancel", user_id: "U", prompt: "fix" };
  const creating = orchestrator.createSession(input);
  await waitFor(() => orchestrator.lookupSession(input)?.run_id, Boolean, () => "the session was never registered");
  const lookup = orchestrator.lookupSession(input)!;
  await orchestrator.cancel(String(lookup.run_id), "U");
  release(); await creating;
  assert.equal(process.prompts.length, 0);
});

test("grader receives the task requirement instead of Slack bootstrap conversation", async () => {
  let requirement = "";
  const { orchestrator, process } = fixture(undefined, {
    evaluate: async (input: any) => { requirement = input.requirement; return { action: "fail", candidate: { hash: "h", path: "/s" }, verification: { status: "unknown", checks: [] } }; },
    assertUnchanged: async () => {}, execution: () => {},
  });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "grader-context", user_id: "U", prompt: "Prior Slack conversation and author claims", routing_text: "Fix the failing counter test" });
  process.emit(SETTLED); await waitFor(() => requirement, Boolean, () => "the grader was never called");
  assert.equal(requirement, "Fix the failing counter test");
});


test('discussion keeps usage/budget but skips quality/commit and follow-up implement still verifies', async () => {
  let evaluations = 0, closed = 0; const roles: string[] = [];
  const { orchestrator, process, calls } = fixture(undefined, {
    execution: (_id: string, _parent: string, role: string) => { roles.push(role); },
    evaluate: async () => { evaluations++; return { action: 'pass', candidate: { hash: 'h', path: '/s' }, verification: { status: 'passed', checks: [] }, grade: { status: 'passed' } }; },
    assertUnchanged: async () => {}, close: () => { closed++; },
  });
  calls.diff = { files: [], bytes: 0, stat: '' };
  const request = { team_id: 'T', channel_id: 'C', thread_ts: 'discussion', user_id: 'U', prompt: 'explain', task_mode: 'discuss' };
  const first = await orchestrator.createSession(request as any);
  assert.equal(process.prompts[0].taskMode,'discuss');
  process.emit({ type: 'answer',text: 'Project explanation' }); process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  const run = orchestrator.getRun(first.run_id);
  assert.equal(run.status,'completed'); assert.equal(run.answer,'Project explanation');
  assert.equal(run.verification_status,'not_applicable'); assert.equal(run.grader_status,'not_applicable');
  assert.equal(run.publication_status,'not_applicable'); assert.equal(run.commit,undefined);
  assert.equal(evaluations,0); assert.equal(calls.commits,0); assert.equal(closed,1);
  assert.deepEqual(roles,['discussion','discussion']);
  const second = await orchestrator.createSession({ ...request, task_mode: undefined, prompt: 'implement fix' } as any);
  assert.equal(second.session_id,first.session_id); assert.equal(process.prompts[1].taskMode,'implement');
  process.emit(SETTLED); await terminalRun(orchestrator, second.run_id);
  assert.equal(evaluations,1); assert.equal(calls.commits,1);
});

test('discussion cannot silently deliver unexpected changes even without quality configuration', async () => {
  const { orchestrator, process, calls } = fixture();
  const r = await orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'drift', user_id: 'U', prompt: 'explain', task_mode: 'discuss' } as any);
  process.emit(SETTLED); await terminalRun(orchestrator, r.run_id);
  assert.equal(orchestrator.getRun(r.run_id).status,'failed'); assert.equal(calls.commits,0);
  assert.equal(orchestrator.getRun(r.run_id).error?.code,'discussion_workspace_changed');
  assert.equal(calls.rollback.length,1);
});

test('invalid task mode fails before dispatch and no-diff discussion needs no profile', async () => {
  const { orchestrator, process, calls } = fixture();
  const request = { team_id: 'T', channel_id: 'C', thread_ts: 'mode', user_id: 'U', prompt: 'explain' };
  await assert.rejects(orchestrator.createSession({ ...request, task_mode: 'auto' } as any), /task_mode/);
  assert.equal(process.prompts.length,0);
  calls.diff = { files: [], bytes: 0, stat: '' };
  const r = await orchestrator.createSession({ ...request, task_mode: 'discuss' } as any);
  process.emit(SETTLED); await terminalRun(orchestrator, r.run_id);
  assert.equal(orchestrator.getRun(r.run_id).status,'completed'); assert.equal(calls.commits,0);
});

test('structured discussion investigates then uses a read-only final responder without a quality gate', async () => {
  let evaluations = 0;
  const { orchestrator, process, calls } = fixture(undefined, {
    structured: { enabled: true, maxCostUsd: 5, activeTimeoutMs: 1_800_000 },
    discover: async () => 'Evidence from independent readers', writerStarted: () => {},
    execution: () => {}, evaluate: async () => { evaluations++; throw new Error('not applicable'); },
    assertUnchanged: async () => {},
  });
  calls.diff = { files: [], bytes: 0, stat: '' };
  const r = await orchestrator.createSession({ team_id: 'T', channel_id: 'C', thread_ts: 'structured-discuss', user_id: 'U', prompt: 'explain', task_mode: 'discuss', workflow_mode: 'structured' } as any);
  await prompted(process);
  assert.match(process.prompts[0].prompt,/Evidence/); assert.equal(process.prompts[0].taskMode,'discuss');
  process.emit(SETTLED); await terminalRun(orchestrator, r.run_id);
  assert.equal(orchestrator.getRun(r.run_id).verification_status,'not_applicable');
  assert.equal(orchestrator.getRun(r.run_id).status,'completed'); assert.equal(evaluations,0); assert.equal(calls.commits,0);
});

test("background persistence failures mark the control store unhealthy instead of rejecting unhandled", async () => {
  let failSaves = false;
  const stateStore = {
    healthy: true,
    load: async () => ({ schema_version: 1, sessions: [], runs: [] }),
    takeExpiredSessions: () => [],
    save: async () => { if (failSaves) throw new Error("disk full"); },
  };
  const { orchestrator, process: pi } = fixture(undefined, undefined, stateStore);
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "persist", user_id: "U", prompt: "inspect" });
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
  process.on("unhandledRejection", onUnhandled);
  const originalError = console.error;
  console.error = () => undefined;
  try {
    failSaves = true;
    pi.emit({ type: "turn" });
    pi.emit({ type: "usage", cost_usd: 0.01 } as PublicRunEvent);
    pi.emit({ type: "tool", phase: "start", tool_call_id: "t1", tool: "read" });
    pi.emit({ type: "answer", text: "partial" });
    pi.emit({ type: "approval", approval_id: "a1", title: "Allow?", detail: "edit" });
    await waitFor(() => stateStore.healthy === false && pi.aborted.includes(accepted.run_id), Boolean, () => "persistence failure never stopped the run");
  } finally {
    process.off("unhandledRejection", onUnhandled);
    console.error = originalError;
  }
  assert.deepEqual(unhandled, []);
  assert.equal(stateStore.healthy, false);
  assert.ok(pi.aborted.includes(accepted.run_id));
  await assert.rejects(
    orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "after", user_id: "U", prompt: "next" }),
    /control store unavailable/i,
  );
});

for (const [outcome, limit, code] of [
  ["truncated", undefined, "agent_output_truncated"],
  ["limit_reached", "tool_calls", "agent_tool_limit"],
  ["limit_reached", "turns", "agent_turn_limit"],
  ["error", undefined, "provider_error"],
  ["aborted", undefined, "agent_aborted"],
] as const) {
  test(`a ${outcome}${limit ? `/${limit}` : ""} settle fails the run as ${code} and never commits`, async () => {
    const { orchestrator, process, calls } = fixture();
    const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: `outcome-${code}`, user_id: "U", prompt: "fix" });
    process.emit({ type: "settled", outcome, ...(limit ? { limit } : {}), ...(outcome === "error" ? { error_message: "upstream 500" } : {}) });
    const run = await settledRun(orchestrator, accepted.run_id, calls);
    assert.equal(run.status, "failed");
    assert.equal(run.error?.code, code);
    assert.equal(run.execution_outcome, outcome);
    assert.equal(calls.commits, 0);
    assert.equal(calls.rollback.length, 1);
  });
}

test("a verified workflow never sends a truncated execution to the quality gate", async () => {
  let evaluated = 0;
  const { orchestrator, process, calls } = fixture(undefined, {
    execution: () => {}, assertUnchanged: async () => {}, maxCostUsd: 5,
    evaluate: async () => { evaluated += 1; return { action: "pass", candidate: { hash: "h", path: "/s" }, verification: { status: "passed", checks: [] } }; },
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "wf-truncated", user_id: "U", prompt: "fix" });
  await prompted(process);
  process.emit({ type: "settled", outcome: "truncated", stop_reason: "length" });
  const run = await settledRun(orchestrator, accepted.run_id, calls);
  assert.equal(evaluated, 0);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "agent_output_truncated");
  assert.equal(run.execution_status, "failed");
  assert.equal(calls.commits, 0);
});

test("a completed settle records its outcome and still commits", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "outcome-ok", user_id: "U", prompt: "fix" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, accepted.run_id);
  assert.equal(orchestrator.getRun(accepted.run_id).execution_outcome, "completed");
  assert.equal(calls.commits, 1);
});

test("an unverified no-profile change is labelled unverified in the run and the commit trailers", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "unverified", user_id: "U", prompt: "fix" });
  assert.equal(orchestrator.getRun(accepted.run_id).verification_status, "unverified");
  process.emit(SETTLED);
  await terminalRun(orchestrator, accepted.run_id);
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "completed");
  assert.equal(run.verification_status, "unverified");
  const message = calls.commitMessages[0];
  // The subject summarises the task ("fix"); the run id moved to the Pi-Run-Id trailer below.
  assert.match(message, /^Pi agent: fix\n\n/);
  assert.match(message, new RegExp(`\nPi-Run-Id: ${accepted.run_id}(\n|$)`));
  assert.match(message, /\nPi-Model: openai\/gpt-5\.6-luna(\n|$)/);
  assert.match(message, /\nPi-Verification: unverified(\n|$)/);
});

test("a verified change records its verification and grader verdicts as commit trailers", async () => {
  const { orchestrator, process, calls } = fixture(undefined, {
    execution: () => {}, assertUnchanged: async () => {}, maxCostUsd: 5,
    evaluate: async () => ({ action: "pass", candidate: { hash: "h", path: "/s" }, verification: { status: "passed", checks: [] }, grade: { status: "passed", candidate_hash: "h", criteria: [], findings: [] } }),
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "verified-trailers", user_id: "U", prompt: "fix" });
  await prompted(process);
  process.emit(SETTLED);
  await settledRun(orchestrator, accepted.run_id, calls);
  assert.match(calls.commitMessages[0] ?? "", /\nPi-Verification: passed\nPi-Grader: passed(\n|$)/);
});

test("requireVerification rejects implement without a trusted profile but still allows discussion", async () => {
  const { orchestrator } = fixture(undefined, undefined, undefined, { requireVerification: true });
  await assert.rejects(
    orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "req-1", user_id: "U", prompt: "fix" }),
    (error: any) => error.code === "verification_profile_required" && error.statusCode === 422,
  );
  const discussed = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "req-2", user_id: "U", prompt: "why?", task_mode: "discuss" });
  assert.equal(discussed.status, "starting");
});

function gatedFixture(evaluate: () => Promise<any>) {
  return fixture(undefined, { execution: () => {}, assertUnchanged: async () => {}, maxCostUsd: 5, evaluate });
}

async function settleWorkflow(f: ReturnType<typeof fixture>, thread: string) {
  const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: thread, user_id: "U", prompt: "fix" });
  await prompted(f.process);
  f.process.emit(SETTLED);
  return settledRun(f.orchestrator, accepted.run_id, f.calls);
}

/** Waits until the workflow has prompted the writer, so a settle reaches a running execution. */
async function prompted(process: FakeProcess) {
  await waitFor(() => process.prompts.length, count => count > 0, () => "the writer was never prompted");
}

/** Waits until the run is terminal and its worktree was either committed or rolled back. */
function settledRun(orchestrator: CodingAgentOrchestrator, runId: string, calls: ReturnType<typeof fixture>["calls"]) {
  return waitFor(() => orchestrator.getRun(runId), run => ["completed", "failed", "timed_out", "cancelled"].includes(run.status) && calls.commits + calls.rollback.length > 0,
    run => `run never settled; status ${run.status}, commits ${calls.commits}, rollbacks ${calls.rollback.length}`);
}

/** Waits until the run reaches a terminal status, which is also when its session is released. */
function terminalRun(orchestrator: CodingAgentOrchestrator, runId: string) {
  return waitFor(() => orchestrator.getRun(runId), run => ["completed", "rejected", "cancelled", "failed", "timed_out", "interrupted"].includes(run.status),
    run => `run never finished; status ${run.status}`);
}

test("a scope violation the gate cannot repair fails as scope_violation, not quality_gate_unavailable", async () => {
  const f = gatedFixture(async () => ({ action: "fail", failure: { code: "scope_violation", files: ["docs/x.md"] }, candidate: { hash: "", path: "/w" }, verification: { status: "failed", checks: [] } }));
  const run = await settleWorkflow(f, "scope-fail");
  assert.equal(run.error?.code, "scope_violation");
  assert.match(run.error?.message ?? "", /docs\/x\.md/);
  assert.equal(f.calls.commits, 0);
});

test("diff limit violations inside the gate report diff_limit_exceeded", async () => {
  const f = gatedFixture(async () => { throw new Error("unreachable"); });
  f.calls.inspectError = new DiffLimitError("Changed file limit exceeded (51/50)");
  const run = await settleWorkflow(f, "diff-limit-gate");
  assert.equal(run.error?.code, "diff_limit_exceeded");
  assert.match(run.error?.message ?? "", /51\/50/);
});

test("protected file changes report protected_file_modified", async () => {
  const f = gatedFixture(async () => { throw new Error("unreachable"); });
  f.calls.inspectError = new ProtectedFileError("Protected file modification detected: .env");
  const run = await settleWorkflow(f, "protected-gate");
  assert.equal(run.error?.code, "protected_file_modified");
});

test("budget exhaustion during the gate reports workflow_budget_exhausted", async () => {
  const f = gatedFixture(async () => { throw new Error("workflow_budget_exhausted"); });
  const run = await settleWorkflow(f, "budget-gate");
  assert.equal(run.error?.code, "workflow_budget_exhausted");
});

test("other gate failures stay quality_gate_unavailable but keep a bounded cause", async () => {
  const f = gatedFixture(async () => { throw new Error("docker daemon unreachable"); });
  const run = await settleWorkflow(f, "gate-cause");
  assert.equal(run.error?.code, "quality_gate_unavailable");
  assert.match(run.error?.message ?? "", /docker daemon unreachable/);
});

test("gate failures with a known cause record cause_code on the run", async () => {
  const f = gatedFixture(async () => { throw new HarnessError("docker_unavailable", "Cannot connect to the Docker daemon"); });
  const run = await settleWorkflow(f, "gate-cause-code");
  assert.equal(run.error?.code, "quality_gate_unavailable");
  assert.equal(run.error?.cause_code, "docker_unavailable");
  assert.equal(f.calls.commits, 0);
});

test("a no-profile diff over the limits fails as diff_limit_exceeded and rolls back", async () => {
  const { orchestrator, process, calls } = fixture();
  calls.inspectError = new DiffLimitError("Total diff limit exceeded (2000000/1048576)");
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "diff-limit-plain", user_id: "U", prompt: "fix" });
  process.emit(SETTLED);
  const run = await settledRun(orchestrator, accepted.run_id, calls);
  assert.equal(run.error?.code, "diff_limit_exceeded");
  assert.equal(calls.commits, 0);
  assert.equal(calls.rollback.length, 1);
});

function countingStore(failing = () => false) {
  const store = { healthy: true, saves: 0, load: async () => ({ schema_version: 1, sessions: [], runs: [] }), takeExpiredSessions: () => [],
    save: async () => { store.saves += 1; if (failing()) { store.healthy = false; throw new Error("disk full"); } } };
  return store;
}

test("answer deltas accumulate into the run answer without becoming events", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "deltas", user_id: "U", prompt: "explain" });
  process.emit({ type: "answer_delta", text: "Hel" });
  process.emit({ type: "answer_delta", text: "lo" });
  assert.equal(orchestrator.getRun(accepted.run_id).answer, "Hello");
  process.emit({ type: "answer", text: "Hello world" });
  assert.equal(orchestrator.getRun(accepted.run_id).answer, "Hello world");
  process.emit({ type: "answer_delta", text: "Next" });
  assert.equal(orchestrator.getRun(accepted.run_id).answer, "Next");
  assert.equal(orchestrator.getRun(accepted.run_id).events.filter((event: any) => event.type === "answer_delta").length, 0);
});

test("observational events are persisted with a debounce; status transitions persist immediately", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const store = countingStore();
  const { orchestrator, process } = fixture(undefined, undefined, store);
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "debounce", user_id: "U", prompt: "explain" });
  const before = store.saves;
  for (let i = 0; i < 100; i++) process.emit({ type: "answer_delta", text: "x" });
  for (let i = 0; i < 10; i++) process.emit({ type: "tool", phase: "start", tool_call_id: `t${i}`, tool: "read" });
  process.emit({ type: "turn" });
  await waitFor(() => orchestrator.getRun(created.run_id).turn_count, count => count > 0, () => "the turn was never recorded");
  assert.equal(store.saves, before);
  t.mock.timers.tick(1_000);
  await waitFor(() => store.saves, saves => saves > before, () => "the debounced save never ran");
  assert.equal(store.saves, before + 1);
  process.emit({ type: "approval", approval_id: "a1", title: "Allow?", detail: "edit" });
  await waitFor(() => store.saves, saves => saves > before + 1, () => "the status transition was never persisted");
  assert.equal(store.saves, before + 2);
});

test("finished runs are not re-observed on every later persist", async () => {
  const observed = new Map<string, number>();
  const { orchestrator, process } = fixture(undefined, undefined, undefined, { observeRun: (run: any) => observed.set(run.run_id, (observed.get(run.run_id) ?? 0) + 1) });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "obs-1", user_id: "U", prompt: "one" });
  process.emit(SETTLED);
  await terminalRun(orchestrator, first.run_id);
  assert.equal(orchestrator.getRun(first.run_id).status, "completed");
  const settledCount = observed.get(first.run_id)!;
  const second = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "obs-2", user_id: "U", prompt: "two" });
  const observedSecond = observed.get(second.run_id) ?? 0;
  process.emit({ type: "approval", approval_id: "a1", title: "Allow?", detail: "edit" });
  await waitFor(() => observed.get(second.run_id) ?? 0, count => count > observedSecond, () => "the approval was never persisted");
  assert.equal(observed.get(first.run_id), settledCount);
});

test("persistence failures during completion, failure and timeout never become unhandled rejections", async () => {
  let failSaves = false;
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
  process.on("unhandledRejection", onUnhandled);
  const originalError = console.error;
  console.error = () => undefined;
  try {
    for (const [thread, trigger] of [["save-complete", "settle"], ["save-fail", "error"], ["save-timeout", "timeout"]] as const) {
      failSaves = false;
      const store = countingStore(() => failSaves);
      const f = fixture(undefined, undefined, store, { deadlineMs: trigger === "timeout" ? 5 : 10_000 });
      const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: thread, user_id: "U", prompt: "fix" });
      failSaves = true;
      if (trigger === "settle") f.process.emit(SETTLED);
      if (trigger === "error") f.process.emit({ type: "error", code: "container_exited", message: "gone" });
      if (trigger === "timeout") await new Promise((resolve) => setTimeout(resolve, 20));
      await settledRun(f.orchestrator, accepted.run_id, f.calls);
    }
  } finally {
    process.off("unhandledRejection", onUnhandled);
    console.error = originalError;
  }
  assert.deepEqual(unhandled, []);
});

test("a failed no-profile run archives its patch before rollback", async () => {
  const order: string[] = [];
  const { orchestrator, process, calls } = fixture(undefined, undefined, undefined, {
    archivePatch: async (input: { run_id: string; worktree: string; baseline: string }) => {
      order.push(`archive:${calls.rollback.length}`);
      assert.equal(input.worktree, "/managed/w1"); assert.equal(input.baseline, "a".repeat(40));
      return `/runtime/artifacts/${input.run_id}/undelivered.patch`;
    },
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "no-profile-archive", user_id: "U", prompt: "edit" });
  process.emit({ type: "settled", outcome: "aborted" });
  await terminalRun(orchestrator, accepted.run_id);
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "agent_aborted");
  assert.deepEqual(order, ["archive:0"]);
  assert.equal(calls.rollback.length, 1);
  assert.equal(run.patch_artifact, `/runtime/artifacts/${accepted.run_id}/undelivered.patch`);
});

test("cancelled runs archive their patch too, and an empty diff records none", async () => {
  const { orchestrator, calls } = fixture(undefined, undefined, undefined, { archivePatch: async () => undefined });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "cancel-archive", user_id: "U", prompt: "edit" });
  await orchestrator.cancel(accepted.run_id, "U");
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "cancelled");
  assert.equal(run.patch_artifact, undefined);
  assert.equal(calls.rollback.length, 1);
});

test("an archive failure keeps the worktree and is not reported as a container stop failure", async () => {
  const { orchestrator, process, calls } = fixture(undefined, undefined, undefined, {
    archivePatch: async () => { throw new Error("disk full"); },
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "archive-failure", user_id: "U", prompt: "edit" });
  process.emit({ type: "settled", outcome: "aborted" });
  await terminalRun(orchestrator, accepted.run_id);
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "archive_failed");
  assert.deepEqual(calls.rollback, []);
  assert.equal(process.aborted.length, 1, "the writer was still stopped");
});

test("a quality archive failure is archive_failed, not container_stop_failed", async () => {
  const { orchestrator, calls } = fixture(undefined, {
    evaluate: async () => { throw new Error("unused"); }, assertUnchanged: async () => {}, execution: () => {},
    stop: async () => {}, archive: async () => { throw new Error("snapshot failed"); },
  });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "quality-archive-failure", user_id: "U", prompt: "fix" });
  await orchestrator.cancel(accepted.run_id, "U");
  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.error?.code, "archive_failed");
  assert.deepEqual(calls.rollback, []);
});

test("a 3 MiB diff over the limit is archived in full and rolled back", async () => {
  const big = Array.from({ length: 3 * 1024 }, (_, i) => `line ${i} ${"x".repeat(1100)}`).join("\n") + "\n";
  const { run, patches, dirty, commits } = await runScenario({ name: "3 MiB diff limit",
    attempts: [{ files: { "src/big.txt": big }, records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", errorCode: "diff_limit_exceeded", committed: false } });
  assert.equal(run.error?.code, "diff_limit_exceeded");
  assert.equal(commits, 0);
  assert.equal(dirty, "");
  assert.equal(patches.length, 1);
  assert.ok(patches[0].content.length > 3 * 1024 * 1024, "the whole diff is kept below the archive cap");
  assert.doesNotMatch(patches[0].content, /patch truncated/);
  assert.match(patches[0].content, /line 3071 x/);
});

test("a failed rollback keeps the run's original error and cause and flags the rollback", async () => {
  const f = gatedFixture(async () => { throw new HarnessError("docker_unavailable", "Cannot connect to the Docker daemon"); });
  f.calls.rollbackError = new Error("worktree locked");
  const run = await settleWorkflow(f, "rollback-keeps-cause");
  assert.equal(run.error?.code, "quality_gate_unavailable");
  assert.equal(run.error?.cause_code, "docker_unavailable");
  assert.equal(run.error?.rollback_failed, true);
});

test("a diff over the limits on an attempt with repairs left becomes a repair naming the limit", async () => {
  const f = fixture(undefined, { execution: () => {}, assertUnchanged: async () => {}, maxCostUsd: 5, maxRepairs: 1,
    evaluate: async () => ({ action: "pass", candidate: { hash: "h", path: "/s" }, verification: { status: "passed", checks: [] }, grade: { status: "passed", candidate_hash: "h", criteria: [], findings: [] } }) });
  f.calls.inspectError = new DiffLimitError("Changed file limit exceeded (51/50)");
  const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "diff-limit-repair", user_id: "U", prompt: "fix the parser" });
  await prompted(f.process);
  f.process.emit(SETTLED);
  await waitFor(() => f.process.prompts.length, count => count > 1, () => "the diff limit was never sent back as a repair");
  assert.match(f.process.prompts[1]?.prompt ?? "", /51\/50/);
  assert.match(f.process.prompts[1]?.prompt ?? "", /fix the parser/);
  f.calls.inspectError = undefined;
  f.process.emit(SETTLED);
  const run = await settledRun(f.orchestrator, accepted.run_id, f.calls);
  assert.equal(run.status, "completed", JSON.stringify(run.error));
  assert.equal(run.attempts?.length, 2);
});

test("publicRun strips host artifact paths from every attempt", () => {
  const shown = publicRun({ run_id: "r", attempts: [{ attempt: 1, execution_id: "e", artifact: "/var/lib/pi-agent/artifacts/r/candidate.patch" }] } as any);
  assert.ok(!JSON.stringify(shown).includes("/var/lib/pi-agent"), JSON.stringify(shown));
});

test("auto-routing falls back to a configured model when Luna is not configured", async () => {
  const { orchestrator, calls } = fixture({ route: async () => { throw new Error("jev down"); } }, undefined, undefined, { availableModelRefs: ["anthropic/claude-sonnet-4-6"] });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "fallback-configured", user_id: "U", prompt: "fix" });
  await waitFor(() => calls.startedModels.length, count => count > 0, () => "no process started");
  assert.equal(calls.startedModels[0], "anthropic/claude-sonnet-4-6");
});

test("a worktree over its disk quota fails the run as disk_quota_exceeded and rolls back", async () => {
  let usage = 10;
  const { orchestrator, calls } = fixture(undefined, undefined, undefined, { diskQuota: { bytes: 100, intervalMs: 5, usage: async () => usage } });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "disk-quota", user_id: "U", prompt: "fix" });
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(orchestrator.getRun(accepted.run_id).status, "running", "under the quota the run continues");
  usage = 101;
  const run = await waitFor(() => orchestrator.getRun(accepted.run_id), current => current.status === "failed" && calls.rollback.length > 0, current => `status ${current.status}`);
  assert.equal(run.error?.code, "disk_quota_exceeded");
  assert.equal(calls.commits, 0);
});

test("the process's startup summary is recorded on the attempt it started, or on the run without a workflow", async () => {
  const summary = { reused: false, queue_ms: 3, container_ms: 900, rpc_ready_ms: 200, configure_ms: 50, dispatch_to_prompt_ms: 1150 };
  const plain = fixture();
  const accepted = await plain.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "startup-plain", user_id: "U", prompt: "fix" });
  plain.process.emit({ type: "startup", startup: summary });
  assert.deepEqual(plain.orchestrator.getRun(accepted.run_id).startup, summary);

  const gated = gatedFixture(async () => new Promise(() => undefined));
  const run = await gated.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "startup-gated", user_id: "U", prompt: "fix" });
  await prompted(gated.process);
  gated.process.emit({ type: "startup", startup: summary });
  assert.deepEqual(gated.orchestrator.getRun(run.run_id).attempts?.[0]?.startup, summary);
});

test("a session's git mirror is persisted and handed to its restored container; legacy sessions restore without one", async () => {
  const saved: any[] = [];
  const started: Array<{ id: string; gitDir?: string; restored?: boolean }> = [];
  const startProcess = async (session: { id: string; gitDir?: string; restored?: boolean }, onEvent: (event: PublicRunEvent) => void) => {
    started.push({ id: session.id, gitDir: session.gitDir, restored: session.restored });
    const process = new FakeProcess(); process.handler = onEvent; return process;
  };
  const worktrees = {
    create: async () => ({ path: "/managed/w1", branch: "pi-agent/20260801-run", baselineCommit: "a".repeat(40), gitDir: "/managed/mirrors/w1.git" }),
    inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }), commit: async () => "a".repeat(40), rollback: async () => undefined, remove: async () => undefined,
  };
  const first = fixture(undefined, undefined, { load: async () => ({ schema_version: 1, sessions: [], runs: [] }), takeExpiredSessions: () => [], save: async (state: any) => { saved.push(structuredClone(state)); } }, { worktrees, startProcess });
  const created = await first.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "mirror", user_id: "U", prompt: "fix" });
  const persisted = saved.at(-1).sessions.find((session: any) => session.id === created.session_id);
  assert.equal(persisted.gitDir, "/managed/mirrors/w1.git");
  assert.deepEqual(started[0], { id: created.session_id, gitDir: "/managed/mirrors/w1.git", restored: undefined });

  const legacy = { id: "L", key: "T:C:legacy", ownerUserId: "U", modelRef: "openai/gpt-5.6-luna", worktreePath: "/managed/w0", branch: "pi-agent/old", baselineCommit: "c".repeat(40), closed: false, lastActivity: Date.now() };
  started.length = 0;
  const second = fixture(undefined, undefined, { load: async () => ({ schema_version: 1, sessions: [{ ...persisted, activeRunId: undefined }, legacy], runs: [] }), takeExpiredSessions: () => [], save: async () => {} }, { worktrees, startProcess });
  await second.orchestrator.restore();
  assert.deepEqual(started.find((session) => session.id === created.session_id), { id: created.session_id, gitDir: "/managed/mirrors/w1.git", restored: true });
  assert.deepEqual(started.find((session) => session.id === "L"), { id: "L", gitDir: undefined, restored: true });
});

async function failedRunWithPatch(thread: string, archived: string | null = "/runtime/artifacts/R/undelivered.patch") {
  const f = fixture(undefined, undefined, undefined, { archivePatch: async () => archived ?? undefined });
  const accepted = await f.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: thread, user_id: "U", prompt: "add a widget counter" });
  f.process.emit({ type: "settled", outcome: "aborted" });
  await terminalRun(f.orchestrator, accepted.run_id);
  assert.equal(f.orchestrator.getRun(accepted.run_id).status, "failed");
  return { ...f, accepted };
}

test("resume from patch applies the retained patch to the session's clean worktree and starts a new run that says so", async () => {
  const { orchestrator, process, calls, accepted } = await failedRunWithPatch("resume-patch");
  const rollbacks = calls.rollback.length;
  const resumed = await orchestrator.resumeFromPatch(accepted.run_id, "U");
  assert.equal(resumed.session_id, accepted.session_id);
  assert.equal(resumed.status, "starting");
  assert.notEqual(resumed.run_id, accepted.run_id);
  assert.deepEqual(calls.applied, [{ path: "/managed/w1", file: "/runtime/artifacts/R/undelivered.patch" }]);
  assert.equal(calls.rollback.length, rollbacks, "the applied patch is not rolled back before the run");
  const prompt = process.prompts.at(-1)!;
  assert.equal(prompt.runId, resumed.run_id);
  assert.match(prompt.prompt, new RegExp(`previous attempt's patch \\(run ${accepted.run_id}\\)`));
  assert.match(prompt.prompt, /already applied/);
  assert.match(prompt.prompt, /agent_aborted/);
  assert.match(prompt.prompt, /add a widget counter/);
  assert.equal(orchestrator.getRun(resumed.run_id).status, "running");
  assert.equal(orchestrator.getRun(accepted.run_id).status, "failed", "the old run is left as it was");
});

test("resume from patch is owner-only, needs a terminal run with a retained patch and a free session", async () => {
  const { orchestrator, accepted, calls } = await failedRunWithPatch("resume-patch-guards");
  await assert.rejects(orchestrator.resumeFromPatch(accepted.run_id, "someone-else"), (error: any) => error.statusCode === 403 && error.code === "patch_forbidden");
  await assert.rejects(orchestrator.resumeFromPatch("missing", "U"), (error: any) => error.statusCode === 404);
  const resumed = await orchestrator.resumeFromPatch(accepted.run_id, "U");
  await assert.rejects(orchestrator.resumeFromPatch(accepted.run_id, "U"), (error: any) => error.statusCode === 409 && error.code === "session_busy");
  await assert.rejects(orchestrator.resumeFromPatch(resumed.run_id, "U"), (error: any) => error.statusCode === 409, "an active run cannot be resumed from");
  assert.equal(calls.applied.length, 1);

  const none = await failedRunWithPatch("resume-patch-none", null);
  await assert.rejects(none.orchestrator.resumeFromPatch(none.accepted.run_id, "U"), (error: any) => error.statusCode === 404 && error.code === "patch_not_found");
  assert.equal(none.calls.applied.length, 0);
});

test("a patch that does not apply cleanly is refused with 409 and leaves the session free", async () => {
  const { PatchApplyError } = await import("../src/patch-archive.js");
  const { orchestrator, process, calls, accepted } = await failedRunWithPatch("resume-patch-conflict");
  const prompts = process.prompts.length;
  calls.applyError = new PatchApplyError("patch_conflict", "The retained patch does not apply cleanly");
  await assert.rejects(orchestrator.resumeFromPatch(accepted.run_id, "U"), (error: any) => error.statusCode === 409 && error.code === "patch_conflict");
  assert.equal(process.prompts.length, prompts);
  calls.applyError = undefined;
  const next = await orchestrator.prompt(accepted.session_id, "try again", "U");
  assert.equal(next.status, "starting");
});

test("a crashed container's stderr tail is kept on the failed run; an error without one adds nothing", async () => {
  const { orchestrator, process } = fixture();
  const crashed = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "stderr-tail", user_id: "U", prompt: "fix" });
  const stderr_tail = "provider refused [REDACTED]\nError: worker crashed\n";
  process.emit({ type: "error", code: "container_exited", message: "Agent container exited unexpectedly", stderr_tail });
  await terminalRun(orchestrator, crashed.run_id);
  assert.equal(orchestrator.getRun(crashed.run_id).status, "failed");
  assert.deepEqual(orchestrator.getRun(crashed.run_id).error, { code: "container_exited", message: "Agent container exited unexpectedly", stderr_tail });
  assert.equal((publicRun(orchestrator.getRun(crashed.run_id)).error as { stderr_tail?: string }).stderr_tail, stderr_tail);

  const other = fixture();
  const rejected = await other.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "no-stderr-tail", user_id: "U", prompt: "fix" });
  other.process.emit({ type: "error", code: "prompt_rejected", message: "Pi rejected the prompt before it started" });
  await terminalRun(other.orchestrator, rejected.run_id);
  assert.deepEqual(other.orchestrator.getRun(rejected.run_id).error, { code: "prompt_rejected", message: "Pi rejected the prompt before it started" });
});
