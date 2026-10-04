import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator, RuntimeConflictError, RuntimeForbiddenError } from "../src/orchestrator.js";

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompts: Array<{ runId: string; prompt: string; route?: { modelRef: string; effort: string } }> = [];
  decisions: Array<{ id: string; approved: boolean }> = [];
  aborted: string[] = [];
  prompt(runId: string, prompt: string, route?: { modelRef: string; effort: string }) { this.prompts.push({ runId, prompt, route }); }
  decide(id: string, approved: boolean) { this.decisions.push({ id, approved }); }
  async abort(runId: string) { this.aborted.push(runId); }
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function fixture(router?: { route(text?: string): Promise<any> }) {
  const process = new FakeProcess();
  const calls = {
    rollback: [] as string[],
    commits: 0,
    commitImpl: async () => "b".repeat(40),
    startedModels: [] as string[],
  };
  const worktrees = {
    create: async () => ({ path: "/managed/w1", branch: "pi-agent/20260801-run", baselineCommit: "a".repeat(40) }),
    inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
    commit: async () => { calls.commits += 1; return calls.commitImpl(); },
    rollback: async (_path: string, baseline: string) => { calls.rollback.push(baseline); },
    remove: async () => undefined,
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
    mcpRepository: "acme/widgets",
  } as any);
  return { orchestrator, process, calls };
}

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
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
  const second = await (orchestrator as any).prompt(first.session_id, "inspect result", "U", "inspect result");
  assert.equal(orchestrator.getRun(second.run_id).model, "gpt-5.6-luna");
  assert.equal(process.prompts[1].route?.effort, "low");
});

test("explicit Sol session ignores Jev and keeps Sol max on follow-up", async () => {
  let calls = 0;
  const { orchestrator, process } = fixture({ route: async () => { calls++; throw new Error("should not classify"); } });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "manual-sol", user_id: "U", prompt: "fix", model: "openai/gpt-5.6-sol", routing_text: "fix" } as any);
  assert.equal(process.prompts[0].route?.effort, "max");
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
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

  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
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
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
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
  assert.equal(JSON.stringify(run.events).includes("arguments"), false);
  assert.deepEqual(run.events.slice(-2).map((event) => event.type), ["mcp", "mcp"]);
  process.emit({ type: "tool", phase: "start", tool_call_id: "m2", tool: "mcp__github__search_code" });
  assert.equal((orchestrator as any).mcpToolStarts.size, 1);
  process.emit({ type: "error", code: "container_exited", message: "stopped" });
  await new Promise((resolve) => setImmediate(resolve));
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
  assert.deepEqual(process.decisions, [{ id: "rpc-ui-1", approved: false }]);
  assert.deepEqual(calls.rollback, ["a".repeat(40)]);
});

test("same thread resumes its session and a busy session rejects overlapping prompts", async () => {
  const { orchestrator, process } = fixture();
  const first = await orchestrator.createSession({ team_id: "T1", channel_id: "C1", thread_ts: "1.0", user_id: "U1", prompt: "first" });
  await assert.rejects(orchestrator.prompt(first.session_id, "second", "U1"), RuntimeConflictError);
  process.emit({ type: "answer", text: "done" });
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
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
  await new Promise((resolve) => setImmediate(resolve));
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
  process.emit({ type: "settled" });
  await assert.rejects(orchestrator.cancel(accepted.run_id, "U"), RuntimeConflictError);
  releaseCommit();
  await new Promise((resolve) => setImmediate(resolve));
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
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));

  const pendingPrompt = orchestrator.prompt(first.session_id, "follow up", "U", "follow up");
  await new Promise((resolve) => setImmediate(resolve));
  await assert.rejects(orchestrator.closeSession(first.session_id, "U"), RuntimeConflictError);
  releaseRoute({ modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev" });
  await pendingPrompt;
});

test("routed follow-up is rejected while session close is pending", async () => {
  const { orchestrator, process } = fixture();
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "prompt-race", user_id: "U", prompt: "first" });
  process.emit({ type: "settled" });
  await new Promise((resolve) => setImmediate(resolve));
  let releaseClose!: () => void;
  const closeGate = new Promise<void>((resolve) => { releaseClose = resolve; });
  process.close = async () => { await closeGate; };

  const pendingClose = orchestrator.closeSession(first.session_id, "U");
  await new Promise((resolve) => setImmediate(resolve));
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
