import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator, type QualityServices } from "../src/orchestrator.js";
import type { WriterChecksRequest, WriterChecksResponse } from "../src/writer-checks.js";

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  responses: Array<{ id: string; response: { value?: string; cancelled?: true } }> = [];
  aborted: string[] = [];
  prompt() {}
  decide() {}
  respondUi(id: string, response: { value?: string; cancelled?: true }) { this.responses.push({ id, response }); }
  async abort(runId: string) { this.aborted.push(runId); }
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
  answer(id: string) { const found = this.responses.find(item => item.id === id); return found?.response.value ? JSON.parse(found.response.value) as WriterChecksResponse : undefined; }
}

const passed: WriterChecksResponse = { status: "passed", checks: [{ name: "lint", status: "passed" }], report: "Verification: passed" };
const tick = () => new Promise(resolve => setImmediate(resolve));

function fixture(options: { runChecks?: (request: WriterChecksRequest) => Promise<WriterChecksResponse>; maxCalls?: number; reserveMs?: number; quality?: boolean; stop?: () => Promise<void> } = {}) {
  const process = new FakeProcess();
  const rollbacks: string[] = []; const requests: WriterChecksRequest[] = [];
  const quality: QualityServices | undefined = options.quality === false ? undefined : {
    evaluate: async () => { throw new Error("not reached"); }, assertUnchanged: async () => undefined, execution: () => undefined,
    maxCostUsd: 10, activeTimeoutMs: 600_000, stop: options.stop,
    runChecks: async request => { requests.push(request); return (options.runChecks ?? (async () => passed))(request); },
    writerChecks: { maxCalls: options.maxCalls ?? 4, checks: ["lint", "unit"], defaultChecks: ["lint"], estimateMs: names => names.length * 1000, reserveMs: options.reserveMs ?? 1000 },
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "pi-agent/run", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }), commit: async () => "b".repeat(40),
      rollback: async (_path: string, baseline: string) => { rollbacks.push(baseline); }, remove: async () => undefined,
    },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
    deadlineMs: 600_000, quality,
  });
  return { orchestrator, process, rollbacks, requests };
}
const request = (id: string, toolCallId = `tc-${id}`, checks?: string[]): PublicRunEvent => ({ type: "checks_request", request_id: id, tool_call_id: toolCallId, ...(checks ? { checks } : {}) });

test("run_checks answers through respondUi and counts the call on the run", async () => {
  const { orchestrator, process, requests } = fixture();
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "ok", user_id: "U", prompt: "fix" });
  process.emit(request("ui-1", "tc1", ["unit"]));
  await tick(); await tick();
  assert.deepEqual(process.answer("ui-1"), passed);
  assert.deepEqual(requests.map(item => [item.run_id, item.checks]), [[created.run_id, ["unit"]]]);
  assert.equal(orchestrator.getRun(created.run_id).writer_checks_calls, 1);
});

test("a second concurrent call is busy and calls beyond maxCalls are refused", async () => {
  let release!: () => void;
  const { orchestrator, process, requests } = fixture({ maxCalls: 2, runChecks: () => new Promise(resolve => { release = () => resolve(passed); }) });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "limits", user_id: "U", prompt: "fix" });
  process.emit(request("ui-1")); await tick();
  process.emit(request("ui-2")); await tick();
  assert.equal(process.answer("ui-2")?.reason, "busy");
  release(); await tick(); await tick();
  assert.equal(process.answer("ui-1")?.status, "passed");
  process.emit(request("ui-3")); await tick(); release(); await tick(); await tick();
  process.emit(request("ui-4")); await tick();
  assert.equal(process.answer("ui-4")?.reason, "max_calls");
  assert.equal(requests.length, 2);
});

test("run_checks is refused when it would eat the time kept for final verification and grading", async () => {
  const { orchestrator, process, requests } = fixture({ reserveMs: 599_500 });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "time", user_id: "U", prompt: "fix" });
  process.emit(request("ui-1")); await tick();
  assert.equal(process.answer("ui-1")?.reason, "insufficient_time_budget");
  assert.equal(requests.length, 0);
});

test("run_checks is unavailable without a profile and in discuss mode", async () => {
  const bare = fixture({ quality: false });
  await bare.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "bare", user_id: "U", prompt: "fix" });
  bare.process.emit(request("ui-1")); await tick();
  assert.equal(bare.process.answer("ui-1")?.reason, "unavailable");
  const discuss = fixture();
  await discuss.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "discuss", user_id: "U", prompt: "explain", task_mode: "discuss" });
  discuss.process.emit(request("ui-1")); await tick();
  assert.equal(discuss.process.answer("ui-1")?.reason, "unavailable");
  assert.equal(discuss.requests.length, 0);
});

test("cancelling a run stops its checks and rolls back", async () => {
  let stopped = 0; let seen!: AbortSignal;
  const { orchestrator, process, rollbacks } = fixture({
    stop: async () => { stopped++; },
    runChecks: request => new Promise((_resolve, reject) => { seen = request.signal; request.signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true }); }),
  });
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "cancel", user_id: "U", prompt: "fix" });
  process.emit(request("ui-1")); await tick();
  await orchestrator.cancel(created.run_id, "U");
  assert.equal(orchestrator.getRun(created.run_id).status, "cancelled");
  assert.ok(seen.aborted);
  assert.equal(stopped, 1);
  assert.deepEqual(rollbacks, ["a".repeat(40)]);
  assert.equal(process.answer("ui-1"), undefined, "a cancelled run never answers the writer");
});
