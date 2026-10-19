import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";

import { AgentExecutionRunner, type InvestigationProcessFactory } from "../src/agent-execution-runner.js";
import { HarnessError } from "../src/error-taxonomy.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import type { InvestigationRequest, WorkflowNode } from "../src/structured-workflow.js";

const stateRoot = mkdtempSync(path.join(os.tmpdir(), "agent-execution-runner-"));
after(async () => { const { rm } = await import("node:fs/promises"); await rm(stateRoot, { recursive: true, force: true }); });

type Step = { record?: Record<string, unknown>; event?: PublicRunEvent };
const submit = (args: unknown): Step[] => [
  { record: { type: "tool_execution_start", toolName: "submit_investigation", toolCallId: "t1", args } },
  { record: { type: "tool_execution_end", toolName: "submit_investigation", toolCallId: "t1", isError: false } },
];
const settled: Step = { event: { type: "settled", outcome: "completed" } as PublicRunEvent };
const usage: Step = { event: { type: "usage", cost_usd: 0.5 } as PublicRunEvent };
const valid = { status: "succeeded", findings: [{ path: "src/a.ts", line: 3, reason: "returns early" }], evidence: ["src/a.ts:3"], unresolved: [], suggested_tasks: [] };

function setup(steps: Step[]) {
  const calls = { closed: 0, ends: [] as Array<{ cost?: number; queueMs?: number }>, costs: [] as number[] };
  const createProcess: InvestigationProcessFactory = (_options, onEvent, onRecord) => ({
    prompt: () => { setImmediate(() => { for (const step of steps) { if (step.record) onRecord(step.record); if (step.event) onEvent(step.event); } }); },
    close: async () => { calls.closed += 1; },
    queueMs: 7,
  });
  const runner = new AgentExecutionRunner({ image: "img", network: "net", gatewayUrl: "http://gw", stateRoot, token: () => "t", createProcess,
    execution: async (_id, _parent, _role, _attempt, phase, cost, queueMs) => { if (phase === "end") calls.ends.push({ cost, queueMs }); },
    cost: async (_parent, cost) => { calls.costs.push(cost); } });
  const node = { id: "explore", role: "code_explorer", purpose: "find it", expected_output: "findings", paths: ["src"], dependencies: [], workflow: "run-1", wave: 1, state: "running", attempt: 1, execution: "exec-0001", queued_at: 0 } as WorkflowNode;
  const request: InvestigationRequest = { input: { run_id: "run-1", owner: "o", requirement: "fix", baseline: "base", snapshot: { hash: "h", path: stateRoot }, fingerprint: "f" }, node, results: [], signal: new AbortController().signal };
  return { runner, request, calls };
}

test("an injected process's valid investigation is returned and its costs recorded", async () => {
  const { runner, request, calls } = setup([usage, ...submit(valid), settled]);
  assert.deepEqual(await runner.run(request), valid);
  assert.equal(calls.closed, 1);
  assert.deepEqual(calls.ends, [{ cost: 0.5, queueMs: 7 }]);
  assert.deepEqual(calls.costs, [0.5]);
  assert.equal(request.node.attempt_queue_ms, 7);
});

test("an invalid investigation keeps its validation error", async () => {
  const { runner, request, calls } = setup([usage, ...submit({ ...valid, findings: [{ path: "docs/x.md", line: 1, reason: "outside" }] }), settled]);
  await assert.rejects(runner.run(request), (error: unknown) => error instanceof HarnessError && error.causeCode === "investigation_invalid_submission"
    && /^investigation_invalid_submission: Invalid investigation evidence/.test(error.message));
  assert.equal(calls.closed, 1);
  assert.deepEqual(calls.costs, [0.5]);
});

test("settling without a submission reports the missing structured result", async () => {
  const { runner, request } = setup([settled]);
  await assert.rejects(runner.run(request), (error: unknown) => error instanceof HarnessError && error.causeCode === "investigation_missing_structured_result");
});
