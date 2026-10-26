import assert from "node:assert/strict";
import { test } from "node:test";

import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import type { RuntimeState } from "../src/runtime-state.js";

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompt() {}
  decide() {}
  respondUi() {}
  async abort() {}
  async close() {}
}

function fixture() {
  const processes = new Map<string, FakeProcess>();
  const saved: RuntimeState[] = [];
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async (runId: string) => ({ path: `/managed/${runId}`, branch: `pi-agent/${runId}`, baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
      commit: async () => "b".repeat(40),
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    startProcess: async (session, onEvent) => {
      const process = new FakeProcess();
      process.handler = onEvent;
      processes.set(session.id, process);
      return process;
    },
    maxActiveSessions: 8,
    stateStore: {
      healthy: true,
      load: async () => ({ schema_version: 1, sessions: [], runs: [] }),
      save: async (state: RuntimeState) => { saved.push(structuredClone(state)); },
      takeExpiredSessions: () => [],
    } as any,
  });
  return { orchestrator, processes, saved };
}
const tick = () => new Promise((resolve) => setImmediate(resolve));

test("a truncated model response becomes a run event and is counted per role", async () => {
  const { orchestrator, saved } = fixture();
  const { run_id } = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trunc-1", user_id: "U", prompt: "fix" });
  await orchestrator.recordTruncation(run_id, "writer", run_id);
  await orchestrator.recordTruncation(run_id, "grader", "exec-grader");
  await orchestrator.recordTruncation(run_id, undefined, "exec-legacy");
  await orchestrator.recordTruncation(run_id, "writer", run_id);
  const run = orchestrator.getRun(run_id);
  assert.deepEqual(run.truncated_requests, { writer: 2, grader: 1, unknown: 1 });
  assert.deepEqual(run.events.filter((event) => event.type === "model_output_truncated"), [
    { type: "model_output_truncated", role: "writer", execution_id: run_id },
    { type: "model_output_truncated", role: "grader", execution_id: "exec-grader" },
    { type: "model_output_truncated", role: "unknown", execution_id: "exec-legacy" },
    { type: "model_output_truncated", role: "writer", execution_id: run_id },
  ]);
  assert.deepEqual(saved.at(-1)?.runs.find((saved) => saved.run_id === run_id)?.truncated_requests, { writer: 2, grader: 1, unknown: 1 }, "the count is persisted");
});

test("a truncation reported for an unknown run is ignored", async () => {
  const { orchestrator } = fixture();
  await orchestrator.recordTruncation("missing", "writer", "missing");
});

test("a truncation reported after the run ended still reaches its harness metrics", async () => {
  const { orchestrator, processes } = fixture();
  const { run_id, session_id } = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "trunc-2", user_id: "U", prompt: "fix" });
  await tick();
  // The writer sees the cut-off turn and settles before the gateway's report arrives.
  processes.get(session_id)!.handler({ type: "settled", outcome: "truncated", stop_reason: "length" } as PublicRunEvent);
  for (let i = 0; i < 20 && orchestrator.getRun(run_id).status !== "failed"; i++) await tick();
  assert.equal(orchestrator.getRun(run_id).status, "failed");
  assert.equal(orchestrator.getRun(run_id).harness_metrics?.truncated_requests, 0);
  await orchestrator.recordTruncation(run_id, "writer", run_id);
  assert.equal(orchestrator.getRun(run_id).harness_metrics?.truncated_requests, 1);
  assert.deepEqual(orchestrator.getRun(run_id).harness_metrics?.truncated_requests_by_role, { writer: 1 });
});
