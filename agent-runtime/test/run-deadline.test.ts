import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import { waitFor } from "./support/wait.js";

const BASELINE = "a".repeat(40);

function fixture(options: { abort?: () => Promise<void> } = {}) {
  const order: string[] = [];
  const process = {
    handler: (_event: PublicRunEvent) => undefined as void,
    prompts: [] as string[],
    aborted: [] as string[],
    prompt(_runId: string, prompt: string) { this.prompts.push(prompt); },
    decide() {},
    respondUi() {},
    async abort(runId: string) { order.push("stop"); this.aborted.push(runId); await options.abort?.(); },
    async close() {},
  };
  const rollback: string[] = [];
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "pi-agent/20260801-run", baselineCommit: BASELINE }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
      commit: async () => { order.push("commit"); return "b".repeat(40); },
      rollback: async (_path: string, baseline: string) => { order.push("rollback"); rollback.push(baseline); },
      remove: async () => undefined,
    },
    startProcess: async (_session: unknown, onEvent: (event: PublicRunEvent) => void) => { process.handler = onEvent; return process; },
    maxActiveSessions: 2,
    deadlineMs: 60_000,
    archivePatch: async () => { order.push("archive"); return undefined; },
  } as any);
  return { orchestrator, process, order, rollback };
}

const settled = (orchestrator: CodingAgentOrchestrator, runId: string) =>
  waitFor(() => orchestrator.getRun(runId), run => run.status !== "running" && run.status !== "starting", run => `run never left ${run.status}`);

test("a run that outlives its deadline is stopped, then rolled back, and times out", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { orchestrator, process, order, rollback } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "deadline", user_id: "U", prompt: "fix" });
  t.mock.timers.tick(59_999);
  assert.deepEqual(process.aborted, []);
  assert.equal(orchestrator.getRun(accepted.run_id).status, "running");
  t.mock.timers.tick(1);
  t.mock.timers.reset();
  const run = await settled(orchestrator, accepted.run_id);
  await waitFor(() => rollback.length, count => count > 0, () => "the timed-out worktree was never rolled back");
  assert.equal(run.status, "timed_out");
  assert.equal(run.error?.code, "run_timeout");
  assert.deepEqual(process.aborted, [accepted.run_id]);
  assert.deepEqual(rollback, [BASELINE]);
  assert.deepEqual(order, ["stop", "archive", "rollback"]);
});

test("a settle after the deadline fired cannot commit the timed-out run", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { orchestrator, process, order } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "deadline-late", user_id: "U", prompt: "fix" });
  t.mock.timers.tick(60_000);
  t.mock.timers.reset();
  process.handler({ type: "settled", outcome: "completed" });
  const run = await settled(orchestrator, accepted.run_id);
  await waitFor(() => order.includes("rollback"), Boolean, () => `the timed-out worktree was never rolled back; ${order.join(",")}`);
  assert.equal(run.status, "timed_out");
  assert.ok(!order.includes("commit"));
});

test("when the writer cannot be confirmed stopped at the deadline, the worktree is kept instead of rolled back", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { orchestrator, order, rollback } = fixture({ abort: async () => { throw new Error("docker stop failed"); } });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "deadline-stuck", user_id: "U", prompt: "fix" });
  t.mock.timers.tick(60_000);
  t.mock.timers.reset();
  const run = await settled(orchestrator, accepted.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "container_stop_failed");
  assert.deepEqual(rollback, []);
  assert.deepEqual(order, ["stop"]);
});
