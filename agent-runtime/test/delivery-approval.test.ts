import assert from "node:assert/strict";
import { test } from "node:test";

import { approvalSettings, DEFAULT_DELIVERY_TTL_MS, MAX_DELIVERY_TTL_MS } from "../src/approval-mode.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";

const SETTLED = { type: "settled", outcome: "completed" } as const;
const tick = () => new Promise(resolve => setImmediate(resolve));

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  aborted: string[] = [];
  prompt() {}
  decide() {}
  respondUi() {}
  async abort(runId: string) { this.aborted.push(runId); }
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function fixture(extra: Record<string, unknown> = {}) {
  const processes: FakeProcess[] = [];
  const calls = { order: [] as string[], commits: 0, patch: "+export const a = 2;\n", started: [] as Array<{ skipDelivery?: boolean }> };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async (id: string) => ({ path: `/managed/${id}`, branch: "b", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: Buffer.byteLength(calls.patch), stat: " src/a.ts | 2 +-" }),
      patch: async () => calls.patch,
      commit: async () => { calls.commits += 1; calls.order.push("commit"); return "c".repeat(40); },
      rollback: async () => { calls.order.push("rollback"); },
      remove: async () => undefined,
    },
    startProcess: async (session: { skipDelivery?: boolean }, onEvent: (event: PublicRunEvent) => void) => {
      calls.started.push(session);
      const process = new FakeProcess();
      process.handler = onEvent;
      processes.push(process);
      return process;
    },
    archivePatch: async () => { calls.order.push("archive"); return "/artifacts/run.patch"; },
    maxActiveSessions: 1,
    deadlineMs: 10_000,
    deliveryApproval: { ttlMs: 10_000, secrets: () => ["sk-live-secret"] },
    ...extra,
  } as any);
  return { orchestrator, processes, calls };
}

const card = (run: { events: Array<Record<string, unknown>> }) => run.events.find(event => event.kind === "delivery") as any;

test("approval settings never turn off both write approval and delivery approval", () => {
  for (const mode of [undefined, "", " ", "delivery", "write", " delivery ", "none", "WRITE", "off"]) {
    for (const ttl of [undefined, "1", "60000", String(MAX_DELIVERY_TTL_MS * 2), "0", "-5", "soon"]) {
      let settings;
      try { settings = approvalSettings({ PI_AGENT_APPROVAL_MODE: mode, PI_AGENT_DELIVERY_TTL_MS: ttl }); } catch { continue; }
      assert.ok(settings.deliveryApproval || settings.writeApproval === "per_prompt", `${mode}/${ttl}`);
      assert.ok(settings.deliveryTtlMs > 0 && settings.deliveryTtlMs <= MAX_DELIVERY_TTL_MS, `${mode}/${ttl}`);
    }
  }
  assert.deepEqual(approvalSettings({}), { mode: "delivery", deliveryApproval: true, writeApproval: "none", deliveryTtlMs: DEFAULT_DELIVERY_TTL_MS });
  assert.equal(approvalSettings({ PI_AGENT_DELIVERY_TTL_MS: String(MAX_DELIVERY_TTL_MS * 2) }).deliveryTtlMs, MAX_DELIVERY_TTL_MS);
  assert.throws(() => approvalSettings({ PI_AGENT_APPROVAL_MODE: "none" }), /must be delivery/);
  assert.throws(() => approvalSettings({ PI_AGENT_DELIVERY_TTL_MS: "0" }), /positive integer/);
});

test("the removed write approval mode fails startup with a pointer to delivery mode", () => {
  for (const mode of ["write", " write ", "WRITE"]) {
    assert.throws(() => approvalSettings({ PI_AGENT_APPROVAL_MODE: mode }), (error: Error) => {
      assert.match(error.message, /write.*removed/i, mode);
      assert.match(error.message, /PI_AGENT_APPROVAL_MODE=delivery/, mode);
      assert.match(error.message, /docs\/agent-verification-ci\.md/, mode);
      return true;
    });
  }
});

test("delivery approval requires a worktree patch source to bind the diff", () => {
  assert.throws(() => new CodingAgentOrchestrator({
    worktrees: { create: async () => ({ path: "", branch: "", baselineCommit: "" }), inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }), commit: async () => "", rollback: async () => undefined, remove: async () => undefined },
    startProcess: async () => new FakeProcess(),
    deliveryApproval: { ttlMs: 1_000 },
  }), /worktrees\.patch/);
});

test("a run waiting for delivery frees its capacity slot and outlives the run deadline", async () => {
  const { orchestrator, processes, calls } = fixture({ deadlineMs: 30 });
  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "1", user_id: "U", prompt: "fix" });
  processes[0]!.emit(SETTLED); await tick(); await tick();
  assert.equal(orchestrator.getRun(first.run_id).status, "awaiting_approval");
  assert.equal(orchestrator.getRun(first.run_id).phase, "awaiting_delivery");
  const second = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "2", user_id: "U", prompt: "other" });
  assert.equal(orchestrator.getRun(second.run_id).status, "running");
  await new Promise(resolve => setTimeout(resolve, 80));
  assert.equal(orchestrator.getRun(first.run_id).status, "awaiting_approval", "the TTL replaces the run deadline");
  assert.equal(calls.commits, 0);
});

test("cancelling a pending delivery archives, rolls back and fails as delivery_rejected", async () => {
  const { orchestrator, processes, calls } = fixture();
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "cancel", user_id: "U", prompt: "fix" });
  processes[0]!.emit(SETTLED); await tick(); await tick();
  await orchestrator.cancel(created.run_id, "U");
  const run = orchestrator.getRun(created.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "delivery_rejected");
  assert.deepEqual(calls.order, ["archive", "rollback"]);
  await assert.rejects(orchestrator.decide(created.run_id, card(run).approval_id, "U", "approve"));
});

test("only the owner can deliver", async () => {
  const { orchestrator, processes, calls } = fixture();
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "owner", user_id: "U", prompt: "fix" });
  processes[0]!.emit(SETTLED); await tick(); await tick();
  const approvalId = card(orchestrator.getRun(created.run_id)).approval_id;
  await assert.rejects(orchestrator.decide(created.run_id, approvalId, "intruder", "approve"));
  await orchestrator.decide(created.run_id, approvalId, "U", "approve");
  assert.equal(orchestrator.getRun(created.run_id).status, "completed");
  assert.equal(calls.commits, 1);
});

test("CI repair sessions skip delivery approval and keep per-write prompts", async () => {
  const { orchestrator, processes, calls } = fixture();
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "ci", user_id: "U", prompt: "fix" }, undefined, undefined, { delivery: "skip" });
  assert.equal(calls.started[0]?.skipDelivery, true);
  processes[0]!.emit(SETTLED); await tick(); await tick();
  assert.equal(orchestrator.getRun(created.run_id).status, "completed");
  assert.equal(calls.commits, 1);
});

test("the delivery card excerpt is redacted and bounded to 1.8 KB", async () => {
  const { orchestrator, processes, calls } = fixture();
  calls.patch = `+OPENAI_API_KEY=sk-live-secret\n+token sk-live-secret\n${"+line with ünïcode\n".repeat(400)}`;
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "excerpt", user_id: "U", prompt: "fix" });
  processes[0]!.emit(SETTLED); await tick(); await tick();
  const delivery = card(orchestrator.getRun(created.run_id)).delivery;
  assert.ok(Buffer.byteLength(delivery.excerpt) <= 1_800, String(Buffer.byteLength(delivery.excerpt)));
  assert.doesNotMatch(delivery.excerpt, /sk-live-secret/);
  assert.match(delivery.excerpt, /truncated/);
  assert.equal(delivery.verification_status, "unverified");
});

test("a restart while waiting for delivery fails the run as delivery_interrupted after archiving", async () => {
  const order: string[] = [];
  const state = {
    schema_version: 1 as const,
    sessions: [{ id: "S", key: "T:C:9", ownerUserId: "U", worktreePath: "/managed/w", branch: "b", baselineCommit: "base", closed: false, lastActivity: Date.now() }],
    // The state store turns any non-terminal run into `interrupted` on load.
    runs: [{ run_id: "R", session_id: "S", status: "interrupted" as const, phase: "awaiting_delivery", error: { code: "runtime_restarted", message: "x" }, answer: "", owner_user_id: "U", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:01Z", tool_count: 1, turn_count: 1, events: [] }],
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "", branch: "", baselineCommit: "" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      patch: async () => "",
      commit: async () => "",
      rollback: async () => { order.push("rollback"); },
      remove: async () => undefined,
    },
    stateStore: { load: async () => state, save: async () => undefined, takeExpiredSessions: () => [] } as any,
    startProcess: async () => { order.push("start"); return new FakeProcess(); },
    archivePatch: async () => { order.push("archive"); return "/artifacts/R.patch"; },
    deliveryApproval: { ttlMs: 1_000 },
  });
  await orchestrator.restore();
  assert.deepEqual(order, ["archive", "rollback", "start"]);
  const run = orchestrator.getRun("R");
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "delivery_interrupted");
});
