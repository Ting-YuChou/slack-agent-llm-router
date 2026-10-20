import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { ApprovalLedger } from "../src/approval-ledger.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { RuntimeStateStore } from "../src/runtime-state.js";

const SETTLED = { type: "settled", outcome: "completed" } as const;
const tick = () => new Promise(resolve => setImmediate(resolve));

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompt() {}
  decide() {}
  respondUi() {}
  async abort() {}
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

/** A runtime state root shared by every orchestrator a test starts, so a second one is a restart. */
async function stateRoot() {
  const root = await mkdtemp(path.join(tmpdir(), "pi-approval-ledger-"));
  const calls = { order: [] as string[], commits: 0, patch: "+export const a = 2;\n" };
  const ledgers: ApprovalLedger[] = [];
  const start = (ttlMs = 10_000) => {
    const processes: FakeProcess[] = [];
    const ledger = new ApprovalLedger(path.join(root, "approvals", "approvals.sqlite"));
    ledgers.push(ledger);
    const orchestrator = new CodingAgentOrchestrator({
      worktrees: {
        create: async (id: string) => ({ path: `/managed/${id}`, branch: "b", baselineCommit: "a".repeat(40) }),
        inspectDiff: async () => ({ files: ["src/a.ts"], bytes: Buffer.byteLength(calls.patch), stat: " src/a.ts | 2 +-" }),
        patch: async () => calls.patch,
        commit: async () => { calls.commits += 1; calls.order.push("commit"); return "c".repeat(40); },
        rollback: async () => { calls.order.push("rollback"); },
        remove: async () => undefined,
      },
      stateStore: new RuntimeStateStore(path.join(root, "runtime-state.json")),
      startProcess: async (_session: unknown, onEvent: (event: PublicRunEvent) => void) => {
        const process = new FakeProcess();
        process.handler = onEvent;
        processes.push(process);
        return process;
      },
      archivePatch: async () => { calls.order.push("archive"); return "/artifacts/run.patch"; },
      deadlineMs: 10_000,
      deliveryApproval: { ttlMs },
      approvalLedger: ledger,
    });
    return { orchestrator, processes, ledger };
  };
  const cleanup = async () => { for (const ledger of ledgers) ledger.close(); await rm(root, { recursive: true, force: true }); };
  return { root, calls, start, cleanup };
}

const card = (run: { events: Array<Record<string, unknown>> }) => run.events.find(event => event.kind === "delivery") as any;

/** Runs one prompt to its delivery card, then shuts the runtime down as a deploy would. */
async function pendingDelivery(env: Awaited<ReturnType<typeof stateRoot>>, ttlMs?: number) {
  const first = env.start(ttlMs);
  const created = await first.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "1", user_id: "U", prompt: "fix" });
  first.processes[0]!.emit(SETTLED); await tick(); await tick();
  const delivery = card(first.orchestrator.getRun(created.run_id));
  assert.equal(first.orchestrator.getRun(created.run_id).status, "awaiting_approval");
  await first.orchestrator.shutdown();
  // Drain the first runtime's queued state writes; two live stores never share a file in production.
  await (first.orchestrator as any).persistState();
  first.ledger.close();
  return { runId: created.run_id, approvalId: delivery.approval_id as string, digest: delivery.delivery.delivery_digest as string };
}

test("a pending delivery survives a restart and approving it commits the persisted patch hash", async () => {
  const env = await stateRoot();
  try {
    const pending = await pendingDelivery(env);
    const second = env.start();
    assert.equal(second.ledger.pendingDelivery(pending.runId)?.patchHash, pending.digest, "the ledger binds the digest shown on the card");
    await second.orchestrator.restore();
    const run = second.orchestrator.getRun(pending.runId);
    assert.equal(run.status, "awaiting_approval");
    assert.equal(run.phase, "awaiting_delivery");
    assert.equal(run.error, undefined);
    assert.deepEqual(env.calls.order, [], "nothing is archived or rolled back");
    await assert.rejects(second.orchestrator.decide(pending.runId, pending.approvalId, "intruder", "approve"));
    await second.orchestrator.decide(pending.runId, pending.approvalId, "U", "approve");
    assert.equal(second.orchestrator.getRun(pending.runId).status, "completed");
    assert.equal(env.calls.commits, 1);
    assert.equal(second.ledger.pendingDelivery(pending.runId), undefined, "a decided approval is used");
    await assert.rejects(second.orchestrator.decide(pending.runId, pending.approvalId, "U", "approve"));
    assert.equal(env.calls.commits, 1);
  } finally { await env.cleanup(); }
});

test("a worktree changed after the restart fails the restored delivery as delivery_candidate_changed", async () => {
  const env = await stateRoot();
  try {
    const pending = await pendingDelivery(env);
    env.calls.patch = "+export const a = 3;\n";
    const second = env.start();
    await second.orchestrator.restore();
    assert.equal(second.orchestrator.getRun(pending.runId).status, "awaiting_approval");
    await second.orchestrator.decide(pending.runId, pending.approvalId, "U", "approve");
    const run = second.orchestrator.getRun(pending.runId);
    assert.equal(run.status, "failed");
    assert.equal(run.error?.code, "delivery_candidate_changed");
    assert.equal(env.calls.commits, 0);
    assert.deepEqual(env.calls.order, ["archive", "rollback"]);
  } finally { await env.cleanup(); }
});

test("a delivery whose approval expired during the restart fails as delivery_expired", async () => {
  const env = await stateRoot();
  try {
    const pending = await pendingDelivery(env, 40);
    await new Promise(resolve => setTimeout(resolve, 60));
    const second = env.start(40);
    await second.orchestrator.restore();
    await tick(); await tick();
    const run = second.orchestrator.getRun(pending.runId);
    assert.equal(run.status, "failed");
    assert.equal(run.error?.code, "delivery_expired");
    assert.equal(run.approvals?.expired, 1);
    assert.deepEqual(env.calls.order, ["archive", "rollback"]);
    await assert.rejects(second.orchestrator.decide(pending.runId, pending.approvalId, "U", "approve"));
    assert.equal(env.calls.commits, 0);
  } finally { await env.cleanup(); }
});

test("a restored delivery re-arms its remaining expiry", async () => {
  const env = await stateRoot();
  try {
    const pending = await pendingDelivery(env, 150);
    const second = env.start(10_000);
    await second.orchestrator.restore();
    assert.equal(second.orchestrator.getRun(pending.runId).status, "awaiting_approval");
    await new Promise(resolve => setTimeout(resolve, 250));
    const run = second.orchestrator.getRun(pending.runId);
    assert.equal(run.status, "failed", "the persisted expiry, not the new TTL, applies");
    assert.equal(run.error?.code, "delivery_expired");
  } finally { await env.cleanup(); }
});

test("a pending delivery with no ledger record still fails as delivery_interrupted", async () => {
  const env = await stateRoot();
  try {
    const pending = await pendingDelivery(env);
    await rm(path.join(env.root, "approvals"), { recursive: true, force: true });
    const second = env.start();
    await second.orchestrator.restore();
    const run = second.orchestrator.getRun(pending.runId);
    assert.equal(run.status, "failed");
    assert.equal(run.error?.code, "delivery_interrupted");
    assert.deepEqual(env.calls.order, ["archive", "rollback"]);
    await assert.rejects(second.orchestrator.decide(pending.runId, pending.approvalId, "U", "approve"));
    assert.equal(env.calls.commits, 0);
  } finally { await env.cleanup(); }
});

test("a pending tool approval still becomes interrupted after a restart", async () => {
  const env = await stateRoot();
  try {
    const first = env.start();
    const created = await first.orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "tool", user_id: "U", prompt: "fix" });
    first.processes[0]!.emit({ type: "approval", approval_id: "rpc-1", title: "Allow shell command?", detail: "make" } as PublicRunEvent);
    await tick(); await tick();
    assert.equal(first.orchestrator.getRun(created.run_id).status, "awaiting_approval");
    first.ledger.close();
    const store = new RuntimeStateStore(path.join(env.root, "runtime-state.json"));
    // Save the awaiting state as a crash would leave it, without the shutdown's cancel.
    await (first.orchestrator as any).persistState();
    const loaded = await store.load();
    assert.equal(loaded.runs.find(run => run.run_id === created.run_id)?.status, "interrupted");
    const second = env.start();
    await second.orchestrator.restore();
    const run = second.orchestrator.getRun(created.run_id);
    assert.equal(run.status, "interrupted");
    assert.equal(run.error?.code, "runtime_restarted");
  } finally { await env.cleanup(); }
});

test("the ledger prunes used and long-expired approvals", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-approval-prune-"));
  let now = 1_000_000;
  const ledger = new ApprovalLedger(path.join(root, "approvals.sqlite"), () => now);
  try {
    const base = { toolCallId: "delivery", userId: "U", action: "deliver", used: false };
    ledger.record({ ...base, id: "used", runId: "R1", expiresAt: now + 60_000, patchHash: "sha256:1" });
    ledger.record({ ...base, id: "stale", runId: "R2", expiresAt: now - 1, patchHash: "sha256:2" });
    ledger.record({ ...base, id: "live", runId: "R3", expiresAt: now + 60_000, patchHash: "sha256:3" });
    ledger.markUsed("used");
    assert.equal(ledger.pendingDelivery("R1"), undefined);
    assert.equal(ledger.pendingDelivery("R2")?.id, "stale", "an expired approval is kept until it is pruned so a restart can report it expired");
    now += 25 * 60 * 60_000;
    ledger.record({ ...base, id: "fresh", runId: "R4", expiresAt: now + 60_000, patchHash: "sha256:4" });
    assert.equal(ledger.prune(), 3);
    assert.deepEqual(["R1", "R2", "R3", "R4"].map(run => ledger.pendingDelivery(run)?.id), [undefined, undefined, undefined, "fresh"]);
  } finally { ledger.close(); await rm(root, { recursive: true, force: true }); }
});

test("a restored structured-workflow delivery completes its execution phase on approval", async () => {
  const env = await stateRoot();
  try {
    const now = Date.now();
    const digest = "sha256:" + (await import("node:crypto")).createHash("sha256").update(JSON.stringify(["src/a.ts"])).update("\0").update(env.calls.patch).digest("hex");
    const { writeFile } = await import("node:fs/promises");
    await writeFile(path.join(env.root, "runtime-state.json"), JSON.stringify({
      schema_version: 1,
      sessions: [{ id: "S", key: "T:C:wf", ownerUserId: "U", worktreePath: "/managed/S", branch: "b", baselineCommit: "a".repeat(40), closed: false, lastActivity: now }],
      runs: [{ run_id: "R", session_id: "S", status: "awaiting_approval", phase: "awaiting_delivery", workflow_mode: "structured", execution_status: "running", verification_status: "passed", grader_status: "passed",
        answer: "", owner_user_id: "U", created_at: new Date(now).toISOString(), updated_at: new Date(now).toISOString(), tool_count: 1, turn_count: 1,
        events: [{ type: "approval", kind: "delivery", approval_id: "A", delivery: { delivery_digest: digest } }] }],
    }));
    const seed = new ApprovalLedger(path.join(env.root, "approvals", "approvals.sqlite"));
    seed.record({ id: "A", runId: "R", toolCallId: "delivery", userId: "U", action: "deliver", expiresAt: now + 60_000, used: false, patchHash: digest });
    seed.close();
    const restarted = env.start();
    await restarted.orchestrator.restore();
    assert.equal(restarted.orchestrator.getRun("R").status, "awaiting_approval");
    await restarted.orchestrator.decide("R", "A", "U", "approve");
    const run = restarted.orchestrator.getRun("R");
    assert.equal(run.status, "completed");
    assert.equal(run.execution_status, "succeeded");
    assert.equal(run.phase, "completed");
    assert.equal(env.calls.commits, 1);
  } finally { await env.cleanup(); }
});
