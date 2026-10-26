import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import { RuntimeStateStore, type RuntimeState } from "../src/runtime-state.js";

test("runtime metadata is atomically persisted and active runs become interrupted on load", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-"));
  const store = new RuntimeStateStore(path.join(root, "state.json"));
  await store.save({
    schema_version: 1,
    sessions: [{ id: "S1", key: "T:C:1", ownerUserId: "U1", modelRef: "anthropic/claude-sonnet-4-6", worktreePath: "/w", branch: "b", baselineCommit: "a", closed: false, lastActivity: Date.now(), activeRunId: "R1" }],
    runs: [{ run_id: "R1", session_id: "S1", status: "running", answer: "", owner_user_id: "U1", provider: "anthropic", model: "claude-sonnet-4-6", reasoning_effort: "max", created_at: "x", updated_at: "x", tool_count: 1, turn_count: 1, events: [] }],
  });

  const loaded = await store.load();

  assert.equal(loaded.runs[0].status, "interrupted");
  assert.equal(loaded.runs[0].error?.code, "runtime_restarted");
  assert.equal(loaded.sessions[0].activeRunId, undefined);
  assert.equal(loaded.sessions[0].modelRef, "anthropic/claude-sonnet-4-6");
});

test("audit records older than seven days and closed sessions older than 24 hours are pruned", async () => {
  let now = 10 * 24 * 60 * 60 * 1000;
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-prune-"));
  const store = new RuntimeStateStore(path.join(root, "state.json"), () => now);
  await store.save({
    schema_version: 1,
    sessions: [{ id: "old", key: "T:C:1", ownerUserId: "U", worktreePath: "/w", branch: "b", baselineCommit: "a", closed: true, lastActivity: 0 }],
    runs: [{ run_id: "old-run", session_id: "old", status: "completed", answer: "", owner_user_id: "U", created_at: new Date(0).toISOString(), updated_at: new Date(0).toISOString(), tool_count: 0, turn_count: 0, events: [] }],
  });

  const loaded = await store.load();
  assert.deepEqual(loaded.sessions, []);
  assert.deepEqual(loaded.runs, []);
});

test("corrupt control state is not silently treated as a fresh runtime", async () => {
  const { writeFile, rm } = await import("node:fs/promises");
  const root = await mkdtemp(path.join(tmpdir(), "pi-corrupt-"));
  try {
    await writeFile(path.join(root, "state.json"), "{broken");
    await assert.rejects(new RuntimeStateStore(path.join(root, "state.json")).load());
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("a failed write does not poison later saves, and a successful save restores health", async () => {
  const { mkdir, readFile, rm } = await import("node:fs/promises");
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-recover-"));
  try {
    const file = path.join(root, "state.json");
    const store = new RuntimeStateStore(file);
    await mkdir(`${file}.tmp`);
    await assert.rejects(store.save({ schema_version: 1, sessions: [], runs: [] }));
    assert.equal(store.healthy, false);
    await rm(`${file}.tmp`, { recursive: true });
    await store.save({ schema_version: 1, sessions: [], runs: [{ run_id: "after" } as any] });
    assert.equal(store.healthy, true);
    assert.match(await readFile(file, "utf8"), /"after"/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("the control state file is written as compact JSON", async () => {
  const { readFile, rm } = await import("node:fs/promises");
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-compact-"));
  try {
    const file = path.join(root, "state.json");
    const state = {
      schema_version: 1 as const,
      sessions: [],
      runs: [{ run_id: "R1", session_id: "S1", status: "completed" as const, answer: "", owner_user_id: "U", created_at: "x", updated_at: "x", tool_count: 0, turn_count: 0, events: [{ type: "status", status: "completed" }] }],
    };
    await new RuntimeStateStore(file).save(state);
    const text = await readFile(file, "utf8");
    assert.equal(text, JSON.stringify(state));
    assert.doesNotMatch(text, /\n/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("new sessions persist sealRequired and restore with it; a session saved before the marker restores as legacy", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-seal-"));
  try {
    const file = path.join(root, "state.json");
    const started: Array<{ id: string; restored?: boolean; sealRequired?: boolean }> = [];
    const orchestrator = () => new CodingAgentOrchestrator({
      worktrees: {
        create: async () => ({ path: "/w/new", branch: "b", baselineCommit: "a".repeat(40) }),
        inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
        commit: async () => "b".repeat(40),
        rollback: async () => undefined,
        remove: async () => undefined,
      },
      startProcess: async ({ id, restored, sealRequired }) => {
        started.push({ id, ...(restored ? { restored } : {}), ...(sealRequired !== undefined ? { sealRequired } : {}) });
        return { prompt() {}, decide() {}, respondUi() {}, async abort() {}, async close() {} };
      },
      stateStore: new RuntimeStateStore(file),
    });
    const accepted = await orchestrator().createSession({ team_id: "T", channel_id: "C", thread_ts: "sealed", user_id: "U", prompt: "fix" });
    assert.deepEqual(started, [{ id: accepted.session_id, sealRequired: true }], "a new session's container must find its seal");
    const saved = JSON.parse(await readFile(file, "utf8")) as RuntimeState;
    assert.equal(saved.sessions.find((session) => session.id === accepted.session_id)?.sealRequired, true);

    // The same state as a runtime from before the marker wrote it: its session has no sealRequired field.
    const legacy = { id: "legacy", key: "T:C:legacy", ownerUserId: "U", worktreePath: "/w/legacy", branch: "b", baselineCommit: "a".repeat(40), closed: false, lastActivity: Date.now() };
    await writeFile(file, JSON.stringify({ ...saved, sessions: [...saved.sessions, legacy] }));
    started.length = 0;
    await orchestrator().restore();
    assert.deepEqual(started, [{ id: accepted.session_id, restored: true, sealRequired: true }, { id: "legacy", restored: true }]);
    const restored = (JSON.parse(await readFile(file, "utf8")) as RuntimeState).sessions;
    assert.equal(restored.find((session) => session.id === accepted.session_id)?.sealRequired, true);
    assert.equal("sealRequired" in restored.find((session) => session.id === "legacy")!, false, "a legacy session stays legacy");
  } finally { await rm(root, { recursive: true, force: true }); }
});
