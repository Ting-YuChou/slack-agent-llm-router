import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { RuntimeStateStore } from "../src/runtime-state.js";

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
