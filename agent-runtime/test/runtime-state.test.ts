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
    sessions: [{ id: "S1", key: "T:C:1", ownerUserId: "U1", worktreePath: "/w", branch: "b", baselineCommit: "a", closed: false, lastActivity: Date.now(), activeRunId: "R1" }],
    runs: [{ run_id: "R1", session_id: "S1", status: "running", answer: "", owner_user_id: "U1", created_at: "x", updated_at: "x", tool_count: 1, turn_count: 1, events: [] }],
  });

  const loaded = await store.load();

  assert.equal(loaded.runs[0].status, "interrupted");
  assert.equal(loaded.runs[0].error?.code, "runtime_restarted");
  assert.equal(loaded.sessions[0].activeRunId, undefined);
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
