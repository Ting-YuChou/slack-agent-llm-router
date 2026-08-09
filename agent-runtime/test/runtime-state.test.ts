import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { RuntimeStateStore } from "../src/runtime-state.js";

test("runtime metadata is atomically persisted and active runs become interrupted on load", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-"));
  const store = new RuntimeStateStore(path.join(root, "state.json"));
  await store.save({
    schema_version: 1,
    sessions: [{ id: "S1", key: "T:C:1", ownerUserId: "U1", modelRef: "anthropic/claude-sonnet-4-6", worktreePath: "/w", branch: "b", baselineCommit: "a", closed: false, lastActivity: Date.now(), activeRunId: "R1", piSessionId: "PI1", piSessionFile: "session.jsonl", piLeafId: "leaf-0", parentSessionId: "parent", forkSourceRunId: "source-run", activeCheckpointRunId: "R0" }],
    runs: [{ run_id: "R1", session_id: "S1", status: "running", answer: "", owner_user_id: "U1", provider: "anthropic", model: "claude-sonnet-4-6", reasoning_effort: "max", created_at: "x", updated_at: "x", tool_count: 1, turn_count: 1, events: [], kind: "prompt", baseline_commit: "a", pi_parent_leaf_id: "leaf-0", parent_run_id: "R0" }],
  });

  const loaded = await store.load();

  assert.equal(loaded.runs[0].status, "interrupted");
  assert.equal(loaded.runs[0].error?.code, "runtime_restarted");
  assert.equal(loaded.sessions[0].activeRunId, undefined);
  assert.equal(loaded.sessions[0].modelRef, "anthropic/claude-sonnet-4-6");
  assert.equal(loaded.sessions[0].piSessionFile, "session.jsonl");
  assert.equal(loaded.sessions[0].parentSessionId, "parent");
  assert.equal(loaded.sessions[0].activeCheckpointRunId, "R0");
  assert.equal(loaded.runs[0].baseline_commit, "a");
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

test("a transient save failure does not poison later tombstone persistence", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-state-retry-"));
  const blockedParent = path.join(root, "blocked");
  await writeFile(blockedParent, "not a directory");
  const store = new RuntimeStateStore(path.join(blockedParent, "state.json"));
  const state = { schema_version: 1 as const, sessions: [], runs: [] };

  await assert.rejects(store.save(state));
  await rm(blockedParent);
  await mkdir(blockedParent);
  await store.save(state);

  assert.deepEqual(await store.load(), state);
});
