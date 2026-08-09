import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import { PiSessionStore } from "../src/pi-session-store.js";

function userMessage(text: string) {
  return { role: "user", content: text, timestamp: Date.now() } as const;
}

function assistantMessage(text: string) {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    provider: "openai",
    model: "gpt-5.6-luna",
    usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: "stop",
    timestamp: Date.now(),
  } as const;
}

async function sessionFixture() {
  const root = await mkdtemp(path.join(tmpdir(), "pi-session-store-"));
  const runtimeSessionId = "runtime-parent";
  const piSessionId = "11111111-1111-4111-8111-111111111111";
  const sessionDir = path.join(root, runtimeSessionId);
  const cwd = path.join(root, "parent-worktree");
  await mkdir(sessionDir, { recursive: true });
  await mkdir(cwd, { recursive: true });
  const manager = SessionManager.create(cwd, sessionDir, { id: piSessionId });
  const firstUser = manager.appendMessage(userMessage("first task"));
  const firstAssistant = manager.appendMessage(assistantMessage("first done") as never);
  const secondUser = manager.appendMessage(userMessage("second task"));
  const secondAssistant = manager.appendMessage(assistantMessage("second done") as never);
  const file = manager.getSessionFile();
  assert.ok(file);
  return {
    root,
    runtimeSessionId,
    piSessionId,
    cwd,
    basename: path.basename(file),
    firstUser,
    firstAssistant,
    secondUser,
    secondAssistant,
  };
}

test("rollback creates a deterministic Pi branch without deleting abandoned history", async () => {
  const fixture = await sessionFixture();
  const store = new PiSessionStore(fixture.root);

  const result = await store.rollback({
    runtimeSessionId: fixture.runtimeSessionId,
    sessionFile: fixture.basename,
    parentLeafId: fixture.firstAssistant,
    runId: "failed-run",
  });
  const snapshot = await store.read(fixture.runtimeSessionId, fixture.basename);

  assert.equal(snapshot.leafId, result.leafId);
  assert.ok(snapshot.entries.some((entry) => entry.id === fixture.secondAssistant));
  const rollbackEntry = snapshot.entries.find((entry) => entry.id === result.leafId);
  assert.equal(rollbackEntry?.type, "branch_summary");
  assert.equal(rollbackEntry?.parentId, fixture.firstAssistant);
  assert.doesNotMatch(JSON.stringify(rollbackEntry), /second task|second done/);
});

test("fork extracts the selected Pi branch into an exact child session and rewrites cwd", async () => {
  const fixture = await sessionFixture();
  const store = new PiSessionStore(fixture.root);
  const parentPath = path.join(fixture.root, fixture.runtimeSessionId, fixture.basename);
  const parentBefore = await readFile(parentPath, "utf8");
  const childCwd = path.join(fixture.root, "child-worktree");
  await mkdir(childCwd);

  const child = await store.fork({
    sourceRuntimeSessionId: fixture.runtimeSessionId,
    sourceSessionFile: fixture.basename,
    sourceLeafId: fixture.firstAssistant,
    childRuntimeSessionId: "runtime-child",
    childPiSessionId: "22222222-2222-4222-8222-222222222222",
    childCwd,
  });
  const childSnapshot = await store.read("runtime-child", child.sessionFile);

  assert.equal(await readFile(parentPath, "utf8"), parentBefore);
  assert.equal(childSnapshot.piSessionId, "22222222-2222-4222-8222-222222222222");
  assert.equal(childSnapshot.cwd, childCwd);
  assert.equal(childSnapshot.leafId, fixture.firstAssistant);
  assert.deepEqual(childSnapshot.entries.map((entry) => entry.id), [fixture.firstUser, fixture.firstAssistant]);
});

test("legacy discovery migrates exactly one JSONL file and fails closed on ambiguity", async () => {
  const fixture = await sessionFixture();
  const store = new PiSessionStore(fixture.root);

  const discovered = await store.discover(fixture.runtimeSessionId);
  assert.equal(discovered.sessionFile, fixture.basename);
  assert.equal(discovered.piSessionId, fixture.piSessionId);

  const manager = SessionManager.create(fixture.cwd, path.join(fixture.root, fixture.runtimeSessionId), {
    id: "33333333-3333-4333-8333-333333333333",
  });
  manager.appendMessage(userMessage("ambiguous"));
  manager.appendMessage(assistantMessage("ambiguous") as never);
  await assert.rejects(store.discover(fixture.runtimeSessionId), /ambiguous/i);
});

test("real SessionManager compaction entries remain valid fork checkpoints", async () => {
  const fixture = await sessionFixture();
  const sessionPath = path.join(fixture.root, fixture.runtimeSessionId, fixture.basename);
  const manager = SessionManager.open(sessionPath, path.dirname(sessionPath));
  const compactionLeaf = manager.appendCompaction(
    "Earlier turns completed the first and second tasks.",
    fixture.secondUser,
    12_000,
  );
  const store = new PiSessionStore(fixture.root);
  const childCwd = path.join(fixture.root, "compaction-child");
  await mkdir(childCwd);

  const child = await store.fork({
    sourceRuntimeSessionId: fixture.runtimeSessionId,
    sourceSessionFile: fixture.basename,
    sourceLeafId: compactionLeaf,
    childRuntimeSessionId: "runtime-compaction-child",
    childPiSessionId: "55555555-5555-4555-8555-555555555555",
    childCwd,
  });
  const snapshot = await store.read("runtime-compaction-child", child.sessionFile);

  assert.equal(snapshot.leafId, compactionLeaf);
  assert.equal(snapshot.entries.at(-1)?.type, "compaction");
});

test("recursive cleanup rejects dot-segment session ids without touching the managed root", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-session-cleanup-"));
  const sentinel = path.join(root, "keep.txt");
  await writeFile(sentinel, "keep", "utf8");
  const store = new PiSessionStore(root);

  await assert.rejects(store.remove("."), /below its managed root/i);
  await assert.rejects(store.remove(".."), /below its managed root/i);

  assert.equal(await readFile(sentinel, "utf8"), "keep");
});
