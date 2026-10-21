import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator, MAX_RUN_EVENTS, RUN_RETENTION_MS, buildCommitMessage, type RunRecord } from "../src/orchestrator.js";
import type { RuntimeState } from "../src/runtime-state.js";

const SETTLED = { type: "settled", outcome: "completed" } as const;
const HOUR = 60 * 60 * 1000;

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  prompt() {}
  decide() {}
  respondUi() {}
  async abort() {}
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function fixture(load?: RuntimeState) {
  const processes = new Map<string, FakeProcess>();
  const saved: RuntimeState[] = [];
  const commitMessages: string[] = [];
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async (runId: string) => ({ path: `/managed/${runId}`, branch: `pi-agent/${runId}`, baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
      commit: async (_path: string, message: string) => { commitMessages.push(message); return "b".repeat(40); },
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
    deadlineMs: 10 * 24 * HOUR,
    stateStore: {
      healthy: true,
      load: async () => load ?? { schema_version: 1, sessions: [], runs: [] },
      save: async (state: RuntimeState) => { saved.push(structuredClone(state)); },
      takeExpiredSessions: () => [],
    } as any,
  });
  const start = (thread: string, prompt = "fix") => orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: thread, user_id: "U", prompt });
  const finish = async (sessionId: string) => {
    processes.get(sessionId)!.emit(SETTLED);
    await new Promise((resolve) => setImmediate(resolve));
  };
  return { orchestrator, processes, saved, commitMessages, start, finish };
}

const savedRunIds = (state: RuntimeState | undefined) => (state?.runs ?? []).map((run) => run.run_id).sort();

test("terminal runs older than 24 hours leave memory and the state file; recent and active runs stay", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: Date.parse("2026-10-01T00:00:00Z") });
  const { orchestrator, saved, start, finish } = fixture();
  const old = await start("old");
  await finish(old.session_id);
  assert.equal(orchestrator.getRun(old.run_id).status, "completed");

  t.mock.timers.tick(2 * HOUR);
  const recent = await start("recent");
  await finish(recent.session_id);
  const active = await start("active");

  t.mock.timers.tick(RUN_RETENTION_MS - 2 * HOUR + 1);
  // Any durable write prunes; a fresh session is the simplest trigger.
  const fresh = await start("fresh");

  assert.throws(() => orchestrator.getRun(old.run_id), /Run not found/);
  assert.throws(() => orchestrator.listEvents(old.run_id), /Run not found/);
  assert.equal(orchestrator.getRun(recent.run_id).status, "completed");
  assert.equal(orchestrator.getRun(active.run_id).status, "running");
  assert.equal(orchestrator.getRun(fresh.run_id).status, "running");
  assert.deepEqual(savedRunIds(saved.at(-1)), [recent.run_id, active.run_id, fresh.run_id].sort());
  assert.equal(RUN_RETENTION_MS, 24 * HOUR);
});

test("restore evicts expired terminal runs but keeps an interrupted run its open session can resume", async () => {
  const now = Date.now();
  const old = new Date(now - RUN_RETENTION_MS - HOUR).toISOString();
  const run = (run_id: string, session_id: string, status: RunRecord["status"], updated_at: string): RunRecord => ({
    run_id, session_id, status, answer: "", owner_user_id: "U", created_at: updated_at, updated_at, tool_count: 0, turn_count: 0, events: [],
  });
  const { orchestrator, saved } = fixture({
    schema_version: 1,
    sessions: [
      { id: "open", key: "T:C:open", ownerUserId: "U", worktreePath: "/managed/open", branch: "b", baselineCommit: "base", closed: false, lastActivity: now },
      { id: "closed", key: "T:C:closed", ownerUserId: "U", worktreePath: "/managed/closed", branch: "b", baselineCommit: "base", closed: true, lastActivity: now },
    ],
    runs: [
      run("old-done", "closed", "completed", old),
      run("old-interrupted", "open", "interrupted", old),
      run("recent-done", "closed", "failed", new Date(now - HOUR).toISOString()),
    ],
  });
  await orchestrator.restore();
  assert.throws(() => orchestrator.getRun("old-done"), /Run not found/);
  assert.equal(orchestrator.getRun("old-interrupted").status, "interrupted");
  assert.equal(orchestrator.getRun("recent-done").status, "failed");
  assert.deepEqual(savedRunIds(saved.at(-1)), ["old-interrupted", "recent-done"]);
});

test("a run keeps at most 2000 events, counts the rest, and absolute event cursors stay correct", async () => {
  const { orchestrator, processes, start } = fixture();
  const accepted = await start("events");
  const process = processes.get(accepted.session_id)!;
  // Event 0 is the starting status; the tool events follow.
  for (let index = 1; index <= 2100; index++) process.emit({ type: "tool", phase: "start", tool: "read", tool_call_id: `c${index}` });

  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(MAX_RUN_EVENTS, 2000);
  assert.equal(run.events.length, 2000);
  assert.equal(run.events_truncated, 101);
  assert.equal(run.tool_count, 2100);
  // The newest events are kept: the last one is the last emitted.
  assert.equal(run.events.at(-1)?.tool_call_id, "c2100");
  assert.equal(run.events[0]?.tool_call_id, "c101");

  // `after` is an absolute position in the full event history.
  const tail = orchestrator.listEvents(accepted.run_id, 2050);
  assert.equal(tail.length, 51);
  assert.equal(tail[0]?.tool_call_id, "c2050");
  // A cursor behind the cap resumes at the oldest event still kept.
  const behind = orchestrator.listEvents(accepted.run_id, 0);
  assert.equal(behind.length, 2000);
  assert.equal(behind[0]?.tool_call_id, "c101");
  assert.deepEqual(orchestrator.listEvents(accepted.run_id, 2101), []);

  process.emit({ type: "tool", phase: "end", tool: "read", tool_call_id: "c2100" });
  const next = orchestrator.listEvents(accepted.run_id, 2101);
  assert.equal(next.length, 1);
  assert.equal(next[0]?.phase, "end");
  assert.equal(orchestrator.getRun(accepted.run_id).events_truncated, 102);
});

test("waitForEvents returns as soon as a new event arrives", async () => {
  const { orchestrator, processes, start } = fixture();
  const accepted = await start("wait");
  const cursor = orchestrator.getRun(accepted.run_id).events.length;
  const started = Date.now();
  const waiting = orchestrator.waitForEvents(accepted.run_id, cursor, 20_000);
  setTimeout(() => processes.get(accepted.session_id)!.emit({ type: "tool", phase: "start", tool: "read", tool_call_id: "c1" }), 20);
  const result = await waiting;
  assert.ok(Date.now() - started < 5_000);
  assert.equal(result.offset, cursor);
  assert.equal(result.events.length, 1);
  assert.equal(result.events[0]?.tool_call_id, "c1");
});

test("waitForEvents returns existing events at once and an empty list on timeout", async () => {
  const { orchestrator, start } = fixture();
  const accepted = await start("timeout");
  const ready = await orchestrator.waitForEvents(accepted.run_id, 0, 20_000);
  assert.equal(ready.events.length, orchestrator.getRun(accepted.run_id).events.length);
  const cursor = ready.events.length;
  const started = Date.now();
  const empty = await orchestrator.waitForEvents(accepted.run_id, cursor, 50);
  assert.ok(Date.now() - started >= 40);
  assert.deepEqual(empty, { events: [], offset: cursor });
  // An aborted wait (client disconnect) resolves empty instead of hanging.
  const controller = new AbortController();
  const aborted = orchestrator.waitForEvents(accepted.run_id, cursor, 20_000, controller.signal);
  controller.abort();
  assert.deepEqual(await aborted, { events: [], offset: cursor });
  await assert.rejects(orchestrator.waitForEvents("missing", 0, 10), /Run not found/);
});

test("the commit subject is a bounded, redacted, single-line task summary followed by the trailers", async () => {
  const { start, finish, commitMessages } = fixture();
  const prompt = "Fix the flaky login test\nAPI_KEY=sk-live-123456 should never leak\n\n" + "and then refactor the session module ".repeat(5);
  const accepted = await start("commit", prompt);
  await finish(accepted.session_id);
  const message = commitMessages[0]!;
  const [subject, blank, ...trailers] = message.split("\n");
  assert.match(subject, /^Pi agent: Fix the flaky login test API_KEY=\[REDACTED\]/);
  assert.ok(subject.length <= 72, `subject is ${subject.length} chars`);
  assert.ok(subject.endsWith("…"));
  assert.doesNotMatch(message, /sk-live/);
  assert.equal(blank, "");
  assert.deepEqual(trailers, [`Pi-Run-Id: ${accepted.run_id}`, "Pi-Model: openai/gpt-5.6-luna", "Pi-Verification: unverified"]);
});

test("a run without a task summary keeps the run id as its subject", () => {
  const message = buildCommitMessage({ run_id: "R1", provider: "openai", model: "m", verification_status: "passed" } as RunRecord);
  assert.equal(message, "Pi agent: R1\n\nPi-Run-Id: R1\nPi-Model: openai/m\nPi-Verification: passed");
});

test("a short summary is kept whole", () => {
  const short = buildCommitMessage({ run_id: "R1", task_summary: "Add a retry", provider: "openai", model: "m" } as RunRecord);
  assert.equal(short.split("\n")[0], "Pi agent: Add a retry");
});
