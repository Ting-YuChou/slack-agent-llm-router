import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent, RpcSessionState, RpcSessionStats, RpcTreeSnapshot } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";

class TreeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  checkpoint = {
    piSessionId: "11111111-1111-4111-8111-111111111111",
    sessionFile: "session.jsonl",
    leafId: "leaf-1",
    userEntryId: "user-1",
  };
  compactCalls: Array<{ runId: string; instructions?: string }> = [];
  closeCalls = 0;
  abortReturnsCheckpoint = true;
  treeSnapshot?: RpcTreeSnapshot;
  prompt() {}
  decide() {}
  async abort() { return this.abortReturnsCheckpoint ? this.checkpoint : undefined; }
  async close() { this.closeCalls += 1; }
  async getCheckpoint() { return this.checkpoint; }
  async getState(): Promise<RpcSessionState> {
    return {
      sessionId: this.checkpoint.piSessionId,
      sessionFile: `/var/lib/pi-session/${this.checkpoint.sessionFile}`,
      autoCompactionEnabled: true,
      messageCount: 2,
      pendingMessageCount: 0,
      isStreaming: false,
      isCompacting: false,
    };
  }
  async getSessionStats(): Promise<RpcSessionStats> {
    return {
      sessionId: this.checkpoint.piSessionId,
      userMessages: 1,
      assistantMessages: 1,
      toolCalls: 2,
      toolResults: 2,
      totalMessages: 6,
      tokens: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, total: 15 },
      cost: 0.01,
      contextUsage: { tokens: 15, contextWindow: 100, percent: 15 },
    };
  }
  async getTree(): Promise<RpcTreeSnapshot> {
    return this.treeSnapshot ?? {
      tree: [
        {
          entry: { type: "message", id: this.checkpoint.userEntryId, parentId: null },
          children: [
            {
              entry: { type: "message", id: this.checkpoint.leafId, parentId: this.checkpoint.userEntryId },
              children: [],
            },
          ],
        },
      ],
      leafId: this.checkpoint.leafId,
    };
  }
  async compact(runId: string, instructions?: string) {
    this.compactCalls.push({ runId, instructions });
    this.handler({ type: "compaction", phase: "start", reason: "manual" });
    this.handler({ type: "compaction", phase: "end", reason: "manual", aborted: false, will_retry: false });
    return this.checkpoint;
  }
  emit(event: PublicRunEvent) { this.handler(event); }
}

function fixture(options?: {
  rollbackPiFails?: boolean;
  inspectFails?: boolean;
  maxActiveSessions?: number;
  maxRetainedSessionsPerOwner?: number;
}) {
  const process = new TreeProcess();
  const calls = {
    creates: [] as Array<{ id: string; commit?: string }>,
    rollbacks: [] as string[],
    piRollbacks: [] as Array<Record<string, unknown>>,
    piForks: [] as Array<Record<string, unknown>>,
    inspectDiff: 0,
    commits: 0,
  };
  const worktrees = {
    create: async (id: string, commit?: string) => {
      calls.creates.push({ id, commit });
      return {
        path: calls.creates.length === 1 ? "/managed/parent" : "/managed/child",
        branch: calls.creates.length === 1 ? "pi-agent/parent" : "pi-agent/child",
        baselineCommit: commit ?? "a".repeat(40),
      };
    },
    inspectDiff: async () => {
      calls.inspectDiff += 1;
      if (options?.inspectFails) throw new Error("diff limit");
      return { files: ["src/a.ts"], bytes: 10, stat: "1 file changed" };
    },
    commit: async () => { calls.commits += 1; return "b".repeat(40); },
    rollback: async (_path: string, baseline: string) => { calls.rollbacks.push(baseline); },
    remove: async () => undefined,
  };
  const piSessions = {
    discover: async () => ({ piSessionId: process.checkpoint.piSessionId, sessionFile: process.checkpoint.sessionFile }),
    rollback: async (input: Record<string, unknown>) => {
      calls.piRollbacks.push(input);
      if (options?.rollbackPiFails) throw new Error("pi rollback failed");
      return { leafId: "rollback-leaf" };
    },
    fork: async (input: Record<string, unknown>) => {
      calls.piForks.push(input);
      return { sessionFile: "child.jsonl", leafId: String(input.sourceLeafId) };
    },
    remove: async () => undefined,
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees,
    piSessions,
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
    deadlineMs: 10_000,
    maxActiveSessions: options?.maxActiveSessions,
    maxRetainedSessionsPerOwner: options?.maxRetainedSessionsPerOwner,
  });
  return { orchestrator, process, calls, piSessions, worktrees };
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

test("completed prompts bind their pre-run Pi/Git baseline to the final leaf and commit", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "1", user_id: "U", prompt: "  fix payment timeout  ",
  });
  process.emit({ type: "settled" });
  await settle();

  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.display_prompt, "fix payment timeout");
  assert.equal(run.baseline_commit, "a".repeat(40));
  assert.equal(run.pi_parent_leaf_id, null);
  assert.equal(run.pi_user_entry_id, "user-1");
  assert.equal(run.pi_leaf_id, "leaf-1");
  assert.equal(run.commit, "b".repeat(40));
});

test("partial rollback failure marks needs-attention and blocks prompts, compact, and fork", async () => {
  const { orchestrator } = fixture({ rollbackPiFails: true });
  const accepted = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "2", user_id: "U", prompt: "unsafe change",
  });
  await orchestrator.cancel(accepted.run_id, "U");

  assert.equal(orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "2" })?.needs_attention, true);
  for (const operation of [
    () => orchestrator.prompt(accepted.session_id, "next", "U"),
    () => orchestrator.compact(accepted.session_id, "U"),
    () => orchestrator.fork(accepted.session_id, { user_id: "U", source_run_id: accepted.run_id, team_id: "T", channel_id: "C", thread_ts: "child" }),
  ]) {
    await assert.rejects(operation(), (error: unknown) => (error as { code?: string }).code === "session_needs_attention");
  }
});

test("tree and stats are sanitized while manual compaction is an isolated maintenance run", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "3", user_id: "U", prompt: "fix" });
  process.emit({ type: "settled" });
  await settle();

  const tree = await orchestrator.getTree(accepted.session_id, "U");
  const stats = await orchestrator.getStats(accepted.session_id, "U");
  const compacted = await orchestrator.compact(accepted.session_id, "U", "Preserve test results");
  await settle();

  assert.equal(tree.turns[0].run_id, accepted.run_id);
  assert.equal(tree.turns[0].forkable, true);
  assert.equal(tree.turns[0].active, true);
  assert.doesNotMatch(JSON.stringify(tree), /session\.jsonl|\/managed|payment|bootstrap/i);
  assert.equal(stats.auto_compaction_enabled, true);
  assert.equal(stats.compaction_count, 0);
  assert.deepEqual(process.compactCalls, [{ runId: compacted.run_id, instructions: "Preserve test results" }]);
  assert.equal(orchestrator.getRun(compacted.run_id).kind, "compaction");
  assert.equal(calls.inspectDiff, 1);
  assert.equal(calls.commits, 1);
});

test("fork creates an independent worktree and Pi child from the same completed checkpoint", async () => {
  const { orchestrator, process, calls } = fixture();
  const parent = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "4", user_id: "U", prompt: "first" });
  process.emit({ type: "settled" });
  await settle();

  const child = await orchestrator.fork(parent.session_id, {
    user_id: "U",
    source_run_id: parent.run_id,
    team_id: "T",
    channel_id: "C",
    thread_ts: "child-thread",
  });

  assert.equal(calls.creates[1].commit, "b".repeat(40));
  assert.equal(calls.piForks[0].sourceLeafId, "leaf-1");
  assert.equal(calls.piForks[0].childCwd, "/managed/child");
  assert.equal(child.baseline_commit, "b".repeat(40));
  assert.equal(child.parent_session_id, parent.session_id);
  assert.equal(child.fork_source_run_id, parent.run_id);
  assert.equal(orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "child-thread" })?.session_id, child.session_id);
});

test("diff validation failure rolls back both Git and Pi instead of leaving needs-attention", async () => {
  const { orchestrator, process, calls } = fixture({ inspectFails: true });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "5", user_id: "U", prompt: "too much" });

  process.emit({ type: "settled" });
  await settle();

  const run = orchestrator.getRun(accepted.run_id);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "diff_validation_failed");
  assert.deepEqual(calls.rollbacks, ["a".repeat(40)]);
  assert.equal(calls.piRollbacks.length, 1);
  assert.equal(orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "5" })?.needs_attention, false);
});

test("cancelling compaction does not inspect, commit, or rollback the worktree", async () => {
  const { orchestrator, process, calls } = fixture();
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "6", user_id: "U", prompt: "first" });
  process.emit({ type: "settled" });
  await settle();
  process.compact = async () => new Promise<never>(() => undefined);
  const maintenance = await orchestrator.compact(accepted.session_id, "U");
  process.checkpoint = { ...process.checkpoint, leafId: "compaction-leaf" };

  await orchestrator.cancel(maintenance.run_id, "U");

  assert.equal(orchestrator.getRun(maintenance.run_id).status, "cancelled");
  assert.equal(orchestrator.getRun(maintenance.run_id).pi_leaf_id, "compaction-leaf");
  assert.deepEqual(calls.rollbacks, []);
  assert.equal(calls.inspectDiff, 1);
  assert.equal(calls.commits, 1);
});

test("native Pi tree determines active ancestry after compaction", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "native", user_id: "U", prompt: "first",
  });
  process.emit({ type: "settled" });
  await settle();
  process.treeSnapshot = {
    tree: [
      {
        entry: { type: "message", id: "user-1", parentId: null },
        children: [
          {
            entry: { type: "message", id: "leaf-1", parentId: "user-1" },
            children: [
              {
                entry: { type: "compaction", id: "compact-1", parentId: "leaf-1" },
                children: [],
              },
              {
                entry: { type: "branch_summary", id: "alternate-1", parentId: "leaf-1" },
                children: [],
              },
            ],
          },
        ],
      },
    ],
    leafId: "compact-1",
  };

  const tree = await orchestrator.getTree(accepted.session_id, "U");

  assert.equal(tree.turns[0].active, true);
  assert.equal(tree.turns[0].forkable, true);
  assert.equal(tree.lineage.native_compactions, 1);
  assert.equal(tree.lineage.native_branch_points, 1);
});

test("uncertain cancelled compaction marks the session needs-attention", async () => {
  const { orchestrator, process } = fixture();
  const accepted = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "uncertain-compact", user_id: "U", prompt: "first",
  });
  process.emit({ type: "settled" });
  await settle();
  process.compact = async () => new Promise<never>(() => undefined);
  process.abortReturnsCheckpoint = false;
  const maintenance = await orchestrator.compact(accepted.session_id, "U");

  await orchestrator.cancel(maintenance.run_id, "U");

  assert.equal(orchestrator.getRun(maintenance.run_id).error?.code, "compaction_state_uncertain");
  assert.equal(
    orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "uncertain-compact" })?.needs_attention,
    true,
  );
});

test("prompt checkpoint is durably saved before Pi receives the task", async () => {
  let saved = false;
  let promptObservedSavedState = false;
  const process = new TreeProcess();
  process.prompt = () => { promptObservedSavedState = saved; };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w", branch: "b", baselineCommit: "base" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => undefined,
      remove: async () => undefined,
    },
    stateStore: {
      load: async () => ({ schema_version: 1, sessions: [], runs: [] }),
      save: async (state: { runs: Array<Record<string, unknown>> }) => {
        assert.equal(state.runs[0]?.baseline_commit, "base");
        saved = true;
      },
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });

  await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "persist-first", user_id: "U", prompt: "fix",
  });

  assert.equal(promptObservedSavedState, true);
});

test("fork reserves the parent until branch extraction completes", async () => {
  const { orchestrator, process, piSessions } = fixture();
  const parent = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "reserve", user_id: "U", prompt: "first",
  });
  process.emit({ type: "settled" });
  await settle();
  let release!: () => void;
  let entered!: () => void;
  const enteredGate = new Promise<void>((resolve) => { entered = resolve; });
  const releaseGate = new Promise<void>((resolve) => { release = resolve; });
  const originalDiscover = piSessions.discover;
  piSessions.discover = async (...args: Parameters<typeof originalDiscover>) => {
    entered();
    await releaseGate;
    return originalDiscover(...args);
  };

  const forking = orchestrator.fork(parent.session_id, {
    user_id: "U", source_run_id: parent.run_id, team_id: "T", channel_id: "C", thread_ts: "reserved-child",
  });
  await enteredGate;
  await assert.rejects(
    orchestrator.prompt(parent.session_id, "racing task", "U"),
    (error: unknown) => (error as { code?: string }).code === "session_busy",
  );
  release();
  await forking;
});

test("child follow-up prompts obey global active session capacity", async () => {
  const { orchestrator, process } = fixture({ maxActiveSessions: 1 });
  const parent = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "capacity-parent", user_id: "U", prompt: "first",
  });
  process.emit({ type: "settled" });
  await settle();
  const child = await orchestrator.fork(parent.session_id, {
    user_id: "U", source_run_id: parent.run_id, team_id: "T", channel_id: "C", thread_ts: "capacity-child",
  });
  await orchestrator.prompt(parent.session_id, "parent active", "U");

  await assert.rejects(
    orchestrator.prompt(String(child.session_id), "child active", "U"),
    (error: unknown) => (error as { code?: string }).code === "runtime_busy",
  );
});

test("retained session bound rejects unbounded forks before allocation", async () => {
  const { orchestrator, process, calls } = fixture({ maxRetainedSessionsPerOwner: 2 });
  const parent = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "bounded-parent", user_id: "U", prompt: "first",
  });
  process.emit({ type: "settled" });
  await settle();
  await orchestrator.fork(parent.session_id, {
    user_id: "U", source_run_id: parent.run_id, team_id: "T", channel_id: "C", thread_ts: "bounded-child-1",
  });

  await assert.rejects(
    orchestrator.fork(parent.session_id, {
      user_id: "U", source_run_id: parent.run_id, team_id: "T", channel_id: "C", thread_ts: "bounded-child-2",
    }),
    (error: unknown) => (error as { code?: string }).code === "runtime_busy",
  );
  assert.equal(calls.creates.length, 2);
});

test("fork persistence and cleanup failure leave a durable retryable tombstone", async () => {
  const parentProcess = new TreeProcess();
  const childProcess = new TreeProcess();
  let starts = 0;
  let failingSaves = 0;
  let piRemovalFails = true;
  const savedStates: Array<{ sessions: Array<{ cleanupPending?: boolean }> }> = [];
  const removedWorktrees: string[] = [];
  const removedPiSessions: string[] = [];
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async (_id: string, commit?: string) => ({
        path: starts === 0 ? "/managed/parent" : "/managed/child",
        branch: starts === 0 ? "parent" : "child",
        baselineCommit: commit ?? "base",
      }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "checkpoint",
      rollback: async () => undefined,
      remove: async (worktree: string) => { removedWorktrees.push(worktree); },
    },
    piSessions: {
      discover: async () => ({ piSessionId: parentProcess.checkpoint.piSessionId, sessionFile: "session.jsonl" }),
      rollback: async () => ({ leafId: "rollback" }),
      fork: async () => ({ sessionFile: "child.jsonl", leafId: "leaf-1" }),
      remove: async (sessionId: string) => {
        removedPiSessions.push(sessionId);
        if (piRemovalFails) throw new Error("Pi storage busy");
      },
    },
    stateStore: {
      load: async () => ({ schema_version: 1, sessions: [], runs: [] }),
      save: async (state: { sessions: Array<{ cleanupPending?: boolean }> }) => {
        if (failingSaves > 0) {
          failingSaves -= 1;
          throw new Error("disk full");
        }
        savedStates.push(structuredClone(state));
      },
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => {
      const process = starts++ === 0 ? parentProcess : childProcess;
      process.handler = onEvent;
      return process;
    },
  });
  const parent = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "persist-parent", user_id: "U", prompt: "first",
  });
  parentProcess.emit({ type: "settled" });
  await settle();
  failingSaves = 1;

  await assert.rejects(orchestrator.fork(parent.session_id, {
    user_id: "U",
    source_run_id: parent.run_id,
    team_id: "T",
    channel_id: "C",
    thread_ts: "persist-child",
  }), /disk full/);

  assert.equal(
    orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "persist-child" }),
    null,
  );
  assert.equal(childProcess.closeCalls, 1);
  assert.deepEqual(removedWorktrees, ["/managed/child"]);
  assert.equal(removedPiSessions.length, 1);
  assert.equal(savedStates.at(-1)?.sessions.some((session) => session.cleanupPending), true);

  piRemovalFails = false;
  await orchestrator.cleanupExpiredSessions();

  assert.equal(childProcess.closeCalls, 2);
  assert.deepEqual(removedWorktrees, ["/managed/child", "/managed/child"]);
  assert.equal(removedPiSessions.length, 2);
  assert.equal(savedStates.at(-1)?.sessions.some((session) => session.cleanupPending), false);
});

test("new session admission reaps expired retained worktrees", async () => {
  const expiredProcess = new TreeProcess();
  const freshProcess = new TreeProcess();
  const removed: string[] = [];
  let starts = 0;
  const orchestrator = new CodingAgentOrchestrator({
    maxRetainedSessionsPerOwner: 1,
    worktrees: {
      create: async () => ({ path: "/managed/fresh", branch: "fresh", baselineCommit: "base" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => undefined,
      remove: async (worktree: string) => { removed.push(worktree); },
    },
    stateStore: {
      load: async () => ({
        schema_version: 1,
        sessions: [
          {
            id: "expired",
            key: "T:C:expired",
            ownerUserId: "U",
            modelRef: "openai/gpt-5.6-luna",
            worktreePath: "/managed/expired",
            branch: "expired",
            baselineCommit: "base",
            closed: false,
            lastActivity: Date.now() - 25 * 60 * 60 * 1_000,
          },
        ],
        runs: [],
      }),
      save: async () => undefined,
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => {
      const process = starts++ === 0 ? expiredProcess : freshProcess;
      process.handler = onEvent;
      return process;
    },
  });
  await orchestrator.restore();

  await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "fresh", user_id: "U", prompt: "new",
  });

  assert.deepEqual(removed, ["/managed/expired"]);
  assert.equal(expiredProcess.closeCalls, 1);
});

test("tree and stats inspection reserve the session and global container slot", async () => {
  const { orchestrator, process } = fixture({ maxActiveSessions: 1 });
  const first = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "inspect-1", user_id: "U", prompt: "one",
  });
  process.emit({ type: "settled" });
  await settle();
  const second = await orchestrator.createSession({
    team_id: "T", channel_id: "C", thread_ts: "inspect-2", user_id: "U", prompt: "two",
  });
  process.emit({ type: "settled" });
  await settle();
  let entered!: () => void;
  let release!: () => void;
  const enteredGate = new Promise<void>((resolve) => { entered = resolve; });
  const releaseGate = new Promise<void>((resolve) => { release = resolve; });
  const originalGetTree = process.getTree.bind(process);
  process.getTree = async () => {
    entered();
    await releaseGate;
    return originalGetTree();
  };

  const inspecting = orchestrator.getTree(first.session_id, "U");
  await enteredGate;
  await assert.rejects(
    orchestrator.prompt(first.session_id, "race", "U"),
    (error: unknown) => (error as { code?: string }).code === "session_busy",
  );
  await assert.rejects(
    orchestrator.getStats(second.session_id, "U"),
    (error: unknown) => (error as { code?: string }).code === "runtime_busy",
  );
  await assert.rejects(
    orchestrator.createSession({
      team_id: "T", channel_id: "C", thread_ts: "inspect-3", user_id: "U", prompt: "three",
    }),
    (error: unknown) => (error as { code?: string }).code === "runtime_busy",
  );
  release();
  await inspecting;
});

test("partial expiry cleanup stays tombstoned and retries without republishing", async () => {
  const process = new TreeProcess();
  let piRemovalFails = true;
  const savedStates: Array<{ sessions: Array<{ cleanupPending?: boolean }> }> = [];
  let worktreeRemovals = 0;
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/new", branch: "new", baselineCommit: "base" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => undefined,
      remove: async () => { worktreeRemovals += 1; },
    },
    piSessions: {
      discover: async () => ({ piSessionId: "PI", sessionFile: "session.jsonl" }),
      rollback: async () => ({ leafId: "rollback" }),
      fork: async () => ({ sessionFile: "child.jsonl", leafId: "leaf" }),
      remove: async () => { if (piRemovalFails) throw new Error("Pi storage busy"); },
    },
    stateStore: {
      load: async () => ({
        schema_version: 1,
        sessions: [{
          id: "old",
          key: "T:C:old",
          ownerUserId: "U",
          modelRef: "openai/gpt-5.6-luna",
          worktreePath: "/managed/old",
          branch: "old",
          baselineCommit: "base",
          closed: false,
          lastActivity: Date.now() - 25 * 60 * 60 * 1_000,
          piSessionId: "PI",
          piSessionFile: "session.jsonl",
        }],
        runs: [],
      }),
      save: async (state: { sessions: Array<{ cleanupPending?: boolean }> }) => {
        savedStates.push(structuredClone(state));
      },
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  await orchestrator.restore();

  await orchestrator.cleanupExpiredSessions();

  assert.equal(
    orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "old" }),
    null,
  );
  assert.equal(savedStates.at(-1)?.sessions[0]?.cleanupPending, true);
  assert.equal(worktreeRemovals, 1);

  piRemovalFails = false;
  await orchestrator.cleanupExpiredSessions();

  assert.deepEqual(savedStates.at(-1)?.sessions, []);
  assert.equal(worktreeRemovals, 2);
});

test("restore reconciles an interrupted compaction from the exact Pi leaf", async () => {
  const process = new TreeProcess();
  let gitRollbacks = 0;
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/new", branch: "new", baselineCommit: "base" }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }),
      commit: async () => "base",
      rollback: async () => { gitRollbacks += 1; },
      remove: async () => undefined,
    },
    piSessions: {
      discover: async () => ({ piSessionId: "PI", sessionFile: "session.jsonl" }),
      read: async () => ({ piSessionId: "PI", sessionFile: "session.jsonl", leafId: "compacted-leaf" }),
      rollback: async () => ({ leafId: "rollback" }),
      fork: async () => ({ sessionFile: "child.jsonl", leafId: "leaf" }),
      remove: async () => undefined,
    },
    stateStore: {
      load: async () => ({
        schema_version: 1,
        sessions: [{
          id: "S",
          key: "T:C:compact-restart",
          ownerUserId: "U",
          modelRef: "openai/gpt-5.6-luna",
          worktreePath: "/managed/w",
          branch: "b",
          baselineCommit: "base",
          closed: false,
          lastActivity: Date.now(),
          piSessionId: "PI",
          piSessionFile: "session.jsonl",
          piLeafId: "old-leaf",
        }],
        runs: [{
          run_id: "RC",
          session_id: "S",
          status: "interrupted",
          answer: "",
          owner_user_id: "U",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          tool_count: 0,
          turn_count: 0,
          events: [],
          kind: "compaction",
        }],
      }),
      save: async () => undefined,
      takeExpiredSessions: () => [],
    } as any,
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
  });
  await orchestrator.restore();

  const next = await orchestrator.prompt("S", "continue", "U");

  assert.equal(orchestrator.getRun(next.run_id).pi_parent_leaf_id, "compacted-leaf");
  assert.equal(gitRollbacks, 0);
});
