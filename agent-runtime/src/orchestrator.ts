import { randomUUID } from "node:crypto";

import { listAgentModels, resolveAgentModel } from "./agent-model.js";
import { ApprovalStore } from "./policy.js";
import type { PublicRunEvent, RpcSessionState, RpcSessionStats, RpcTreeSnapshot } from "./pi-rpc.js";
import type { PersistedSession, RuntimeStateStore } from "./runtime-state.js";

export type RunStatus =
  | "starting"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "rejected"
  | "cancelled"
  | "failed"
  | "timed_out"
  | "interrupted";

export class RuntimeError extends Error {
  constructor(message: string, public readonly code: string, public readonly statusCode: number, options?: ErrorOptions) {
    super(message, options);
  }
}
export class RuntimeConflictError extends RuntimeError {
  constructor(message = "The session already has an active run") { super(message, "session_busy", 409); }
}
export class RuntimeForbiddenError extends RuntimeError {
  constructor(message = "This action is not allowed for this user") { super(message, "approval_forbidden", 401); }
}
export class RuntimeNotFoundError extends RuntimeError {
  constructor(message = "Session or run not found") { super(message, "not_found", 404); }
}
export class RuntimeBusyError extends RuntimeError {
  constructor() { super("Agent runtime is busy", "runtime_busy", 429); }
}
export class RuntimeNeedsAttentionError extends RuntimeError {
  constructor() { super("Session needs operator attention before it can continue", "session_needs_attention", 409); }
}

export interface SessionInput {
  team_id: string;
  channel_id: string;
  thread_ts: string;
  user_id: string;
  prompt: string;
  display_prompt?: string;
  model?: string;
}

export interface ProcessCheckpoint {
  piSessionId: string;
  sessionFile: string;
  leafId: string | null;
  userEntryId?: string;
}

export interface RunRecord {
  run_id: string;
  session_id: string;
  status: RunStatus;
  answer: string;
  owner_user_id: string;
  provider?: string;
  model?: string;
  reasoning_effort?: string;
  created_at: string;
  updated_at: string;
  tool_count: number;
  turn_count: number;
  events: Array<Record<string, unknown>>;
  kind?: "prompt" | "compaction";
  display_prompt?: string;
  baseline_commit?: string;
  pi_parent_leaf_id?: string | null;
  pi_user_entry_id?: string;
  pi_leaf_id?: string | null;
  parent_run_id?: string | null;
  changed_files?: string[];
  diff_stat?: string;
  branch?: string;
  commit?: string;
  cherry_pick?: string;
  error?: { code: string; message: string };
}

export interface AgentProcess {
  prompt(runId: string, prompt: string): void;
  decide(rpcUiId: string, approved: boolean): void;
  abort(runId: string): Promise<ProcessCheckpoint | void>;
  close(): Promise<void>;
  getCheckpoint?(): Promise<ProcessCheckpoint>;
  getState?(): Promise<RpcSessionState>;
  getSessionStats?(): Promise<RpcSessionStats>;
  getTree?(): Promise<RpcTreeSnapshot>;
  compact?(runId: string, customInstructions?: string): Promise<ProcessCheckpoint>;
}

interface Worktrees {
  create(runId: string, startCommit?: string): Promise<{ path: string; branch: string; baselineCommit: string }>;
  inspectDiff(path: string, limits: { maxFiles: number; maxBytes: number; maxSingleFileBytes: number }): Promise<{ files: string[]; bytes: number; stat: string }>;
  commit(path: string, message: string): Promise<string>;
  rollback(path: string, baseline: string): Promise<void>;
  remove(path: string): Promise<void>;
}

export interface PiSessionOperations {
  discover(runtimeSessionId: string, sessionFile?: string): Promise<{ piSessionId: string; sessionFile: string }>;
  read?(runtimeSessionId: string, sessionFile: string): Promise<{
    piSessionId: string;
    sessionFile: string;
    leafId: string | null;
  }>;
  rollback(input: { runtimeSessionId: string; sessionFile: string; parentLeafId: string | null; runId: string }): Promise<{ leafId: string }>;
  fork(input: {
    sourceRuntimeSessionId: string;
    sourceSessionFile: string;
    sourceLeafId: string;
    childRuntimeSessionId: string;
    childPiSessionId: string;
    childCwd: string;
  }): Promise<{ sessionFile: string; leafId: string | null }>;
  remove(runtimeSessionId: string): Promise<void>;
}

interface SessionRecord {
  id: string;
  key: string;
  ownerUserId: string;
  modelRef: string;
  worktreePath: string;
  branch: string;
  baselineCommit: string;
  process: AgentProcess;
  activeRunId?: string;
  closed: boolean;
  lastActivity: number;
  piSessionId: string;
  piSessionFile?: string;
  piLeafId: string | null;
  parentSessionId?: string;
  forkSourceRunId?: string;
  needsAttention?: boolean;
  activeCheckpointRunId?: string;
  cleanupPending?: boolean;
}

export class CodingAgentOrchestrator {
  private readonly worktrees: Worktrees;
  private readonly startProcess: (session: {
    id: string;
    worktreePath: string;
    modelRef: string;
    piSessionId: string;
    piSessionFile?: string;
    restored?: boolean;
  }, onEvent: (event: PublicRunEvent) => void) => Promise<AgentProcess>;
  private readonly maxActiveSessions: number;
  private readonly maxRetainedSessionsPerOwner: number;
  private readonly deadlineMs: number;
  private readonly availableModelRefs: Set<string>;
  private readonly stateStore?: RuntimeStateStore;
  private readonly piSessions?: PiSessionOperations;
  private readonly approvals = new ApprovalStore();
  private readonly sessions = new Map<string, SessionRecord>();
  private readonly threadSessions = new Map<string, string>();
  private readonly runs = new Map<string, RunRecord>();
  private readonly cleanupTombstones = new Map<string, PersistedSession>();
  private readonly cleanupProcesses = new Map<string, AgentProcess>();
  private readonly deadlines = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly terminalClaims = new Set<string>();
  private readonly pendingThreadKeys = new Set<string>();
  private readonly sessionReservations = new Set<string>();
  private readonly containerReservations = new Set<string>();
  private readonly retainedReservations = new Map<string, number>();
  private cleanupInFlight?: Promise<void>;
  private startingSessions = 0;

  constructor(options: {
    worktrees: Worktrees;
    startProcess: CodingAgentOrchestrator["startProcess"];
    maxActiveSessions?: number;
    maxRetainedSessionsPerOwner?: number;
    deadlineMs?: number;
    stateStore?: RuntimeStateStore;
    availableModelRefs?: string[];
    piSessions?: PiSessionOperations;
  }) {
    this.worktrees = options.worktrees;
    this.startProcess = options.startProcess;
    this.maxActiveSessions = options.maxActiveSessions ?? 2;
    this.maxRetainedSessionsPerOwner = options.maxRetainedSessionsPerOwner ?? 20;
    this.deadlineMs = options.deadlineMs ?? 15 * 60_000;
    this.availableModelRefs = new Set(options.availableModelRefs ?? listAgentModels().map((model) => model.ref));
    this.stateStore = options.stateStore;
    this.piSessions = options.piSessions;
  }

  async restore(): Promise<void> {
    if (!this.stateStore) return;
    const state = await this.stateStore.load();
    for (const run of state.runs) this.runs.set(run.run_id, run);
    for (const expired of this.stateStore.takeExpiredSessions()) {
      this.cleanupTombstones.set(expired.id, {
        ...expired,
        activeRunId: undefined,
        closed: true,
        cleanupPending: true,
      });
    }
    for (const saved of state.sessions) {
      if (saved.cleanupPending) {
        this.cleanupTombstones.set(saved.id, saved);
        continue;
      }
      if (saved.closed) continue;
      const latestRun = state.runs
        .filter((run) => run.session_id === saved.id)
        .sort((left, right) => Date.parse(left.updated_at) - Date.parse(right.updated_at))
        .at(-1);
      if (this.piSessions && !saved.piSessionFile) {
        try {
          const migrated = await this.piSessions.discover(saved.id);
          saved.piSessionId = migrated.piSessionId;
          saved.piSessionFile = migrated.sessionFile;
        } catch {
          saved.needsAttention = true;
        }
      }
      if (latestRun?.status === "interrupted") {
        let rollbackFailed = false;
        if (latestRun.kind === "compaction") {
          try {
            if (!this.piSessions?.read) throw new Error("Pi compaction checkpoint reader is unavailable");
            const located = await this.piSessions.discover(saved.id, saved.piSessionFile);
            const checkpoint = await this.piSessions.read(saved.id, located.sessionFile);
            saved.piSessionId = checkpoint.piSessionId;
            saved.piSessionFile = checkpoint.sessionFile;
            saved.piLeafId = checkpoint.leafId;
          } catch {
            rollbackFailed = true;
          }
        } else {
          try {
            await this.worktrees.rollback(saved.worktreePath, latestRun.baseline_commit ?? saved.baselineCommit);
          } catch {
            rollbackFailed = true;
          }
          if (this.piSessions) try {
            const located = await this.piSessions.discover(saved.id, saved.piSessionFile);
            const rollback = await this.piSessions.rollback({
              runtimeSessionId: saved.id,
              sessionFile: located.sessionFile,
              parentLeafId: latestRun.pi_parent_leaf_id ?? null,
              runId: latestRun.run_id,
            });
            saved.piSessionId = located.piSessionId;
            saved.piSessionFile = located.sessionFile;
            saved.piLeafId = rollback.leafId;
            saved.activeCheckpointRunId = latestRun.parent_run_id ?? undefined;
          } catch {
            rollbackFailed = true;
          }
        }
        if (rollbackFailed) saved.needsAttention = true;
      }
      let session!: SessionRecord;
      const modelRef = saved.modelRef ?? resolveAgentModel().ref;
      const piSessionId = saved.piSessionId ?? saved.id;
      const process = await this.startProcess(
        {
          id: saved.id,
          worktreePath: saved.worktreePath,
          modelRef,
          piSessionId,
          piSessionFile: saved.piSessionFile,
          restored: true,
        },
        (event) => this.handleProcessEvent(session.id, event),
      );
      session = {
        ...saved,
        modelRef,
        piSessionId,
        piLeafId: saved.piLeafId ?? latestRun?.pi_leaf_id ?? null,
        activeCheckpointRunId: saved.activeCheckpointRunId,
        process,
      };
      this.sessions.set(session.id, session);
      this.threadSessions.set(session.key, session.id);
    }
    await this.persistState();
    await this.retryCleanupTombstones();
  }

  async createSession(input: SessionInput): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    validateSessionInput(input);
    await this.cleanupExpiredSessions();
    let requestedModel;
    try {
      requestedModel = resolveAgentModel(input.model);
    } catch (error) {
      throw new RuntimeError("Requested Agent model is not allowlisted", "invalid_model", 400, { cause: error });
    }
    if (!this.availableModelRefs.has(requestedModel.ref)) {
      throw new RuntimeError("Requested model provider is not configured", "provider_not_configured", 503);
    }
    const key = `${input.team_id}:${input.channel_id}:${input.thread_ts}`;
    const existingId = this.threadSessions.get(key);
    if (existingId) {
      const existing = this.sessions.get(existingId);
      if (existing && !existing.closed) {
        if (existing.ownerUserId !== input.user_id) throw new RuntimeForbiddenError("Only the session owner can submit prompts");
        if (input.model && existing.modelRef !== requestedModel.ref) {
          throw new RuntimeConflictError("An existing Agent session cannot switch models");
        }
        return this.prompt(existing.id, input.prompt, input.user_id);
      }
    }
    if (this.pendingThreadKeys.has(key)) throw new RuntimeConflictError("This Slack thread is already creating a session");
    this.requireRetainedCapacity(input.user_id);
    this.requireActiveCapacity();

    const sessionId = randomUUID();
    const piSessionId = randomUUID();
    const runId = randomUUID();
    this.pendingThreadKeys.add(key);
    this.startingSessions += 1;
    this.reserveRetainedSession(input.user_id);
    try {
      const worktree = await this.worktrees.create(runId);
      let session!: SessionRecord;
      let process: AgentProcess;
      try {
        process = await this.startProcess(
          { id: sessionId, worktreePath: worktree.path, modelRef: requestedModel.ref, piSessionId },
          (event) => this.handleProcessEvent(session.id, event),
        );
      } catch (error) {
        try { await this.worktrees.remove(worktree.path); } catch { /* isolated cleanup is best effort */ }
        throw new RuntimeError("Agent container could not start", "container_start_failed", 502, { cause: error });
      }
      session = {
        id: sessionId,
        key,
        ownerUserId: input.user_id,
        modelRef: requestedModel.ref,
        worktreePath: worktree.path,
        branch: worktree.branch,
        baselineCommit: worktree.baselineCommit,
        process,
        closed: false,
        lastActivity: Date.now(),
        piSessionId,
        piLeafId: null,
      };
      this.sessions.set(sessionId, session);
      this.threadSessions.set(key, sessionId);
      try {
        await this.startRun(session, runId, input.prompt, input.display_prompt);
      } catch (error) {
        this.sessions.delete(session.id);
        this.threadSessions.delete(session.key);
        try { await session.process.close(); } catch { /* no run was accepted */ }
        try { await this.worktrees.remove(session.worktreePath); } catch { /* isolated cleanup is best effort */ }
        if (this.piSessions) {
          try { await this.piSessions.remove(session.id); } catch { /* no durable session remains */ }
        }
        throw error;
      }
      return { session_id: sessionId, run_id: runId, status: "starting" };
    } finally {
      this.pendingThreadKeys.delete(key);
      this.startingSessions -= 1;
      this.releaseRetainedSession(input.user_id);
    }
  }

  async prompt(sessionId: string, prompt: string, userId: string): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    const session = this.requireSession(sessionId);
    if (session.ownerUserId !== userId) throw new RuntimeForbiddenError("Only the session owner can submit prompts");
    this.requireSessionReady(session);
    this.requireSessionIdle(session);
    this.requireActiveCapacity();
    if (!this.availableModelRefs.has(session.modelRef)) {
      throw new RuntimeError("Session model provider is not configured", "provider_not_configured", 503);
    }
    if (!prompt.trim()) throw new RuntimeError("prompt is required", "invalid_request", 400);
    const runId = randomUUID();
    await this.startRun(session, runId, prompt);
    return { session_id: sessionId, run_id: runId, status: "starting" };
  }

  async getTree(sessionId: string, userId: string): Promise<{
    session_id: string;
    active_leaf_id: string | null;
    turns: Array<Record<string, unknown>>;
    lineage: Record<string, unknown>;
    truncated: boolean;
  }> {
    const session = this.requireOwnedSession(sessionId, userId);
    this.requireSessionIdle(session);
    const snapshot = !session.needsAttention && session.process.getTree
      ? await this.withContainerReservation(session, () => session.process.getTree!())
      : { tree: [], leafId: session.piLeafId };
    const nativeTree = indexNativeTree(snapshot);
    const promptRuns = [...this.runs.values()]
      .filter((run) => run.session_id === session.id && (run.kind ?? "prompt") === "prompt")
      .sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at));
    const start = Math.max(0, promptRuns.length - 100);
    const runByUserEntry = new Map(
      promptRuns
        .filter((run): run is RunRecord & { pi_user_entry_id: string } => Boolean(run.pi_user_entry_id))
        .map((run) => [run.pi_user_entry_id, run]),
    );
    const activeRunId = closestRunAncestor(snapshot.leafId, nativeTree.parents, runByUserEntry)?.run_id;
    const turns = promptRuns.slice(start).map((run, index) => {
      const nativeParent = run.pi_user_entry_id
        ? closestRunAncestor(nativeTree.parents.get(run.pi_user_entry_id) ?? null, nativeTree.parents, runByUserEntry)
        : undefined;
      const checkpointExists = Boolean(run.pi_leaf_id && nativeTree.entries.has(run.pi_leaf_id));
      return {
        number: start + index + 1,
        run_id: run.run_id,
        pi_entry_id: run.pi_user_entry_id,
        parent_run_id: nativeParent?.run_id ?? run.parent_run_id ?? null,
        task_preview: (run.display_prompt ?? "").replace(/\s+/g, " ").trim().slice(0, 160),
        status: run.status,
        timestamp: run.created_at,
        commit: run.commit,
        active: run.run_id === activeRunId,
        forkable: run.status === "completed" && Boolean(run.commit) && checkpointExists,
      };
    });
    return {
      session_id: session.id,
      active_leaf_id: snapshot.leafId,
      turns,
      lineage: {
        parent_session_id: session.parentSessionId,
        fork_source_run_id: session.forkSourceRunId,
        native_branch_points: nativeTree.branchPoints,
        native_compactions: nativeTree.compactions,
      },
      truncated: promptRuns.length > 100,
    };
  }

  async getStats(sessionId: string, userId: string): Promise<Record<string, unknown>> {
    const session = this.requireOwnedSession(sessionId, userId);
    this.requireSessionReady(session);
    this.requireSessionIdle(session);
    if (!session.process.getSessionStats || !session.process.getState) {
      throw new RuntimeError("Pi session statistics are unavailable", "rpc_unavailable", 502);
    }
    const { stats, state } = await this.withContainerReservation(session, async () => ({
      stats: await session.process.getSessionStats!(),
      state: await session.process.getState!(),
    }));
    const compactionCount = [...this.runs.values()]
      .filter((run) => run.session_id === session.id)
      .flatMap((run) => run.events)
      .filter((event) => event.type === "compaction" && event.phase === "end" && event.aborted !== true).length;
    return {
      session_id: session.id,
      user_messages: stats.userMessages,
      assistant_messages: stats.assistantMessages,
      tool_calls: stats.toolCalls,
      tool_results: stats.toolResults,
      total_messages: stats.totalMessages,
      tokens: stats.tokens,
      cost: stats.cost,
      context_usage: stats.contextUsage,
      auto_compaction_enabled: state.autoCompactionEnabled,
      compaction_count: compactionCount,
    };
  }

  async compact(sessionId: string, userId: string, customInstructions?: string): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    const session = this.requireOwnedSession(sessionId, userId);
    this.requireSessionReady(session);
    this.requireSessionIdle(session);
    this.requireActiveCapacity();
    if (!session.process.compact) throw new RuntimeError("Pi compaction is unavailable", "rpc_unavailable", 502);
    if (customInstructions && customInstructions.length > 2_000) {
      throw new RuntimeError("Compaction instructions are too large", "invalid_request", 400);
    }
    const runId = randomUUID();
    const now = new Date().toISOString();
    const model = resolveAgentModel(session.modelRef);
    const run: RunRecord = {
      run_id: runId,
      session_id: session.id,
      status: "running",
      answer: "",
      owner_user_id: session.ownerUserId,
      provider: model.provider,
      model: model.id,
      reasoning_effort: model.reasoningEffort,
      created_at: now,
      updated_at: now,
      tool_count: 0,
      turn_count: 0,
      events: [{ type: "status", status: "starting" }],
      kind: "compaction",
      branch: session.branch,
    };
    this.runs.set(runId, run);
    session.activeRunId = runId;
    session.lastActivity = Date.now();
    try {
      await this.persistState();
    } catch (error) {
      this.runs.delete(runId);
      session.activeRunId = undefined;
      throw new RuntimeError(
        "Compaction checkpoint could not be persisted before maintenance",
        "checkpoint_persist_failed",
        502,
        { cause: error },
      );
    }
    const deadline = setTimeout(() => void this.failCompaction(session, run, "compaction_timeout", "Compaction exceeded its deadline"), 5 * 60_000);
    deadline.unref();
    this.deadlines.set(runId, deadline);
    void session.process.compact(runId, customInstructions?.trim() || undefined).then(
      (checkpoint) => this.completeCompaction(session, run, checkpoint),
      () => this.failCompaction(session, run, "compaction_failed", "Pi could not compact this session"),
    );
    return { session_id: session.id, run_id: runId, status: "starting" };
  }

  async fork(sessionId: string, input: {
    user_id: string;
    source_run_id: string;
    team_id: string;
    channel_id: string;
    thread_ts: string;
  }): Promise<Record<string, unknown>> {
    await this.cleanupExpiredSessions();
    const parent = this.requireOwnedSession(sessionId, input.user_id);
    this.requireSessionReady(parent);
    this.requireSessionIdle(parent);
    this.requireRetainedCapacity(parent.ownerUserId);
    if (!this.piSessions) throw new RuntimeError("Pi session fork is unavailable", "rpc_unavailable", 502);
    for (const field of [input.team_id, input.channel_id, input.thread_ts]) {
      if (!field?.trim()) throw new RuntimeError("Fork destination is required", "invalid_request", 400);
    }
    const source = this.runs.get(input.source_run_id);
    if (!source || source.session_id !== parent.id) throw new RuntimeNotFoundError("Fork source run not found");
    if (source.status !== "completed" || !source.commit || !source.pi_leaf_id) {
      throw new RuntimeConflictError("Fork source does not have a completed checkpoint");
    }
    const key = `${input.team_id}:${input.channel_id}:${input.thread_ts}`;
    if (this.threadSessions.has(key) || this.pendingThreadKeys.has(key)) {
      throw new RuntimeConflictError("Fork destination already has an Agent session");
    }
    const childSessionId = randomUUID();
    const childPiSessionId = randomUUID();
    this.pendingThreadKeys.add(key);
    this.sessionReservations.add(parent.id);
    this.reserveRetainedSession(parent.ownerUserId);
    try {
      const parentLocation = await this.piSessions.discover(parent.id, parent.piSessionFile);
      const worktree = await this.worktrees.create(childSessionId, source.commit);
      let child: SessionRecord | undefined;
      let process: AgentProcess | undefined;
      let cleanupRecord: PersistedSession = {
        id: childSessionId,
        key,
        ownerUserId: parent.ownerUserId,
        modelRef: parent.modelRef,
        worktreePath: worktree.path,
        branch: worktree.branch,
        baselineCommit: source.commit,
        closed: true,
        lastActivity: Date.now(),
        piSessionId: childPiSessionId,
        parentSessionId: parent.id,
        forkSourceRunId: source.run_id,
        cleanupPending: true,
      };
      try {
        const piChild = await this.piSessions.fork({
          sourceRuntimeSessionId: parent.id,
          sourceSessionFile: parentLocation.sessionFile,
          sourceLeafId: source.pi_leaf_id,
          childRuntimeSessionId: childSessionId,
          childPiSessionId,
          childCwd: worktree.path,
        });
        cleanupRecord = {
          ...cleanupRecord,
          piSessionFile: piChild.sessionFile,
          piLeafId: piChild.leafId,
        };
        process = await this.startProcess({
          id: childSessionId,
          worktreePath: worktree.path,
          modelRef: parent.modelRef,
          piSessionId: childPiSessionId,
          piSessionFile: piChild.sessionFile,
          restored: true,
        }, (event) => this.handleProcessEvent(child!.id, event));
        child = {
          id: childSessionId,
          key,
          ownerUserId: parent.ownerUserId,
          modelRef: parent.modelRef,
          worktreePath: worktree.path,
          branch: worktree.branch,
          baselineCommit: source.commit,
          process,
          closed: false,
          lastActivity: Date.now(),
          piSessionId: childPiSessionId,
          piSessionFile: piChild.sessionFile,
          piLeafId: piChild.leafId,
          parentSessionId: parent.id,
          forkSourceRunId: source.run_id,
        };
        this.sessions.set(child.id, child);
        this.threadSessions.set(key, child.id);
        await this.persistState();
        return {
          session_id: child.id,
          branch: child.branch,
          baseline_commit: child.baselineCommit,
          parent_session_id: parent.id,
          fork_source_run_id: source.run_id,
          model_ref: child.modelRef,
        };
      } catch (error) {
        if (child) {
          cleanupRecord = {
            ...this.persistedSession(child),
            activeRunId: undefined,
            closed: true,
            cleanupPending: true,
          };
        }
        this.sessions.delete(childSessionId);
        this.threadSessions.delete(key);
        this.cleanupTombstones.set(childSessionId, cleanupRecord);
        if (process) this.cleanupProcesses.set(childSessionId, process);
        try {
          await this.persistState();
          await this.cleanupTombstone(childSessionId);
        } catch {
          /* durable/in-memory tombstone is retried by the expiry sweep */
        }
        throw error;
      }
    } finally {
      this.pendingThreadKeys.delete(key);
      this.sessionReservations.delete(parent.id);
      this.releaseRetainedSession(parent.ownerUserId);
    }
  }

  getRun(runId: string): RunRecord {
    const run = this.runs.get(runId);
    if (!run) throw new RuntimeNotFoundError("Run not found");
    return structuredClone(run);
  }

  findSession(input: { team_id: string; channel_id: string; thread_ts: string }): string | undefined {
    return this.threadSessions.get(`${input.team_id}:${input.channel_id}:${input.thread_ts}`);
  }

  lookupSession(input: { team_id: string; channel_id: string; thread_ts: string }): Record<string, unknown> | null {
    const sessionId = this.findSession(input);
    if (!sessionId) return null;
    const session = this.sessions.get(sessionId);
    if (!session || session.closed) return null;
    return {
      session_id: session.id,
      run_id: session.activeRunId,
      owner_user_id: session.ownerUserId,
      branch: session.branch,
      model_ref: session.modelRef,
      active: Boolean(session.activeRunId),
      needs_attention: Boolean(session.needsAttention),
      parent_session_id: session.parentSessionId,
      fork_source_run_id: session.forkSourceRunId,
    };
  }

  async decide(runId: string, approvalId: string, userId: string, decision: "approve" | "reject"): Promise<void> {
    const run = this.runs.get(runId);
    if (!run) throw new RuntimeNotFoundError("Run not found");
    const session = this.requireSession(run.session_id);
    if (run.status !== "awaiting_approval" || session.activeRunId !== runId || this.terminalClaims.has(runId)) {
      throw new RuntimeConflictError("Run is not awaiting this approval");
    }
    const result = this.approvals.decide(approvalId, userId, decision, runId);
    if (result.code === "wrong_user") throw new RuntimeForbiddenError();
    if (result.code === "not_found") throw new RuntimeNotFoundError("Approval not found");
    if (result.code === "wrong_run" || result.code === "expired" || result.code === "already_used") {
      throw new RuntimeConflictError(`Approval is ${result.code.replace("_", " ")}`);
    }
    if (decision === "reject" && !this.claimTerminal(run)) {
      throw new RuntimeConflictError("Run is already finalizing");
    }
    session.process.decide(result.approval!.toolCallId, decision === "approve");
    if (decision === "approve") {
      this.updateRun(run, "running");
      run.events.push({ type: "approval_decision", decision: "approved", approval_id: approvalId });
    } else {
      if (!await this.stopBeforeRollback(session, run)) {
        await this.persistState();
        return;
      }
      this.updateRun(run, "rejected");
      run.events.push({ type: "approval_decision", decision: "rejected", approval_id: approvalId });
      this.finishActiveRun(session, runId);
      await this.safeRollback(session, run);
    }
    await this.persistState();
  }

  async cancel(runId: string, userId: string): Promise<void> {
    const run = this.runs.get(runId);
    if (!run) throw new RuntimeNotFoundError("Run not found");
    if (run.owner_user_id !== userId) throw new RuntimeForbiddenError();
    if (isTerminal(run.status)) throw new RuntimeConflictError("Run is already terminal");
    const session = this.requireSession(run.session_id);
    if (!this.claimTerminal(run)) throw new RuntimeConflictError("Run is already finalizing");
    if (run.kind === "compaction") {
      const reconciled = await this.stopAndReconcileCompaction(session, run);
      if (!reconciled) {
        run.error = {
          code: "compaction_state_uncertain",
          message: "Pi compaction state needs operator attention",
        };
      }
      this.updateRun(run, reconciled ? "cancelled" : "failed");
      this.finishActiveRun(session, runId);
      await this.persistState();
      return;
    }
    if (!await this.stopBeforeRollback(session, run)) {
      await this.persistState();
      return;
    }
    this.updateRun(run, "cancelled");
    this.finishActiveRun(session, runId);
    await this.safeRollback(session, run);
    await this.persistState();
  }

  async closeSession(sessionId: string, userId: string): Promise<void> {
    const session = this.requireSession(sessionId);
    if (session.ownerUserId !== userId) throw new RuntimeForbiddenError();
    if (this.sessionReservations.has(session.id)) throw new RuntimeConflictError();
    if (session.activeRunId) await this.cancel(session.activeRunId, userId);
    await session.process.close();
    const tombstone: PersistedSession = {
      ...this.persistedSession(session),
      activeRunId: undefined,
      closed: true,
      cleanupPending: true,
    };
    session.closed = true;
    this.sessions.delete(session.id);
    this.threadSessions.delete(session.key);
    this.cleanupTombstones.set(session.id, tombstone);
    try {
      await this.persistState();
    } catch (error) {
      this.cleanupTombstones.delete(session.id);
      session.closed = false;
      this.sessions.set(session.id, session);
      this.threadSessions.set(session.key, session.id);
      throw error;
    }
    await this.cleanupTombstone(session.id).catch(() => undefined);
  }

  listEvents(runId: string, after = 0): Array<Record<string, unknown>> {
    return this.getRun(runId).events.slice(Math.max(0, after));
  }

  private async startRun(
    session: SessionRecord,
    runId: string,
    prompt: string,
    displayPrompt?: string,
  ): Promise<void> {
    const now = new Date().toISOString();
    const model = resolveAgentModel(session.modelRef);
    const run: RunRecord = {
      run_id: runId,
      session_id: session.id,
      status: "running",
      answer: "",
      owner_user_id: session.ownerUserId,
      provider: model.provider,
      model: model.id,
      reasoning_effort: model.reasoningEffort,
      created_at: now,
      updated_at: now,
      tool_count: 0,
      turn_count: 0,
      events: [{ type: "status", status: "starting" }],
      branch: session.branch,
      kind: "prompt",
      display_prompt: (displayPrompt ?? prompt).trim().slice(0, 2_000),
      baseline_commit: session.baselineCommit,
      pi_parent_leaf_id: session.piLeafId,
      parent_run_id: session.activeCheckpointRunId ?? null,
    };
    this.runs.set(runId, run);
    session.activeRunId = runId;
    session.lastActivity = Date.now();
    try {
      await this.persistState();
    } catch (error) {
      this.runs.delete(runId);
      session.activeRunId = undefined;
      throw new RuntimeError(
        "Agent checkpoint could not be persisted before model execution",
        "checkpoint_persist_failed",
        502,
        { cause: error },
      );
    }
    const deadline = setTimeout(() => void this.timeoutRun(session, run), this.deadlineMs);
    deadline.unref();
    this.deadlines.set(runId, deadline);
    session.process.prompt(runId, prompt.trim());
  }

  private handleProcessEvent(sessionId: string, event: PublicRunEvent): void {
    const session = this.sessions.get(sessionId);
    if (!session?.activeRunId) return;
    const run = this.runs.get(session.activeRunId);
    if (!run || isTerminal(run.status)) return;
    if (event.type === "turn") {
      run.turn_count += 1;
      void this.persistState();
      return;
    }
    if (event.type === "approval") {
      const approval = this.approvals.create({
        runId: run.run_id,
        toolCallId: event.approval_id,
        userId: run.owner_user_id,
        action: `${event.title}\n${event.detail}`,
      });
      this.updateRun(run, "awaiting_approval");
      run.events.push({ ...event, approval_id: approval.id, expires_at: new Date(approval.expiresAt).toISOString() });
      void this.persistState();
      return;
    }
    if (event.type === "tool") {
      if (event.phase === "start") run.tool_count += 1;
      run.events.push(event);
      void this.persistState();
      return;
    }
    if (event.type === "answer") {
      run.answer = event.text;
      run.events.push(event);
      void this.persistState();
      return;
    }
    if (event.type === "compaction") {
      run.events.push(event);
      void this.persistState();
      return;
    }
    if (event.type === "error") {
      void this.failRun(session, run, event.code, event.message);
      return;
    }
    if (event.type === "settled") void this.completeRun(session, run);
  }

  private async completeRun(session: SessionRecord, run: RunRecord): Promise<void> {
    if (!this.claimTerminal(run)) return;
    let checkpoint: ProcessCheckpoint | undefined;
    try {
      checkpoint = await session.process.getCheckpoint?.();
      if (this.piSessions && !checkpoint) throw new Error("Pi checkpoint unavailable");
    } catch {
      run.error = { code: "checkpoint_failed", message: "Pi checkpoint could not be verified" };
      this.updateRun(run, "failed");
      this.finishActiveRun(session, run.run_id);
      await this.safeRollback(session, run);
      await this.persistState();
      return;
    }
    let diff: { files: string[]; bytes: number; stat: string };
    try {
      diff = await this.worktrees.inspectDiff(session.worktreePath, {
        maxFiles: 50,
        maxBytes: 1024 * 1024,
        maxSingleFileBytes: 256 * 1024,
      });
    } catch {
      run.error = { code: "diff_validation_failed", message: "Agent changes failed final diff validation" };
      this.updateRun(run, "failed");
      this.finishActiveRun(session, run.run_id);
      await this.safeRollback(session, run);
      await this.persistState();
      return;
    }
    try {
      const commit = await this.worktrees.commit(session.worktreePath, `Pi agent: ${run.run_id}`);
      session.baselineCommit = commit;
      run.changed_files = diff.files;
      run.diff_stat = diff.stat;
      run.commit = commit;
      run.cherry_pick = `git cherry-pick ${commit}`;
      run.events.push({ type: "diff", files: diff.files, stat: diff.stat, bytes: diff.bytes });
      if (checkpoint) {
        session.piSessionId = checkpoint.piSessionId;
        session.piSessionFile = checkpoint.sessionFile;
        session.piLeafId = checkpoint.leafId;
        run.pi_user_entry_id = checkpoint.userEntryId;
        run.pi_leaf_id = checkpoint.leafId;
        session.activeCheckpointRunId = run.run_id;
      }
      this.updateRun(run, "completed");
      this.finishActiveRun(session, run.run_id);
    } catch {
      run.error = { code: "commit_failed", message: "Changes remain in the isolated worktree and need attention" };
      session.needsAttention = true;
      this.updateRun(run, "failed");
      this.finishActiveRun(session, run.run_id);
    }
    await this.persistState();
  }

  private async completeCompaction(session: SessionRecord, run: RunRecord, checkpoint: ProcessCheckpoint): Promise<void> {
    if (!this.claimTerminal(run)) return;
    this.applyCheckpoint(session, checkpoint);
    run.pi_leaf_id = checkpoint.leafId;
    this.updateRun(run, "completed");
    this.finishActiveRun(session, run.run_id);
    await this.persistState();
  }

  private async failCompaction(session: SessionRecord, run: RunRecord, code: string, message: string): Promise<void> {
    if (!this.claimTerminal(run)) return;
    const reconciled = await this.stopAndReconcileCompaction(session, run);
    run.error = reconciled
      ? { code, message }
      : { code: "compaction_state_uncertain", message: "Pi compaction state needs operator attention" };
    this.updateRun(run, "failed");
    this.finishActiveRun(session, run.run_id);
    await this.persistState();
  }

  private async failRun(session: SessionRecord, run: RunRecord, code: string, message: string): Promise<void> {
    if (!this.claimTerminal(run)) return;
    if (!await this.stopBeforeRollback(session, run)) {
      await this.persistState();
      return;
    }
    run.error = { code, message };
    this.updateRun(run, "failed");
    this.finishActiveRun(session, run.run_id);
    await this.safeRollback(session, run);
    await this.persistState();
  }

  private async timeoutRun(session: SessionRecord, run: RunRecord): Promise<void> {
    if (!this.claimTerminal(run)) return;
    if (!await this.stopBeforeRollback(session, run)) {
      await this.persistState();
      return;
    }
    run.error = { code: "run_timeout", message: "Agent prompt exceeded its deadline" };
    this.updateRun(run, "timed_out");
    this.finishActiveRun(session, run.run_id);
    await this.safeRollback(session, run);
    await this.persistState();
  }

  private async safeRollback(session: SessionRecord, run: RunRecord): Promise<void> {
    if (run.kind === "compaction") return;
    let failed = false;
    try {
      await this.worktrees.rollback(session.worktreePath, run.baseline_commit ?? session.baselineCommit);
    } catch {
      failed = true;
    }
    if (this.piSessions) {
      try {
        const checkpoint = await session.process.getCheckpoint?.();
        const located = await this.piSessions.discover(
          session.id,
          checkpoint?.sessionFile ?? session.piSessionFile,
        );
        const rollback = await this.piSessions.rollback({
          runtimeSessionId: session.id,
          sessionFile: located.sessionFile,
          parentLeafId: run.pi_parent_leaf_id ?? null,
          runId: run.run_id,
        });
        session.piSessionId = located.piSessionId;
        session.piSessionFile = located.sessionFile;
        session.piLeafId = rollback.leafId;
        session.activeCheckpointRunId = run.parent_run_id ?? undefined;
      } catch {
        failed = true;
      }
    }
    if (failed) {
      session.needsAttention = true;
      run.error = { code: "rollback_failed", message: "Pi session or isolated worktree needs attention" };
    }
  }

  private async stopBeforeRollback(session: SessionRecord, run: RunRecord): Promise<boolean> {
    try {
      await session.process.abort(run.run_id);
      return true;
    } catch {
      run.error = {
        code: "container_stop_failed",
        message: "Changes remain isolated because the Agent container could not be confirmed stopped",
      };
      this.updateRun(run, "failed");
      session.needsAttention = true;
      this.finishActiveRun(session, run.run_id);
      return false;
    }
  }

  private async stopAndReconcileCompaction(
    session: SessionRecord,
    run: RunRecord,
  ): Promise<boolean> {
    try {
      const checkpoint = await session.process.abort(run.run_id);
      if (!checkpoint) throw new Error("Compaction checkpoint is unavailable");
      this.applyCheckpoint(session, checkpoint);
      run.pi_leaf_id = checkpoint.leafId;
      return true;
    } catch {
      session.needsAttention = true;
      return false;
    }
  }

  private applyCheckpoint(session: SessionRecord, checkpoint: ProcessCheckpoint): void {
    session.piSessionId = checkpoint.piSessionId;
    session.piSessionFile = checkpoint.sessionFile;
    session.piLeafId = checkpoint.leafId;
  }

  private finishActiveRun(session: SessionRecord, runId: string): void {
    const timer = this.deadlines.get(runId);
    if (timer) clearTimeout(timer);
    this.deadlines.delete(runId);
    if (session.activeRunId === runId) session.activeRunId = undefined;
  }

  private claimTerminal(run: RunRecord): boolean {
    if (isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return false;
    this.terminalClaims.add(run.run_id);
    this.approvals.invalidateRun(run.run_id);
    const timer = this.deadlines.get(run.run_id);
    if (timer) clearTimeout(timer);
    this.deadlines.delete(run.run_id);
    return true;
  }

  private updateRun(run: RunRecord, status: RunStatus): void {
    run.status = status;
    run.updated_at = new Date().toISOString();
    run.events.push({ type: "status", status });
  }

  private requireSession(sessionId: string): SessionRecord {
    const session = this.sessions.get(sessionId);
    if (!session || session.closed) throw new RuntimeNotFoundError("Session not found");
    return session;
  }

  private requireOwnedSession(sessionId: string, userId: string): SessionRecord {
    const session = this.requireSession(sessionId);
    if (session.ownerUserId !== userId) throw new RuntimeForbiddenError("Only the session owner can access this session");
    return session;
  }

  private requireSessionReady(session: SessionRecord): void {
    if (session.needsAttention) throw new RuntimeNeedsAttentionError();
  }

  private requireSessionIdle(session: SessionRecord): void {
    if (session.activeRunId || this.sessionReservations.has(session.id)) {
      throw new RuntimeConflictError();
    }
  }

  private requireActiveCapacity(): void {
    const active = [...this.sessions.values()].filter(
      (session) => !session.closed && Boolean(session.activeRunId),
    ).length;
    if (active + this.startingSessions + this.containerReservations.size >= this.maxActiveSessions) {
      throw new RuntimeBusyError();
    }
  }

  private async withContainerReservation<T>(
    session: SessionRecord,
    operation: () => Promise<T>,
  ): Promise<T> {
    this.requireSessionIdle(session);
    this.requireActiveCapacity();
    this.sessionReservations.add(session.id);
    this.containerReservations.add(session.id);
    try {
      return await operation();
    } finally {
      this.containerReservations.delete(session.id);
      this.sessionReservations.delete(session.id);
    }
  }

  private requireRetainedCapacity(userId: string): void {
    const retained = [...this.sessions.values()].filter(
      (session) => !session.closed && session.ownerUserId === userId,
    ).length + [...this.cleanupTombstones.values()].filter(
      (session) => session.ownerUserId === userId,
    ).length + (this.retainedReservations.get(userId) ?? 0);
    if (retained >= this.maxRetainedSessionsPerOwner) throw new RuntimeBusyError();
  }

  private reserveRetainedSession(userId: string): void {
    this.retainedReservations.set(
      userId,
      (this.retainedReservations.get(userId) ?? 0) + 1,
    );
  }

  private releaseRetainedSession(userId: string): void {
    const next = (this.retainedReservations.get(userId) ?? 1) - 1;
    if (next > 0) this.retainedReservations.set(userId, next);
    else this.retainedReservations.delete(userId);
  }

  cleanupExpiredSessions(now = Date.now()): Promise<void> {
    if (this.cleanupInFlight) return this.cleanupInFlight;
    this.cleanupInFlight = this.cleanupExpiredSessionsOnce(now).finally(() => {
      this.cleanupInFlight = undefined;
    });
    return this.cleanupInFlight;
  }

  private async cleanupExpiredSessionsOnce(now: number): Promise<void> {
    await this.retryCleanupTombstones();
    const cutoff = now - 24 * 60 * 60 * 1_000;
    for (const session of [...this.sessions.values()]) {
      if (
        session.closed
        || session.activeRunId
        || this.sessionReservations.has(session.id)
        || session.lastActivity >= cutoff
      ) {
        continue;
      }
      this.sessionReservations.add(session.id);
      try {
        await session.process.close();
      } catch {
        session.needsAttention = true;
        this.sessionReservations.delete(session.id);
        await this.persistState();
        continue;
      }
      const tombstone = {
        ...this.persistedSession(session),
        activeRunId: undefined,
        closed: true,
        cleanupPending: true,
      };
      session.closed = true;
      this.sessions.delete(session.id);
      this.threadSessions.delete(session.key);
      this.cleanupTombstones.set(session.id, tombstone);
      try {
        await this.persistState();
      } catch {
        this.cleanupTombstones.delete(session.id);
        session.closed = false;
        this.sessions.set(session.id, session);
        this.threadSessions.set(session.key, session.id);
        this.sessionReservations.delete(session.id);
        continue;
      }
      this.sessionReservations.delete(session.id);
      await this.cleanupTombstone(session.id).catch(() => undefined);
    }
  }

  private async retryCleanupTombstones(): Promise<void> {
    if (this.cleanupTombstones.size > 0) {
      try {
        await this.persistState();
      } catch {
        return;
      }
    }
    for (const sessionId of [...this.cleanupTombstones.keys()]) {
      await this.cleanupTombstone(sessionId).catch(() => undefined);
    }
  }

  private async cleanupTombstone(sessionId: string): Promise<void> {
    const tombstone = this.cleanupTombstones.get(sessionId);
    if (!tombstone) return;
    const process = this.cleanupProcesses.get(sessionId);
    if (process) await process.close();
    await this.worktrees.remove(tombstone.worktreePath);
    if (this.piSessions) await this.piSessions.remove(sessionId);
    this.cleanupTombstones.delete(sessionId);
    this.cleanupProcesses.delete(sessionId);
    try {
      await this.persistState();
    } catch (error) {
      this.cleanupTombstones.set(sessionId, tombstone);
      if (process) this.cleanupProcesses.set(sessionId, process);
      throw error;
    }
  }

  private persistedSession(session: SessionRecord): PersistedSession {
    return {
      id: session.id,
      key: session.key,
      ownerUserId: session.ownerUserId,
      modelRef: session.modelRef,
      worktreePath: session.worktreePath,
      branch: session.branch,
      baselineCommit: session.baselineCommit,
      activeRunId: session.activeRunId,
      closed: session.closed,
      lastActivity: session.lastActivity,
      piSessionId: session.piSessionId,
      piSessionFile: session.piSessionFile,
      piLeafId: session.piLeafId,
      parentSessionId: session.parentSessionId,
      forkSourceRunId: session.forkSourceRunId,
      needsAttention: session.needsAttention,
      activeCheckpointRunId: session.activeCheckpointRunId,
      cleanupPending: session.cleanupPending,
    };
  }

  private async persistState(): Promise<void> {
    if (!this.stateStore) return;
    await this.stateStore.save({
      schema_version: 1,
      sessions: [
        ...[...this.sessions.values()].map((session) => this.persistedSession(session)),
        ...[...this.cleanupTombstones.values()].map((session) => structuredClone(session)),
      ],
      runs: [...this.runs.values()].map((run) => structuredClone(run)),
    });
  }
}

function indexNativeTree(snapshot: RpcTreeSnapshot): {
  entries: Set<string>;
  parents: Map<string, string | null>;
  branchPoints: number;
  compactions: number;
} {
  const entries = new Set<string>();
  const parents = new Map<string, string | null>();
  let branchPoints = 0;
  let compactions = 0;
  const visit = (node: Record<string, unknown>) => {
    const entry = isRecord(node.entry) ? node.entry : undefined;
    const children = Array.isArray(node.children)
      ? node.children.filter(isRecord)
      : [];
    if (entry && typeof entry.id === "string") {
      entries.add(entry.id);
      parents.set(
        entry.id,
        typeof entry.parentId === "string" ? entry.parentId : null,
      );
      if (entry.type === "compaction") compactions += 1;
      if (children.length > 1) branchPoints += 1;
    }
    for (const child of children) visit(child);
  };
  for (const root of snapshot.tree) visit(root);
  return { entries, parents, branchPoints, compactions };
}

function closestRunAncestor(
  startId: string | null | undefined,
  parents: Map<string, string | null>,
  runByUserEntry: Map<string, RunRecord>,
): RunRecord | undefined {
  let current = startId ?? null;
  const visited = new Set<string>();
  while (current && !visited.has(current)) {
    visited.add(current);
    const run = runByUserEntry.get(current);
    if (run) return run;
    current = parents.get(current) ?? null;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateSessionInput(input: SessionInput): void {
  for (const field of [input.team_id, input.channel_id, input.thread_ts, input.user_id, input.prompt]) {
    if (typeof field !== "string" || !field.trim()) throw new RuntimeError("Required session field is missing", "invalid_request", 400);
  }
  if (input.prompt.length > 20_000) throw new RuntimeError("Prompt is too large", "invalid_request", 400);
  if (input.display_prompt !== undefined && (
    typeof input.display_prompt !== "string" || !input.display_prompt.trim() || input.display_prompt.length > 2_000
  )) {
    throw new RuntimeError("display_prompt is invalid", "invalid_request", 400);
  }
}

function isTerminal(status: RunStatus): boolean {
  return ["completed", "rejected", "cancelled", "failed", "timed_out", "interrupted"].includes(status);
}
