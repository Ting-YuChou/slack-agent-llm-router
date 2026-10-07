import { randomUUID } from "node:crypto";

import { listAgentModels, resolveAgentModel } from "./agent-model.js";
import type { RouteDecision } from "./jev-router.js";
import { ApprovalStore } from "./policy.js";
import type { PublicRunEvent } from "./pi-rpc.js";
import type { RuntimeStateStore } from "./runtime-state.js";

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

export interface SessionInput {
  team_id: string;
  channel_id: string;
  thread_ts: string;
  user_id: string;
  prompt: string;
  model?: string;
  routing_text?: string;
  task_id?: string;
}

export interface RunRecord {
  run_id: string;
  session_id: string;
  status: RunStatus;
  answer: string;
  owner_user_id: string;
  task_id?: string;
  baseline_commit?: string;
  capture_complete?: boolean | null;
  capture_reasons?: string[];
  provider?: string;
  model?: string;
  reasoning_effort?: string;
  routing?: RouteDecision;
  estimated_cost_usd?: number;
  mcp_server?: string;
  mcp_tool?: string;
  mcp_repository?: string;
  mcp_latency_ms?: number;
  mcp_success?: boolean;
  mcp_error_code?: string | null;
  mcp_result_bytes?: number;
  mcp_summary?: {
    total_calls: number;
    failures: number;
    total_result_bytes: number;
    servers: Record<string, { calls: number; failures: number; total_latency_ms: number; result_bytes: number }>;
  };
  created_at: string;
  updated_at: string;
  tool_count: number;
  turn_count: number;
  events: Array<Record<string, unknown>>;
  changed_files?: string[];
  diff_stat?: string;
  branch?: string;
  commit?: string;
  cherry_pick?: string;
  error?: { code: string; message: string };
}

export interface AgentProcess {
  prompt(runId: string, prompt: string, route: RouteDecision): void;
  decide(rpcUiId: string, approved: boolean): void;
  abort(runId: string): Promise<void>;
  close(): Promise<void>;
}

interface Worktrees {
  create(runId: string): Promise<{ path: string; branch: string; baselineCommit: string }>;
  inspectDiff(path: string, limits: { maxFiles: number; maxBytes: number; maxSingleFileBytes: number }): Promise<{ files: string[]; bytes: number; stat: string }>;
  commit(path: string, message: string): Promise<string>;
  rollback(path: string, baseline: string): Promise<void>;
  remove(path: string): Promise<void>;
}

interface SessionRecord {
  id: string;
  key: string;
  ownerUserId: string;
  modelRef: string;
  autoRouting: boolean;
  worktreePath: string;
  branch: string;
  baselineCommit: string;
  process: AgentProcess;
  activeRunId?: string;
  closed: boolean;
  lastActivity: number;
}

export class CodingAgentOrchestrator {
  private readonly worktrees: Worktrees;
  private readonly startProcess: (session: { id: string; ownerUserId: string; worktreePath: string; modelRef: string; restored?: boolean }, onEvent: (event: PublicRunEvent) => void) => Promise<AgentProcess>;
  private readonly maxActiveSessions: number;
  private readonly deadlineMs: number;
  private readonly availableModelRefs: Set<string>;
  private readonly stateStore?: RuntimeStateStore;
  private readonly router?: { route(text?: string): Promise<RouteDecision> };
  private readonly mcpRepository?: string;
  private readonly approvals = new ApprovalStore();
  private readonly sessions = new Map<string, SessionRecord>();
  private readonly threadSessions = new Map<string, string>();
  private readonly runs = new Map<string, RunRecord>();
  private readonly deadlines = new Map<string, ReturnType<typeof setTimeout>>();
  private readonly terminalClaims = new Set<string>();
  private readonly pendingThreadKeys = new Set<string>();
  private readonly pendingSessionIds = new Set<string>();
  private readonly closingSessionIds = new Set<string>();
  private readonly mcpToolStarts = new Map<string, number>();
  private startingSessions = 0;

  private readonly captureState?: (runId: string) => { complete: boolean | null; reasons: string[] };
  private readonly observeRun?: (run: RunRecord) => void;
  private readonly observeLifecycle?: (runId: string, name: string, phase: "start" | "end", failed?: boolean, sessionId?: string) => void;

  constructor(options: {
    captureState?: CodingAgentOrchestrator["captureState"];
    observeRun?: (run: RunRecord) => void;
    observeLifecycle?: CodingAgentOrchestrator["observeLifecycle"];
    worktrees: Worktrees;
    startProcess: CodingAgentOrchestrator["startProcess"];
    maxActiveSessions?: number;
    deadlineMs?: number;
    stateStore?: RuntimeStateStore;
    availableModelRefs?: string[];
    router?: { route(text?: string): Promise<RouteDecision> };
    mcpRepository?: string;
  }) {
    this.captureState = options.captureState;
    this.observeRun = options.observeRun; this.observeLifecycle = options.observeLifecycle;
    this.worktrees = options.worktrees;
    this.startProcess = options.startProcess;
    this.maxActiveSessions = options.maxActiveSessions ?? 2;
    this.deadlineMs = options.deadlineMs ?? 15 * 60_000;
    this.availableModelRefs = new Set(options.availableModelRefs ?? listAgentModels().map((model) => model.ref));
    this.stateStore = options.stateStore;
    this.router = options.router;
    this.mcpRepository = options.mcpRepository;
  }

  async restore(): Promise<void> {
    if (!this.stateStore) return;
    const state = await this.stateStore.load();
    for (const expired of this.stateStore.takeExpiredSessions()) {
      try { await this.worktrees.remove(expired.worktreePath); } catch { /* branch and audit metadata remain */ }
    }
    for (const run of state.runs) {
      if (!["running", "awaiting_approval"].includes(run.status) && run.capture_complete === undefined) {
        const recovered = run.status === "interrupted" ? this.captureState?.(run.run_id) : undefined;
        run.capture_complete = recovered?.complete === false ? false : null;
        run.capture_reasons = recovered?.complete === false ? recovered.reasons : ["historical_capture_unknown"];
      }
      this.runs.set(run.run_id, run);
    }
    for (const saved of state.sessions) {
      if (saved.closed) continue;
      const latestRun = state.runs
        .filter((run) => run.session_id === saved.id)
        .sort((left, right) => Date.parse(left.updated_at) - Date.parse(right.updated_at))
        .at(-1);
      if (latestRun?.status === "interrupted") {
        await this.worktrees.rollback(saved.worktreePath, saved.baselineCommit);
      }
      let session!: SessionRecord;
      const modelRef = saved.modelRef ?? resolveAgentModel().ref;
      const process = await this.startProcess(
        { id: saved.id, ownerUserId: saved.ownerUserId, worktreePath: saved.worktreePath, modelRef, restored: true },
        (event) => this.handleProcessEvent(session.id, event),
      );
      session = { ...saved, modelRef, autoRouting: saved.autoRouting === true, process };
      this.sessions.set(session.id, session);
      this.threadSessions.set(session.key, session.id);
    }
    await this.persistState();
  }

  async createSession(input: SessionInput): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    const requestedAt = new Date().toISOString();
    validateSessionInput(input);
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
        if (input.model && (existing.autoRouting || existing.modelRef !== requestedModel.ref)) {
          throw new RuntimeConflictError("An existing Agent session cannot switch models");
        }
        return this.prompt(existing.id, input.prompt, input.user_id, input.routing_text, input.task_id);
      }
    }
    if (this.pendingThreadKeys.has(key)) throw new RuntimeConflictError("This Slack thread is already creating a session");
    const activeSessions = [...this.sessions.values()].filter((session) => !session.closed && session.activeRunId).length;
    if (activeSessions + this.startingSessions >= this.maxActiveSessions) throw new RuntimeBusyError();

    const sessionId = randomUUID();
    const runId = randomUUID();
    this.pendingThreadKeys.add(key);
    this.startingSessions += 1;
    try {
      const autoRouting = !input.model;
      this.notifyLifecycle(runId, "routing", "start", false, sessionId);
      const route = await this.selectRoute(autoRouting, requestedModel.ref, input.routing_text);
      this.notifyLifecycle(runId, "routing", "end");
      this.notifyLifecycle(runId, "worktree_create", "start");
      const worktree = await this.worktrees.create(runId);
      this.notifyLifecycle(runId, "worktree_create", "end");
      let session!: SessionRecord;
      let process: AgentProcess;
      try {
        process = await this.startProcess(
          { id: sessionId, ownerUserId: input.user_id, worktreePath: worktree.path, modelRef: route.modelRef },
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
        modelRef: route.modelRef,
        autoRouting,
        worktreePath: worktree.path,
        branch: worktree.branch,
        baselineCommit: worktree.baselineCommit,
        process,
        closed: false,
        lastActivity: Date.now(),
      };
      this.sessions.set(sessionId, session);
      this.threadSessions.set(key, sessionId);
      this.startRun(session, runId, input.prompt, route, requestedAt, input.task_id);
      await this.persistState();
      return { session_id: sessionId, run_id: runId, status: "starting" };
    } catch (error) {
      this.notifyLifecycle(runId, "preparation", "end", true); throw error;
    } finally {
      this.pendingThreadKeys.delete(key);
      this.startingSessions -= 1;
    }
  }

  async prompt(sessionId: string, prompt: string, userId: string, routingText?: string, taskId?: string): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    const requestedAt = new Date().toISOString();
    const session = this.requireSession(sessionId);
    if (session.ownerUserId !== userId) throw new RuntimeForbiddenError("Only the session owner can submit prompts");
    if (session.activeRunId || this.pendingSessionIds.has(sessionId) || this.closingSessionIds.has(sessionId)) {
      throw new RuntimeConflictError();
    }
    if (!this.availableModelRefs.has(session.modelRef)) {
      throw new RuntimeError("Session model provider is not configured", "provider_not_configured", 503);
    }
    if (!prompt.trim()) throw new RuntimeError("prompt is required", "invalid_request", 400);
    validateTaskId(taskId);
    this.pendingSessionIds.add(sessionId);
    const runId = randomUUID();
    try {
      this.notifyLifecycle(runId, "routing", "start", false, sessionId);
      const route = await this.selectRoute(session.autoRouting, session.modelRef, routingText);
      this.notifyLifecycle(runId, "routing", "end");
      if (session.autoRouting) session.modelRef = route.modelRef;
      this.startRun(session, runId, prompt, route, requestedAt, taskId);
      await this.persistState();
      return { session_id: sessionId, run_id: runId, status: "starting" };
    } finally {
      this.pendingSessionIds.delete(sessionId);
    }
  }

  private async selectRoute(autoRouting: boolean, modelRef: string, routingText?: string): Promise<RouteDecision> {
    if (!autoRouting) return { modelRef, effort: "max", source: "explicit" };
    let decision: RouteDecision;
    try {
      decision = this.router ? await this.router.route(routingText)
        : { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "off" };
    } catch {
      return { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "fallback", reason: "jev_unavailable" };
    }
    if (!this.availableModelRefs.has(decision.modelRef)) {
      return { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "fallback", reason: "model_not_configured" };
    }
    return decision;
  }

  async markCaptureIncomplete(runId: string, reason: string): Promise<void> {
    const run = this.runs.get(runId);
    if (!run || run.capture_reasons?.includes(reason)) return;
    run.capture_complete = false;
    run.capture_reasons = [...(run.capture_reasons ?? []), reason];
    await this.persistState();
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
    if (this.pendingSessionIds.has(sessionId)) {
      throw new RuntimeConflictError("Session is starting a prompt");
    }
    if (this.closingSessionIds.has(sessionId)) throw new RuntimeConflictError("Session is already closing");
    this.closingSessionIds.add(sessionId);
    try {
      if (session.activeRunId) await this.cancel(session.activeRunId, userId);
      await session.process.close();
      await this.worktrees.remove(session.worktreePath);
      session.closed = true;
      this.threadSessions.delete(session.key);
      await this.persistState();
    } finally {
      this.closingSessionIds.delete(sessionId);
    }
  }

  listEvents(runId: string, after = 0): Array<Record<string, unknown>> {
    return this.getRun(runId).events.slice(Math.max(0, after));
  }

  private startRun(session: SessionRecord, runId: string, prompt: string, route: RouteDecision, requestedAt: string, taskId?: string): void {
    const model = resolveAgentModel(route.modelRef);
    const run: RunRecord = {
      run_id: runId,
      session_id: session.id,
      status: "running",
      answer: "",
      owner_user_id: session.ownerUserId,
      task_id: taskId,
      baseline_commit: session.baselineCommit,
      provider: model.provider,
      model: model.id,
      reasoning_effort: route.effort,
      routing: route,
      estimated_cost_usd: route.jevCostUsd ?? 0,
      created_at: requestedAt,
      updated_at: requestedAt,
      tool_count: 0,
      turn_count: 0,
      events: [{ type: "status", status: "starting" }],
      branch: session.branch,
    };
    this.runs.set(runId, run);
    session.activeRunId = runId;
    session.lastActivity = Date.now();
    this.notifyRun(run);
    session.process.prompt(runId, prompt.trim(), route);
    const deadline = setTimeout(() => void this.timeoutRun(session, run), this.deadlineMs);
    deadline.unref();
    this.deadlines.set(runId, deadline);
  }

  private handleProcessEvent(sessionId: string, event: PublicRunEvent): void {
    const session = this.sessions.get(sessionId);
    if (!session?.activeRunId) return;
    const run = this.runs.get(session.activeRunId);
    if (!run || isTerminal(run.status)) return;
    if (event.type === "usage") {
      run.estimated_cost_usd = (run.estimated_cost_usd ?? 0) + event.cost_usd;
      void this.persistState();
      return;
    }
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
      const mcpMatch = /^mcp__([^_]+)__(.+)$/.exec(event.tool);
      const mcpServer = mcpMatch?.[1];
      const mcpTool = mcpMatch?.[2];
      if (mcpTool && mcpServer) {
        const timingKey = `${sessionId}:${event.tool_call_id}`;
        if (event.phase === "start") this.mcpToolStarts.set(timingKey, Date.now());
        const started = this.mcpToolStarts.get(timingKey);
        if (event.phase === "end") this.mcpToolStarts.delete(timingKey);
        run.mcp_server = mcpServer;
        run.mcp_tool = mcpTool;
        if (mcpServer === "github" && this.mcpRepository) run.mcp_repository = this.mcpRepository;
        if (event.phase === "end") {
          run.mcp_latency_ms = started === undefined ? 0 : Math.max(0, Date.now() - started);
          run.mcp_success = event.is_error !== true;
          run.mcp_error_code = event.is_error === true ? "tool_error" : null;
          run.mcp_result_bytes = event.result_bytes ?? 0;
          const summary = run.mcp_summary ??= { total_calls: 0, failures: 0, total_result_bytes: 0, servers: {} };
          const serverSummary = summary.servers[mcpServer] ??= { calls: 0, failures: 0, total_latency_ms: 0, result_bytes: 0 };
          summary.total_calls += 1;
          summary.failures += event.is_error === true ? 1 : 0;
          summary.total_result_bytes += event.result_bytes ?? 0;
          serverSummary.calls += 1;
          serverSummary.failures += event.is_error === true ? 1 : 0;
          serverSummary.total_latency_ms += run.mcp_latency_ms;
          serverSummary.result_bytes += event.result_bytes ?? 0;
        }
        run.events.push({
          type: "mcp",
          phase: event.phase,
          server: mcpServer,
          tool: mcpTool,
          ...(mcpServer === "github" && this.mcpRepository ? { repository: this.mcpRepository } : {}),
          ...(event.phase === "end" ? { success: event.is_error !== true } : {}),
        });
      } else {
        run.events.push(event);
      }
      void this.persistState();
      return;
    }
    if (event.type === "answer") {
      run.answer = event.text;
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
    try {
      const diff = await this.worktrees.inspectDiff(session.worktreePath, {
        maxFiles: 50,
        maxBytes: 1024 * 1024,
        maxSingleFileBytes: 256 * 1024,
      });
      this.notifyLifecycle(run.run_id, "commit", "start");
      const commit = await this.worktrees.commit(session.worktreePath, `Pi agent: ${run.run_id}`);
      this.notifyLifecycle(run.run_id, "commit", "end");
      session.baselineCommit = commit;
      run.changed_files = diff.files;
      run.diff_stat = diff.stat;
      run.commit = commit;
      run.cherry_pick = `git cherry-pick ${commit}`;
      run.events.push({ type: "diff", files: diff.files, stat: diff.stat, bytes: diff.bytes });
      this.updateRun(run, "completed");
      this.finishActiveRun(session, run.run_id);
    } catch {
      this.notifyLifecycle(run.run_id, "commit", "end", true);
      run.error = { code: "commit_failed", message: "Changes remain in the isolated worktree and need attention" };
      this.updateRun(run, "failed");
      this.finishActiveRun(session, run.run_id);
    }
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
    this.notifyLifecycle(run.run_id, "rollback", "start");
    try {
      await this.worktrees.rollback(session.worktreePath, session.baselineCommit);
      this.notifyLifecycle(run.run_id, "rollback", "end");
    } catch {
      this.notifyLifecycle(run.run_id, "rollback", "end", true);
      run.error = { code: "rollback_failed", message: "Isolated worktree needs attention" };
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
      this.finishActiveRun(session, run.run_id);
      return false;
    }
  }

  private finishActiveRun(session: SessionRecord, runId: string): void {
    const timer = this.deadlines.get(runId);
    if (timer) clearTimeout(timer);
    this.deadlines.delete(runId);
    for (const key of this.mcpToolStarts.keys()) {
      if (key.startsWith(`${session.id}:`)) this.mcpToolStarts.delete(key);
    }
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

  async shutdown(): Promise<void> {
    for (const session of this.sessions.values()) {
      if (session.activeRunId) { try { await this.cancel(session.activeRunId, session.ownerUserId); } catch { /* retain worktree if stop fails */ } }
      try { await session.process.close(); } catch { /* stop failure already audited */ }
    }
  }

  private notifyLifecycle(runId: string, name: string, phase: "start" | "end", failed = false, sessionId?: string): void {
    try { this.observeLifecycle?.(runId, name, phase, failed, sessionId); } catch { /* monitoring must not fail work */ }
  }

  private notifyRun(run: RunRecord): void {
    try {
      if (!["running", "awaiting_approval"].includes(run.status) && run.capture_complete === undefined) {
        const capture = this.captureState?.(run.run_id);
        run.capture_complete = capture?.complete ?? null;
        run.capture_reasons = capture?.reasons ?? [];
      }
      this.observeRun?.(structuredClone(run)); } catch { /* monitoring cannot fail agent runs */ }
  }

  private async persistState(): Promise<void> {
    for (const run of this.runs.values()) this.notifyRun(run);
    if (!this.stateStore) return;
    await this.stateStore.save({
      schema_version: 1,
      sessions: [...this.sessions.values()].map((session) => ({
        id: session.id,
        key: session.key,
        ownerUserId: session.ownerUserId,
        modelRef: session.modelRef,
        autoRouting: session.autoRouting,
        worktreePath: session.worktreePath,
        branch: session.branch,
        baselineCommit: session.baselineCommit,
        activeRunId: session.activeRunId,
        closed: session.closed,
        lastActivity: session.lastActivity,
      })),
      runs: [...this.runs.values()].map((run) => structuredClone(run)),
    });
  }
}

function validateTaskId(value: unknown): void {
  if (value !== undefined && (typeof value !== "string" || !value.trim() || value.length > 128)) throw new RuntimeError("Invalid task_id", "invalid_request", 400);
}

function validateSessionInput(input: SessionInput): void {
  validateTaskId(input.task_id);
  for (const field of [input.team_id, input.channel_id, input.thread_ts, input.user_id, input.prompt]) {
    if (typeof field !== "string" || !field.trim()) throw new RuntimeError("Required session field is missing", "invalid_request", 400);
  }
  if (input.prompt.length > 20_000) throw new RuntimeError("Prompt is too large", "invalid_request", 400);
  if (input.routing_text !== undefined && typeof input.routing_text !== "string") {
    throw new RuntimeError("routing_text must be a string", "invalid_request", 400);
  }
}

function isTerminal(status: RunStatus): boolean {
  return ["completed", "rejected", "cancelled", "failed", "timed_out", "interrupted"].includes(status);
}
