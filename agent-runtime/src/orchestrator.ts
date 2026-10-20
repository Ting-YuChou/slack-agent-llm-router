import { createHash, randomUUID } from "node:crypto";
import { taskMode, type TaskMode } from "./task-mode.js";

import { listAgentModels, resolveAgentModel } from "./agent-model.js";
import type { RouteDecision } from "./jev-router.js";
import type { AgentLimits } from "./agent-limits.js";
import { ApprovalStore } from "./policy.js";
import type { PublicRunEvent, RunOutcome, UiResponse } from "./pi-rpc.js";
import type { RuntimeStateStore } from "./runtime-state.js";
import { workflowMode, type WorkflowMode, type WorkflowNode } from "./structured-workflow.js";
import { WorkflowBudget } from "./workflow-budget.js";
import type { Candidate, GateInput, GateResult } from "./quality-gate.js";
import { classifyGateError, failureCause, type RunFailure } from "./error-taxonomy.js";
import { promptTelemetry, renderPrompt } from "./prompts.js";
import type { FixResult } from "./fixers.js";
import { computeHarnessMetrics, type HarnessMetrics } from "./harness-metrics.js";
import { summarizePolicy, type PolicyDecision, type PolicySummary } from "./policy-decision.js";
import { redact } from "./telemetry.js";
import { DEFAULT_DIFF_LIMITS, DiffLimitError } from "./worktree-manager.js";
import { MIN_CHECKS_WINDOW_MS, rejected, type WriterChecksRequest, type WriterChecksResponse } from "./writer-checks.js";

export interface QualityServices {
  evaluate(input: GateInput): Promise<GateResult>;
  assertUnchanged(candidate: Candidate, input: GateInput): Promise<void>;
  execution(id: string, parent: string, role: "implementation" | "discussion", attempt: number, phase: "start" | "end", queueMs?: number): void | Promise<void>;
  stop?(id: string): Promise<void>;
  open?(id: string, initialCost: number): void;
  close?(id: string): void;
  archive?(input: GateInput): Promise<void>;
  elapsedMs?(id: string): number;
  activeTimeoutMs?: number;
  maxCostUsd?: number;
  structured?: { enabled: boolean; maxCostUsd: number; activeTimeoutMs: number };
  prepare?(id: string, mode: WorkflowMode): void;
  discover?(input: GateInput & { owner: string; initial_active_ms: number }): Promise<string>;
  writerStarted?(id: string): void;
  canResume?(id: string): boolean;
  resumeElapsed?(id: string): number;
  resumeCheckpoint?(id: string): number;
  stopDiscovery?(id: string): Promise<void>;
  validateResume?(input: GateInput): Promise<void>;
  tasks?(id: string): WorkflowNode[];
  /** The writer's own candidate-only check runs (run_checks). */
  runChecks?(request: WriterChecksRequest): Promise<WriterChecksResponse>;
  /** Repairs the profile allows; a diff over the limits on an attempt within it is repaired, not failed. */
  maxRepairs?: number;
  writerChecks?: {
    maxCalls: number; checks: string[]; defaultChecks: string[];
    /** Worst-case run time of these checks. */
    estimateMs(checks: string[]): number;
    /** Time kept back for final verification and grading. */
    reserveMs: number;
  };
}
interface WorkflowState { mode: WorkflowMode; discovery?: Promise<string>; priorActiveMs: number; initialActiveMs: number; requirement: string; route: RouteDecision; attempt: number; executionId: string; budget: WorkflowBudget; controller: AbortController; checking: boolean; operation?: Promise<GateResult>; checksOperation?: Promise<WriterChecksResponse>; checksCalls?: number }

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
  task_mode?: TaskMode;
  workflow_mode?: WorkflowMode;
  routing_text?: string;
  task_id?: string;
}

export interface RunRecord {
  /** Aggregate hash of prompts/*.md this run was built with (PROMPTS_VERSION). */
  prompts_version?: string;
  /** `prompt_id@hash` of each prompt the run's role uses. */
  prompts?: string[];
  task_mode?: TaskMode;
  workflow_mode?: WorkflowMode;
  workflow_requirement?: string;
  workflow_prompt?: string;
  traceparent?: string;
  interrupted_at?: string;
  suspended_ms?: number;
  prior_active_ms?: number;
  workflow_summary?: { revision: number; total: number; completed: number; unresolved?: number; agent_work_ms: number; queue_ms: number };
  budget?: ReturnType<WorkflowBudget["snapshot"]>;
  timing?: { wall_ms: number; active_ms: number; human_wait_ms: number };
  phase?: string;
  execution_status?: "running" | "succeeded" | "failed";
  /** run_checks calls the writer made in this run. */
  writer_checks_calls?: number;
  /** How the latest writer execution ended (Pi stop reason classification). */
  execution_outcome?: RunOutcome;
  grader_findings?: Array<{ path: string; line: number; severity: string; reason: string }>;
  verification_status?: string;
  grader_status?: string;
  /** Why the last grade was a harness-made unknown, when it was. */
  grader_cause?: { code: string; message: string };
  publication_status?: string;
  attempts?: Array<{ attempt: number; execution_id: string; candidate_hash?: string; verification_status?: string; grader_status?: string; artifact?: string; fixes?: FixResult[]; fixer_cause?: string }>;
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
  limits?: AgentLimits;
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
  /** Counts of the container's policy decisions (allow, block, approval) for this run. */
  policy_summary?: PolicySummary;
  /** Owner approvals this run asked for (tool writes, shell commands, delivery) and how they ended. */
  approvals?: ApprovalCounts;
  /** When the writer first started a write or edit tool call, for time-to-first-edit. */
  first_edit_at?: string;
  /** Computed once the run is terminal; see harness-metrics.ts. */
  harness_metrics?: HarnessMetrics;
  events: Array<Record<string, unknown>>;
  changed_files?: string[];
  diff_stat?: string;
  branch?: string;
  commit?: string;
  cherry_pick?: string;
  error?: RunFailure;
  /** Host path of the archived patch of an undelivered run; never leaves the host (see publicRun). */
  patch_artifact?: string;
}

/** The run as clients (HTTP, Slack, telemetry) see it: host artifact paths become a flag. */
export interface ApprovalCounts { requested: number; approved: number; rejected: number; expired: number; wait_ms: number }

export function publicRun<T extends Partial<RunRecord>>(run: T): Omit<T, "patch_artifact"> & { patch_retained?: true } {
  const { patch_artifact, ...rest } = run;
  const shown = rest.attempts ? { ...rest, attempts: rest.attempts.map(({ artifact: _artifact, ...attempt }) => attempt) } : rest;
  return patch_artifact ? { ...shown, patch_retained: true } : shown;
}

/** Archives the worktree's change before rollback; resolves to the archive path, or undefined when nothing changed. */
export type PatchArchiver = (input: { run_id: string; worktree: string; baseline: string }) => Promise<string | undefined>;

export interface AgentProcess {
  queueMs?: number;
  prompt(runId: string, prompt: string, route: RouteDecision, taskMode?: TaskMode): void;
  /** Limits the process applies for this route, recorded on the run as `limits`. */
  limitsFor?(route: RouteDecision): AgentLimits | undefined;
  decide(rpcUiId: string, approved: boolean): void;
  /** Answers an extension UI dialog other than an approval (run_checks). */
  respondUi(rpcUiId: string, response: UiResponse): void;
  abort(runId: string): Promise<void>;
  close(): Promise<void>;
}

interface Worktrees {
  /** Full patch against HEAD; delivery approval binds its digest when available. */
  patch?(path: string): Promise<string>;
  create(runId: string, trustedBase?: string): Promise<{ path: string; branch: string; baselineCommit: string }>;
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
  /** Trusted callers (CI repair) gate delivery themselves. */
  skipDelivery?: boolean;
  activeRunId?: string;
  closed: boolean;
  lastActivity: number;
}

export class CodingAgentOrchestrator {
  private readonly worktrees: Worktrees;
  private readonly startProcess: (session: { id: string; ownerUserId: string; worktreePath: string; modelRef: string; restored?: boolean; skipDelivery?: boolean }, onEvent: (event: PublicRunEvent) => void) => Promise<AgentProcess>;
  private readonly maxActiveSessions: number;
  private readonly deadlineMs: number;
  private readonly availableModelRefs: Set<string>;
  private readonly stateStore?: RuntimeStateStore;
  private readonly router?: { route(text?: string, runId?: string): Promise<RouteDecision> };
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
  private readonly workflows = new Map<string, WorkflowState>();
  private readonly quality?: QualityServices;
  private readonly captureTraceContext?: (id: string) => string | undefined;
  private readonly requireVerification: boolean;
  private readonly archivePatch?: PatchArchiver;
  private readonly deliveryApproval?: DeliveryApprovalOptions;
  /** Runs waiting for the owner to approve their diff, with the approval and the digest it binds. */
  private readonly deliveries = new Map<string, { approvalId: string; digest: string }>();
  /** Text of the assistant message currently streaming, per run; replaced by the final answer. */
  private readonly streamingAnswers = new Map<string, string>();
  /** When each open approval was shown, for the run's approval wait time. */
  private readonly approvalOpened = new Map<string, number>();
  /** Terminal runs already reported to observers, keyed to the update they were reported at (resume revives runs). */
  private readonly observedFinal = new Map<string, string>();
  private persistTimer?: ReturnType<typeof setTimeout>;

  private readonly captureState?: (runId: string) => { complete: boolean | null; reasons: string[] };
  private readonly observeRun?: (run: RunRecord) => void;
  private readonly observePolicy?: (runId: string, sessionId: string, decision: PolicyDecision) => void;
  private readonly observeLifecycle?: (runId: string, name: string, phase: "start" | "end", failed?: boolean, sessionId?: string) => void;

  constructor(options: {
    captureState?: CodingAgentOrchestrator["captureState"];
    observeRun?: (run: RunRecord) => void;
    observePolicy?: CodingAgentOrchestrator["observePolicy"];
    observeLifecycle?: CodingAgentOrchestrator["observeLifecycle"];
    worktrees: Worktrees;
    startProcess: CodingAgentOrchestrator["startProcess"];
    maxActiveSessions?: number;
    deadlineMs?: number;
    stateStore?: RuntimeStateStore;
    availableModelRefs?: string[];
    router?: { route(text?: string, runId?: string): Promise<RouteDecision> };
    mcpRepository?: string;
    quality?: QualityServices;
    captureTraceContext?: (id: string) => string | undefined;
    /** Production deployments refuse implement runs that no trusted profile can verify. */
    requireVerification?: boolean;
    /** Keeps the patch of every run that is rolled back instead of delivered, profile or not. */
    archivePatch?: PatchArchiver;
    /** Off by default: the owner approves the final diff before it is committed. */
    deliveryApproval?: DeliveryApprovalOptions;
  }) {
    this.captureState = options.captureState;
    this.observeRun = options.observeRun; this.observePolicy = options.observePolicy; this.observeLifecycle = options.observeLifecycle;
    this.worktrees = options.worktrees;
    this.startProcess = options.startProcess;
    this.maxActiveSessions = options.maxActiveSessions ?? 2;
    this.deadlineMs = options.deadlineMs ?? 15 * 60_000;
    this.availableModelRefs = new Set(options.availableModelRefs ?? listAgentModels().map((model) => model.ref));
    this.stateStore = options.stateStore;
    this.router = options.router;
    this.mcpRepository = options.mcpRepository;
    this.quality = options.quality;
    this.captureTraceContext = options.captureTraceContext;
    this.requireVerification = options.requireVerification === true;
    this.archivePatch = options.archivePatch;
    this.deliveryApproval = options.deliveryApproval;
    if (this.deliveryApproval && !this.worktrees.patch) throw new Error("Delivery approval needs worktrees.patch to bind the approved diff");
  }

  async restore(): Promise<void> {
    if (!this.stateStore) return;
    const state = await this.stateStore.load();
    for (const expired of this.stateStore.takeExpiredSessions()) {
      try {
        await this.quality?.archive?.({ run_id: `expired-${expired.id}`, worktree: expired.worktreePath, baseline: expired.baselineCommit, requirement: "Expired session", attempt: 0, signal: new AbortController().signal });
        await this.worktrees.remove(expired.worktreePath);
      } catch { /* Preserve isolated changes when evidence cannot be archived. */ }
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
      if (latestRun?.status === "interrupted" && !(latestRun.workflow_mode === "structured" && this.quality?.canResume?.(latestRun.run_id))) {
        if (latestRun.attempts?.length && !this.quality) throw new Error("Interrupted verified workflow requires its trusted profile before recovery");
        await this.quality?.archive?.({ run_id: latestRun.run_id, worktree: saved.worktreePath, baseline: saved.baselineCommit, requirement: "Interrupted workflow", attempt: 0, signal: new AbortController().signal });
        await this.retainPatch(latestRun, saved.worktreePath, saved.baselineCommit);
        await this.worktrees.rollback(saved.worktreePath, saved.baselineCommit);
        if (latestRun.phase === "awaiting_delivery") {
          latestRun.error = { code: "delivery_interrupted", message: "The runtime restarted before the changes were delivered; the patch was kept" };
          this.updateRun(latestRun, "failed");
        }
      }
      let session!: SessionRecord;
      const modelRef = saved.modelRef ?? resolveAgentModel().ref;
      const process = await this.startProcess(
        { id: saved.id, ownerUserId: saved.ownerUserId, worktreePath: saved.worktreePath, modelRef, restored: true, ...(saved.skipDelivery ? { skipDelivery: true } : {}) },
        (event) => this.handleProcessEvent(session.id, event),
      );
      session = { ...saved, modelRef, autoRouting: saved.autoRouting === true, process };
      this.sessions.set(session.id, session);
      this.threadSessions.set(session.key, session.id);
    }
    await this.persistState();
  }

  async createSession(input: SessionInput, trustedBase?: string, trustedRunId?: string, trusted: { delivery?: "skip" } = {}): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    if (this.stateStore?.healthy === false) throw new RuntimeError("Runtime control store unavailable", "control_unavailable", 503);
    const requestedAt = new Date().toISOString();
    validateSessionInput(input);
    const mode = this.requireWorkflowMode(input.workflow_mode);
    const task = this.requireTaskMode(input.task_mode);
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
        return this.prompt(existing.id, input.prompt, input.user_id, input.routing_text, input.task_id, mode, task);
      }
    }
    if (this.pendingThreadKeys.has(key)) throw new RuntimeConflictError("This Slack thread is already creating a session");
    // A run waiting for its owner to deliver holds no container work, so it does not use a slot.
    const activeSessions = [...this.sessions.values()].filter((session) => !session.closed && session.activeRunId && !this.deliveries.has(session.activeRunId)).length;
    if (activeSessions + this.startingSessions >= this.maxActiveSessions) throw new RuntimeBusyError();

    const sessionId = randomUUID();
    const runId = trustedRunId ?? randomUUID();
    this.pendingThreadKeys.add(key);
    this.startingSessions += 1;
    try {
      this.quality?.prepare?.(runId, mode);
      const autoRouting = !input.model;
      this.notifyLifecycle(runId, "routing", "start", false, sessionId);
      const route = await this.selectRoute(autoRouting, requestedModel.ref, input.routing_text, runId);
      this.notifyLifecycle(runId, "routing", "end");
      this.notifyLifecycle(runId, "worktree_create", "start");
      const worktree = await this.worktrees.create(runId, trustedBase);
      this.notifyLifecycle(runId, "worktree_create", "end");
      let session!: SessionRecord;
      let process: AgentProcess;
      try {
        process = await this.startProcess(
          { id: sessionId, ownerUserId: input.user_id, worktreePath: worktree.path, modelRef: route.modelRef, ...(trusted.delivery === "skip" ? { skipDelivery: true } : {}) },
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
        ...(trusted.delivery === "skip" ? { skipDelivery: true } : {}),
        closed: false,
        lastActivity: Date.now(),
      };
      this.sessions.set(sessionId, session);
      this.threadSessions.set(key, sessionId);
      await this.startRun(session, runId, input.prompt, route, requestedAt, input.task_id, input.routing_text, mode, task);
      await this.persistState();
      return { session_id: sessionId, run_id: runId, status: "starting" };
    } catch (error) {
      this.notifyLifecycle(runId, "preparation", "end", true); throw error;
    } finally {
      this.pendingThreadKeys.delete(key);
      this.startingSessions -= 1;
    }
  }

  async prompt(sessionId: string, prompt: string, userId: string, routingText?: string, taskId?: string, requestedMode?: WorkflowMode, requestedTaskMode?: TaskMode): Promise<{ session_id: string; run_id: string; status: "starting" }> {
    if (this.stateStore?.healthy === false) throw new RuntimeError("Runtime control store unavailable", "control_unavailable", 503);
    const mode = this.requireWorkflowMode(requestedMode);
    const task = this.requireTaskMode(requestedTaskMode);
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
    this.quality?.prepare?.(runId, mode);
    try {
      this.notifyLifecycle(runId, "routing", "start", false, sessionId);
      const route = await this.selectRoute(session.autoRouting, session.modelRef, routingText, runId);
      this.notifyLifecycle(runId, "routing", "end");
      if (session.autoRouting) session.modelRef = route.modelRef;
      await this.startRun(session, runId, prompt, route, requestedAt, taskId, routingText, mode, task);
      await this.persistState();
      return { session_id: sessionId, run_id: runId, status: "starting" };
    } finally {
      this.pendingSessionIds.delete(sessionId);
    }
  }

  private async selectRoute(autoRouting: boolean, modelRef: string, routingText?: string, runId?: string): Promise<RouteDecision> {
    if (!autoRouting) return { modelRef, effort: "max", source: "explicit" };
    let decision: RouteDecision;
    try {
      decision = this.router ? await this.router.route(routingText, runId)
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
    this.countApproval(run, approvalId, decision === "approve" ? "approved" : "rejected");
    if (this.deliveries.get(runId)?.approvalId === approvalId) {
      await this.decideDelivery(session, run, approvalId, decision);
      return;
    }
    if (decision === "reject" && !this.claimTerminal(run)) {
      throw new RuntimeConflictError("Run is already finalizing");
    }
    session.process.decide(result.approval!.toolCallId, decision === "approve");
    if (decision === "approve") {
      const workflow = this.workflows.get(runId);
      workflow?.budget.resume();
      if (workflow) this.armDeadline(session, run, workflow.budget.remainingMs());
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
    if (this.deliveries.delete(runId)) {
      await this.failClaimedRun(session, run, { code: "delivery_rejected", message: "The owner cancelled the run before delivery; the patch was kept" });
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

  private async startRun(session: SessionRecord, runId: string, prompt: string, route: RouteDecision, requestedAt: string, taskId?: string, requirement?: string, mode: WorkflowMode = "single", task: TaskMode = "implement"): Promise<void> {
    const model = resolveAgentModel(route.modelRef);
    const run: RunRecord = {
      run_id: runId,
      workflow_mode: mode,
      task_mode: task,
      ...promptTelemetry(task === "discuss" ? "discussion" : "implementation"),
      ...(mode === "structured" ? { workflow_requirement: requirement ?? prompt, workflow_prompt: prompt } : {}),
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
    const limits = session.process.limitsFor?.(route);
    if (limits) run.limits = limits;
    if (task === "implement" && !this.quality) run.verification_status = "unverified";
    if (task === "discuss") {
      run.phase = "discussing"; run.execution_status = "running";
      run.verification_status = "not_applicable"; run.grader_status = "not_applicable"; run.publication_status = "not_applicable";
    }
    this.runs.set(runId, run);
    session.activeRunId = runId;
    session.lastActivity = Date.now();
    this.notifyRun(run);
    try {
    if (this.quality) {
      const priorActiveMs = this.quality.elapsedMs?.(runId) ?? 0;
      if (mode === 'structured') run.prior_active_ms = priorActiveMs;
      const initialActiveMs = priorActiveMs + Math.max(0, Date.now() - Date.parse(requestedAt));
      const limits = mode === "structured" ? this.quality.structured! : this.quality;
      const workflow: WorkflowState = { mode, priorActiveMs, initialActiveMs, requirement: requirement ?? prompt, route, attempt: 1, executionId: randomUUID(), budget: new WorkflowBudget(limits.maxCostUsd, (limits.activeTimeoutMs ?? this.deadlineMs) - initialActiveMs), controller: new AbortController(), checking: false };
      workflow.budget.report(route.jevCostUsd ?? 0);
      this.quality.open?.(runId, route.jevCostUsd ?? 0);
      this.workflows.set(runId, workflow);
      run.execution_status = "running"; run.publication_status = task === "discuss" ? "not_applicable" : "local_only"; run.phase = task === "discuss" ? "discussing" : "implementing"; run.verification_status = task === "discuss" ? "not_applicable" : "pending"; run.grader_status = task === "discuss" ? "not_applicable" : "pending"; run.attempts = [];
    }
    // Persist the busy state and exact execution identity before launching any model work.
    await this.persistState();
    const workflow = this.workflows.get(runId);
    if (isTerminal(run.status) || this.terminalClaims.has(runId)) return;
    if (workflow?.mode === 'structured') this.launchDiscovery(session, run, workflow, prompt.trim());
    else if (workflow) await this.startExecution(session, run, workflow, prompt.trim());
    else session.process.prompt(runId, prompt.trim(), route, task);
    if (!isTerminal(run.status)) this.armDeadline(session, run, this.workflows.get(runId)?.budget.remainingMs() ?? this.deadlineMs);
    } catch (error) {
      if (this.stateStore?.healthy !== false && !isTerminal(run.status) && !this.terminalClaims.has(runId)) await this.failRun(session, run, { code: "execution_start_failed", message: "Execution could not start within its configured controls" });
      throw error;
    }
  }

  private requireTaskMode(value: unknown): TaskMode {
    let task: TaskMode;
    try { task = taskMode(value); } catch { throw new RuntimeError("Invalid task_mode; expected discuss or implement", "invalid_request", 400); }
    if (task === "implement" && this.requireVerification && !this.quality) {
      throw new RuntimeError("Implement runs require a trusted verification profile; use discuss mode", "verification_profile_required", 422);
    }
    return task;
  }

  private requireWorkflowMode(value: unknown): WorkflowMode {
    let mode: WorkflowMode;
    try { mode = workflowMode(value); } catch { throw new RuntimeError('Invalid workflow_mode', 'invalid_request', 400); }
    if (mode === 'structured' && (!this.quality?.structured?.enabled || !this.quality.discover)) throw new RuntimeError('Structured workflow requires an enabled trusted profile', 'workflow_setup_required', 503);
    return mode;
  }

  private launchDiscovery(session: SessionRecord, run: RunRecord, state: WorkflowState, prompt: string) {
    run.phase = 'planning';
    state.discovery = this.quality!.discover!({ run_id: run.run_id, owner: run.owner_user_id, worktree: session.worktreePath, baseline: session.baselineCommit, requirement: state.requirement, attempt: 0, signal: state.controller.signal, initial_active_ms: state.initialActiveMs });
    void state.discovery.then(async evidence => {
      state.controller.signal.throwIfAborted();
      if (isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
      this.quality!.writerStarted?.(run.run_id);
      run.phase = run.task_mode === 'discuss' ? 'discussing' : 'implementing'; await this.persistState();
      await this.startExecution(session, run, state, `${prompt}\n\n${evidence}`);
    }).catch(async (error: unknown) => {
      const cause = failureCause(error);
      if (!isTerminal(run.status) && !this.terminalClaims.has(run.run_id)) await this.failRun(session, run, { code: 'investigation_unknown', message: 'Required investigation could not complete; evidence retained', ...(cause ? { cause_code: cause } : {}) });
    });
  }

  tasks(runId: string): WorkflowNode[] { this.getRun(runId); return this.quality?.tasks?.(runId) ?? []; }

  async resume(runId: string, user: string) {
    const run = this.runs.get(runId); if (!run) throw new RuntimeNotFoundError();
    if (run.owner_user_id !== user) throw new RuntimeForbiddenError();
    this.requireWorkflowMode(run.workflow_mode);
    if (run.workflow_mode !== 'structured' || run.status !== 'interrupted' || !this.quality?.canResume?.(runId)) throw new RuntimeConflictError('Only interrupted read-only workflow stages can resume');
    const session = this.requireSession(run.session_id);
    if (session.activeRunId || this.pendingSessionIds.has(session.id) || this.closingSessionIds.has(session.id)) throw new RuntimeConflictError();
    if (session.baselineCommit !== run.baseline_commit) throw new RuntimeConflictError('Baseline changed; start a new task');
    this.pendingSessionIds.add(session.id);
    try { await this.quality.validateResume?.({ run_id: runId, worktree: session.worktreePath, baseline: session.baselineCommit, requirement: run.workflow_requirement ?? '', attempt: 0, signal: new AbortController().signal }); }
    catch { throw new RuntimeConflictError('Workflow identity or artifacts changed; start a new task'); }
    finally { this.pendingSessionIds.delete(session.id); }
    const limits = this.quality.structured!;
    const consumed = Math.max(this.quality.resumeElapsed?.(runId) ?? 0,(run.timing?.active_ms ?? 0) + (run.prior_active_ms ?? 0));
    if (consumed >= limits.activeTimeoutMs) throw new RuntimeConflictError('Workflow time budget exhausted');
    const state: WorkflowState = { mode: 'structured', priorActiveMs: run.prior_active_ms ?? 0, initialActiveMs: consumed, requirement: '', route: run.routing!, attempt: 1, executionId: randomUUID(), budget: new WorkflowBudget(limits.maxCostUsd, limits.activeTimeoutMs - consumed), controller: new AbortController(), checking: false };
    state.budget.report(run.estimated_cost_usd ?? 0);
    const saved = this.quality.tasks?.(runId); if (!saved?.length) throw new RuntimeConflictError('Workflow artifacts missing');
    state.requirement = run.workflow_requirement ?? '';
    if (!state.requirement) throw new RuntimeConflictError('Workflow requirement missing');
    this.terminalClaims.delete(runId); this.workflows.set(runId,state); session.activeRunId = runId;
    run.suspended_ms = (run.suspended_ms ?? 0) + Math.max(0, Date.now() - Math.max(this.quality.resumeCheckpoint?.(runId) ?? 0,Date.parse(run.interrupted_at ?? run.updated_at)));
    run.status = 'running'; run.updated_at = new Date().toISOString();
    run.execution_status = 'running'; run.verification_status = run.task_mode === 'discuss' ? 'not_applicable' : 'pending'; run.grader_status = run.task_mode === 'discuss' ? 'not_applicable' : 'pending';
    this.notifyLifecycle(runId,'resume','start',false,session.id);
    run.traceparent = this.captureTraceContext?.(runId) ?? run.traceparent;
    this.notifyLifecycle(runId,'resume','end',false,session.id);
    await this.persistState(); this.launchDiscovery(session,run,state,run.workflow_prompt ?? state.requirement);
    this.armDeadline(session,run,state.budget.remainingMs());
    return { run_id: runId, session_id: session.id, status: 'starting' as const };
  }

  async qualityPhase(runId: string, phase: string, cost = 0): Promise<void> {
    const run = this.runs.get(runId); const workflow = this.workflows.get(runId);
    if (!run || !workflow || isTerminal(run.status)) return;
    workflow.budget.report(cost); run.estimated_cost_usd = (run.estimated_cost_usd ?? 0) + cost;
    run.phase = phase; await this.persistState(); workflow.budget.check();
  }

  private armDeadline(session: SessionRecord, run: RunRecord, ms: number): void {
    const previous = this.deadlines.get(run.run_id); if (previous) clearTimeout(previous);
    const runId = run.run_id;
    const deadline = setTimeout(() => this.background(this.timeoutRun(session, run)), ms);
    deadline.unref();
    this.deadlines.set(runId, deadline);
  }

  private async startExecution(session: SessionRecord, run: RunRecord, state: WorkflowState, prompt: string) {
    state.budget.check();
    run.execution_status = "running"; run.answer = "";
    run.attempts!.push({ attempt: state.attempt, execution_id: state.executionId });
    await this.persistState();
    if (state.controller.signal.aborted || isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
    await this.quality!.execution(state.executionId, run.run_id, run.task_mode === "discuss" ? "discussion" : "implementation", state.attempt, "start");
    session.process.prompt(state.executionId, prompt, state.route, run.task_mode ?? "implement");
  }

  private handleProcessEvent(sessionId: string, event: PublicRunEvent): void {
    const session = this.sessions.get(sessionId);
    if (!session?.activeRunId) return;
    const run = this.runs.get(session.activeRunId);
    if (!run || isTerminal(run.status)) return;
    if (event.type === "usage") {
      const state = this.workflows.get(run.run_id); state?.budget.report(event.cost_usd);
      run.estimated_cost_usd = (run.estimated_cost_usd ?? 0) + event.cost_usd;
      this.persistSoon();
      if (state && state.budget.estimatedUsd > state.budget.limitUsd) this.background(this.failRun(session, run, { code: "workflow_budget_exhausted", message: "Model budget exhausted" }));
      return;
    }
    if (event.type === "turn") {
      run.turn_count += 1;
      this.persistSoon();
      return;
    }
    if (event.type === "policy_decision") {
      // Counted, not stored: one run can make hundreds of decisions.
      const { type: _type, ...decision } = event;
      run.policy_summary = summarizePolicy(run.policy_summary, decision);
      try { this.observePolicy?.(run.run_id, run.session_id, decision); } catch { /* monitoring cannot fail agent runs */ }
      this.persistSoon();
      return;
    }
    if (event.type === "approval") {
      const state = this.workflows.get(run.run_id);
      if (state) { state.budget.pause(); const timer = this.deadlines.get(run.run_id); if (timer) clearTimeout(timer); this.armDeadline(session, run, 300_000); }
      const approval = this.approvals.create({
        runId: run.run_id,
        toolCallId: event.approval_id,
        userId: run.owner_user_id,
        action: `${event.title}\n${event.detail}`,
      });
      this.countApproval(run, approval.id, "requested");
      this.updateRun(run, "awaiting_approval");
      run.events.push({ ...event, approval_id: approval.id, expires_at: new Date(approval.expiresAt).toISOString() });
      this.persistInBackground();
      return;
    }
    if (event.type === "checks_request") {
      this.background(this.answerChecks(session, run, event));
      return;
    }
    if (event.type === "budget_request") {
      // Only dollar figures cross the dialog; runs without a workflow budget have none to give.
      const budget = this.workflows.get(run.run_id)?.budget;
      session.process.respondUi(event.request_id, { value: JSON.stringify(budget ? { used_usd: budget.estimatedUsd, limit_usd: budget.limitUsd } : { unavailable: true }) });
      return;
    }
    if (event.type === "tool") {
      if (event.phase === "start") run.tool_count += 1;
      if (event.phase === "start" && (event.tool === "write" || event.tool === "edit")) run.first_edit_at ??= new Date().toISOString();
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
      this.persistSoon();
      return;
    }
    if (event.type === "answer_delta") {
      const text = (this.streamingAnswers.get(run.run_id) ?? "") + event.text;
      this.streamingAnswers.set(run.run_id, text);
      run.answer = text;
      this.persistSoon();
      return;
    }
    if (event.type === "answer") {
      this.streamingAnswers.delete(run.run_id);
      run.answer = event.text;
      run.events.push(event);
      this.persistSoon();
      return;
    }
    if (event.type === "error") {
      this.background(this.failRun(session, run, { code: event.code, message: event.message }));
      return;
    }
    if (event.type === "settled") {
      run.execution_outcome = event.outcome;
      if (event.outcome !== "completed") {
        if (this.workflows.has(run.run_id)) run.execution_status = "failed";
        this.background(this.failRun(session, run, settleFailure(event)));
        return;
      }
      if (this.workflows.has(run.run_id)) run.execution_status = "succeeded";
      this.background(this.completeRun(session, run));
    }
  }

  /** Runs the writer's run_checks request within its per-run limits and answers the dialog. */
  private async answerChecks(session: SessionRecord, run: RunRecord, event: Extract<PublicRunEvent, { type: "checks_request" }>): Promise<void> {
    const respond = (response: WriterChecksResponse) => session.process.respondUi(event.request_id, { value: JSON.stringify(response) });
    const workflow = this.workflows.get(run.run_id);
    const writer = this.quality?.writerChecks;
    if (run.task_mode === "discuss" || !workflow || !this.quality?.runChecks || !writer) { respond(rejected("unavailable", "run_checks needs a trusted verification profile and is not available in this run")); return; }
    if (workflow.checksOperation) { respond(rejected("busy", "A run_checks call is already running; wait for its result")); return; }
    const calls = workflow.checksCalls ?? 0;
    if (calls >= writer.maxCalls) { respond(rejected("max_calls", `run_checks may be called at most ${writer.maxCalls} times per run`)); return; }
    const names = event.checks?.length ? event.checks : writer.defaultChecks;
    const windowMs = workflow.budget.remainingMs() - writer.reserveMs;
    // Enough room for the requested checks, or at least a minute to run them clamped to the window.
    if (windowMs < Math.min(MIN_CHECKS_WINDOW_MS, writer.estimateMs(names))) { respond(rejected("insufficient_time_budget", "Not enough time left to run checks and still verify the final result; finish without them")); return; }
    workflow.checksCalls = calls + 1; run.writer_checks_calls = workflow.checksCalls;
    run.events.push({ type: "checks", phase: "start", checks: names });
    this.persistSoon();
    const operation = this.quality.runChecks({ run_id: run.run_id, worktree: session.worktreePath, baseline: session.baselineCommit, ...(event.checks ? { checks: event.checks } : {}), signal: workflow.controller.signal, deadlineMs: windowMs });
    workflow.checksOperation = operation;
    try {
      const response = await operation;
      if (workflow.controller.signal.aborted || isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
      run.events.push({ type: "checks", phase: "end", status: response.status });
      respond(response);
    } catch {
      if (workflow.controller.signal.aborted || isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
      run.events.push({ type: "checks", phase: "end", status: "unknown" });
      respond({ status: "unknown", reason: "run_checks_failed", checks: [], report: "The checks could not run. Continue; the final verification still runs." });
    } finally {
      if (workflow.checksOperation === operation) workflow.checksOperation = undefined;
      this.persistSoon();
    }
  }

  private async completeDiscussion(session: SessionRecord, run: RunRecord): Promise<void> {
    const workflow = this.workflows.get(run.run_id);
    if (workflow?.checking || isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
    if (workflow) workflow.checking = true;
    try {
      if (workflow) {
        await this.quality!.execution(workflow.executionId, run.run_id, "discussion", workflow.attempt, "end", session.process.queueMs);
        workflow.budget.check();
      }
      const diff = await this.worktrees.inspectDiff(session.worktreePath, DEFAULT_DIFF_LIMITS);
      if (isTerminal(run.status) || this.terminalClaims.has(run.run_id) || workflow?.controller.signal.aborted) return;
      if (diff.files.length || diff.bytes) {
        await this.failRun(session, run, { code: "discussion_workspace_changed", message: "Read-only discussion produced unexpected changes; delivery blocked" }); return;
      }
      if (!this.claimTerminal(run)) return;
      run.changed_files = []; run.phase = "completed";
      run.verification_status = "not_applicable"; run.grader_status = "not_applicable"; run.publication_status = "not_applicable";
      this.updateRun(run, "completed"); this.finishActiveRun(session, run.run_id);
      await this.persistState();
    } catch {
      if (!isTerminal(run.status) && !this.terminalClaims.has(run.run_id)) await this.failRun(session, run, { code: "discussion_completion_failed", message: "Read-only completion checks could not finish" });
    }
  }

  private async completeRun(session: SessionRecord, run: RunRecord): Promise<void> {
    if (run.task_mode === "discuss") { await this.completeDiscussion(session, run); return; }
    const workflow = this.workflows.get(run.run_id);
    if (workflow && this.quality) {
      if (workflow.checking || this.terminalClaims.has(run.run_id) || isTerminal(run.status)) return;
      workflow.checking = true;
      const input: GateInput = { run_id: run.run_id, worktree: session.worktreePath, baseline: session.baselineCommit, requirement: workflow.requirement, attempt: workflow.attempt, signal: workflow.controller.signal };
      try {
        await this.quality.execution(workflow.executionId, run.run_id, "implementation", workflow.attempt, "end", session.process.queueMs);
        workflow.budget.check();
        try { await this.worktrees.inspectDiff(session.worktreePath, DEFAULT_DIFF_LIMITS); }
        catch (error) {
          if (!(error instanceof DiffLimitError) || workflow.attempt > (this.quality.maxRepairs ?? 0)) throw error;
          workflow.attempt++; workflow.executionId = randomUUID(); workflow.checking = false;
          run.phase = "repairing"; await this.persistState();
          await this.startExecution(session, run, workflow, renderPrompt("diff-limit-repair", { requirement: workflow.requirement, limit: error.message.slice(0, 200) })); return;
        }
        run.phase = "verifying"; await this.persistState();
        workflow.operation = this.quality.evaluate(input);
        const result = await workflow.operation; workflow.operation = undefined;
        if (workflow.controller.signal.aborted || isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
        workflow.budget.check();
        run.grader_findings = result.grade?.findings;
        run.verification_status = result.verification.status; run.grader_status = result.grade?.status ?? "pending";
        Object.assign(run.attempts!.at(-1)!, { candidate_hash: result.candidate.hash, verification_status: run.verification_status, grader_status: run.grader_status, artifact: result.candidate.artifact,
          ...(result.candidate.fixes?.length ? { fixes: result.candidate.fixes } : {}), ...(result.candidate.fixer_cause ? { fixer_cause: result.candidate.fixer_cause } : {}) });
        if (result.action === "repair") {
          workflow.attempt++; workflow.executionId = randomUUID(); workflow.checking = false;
          run.phase = "repairing"; await this.persistState();
          await this.startExecution(session, run, workflow, result.repair_prompt!); return;
        }
        if (result.action === "fail") {
          if (result.failure?.code === "scope_violation") {
            await this.failRun(session, run, { code: "scope_violation", message: `Changes outside the allowed paths were not delivered: ${result.failure.files.slice(0, 10).join(", ")}` }); return;
          }
          if (result.grade?.cause) run.grader_cause = result.grade.cause;
          const cause = result.failure?.code === "verification_failed" ? result.failure.cause_code : undefined;
          await this.failRun(session, run, { code: "verification_failed", message: "Candidate retained as an isolated artifact; verification/grading did not pass", ...(cause ? { cause_code: cause } : {}) }); return;
        }
        await this.quality.assertUnchanged(result.candidate, input);
        workflow.controller.signal.throwIfAborted();
        run.phase = "finalizing"; await this.persistState();
      } catch (error) {
        workflow.operation = undefined;
        if (!isTerminal(run.status) && !this.terminalClaims.has(run.run_id)) {
          await this.failRun(session, run, classifyGateError(error));
        }
        return;
      }
    }
    if (this.deliveryApproval && !session.skipDelivery) {
      await this.requestDelivery(session, run);
      return;
    }
    // Claim synchronously after settle so cancellation cannot race finalization.
    if (!this.claimTerminal(run)) return;
    await this.finalizeClaimedRun(session, run);
  }

  /** Commits the worktree for a run whose terminal claim the caller already holds. */
  private async finalizeClaimedRun(session: SessionRecord, run: RunRecord): Promise<void> {
    const workflow = this.workflows.get(run.run_id);
    try {
      let diff: Awaited<ReturnType<Worktrees["inspectDiff"]>>;
      try { diff = await this.worktrees.inspectDiff(session.worktreePath, DEFAULT_DIFF_LIMITS); }
      catch (error) {
        if (!(error instanceof DiffLimitError)) throw error;
        await this.failClaimedRun(session, run, classifyGateError(error));
        return;
      }
      this.notifyLifecycle(run.run_id, "commit", "start");
      const commit = await this.worktrees.commit(session.worktreePath, buildCommitMessage(run));
      this.notifyLifecycle(run.run_id, "commit", "end");
      session.baselineCommit = commit;
      run.changed_files = diff.files;
      run.diff_stat = diff.stat;
      run.commit = commit;
      run.cherry_pick = `git cherry-pick ${commit}`;
      run.events.push({ type: "diff", files: diff.files, stat: diff.stat, bytes: diff.bytes });
      this.updateRun(run, "completed");
      if (workflow) { run.execution_status = "succeeded"; run.phase = "completed"; }
      this.finishActiveRun(session, run.run_id);
    } catch {
      this.notifyLifecycle(run.run_id, "commit", "end", true);
      run.error = { code: "commit_failed", message: "Changes remain in the isolated worktree and need attention" };
      this.updateRun(run, "failed");
      this.finishActiveRun(session, run.run_id);
    }
    await this.persistState();
  }

  /** Pauses a settled run until its owner approves the exact diff; an empty diff completes directly. */
  private async requestDelivery(session: SessionRecord, run: RunRecord): Promise<void> {
    if (isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
    let diff: Awaited<ReturnType<Worktrees["inspectDiff"]>>;
    let digest: string;
    let patch: string;
    try {
      diff = await this.worktrees.inspectDiff(session.worktreePath, DEFAULT_DIFF_LIMITS);
      ({ digest, patch } = await this.deliveryCandidate(session, diff));
    } catch (error) {
      if (!isTerminal(run.status) && !this.terminalClaims.has(run.run_id)) await this.failRun(session, run, classifyGateError(error));
      return;
    }
    if (isTerminal(run.status) || this.terminalClaims.has(run.run_id)) return;
    if (!diff.files.length && !diff.bytes) {
      if (!this.claimTerminal(run)) return;
      await this.finalizeClaimedRun(session, run);
      return;
    }
    const ttlMs = this.deliveryApproval!.ttlMs;
    const approval = this.approvals.create({ runId: run.run_id, toolCallId: "delivery", userId: run.owner_user_id, action: `Deliver ${diff.files.length} changed files` }, ttlMs);
    this.countApproval(run, approval.id, "requested");
    this.deliveries.set(run.run_id, { approvalId: approval.id, digest });
    this.workflows.get(run.run_id)?.budget.pause();
    const previous = this.deadlines.get(run.run_id);
    if (previous) clearTimeout(previous);
    const expiry = setTimeout(() => this.background(this.expireDelivery(session, run, approval.id)), ttlMs);
    expiry.unref();
    this.deadlines.set(run.run_id, expiry);
    run.phase = "awaiting_delivery";
    run.changed_files = diff.files;
    run.diff_stat = diff.stat;
    this.updateRun(run, "awaiting_approval");
    run.events.push({
      type: "approval",
      kind: "delivery",
      approval_id: approval.id,
      title: `Deliver ${diff.files.length} changed file${diff.files.length === 1 ? "" : "s"}?`,
      detail: diff.stat,
      expires_at: new Date(approval.expiresAt).toISOString(),
      delivery: {
        files: diff.files,
        stat: diff.stat,
        verification_status: run.verification_status ?? "unverified",
        grader_status: run.grader_status ?? "not_run",
        excerpt: truncateUtf8(String(redact(patch, this.deliveryApproval!.secrets?.() ?? [])), DELIVERY_EXCERPT_BYTES),
        delivery_digest: digest,
      },
    });
    await this.persistState();
  }

  private async deliveryCandidate(session: SessionRecord, diff: Awaited<ReturnType<Worktrees["inspectDiff"]>>): Promise<{ digest: string; patch: string }> {
    const patch = await this.worktrees.patch!(session.worktreePath);
    const digest = createHash("sha256").update(JSON.stringify(diff.files)).update("\0").update(patch).digest("hex");
    return { digest: `sha256:${digest}`, patch };
  }

  private async decideDelivery(session: SessionRecord, run: RunRecord, approvalId: string, decision: "approve" | "reject"): Promise<void> {
    const pending = this.deliveries.get(run.run_id)!;
    if (!this.claimTerminal(run)) throw new RuntimeConflictError("Run is already finalizing");
    this.deliveries.delete(run.run_id);
    run.events.push({ type: "approval_decision", decision: decision === "approve" ? "approved" : "rejected", approval_id: approvalId });
    if (decision === "reject") {
      await this.failClaimedRun(session, run, { code: "delivery_rejected", message: "The owner discarded the changes; the patch was kept" });
      return;
    }
    let current: string;
    try {
      current = (await this.deliveryCandidate(session, await this.worktrees.inspectDiff(session.worktreePath, DEFAULT_DIFF_LIMITS))).digest;
    } catch (error) {
      await this.failClaimedRun(session, run, classifyGateError(error));
      return;
    }
    if (current !== pending.digest) {
      await this.failClaimedRun(session, run, { code: "delivery_candidate_changed", message: "The worktree changed after the delivery card was posted; nothing was committed" });
      return;
    }
    run.phase = "finalizing";
    await this.finalizeClaimedRun(session, run);
  }

  private countApproval(run: RunRecord, approvalId: string, outcome: "requested" | "approved" | "rejected" | "expired"): void {
    const counts = run.approvals ??= { requested: 0, approved: 0, rejected: 0, expired: 0, wait_ms: 0 };
    counts[outcome] += 1;
    if (outcome === "requested") { this.approvalOpened.set(approvalId, Date.now()); return; }
    const opened = this.approvalOpened.get(approvalId);
    this.approvalOpened.delete(approvalId);
    if (opened !== undefined) counts.wait_ms += Math.max(0, Date.now() - opened);
  }

  private async expireDelivery(session: SessionRecord, run: RunRecord, approvalId: string): Promise<void> {
    if (this.deliveries.get(run.run_id)?.approvalId !== approvalId || !this.claimTerminal(run)) return;
    this.deliveries.delete(run.run_id);
    this.countApproval(run, approvalId, "expired");
    await this.failClaimedRun(session, run, { code: "delivery_expired", message: "Nobody delivered the changes before the approval expired; the patch was kept" });
  }

  private async failRun(session: SessionRecord, run: RunRecord, failure: RunFailure): Promise<void> {
    if (!this.claimTerminal(run)) return;
    await this.failClaimedRun(session, run, failure);
  }

  private async failClaimedRun(session: SessionRecord, run: RunRecord, failure: RunFailure): Promise<void> {
    if (!await this.stopBeforeRollback(session, run)) {
      await this.persistState();
      return;
    }
    run.error = { ...failure };
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
      // Keep why the run failed; the rollback failure is a flag on it, not a replacement.
      run.error = run.error ? { ...run.error, rollback_failed: true } : { code: "rollback_failed", message: "Isolated worktree needs attention" };
    }
  }

  /** Stops every writer and keeps the undelivered change; false means the worktree must not be rolled back. */
  private async stopBeforeRollback(session: SessionRecord, run: RunRecord): Promise<boolean> {
    const workflow = this.workflows.get(run.run_id);
    try {
      workflow?.controller.abort();
      if (workflow?.discovery) await workflow.discovery.catch(() => {});
      if (workflow?.operation) await workflow.operation.catch(() => {});
      if (workflow?.checksOperation) await workflow.checksOperation.catch(() => {});
      await session.process.abort(workflow?.executionId ?? run.run_id);
      await this.quality?.stop?.(run.run_id);
    } catch {
      return this.refuseRollback(session, run, {
        code: "container_stop_failed",
        message: "Changes remain isolated because the Agent container could not be confirmed stopped",
      });
    }
    try {
      if (this.quality?.archive) await this.quality.archive({ run_id: run.run_id, worktree: session.worktreePath, baseline: session.baselineCommit, requirement: workflow?.requirement ?? "", attempt: workflow?.attempt ?? 0, signal: new AbortController().signal });
      await this.retainPatch(run, session.worktreePath, session.baselineCommit);
    } catch {
      return this.refuseRollback(session, run, {
        code: "archive_failed",
        message: "Changes remain in the isolated worktree because their patch could not be archived",
      });
    }
    return true;
  }

  private async retainPatch(run: RunRecord, worktree: string, baseline: string): Promise<void> {
    const archived = await this.archivePatch?.({ run_id: run.run_id, worktree, baseline });
    if (archived) run.patch_artifact = archived;
  }

  private refuseRollback(session: SessionRecord, run: RunRecord, failure: RunFailure): false {
    run.error = failure;
    this.updateRun(run, "failed");
    this.finishActiveRun(session, run.run_id);
    return false;
  }

  private finishActiveRun(session: SessionRecord, runId: string): void {
    const timer = this.deadlines.get(runId);
    if (timer) clearTimeout(timer);
    this.deadlines.delete(runId);
    if (this.workflows.has(runId)) this.quality?.close?.(runId);
    this.workflows.delete(runId);
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
    if (run.execution_status && isTerminal(status)) {
      if (run.execution_status === "running") run.execution_status = status === "completed" ? "succeeded" : "failed";
      if (status !== "completed") run.phase = status;
      if (run.verification_status === "pending") run.verification_status = "unknown";
      if (run.grader_status === "pending") run.grader_status = "unknown";
    }
    run.updated_at = new Date().toISOString();
    run.events.push({ type: "status", status });
    // Recomputed in notifyRun once the final timing is known.
    if (isTerminal(status)) run.harness_metrics = computeHarnessMetrics(run);
  }

  private requireSession(sessionId: string): SessionRecord {
    const session = this.sessions.get(sessionId);
    if (!session || session.closed) throw new RuntimeNotFoundError("Session not found");
    return session;
  }

  async shutdown(): Promise<void> {
    for (const session of this.sessions.values()) {
      const run = session.activeRunId ? this.runs.get(session.activeRunId) : undefined;
      const state = run ? this.workflows.get(run.run_id) : undefined;
      if (run?.workflow_mode === 'structured' && state && this.quality?.canResume?.(run.run_id) && this.claimTerminal(run)) {
        try {
          state.controller.abort(); await state.discovery?.catch(() => {}); await this.quality.stopDiscovery?.(run.run_id);
          this.updateRun(run,'interrupted'); run.interrupted_at = new Date().toISOString();
          await this.persistState();
          const timer = this.deadlines.get(run.run_id); if (timer) clearTimeout(timer);
          this.deadlines.delete(run.run_id); this.workflows.delete(run.run_id); session.activeRunId = undefined;
          await this.persistState();
        } catch { run.error = { code: 'shutdown_stop_unconfirmed',message: 'Read-only workers could not be confirmed stopped; evidence retained' }; this.updateRun(run,'failed'); await this.persistState(); }
      } else if (session.activeRunId && !this.deliveries.has(session.activeRunId)) { try { await this.cancel(session.activeRunId, session.ownerUserId); } catch { /* retain worktree if stop fails */ } }
      try { await session.process.close(); } catch { /* stop failure already audited */ }
    }
  }

  private notifyLifecycle(runId: string, name: string, phase: "start" | "end", failed = false, sessionId?: string): void {
    try { this.observeLifecycle?.(runId, name, phase, failed, sessionId); } catch { /* monitoring must not fail work */ }
  }

  private notifyRun(run: RunRecord): void {
    try {
      run.traceparent ??= this.captureTraceContext?.(run.run_id);
      if (run.workflow_mode === 'structured') {
        const nodes = this.quality?.tasks?.(run.run_id) ?? [];
        run.workflow_summary = { revision: Math.max(0,...nodes.map(node => node.wave)), total: nodes.length, completed: nodes.filter(node => node.state === 'succeeded').length, unresolved: nodes.filter(node => node.state === 'unresolved').length, agent_work_ms: nodes.reduce((sum,node) => sum + (node.agent_work_ms ?? node.duration_ms ?? 0),0), queue_ms: nodes.reduce((sum,node) => sum + (node.queue_ms ?? 0),0) };
      }
      const workflow = this.workflows.get(run.run_id);
      if (workflow) {
        run.budget = workflow.budget.snapshot();
        const wall = Date.now() - Date.parse(run.created_at);
        // Reproduction consumes the shared budget but predates this execution's wall clock.
        const active = workflow.budget.activeMs() + workflow.initialActiveMs - workflow.priorActiveMs;
        run.timing = { wall_ms: wall, active_ms: active, human_wait_ms: Math.max(0, wall - active - (run.suspended_ms ?? 0)) };
      }
      if (!["running", "awaiting_approval"].includes(run.status) && run.capture_complete === undefined) {
        const capture = this.captureState?.(run.run_id);
        run.capture_complete = capture?.complete ?? null;
        run.capture_reasons = capture?.reasons ?? [];
      }
      if (isTerminal(run.status)) run.harness_metrics = computeHarnessMetrics(run);
      this.observeRun?.(structuredClone(run)); } catch { /* monitoring cannot fail agent runs */ }
  }

  /** Event handlers cannot await persistence; a failure must fail closed rather than become an unhandled rejection. */
  private persistInBackground(): void {
    this.background(this.persistState());
  }

  /** Fire-and-forget control work: persistence failures mark the runtime unhealthy instead of crashing Node. */
  private background(work: Promise<unknown>): void {
    work.catch((error: unknown) => {
      if (this.stateStore) this.stateStore.healthy = false;
      console.error("Agent runtime control state could not be persisted; runtime marked unhealthy", error instanceof Error ? error.message : error);
    });
  }

  /** Observational updates (tokens, tools, turns, usage) coalesce into at most one write per second. */
  private persistSoon(): void {
    if (this.persistTimer) return;
    this.persistTimer = setTimeout(() => { this.persistTimer = undefined; this.persistInBackground(); }, 1_000);
    this.persistTimer.unref?.();
  }

  private async persistState(): Promise<void> {
    if (this.persistTimer) { clearTimeout(this.persistTimer); this.persistTimer = undefined; }
    for (const run of this.runs.values()) {
      if (isTerminal(run.status) && this.observedFinal.has(run.run_id) && this.observedFinal.get(run.run_id) === run.updated_at) continue;
      this.notifyRun(run);
      if (isTerminal(run.status)) { this.observedFinal.set(run.run_id, run.updated_at); this.streamingAnswers.delete(run.run_id); }
      else this.observedFinal.delete(run.run_id);
    }
    if (!this.stateStore) return;
    try { await this.stateStore.save({
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
        ...(session.skipDelivery ? { skipDelivery: true } : {}),
        closed: session.closed,
        lastActivity: session.lastActivity,
      })),
      runs: [...this.runs.values()].map((run) => structuredClone(run)),
    });
    } catch (error) {
      // Retain isolated changes and stop all work if execution state cannot be durable.
      for (const workflow of this.workflows.values()) workflow.controller.abort();
      await Promise.allSettled([...this.sessions.values()].filter(session => session.activeRunId).map(async session => {
        const runId = session.activeRunId!;
        await session.process.abort(this.workflows.get(runId)?.executionId ?? runId);
        await this.quality?.stop?.(runId);
      }));
      throw error;
    }
  }
}

/** Classify a quality-gate exception so users see what actually blocked delivery. */
/** Trailers let reviewers trace a delivered commit back to its run and how (or whether) it was verified. */
export function buildCommitMessage(run: RunRecord): string {
  const trailers = [`Pi-Run-Id: ${run.run_id}`, `Pi-Model: ${run.provider}/${run.model}`, `Pi-Verification: ${run.verification_status ?? "unverified"}`];
  if (run.grader_status) trailers.push(`Pi-Grader: ${run.grader_status}`);
  return `Pi agent: ${run.run_id}\n\n${trailers.join("\n")}`;
}

/** Non-completed executions are never delivered; each outcome gets its own actionable code. */
function settleFailure(event: Extract<PublicRunEvent, { type: "settled" }>): RunFailure {
  if (event.outcome === "limit_reached") {
    return event.limit === "turns"
      ? { code: "agent_turn_limit", message: "Agent stopped at its turn limit before finishing; changes were not delivered" }
      : { code: "agent_tool_limit", message: "Agent stopped at its tool-call limit before finishing; changes were not delivered" };
  }
  if (event.outcome === "truncated") return { code: "agent_output_truncated", message: "Model output hit its token limit; changes were not delivered" };
  if (event.outcome === "aborted") return { code: "agent_aborted", message: "Agent execution was aborted before finishing; changes were not delivered" };
  return { code: "provider_error", message: event.error_message ? `Model provider error: ${event.error_message}` : "Model provider returned an error; changes were not delivered" };
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

export interface DeliveryApprovalOptions {
  /** How long the owner has to deliver before the changes are discarded (patch kept). */
  ttlMs: number;
  /** Extra secret values to mask in the diff excerpt shown on the delivery card. */
  secrets?: () => string[];
}

const DELIVERY_EXCERPT_BYTES = 1_800;

function truncateUtf8(text: string, maxBytes: number): string {
  if (Buffer.byteLength(text) <= maxBytes) return text;
  const marker = "\n… (truncated)";
  let cut = Buffer.from(text).subarray(0, maxBytes - Buffer.byteLength(marker)).toString("utf8");
  // A split multi-byte character decodes to U+FFFD; drop it.
  cut = cut.replace(/\uFFFD$/, "");
  return `${cut}${marker}`;
}
