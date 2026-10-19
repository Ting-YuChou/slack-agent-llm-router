import { randomUUID } from "node:crypto";
import { ControlStore } from "./ci-control.js";
import { RuntimeError } from "./orchestrator.js";
import { workflowMode, type WorkflowMode } from "./structured-workflow.js";
export interface RepairTarget {
  repository: string; pr: number; head: string; tested: string; branch: string;
  failed_jobs?: string[];
  workflow: string; run: number; attempt: number; diagnostic: string;
}
interface ExecutionSummary {
  workflow_summary?: { revision: number; total: number; completed: number; agent_work_ms: number; queue_ms: number }; suspended_ms?: number;
  status: string; phase?: string; commit?: string; branch?: string;
  attempts?: Array<{ candidate_hash?: string }>;
  events?: Array<Record<string, unknown>>; answer?: string;
  estimated_cost_usd?: number; timing?: { wall_ms: number; active_ms: number; human_wait_ms: number };
  grader_findings?: Array<{path: string; line: number; reason: string}>;
  verification_status?: string; grader_status?: string; execution_status?: string;
}
interface RepairJob {
  workflow_mode?: WorkflowMode;
  workflow_summary?: ExecutionSummary["workflow_summary"]; suspended_ms?: number;
  run_id: string; session_id: string; owner_user_id: string; status: string; phase: string;
  approval_expires_at?: number;
  approval_id?: string; approval_kind?: "modify" | "publish"; target: RepairTarget;
  child?: string; candidate_hash?: string; commit?: string; branch?: string;
  verification_status?: string; grader_status?: string; publication_status: string; url?: string; error?: string;
  events?: Array<Record<string, unknown>>; answer?: string; estimated_cost_usd?: number;
  grader_findings?: ExecutionSummary["grader_findings"];
  attempts?: ExecutionSummary["attempts"]; timing?: ExecutionSummary["timing"]; execution_status?: string;
  approval_wait_ms?: number; approval_wait_started_at?: number; execution_wait_ms?: number;
  created_at: string; updated_at: string;
}
interface RepairOptions {
  store: ControlStore; repository: string; owner: { team_id: string; channel_id: string; user_id: string };
  investigate(input: string): Promise<RepairTarget>;
  reproduce(target: RepairTarget, signal: AbortSignal, id: string): Promise<"passed" | "failed" | "unknown">;
  execute(target: RepairTarget, id: string, mode?: WorkflowMode): Promise<{ run_id: string; session_id: string }>;
  validateMode?(mode: WorkflowMode): void;
  resumeExecution?(id: string, owner: string): Promise<{ run_id: string; session_id: string; status: 'starting' }>;
  result(id: string): ExecutionSummary;
  decideExecution?(id: string, approval: string, owner: string, decision: "approve" | "reject"): Promise<void>;
  publish(target: RepairTarget, commit: string, branch: string, id: string): Promise<string>;
  cancel?(id: string, owner: string): Promise<void>;
  observe?(job: RepairJob): void;
  record?(id: string, payload: unknown): Promise<void>;
}
/** Durable transitions are written before any execution or external publication. */
export class CiRepairs {
  private readonly busy = new Set<string>();
  private readonly controllers = new Map<string, AbortController>();
  constructor(private readonly options: RepairOptions) {}
  get healthy() { return this.options.store.healthy; }
  checkContext(team: unknown, channel: unknown, owner: unknown) {
    const configured = this.options.owner;
    if (team !== configured.team_id || channel !== configured.channel_id || owner !== configured.user_id) throw new RuntimeError("CI repair must use configured Slack team, channel and owner", "repair_forbidden", 401);
  }
  async notifications() {
    const jobs = this.options.store.all<RepairJob>("repair");
    const result = [];
    for (const job of jobs) {
      const current = await this.get(job.run_id);
      if (current.status === "awaiting_approval" || current.status === "running") result.push({ ...current, team_id: this.options.owner.team_id, channel_id: this.options.owner.channel_id });
    }
    return result;
  }
  pruneDiagnostics(now = Date.now()) {
    for (const job of this.options.store.all<RepairJob>("repair")) {
      if (["running", "awaiting_approval"].includes(job.status) || Date.parse(job.updated_at) >= now - 30 * 86400000 || job.target.diagnostic === "[expired]") continue;
      job.target.diagnostic = "[expired]"; this.options.store.put("repair", job.run_id, job);
    }
  }
  has(id: string) { return Boolean(this.options.store.get("repair", id)); }
  private save(job: RepairJob) {
    const now = Date.now();
    if (job.status === "awaiting_approval") job.approval_wait_started_at ??= now;
    else if (job.approval_wait_started_at !== undefined) {
      job.approval_wait_ms = (job.approval_wait_ms ?? 0) + Math.max(0, now - job.approval_wait_started_at);
      delete job.approval_wait_started_at;
    }
    this.updateTiming(job, now);
    job.events = job.approval_id ? [{ type: "approval", kind: job.approval_kind, approval_id: job.approval_id, title: job.approval_kind === "publish" ? "Publish repair Draft PR" : "Approve CI reproduction and modification", detail: `Repository: ${job.target.repository}, branch: ${job.target.branch}, head: ${job.target.head}. Verification: ${job.verification_status ?? "pending"}; grader: ${job.grader_status ?? "pending"}` }] : [];
    job.answer = job.url ? `Repair draft PR: ${job.url}` : job.error ?? `CI repair: ${job.phase}`;
    job.updated_at = new Date().toISOString(); this.options.store.put("repair", job.run_id, job); try { this.options.observe?.(job); } catch { /* monitoring never changes authorization */ } }
  async create(input: string, user: string, receipt?: string, verified?: RepairTarget, requestedMode?: WorkflowMode): Promise<RepairJob> {
    let mode: WorkflowMode; try { mode = workflowMode(requestedMode); } catch { throw new RuntimeError("Invalid workflow_mode","invalid_request",400); }
    this.options.validateMode?.(mode);
    if (user !== this.options.owner.user_id) throw new RuntimeError("Only configured repository owner can request repair", "repair_forbidden", 401);
    if (receipt && this.has(receipt)) return this.require(receipt);
    const target = verified ?? await this.options.investigate(input);
    if (target.repository !== this.options.repository || target.branch.startsWith("pi-agent/")) throw new RuntimeError("Repository or repair branch is not allowed", "repair_not_allowed", 400);
    const id = receipt ?? randomUUID(); const now = new Date().toISOString();
    const job: RepairJob = { workflow_mode: mode, run_id: id, session_id: id, owner_user_id: user, status: "awaiting_approval", phase: "awaiting_modification", target, publication_status: "pending", created_at: now, updated_at: now };
    job.approval_expires_at = Date.now() + 300_000; job.approval_kind = "modify"; job.approval_id = this.options.store.approval(id, user, "modify", target.head, this.targetKey(target)); this.save(job); return job;
  }
  async get(id: string): Promise<RepairJob> {
    const job = this.require(id);
    if (["running", "awaiting_approval"].includes(job.status)) this.updateTiming(job, Date.now());
    if (job.status === "awaiting_approval" && (job.approval_expires_at ?? 0) <= Date.now()) {
      job.status = job.phase === "awaiting_publication" ? "completed" : "timed_out";
      job.publication_status = job.phase === "awaiting_publication" ? "expired" : job.publication_status;
      job.phase = "finished"; delete job.approval_id; delete job.approval_kind; this.save(job);
    }
    if (job.child && job.phase === "executing") {
      const result = this.options.result(job.child);
      job.workflow_summary = result.workflow_summary; job.suspended_ms = result.suspended_ms;
      job.grader_findings = result.grader_findings; job.attempts = result.attempts; job.estimated_cost_usd = result.estimated_cost_usd; job.execution_status = result.execution_status;
      job.execution_wait_ms = result.timing?.human_wait_ms ?? job.execution_wait_ms;
      this.updateTiming(job, Date.now());
      if (["running", "awaiting_approval"].includes(result.status)) return { ...job, status: result.status, phase: result.phase ?? job.phase, events: result.events, answer: result.answer };
      if (["completed", "failed", "cancelled", "timed_out", "rejected", "interrupted"].includes(result.status)) {
        job.verification_status = result.verification_status; job.grader_status = result.grader_status;
        if (result.status === "completed" && result.commit && result.commit !== job.target.head && result.branch && result.verification_status === "passed" && result.grader_status === "passed") {
          job.commit = result.commit; job.branch = result.branch; job.candidate_hash = result.attempts?.at(-1)?.candidate_hash;
          if (!job.candidate_hash) { job.status = "failed"; job.phase = "failed"; job.error = "missing_candidate_identity"; }
          else { job.status = "awaiting_approval"; job.phase = "awaiting_publication"; job.approval_expires_at = Date.now() + 300_000; job.approval_kind = "publish"; job.approval_id = this.options.store.approval(id, job.owner_user_id, "publish", job.candidate_hash, this.targetKey(job.target)); }
        } else { job.status = result.status === "completed" ? "failed" : result.status; job.phase = "finished"; if (result.status === "completed") job.error = "verified_repair_evidence_missing_or_no_changes"; }
        this.save(job);
      }
    }
    return job;
  }
  async decide(id: string, approval: string, user: string, decision: "approve" | "reject", background = false) {
    const job = await this.get(id);
    const internal = this.require(id);
    if (internal.child && internal.phase === "executing" && job.status === "awaiting_approval") {
      if (!this.options.decideExecution) throw new RuntimeError("Execution approval unavailable", "approval_conflict", 409);
      await this.options.decideExecution(internal.child, approval, user, decision); return;
    }
    if (this.busy.has(id)) throw new RuntimeError("Repair transition is in progress", "repair_busy", 409);
    const kind = job.approval_kind;
    if (!kind || job.status !== "awaiting_approval" || approval !== job.approval_id) throw new RuntimeError("Approval is no longer valid", "approval_conflict", 409);
    if (user !== job.owner_user_id) throw new RuntimeError("Only owner can decide", "approval_forbidden", 401);
    try { this.options.store.consume(approval, id, user, kind, kind === "modify" ? job.target.head : job.candidate_hash!, this.targetKey(job.target)); }
    catch { throw new RuntimeError("Approval expired or consumed", "approval_conflict", 409); }
    this.busy.add(id); delete job.approval_id; delete job.approval_kind;
    job.status = "running"; job.phase = `authorized_${kind}`; this.save(job);
    const perform = async () => {
    try {
      if (decision === "reject") {
        job.status = kind === "modify" ? "rejected" : "completed"; job.phase = "finished"; job.publication_status = "declined"; this.save(job); return;
      }
      const current = await this.options.investigate(String(job.target.run));
      if (this.require(id).status === "cancelled") return;
      if (current.head !== job.target.head || current.tested !== job.target.tested || current.attempt !== job.target.attempt || current.branch !== job.target.branch) {
        job.status = "failed"; job.phase = "finished"; job.publication_status = "stale"; job.error = "PR head or CI identity changed; revalidation required"; this.save(job); return;
      }
      if (kind === "modify") {
        job.status = "running"; job.phase = "reproducing"; this.save(job);
        const controller = new AbortController(); this.controllers.set(id, controller);
        const reproduction = await this.options.reproduce(job.target, controller.signal, id); controller.signal.throwIfAborted();
        if (reproduction !== "failed") { job.status = "failed"; job.phase = "finished"; job.error = reproduction === "passed" ? "ci_failure_not_reproduced" : "reproduction_unknown"; this.save(job); return; }
        job.phase = "starting_writer"; this.save(job);
        const child = await this.options.execute(job.target, id, job.workflow_mode);
        job.child = child.run_id; job.session_id = child.session_id; job.phase = "executing"; this.save(job);
      } else {
        job.status = "running"; job.phase = "publishing"; job.publication_status = "publishing"; this.save(job);
        job.url = await this.options.publish(job.target, job.commit!, job.branch!, id);
        job.publication_status = "published"; job.phase = "finished"; job.status = "completed"; this.save(job);
        await this.options.record?.(id, { status: "published", url: job.url, candidate_hash: job.candidate_hash, commit: job.commit });
      }
    } catch (error) {
      job.status = this.require(id).status === "cancelled" ? "cancelled" : "failed"; job.phase = "finished"; job.error = error instanceof Error ? error.message : "ci_repair_failed";
      if (kind === "publish") job.publication_status = "failed";
      this.save(job); throw error;
    } finally { this.controllers.delete(id); this.busy.delete(id); }
    };
    if (background) { void perform().catch(() => {}); return; }
    await perform();
  }
  executionId(id: string) { return this.require(id).child; }
  async resume(id: string, owner: string) {
    const job = await this.get(id);
    if (job.owner_user_id !== owner) throw new RuntimeError('Only owner can resume','repair_forbidden',401);
    if (!job.child || job.status !== 'interrupted' || job.workflow_mode !== 'structured' || !this.options.resumeExecution) throw new RuntimeError('Repair resume unavailable','repair_conflict',409);
    const current = await this.options.investigate(String(job.target.run));
    if (current.head !== job.target.head || current.tested !== job.target.tested || current.attempt !== job.target.attempt || current.branch !== job.target.branch) throw new RuntimeError('CI identity changed; create a new repair','repair_stale',409);
    const result = await this.options.resumeExecution(job.child,owner);
    job.status = 'running'; job.phase = 'executing'; this.save(job); return result;
  }
  retryPublication(id: string, user: string) {
    const job = this.require(id);
    if (job.owner_user_id !== user || job.publication_status !== "failed" || !job.candidate_hash) throw new RuntimeError("Publication retry unavailable", "publication_conflict", 409);
    job.status = "awaiting_approval"; job.phase = "awaiting_publication"; job.approval_expires_at = Date.now() + 300_000; job.approval_kind = "publish";
    job.approval_id = this.options.store.approval(id, user, "publish", job.candidate_hash, this.targetKey(job.target)); this.save(job); return job;
  }
  async cancel(id: string, owner: string) {
    const job = this.require(id); if (owner !== job.owner_user_id) throw new RuntimeError("Only owner can cancel", "repair_forbidden", 401);
    if (job.phase === "publishing" || job.phase === "starting_writer") throw new RuntimeError("Transition in progress", "repair_busy", 409);
    this.controllers.get(id)?.abort();
    if (job.child) await this.options.cancel?.(job.child, owner);
    job.status = "cancelled"; job.phase = "finished"; delete job.approval_id; this.save(job);
  }
  recover() {
    for (const job of this.options.store.all<RepairJob>("repair")) {
      if (["authorized_modify", "authorized_publish", "reproducing", "starting_writer"].includes(job.phase)) { job.status = "interrupted"; job.phase = "finished"; job.error = "runtime_restarted"; this.save(job); }
      if (job.phase === "publishing") { job.status = "failed"; job.phase = "finished"; job.publication_status = "failed"; this.save(job); }
    }
  }
  private targetKey(target: RepairTarget) { return `${target.repository}:${target.branch}`; }
  private updateTiming(job: RepairJob, now: number) {
    const wall = Math.max(0, now - Date.parse(job.created_at));
    const pending = job.approval_wait_started_at === undefined ? 0 : Math.max(0, now - job.approval_wait_started_at);
    const waiting = Math.min(wall, (job.approval_wait_ms ?? 0) + pending + (job.execution_wait_ms ?? 0));
    job.timing = { wall_ms: wall, active_ms: Math.max(0,wall - waiting - (job.suspended_ms ?? 0)), human_wait_ms: waiting };
  }
  private require(id: string) {
    const job = this.options.store.get<RepairJob>("repair", id);
    if (!job) throw new RuntimeError("Repair not found", "not_found", 404); return job;
  }
}
