import { ScopeViolationError } from "./candidate-scope.js";
import { formatCheckFeedback } from "./check-report.js";
import type { Candidate, CheckStatus, VerificationResult } from "./quality-gate.js";
import type { VerificationCheckSpec, VerificationProfile } from "./verification.js";

export const MAX_WRITER_CHECKS_BYTES = 16 * 1024;
const CHECK_OVERHEAD_MS = 15_000;
const MODE_OVERHEAD_MS = 30_000;
const GRADER_RESERVE_MS = 150_000;
/** A call whose checks do not fit beyond the reserve still runs, clamped, if at least this much is left. */
export const MIN_CHECKS_WINDOW_MS = 60_000;

export interface WriterChecksRequest { run_id: string; worktree: string; baseline: string; checks?: string[]; signal: AbortSignal; deadlineMs?: number }
/** What the writer's run_checks tool receives. Never contains host paths; at most 16 KiB as JSON. */
export interface WriterChecksResponse {
  status: CheckStatus | "scope_violation" | "rejected";
  /** Why the request was refused (`unknown_check`, `busy`, `max_calls`, ...). */
  reason?: string;
  checks: Array<{ name: string; status: CheckStatus }>;
  /** Files outside the allowed paths, for a scope violation. */
  files?: string[];
  report: string;
}

interface Verifier { verify(candidate: Candidate, signal: AbortSignal, modes: Array<"candidate">, checks: VerificationCheckSpec[]): Promise<VerificationResult>; stop(): Promise<void> }

export function rejected(reason: string, report: string): WriterChecksResponse {
  return { status: "rejected", reason, checks: [], report: report.slice(0, 2000) };
}

function rejectedForTime(): WriterChecksResponse {
  return { status: "unknown", reason: "time_budget", checks: [], report: "The checks did not finish in the time left before final verification. Finish without them; the final verification still runs." };
}

/** Bounds the serialized response by shortening the report (JSON escaping can grow it, hence the loop). */
function bounded(response: WriterChecksResponse): WriterChecksResponse {
  let out = response;
  for (let over = Buffer.byteLength(JSON.stringify(out)) - MAX_WRITER_CHECKS_BYTES; over > 0; over = Buffer.byteLength(JSON.stringify(out)) - MAX_WRITER_CHECKS_BYTES) {
    const bytes = Buffer.from(out.report);
    out = { ...out, report: `${bytes.subarray(0, Math.max(0, bytes.length - over * 2 - 64)).toString().replace(/\uFFFD$/, "")}\n[report truncated]` };
  }
  return out;
}

/**
 * The writer's own check runs: candidate mode only, a subset of the trusted profile's checks,
 * on a snapshot of the worktree as it is now. Problems the writer can fix come back as results.
 */
export class WriterChecks {
  private readonly active = new Map<string, { controller: AbortController; verifier: Verifier }>();
  constructor(private readonly options: {
    profile: Pick<VerificationProfile, "checks" | "writerChecks" | "isolation">;
    /** A candidate-only snapshot; throws ScopeViolationError for files outside the allowed paths. */
    snapshot(request: WriterChecksRequest): Promise<Candidate>;
    verifier(): Verifier;
  }) {}

  get checkNames(): string[] { return this.options.profile.checks.map(check => check.name); }
  get defaultChecks(): string[] { return this.options.profile.writerChecks?.defaultChecks ?? this.checkNames; }
  /** Worst case for one candidate-mode run of these checks; per-check set-up only when each check has its own container. */
  estimateMs(names: string[]): number {
    const perCheck = this.options.profile.isolation === "per_check" ? CHECK_OVERHEAD_MS : 0;
    return this.options.profile.checks.filter(check => names.includes(check.name)).reduce((total, check) => total + check.timeoutMs + perCheck, MODE_OVERHEAD_MS);
  }
  /** Kept back for the final verification (all checks; modes run in parallel) and the grader, unless the profile sets it. */
  get reserveMs(): number { return this.options.profile.writerChecks?.reserveMs ?? this.estimateMs(this.checkNames) + GRADER_RESERVE_MS; }

  async run(request: WriterChecksRequest): Promise<WriterChecksResponse> {
    const names = [...new Set(request.checks?.length ? request.checks : this.defaultChecks)];
    const unknown = names.filter(name => !this.checkNames.includes(name));
    if (unknown.length) return rejected("unknown_check", `Unknown checks: ${unknown.slice(0, 10).map(name => JSON.stringify(name.slice(0, 80))).join(", ")}. Available: ${this.checkNames.join(", ")}`);
    if (this.active.has(request.run_id)) return rejected("busy", "A run_checks call is already running for this task");
    const controller = new AbortController();
    const deadline = new AbortController();
    const timer = request.deadlineMs === undefined ? undefined : setTimeout(() => deadline.abort(new Error("run_checks_time_budget")), request.deadlineMs);
    const signal = AbortSignal.any([request.signal, controller.signal, deadline.signal]);
    const verifier = this.options.verifier();
    this.active.set(request.run_id, { controller, verifier });
    try {
      let candidate: Candidate;
      try { candidate = await this.options.snapshot(request); }
      catch (error) {
        if (!(error instanceof ScopeViolationError)) throw error;
        const files = error.files.slice(0, 50).map(file => file.slice(0, 256));
        return bounded({ status: "scope_violation", checks: [], files, report: "Changes outside the allowed paths; revert or move them before running checks." });
      }
      signal.throwIfAborted();
      const selected = this.options.profile.checks.filter(check => names.includes(check.name))
        .map(check => request.deadlineMs === undefined ? check : { ...check, timeoutMs: Math.max(1, Math.min(check.timeoutMs, request.deadlineMs)) });
      // Race the deadline: a verifier stuck on a container must not hold the writer past its window.
      const timedOut = new Promise<undefined>(resolve => { deadline.signal.addEventListener("abort", () => resolve(undefined), { once: true }); });
      const verification = await Promise.race([verifier.verify(candidate, signal, ["candidate"], selected), timedOut]).catch(error => { if (deadline.signal.aborted) return undefined; throw error; });
      if (!verification || deadline.signal.aborted) return rejectedForTime();
      signal.throwIfAborted();
      return bounded({
        status: verification.status,
        checks: verification.checks.map(check => ({ name: check.name.replace(/^candidate:/, ""), status: check.status })),
        report: formatCheckFeedback(verification, { maxBytes: MAX_WRITER_CHECKS_BYTES - 2048 }),
      });
    } finally {
      clearTimeout(timer);
      this.active.delete(request.run_id);
      await verifier.stop();
    }
  }

  /** Stops the run's in-flight checks and removes their containers. */
  async stop(runId: string): Promise<void> {
    const active = this.active.get(runId);
    if (!active) return;
    active.controller.abort(new Error("run_checks_stopped"));
    await active.verifier.stop();
  }
}
