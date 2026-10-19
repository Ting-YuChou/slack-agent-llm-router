import { randomBytes } from "node:crypto";

import { ScopeViolationError } from "./candidate-scope.js";
import { formatCheckFeedback, type CheckFailure } from "./check-report.js";
import { HarnessError, type CauseCode } from "./error-taxonomy.js";
import { promptTelemetry, renderPrompt } from "./prompts.js";

export type CheckStatus = "passed" | "failed" | "unknown";
export interface VerificationCheck {
  name: string; status: CheckStatus; exit_code: number | null; duration_ms?: number; passed?: number | null; failed?: number | null; artifact?: string;
  /** Parsed diagnostics, repository-relative. */
  failures?: CheckFailure[];
  /** Redacted, bounded log tail; never a host path. */
  excerpt?: string;
  /** Why the check has no trustworthy result (for example `check_timeout`). */
  cause?: string;
  timed_out?: boolean;
}
export interface VerificationResult { status: CheckStatus; checks: VerificationCheck[] }
export interface Grade {
  status: "passed" | "needs_changes" | "unknown";
  candidate_hash: string;
  criteria: Array<{ id: string; passed: boolean; evidence?: string }>;
  findings: Array<{ path: string; line: number; severity: "error" | "warning"; reason: string }>;
  /** Set on a harness-made unknown: why no valid verdict exists. Never part of a grader's own submission. */
  cause?: { code: CauseCode; message: string };
  /** Grader executions spent on this candidate, retries included. */
  grader_attempts?: number;
}
export interface Candidate { sourceHash?: string; testedBaseline?: string; hash: string; path: string; patch?: string; artifact?: string }
export interface GateInput { run_id: string; worktree: string; baseline: string; requirement: string; attempt: number; signal: AbortSignal }
export interface GateResult { action: "pass" | "repair" | "fail"; candidate: Candidate; verification: VerificationResult; grade?: Grade; repair_prompt?: string; failure?: { code: "scope_violation"; files: string[] } | { code: "verification_failed"; cause_code?: CauseCode } }
export interface QualityGateOptions {
  profile: { id: string; rubric: string[]; maxRepairs: number };
  snapshot(input: GateInput): Promise<Candidate>;
  verify(candidate: Candidate, input: GateInput): Promise<VerificationResult>;
  grade(candidate: Candidate, verification: VerificationResult, input: GateInput): Promise<unknown>;
  record?(kind: string, payload: unknown, input: GateInput): Promise<void>;
}
/** Verification as models may see it: host artifact paths stay on the host. */
export function publicVerification(verification: VerificationResult): VerificationResult {
  return { status: verification.status, checks: verification.checks.map(({ artifact: _artifact, ...check }) => check) };
}
export function validateGrade(value: unknown, hash: string, rubric: string[]): Grade {
  const g = value as Grade;
  if (!g || !["passed", "needs_changes", "unknown"].includes(g.status) || g.candidate_hash !== hash || !Array.isArray(g.criteria) || g.criteria.some(c => !c || typeof c !== "object") || !Array.isArray(g.findings) || g.findings.length > 50 || g.criteria.length !== rubric.length) throw new Error("Invalid grader result");
  if (new Set(g.criteria.map(c => c.id)).size !== rubric.length || !rubric.every(id => g.criteria.some(c => c.id === id && typeof c.passed === "boolean"))) throw new Error("Missing grader criteria");
  if (g.findings.some(f => !f || typeof f.path !== "string" || f.path.startsWith("/") || f.path.split(/[\\/]/).includes("..") || !Number.isInteger(f.line) || f.line < 1 || !["error", "warning"].includes(f.severity) || typeof f.reason !== "string" || f.reason.length > 4000)) throw new Error("Invalid grader finding");
  if (g.criteria.some(c => c.evidence !== undefined && (typeof c.evidence !== "string" || c.evidence.length > 4000))) throw new Error("Invalid grader evidence");
  if (g.status === "passed" && (g.criteria.some(c => !c.passed || !c.evidence?.trim()) || g.findings.some(f => f.severity === "error"))) throw new Error("Inconsistent grader verdict");
  return g;
}
/** A harness-made unknown verdict: no criterion passes, and `cause` says why the grader gave none. */
export function syntheticUnknown(rubric: string[], hash: string, cause: NonNullable<Grade["cause"]>): Grade {
  return { status: "unknown", candidate_hash: hash, criteria: rubric.map(id => ({ id, passed: false })), findings: [], cause };
}

function graderCause(error: unknown): NonNullable<Grade["cause"]> {
  if (error instanceof HarnessError) return { code: error.causeCode, message: error.message.slice(0, 300) };
  const detail = error instanceof Error ? error.message.slice(0, 200) : "unknown error";
  // validateGrade throws plain errors for malformed verdicts; anything else broke the grader itself.
  return /^(Invalid grader|Missing grader|Inconsistent grader)/.test(detail)
    ? { code: "grader_invalid_submission", message: `Grader result was rejected: ${detail}` }
    : { code: "grader_execution_failed", message: `Grader failed: ${detail}` };
}

const GRADER_ATTEMPTS = 2;

function isBudgetExhausted(error: unknown): boolean {
  return error instanceof Error && error.message === "workflow_budget_exhausted";
}

export class QualityGate {
  constructor(private readonly options: QualityGateOptions) {}
  /** Out-of-scope edits are a writer mistake, not missing infrastructure: give it the file list to revert. */
  private scopeViolation(input: GateInput, files: string[]): GateResult {
    const listed = files.slice(0, 50);
    return {
      action: input.attempt <= this.options.profile.maxRepairs ? "repair" : "fail",
      candidate: { hash: "", path: input.worktree },
      verification: { status: "failed", checks: [{ name: "allowed_paths", status: "failed", exit_code: null }] },
      failure: { code: "scope_violation", files: listed },
      repair_prompt: renderPrompt("scope-repair", { requirement: input.requirement, files: JSON.stringify(listed) }),
    };
  }
  /** Grader infrastructure errors get one fresh grader; a grader that judges the evidence insufficient does not. */
  private async gradeWithRetry(candidate: Candidate, verification: VerificationResult, input: GateInput): Promise<Grade> {
    const { rubric } = this.options.profile;
    for (let attempt = 1; ; attempt++) {
      try {
        const grade = { ...validateGrade(await this.options.grade(candidate, verification, input), candidate.hash, rubric), grader_attempts: attempt };
        // `cause` is harness-owned; never let a submission supply one.
        if (grade.status === "unknown") grade.cause = { code: "insufficient_evidence", message: "Grader judged the evidence insufficient for a verdict" };
        else delete grade.cause;
        return grade;
      } catch (error) {
        if (input.signal.aborted || isBudgetExhausted(error)) throw error;
        if (attempt < GRADER_ATTEMPTS) continue;
        return { ...syntheticUnknown(rubric, candidate.hash, graderCause(error)), grader_attempts: attempt };
      }
    }
  }

  async evaluate(input: GateInput): Promise<GateResult> {
    input.signal.throwIfAborted();
    let candidate: Candidate;
    try { candidate = await this.options.snapshot(input); }
    catch (error) {
      if (!(error instanceof ScopeViolationError)) throw error;
      return this.scopeViolation(input, error.files);
    }
    const verification = await this.options.verify(candidate, input);
    input.signal.throwIfAborted();
    await this.options.record?.("verification", { candidate_hash: candidate.hash, source_hash: candidate.sourceHash ?? candidate.hash, tested_baseline: candidate.testedBaseline ?? input.baseline, ...verification, ...promptTelemetry("implementation") }, input);
    let grade: Grade | undefined;
    if (verification.status === "passed") {
      grade = await this.gradeWithRetry(candidate, verification, input);
      await this.options.record?.("grade", { ...grade, ...promptTelemetry("grader") }, input);
    }
    input.signal.throwIfAborted();
    if (verification.status === "passed" && grade?.status === "passed") return { action: "pass", candidate, verification, grade };
    const retry = verification.status !== "unknown" && grade?.status !== "unknown" && input.attempt <= this.options.profile.maxRepairs;
    const failure = !retry && grade?.cause ? { code: "verification_failed" as const, cause_code: grade.cause.code } : undefined;
    return { action: retry ? "repair" : "fail", candidate, verification, grade, ...(failure ? { failure } : {}),
      repair_prompt: repairPrompt(input.requirement, verification, grade) };
  }
}

/** Diagnostics go inside a block whose random boundary the candidate cannot predict or close early. */
function repairPrompt(requirement: string, verification: VerificationResult, grade?: Grade): string {
  const findings = grade ? [`Grader: ${grade.status}`, ...grade.criteria.filter(c => !c.passed).map(c => `- unmet: ${c.id}`), ...grade.findings.slice(0, 50).map(f => `- ${f.path}:${f.line} [${f.severity}] ${f.reason.slice(0, 500)}`)] : [];
  const body = [formatCheckFeedback(verification), ...findings].join("\n");
  let boundary: string;
  do boundary = `UNTRUSTED-${randomBytes(12).toString("hex")}`; while (body.includes(boundary));
  return renderPrompt("repair", { requirement, boundary, body });
}
