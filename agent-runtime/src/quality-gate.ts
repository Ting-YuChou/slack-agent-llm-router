import type { FixResult } from "./fixers.js";
import { randomBytes } from "node:crypto";

import { ScopeViolationError } from "./candidate-scope.js";
import { formatCheckFeedback, type CheckFailure } from "./check-report.js";
import { HarnessError, type CauseCode } from "./error-taxonomy.js";
import { promptTelemetry, renderPrompt } from "./prompts.js";

export type CheckStatus = "passed" | "failed" | "unknown";
export interface VerificationCheck {
  name: string; status: CheckStatus; exit_code: number | null; duration_ms?: number; passed?: number | null; failed?: number | null; artifact?: string;
  /** The profile check's kind, for failure categories in analytics. */
  kind?: "static" | "test";
  /** Parsed diagnostics, repository-relative. */
  failures?: CheckFailure[];
  /** Redacted, bounded log tail; never a host path. */
  excerpt?: string;
  /** Why the check has no trustworthy result (for example `check_timeout`). */
  cause?: string;
  timed_out?: boolean;
}
export interface VerificationResult { status: CheckStatus; checks: VerificationCheck[] }
export type GraderAggregate = "unanimous" | "majority";
/** How many independent graders judge one candidate, and how their verdicts combine. */
export interface GraderSampling { samples: 1 | 3; aggregate: GraderAggregate }
export const DEFAULT_GRADER_SAMPLING: GraderSampling = { samples: 1, aggregate: "unanimous" };
/** One sample's own verdict; `cause_code` says why an unknown sample gave none. */
export interface SampleVerdict { status: Grade["status"]; grader_attempts: number; cause_code?: CauseCode }
export interface Grade {
  status: "passed" | "needs_changes" | "unknown";
  candidate_hash: string;
  criteria: Array<{ id: string; passed: boolean; evidence?: string }>;
  findings: Array<{ path: string; line: number; severity: "error" | "warning"; reason: string }>;
  /** Set on a harness-made unknown: why no valid verdict exists. Never part of a grader's own submission. */
  cause?: { code: CauseCode; message: string };
  /** Grader executions spent on this candidate, retries included. */
  grader_attempts?: number;
  /** The sampling this verdict was aggregated under. */
  samples?: GraderSampling["samples"];
  aggregate?: GraderAggregate;
  /** Every sample that ran, in order; sampling stops once no later sample can change the outcome. */
  sample_verdicts?: SampleVerdict[];
}
export interface Candidate { sourceHash?: string; testedBaseline?: string; hash: string; path: string; patch?: string; artifact?: string;
  /** What the profile's fixers changed before this snapshot, and why they gave up if they did. */
  fixes?: FixResult[]; fixer_cause?: string }
export interface GateInput { run_id: string; worktree: string; baseline: string; requirement: string; attempt: number; signal: AbortSignal }
export interface GateResult { action: "pass" | "repair" | "fail"; candidate: Candidate; verification: VerificationResult; grade?: Grade; repair_prompt?: string; failure?: { code: "scope_violation"; files: string[] } | { code: "verification_failed"; cause_code?: CauseCode } }
export interface QualityGateOptions {
  profile: { id: string; rubric: string[]; maxRepairs: number; grader?: GraderSampling };
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

/**
 * The aggregated status once no remaining sample can change it, otherwise undefined.
 * unanimous: any needs_changes rejects; passed needs every sample to pass.
 * majority: a strict majority of the valid (passed | needs_changes) samples; unknown samples do not vote.
 * Either way the result is unknown only when no decided outcome remains possible.
 */
export function decideSamples(statuses: Array<Grade["status"]>, sampling: GraderSampling): Grade["status"] | undefined {
  const passed = statuses.filter(status => status === "passed").length;
  const rejected = statuses.filter(status => status === "needs_changes").length;
  const remaining = sampling.samples - statuses.length;
  if (sampling.aggregate === "unanimous") {
    if (rejected) return "needs_changes";
    if (remaining > 0) return undefined;
    return passed === statuses.length ? "passed" : "unknown";
  }
  if (passed > rejected + remaining) return "passed";
  if (rejected > passed + remaining) return "needs_changes";
  return remaining > 0 ? undefined : "unknown";
}

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
  /** Samples independent graders until the profile's aggregate is decided; the verdict carries the deciding sample's criteria and findings. */
  private async gradeSamples(candidate: Candidate, verification: VerificationResult, input: GateInput): Promise<Grade> {
    const sampling = this.options.profile.grader ?? DEFAULT_GRADER_SAMPLING;
    const grades: Grade[] = [];
    let status: Grade["status"] | undefined;
    while ((status = decideSamples(grades.map(grade => grade.status), sampling)) === undefined) {
      input.signal.throwIfAborted();
      grades.push(await this.gradeWithRetry(candidate, verification, input));
    }
    // An unknown outcome takes the first sample that gave no verdict, and so its cause.
    const deciding = grades.find(grade => grade.status === status) ?? syntheticUnknown(this.options.profile.rubric, candidate.hash, { code: "insufficient_evidence", message: "Grader samples reached no verdict" });
    return {
      ...deciding, grader_attempts: grades.reduce((total, grade) => total + (grade.grader_attempts ?? 1), 0),
      samples: sampling.samples, aggregate: sampling.aggregate,
      sample_verdicts: grades.map(grade => ({ status: grade.status, grader_attempts: grade.grader_attempts ?? 1, ...(grade.cause ? { cause_code: grade.cause.code } : {}) })),
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
      grade = await this.gradeSamples(candidate, verification, input);
      await this.options.record?.("grade", { ...grade, ...promptTelemetry("grader") }, input);
    }
    input.signal.throwIfAborted();
    if (verification.status === "passed" && grade?.status === "passed") return { action: "pass", candidate, verification, grade };
    const retry = verification.status !== "unknown" && grade?.status !== "unknown" && input.attempt <= this.options.profile.maxRepairs;
    const cause = verification.status === "unknown" ? verificationCause(verification) : grade?.cause?.code;
    const failure = !retry && cause ? { code: "verification_failed" as const, cause_code: cause } : undefined;
    return { action: retry ? "repair" : "fail", candidate, verification, grade, ...(failure ? { failure } : {}),
      repair_prompt: repairPrompt(input.requirement, verification, grade) };
  }
}

/** Why verification has no trustworthy result: it ran out of time, or a check never reported. */
function verificationCause(verification: VerificationResult): CauseCode {
  return verification.checks.some(check => check.status === "unknown" && (check.timed_out || check.cause === "check_timeout" || check.cause === "mode_timeout")) ? "verification_timeout" : "verification_incomplete";
}

/** Diagnostics go inside a block whose random boundary the candidate cannot predict or close early. */
function repairPrompt(requirement: string, verification: VerificationResult, grade?: Grade): string {
  const findings = grade ? [`Grader: ${grade.status}`, ...grade.criteria.filter(c => !c.passed).map(c => `- unmet: ${c.id}`), ...grade.findings.slice(0, 50).map(f => `- ${f.path}:${f.line} [${f.severity}] ${f.reason.slice(0, 500)}`)] : [];
  const body = [formatCheckFeedback(verification), ...findings].join("\n");
  let boundary: string;
  do boundary = `UNTRUSTED-${randomBytes(12).toString("hex")}`; while (body.includes(boundary));
  return renderPrompt("repair", { requirement, boundary, body });
}
