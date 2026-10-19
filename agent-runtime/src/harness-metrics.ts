import type { RunRecord } from "./orchestrator.js";

/** Per-run signals for tuning the harness itself: limits, the repair loop, friction, cost. */
export interface HarnessMetrics {
  outcome: string | null;
  turn_util: number | null; tool_util: number | null;
  attempts: number; first_attempt_pass: boolean | null; repair_success: boolean | null;
  writer_checks_calls: number; approvals: number; policy_blocks: number;
  /** Files the profile's fixers rewrote, summed over attempts. */
  fixed_files: number;
  cost_usd: number | null; wall_s: number; active_s: number | null; human_wait_s: number | null;
  time_to_first_edit_s: number | null;
}

const ratio = (used: number, max: number | undefined) => max ? Math.round((used / max) * 1000) / 1000 : null;
const seconds = (ms: number) => Math.round(ms / 100) / 10;

/** Derived from the run alone, so it can be recomputed from any persisted record. */
export function computeHarnessMetrics(run: RunRecord, now = Date.now()): HarnessMetrics {
  const attempts = run.attempts ?? [];
  // Unverified runs have no gate to pass, so they have no attempt verdicts either.
  const gated = attempts.length > 0 && run.verification_status !== "unverified";
  const first = attempts[0];
  const created = Date.parse(run.created_at);
  return {
    outcome: run.execution_outcome ?? null,
    turn_util: ratio(run.turn_count, run.limits?.maxTurns),
    tool_util: ratio(run.tool_count, run.limits?.maxToolCalls),
    attempts: attempts.length,
    first_attempt_pass: gated ? first!.verification_status === "passed" && first!.grader_status === "passed" : null,
    repair_success: gated && attempts.length > 1 ? run.status === "completed" : null,
    writer_checks_calls: run.writer_checks_calls ?? 0,
    approvals: run.approvals?.requested ?? 0,
    policy_blocks: run.policy_summary?.block ?? 0,
    fixed_files: attempts.reduce((sum, attempt) => sum + (attempt.fixes ?? []).reduce((files, fix) => files + fix.files.length, 0), 0),
    cost_usd: run.estimated_cost_usd ?? null,
    wall_s: seconds(run.timing?.wall_ms ?? Math.max(0, (Date.parse(run.updated_at) || now) - created)),
    active_s: run.timing ? seconds(run.timing.active_ms) : null,
    human_wait_s: run.timing ? seconds(run.timing.human_wait_ms) : null,
    time_to_first_edit_s: run.first_edit_at ? seconds(Date.parse(run.first_edit_at) - created) : null,
  };
}
