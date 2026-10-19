import { DiffLimitError, ProtectedFileError } from "./worktree-manager.js";

/** Why a harness step failed, finer than the run's error code (which stays stable for clients). */
export type CauseCode =
  | "docker_unavailable"
  | "snapshot_too_large"
  | "candidate_changed"
  | "candidate_artifact_missing"
  | "verification_stop_unconfirmed"
  | "grader_timeout"
  | "grader_did_not_submit"
  | "grader_invalid_submission"
  | "grader_execution_failed"
  | "insufficient_evidence";

/** A failure whose cause the harness knows; the gate reports it instead of a bare message. */
export class HarnessError extends Error {
  constructor(readonly causeCode: CauseCode, message: string) {
    super(message);
    this.name = "HarnessError";
  }
}

export interface RunFailure { code: string; message: string; cause_code?: CauseCode }

/** Maps an exception escaping the quality gate to the run's public error. */
export function classifyGateError(error: unknown): RunFailure {
  const detail = error instanceof Error ? error.message.slice(0, 200) : "unknown error";
  if (error instanceof ProtectedFileError) return { code: "protected_file_modified", message: `${detail}; changes were not delivered` };
  if (error instanceof DiffLimitError) return { code: "diff_limit_exceeded", message: `${detail}; split the change or reduce its size` };
  if (detail === "workflow_budget_exhausted") return { code: "workflow_budget_exhausted", message: "Workflow cost or time budget was exhausted before verification finished" };
  const failure: RunFailure = { code: "quality_gate_unavailable", message: `Required verification could not complete (${detail})` };
  if (error instanceof HarnessError) failure.cause_code = error.causeCode;
  return failure;
}
