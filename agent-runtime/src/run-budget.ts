import { resolveAgentModel } from "./agent-model.js";
import type { ModelPrice } from "./gateway-budget.js";
import type { GatewayRole } from "./gateway-token.js";

/** Reasoning tokens count against the output cap, so writers need far more room than reviewers. */
export const OUTPUT_TOKEN_LIMITS: Record<GatewayRole, number> = { writer: 32_768, grader: 16_384, investigator: 8_192 };

export function outputTokenLimitFor(role: GatewayRole): number {
  return OUTPUT_TOKEN_LIMITS[role];
}

/** Workflow executions spend their parent's budget; any other run is metered under a default budget of its own. */
export function budgetIdFor(execution: string, parents: ReadonlyMap<string, string>, ledger: { open(id: string, usd: number): void }, defaultUsd: number, recorded?: (execution: string) => string | undefined): string {
  // The map is in memory; the control store's execution record outlives a restart.
  const parent = parents.get(execution) ?? recorded?.(execution);
  if (parent) return parent;
  ledger.open(execution, defaultUsd);
  return execution;
}

export function parseDefaultRunBudget(env: NodeJS.ProcessEnv): number {
  const raw = env.PI_AGENT_DEFAULT_RUN_BUDGET_USD;
  // Reservations count request bytes as tokens, so the cap must leave room for large contexts.
  if (raw === undefined || raw === "") return 5;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error("PI_AGENT_DEFAULT_RUN_BUDGET_USD must be a positive number of USD");
  return value;
}

/** Operator-configured prices win; otherwise fall back to the pinned model catalog (USD per million tokens). */
export function priceFor(ref: string, configured: Record<string, ModelPrice>): ModelPrice | undefined {
  if (configured[ref]) return configured[ref];
  try {
    const { cost } = resolveAgentModel(ref);
    return cost.input > 0 ? { input: cost.input, output: cost.output, cacheRead: cost.cacheRead } : undefined;
  } catch { return undefined; }
}

// Not "interrupted": an interrupted structured run can resume and must keep its budget.
const TERMINAL = new Set(["completed", "failed", "cancelled", "timed_out"]);

/** A run metered under its own default budget stops that budget when it ends, so its token can spend nothing more. */
export function releaseRunBudget(run: { run_id: string; status: string }, parents: ReadonlyMap<string, string>, ledger: { stop(id: string): void }): void {
  if (TERMINAL.has(run.status) && !parents.has(run.run_id)) ledger.stop(run.run_id);
}

/** Whether the budget an execution spends (its parent run's, or its own) is still open. */
export function executionActive(execution: string, parents: ReadonlyMap<string, string>, ledger: { active(id: string): boolean }, recorded?: (execution: string) => string | undefined): boolean {
  return ledger.active(parents.get(execution) ?? recorded?.(execution) ?? execution);
}
