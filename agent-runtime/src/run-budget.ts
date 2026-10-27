import { resolveAgentModel } from "./agent-model.js";
import type { BudgetLedger } from "./budget-ledger.js";
import type { ModelPrice } from "./gateway-budget.js";
import type { GatewayRole } from "./gateway-token.js";

/** Reasoning tokens count against the output cap, so writers need far more room than reviewers. */
export const OUTPUT_TOKEN_LIMITS: Record<GatewayRole, number> = { writer: 32_768, grader: 16_384, investigator: 8_192 };

export function isGatewayRole(value: unknown): value is GatewayRole {
  return typeof value === "string" && Object.hasOwn(OUTPUT_TOKEN_LIMITS, value);
}

/** No role's output cap goes past this, whatever a profile asks; a larger one also reserves more of the run's budget per request. */
export const OUTPUT_TOKEN_CEILING = 32_768;

/** A verification profile's `limits`: per-role output caps that replace the built-in ones. */
export interface ProfileLimits { outputTokens?: Partial<Record<GatewayRole, number>> }

export function outputTokenLimitFor(role: GatewayRole, limits?: ProfileLimits): number {
  return Math.min(OUTPUT_TOKEN_CEILING, limits?.outputTokens?.[role] ?? OUTPUT_TOKEN_LIMITS[role]);
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
// "rejected" is a run whose owner turned down a tool approval; it never resumes.
const TERMINAL = new Set(["completed", "failed", "cancelled", "timed_out", "rejected"]);

/** A run metered under its own default budget stops that budget when it ends, so its token can spend nothing more. */
export function releaseRunBudget(run: { run_id: string; status: string }, parents: ReadonlyMap<string, string>, ledger: { stop(id: string): void }): void {
  if (TERMINAL.has(run.status) && !parents.has(run.run_id)) ledger.stop(run.run_id);
}

/** Whether the budget an execution spends (its parent run's, or its own) is still open. */
export function executionActive(execution: string, parents: ReadonlyMap<string, string>, ledger: { active(id: string): boolean }, recorded?: (execution: string) => string | undefined): boolean {
  return ledger.active(parents.get(execution) ?? recorded?.(execution) ?? execution);
}

/** Dollars spent of the ledger budget an execution is metered under, for its writer's budget_status dialog. */
export function runBudgetStatus(execution: string, parents: ReadonlyMap<string, string>, ledger: Pick<BudgetLedger, "open" | "summary">, defaultUsd: number, recorded?: (execution: string) => string | undefined): { used_usd: number; limit_usd: number } | undefined {
  // The same budget its gateway token spends; the gateway refuses a request once this reaches the limit.
  const { estimated_usd, initial_cost_usd = 0, limit_usd } = ledger.summary(budgetIdFor(execution, parents, ledger, defaultUsd, recorded));
  return limit_usd === undefined ? undefined : { used_usd: initial_cost_usd + estimated_usd, limit_usd };
}
