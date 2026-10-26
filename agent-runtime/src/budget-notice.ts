import { renderPrompt } from "./prompts.js";

/** The policy extension's budget query; the host answers it at most once per turn. */
export const BUDGET_STATUS_DIALOG = "pi-agent/budget_status";

/** The host adds the run's active time to its budget_status reply once less than this share of it is left. */
export const SHORT_TIME_SHARE = 0.25;

export interface BudgetNotice { stage: "notice" | "final"; text: string }

function parseReply(reply: string | undefined): Record<string, unknown> | undefined {
  if (reply === undefined) return undefined;
  try {
    const parsed: unknown = JSON.parse(reply);
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : undefined;
  } catch { return undefined; }
}

/** A finite amount used of a positive limit, or nothing. */
function budgetFigures(used: unknown, limit: unknown): [used: number, limit: number] | undefined {
  if (typeof used !== "number" || typeof limit !== "number" || !Number.isFinite(used) || !Number.isFinite(limit) || used < 0 || limit <= 0) return undefined;
  return [used, limit];
}

/**
 * The notice for the host's budget_status reply: a notice at 75% of the run's model budget, a final
 * warning at 90%. Anything but finite dollar figures, such as {unavailable:true}, gets no notice.
 */
export function usdBudgetNotice(reply: string | undefined): BudgetNotice | undefined {
  const parsed = parseReply(reply);
  const figures = budgetFigures(parsed?.used_usd, parsed?.limit_usd);
  if (!figures) return undefined;
  const [used, limit] = figures;
  const usd = (value: number) => `$${value.toFixed(2)}`;
  const vars = { used: usd(used), max: usd(limit), unit: "of the model budget", left: usd(Math.max(0, limit - used)) };
  if (used >= 0.9 * limit) return { stage: "final", text: renderPrompt("budget-final", vars) };
  if (used >= 0.75 * limit) return { stage: "notice", text: renderPrompt("budget-notice", vars) };
  return undefined;
}

/**
 * The notice for the run's active time in the same reply, which carries it only once less than a quarter
 * is left: a notice naming the minutes left, then a final warning once a tenth or less is left.
 */
export function activeTimeNotice(reply: string | undefined): BudgetNotice | undefined {
  const parsed = parseReply(reply);
  const figures = budgetFigures(parsed?.used_ms, parsed?.limit_ms);
  if (!figures) return undefined;
  const [usedMs, limitMs] = figures;
  const leftMs = Math.max(0, limitMs - usedMs);
  if (leftMs >= SHORT_TIME_SHARE * limitMs) return undefined;
  // Tenths of a minute, with what is left rounded down so the agent never counts on time it does not have.
  const max = Math.round(limitMs / 6_000) / 10;
  const left = Math.floor(leftMs / 6_000) / 10;
  const vars = { used: Math.max(0, Math.round((max - left) * 10) / 10), max, unit: "minutes of the time budget", left: `${left} minute${left === 1 ? "" : "s"}` };
  if (leftMs <= 0.1 * limitMs) return { stage: "final", text: renderPrompt("budget-final", vars) };
  return { stage: "notice", text: renderPrompt("budget-notice", vars) };
}
