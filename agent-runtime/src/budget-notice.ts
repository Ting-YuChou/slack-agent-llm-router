import { renderPrompt } from "./prompts.js";

/** The policy extension's dollar budget query; the host answers it at most once per turn. */
export const BUDGET_STATUS_DIALOG = "pi-agent/budget_status";

export interface UsdBudgetNotice { stage: "notice" | "final"; text: string }

/**
 * The notice for the host's budget_status reply: a notice at 75% of the run's model budget, a final
 * warning at 90%. Anything but finite dollar figures, such as {unavailable:true}, gets no notice.
 */
export function usdBudgetNotice(reply: string | undefined): UsdBudgetNotice | undefined {
  if (reply === undefined) return undefined;
  let parsed: { used_usd?: unknown; limit_usd?: unknown };
  try { parsed = JSON.parse(reply) as typeof parsed; } catch { return undefined; }
  const { used_usd: used, limit_usd: limit } = parsed ?? {};
  if (typeof used !== "number" || typeof limit !== "number" || !Number.isFinite(used) || !Number.isFinite(limit) || used < 0 || limit <= 0) return undefined;
  const usd = (value: number) => `$${value.toFixed(2)}`;
  const unit = "of the model budget";
  if (used >= 0.9 * limit) return { stage: "final", text: renderPrompt("budget-final", { used: usd(used), max: usd(limit), unit, left: usd(Math.max(0, limit - used)) }) };
  if (used >= 0.75 * limit) return { stage: "notice", text: renderPrompt("budget-notice", { used: usd(used), max: usd(limit), unit }) };
  return undefined;
}
